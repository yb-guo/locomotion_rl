"""Minimal PPO core for task014 smoke runs.

Torch is imported only inside torch-required functions so local no-torch pytest
can still import this module.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

REWARD_COMPONENT_NAMES = (
    "tracking_lin_vel",
    "tracking_yaw_rate",
    "upright",
    "tracking_base_height",
    "action_rate_penalty",
    "joint_velocity_penalty",
    "joint_deviation_penalty",
    "termination_penalty",
)


@dataclass(frozen=True, slots=True)
class PPOConfig:
    obs_dim: int = 90
    action_dim: int = 27
    hidden_dim: int = 128
    hidden_layers: int = 2
    n_envs: int = 1024
    rollout_steps: int = 32
    ppo_updates: int = 5
    epochs: int = 2
    minibatch_size: int = 8192
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    log_std_init: float = -0.5
    tanh_eps: float = 1e-6

    def __post_init__(self) -> None:
        positive_ints = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": self.hidden_layers,
            "n_envs": self.n_envs,
            "rollout_steps": self.rollout_steps,
            "ppo_updates": self.ppo_updates,
            "epochs": self.epochs,
            "minibatch_size": self.minibatch_size,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.minibatch_size > self.n_envs * self.rollout_steps:
            raise ValueError("minibatch_size must not exceed rollout batch size")
        for name in ("lr", "gamma", "gae_lambda", "clip", "value_coef", "max_grad_norm"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.entropy_coef < 0.0:
            raise ValueError("entropy_coef must be non-negative")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in (0, 1]")
        if self.tanh_eps <= 0.0:
            raise ValueError("tanh_eps must be positive")


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    observations: Any
    actions: Any
    rewards: Any
    dones: Any
    values: Any
    log_probs: Any
    next_observation: Any
    next_value: Any
    collect_time_s: float
    env_steps: int
    reward_mean: float
    reward_component_means: dict[str, float]
    done_count: int
    timeout_count: int
    fallen_count: int
    reset_count: int
    height_bad_count: int
    termination_height_bad_count: int
    tilt_bad_count: int
    height_reset_count: int
    tilt_reset_count: int
    full_env_reset_wave: bool
    full_env_reset_wave_count: int
    episode_length_mean: float
    episode_length_min: float
    episode_length_max: float
    completed_episode_length_mean: float
    completed_episode_count: int
    root_height_mean: float
    root_height_min: float
    upright_mean: float
    terminated: Any | None = None
    truncated: Any | None = None
    terminal_values: Any | None = None


@dataclass(frozen=True, slots=True)
class PPODiagnostics:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float
    update_time_s: float
    update_samples_per_sec: float


def describe_training_plan() -> tuple[str, ...]:
    return (
        "Validate simulator reset/step API.",
        "Freeze observation and action schema.",
        "Run a small PPO baseline.",
        "Swap in transformer policy after baseline reward improves.",
    )


def require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - local no-torch path.
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch


def build_actor_critic(config: PPOConfig, *, device: str) -> Any:
    torch = require_torch()
    nn = torch.nn

    class ActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.actor = make_mlp(nn, config.obs_dim, config.action_dim, config)
            self.value = make_mlp(nn, config.obs_dim, 1, config)
            self.log_std = nn.Parameter(
                torch.full((config.action_dim,), config.log_std_init, device=device)
            )

        def forward(self, observation: Any) -> tuple[Any, Any]:
            mean = self.actor(observation)
            value = self.value(observation).squeeze(-1)
            return mean, value

        def act(self, observation: Any) -> tuple[Any, Any, Any, Any]:
            mean, value = self.forward(observation)
            log_std = self.log_std.expand_as(mean)
            std = log_std.exp()
            raw_action = mean + std * torch.randn_like(mean)
            action = torch.tanh(raw_action)
            log_prob = tanh_gaussian_log_prob_from_raw(raw_action, mean, log_std, config)
            entropy = gaussian_entropy(log_std)
            return action, log_prob, value, entropy

        def evaluate_actions(self, observation: Any, action: Any) -> tuple[Any, Any, Any]:
            mean, value = self.forward(observation)
            log_std = self.log_std.expand_as(mean)
            log_prob = tanh_gaussian_log_prob_from_action(action, mean, log_std, config)
            entropy = gaussian_entropy(log_std)
            return log_prob, entropy, value

    return ActorCritic().to(device)


def make_mlp(nn: Any, input_dim: int, output_dim: int, config: PPOConfig) -> Any:
    layers: list[Any] = []
    current_dim = input_dim
    for _ in range(config.hidden_layers):
        layer = nn.Linear(current_dim, config.hidden_dim)
        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        layers.append(nn.Tanh())
        current_dim = config.hidden_dim
    output = nn.Linear(current_dim, output_dim)
    nn.init.orthogonal_(output.weight, gain=0.01)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


def tanh_gaussian_log_prob_from_raw(
    raw_action: Any,
    mean: Any,
    log_std: Any,
    config: PPOConfig,
) -> Any:
    torch = require_torch()
    std = log_std.exp()
    normal = torch.distributions.Normal(mean, std)
    action = torch.tanh(raw_action)
    log_prob = normal.log_prob(raw_action) - torch.log(
        1.0 - action.square() + config.tanh_eps
    )
    return log_prob.sum(dim=-1)


def tanh_gaussian_log_prob_from_action(
    action: Any,
    mean: Any,
    log_std: Any,
    config: PPOConfig,
) -> Any:
    torch = require_torch()
    clipped = action.clamp(-1.0 + config.tanh_eps, 1.0 - config.tanh_eps)
    raw_action = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))
    return tanh_gaussian_log_prob_from_raw(raw_action, mean, log_std, config)


def gaussian_entropy(log_std: Any) -> Any:
    return (log_std + 0.5 * math.log(2.0 * math.pi * math.e)).sum(dim=-1)


def collect_rollout(env: Any, model: Any, observation: Any, config: PPOConfig) -> RolloutBatch:
    torch = require_torch()
    observations: list[Any] = []
    actions: list[Any] = []
    rewards: list[Any] = []
    dones: list[Any] = []
    truncated_flags: list[Any] = []
    terminated_flags: list[Any] = []
    height_bad_flags: list[Any] = []
    termination_height_bad_flags: list[Any] = []
    tilt_bad_flags: list[Any] = []
    episode_lengths: list[Any] = []
    completed_episode_lengths: list[Any] = []
    root_height_values: list[Any] = []
    upright_values: list[Any] = []
    reward_component_values: dict[str, list[Any]] = {
        name: [] for name in REWARD_COMPONENT_NAMES
    }
    values: list[Any] = []
    log_probs: list[Any] = []
    terminal_observations: list[Any] = []
    reset_count = 0
    full_env_reset_wave_count = 0
    synchronize_device(getattr(observation, "device", None))
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(config.rollout_steps):
            action, log_prob, value, _entropy = model.act(observation)
            transition = env.step(action)
            reward = transition.reward
            done = transition.done
            terminal_observation = getattr(transition, "terminal_observation", None)
            if terminal_observation is None:
                terminal_observation = transition.info.get("terminal_observation")
            if terminal_observation is None:
                terminal_observation = transition.observation
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            truncated_flags.append(transition.truncated)
            terminated_flags.append(transition.terminated)
            terminal_observations.append(terminal_observation)
            components = transition.info.get("components", {})
            if "height_bad" in components:
                height_bad_flags.append(components["height_bad"])
            if "termination_height_bad" in components:
                termination_height_bad_flags.append(components["termination_height_bad"])
            if "tilt_bad" in components:
                tilt_bad_flags.append(components["tilt_bad"])
            if "episode_lengths" in transition.info:
                episode_lengths.append(transition.info["episode_lengths"])
            if "completed_episode_lengths" in transition.info:
                completed = transition.info["completed_episode_lengths"]
                if tensor_length(completed) > 0:
                    completed_episode_lengths.append(completed)
            if "root_height" in components:
                root_height_values.append(components["root_height"])
            if "upright" in components:
                upright_values.append(components["upright"])
            for name in REWARD_COMPONENT_NAMES:
                if name in components:
                    reward_component_values[name].append(components[name])
            values.append(value)
            log_probs.append(log_prob)
            reset_count += int(transition.info.get("reset_count", 0))
            full_env_reset_wave_count += int(bool(transition.info.get("full_env_reset_wave", False)))
            observation = transition.observation
        next_value = model.forward(observation)[1]
        terminal_observation_tensor = torch.stack(terminal_observations)
        terminal_values = model.forward(
            terminal_observation_tensor.reshape(-1, config.obs_dim)
        )[1].reshape(config.rollout_steps, config.n_envs)
    synchronize_device(getattr(observation, "device", None))
    collect_time_s = time.perf_counter() - started
    observation_tensor = torch.stack(observations)
    action_tensor = torch.stack(actions)
    reward_tensor = torch.stack(rewards)
    done_tensor = torch.stack(dones)
    terminated_tensor = torch.stack(terminated_flags)
    truncated_tensor = torch.stack(truncated_flags)
    value_tensor = torch.stack(values)
    log_prob_tensor = torch.stack(log_probs)
    assert_finite_tensor(observation_tensor, "observation")
    assert_finite_tensor(action_tensor, "action")
    assert_finite_tensor(reward_tensor, "reward")
    assert_finite_tensor(value_tensor, "value")
    assert_finite_tensor(log_prob_tensor, "log_prob")
    env_steps = config.rollout_steps * config.n_envs
    return RolloutBatch(
        observations=observation_tensor,
        actions=action_tensor,
        rewards=reward_tensor,
        dones=done_tensor,
        values=value_tensor,
        log_probs=log_prob_tensor,
        next_observation=observation,
        next_value=next_value,
        collect_time_s=collect_time_s,
        env_steps=env_steps,
        reward_mean=float(reward_tensor.mean().item()),
        reward_component_means={
            name: mean_tensors(torch, values)
            for name, values in reward_component_values.items()
            if values
        },
        done_count=int(done_tensor.sum().item()),
        timeout_count=int(torch.stack(truncated_flags).sum().item()),
        fallen_count=int(torch.stack(terminated_flags).sum().item()),
        reset_count=reset_count,
        height_bad_count=sum_bool_tensors(torch, height_bad_flags),
        termination_height_bad_count=sum_bool_tensors(torch, termination_height_bad_flags),
        tilt_bad_count=sum_bool_tensors(torch, tilt_bad_flags),
        height_reset_count=sum_bool_tensors(torch, termination_height_bad_flags),
        tilt_reset_count=sum_bool_tensors(torch, tilt_bad_flags),
        full_env_reset_wave=full_env_reset_wave_count > 0,
        full_env_reset_wave_count=full_env_reset_wave_count,
        episode_length_mean=mean_tensors(torch, episode_lengths),
        episode_length_min=min_tensors(torch, episode_lengths),
        episode_length_max=max_tensors(torch, episode_lengths),
        completed_episode_length_mean=mean_cat_tensors(torch, completed_episode_lengths),
        completed_episode_count=sum_tensor_lengths(completed_episode_lengths),
        root_height_mean=mean_tensors(torch, root_height_values),
        root_height_min=min_tensors(torch, root_height_values),
        upright_mean=mean_tensors(torch, upright_values),
        terminated=terminated_tensor,
        truncated=truncated_tensor,
        terminal_values=terminal_values,
    )


def compute_gae(batch: RolloutBatch, config: PPOConfig) -> tuple[Any, Any]:
    torch = require_torch()
    advantages = torch.zeros_like(batch.rewards)
    last_advantage = torch.zeros_like(batch.next_value)
    terminated = _rollout_mask_or_default(
        torch,
        batch.terminated,
        default=batch.dones,
        like=batch.dones,
    )
    truncated = _rollout_mask_or_default(
        torch,
        batch.truncated,
        default=torch.zeros_like(batch.dones, dtype=torch.bool),
        like=batch.dones,
    )
    terminal_values = batch.terminal_values
    if terminal_values is None:
        terminal_values = torch.zeros_like(batch.values)
    for step in reversed(range(config.rollout_steps)):
        next_value = batch.next_value if step == config.rollout_steps - 1 else batch.values[step + 1]
        bootstrap_value = torch.where(truncated[step], terminal_values[step], next_value)
        not_terminated = 1.0 - terminated[step].float()
        continue_mask = 1.0 - batch.dones[step].float()
        delta = batch.rewards[step] + config.gamma * bootstrap_value * not_terminated - batch.values[step]
        last_advantage = delta + config.gamma * config.gae_lambda * continue_mask * last_advantage
        advantages[step] = last_advantage
    returns = advantages + batch.values
    assert_finite_tensor(advantages, "advantages")
    assert_finite_tensor(returns, "returns")
    return advantages, returns


def _rollout_mask_or_default(torch: Any, value: Any | None, *, default: Any, like: Any) -> Any:
    if value is None:
        value = default
    if hasattr(value, "to"):
        return value.to(device=like.device, dtype=torch.bool)
    return torch.as_tensor(value, device=like.device, dtype=torch.bool)


def ppo_update(
    model: Any,
    optimizer: Any,
    batch: RolloutBatch,
    advantages: Any,
    returns: Any,
    config: PPOConfig,
) -> PPODiagnostics:
    torch = require_torch()
    flat_observations = batch.observations.reshape(-1, config.obs_dim)
    flat_actions = batch.actions.reshape(-1, config.action_dim)
    flat_old_log_probs = batch.log_probs.reshape(-1)
    flat_advantages = advantages.reshape(-1)
    flat_returns = returns.reshape(-1)
    flat_advantages = (flat_advantages - flat_advantages.mean()) / (
        flat_advantages.std(unbiased=False) + 1e-8
    )
    device = flat_observations.device
    synchronize_device(device)
    started = time.perf_counter()
    batch_size = flat_observations.shape[0]
    metric_sums = {
        "policy_loss": torch.zeros((), device=device),
        "value_loss": torch.zeros((), device=device),
        "entropy": torch.zeros((), device=device),
        "approx_kl": torch.zeros((), device=device),
        "clip_fraction": torch.zeros((), device=device),
        "grad_norm": torch.zeros((), device=device),
    }
    finite_loss_flags: list[Any] = []
    minibatches = 0
    for _epoch in range(config.epochs):
        indices = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, config.minibatch_size):
            minibatch = indices[start : start + config.minibatch_size]
            new_log_prob, entropy, value = model.evaluate_actions(
                flat_observations[minibatch],
                flat_actions[minibatch],
            )
            old_log_prob = flat_old_log_probs[minibatch]
            log_ratio = new_log_prob - old_log_prob
            ratio = log_ratio.exp()
            advantage = flat_advantages[minibatch]
            policy_loss = -torch.min(
                ratio * advantage,
                ratio.clamp(1.0 - config.clip, 1.0 + config.clip) * advantage,
            ).mean()
            value_loss = 0.5 * (value - flat_returns[minibatch]).square().mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean
            finite_loss_flags.append(torch.isfinite(loss.detach()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > config.clip).float().mean()
                metric_sums["policy_loss"] += policy_loss.detach()
                metric_sums["value_loss"] += value_loss.detach()
                metric_sums["entropy"] += entropy_mean.detach()
                metric_sums["approx_kl"] += approx_kl.detach()
                metric_sums["clip_fraction"] += clip_fraction.detach()
                metric_sums["grad_norm"] += grad_norm.detach()
            minibatches += 1
    assert_finite_flags(torch.stack(finite_loss_flags), "ppo_loss")
    synchronize_device(device)
    update_time_s = time.perf_counter() - started
    scale = 1.0 / max(1, minibatches)
    samples = batch_size * config.epochs
    return PPODiagnostics(
        policy_loss=float((metric_sums["policy_loss"] * scale).item()),
        value_loss=float((metric_sums["value_loss"] * scale).item()),
        entropy=float((metric_sums["entropy"] * scale).item()),
        approx_kl=float((metric_sums["approx_kl"] * scale).item()),
        clip_fraction=float((metric_sums["clip_fraction"] * scale).item()),
        grad_norm=float((metric_sums["grad_norm"] * scale).item()),
        update_time_s=update_time_s,
        update_samples_per_sec=samples / update_time_s if update_time_s > 0.0 else 0.0,
    )


def parameter_l1_sum(module: Any) -> float:
    total = 0.0
    for parameter in module.parameters():
        total += float(parameter.detach().abs().sum().item())
    return total


def assert_finite_tensor(value: Any, label: str) -> None:
    torch = require_torch()
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains NaN or Inf")


def assert_finite_flags(flags: Any, label: str) -> None:
    if not flags.all():
        raise ValueError(f"{label} contains NaN or Inf")


def synchronize_device(device: Any) -> None:
    if device is None:
        return
    if getattr(device, "type", None) != "cuda":
        return
    torch = require_torch()
    torch.cuda.synchronize(device)


def sum_bool_tensors(torch: Any, values: list[Any]) -> int:
    if not values:
        return 0
    return int(torch.stack(values).sum().item())


def mean_tensors(torch: Any, values: list[Any]) -> float:
    if not values:
        return 0.0
    return float(torch.stack(values).float().mean().item())


def min_tensors(torch: Any, values: list[Any]) -> float:
    if not values:
        return 0.0
    return float(torch.stack(values).float().min().item())


def max_tensors(torch: Any, values: list[Any]) -> float:
    if not values:
        return 0.0
    return float(torch.stack(values).float().max().item())


def mean_cat_tensors(torch: Any, values: list[Any]) -> float:
    if not values:
        return 0.0
    return float(torch.cat(values).float().mean().item())


def sum_tensor_lengths(values: list[Any]) -> int:
    return sum(tensor_length(value) for value in values)


def tensor_length(value: Any) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    return len(value)


def tensor_device_ok(values: dict[str, Any], expected: str) -> bool:
    return all(str(getattr(value, "device", "")) == expected for value in values.values())
