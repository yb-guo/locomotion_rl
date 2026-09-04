"""Minimal PPO core for task014 smoke runs.

Torch is imported only inside torch-required functions so local no-torch pytest
can still import this module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from h200_locomotion_lab.algorithms.ppo import (
    PPODiagnostics,
    compute_gae,
    ppo_update,
)
from h200_locomotion_lab.policies.tanh_gaussian_actor_critic import (
    build_tanh_gaussian_actor_critic as build_actor_critic,
)
from h200_locomotion_lab.policies.tanh_gaussian_actor_critic import (
    gaussian_entropy,
    make_mlp,
    tanh_gaussian_log_prob_from_action,
    tanh_gaussian_log_prob_from_raw,
)
from h200_locomotion_lab.tasks.g1_velocity_tracking import G1_REWARD_COMPONENT_NAMES

__all__ = [
    "REWARD_COMPONENT_NAMES",
    "PPOConfig",
    "PPODiagnostics",
    "RolloutBatch",
    "assert_finite_flags",
    "assert_finite_tensor",
    "build_actor_critic",
    "collect_rollout",
    "compute_gae",
    "describe_training_plan",
    "gaussian_entropy",
    "make_mlp",
    "max_tensors",
    "mean_cat_tensors",
    "mean_tensors",
    "min_tensors",
    "parameter_l1_sum",
    "ppo_update",
    "require_torch",
    "sum_bool_tensors",
    "sum_tensor_lengths",
    "synchronize_device",
    "tanh_gaussian_log_prob_from_action",
    "tanh_gaussian_log_prob_from_raw",
    "tensor_device_ok",
    "tensor_length",
]

REWARD_COMPONENT_NAMES = G1_REWARD_COMPONENT_NAMES


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
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            truncated_flags.append(transition.truncated)
            terminated_flags.append(transition.terminated)
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
    synchronize_device(getattr(observation, "device", None))
    collect_time_s = time.perf_counter() - started
    observation_tensor = torch.stack(observations)
    action_tensor = torch.stack(actions)
    reward_tensor = torch.stack(rewards)
    done_tensor = torch.stack(dones)
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
