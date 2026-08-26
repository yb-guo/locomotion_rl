"""Task-independent generalized-advantage and clipped-PPO update kernels."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol


class PPORollout(Protocol):
    """Only rollout fields used by PPO; task diagnostics are intentionally absent."""

    observations: Any
    actions: Any
    rewards: Any
    dones: Any
    values: Any
    log_probs: Any
    next_value: Any


class PPOSettings(Protocol):
    """Structural settings accepted from any experiment config object."""

    epochs: int
    minibatch_size: int
    gamma: float
    gae_lambda: float
    clip: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float


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


def compute_gae(batch: PPORollout, config: PPOSettings) -> tuple[Any, Any]:
    """Compute GAE from generic rollout tensors, independent of any task."""

    torch = _require_torch()
    advantages = torch.zeros_like(batch.rewards)
    last_advantage = torch.zeros_like(batch.next_value)
    rollout_steps = int(batch.rewards.shape[0])
    for step in reversed(range(rollout_steps)):
        next_value = batch.next_value if step == rollout_steps - 1 else batch.values[step + 1]
        next_not_done = 1.0 - batch.dones[step].float()
        delta = batch.rewards[step] + config.gamma * next_value * next_not_done - batch.values[step]
        last_advantage = delta + config.gamma * config.gae_lambda * next_not_done * last_advantage
        advantages[step] = last_advantage
    returns = advantages + batch.values
    _assert_finite_tensor(advantages, "advantages")
    _assert_finite_tensor(returns, "returns")
    return advantages, returns


def ppo_update(
    model: Any,
    optimizer: Any,
    batch: PPORollout,
    advantages: Any,
    returns: Any,
    config: PPOSettings,
) -> PPODiagnostics:
    """Apply one clipped-PPO update without inspecting task names or metrics."""

    torch = _require_torch()
    flat_observations = batch.observations.reshape(-1, batch.observations.shape[-1])
    flat_actions = batch.actions.reshape(-1, batch.actions.shape[-1])
    flat_old_log_probs = batch.log_probs.reshape(-1)
    flat_advantages = advantages.reshape(-1)
    flat_returns = returns.reshape(-1)
    flat_advantages = (flat_advantages - flat_advantages.mean()) / (
        flat_advantages.std(unbiased=False) + 1e-8
    )
    device = flat_observations.device
    _synchronize_device(device)
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
            evaluate_kwargs = {}
            active_mask = getattr(batch, "active_action_mask", None)
            if active_mask is not None:
                evaluate_kwargs["active_action_mask"] = active_mask.reshape(-1, active_mask.shape[-1])[minibatch]
            try:
                new_log_prob, entropy, value = model.evaluate_actions(
                    flat_observations[minibatch],
                    flat_actions[minibatch],
                    **evaluate_kwargs,
                )
            except TypeError:
                if not evaluate_kwargs:
                    raise
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
    _assert_finite_flags(torch.stack(finite_loss_flags), "ppo_loss")
    _synchronize_device(device)
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


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - local no-torch path.
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch


def _assert_finite_tensor(value: Any, label: str) -> None:
    torch = _require_torch()
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains NaN or Inf")


def _assert_finite_flags(flags: Any, label: str) -> None:
    if not flags.all():
        raise ValueError(f"{label} contains NaN or Inf")


def _synchronize_device(device: Any) -> None:
    if device is None or getattr(device, "type", None) != "cuda":
        return
    torch = _require_torch()
    torch.cuda.synchronize(device)
