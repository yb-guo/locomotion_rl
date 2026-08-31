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
    raw_actions: Any
    old_means: Any
    old_log_stds: Any


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
    target_kl: float | None
    hard_kl_stop: bool
    adaptive_kl: bool
    desired_kl: float | None


@dataclass(frozen=True, slots=True)
class PPOMinibatchDiagnostics:
    epoch: int
    index: int
    approx_kl: float
    clip_fraction: float
    applied: bool
    scheduler_kl: float | None = None
    learning_rate_before: float | None = None
    learning_rate_after: float | None = None
    scheduler_decision: str = "disabled"
    same_policy_identity_error: float | None = None


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
    early_stopped: bool
    minibatches_attempted: int
    minibatches_completed: int
    epochs_completed: int
    learning_rate: float
    minibatches: tuple[PPOMinibatchDiagnostics, ...]
    scheduler_decision: str
    learning_rate_before: float
    learning_rate_after: float
    scheduler_kl: float | None
    desired_kl: float | None


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
    using_raw_actions = getattr(batch, "raw_actions", None) is not None
    flat_actions = getattr(batch, "raw_actions", batch.actions).reshape(-1, batch.actions.shape[-1])
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
    active_mask_flat = getattr(batch, "active_action_mask", None)
    if active_mask_flat is None:
        active_mask_flat = getattr(model, "action_mask", None)
    if active_mask_flat is not None:
        active_mask_flat = active_mask_flat.reshape(-1, flat_actions.shape[-1])
        if active_mask_flat.shape[0] == 1:
            active_mask_flat = active_mask_flat.expand(batch_size, -1)
    learning_rate_before = float(optimizer.param_groups[0]["lr"])
    scheduler_kl: float | None = None
    scheduler_decision = "disabled"
    saved_old_params = getattr(batch, "old_means", None) is not None and getattr(batch, "old_log_stds", None) is not None
    desired_kl = getattr(config, "desired_kl", None)
    if desired_kl is not None:
        desired_kl = float(desired_kl)
    if getattr(config, "adaptive_kl", False):
        if desired_kl is None or desired_kl <= 0.0:
            raise ValueError("adaptive KL requires positive desired_kl")
        if not using_raw_actions or not saved_old_params:
            raise ValueError("adaptive KL requires rollout raw_actions and old distribution params")
        with torch.no_grad():
            old_mean = getattr(batch, "old_means", None)
            old_log_std = getattr(batch, "old_log_stds", None)
            old_mean = old_mean.reshape(-1, flat_actions.shape[-1])
            old_log_std = old_log_std.reshape(-1, flat_actions.shape[-1])
            old_mean = old_mean.detach().clone()
            old_log_std = old_log_std.detach().clone()
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
    minibatch_records: list[PPOMinibatchDiagnostics] = []
    early_stopped = False
    epochs_completed = 0
    for _epoch in range(config.epochs):
        indices = torch.randperm(batch_size, device=device)
        for minibatch_index, start in enumerate(range(0, batch_size, config.minibatch_size)):
            minibatch = indices[start : start + config.minibatch_size]
            evaluate_kwargs = {}
            active_mask = getattr(batch, "active_action_mask", None)
            if active_mask is not None:
                evaluate_kwargs["active_action_mask"] = active_mask.reshape(-1, active_mask.shape[-1])[minibatch]
            if using_raw_actions and not hasattr(model, "evaluate_raw_actions"):
                raise ValueError("raw PPO action replay requires model.evaluate_raw_actions")
            evaluator = model.evaluate_raw_actions if using_raw_actions else model.evaluate_actions
            try:
                new_log_prob, entropy, value = evaluator(
                    flat_observations[minibatch],
                    flat_actions[minibatch],
                    **evaluate_kwargs,
                )
            except TypeError as error:
                if using_raw_actions:
                    raise ValueError("raw PPO action replay requires masked evaluate_raw_actions support") from error
                if not evaluate_kwargs:
                    raise
                new_log_prob, entropy, value = model.evaluate_actions(
                    flat_observations[minibatch],
                    flat_actions[minibatch],
                )
            old_log_prob = flat_old_log_probs[minibatch]
            log_ratio = new_log_prob - old_log_prob
            ratio = log_ratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > config.clip).float().mean()
                mb_scheduler_kl = mb_lr_before = mb_lr_after = identity_error = None
                mb_decision = "disabled"
                if getattr(config, "adaptive_kl", False) and saved_old_params:
                    current_mean, _ = _policy_mean_value(model, flat_observations[minibatch])
                    current_log_std = model.log_std.expand_as(current_mean)
                    mb_scheduler_kl = float(_joint_gaussian_kl(
                        old_mean[minibatch], old_log_std[minibatch], current_mean, current_log_std,
                        None if active_mask_flat is None else active_mask_flat[minibatch]).mean().item())
                    identity_log_prob = _raw_tanh_gaussian_log_prob(
                        flat_actions[minibatch],
                        old_mean[minibatch],
                        old_log_std[minibatch],
                        None if active_mask_flat is None else active_mask_flat[minibatch],
                        tanh_eps=float(getattr(getattr(model, "config", None), "tanh_eps", 1e-6)),
                    )
                    identity_error = float((identity_log_prob - old_log_prob).abs().max().item())
                    mb_lr_before = float(optimizer.param_groups[0]["lr"])
                    factor, mb_decision = _adaptive_lr_factor(mb_scheduler_kl, desired_kl)
                    mb_lr_after = min(1e-2, max(1e-5, mb_lr_before * factor))
                    for group in optimizer.param_groups:
                        group["lr"] = min(1e-2, max(1e-5, float(group["lr"]) * factor))
                target_kl = getattr(config, "target_kl", None)
                should_stop = (
                    getattr(config, "hard_kl_stop", True)
                    and target_kl is not None
                    and minibatches > 0
                    and float(approx_kl.item()) > 1.5 * target_kl
                )
                minibatch_records.append(PPOMinibatchDiagnostics(
                    epoch=_epoch,
                    index=minibatch_index,
                    approx_kl=float(approx_kl.item()),
                    clip_fraction=float(clip_fraction.item()),
                    applied=not should_stop,
                    scheduler_kl=mb_scheduler_kl,
                    learning_rate_before=mb_lr_before,
                    learning_rate_after=mb_lr_after,
                    scheduler_decision=mb_decision,
                    same_policy_identity_error=identity_error,
                ))
            if should_stop:
                early_stopped = True
                break
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
            # clip_grad_norm_ returns the total norm before clipping.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                metric_sums["policy_loss"] += policy_loss.detach()
                metric_sums["value_loss"] += value_loss.detach()
                metric_sums["entropy"] += entropy_mean.detach()
                metric_sums["approx_kl"] += approx_kl.detach()
                metric_sums["clip_fraction"] += clip_fraction.detach()
                metric_sums["grad_norm"] += grad_norm.detach()
            minibatches += 1
        if early_stopped:
            break
        epochs_completed += 1
    _assert_finite_flags(torch.stack(finite_loss_flags), "ppo_loss")
    if getattr(config, "adaptive_kl", False) and minibatch_records:
        scheduler_kl = minibatch_records[-1].scheduler_kl
        scheduler_decision = minibatch_records[-1].scheduler_decision
    learning_rate_after = float(optimizer.param_groups[0]["lr"])
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
        early_stopped=early_stopped,
        minibatches_attempted=len(minibatch_records),
        minibatches_completed=minibatches,
        epochs_completed=epochs_completed,
        learning_rate=float(optimizer.param_groups[0]["lr"]),
        minibatches=tuple(minibatch_records),
        scheduler_decision=scheduler_decision,
        learning_rate_before=learning_rate_before,
        learning_rate_after=learning_rate_after,
        scheduler_kl=scheduler_kl,
        desired_kl=desired_kl,
    )


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - local no-torch path.
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch


def _policy_mean_value(model: Any, observations: Any) -> tuple[Any, Any]:
    forward = getattr(model, "forward", None)
    if forward is not None:
        return forward(observations)
    return model(observations)


def _masked_tensor_mean(value: Any, mask: Any | None) -> Any:
    if mask is None:
        return value.mean()
    typed_mask = mask.to(device=value.device, dtype=value.dtype)
    return (value * typed_mask).sum().div(typed_mask.sum().clamp_min(1.0))


def _raw_tanh_gaussian_log_prob(
    raw: Any,
    mean: Any,
    log_std: Any,
    active_mask: Any | None,
    *,
    tanh_eps: float,
) -> Any:
    torch = _require_torch()
    normal = torch.distributions.Normal(mean, log_std.exp())
    action = torch.tanh(raw).clamp(-1.0 + tanh_eps, 1.0 - tanh_eps)
    per_dim = normal.log_prob(raw) - torch.log(1.0 - action.square() + tanh_eps)
    if active_mask is not None:
        per_dim = per_dim * active_mask.to(device=per_dim.device, dtype=per_dim.dtype)
    return per_dim.sum(dim=-1)


def _joint_gaussian_kl(old_mean: Any, old_log_std: Any, new_mean: Any, new_log_std: Any, active_mask: Any | None) -> Any:
    variance_ratio = ((old_log_std - new_log_std) * 2.0).exp()
    mean_delta = (old_mean - new_mean).square() / new_log_std.mul(2.0).exp()
    per_dim = 0.5 * (variance_ratio + mean_delta - 1.0 + 2.0 * (new_log_std - old_log_std))
    if active_mask is not None:
        per_dim = per_dim * active_mask.to(device=per_dim.device, dtype=per_dim.dtype)
    return per_dim.sum(dim=-1)


def _adaptive_lr_factor(kl: float, desired_kl: float | None) -> tuple[float, str]:
    if desired_kl is not None and kl > 2.0 * desired_kl:
        return 1.0 / 1.5, "decrease"
    if desired_kl is not None and 0.0 < kl < 0.5 * desired_kl:
        return 1.5, "increase"
    return 1.0, "hold"


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
