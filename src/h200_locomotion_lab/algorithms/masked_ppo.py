"""Active-slot action distribution helpers for whole-body PPO.

The policy still owns a 45-dimensional head.  A topology binding supplies a
boolean mask; inactive dimensions are forced to zero and contribute neither
likelihood nor entropy.  This keeps the PPO objective invariant to how many
actuators a particular morphology happens to expose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MaskedPPOLoss:
    policy_loss: Any
    entropy: Any
    approx_kl: Any
    clip_fraction: Any
    ratio: Any


def mask_action(action: Any, active_mask: Any) -> Any:
    """Force inactive action slots to exactly zero."""

    return action * _mask_like(active_mask, action)


def masked_log_prob(log_prob_per_dim: Any, active_mask: Any) -> Any:
    """Sum per-dimension log probabilities over active slots only."""

    mask = _mask_like(active_mask, log_prob_per_dim)
    return (log_prob_per_dim * mask).sum(dim=-1)


def masked_entropy(entropy_per_dim: Any, active_mask: Any) -> Any:
    """Sum per-dimension entropy over active slots only."""

    mask = _mask_like(active_mask, entropy_per_dim)
    return (entropy_per_dim * mask).sum(dim=-1)


def sample_masked_tanh_gaussian(
    mean: Any,
    log_std: Any,
    active_mask: Any,
    *,
    deterministic: bool = False,
    tanh_eps: float = 1e-6,
) -> tuple[Any, Any, Any]:
    """Sample a tanh-Gaussian action and return masked action/log-prob/entropy."""

    torch = _require_torch()
    if mean.shape != log_std.shape:
        raise ValueError("mean and log_std must have equal shape")
    normal = torch.distributions.Normal(mean, log_std.exp())
    raw = mean if deterministic else normal.rsample()
    action = torch.tanh(raw)
    clipped = action.clamp(-1.0 + tanh_eps, 1.0 - tanh_eps)
    per_dim_log_prob = normal.log_prob(raw) - torch.log(1.0 - clipped.square() + tanh_eps)
    per_dim_entropy = normal.entropy()
    mask = _mask_like(active_mask, action)
    return (
        action * mask,
        masked_log_prob(per_dim_log_prob, mask),
        masked_entropy(per_dim_entropy, mask),
    )


def masked_tanh_gaussian_log_prob(
    action: Any,
    mean: Any,
    log_std: Any,
    active_mask: Any,
    *,
    tanh_eps: float = 1e-6,
) -> Any:
    """Evaluate the likelihood of a masked action using active slots only."""

    torch = _require_torch()
    clipped = action.clamp(-1.0 + tanh_eps, 1.0 - tanh_eps)
    raw = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))
    normal = torch.distributions.Normal(mean, log_std.exp())
    per_dim = normal.log_prob(raw) - torch.log(1.0 - clipped.square() + tanh_eps)
    return masked_log_prob(per_dim, active_mask)


def masked_ppo_surrogate(
    new_log_prob_per_dim: Any,
    old_log_prob_per_dim: Any,
    advantages: Any,
    active_mask: Any,
    *,
    clip: float = 0.2,
) -> MaskedPPOLoss:
    """Compute clipped PPO terms after applying the slot mask."""

    new_log_prob = _collapse_log_prob(new_log_prob_per_dim, active_mask)
    old_log_prob = _collapse_log_prob(old_log_prob_per_dim, active_mask)
    ratio = (new_log_prob - old_log_prob).exp()
    clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
    policy_loss = -torch_min(ratio * advantages, clipped_ratio * advantages).mean()
    approx_kl = ((ratio - 1.0) - (new_log_prob - old_log_prob)).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip).float().mean()
    return MaskedPPOLoss(
        policy_loss=policy_loss,
        entropy=None,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        ratio=ratio,
    )


def _collapse_log_prob(value: Any, active_mask: Any) -> Any:
    if value.ndim >= 1 and active_mask.ndim >= 1 and value.shape[-1] == active_mask.shape[-1]:
        return masked_log_prob(value, active_mask)
    return value


def _mask_like(mask: Any, value: Any) -> Any:
    if tuple(mask.shape) != tuple(value.shape):
        if mask.ndim == 1 and value.ndim >= 2 and value.shape[-1] == mask.shape[0]:
            mask = mask.expand(*value.shape[:-1], mask.shape[0])
        else:
            raise ValueError(f"active mask shape {tuple(mask.shape)} does not match {tuple(value.shape)}")
    return mask.to(device=value.device, dtype=value.dtype)


def torch_min(left: Any, right: Any) -> Any:
    return _require_torch().minimum(left, right)


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch
