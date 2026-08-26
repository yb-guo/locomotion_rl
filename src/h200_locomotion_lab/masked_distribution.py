"""Optional-Torch masked action distribution primitives.

This neutral numerical helper is shared by policy implementations without
making the dependency-neutral ``core`` contracts import Torch or PPO.
"""

from __future__ import annotations

from typing import Any


def masked_log_prob(log_prob_per_dim: Any, active_mask: Any) -> Any:
    mask = _mask_like(active_mask, log_prob_per_dim)
    return (log_prob_per_dim * mask).sum(dim=-1)


def masked_entropy(entropy_per_dim: Any, active_mask: Any) -> Any:
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
    torch = _require_torch()
    normal = torch.distributions.Normal(mean, log_std.exp())
    raw = mean if deterministic else normal.rsample()
    action = torch.tanh(raw)
    clipped = action.clamp(-1.0 + tanh_eps, 1.0 - tanh_eps)
    per_dim_log_prob = normal.log_prob(raw) - torch.log(1.0 - clipped.square() + tanh_eps)
    return (
        action * _mask_like(active_mask, action),
        masked_log_prob(per_dim_log_prob, active_mask),
        masked_entropy(normal.entropy(), active_mask),
    )


def masked_tanh_gaussian_log_prob(
    action: Any,
    mean: Any,
    log_std: Any,
    active_mask: Any,
    *,
    tanh_eps: float = 1e-6,
) -> Any:
    torch = _require_torch()
    clipped = action.clamp(-1.0 + tanh_eps, 1.0 - tanh_eps)
    raw = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))
    normal = torch.distributions.Normal(mean, log_std.exp())
    per_dim = normal.log_prob(raw) - torch.log(1.0 - clipped.square() + tanh_eps)
    return masked_log_prob(per_dim, active_mask)


def _mask_like(mask: Any, value: Any) -> Any:
    if tuple(mask.shape) != tuple(value.shape):
        if mask.ndim == 1 and value.ndim >= 2 and value.shape[-1] == mask.shape[0]:
            mask = mask.expand(*value.shape[:-1], mask.shape[0])
        else:
            raise ValueError(f"active mask shape {tuple(mask.shape)} does not match {tuple(value.shape)}")
    return mask.to(device=value.device, dtype=value.dtype)


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch

