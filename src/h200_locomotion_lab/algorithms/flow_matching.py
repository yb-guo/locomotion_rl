"""RL adapters for MIP/JiT/flow-matching policies.

Flow matching is kept orthogonal to the task and 45D schema.  A policy with a
tractable action likelihood can use PPO; a deterministic or intractable flow
must use the advantage/Q-weighted regression fallback instead of inventing a
log-probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol


class FlowMatchingPolicy(Protocol):
    has_tractable_log_prob: bool

    def vector_field(self, state: Any, time: Any, condition: Any) -> Any: ...

    def log_prob(self, action: Any, condition: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class FlowMatchingRLConfig:
    temperature: float = 1.0
    clip_weight: float = 20.0
    use_q_values: bool = False

    def __post_init__(self) -> None:
        if self.temperature <= 0.0 or self.clip_weight <= 0.0:
            raise ValueError("temperature and clip_weight must be positive")


class FlowMatchingPPOAdapter:
    """Guarded adapter that only exposes PPO when likelihood is real."""

    def __init__(self, policy: FlowMatchingPolicy) -> None:
        self.policy = policy
        if not getattr(policy, "has_tractable_log_prob", False):
            raise ValueError(
                "flow policy has no tractable log_prob; use AdvantageWeightedFlowMatching"
            )

    def log_prob(self, action: Any, condition: Any) -> Any:
        value = self.policy.log_prob(action, condition)
        if not _is_finite(value):
            raise ValueError("flow policy returned a non-finite log_prob")
        return value


class AdvantageWeightedFlowMatching:
    """Advantage/Q-weighted vector-field regression for intractable flows."""

    def __init__(self, policy: FlowMatchingPolicy, config: FlowMatchingRLConfig | None = None) -> None:
        self.policy = policy
        self.config = config or FlowMatchingRLConfig()
        if getattr(policy, "has_tractable_log_prob", False):
            raise ValueError("use FlowMatchingPPOAdapter when log_prob is tractable")

    def loss(
        self,
        predicted_vector_field: Any,
        target_vector_field: Any,
        advantages: Any,
    ) -> Any:
        torch = _require_torch()
        weights = torch.exp(advantages / self.config.temperature).clamp(max=self.config.clip_weight)
        while weights.ndim < predicted_vector_field.ndim:
            weights = weights.unsqueeze(-1)
        squared_error = (predicted_vector_field - target_vector_field).square()
        return (weights * squared_error).mean()

    def training_mode(self) -> str:
        return "q_weighted_flow_matching" if self.config.use_q_values else "advantage_weighted_flow_matching"


def _is_finite(value: Any) -> bool:
    if hasattr(value, "isfinite"):
        return bool(value.isfinite().all().item())
    return all(math.isfinite(float(item)) for item in value)


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch
