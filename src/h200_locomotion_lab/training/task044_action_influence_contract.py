"""Pure Task044 action-influence diagnostic contract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Task044ActionInfluenceThresholds:
    min_mean_abs_l1_delta: float = 0.01
    min_mean_l2_delta: float = 0.01
    metric_scope: str = "final_trial_window"


def evaluate_task044_action_influence_triplet(
    normal: Mapping[str, Any],
    zero_residual: Mapping[str, Any],
    stateless_memory: Mapping[str, Any],
    thresholds: Task044ActionInfluenceThresholds | None = None,
) -> dict[str, Any]:
    """Compare action statistics across memory-ablation eval JSONs.

    This is diagnostic-only. It can show that memory ablations changed the
    actor output distribution, but it cannot prove useful memory causality.
    """

    thresholds = thresholds or Task044ActionInfluenceThresholds()
    reasons: list[str] = []
    normal_stats = _action_stats(normal, "normal", thresholds.metric_scope, reasons)
    zero_stats = _action_stats(
        zero_residual,
        "zero_residual",
        thresholds.metric_scope,
        reasons,
    )
    stateless_stats = _action_stats(
        stateless_memory,
        "stateless_memory",
        thresholds.metric_scope,
        reasons,
    )
    if reasons:
        return _result(False, thresholds, reasons, {}, {}, {})

    zero_delta = _action_delta(normal_stats, zero_stats)
    stateless_delta = _action_delta(normal_stats, stateless_stats)
    zero_changed = _delta_changed(zero_delta, thresholds)
    stateless_changed = _delta_changed(stateless_delta, thresholds)
    if not zero_changed:
        reasons.append("zero_residual_action_stats_tied")
    if not stateless_changed:
        reasons.append("stateless_memory_action_stats_tied")
    influence_detected = zero_changed or stateless_changed
    return _result(
        influence_detected,
        thresholds,
        reasons,
        zero_delta,
        stateless_delta,
        {
            "normal": normal_stats,
            "zero_residual": zero_stats,
            "stateless_memory": stateless_stats,
        },
    )


def _result(
    influence_detected: bool,
    thresholds: Task044ActionInfluenceThresholds,
    reasons: list[str],
    zero_delta: Mapping[str, Any],
    stateless_delta: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "task044_action_influence_contract": True,
        "action_influence_detected": influence_detected,
        "diagnostic_pass": influence_detected,
        "memory_causality_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "thresholds": {
            "min_mean_abs_l1_delta": thresholds.min_mean_abs_l1_delta,
            "min_mean_l2_delta": thresholds.min_mean_l2_delta,
            "metric_scope": thresholds.metric_scope,
        },
        "zero_residual_delta": dict(zero_delta),
        "stateless_memory_delta": dict(stateless_delta),
        "action_stats": dict(stats),
        "failure_reasons": list(reasons),
    }


def _action_stats(
    summary: Mapping[str, Any],
    label: str,
    metric_scope: str,
    reasons: list[str],
) -> dict[str, Any]:
    container: Any
    if metric_scope == "root":
        container = summary
    else:
        container = summary.get(metric_scope)
    if not isinstance(container, Mapping):
        reasons.append(f"{label}_{metric_scope}_missing")
        return {}
    stats = container.get("action_stats")
    if not isinstance(stats, Mapping):
        reasons.append(f"{label}_{metric_scope}_action_stats_missing")
        return {}

    sample_count = _finite_number(stats.get("sample_count"))
    mean_l2 = _finite_number(stats.get("mean_l2"))
    max_abs = _finite_number(stats.get("max_abs"))
    mean_abs_by_dim = _number_list(stats.get("mean_abs_by_dim"))
    action_dim = _finite_number(stats.get("action_dim"))
    if sample_count is None or sample_count <= 0:
        reasons.append(f"{label}_action_sample_count_missing")
    if mean_l2 is None:
        reasons.append(f"{label}_action_mean_l2_missing")
    if max_abs is None:
        reasons.append(f"{label}_action_max_abs_missing")
    if not mean_abs_by_dim:
        reasons.append(f"{label}_action_mean_abs_by_dim_missing")
    if action_dim is not None and mean_abs_by_dim and int(action_dim) != len(mean_abs_by_dim):
        reasons.append(f"{label}_action_dim_mismatch")

    return {
        "sample_count": int(sample_count) if sample_count is not None else None,
        "action_dim": int(action_dim) if action_dim is not None else len(mean_abs_by_dim),
        "mean_l2": mean_l2,
        "max_abs": max_abs,
        "mean_abs_by_dim": mean_abs_by_dim,
        "top_abs_dims": list(stats.get("top_abs_dims", [])),
    }


def _action_delta(normal: Mapping[str, Any], ablation: Mapping[str, Any]) -> dict[str, Any]:
    normal_mean_abs = normal["mean_abs_by_dim"]
    ablation_mean_abs = ablation["mean_abs_by_dim"]
    if len(normal_mean_abs) != len(ablation_mean_abs):
        raise ValueError("action stat dimensions must match")
    per_dim_delta = [
        float(ablation_value - normal_value)
        for normal_value, ablation_value in zip(normal_mean_abs, ablation_mean_abs)
    ]
    per_dim_abs_delta = [abs(value) for value in per_dim_delta]
    max_delta = max(per_dim_abs_delta) if per_dim_abs_delta else 0.0
    max_dim = per_dim_abs_delta.index(max_delta) if per_dim_abs_delta else None
    return {
        "mean_l2_delta": float(ablation["mean_l2"] - normal["mean_l2"]),
        "mean_abs_l1_delta": float(sum(per_dim_abs_delta) / max(len(per_dim_abs_delta), 1)),
        "max_mean_abs_dim_delta": float(max_delta),
        "max_mean_abs_delta_dim": max_dim,
        "per_dim_mean_abs_delta": per_dim_delta,
    }


def _delta_changed(
    delta: Mapping[str, Any],
    thresholds: Task044ActionInfluenceThresholds,
) -> bool:
    return (
        abs(float(delta["mean_l2_delta"])) >= thresholds.min_mean_l2_delta
        or float(delta["mean_abs_l1_delta"]) >= thresholds.min_mean_abs_l1_delta
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _number_list(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    numbers: list[float] = []
    for item in value:
        number = _finite_number(item)
        if number is None:
            return []
        numbers.append(number)
    return numbers
