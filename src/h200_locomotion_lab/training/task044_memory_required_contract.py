"""Pure Task044 memory-required triplet contract.

The contract intentionally has no simulator dependency. It evaluates the three
JSON summaries produced by normal, zero-residual, and stateless-memory evals.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


ZERO_RESIDUAL_MODES = frozenset({"zero_txl_residual", "zero_residual", "zero_memory_latent"})
STATELESS_MEMORY_MODES = frozenset({"stateless_txl_memory", "stateless_memory"})


@dataclass(frozen=True, slots=True)
class Task044TripletThresholds:
    min_ablation_lin_vel_error_delta: float = 0.03
    min_ablation_fall_ratio_delta: float = 0.10
    min_ablation_completion_ratio_drop: float = 0.10
    metric_scope: str = "final_trial"


def evaluate_task044_memory_required_triplet(
    normal: Mapping[str, Any],
    zero_residual: Mapping[str, Any],
    stateless_memory: Mapping[str, Any],
    thresholds: Task044TripletThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate whether an eval triplet proves memory-required behavior."""

    thresholds = thresholds or Task044TripletThresholds()
    reasons: list[str] = []

    hidden_contract = evaluate_hidden_fault_observation_contract(normal)
    reasons.extend(hidden_contract["failure_reasons"])

    normal_contract = _evaluate_normal_record(normal)
    reasons.extend(normal_contract["failure_reasons"])

    zero_record = _evaluate_ablation_record(
        zero_residual,
        ZERO_RESIDUAL_MODES,
        "zero_residual",
    )
    stateless_record = _evaluate_ablation_record(
        stateless_memory,
        STATELESS_MEMORY_MODES,
        "stateless_memory",
    )
    reasons.extend(zero_record["failure_reasons"])
    reasons.extend(stateless_record["failure_reasons"])

    zero_degradation = _evaluate_ablation_degradation(
        normal,
        zero_residual,
        thresholds,
        "zero_residual",
    )
    stateless_degradation = _evaluate_ablation_degradation(
        normal,
        stateless_memory,
        thresholds,
        "stateless_memory",
    )
    reasons.extend(zero_degradation["failure_reasons"])
    reasons.extend(stateless_degradation["failure_reasons"])

    pass_gate = not reasons
    return {
        "task044_memory_required_contract": True,
        "task044_memory_required_pass": pass_gate,
        "hidden_fault_contract": hidden_contract,
        "normal_contract": normal_contract,
        "zero_residual_ablation": {
            **zero_record,
            "degradation": zero_degradation,
        },
        "stateless_memory_ablation": {
            **stateless_record,
            "degradation": stateless_degradation,
        },
        "thresholds": {
            "min_ablation_lin_vel_error_delta": thresholds.min_ablation_lin_vel_error_delta,
            "min_ablation_fall_ratio_delta": thresholds.min_ablation_fall_ratio_delta,
            "min_ablation_completion_ratio_drop": thresholds.min_ablation_completion_ratio_drop,
            "metric_scope": thresholds.metric_scope,
        },
        "memory_required_evidence_pass": pass_gate,
        "memory_causality_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "failure_reasons": reasons,
    }


def evaluate_hidden_fault_observation_contract(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Require fault labels to be hidden from actor observations."""

    keys = (
        "fault_identity_in_actor_obs",
        "fault_severity_in_actor_obs",
        "fault_onset_in_actor_obs",
        "fault_recovery_in_actor_obs",
        "fault_schedule_in_actor_obs",
    )
    meta = summary.get("task044_hidden_fault_contract")
    if not isinstance(meta, Mapping):
        meta = summary

    reasons: list[str] = []
    checked: dict[str, Any] = {}
    for key in keys:
        value = meta.get(key)
        checked[key] = value
        if value is not False:
            reasons.append(f"{key}_not_false")

    return {
        "pass": not reasons,
        "checked": checked,
        "failure_reasons": reasons,
    }


def _evaluate_normal_record(summary: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if summary.get("pipeline_pass") is not True:
        reasons.append("normal_pipeline_not_passed")
    if summary.get("quality_gate_pass") is not True:
        reasons.append("normal_quality_gate_not_passed")
    if not _memory_debug_active(summary):
        reasons.append("normal_memory_debug_not_active")
    if _ablation_mode(summary) not in {"", "none"}:
        reasons.append("normal_ablation_mode_not_none")
    return {
        "pass": not reasons,
        "memory_debug_active": _memory_debug_active(summary),
        "ablation_mode": _ablation_mode(summary) or "none",
        "failure_reasons": reasons,
    }


def _evaluate_ablation_record(
    summary: Mapping[str, Any],
    expected_modes: frozenset[str],
    label: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    mode = _ablation_mode(summary)
    if mode not in expected_modes:
        reasons.append(f"{label}_ablation_mode_mismatch")
    if summary.get("memory_ablation_mode_match") is False:
        reasons.append(f"{label}_ablation_mode_match_false")
    return {
        "recorded": not reasons,
        "ablation_mode": mode,
        "failure_reasons": reasons,
    }


def _evaluate_ablation_degradation(
    normal: Mapping[str, Any],
    ablation: Mapping[str, Any],
    thresholds: Task044TripletThresholds,
    label: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    normal_metrics = _final_trial_metrics(normal, "normal", reasons, thresholds.metric_scope)
    ablation_metrics = _final_trial_metrics(ablation, label, reasons, thresholds.metric_scope)
    if reasons:
        return {
            "degraded": False,
            "deltas": {},
            "failure_reasons": reasons,
        }

    lin_vel_error_delta = (
        ablation_metrics["lin_vel_error_mean"] - normal_metrics["lin_vel_error_mean"]
    )
    fall_ratio_delta = ablation_metrics["fall_ratio"] - normal_metrics["fall_ratio"]
    completion_ratio_drop = (
        normal_metrics["completion_ratio"] - ablation_metrics["completion_ratio"]
    )
    degraded = (
        lin_vel_error_delta >= thresholds.min_ablation_lin_vel_error_delta
        or fall_ratio_delta >= thresholds.min_ablation_fall_ratio_delta
        or completion_ratio_drop >= thresholds.min_ablation_completion_ratio_drop
    )
    if not degraded:
        reasons.append(f"{label}_ablation_not_degraded")
    return {
        "degraded": degraded,
        "deltas": {
            "lin_vel_error_delta": lin_vel_error_delta,
            "fall_ratio_delta": fall_ratio_delta,
            "completion_ratio_drop": completion_ratio_drop,
        },
        "normal_metrics": normal_metrics,
        "ablation_metrics": ablation_metrics,
        "failure_reasons": reasons,
    }


def _final_trial_metrics(
    summary: Mapping[str, Any],
    label: str,
    reasons: list[str],
    metric_scope: str,
) -> dict[str, float]:
    final_trial = summary.get(metric_scope)
    if not isinstance(final_trial, Mapping):
        reasons.append(f"{label}_{metric_scope}_missing")
        return {}

    metrics = {
        "completion_ratio": _number_at(final_trial, ("completion_ratio",)),
        "fall_ratio": _number_at(final_trial, ("fall_ratio",)),
        "lin_vel_error_mean": _number_at(final_trial, ("lin_vel_error", "mean")),
    }
    for key, value in metrics.items():
        if value is None:
            reasons.append(f"{label}_{key}_missing_or_nonfinite")
    return {key: value for key, value in metrics.items() if value is not None}


def _number_at(source: Mapping[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = source
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if isinstance(current, bool) or current is None:
        return None
    if isinstance(current, int):
        return float(current)
    if isinstance(current, float) and math.isfinite(current):
        return current
    return None


def _ablation_mode(summary: Mapping[str, Any]) -> str:
    mode = summary.get("memory_ablation_mode")
    if mode is None:
        txl_debug = summary.get("txl_debug")
        if isinstance(txl_debug, Mapping):
            mode = txl_debug.get("task042_memory_ablation_mode")
    if mode is None:
        return ""
    return str(mode)


def _memory_debug_active(summary: Mapping[str, Any]) -> bool:
    if summary.get("memory_debug_active") is True:
        return True
    contract = summary.get("true_txl_memory_debug_contract")
    if isinstance(contract, Mapping) and contract.get("active") is True:
        return True
    txl_debug = summary.get("txl_debug")
    if not isinstance(txl_debug, Mapping):
        return False
    if txl_debug.get("stateful_memory_enabled") is False:
        return False
    previous_lengths = txl_debug.get("last_attended_previous_memory_lengths")
    if isinstance(previous_lengths, list) and any(_positive_number(value) for value in previous_lengths):
        return True
    return _positive_number(txl_debug.get("incremental_steps"))


def _positive_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return value > 0
    return False
