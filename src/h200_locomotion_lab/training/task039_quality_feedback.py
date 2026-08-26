"""Task039 local quality feedback gate.

This module intentionally has no simulator or training dependency. It validates
one eval-summary mapping with the metric shape produced by the Task037/Task038
multi-trial smoke tools, then returns diagnostic quality feedback.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

REQUIRED_FINAL_TRIAL_METRICS = (
    ("completion_ratio",),
    ("fall_ratio",),
    ("gravity_xy", "max"),
    ("root_z", "min"),
    ("lin_vel_error", "mean"),
    ("yaw_vel_error", "mean"),
)

REQUIRED_TRIAL0_TREND_METRICS = REQUIRED_FINAL_TRIAL_METRICS

REQUIRED_AGGREGATE_METRICS = (
    ("completion_ratio_per_trial_mean",),
    ("fall_ratio",),
    ("gravity_xy_max",),
    ("root_z_min",),
    ("lin_vel_error_mean",),
    ("yaw_vel_error_mean",),
)

NO_OVERCLAIM_FIELDS = (
    "quality_claim",
    "training_claim",
    "eval_claim",
    "reproduction_claim",
    "superiority_claim",
)


@dataclass(frozen=True, slots=True)
class QualityGate:
    min_final_completion_ratio: float = 0.95
    max_final_fall_ratio: float = 0.10
    max_final_gravity_xy: float = 0.75
    min_final_root_z: float = 0.55
    max_final_lin_vel_error: float = 0.45
    max_final_yaw_vel_error: float = 0.35
    require_trend_context: bool = True
    enforce_trial0_non_regression: bool = True
    trend_tolerance: float = 1.0e-3


@dataclass(frozen=True, slots=True)
class QualityFeedbackResult:
    pipeline_pass: bool
    quality_gate_pass: bool
    diagnostic_only: bool
    no_reproduction_claim: bool
    no_superiority_claim: bool
    no_training_success_claim: bool
    failure_reasons: tuple[str, ...] = ()
    checked_metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.quality_gate_pass

    def to_json(self) -> dict[str, Any]:
        return {
            "pipeline_pass": self.pipeline_pass,
            "quality_gate_pass": self.quality_gate_pass,
            "diagnostic_only": self.diagnostic_only,
            "no_reproduction_claim": self.no_reproduction_claim,
            "no_superiority_claim": self.no_superiority_claim,
            "no_training_success_claim": self.no_training_success_claim,
            "failure_reasons": list(self.failure_reasons),
            "checked_metrics": dict(self.checked_metrics),
        }


def evaluate_quality_feedback(
    summary: Mapping[str, Any],
    gate: QualityGate | None = None,
) -> QualityFeedbackResult:
    """Evaluate one local eval summary for diagnostic quality feedback.

    ``pipeline_pass`` is copied from the input summary only as a health signal.
    The quality gate is computed independently from final-trial locomotion
    metrics and cannot pass from pipeline health alone.
    """

    gate = gate or QualityGate()
    reasons: list[str] = []
    checked: dict[str, float] = {}
    pipeline_pass = bool(summary.get("pipeline_pass", summary.get("pass", False)))

    final_trial = summary.get("final_trial")
    if not isinstance(final_trial, Mapping):
        reasons.append("final_trial_missing")
        final_trial = {}

    trial0 = summary.get("trial_0", summary.get("trial0"))
    aggregate = summary.get("aggregate")
    has_trial0 = isinstance(trial0, Mapping)
    has_aggregate = isinstance(aggregate, Mapping)
    if gate.require_trend_context and not has_trial0 and not has_aggregate:
        reasons.append("trend_context_missing")

    _collect_required_metrics(
        final_trial,
        REQUIRED_FINAL_TRIAL_METRICS,
        "final_trial",
        reasons,
        checked,
    )

    if has_trial0:
        _collect_required_metrics(
            trial0,
            REQUIRED_TRIAL0_TREND_METRICS,
            "trial_0",
            reasons,
            checked,
        )
    elif has_aggregate:
        _collect_required_metrics(
            aggregate,
            REQUIRED_AGGREGATE_METRICS,
            "aggregate",
            reasons,
            checked,
        )

    if _has_final_metrics(checked):
        _check_final_thresholds(checked, gate, reasons)
    if (
        has_trial0
        and gate.enforce_trial0_non_regression
        and _has_final_metrics(checked)
        and _has_trial0_metrics(checked)
    ):
        _check_trial0_non_regression(checked, gate, reasons)

    claim_reasons = _claim_field_reasons(summary)
    reasons.extend(claim_reasons)
    diagnostic_only = not claim_reasons
    no_reproduction_claim = summary.get("reproduction_claim") is False
    no_superiority_claim = summary.get("superiority_claim") is False
    no_training_success_claim = summary.get("training_claim") is False
    if not diagnostic_only:
        reasons.append("claim_boundary_violation")

    quality_gate_pass = not reasons
    return QualityFeedbackResult(
        pipeline_pass=pipeline_pass,
        quality_gate_pass=quality_gate_pass,
        diagnostic_only=diagnostic_only,
        no_reproduction_claim=no_reproduction_claim,
        no_superiority_claim=no_superiority_claim,
        no_training_success_claim=no_training_success_claim,
        failure_reasons=tuple(reasons),
        checked_metrics=checked,
    )


def _collect_required_metrics(
    source: Mapping[str, Any],
    paths: tuple[tuple[str, ...], ...],
    label: str,
    reasons: list[str],
    checked: dict[str, float],
) -> None:
    for path in paths:
        key = f"{label}.{'.'.join(path)}"
        found, value = _get_path(source, path)
        if not found:
            reasons.append(f"{key}_missing")
            continue
        if not _is_finite_number(value):
            reasons.append(f"{key}_nonfinite")
            continue
        checked[key] = float(value)


def _check_final_thresholds(
    checked: Mapping[str, float],
    gate: QualityGate,
    reasons: list[str],
) -> None:
    _max_fail(
        checked["final_trial.completion_ratio"],
        gate.min_final_completion_ratio,
        "final_completion_ratio_too_low",
        reasons,
        minimum=True,
    )
    _max_fail(
        checked["final_trial.fall_ratio"],
        gate.max_final_fall_ratio,
        "final_fall_ratio_too_high",
        reasons,
    )
    _max_fail(
        checked["final_trial.gravity_xy.max"],
        gate.max_final_gravity_xy,
        "final_gravity_xy_too_high",
        reasons,
    )
    _max_fail(
        checked["final_trial.root_z.min"],
        gate.min_final_root_z,
        "final_root_z_too_low",
        reasons,
        minimum=True,
    )
    _max_fail(
        checked["final_trial.lin_vel_error.mean"],
        gate.max_final_lin_vel_error,
        "final_lin_vel_error_too_high",
        reasons,
    )
    _max_fail(
        checked["final_trial.yaw_vel_error.mean"],
        gate.max_final_yaw_vel_error,
        "final_yaw_vel_error_too_high",
        reasons,
    )


def _check_trial0_non_regression(
    checked: Mapping[str, float],
    gate: QualityGate,
    reasons: list[str],
) -> None:
    tol = gate.trend_tolerance
    trend_pairs = (
        ("completion_ratio", True),
        ("fall_ratio", False),
        ("gravity_xy.max", False),
        ("root_z.min", True),
        ("lin_vel_error.mean", False),
        ("yaw_vel_error.mean", False),
    )
    for metric, higher_is_better in trend_pairs:
        final_value = checked[f"final_trial.{metric}"]
        trial0_value = checked[f"trial_0.{metric}"]
        if higher_is_better and final_value + tol < trial0_value:
            reasons.append(f"{metric.replace('.', '_')}_regressed_from_trial0")
        if not higher_is_better and final_value > trial0_value + tol:
            reasons.append(f"{metric.replace('.', '_')}_regressed_from_trial0")


def _max_fail(
    value: float,
    threshold: float,
    reason: str,
    reasons: list[str],
    *,
    minimum: bool = False,
) -> None:
    if minimum:
        if value < threshold:
            reasons.append(reason)
    elif value > threshold:
        reasons.append(reason)


def _has_final_metrics(checked: Mapping[str, float]) -> bool:
    return all(f"final_trial.{'.'.join(path)}" in checked for path in REQUIRED_FINAL_TRIAL_METRICS)


def _has_trial0_metrics(checked: Mapping[str, float]) -> bool:
    return all(f"trial_0.{'.'.join(path)}" in checked for path in REQUIRED_TRIAL0_TREND_METRICS)


def _claim_field_reasons(summary: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for claim_field in NO_OVERCLAIM_FIELDS:
        if claim_field not in summary:
            reasons.append(f"claim_flag_missing:{claim_field}")
        elif summary[claim_field] is not False:
            reasons.append(f"claim_flag_not_false:{claim_field}")
    return reasons


def _get_path(source: Mapping[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    current: Any = source
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False
