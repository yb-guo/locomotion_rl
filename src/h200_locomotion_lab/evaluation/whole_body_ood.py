"""Task057 frozen heldout/OOD evaluation protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from h200_locomotion_lab.core.checkpoint import WholeBodyCheckpointMetadata
from h200_locomotion_lab.robots.whole_body_slots import (
    WholeBodySlotMapping,
    build_anymal_c_mapping,
    build_berkeley_humanoid_mapping,
    build_g1_whole_body_mapping,
    build_go2_mapping,
)

OODSplit = Literal["heldout", "ood", "regression"]


@dataclass(frozen=True, slots=True)
class OODCase:
    name: str
    split: OODSplit
    robot: str | None = None
    dynamics_multiplier: float = 1.0
    dynamic_motor_events: bool = False
    locked_motor: bool = False
    mapping_builder: Callable[[], WholeBodySlotMapping] | None = None

    def __post_init__(self) -> None:
        if self.split not in {"heldout", "ood", "regression"}:
            raise ValueError("unsupported OOD split")
        if self.dynamics_multiplier < 1.0:
            raise ValueError("OOD dynamics multiplier must be >= 1")
        if self.locked_motor and self.dynamic_motor_events:
            raise ValueError("locked and dynamic motor cases are separate conditions")

    @property
    def mapping(self) -> WholeBodySlotMapping | None:
        return self.mapping_builder() if self.mapping_builder else None


DEFAULT_WHOLE_BODY_OOD_CASES = (
    OODCase("procedural_heldout_topology", "heldout"),
    OODCase("doubled_dynamics_range", "ood", dynamics_multiplier=2.0),
    OODCase("dynamic_weak_dead_delay", "ood", dynamic_motor_events=True),
    OODCase("unseen_locked_stuck_motor", "ood", locked_motor=True),
    OODCase(
        "berkeley_humanoid",
        "regression",
        robot="Berkeley Humanoid",
        mapping_builder=build_berkeley_humanoid_mapping,
    ),
    OODCase(
        "anymal_c",
        "regression",
        robot="ANYmal C",
        mapping_builder=build_anymal_c_mapping,
    ),
    OODCase(
        "g1_29dof_whole_body",
        "regression",
        robot="G1",
        mapping_builder=build_g1_whole_body_mapping,
    ),
    OODCase(
        "go2_quadruped",
        "regression",
        robot="Go2",
        mapping_builder=build_go2_mapping,
    ),
)


@dataclass(frozen=True, slots=True)
class WholeBodyOODThresholds:
    zero_fall_ratio: float = 0.80
    normalized_velocity_error: float = 0.35
    dynamic_few_shot_improvement: float = 0.05


@dataclass(frozen=True, slots=True)
class OODResult:
    case: str
    metrics: Mapping[str, float]
    passed: bool
    bootstrap_ci: tuple[float, float] | None = None


def evaluate_ood_gate(
    metrics: Mapping[str, float],
    thresholds: WholeBodyOODThresholds | None = None,
) -> tuple[bool, list[str]]:
    """Evaluate the frozen checkpoint's zero-shot/few-shot thresholds."""

    thresholds = thresholds or WholeBodyOODThresholds()
    reasons: list[str] = []
    if metrics.get("zero_fall_ratio", 0.0) < thresholds.zero_fall_ratio:
        reasons.append("zero_fall_ratio_below_threshold")
    if metrics.get("normalized_velocity_error", float("inf")) > thresholds.normalized_velocity_error:
        reasons.append("normalized_velocity_error_above_threshold")
    if (
        "few_shot_dynamic_improvement" in metrics
        and metrics["few_shot_dynamic_improvement"] < thresholds.dynamic_few_shot_improvement
    ):
        reasons.append("few_shot_dynamic_improvement_below_threshold")
    if "paired_bootstrap_ci_lower" in metrics and metrics["paired_bootstrap_ci_lower"] <= 0.0:
        reasons.append("paired_bootstrap_ci_includes_zero")
    return not reasons, reasons


def build_whole_body_ood_plan() -> tuple[OODCase, ...]:
    """Return a fresh tuple so callers cannot mutate the frozen evaluation set."""

    return tuple(DEFAULT_WHOLE_BODY_OOD_CASES)


def validate_checkpoint_selection_metadata(metadata: WholeBodyCheckpointMetadata) -> None:
    """Reject heldout/OOD metadata from validation-checkpoint selection."""

    if metadata.topology_split not in {"train", "validation"}:
        raise ValueError("heldout/OOD/regression checkpoints cannot select a model")


def run_ood_suite(
    runner: Callable[[OODCase], Mapping[str, float]],
    *,
    cases: Sequence[OODCase] = DEFAULT_WHOLE_BODY_OOD_CASES,
    thresholds: WholeBodyOODThresholds | None = None,
) -> tuple[OODResult, ...]:
    thresholds = thresholds or WholeBodyOODThresholds()
    results: list[OODResult] = []
    for case in cases:
        metrics = dict(runner(case))
        passed = (
            metrics.get("zero_fall_ratio", 0.0) >= thresholds.zero_fall_ratio
            and metrics.get("normalized_velocity_error", float("inf"))
            <= thresholds.normalized_velocity_error
        )
        results.append(OODResult(case.name, metrics, passed))
    return tuple(results)


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap a paired improvement and return its percentile CI."""

    if not differences:
        raise ValueError("at least one paired difference is required")
    if not 0.0 < confidence < 1.0 or samples <= 0:
        raise ValueError("confidence must be in (0,1) and samples must be positive")
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional evaluation dependency
        raise RuntimeError("NumPy is required for paired bootstrap") from exc
    values = np.asarray(tuple(float(value) for value in differences), dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) * 0.5
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))
