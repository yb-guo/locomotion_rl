"""Frozen whole-body generalization and ablation evaluation protocols."""

from h200_locomotion_lab.evaluation.whole_body_ood import (
    DEFAULT_WHOLE_BODY_OOD_CASES,
    OODCase,
    OODResult,
    WholeBodyOODThresholds,
    build_whole_body_ood_plan,
    evaluate_ood_gate,
    paired_bootstrap_ci,
    run_ood_suite,
    validate_checkpoint_selection_metadata,
)

__all__ = [
    "DEFAULT_WHOLE_BODY_OOD_CASES",
    "OODCase",
    "OODResult",
    "WholeBodyOODThresholds",
    "build_whole_body_ood_plan",
    "evaluate_ood_gate",
    "paired_bootstrap_ci",
    "run_ood_suite",
    "validate_checkpoint_selection_metadata",
]
