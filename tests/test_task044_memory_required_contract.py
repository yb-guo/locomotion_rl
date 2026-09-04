import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_task044_triplet_passes_when_both_ablations_materially_degrade() -> None:
    module = _load_contract()

    result = module.evaluate_task044_memory_required_triplet(
        _summary("none", lin_vel_error=0.20),
        _summary("zero_txl_residual", lin_vel_error=0.25),
        _summary("stateless_txl_memory", lin_vel_error=0.24, pipeline_pass=False),
    )

    assert result["task044_memory_required_pass"] is True
    assert result["memory_required_evidence_pass"] is True
    assert result["zero_residual_ablation"]["degradation"]["degraded"] is True
    assert result["stateless_memory_ablation"]["degradation"]["degraded"] is True
    assert result["memory_causality_claim"] is False
    assert result["reproduction_claim"] is False


def test_task044_triplet_fails_when_ablations_are_tied_with_normal() -> None:
    module = _load_contract()

    result = module.evaluate_task044_memory_required_triplet(
        _summary("none", lin_vel_error=0.20),
        _summary("zero_txl_residual", lin_vel_error=0.21),
        _summary("stateless_txl_memory", lin_vel_error=0.205, pipeline_pass=False),
    )

    assert result["task044_memory_required_pass"] is False
    assert "zero_residual_ablation_not_degraded" in result["failure_reasons"]
    assert "stateless_memory_ablation_not_degraded" in result["failure_reasons"]


def test_task044_triplet_accepts_zero_memory_latent_as_stronger_zero_ablation() -> None:
    module = _load_contract()

    result = module.evaluate_task044_memory_required_triplet(
        _summary("none", lin_vel_error=0.20),
        _summary("zero_memory_latent", lin_vel_error=0.26),
        _summary("stateless_txl_memory", lin_vel_error=0.24, pipeline_pass=False),
    )

    assert result["task044_memory_required_pass"] is True
    assert result["zero_residual_ablation"]["ablation_mode"] == "zero_memory_latent"


def test_task044_triplet_can_use_final_trial_window_for_degradation() -> None:
    module = _load_contract()
    thresholds = module.Task044TripletThresholds(metric_scope="final_trial_window")

    result = module.evaluate_task044_memory_required_triplet(
        _summary("none", lin_vel_error=0.20, window_lin_vel_error=0.20),
        _summary("zero_txl_residual", lin_vel_error=0.20, window_lin_vel_error=0.25),
        _summary(
            "stateless_txl_memory",
            lin_vel_error=0.20,
            window_lin_vel_error=0.24,
            pipeline_pass=False,
        ),
        thresholds,
    )

    assert result["task044_memory_required_pass"] is True
    assert result["thresholds"]["metric_scope"] == "final_trial_window"
    assert result["zero_residual_ablation"]["degradation"]["deltas"][
        "lin_vel_error_delta"
    ] == pytest.approx(0.05)


def test_task044_triplet_can_use_final_trial_tail_window_for_degradation() -> None:
    module = _load_contract()
    thresholds = module.Task044TripletThresholds(metric_scope="final_trial_tail_window")

    result = module.evaluate_task044_memory_required_triplet(
        _summary("none", lin_vel_error=0.20, tail_lin_vel_error=0.18),
        _summary("zero_txl_residual", lin_vel_error=0.20, tail_lin_vel_error=0.23),
        _summary(
            "stateless_txl_memory",
            lin_vel_error=0.20,
            tail_lin_vel_error=0.22,
            pipeline_pass=False,
        ),
        thresholds,
    )

    assert result["task044_memory_required_pass"] is True
    assert result["thresholds"]["metric_scope"] == "final_trial_tail_window"
    assert result["zero_residual_ablation"]["degradation"]["deltas"][
        "lin_vel_error_delta"
    ] == pytest.approx(0.05)


def test_task044_triplet_can_use_post_fault_window_for_degradation() -> None:
    module = _load_contract()
    thresholds = module.Task044TripletThresholds(metric_scope="post_fault_window")

    result = module.evaluate_task044_memory_required_triplet(
        _summary("none", lin_vel_error=0.20, post_fault_lin_vel_error=0.18),
        _summary("zero_memory_latent", lin_vel_error=0.20, post_fault_lin_vel_error=0.24),
        _summary(
            "stateless_txl_memory",
            lin_vel_error=0.20,
            post_fault_lin_vel_error=0.22,
            pipeline_pass=False,
        ),
        thresholds,
    )

    assert result["task044_memory_required_pass"] is True
    assert result["thresholds"]["metric_scope"] == "post_fault_window"


def test_task044_triplet_requires_normal_quality_pass() -> None:
    module = _load_contract()

    result = module.evaluate_task044_memory_required_triplet(
        _summary("none", quality_gate_pass=False),
        _summary("zero_txl_residual", lin_vel_error=0.30),
        _summary("stateless_txl_memory", lin_vel_error=0.30),
    )

    assert result["task044_memory_required_pass"] is False
    assert "normal_quality_gate_not_passed" in result["failure_reasons"]


def test_task044_triplet_requires_fault_labels_hidden_from_actor_obs() -> None:
    module = _load_contract()
    normal = _summary("none")
    normal["task044_hidden_fault_contract"]["fault_identity_in_actor_obs"] = True

    result = module.evaluate_task044_memory_required_triplet(
        normal,
        _summary("zero_txl_residual", lin_vel_error=0.30),
        _summary("stateless_txl_memory", lin_vel_error=0.30),
    )

    assert result["task044_memory_required_pass"] is False
    assert "fault_identity_in_actor_obs_not_false" in result["failure_reasons"]


def test_task044_triplet_rejects_missing_ablation_mode() -> None:
    module = _load_contract()
    zero = _summary("zero_txl_residual", lin_vel_error=0.30)
    zero.pop("memory_ablation_mode")
    zero["txl_debug"].pop("task042_memory_ablation_mode")

    result = module.evaluate_task044_memory_required_triplet(
        _summary("none"),
        zero,
        _summary("stateless_txl_memory", lin_vel_error=0.30),
    )

    assert result["task044_memory_required_pass"] is False
    assert "zero_residual_ablation_mode_mismatch" in result["failure_reasons"]


def test_task044_stateless_pipeline_false_still_counts_as_recorded_ablation() -> None:
    module = _load_contract()

    result = module.evaluate_task044_memory_required_triplet(
        _summary("none", lin_vel_error=0.20),
        _summary("zero_txl_residual", lin_vel_error=0.30),
        _summary("stateless_txl_memory", lin_vel_error=0.30, pipeline_pass=False),
    )

    assert result["task044_memory_required_pass"] is True
    assert result["stateless_memory_ablation"]["recorded"] is True
    assert result["stateless_memory_ablation"]["ablation_mode"] == "stateless_txl_memory"


def _summary(
    mode: str,
    *,
    lin_vel_error: float = 0.20,
    window_lin_vel_error: float | None = None,
    tail_lin_vel_error: float | None = None,
    post_fault_lin_vel_error: float | None = None,
    fall_ratio: float = 0.0,
    completion_ratio: float = 1.0,
    pipeline_pass: bool = True,
    quality_gate_pass: bool = True,
) -> dict:
    return {
        "pipeline_pass": pipeline_pass,
        "quality_gate_pass": quality_gate_pass,
        "pass": pipeline_pass and quality_gate_pass,
        "memory_ablation_mode": mode,
        "memory_ablation_mode_match": True,
        "memory_debug_active": mode == "none",
        "task044_hidden_fault_contract": {
            "fault_identity_in_actor_obs": False,
            "fault_severity_in_actor_obs": False,
            "fault_onset_in_actor_obs": False,
            "fault_recovery_in_actor_obs": False,
            "fault_schedule_in_actor_obs": False,
        },
        "final_trial": {
            "completion_ratio": completion_ratio,
            "fall_ratio": fall_ratio,
            "lin_vel_error": {"mean": lin_vel_error},
        },
        "final_trial_window": {
            "completion_ratio": completion_ratio,
            "fall_ratio": fall_ratio,
            "lin_vel_error": {
                "mean": window_lin_vel_error
                if window_lin_vel_error is not None
                else lin_vel_error
            },
        },
        "final_trial_tail_window": {
            "completion_ratio": completion_ratio,
            "fall_ratio": fall_ratio,
            "lin_vel_error": {
                "mean": tail_lin_vel_error if tail_lin_vel_error is not None else lin_vel_error
            },
        },
        "post_fault_window": {
            "completion_ratio": completion_ratio,
            "fall_ratio": fall_ratio,
            "lin_vel_error": {
                "mean": post_fault_lin_vel_error
                if post_fault_lin_vel_error is not None
                else lin_vel_error
            },
        },
        "txl_debug": {
            "task042_memory_ablation_mode": mode,
            "stateful_memory_enabled": mode != "stateless_txl_memory",
            "last_attended_previous_memory_lengths": [64, 64],
            "incremental_steps": 8,
        },
    }


def _load_contract():
    path = ROOT / "src" / "h200_locomotion_lab" / "training" / "task044_memory_required_contract.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module
