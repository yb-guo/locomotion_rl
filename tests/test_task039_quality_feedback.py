import math

from h200_locomotion_lab.training.task039_quality_feedback import (
    evaluate_quality_feedback,
)


def _trial(
    *,
    completion_ratio=1.0,
    fall_ratio=0.0,
    gravity_xy_max=0.20,
    root_z_min=0.72,
    lin_vel_error_mean=0.18,
    yaw_vel_error_mean=0.08,
):
    return {
        "completion_ratio": completion_ratio,
        "fall_ratio": fall_ratio,
        "gravity_xy": {"mean": gravity_xy_max * 0.5, "max": gravity_xy_max},
        "root_z": {"mean": root_z_min + 0.05, "min": root_z_min},
        "lin_vel_error": {"mean": lin_vel_error_mean},
        "yaw_vel_error": {"mean": yaw_vel_error_mean},
    }


def _aggregate():
    return {
        "trial_count": 2,
        "completion_ratio_per_trial_mean": 0.85,
        "fall_ratio": 0.1,
        "gravity_xy_max": 0.6,
        "root_z_min": 0.6,
        "lin_vel_error_mean": 0.45,
        "yaw_vel_error_mean": 0.25,
    }


def _improved_summary():
    return {
        "pipeline_pass": True,
        "pass": True,
        "eval_pipeline_smoke_only": True,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "trial_0": _trial(
            completion_ratio=0.70,
            fall_ratio=0.35,
            gravity_xy_max=0.70,
            root_z_min=0.58,
            lin_vel_error_mean=0.80,
            yaw_vel_error_mean=0.45,
        ),
        "final_trial": _trial(),
        "aggregate": _aggregate(),
    }


def test_missing_metrics_fail_with_clear_reasons():
    summary = _improved_summary()
    del summary["final_trial"]["root_z"]

    result = evaluate_quality_feedback(summary)

    assert not result.quality_gate_pass
    assert "final_trial.root_z.min_missing" in result.failure_reasons


def test_nonfinite_metrics_fail():
    summary = _improved_summary()
    summary["final_trial"]["lin_vel_error"]["mean"] = math.inf

    result = evaluate_quality_feedback(summary)

    assert not result.quality_gate_pass
    assert "final_trial.lin_vel_error.mean_nonfinite" in result.failure_reasons


def test_low_root_z_fails():
    summary = _improved_summary()
    summary["final_trial"]["root_z"]["min"] = 0.30

    result = evaluate_quality_feedback(summary)

    assert not result.quality_gate_pass
    assert "final_root_z_too_low" in result.failure_reasons


def test_high_gravity_xy_fails():
    summary = _improved_summary()
    summary["final_trial"]["gravity_xy"]["max"] = 0.95

    result = evaluate_quality_feedback(summary)

    assert not result.quality_gate_pass
    assert "final_gravity_xy_too_high" in result.failure_reasons


def test_high_fall_ratio_fails():
    summary = _improved_summary()
    summary["final_trial"]["fall_ratio"] = 0.25

    result = evaluate_quality_feedback(summary)

    assert not result.quality_gate_pass
    assert "final_fall_ratio_too_high" in result.failure_reasons


def test_high_tracking_errors_fail():
    summary = _improved_summary()
    summary["final_trial"]["lin_vel_error"]["mean"] = 0.90
    summary["final_trial"]["yaw_vel_error"]["mean"] = 0.80

    result = evaluate_quality_feedback(summary)

    assert not result.quality_gate_pass
    assert "final_lin_vel_error_too_high" in result.failure_reasons
    assert "final_yaw_vel_error_too_high" in result.failure_reasons


def test_pipeline_pass_only_does_not_pass_quality_gate():
    result = evaluate_quality_feedback(
        {
            "pipeline_pass": True,
            "pass": True,
            "quality_claim": False,
            "training_claim": False,
            "eval_claim": False,
            "reproduction_claim": False,
            "superiority_claim": False,
        }
    )

    assert result.pipeline_pass
    assert not result.quality_gate_pass
    assert "final_trial_missing" in result.failure_reasons


def test_complete_improved_fixture_passes_as_diagnostic_only():
    result = evaluate_quality_feedback(_improved_summary())

    assert result.pipeline_pass
    assert result.quality_gate_pass
    assert result.failure_reasons == ()
    assert result.diagnostic_only
    assert result.no_reproduction_claim
    assert result.no_superiority_claim
    assert result.no_training_success_claim
    assert result.to_json()["quality_gate_pass"] is True


def test_trial0_non_regression_ignores_submillimeter_root_z_noise():
    summary = _improved_summary()
    summary["trial_0"]["root_z"]["min"] = 0.7562705
    summary["final_trial"]["root_z"]["min"] = 0.7557917

    result = evaluate_quality_feedback(summary)

    assert result.quality_gate_pass
    assert "root_z_min_regressed_from_trial0" not in result.failure_reasons


def test_trial0_non_regression_still_rejects_material_root_z_regression():
    summary = _improved_summary()
    summary["trial_0"]["root_z"]["min"] = 0.76
    summary["final_trial"]["root_z"]["min"] = 0.74

    result = evaluate_quality_feedback(summary)

    assert not result.quality_gate_pass
    assert "root_z_min_regressed_from_trial0" in result.failure_reasons


def test_missing_claim_flag_fails_with_clear_reason():
    summary = _improved_summary()
    del summary["quality_claim"]

    result = evaluate_quality_feedback(summary)

    assert not result.quality_gate_pass
    assert not result.diagnostic_only
    assert "claim_flag_missing:quality_claim" in result.failure_reasons
    assert "claim_boundary_violation" in result.failure_reasons


def test_true_claim_flag_fails_with_clear_reason():
    summary = _improved_summary()
    summary["reproduction_claim"] = True

    result = evaluate_quality_feedback(summary)

    assert not result.quality_gate_pass
    assert not result.diagnostic_only
    assert not result.no_reproduction_claim
    assert "claim_flag_not_false:reproduction_claim" in result.failure_reasons
    assert "claim_boundary_violation" in result.failure_reasons


def test_aggregate_can_supply_trend_context_when_trial0_absent():
    summary = _improved_summary()
    del summary["trial_0"]

    result = evaluate_quality_feedback(summary)

    assert result.quality_gate_pass
    assert "aggregate.root_z_min" in result.checked_metrics
