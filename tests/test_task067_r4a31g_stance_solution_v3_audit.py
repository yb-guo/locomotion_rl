from __future__ import annotations

import pytest

from h200_locomotion_lab.tools import whole_body_stance_solution_v3_audit as audit


def _passing_record(*, family: str, seed: int, range_fraction: float, horizon_steps: int) -> dict:
    del horizon_steps
    return {
        "family": family,
        "seed": seed,
        "range_fraction": float(range_fraction),
        "label": f"{family}:rf{range_fraction:g}:seed{seed}",
        "solve_seconds": 0.1,
        "root_gauge": {"x_y_yaw_zero": True},
        "margins": {
            "min_joint_margin": 0.06,
            "min_ctrl_margin": 0.02,
            "max_abs_ctrl_minus_qpos": 0.1,
        },
        "geometry": {
            "feet_near_floor": 2,
            "foot_height_spread": 0.0,
            "support_all_feet": {
                "degenerate": False,
                "inside": True,
                "hull_area": 0.1,
                "margin": 0.01,
            },
        },
        "public_zero_action_hold": {"survived": True},
        "v3_acceptance": {
            "biped_strict_initial_and_hold": family == "biped",
            "stance_matrix_hold": True,
        },
    }


def test_r4a31g_audit_records_solver_failure_without_physical_infeasibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_record(*, family: str, seed: int, range_fraction: float, horizon_steps: int) -> dict:
        if family == "biped" and seed == 1:
            raise RuntimeError("actual-dynamics stance solver failed strict equilibrium")
        return _passing_record(
            family=family,
            seed=seed,
            range_fraction=range_fraction,
            horizon_steps=horizon_steps,
        )

    monkeypatch.setattr(audit, "run_record", fake_run_record)

    payload = audit.run_audit(
        endpoint_seeds=2,
        matrix_seeds=2,
        range_fractions=(0.0,),
        horizon_steps=100,
        include_quadruped_matrix=False,
    )

    failures = [
        record
        for record in payload["stance_matrix_records"]
        if record.get("record_status") == "record_build_failed"
    ]
    assert [record["label"] for record in failures] == ["biped:rf0:seed1"]
    assert failures[0]["failure_classification"] == "search_exhausted_without_certificate"
    assert failures[0]["physical_infeasibility_claimed"] is False
    assert payload["summary"]["stance_matrix"]["record_build_failures"] == 1
    assert payload["summary"]["endpoint_4x2"]["biped_record_build_failures"] == 1
    assert payload["decision"]["status"] == "stance_solution_v3_biped_strict_feedforward_incomplete"
    assert payload["assertions"]["endpoint_record_build_failures_are_zero"] is False
