from __future__ import annotations

import json
from pathlib import Path

import pytest

from h200_locomotion_lab.tools.whole_body_contact_preserving_continuation import (
    _build_shard_for,
)
from h200_locomotion_lab.tools.whole_body_equilibrium_audit import (
    _STRICT_JOINT_QACC_MAX,
    _STRICT_ROOT_QACC_NORM,
)
from h200_locomotion_lab.tools.whole_body_lexicographic_collision_free_equilibrium import (
    _DEFAULT_R4A31D_INPUT,
    _DEFAULT_R4A31E_INPUT,
    continuous_collision_clearance_report,
)

_R4A31F_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31f_lexicographic_collision_free_equilibrium_realization.json"
)


def _artifact() -> dict:
    if not _R4A31F_ARTIFACT.exists():
        pytest.fail("Task067 R4a.3.1f artifact is required for lexicographic regression")
    return json.loads(_R4A31F_ARTIFACT.read_text(encoding="utf-8"))


def test_continuous_clearance_report_uses_signed_distance_not_contact_count() -> None:
    pytest.importorskip("mujoco")
    shard, _ = _build_shard_for("biped", 3, 0.5)
    data = shard.data[0]
    shard.mujoco.mj_forward(shard.model, data)

    report = continuous_collision_clearance_report(shard, data)

    assert report["uses_integer_contact_count_for_clearance"] is False
    assert report["self_collision_geom_pair_count"] >= 1
    assert report["minimum_self_pair_distance"] is not None
    assert all("distance" in pair for pair in report["self_collision_geom_pairs"])
    assert all("contact_count" not in pair for pair in report["self_collision_geom_pairs"])


def test_r4a31f_artifact_contract_and_full_coverage() -> None:
    payload = _artifact()

    assert payload["schema"] == (
        "task067_r4a31f_lexicographic_collision_free_equilibrium_realization_v1"
    )
    assert payload["source_artifacts"]["r4a31d"].endswith(_DEFAULT_R4A31D_INPUT.name)
    assert payload["source_artifacts"]["r4a31e"].endswith(_DEFAULT_R4A31E_INPUT.name)
    assert payload["assertions"]["uses_lexicographic_phases"]
    assert payload["assertions"]["kinematic_phase_uses_continuous_signed_distance_not_contact_count"]
    assert payload["assertions"]["dynamics_phase_uses_no_integer_contact_count_residual"]
    assert payload["assertions"]["final_acceptance_does_not_require_exactly_8_contacts"]
    assert payload["assertions"]["search_failure_not_promoted_to_physical_infeasible"]
    assert payload["summary"]["failed_endpoints_tested"] == 3
    assert payload["summary"]["endpoints_recovered"] == 3
    assert payload["summary"]["combined_collision_free_strict_passed"] == 8
    assert payload["summary"]["combined_incomplete_labels"] == []
    assert payload["decision"]["status"] == "r4a31f_collision_free_strict_coverage_restored_8_of_8"
    assert payload["provenance"]["diagnostic_scope"]["prepares_stance_solution_v3"] is False
    assert payload["provenance"]["diagnostic_scope"]["restores_feedback_or_task061_062"] is False


def test_r4a31f_kinematic_phase_reaches_flat_collision_free_qpos() -> None:
    payload = _artifact()

    assert payload["summary"]["kinematic_flat_collision_free_reachable"] == 3
    for endpoint in payload["endpoints"]:
        phase = endpoint["kinematic_phase"]
        best = phase["best"]
        clearance = best["continuous_clearance"]
        support = best["support"]["support_margin"]

        assert phase["passed_attempt_count"] >= 1
        assert best["kinematic_constraints_passed"]
        assert best["flat_patch"]["height_error_to_penetration_max_abs"] <= 1e-3
        assert best["flat_patch"]["global_height_spread"] <= 1e-3
        assert clearance["minimum_self_pair_distance"] is None or clearance["minimum_self_pair_distance"] >= 0.0
        assert (
            clearance["minimum_nonfoot_floor_distance"] is None
            or clearance["minimum_nonfoot_floor_distance"] >= 0.0
        )
        assert support["inside"]
        assert support["margin"] >= 1e-3


def test_r4a31f_dynamics_phase_actual_strict_hold_for_every_recovered_endpoint() -> None:
    payload = _artifact()
    records = [payload["positive_control"], *payload["endpoints"]]

    for record in records:
        best = record["dynamics_phase"]["best"]
        snapshot = best["actual_snapshot"]
        hold = best["strict_nominal_hold_2s"]

        assert record["strict_recovered"]
        assert best["strict_gate_passed"]
        assert best["flat_geometry_realized"]
        assert best["variable_contract"]["uses_integer_contact_count_residual"] is False
        assert best["bounded_ctrl_subproblem"]["fixed_qpos"]
        assert "affine_prediction_error_max_abs" in best["bounded_ctrl_subproblem"]
        assert snapshot["root_qacc_norm"] <= _STRICT_ROOT_QACC_NORM
        assert snapshot["joint_qacc_max"] <= _STRICT_JOINT_QACC_MAX
        assert snapshot["support_mode"] == "double_support"
        assert snapshot["contact"]["non_foot_contacts"] == 0
        assert snapshot["contact"].get("self_contacts", 0) == 0
        assert hold["hold_executed"]
        assert hold["passed"]
        assert hold["self_contact_steps"] == 0
