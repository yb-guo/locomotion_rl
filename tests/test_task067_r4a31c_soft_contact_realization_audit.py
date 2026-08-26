from __future__ import annotations

import json
from pathlib import Path

import pytest

from h200_locomotion_lab.tools.whole_body_contact_preserving_continuation import (
    _build_shard_for,
    _load_r4a3_records,
)
from h200_locomotion_lab.tools.whole_body_soft_contact_realization_audit import (
    _DEFAULT_R4A31B_INPUT,
    _decide,
    force_closure_audit_for_state,
    strict_positive_control_audits,
)

_R4A3_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a3_strict_equilibrium_coverage_4x2.json"
)
_R4A31C_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31c_soft_contact_force_closure_realization_audit.json"
)


def test_strict_positive_control_dynamics_and_internal_efc_close() -> None:
    pytest.importorskip("mujoco")
    records = _load_r4a3_records(_R4A3_ARTIFACT, family="biped")
    record = next(
        item
        for item in records
        if item["strict_contract_passed"] and int(item["seed"]) == 0 and float(item["range_fraction"]) == 0.0
    )
    shard, _ = _build_shard_for("biped", 0, 0.0)
    best = record["strict_refinement"]["best"]

    audit = force_closure_audit_for_state(shard, best["qpos"], best["ctrl"])

    assert audit["actuator_generalized_force_vs_data_qfrc_actuator"]["passed"]
    assert audit["contact_force_jacobian_and_efc_closure"][
        "full_mujoco_efc_jacobian_vs_full_qfrc_constraint"
    ]["passed"]
    assert audit["contact_force_jacobian_and_efc_closure"][
        "foot_hand_force_jacobian_vs_filtered_foot_floor_efc"
    ]["passed"]
    assert audit["full_dynamics_closure"]["passed"]


def test_positive_controls_are_exactly_the_five_strict_records() -> None:
    pytest.importorskip("mujoco")
    records = _load_r4a3_records(_R4A3_ARTIFACT, family="biped")

    audits = strict_positive_control_audits(records)

    assert len(audits) == 5
    assert {row["label"] for row in audits} == {
        "biped:rf0:seed0",
        "biped:rf0:seed2",
        "biped:rf0:seed3",
        "biped:rf0.5:seed1",
        "biped:rf0.5:seed2",
    }


def test_decision_pauses_rigid_candidates_when_positive_controls_do_not_close() -> None:
    summary = {
        "positive_control_closure": {
            "strict_positive_controls": 5,
            "full_closure_passed": 2,
        },
        "failed_active_set": {"inconsistent_endpoint_count": 0},
        "penetration_sweep": {"strict_recovered_count": 0},
    }

    decision = _decide(summary)

    assert decision["status"] == "r4a31c_force_jacobian_mapping_bug_pause_rigid_candidates"
    assert "pause" in decision["decision"].lower()


def test_artifact_records_seed3_unselected_corner_deeper_than_selected() -> None:
    if not _R4A31C_ARTIFACT.exists():
        pytest.fail("Task067 R4a.3.1c artifact is required for active-set regression")

    payload = json.loads(_R4A31C_ARTIFACT.read_text(encoding="utf-8"))
    seed3 = next(
        row
        for row in payload["failed_candidate_realization_audits"]
        if row["endpoint_label"] == "biped:rf0.5:seed3"
    )

    assert seed3["corner_height_audit"]["unselected_deeper_than_selected"]
    assert seed3["corner_height_audit"]["unselected_deeper_margin"] > 0.002


def test_artifact_has_penetration_sweep_and_no_physical_infeasibility_certificate() -> None:
    if not _R4A31C_ARTIFACT.exists():
        pytest.fail("Task067 R4a.3.1c artifact is required for sweep regression")

    payload = json.loads(_R4A31C_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["schema"] == "task067_r4a31c_soft_contact_force_closure_realization_audit_v2_same_set_efc"
    assert payload["fixed_contact_artifact"].endswith(_DEFAULT_R4A31B_INPUT.name)
    assert payload["assertions"]["strict_positive_control_count_is_5"]
    assert payload["assertions"]["full_dynamics_closure_passes_all_positive_controls"]
    assert payload["assertions"]["search_failure_not_promoted_to_physical_infeasible"]
    assert payload["provenance"]["diagnostic_scope"][
        "foot_hand_reconstruction_compared_only_to_filtered_foot_floor_efc"
    ]
    assert payload["provenance"]["diagnostic_scope"]["full_efc_compared_to_full_qfrc_constraint"]
    assert all(len(row["samples"]) == 13 for row in payload["penetration_sweeps"])
