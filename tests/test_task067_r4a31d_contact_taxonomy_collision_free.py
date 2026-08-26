from __future__ import annotations

import json
from pathlib import Path

import pytest

from h200_locomotion_lab.tools.whole_body_contact_preserving_continuation import (
    _build_shard_for,
)
from h200_locomotion_lab.tools.whole_body_dynamic_balance_diagnosis import _contact_report
from h200_locomotion_lab.tools.whole_body_equilibrium_audit import strict_actual_equilibrium

_R4A31D_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31d_contact_taxonomy_collision_free_strict_coverage_4x2.json"
)


def _strict_snapshot() -> dict:
    return {
        "root_qacc_norm": 0.0,
        "joint_qacc_max": 0.0,
        "double_support": True,
        "contact": {"non_foot_contacts": 0, "self_contacts": 0},
        "actuator_saturation_events": 0,
    }


def test_strict_actual_equilibrium_rejects_self_contact() -> None:
    snapshot = _strict_snapshot()
    snapshot["contact"]["self_contacts"] = 1

    assert not strict_actual_equilibrium(snapshot)


def test_contact_report_exposes_taxonomy_buckets() -> None:
    pytest.importorskip("mujoco")
    shard, _ = _build_shard_for("biped", 0, 0.0)
    data = shard.data[0]
    shard.mujoco.mj_forward(shard.model, data)

    report = _contact_report(shard, data)

    assert "support_foot_floor_contacts" in report
    assert "forbidden_nonfoot_floor_contacts" in report
    assert "self_contacts" in report
    assert set(report["taxonomy"]) >= {
        "support_foot_floor_contacts",
        "forbidden_nonfoot_floor_contacts",
        "self_contacts",
        "geom_pairs",
    }


def test_r4a31d_artifact_records_collision_free_contract() -> None:
    if not _R4A31D_ARTIFACT.exists():
        pytest.fail("Task067 R4a.3.1d artifact is required for collision-free regression")

    payload = json.loads(_R4A31D_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["schema"] == "task067_r4a3_strict_equilibrium_coverage_v2_contact_taxonomy_collision_free"
    assert payload["summary"]["source_records"] == 8
    assert payload["provenance"]["parameters"]["nominal_stance_contract"] == {
        "double_support": True,
        "forbidden_nonfoot_floor_contacts": 0,
        "self_contacts": 0,
    }
    assert payload["summary"]["strict_nominal_hold_self_collision_free"] == 4
    assert payload["summary"]["strict_nominal_hold_self_collision_free_denominator"] == 4
    for record in payload["records"]:
        contact = record["strict_refinement"]["best"]["snapshot"]["contact"]
        assert set(contact["taxonomy"]) >= {
            "support_foot_floor_contacts",
            "forbidden_nonfoot_floor_contacts",
            "self_contacts",
            "geom_pairs",
        }
        if record["strict_contract_passed"]:
            assert contact["self_contacts"] == 0
            assert record["strict_nominal_hold"]["self_contact_steps"] == 0


def test_r4a31d_refinement_allows_upper_body_adjustment_and_self_clearance() -> None:
    if not _R4A31D_ARTIFACT.exists():
        pytest.fail("Task067 R4a.3.1d artifact is required for optimizer contract regression")

    payload = json.loads(_R4A31D_ARTIFACT.read_text(encoding="utf-8"))
    optimizer = payload["records"][0]["strict_refinement"]["optimizer"]

    assert optimizer["upper_body_joint_bound_policy"] == "compiled_physical_joint_limits"
    assert optimizer["lower_body_joint_adjustment"] == pytest.approx(0.08)
    assert optimizer["self_contact_clearance"] > 0.0
