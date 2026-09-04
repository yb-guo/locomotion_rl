from __future__ import annotations

from pathlib import Path

import pytest

from h200_locomotion_lab.tools.whole_body_strict_equilibrium_coverage import (
    load_coverage_records,
    run_strict_coverage,
    strict_equilibrium_contract_passed,
)

_R4A2_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a2_joint_aware_equilibrium_diagnosis_4x2.json"
)


def _strict_snapshot() -> dict:
    return {
        "root_qacc_norm": 0.0,
        "joint_qacc_max": 0.0,
        "double_support": True,
        "contact": {"non_foot_contacts": 0, "self_contacts": 0},
        "actuator_saturation_events": 0,
    }


def test_strict_coverage_loads_all_biped_records_not_only_r4a2_feasible() -> None:
    if not _R4A2_ARTIFACT.exists():
        pytest.fail("Task067 R4a.2 coverage artifact is required, not optional")

    records = load_coverage_records(_R4A2_ARTIFACT, families=("biped",))

    assert len(records) == 8
    assert sum(record["contact_equilibrium"]["status"] == "feasible" for record in records) == 5
    assert any(record["contact_equilibrium"]["status"] != "feasible" for record in records)


def test_strict_contract_requires_nominal_hold_acceptance() -> None:
    hold_failed = {"passed": False}
    hold_passed = {"passed": True}

    assert not strict_equilibrium_contract_passed(_strict_snapshot(), hold_failed)
    assert strict_equilibrium_contract_passed(_strict_snapshot(), hold_passed)


def test_strict_contract_rejects_loose_actual_qacc() -> None:
    snapshot = _strict_snapshot()
    snapshot["root_qacc_norm"] = 0.25

    assert not strict_equilibrium_contract_passed(snapshot, {"passed": True})


def test_strict_contract_rejects_self_contact() -> None:
    snapshot = _strict_snapshot()
    snapshot["contact"]["self_contacts"] = 1

    assert not strict_equilibrium_contract_passed(snapshot, {"passed": True})


def test_seed_zero_strict_coverage_finds_known_nominal_hold_positive() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("scipy")
    if not _R4A2_ARTIFACT.exists():
        pytest.fail("Task067 R4a.2 coverage artifact is required, not optional")

    payload = run_strict_coverage(
        input_json=_R4A2_ARTIFACT,
        families=("biped",),
        seeds=(0,),
        range_fractions=(0.0,),
        horizon_steps=20,
        max_nfev=1500,
    )

    assert payload["summary"]["source_records"] == 1
    assert payload["summary"]["strict_refined_actual_equilibria"] == 1
    assert payload["summary"]["strict_contract_passed"] == 1
    assert payload["records"][0]["strict_nominal_hold"]["passed"]
