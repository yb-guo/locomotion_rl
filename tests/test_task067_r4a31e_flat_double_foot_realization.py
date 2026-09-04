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
from h200_locomotion_lab.tools.whole_body_flat_double_foot_realization import (
    _DEFAULT_INPUT,
    flat_patch_report,
)

_R4A31E_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31e_flat_double_foot_active_set_realization.json"
)


def test_flat_patch_report_tracks_all_nominal_sole_corners_without_exact_contact_requirement() -> None:
    pytest.importorskip("mujoco")
    shard, _ = _build_shard_for("biped", 0, 0.0)
    data = shard.data[0]
    shard.mujoco.mj_forward(shard.model, data)

    report = flat_patch_report(shard, data, penetration=0.0)

    assert report["corner_count"] == 8
    assert report["does_not_require_exact_mujoco_contact_count"]
    assert set(report["feet"]) == set(shard._foot_geoms)
    assert all(foot["corner_count"] == 4 for foot in report["feet"].values())


def test_r4a31e_artifact_runs_all_failures_and_positive_same_path() -> None:
    if not _R4A31E_ARTIFACT.exists():
        pytest.fail("Task067 R4a.3.1e artifact is required for flat realization regression")

    payload = json.loads(_R4A31E_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["schema"] == "task067_r4a31e_flat_double_foot_active_set_realization_v1"
    assert payload["source_artifact"].endswith(_DEFAULT_INPUT.name)
    assert payload["assertions"]["all_four_failed_endpoints_tested"]
    assert payload["assertions"]["positive_control_same_path_tested"]
    assert payload["assertions"]["final_acceptance_does_not_require_exactly_8_contacts"]
    assert payload["assertions"]["search_failure_not_promoted_to_physical_infeasible"]
    assert payload["summary"]["failed_endpoints_tested"] == 4
    assert payload["summary"]["positive_controls_tested"] == 1
    assert {row["role"] for row in payload["endpoints"]} == {"failed_endpoint"}
    assert payload["positive_control"]["role"] == "positive_control"
    assert payload["positive_control"]["strict_recovered"]
    assert payload["positive_control"]["best"]["strict_gate_passed"]
    assert payload["positive_control"]["best"]["actual_strict_gate_and_hold_passed"]


def test_r4a31e_positive_control_actual_strict_and_hold_pass() -> None:
    if not _R4A31E_ARTIFACT.exists():
        pytest.fail("Task067 R4a.3.1e artifact is required for positive-control regression")

    payload = json.loads(_R4A31E_ARTIFACT.read_text(encoding="utf-8"))
    best = payload["positive_control"]["best"]
    snapshot = best["actual_snapshot"]
    hold = best["strict_nominal_hold_2s"]

    assert snapshot["root_qacc_norm"] <= _STRICT_ROOT_QACC_NORM
    assert snapshot["joint_qacc_max"] <= _STRICT_JOINT_QACC_MAX
    assert snapshot["support_mode"] == "double_support"
    assert snapshot["contact"]["non_foot_contacts"] == 0
    assert snapshot["contact"].get("self_contacts", 0) == 0
    assert hold["hold_executed"]
    assert hold["passed"]
    assert hold["self_contact_steps"] == 0


def test_r4a31e_full_limit_fallback_available_after_local_search() -> None:
    if not _R4A31E_ARTIFACT.exists():
        pytest.fail("Task067 R4a.3.1e artifact is required for fallback regression")

    payload = json.loads(_R4A31E_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["provenance"]["parameters"]["full_limit_fallback_required_after_local_failure"]
    assert payload["summary"]["full_limit_fallback_executed_count"] >= 1
    for endpoint in payload["endpoints"]:
        local_attempts = [
            preview
            for preview in endpoint["attempt_preview"]
            if preview["stage"] == "local_trust_region"
        ]
        if all(not preview["strict_gate_passed"] for preview in local_attempts):
            assert endpoint["full_limit_fallback_executed"]
        if endpoint["full_limit_fallback_executed"]:
            assert any(
                preview["stage"] == "full_limit_fallback"
                for preview in endpoint["attempt_preview"]
            )


def test_r4a31e_seed3_input_only_probe_is_recorded() -> None:
    if not _R4A31E_ARTIFACT.exists():
        pytest.fail("Task067 R4a.3.1e artifact is required for input-only regression")

    payload = json.loads(_R4A31E_ARTIFACT.read_text(encoding="utf-8"))
    probe = payload["input_only_probe"]

    assert probe["label"] == "biped:rf0:seed3"
    assert probe["fixed_qpos"]
    assert probe["solves_ctrl_only"]
    assert probe["conclusion"] in {"input_only_recovered_strict", "input_only_not_sufficient"}
    assert not probe["strict_gate_passed"]
