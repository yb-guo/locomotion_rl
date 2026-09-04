from __future__ import annotations

from pathlib import Path

import pytest

from h200_locomotion_lab.tools.whole_body_bounded_feedback_diagnosis import (
    _build_shard,
    _load_feasible_records,
)
from h200_locomotion_lab.tools.whole_body_feedback_authority_diagnosis import (
    _HIGH_GAIN,
    _authority_probes,
    decide_r4b2,
    restoring_score,
    run_authority_probe,
)

_R4A2_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/r4a2_joint_aware_equilibrium_diagnosis_4x2.json"
)


def test_r4b2_restoring_score_is_positive_for_opposing_acceleration() -> None:
    assert restoring_score(perturb_sign=1, qacc_value=-0.25) > 0.0
    assert restoring_score(perturb_sign=-1, qacc_value=0.25) > 0.0
    assert restoring_score(perturb_sign=1, qacc_value=0.25) < 0.0


def test_r4b2_decision_flags_inverted_mapping_suspect() -> None:
    authority_summary = {
        "angular_probes": 10,
        "current_best_angular": 1,
        "inverted_best_angular": 7,
        "static_best_angular": 2,
        "current_improvement_median": 0.1,
        "inverted_improvement_median": 0.5,
        "best_static_improvement_median": 0.3,
    }
    timeline_summary = {"hold_baseline": {"seeds": 5, "contact_loss_precedes_tilt_warning": 0}}

    decision = decide_r4b2(authority_summary, timeline_summary)

    assert decision["status"] == "feedback_sign_or_mapping_suspect"


def test_r4b2_decision_flags_contact_loss_before_tilt() -> None:
    authority_summary = {
        "angular_probes": 10,
        "current_best_angular": 4,
        "inverted_best_angular": 4,
        "static_best_angular": 2,
        "current_improvement_median": 0.2,
        "inverted_improvement_median": 0.1,
        "best_static_improvement_median": 0.15,
    }
    timeline_summary = {"hold_baseline": {"seeds": 5, "contact_loss_precedes_tilt_warning": 3}}

    decision = decide_r4b2(authority_summary, timeline_summary)

    assert decision["status"] == "contact_mode_loss_precedes_tilt"


def test_r4b2_authority_probe_records_bounded_lower_body_deltas() -> None:
    pytest.importorskip("mujoco")
    if not _R4A2_ARTIFACT.exists():
        pytest.skip("R4a.2 artifact is required for R4b-2 authority regression")
    record = next(item for item in _load_feasible_records(_R4A2_ARTIFACT, family="biped") if item["seed"] == 0)
    shard = _build_shard(record)
    probe = next(item for item in _authority_probes() if item.kind == "angle" and item.axis == "roll" and item.sign == 1)

    result = run_authority_probe(shard, record["contact_equilibrium"], probe)

    by_name = {response["name"]: response for response in result["responses"]}
    assert by_name["current_combined_high"]["max_abs_delta"] <= _HIGH_GAIN.max_delta + 1e-12
    assert by_name["inverted_combined_high"]["max_abs_delta"] <= _HIGH_GAIN.max_delta + 1e-12
    assert result["best_any_name"] in by_name
    assert result["probe"]["qacc_dof"] == 3
