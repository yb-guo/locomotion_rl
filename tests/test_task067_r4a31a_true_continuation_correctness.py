from __future__ import annotations

import json
from pathlib import Path

import pytest

from h200_locomotion_lab.tools.whole_body_contact_preserving_continuation import (
    SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE,
    _build_shard_for,
    _load_r4a3_records,
    _strict_source_state,
)
from h200_locomotion_lab.tools.whole_body_true_continuation_correctness import (
    _decide,
    artificial_warm_start_clip_violations,
    true_warm_start_from_previous_solution,
)

_R4A3_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a3_strict_equilibrium_coverage_4x2.json"
)
_R4A31A_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31a_true_continuation_correctness_3fail.json"
)


def _strict_source(seed: int, range_fraction: float) -> dict:
    if not _R4A3_ARTIFACT.exists():
        pytest.fail("Task067 R4a.3 artifact is required for R4a.3.1a regression")
    records = _load_r4a3_records(_R4A3_ARTIFACT, family="biped")
    return _strict_source_state(
        records,
        family="biped",
        seed=seed,
        range_fraction=range_fraction,
    )


def test_seed3_min_step_does_not_apply_target_stance_artificial_clip() -> None:
    pytest.importorskip("mujoco")
    source = _strict_source(seed=3, range_fraction=0.0)
    target_shard, _ = _build_shard_for("biped", 3, 0.00625)

    warm_start, _ = true_warm_start_from_previous_solution(
        target_shard,
        source["qpos"],
        source["ctrl"],
    )

    artificial_clip = warm_start.report["artificial_trust_region_clip"]
    target_distance = warm_start.report["target_r2_stance_distance_to_previous_solution"]
    assert target_distance["joint_abs_max"] > 0.14
    assert artificial_clip["count"] == 0
    assert artificial_clip["max_abs_clip"] == 0.0
    assert not any(abs(row["abs_clip"] - 0.145) < 1e-3 for row in artificial_clip["rows"])


def test_target_r2_stance_distance_does_not_change_warm_start_branch() -> None:
    pytest.importorskip("mujoco")
    source = _strict_source(seed=3, range_fraction=0.0)
    target_shard, _ = _build_shard_for("biped", 3, 0.00625)

    warm_start, bounds = true_warm_start_from_previous_solution(
        target_shard,
        source["qpos"],
        source["ctrl"],
    )

    joint_slice = slice(4, 4 + bounds.joint_count)
    assert warm_start.report["target_r2_stance_distance_to_previous_solution"]["joint_abs_max"] > 0.14
    assert warm_start.report["target_r2_stance_reanchoring_forbidden"]
    assert warm_start.report["joint_bounds_policy"]["target_r2_stance_centered_plus_minus_0p08_rad"] is False
    assert list(warm_start.solver_start_vector[joint_slice]) == list(
        warm_start.physical_clipped_vector[joint_slice]
    )


def test_solver_search_failure_is_not_promoted_to_physical_infeasible() -> None:
    summary = {
        "combined_source_records": 8,
        "combined_strict_contract_passed": 5,
        "infeasibility_certificates": {
            "kinematic_double_support_infeasible": 0,
            "wrench_or_actuation_infeasible": 0,
        },
    }

    decision = _decide(summary)

    assert decision["status"] == "r4a31a_still_incomplete_enter_fixed_contact_mode_solver"
    assert "do not modify env, controller, generator, kp/kv" in decision["next_allowed_work"].lower()
    assert SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE == "search_exhausted_without_certificate"


def test_artifact_asserts_all_continuation_artificial_warm_start_clips_are_zero() -> None:
    if not _R4A31A_ARTIFACT.exists():
        pytest.fail("Task067 R4a.3.1a artifact is required for the clip-zero assertion")

    payload = json.loads(_R4A31A_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["assertions"]["all_continuation_step_artificial_warm_start_clip_zero"]
    assert payload["assertions"]["artificial_warm_start_clip_violations"] == []
    assert artificial_warm_start_clip_violations(payload) == []
