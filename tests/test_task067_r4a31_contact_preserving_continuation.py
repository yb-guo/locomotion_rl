from __future__ import annotations

from pathlib import Path

import pytest

from h200_locomotion_lab.tools.whole_body_contact_preserving_continuation import (
    SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE,
    SINGLE_SUPPORT_EQUILIBRIUM_ONLY_FOUND,
    _build_shard_for,
    _choose_attempt_result,
    _load_r4a3_records,
    _step_deltas,
    classify_snapshot_without_certificates,
    warm_start_mapping_manifest,
)

_R4A3_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a3_strict_equilibrium_coverage_4x2.json"
)


def test_same_topology_warm_start_mapping_matches_across_range_fraction() -> None:
    pytest.importorskip("mujoco")
    source_shard, _ = _build_shard_for("biped", 0, 0.0)
    target_shard, _ = _build_shard_for("biped", 0, 0.5)

    mapping = warm_start_mapping_manifest(source_shard, target_shard)

    assert mapping["same_topology"]
    assert mapping["joint_names_match"]
    assert mapping["semantic_slots_match"]
    assert mapping["actuator_names_match"]
    assert mapping["source_nq"] == mapping["target_nq"]
    assert mapping["source_nu"] == mapping["target_nu"]


def test_continuation_bisection_schedule_is_reproducible() -> None:
    assert _step_deltas(0.0, 0.5, 0.05, 0.00625) == [0.05, 0.025, 0.0125, 0.00625]
    assert _step_deltas(0.475, 0.5, 0.05, 0.00625) == [0.025, 0.0125, 0.00625]
    assert _step_deltas(0.5, 0.0, 0.05, 0.00625) == [0.05, 0.025, 0.0125, 0.00625]


def test_rf05_seed0_exact_single_support_qacc_zero_is_not_strict() -> None:
    if not _R4A3_ARTIFACT.exists():
        pytest.fail("Task067 R4a.3 artifact is required for R4a.3.1 regression")

    records = _load_r4a3_records(_R4A3_ARTIFACT, family="biped")
    record = next(
        item
        for item in records
        if int(item["seed"]) == 0 and float(item["range_fraction"]) == 0.5
    )
    snapshot = record["strict_refinement"]["best"]["snapshot"]

    assert snapshot["root_qacc_norm"] <= 1e-5
    assert snapshot["joint_qacc_max"] <= 1e-4
    assert not snapshot["double_support"]
    assert classify_snapshot_without_certificates(snapshot) == SINGLE_SUPPORT_EQUILIBRIUM_ONLY_FOUND


def test_search_failure_is_not_promoted_to_physical_infeasible_without_certificate() -> None:
    fake_attempt = {
        "phase": "qacc_only_continuation",
        "strict_gate_passed": False,
        "classification": SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE,
    }

    _, selected = _choose_attempt_result(fake_attempt, None)

    assert selected["classification"] == SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE
