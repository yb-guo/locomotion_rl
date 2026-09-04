from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from h200_locomotion_lab.robots.procedural_morphology import (
    PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
    PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
)
from h200_locomotion_lab.tools.whole_body_bounded_feedback_diagnosis import (
    _TASK067_R4A2_HISTORICAL_CONTRACT_HASH,
    _TASK067_R4A2_HISTORICAL_CONTRACT_VERSION,
    _build_shard,
    _build_shard_with_replay_binding,
    _load_feasible_records,
)
from h200_locomotion_lab.tools.whole_body_dynamic_balance_diagnosis import (
    _center_of_mass,
    _contact_report,
    _reset_to_qpos,
)
from h200_locomotion_lab.tools.whole_body_equilibrium_audit import (
    _state_snapshot,
    refine_actual_equilibrium,
    strict_actual_equilibrium,
)

_R4A2_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a2_joint_aware_equilibrium_diagnosis_4x2.json"
)


def _seed_zero_record() -> dict:
    if not _R4A2_ARTIFACT.exists():
        pytest.fail("Task067 R4a.2 evidence artifact is required, not optional")
    return next(
        item
        for item in _load_feasible_records(_R4A2_ARTIFACT, family="biped")
        if float(item["range_fraction"]) == 0.0 and int(item["seed"]) == 0
    )


def test_r4a2_feasible_label_is_not_a_strict_actual_equilibrium() -> None:
    pytest.importorskip("mujoco")
    record = _seed_zero_record()
    shard = _build_shard(record)
    best = record["contact_equilibrium"]["best"]
    snapshot = _state_snapshot(shard, best["qpos"], best["ctrl"])

    assert record["contact_equilibrium"]["status"] == "feasible"
    assert not strict_actual_equilibrium(snapshot)
    assert snapshot["root_qacc_norm"] > 0.1
    assert snapshot["com_minus_cop_xy"] is not None
    assert snapshot["com_minus_cop_xy"][0] < -0.003


def test_task067_historical_replay_contract_is_explicitly_bound_to_current_runtime() -> None:
    pytest.importorskip("mujoco")
    record = _seed_zero_record()

    _, binding = _build_shard_with_replay_binding(record)
    manifest = binding.manifest()

    assert manifest["replay_mode"] == "task067_r4a2_historical_artifact_replay"
    assert manifest["source_contract_version"] == _TASK067_R4A2_HISTORICAL_CONTRACT_VERSION
    assert manifest["source_contract_hash"] == _TASK067_R4A2_HISTORICAL_CONTRACT_HASH
    assert manifest["runtime_contract_version"] == PROCEDURAL_EMBODIMENT_CONTRACT_VERSION
    assert manifest["runtime_contract_hash"] == PROCEDURAL_EMBODIMENT_CONTRACT_HASH
    assert manifest["source_matches_runtime"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("embodiment_contract_version", "procedural_whole_body_unknown_contract"),
        ("embodiment_contract_hash", "0" * 64),
    ),
)
def test_task067_historical_replay_rejects_unknown_or_wrong_contract(
    field: str,
    value: str,
) -> None:
    pytest.importorskip("mujoco")
    record = deepcopy(_seed_zero_record())
    record["morphology_instance_key"][field] = value

    with pytest.raises(ValueError, match="unsupported embodiment contract"):
        _build_shard(record)


@pytest.mark.parametrize("field", ("blueprint_hash", "physical_hash"))
def test_task067_historical_replay_rejects_geometry_or_physical_tamper(field: str) -> None:
    pytest.importorskip("mujoco")
    record = deepcopy(_seed_zero_record())
    record["morphology_instance_key"][field] = "f" * 64

    with pytest.raises(ValueError, match=field):
        _build_shard(record)


def test_root_xy_translation_is_not_a_com_support_offset_probe() -> None:
    pytest.importorskip("mujoco")
    record = _seed_zero_record()
    shard = _build_shard(record)
    best = record["contact_equilibrium"]["best"]
    qpos = shard.np.asarray(best["qpos"], dtype=shard.np.float64)
    ctrl = shard.np.asarray(best["ctrl"], dtype=shard.np.float64)

    def snapshot(offset_x: float) -> tuple:
        translated = qpos.copy()
        translated[0] += offset_x
        _reset_to_qpos(shard, shard.data[0], translated)
        shard.data[0].ctrl[:] = ctrl
        shard.mujoco.mj_forward(shard.model, shard.data[0])
        com = _center_of_mass(shard, shard.data[0])
        contact = _contact_report(shard, shard.data[0])
        cop = contact["center_of_pressure_xy"]
        assert cop is not None
        return (
            shard.np.asarray(shard.data[0].qacc).copy(),
            shard.np.asarray([com[0] - cop[0], com[1] - cop[1]]),
        )

    qacc_zero, com_cop_zero = snapshot(0.0)
    qacc_offset, com_cop_offset = snapshot(0.01)
    assert shard.np.allclose(qacc_zero, qacc_offset, atol=1e-9, rtol=1e-9)
    assert shard.np.allclose(com_cop_zero, com_cop_offset, atol=1e-9, rtol=1e-9)


def test_actual_qacc_refinement_closes_the_seed_zero_equilibrium_gap() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("scipy")
    record = _seed_zero_record()
    shard = _build_shard(record)
    result = refine_actual_equilibrium(shard, record["contact_equilibrium"], max_nfev=1500)

    assert result["status"] == "feasible"
    assert result["best"]["snapshot"]["root_qacc_norm"] <= 1e-5
    assert result["best"]["snapshot"]["joint_qacc_max"] <= 1e-4
