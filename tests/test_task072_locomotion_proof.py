from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import numpy as np
import pytest

from h200_locomotion_lab.algorithms.ppo import (
    _adaptive_lr_factor,
    _joint_gaussian_kl,
    _masked_tensor_mean,
    ppo_update,
)
from h200_locomotion_lab.core.whole_body import WholeBodyStep
from h200_locomotion_lab.masked_distribution import (
    masked_raw_gaussian_log_prob,
    masked_tanh_gaussian_log_prob,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py"
SPEC = importlib.util.spec_from_file_location("task072_locomotion_proof", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TASK072 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TASK072
SPEC.loader.exec_module(TASK072)

REPAIR_SCRIPT = ROOT / ".agent/task/task072-bound-g1-go2-locomotion-proof/task072_e3a_repair_verifier.py"
REPAIR_SPEC = importlib.util.spec_from_file_location("task072_e3a_repair_verifier", REPAIR_SCRIPT)
assert REPAIR_SPEC is not None and REPAIR_SPEC.loader is not None
REPAIR = importlib.util.module_from_spec(REPAIR_SPEC)
REPAIR_SPEC.loader.exec_module(REPAIR)

MJLAB_RUNNER_SCRIPT = ROOT / ".agent/task/task072-bound-g1-go2-locomotion-proof/task072_mjlab_contact_runner.py"
MJLAB_RUNNER_SPEC = importlib.util.spec_from_file_location("task072_mjlab_contact_runner", MJLAB_RUNNER_SCRIPT)
assert MJLAB_RUNNER_SPEC is not None and MJLAB_RUNNER_SPEC.loader is not None
MJLAB_RUNNER = importlib.util.module_from_spec(MJLAB_RUNNER_SPEC)
MJLAB_RUNNER_SPEC.loader.exec_module(MJLAB_RUNNER)


def _passing_gate_metrics() -> dict[str, object]:
    return {
        "zero_fall_ratio": 1.0,
        "planar_velocity_error": 0.10,
        "yaw_error": 0.10,
        "gravity_xy": 0.10,
        "nonfall_forward_displacement_mean": 4.0,
        "zero_baseline_planar_velocity_error": 0.50,
        "update0_baseline_planar_velocity_error": 0.45,
        "zero_action_common_prefix_planar_margin": 0.20,
        "update0_common_prefix_planar_margin": 0.15,
        "zero_action_forward_displacement_margin": 3.0,
        "update0_forward_displacement_margin": 2.5,
        "finite": True,
        "checkpoint_verified": True,
        "progression_verified": True,
        "paired_baselines_verified": True,
        "video_verified": True,
        "final_eval_configuration_verified": True,
    }


def test_quality_gate_is_fail_closed() -> None:
    passing = _passing_gate_metrics()
    assert TASK072.quality_gate(passing) == (True, [])

    for key, value in (
        ("zero_fall_ratio", 0.90),
        ("planar_velocity_error", 0.36),
        ("yaw_error", float("nan")),
        ("nonfall_forward_displacement_mean", 0.0),
        ("video_verified", False),
        ("update0_common_prefix_planar_margin", 0.049),
        ("zero_action_forward_displacement_margin", 1.99),
    ):
        candidate = dict(passing)
        candidate[key] = value
        passed, reasons = TASK072.quality_gate(candidate)
        assert not passed
        assert reasons


def test_mjlab_runtime_binding_uses_explicit_29_joint_mapping() -> None:
    mapping = MJLAB_RUNNER.SEMANTIC_TO_ANON_JOINT
    assert len(mapping) == 29
    assert len(set(mapping.values())) == 29
    assert mapping["limb0_ankle_pitch"] == "anon_limb0_ankle_pitch_link_joint"
    assert mapping["right_arm_wrist_yaw"] == "anon_right_arm_wrist_yaw_link_joint"


def test_mjlab_runtime_binding_material_contract_matches_unitree_g1_flat() -> None:
    assert MJLAB_RUNNER.LINEAGE_ID == "mjlab_g1_7capsule_task_v3_single_ground"
    assert MJLAB_RUNNER.CONTACT_PROFILE_ID == "mjlab_g1_7capsule_task_v2"
    assert MJLAB_RUNNER._runtime_material_contract() == {
        "foot": {
            "condim": 3,
            "priority": 1,
            "nominal_sliding_friction": 0.6,
            "mujoco_friction": [0.6, 0.005, 0.0001],
            "contype": 1,
            "conaffinity": 1,
        },
        "non_foot_collision": {"condim": 1, "contype": 1, "conaffinity": 1},
        "logical_feet": 2,
        "capsules_per_foot": 7,
        "legacy_foot_boxes": 0,
    }


def test_mjlab_runtime_spec_removes_floor_without_mutating_asset() -> None:
    import hashlib
    before = hashlib.sha256(MJLAB_RUNNER.ASSET_XML.read_bytes()).hexdigest()
    xml = MJLAB_RUNNER.runtime_spec_xml()
    after = hashlib.sha256(MJLAB_RUNNER.ASSET_XML.read_bytes()).hexdigest()
    assert before == after
    root = __import__("xml.etree.ElementTree", fromlist=["ElementTree"]).fromstring(xml)
    assert not any(geom.get("name") == "floor" for geom in root.find("worldbody").findall("geom"))


def test_mjlab_ground_plane_checker_is_single_ground_fail_closed() -> None:
    import mujoco
    geoms = "".join(
        f'<geom name="{name}" type="sphere" size="0.02" pos="0 0 0.01" contype="1" conaffinity="1"/>'
        for name in MJLAB_RUNNER.FOOT_GEOMS
    )
    def audit(extra: str) -> dict[str, object]:
        model = mujoco.MjModel.from_xml_string(
            f'<mujoco><worldbody><geom name="terrain" type="plane" size="2 2 0.1"/>{extra}<body><joint type="free"/>{geoms}</body></worldbody></mujoco>'
        )
        return MJLAB_RUNNER._ground_plane_audit(model, mujoco.MjData(model))
    assert all(audit("")["checks"].values())
    assert not audit('<geom name="floor" type="plane" size="2 2 0.1"/>')["checks"]["exactly_one_collision_enabled_plane"]
    half_enabled = audit(
        '<geom name="floor" type="plane" size="2 2 0.1" contype="0" conaffinity="1"/>'
    )
    assert not half_enabled["checks"]["exactly_one_collision_enabled_plane"]


def test_mjlab_runner_capacity_defaults_require_4096_equivalence() -> None:
    args = MJLAB_RUNNER.parse_args(["capacity-smoke", "--output", "out.json"])
    assert args.candidates == [2048, 4096, 6144]
    train = MJLAB_RUNNER.parse_args(["one-update-train", "--run-dir", "run"])
    assert train.num_envs == 4096
    assert train.rollout_steps == 24
    assert train.capacity_artifact == MJLAB_RUNNER.RUNTIME_BINDING_ROOT / "capacity_smoke_2048_4096_6144.json"
    assert MJLAB_RUNNER.REQUIRED_TRANSITIONS_PER_UPDATE == 4096 * 24


def test_mjlab_defaults_are_v3_single_ground_paths() -> None:
    assert MJLAB_RUNNER.DEFAULT_OUTPUT_ROOT == MJLAB_RUNNER.RUNTIME_BINDING_ROOT
    assert MJLAB_RUNNER.DEFAULT_OUTPUT_ROOT.is_relative_to(MJLAB_RUNNER.RUNTIME_BINDING_ROOT)
    assert MJLAB_RUNNER.RUNTIME_BINDING_ROOT.name == "mjlab_g1_7capsule_task_v3_single_ground"
    assert "mjlab_contact_training/g1" not in str(MJLAB_RUNNER.DEFAULT_OUTPUT_ROOT)
    assert MJLAB_RUNNER.parse_args(["registration-smoke"]).output.is_relative_to(MJLAB_RUNNER.RUNTIME_BINDING_ROOT)
    assert MJLAB_RUNNER.parse_args(["capacity-smoke"]).output.is_relative_to(MJLAB_RUNNER.RUNTIME_BINDING_ROOT)
    assert MJLAB_RUNNER.parse_args(["one-update-train"]).run_dir.is_relative_to(MJLAB_RUNNER.RUNTIME_BINDING_ROOT)


def test_mjlab_capacity_evidence_is_fail_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "capacity.json"
    valid = {
        "passed": True,
        "lineage_id": MJLAB_RUNNER.LINEAGE_ID,
        "required_capacity": {
            "num_envs": 4096,
            "rollout_steps_per_env": 24,
            "transitions_per_update": 4096 * 24,
        },
        "selected": {
            "num_envs": 4096,
            "rollout_steps_per_env": 24,
            "transitions_per_update": 4096 * 24,
            "passed": True,
        },
        "gpu_lock": {"held_by_ancestor": True},
    }
    artifact.write_text(json.dumps(valid), encoding="utf-8")
    loaded = MJLAB_RUNNER._load_capacity_evidence(artifact, num_envs=4096, rollout_steps=24)
    assert all(loaded["consumption_checks"].values())

    selected_lineage = copy.deepcopy(valid)
    del selected_lineage["lineage_id"]
    selected_lineage["selected"]["lineage_id"] = MJLAB_RUNNER.LINEAGE_ID
    artifact.write_text(json.dumps(selected_lineage), encoding="utf-8")
    loaded = MJLAB_RUNNER._load_capacity_evidence(artifact, num_envs=4096, rollout_steps=24)
    assert loaded["consumption_checks"]["lineage"] is True

    old_runtime = copy.deepcopy(valid)
    old_runtime["lineage_id"] = MJLAB_RUNNER.CONTACT_PROFILE_ID
    artifact.write_text(json.dumps(old_runtime), encoding="utf-8")
    with pytest.raises(ValueError, match="capacity evidence failed checks"):
        MJLAB_RUNNER._load_capacity_evidence(artifact, num_envs=4096, rollout_steps=24)

    for path, value in (
        (("gpu_lock", "held_by_ancestor"), False),
        (("selected", "num_envs"), 2048),
        (("required_capacity", "rollout_steps_per_env"), 12),
    ):
        forged = copy.deepcopy(valid)
        target = forged
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        artifact.write_text(json.dumps(forged), encoding="utf-8")
        with pytest.raises(ValueError, match="capacity evidence failed checks"):
            MJLAB_RUNNER._load_capacity_evidence(artifact, num_envs=4096, rollout_steps=24)


def test_mjlab_gpu_lock_required_for_cuda_only() -> None:
    assert MJLAB_RUNNER._device_requires_gpu_lock("cuda:0") is True
    assert MJLAB_RUNNER._device_requires_gpu_lock("CUDA") is True
    assert MJLAB_RUNNER._device_requires_gpu_lock("cpu") is False


def test_repaired_smoke_verifier_accepts_existing_evidence(tmp_path: Path) -> None:
    base = TASK072.TASK_DIR / "artifacts/nominal_v4/unitree_g1/E3a_mjlab_kl_repair"
    args = SimpleNamespace(run_manifest=base / "smoke/run_manifest.json", no_update_gate=base / "no_update_correctness_gate.json", output=tmp_path / "r4.json")
    assert REPAIR.verify_smoke(args) == 0
    assert __import__("json").loads(args.output.read_text())["r4_repaired_smoke_passed"] is True


def test_repaired_smoke_verifier_rejects_tampered_binding(tmp_path: Path) -> None:
    import shutil
    gate = TASK072.TASK_DIR / "artifacts/nominal_v4/unitree_g1/E3a_mjlab_kl_repair/no_update_correctness_gate.json"
    tampered = tmp_path / "no_update.json"
    shutil.copyfile(gate, tampered)
    payload = __import__("json").loads(tampered.read_text())
    payload["r0"]["fixed_artifacts"]["e2_gate"]["sha256"] = "0" * 64
    tampered.write_text(__import__("json").dumps(payload))
    base = TASK072.TASK_DIR / "artifacts/nominal_v4/unitree_g1/E3a_mjlab_kl_repair"
    args = SimpleNamespace(run_manifest=base / "smoke/run_manifest.json", no_update_gate=tampered, output=tmp_path / "r4.json")
    assert REPAIR.verify_smoke(args) != 0
    assert __import__("json").loads(args.output.read_text())["r4_repaired_smoke_passed"] is False


def test_repaired_smoke_gate_binds_full_lineage_and_parameter_delta() -> None:
    payload = REPAIR.check_smoke(REPAIR.R4_MANIFEST_PATH, REPAIR.NO_UPDATE_PATH)
    manifest = json.loads(REPAIR.R4_MANIFEST_PATH.read_text(encoding="utf-8"))
    sources = manifest["static_lineage"]["sources"]
    assert manifest["source_sha256"] == REPAIR.payload_sha(sources)
    assert manifest["source_sha256"] != sources["task072_cli"]["sha256"]
    assert set(payload["controlled_sources"]) == set(REPAIR.SOURCE_PATHS)
    assert payload["old_rejected_E3a_preserved"] is True
    assert payload["random_initialization_matches_e2"] is True
    assert payload["parameter_update"]["parameter_count_compared"] == 243803
    assert payload["parameter_update"]["delta_max_abs"] == pytest.approx(
        0.0006282080430537462,
        abs=0.0,
    )
    assert payload["parameter_update"]["delta_l2"] == pytest.approx(
        0.19559913081391125,
        abs=0.0,
    )


def test_repaired_r4_gate_requires_entire_recomputed_payload() -> None:
    payload = REPAIR.check_smoke(REPAIR.R4_MANIFEST_PATH, REPAIR.NO_UPDATE_PATH)
    assert REPAIR.validate_r4_gate_payload(payload) == payload
    assert REPAIR.require_r4_gate(REPAIR.R4_GATE_PATH) == payload
    for path, value in (
        (("old_rejected_E3a_preserved",), False),
        (("claim_boundary", "r5_training_started"), True),
        (("artifact_sha256", "progression"), "0" * 64),
    ):
        forged = copy.deepcopy(payload)
        target = forged
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match="deterministic recomputation"):
            REPAIR.validate_r4_gate_payload(forged)


def _valid_optimizer_reports(*, updates: int, minibatches_per_epoch: int) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    count = 4 * minibatches_per_epoch
    for update in range(1, updates + 1):
        minibatches: list[dict[str, object]] = []
        for epoch in range(4):
            for index in range(minibatches_per_epoch):
                first = epoch == 0 and index == 0
                minibatches.append(
                    {
                        "applied": True,
                        "epoch": epoch,
                        "index": index,
                        "scheduler_kl": 0.0 if first else 0.01,
                        "scheduler_decision": "hold",
                        "learning_rate_before": 0.0001,
                        "learning_rate_after": 0.0001,
                        "approx_kl": 0.0 if first else 0.001,
                        "clip_fraction": 0.0 if first else 0.01,
                        "same_policy_identity_error": 0.0,
                    }
                )
        reports.append(
            {
                "global_update": update,
                "env_steps": update * 2048,
                "epochs_completed": 4,
                "minibatches_attempted": count,
                "minibatches_completed": count,
                "early_stopped": False,
                "desired_kl": 0.01,
                "approx_kl": sum(float(item["approx_kl"]) for item in minibatches) / count,
                "clip_fraction": sum(float(item["clip_fraction"]) for item in minibatches) / count,
                "scheduler_kl": 0.01,
                "scheduler_decision": "hold",
                "learning_rate_before": 0.0001,
                "learning_rate_after": 0.0001,
                "learning_rate": 0.0001,
                "minibatches": minibatches,
            }
        )
    return reports


def test_repaired_optimizer_validates_full_1000_by_32_telemetry() -> None:
    reports = _valid_optimizer_reports(updates=1000, minibatches_per_epoch=8)
    minibatches = REPAIR.validate_reports(
        reports,
        updates=1000,
        minibatches_per_epoch=8,
        env_steps_per_update=2048,
        initial_learning_rate=0.0001,
    )
    assert len(minibatches) == 32000
    metrics = REPAIR.compute_metrics(minibatches)
    assert REPAIR.metrics_pass(metrics) is True
    assert REPAIR.THRESHOLDS == {
        "approx_kl_mean": 0.015,
        "approx_kl_p95": 0.03,
        "approx_kl_max": 0.05,
        "clip_fraction_mean": 0.20,
        "clip_fraction_p95": 0.35,
    }


def test_repaired_optimizer_telemetry_mutations_fail_closed() -> None:
    base = _valid_optimizer_reports(updates=1, minibatches_per_epoch=2)
    mutations = []
    identity = copy.deepcopy(base)
    identity[0]["minibatches"][1]["same_policy_identity_error"] = -1.0
    mutations.append(identity)
    decision = copy.deepcopy(base)
    decision[0]["minibatches"][1]["scheduler_decision"] = "increase"
    mutations.append(decision)
    negative_kl = copy.deepcopy(base)
    negative_kl[0]["minibatches"][1]["approx_kl"] = -0.1
    mutations.append(negative_kl)
    bad_clip = copy.deepcopy(base)
    bad_clip[0]["minibatches"][1]["clip_fraction"] = 1.1
    mutations.append(bad_clip)
    bad_layout = copy.deepcopy(base)
    bad_layout[0]["minibatches"][1]["index"] = 0
    mutations.append(bad_layout)
    bad_lr = copy.deepcopy(base)
    bad_lr[0]["minibatches"][1]["learning_rate_before"] = 0.0002
    mutations.append(bad_lr)
    boolean = copy.deepcopy(base)
    boolean[0]["minibatches"][1]["scheduler_kl"] = True
    mutations.append(boolean)
    for reports in mutations:
        with pytest.raises(ValueError):
            REPAIR.validate_reports(
                reports,
                updates=1,
                minibatches_per_epoch=2,
                env_steps_per_update=2048,
                initial_learning_rate=0.0001,
            )


def test_repaired_optimizer_approximate_kl_zero_point_tolerance_is_narrow() -> None:
    tiny = _valid_optimizer_reports(updates=1, minibatches_per_epoch=2)
    tiny[0]["minibatches"][1]["approx_kl"] = -5e-9
    tiny[0]["approx_kl"] = sum(m["approx_kl"] for m in tiny[0]["minibatches"]) / len(tiny[0]["minibatches"])
    assert len(REPAIR.validate_reports(tiny, updates=1, minibatches_per_epoch=2, env_steps_per_update=2048, initial_learning_rate=0.0001)) == 8
    assert REPAIR.compute_metrics([{"approx_kl": -5e-9, "clip_fraction": 0.0}])["approx_kl_mean"] == pytest.approx(-5e-9)
    material = copy.deepcopy(tiny)
    material[0]["minibatches"][1]["approx_kl"] = -2e-8
    with pytest.raises(ValueError, match="negative approximate KL"):
        REPAIR.validate_reports(material, updates=1, minibatches_per_epoch=2, env_steps_per_update=2048, initial_learning_rate=0.0001)


def test_repaired_optimizer_approx_aggregate_tolerance_does_not_widen_clip() -> None:
    reports = _valid_optimizer_reports(updates=1, minibatches_per_epoch=2)
    reports[0]["approx_kl"] += 5e-9
    assert REPAIR.validate_reports(reports, updates=1, minibatches_per_epoch=2, env_steps_per_update=2048, initial_learning_rate=0.0001)
    reports = _valid_optimizer_reports(updates=1, minibatches_per_epoch=2)
    reports[0]["approx_kl"] += 2e-8
    with pytest.raises(ValueError, match="approximate KL aggregate"):
        REPAIR.validate_reports(reports, updates=1, minibatches_per_epoch=2, env_steps_per_update=2048, initial_learning_rate=0.0001)
    reports = _valid_optimizer_reports(updates=1, minibatches_per_epoch=2)
    reports[0]["clip_fraction"] += 2e-9
    with pytest.raises(ValueError, match="clip aggregate"):
        REPAIR.validate_reports(reports, updates=1, minibatches_per_epoch=2, env_steps_per_update=2048, initial_learning_rate=0.0001)


def test_repaired_r5_policy_parameter_count_guard_fails_closed() -> None:
    valid = {
        "nonzero": True,
        "parameter_count_compared": REPAIR.EXPECTED_POLICY_PARAMETER_COUNT,
    }
    assert REPAIR.validate_policy_update(valid, stage="R5") is valid

    wrong_count = dict(valid, parameter_count_compared=REPAIR.EXPECTED_POLICY_PARAMETER_COUNT - 1)
    with pytest.raises(ValueError, match="R5 policy parameter count mismatch"):
        REPAIR.validate_policy_update(wrong_count, stage="R5")


def test_repaired_optimizer_config_diff_is_exact_and_missing_is_not_null() -> None:
    e2 = json.loads((REPAIR.E2_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    repaired = REPAIR.repaired_config(e2["configuration"], stage="pilot")
    assert set(REPAIR.config_diff(e2["configuration"], repaired)) == REPAIR.CONFIG_DIFF_PATHS
    missing = {"value": None}
    assert REPAIR.config_diff(missing, {}) == ["configuration.value"]
    assert REPAIR.config_diff(missing, {"value": None}) == []


def test_repaired_manifest_rejects_run_identity_and_source_drift() -> None:
    no_update, e2, _ = REPAIR.load_fixed_evidence(REPAIR.NO_UPDATE_PATH)
    manifest = json.loads(REPAIR.R4_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = REPAIR.repaired_config(e2["configuration"], stage="smoke")
    REPAIR.validate_manifest(
        manifest,
        expected_config=expected,
        e2_config=e2["configuration"],
        no_update=no_update,
    )
    run_identity = copy.deepcopy(manifest)
    run_identity["run_identity"]["case"] = "forged"
    with pytest.raises(ValueError, match="run identity"):
        REPAIR.validate_manifest(
            run_identity,
            expected_config=expected,
            e2_config=e2["configuration"],
            no_update=no_update,
        )
    source = copy.deepcopy(manifest)
    source["static_lineage"]["sources"]["environment"]["sha256"] = "0" * 64
    source["static_lineage_sha256"] = REPAIR.payload_sha(source["static_lineage"])
    source["run_identity"]["static_lineage_sha256"] = source["static_lineage_sha256"]
    source["run_identity_sha256"] = REPAIR.payload_sha(source["run_identity"])
    with pytest.raises(ValueError, match="source environment SHA drift"):
        REPAIR.validate_manifest(
            source,
            expected_config=expected,
            e2_config=e2["configuration"],
            no_update=no_update,
        )


def test_repaired_optimizer_threshold_boundaries_are_unchanged() -> None:
    assert REPAIR.metrics_pass(dict(REPAIR.THRESHOLDS)) is True
    for key, boundary in REPAIR.THRESHOLDS.items():
        candidate = dict(REPAIR.THRESHOLDS)
        candidate[key] = boundary + 1e-12
        assert REPAIR.metrics_pass(candidate) is False


def test_repaired_optimizer_gate_handles_current_r5_state(tmp_path: Path) -> None:
    args = SimpleNamespace(
        run_manifest=REPAIR.R5_MANIFEST_PATH,
        r4_gate=REPAIR.R4_GATE_PATH,
        output=tmp_path / "optimizer_gate.json",
    )
    result = REPAIR.verify_optimizer(args)
    assert result in (0, 1)
    payload = json.loads(args.output.read_text(encoding="utf-8"))
    assert "r2_e3a_optimizer_gate_passed" in payload


def test_e1_phase_clock_reset_and_quarter_cycle() -> None:
    phase = TASK072.phase_from_trial_step
    values = [
        (0, (0.0, 1.0)),
        (10, (1.0, 0.0)),
        (20, (0.0, -1.0)),
        (30, (-1.0, 0.0)),
    ]
    for trial_step, expected in values:
        angle = 2.0 * np.pi * float(phase(trial_step, 50.0, 0.8))
        observed = (float(np.sin(angle)), float(np.cos(angle)))
        assert np.allclose(observed, expected, atol=1e-12)
        assert 0.0 <= float(phase(trial_step, 50.0, 0.8)) < 1.0


def test_e2_cli_and_parent_metadata_contract() -> None:
    args = TASK072._parser().parse_args(
        ["train", "--case", "unitree_g1", "--stage", "smoke", "--variant", "E2_reward_dt"]
    )
    assert args.variant == "E2_reward_dt"
    parent = TASK072.TASK_DIR / "artifacts/nominal_v4/unitree_g1/E1_phase/run_manifest.json"
    manifest = __import__("json").loads(parent.read_text(encoding="utf-8"))
    assert manifest["variant_id"] == "E1_phase"
    assert TASK072._parent_config_sha256("E2_reward_dt") == TASK072.payload_sha256(
        manifest["configuration"]
    )


def test_e3a_cli_and_adaptive_schedule_increase() -> None:
    torch = pytest.importorskip("torch")
    args = TASK072._parser().parse_args(
        ["train", "--case", "unitree_g1", "--stage", "smoke", "--variant", "E3a_adaptive_kl"]
    )
    assert args.variant == "E3a_adaptive_kl"
    from h200_locomotion_lab.policies.whole_body_mlp import WholeBodyMLPActorCritic, WholeBodyMLPConfig

    model = WholeBodyMLPActorCritic(
        WholeBodyMLPConfig(obs_dim=1, action_dim=1, hidden_dim=4, hidden_layers=1),
        action_mask=np.ones(1, dtype=bool),
    )
    config = SimpleNamespace(
        epochs=1, minibatch_size=4, gamma=0.99, gae_lambda=0.95, clip=0.2,
        value_coef=0.5, entropy_coef=0.0, max_grad_norm=1.0, target_kl=None,
        hard_kl_stop=False, adaptive_kl=True, desired_kl=0.01,
    )
    observations = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    with torch.no_grad():
        current_mean, current_value = model.forward(observations.reshape(-1, 1))
        old_log_std = model.log_std.expand_as(current_mean)
        old_mean = current_mean + old_log_std.exp() * (2.0 * 0.004) ** 0.5
        raw_actions = old_mean.clone()
        old_log_prob = masked_raw_gaussian_log_prob(
            raw_actions,
            old_mean,
            old_log_std,
            torch.ones_like(raw_actions, dtype=torch.bool),
        )
    batch = SimpleNamespace(
        observations=observations,
        actions=torch.tanh(raw_actions).reshape(2, 2, 1),
        raw_actions=raw_actions.reshape(2, 2, 1),
        old_means=old_mean.reshape(2, 2, 1),
        old_log_stds=old_log_std.reshape(2, 2, 1),
        log_probs=old_log_prob.reshape(2, 2),
        rewards=torch.ones((2, 2)),
        dones=torch.zeros((2, 2), dtype=torch.bool), values=current_value.reshape(2, 2),
        next_value=torch.zeros(2), active_action_mask=torch.ones((2, 2, 1), dtype=torch.bool),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    diagnostics = ppo_update(model, optimizer, batch, torch.tensor([[1.0, -1.0], [1.0, -1.0]]), torch.ones((2, 2)), config)
    assert diagnostics.scheduler_decision == "increase"
    assert diagnostics.learning_rate_before == 1e-3
    assert diagnostics.learning_rate_after == 1.5e-3
    assert 0.0 < diagnostics.scheduler_kl < 0.005
    assert diagnostics.minibatches_attempted == diagnostics.minibatches_completed == 1


def test_e3a_adaptive_kl_requires_saved_old_distribution() -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        action_mask = torch.ones(1, dtype=torch.bool)

        def __init__(self) -> None:
            super().__init__()
            self.log_std = torch.nn.Parameter(torch.zeros(1))
            self.parameter = torch.nn.Parameter(torch.zeros(()))

        def evaluate_actions(self, observations: object, _actions: object, **_kwargs: object) -> tuple[object, object, object]:
            values = observations[:, 0] * 0.0 + self.parameter
            return values, values + 1.0, values

    config = SimpleNamespace(
        epochs=1, minibatch_size=2, gamma=0.99, gae_lambda=0.95, clip=0.2,
        value_coef=0.5, entropy_coef=0.0, max_grad_norm=1.0, target_kl=None,
        hard_kl_stop=False, adaptive_kl=True, desired_kl=0.01,
    )
    batch = SimpleNamespace(
        observations=torch.zeros((1, 2, 1)), actions=torch.zeros((1, 2, 1)),
        log_probs=torch.zeros((1, 2)), rewards=torch.ones((1, 2)),
        dones=torch.zeros((1, 2), dtype=torch.bool), values=torch.zeros((1, 2)),
        next_value=torch.zeros(2), active_action_mask=torch.ones((1, 2, 1), dtype=torch.bool),
    )
    model = Model()
    with pytest.raises(ValueError, match="raw_actions and old distribution"):
        ppo_update(
            model, torch.optim.Adam(model.parameters(), lr=1e-3), batch,
            torch.zeros((1, 2)), torch.zeros((1, 2)), config,
        )


def test_e3a_raw_replay_does_not_fallback_to_squashed_likelihood() -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        action_mask = torch.ones(1, dtype=torch.bool)

        def __init__(self) -> None:
            super().__init__()
            self.log_std = torch.nn.Parameter(torch.zeros(1))
            self.parameter = torch.nn.Parameter(torch.zeros(()))

        def forward(self, observations: object) -> tuple[object, object]:
            values = observations[:, 0] * 0.0 + self.parameter
            return values[:, None], values

        def evaluate_raw_actions(self, observations: object, raw_actions: object) -> tuple[object, object, object]:
            values = observations[:, 0] * 0.0 + self.parameter
            return values, values, values

        def evaluate_actions(self, observations: object, _actions: object, **_kwargs: object) -> tuple[object, object, object]:
            values = observations[:, 0] * 0.0 + self.parameter
            return values + 99.0, values, values

    config = SimpleNamespace(
        epochs=1, minibatch_size=2, gamma=0.99, gae_lambda=0.95, clip=0.2,
        value_coef=0.5, entropy_coef=0.0, max_grad_norm=1.0, target_kl=None,
        hard_kl_stop=False, adaptive_kl=True, desired_kl=0.01,
    )
    batch = SimpleNamespace(
        observations=torch.zeros((1, 2, 1)), actions=torch.zeros((1, 2, 1)),
        raw_actions=torch.zeros((1, 2, 1)), old_means=torch.zeros((1, 2, 1)),
        old_log_stds=torch.zeros((1, 2, 1)), log_probs=torch.zeros((1, 2)),
        rewards=torch.ones((1, 2)), dones=torch.zeros((1, 2), dtype=torch.bool),
        values=torch.zeros((1, 2)), next_value=torch.zeros(2),
        active_action_mask=torch.ones((1, 2, 1), dtype=torch.bool),
    )
    model = Model()
    with pytest.raises(ValueError, match="masked evaluate_raw_actions"):
        ppo_update(
            model, torch.optim.Adam(model.parameters(), lr=1e-3), batch,
            torch.zeros((1, 2)), torch.zeros((1, 2)), config,
        )


def test_e3a_scheduler_kl_mask_average_does_not_double_divide_batch() -> None:
    torch = pytest.importorskip("torch")
    value = torch.ones((4, 3))
    mask = torch.ones((4, 3), dtype=torch.bool)
    assert float(_masked_tensor_mean(value, mask).item()) == pytest.approx(1.0)


def test_e3a_raw_likelihood_identity_and_joint_kl_semantics() -> None:
    torch = pytest.importorskip("torch")
    raw = torch.tensor([[0.2, 20.0, -20.0]])
    mean = torch.zeros_like(raw)
    log_std = torch.zeros_like(raw)
    mask = torch.tensor([[True, True, False]])
    old = masked_raw_gaussian_log_prob(raw, mean, log_std, mask)
    assert float((old - masked_raw_gaussian_log_prob(raw, mean, log_std, mask)).abs().max()) <= 1e-6
    inverse_reconstructed = masked_tanh_gaussian_log_prob(torch.tanh(raw), mean, log_std, mask)
    assert float((inverse_reconstructed - old).abs().max()) > 100.0
    joint = _joint_gaussian_kl(mean, log_std, torch.full_like(raw, 0.2), log_std, mask)
    assert float(joint) == pytest.approx(0.04)
    assert [_adaptive_lr_factor(value, 0.01)[1] for value in (0.004, 0.01, 0.021)] == ["increase", "hold", "decrease"]


def test_e3a_ppo_first_minibatch_identity_uses_raw_actions() -> None:
    torch = pytest.importorskip("torch")
    from h200_locomotion_lab.policies.whole_body_mlp import (
        WholeBodyMLPActorCritic,
        WholeBodyMLPConfig,
    )

    torch.manual_seed(72)
    action_mask = torch.zeros(45, dtype=torch.bool)
    action_mask[:29] = True
    batch_mask = action_mask.expand(2, -1)
    model = WholeBodyMLPActorCritic(
        WholeBodyMLPConfig(obs_dim=3, action_dim=45, hidden_dim=8, hidden_layers=1),
        action_mask=action_mask,
    )
    observations = torch.tensor([[0.1, 0.2, -0.3], [0.4, -0.5, 0.6]], dtype=torch.float32)
    _action, raw, _log_prob, _entropy, _value, mean, log_std = model.act_with_details(
        observations,
        active_mask=batch_mask,
    )
    raw = raw.detach().clone()
    mean = mean.detach()
    log_std = log_std.detach()
    raw[:, 0] = 20.0
    raw[:, 1] = -20.0
    old_log_prob = masked_raw_gaussian_log_prob(raw, mean, log_std, batch_mask)
    batch = SimpleNamespace(
        observations=observations.reshape(1, 2, 3),
        actions=torch.tanh(raw).reshape(1, 2, 45),
        raw_actions=raw.reshape(1, 2, 45),
        old_means=mean.reshape(1, 2, 45),
        old_log_stds=log_std.reshape(1, 2, 45),
        log_probs=old_log_prob.reshape(1, 2),
        rewards=torch.ones((1, 2)),
        dones=torch.zeros((1, 2), dtype=torch.bool),
        values=torch.zeros((1, 2)),
        next_value=torch.zeros(2),
        active_action_mask=batch_mask.reshape(1, 2, 45),
    )
    config = SimpleNamespace(
        epochs=1,
        minibatch_size=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=1.0,
        target_kl=None,
        hard_kl_stop=False,
        adaptive_kl=True,
        desired_kl=0.01,
    )
    diagnostics = ppo_update(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-4),
        batch,
        torch.tensor([[1.0, -1.0]]),
        torch.zeros((1, 2)),
        config,
    )
    first = diagnostics.minibatches[0]
    assert first.same_policy_identity_error <= 1e-5
    assert first.approx_kl == pytest.approx(0.0, abs=1e-6)
    assert first.clip_fraction == 0.0
    assert first.scheduler_kl == pytest.approx(0.0, abs=1e-6)


def test_e3a_verifier_writes_failed_gate_on_missing_manifest(tmp_path: Path) -> None:
    output = tmp_path / "e3a_gate.json"
    args = SimpleNamespace(run_manifest=tmp_path / "missing.json", output=output)
    assert TASK072.verify_e3_optimizer_command(args) == 1
    payload = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert payload["r2_e3a_optimizer_gate_passed"] is False
    assert payload["r2_next_variant_allowed"] is None


def test_e3a_kl_repair_no_update_gate_writes_current_evidence(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    output = tmp_path / "no_update_gate.json"
    args = SimpleNamespace(output=output)
    assert TASK072.verify_e3a_kl_repair_command(args) == 0
    payload = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert payload["r0_r3_passed"] is True
    assert payload["training_started"] is False
    assert payload["smoke_started"] is False
    assert payload["r1"]["regular_max_abs_log_prob_delta"] <= 1e-5
    assert payload["r1"]["saturated_max_abs_log_prob_delta"] <= 1e-5
    assert payload["r1"]["same_policy_clip_fraction"] == 0.0
    assert payload["r2"]["joint_kl_by_target"]["0.021"] == pytest.approx(0.021, abs=1e-6)
    assert payload["r2"]["scheduler_decisions"] == {
        "0.004": "increase",
        "0.01": "hold",
        "0.021": "decrease",
    }
    assert set(payload["r3"]["config_diff_paths"]) <= set(payload["r3"]["config_diff_allowlist"])
    assert payload["r3"]["original_optimizer_gate_thresholds"]["approx_kl_max"] == 0.05
    assert payload["r0"]["rejected_e3a_remains_rejected"] is True


def test_e2_gate_must_authorize_e3a(tmp_path: Path) -> None:
    import json

    manifest = tmp_path / "run_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    gate = tmp_path / "e2_gate.json"
    gate.write_text(
        json.dumps(
            {
                "variant_id": "E2_reward_dt",
                "r2_e2_scale_gate_passed": True,
                "r2_next_variant_allowed": "E3a_adaptive_kl",
                "e2_run_manifest_sha256": TASK072.sha256_path(manifest),
            }
        ),
        encoding="utf-8",
    )
    assert TASK072._require_e2_gate_authorizes_e3a(tmp_path)["r2_e2_scale_gate_passed"] is True
    gate.write_text(
        json.dumps(
            {
                "variant_id": "E2_reward_dt",
                "r2_e2_scale_gate_passed": True,
                "r2_next_variant_allowed": "E4a_roll_authority",
                "e2_run_manifest_sha256": TASK072.sha256_path(manifest),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not authorize E3a"):
        TASK072._require_e2_gate_authorizes_e3a(tmp_path)


def test_e2_reward_ratio_exact_and_e1_scale_unchanged() -> None:
    raw = np.asarray([1.25, -3.0])
    scaled = TASK072.ppo_reward_from_raw(raw, TASK072.CONTROL_DT)
    assert np.array_equal(scaled / raw, np.asarray([0.02, 0.02]))
    assert TASK072.ppo_reward_from_raw(np.asarray([0.0]), TASK072.CONTROL_DT)[0] == 0.0
    assert np.array_equal(TASK072.ppo_reward_from_raw(raw), raw)


def test_e2_scale_gate_recomputes_first_100_and_rejects_ratio_drift() -> None:
    row = {
        "grad_norm": 10.0,
        "value_loss": 0.5,
        "pre_clip_grad_norm": 1.0,
        "grad_norm_is_pre_clip": True,
        "policy_loss": 0.01,
        "raw_reward_mean": 2.0,
        "ppo_reward_mean": 0.04,
        "reward_mean": 0.04,
        "return_mean": 0.4,
        "return_std": 0.1,
        "return_p95": 0.5,
        "gae_target_mean": 0.2,
        "gae_target_std": 0.1,
        "gae_target_p95": 0.3,
        "value_prediction_mean": 0.02,
        "value_prediction_std": 0.01,
        "reward_sample_count": 1,
        "raw_reward_nonzero_count": 1,
        "reward_zero_mismatch_count": 0,
        "reward_scale_error_max": 0.0,
    }
    e1 = {"reports": [dict(row, grad_norm=10.0, value_loss=10.0) for _ in range(100)]}
    e2 = {"reports": [dict(row) for _ in range(100)]}
    passed, details = TASK072._recompute_e2_scale_gate(
        {"configuration": {}}, {"configuration": {"reward_scale": 0.02}}, e1, e2
    )
    assert passed
    assert details["e2_mean_reward_ratio"] == pytest.approx(0.02)
    e2["reports"][0]["ppo_reward_mean"] = 0.2
    passed, _details = TASK072._recompute_e2_scale_gate(
        {"configuration": {}}, {"configuration": {"reward_scale": 0.02}}, e1, e2
    )
    assert not passed
    e2 = {"reports": [dict(row) for _ in range(100)]}
    e2["reports"][0]["return_mean"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        TASK072._recompute_e2_scale_gate(
            {"configuration": {}}, {"configuration": {"reward_scale": 0.02}}, e1, e2
        )


def test_e2_config_diff_is_fail_closed_for_unallowlisted_change() -> None:
    assert TASK072._config_diff(
        {"reward_scale": 1.0, "learning_rate": 1e-4},
        {"reward_scale": 0.02, "learning_rate": 2e-4},
    ) == ["configuration.learning_rate", "configuration.reward_scale"]


def test_static_lineage_binds_task_independent_ppo_kernel() -> None:
    pytest.importorskip("mujoco")
    context = TASK072._load_bound_context("unitree_g1")
    shard = TASK072._build_shard(
        context,
        num_envs=1,
        trial_seconds=1.0,
        seed=72072,
        action_scale=0.35,
        phase_observation=True,
    )
    lineage = TASK072._static_lineage(
        "unitree_g1",
        context,
        shard,
        action_scale=0.35,
        phase_observation=True,
    )
    assert lineage["sources"]["ppo_kernel"]["path"] == "src/h200_locomotion_lab/algorithms/ppo.py"


def test_e1_observation_schema_and_legacy_trainer_dimension() -> None:
    torch = pytest.importorskip("torch")
    from h200_locomotion_lab.training.whole_body_ppo import WholeBodyPPOConfig, WholeBodyPPOTrainer

    class Env:
        active_action_mask = np.ones((1, 45), dtype=bool)

        def __init__(self, dimension: int) -> None:
            self.dimension = dimension

        def reset(self) -> np.ndarray:
            return np.zeros((1, self.dimension), dtype=np.float32)

    for dimension in (193, 195):
        trainer = WholeBodyPPOTrainer(
            Env(dimension), action_mask=np.ones((1, 45), dtype=bool),
            config=WholeBodyPPOConfig(device="cpu"),
        )
        assert trainer.observation.shape[-1] == dimension
        assert trainer.policy.config.obs_dim == dimension


def test_e1_make_policy_uses_manifest_observation_dim() -> None:
    pytest.importorskip("torch")
    shard = SimpleNamespace(active_action_mask=np.ones((1, 45), dtype=bool))
    policy = TASK072._make_policy(
        shard,
        {
            "actor_observation_dim": 195,
            "hidden_dim": 32,
            "hidden_layers": 1,
            "log_std_init": -1.0,
        },
        "cpu",
    )
    assert policy.config.obs_dim == 195


def test_e1_eval_helpers_rebuild_phase_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    shard = SimpleNamespace()

    def build_shard(_context: object, **kwargs: object) -> object:
        calls.append(bool(kwargs["phase_observation"]))
        return shard

    monkeypatch.setattr(TASK072, "_build_shard", build_shard)
    monkeypatch.setattr(
        TASK072,
        "_make_policy",
        lambda _shard, _config, _device: SimpleNamespace(load_state_dict=lambda _state: None),
    )
    monkeypatch.setattr(TASK072, "_policy_action_provider", lambda _policy, _shard, _device: object())
    monkeypatch.setattr(TASK072, "evaluate_first_trials", lambda _shard, _provider: {"ok": True})

    config = {
        "variant_id": "E1_phase",
        "action_scale": 0.35,
        "actor_observation_dim": 195,
    }
    TASK072._evaluate_policy_payload(
        {"policy": {}},
        context={},
        config=config,
        num_envs=1,
        trial_seconds=1.0,
        seed=1,
        device="cpu",
    )
    TASK072._evaluate_zero_action(
        context={},
        config=config,
        num_envs=1,
        trial_seconds=1.0,
        seed=1,
    )
    assert calls == [True, True]


def test_motor_tuple_action_amplitude_formula_and_fail_closed() -> None:
    tuples = (
        TASK072.MotorTuple(
            semantic_slot="limb0_hip_pitch",
            control_mode="builtin_position_pd",
            effective_effort=88.0,
            velocity_limit=32.0,
            kp=44.0,
            kd=2.0,
            armature=0.01,
            friction=0.2,
            transmission_group="motor_family_00",
            provenance_sha256="a" * 64,
        ),
    )
    assert TASK072.derive_position_action_amplitudes(tuples) == {
        "limb0_hip_pitch": pytest.approx(0.5)
    }
    bad = TASK072.MotorTuple(
        semantic_slot="limb0_hip_pitch",
        control_mode="torque",
        effective_effort=88.0,
        velocity_limit=None,
        kp=44.0,
        kd=2.0,
        armature=0.01,
        friction=0.2,
        transmission_group="motor_family_00",
        provenance_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="unsupported control mode"):
        TASK072.derive_position_action_amplitudes((bad,))


def test_action_contract_loads_exact_g1_and_go2_slot_counts() -> None:
    payload = TASK072.action_contract_payload(("unitree_g1", "unitree_go2"))
    by_case = {record["case_id"]: record for record in payload["cases"]}
    assert by_case["unitree_g1"]["slot_count"] == 29
    assert by_case["unitree_go2"]["slot_count"] == 12
    for record in by_case.values():
        assert all(value > 0.0 for value in record["action_amplitudes"].values())
        assert len(record["action_residual_bounds_by_slot"]) == record["slot_count"]
        assert all(
            negative > 0.0 and positive > 0.0
            for negative, positive in record["action_residual_bounds_by_slot"].values()
        )
    assert payload["formula_version"] == "motor_tuple_headroom_residual_v2"


def test_train_stage_contract_is_fixed_command_without_randomization() -> None:
    assert TASK072.TRAIN_STAGES["smoke"].transitions == 4 * 32 * 2
    assert TASK072.TRAIN_STAGES["pilot"].transitions == 32 * 64 * 1000
    assert TASK072.TRAIN_STAGES["proof"].transitions == 32 * 64 * 31200
    parser = TASK072._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--case", "unitree_g1", "--run-dir", "x"])


class _FakeShard:
    def __init__(self) -> None:
        self.num_envs = 2
        self.config = SimpleNamespace(trial_steps=3, trial_seconds=0.06, control_hz=50.0)
        self.data = [SimpleNamespace(env_id=0), SimpleNamespace(env_id=1)]
        self._step = 0
        self.actions: list[np.ndarray] = []

    def _canonical_state(self, data: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(
            world_position=(0.0, float(data.env_id), 1.0),
            world_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        )

    def reset(self) -> np.ndarray:
        self._step = 0
        return np.zeros((2, 193), dtype=np.float32)

    def step(self, action: np.ndarray) -> SimpleNamespace:
        self._step += 1
        self.actions.append(action.copy())
        positions = np.asarray(((0.1, 0.0, 0.3), (0.1 * self._step, 1.0, 1.0)), dtype=float)
        done = np.asarray((self._step == 1, self._step == 3), dtype=bool)
        fall = np.asarray((self._step == 1, False), dtype=bool)
        linear = np.asarray(((0.4, 0.0, 0.0), (0.5, 0.0, 0.0)), dtype=float)
        angular = np.zeros((2, 3), dtype=float)
        gravity = np.asarray(((0.0, 0.0, -1.0), (0.0, 0.0, -1.0)), dtype=float)
        return SimpleNamespace(
            actor_observation=np.zeros((2, 193), dtype=np.float32),
            trial_done=done,
            metrics={
                "fall": fall,
                "post_step_pre_reset_world_position": positions,
                "post_step_pre_reset_world_quaternion_wxyz": np.asarray(
                    ((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
                ),
                "post_step_pre_reset_local_linear_velocity": linear,
                "post_step_pre_reset_local_angular_velocity": angular,
                "post_step_pre_reset_projected_gravity": gravity,
            },
        )


def test_first_trial_evaluation_uses_terminal_state_and_equal_trial_weight() -> None:
    shard = _FakeShard()

    def actions(_observation: np.ndarray) -> np.ndarray:
        return np.ones((2, 45), dtype=np.float32)

    metrics = TASK072.evaluate_first_trials(shard, actions)
    assert metrics["zero_fall_ratio"] == 0.5
    assert metrics["fall_count"] == 1
    assert metrics["trial_step_counts"] == [1, 3]
    assert metrics["forward_displacement_mean"] == pytest.approx(0.2)
    assert metrics["nonfall_forward_displacement_mean"] == pytest.approx(0.3)
    assert metrics["planar_velocity_error"] == pytest.approx(0.05)
    assert np.all(shard.actions[1][0] == 0.0)
    assert np.all(shard.actions[2][0] == 0.0)


def test_checkpoint_lineage_rejects_any_drift() -> None:
    expected = {
        "case": "unitree_g1",
        "static_lineage_sha256": "a" * 64,
        "global_update": 2,
    }
    TASK072.validate_checkpoint_lineage(dict(expected), expected)
    tampered = dict(expected)
    tampered["global_update"] = 1
    with pytest.raises(ValueError, match="lineage mismatch"):
        TASK072.validate_checkpoint_lineage(tampered, expected)


def test_task_reward_prefers_upright_forward_tracking_over_static_standing() -> None:
    metrics = {
        "post_step_pre_reset_local_linear_velocity": np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))),
        "post_step_pre_reset_local_angular_velocity": np.zeros((2, 3)),
        "post_step_pre_reset_projected_gravity": np.asarray(((0.0, 0.0, -1.0), (0.0, 0.0, -1.0))),
        "post_step_pre_reset_world_quaternion_wxyz": np.asarray(
            ((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        ),
        "post_step_pre_reset_world_position": np.asarray(
            ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
        ),
        "non_foot_contact_fraction": np.zeros(2),
        "fall": np.zeros(2, dtype=bool),
        "previous_action": np.zeros((2, 45)),
    }
    step = WholeBodyStep(
        actor_observation=np.zeros((2, 193)),
        critic_observation=np.zeros((2, 193)),
        reward=np.zeros(2),
        trial_done=np.zeros(2, dtype=bool),
        context_done=np.zeros(2, dtype=bool),
        active_action_mask=np.ones((2, 45), dtype=bool),
        metrics=metrics,
    )
    shard = SimpleNamespace(
        np=np,
        blueprint=SimpleNamespace(family="quadruped"),
        data=(SimpleNamespace(), SimpleNamespace()),
        _commands=np.zeros((2, 3)),
        _canonical_state=lambda _data: SimpleNamespace(
            world_position=(0.0, 0.0, 1.0),
            world_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        active_action_mask=step.active_action_mask,
        reset=lambda: step.actor_observation,
        step=lambda _action: step,
    )
    fixed = TASK072.Task072LocomotionReward(shard)
    reward = fixed.step(np.zeros((2, 45))).reward
    assert reward[1] > reward[0] + 2.5
    assert fixed.target_vx() == 0.5
    assert set(fixed.step(np.zeros((2, 45))).metrics["reward_components"]) == {
        "track_xy",
        "track_yaw",
        "heading",
        "upright",
        "tilt",
        "nonfoot_contact",
        "action_rate",
    }
    assert "fall" not in inspect.getsource(TASK072.Task072QuadrupedReward.step).split("reward =")[1]


def test_ppo_target_kl_telemetry_preserves_unbounded_update() -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(()))

        def evaluate_actions(self, observations: object, _actions: object, **_kwargs: object) -> tuple[object, object, object]:
            values = observations[:, 0] * 0.0 + self.parameter
            return values, values + 1.0, values

    config = SimpleNamespace(
        epochs=2, minibatch_size=2, gamma=0.99, gae_lambda=0.95, clip=0.2,
        value_coef=0.5, entropy_coef=0.0, max_grad_norm=1.0, target_kl=None,
    )
    batch = SimpleNamespace(
        observations=torch.zeros((2, 2, 1)), actions=torch.zeros((2, 2, 1)),
        log_probs=torch.zeros((2, 2)), rewards=torch.ones((2, 2)),
        dones=torch.zeros((2, 2), dtype=torch.bool), values=torch.zeros((2, 2)),
        next_value=torch.zeros(2),
    )
    model = Model()
    diagnostics = ppo_update(
        model, torch.optim.Adam(model.parameters(), lr=1e-3), batch,
        torch.ones((2, 2)), torch.ones((2, 2)), config,
    )
    assert not diagnostics.early_stopped
    assert diagnostics.minibatches_attempted == diagnostics.minibatches_completed == 4
    assert diagnostics.epochs_completed == 2
    assert all(record.applied for record in diagnostics.minibatches)


def test_biped_reward_foot_contact_reward_components_no_fall_penalty() -> None:
    pytest.importorskip("mujoco")
    context = TASK072._load_bound_context("unitree_g1")
    shard = TASK072._build_shard(
        context,
        num_envs=2,
        trial_seconds=1.0,
        seed=72072,
        action_scale=0.35,
    )
    reward = TASK072.Task072BipedReward(shard)
    step = reward.step(np.zeros((2, 45), dtype=np.float32))
    components = step.metrics["reward_components"]
    assert {
        "track_xy",
        "track_yaw",
        "upright",
        "tilt",
        "height",
        "stand_support",
        "phase_gait",
        "out_of_phase_double_support",
        "clearance",
        "touchdown_airtime",
        "soft_landing",
        "foot_slip",
        "nonfoot_contact",
        "pose_hip",
        "pose_knee",
        "pose_ankle",
        "pose_waist",
        "pose_arm_wrist",
        "joint_velocity",
        "joint_limit",
        "action_magnitude",
        "action_rate",
        "base_angvel_xy",
    } == set(components)
    weighted_sum = sum(value["weighted"] for value in components.values())
    assert np.allclose(step.reward, weighted_sum)
    assert step.metrics["foot_contact"].shape == (2, 2)
    assert step.metrics["foot_air_time"].shape == (2, 2)
    assert np.isfinite(step.reward).all()
    assert "- (200.0" not in inspect.getsource(TASK072.Task072BipedReward.step)


def test_contact_aligned_g1_groups_14_capsules_as_two_logical_feet() -> None:
    pytest.importorskip("mujoco")
    context = TASK072._load_bound_context(
        "unitree_g1", contact_profile=TASK072.MJLAB_G1_7CAPSULE_PROFILE_ID
    )
    shard = TASK072._build_shard(
        context,
        num_envs=2,
        trial_seconds=0.2,
        seed=72072,
        action_scale=0.35,
    )

    assert tuple(shard._logical_foot_names) == ("left_foot", "right_foot")
    assert [len(group) for group in shard._foot_geom_ids_by_foot] == [7, 7]
    assert len(shard._foot_geom_ids) == 14
    assert not any(name.endswith("_footpad") for name in shard._foot_geoms)

    step = shard.step(np.zeros((2, 45), dtype=np.float32))
    assert step.metrics["foot_contact"].shape == (2, 2)
    assert step.metrics["touchdown"].shape == (2, 2)
    assert step.metrics["foot_air_time"].shape == (2, 2)
    assert step.metrics["foot_normal_force"].shape == (2, 2)
    assert np.isfinite(step.metrics["foot_normal_force"]).all()
    assert np.all(step.metrics["non_foot_contact_fraction"] == 0.0)


def test_biped_phase_contact_prefers_alternating_over_static_double_support() -> None:
    config = TASK072.Task072BipedRewardConfig()
    samples = 40
    static_subtotal = []
    alternating_subtotal = []
    for step in range(samples):
        phase = ((step / TASK072.CONTROL_HZ) % config.period_s) / config.period_s
        leg_phase = np.asarray(
            (
                (phase + config.left_phase_offset) % 1.0,
                (phase + config.right_phase_offset) % 1.0,
            )
        )
        desired = leg_phase < config.stance_threshold
        static_contact = np.asarray((True, True))
        alternating_contact = desired
        for contact, sink in (
            (static_contact, static_subtotal),
            (alternating_contact, alternating_subtotal),
        ):
            phase_gait = float(np.mean(contact == desired)) * config.phase_gait
            out_of_phase = -float(bool(contact.all() and not desired.all())) * config.out_of_phase_double_support
            stand_support = 0.0
            sink.append(phase_gait + out_of_phase + stand_support)
    assert np.mean(static_subtotal) <= 0.0
    assert np.mean(alternating_subtotal) > 0.0
    assert np.mean(alternating_subtotal) > np.mean(static_subtotal)


def test_progression_requires_update0_and_monotonic_final() -> None:
    progression = {
        "reports": [
            {"global_update": 1, "reward_mean": 1.0},
            {"global_update": 2, "reward_mean": 1.1},
        ],
        "checkpoints": [
            {"global_update": 0, "env_steps": 0},
            {"global_update": 2, "env_steps": 64},
        ],
    }
    TASK072.validate_progression(progression, expected_updates=2, expected_env_steps=64)
    progression["checkpoints"][0]["global_update"] = 1
    with pytest.raises(ValueError, match="update0"):
        TASK072.validate_progression(progression, expected_updates=2, expected_env_steps=64)


class _FakeWriter:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.frames = 0

    def __enter__(self) -> Self:
        return self

    def append_data(self, _frame: np.ndarray) -> None:
        self.frames += 1

    def __exit__(self, *_args: object) -> None:
        self.output.write_bytes(str(self.frames).encode())


class _FakeImageIO:
    def get_writer(self, output: Path, **_kwargs: object) -> _FakeWriter:
        return _FakeWriter(output)

    def imwrite(self, output: Path, _frame: np.ndarray) -> None:
        output.write_bytes(b"midframe")


def test_video_writer_requires_and_writes_exactly_400_frames(tmp_path: Path) -> None:
    output = tmp_path / "walk.mp4"
    frames = (np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(400))
    count, midframe = TASK072._write_video_frames(_FakeImageIO(), output, frames)
    assert count == 400
    assert output.read_bytes() == b"400"
    assert midframe.read_bytes() == b"midframe"


def test_video_horizon_does_not_timeout_on_frame_400() -> None:
    trial_steps = round(TASK072.CONTROL_HZ * TASK072.VIDEO_TRIAL_SECONDS)
    assert trial_steps > TASK072.VIDEO_FRAMES


def test_eval_path_does_not_construct_a_trainer() -> None:
    assert "WholeBodyPPOTrainer" not in inspect.getsource(TASK072.eval_command)
