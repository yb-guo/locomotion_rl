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

# Independent audit literal: never generated from the runner's table builder.
EXPECTED_TASK072_REWARD_V4_ACTIVE_TABLE = (
    ("track_xy_centered", "task072_reward_track_xy_centered", 2.0),
    ("track_yaw", "task072_reward_track_yaw", 0.50),
    ("upright", "task072_reward_upright", 0.25),
    ("tilt", "task072_reward_tilt", 5.0),
    ("height", "task072_reward_height", 0.25),
    ("stand_support", "task072_reward_stand_support", 0.30),
    ("phase_gait", "task072_reward_phase_gait", 0.50),
    ("out_of_phase_double_support", "task072_reward_out_of_phase_double_support", 0.35),
    ("clearance", "task072_reward_clearance", 0.50),
    ("touchdown_airtime", "task072_reward_touchdown_airtime", 0.10),
    ("soft_landing", "task072_reward_soft_landing", 0.10),
    ("foot_slip", "task072_reward_foot_slip", 0.20),
    ("nonfoot_contact", "task072_reward_nonfoot_contact", 0.20),
    ("pose_hip", "task072_reward_pose_hip", 0.20),
    ("pose_knee", "task072_reward_pose_knee", 0.30),
    ("pose_ankle", "task072_reward_pose_ankle", 0.20),
    ("pose_waist", "task072_reward_pose_waist", 0.10),
    ("pose_arm_wrist", "task072_reward_pose_arm_wrist", 0.05),
    ("joint_velocity", "task072_reward_joint_velocity", 0.02),
    ("joint_limit", "task072_reward_joint_limit", 0.05),
    ("action_magnitude", "task072_reward_action_magnitude", 0.01),
    ("action_rate", "task072_reward_action_rate", 0.01),
    ("base_angvel_xy", "task072_reward_base_angvel_xy", 0.02),
    ("fall_terminated", "task072_reward_fall_terminated", 300.0),
)
EXPECTED_TASK072_REWARD_V4_PARAM_KEYS = {
    "track_xy_centered": ("asset_name", "body_id", "body_name", "command_name", "denominator"),
    "track_yaw": ("asset_name", "body_id", "body_name", "command_name", "denominator"),
    "upright": ("asset_name", "body_id", "body_name"),
    "tilt": ("asset_name", "body_id", "body_name"),
    "height": ("asset_name", "body_id", "body_name", "stance_height", "stance_payload_sha256"),
    "stand_support": ("command_name", "command_threshold", "sensor_name"),
    "phase_gait": ("command_name", "command_threshold", "offsets", "period", "sensor_name", "stance_fraction"),
    "out_of_phase_double_support": ("command_name", "command_threshold", "offsets", "period", "sensor_name", "stance_fraction"),
    "clearance": ("clearance_height", "clearance_sigma", "command_name", "command_threshold", "offsets", "period", "sensor_name", "site_ids", "site_names", "stance_fraction"),
    "touchdown_airtime": ("airtime_clip", "sensor_name", "site_ids", "site_names"),
    "soft_landing": ("landing_velocity_sigma", "sensor_name", "site_ids", "site_names"),
    "foot_slip": ("sensor_name", "site_ids", "site_names"),
    "nonfoot_contact": ("body_ids", "body_names", "sensor_name", "terrain_name"),
    "pose_hip": ("anonymous_joint_names", "asset_name", "joint_ids", "q_ref", "semantic_joint_names", "stance_payload_sha256"),
    "pose_knee": ("anonymous_joint_names", "asset_name", "joint_ids", "q_ref", "semantic_joint_names", "stance_payload_sha256"),
    "pose_ankle": ("anonymous_joint_names", "asset_name", "joint_ids", "q_ref", "semantic_joint_names", "stance_payload_sha256"),
    "pose_waist": ("anonymous_joint_names", "asset_name", "joint_ids", "q_ref", "semantic_joint_names", "stance_payload_sha256"),
    "pose_arm_wrist": ("anonymous_joint_names", "asset_name", "joint_ids", "q_ref", "semantic_joint_names", "stance_payload_sha256"),
    "joint_velocity": ("anonymous_joint_names", "asset_name", "joint_ids", "semantic_joint_names"),
    "joint_limit": ("anonymous_joint_names", "asset_name", "joint_ids", "lower", "semantic_joint_names", "soft_fraction", "upper"),
    "action_magnitude": ("action_name",),
    "action_rate": ("action_name", "previous_action_reset"),
    "base_angvel_xy": ("asset_name", "body_id", "body_name"),
    "fall_terminated": (),
}
EXPECTED_TASK072_STANCE_SHA = "cc522b67380713954480c3e9781be01fc6ad96445fb133d410f213f551f5ce9a"
EXPECTED_TASK072_FOOT_SITE_NAMES = ["anon_limb0_ankle_roll_link_foot", "anon_limb1_ankle_roll_link_foot"]
EXPECTED_TASK072_NONFOOT_BODY_IDS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
EXPECTED_TASK072_POSE_GROUP_IDS = {
    "pose_hip": [0, 1, 2, 6, 7, 8],
    "pose_knee": [3, 9],
    "pose_ankle": [4, 5, 10, 11],
    "pose_waist": [12, 13, 14],
    "pose_arm_wrist": list(range(15, 29)),
}
EXPECTED_TASK072_JOINT_LIMIT_LOWER = [-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618, -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618, -2.618, -0.52, -0.52, -3.0892, -1.5882, -2.618, -1.0472, -1.97222, -1.61443, -1.61443, -3.0892, -2.2515, -2.618, -1.0472, -1.97222, -1.61443, -1.61443]
EXPECTED_TASK072_JOINT_LIMIT_UPPER = [2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618, 2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618, 2.618, 0.52, 0.52, 2.6704, 2.2515, 2.618, 2.0944, 1.97222, 1.61443, 1.61443, 2.6704, 1.5882, 2.618, 2.0944, 1.97222, 1.61443, 1.61443]


def test_task072_reward_v4_literal_has_exact_active_order() -> None:
    env_cfg, _agent_cfg, _runner_cls, registration = MJLAB_RUNNER.build_task_cfg(1, 24, MJLAB_RUNNER.DEFAULT_SEED, 1)
    table = MJLAB_RUNNER.task072_reward_active_table_from_cfg(env_cfg.rewards)
    actual = tuple((row["name"], row["qualname"], row["weight"]) for row in table)

    assert actual == EXPECTED_TASK072_REWARD_V4_ACTIVE_TABLE
    assert len(table) == 24
    assert registration["reward_terms"] == [row[0] for row in EXPECTED_TASK072_REWARD_V4_ACTIVE_TABLE]
    assert set(env_cfg.rewards) == {row[0] for row in EXPECTED_TASK072_REWARD_V4_ACTIVE_TABLE}
    assert {"foot_gait", "feet_gait", "is_terminated"}.isdisjoint(env_cfg.rewards)
    assert env_cfg.episode_length_s == 10_000.0
    assert all(row["module"] == "task072_mjlab_contact_runner" for row in table)
    assert all("<locals>" not in row["qualname"] for row in table)
    assert all(row["source_file"] == str(MJLAB_RUNNER_SCRIPT.resolve()) for row in table)
    assert all(json.loads(json.dumps(row["params"])) == row["params"] for row in table)
    assert {row["name"]: tuple(sorted(row["params"])) for row in table} == EXPECTED_TASK072_REWARD_V4_PARAM_KEYS

    active_sha = MJLAB_RUNNER.task072_validate_reward_active_table(table)
    reward_payload = MJLAB_RUNNER.task072_canonical_reward_payload(table)
    assert active_sha == registration["reward_active_table_sha256"]
    assert reward_payload["payload_sha256"] == registration["reward_payload_sha256"]
    assert reward_payload["phase"] == {
        "clock": "episode_length_buf",
        "first_action_k": 1,
        "reset_k": 0,
        "period": MJLAB_RUNNER.TASK072_GAIT_PERIOD_S,
        "offsets": [0.0, 0.5],
        "stance_fraction": 0.55,
    }

    by_name = {row["name"]: row for row in table}
    for name in ("track_xy_centered", "track_yaw", "upright", "tilt", "height", "base_angvel_xy"):
        assert by_name[name]["params"]["asset_name"] == "robot"
        assert by_name[name]["params"]["body_name"] == "anon_waist_pitch_link"
        assert by_name[name]["params"]["body_id"] == 16
    for name in ("track_xy_centered", "track_yaw"):
        assert by_name[name]["params"]["command_name"] == "twist"
        assert by_name[name]["params"]["denominator"] == 0.25
    assert by_name["height"]["params"]["stance_height"] == pytest.approx(0.8533691099183076)
    assert by_name["height"]["params"]["stance_payload_sha256"] == EXPECTED_TASK072_STANCE_SHA
    assert by_name["stand_support"]["params"] == {
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
        "command_threshold": 0.1,
    }
    for name in ("phase_gait", "out_of_phase_double_support", "clearance"):
        assert by_name[name]["params"]["period"] == 0.8
        assert by_name[name]["params"]["offsets"] == [0.0, 0.5]
        assert by_name[name]["params"]["stance_fraction"] == 0.55
        assert by_name[name]["params"]["command_name"] == "twist"
        assert by_name[name]["params"]["command_threshold"] == 0.1
    for name in ("clearance", "touchdown_airtime", "soft_landing", "foot_slip"):
        assert by_name[name]["params"]["site_ids"] == [8, 15]
        assert by_name[name]["params"]["site_names"] == EXPECTED_TASK072_FOOT_SITE_NAMES
        assert by_name[name]["params"]["sensor_name"] == "feet_ground_contact"
    assert by_name["clearance"]["params"]["clearance_height"] == 0.10
    assert by_name["clearance"]["params"]["clearance_sigma"] == 0.05
    assert by_name["touchdown_airtime"]["params"]["airtime_clip"] == 0.5
    assert by_name["soft_landing"]["params"]["landing_velocity_sigma"] == 0.5
    assert by_name["nonfoot_contact"]["params"]["body_ids"] == EXPECTED_TASK072_NONFOOT_BODY_IDS
    assert set(by_name["nonfoot_contact"]["params"]["body_names"]).isdisjoint({
        "anon_limb0_ankle_roll_link",
        "anon_limb1_ankle_roll_link",
    })
    assert by_name["nonfoot_contact"]["params"]["sensor_name"] == "nonfoot_ground_contact"
    assert by_name["nonfoot_contact"]["params"]["terrain_name"] == "terrain"
    for name, expected_ids in EXPECTED_TASK072_POSE_GROUP_IDS.items():
        assert by_name[name]["params"]["joint_ids"] == expected_ids
        assert len(by_name[name]["params"]["q_ref"]) == len(expected_ids)
        assert by_name[name]["params"]["stance_payload_sha256"] == EXPECTED_TASK072_STANCE_SHA
    assert sorted(sum((row["params"]["joint_ids"] for row in (by_name[name] for name in EXPECTED_TASK072_POSE_GROUP_IDS)), [])) == list(range(29))
    assert by_name["joint_velocity"]["params"]["joint_ids"] == list(range(29))
    assert by_name["joint_limit"]["params"]["joint_ids"] == list(range(29))
    assert by_name["joint_limit"]["params"]["lower"] == EXPECTED_TASK072_JOINT_LIMIT_LOWER
    assert by_name["joint_limit"]["params"]["upper"] == EXPECTED_TASK072_JOINT_LIMIT_UPPER
    assert by_name["joint_limit"]["params"]["soft_fraction"] == 0.9
    assert by_name["action_magnitude"]["params"] == {"action_name": "joint_pos"}
    assert by_name["action_rate"]["params"] == {"action_name": "joint_pos", "previous_action_reset": 0.0}

    missing = table[:-1]
    with pytest.raises(ValueError, match="wrong key/order/count"):
        MJLAB_RUNNER.task072_validate_reward_active_table(missing)
    parent_callable = copy.deepcopy(table)
    parent_callable[0]["module"] = "mjlab.parent_rewards"
    with pytest.raises(ValueError, match="parent or alias"):
        MJLAB_RUNNER.task072_validate_reward_active_table(parent_callable)
    weight_drift = copy.deepcopy(table)
    weight_drift[0]["weight"] = 99.0
    with pytest.raises(ValueError, match="weight drift"):
        MJLAB_RUNNER.task072_validate_reward_active_table(weight_drift)
    phase_drift = copy.deepcopy(table)
    phase_drift[6]["params"]["period"] = 1.0
    with pytest.raises(ValueError, match="phase params drift"):
        MJLAB_RUNNER.task072_validate_reward_active_table(phase_drift)
    q_ref_drift = copy.deepcopy(table)
    q_ref_drift[13]["params"]["q_ref"][0] += 0.01
    with pytest.raises(ValueError, match="pose q_ref drift"):
        MJLAB_RUNNER.task072_validate_reward_active_table(q_ref_drift)
    source_hash_drift = copy.deepcopy(table)
    source_hash_drift[0]["function_source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source hash drift"):
        MJLAB_RUNNER.task072_validate_reward_active_table(source_hash_drift)
    with pytest.raises(ValueError, match="reward active-table SHA drift"):
        MJLAB_RUNNER.task072_require_train_eval_reward_match("a" * 64, "b" * 64)


def test_task072_reward_v4_oracle_and_manager_fixture_probe() -> None:
    env_cfg, _agent_cfg, _runner_cls, _registration = MJLAB_RUNNER.build_task_cfg(1, 24, MJLAB_RUNNER.DEFAULT_SEED, 1)
    oracle = MJLAB_RUNNER.task072_reward_v3_oracle_pre_dt_means()
    assert oracle == {
        "static_both": pytest.approx(-0.5542411176571156),
        "ideal_phase_matched": pytest.approx(1.7),
        "persistent_left_only": pytest.approx(-0.26012009890715004),
        "ideal_static_margin": pytest.approx(2.2542411176571155),
    }
    probe = MJLAB_RUNNER.task072_reward_fixture_probe(env_cfg.rewards)
    assert probe["passed"] is True
    for key in ("static_both", "ideal_phase_matched", "persistent_left_only"):
        assert probe[key]["weighted_pre_dt_mean"] == pytest.approx(oracle[key], abs=1.0e-6)
        assert probe[key]["dt_contribution_mean"] == pytest.approx(oracle[key] * 0.02, abs=1.0e-6)
        assert probe[key]["iterable_abs_diff_max"] <= 1.0e-6
    assert probe["ideal_static_margin"] == pytest.approx(oracle["ideal_static_margin"], abs=1.0e-6)
    assert probe["terminal"]["normal"]["fall_raw"] == 0.0
    assert probe["terminal"]["fell_over"]["fall_raw"] == -1.0
    assert probe["terminal"]["fell_over"]["pre_dt"] == -300.0
    assert probe["terminal"]["fell_over"]["dt_contribution"] == -6.0
    assert probe["terminal"]["timeout"]["fall_raw"] == 0.0
    assert probe["terminal"]["passed"] is True


def test_task072_v4_cross_manager_contract_closes_phase_and_fall() -> None:
    env_cfg, agent_cfg, _runner_cls, registration = MJLAB_RUNNER.build_task_cfg(1, 24, MJLAB_RUNNER.DEFAULT_SEED, 1)
    table = MJLAB_RUNNER.task072_reward_active_table_from_cfg(env_cfg.rewards)
    by_name = {row["name"]: row for row in table}
    assert by_name["phase_gait"]["params"]["period"] == MJLAB_RUNNER.TASK072_GAIT_PERIOD_S
    assert by_name["clearance"]["params"]["period"] == MJLAB_RUNNER.TASK072_GAIT_PERIOD_S
    assert table[-1]["name"] == "fall_terminated"
    assert table[-1]["params"] == {}
    assert table[-1]["weight"] == 300.0
    assert registration["reward_terms"][-1] == "fall_terminated"
    semantic = MJLAB_RUNNER.task072_runtime_semantic_payload(env_cfg, agent_cfg, registration, render_mode=None)
    assert semantic["active_subtask"] == "003k"
    assert semantic["lineage_id"] == MJLAB_RUNNER.LINEAGE_ID
    assert semantic["episode"]["episode_length_s"] == 10_000.0
    assert semantic["rewards"][-1]["name"] == "fall_terminated"


def test_task072_clip_logging_and_eval_cause_separation() -> None:
    torch = pytest.importorskip("torch")

    class Base:
        num_actions = 29
        num_envs = 2
        cfg = {}

        def __init__(self) -> None:
            self.last_action = None

        @property
        def unwrapped(self) -> Self:
            return self

        def reset(self) -> tuple[str, dict[str, object]]:
            return "obs", {}

        def step(self, action):
            self.last_action = action
            return "obs", action, torch.zeros(2, dtype=torch.bool), {}

    names = list(MJLAB_RUNNER.SEMANTIC_TO_ANON_JOINT)
    base = Base()
    wrapper = MJLAB_RUNNER.Task072ClipLoggingVecEnvWrapper(base, names, 2)
    raw0 = torch.zeros((2, 29))
    raw1 = torch.zeros((2, 29))
    raw0[0, 0] = 1.1
    raw1[1, 1] = -2.0
    wrapper.step(raw0)
    wrapper.step(raw1)
    records = MJLAB_RUNNER.validate_task072_clip_records(
        wrapper.drain_task072_clip_update_records(),
        expected_updates=1,
    )
    record = records[0]
    assert record["clipped_scalars"] == 2
    assert record["scalar_denominator"] == 116
    assert record["scalar_clip_fraction"] == pytest.approx(2 / 116)
    assert record["env_steps_with_any_clip"] == 2
    assert record["env_step_denominator"] == 4
    assert record["per_joint_denominator"] == 4
    assert record["per_joint_clipped_scalars"][names[0]] == 1
    assert record["per_joint_clipped_scalars"][names[1]] == 1
    assert record["per_joint_clip_fraction"][names[0]] == pytest.approx(0.25)
    assert record["max_abs_raw_action"] == pytest.approx(2.0)
    assert torch.equal(base.last_action, raw1)
    assert wrapper.drain_task072_clip_update_records() == []

    missing_field = copy.deepcopy(record)
    del missing_field["per_joint_clip_fraction"]
    with pytest.raises(ValueError, match="missing fields"):
        MJLAB_RUNNER.validate_task072_clip_records([missing_field], expected_updates=1)

    metrics = MJLAB_RUNNER.task072_eval_cause_metrics([
        {
            "reset_terminated": [False, False],
            "reset_time_outs": [False, False],
            "done": [False, False],
            "x": [0.1, 0.2],
            "vx": [0.5, 0.4],
            "planar_tracking_error": [0.0, 0.1],
            "yaw_error": [0.0, 0.0],
            "gravity_xy": [0.0, 0.0],
            "contact": [[True, False], [False, True]],
        },
        {
            "reset_terminated": [True, False],
            "reset_time_outs": [False, True],
            "done": [True, True],
            "x": [99.0, 99.0],
            "vx": [99.0, 99.0],
            "planar_tracking_error": [99.0, 99.0],
            "yaw_error": [99.0, 99.0],
            "gravity_xy": [99.0, 99.0],
            "contact": [[True, True], [True, True]],
        },
    ], 2)
    assert metrics["reset_terminated"]["count"] == 1
    assert metrics["reset_time_outs"]["count"] == 1
    assert metrics["zero_fall_ratio"] == pytest.approx(0.5)
    assert metrics["common_prefix"]["mean_x_displacement"] == pytest.approx(0.15)
    assert metrics["common_prefix"]["mean_vx"] == pytest.approx(0.45)
    assert metrics["common_prefix"]["planar_tracking_error"] == pytest.approx(0.05)
    assert metrics["survivor_full_horizon"]["survivor_count"] == 0
    assert metrics["survivor_full_horizon"]["mean_vx"] is None
    with pytest.raises(ValueError, match="overlap"):
        MJLAB_RUNNER.task072_eval_cause_metrics([
            {"reset_terminated": [True], "reset_time_outs": [True], "done": [True], "x": [0.0], "vx": [0.0]},
        ], 1)


def _task072_clip_record(
    update_index: int,
    *,
    clipped_by_joint: dict[str, int] | None = None,
    env_steps_with_any_clip: int = 0,
    max_abs_raw_action: float = 0.0,
) -> dict[str, object]:
    names = list(MJLAB_RUNNER.SEMANTIC_TO_ANON_JOINT)
    env_step_denominator = 4096 * 24
    per_joint = {name: int((clipped_by_joint or {}).get(name, 0)) for name in names}
    clipped_scalars = sum(per_joint.values())
    return {
        "update_index": update_index,
        "num_envs": 4096,
        "rollout_steps": 24,
        "joint_count": 29,
        "clipped_scalars": clipped_scalars,
        "scalar_denominator": env_step_denominator * 29,
        "scalar_clip_fraction": clipped_scalars / (env_step_denominator * 29),
        "env_steps_with_any_clip": env_steps_with_any_clip,
        "env_step_denominator": env_step_denominator,
        "env_step_any_clip_fraction": env_steps_with_any_clip / env_step_denominator,
        "per_joint_clipped_scalars": per_joint,
        "per_joint_denominator": env_step_denominator,
        "per_joint_clip_fraction": {name: value / env_step_denominator for name, value in per_joint.items()},
        "max_abs_raw_action": max_abs_raw_action,
    }


def _task072_fake_training_manifest(tmp_path: Path, records: list[dict[str, object]] | None = None) -> dict[str, object]:
    records = records or [_task072_clip_record(index) for index in range(MJLAB_RUNNER.TASK072_PILOT_UPDATES)]
    summary = MJLAB_RUNNER.pool_task072_clip_records(records, last_n=7)
    progression = tmp_path / "progression.json"
    progression.write_text(json.dumps({"passed": True}), encoding="utf-8")
    checkpoint_sha = {}
    for update in MJLAB_RUNNER.TASK072_PILOT_EVAL_UPDATES:
        checkpoint = tmp_path / f"model_{update}.pt"
        checkpoint.write_bytes(f"model-{update}".encode())
        checkpoint_sha[str(checkpoint.resolve())] = MJLAB_RUNNER.sha256_path(checkpoint)
    env_cfg, _agent_cfg, _runner_cls, registration = MJLAB_RUNNER.build_task_cfg(
        1, 24, MJLAB_RUNNER.DEFAULT_SEED, 1
    )
    runtime_table = MJLAB_RUNNER.task072_reward_active_table_from_cfg(env_cfg.rewards)
    losses = [{"value": 0.1 + update * 0.001} for update in range(MJLAB_RUNNER.TASK072_PILOT_UPDATES)]
    update_reports = [
        {
            "update_index": update,
            "losses": loss,
            "pre_update_std": {"shape": [29], "min": 1.0, "mean": 1.0, "max": 1.0, "finite": True},
            "post_update_std": {"shape": [29], "min": 0.9, "mean": 0.9, "max": 0.9, "finite": True},
        }
        for update, loss in enumerate(losses)
    ]
    return {
        "schema_version": 3,
        "subtask": MJLAB_RUNNER.TASK072_ACTIVE_SUBTASK,
        "lineage_id": MJLAB_RUNNER.LINEAGE_ID,
        "runtime_lineage_id": MJLAB_RUNNER.LINEAGE_ID,
        "seed": MJLAB_RUNNER.DEFAULT_SEED,
        "num_envs": 4096,
        "rollout_steps_per_env": 24,
        "updates": 21,
        "observed_transitions": 4096 * 24 * 21,
        "checkpoint_sha256": checkpoint_sha,
        "action_clip_update_records": records,
        "action_clip_last_7_summary": summary,
        "policy_distribution_lineage": {
            "actor_distribution_class": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 0.001,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "clip_actions": 1.0,
            "action_dimension": 29,
        },
        "optimizer_step_count": 5 * 4 * 21,
        "expected_optimizer_step_count": 5 * 4 * 21,
        "parameter_delta": {"finite": True, "max_abs": 0.1, "changed_parameter_count": 1},
        "losses": losses,
        "update_reports": update_reports,
        "runtime_reward_active_table": runtime_table,
        "runtime_reward_active_table_sha256": registration["reward_active_table_sha256"],
        "runtime_rollout_evidence": {
            "check_for_nan_enabled": True,
            "expected_rollout_steps": 24 * 21,
            "observed_rollout_steps": 24 * 21,
            "expected_transitions": 4096 * 24 * 21,
            "observed_transitions": 4096 * 24 * 21,
            "obs_finite": True,
            "rewards_finite": True,
            "dones_finite": True,
            "finite": True,
        },
        "check_for_nan_enabled": True,
        "runtime_reward_active_term_count": len(runtime_table),
        "reward_contract": {
            "version": MJLAB_RUNNER.REWARD_CONTRACT_VERSION,
            "canonical_payload_sha256": registration["reward_payload_sha256"],
            "config_active_table_sha256": registration["reward_active_table_sha256"],
        },
        "progression": {"path": str(progression), "sha256": MJLAB_RUNNER.sha256_path(progression)},
        "capacity_evidence": {"consumption_checks": {"ok": True}},
        "external_mjlab_checks": {"frame_local": True, "commit_pinned": True, "tracked_clean": True},
        "training_execution_complete": True,
        "acceptance_checks": {
            name: True for name in reversed(MJLAB_RUNNER.TASK072_ACCEPTANCE_CHECK_NAMES)
        },
        "passed": True,
    }


def _task072_fake_eval(
    manifest: dict[str, object],
    update: int,
    *,
    median_first_fall: float,
    common_vx: float = 0.06,
    common_x: float = 0.11,
    timeouts: int = 0,
    finite: bool = True,
    manifest_sha: str | None = None,
) -> dict[str, object]:
    checkpoint_path = next(path for path in manifest["checkpoint_sha256"] if Path(path).stem == f"model_{update}")
    return {
        "schema_version": 3,
        "subtask": MJLAB_RUNNER.TASK072_ACTIVE_SUBTASK,
        "lineage_id": MJLAB_RUNNER.LINEAGE_ID,
        "checkpoint": {"path": checkpoint_path, "sha256": manifest["checkpoint_sha256"][checkpoint_path]},
        "training_manifest": {"sha256": manifest_sha},
        "metrics": {
            "eval_seconds": 20.0,
            "eval_envs": 256,
            "fixed_command": {"vx": 0.5, "vy": 0.0, "yaw": 0.0},
            "reward_finite": finite,
            "obs_finite": finite,
            "reset_time_outs": {"count": timeouts},
            "first_fall_seconds": {"median": median_first_fall},
            "common_prefix": {
                "mean_vx": common_vx,
                "median_x_displacement": common_x,
            },
        },
        "passed": False,
    }


def test_task072_clip_summary_and_training_manifest_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = [_task072_clip_record(index) for index in range(MJLAB_RUNNER.TASK072_PILOT_UPDATES)]
    summary = MJLAB_RUNNER.pool_task072_clip_records(records, last_n=7)
    assert MJLAB_RUNNER.validate_task072_clip_summary(summary, expected_update_indices=list(range(14, 21))) is summary

    bad_fraction = copy.deepcopy(records[0])
    bad_fraction["scalar_clip_fraction"] = 0.5
    with pytest.raises(ValueError, match="scalar fraction mismatch"):
        MJLAB_RUNNER.validate_task072_clip_records([bad_fraction])
    fractional_counter = copy.deepcopy(records[0])
    fractional_counter["clipped_scalars"] = 0.5
    with pytest.raises(ValueError, match="must be an integer"):
        MJLAB_RUNNER.validate_task072_clip_records([fractional_counter])
    fractional_summary = copy.deepcopy(summary)
    fractional_summary["env_steps_with_any_clip"] = 0.5
    with pytest.raises(ValueError, match="must be an integer"):
        MJLAB_RUNNER.validate_task072_clip_summary(fractional_summary, expected_update_indices=list(range(14, 21)))

    names = list(MJLAB_RUNNER.SEMANTIC_TO_ANON_JOINT)
    heavy = copy.deepcopy(records)
    for update_index in range(14, 21):
        heavy[update_index] = _task072_clip_record(
            update_index,
            clipped_by_joint={name: 24_576 for name in names[:12]},
            env_steps_with_any_clip=1_000,
            max_abs_raw_action=1.5,
        )
    heavy_summary = MJLAB_RUNNER.pool_task072_clip_records(heavy, last_n=7)
    assert MJLAB_RUNNER.validate_task072_clip_summary(heavy_summary, expected_update_indices=list(range(14, 21))) is heavy_summary

    current = {
        "action_contract": {},
        "reward_contract": {},
        "canonical_train_eval_config": {"payload_sha256": "c" * 64, "passed": True},
        "runner_source_sha256": "r" * 64,
        "runtime_spec_sha256": "s" * 64,
        "asset_xml": {"sha256": "a" * 64},
        "contact_profile": {"payload_sha256": "p" * 64},
        "stance": {"payload_sha256": "t" * 64},
        "external_mjlab_checks": {"frame_local": True, "commit_pinned": True, "tracked_clean": True},
    }
    monkeypatch.setattr(MJLAB_RUNNER, "_common_manifest", lambda _args: current)
    fake_manifest = _task072_fake_training_manifest(tmp_path, records)
    current["reward_contract"] = fake_manifest["reward_contract"]
    manifest = {
        **current,
        **fake_manifest,
        "action_contract": current["action_contract"],
        "reward_contract": current["reward_contract"],
        "canonical_train_eval_config": current["canonical_train_eval_config"],
    }
    MJLAB_RUNNER._validate_training_manifest_for_eval(manifest)
    forged = copy.deepcopy(manifest)
    forged["action_clip_update_records"][0]["clipped_scalars"] = 1
    with pytest.raises(ValueError, match="training manifest failed"):
        MJLAB_RUNNER._validate_training_manifest_for_eval(forged)
    no_capacity = copy.deepcopy(manifest)
    no_capacity["capacity_evidence"] = {"consumption_checks": {}}
    with pytest.raises(ValueError, match="training manifest failed"):
        MJLAB_RUNNER._validate_training_manifest_for_eval(no_capacity)
    high_clip_manifest = {
        **manifest,
        "action_clip_update_records": heavy,
        "action_clip_last_7_summary": heavy_summary,
        "passed": True,
    }
    MJLAB_RUNNER._validate_training_manifest_for_eval(high_clip_manifest)


def test_task072_one_update_acceptance_evidence_is_complete_and_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = {
        "action_contract": {}, "reward_contract": {},
        "canonical_train_eval_config": {"payload_sha256": "c" * 64, "passed": True},
        "runner_source_sha256": "r" * 64, "runtime_spec_sha256": "s" * 64,
        "asset_xml": {"sha256": "a" * 64}, "contact_profile": {"payload_sha256": "p" * 64},
        "stance": {"payload_sha256": "t" * 64},
        "external_mjlab_checks": {"frame_local": True, "commit_pinned": True, "tracked_clean": True},
    }
    monkeypatch.setattr(MJLAB_RUNNER, "_common_manifest", lambda _args: current)
    fake_manifest = _task072_fake_training_manifest(tmp_path)
    current["reward_contract"] = fake_manifest["reward_contract"]
    manifest = {**current, **fake_manifest}
    MJLAB_RUNNER._validate_training_manifest_for_eval(manifest)
    for missing_field in (
        "optimizer_step_count",
        "parameter_delta",
        "losses",
        "runtime_reward_active_table",
    ):
        incomplete = copy.deepcopy(manifest)
        del incomplete[missing_field]
        with pytest.raises(ValueError, match="training manifest failed"):
            MJLAB_RUNNER._validate_training_manifest_for_eval(incomplete)


def test_task072_pilot_gate_requires_survival_non_regression_and_clip_updates(tmp_path: Path) -> None:
    manifest = _task072_fake_training_manifest(tmp_path)
    evals = [
        _task072_fake_eval(manifest, 0, median_first_fall=1.4),
        _task072_fake_eval(manifest, 7, median_first_fall=2.7),
        _task072_fake_eval(manifest, 14, median_first_fall=2.5),
        _task072_fake_eval(manifest, 20, median_first_fall=2.8),
    ]
    gate = MJLAB_RUNNER.task072_pilot_continuation_gate(manifest, evals)
    assert gate["passed"] is True
    assert gate["comparison_checks"] == {
        "model20_median_first_fall_ge_2p5": True,
        "model20_median_first_fall_ge_model0_plus_0p5": True,
        "model14_median_first_fall_ge_model7_minus_0p25": True,
        "model20_median_first_fall_ge_model7_minus_0p10": True,
        "model20_median_first_fall_ge_model14_minus_0p25": True,
        "model20_common_prefix_mean_vx_ge_0p05": True,
        "model20_common_prefix_median_x_ge_0p10": True,
    }

    regressed = copy.deepcopy(evals)
    regressed[3]["metrics"]["first_fall_seconds"]["median"] = 2.4
    gate = MJLAB_RUNNER.task072_pilot_continuation_gate(manifest, regressed)
    assert gate["passed"] is False
    assert "continuation.model20_median_first_fall_ge_2p5" in gate["failure_reasons"]

    timeout = copy.deepcopy(evals)
    timeout[1]["metrics"]["reset_time_outs"]["count"] = 1
    gate = MJLAB_RUNNER.task072_pilot_continuation_gate(manifest, timeout)
    assert gate["passed"] is False
    assert "eval.model_7.no_time_outs" in gate["failure_reasons"]

    low_forward = copy.deepcopy(evals)
    low_forward[3]["metrics"]["common_prefix"]["mean_vx"] = 0.049
    gate = MJLAB_RUNNER.task072_pilot_continuation_gate(manifest, low_forward)
    assert gate["passed"] is False
    assert "continuation.model20_common_prefix_mean_vx_ge_0p05" in gate["failure_reasons"]

    missing_eval = MJLAB_RUNNER.task072_pilot_continuation_gate(manifest, evals[:3])
    assert missing_eval["passed"] is False
    assert "eval_set.exact_updates" in missing_eval["failure_reasons"]
    no_capacity_manifest = copy.deepcopy(manifest)
    no_capacity_manifest["capacity_evidence"] = {"consumption_checks": {}}
    gate = MJLAB_RUNNER.task072_pilot_continuation_gate(no_capacity_manifest, evals)
    assert gate["passed"] is False
    assert "training.capacity_consumed" in gate["failure_reasons"]

    names = list(MJLAB_RUNNER.SEMANTIC_TO_ANON_JOINT)
    heavy_records = [_task072_clip_record(index) for index in range(MJLAB_RUNNER.TASK072_PILOT_UPDATES)]
    for update_index in range(14, 21):
        heavy_records[update_index] = _task072_clip_record(
            update_index,
            clipped_by_joint={name: 24_576 for name in names[:12]},
            env_steps_with_any_clip=1_000,
            max_abs_raw_action=1.5,
        )
    bad_manifest = _task072_fake_training_manifest(tmp_path, heavy_records)
    bad_manifest["passed"] = True
    gate = MJLAB_RUNNER.task072_pilot_continuation_gate(bad_manifest, [
        _task072_fake_eval(bad_manifest, 0, median_first_fall=1.4),
        _task072_fake_eval(bad_manifest, 7, median_first_fall=2.7),
        _task072_fake_eval(bad_manifest, 14, median_first_fall=2.5),
        _task072_fake_eval(bad_manifest, 20, median_first_fall=2.8),
    ])
    assert gate["passed"] is True
    assert "training.clip_last_7_valid" not in gate["failure_reasons"]


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
    assert MJLAB_RUNNER.LINEAGE_ID == "mjlab_g1_7capsule_task_v4_semantic_closed"
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
    assert train.capacity_artifact == MJLAB_RUNNER.RUNTIME_BINDING_ROOT / "003k_capacity_smoke_2048_4096_6144.json"
    assert MJLAB_RUNNER.REQUIRED_TRANSITIONS_PER_UPDATE == 4096 * 24


def test_mjlab_runner_records_003f_only_for_runtime_verifier_then_003k(tmp_path: Path) -> None:
    verify = MJLAB_RUNNER.parse_args(["verify-runtime-binding", "--output", "verify.json"])
    capacity = MJLAB_RUNNER.parse_args(["capacity-smoke", "--output", "capacity.json"])
    one_update = MJLAB_RUNNER.parse_args(["one-update-train", "--run-dir", "run"])
    evaluate = MJLAB_RUNNER.parse_args(["evaluate", "--checkpoint", "model_100.pt", "--run-manifest", "run_manifest.json", "--output", "eval.json"])
    assert verify.command == "verify-runtime-binding"
    assert MJLAB_RUNNER._device_requires_gpu_lock(evaluate.device) is True

    contact_payload = {"contact_profile_id": MJLAB_RUNNER.CONTACT_PROFILE_ID}
    stance_payload = {"contact_profile_id": MJLAB_RUNNER.CONTACT_PROFILE_ID}

    def fake_sha(_path: Path) -> str:
        return "a" * 64

    def fake_payload_sha(_payload: object) -> str:
        return "b" * 64

    original_contact = MJLAB_RUNNER.CONTACT_PROFILE
    original_stance = MJLAB_RUNNER.STANCE
    original_sha = MJLAB_RUNNER.sha256_path
    original_payload_sha = MJLAB_RUNNER.payload_sha256
    original_runtime_spec = MJLAB_RUNNER.runtime_spec_xml
    original_runtime = MJLAB_RUNNER._runtime_metadata
    original_canonical = MJLAB_RUNNER.canonical_train_eval_config_payload
    original_action_contract = MJLAB_RUNNER.action_contract_from_asset_xml
    original_stance_dict = MJLAB_RUNNER._stance_dict
    original_reward_table = MJLAB_RUNNER.task072_reward_v3_table
    original_active_table = MJLAB_RUNNER.task072_reward_active_table_from_cfg
    original_reward_payload = MJLAB_RUNNER.task072_canonical_reward_payload
    original_reward_validate = MJLAB_RUNNER.task072_validate_reward_active_table
    original_ensure = MJLAB_RUNNER.ensure_v2_artifacts
    contact = tmp_path / "contact.json"
    stance = tmp_path / "stance.json"
    contact.write_text(json.dumps(contact_payload), encoding="utf-8")
    stance.write_text(json.dumps(stance_payload), encoding="utf-8")
    try:
        MJLAB_RUNNER.CONTACT_PROFILE = contact
        MJLAB_RUNNER.STANCE = stance
        MJLAB_RUNNER.ensure_v2_artifacts = lambda: None
        MJLAB_RUNNER.sha256_path = fake_sha
        MJLAB_RUNNER.payload_sha256 = fake_payload_sha
        MJLAB_RUNNER.runtime_spec_xml = lambda: "<mujoco/>"
        MJLAB_RUNNER._runtime_metadata = lambda _command: {
            "external_mjlab": {
                "actual_commit": "1425b15f73bd4095f0df53709d7c389c3eb9e790",
                "expected_commit": "1425b15f73bd4095f0df53709d7c389c3eb9e790",
                "tracked_clean": True,
            }
        }
        MJLAB_RUNNER.canonical_train_eval_config_payload = lambda: {
            "payload_sha256": "c" * 64,
            "diff": [],
            "eval_diff_allowlist": [],
            "non_allowlisted_diff": [],
            "passed": True,
        }
        MJLAB_RUNNER.action_contract_from_asset_xml = lambda: {
            "payload_sha256": "d" * 64,
        }
        MJLAB_RUNNER._stance_dict = lambda: {"stub": True}
        MJLAB_RUNNER.task072_reward_v3_table = lambda _stance: {"stub": object()}
        MJLAB_RUNNER.task072_reward_active_table_from_cfg = lambda _cfg: []
        MJLAB_RUNNER.task072_canonical_reward_payload = lambda _table: {"payload_sha256": "e" * 64}
        MJLAB_RUNNER.task072_validate_reward_active_table = lambda _table: "f" * 64
        assert MJLAB_RUNNER._common_manifest(verify)["subtask"] == "003f"
        assert MJLAB_RUNNER._common_manifest(capacity)["subtask"] == MJLAB_RUNNER.TASK072_ACTIVE_SUBTASK
        assert MJLAB_RUNNER._common_manifest(one_update)["subtask"] == MJLAB_RUNNER.TASK072_ACTIVE_SUBTASK
        assert MJLAB_RUNNER._common_manifest(evaluate)["subtask"] == MJLAB_RUNNER.TASK072_ACTIVE_SUBTASK
    finally:
        MJLAB_RUNNER.CONTACT_PROFILE = original_contact
        MJLAB_RUNNER.STANCE = original_stance
        MJLAB_RUNNER.sha256_path = original_sha
        MJLAB_RUNNER.payload_sha256 = original_payload_sha
        MJLAB_RUNNER.runtime_spec_xml = original_runtime_spec
        MJLAB_RUNNER._runtime_metadata = original_runtime
        MJLAB_RUNNER.canonical_train_eval_config_payload = original_canonical
        MJLAB_RUNNER.action_contract_from_asset_xml = original_action_contract
        MJLAB_RUNNER._stance_dict = original_stance_dict
        MJLAB_RUNNER.task072_reward_v3_table = original_reward_table
        MJLAB_RUNNER.task072_reward_active_table_from_cfg = original_active_table
        MJLAB_RUNNER.task072_canonical_reward_payload = original_reward_payload
        MJLAB_RUNNER.task072_validate_reward_active_table = original_reward_validate
        MJLAB_RUNNER.ensure_v2_artifacts = original_ensure


def test_mjlab_runtime_defaults_are_v4_semantic_closed_paths() -> None:
    assert MJLAB_RUNNER.DEFAULT_OUTPUT_ROOT == MJLAB_RUNNER.RUNTIME_BINDING_ROOT
    assert MJLAB_RUNNER.DEFAULT_OUTPUT_ROOT.is_relative_to(MJLAB_RUNNER.RUNTIME_BINDING_ROOT)
    assert MJLAB_RUNNER.RUNTIME_BINDING_ROOT.name == "mjlab_g1_7capsule_task_v4_semantic_closed"
    assert "mjlab_contact_training/g1" not in str(MJLAB_RUNNER.DEFAULT_OUTPUT_ROOT)
    assert MJLAB_RUNNER.parse_args(["r0-smoke"]).output.is_relative_to(MJLAB_RUNNER.RUNTIME_BINDING_ROOT)
    assert MJLAB_RUNNER.parse_args(["capacity-smoke"]).output.is_relative_to(MJLAB_RUNNER.RUNTIME_BINDING_ROOT)
    assert MJLAB_RUNNER.parse_args(["one-update-train"]).run_dir.is_relative_to(MJLAB_RUNNER.RUNTIME_BINDING_ROOT)
    pilot_gate = MJLAB_RUNNER.parse_args(["pilot-gate"])
    assert pilot_gate.output == MJLAB_RUNNER.RUNTIME_BINDING_ROOT / "003k_pilot_gate.json"
    assert [path.name for path in pilot_gate.eval] == [
        "003k_eval_pilot_model_0_fixed_vx0p5_seed720400.json",
        "003k_eval_pilot_model_7_fixed_vx0p5_seed720400.json",
        "003k_eval_pilot_model_14_fixed_vx0p5_seed720400.json",
        "003k_eval_pilot_model_20_fixed_vx0p5_seed720400.json",
    ]


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


def test_mjlab_action_contract_signed_bounds_cover_all_targets() -> None:
    torch = pytest.importorskip("torch")
    contract = MJLAB_RUNNER.action_contract_from_asset_xml()
    assert contract["version"] == "task072_mjlab_signed_headroom_v1"
    assert len(contract["rows"]) == 29
    assert len({row["semantic_joint"] for row in contract["rows"]}) == 29
    assert contract["policy_action_domain"] == {"transform": "clip", "lower": -1.0, "upper": 1.0}
    for row in contract["rows"]:
        lower, upper = row["joint_range"]
        offset = row["stance_action_offset"]
        for raw, key in ((-1.0, "signed_negative_amplitude"), (0.0, "signed_positive_amplitude"), (1.0, "signed_positive_amplitude")):
            target = offset + raw * row[key]
            assert lower < target < upper
            assert target >= lower + row["safety_margin"] - 1e-12
            assert target <= upper - row["safety_margin"] + 1e-12
    raw = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    neg = torch.ones_like(raw) * 0.1
    pos = torch.ones_like(raw) * 0.2
    offset = torch.zeros_like(raw) + 0.5
    target, clipped = MJLAB_RUNNER.apply_signed_action_contract(raw, neg, pos, offset)
    assert torch.allclose(clipped, torch.tensor([[-1.0, -1.0, 0.0, 1.0, 1.0]]))
    assert torch.allclose(target, torch.tensor([[0.4, 0.4, 0.5, 0.7, 0.7]]))


def test_mjlab_canonical_train_eval_config_diff_is_allowlisted() -> None:
    payload = MJLAB_RUNNER.canonical_train_eval_config_payload()
    assert payload["passed"] is True
    assert payload["non_allowlisted_diff"] == []
    assert payload["train"]["env"]["command"]["lin_vel_x"] == [0.5, 0.5]
    assert payload["train"]["env"]["events"] == ["reset_base", "reset_robot_joints"]
    assert "is_terminated" not in payload["train"]["env"]["reward_names"]
    assert "feet_gait" not in payload["train"]["env"]["reward_names"]
    assert payload["train"]["action"]["policy_action_domain"]["transform"] == "clip"


def test_mjlab_evaluate_requires_manifest_bound_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_1.pt"
    checkpoint.write_bytes(b"bad")
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({"checkpoint_sha256": {}}), encoding="utf-8")
    args = MJLAB_RUNNER.parse_args([
        "evaluate",
        "--checkpoint",
        str(checkpoint),
        "--run-manifest",
        str(manifest),
        "--output",
        str(tmp_path / "eval.json"),
        "--device",
        "cpu",
    ])
    with pytest.raises(ValueError, match="training manifest failed Task072 eval lineage checks"):
        MJLAB_RUNNER.evaluate_checkpoint(args)

    noncanonical = MJLAB_RUNNER.parse_args([
        "evaluate",
        "--checkpoint",
        str(checkpoint),
        "--run-manifest",
        str(manifest),
        "--output",
        str(tmp_path / "eval_bad_rollout.json"),
        "--rollout-steps",
        "12",
        "--device",
        "cpu",
    ])
    with pytest.raises(ValueError, match="24 rollout steps"):
        MJLAB_RUNNER.evaluate_checkpoint(noncanonical)


def test_mjlab_render_reload_and_freeze_are_fail_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_1.pt"
    checkpoint.write_bytes(b"bad")
    eval_path = tmp_path / "eval.json"
    eval_path.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    video = tmp_path / "walk.json"
    video.write_text("{}", encoding="utf-8")
    assert MJLAB_RUNNER.render_command(
        MJLAB_RUNNER.parse_args([
            "render",
            "--checkpoint",
            str(checkpoint),
            "--run-manifest",
            str(manifest),
            "--eval",
            str(eval_path),
            "--output",
            str(tmp_path / "walk.mp4"),
            "--device",
            "cpu",
        ])
    ) == 1
    assert MJLAB_RUNNER.verify_reload_command(
        MJLAB_RUNNER.parse_args([
            "verify-reload",
            "--checkpoint",
            str(checkpoint),
            "--run-manifest",
            str(manifest),
            "--eval",
            str(eval_path),
            "--video",
            str(video),
            "--output",
            str(tmp_path / "reload.json"),
        ])
    ) == 1
    assert MJLAB_RUNNER.freeze_command(
        MJLAB_RUNNER.parse_args([
            "freeze",
            "--eval",
            str(eval_path),
            "--video",
            str(video),
            "--reload-verifier",
            str(tmp_path / "reload.json"),
            "--output",
            str(tmp_path / "freeze.json"),
        ])
    ) == 1


def test_repaired_smoke_verifier_rejects_historical_source_drift(tmp_path: Path) -> None:
    base = TASK072.TASK_DIR / "artifacts/nominal_v4/unitree_g1/E3a_mjlab_kl_repair"
    args = SimpleNamespace(run_manifest=base / "smoke/run_manifest.json", no_update_gate=base / "no_update_correctness_gate.json", output=tmp_path / "r4.json")
    assert REPAIR.verify_smoke(args) != 0
    payload = __import__("json").loads(args.output.read_text())
    assert payload["r4_repaired_smoke_passed"] is False
    assert "source environment SHA drift" in payload["failure_reasons"]


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
    with pytest.raises(ValueError, match="source environment SHA drift"):
        REPAIR.check_smoke(REPAIR.R4_MANIFEST_PATH, REPAIR.NO_UPDATE_PATH)


def test_repaired_r4_gate_requires_entire_recomputed_payload() -> None:
    with pytest.raises(ValueError, match="source environment SHA drift"):
        REPAIR.require_r4_gate(REPAIR.R4_GATE_PATH)


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
    with pytest.raises(ValueError, match="source environment SHA drift"):
        REPAIR.validate_manifest(
            manifest,
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
    from h200_locomotion_lab.policies.whole_body_mlp import (
        WholeBodyMLPActorCritic,
        WholeBodyMLPConfig,
    )

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
    pytest.importorskip("torch")
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
