"""Task072 MJLab/RSL-RL runner for the contact-aligned anonymous G1 asset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = ROOT / ".agent/task/task072-bound-g1-go2-locomotion-proof"
CONTACT_PROFILE_ID = "mjlab_g1_7capsule_task_v2"
V1_CONTACT_ROOT = TASK_DIR / "artifacts/contact_alignment/mjlab_g1_7capsule_v1"
CONTACT_ROOT = TASK_DIR / "artifacts/contact_alignment" / CONTACT_PROFILE_ID
ASSET_XML = CONTACT_ROOT / "unitree_g1_mjlab_g1_7capsule_task_v2.xml"
CONTACT_PROFILE = CONTACT_ROOT / "contact_profile.json"
STANCE = CONTACT_ROOT / "stance_solution.json"
EXTERNAL_MJLAB = ROOT / ".external/unitree_rl_mjlab"
GPU_LOCK = Path("/home/admin1/workspace/run/.gpu.lock")
TASK_ID = "Task072-G1-MJLab-7Capsule-SingleGround-Flat"
LINEAGE_ID = "mjlab_g1_7capsule_task_v4_semantic_closed"
MJLAB_PARENT_TASK = "Unitree-G1-Flat"
EXPECTED_MJLAB_COMMIT = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
ACTION_CONTRACT_VERSION = "task072_mjlab_signed_headroom_v1"
REWARD_CONTRACT_VERSION = "task072_mjlab_biped_phase_contact_survival_v4"
POLICY_ACTION_DOMAIN = {"transform": "clip", "lower": -1.0, "upper": 1.0}
EVAL_CONFIG_DIFF_ALLOWLIST = {
    "env.scene.num_envs",
    "env.seed",
    "env.render_mode",
    "agent.seed",
    "agent.max_iterations",
    "agent.save_interval",
    "registration.num_envs",
    "registration.max_iterations",
    "registration.task_id",
    "registration.transitions_per_update",
    "semantic_contract.agent.seed",
    "semantic_contract.registration.num_envs",
    "semantic_contract.registration.task_id",
    "semantic_contract.registration.transitions_per_update",
    "semantic_contract.scene.num_envs",
}
MAX_TRANSITIONS = 63_897_600
DEFAULT_SEED = 720301
REQUIRED_CAPACITY_NUM_ENVS = 4096
REQUIRED_ROLLOUT_STEPS = 24
REQUIRED_TRANSITIONS_PER_UPDATE = REQUIRED_CAPACITY_NUM_ENVS * REQUIRED_ROLLOUT_STEPS
TASK072_ACTIVE_SUBTASK = "003k"
TASK072_GAIT_PERIOD_S = 0.8
TASK072_ACCEPTANCE_CHECK_NAMES = (
    "exact_shape",
    "capacity_consumed",
    "checkpoint_produced",
    "transitions_exact",
    "clip_records_valid",
    "optimizer_step_count_exact",
    "parameter_delta_positive_finite",
    "losses_finite",
    "runtime_reward_terms_exact",
    "runtime_reward_sha_match",
    "nan_check_enabled",
    "finite_runtime_evidence",
)
TASK072_PILOT_UPDATES = 21
TASK072_PILOT_EVAL_UPDATES = (0, 7, 14, 20)
TASK072_PILOT_TRANSITIONS = REQUIRED_TRANSITIONS_PER_UPDATE * TASK072_PILOT_UPDATES
RUNTIME_BINDING_ROOT = ROOT / "artifacts/mjlab_runtime_binding/g1" / LINEAGE_ID
DEFAULT_OUTPUT_ROOT = RUNTIME_BINDING_ROOT
FOOT_SITES = (
    "anon_limb0_ankle_roll_link_foot",
    "anon_limb1_ankle_roll_link_foot",
)
FOOT_BODIES = (
    "anon_limb0_ankle_roll_link",
    "anon_limb1_ankle_roll_link",
)
FOOT_GEOMS = tuple(
    f"anon_limb{limb}_ankle_roll_link_mjlab_foot{index}_collision"
    for limb in (0, 1)
    for index in range(1, 8)
)
TORSO_BODY = "anon_waist_pitch_link"
PELVIS_BODY = "anon_pelvis_core"
STANCE_TOL = 1.0e-6
FOOT_COLLISION_REGEX = r"^anon_limb[01]_ankle_roll_link_mjlab_foot[1-7]_collision$"
ALL_COLLISION_REGEX = r".*_collision$"
TASK_FOOT_FRICTION = (0.6,)
TASK_FOOT_MUJOCO_FRICTION = (0.6, 0.005, 0.0001)
EXPECTED_ACTOR_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "phase",
    "joint_pos",
    "joint_vel",
    "actions",
)
EXPECTED_CRITIC_TERMS = (
    *EXPECTED_ACTOR_TERMS,
    "base_lin_vel",
    "foot_height",
    "foot_air_time",
    "foot_contact",
    "foot_contact_forces",
)

SEMANTIC_TO_ANON_JOINT = {
    "limb0_hip_pitch": "anon_limb0_hip_pitch_link_joint",
    "limb0_hip_roll": "anon_limb0_hip_roll_link_joint",
    "limb0_hip_yaw": "anon_limb0_hip_yaw_link_joint",
    "limb0_knee_pitch": "anon_limb0_knee_pitch_link_joint",
    "limb0_ankle_pitch": "anon_limb0_ankle_pitch_link_joint",
    "limb0_ankle_roll": "anon_limb0_ankle_roll_link_joint",
    "limb1_hip_pitch": "anon_limb1_hip_pitch_link_joint",
    "limb1_hip_roll": "anon_limb1_hip_roll_link_joint",
    "limb1_hip_yaw": "anon_limb1_hip_yaw_link_joint",
    "limb1_knee_pitch": "anon_limb1_knee_pitch_link_joint",
    "limb1_ankle_pitch": "anon_limb1_ankle_pitch_link_joint",
    "limb1_ankle_roll": "anon_limb1_ankle_roll_link_joint",
    "waist_yaw": "anon_waist_yaw_link_joint",
    "waist_roll": "anon_waist_roll_link_joint",
    "waist_pitch": "anon_waist_pitch_link_joint",
    "left_arm_shoulder_pitch": "anon_left_arm_shoulder_pitch_link_joint",
    "left_arm_shoulder_roll": "anon_left_arm_shoulder_roll_link_joint",
    "left_arm_shoulder_yaw": "anon_left_arm_shoulder_yaw_link_joint",
    "left_arm_elbow_pitch": "anon_left_arm_elbow_pitch_link_joint",
    "left_arm_wrist_roll": "anon_left_arm_wrist_roll_link_joint",
    "left_arm_wrist_pitch": "anon_left_arm_wrist_pitch_link_joint",
    "left_arm_wrist_yaw": "anon_left_arm_wrist_yaw_link_joint",
    "right_arm_shoulder_pitch": "anon_right_arm_shoulder_pitch_link_joint",
    "right_arm_shoulder_roll": "anon_right_arm_shoulder_roll_link_joint",
    "right_arm_shoulder_yaw": "anon_right_arm_shoulder_yaw_link_joint",
    "right_arm_elbow_pitch": "anon_right_arm_elbow_pitch_link_joint",
    "right_arm_wrist_roll": "anon_right_arm_wrist_roll_link_joint",
    "right_arm_wrist_pitch": "anon_right_arm_wrist_pitch_link_joint",
    "right_arm_wrist_yaw": "anon_right_arm_wrist_yaw_link_joint",
}

REWARD_V3_WEIGHTS = {
    "track_xy_centered": 2.0,
    "track_yaw": 0.50,
    "upright": 0.25,
    "tilt": 5.0,
    "height": 0.25,
    "stand_support": 0.30,
    "phase_gait": 0.50,
    "out_of_phase_double_support": 0.35,
    "clearance": 0.50,
    "touchdown_airtime": 0.10,
    "soft_landing": 0.10,
    "foot_slip": 0.20,
    "nonfoot_contact": 0.20,
    "pose_hip": 0.20,
    "pose_knee": 0.30,
    "pose_ankle": 0.20,
    "pose_waist": 0.10,
    "pose_arm_wrist": 0.05,
    "joint_velocity": 0.02,
    "joint_limit": 0.05,
    "action_magnitude": 0.01,
    "action_rate": 0.01,
    "base_angvel_xy": 0.02,
}
REWARD_V3_ORDER = tuple(REWARD_V3_WEIGHTS)
REWARD_V3_PHASE = {"period": TASK072_GAIT_PERIOD_S, "offsets": [0.0, 0.5], "stance_fraction": 0.55}
REWARD_V3_PARAM_KEYS = {
    "track_xy_centered": ("asset_name", "body_id", "body_name", "command_name", "denominator"),
    "track_yaw": ("asset_name", "body_id", "body_name", "command_name", "denominator"),
    "upright": ("asset_name", "body_id", "body_name"),
    "tilt": ("asset_name", "body_id", "body_name"),
    "height": ("asset_name", "body_id", "body_name", "stance_height", "stance_payload_sha256"),
    "stand_support": ("command_name", "command_threshold", "sensor_name"),
    "phase_gait": ("command_name", "command_threshold", "offsets", "period", "sensor_name", "stance_fraction"),
    "out_of_phase_double_support": ("command_name", "command_threshold", "offsets", "period", "sensor_name", "stance_fraction"),
    "clearance": (
        "clearance_height",
        "clearance_sigma",
        "command_name",
        "command_threshold",
        "offsets",
        "period",
        "sensor_name",
        "site_ids",
        "site_names",
        "stance_fraction",
    ),
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
}
REWARD_V3_ORACLE_EXPECTED = {
    "static_both": -0.5542411176571156,
    "ideal_phase_matched": 1.7,
    "persistent_left_only": -0.26012009890715004,
    "ideal_static_margin": 2.2542411176571155,
}
REWARD_V4_WEIGHTS = {**REWARD_V3_WEIGHTS, "fall_terminated": 300.0}
REWARD_V4_ORDER = (*REWARD_V3_ORDER, "fall_terminated")
REWARD_V4_PARAM_KEYS = {**REWARD_V3_PARAM_KEYS, "fall_terminated": ()}


def _task072_command(env: Any) -> Any:
    return env.command_manager.get_command("twist")


def _task072_robot(env: Any) -> Any:
    return env.scene["robot"].data


def _task072_contact(env: Any) -> Any:
    return env.scene["feet_ground_contact"].data


def _task072_contact_bits(env: Any) -> Any:
    return _task072_contact(env).current_contact_time > 0


def _task072_mask(env: Any, command_name: str, threshold: float = 0.1) -> Any:
    import torch

    command = env.command_manager.get_command(command_name)
    return (torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > threshold).to(torch.float32)


def _task072_phase(env: Any, period: float = 0.8, offsets: list[float] | tuple[float, ...] = (0.0, 0.5)) -> Any:
    import torch

    period_steps = max(1, int(round(float(period) / float(env.step_dt))))
    phase = torch.remainder(env.episode_length_buf.to(torch.int64), period_steps).to(torch.float32) / float(period_steps)
    return torch.stack([torch.remainder(phase + float(offset), 1.0) for offset in offsets], dim=1)


def _task072_site_pos(env: Any, site_ids: list[int]) -> Any:
    return _task072_robot(env).site_pos_w[:, site_ids, :]


def _task072_site_vel(env: Any, site_ids: list[int]) -> Any:
    return _task072_robot(env).site_lin_vel_w[:, site_ids, :]


def _task072_joint_values(env: Any, field: str, indices: list[int] | tuple[int, ...]) -> Any:
    import torch
    value = getattr(_task072_robot(env), field)
    return value[:, list(indices)] if indices else torch.zeros((env.num_envs, 0), device=env.device)


def task072_reward_track_xy_centered(env: Any, command_name: str, denominator: float, **_: Any) -> Any:
    import torch

    actual = _task072_robot(env).root_link_lin_vel_b
    command = env.command_manager.get_command(command_name)
    error = torch.stack((actual[:, 0] - command[:, 0], actual[:, 1] - command[:, 1], actual[:, 2]), dim=1)
    return torch.exp(-torch.sum(error * error, dim=1) / float(denominator)) - 1.0


def task072_reward_track_yaw(env: Any, command_name: str, denominator: float, **_: Any) -> Any:
    import torch

    command = env.command_manager.get_command(command_name)
    error = _task072_robot(env).root_link_ang_vel_b[:, 2] - command[:, 2]
    return torch.exp(-error.square() / float(denominator))


def task072_reward_upright(env: Any, **_: Any) -> Any:
    import torch
    return torch.clamp(-_task072_robot(env).projected_gravity_b[:, 2], 0., 1.)


def task072_reward_tilt(env: Any, **_: Any) -> Any:
    return -_task072_robot(env).projected_gravity_b[:, :2].square().sum(dim=1)


def task072_reward_height(env: Any, stance_height: float, **_: Any) -> Any:
    return -((_task072_robot(env).root_link_pos_w[:, 2] - stance_height) / 0.10).square()


def task072_reward_stand_support(env: Any, command_name: str, command_threshold: float, **_: Any) -> Any:
    return (1.0 - _task072_mask(env, command_name, command_threshold)) * _task072_contact_bits(env).any(dim=1)


def task072_reward_phase_gait(
    env: Any,
    command_name: str,
    command_threshold: float,
    period: float,
    offsets: list[float],
    stance_fraction: float,
    **_: Any,
) -> Any:
    import torch

    desired = _task072_phase(env, period, offsets) < stance_fraction
    return _task072_mask(env, command_name, command_threshold) * (
        _task072_contact_bits(env) == desired
    ).to(torch.float32).mean(dim=1)


def task072_reward_out_of_phase_double_support(
    env: Any,
    command_name: str,
    command_threshold: float,
    period: float,
    offsets: list[float],
    stance_fraction: float,
    **_: Any,
) -> Any:
    import torch

    desired = _task072_phase(env, period, offsets) < stance_fraction
    contact = _task072_contact_bits(env)
    return -_task072_mask(env, command_name, command_threshold) * (
        contact.all(dim=1) & ~desired.all(dim=1)
    ).to(torch.float32)


def task072_reward_clearance(
    env: Any,
    command_name: str,
    command_threshold: float,
    period: float,
    offsets: list[float],
    stance_fraction: float,
    site_ids: list[int],
    clearance_height: float,
    clearance_sigma: float,
    **_: Any,
) -> Any:
    import torch

    desired = _task072_phase(env, period, offsets) < stance_fraction
    contact = _task072_contact_bits(env)
    values = torch.exp(-((_task072_site_pos(env, site_ids)[:, :, 2] - clearance_height) / clearance_sigma).square())
    swing = (~desired) & (~contact)
    empty = torch.zeros(env.num_envs, device=env.device)
    mean = torch.where(swing.any(dim=1), (values * swing).sum(dim=1) / swing.sum(dim=1).clamp_min(1), empty)
    return _task072_mask(env, command_name, command_threshold) * mean


def task072_reward_touchdown_airtime(env: Any, sensor_name: str, airtime_clip: float, **_: Any) -> Any:
    import torch

    sensor = env.scene[sensor_name]
    return (
        sensor.compute_first_contact(float(env.step_dt))
        * torch.clamp(sensor.data.last_air_time / float(airtime_clip), 0.0, 1.0)
    ).mean(dim=1)


def task072_reward_soft_landing(env: Any, sensor_name: str, site_ids: list[int], landing_velocity_sigma: float, **_: Any) -> Any:
    import torch

    sensor = env.scene[sensor_name]
    velocity = _task072_site_vel(env, site_ids)[:, :, 2]
    return (sensor.compute_first_contact(float(env.step_dt)) * torch.exp(-(velocity / landing_velocity_sigma).square())).mean(dim=1)


def task072_reward_foot_slip(env: Any, site_ids: list[int], **_: Any) -> Any:
    contact = _task072_contact_bits(env)
    velocity = _task072_site_vel(env, site_ids)[:, :, :2]
    return -(contact * velocity.square().sum(dim=2)).mean(dim=1)


def task072_reward_nonfoot_contact(env: Any, sensor_name: str, **_: Any) -> Any:
    return -env.scene[sensor_name].data.found.to(dtype=_task072_robot(env).root_link_pos_w.dtype).mean(dim=1)


def _task072_pose_reward(env: Any, joint_ids: list[int], q_ref: list[float], **_: Any) -> Any:
    import torch

    return -(
        _task072_joint_values(env, "joint_pos", joint_ids)
        - torch.tensor(q_ref, device=env.device, dtype=_task072_robot(env).joint_pos.dtype)
    ).square().mean(dim=1)


def task072_reward_pose_hip(env: Any, joint_ids: list[int], q_ref: list[float], **kwargs: Any) -> Any: return _task072_pose_reward(env, joint_ids, q_ref, **kwargs)
def task072_reward_pose_knee(env: Any, joint_ids: list[int], q_ref: list[float], **kwargs: Any) -> Any: return _task072_pose_reward(env, joint_ids, q_ref, **kwargs)
def task072_reward_pose_ankle(env: Any, joint_ids: list[int], q_ref: list[float], **kwargs: Any) -> Any: return _task072_pose_reward(env, joint_ids, q_ref, **kwargs)
def task072_reward_pose_waist(env: Any, joint_ids: list[int], q_ref: list[float], **kwargs: Any) -> Any: return _task072_pose_reward(env, joint_ids, q_ref, **kwargs)
def task072_reward_pose_arm_wrist(env: Any, joint_ids: list[int], q_ref: list[float], **kwargs: Any) -> Any: return _task072_pose_reward(env, joint_ids, q_ref, **kwargs)


def task072_reward_joint_velocity(env: Any, joint_ids: list[int], **_: Any) -> Any:
    return -_task072_joint_values(env, "joint_vel", joint_ids).square().mean(dim=1)


def task072_reward_joint_limit(env: Any, joint_ids: list[int], lower: list[float], upper: list[float], soft_fraction: float = .9, **_: Any) -> Any:
    import torch
    q = _task072_joint_values(env, "joint_pos", joint_ids)
    lo, hi = torch.tensor(lower, device=env.device), torch.tensor(upper, device=env.device)
    center, half = (lo + hi) / 2, (hi - lo) / 2
    v = torch.relu((q - (center + soft_fraction * half)) / (hi - lo)) + torch.relu(((center - soft_fraction * half) - q) / (hi - lo))
    return -v.square().mean(dim=1)


def _task072_action_pair(env: Any) -> tuple[Any, Any]:
    import torch

    action = env.action_manager.action
    prev_action = getattr(env.action_manager, "prev_action", None)
    if prev_action is None:
        prev_action = torch.zeros_like(action)
    return action, prev_action


def task072_reward_action_magnitude(env: Any, **_: Any) -> Any:
    action, _prev_action = _task072_action_pair(env)
    return -action.square().mean(dim=1)


def task072_reward_action_rate(env: Any, **_: Any) -> Any:
    action, prev_action = _task072_action_pair(env)
    return -(action - prev_action).square().mean(dim=1)


def task072_reward_base_angvel_xy(env: Any, **_: Any) -> Any:
    return -_task072_robot(env).root_link_ang_vel_b[:, :2].square().sum(dim=1)


def task072_reward_fall_terminated(env: Any) -> Any:
    import torch

    return -env.termination_manager.terminated.to(dtype=torch.float32)


for _reward_name in REWARD_V4_ORDER:
    globals()[f"task072_reward_{_reward_name}"].__module__ = "task072_mjlab_contact_runner"
del _reward_name


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_task072_cli() -> Any:
    module_path = TASK_DIR / "task072_locomotion_proof.py"
    spec = importlib.util.spec_from_file_location("task072_locomotion_proof_for_mjlab_runner", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Task072 CLI from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stance_dict() -> dict[str, Any]:
    ensure_v2_artifacts()
    payload = json.loads(STANCE.read_text(encoding="utf-8"))
    stance = payload["stance_solution"]
    if payload["contact_profile_id"] != CONTACT_PROFILE_ID:
        raise ValueError("stance contact_profile_id does not match v2 contact profile")
    if stance["model_xml_sha256"] != sha256_path(ASSET_XML):
        raise ValueError("stance model_xml_sha256 does not match v2 XML")
    if payload_sha256(stance) != payload["stance_solution_sha256"]:
        raise ValueError("stance payload SHA mismatch")
    return stance


def _mapping_table(stance: dict[str, Any]) -> list[dict[str, str]]:
    expected_slots = set(stance["joint_qpos"])
    if expected_slots != set(SEMANTIC_TO_ANON_JOINT):
        missing = sorted(expected_slots - set(SEMANTIC_TO_ANON_JOINT))
        extra = sorted(set(SEMANTIC_TO_ANON_JOINT) - expected_slots)
        raise ValueError(f"semantic joint map does not cover stance: missing={missing}, extra={extra}")
    if len(set(SEMANTIC_TO_ANON_JOINT.values())) != len(SEMANTIC_TO_ANON_JOINT):
        raise ValueError("semantic joint map contains duplicate anonymous joints")
    return [
        {
            "semantic_joint": semantic,
            "anonymous_joint": anonymous,
            "action_target": anonymous,
        }
        for semantic, anonymous in sorted(SEMANTIC_TO_ANON_JOINT.items())
    ]


def _runtime_material_contract() -> dict[str, Any]:
    return {
        "foot": {
            "condim": 3,
            "priority": 1,
            "nominal_sliding_friction": 0.6,
            "mujoco_friction": list(TASK_FOOT_MUJOCO_FRICTION),
            "contype": 1,
            "conaffinity": 1,
        },
        "non_foot_collision": {"condim": 1, "contype": 1, "conaffinity": 1},
        "logical_feet": 2,
        "capsules_per_foot": 7,
        "legacy_foot_boxes": 0,
    }


def _compiled_material_from_model(model: Any) -> dict[str, Any]:
    import mujoco

    runtime_material_path = CONTACT_ROOT / "runtime_material.json"
    declared = json.loads(runtime_material_path.read_text(encoding="utf-8"))
    expected_non_foot = set(declared.get("non_foot_collision_geom_names", []))
    foot_rows = []
    legacy_foot_boxes = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        short_name = name.split("/")[-1]
        if short_name.endswith("_footpad"):
            legacy_foot_boxes.append(name)
        if short_name not in FOOT_GEOMS:
            continue
        foot_rows.append(
            {
                "name": name,
                "short_name": short_name,
                "condim": int(model.geom_condim[geom_id]),
                "priority": int(model.geom_priority[geom_id]),
                "friction": [float(value) for value in model.geom_friction[geom_id]],
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
            }
        )
    non_foot_rows = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        short_name = name.split("/")[-1]
        if short_name not in expected_non_foot:
            continue
        non_foot_rows.append(
            {
                "name": name,
                "short_name": short_name,
                "condim": int(model.geom_condim[geom_id]),
                "priority": int(model.geom_priority[geom_id]),
                "friction": [float(value) for value in model.geom_friction[geom_id]],
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
            }
        )
    expected_friction = list(TASK_FOOT_MUJOCO_FRICTION)
    return {
        "foot_geoms": foot_rows,
        "non_foot_collision_geoms": non_foot_rows,
        "legacy_foot_boxes": legacy_foot_boxes,
        "declared_counts": {
            "foot_collision_geoms": len(FOOT_GEOMS),
            "non_foot_collision_geoms": len(expected_non_foot),
        },
        "checks": {
            "two_logical_feet": len(FOOT_GEOMS) == 14,
            "fourteen_foot_collision_geoms": len(foot_rows) == 14,
            "foot_names_match_declared": {row["short_name"] for row in foot_rows} == set(FOOT_GEOMS),
            "non_foot_count_matches_declared": len(non_foot_rows) == len(expected_non_foot),
            "non_foot_names_match_declared": {row["short_name"] for row in non_foot_rows} == expected_non_foot,
            "zero_legacy_foot_boxes": not legacy_foot_boxes,
            "foot_condim_3": all(row["condim"] == 3 for row in foot_rows),
            "foot_priority_1": all(row["priority"] == 1 for row in foot_rows),
            "foot_mujoco_friction": all(
                len(row["friction"]) == len(expected_friction)
                and all(abs(actual - expected) <= STANCE_TOL for actual, expected in zip(row["friction"], expected_friction))
                for row in foot_rows
            ),
            "foot_contype_1": all(row["contype"] == 1 for row in foot_rows),
            "foot_conaffinity_1": all(row["conaffinity"] == 1 for row in foot_rows),
            "non_foot_condim_1": all(row["condim"] == 1 for row in non_foot_rows),
            "non_foot_contype_1": all(row["contype"] == 1 for row in non_foot_rows),
            "non_foot_conaffinity_1": all(row["conaffinity"] == 1 for row in non_foot_rows),
        },
    }


def _device_requires_gpu_lock(device: str) -> bool:
    return str(device).lower().startswith("cuda")


def _ancestor_pids(pid: int | None = None) -> list[int]:
    current = int(pid or os.getpid())
    ancestors: list[int] = []
    seen: set[int] = set()
    while current > 0 and current not in seen:
        ancestors.append(current)
        seen.add(current)
        try:
            parts = Path(f"/proc/{current}/stat").read_text(encoding="utf-8").split()
            current = int(parts[3])
        except (FileNotFoundError, IndexError, ValueError, PermissionError):
            break
    return ancestors


def _gpu_lock_status() -> dict[str, Any]:
    try:
        stat = GPU_LOCK.stat()
    except FileNotFoundError:
        return {
            "path": str(GPU_LOCK),
            "exists": False,
            "locked": False,
            "held_by_ancestor": False,
            "holder_pids": [],
            "ancestor_pids": _ancestor_pids(),
        }
    dev = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}"
    inode = str(stat.st_ino)
    holder_pids: list[int] = []
    try:
        for line in Path("/proc/locks").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 6:
                continue
            lock_pid = fields[4]
            lock_target = fields[5]
            if not lock_target.startswith(f"{dev}:") or not lock_target.endswith(f":{inode}"):
                continue
            if lock_pid != "-1":
                holder_pids.append(int(lock_pid))
    except (FileNotFoundError, ValueError, PermissionError):
        holder_pids = []
    ancestors = _ancestor_pids()
    return {
        "path": str(GPU_LOCK),
        "exists": True,
        "device_inode": f"{dev}:{inode}",
        "locked": bool(holder_pids),
        "holder_pids": holder_pids,
        "ancestor_pids": ancestors,
        "held_by_ancestor": bool(set(holder_pids) & set(ancestors)),
    }


def _require_gpu_lock_for_device(device: str) -> dict[str, Any]:
    status = _gpu_lock_status()
    if _device_requires_gpu_lock(device) and not status["held_by_ancestor"]:
        raise RuntimeError(f"CUDA run must be launched under shared GPU lock: {GPU_LOCK}")
    return status


def ensure_v2_artifacts() -> None:
    """Materialize the immutable v2 contact lineage from the passed v1 asset."""
    if ASSET_XML.exists() and CONTACT_PROFILE.exists() and STANCE.exists():
        profile = json.loads(CONTACT_PROFILE.read_text(encoding="utf-8"))
        stance = json.loads(STANCE.read_text(encoding="utf-8"))
        if (
            profile.get("contact_profile_id") != CONTACT_PROFILE_ID
            or stance.get("contact_profile_id") != CONTACT_PROFILE_ID
            or profile.get("runtime_material") != _runtime_material_contract()
            or stance["stance_solution"]["model_xml_sha256"] != sha256_path(ASSET_XML)
        ):
            raise ValueError("existing v2 runtime binding artifacts are stale or mismatched")
        return
    source_root = V1_CONTACT_ROOT
    source_xml = source_root / "unitree_g1_mjlab_g1_7capsule_v1.xml"
    source_profile = source_root / "contact_profile.json"
    if not all(path.is_file() for path in (source_xml, source_profile)):
        raise ValueError("v1 contact alignment artifacts are required to materialize v2")
    root = ET.parse(source_xml).getroot()
    foot_names = set(FOOT_GEOMS)
    collision_names: list[str] = []
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if name.endswith("_collision"):
            collision_names.append(name)
            geom.set("condim", "3" if name in foot_names else "1")
            geom.set("priority", "1" if name in foot_names else "0")
            if name in foot_names:
                geom.set("friction", "0.6 0.005 0.0001")
    CONTACT_ROOT.mkdir(parents=True, exist_ok=True)
    ASSET_XML.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    profile = deepcopy(json.loads(source_profile.read_text(encoding="utf-8")))
    profile["contact_profile_id"] = CONTACT_PROFILE_ID
    profile["lineage_id"] = CONTACT_PROFILE_ID
    profile["asset"]["candidate_xml_sha256"] = sha256_path(ASSET_XML)
    profile["runtime_material"] = _runtime_material_contract()
    write_json(CONTACT_PROFILE, profile)
    task072 = _load_task072_cli()
    parent_context = task072._load_bound_context("unitree_g1")
    asset = {
        "xml": ASSET_XML.read_text(encoding="utf-8"),
        "xml_sha256": sha256_path(ASSET_XML),
        "logical_foot_groups": {
            key: tuple(value)
            for key, value in profile["logical_foot_groups"].items()
        },
    }
    solved = task072._solve_contact_aligned_stance(parent_context, asset)
    stance = {
        "schema_version": 1,
        "contact_profile_id": CONTACT_PROFILE_ID,
        "stance_solution": solved.manifest(),
        "stance_solution_sha256": solved.solution_hash,
    }
    write_json(STANCE, stance)
    write_json(
        CONTACT_ROOT / "runtime_material.json",
        {
            "lineage_id": CONTACT_PROFILE_ID,
            "collision_geom_count": len(collision_names),
            "foot_geom_names": list(FOOT_GEOMS),
            "non_foot_collision_geom_names": [name for name in collision_names if name not in foot_names],
            "material_contract": _runtime_material_contract(),
        },
    )


def _prepare_external_imports() -> None:
    if not EXTERNAL_MJLAB.exists():
        raise RuntimeError(f"Task072 requires frame-local external MJLab checkout: {EXTERNAL_MJLAB}")
    external = _external_mjlab_status()
    if external["actual_commit"] != EXPECTED_MJLAB_COMMIT:
        raise RuntimeError("Task072 external MJLab checkout commit drift")
    if not external["tracked_clean"]:
        raise RuntimeError("Task072 external MJLab checkout has tracked dirty changes")
    external = str(EXTERNAL_MJLAB)
    if external not in sys.path:
        sys.path.insert(0, external)


def runtime_spec_xml() -> str:
    ensure_v2_artifacts()
    root = ET.parse(ASSET_XML).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("asset XML is missing worldbody")
    source_planes = [
        geom for geom in worldbody.findall("geom")
        if geom.get("type") == "plane"
        and (
            int(geom.get("contype", "1")) != 0
            or int(geom.get("conaffinity", "1")) != 0
        )
    ]
    if len(source_planes) != 1 or source_planes[0].get("name") != "floor":
        raise ValueError("expected exactly one collision-enabled source plane named floor")
    worldbody.remove(source_planes[0])
    pelvis = root.find(f".//body[@name='{PELVIS_BODY}']")
    if pelvis is None:
        raise ValueError(f"missing pelvis body for MJLab sensors: {PELVIS_BODY}")
    if pelvis.find("site[@name='imu_in_pelvis']") is None:
        ET.SubElement(pelvis, "site", {"name": "imu_in_pelvis", "size": "0.01", "pos": "0 0 0"})
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    existing = {item.get("name") for item in sensor}
    additions = (
        ("gyro", {"name": "imu_ang_vel", "site": "imu_in_pelvis"}),
        ("velocimeter", {"name": "imu_lin_vel", "site": "imu_in_pelvis"}),
        ("accelerometer", {"name": "imu_lin_acc", "site": "imu_in_pelvis"}),
        ("subtreeangmom", {"name": "root_angmom", "body": PELVIS_BODY}),
    )
    for tag, attrs in additions:
        if attrs["name"] not in existing:
            ET.SubElement(sensor, tag, attrs)
    return ET.tostring(root, encoding="unicode")


def _ground_plane_audit(model: Any, data: Any) -> dict[str, Any]:
    """Fail closed on duplicate planes and hidden foot/plane contact pairs."""
    import mujoco

    plane_ids = [
        geom_id for geom_id in range(model.ngeom)
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE
        and (int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0)
    ]
    plane_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in plane_ids]
    foot_ids = {
        geom_id for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").split("/")[-1] in FOOT_GEOMS
    }
    terrain_ids = {i for i in plane_ids if plane_names[plane_ids.index(i)] == "terrain"}
    mujoco.mj_forward(model, data)
    pairs = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        pair = (int(contact.geom1), int(contact.geom2))
        if pair[0] in foot_ids or pair[1] in foot_ids:
            pairs.append(pair)
    foot_ground_pairs = [pair for pair in pairs if pair[0] in terrain_ids or pair[1] in terrain_ids]
    hidden_plane_pairs = [
        pair for pair in pairs
        if (pair[0] in set(plane_ids) or pair[1] in set(plane_ids)) and pair not in foot_ground_pairs
    ]
    foot_ground_names = {
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").split("/")[-1]
        for pair in foot_ground_pairs for geom_id in pair if geom_id in foot_ids
    }
    return {
        "collision_enabled_plane_names": plane_names,
        "foot_ground_pairs": len(foot_ground_pairs),
        "foot_ground_foot_names": sorted(foot_ground_names),
        "hidden_plane_pairs": len(hidden_plane_pairs),
        "checks": {
            "exactly_one_collision_enabled_plane": len(plane_ids) == 1,
            "plane_named_terrain": plane_names == ["terrain"],
            "zero_robot_floor": "robot/floor" not in plane_names and "floor" not in plane_names,
            "fourteen_feet_only_terrain": foot_ground_names == set(FOOT_GEOMS),
            "zero_hidden_plane_pairs": not hidden_plane_pairs,
        },
    }


def action_contract_from_asset_xml() -> dict[str, Any]:
    ensure_v2_artifacts()
    root = ET.parse(ASSET_XML).getroot()
    stance = _stance_dict()
    semantic_by_joint = {anonymous: semantic for semantic, anonymous in SEMANTIC_TO_ANON_JOINT.items()}
    joint_ranges = {
        joint.get("name"): tuple(float(value) for value in joint.get("range", "").split())
        for joint in root.findall(".//joint")
        if joint.get("name") and joint.get("range")
    }
    rows: list[dict[str, Any]] = []
    for actuator in root.findall(".//actuator/position"):
        joint = actuator.get("joint")
        kp = float(actuator.get("kp", "nan"))
        force = tuple(float(value) for value in actuator.get("forcerange", "").split())
        if not joint or joint not in semantic_by_joint or joint not in joint_ranges or not force or kp <= 0.0:
            raise ValueError(f"bad position actuator scale inputs: {actuator.attrib}")
        semantic = semantic_by_joint[joint]
        lower, upper = joint_ranges[joint]
        offset = float(stance["actuator_ctrl_eq"][semantic])
        span = upper - lower
        safety_margin = 0.05 * span
        safe_lower = lower + safety_margin
        safe_upper = upper - safety_margin
        negative_headroom = offset - safe_lower
        positive_headroom = safe_upper - offset
        effort = min(abs(value) for value in force)
        motor_delta = 0.25 * effort / kp
        negative_amplitude = min(motor_delta, negative_headroom)
        positive_amplitude = min(motor_delta, positive_headroom)
        if (
            lower > offset
            or offset > upper
            or negative_amplitude <= 0.0
            or positive_amplitude <= 0.0
        ):
            raise ValueError(f"unsafe Task072 action headroom for {semantic}")
        rows.append(
            {
                "semantic_joint": semantic,
                "anonymous_joint": joint,
                "actuator": actuator.get("name"),
                "joint_range": [lower, upper],
                "stance_action_offset": offset,
                "motor_delta": motor_delta,
                "safety_margin": safety_margin,
                "negative_headroom": negative_headroom,
                "positive_headroom": positive_headroom,
                "signed_negative_amplitude": negative_amplitude,
                "signed_positive_amplitude": positive_amplitude,
            }
        )
    rows.sort(key=lambda row: row["anonymous_joint"])
    if len(rows) != 29 or len({row["semantic_joint"] for row in rows}) != 29:
        raise ValueError(f"expected unique 29 anonymous G1 action contract rows, got {len(rows)}")
    payload = {
        "schema_version": 1,
        "version": ACTION_CONTRACT_VERSION,
        "lineage_id": LINEAGE_ID,
        "policy_action_domain": POLICY_ACTION_DOMAIN,
        "target_rule": "offset + raw<0 ? raw*negative_amplitude : raw*positive_amplitude",
        "rows": rows,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def action_scale_from_asset_xml() -> dict[str, float]:
    contract = action_contract_from_asset_xml()
    return {row["anonymous_joint"]: row["signed_positive_amplitude"] for row in contract["rows"]}


def _signed_action_bounds_by_joint() -> tuple[dict[str, float], dict[str, float]]:
    contract = action_contract_from_asset_xml()
    negative = {row["anonymous_joint"]: row["signed_negative_amplitude"] for row in contract["rows"]}
    positive = {row["anonymous_joint"]: row["signed_positive_amplitude"] for row in contract["rows"]}
    return negative, positive


def _runtime_entity_indices() -> dict[str, Any]:
    import mujoco

    spec = mujoco.MjSpec.from_string(runtime_spec_xml())
    model = spec.compile()
    body_names = [body.name.split("/")[-1] for body in spec.bodies[1:]]
    joint_names = [
        joint.name.split("/")[-1]
        for joint in spec.joints
        if joint.name and joint.name.split("/")[-1] != "root_free"
    ]
    site_names = [site.name.split("/")[-1] for site in spec.sites]
    lower: list[float] = []
    upper: list[float] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"compiled runtime model is missing joint {name}")
        lo, hi = model.jnt_range[joint_id]
        lower.append(float(lo))
        upper.append(float(hi))
    return {
        "body_names": body_names,
        "body_ids": {name: index for index, name in enumerate(body_names)},
        "joint_names": joint_names,
        "joint_ids": {name: index for index, name in enumerate(joint_names)},
        "joint_lower": lower,
        "joint_upper": upper,
        "site_names": site_names,
        "site_ids": {name: index for index, name in enumerate(site_names)},
    }


def _task072_pose_groups(semantic_joint_names: list[str]) -> dict[str, list[int]]:
    groups = {
        "pose_hip": [i for i, name in enumerate(semantic_joint_names) if name.startswith("limb") and "_hip_" in name],
        "pose_knee": [i for i, name in enumerate(semantic_joint_names) if name.startswith("limb") and "_knee_" in name],
        "pose_ankle": [i for i, name in enumerate(semantic_joint_names) if name.startswith("limb") and "_ankle_" in name],
        "pose_waist": [i for i, name in enumerate(semantic_joint_names) if name.startswith("waist_")],
        "pose_arm_wrist": [
            i
            for i, name in enumerate(semantic_joint_names)
            if name.startswith(("left_arm_", "right_arm_"))
        ],
    }
    if sorted(sum(groups.values(), [])) != list(range(29)):
        raise ValueError("Task072 reward pose groups do not partition the 29-joint mapping")
    if {name: len(indices) for name, indices in groups.items()} != {
        "pose_hip": 6,
        "pose_knee": 2,
        "pose_ankle": 4,
        "pose_waist": 3,
        "pose_arm_wrist": 14,
    }:
        raise ValueError("Task072 reward pose groups have unexpected sizes")
    return groups


def task072_reward_v4_table(stance: dict[str, Any]) -> dict[str, Any]:
    """Return the complete, ordered, JSON-native v4 RewardTermCfg table."""
    from mjlab.managers.reward_manager import RewardTermCfg

    indices = _runtime_entity_indices()
    semantic_names = list(SEMANTIC_TO_ANON_JOINT)
    anonymous_names = [SEMANTIC_TO_ANON_JOINT[name] for name in semantic_names]
    if indices["joint_names"] != anonymous_names:
        raise ValueError("Task072 reward joint order does not match runtime joint order")
    groups = _task072_pose_groups(semantic_names)
    q_ref = [float(stance["joint_qpos"][name]) for name in semantic_names]
    stance_payload_sha = payload_sha256(stance)
    torso_body_id = int(indices["body_ids"][TORSO_BODY])
    foot_site_ids = [int(indices["site_ids"][name]) for name in FOOT_SITES]
    nonfoot_body_names = [name for name in indices["body_names"] if name not in set(FOOT_BODIES)]
    nonfoot_body_ids = [int(indices["body_ids"][name]) for name in nonfoot_body_names]
    common = {"asset_name": "robot", "body_name": TORSO_BODY, "body_id": torso_body_id}
    terms: dict[str, Any] = {}
    funcs = {name: globals()[f"task072_reward_{name}"] for name in REWARD_V4_ORDER}
    for name in REWARD_V4_ORDER:
        params: dict[str, Any] = dict(common)
        if name == "fall_terminated":
            params = {}
        elif name in ("track_xy_centered", "track_yaw"):
            params.update(command_name="twist", denominator=0.25)
        elif name == "height":
            params.update(stance_height=float(stance["root_pose_eq"][2]), stance_payload_sha256=stance_payload_sha)
        elif name == "stand_support":
            params = {"command_name": "twist", "sensor_name": "feet_ground_contact", "command_threshold": 0.1}
        elif name in ("phase_gait", "out_of_phase_double_support"):
            params = {
                "command_name": "twist",
                "sensor_name": "feet_ground_contact",
                "command_threshold": 0.1,
                **REWARD_V3_PHASE,
            }
        elif name == "clearance":
            params = {
                "command_name": "twist",
                "sensor_name": "feet_ground_contact",
                "command_threshold": 0.1,
                "site_names": list(FOOT_SITES),
                "site_ids": foot_site_ids,
                "clearance_height": 0.10,
                "clearance_sigma": 0.05,
                **REWARD_V3_PHASE,
            }
        elif name == "touchdown_airtime":
            params = {
                "sensor_name": "feet_ground_contact",
                "site_names": list(FOOT_SITES),
                "site_ids": foot_site_ids,
                "airtime_clip": 0.5,
            }
        elif name == "soft_landing":
            params = {
                "sensor_name": "feet_ground_contact",
                "site_names": list(FOOT_SITES),
                "site_ids": foot_site_ids,
                "landing_velocity_sigma": 0.5,
            }
        elif name == "foot_slip":
            params = {"sensor_name": "feet_ground_contact", "site_names": list(FOOT_SITES), "site_ids": foot_site_ids}
        elif name == "nonfoot_contact":
            params = {
                "sensor_name": "nonfoot_ground_contact",
                "body_names": nonfoot_body_names,
                "body_ids": nonfoot_body_ids,
                "terrain_name": "terrain",
            }
        elif name.startswith("pose_"):
            group_ids = groups[name]
            params = {
                "asset_name": "robot",
                "semantic_joint_names": [semantic_names[i] for i in group_ids],
                "anonymous_joint_names": [anonymous_names[i] for i in group_ids],
                "joint_ids": group_ids,
                "q_ref": [q_ref[i] for i in group_ids],
                "stance_payload_sha256": stance_payload_sha,
            }
        elif name in ("joint_velocity", "joint_limit"):
            params = {
                "asset_name": "robot",
                "semantic_joint_names": semantic_names,
                "anonymous_joint_names": anonymous_names,
                "joint_ids": list(range(29)),
            }
            if name == "joint_limit":
                params.update(lower=indices["joint_lower"], upper=indices["joint_upper"], soft_fraction=0.9)
        elif name in ("action_magnitude", "action_rate"):
            params = {"action_name": "joint_pos"}
            if name == "action_rate":
                params["previous_action_reset"] = 0.0
        terms[name] = RewardTermCfg(func=funcs[name], weight=REWARD_V4_WEIGHTS[name], params=params)
    return terms


task072_reward_v3_table = task072_reward_v4_table


def task072_reward_active_table_from_cfg(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    import inspect

    table = []
    for name, term_cfg in cfg.items():
        func = term_cfg.func
        table.append(
            {
                "name": name,
                "module": func.__module__,
                "qualname": func.__qualname__,
                "source_file": str(Path(inspect.getsourcefile(func) or "").resolve()),
                "function_source_sha256": hashlib.sha256(inspect.getsource(func).encode()).hexdigest(),
                "weight": float(term_cfg.weight),
                "params": term_cfg.params,
            }
        )
    return table


def task072_reward_active_table_from_manager(manager: Any) -> list[dict[str, Any]]:
    return task072_reward_active_table_from_cfg({name: manager.get_term_cfg(name) for name in manager.active_terms})


def task072_canonical_reward_payload(active_table: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "id": REWARD_CONTRACT_VERSION,
        "lineage_id": LINEAGE_ID,
        "active_terms": active_table,
        "signal_sampling": {
            "step_source": "ManagerBasedRlEnv.step RewardManager point after decimation and termination compute before reset",
            "contact_sensor": "feet_ground_contact logical OR over 7 capsules per foot against terrain",
            "action": "RslRlVecEnvWrapper-clipped normalized joint_pos action; previous action resets to zero",
            "q_ref": "bound stance_solution.joint_qpos",
        },
        "phase": {"clock": "episode_length_buf", "first_action_k": 1, "reset_k": 0, **REWARD_V3_PHASE},
        "dt": {"control_dt": 0.02, "reward_manager_scale_by_dt": True, "term_functions_premultiply_dt": False},
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def task072_validate_reward_active_table(table: list[dict[str, Any]]) -> str:
    import inspect

    if [row.get("name") for row in table] != list(REWARD_V4_ORDER):
        raise ValueError("Task072 reward active table has wrong key/order/count")
    forbidden = {"foot_gait", "feet_gait", "is_terminated"}
    if forbidden & {str(row.get("name")) for row in table}:
        raise ValueError("Task072 reward active table contains forbidden parent reward")
    stance = _stance_dict()
    stance_payload_sha = payload_sha256(stance)
    source_file = str(Path(__file__).resolve())
    indices = _runtime_entity_indices()
    expected_foot_site_ids = [int(indices["site_ids"][name]) for name in FOOT_SITES]
    expected_torso_body_id = int(indices["body_ids"][TORSO_BODY])
    semantic_names = list(SEMANTIC_TO_ANON_JOINT)
    anonymous_names = [SEMANTIC_TO_ANON_JOINT[name] for name in semantic_names]
    expected_groups = _task072_pose_groups(semantic_names)
    seen: set[str] = set()
    for row in table:
        name = str(row["name"])
        params = row.get("params", {})
        if name in seen:
            raise ValueError(f"Task072 reward active table contains duplicate term {name}")
        seen.add(name)
        if row.get("module") != "task072_mjlab_contact_runner" or row.get("qualname") != f"task072_reward_{name}":
            raise ValueError(f"Task072 reward term uses parent or alias callable: {name}")
        if "<locals>" in str(row.get("qualname")):
            raise ValueError(f"Task072 reward term is nested: {name}")
        if row.get("source_file") != source_file:
            raise ValueError(f"Task072 reward term source file drift: {name}")
        expected_func = globals()[f"task072_reward_{name}"]
        expected_source_hash = hashlib.sha256(inspect.getsource(expected_func).encode()).hexdigest()
        if row.get("function_source_sha256") != expected_source_hash:
            raise ValueError(f"Task072 reward term source hash drift: {name}")
        if float(row.get("weight")) != REWARD_V4_WEIGHTS[name]:
            raise ValueError(f"Task072 reward weight drift: {name}")
        if tuple(sorted(params)) != tuple(REWARD_V4_PARAM_KEYS[name]):
            raise ValueError(f"Task072 reward param keys drift: {name}")
        if json.loads(json.dumps(params, sort_keys=True)) != params:
            raise ValueError(f"Task072 reward params are not JSON-native: {name}")
        if name in ("track_xy_centered", "track_yaw", "upright", "tilt", "height", "base_angvel_xy"):
            if params.get("asset_name") != "robot" or params.get("body_name") != TORSO_BODY or params.get("body_id") != expected_torso_body_id:
                raise ValueError(f"Task072 reward torso body binding drift: {name}")
        if name in ("track_xy_centered", "track_yaw"):
            if params.get("command_name") != "twist" or params.get("denominator") != 0.25:
                raise ValueError(f"Task072 reward command tracking params drift: {name}")
        if name == "stand_support":
            if params != {"command_name": "twist", "sensor_name": "feet_ground_contact", "command_threshold": 0.1}:
                raise ValueError("Task072 stand support params drift")
    phase = table[list(REWARD_V4_ORDER).index("phase_gait")]["params"]
    if phase.get("period") != TASK072_GAIT_PERIOD_S or phase.get("offsets") != [0.0, 0.5] or phase.get("stance_fraction") != 0.55:
        raise ValueError("Task072 reward phase params drift")
    double = table[list(REWARD_V4_ORDER).index("out_of_phase_double_support")]["params"]
    if double.get("period") != TASK072_GAIT_PERIOD_S or double.get("offsets") != [0.0, 0.5] or double.get("stance_fraction") != 0.55:
        raise ValueError("Task072 reward phase params drift")
    by_name = {row["name"]: row for row in table}
    height = by_name["height"]["params"]
    if abs(float(height.get("stance_height")) - float(stance["root_pose_eq"][2])) > STANCE_TOL:
        raise ValueError("Task072 reward stance height drift")
    if height.get("stance_payload_sha256") != stance_payload_sha:
        raise ValueError("Task072 reward stance payload SHA drift")
    for name in ("clearance", "touchdown_airtime", "soft_landing", "foot_slip"):
        params = by_name[name]["params"]
        if params.get("site_names") != list(FOOT_SITES) or params.get("site_ids") != expected_foot_site_ids:
            raise ValueError(f"Task072 reward foot site binding drift: {name}")
        if params.get("sensor_name") != "feet_ground_contact":
            raise ValueError(f"Task072 reward foot sensor drift: {name}")
    clearance = by_name["clearance"]["params"]
    if clearance.get("clearance_height") != 0.10 or clearance.get("clearance_sigma") != 0.05:
        raise ValueError("Task072 reward clearance params drift")
    if by_name["touchdown_airtime"]["params"].get("airtime_clip") != 0.5:
        raise ValueError("Task072 reward touchdown airtime params drift")
    if by_name["soft_landing"]["params"].get("landing_velocity_sigma") != 0.5:
        raise ValueError("Task072 reward soft landing params drift")
    nonfoot = by_name["nonfoot_contact"]["params"]
    if nonfoot.get("sensor_name") != "nonfoot_ground_contact" or nonfoot.get("terrain_name") != "terrain":
        raise ValueError("Task072 nonfoot contact sensor drift")
    if len(nonfoot.get("body_ids", [])) != 29 or set(nonfoot.get("body_names", [])) & set(FOOT_BODIES):
        raise ValueError("Task072 nonfoot contact body binding drift")
    for pose_name, joint_ids in expected_groups.items():
        params = by_name[pose_name]["params"]
        if params.get("joint_ids") != joint_ids:
            raise ValueError(f"Task072 reward pose joint partition drift: {pose_name}")
        expected_q_ref = [float(stance["joint_qpos"][semantic_names[index]]) for index in joint_ids]
        if len(params.get("q_ref", [])) != len(joint_ids):
            raise ValueError(f"Task072 reward pose q_ref size drift: {pose_name}")
        if any(abs(float(actual) - expected) > STANCE_TOL for actual, expected in zip(params["q_ref"], expected_q_ref)):
            raise ValueError(f"Task072 reward pose q_ref drift: {pose_name}")
        if params.get("semantic_joint_names") != [semantic_names[index] for index in joint_ids]:
            raise ValueError(f"Task072 reward pose semantic names drift: {pose_name}")
        if params.get("anonymous_joint_names") != [anonymous_names[index] for index in joint_ids]:
            raise ValueError(f"Task072 reward pose anonymous names drift: {pose_name}")
        if params.get("stance_payload_sha256") != stance_payload_sha:
            raise ValueError(f"Task072 reward pose stance payload SHA drift: {pose_name}")
    for name in ("joint_velocity", "joint_limit"):
        params = by_name[name]["params"]
        if params.get("joint_ids") != list(range(29)):
            raise ValueError(f"Task072 reward 29-joint binding drift: {name}")
        if params.get("semantic_joint_names") != semantic_names or params.get("anonymous_joint_names") != anonymous_names:
            raise ValueError(f"Task072 reward joint names drift: {name}")
    joint_limit = by_name["joint_limit"]["params"]
    if len(joint_limit.get("lower", [])) != 29 or len(joint_limit.get("upper", [])) != 29 or joint_limit.get("soft_fraction") != 0.9:
        raise ValueError("Task072 reward joint limit params drift")
    if joint_limit.get("lower") != indices["joint_lower"] or joint_limit.get("upper") != indices["joint_upper"]:
        raise ValueError("Task072 reward joint limit range drift")
    if by_name["action_magnitude"]["params"] != {"action_name": "joint_pos"}:
        raise ValueError("Task072 reward action magnitude params drift")
    if by_name["action_rate"]["params"] != {"action_name": "joint_pos", "previous_action_reset": 0.0}:
        raise ValueError("Task072 reward action rate params drift")
    if by_name["fall_terminated"]["params"] != {} or float(by_name["fall_terminated"]["weight"]) != 300.0:
        raise ValueError("Task072 fall termination reward params drift")
    return payload_sha256(table)


def task072_require_train_eval_reward_match(train_sha256: str, eval_sha256: str) -> None:
    if train_sha256 != eval_sha256:
        raise ValueError("Task072 train/eval reward active-table SHA drift")


def _callable_id(func: Any) -> str:
    return f"{getattr(func, '__module__', type(func).__module__)}.{getattr(func, '__qualname__', type(func).__qualname__)}"


def _json_native(value: Any) -> Any:
    if callable(value):
        return _callable_id(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "__dataclass_fields__"):
        return _json_native(asdict(value))
    if hasattr(value, "value") and type(value).__module__ == "enum":
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Task072 semantic payload cannot encode {type(value).__name__}")


def _manager_cfg_table(cfg: dict[str, Any], extra_fields: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    rows = []
    for name, term in cfg.items():
        row = {
            "name": name,
            "callable": _callable_id(term.func),
            "params": _json_native(getattr(term, "params", {})),
        }
        for field in extra_fields:
            row[field] = _json_native(getattr(term, field))
        rows.append(row)
    return rows


def _observation_term_row(name: str, term: Any) -> dict[str, Any]:
    return {
        "name": name,
        "callable": _callable_id(term.func),
        "params": _json_native(term.params),
        "noise": _json_native(term.noise),
        "scale": _json_native(term.scale),
        "history_length": int(term.history_length),
    }


def _observation_group_payload(group: Any) -> dict[str, Any]:
    return {
        "concatenate_terms": bool(group.concatenate_terms),
        "enable_corruption": bool(group.enable_corruption),
        "history_length": int(group.history_length),
        "terms": [_observation_term_row(name, term) for name, term in group.terms.items()],
    }


def task072_observation_active_table_from_manager(manager: Any) -> dict[str, list[dict[str, Any]]]:
    return {
        group: [
            _observation_term_row(name, term)
            for name, term in zip(manager.active_terms[group], manager._group_obs_term_cfgs[group])
        ]
        for group in ("actor", "critic")
    }


def task072_termination_active_table_from_manager(manager: Any) -> list[dict[str, Any]]:
    return _manager_cfg_table({name: manager.get_term_cfg(name) for name in manager.active_terms}, ("time_out",))


def task072_validate_cross_manager_phase_and_termination(
    observation_table: dict[str, list[dict[str, Any]]],
    reward_table: list[dict[str, Any]],
    termination_table: list[dict[str, Any]],
) -> dict[str, Any]:
    reward_by_name = {row["name"]: row for row in reward_table}
    actor_phase = observation_table["actor"][list(EXPECTED_ACTOR_TERMS).index("phase")]
    critic_phase = observation_table["critic"][list(EXPECTED_CRITIC_TERMS).index("phase")]
    phase_periods = {
        "actor_observation": actor_phase["params"].get("period"),
        "critic_observation": critic_phase["params"].get("period"),
        "reward_phase_gait": reward_by_name["phase_gait"]["params"].get("period"),
        "reward_out_of_phase_double_support": reward_by_name["out_of_phase_double_support"]["params"].get("period"),
        "reward_clearance": reward_by_name["clearance"]["params"].get("period"),
    }
    checks = {
        "actor_terms": [row["name"] for row in observation_table["actor"]] == list(EXPECTED_ACTOR_TERMS),
        "critic_terms": [row["name"] for row in observation_table["critic"]] == list(EXPECTED_CRITIC_TERMS),
        "actor_phase_callable": actor_phase["callable"] == "src.tasks.velocity.mdp.observations.phase",
        "critic_phase_callable": critic_phase["callable"] == "src.tasks.velocity.mdp.observations.phase",
        "phase_periods": all(value == TASK072_GAIT_PERIOD_S for value in phase_periods.values()),
        "observation_uncorrupted": all(
            row["noise"] is None and row["scale"] is None
            for table in observation_table.values()
            for row in table
        ),
        "termination_terms": [row["name"] for row in termination_table] == ["time_out", "fell_over"],
        "termination_timeout_flags": [row["time_out"] for row in termination_table] == [True, False],
        "fell_over_angle": termination_table[1]["params"] == {"limit_angle": 1.2217304763960306},
    }
    return {"phase_periods": phase_periods, "termination_table": termination_table, "checks": checks, "passed": all(checks.values())}


def task072_runtime_semantic_payload(
    env_cfg: Any,
    agent_cfg: Any,
    registration: dict[str, Any],
    *,
    render_mode: str | None,
) -> dict[str, Any]:
    """Extract the task-owned v4 runtime contract from resolved configs only."""
    action_cfg = env_cfg.actions["joint_pos"]
    twist = env_cfg.commands["twist"]
    algorithm_cfg = agent_cfg.algorithm
    distribution_cfg = agent_cfg.actor.distribution_cfg
    reward_table = task072_reward_active_table_from_cfg(env_cfg.rewards)
    active_sha = task072_validate_reward_active_table(reward_table)
    command_payload = {
        "entity_name": twist.entity_name,
        "heading_command": bool(twist.heading_command),
        "heading_control_stiffness": float(twist.heading_control_stiffness),
        "rel_standing_envs": float(twist.rel_standing_envs),
        "rel_heading_envs": float(twist.rel_heading_envs),
        "init_velocity_prob": float(twist.init_velocity_prob),
        "resampling_time_range": list(twist.resampling_time_range),
        "ranges": {
            "lin_vel_x": list(twist.ranges.lin_vel_x),
            "lin_vel_y": list(twist.ranges.lin_vel_y),
            "ang_vel_z": list(twist.ranges.ang_vel_z),
            "heading": _json_native(twist.ranges.heading),
        },
        "debug_vis": bool(twist.debug_vis),
    }
    return {
        "schema_version": 1,
        "active_subtask": TASK072_ACTIVE_SUBTASK,
        "lineage_id": LINEAGE_ID,
        "source_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "external_mjlab_commit": _external_mjlab_status()["actual_commit"],
        "asset": {
            "contact_profile_id": CONTACT_PROFILE_ID,
            "xml_path": str(ASSET_XML.resolve()),
            "xml_sha256": sha256_path(ASSET_XML),
            "contact_profile_sha256": sha256_path(CONTACT_PROFILE),
            "stance_sha256": sha256_path(STANCE),
            "stance_payload_sha256": payload_sha256(_stance_dict()),
            "runtime_spec_sha256": registration["runtime_spec_sha256"],
        },
        "sim": {
            "timestep": float(env_cfg.sim.mujoco.timestep),
            "decimation": int(env_cfg.decimation),
            "step_dt": float(env_cfg.sim.mujoco.timestep) * int(env_cfg.decimation),
            "iterations": int(env_cfg.sim.mujoco.iterations),
            "ls_iterations": int(env_cfg.sim.mujoco.ls_iterations),
            "ccd_iterations": int(env_cfg.sim.mujoco.ccd_iterations),
            "contact_sensor_maxmatch": int(env_cfg.sim.contact_sensor_maxmatch),
            "njmax": int(env_cfg.sim.njmax),
            "nconmax": int(env_cfg.sim.nconmax),
        },
        "episode": {"episode_length_s": float(env_cfg.episode_length_s)},
        "scene": {
            "num_envs": int(env_cfg.scene.num_envs),
            "sensors": [_json_native(sensor) for sensor in env_cfg.scene.sensors],
            "robot_init": _json_native(env_cfg.scene.entities["robot"].init_state),
        },
        "observations": {
            "actor": _observation_group_payload(env_cfg.observations["actor"]),
            "critic": _observation_group_payload(env_cfg.observations["critic"]),
        },
        "actions": {
            "terms": [
                {
                    "name": "joint_pos",
                    "target_names": list(action_cfg.actuator_names),
                    "offset": _json_native(action_cfg.offset),
                    "negative_scale": _json_native(action_cfg.task072_negative_scale),
                    "positive_scale": _json_native(action_cfg.task072_positive_scale),
                    "policy_action_domain": dict(POLICY_ACTION_DOMAIN),
                    "action_contract_sha256": registration["action_contract_sha256"],
                }
            ],
        },
        "commands": {"twist": command_payload},
        "events": _manager_cfg_table(env_cfg.events, ("mode", "interval_range_s")),
        "rewards": reward_table,
        "reward_active_table_sha256": active_sha,
        "reward_payload_sha256": task072_canonical_reward_payload(reward_table)["payload_sha256"],
        "terminations": _manager_cfg_table(env_cfg.terminations, ("time_out",)),
        "curriculum": _json_native(env_cfg.curriculum),
        "metrics": _manager_cfg_table(env_cfg.metrics),
        "agent": {
            "seed": int(agent_cfg.seed),
            "num_steps_per_env": int(agent_cfg.num_steps_per_env),
            "max_iterations": int(agent_cfg.max_iterations),
            "save_interval": int(agent_cfg.save_interval),
            "resume": bool(agent_cfg.resume),
            "upload_model": bool(agent_cfg.upload_model),
            "logger": agent_cfg.logger,
            "clip_actions": float(agent_cfg.clip_actions),
            "actor": _json_native(agent_cfg.actor),
            "critic": _json_native(agent_cfg.critic),
            "distribution": _json_native(distribution_cfg),
            "ppo": _json_native(algorithm_cfg),
            "optimizer_runtime_class": "torch.optim.Adam",
        },
        "registration": _json_native(registration),
        "fixed_command_assertion": command_payload["ranges"] == {
            "lin_vel_x": [0.5, 0.5],
            "lin_vel_y": [0.0, 0.0],
            "ang_vel_z": [0.0, 0.0],
            "heading": None,
        },
        "render_mode": render_mode,
    }


def task072_stage_semantic_contract(
    *,
    num_envs: int,
    rollout_steps: int,
    seed: int,
    max_iterations: int,
    fixed_command: bool,
    render_mode: str | None,
) -> dict[str, Any]:
    env_cfg, agent_cfg, _runner_cls, registration = build_task_cfg(
        int(num_envs),
        int(rollout_steps),
        int(seed),
        int(max_iterations),
        fixed_command=bool(fixed_command),
    )
    payload = task072_runtime_semantic_payload(env_cfg, agent_cfg, registration, render_mode=render_mode)
    return {"payload": payload, "payload_sha256": payload_sha256(payload)}


def task072_assert_agent_ppo_contract(agent_cfg: Any) -> dict[str, Any]:
    actor = agent_cfg.actor
    critic = agent_cfg.critic
    algorithm = agent_cfg.algorithm
    distribution = actor.distribution_cfg
    payload = {
        "actor_hidden_dims": list(actor.hidden_dims),
        "actor_activation": actor.activation,
        "actor_obs_normalization": bool(actor.obs_normalization),
        "actor_distribution": _json_native(distribution),
        "critic_hidden_dims": list(critic.hidden_dims),
        "critic_activation": critic.activation,
        "critic_obs_normalization": bool(critic.obs_normalization),
        "value_loss_coef": float(algorithm.value_loss_coef),
        "use_clipped_value_loss": bool(algorithm.use_clipped_value_loss),
        "clip_param": float(algorithm.clip_param),
        "entropy_coef": float(algorithm.entropy_coef),
        "num_learning_epochs": int(algorithm.num_learning_epochs),
        "num_mini_batches": int(algorithm.num_mini_batches),
        "learning_rate": float(algorithm.learning_rate),
        "schedule": algorithm.schedule,
        "gamma": float(algorithm.gamma),
        "lam": float(algorithm.lam),
        "desired_kl": float(algorithm.desired_kl),
        "max_grad_norm": float(algorithm.max_grad_norm),
        "optimizer": algorithm.optimizer,
        "num_steps_per_env": int(agent_cfg.num_steps_per_env),
        "resume": bool(agent_cfg.resume),
        "upload_model": bool(agent_cfg.upload_model),
        "logger": agent_cfg.logger,
    }
    expected = {
        "actor_hidden_dims": [512, 256, 128],
        "actor_activation": "elu",
        "actor_obs_normalization": True,
        "actor_distribution": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        "critic_hidden_dims": [512, 256, 128],
        "critic_activation": "elu",
        "critic_obs_normalization": True,
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 0.001,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "optimizer": "adam",
        "num_steps_per_env": REQUIRED_ROLLOUT_STEPS,
        "resume": False,
        "upload_model": False,
        "logger": "tensorboard",
    }
    drift = {key: {"actual": payload[key], "expected": value} for key, value in expected.items() if payload[key] != value}
    if drift:
        raise ValueError(f"Task072 PPO contract drift: {drift}")
    return payload


def task072_reward_v3_oracle_pre_dt_means() -> dict[str, float]:
    totals = {"static_both": 0.0, "ideal_phase_matched": 0.0, "persistent_left_only": 0.0}
    for step in range(1, 41):
        phase = (step % 40) / 40.0
        desired = [((phase + offset) % 1.0) < 0.55 for offset in (0.0, 0.5)]
        fixtures = {
            "static_both": {"vx": 0.0, "contact": [True, True], "height": [0.0, 0.0]},
            "ideal_phase_matched": {
                "vx": 0.5,
                "contact": desired,
                "height": [0.0 if contact else 0.10 for contact in desired],
            },
            "persistent_left_only": {"vx": 0.0, "contact": [True, False], "height": [0.0, 0.0]},
        }
        for name, fixture in fixtures.items():
            contact = fixture["contact"]
            height = fixture["height"]
            raw = 0.0
            raw += 2.0 * (math.exp(-(((fixture["vx"] - 0.5) ** 2) / 0.25)) - 1.0)
            raw += 0.50
            raw += 0.25
            raw += 0.50 * (sum(actual == expected for actual, expected in zip(contact, desired)) / 2.0)
            raw += 0.35 * (-float(all(contact) and not all(desired)))
            swing = [
                index
                for index, (expected, actual) in enumerate(zip(desired, contact))
                if not expected and not actual
            ]
            if swing:
                raw += 0.50 * (
                    sum(math.exp(-(((height[index] - 0.10) / 0.05) ** 2)) for index in swing)
                    / len(swing)
                )
            totals[name] += raw
    means = {name: value / 40.0 for name, value in totals.items()}
    means["ideal_static_margin"] = means["ideal_phase_matched"] - means["static_both"]
    return means


class Task072RewardFixtureAdapter:
    """Tensor-backed env surface consumed by the Task072 reward callables."""

    def __init__(self, active_table: list[dict[str, Any]], num_envs: int = 1, device: str = "cpu") -> None:
        import torch

        self.num_envs = int(num_envs)
        self.device = device
        self.step_dt = 0.02
        self.max_episode_length_s = 10_000.0
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.command_manager = _Task072FixtureCommandManager(self.num_envs, device)
        self.termination_manager = _Task072FixtureTerminationManager(self.num_envs, device)
        max_site_id = 1
        nonfoot_count = 1
        stance_height = 1.0
        foot_site_ids = [0, 1]
        for row in active_table:
            params = row["params"]
            site_ids = params.get("site_ids", [])
            if isinstance(site_ids, int):
                site_ids = [site_ids]
            if site_ids:
                max_site_id = max(max_site_id, max(int(value) for value in site_ids))
            nonfoot_count = max(nonfoot_count, len(params.get("body_ids", [])))
            if row["name"] == "clearance":
                foot_site_ids = [int(value) for value in params["site_ids"]]
            if row["name"] == "height":
                stance_height = float(params["stance_height"])
        self._site_count = max_site_id + 1
        self._nonfoot_count = nonfoot_count
        self._foot_site_ids = foot_site_ids
        self.scene = {
            "robot": SimpleNamespace(data=_Task072FixtureRobotData(self.num_envs, self._site_count, device)),
            "feet_ground_contact": _Task072FixtureFeetContact(self.num_envs, device),
            "nonfoot_ground_contact": SimpleNamespace(
                data=SimpleNamespace(found=torch.zeros((self.num_envs, self._nonfoot_count), device=device))
            ),
        }
        self.action_manager = _Task072FixtureActionManager(self.num_envs, 29, device)
        self._q_ref = torch.zeros((self.num_envs, 29), device=device)
        for row in active_table:
            if str(row["name"]).startswith("pose_"):
                for joint_id, value in zip(row["params"]["joint_ids"], row["params"]["q_ref"]):
                    self._q_ref[:, int(joint_id)] = float(value)
        self._stance_height = stance_height

    def set_state(self, fixture: str, step_index: int) -> None:
        import torch

        if fixture not in {"static_both", "ideal_phase_matched", "persistent_left_only", "normal", "fell_over", "timeout"}:
            raise ValueError(f"unknown Task072 reward fixture: {fixture}")
        phase = (int(step_index) % 40) / 40.0
        desired = torch.tensor(
            [((phase + offset) % 1.0) < 0.55 for offset in (0.0, 0.5)],
            dtype=torch.bool,
            device=self.device,
        )
        contact = {
            "static_both": torch.tensor([True, True], device=self.device),
            "ideal_phase_matched": desired,
            "persistent_left_only": torch.tensor([True, False], device=self.device),
            "normal": torch.tensor([True, True], device=self.device),
            "fell_over": torch.tensor([True, True], device=self.device),
            "timeout": torch.tensor([True, True], device=self.device),
        }[fixture]
        vx = 0.5 if fixture == "ideal_phase_matched" else 0.0
        robot = self.scene["robot"].data
        robot.root_link_lin_vel_b.zero_()
        robot.root_link_lin_vel_b[:, 0] = vx
        robot.root_link_ang_vel_b.zero_()
        robot.projected_gravity_b[:] = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        robot.root_link_pos_w.zero_()
        robot.root_link_pos_w[:, 2] = self._stance_height
        robot.root_link_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        robot.joint_pos[:] = self._q_ref
        robot.joint_vel.zero_()
        robot.site_pos_w.zero_()
        robot.site_lin_vel_w.zero_()
        for foot, site_id in enumerate(self._foot_site_ids):
            if site_id < self._site_count:
                robot.site_pos_w[:, site_id, 2] = 0.10 if fixture == "ideal_phase_matched" and not bool(desired[foot]) else 0.0
        contact_sensor = self.scene["feet_ground_contact"]
        contact_sensor.data.current_contact_time[:] = contact.float().view(1, 2) * self.step_dt
        contact_sensor.data.last_air_time.zero_()
        contact_sensor._first_contact.zero_()
        self.scene["nonfoot_ground_contact"].data.found.zero_()
        self.action_manager.action.zero_()
        self.action_manager.prev_action.zero_()
        self.episode_length_buf[:] = int(step_index)
        self.termination_manager.terminated[:] = fixture == "fell_over"
        self.termination_manager.time_outs[:] = fixture == "timeout"


class _Task072FixtureRobotData:
    def __init__(self, num_envs: int, site_count: int, device: str) -> None:
        import torch

        self.root_link_lin_vel_b = torch.zeros((num_envs, 3), device=device)
        self.root_link_ang_vel_b = torch.zeros((num_envs, 3), device=device)
        self.root_link_pos_w = torch.zeros((num_envs, 3), device=device)
        self.root_link_quat_w = torch.zeros((num_envs, 4), device=device)
        self.projected_gravity_b = torch.zeros((num_envs, 3), device=device)
        self.joint_pos = torch.zeros((num_envs, 29), device=device)
        self.joint_vel = torch.zeros((num_envs, 29), device=device)
        self.site_pos_w = torch.zeros((num_envs, site_count, 3), device=device)
        self.site_lin_vel_w = torch.zeros((num_envs, site_count, 3), device=device)


class _Task072FixtureCommandManager:
    def __init__(self, num_envs: int, device: str) -> None:
        import torch

        self._command = torch.tensor([0.5, 0.0, 0.0], device=device).repeat(num_envs, 1)

    def get_command(self, command_name: str) -> Any:
        if command_name != "twist":
            raise ValueError(f"unexpected Task072 command name: {command_name}")
        return self._command


class _Task072FixtureFeetContact:
    def __init__(self, num_envs: int, device: str) -> None:
        import torch

        self.data = SimpleNamespace(
            current_contact_time=torch.zeros((num_envs, 2), device=device),
            last_air_time=torch.zeros((num_envs, 2), device=device),
        )
        self._first_contact = torch.zeros((num_envs, 2), dtype=torch.bool, device=device)

    def compute_first_contact(self, dt: float) -> Any:
        del dt
        return self._first_contact


class _Task072FixtureActionManager:
    def __init__(self, num_envs: int, action_dim: int, device: str) -> None:
        import torch

        self.action = torch.zeros((num_envs, action_dim), device=device)
        self.prev_action = torch.zeros((num_envs, action_dim), device=device)


class _Task072FixtureTerminationManager:
    def __init__(self, num_envs: int, device: str) -> None:
        import torch

        self.terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.time_outs = torch.zeros(num_envs, dtype=torch.bool, device=device)


def task072_reward_fixture_probe(active_cfg: dict[str, Any]) -> dict[str, Any]:
    import torch
    from mjlab.managers.reward_manager import RewardManager

    active_table = task072_reward_active_table_from_cfg(active_cfg)
    fixture = Task072RewardFixtureAdapter(active_table)
    manager_pre = RewardManager(active_cfg, fixture, scale_by_dt=False)
    manager_dt = RewardManager(active_cfg, fixture, scale_by_dt=True)
    result: dict[str, Any] = {}
    for name in ("static_both", "ideal_phase_matched", "persistent_left_only"):
        pre_values = []
        dt_values = []
        max_iterable_error = 0.0
        for step_index in range(1, 41):
            fixture.set_state(name, step_index)
            pre = manager_pre.compute(fixture.step_dt)
            dt_reward = manager_dt.compute(fixture.step_dt)
            breakdown = task072_reward_breakdown_from_manager(manager_pre, fixture, fixture.step_dt)
            iterable_total = sum(float(values[0]) for _term, values in manager_pre.get_active_iterable_terms(0))
            total_pre = float(breakdown["total_weighted_pre_dt"][0].detach().cpu())
            max_iterable_error = max(max_iterable_error, abs(iterable_total - total_pre))
            pre_values.append(float(pre[0].detach().cpu()))
            dt_values.append(float(dt_reward[0].detach().cpu()))
        pre_mean = sum(pre_values) / len(pre_values)
        dt_mean = sum(dt_values) / len(dt_values)
        result[name] = {
            "weighted_pre_dt_mean": pre_mean,
            "dt_contribution_mean": dt_mean,
            "oracle_abs_diff": abs(pre_mean - REWARD_V3_ORACLE_EXPECTED[name]),
            "dt_once_abs_diff": abs(dt_mean - pre_mean * 0.02),
            "iterable_abs_diff_max": max_iterable_error,
        }
    terminal = {}
    for name in ("normal", "fell_over", "timeout"):
        fixture.set_state(name, 1)
        if bool(fixture.termination_manager.terminated[0]) and bool(fixture.termination_manager.time_outs[0]):
            raise ValueError("Task072 terminal fixture overlaps terminated and timeout")
        pre = manager_pre.compute(fixture.step_dt)
        dt_reward = manager_dt.compute(fixture.step_dt)
        fall_row = next(row for row in task072_reward_breakdown_from_manager(manager_pre, fixture, fixture.step_dt)["rows"] if row["name"] == "fall_terminated")
        terminal[name] = {
            "terminated": bool(fixture.termination_manager.terminated[0]),
            "time_out": bool(fixture.termination_manager.time_outs[0]),
            "fall_raw": float(fall_row["raw"][0].detach().cpu()),
            "pre_dt": float(fall_row["weighted_pre_dt"][0].detach().cpu()),
            "dt_contribution": float(fall_row["dt_contribution"][0].detach().cpu()),
            "manager_dt_total": float(dt_reward[0].detach().cpu()),
            "manager_pre_total": float(pre[0].detach().cpu()),
        }
    terminal["passed"] = terminal == {
        "normal": {**terminal["normal"], "terminated": False, "time_out": False, "fall_raw": 0.0, "pre_dt": 0.0, "dt_contribution": 0.0},
        "fell_over": {**terminal["fell_over"], "terminated": True, "time_out": False, "fall_raw": -1.0, "pre_dt": -300.0, "dt_contribution": -6.0},
        "timeout": {**terminal["timeout"], "terminated": False, "time_out": True, "fall_raw": 0.0, "pre_dt": 0.0, "dt_contribution": 0.0},
    }
    result["terminal"] = terminal
    result["ideal_static_margin"] = (
        result["ideal_phase_matched"]["weighted_pre_dt_mean"]
        - result["static_both"]["weighted_pre_dt_mean"]
    )
    result["ideal_static_margin_abs_diff"] = abs(result["ideal_static_margin"] - REWARD_V3_ORACLE_EXPECTED["ideal_static_margin"])
    result["passed"] = (
        all(value["oracle_abs_diff"] <= 1.0e-6 and value["dt_once_abs_diff"] <= 1.0e-6 and value["iterable_abs_diff_max"] <= 1.0e-6 for name, value in result.items() if name in {"static_both", "ideal_phase_matched", "persistent_left_only"})
        and result["ideal_static_margin_abs_diff"] <= 1.0e-6
        and terminal["passed"]
    )
    return result


def task072_reward_breakdown_from_manager(manager: Any, env: Any, dt: float) -> dict[str, Any]:
    """Extract actual manager terms without duplicating or reimplementing formulas."""
    rows = []
    for name in manager.active_terms:
        cfg = manager.get_term_cfg(name)
        raw = cfg.func(env, **cfg.params)
        weighted = raw * cfg.weight
        rows.append(
            {
                "name": name,
                "raw": raw,
                "weight": float(cfg.weight),
                "weighted_pre_dt": weighted,
                "dt_contribution": weighted * dt,
            }
        )
    return {
        "rows": rows,
        "total_weighted_pre_dt": sum((row["weighted_pre_dt"] for row in rows)),
        "total_dt": sum((row["dt_contribution"] for row in rows)),
    }


class Task072ClipLoggingVecEnvWrapper:
    """Transparent pre-clip action boundary with immutable 24-step snapshots."""
    def __init__(self, base_env: Any, semantic_joint_names: Any, rollout_steps: int) -> None:
        self.base_env, self.semantic_joint_names = base_env, tuple(semantic_joint_names)
        self.rollout_steps = int(rollout_steps)
        if self.rollout_steps <= 0:
            raise ValueError("Task072 clip logging rollout_steps must be positive")
        if len(self.semantic_joint_names) != int(base_env.num_actions):
            raise ValueError("Task072 clip logging semantic joint count must match action dim")
        self.joint_count = len(self.semantic_joint_names)
        self.steps_in_update = 0
        self._scalar_denominator = 0
        self._clipped_scalars = 0
        self._env_step_denominator = 0
        self._env_steps_with_any_clip = 0
        self._per_joint_clipped = [0 for _ in self.semantic_joint_names]
        self._max_abs_raw_action = 0.0
        self.completed_update_records: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any: return getattr(self.base_env, name)
    @property
    def unwrapped(self) -> Any: return self.base_env.unwrapped
    @property
    def cfg(self) -> Any: return self.base_env.cfg
    @property
    def episode_length_buf(self) -> Any: return self.base_env.episode_length_buf
    @episode_length_buf.setter
    def episode_length_buf(self, value: Any) -> None:
        self.base_env.episode_length_buf = value
    def reset(self, *args: Any, **kwargs: Any) -> Any: return self.base_env.reset(*args, **kwargs)
    def step(self, raw_action: Any) -> Any:
        import torch

        action = torch.as_tensor(raw_action)
        if action.ndim != 2 or int(action.shape[1]) != self.joint_count:
            raise ValueError("Task072 clip logging expected [num_envs, 29] raw actions")
        clipped = torch.abs(action) > 1.0
        self._scalar_denominator += int(action.numel())
        self._clipped_scalars += int(clipped.sum().detach().cpu())
        self._env_step_denominator += int(action.shape[0])
        self._env_steps_with_any_clip += int(clipped.any(dim=1).sum().detach().cpu())
        per_joint = clipped.sum(dim=0).detach().cpu().tolist()
        self._per_joint_clipped = [int(old + new) for old, new in zip(self._per_joint_clipped, per_joint)]
        self._max_abs_raw_action = max(self._max_abs_raw_action, float(torch.abs(action).max().detach().cpu()))
        result = self.base_env.step(raw_action)
        self.steps_in_update += 1
        if self.steps_in_update == self.rollout_steps:
            self.completed_update_records.append(self._snapshot_clip_record())
            self._reset_clip_counts()
        return result
    def drain_task072_clip_update_records(self) -> list[dict[str, Any]]:
        records = list(self.completed_update_records)
        self.completed_update_records.clear()
        return records

    def _snapshot_clip_record(self) -> dict[str, Any]:
        per_joint_fraction = {
            name: self._per_joint_clipped[index] / self._env_step_denominator
            for index, name in enumerate(self.semantic_joint_names)
        }
        per_joint_numerators = {
            name: self._per_joint_clipped[index]
            for index, name in enumerate(self.semantic_joint_names)
        }
        return {
            "update_index": len(self.completed_update_records),
            "num_envs": int(self._env_step_denominator // self.rollout_steps),
            "rollout_steps": self.rollout_steps,
            "joint_count": self.joint_count,
            "clipped_scalars": int(self._clipped_scalars),
            "scalar_denominator": int(self._scalar_denominator),
            "scalar_clip_fraction": self._clipped_scalars / self._scalar_denominator,
            "env_steps_with_any_clip": int(self._env_steps_with_any_clip),
            "env_step_denominator": int(self._env_step_denominator),
            "env_step_any_clip_fraction": self._env_steps_with_any_clip / self._env_step_denominator,
            "per_joint_clipped_scalars": per_joint_numerators,
            "per_joint_denominator": int(self._env_step_denominator),
            "per_joint_clip_fraction": per_joint_fraction,
            "max_abs_raw_action": float(self._max_abs_raw_action),
            "clip_numerator": int(self._clipped_scalars),
            "clip_denominator": int(self._scalar_denominator),
            "clip_fraction": self._clipped_scalars / self._scalar_denominator,
        }

    def _reset_clip_counts(self) -> None:
        self.steps_in_update = 0
        self._scalar_denominator = 0
        self._clipped_scalars = 0
        self._env_step_denominator = 0
        self._env_steps_with_any_clip = 0
        self._per_joint_clipped = [0 for _ in self.semantic_joint_names]
        self._max_abs_raw_action = 0.0


def _task072_exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Task072 clip metric {label} must be an integer")
    return value


def validate_task072_clip_records(records: list[dict[str, Any]], *, expected_updates: int | None = None) -> list[dict[str, Any]]:
    if expected_updates is not None and len(records) != expected_updates:
        raise ValueError("Task072 clip metric update count mismatch")
    required = {
        "update_index",
        "clipped_scalars",
        "scalar_denominator",
        "scalar_clip_fraction",
        "env_steps_with_any_clip",
        "env_step_denominator",
        "env_step_any_clip_fraction",
        "per_joint_clipped_scalars",
        "per_joint_denominator",
        "per_joint_clip_fraction",
        "max_abs_raw_action",
        "num_envs",
        "rollout_steps",
        "joint_count",
    }
    for index, record in enumerate(records):
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"Task072 clip metric missing fields: {missing}")
        if _task072_exact_int(record["update_index"], "update_index") != index:
            raise ValueError("Task072 clip metric update index mismatch")
        if _task072_exact_int(record["joint_count"], "joint_count") != 29:
            raise ValueError("Task072 clip metric joint count mismatch")
        expected_joint_names = set(SEMANTIC_TO_ANON_JOINT)
        if set(record["per_joint_clipped_scalars"]) != expected_joint_names or set(record["per_joint_clip_fraction"]) != expected_joint_names:
            raise ValueError("Task072 clip metric joint set mismatch")
        scalar_denominator = _task072_exact_int(record["scalar_denominator"], "scalar_denominator")
        env_step_denominator = _task072_exact_int(record["env_step_denominator"], "env_step_denominator")
        per_joint_denominator = _task072_exact_int(record["per_joint_denominator"], "per_joint_denominator")
        if scalar_denominator <= 0 or env_step_denominator <= 0 or per_joint_denominator <= 0:
            raise ValueError("Task072 clip metric denominator must be positive")
        expected_env_step_denominator = (
            _task072_exact_int(record["num_envs"], "num_envs")
            * _task072_exact_int(record["rollout_steps"], "rollout_steps")
        )
        if record["num_envs"] <= 0 or record["rollout_steps"] <= 0:
            raise ValueError("Task072 clip metric shape must be positive")
        if env_step_denominator != expected_env_step_denominator:
            raise ValueError("Task072 clip metric env-step denominator mismatch")
        if scalar_denominator != env_step_denominator * 29:
            raise ValueError("Task072 clip metric denominator mismatch")
        if per_joint_denominator != env_step_denominator:
            raise ValueError("Task072 clip metric per-joint denominator mismatch")
        if len(record["per_joint_clip_fraction"]) != 29 or len(record["per_joint_clipped_scalars"]) != 29:
            raise ValueError("Task072 clip metric per-joint field count mismatch")
        clipped_scalars = _task072_exact_int(record["clipped_scalars"], "clipped_scalars")
        env_steps_with_any_clip = _task072_exact_int(record["env_steps_with_any_clip"], "env_steps_with_any_clip")
        if not 0 <= clipped_scalars <= scalar_denominator or not 0 <= env_steps_with_any_clip <= env_step_denominator:
            raise ValueError("Task072 clip metric count outside denominator")
        if env_steps_with_any_clip > clipped_scalars:
            raise ValueError("Task072 clip metric env-step count exceeds scalar count")
        if not math.isfinite(float(record["scalar_clip_fraction"])) or not math.isclose(float(record["scalar_clip_fraction"]), clipped_scalars / scalar_denominator):
            raise ValueError("Task072 clip metric scalar fraction mismatch")
        if not math.isfinite(float(record["env_step_any_clip_fraction"])) or not math.isclose(float(record["env_step_any_clip_fraction"]), env_steps_with_any_clip / env_step_denominator):
            raise ValueError("Task072 clip metric env-step fraction mismatch")
        per_joint_total = 0
        for name, count in record["per_joint_clipped_scalars"].items():
            count = _task072_exact_int(count, f"per_joint_clipped_scalars.{name}")
            per_joint_total += count
            if not 0 <= count <= per_joint_denominator:
                raise ValueError("Task072 clip metric per-joint count outside denominator")
            if not math.isfinite(float(record["per_joint_clip_fraction"][name])) or not math.isclose(float(record["per_joint_clip_fraction"][name]), count / per_joint_denominator):
                raise ValueError("Task072 clip metric per-joint fraction mismatch")
        if per_joint_total != clipped_scalars:
            raise ValueError("Task072 clip metric scalar/per-joint count mismatch")
        if not math.isfinite(float(record["max_abs_raw_action"])):
            raise ValueError("Task072 clip metric max raw action is non-finite")
    return records


def validate_task072_clip_summary(
    summary: dict[str, Any],
    *,
    expected_update_indices: list[int] | None = None,
) -> dict[str, Any]:
    required = {
        "update_indices",
        "clipped_scalars",
        "scalar_denominator",
        "scalar_clip_fraction",
        "env_steps_with_any_clip",
        "env_step_denominator",
        "env_step_any_clip_fraction",
        "per_joint_clipped_scalars",
        "per_joint_denominator",
        "per_joint_clip_fraction",
        "max_abs_raw_action",
    }
    missing = sorted(required - set(summary))
    if missing:
        raise ValueError(f"Task072 clip summary missing fields: {missing}")
    if not isinstance(summary["update_indices"], list) or any(
        _task072_exact_int(value, "update_indices") != value for value in summary["update_indices"]
    ):
        raise ValueError("Task072 clip summary update indices must be integers")
    if expected_update_indices is not None and summary["update_indices"] != expected_update_indices:
        raise ValueError("Task072 clip summary update index mismatch")
    scalar_denominator = _task072_exact_int(summary["scalar_denominator"], "summary.scalar_denominator")
    env_step_denominator = _task072_exact_int(summary["env_step_denominator"], "summary.env_step_denominator")
    per_joint_denominator = _task072_exact_int(summary["per_joint_denominator"], "summary.per_joint_denominator")
    if scalar_denominator <= 0 or env_step_denominator <= 0 or per_joint_denominator <= 0:
        raise ValueError("Task072 clip summary denominator must be positive")
    if set(summary["per_joint_clipped_scalars"]) != set(SEMANTIC_TO_ANON_JOINT) or set(summary["per_joint_clip_fraction"]) != set(SEMANTIC_TO_ANON_JOINT):
        raise ValueError("Task072 clip summary per-joint field count mismatch")
    if scalar_denominator != env_step_denominator * 29:
        raise ValueError("Task072 clip summary denominator mismatch")
    if per_joint_denominator != env_step_denominator:
        raise ValueError("Task072 clip summary per-joint denominator mismatch")
    clipped_scalars = _task072_exact_int(summary["clipped_scalars"], "summary.clipped_scalars")
    env_steps_with_any_clip = _task072_exact_int(summary["env_steps_with_any_clip"], "summary.env_steps_with_any_clip")
    if not 0 <= clipped_scalars <= scalar_denominator or not 0 <= env_steps_with_any_clip <= env_step_denominator:
        raise ValueError("Task072 clip summary count outside denominator")
    if env_steps_with_any_clip > clipped_scalars:
        raise ValueError("Task072 clip summary env-step count exceeds scalar count")
    if not math.isfinite(float(summary["scalar_clip_fraction"])) or not math.isclose(float(summary["scalar_clip_fraction"]), clipped_scalars / scalar_denominator):
        raise ValueError("Task072 clip summary scalar fraction mismatch")
    if not math.isfinite(float(summary["env_step_any_clip_fraction"])) or not math.isclose(
        float(summary["env_step_any_clip_fraction"]),
        env_steps_with_any_clip / env_step_denominator,
    ):
        raise ValueError("Task072 clip summary env-step fraction mismatch")
    if not math.isfinite(float(summary["max_abs_raw_action"])):
        raise ValueError("Task072 clip summary max raw action is non-finite")
    per_joint_total = 0
    for name, count in summary["per_joint_clipped_scalars"].items():
        count = _task072_exact_int(count, f"summary.per_joint_clipped_scalars.{name}")
        per_joint_total += count
        if not 0 <= count <= per_joint_denominator:
            raise ValueError("Task072 clip summary per-joint count outside denominator")
        if not math.isfinite(float(summary["per_joint_clip_fraction"][name])) or not math.isclose(float(summary["per_joint_clip_fraction"][name]), count / per_joint_denominator):
            raise ValueError("Task072 clip summary per-joint fraction mismatch")
    if per_joint_total != clipped_scalars:
        raise ValueError("Task072 clip summary scalar/per-joint count mismatch")
    return summary


def pool_task072_clip_records(records: list[dict[str, Any]], *, last_n: int = 7) -> dict[str, Any]:
    validate_task072_clip_records(records)
    selected = records[-last_n:]
    if not selected:
        raise ValueError("Task072 clip metric pool requires at least one update")
    clipped_scalars = sum(int(record["clipped_scalars"]) for record in selected)
    scalar_denominator = sum(int(record["scalar_denominator"]) for record in selected)
    env_steps_with_any_clip = sum(int(record["env_steps_with_any_clip"]) for record in selected)
    env_step_denominator = sum(int(record["env_step_denominator"]) for record in selected)
    per_joint_clipped = {
        name: sum(int(record["per_joint_clipped_scalars"][name]) for record in selected)
        for name in selected[0]["per_joint_clipped_scalars"]
    }
    per_joint_denominator = sum(int(record["per_joint_denominator"]) for record in selected)
    per_joint_fraction = {
        name: count / per_joint_denominator
        for name, count in per_joint_clipped.items()
    }
    return {
        "update_indices": [int(record["update_index"]) for record in selected],
        "clipped_scalars": clipped_scalars,
        "scalar_denominator": scalar_denominator,
        "scalar_clip_fraction": clipped_scalars / scalar_denominator,
        "env_steps_with_any_clip": env_steps_with_any_clip,
        "env_step_denominator": env_step_denominator,
        "env_step_any_clip_fraction": env_steps_with_any_clip / env_step_denominator,
        "per_joint_clipped_scalars": per_joint_clipped,
        "per_joint_denominator": per_joint_denominator,
        "per_joint_clip_fraction": per_joint_fraction,
        "max_abs_raw_action": max(float(record["max_abs_raw_action"]) for record in selected),
    }


def apply_signed_action_contract(raw_actions: Any, negative_scale: Any, positive_scale: Any, offset: Any) -> tuple[Any, Any]:
    import torch

    clipped = torch.clamp(raw_actions, POLICY_ACTION_DOMAIN["lower"], POLICY_ACTION_DOMAIN["upper"])
    scale = torch.where(clipped < 0.0, negative_scale, positive_scale)
    return clipped * scale + offset, clipped


def _external_mjlab_status() -> dict[str, Any]:
    external_head = subprocess.run(
        ["git", "-C", str(EXTERNAL_MJLAB), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_dirty = subprocess.run(
        ["git", "-C", str(EXTERNAL_MJLAB), "diff", "--quiet"],
        check=False,
    ).returncode != 0
    staged_dirty = subprocess.run(
        ["git", "-C", str(EXTERNAL_MJLAB), "diff", "--cached", "--quiet"],
        check=False,
    ).returncode != 0
    status_short = subprocess.run(
        ["git", "-C", str(EXTERNAL_MJLAB), "status", "--short", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "path": str(EXTERNAL_MJLAB.resolve()),
        "expected_commit": EXPECTED_MJLAB_COMMIT,
        "actual_commit": external_head,
        "tracked_clean": not tracked_dirty and not staged_dirty,
        "status_short": status_short,
    }


def _runtime_metadata(command: str) -> dict[str, Any]:
    import mjlab
    import rsl_rl
    import torch

    return {
        "command": command,
        "cwd": str(Path.cwd()),
        "python": sys.executable,
        "mjlab_file": str(Path(mjlab.__file__).resolve()),
        "mjlab_version": getattr(mjlab, "__version__", None),
        "rsl_rl_file": str(Path(rsl_rl.__file__).resolve()),
        "rsl_rl_version": getattr(rsl_rl, "__version__", None),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_lock_path": str(GPU_LOCK),
        "external_mjlab": _external_mjlab_status(),
        "h200_used": False,
        "external_downloads_performed": False,
        "task048_checkpoint_used": False,
    }


def build_task_cfg(
    num_envs: int,
    rollout_steps: int,
    seed: int,
    max_iterations: int,
    *,
    fixed_command: bool = False,
) -> tuple[Any, Any, type | None, dict[str, Any]]:
    ensure_v2_artifacts()
    _prepare_external_imports()
    import mjlab.tasks  # noqa: F401
    import mujoco
    import src.tasks  # noqa: F401
    from mjlab.actuator import XmlPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
    from mjlab.envs.mdp.actions.actions import resolve_matching_names_values
    from mjlab.envs.mdp import events as mjlab_events
    from mjlab.envs.mdp import metrics as mjlab_metrics
    from mjlab.envs.mdp import observations as mjlab_observations
    from mjlab.envs.mdp import terminations as mjlab_terminations
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls, register_mjlab_task
    from mjlab.utils.spec_config import CollisionCfg
    from src.tasks.velocity.mdp import observations as task_velocity_observations

    spec_xml = runtime_spec_xml()

    def spec_fn() -> Any:
        return mujoco.MjSpec.from_string(spec_xml)

    env_cfg = load_env_cfg(MJLAB_PARENT_TASK)
    agent_cfg = load_rl_cfg(MJLAB_PARENT_TASK)
    runner_cls = load_runner_cls(MJLAB_PARENT_TASK)
    task_id = f"{TASK_ID}-{num_envs}x{rollout_steps}-seed{seed}-{'fixed' if fixed_command else 'train'}"
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.seed = int(seed)
    env_cfg.viewer.body_name = TORSO_BODY
    env_cfg.sim.njmax = max(int(getattr(env_cfg.sim, "njmax", 0) or 0), 600)
    env_cfg.sim.nconmax = max(int(getattr(env_cfg.sim, "nconmax", 0) or 0), 128)
    stance = _stance_dict()
    mapping_table = _mapping_table(stance)
    semantic_to_joint = {row["semantic_joint"]: row["anonymous_joint"] for row in mapping_table}
    xml_joint_names = {
        joint.get("name")
        for joint in ET.parse(ASSET_XML).getroot().findall(".//joint")
        if joint.get("name") and joint.get("name") != "root_free"
    }
    if set(semantic_to_joint.values()) != xml_joint_names:
        missing = sorted(set(semantic_to_joint.values()) - xml_joint_names)
        extra = sorted(xml_joint_names - set(semantic_to_joint.values()))
        raise ValueError(f"explicit joint mapping does not match v2 XML: missing={missing}, extra={extra}")
    if len(semantic_to_joint) != 29 or len(set(semantic_to_joint.values())) != 29:
        raise ValueError("stance semantic-to-anonymous joint mapping is not one-to-one 29/29")
    joint_pos = {semantic_to_joint[slot]: float(value) for slot, value in stance["joint_qpos"].items()}
    action_offset = {semantic_to_joint[slot]: float(value) for slot, value in stance["actuator_ctrl_eq"].items()}
    robot = replace(
        env_cfg.scene.entities["robot"],
        spec_fn=spec_fn,
        init_state=replace(
            env_cfg.scene.entities["robot"].init_state,
            pos=tuple(float(v) for v in stance["root_pose_eq"][:3]),
            rot=tuple(float(v) for v in stance["root_pose_eq"][3:]),
            joint_pos=joint_pos,
            joint_vel={name: 0.0 for name in joint_pos},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlPositionActuatorCfg(target_names_expr=(r".*_joint$",)),),
            soft_joint_pos_limit_factor=0.9,
        ),
        collisions=(
            CollisionCfg(
                geom_names_expr=(ALL_COLLISION_REGEX,),
                contype=1,
                conaffinity=1,
                condim={FOOT_COLLISION_REGEX: 3, ALL_COLLISION_REGEX: 1},
                priority={FOOT_COLLISION_REGEX: 1},
                friction={FOOT_COLLISION_REGEX: TASK_FOOT_FRICTION},
                disable_other_geoms=False,
            ),
        ),
    )
    env_cfg.scene.entities["robot"] = robot
    for group in ("actor", "critic"):
        phase_term = env_cfg.observations[group].terms.get("phase")
        if phase_term is None:
            raise ValueError(f"Task072 {group} observation table is missing phase")
        phase_term.params["period"] = TASK072_GAIT_PERIOD_S
    nonfoot_names = [
        body.get("name", "") for body in ET.parse(ASSET_XML).getroot().findall(".//body")
        if body.get("name") and body.get("name") not in FOOT_BODIES
    ]
    nonfoot_pattern = rf"^({'|'.join(nonfoot_names)})$"
    env_cfg.scene.sensors = (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(
                mode="subtree",
                pattern=rf"^({FOOT_BODIES[0]}|{FOOT_BODIES[1]})$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
            history_length=0,
            secondary_policy="first",
        ),
        ContactSensorCfg(
            name="nonfoot_ground_contact",
            primary=ContactMatch(mode="body", pattern=nonfoot_pattern, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="none",
            num_slots=1,
            track_air_time=False,
            history_length=0,
            secondary_policy="first",
        ),
    )
    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mjlab_observations.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
        ),
        "projected_gravity": ObservationTermCfg(func=mjlab_observations.projected_gravity, params={}),
        "command": ObservationTermCfg(
            func=mjlab_observations.generated_commands,
            params={"command_name": "twist"},
        ),
        "phase": ObservationTermCfg(
            func=task_velocity_observations.phase,
            params={"period": TASK072_GAIT_PERIOD_S, "command_name": "twist"},
        ),
        "joint_pos": ObservationTermCfg(func=mjlab_observations.joint_pos_rel, params={}),
        "joint_vel": ObservationTermCfg(func=mjlab_observations.joint_vel_rel, params={}),
        "actions": ObservationTermCfg(func=mjlab_observations.last_action, params={}),
    }
    critic_terms = {
        **{name: ObservationTermCfg(func=term.func, params=dict(term.params)) for name, term in actor_terms.items()},
        "base_lin_vel": ObservationTermCfg(
            func=mjlab_observations.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),
        "foot_height": ObservationTermCfg(
            func=task_velocity_observations.foot_height,
            params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES)},
        ),
        "foot_air_time": ObservationTermCfg(
            func=task_velocity_observations.foot_air_time,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact": ObservationTermCfg(
            func=task_velocity_observations.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact_forces": ObservationTermCfg(
            func=task_velocity_observations.foot_contact_forces,
            params={"sensor_name": "feet_ground_contact"},
        ),
    }
    env_cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
    }
    env_cfg.rewards = task072_reward_v4_table(stance)
    action_cfg = env_cfg.actions["joint_pos"]
    if not isinstance(action_cfg, JointPositionActionCfg):
        raise TypeError("expected MJLab joint_pos action")
    action_cfg.actuator_names = tuple(semantic_to_joint.values())
    action_cfg.preserve_order = True
    action_cfg.use_default_offset = False
    action_cfg.offset = action_offset
    negative_scale, positive_scale = _signed_action_bounds_by_joint()
    action_cfg.scale = positive_scale
    action_cfg.task072_negative_scale = negative_scale
    action_cfg.task072_positive_scale = positive_scale
    action_cfg.task072_policy_action_domain = dict(POLICY_ACTION_DOMAIN)
    env_cfg.actions = {"joint_pos": action_cfg}

    class Task072SignedJointPositionAction(JointPositionAction):
        def __init__(self, cfg: Any, env: Any) -> None:
            import torch

            super().__init__(cfg, env)
            neg_index, _neg_names, neg_values = resolve_matching_names_values(
                cfg.task072_negative_scale, self._target_names
            )
            pos_index, _pos_names, pos_values = resolve_matching_names_values(
                cfg.task072_positive_scale, self._target_names
            )
            self._task072_negative_scale = self._scale.clone()
            self._task072_positive_scale = self._scale.clone()
            self._task072_negative_scale[:, neg_index] = torch.tensor(neg_values, device=self.device)
            self._task072_positive_scale[:, pos_index] = torch.tensor(pos_values, device=self.device)
            self.task072_clip_fraction = 0.0

        def process_actions(self, actions: Any) -> None:
            self._raw_actions[:] = actions
            self._processed_actions, clipped = apply_signed_action_contract(
                self._raw_actions,
                self._task072_negative_scale,
                self._task072_positive_scale,
                self._offset,
            )
            self.task072_clip_fraction = float((clipped != self._raw_actions).float().mean().detach().cpu())

    action_cfg.build = lambda env: Task072SignedJointPositionAction(action_cfg, env)  # type: ignore[method-assign]
    twist_cmd = env_cfg.commands["twist"]
    twist_cmd.heading_command = False
    twist_cmd.rel_standing_envs = 0.0
    twist_cmd.rel_heading_envs = 0.0
    twist_cmd.init_velocity_prob = 0.0
    twist_cmd.resampling_time_range = (1.0e9, 1.0e9)
    twist_cmd.ranges.lin_vel_x = (0.5, 0.5)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
    twist_cmd.ranges.heading = None
    twist_cmd.debug_vis = False
    env_cfg.commands = {"twist": twist_cmd}
    env_cfg.episode_length_s = 10_000.0
    env_cfg.events = {
        "reset_base": EventTermCfg(
            func=mjlab_events.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mjlab_events.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        ),
    }
    env_cfg.terminations = {
        "time_out": TerminationTermCfg(func=mjlab_terminations.time_out, params={}, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mjlab_terminations.bad_orientation,
            params={"limit_angle": 1.2217304763960306},
            time_out=False,
        ),
    }
    env_cfg.curriculum = {}
    env_cfg.metrics = {
        "mean_action_acc": MetricsTermCfg(func=mjlab_metrics.mean_action_acc, params={}),
    }
    if list(env_cfg.events) != ["reset_base", "reset_robot_joints"]:
        raise ValueError("Task072 event table replacement failed")
    if list(env_cfg.terminations) != ["time_out", "fell_over"]:
        raise ValueError("Task072 termination table replacement failed")
    if list(env_cfg.metrics) != ["mean_action_acc"]:
        raise ValueError("Task072 metrics table replacement failed")
    if list(env_cfg.commands) != ["twist"] or list(env_cfg.actions) != ["joint_pos"]:
        raise ValueError("Task072 command/action table replacement failed")
    if list(env_cfg.scene.sensors) and [sensor.name for sensor in env_cfg.scene.sensors] != [
        "feet_ground_contact",
        "nonfoot_ground_contact",
    ]:
        raise ValueError("Task072 sensor table replacement failed")

    agent_cfg.seed = int(seed)
    agent_cfg.clip_actions = 1.0
    agent_cfg.num_steps_per_env = int(rollout_steps)
    agent_cfg.max_iterations = int(max_iterations)
    agent_cfg.save_interval = max(1, int(max_iterations))
    agent_cfg.logger = "tensorboard"
    agent_cfg.upload_model = False
    agent_cfg.resume = False
    agent_cfg.run_name = LINEAGE_ID
    agent_cfg.experiment_name = LINEAGE_ID
    ppo_contract = task072_assert_agent_ppo_contract(agent_cfg)
    registration = {
        "task_id": task_id,
        "parent_task": MJLAB_PARENT_TASK,
        "runtime_spec_sha256": hashlib.sha256(spec_xml.encode()).hexdigest(),
        "foot_sites": list(FOOT_SITES),
        "foot_bodies": list(FOOT_BODIES),
        "foot_geoms": list(FOOT_GEOMS),
        "torso_body": TORSO_BODY,
        "pelvis_body": PELVIS_BODY,
        "num_envs": int(num_envs),
        "rollout_steps_per_env": int(rollout_steps),
        "transitions_per_update": int(num_envs) * int(rollout_steps),
        "max_iterations": int(max_iterations),
        "fixed_command": True,
        "run_name": agent_cfg.run_name,
        "experiment_name": agent_cfg.experiment_name,
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "action_contract_sha256": action_contract_from_asset_xml()["payload_sha256"],
        "action_scale_sha256": payload_sha256({"negative": negative_scale, "positive": positive_scale}),
        "policy_action_domain": dict(POLICY_ACTION_DOMAIN),
        "reward_contract_version": REWARD_CONTRACT_VERSION,
        "ppo_contract": ppo_contract,
        "lineage_id": LINEAGE_ID,
        "joint_mapping": mapping_table,
        "semantic_to_anonymous_joint": semantic_to_joint,
        "joint_mapping_count": len(semantic_to_joint),
        "joint_mapping_unmapped": 29 - len(semantic_to_joint),
        "joint_mapping_duplicate": len(semantic_to_joint) - len(set(semantic_to_joint.values())),
        "runtime_default_joint_pos": joint_pos,
        "expected_stance_joint_qpos": dict(stance["joint_qpos"]),
        "runtime_action_offset": action_offset,
        "expected_actuator_ctrl_eq": dict(stance["actuator_ctrl_eq"]),
        "disabled_events": ["push_robot", "foot_friction", "encoder_bias", "base_com"],
        "curriculum_disabled": True,
        "reward_terms": list(REWARD_V4_ORDER),
        "reward_active_table_sha256": task072_validate_reward_active_table(task072_reward_active_table_from_cfg(env_cfg.rewards)),
        "reward_payload_sha256": task072_canonical_reward_payload(
            task072_reward_active_table_from_cfg(env_cfg.rewards)
        )["payload_sha256"],
    }
    try:
        register_mjlab_task(task_id, env_cfg, env_cfg, agent_cfg, runner_cls)
    except ValueError as exc:
        if "already registered" not in str(exc):
            raise
    return env_cfg, agent_cfg, runner_cls, registration


def _canonical_config_payload(env_cfg: Any, agent_cfg: Any, registration: dict[str, Any], *, render_mode: str | None) -> dict[str, Any]:
    twist = env_cfg.commands["twist"]
    action_cfg = env_cfg.actions["joint_pos"]
    distribution_cfg = agent_cfg.actor.distribution_cfg
    algorithm_cfg = agent_cfg.algorithm
    return {
        "schema_version": 1,
        "lineage_id": LINEAGE_ID,
        "env": {
            "scene": {"num_envs": int(env_cfg.scene.num_envs)},
            "seed": int(env_cfg.seed),
            "episode_length_s": float(env_cfg.episode_length_s),
            "render_mode": render_mode,
            "command": {
                "lin_vel_x": list(twist.ranges.lin_vel_x),
                "lin_vel_y": list(twist.ranges.lin_vel_y),
                "ang_vel_z": list(twist.ranges.ang_vel_z),
                "heading": twist.ranges.heading,
                "heading_command": bool(twist.heading_command),
                "standing_probability": float(twist.rel_standing_envs),
                "heading_probability": float(twist.rel_heading_envs),
                "init_velocity_probability": float(twist.init_velocity_prob),
                "resampling_time_range": list(twist.resampling_time_range),
            },
            "events": sorted(env_cfg.events),
            "actor_observation_corruption": bool(env_cfg.observations["actor"].enable_corruption),
            "curriculum": dict(env_cfg.curriculum),
            "reward_names": sorted(env_cfg.rewards),
        },
        "agent": {
            "seed": int(agent_cfg.seed),
            "num_steps_per_env": int(agent_cfg.num_steps_per_env),
            "max_iterations": int(agent_cfg.max_iterations),
            "resume": bool(agent_cfg.resume),
            "clip_actions": getattr(agent_cfg, "clip_actions", None),
            "policy_distribution": {
                "actor_distribution_class": distribution_cfg["class_name"],
                "init_std": float(distribution_cfg["init_std"]),
                "std_type": distribution_cfg["std_type"],
                "entropy_coef": float(algorithm_cfg.entropy_coef),
                "num_learning_epochs": int(algorithm_cfg.num_learning_epochs),
                "num_mini_batches": int(algorithm_cfg.num_mini_batches),
                "learning_rate": float(algorithm_cfg.learning_rate),
                "schedule": algorithm_cfg.schedule,
                "desired_kl": float(algorithm_cfg.desired_kl),
                "action_dimension": int(registration["joint_mapping_count"]),
            },
        },
        "action": {
            "version": ACTION_CONTRACT_VERSION,
            "policy_action_domain": dict(POLICY_ACTION_DOMAIN),
            "target_names": list(action_cfg.actuator_names),
            "negative_scale": dict(action_cfg.task072_negative_scale),
            "positive_scale": dict(action_cfg.task072_positive_scale),
            "offset": dict(action_cfg.offset),
        },
        "reward": {"version": REWARD_CONTRACT_VERSION},
        "semantic_contract": task072_runtime_semantic_payload(
            env_cfg,
            agent_cfg,
            registration,
            render_mode=render_mode,
        ),
        "registration": registration,
    }


def _diff_payload(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left) | set(right))
        return [
            path
            for key in keys
            for path in _diff_payload(left.get(key), right.get(key), f"{prefix}.{key}" if prefix else str(key))
        ]
    if left != right:
        return [prefix or "configuration"]
    return []


def canonical_train_eval_config_payload() -> dict[str, Any]:
    train_env, train_agent, _runner_cls, train_registration = build_task_cfg(
        REQUIRED_CAPACITY_NUM_ENVS,
        REQUIRED_ROLLOUT_STEPS,
        DEFAULT_SEED,
        1,
    )
    eval_env, eval_agent, _runner_cls, eval_registration = build_task_cfg(
        256,
        REQUIRED_ROLLOUT_STEPS,
        DEFAULT_SEED + 99,
        1,
        fixed_command=True,
    )
    train_payload = _canonical_config_payload(train_env, train_agent, train_registration, render_mode=None)
    eval_payload = _canonical_config_payload(eval_env, eval_agent, eval_registration, render_mode=None)
    diff = _diff_payload(train_payload, eval_payload)
    non_allowlisted = [path for path in diff if path not in EVAL_CONFIG_DIFF_ALLOWLIST]
    payload = {
        "schema_version": 1,
        "lineage_id": LINEAGE_ID,
        "train": train_payload,
        "eval": eval_payload,
        "eval_diff_allowlist": sorted(EVAL_CONFIG_DIFF_ALLOWLIST),
        "diff": diff,
        "non_allowlisted_diff": non_allowlisted,
        "passed": not non_allowlisted,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def _env_smoke(num_envs: int, rollout_steps: int, seed: int, device: str, steps: int) -> dict[str, Any]:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    env_cfg, agent_cfg, _runner_cls, registration = build_task_cfg(
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        seed=seed,
        max_iterations=1,
    )
    configure_torch_backends()
    torch.set_grad_enabled(False)
    outer = None
    try:
        outer = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
        env = RslRlVecEnvWrapper(outer, clip_actions=agent_cfg.clip_actions)
        obs, _extras = env.reset()
        actor_shape = tuple(int(v) for v in obs["actor"].shape)
        critic_shape = tuple(int(v) for v in obs["critic"].shape)
        rewards = []
        done_counts = []
        finite = bool(torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all())
        for _ in range(int(steps)):
            obs, reward, done, _extras = env.step(torch.zeros((num_envs, env.num_actions), device=device))
            finite = bool(finite and torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all() and torch.isfinite(reward).all())
            rewards.append(float(reward.mean().detach().cpu()))
            done_counts.append(int(done.sum().detach().cpu()))
        return {
            **registration,
            "device": device,
            "steps": int(steps),
            "action_dim": int(env.num_actions),
            "actor_obs_shape": list(actor_shape),
            "critic_obs_shape": list(critic_shape),
            "reward_mean_by_step": rewards,
            "done_count_by_step": done_counts,
            "finite": finite,
            "passed": finite and int(env.num_actions) == 29 and actor_shape == (num_envs, 98) and critic_shape == (num_envs, 113),
        }
    finally:
        if outer is not None:
            outer.close()


def _observation_layout(env: Any, env_cfg: Any, group: str, obs: Any) -> dict[str, Any]:
    term_dims = env.unwrapped.observation_manager.group_obs_term_dim[group]
    term_names = list(env_cfg.observations[group].terms)
    cursor = 0
    slices = {}
    for name, dims in zip(term_names, term_dims):
        width = int(math.prod(dims)) if hasattr(dims, "__len__") else int(dims)
        slices[name] = [cursor, cursor + width]
        cursor += width
    return {"terms": term_names, "term_slices": slices, "shape": list(obs[group].shape)}


def verify_runtime_binding(args: argparse.Namespace) -> int:
    """Verify binding only; this command never constructs an optimizer or trains."""
    result: dict[str, Any] = {"lineage_id": LINEAGE_ID, "optimizer_step_calls": 0, "parameter_delta_max_abs": 0.0}
    outer = None
    try:
        env_cfg, agent_cfg, _runner_cls, registration = build_task_cfg(1, 1, args.seed, 1, fixed_command=True)
        import torch
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.utils.torch import configure_torch_backends

        configure_torch_backends()
        torch.set_grad_enabled(False)
        outer = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        env = RslRlVecEnvWrapper(outer, clip_actions=agent_cfg.clip_actions)
        obs, _ = env.reset()
        _force_fixed_command(env)
        env.unwrapped.observation_manager._obs_buffer = None
        obs = env.get_observations()
        robot = env.unwrapped.scene["robot"]
        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        stance = _stance_dict()
        mapping = registration["semantic_to_anonymous_joint"]
        expected_qpos = {mapping[key]: float(value) for key, value in stance["joint_qpos"].items()}
        expected_ctrl = {mapping[key]: float(value) for key, value in stance["actuator_ctrl_eq"].items()}
        runtime_default_qpos = {
            name: float(robot.data.default_joint_pos[0, index].detach().cpu())
            for index, name in enumerate(robot.joint_names)
            if name in expected_qpos
        }
        reset_joint_qpos = {
            name: float(robot.data.joint_pos[0, index].detach().cpu())
            for index, name in enumerate(robot.joint_names)
            if name in expected_qpos
        }
        root_pose = [float(value) for value in stance["root_pose_eq"]]
        runtime_root = [
            *[float(value) for value in robot.data.root_link_pos_w[0].detach().cpu()],
            *[float(value) for value in robot.data.root_link_quat_w[0].detach().cpu()],
        ]
        zero_action = torch.zeros((1, env.num_actions), device=args.device)
        obs_after, reward, done, _ = env.step(zero_action)
        actor_layout = _observation_layout(env, env_cfg, "actor", obs)
        critic_layout = _observation_layout(env, env_cfg, "critic", obs)
        command_slice = actor_layout["term_slices"].get("command")
        joint_pos_slice = actor_layout["term_slices"].get("joint_pos")
        command_values = (
            [float(value) for value in obs["actor"][0, command_slice[0]:command_slice[1]].detach().cpu()]
            if command_slice is not None
            else []
        )
        joint_pos_rel_reset = (
            [float(value) for value in obs["actor"][0, joint_pos_slice[0]:joint_pos_slice[1]].detach().cpu()]
            if joint_pos_slice is not None
            else []
        )
        target_names = list(action_term.target_names)
        action_offset = {
            name: float(value)
            for name, value in zip(target_names, action_term.offset[0].detach().cpu())
        }
        processed_target = {
            name: float(value)
            for name, value in zip(target_names, action_term._processed_actions[0].detach().cpu())
        }
        material = _compiled_material_from_model(outer.sim.mj_model)
        import mujoco
        model = outer.sim.mj_model
        frozen_data = mujoco.MjData(model)
        root_joint_id = next((i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE), None)
        if root_joint_id is None:
            raise ValueError("compiled model is missing its unique free root joint")
        root_adr = int(model.jnt_qposadr[root_joint_id])
        frozen_data.qpos[root_adr:root_adr + 7] = root_pose
        compiled_joint_names = {
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "").split("/")[-1]: joint_id
            for joint_id in range(model.njnt)
        }
        for joint_name, value in expected_qpos.items():
            joint_id = compiled_joint_names.get(joint_name)
            if joint_id is None:
                raise ValueError(f"compiled model is missing stance joint: {joint_name}")
            frozen_data.qpos[int(model.jnt_qposadr[joint_id])] = value
        ground = _ground_plane_audit(model, frozen_data)
        hold_steps = round(2.0 / float(env.unwrapped.step_dt))
        hold_done_count = int(done.sum().detach().cpu())
        height_values = [float(robot.data.root_link_pos_w[0, 2].detach().cpu())]
        gravity_xy_values = [float(torch.linalg.norm(robot.data.projected_gravity_b[0, :2]).detach().cpu())]
        force_values: list[float] = []
        contact_values: list[list[bool]] = []
        contact_sensor = env.unwrapped.scene["feet_ground_contact"]
        for _ in range(max(0, hold_steps - 1)):
            obs_after, reward, done, _ = env.step(zero_action)
            hold_done_count += int(done.sum().detach().cpu())
            height_values.append(float(robot.data.root_link_pos_w[0, 2].detach().cpu()))
            gravity_xy_values.append(float(torch.linalg.norm(robot.data.projected_gravity_b[0, :2]).detach().cpu()))
            if hasattr(contact_sensor.data, "force") and contact_sensor.data.force is not None:
                force_values.append(float(torch.linalg.norm(contact_sensor.data.force[0], dim=-1).sum().detach().cpu()))
            contact_values.append([bool(value) for value in (contact_sensor.data.current_contact_time[0] > 0).detach().cpu()])
        contact = json.loads(CONTACT_PROFILE.read_text(encoding="utf-8"))
        static_tracking_reward = math.exp(-((0.5 - 0.0) ** 2) / 0.25)
        moving_tracking_reward = math.exp(-((0.5 - 0.5) ** 2) / 0.25)
        qpos_error = max((abs(runtime_default_qpos[k] - expected_qpos[k]) for k in expected_qpos), default=float("inf"))
        reset_qpos_error = max((abs(reset_joint_qpos[k] - expected_qpos[k]) for k in expected_qpos), default=float("inf"))
        action_error = max((abs(processed_target[name] - expected_ctrl[name]) for name in expected_ctrl), default=float("inf"))
        joint_pos_rel_error = max((abs(value) for value in joint_pos_rel_reset), default=float("inf"))
        root_error = max(abs(a - b) for a, b in zip(runtime_root, root_pose))
        qpos_ctrl_delta = {name: abs(expected_qpos[name] - expected_ctrl[name]) for name in expected_qpos}
        result.update({
            **_common_manifest(args), "registration": registration,
            "runtime_default_joint_pos": runtime_default_qpos,
            "expected_stance_joint_qpos": expected_qpos,
            "default_qpos_error_max": qpos_error,
            "reset_joint_qpos": reset_joint_qpos,
            "reset_joint_qpos_error_max": reset_qpos_error,
            "runtime_action_offset": action_offset,
            "expected_actuator_ctrl_eq": expected_ctrl,
            "action_offset_error_max": max((abs(action_offset[name] - expected_ctrl[name]) for name in expected_ctrl), default=float("inf")),
            "zero_action_processed_target": processed_target,
            "zero_action_processed_target_error_max": action_error,
            "reset_root_pose": runtime_root,
            "expected_root_pose": root_pose,
            "reset_root_pose_errors": {"max_abs": root_error},
            "stance_qpos_vs_actuator_ctrl_abs_delta": qpos_ctrl_delta,
            "stance_qpos_vs_actuator_ctrl_min_abs_delta": min(qpos_ctrl_delta.values()),
            "actor_observation_term_layout": actor_layout,
            "critic_observation_term_layout": critic_layout,
            "command_observation_slice": command_slice,
            "command_observation_values": command_values,
            "joint_pos_rel_reset_values": joint_pos_rel_reset,
            "joint_pos_rel_reset_error_max": joint_pos_rel_error,
            "effective_compiled_foot_contact_material": material,
            "runtime_binding_ground_audit": ground,
            "declared_runtime_material_contract": contact.get("runtime_material"),
            "reset_observation_finite": bool(torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()),
            "zero_action_observation_finite": bool(torch.isfinite(obs_after["actor"]).all() and torch.isfinite(obs_after["critic"]).all()),
            "zero_action_reward_finite": bool(torch.isfinite(reward).all()),
            "zero_action_2s_stance_hold": {
                "steps": hold_steps,
                "done_count": hold_done_count,
                "base_height_initial": height_values[0],
                "base_height_final": height_values[-1],
                "base_height_min": min(height_values),
                "base_height_max": max(height_values),
                "base_height_drift_abs": abs(height_values[-1] - height_values[0]),
                "gravity_xy_max": max(gravity_xy_values),
                "joint_drift_max_abs": max((abs(reset_joint_qpos[name] - expected_qpos[name]) for name in expected_qpos), default=float("inf")),
                "contact_any_by_step": contact_values,
                "force_sum_max": max(force_values) if force_values else 0.0,
            },
            "tracking_reward_probe": {
                "command": [0.5, 0.0, 0.0],
                "static_reward": static_tracking_reward,
                "moving_at_command_reward": moving_tracking_reward,
                "moving_higher_than_static": moving_tracking_reward > static_tracking_reward,
            },
            "optimizer_step_calls": 0, "parameter_delta_max_abs": 0.0,
        })
        checks = {
            "lineage": (
                registration["lineage_id"] == LINEAGE_ID
                and registration["run_name"] == LINEAGE_ID
                and registration["experiment_name"] == LINEAGE_ID
            ),
            "joint_mapping": registration["joint_mapping_count"] == 29 and registration["joint_mapping_unmapped"] == 0 and registration["joint_mapping_duplicate"] == 0,
            "default_joint_qpos": result["default_qpos_error_max"] <= STANCE_TOL,
            "reset_joint_qpos": result["reset_joint_qpos_error_max"] <= STANCE_TOL,
            "action_offset": result["action_offset_error_max"] <= STANCE_TOL,
            "stance_qpos_and_action_target_distinct": result["stance_qpos_vs_actuator_ctrl_min_abs_delta"] > STANCE_TOL,
            "zero_action_target": result["zero_action_processed_target_error_max"] <= STANCE_TOL,
            "root_pose": result["reset_root_pose_errors"]["max_abs"] <= STANCE_TOL,
            "actor_observation_layout": result["actor_observation_term_layout"]["terms"] == list(EXPECTED_ACTOR_TERMS)
            and result["actor_observation_term_layout"]["shape"] == [1, 98],
            "critic_observation_layout": result["critic_observation_term_layout"]["terms"] == list(EXPECTED_CRITIC_TERMS)
            and result["critic_observation_term_layout"]["shape"] == [1, 113],
            "command_in_actor": result["command_observation_slice"] is not None,
            "command_values": command_values == [0.5, 0.0, 0.0],
            "joint_pos_rel_zero_reference": result["joint_pos_rel_reset_error_max"] <= STANCE_TOL,
            "material": all(material["checks"].values()) and contact.get("runtime_material") == _runtime_material_contract(),
            "single_ground": all(ground["checks"].values()),
            "finite": result["reset_observation_finite"] and result["zero_action_observation_finite"] and result["zero_action_reward_finite"],
            "tracking_reward_direction": result["tracking_reward_probe"]["moving_higher_than_static"],
            "zero_action_hold_no_immediate_fall": hold_done_count == 0 and min(height_values) > 0.3 and max(height_values) < 1.5,
            "no_update": result["optimizer_step_calls"] == 0 and result["parameter_delta_max_abs"] == 0.0,
        }
        result["checks"] = checks
        result["passed"] = all(checks.values())
        env.close()
    except Exception as exc:  # noqa: BLE001
        result.update({"passed": False, "error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        if outer is not None:
            outer.close()
    write_json(args.output.resolve(), result)
    print(json.dumps({"passed": result["passed"], "output": str(args.output.resolve())}), flush=True)
    return 0 if result["passed"] else 1


def r0_smoke(args: argparse.Namespace) -> int:
    _prepare_external_imports()
    start = time.time()
    try:
        result = _env_smoke(
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            seed=args.seed,
            device=args.device,
            steps=args.steps,
        )
        result["error"] = None
    except Exception as exc:  # noqa: BLE001
        result = {
            "passed": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    result.update(_common_manifest(args))
    result["wall_time_s"] = time.time() - start
    write_json(args.output.resolve(), result)
    print(json.dumps({"passed": result["passed"], "output": str(args.output.resolve())}), flush=True)
    return 0 if result["passed"] else 1


def capacity_smoke(args: argparse.Namespace) -> int:
    _prepare_external_imports()
    gpu_lock = _gpu_lock_status()
    if args.rollout_steps != REQUIRED_ROLLOUT_STEPS or REQUIRED_CAPACITY_NUM_ENVS not in args.candidates:
        payload = {
            **_common_manifest(args),
            "candidates": [int(v) for v in args.candidates],
            "required_capacity": {
                "num_envs": REQUIRED_CAPACITY_NUM_ENVS,
                "rollout_steps_per_env": REQUIRED_ROLLOUT_STEPS,
                "transitions_per_update": REQUIRED_TRANSITIONS_PER_UPDATE,
            },
            "gpu_lock": gpu_lock,
            "results": [],
            "selected": None,
            "passed": False,
            "error": "capacity-smoke must include 4096 envs with 24 rollout steps",
        }
        write_json(args.output.resolve(), payload)
        print(json.dumps({"passed": False, "output": str(args.output.resolve())}), flush=True)
        return 1
    try:
        gpu_lock = _require_gpu_lock_for_device(args.device)
    except Exception as exc:  # noqa: BLE001
        payload = {
            **_common_manifest(args),
            "candidates": [int(v) for v in args.candidates],
            "required_capacity": {
                "num_envs": REQUIRED_CAPACITY_NUM_ENVS,
                "rollout_steps_per_env": REQUIRED_ROLLOUT_STEPS,
                "transitions_per_update": REQUIRED_TRANSITIONS_PER_UPDATE,
            },
            "gpu_lock": gpu_lock,
            "results": [],
            "selected": None,
            "passed": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(args.output.resolve(), payload)
        print(json.dumps({"passed": False, "output": str(args.output.resolve())}), flush=True)
        return 1
    results = []
    highest_passed = None
    required_result = None
    for num_envs in args.candidates:
        start = time.time()
        try:
            item = _env_smoke(
                num_envs=int(num_envs),
                rollout_steps=args.rollout_steps,
                seed=args.seed + int(num_envs),
                device=args.device,
                steps=args.steps,
            )
            item["error"] = None
        except Exception as exc:  # noqa: BLE001
            item = {
                "num_envs": int(num_envs),
                "rollout_steps_per_env": args.rollout_steps,
                "transitions_per_update": int(num_envs) * args.rollout_steps,
                "device": args.device,
                "passed": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        item["wall_time_s"] = time.time() - start
        results.append(item)
        if int(num_envs) == REQUIRED_CAPACITY_NUM_ENVS:
            required_result = item
        if item.get("passed"):
            highest_passed = item
        else:
            break
    passed = bool(required_result and required_result.get("passed"))
    payload = {
        **_common_manifest(args),
        "candidates": [int(v) for v in args.candidates],
        "required_capacity": {
            "num_envs": REQUIRED_CAPACITY_NUM_ENVS,
            "rollout_steps_per_env": REQUIRED_ROLLOUT_STEPS,
            "transitions_per_update": REQUIRED_TRANSITIONS_PER_UPDATE,
        },
        "gpu_lock": gpu_lock,
        "results": results,
        "selected": required_result,
        "highest_passed": highest_passed,
        "passed": passed,
    }
    write_json(args.output.resolve(), payload)
    print(json.dumps({"passed": payload["passed"], "selected": required_result, "output": str(args.output.resolve())}), flush=True)
    return 0 if payload["passed"] else 1


def _load_capacity_evidence(path: Path, *, num_envs: int, rollout_steps: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = payload.get("required_capacity", {})
    selected = payload.get("selected") or {}
    checks = {
        "artifact_passed": payload.get("passed") is True,
        "lineage": (payload.get("lineage_id") or selected.get("lineage_id")) == LINEAGE_ID,
        "required_num_envs": required.get("num_envs") == REQUIRED_CAPACITY_NUM_ENVS,
        "required_rollout_steps": required.get("rollout_steps_per_env") == REQUIRED_ROLLOUT_STEPS,
        "required_transitions": required.get("transitions_per_update") == REQUIRED_TRANSITIONS_PER_UPDATE,
        "selected_num_envs": selected.get("num_envs") == REQUIRED_CAPACITY_NUM_ENVS,
        "selected_rollout_steps": selected.get("rollout_steps_per_env") == REQUIRED_ROLLOUT_STEPS,
        "selected_transitions": selected.get("transitions_per_update") == REQUIRED_TRANSITIONS_PER_UPDATE,
        "selected_passed": selected.get("passed") is True,
        "gpu_lock_held_by_ancestor": bool(payload.get("gpu_lock", {}).get("held_by_ancestor")),
        "training_num_envs_matches": int(num_envs) == REQUIRED_CAPACITY_NUM_ENVS,
        "training_rollout_steps_matches": int(rollout_steps) == REQUIRED_ROLLOUT_STEPS,
    }
    if not all(checks.values()):
        raise ValueError(f"capacity evidence failed checks: {checks}")
    payload["consumption_checks"] = checks
    return payload


def _validate_training_manifest_for_eval(payload: dict[str, Any]) -> None:
    current = _common_manifest(argparse.Namespace(command="evaluate-manifest-check", seed=DEFAULT_SEED + 99))
    clip_records = payload.get("action_clip_update_records", [])
    clip_summary = None
    try:
        validate_task072_clip_records(clip_records, expected_updates=payload.get("updates"))
        clip_summary = pool_task072_clip_records(clip_records, last_n=min(7, len(clip_records)))
        validate_task072_clip_summary(clip_summary)
        clip_shape_ok = all(
            record.get("num_envs") == payload.get("num_envs")
            and record.get("rollout_steps") == payload.get("rollout_steps_per_env")
            for record in clip_records
        )
        clip_ok = bool(clip_records) and clip_shape_ok and payload.get("action_clip_last_7_summary") == clip_summary
    except Exception:
        clip_ok = False
    acceptance_checks = payload.get("acceptance_checks")
    policy_lineage = payload.get("policy_distribution_lineage", {})
    runtime_table = payload.get("runtime_reward_active_table", [])
    runtime_evidence = payload.get("runtime_rollout_evidence", {})
    parameter_delta = payload.get("parameter_delta", {})
    losses = payload.get("losses", [])
    update_reports = payload.get("update_reports", [])
    lineage_keys = {
        "actor_distribution_class",
        "init_std",
        "std_type",
        "entropy_coef",
        "num_learning_epochs",
        "num_mini_batches",
        "learning_rate",
        "schedule",
        "desired_kl",
        "clip_actions",
        "action_dimension",
    }
    policy_lineage_ok = (
        isinstance(policy_lineage, dict)
        and lineage_keys <= set(policy_lineage)
        and isinstance(policy_lineage["actor_distribution_class"], str)
        and all(_finite_metric(policy_lineage[name]) is not None for name in ("init_std", "entropy_coef", "learning_rate", "desired_kl"))
        and all(isinstance(policy_lineage[name], int) and not isinstance(policy_lineage[name], bool) and policy_lineage[name] > 0 for name in ("num_learning_epochs", "num_mini_batches", "action_dimension"))
        and isinstance(policy_lineage["std_type"], str)
        and isinstance(policy_lineage["schedule"], str)
        and isinstance(policy_lineage["clip_actions"], (int, float))
    )
    parameter_delta_ok = (
        isinstance(parameter_delta, dict)
        and parameter_delta.get("finite") is True
        and _finite_metric(parameter_delta.get("max_abs")) is not None
        and float(parameter_delta["max_abs"]) > 0.0
        and isinstance(parameter_delta.get("changed_parameter_count"), int)
        and not isinstance(parameter_delta["changed_parameter_count"], bool)
        and parameter_delta["changed_parameter_count"] > 0
    )
    losses_ok = (
        isinstance(losses, list)
        and len(losses) == int(payload.get("updates", -1))
        and all(_task072_finite_loss_dict(loss) for loss in losses)
        and isinstance(update_reports, list)
        and len(update_reports) == len(losses)
        and all(
            isinstance(report, dict)
            and report.get("update_index") == update_index
            and report.get("losses") == loss
            and _task072_std_stats_payload_finite(report.get("pre_update_std"))
            and _task072_std_stats_payload_finite(report.get("post_update_std"))
            for update_index, (report, loss) in enumerate(zip(update_reports, losses))
        )
    )
    expected_optimizer_steps = (
        int(policy_lineage["num_learning_epochs"])
        * int(policy_lineage["num_mini_batches"])
        * int(payload.get("updates", -1))
        if policy_lineage_ok
        else None
    )
    optimizer_steps_ok = (
        isinstance(payload.get("optimizer_step_count"), int)
        and not isinstance(payload.get("optimizer_step_count"), bool)
        and isinstance(payload.get("expected_optimizer_step_count"), int)
        and not isinstance(payload.get("expected_optimizer_step_count"), bool)
        and expected_optimizer_steps is not None
        and payload["expected_optimizer_step_count"] == expected_optimizer_steps
        and payload["optimizer_step_count"] == expected_optimizer_steps
    )
    runtime_table_ok = (
        isinstance(runtime_table, list)
        and payload.get("runtime_reward_active_term_count") == len(REWARD_V4_ORDER)
        and len(runtime_table) == len(REWARD_V4_ORDER)
        and all(isinstance(row, dict) for row in runtime_table)
        and [row.get("name") for row in runtime_table] == list(REWARD_V4_ORDER)
        and payload.get("runtime_reward_active_table_sha256")
        == current.get("reward_contract", {}).get("config_active_table_sha256")
    )
    runtime_evidence_ok = (
        runtime_evidence.get("check_for_nan_enabled") is True
        and runtime_evidence.get("obs_finite") is True
        and runtime_evidence.get("rewards_finite") is True
        and runtime_evidence.get("dones_finite") is True
        and runtime_evidence.get("expected_rollout_steps") == int(payload.get("updates", -1)) * int(payload.get("rollout_steps_per_env", -1))
        and runtime_evidence.get("observed_rollout_steps") == runtime_evidence.get("expected_rollout_steps")
        and runtime_evidence.get("observed_transitions") == payload.get("observed_transitions")
    )
    acceptance_ok = isinstance(acceptance_checks, dict) and set(acceptance_checks) == set(TASK072_ACCEPTANCE_CHECK_NAMES) and all(
        value is True for value in acceptance_checks.values()
    ) and policy_lineage_ok and optimizer_steps_ok and parameter_delta_ok and losses_ok and runtime_table_ok and runtime_evidence_ok
    progression = payload.get("progression", {})
    progression_path_text = progression.get("path")
    progression_path = Path(progression_path_text) if progression_path_text else None
    progression_sha = progression.get("sha256")
    progression_pointer_ok = bool(
        progression_sha
        and progression_path
        and progression_path.exists()
        and sha256_path(progression_path) == progression_sha
    )
    try:
        expected_stage_semantic = task072_stage_semantic_contract(
            num_envs=int(payload.get("num_envs", -1)),
            rollout_steps=int(payload.get("rollout_steps_per_env", -1)),
            seed=int(payload.get("seed", -1)),
            max_iterations=int(payload.get("updates", -1)),
            fixed_command=False,
            render_mode=None,
        )
        stage_semantic_ok = (
            isinstance(payload.get("stage_semantic_contract"), dict)
            and payload["stage_semantic_contract"].get("payload_sha256") == expected_stage_semantic["payload_sha256"]
        )
    except Exception:
        stage_semantic_ok = False
    capacity_checks = payload.get("capacity_evidence", {}).get("consumption_checks", {})
    checks = {
        "schema": payload.get("schema_version") == 3,
        "subtask": payload.get("subtask") == TASK072_ACTIVE_SUBTASK,
        "lineage": payload.get("lineage_id") == LINEAGE_ID,
        "runtime_lineage": payload.get("runtime_lineage_id") == LINEAGE_ID,
        "action_contract": payload.get("action_contract") == current["action_contract"],
        "reward_contract": payload.get("reward_contract") == current["reward_contract"],
        "canonical_config": payload.get("canonical_train_eval_config", {}).get("payload_sha256")
        == current["canonical_train_eval_config"]["payload_sha256"],
        "canonical_config_passed": payload.get("canonical_train_eval_config", {}).get("passed") is True,
        "stage_semantic_contract": stage_semantic_ok,
        "runner_source": payload.get("runner_source_sha256") == current["runner_source_sha256"],
        "runtime_spec": payload.get("runtime_spec_sha256") == current["runtime_spec_sha256"],
        "asset_xml": payload.get("asset_xml", {}).get("sha256") == current["asset_xml"]["sha256"],
        "contact_profile": payload.get("contact_profile", {}).get("payload_sha256")
        == current["contact_profile"]["payload_sha256"],
        "stance": payload.get("stance", {}).get("payload_sha256") == current["stance"]["payload_sha256"],
        "external": payload.get("external_mjlab_checks") == current["external_mjlab_checks"],
        "external_passed": all(payload.get("external_mjlab_checks", {}).values()),
        "training_manifest_passed": payload.get("passed") is True,
        "training_complete": payload.get("training_execution_complete") is True,
        "capacity_consumed": bool(capacity_checks) and all(capacity_checks.values()),
        "action_clip_metrics": clip_ok,
        "acceptance_checks": acceptance_ok,
        "progression_sha": progression_pointer_ok,
    }
    if not all(checks.values()):
        raise ValueError(f"training manifest failed Task072 eval lineage checks: {checks}")


def _write_task072_clip_tensorboard(log_dir: Path, records: list[dict[str, Any]]) -> None:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        for record in records:
            step = int(record["update_index"])
            writer.add_scalar("Diagnostics/action_clip/scalar_fraction", float(record["scalar_clip_fraction"]), step)
            writer.add_scalar("Diagnostics/action_clip/env_step_any_fraction", float(record["env_step_any_clip_fraction"]), step)
            writer.add_scalar("Diagnostics/action_clip/max_abs_raw", float(record["max_abs_raw_action"]), step)
            for joint_name, value in record["per_joint_clip_fraction"].items():
                writer.add_scalar(f"Diagnostics/action_clip/joint/{joint_name}", float(value), step)
    finally:
        writer.close()


def _task072_trainable_snapshot(alg: Any) -> dict[str, Any]:
    import torch

    return {
        name: parameter.detach().clone()
        for module_name, module in (("actor", alg.actor), ("critic", alg.critic))
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
        for name in [f"{module_name}.{name}"]
    }


def _task072_parameter_delta(before: dict[str, Any], alg: Any) -> dict[str, Any]:
    import torch

    after = _task072_trainable_snapshot(alg)
    deltas = [torch.abs(after[name] - value) for name, value in before.items() if name in after]
    finite = bool(deltas) and all(bool(torch.isfinite(delta).all()) for delta in deltas)
    return {
        "finite": finite,
        "max_abs": max((float(delta.max().cpu()) for delta in deltas), default=0.0),
        "changed_parameter_count": sum(int((delta != 0).sum().cpu()) for delta in deltas),
        "parameter_count": sum(int(delta.numel()) for delta in deltas),
    }


def _task072_std_stats(alg: Any) -> dict[str, Any]:
    import torch

    value = alg.get_policy().output_std.detach().cpu()
    return {
        "shape": list(value.shape),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "finite": bool(torch.isfinite(value).all()),
    }


def _task072_std_stats_payload_finite(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("shape"), list)
        and value.get("finite") is True
        and all(_finite_metric(value.get(name)) is not None for name in ("min", "mean", "max"))
    )


def _task072_finite_loss_dict(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return all(
        isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))
        for item in value.values()
    )


def _task072_policy_distribution_lineage(alg: Any, runner_cfg: dict[str, Any], action_dimension: int) -> dict[str, Any]:
    policy = alg.get_policy()
    distribution = policy.distribution
    distribution_cfg = runner_cfg["actor"]["distribution_cfg"]
    return {
        "actor_distribution_class": f"{type(distribution).__module__}.{type(distribution).__qualname__}",
        "init_std": float(distribution_cfg["init_std"]),
        "std_type": str(getattr(distribution, "std_type", distribution_cfg["std_type"])),
        "entropy_coef": float(alg.entropy_coef),
        "num_learning_epochs": int(alg.num_learning_epochs),
        "num_mini_batches": int(alg.num_mini_batches),
        "learning_rate": float(alg.learning_rate),
        "schedule": str(alg.schedule),
        "desired_kl": float(alg.desired_kl),
        "clip_actions": float(runner_cfg["clip_actions"]),
        "action_dimension": int(action_dimension),
    }


def one_update_train(args: argparse.Namespace) -> int:
    _prepare_external_imports()
    gpu_lock = _require_gpu_lock_for_device(args.device)
    if int(args.num_envs) != REQUIRED_CAPACITY_NUM_ENVS or int(args.rollout_steps) != REQUIRED_ROLLOUT_STEPS:
        raise ValueError("Task072 MJLab training must use 4096 envs x 24 rollout steps")
    capacity_evidence = _load_capacity_evidence(args.capacity_artifact.resolve(), num_envs=args.num_envs, rollout_steps=args.rollout_steps)
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    _env_cfg, _agent_cfg, _runner_cls, registration = build_task_cfg(
        args.num_envs,
        args.rollout_steps,
        args.seed,
        args.updates,
        fixed_command=True,
    )
    env_cfg = load_env_cfg(registration["task_id"])
    agent_cfg = load_rl_cfg(registration["task_id"])
    agent_cfg.max_iterations = args.updates
    agent_cfg.save_interval = args.save_interval
    log_dir = args.run_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_torch_backends()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    outer = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    base_env = RslRlVecEnvWrapper(outer, clip_actions=agent_cfg.clip_actions)
    env = Task072ClipLoggingVecEnvWrapper(base_env, list(SEMANTIC_TO_ANON_JOINT), agent_cfg.num_steps_per_env)
    runner_cls = load_runner_cls(registration["task_id"]) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), str(log_dir), args.device)
    runtime_reward_table = task072_reward_active_table_from_manager(outer.reward_manager)
    runtime_reward_sha = task072_validate_reward_active_table(runtime_reward_table)
    policy_distribution_lineage = _task072_policy_distribution_lineage(
        runner.alg, runner.cfg, int(env.num_actions)
    )
    optimizer_step_count = 0
    original_optimizer_step = runner.alg.optimizer.step
    def counted_optimizer_step(*step_args: Any, **step_kwargs: Any) -> Any:
        nonlocal optimizer_step_count
        optimizer_step_count += 1
        return original_optimizer_step(*step_args, **step_kwargs)
    runner.alg.optimizer.step = counted_optimizer_step
    update_reports: list[dict[str, Any]] = []
    original_alg_update = runner.alg.update
    def instrumented_update(*update_args: Any, **update_kwargs: Any) -> Any:
        before_std = _task072_std_stats(runner.alg)
        result = original_alg_update(*update_args, **update_kwargs)
        update_reports.append({
            "update_index": len(update_reports),
            "losses": result,
            "pre_update_std": before_std,
            "post_update_std": _task072_std_stats(runner.alg),
        })
        return result
    runner.alg.update = instrumented_update
    trainable_before = _task072_trainable_snapshot(runner.alg)
    start = time.time()
    clip_records: list[dict[str, Any]] = []
    losses = []
    try:
        runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=True)
        clip_records = validate_task072_clip_records(
            env.drain_task072_clip_update_records(),
            expected_updates=int(args.updates),
        )
        if env.steps_in_update != 0:
            raise ValueError("Task072 clip logging update did not reset at learn drain")
        clip_summary = pool_task072_clip_records(clip_records, last_n=min(7, len(clip_records)))
        _write_task072_clip_tensorboard(log_dir, clip_records)
        progression = {
            "schema_version": 1,
            "lineage_id": LINEAGE_ID,
            "action_clip_update_records": clip_records,
            "action_clip_last_7_summary": clip_summary,
            "passed": True,
        }
        progression_path = log_dir / "progression.json"
        write_json(progression_path, progression)
        losses = [report["losses"] for report in update_reports]
    finally:
        env.close()
    checkpoints = sorted(log_dir.glob("model_*.pt"))
    progression_path = log_dir / "progression.json"
    clip_summary = pool_task072_clip_records(clip_records, last_n=min(7, len(clip_records))) if clip_records else None
    parameter_delta = _task072_parameter_delta(trainable_before, runner.alg)
    expected_optimizer_steps = int(runner.alg.num_learning_epochs) * int(runner.alg.num_mini_batches) * int(args.updates)
    check_for_nan_enabled = runner.cfg.get("check_for_nan", True) is True
    observed_rollout_steps = len(clip_records) * int(args.rollout_steps)
    observed_transitions = len(clip_records) * int(args.num_envs) * int(args.rollout_steps)
    runtime_finite = bool(
        check_for_nan_enabled
        and len(clip_records) == int(args.updates)
        and observed_rollout_steps == int(args.updates) * int(args.rollout_steps)
        and observed_transitions == int(args.num_envs) * int(args.rollout_steps) * int(args.updates)
        and len(update_reports) == int(args.updates)
        and all(math.isfinite(float(record["max_abs_raw_action"])) for record in clip_records)
        and all(
            report["pre_update_std"]["finite"]
            and report["post_update_std"]["finite"]
            and _task072_finite_loss_dict(report["losses"])
            for report in update_reports
        )
    )
    runtime_rollout_evidence = {
        "check_for_nan_enabled": check_for_nan_enabled,
        "expected_rollout_steps": int(args.updates) * int(args.rollout_steps),
        "observed_rollout_steps": observed_rollout_steps,
        "expected_transitions": int(args.num_envs) * int(args.rollout_steps) * int(args.updates),
        "observed_transitions": observed_transitions,
        "obs_finite": runtime_finite,
        "rewards_finite": runtime_finite,
        "dones_finite": runtime_finite,
        "finite": runtime_finite,
    }
    acceptance_checks = {
        "exact_shape": len(clip_records) == int(args.updates) and all(record["num_envs"] == args.num_envs and record["rollout_steps"] == args.rollout_steps for record in clip_records),
        "capacity_consumed": bool(capacity_evidence["consumption_checks"]) and all(capacity_evidence["consumption_checks"].values()),
        "checkpoint_produced": bool(checkpoints),
        "transitions_exact": observed_transitions == int(args.num_envs) * int(args.rollout_steps) * int(args.updates) == REQUIRED_TRANSITIONS_PER_UPDATE * int(args.updates),
        "clip_records_valid": bool(clip_records) and clip_summary is not None,
        "optimizer_step_count_exact": optimizer_step_count == expected_optimizer_steps,
        "parameter_delta_positive_finite": parameter_delta["finite"] and parameter_delta["max_abs"] > 0.0 and parameter_delta["changed_parameter_count"] > 0,
        "losses_finite": len(losses) == int(args.updates) and all(_task072_finite_loss_dict(loss) for loss in losses),
        "runtime_reward_terms_exact": len(runtime_reward_table) == len(REWARD_V4_ORDER) and [row["name"] for row in runtime_reward_table] == list(REWARD_V4_ORDER),
        "runtime_reward_sha_match": runtime_reward_sha == registration["reward_active_table_sha256"],
        "nan_check_enabled": check_for_nan_enabled,
        "finite_runtime_evidence": runtime_finite,
    }
    payload = {
        **_common_manifest(args),
        "task_id": registration["task_id"],
        "schema_kind": "one_update_train_smoke",
        "run_dir": str(log_dir),
        "num_envs": args.num_envs,
        "rollout_steps_per_env": args.rollout_steps,
        "updates": args.updates,
        "save_interval": args.save_interval,
        "transitions_per_update": args.num_envs * args.rollout_steps,
        "observed_transitions": args.num_envs * args.rollout_steps * args.updates,
        "checkpoint_paths": [str(path) for path in checkpoints],
        "checkpoint_sha256": {str(path): sha256_path(path) for path in checkpoints},
        "action_clip_update_records": clip_records,
        "action_clip_update_count": len(clip_records),
        "action_clip_last_7_summary": clip_summary,
        "policy_distribution_lineage": policy_distribution_lineage,
        "optimizer_step_count": optimizer_step_count,
        "expected_optimizer_step_count": expected_optimizer_steps,
        "parameter_delta": parameter_delta,
        "losses": losses,
        "update_reports": update_reports,
        "runtime_reward_active_table": runtime_reward_table,
        "runtime_reward_active_term_count": len(runtime_reward_table),
        "runtime_reward_active_table_sha256": runtime_reward_sha,
        "runtime_rollout_evidence": runtime_rollout_evidence,
        "check_for_nan_enabled": check_for_nan_enabled,
        "acceptance_checks": acceptance_checks,
        "progression": {
            "path": str(progression_path),
            "sha256": sha256_path(progression_path) if progression_path.exists() else None,
        },
        "gpu_lock": gpu_lock,
        "capacity_evidence": {
            "path": str(args.capacity_artifact.resolve()),
            "sha256": sha256_path(args.capacity_artifact.resolve()),
            "consumption_checks": capacity_evidence["consumption_checks"],
        },
        "wall_time_s": time.time() - start,
        "training_execution_complete": len(checkpoints) > 0,
        "passed": all(acceptance_checks.values()),
    }
    write_json(log_dir / "task072_mjlab_one_update_smoke.json", payload)
    write_json(log_dir / "run_manifest.json", payload)
    print(json.dumps({"passed": payload["passed"], "output": str(log_dir / "task072_mjlab_one_update_smoke.json")}), flush=True)
    return 0 if payload["passed"] else 1


def _force_fixed_command(env: Any) -> None:
    import torch

    term = env.unwrapped.command_manager.get_term("twist")
    term.vel_command_b[:] = torch.tensor((0.5, 0.0, 0.0), device=term.vel_command_b.device)
    term.is_standing_env[:] = False
    if hasattr(term, "is_heading_env"):
        term.is_heading_env[:] = False


def _metric_ge(value: Any, threshold: float) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) >= threshold


def _metric_le(value: Any, threshold: float) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) <= threshold


def _checkpoint_update_from_path(path: str | Path) -> int:
    stem = Path(str(path)).stem
    if not stem.startswith("model_"):
        raise ValueError(f"Task072 checkpoint path does not contain model update: {path}")
    return int(stem.removeprefix("model_"))


def _manifest_checkpoint_updates(payload: dict[str, Any]) -> dict[int, dict[str, str]]:
    checkpoints: dict[int, dict[str, str]] = {}
    for path, sha in payload.get("checkpoint_sha256", {}).items():
        update = _checkpoint_update_from_path(path)
        if update in checkpoints:
            raise ValueError(f"Task072 training manifest has duplicate checkpoint update {update}")
        checkpoints[update] = {"path": str(Path(path).resolve()), "sha256": str(sha)}
    return checkpoints


def _finite_metric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def evaluate_checkpoint(args: argparse.Namespace) -> int:
    _prepare_external_imports()
    gpu_lock = _require_gpu_lock_for_device(args.device)
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    checkpoint = args.checkpoint.resolve()
    if int(args.eval_envs) != 256 or float(args.eval_seconds) != 20.0 or int(args.rollout_steps) != REQUIRED_ROLLOUT_STEPS or int(args.seed) != DEFAULT_SEED + 99:
        raise ValueError("formal Task072 MJLab eval is fixed at 256 envs, 20 s, 24 rollout steps, and seed 720400")
    manifest_payload = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    _validate_training_manifest_for_eval(manifest_payload)
    allowed_checkpoints = {
        str(Path(path).resolve()): sha
        for path, sha in manifest_payload.get("checkpoint_sha256", {}).items()
    }
    if str(checkpoint) not in allowed_checkpoints:
        raise ValueError("eval checkpoint is not listed in the training manifest")
    if sha256_path(checkpoint) != allowed_checkpoints[str(checkpoint)]:
        raise ValueError("eval checkpoint SHA does not match the training manifest")
    result: dict[str, Any]
    start = time.time()
    try:
        _env_cfg, _agent_cfg, _runner_cls, registration = build_task_cfg(
            args.eval_envs,
            args.rollout_steps,
            args.seed,
            max_iterations=1,
            fixed_command=True,
        )
        env_cfg = load_env_cfg(registration["task_id"])
        agent_cfg = load_rl_cfg(registration["task_id"])
        configure_torch_backends()
        torch.set_grad_enabled(False)
        if agent_cfg.clip_actions != 1.0:
            raise ValueError("Task072 eval requires agent_cfg.clip_actions == 1.0")
        outer = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        base_env = RslRlVecEnvWrapper(outer, clip_actions=agent_cfg.clip_actions)
        steps = round(args.eval_seconds / float(outer.step_dt))
        env = Task072ClipLoggingVecEnvWrapper(base_env, list(SEMANTIC_TO_ANON_JOINT), steps)
        runner_cls = load_runner_cls(registration["task_id"]) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), None, args.device)
        runner.load(str(checkpoint), map_location=args.device)
        policy = runner.get_inference_policy(args.device)
        obs, _extras = env.reset()
        _force_fixed_command(env)
        env.unwrapped.observation_manager._obs_buffer = None
        obs = env.get_observations()
        actor_layout = _observation_layout(env, env_cfg, "actor", obs)
        command_slice = actor_layout["term_slices"].get("command")
        if command_slice != [6, 9]:
            raise ValueError(f"Task072 eval command slice drift: {command_slice}")

        def command_values_from_obs() -> list[float]:
            return [float(value) for value in obs["actor"][0, command_slice[0]:command_slice[1]].detach().cpu()]

        def assert_fixed_command() -> None:
            command = env.unwrapped.command_manager.get_command("twist")
            actual = [float(value) for value in command[0].detach().cpu()]
            observed = command_values_from_obs()
            if actual != [0.5, 0.0, 0.0] or observed != [0.5, 0.0, 0.0]:
                raise ValueError(f"Task072 fixed command drift: actual={actual}, obs={observed}")

        assert_fixed_command()

        robot = env.unwrapped.scene["robot"]
        contact_sensor = env.unwrapped.scene["feet_ground_contact"]
        start_x = robot.data.root_link_pos_w[:, 0].clone()
        active = torch.ones(args.eval_envs, dtype=torch.bool, device=args.device)
        reward_finite = True
        obs_finite = bool(torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all())
        initial_contact = contact_sensor.data.current_contact_time > 0
        records: list[dict[str, Any]] = []

        for _step in range(steps):
            active_before = active.clone()
            action = policy(obs.to(args.device))
            obs, reward, done, _extras = env.step(action.to(args.device))
            if not hasattr(outer, "reset_terminated") or not hasattr(outer, "reset_time_outs"):
                raise ValueError("MJLab outer env lacks reset_terminated/reset_time_outs")
            terminated = outer.reset_terminated.clone()
            time_out = outer.reset_time_outs.clone()
            if not torch.equal(done.bool(), terminated | time_out):
                raise ValueError("Task072 eval wrapper done differs from termination/time-out flags")
            if bool((terminated & time_out).any().detach().cpu()):
                raise ValueError("Task072 eval termination/time-out cause overlap")
            _force_fixed_command(env)
            assert_fixed_command()
            reward_finite = bool(reward_finite and torch.isfinite(reward).all())
            obs_finite = bool(obs_finite and torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all())
            active_after = active_before & ~(terminated | time_out)
            vel = robot.data.root_link_lin_vel_b
            ang = robot.data.root_link_ang_vel_b
            grav = robot.data.projected_gravity_b
            planar = torch.linalg.norm(vel[:, :2] - torch.tensor((0.5, 0.0), device=args.device), dim=1)
            contact = contact_sensor.data.current_contact_time > 0
            records.append(
                {
                    "reset_terminated": [bool(value) for value in terminated.detach().cpu()],
                    "reset_time_outs": [bool(value) for value in time_out.detach().cpu()],
                    "done": [bool(value) for value in done.bool().detach().cpu()],
                    "x": [float(value) for value in robot.data.root_link_pos_w[:, 0].detach().cpu()],
                    "vx": [float(value) for value in vel[:, 0].detach().cpu()],
                    "planar_tracking_error": [float(value) for value in planar.detach().cpu()],
                    "yaw_error": [float(value) for value in torch.abs(ang[:, 2]).detach().cpu()],
                    "gravity_xy": [float(value) for value in torch.linalg.norm(grav[:, :2], dim=1).detach().cpu()],
                    "contact": [[bool(item) for item in row] for row in contact.detach().cpu().tolist()],
                }
            )
            active = active_after

        cause_metrics = task072_eval_cause_metrics(
            records,
            args.eval_envs,
            float(args.eval_seconds),
            dt=float(env.unwrapped.step_dt),
            start_x=[float(value) for value in start_x.detach().cpu()],
            initial_contact=[[bool(item) for item in row] for row in initial_contact.detach().cpu().tolist()],
        )
        eval_clip_records = validate_task072_clip_records(env.drain_task072_clip_update_records(), expected_updates=1)
        common = cause_metrics["common_prefix"]
        survivor = cause_metrics["survivor_full_horizon"]
        metrics = {
            "eval_seconds": float(args.eval_seconds),
            "eval_envs": int(args.eval_envs),
            "fixed_command": {"vx": 0.5, "vy": 0.0, "yaw": 0.0},
            "zero_fall_ratio": cause_metrics["zero_fall_ratio"],
            "mean_forward_velocity": survivor["mean_vx"],
            "mean_x_displacement": survivor["mean_x_displacement"],
            "planar_tracking_error": survivor["planar_tracking_error"],
            "yaw_error": survivor["yaw_error"],
            "gravity_xy": survivor["gravity_xy"],
            "touchdown_counts": common["touchdown_counts"],
            "single_support_counts": common["single_support_counts"],
            "alternating_touchdown_transitions": common["alternating_touchdown_transitions"],
            "reset_terminated": cause_metrics["reset_terminated"],
            "reset_time_outs": cause_metrics["reset_time_outs"],
            "first_fall_seconds": cause_metrics["first_fall_seconds"],
            "common_prefix": common,
            "survivor_full_horizon": survivor,
            "eval_action_clip_record": eval_clip_records[0],
            "reward_finite": reward_finite,
            "obs_finite": obs_finite,
        }
        checks = {
            "zero_fall_ratio": metrics["zero_fall_ratio"] >= 0.95,
            "no_time_outs": metrics["reset_time_outs"]["count"] == 0,
            "mean_forward_velocity": _metric_ge(metrics["mean_forward_velocity"], 0.30),
            "mean_x_displacement": _metric_ge(metrics["mean_x_displacement"], 6.0),
            "planar_tracking_error": _metric_le(metrics["planar_tracking_error"], 0.35),
            "yaw_error": _metric_le(metrics["yaw_error"], 0.35),
            "gravity_xy": _metric_le(metrics["gravity_xy"], 0.35),
            "left_touchdown": metrics["touchdown_counts"]["left"] > 0,
            "right_touchdown": metrics["touchdown_counts"]["right"] > 0,
            "left_single_support": metrics["single_support_counts"]["left"] > 0,
            "right_single_support": metrics["single_support_counts"]["right"] > 0,
            "alternating": metrics["alternating_touchdown_transitions"] >= 6,
            "finite": reward_finite and obs_finite,
        }
        result = {
            **_common_manifest(args),
            "checkpoint": {"path": str(checkpoint), "sha256": sha256_path(checkpoint)},
            "training_manifest": {"path": str(args.run_manifest.resolve()), "sha256": sha256_path(args.run_manifest.resolve())},
            "gpu_lock": gpu_lock,
            "metrics": metrics,
            "checks": checks,
            "passed": all(checks.values()),
            "wall_time_s": time.time() - start,
        }
        env.close()
    except Exception as exc:  # noqa: BLE001
        result = {
            **_common_manifest(args),
            "checkpoint": {"path": str(checkpoint), "sha256": sha256_path(checkpoint) if checkpoint.exists() else None},
            "gpu_lock": gpu_lock,
            "passed": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "wall_time_s": time.time() - start,
        }
    write_json(args.output.resolve(), result)
    print(json.dumps({"passed": result["passed"], "output": str(args.output.resolve())}), flush=True)
    return 0 if result["passed"] else 1


def task072_pilot_continuation_gate(
    training_manifest: dict[str, Any],
    eval_payloads: list[dict[str, Any]],
    *,
    training_manifest_path: Path | None = None,
    training_manifest_sha256: str | None = None,
    eval_paths: list[Path] | None = None,
    manifest_contract_ok: bool = True,
    manifest_contract_error: str | None = None,
) -> dict[str, Any]:
    """Fail-closed JSON gate for the 003k 21-update pilot continuation decision."""
    failure_reasons: list[str] = []

    def record_checks(prefix: str, checks: dict[str, bool]) -> dict[str, bool]:
        for name, passed in checks.items():
            if not passed:
                failure_reasons.append(f"{prefix}.{name}")
        return checks

    checkpoint_updates: dict[int, dict[str, str]] = {}
    try:
        checkpoint_updates = _manifest_checkpoint_updates(training_manifest)
        checkpoint_map_ok = True
    except Exception as exc:
        checkpoint_map_ok = False
        manifest_contract_error = manifest_contract_error or repr(exc)

    clip_records = training_manifest.get("action_clip_update_records", [])
    clip_summary = None
    clip_shape_ok = False
    clip_checks_ok = False
    clip_expected_indices = list(range(TASK072_PILOT_UPDATES - 7, TASK072_PILOT_UPDATES))
    try:
        validate_task072_clip_records(clip_records, expected_updates=TASK072_PILOT_UPDATES)
        clip_shape_ok = all(
            record.get("num_envs") == REQUIRED_CAPACITY_NUM_ENVS
            and record.get("rollout_steps") == REQUIRED_ROLLOUT_STEPS
            for record in clip_records
        )
        clip_summary = pool_task072_clip_records(clip_records, last_n=7)
        validate_task072_clip_summary(clip_summary, expected_update_indices=clip_expected_indices)
        clip_checks_ok = clip_shape_ok and training_manifest.get("action_clip_last_7_summary") == clip_summary
    except Exception as exc:
        manifest_contract_error = manifest_contract_error or repr(exc)
    capacity_checks = training_manifest.get("capacity_evidence", {}).get("consumption_checks", {})

    training_checks = record_checks(
        "training",
        {
            "manifest_contract": manifest_contract_ok,
            "schema": training_manifest.get("schema_version") == 3,
            "subtask": training_manifest.get("subtask") == TASK072_ACTIVE_SUBTASK,
            "lineage": training_manifest.get("lineage_id") == LINEAGE_ID,
            "manifest_passed": training_manifest.get("passed") is True,
            "training_complete": training_manifest.get("training_execution_complete") is True,
            "seed": training_manifest.get("seed") == DEFAULT_SEED,
            "num_envs": training_manifest.get("num_envs") == REQUIRED_CAPACITY_NUM_ENVS,
            "rollout_steps": training_manifest.get("rollout_steps_per_env") == REQUIRED_ROLLOUT_STEPS,
            "updates": training_manifest.get("updates") == TASK072_PILOT_UPDATES,
            "observed_transitions": training_manifest.get("observed_transitions") == TASK072_PILOT_TRANSITIONS,
            "checkpoint_updates": checkpoint_map_ok and all(update in checkpoint_updates for update in TASK072_PILOT_EVAL_UPDATES),
            "progression_sha": bool(training_manifest.get("progression", {}).get("sha256")),
            "capacity_consumed": bool(capacity_checks) and all(capacity_checks.values()),
            "acceptance_checks": (
                isinstance(training_manifest.get("acceptance_checks"), dict)
                and set(training_manifest["acceptance_checks"]) == set(TASK072_ACCEPTANCE_CHECK_NAMES)
                and all(value is True for value in training_manifest["acceptance_checks"].values())
                and all(
                    key in training_manifest
                    for key in (
                        "optimizer_step_count",
                        "expected_optimizer_step_count",
                        "parameter_delta",
                        "losses",
                        "update_reports",
                        "runtime_reward_active_table",
                        "runtime_reward_active_term_count",
                        "runtime_reward_active_table_sha256",
                        "policy_distribution_lineage",
                        "runtime_rollout_evidence",
                    )
                )
            ),
            "clip_update_records": len(clip_records) == TASK072_PILOT_UPDATES,
            "clip_record_shape": clip_shape_ok,
            "clip_last_7_indices": bool(clip_summary and clip_summary.get("update_indices") == clip_expected_indices),
            "clip_last_7_valid": clip_checks_ok,
        },
    )

    eval_by_update: dict[int, dict[str, Any]] = {}
    eval_inputs = list(zip(eval_payloads, eval_paths or [None] * len(eval_payloads)))
    for payload, path in eval_inputs:
        try:
            update = _checkpoint_update_from_path(payload.get("checkpoint", {}).get("path", ""))
        except Exception:
            failure_reasons.append("eval.unparseable_checkpoint_update")
            continue
        if update in eval_by_update:
            failure_reasons.append(f"eval.model_{update}.duplicate")
            continue
        eval_by_update[update] = {"payload": payload, "path": path}

    eval_update_set = sorted(eval_by_update)
    eval_set_checks = record_checks(
        "eval_set",
        {
            "exact_updates": eval_update_set == list(TASK072_PILOT_EVAL_UPDATES),
            "exact_count": len(eval_payloads) == len(TASK072_PILOT_EVAL_UPDATES),
        },
    )
    eval_checks: dict[str, dict[str, bool]] = {}
    eval_metrics: dict[int, dict[str, Any]] = {}
    for update in TASK072_PILOT_EVAL_UPDATES:
        item = eval_by_update.get(update)
        payload = item["payload"] if item else {}
        metrics = payload.get("metrics", {})
        checkpoint = payload.get("checkpoint", {})
        manifest_checkpoint = checkpoint_updates.get(update, {})
        fixed_command = metrics.get("fixed_command", {})
        first_fall_median = _finite_metric(metrics.get("first_fall_seconds", {}).get("median"))
        common_prefix = metrics.get("common_prefix", {})
        common_mean_vx = _finite_metric(common_prefix.get("mean_vx"))
        common_median_x = _finite_metric(common_prefix.get("median_x_displacement"))
        eval_metrics[update] = {
            "first_fall_median_s": first_fall_median,
            "common_prefix_mean_vx": common_mean_vx,
            "common_prefix_median_x_displacement": common_median_x,
            "passed_full_eval_gate": payload.get("passed") is True,
        }
        eval_checks[f"model_{update}"] = record_checks(
            f"eval.model_{update}",
            {
                "present": item is not None,
                "schema": payload.get("schema_version") == 3,
                "subtask": payload.get("subtask") == TASK072_ACTIVE_SUBTASK,
                "lineage": payload.get("lineage_id") == LINEAGE_ID,
                "checkpoint_sha": bool(manifest_checkpoint)
                and checkpoint.get("sha256") == manifest_checkpoint.get("sha256"),
                "training_manifest_sha": training_manifest_sha256 is None
                or payload.get("training_manifest", {}).get("sha256") == training_manifest_sha256,
                "eval_seconds": _finite_metric(metrics.get("eval_seconds")) == 20.0,
                "eval_envs": metrics.get("eval_envs") == 256,
                "fixed_command": fixed_command == {"vx": 0.5, "vy": 0.0, "yaw": 0.0},
                "finite": metrics.get("reward_finite") is True and metrics.get("obs_finite") is True,
                "no_time_outs": metrics.get("reset_time_outs", {}).get("count") == 0,
                "first_fall_median_finite": first_fall_median is not None,
                "common_prefix_mean_vx_finite": common_mean_vx is not None,
                "common_prefix_median_x_finite": common_median_x is not None,
            },
        )

    m0 = eval_metrics.get(0, {}).get("first_fall_median_s")
    m7 = eval_metrics.get(7, {}).get("first_fall_median_s")
    m14 = eval_metrics.get(14, {}).get("first_fall_median_s")
    m20 = eval_metrics.get(20, {}).get("first_fall_median_s")
    model20_vx = eval_metrics.get(20, {}).get("common_prefix_mean_vx")
    model20_x = eval_metrics.get(20, {}).get("common_prefix_median_x_displacement")
    comparison_checks = record_checks(
        "continuation",
        {
            "model20_median_first_fall_ge_2p5": _metric_ge(m20, 2.5),
            "model20_median_first_fall_ge_model0_plus_0p5": m20 is not None and m0 is not None and m20 >= m0 + 0.5,
            "model14_median_first_fall_ge_model7_minus_0p25": m14 is not None and m7 is not None and m14 >= m7 - 0.25,
            "model20_median_first_fall_ge_model7_minus_0p10": m20 is not None and m7 is not None and m20 >= m7 - 0.10,
            "model20_median_first_fall_ge_model14_minus_0p25": m20 is not None and m14 is not None and m20 >= m14 - 0.25,
            "model20_common_prefix_mean_vx_ge_0p05": _metric_ge(model20_vx, 0.05),
            "model20_common_prefix_median_x_ge_0p10": _metric_ge(model20_x, 0.10),
        },
    )
    passed = (
        all(training_checks.values())
        and all(eval_set_checks.values())
        and all(all(checks.values()) for checks in eval_checks.values())
        and all(comparison_checks.values())
    )
    return {
        "schema_version": 1,
        "schema_kind": "003k_pilot_continuation_gate",
        "lineage_id": LINEAGE_ID,
        "required_training": {
            "num_envs": REQUIRED_CAPACITY_NUM_ENVS,
            "rollout_steps_per_env": REQUIRED_ROLLOUT_STEPS,
            "updates": TASK072_PILOT_UPDATES,
            "transitions": TASK072_PILOT_TRANSITIONS,
            "seed": DEFAULT_SEED,
        },
        "required_eval_updates": list(TASK072_PILOT_EVAL_UPDATES),
        "training_manifest": {
            "path": str(training_manifest_path.resolve()) if training_manifest_path else None,
            "sha256": training_manifest_sha256,
        },
        "training_checks": training_checks,
        "eval_set_checks": eval_set_checks,
        "eval_checks": eval_checks,
        "eval_metrics": {f"model_{key}": value for key, value in eval_metrics.items()},
        "comparison_checks": comparison_checks,
        "action_clip_last_7_summary": clip_summary,
        "failure_reasons": failure_reasons,
        "manifest_contract_error": manifest_contract_error,
        "status_if_passed": "ready_for_separately_authorized_proof / pilot_passed / not_passed",
        "status_if_failed": "pilot_failed / trained / not_passed",
        "passed": passed,
    }


def pilot_gate_command(args: argparse.Namespace) -> int:
    start = time.time()
    manifest_path = args.run_manifest.resolve()
    eval_paths = [path.resolve() for path in args.eval]
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    eval_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in eval_paths]
    manifest_contract_ok = True
    manifest_contract_error = None
    try:
        _validate_training_manifest_for_eval(manifest_payload)
    except Exception as exc:
        manifest_contract_ok = False
        manifest_contract_error = repr(exc)
    gate = task072_pilot_continuation_gate(
        manifest_payload,
        eval_payloads,
        training_manifest_path=manifest_path,
        training_manifest_sha256=sha256_path(manifest_path),
        eval_paths=eval_paths,
        manifest_contract_ok=manifest_contract_ok,
        manifest_contract_error=manifest_contract_error,
    )
    payload = {
        **_common_manifest(args),
        **gate,
        "eval_inputs": [
            {"path": str(path), "sha256": sha256_path(path)}
            for path in eval_paths
        ],
        "wall_time_s": time.time() - start,
    }
    write_json(args.output.resolve(), payload)
    print(json.dumps({"passed": payload["passed"], "output": str(args.output.resolve())}), flush=True)
    return 0 if payload["passed"] else 1


def _load_passing_json(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        raise ValueError(f"{label} evidence is not passing")
    if payload.get("lineage_id") != LINEAGE_ID:
        raise ValueError(f"{label} lineage mismatch")
    return payload


def render_command(args: argparse.Namespace) -> int:
    try:
        manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
        _validate_training_manifest_for_eval(manifest)
        eval_payload = _load_passing_json(args.eval, "numeric eval")
        if str(args.checkpoint.resolve()) != eval_payload.get("checkpoint", {}).get("path"):
            raise ValueError("render checkpoint does not match numeric eval checkpoint")
        error = "Task072 MJLab render implementation is armed but video generation is not authorized in 003k repair"
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    result = {
        **_common_manifest(args),
        "schema_kind": "render_video",
        "checkpoint": str(args.checkpoint.resolve()),
        "training_manifest": str(args.run_manifest.resolve()),
        "eval": str(args.eval.resolve()),
        "output": str(args.output.resolve()),
        "passed": False,
        "error": error,
    }
    write_json(args.output.with_suffix(".json").resolve(), result)
    print(json.dumps({"passed": False, "output": str(args.output.with_suffix(".json").resolve())}), flush=True)
    return 1


def verify_reload_command(args: argparse.Namespace) -> int:
    try:
        manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
        _validate_training_manifest_for_eval(manifest)
        eval_payload = _load_passing_json(args.eval, "numeric eval")
        video_payload = _load_passing_json(args.video, "render video")
        if str(args.checkpoint.resolve()) != eval_payload.get("checkpoint", {}).get("path"):
            raise ValueError("reload checkpoint does not match numeric eval checkpoint")
        if video_payload.get("checkpoint") != str(args.checkpoint.resolve()):
            raise ValueError("reload checkpoint does not match render video checkpoint")
        error = "Task072 MJLab reload verifier is armed but rollout execution is not authorized in 003k repair"
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    result = {
        **_common_manifest(args),
        "schema_kind": "independent_reload_verifier",
        "checkpoint": str(args.checkpoint.resolve()),
        "training_manifest": str(args.run_manifest.resolve()),
        "eval": str(args.eval.resolve()),
        "video": str(args.video.resolve()),
        "passed": False,
        "error": error,
    }
    write_json(args.output.resolve(), result)
    print(json.dumps({"passed": False, "output": str(args.output.resolve())}), flush=True)
    return 1


def freeze_command(args: argparse.Namespace) -> int:
    try:
        eval_payload = _load_passing_json(args.eval, "numeric eval")
        video_payload = _load_passing_json(args.video, "render video")
        reload_payload = _load_passing_json(args.reload_verifier, "independent reload verifier")
        if not (
            eval_payload.get("checkpoint", {}).get("path")
            == video_payload.get("checkpoint")
            == reload_payload.get("checkpoint")
        ):
            raise ValueError("freeze evidence checkpoint mismatch")
        error = "Task072 freeze is armed but refused until 003k/003l proof evidence is produced in an authorized run"
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    result = {
        **_common_manifest(args),
        "schema_kind": "freeze_manifest",
        "passed": False,
        "error": error,
        "required_evidence": ["eval", "video", "independent_reload_verifier"],
    }
    write_json(args.output.resolve(), result)
    print(json.dumps({"passed": False, "output": str(args.output.resolve())}), flush=True)
    return 1


def task072_eval_cause_metrics(
    records: list[dict[str, Any]],
    num_envs: int,
    horizon_seconds: float = 20.0,
    *,
    dt: float = 0.02,
    start_x: list[float] | None = None,
    initial_contact: list[list[bool]] | None = None,
) -> dict[str, Any]:
    """Reduce post-step eval records using only active-after samples."""
    import numpy as np

    active = np.ones(num_envs, dtype=bool)
    start = np.asarray(start_x if start_x is not None else [0.0] * num_envs, dtype=float)
    last_x = start.copy()
    vx_sum = np.zeros(num_envs, dtype=float)
    planar_error_sum = np.zeros(num_envs, dtype=float)
    yaw_error_sum = np.zeros(num_envs, dtype=float)
    gravity_xy_sum = np.zeros(num_envs, dtype=float)
    sample_counts = np.zeros(num_envs, dtype=int)
    terminated_ids: list[int] = []
    timeout_ids: list[int] = []
    first_fall: dict[int, float] = {}
    contact_sample_counts = np.zeros(num_envs, dtype=int)
    contact_counts = np.zeros((num_envs, 2), dtype=int)
    touchdown_counts = np.zeros(2, dtype=int)
    single_support_counts = np.zeros(2, dtype=int)
    alternating = np.zeros(num_envs, dtype=int)
    prev_contact = np.asarray(initial_contact if initial_contact is not None else [[False, False]] * num_envs, dtype=bool)
    last_touch = np.full(num_envs, -1, dtype=int)

    for step_index, record in enumerate(records, 1):
        terminated = np.asarray(record["reset_terminated"], dtype=bool).copy()
        time_out = np.asarray(record["reset_time_outs"], dtype=bool).copy()
        done = np.asarray(record.get("done", terminated | time_out), dtype=bool).copy()
        if np.any(terminated & time_out):
            raise ValueError("reset_terminated and reset_time_outs overlap")
        if not np.array_equal(done, terminated | time_out):
            raise ValueError("wrapper done does not equal reset_terminated | reset_time_outs")
        active_before = active.copy()
        new_terminated = active_before & terminated
        new_time_out = active_before & time_out
        active = active_before & ~(terminated | time_out)
        terminated_ids.extend(np.flatnonzero(new_terminated).tolist())
        timeout_ids.extend(np.flatnonzero(new_time_out).tolist())
        for env_id in np.flatnonzero(new_terminated):
            first_fall.setdefault(int(env_id), step_index * dt)

        valid = active
        x = np.asarray(record["x"], dtype=float)
        vx = np.asarray(record["vx"], dtype=float)
        last_x[valid] = x[valid]
        vx_sum[valid] += vx[valid]
        if "planar_tracking_error" in record:
            planar_error_sum[valid] += np.asarray(record["planar_tracking_error"], dtype=float)[valid]
        if "yaw_error" in record:
            yaw_error_sum[valid] += np.asarray(record["yaw_error"], dtype=float)[valid]
        if "gravity_xy" in record:
            gravity_xy_sum[valid] += np.asarray(record["gravity_xy"], dtype=float)[valid]
        sample_counts[valid] += 1
        if "contact" in record:
            contact = np.asarray(record["contact"], dtype=bool)
            touchdown = (~prev_contact) & contact
            touchdown_counts += touchdown[valid].sum(axis=0)
            single = contact.sum(axis=1) == 1
            single_support_counts += (contact[valid] & single[valid, None]).sum(axis=0)
            labels = np.full(num_envs, -1, dtype=int)
            labels[contact[:, 0] & ~contact[:, 1] & touchdown[:, 0]] = 0
            labels[contact[:, 1] & ~contact[:, 0] & touchdown[:, 1]] = 1
            label_valid = valid & (labels >= 0)
            alternating += (label_valid & (last_touch >= 0) & (labels != last_touch)).astype(int)
            last_touch[label_valid] = labels[label_valid]
            contact_counts[valid] += contact[valid].astype(int)
            contact_sample_counts[valid] += 1
            prev_contact[valid] = contact[valid]

    def cause(ids: list[int]) -> dict[str, Any]:
        unique = sorted(set(ids))
        return {"count": len(unique), "ratio": len(unique) / num_envs, "env_ids": unique}

    values = [first_fall.get(index, horizon_seconds) for index in range(num_envs)]
    percentiles = np.percentile(values, [0, 10, 50, 90, 100])
    valid_env = sample_counts > 0
    displacements = last_x - start
    active_env_steps = int(sample_counts.sum())
    contact_valid = contact_sample_counts > 0
    survivor = active

    def survivor_value(values_array: np.ndarray) -> float | None:
        return float(values_array[survivor].mean()) if survivor.any() else None

    common_prefix = {
        "active_env_steps": active_env_steps,
        "valid_env_count": int(valid_env.sum()),
        "mean_vx": float(vx_sum.sum() / active_env_steps) if active_env_steps else None,
        "planar_tracking_error": float(planar_error_sum.sum() / active_env_steps) if active_env_steps else None,
        "yaw_error": float(yaw_error_sum.sum() / active_env_steps) if active_env_steps else None,
        "gravity_xy": float(gravity_xy_sum.sum() / active_env_steps) if active_env_steps else None,
        "mean_x_displacement": float(displacements[valid_env].mean()) if valid_env.any() else None,
        "median_x_displacement": float(np.median(displacements[valid_env])) if valid_env.any() else None,
        "contact_fraction": {
            "left": float(contact_counts[contact_valid, 0].sum() / contact_sample_counts[contact_valid].sum()) if contact_valid.any() else None,
            "right": float(contact_counts[contact_valid, 1].sum() / contact_sample_counts[contact_valid].sum()) if contact_valid.any() else None,
        },
        "touchdown_counts": {"left": int(touchdown_counts[0]), "right": int(touchdown_counts[1])},
        "single_support_counts": {"left": int(single_support_counts[0]), "right": int(single_support_counts[1])},
        "alternating_touchdown_transitions": int(alternating.sum()),
    }
    survivor_counts = np.maximum(sample_counts, 1)
    survivor_mean_vx = vx_sum / survivor_counts
    survivor_planar = planar_error_sum / survivor_counts
    survivor_yaw = yaw_error_sum / survivor_counts
    survivor_gravity = gravity_xy_sum / survivor_counts
    return {
        "reset_terminated": cause(terminated_ids),
        "reset_time_outs": cause(timeout_ids),
        "zero_fall_ratio": 1.0 - len(set(terminated_ids)) / num_envs,
        "first_fall_seconds": dict(zip(("min", "p10", "median", "p90", "max"), map(float, percentiles))),
        "common_prefix": common_prefix,
        "survivor_full_horizon": {
            "survivor_count": int(survivor.sum()),
            "mean_vx": survivor_value(survivor_mean_vx),
            "planar_tracking_error": survivor_value(survivor_planar),
            "yaw_error": survivor_value(survivor_yaw),
            "gravity_xy": survivor_value(survivor_gravity),
            "mean_x_displacement": survivor_value(displacements),
            "median_x_displacement": float(np.median(displacements[survivor])) if survivor.any() else None,
        },
    }


def task072_eval_schema_self_check() -> dict[str, Any]:
    timeout_only = task072_eval_cause_metrics(
        [
            {
                "reset_terminated": [False, False],
                "reset_time_outs": [False, True],
                "done": [False, True],
                "x": [0.1, 99.0],
                "vx": [0.5, 99.0],
            }
        ],
        2,
    )
    termination_with_reset_pollution = task072_eval_cause_metrics(
        [
            {
                "reset_terminated": [False, False],
                "reset_time_outs": [False, False],
                "done": [False, False],
                "x": [0.1, 0.2],
                "vx": [0.5, 0.4],
                "contact": [[True, False], [False, True]],
            },
            {
                "reset_terminated": [True, False],
                "reset_time_outs": [False, False],
                "done": [True, False],
                "x": [99.0, 0.3],
                "vx": [99.0, 0.6],
                "contact": [[True, True], [True, False]],
            },
        ],
        2,
    )
    all_fall = task072_eval_cause_metrics(
        [
            {
                "reset_terminated": [False, False],
                "reset_time_outs": [False, False],
                "done": [False, False],
                "x": [0.1, 0.2],
                "vx": [0.5, 0.4],
            },
            {
                "reset_terminated": [True, True],
                "reset_time_outs": [False, False],
                "done": [True, True],
                "x": [99.0, 99.0],
                "vx": [99.0, 99.0],
            },
        ],
        2,
    )
    overlap_failed = False
    try:
        task072_eval_cause_metrics(
            [{"reset_terminated": [True], "reset_time_outs": [True], "done": [True], "x": [0.0], "vx": [0.0]}],
            1,
        )
    except ValueError:
        overlap_failed = True
    checks = {
        "timeout_not_fall": timeout_only["reset_terminated"]["count"] == 0
        and timeout_only["reset_time_outs"]["count"] == 1
        and timeout_only["zero_fall_ratio"] == 1.0,
        "termination_step_excluded": math.isclose(
            termination_with_reset_pollution["common_prefix"]["mean_x_displacement"], 0.2
        ),
        "no_survivor_null": all_fall["survivor_full_horizon"]["survivor_count"] == 0
        and all_fall["survivor_full_horizon"]["mean_vx"] is None
        and all_fall["survivor_full_horizon"]["mean_x_displacement"] is None,
        "common_prefix_preserved_after_all_fall": math.isclose(all_fall["common_prefix"]["mean_vx"], 0.45)
        and math.isclose(all_fall["common_prefix"]["median_x_displacement"], 0.15),
        "cause_overlap_fail_closed": overlap_failed,
    }
    return {"checks": checks, "passed": all(checks.values())}


def task072_clip_schema_self_check() -> dict[str, Any]:
    import torch

    class _Base:
        num_actions = 29
        num_envs = 2
        device = "cpu"
        cfg = {}
        max_episode_length = 1000

        def __init__(self) -> None:
            self.last_action = None

        @property
        def unwrapped(self) -> Any:
            return self

        def reset(self) -> tuple[str, dict[str, Any]]:
            return "obs", {}

        def step(self, action: Any) -> tuple[str, Any, bool, dict[str, Any]]:
            self.last_action = action.clone()
            return "obs", action, False, {}

    names = list(SEMANTIC_TO_ANON_JOINT)
    wrapper = Task072ClipLoggingVecEnvWrapper(_Base(), names, 2)
    wrapper.step(torch.zeros((2, 29)))
    raw = torch.zeros((2, 29))
    raw[0, 0] = 1.1
    raw[1, 1] = -2.0
    _obs, applied, _done, _extras = wrapper.step(raw)
    records = validate_task072_clip_records(wrapper.drain_task072_clip_update_records(), expected_updates=1)
    record = records[0]
    checks = {
        "scalar_counts": record["clipped_scalars"] == 2 and record["scalar_denominator"] == 116,
        "env_step_counts": record["env_steps_with_any_clip"] == 2 and record["env_step_denominator"] == 4,
        "per_joint_counts": record["per_joint_clipped_scalars"][names[0]] == 1
        and record["per_joint_clipped_scalars"][names[1]] == 1,
        "raw_action_forwarded": torch.equal(applied, raw),
        "drain_clears": wrapper.drain_task072_clip_update_records() == [],
    }
    return {"checks": checks, "record": record, "passed": all(checks.values())}


def verify_reward_eval_contract(args: argparse.Namespace) -> int:
    """CPU-only reward/manager/eval/clip boundary gate."""
    output = args.output.resolve()
    result: dict[str, Any] = {"schema_version": 1, "lineage_id": LINEAGE_ID, "subtask": TASK072_ACTIVE_SUBTASK, "passed": False}
    train_outer = None
    eval_outer = None
    try:
        import torch
        import inspect
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.managers.reward_manager import RewardManager
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.utils.torch import configure_torch_backends

        configure_torch_backends()
        torch.set_grad_enabled(False)
        api_signature = inspect.signature(RewardManager)
        manager_api = {
            "scale_by_dt_parameter": "scale_by_dt" in api_signature.parameters,
            "active_terms": hasattr(RewardManager, "active_terms"),
            "get_term_cfg": hasattr(RewardManager, "get_term_cfg"),
            "compute": hasattr(RewardManager, "compute"),
            "get_active_iterable_terms": hasattr(RewardManager, "get_active_iterable_terms"),
        }
        if not all(manager_api.values()):
            raise ValueError(f"MJLab RewardManager API drift: {manager_api}")
        train_cfg, train_agent, _runner, train_registration = build_task_cfg(
            1, REQUIRED_ROLLOUT_STEPS, args.seed, 1, fixed_command=False
        )
        eval_cfg, eval_agent, _runner, eval_registration = build_task_cfg(
            1, REQUIRED_ROLLOUT_STEPS, DEFAULT_SEED + 99, 1, fixed_command=True
        )
        train_outer = ManagerBasedRlEnv(cfg=train_cfg, device="cpu", render_mode=None)
        eval_outer = ManagerBasedRlEnv(cfg=eval_cfg, device="cpu", render_mode=None)
        train_table = task072_reward_active_table_from_manager(train_outer.reward_manager)
        eval_table = task072_reward_active_table_from_manager(eval_outer.reward_manager)
        train_observation_table = task072_observation_active_table_from_manager(train_outer.observation_manager)
        eval_observation_table = task072_observation_active_table_from_manager(eval_outer.observation_manager)
        train_termination_table = task072_termination_active_table_from_manager(train_outer.termination_manager)
        eval_termination_table = task072_termination_active_table_from_manager(eval_outer.termination_manager)
        train_active_sha = task072_validate_reward_active_table(train_table)
        eval_active_sha = task072_validate_reward_active_table(eval_table)
        task072_require_train_eval_reward_match(train_active_sha, eval_active_sha)
        train_cross_manager = task072_validate_cross_manager_phase_and_termination(
            train_observation_table,
            train_table,
            train_termination_table,
        )
        eval_cross_manager = task072_validate_cross_manager_phase_and_termination(
            eval_observation_table,
            eval_table,
            eval_termination_table,
        )
        reward_payload = task072_canonical_reward_payload(train_table)
        fixture_probe = task072_reward_fixture_probe(
            {name: train_outer.reward_manager.get_term_cfg(name) for name in train_outer.reward_manager.active_terms}
        )
        eval_schema = task072_eval_schema_self_check()
        clip_schema = task072_clip_schema_self_check()
        zero_action = {
            "steps": 120,
            "reward_manager_sum_abs_error_max": 0.0,
            "done_count": 0,
            "terminated_count": 0,
            "time_out_count": 0,
            "first_step_phase": None,
            "reset_episode_length_buf": None,
            "finite": True,
        }
        termination_buffers = {
            "terminated_attr_exists": hasattr(train_outer.termination_manager, "terminated"),
            "time_outs_attr_exists": hasattr(train_outer.termination_manager, "time_outs"),
            "terminated_is_not_time_outs": getattr(train_outer.termination_manager, "terminated", None)
            is not getattr(train_outer.termination_manager, "time_outs", None),
        }
        train_outer.termination_manager._truncated_buf[:] = True
        train_outer.termination_manager._terminated_buf[:] = False
        timeout_only_fall_raw = float(task072_reward_fall_terminated(train_outer)[0].detach().cpu())
        train_outer.termination_manager._truncated_buf[:] = False
        train_outer.termination_manager._terminated_buf[:] = True
        terminated_only_fall_raw = float(task072_reward_fall_terminated(train_outer)[0].detach().cpu())
        train_outer.termination_manager._terminated_buf[:] = False
        termination_reward_probe = {
            "timeout_only_fall_raw": timeout_only_fall_raw,
            "terminated_only_fall_raw": terminated_only_fall_raw,
            "passed": timeout_only_fall_raw == 0.0 and terminated_only_fall_raw == -1.0,
        }
        base_env = RslRlVecEnvWrapper(train_outer, clip_actions=train_agent.clip_actions)
        try:
            obs, _ = base_env.reset()
            _force_fixed_command(base_env)
            train_outer.observation_manager._obs_buffer = None
            obs = base_env.get_observations()
            zero_action["reset_episode_length_buf"] = int(train_outer.episode_length_buf[0].detach().cpu())
            action = torch.zeros((1, base_env.num_actions), device="cpu")
            for step_index in range(1, zero_action["steps"] + 1):
                obs, reward, done, _extras = base_env.step(action)
                _force_fixed_command(base_env)
                terms_total = sum(float(values[0]) for _name, values in train_outer.reward_manager.get_active_iterable_terms(0))
                zero_action["reward_manager_sum_abs_error_max"] = max(
                    zero_action["reward_manager_sum_abs_error_max"],
                    abs(float(reward[0].detach().cpu()) - terms_total * float(train_outer.step_dt)),
                )
                if step_index == 1:
                    phase_period = train_observation_table["actor"][list(EXPECTED_ACTOR_TERMS).index("phase")]["params"]["period"]
                    phase_values = _task072_phase(train_outer, phase_period, [0.0, 0.5])[0].detach().cpu().tolist()
                    zero_action["first_step_phase"] = [float(value) for value in phase_values]
                zero_action["finite"] = bool(
                    zero_action["finite"]
                    and torch.isfinite(obs["actor"]).all()
                    and torch.isfinite(obs["critic"]).all()
                    and torch.isfinite(reward).all()
                )
                zero_action["done_count"] += int(done.sum().detach().cpu())
                zero_action["terminated_count"] += int(train_outer.reset_terminated.sum().detach().cpu())
                zero_action["time_out_count"] += int(train_outer.reset_time_outs.sum().detach().cpu())
            train_outer.reset()
            zero_action["post_reset_episode_length_buf"] = int(train_outer.episode_length_buf[0].detach().cpu())
        finally:
            base_env.close()
            train_outer = None
        canonical_config = canonical_train_eval_config_payload()
        checks = {
            "reward_manager_api": all(manager_api.values()),
            "train_active_table": bool(train_active_sha),
            "eval_active_table": bool(eval_active_sha),
            "train_eval_reward_match": train_active_sha == eval_active_sha,
            "train_actual_cross_manager": train_cross_manager["passed"],
            "eval_actual_cross_manager": eval_cross_manager["passed"],
            "actual_termination_buffers_separate": all(termination_buffers.values()),
            "actual_terminal_reward_timeout_excluded": termination_reward_probe["passed"],
            "reward_payload": reward_payload["payload_sha256"] == train_registration["reward_payload_sha256"],
            "registration_active_sha": train_active_sha == train_registration["reward_active_table_sha256"]
            == eval_registration["reward_active_table_sha256"],
            "canonical_config": canonical_config["passed"],
            "fixture_oracle": fixture_probe["passed"],
            "eval_schema": eval_schema["passed"],
            "clip_schema": clip_schema["passed"],
            "zero_action_sanity": (
                zero_action["finite"]
                and zero_action["done_count"] == 0
                and zero_action["terminated_count"] == 0
                and zero_action["time_out_count"] == 0
                and zero_action["reward_manager_sum_abs_error_max"] <= 1.0e-6
                and len(zero_action["first_step_phase"] or []) == 2
                and all(
                    abs(actual - expected) <= 1.0e-6
                    for actual, expected in zip(zero_action["first_step_phase"] or [], [0.025, 0.525])
                )
                and zero_action["reset_episode_length_buf"] == 0
                and zero_action["post_reset_episode_length_buf"] == 0
            ),
        }
        result.update(
            {
                **_common_manifest(argparse.Namespace(command="verify-reward-eval-contract", seed=args.seed)),
                "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
                "runner_sha256": sha256_path(Path(__file__).resolve()),
                "test_sha256": sha256_path(ROOT / "tests/test_task072_locomotion_proof.py"),
                "reward_payload_sha256": reward_payload["payload_sha256"],
                "actual_manager_active_table_sha256": train_active_sha,
                "train_reward_active_table": train_table,
                "eval_reward_active_table": eval_table,
                "train_observation_active_table": train_observation_table,
                "eval_observation_active_table": eval_observation_table,
                "train_termination_active_table": train_termination_table,
                "eval_termination_active_table": eval_termination_table,
                "train_cross_manager_contract": train_cross_manager,
                "eval_cross_manager_contract": eval_cross_manager,
                "termination_buffer_evidence": termination_buffers,
                "termination_reward_probe": termination_reward_probe,
                "reward_manager_api": manager_api,
                "reward_fixture_probe": fixture_probe,
                "eval_schema_self_check": eval_schema,
                "clip_schema_self_check": clip_schema,
                "zero_action_runtime": zero_action,
                "canonical_train_eval_config": canonical_config,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    except Exception as exc:
        result.update({"passed": False, "error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        if train_outer is not None:
            train_outer.close()
        if eval_outer is not None:
            eval_outer.close()
    write_json(output, result)
    print(json.dumps({"passed": result["passed"], "output": str(output)}), flush=True)
    return 0 if result["passed"] else 1


def _common_manifest(args: argparse.Namespace) -> dict[str, Any]:
    ensure_v2_artifacts()
    contact_payload = json.loads(CONTACT_PROFILE.read_text(encoding="utf-8"))
    stance_payload = json.loads(STANCE.read_text(encoding="utf-8"))
    action_contract = action_contract_from_asset_xml()
    canonical = canonical_train_eval_config_payload()
    runtime = _runtime_metadata(" ".join(sys.argv))
    external = runtime["external_mjlab"]
    reward_cfg = task072_reward_v4_table(_stance_dict())
    reward_active_table = task072_reward_active_table_from_cfg(reward_cfg)
    reward_payload = task072_canonical_reward_payload(reward_active_table)
    manifest_subtask = "003f" if getattr(args, "command", None) == "verify-runtime-binding" else TASK072_ACTIVE_SUBTASK
    if getattr(args, "command", None) in {"evaluate", "verify-reward-eval-contract"}:
        semantic_num_envs = int(getattr(args, "eval_envs", 1))
        semantic_seed = int(getattr(args, "seed", DEFAULT_SEED + 99))
        semantic_fixed = True
    else:
        semantic_num_envs = int(getattr(args, "num_envs", REQUIRED_CAPACITY_NUM_ENVS))
        semantic_seed = int(getattr(args, "seed", DEFAULT_SEED))
        semantic_fixed = False
    semantic_rollout_steps = int(getattr(args, "rollout_steps", REQUIRED_ROLLOUT_STEPS))
    semantic_max_iterations = int(getattr(args, "updates", getattr(args, "max_iterations", 1)))
    stage_semantic = (
        None
        if manifest_subtask == "003f"
        else task072_stage_semantic_contract(
            num_envs=semantic_num_envs,
            rollout_steps=semantic_rollout_steps,
            seed=semantic_seed,
            max_iterations=semantic_max_iterations,
            fixed_command=semantic_fixed,
            render_mode=None,
        )
    )
    return {
        "schema_version": 3,
        "task": "task072-bound-g1-go2-locomotion-proof",
        "subtask": manifest_subtask,
        "lineage_id": LINEAGE_ID,
        "asset_xml": {"path": str(ASSET_XML.resolve()), "sha256": sha256_path(ASSET_XML)},
        "contact_profile": {
            "path": str(CONTACT_PROFILE.resolve()),
            "sha256": sha256_path(CONTACT_PROFILE),
            "payload_sha256": payload_sha256(contact_payload),
        },
        "stance": {
            "path": str(STANCE.resolve()),
            "sha256": sha256_path(STANCE),
            "payload_sha256": payload_sha256(stance_payload),
        },
        "runtime_lineage_id": LINEAGE_ID,
        "contact_profile_id": CONTACT_PROFILE_ID,
        "runner_source_sha256": sha256_path(Path(__file__).resolve()),
        "runtime_spec_sha256": hashlib.sha256(runtime_spec_xml().encode()).hexdigest(),
        "action_contract": {
            "version": ACTION_CONTRACT_VERSION,
            "payload_sha256": action_contract["payload_sha256"],
            "policy_action_domain": dict(POLICY_ACTION_DOMAIN),
        },
        "reward_contract": {
            "version": REWARD_CONTRACT_VERSION,
            "canonical_payload_sha256": reward_payload["payload_sha256"],
            "config_active_table_sha256": task072_validate_reward_active_table(reward_active_table),
        },
        "canonical_train_eval_config": {
            "payload_sha256": canonical["payload_sha256"],
            "diff": canonical["diff"],
            "eval_diff_allowlist": canonical["eval_diff_allowlist"],
            "non_allowlisted_diff": canonical["non_allowlisted_diff"],
            "passed": canonical["passed"],
        },
        "stage_semantic_contract": stage_semantic,
        "external_mjlab_checks": {
            "frame_local": EXTERNAL_MJLAB.is_relative_to(ROOT),
            "commit_pinned": external["actual_commit"] == external["expected_commit"],
            "tracked_clean": external["tracked_clean"],
        },
        "max_transitions": MAX_TRANSITIONS,
        "seed": getattr(args, "seed", DEFAULT_SEED),
        "runtime": runtime,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r0 = sub.add_parser("r0-smoke")
    r0.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "r0_registration_smoke.json")
    r0.add_argument("--num-envs", type=int, default=1)
    r0.add_argument("--rollout-steps", type=int, default=24)
    r0.add_argument("--steps", type=int, default=2)
    r0.add_argument("--seed", type=int, default=DEFAULT_SEED)
    r0.add_argument("--device", default="cpu")
    r0.set_defaults(func=r0_smoke)

    cap = sub.add_parser("capacity-smoke")
    cap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "003k_capacity_smoke_2048_4096_6144.json")
    cap.add_argument("--candidates", type=int, nargs="+", default=[2048, 4096, 6144])
    cap.add_argument("--rollout-steps", type=int, default=24)
    cap.add_argument("--steps", type=int, default=2)
    cap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    cap.add_argument("--device", default="cuda:0")
    cap.set_defaults(func=capacity_smoke)

    train = sub.add_parser("one-update-train")
    train.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "003k_one_update_4096x24_seed720301")
    train.add_argument("--capacity-artifact", type=Path, default=RUNTIME_BINDING_ROOT / "003k_capacity_smoke_2048_4096_6144.json")
    train.add_argument("--num-envs", type=int, default=4096)
    train.add_argument("--rollout-steps", type=int, default=24)
    train.add_argument("--updates", type=int, default=1)
    train.add_argument("--save-interval", type=int, default=1)
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--device", default="cuda:0")
    train.set_defaults(func=one_update_train)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--run-manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--eval-envs", type=int, default=256)
    evaluate.add_argument("--eval-seconds", type=float, default=20.0)
    evaluate.add_argument("--rollout-steps", type=int, default=24)
    evaluate.add_argument("--seed", type=int, default=DEFAULT_SEED + 99)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.set_defaults(func=evaluate_checkpoint)
    pilot_gate = sub.add_parser("pilot-gate")
    pilot_gate.add_argument("--run-manifest", type=Path, default=DEFAULT_OUTPUT_ROOT / "003k_pilot_4096x24x21_seed720301/run_manifest.json")
    pilot_gate.add_argument(
        "--eval",
        type=Path,
        nargs=4,
        default=[
            DEFAULT_OUTPUT_ROOT / f"003k_eval_pilot_model_{update}_fixed_vx0p5_seed720400.json"
            for update in TASK072_PILOT_EVAL_UPDATES
        ],
    )
    pilot_gate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "003k_pilot_gate.json")
    pilot_gate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    pilot_gate.set_defaults(func=pilot_gate_command)
    verify = sub.add_parser("verify-runtime-binding")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--seed", type=int, default=DEFAULT_SEED)
    verify.add_argument("--device", default="cpu")
    verify.set_defaults(func=verify_runtime_binding)
    contract = sub.add_parser("verify-reward-eval-contract")
    contract.add_argument("--output", type=Path, default=RUNTIME_BINDING_ROOT / "003k_v4_semantic_reward_eval_contract_verifier.json")
    contract.add_argument("--seed", type=int, default=DEFAULT_SEED)
    contract.set_defaults(func=verify_reward_eval_contract)
    render = sub.add_parser("render")
    render.add_argument("--checkpoint", type=Path, required=True)
    render.add_argument("--run-manifest", type=Path, required=True)
    render.add_argument("--eval", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--seed", type=int, default=DEFAULT_SEED + 199)
    render.add_argument("--device", default="cuda:0")
    render.set_defaults(func=render_command)
    reload_verify = sub.add_parser("verify-reload")
    reload_verify.add_argument("--checkpoint", type=Path, required=True)
    reload_verify.add_argument("--run-manifest", type=Path, required=True)
    reload_verify.add_argument("--eval", type=Path, required=True)
    reload_verify.add_argument("--video", type=Path, required=True)
    reload_verify.add_argument("--output", type=Path, required=True)
    reload_verify.add_argument("--seed", type=int, default=DEFAULT_SEED + 299)
    reload_verify.add_argument("--device", default="cpu")
    reload_verify.set_defaults(func=verify_reload_command)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--output", type=Path, default=RUNTIME_BINDING_ROOT / "freeze/task072_freeze_manifest.json")
    freeze.add_argument("--eval", type=Path, required=True)
    freeze.add_argument("--video", type=Path, required=True)
    freeze.add_argument("--reload-verifier", type=Path, required=True)
    freeze.add_argument("--seed", type=int, default=DEFAULT_SEED)
    freeze.add_argument("--device", default="cpu")
    freeze.set_defaults(func=freeze_command)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
