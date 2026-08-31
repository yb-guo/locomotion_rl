"""Train and verify locomotion on the exact Task071-bound G1 and Go2 assets."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from xml.etree import ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from h200_locomotion_lab.envs.whole_body_mujoco import phase_from_trial_step

ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parent
TASK071_DIR = ROOT / ".agent/task/task071-multimorphology-training-readiness"
DEFAULT_ARTIFACT_DIR = TASK_DIR / "artifacts"
CASES = ("unitree_g1", "unitree_go2")
COMMAND = (0.5, 0.0, 0.0)
CONTROL_HZ = 50.0
CONTROL_DT = 0.02
PHYSICS_HZ = 500.0
FINAL_EVAL_SECONDS = 20.0
FINAL_EVAL_ENVS = 20
VIDEO_SECONDS = 8.0
VIDEO_FRAMES = 400
VIDEO_TRIAL_SECONDS = VIDEO_SECONDS + 1.0 / CONTROL_HZ
ARTIFACT_VERSION = "task072_bound_locomotion_proof_v1"
ACTION_CONTRACT_VERSION = "motor_tuple_headroom_residual_v2"
BIPED_REWARD_VERSION = "task072_biped_phase_contact_v2"
QUADRUPED_REWARD_VERSION = "task072_quadruped_fixed_command_v1"
TRAINING_REWARD_VERSION = "task072_case_specific_fixed_command_reward_v1"
OBSERVATION_TRANSFORM_VERSION = "task072_relative_scaled_observation_v1"
TASK070_ATTEMPT010_ROOT = Path(
    "/home/admin1/workspace/proj/locomotion_rl/.agent/task/"
    "task070-archetype-constrained-standable-morphology/artifacts/"
    "preview_task070_v2_descriptor_driven_attempt010"
)
TASK070_FROZEN_INPUTS = {
    "unitree_g1": {
        "descriptor": {
            "relative_path": "unitree_g1_seed000/unitree_g1_29dof_structural_descriptor.json",
            "raw_sha256": "6464ad8af464956ca8c722a95fddd94b7183c0cdd153134b0cbda12f6199662e",
            "payload_sha256": "cd16bbb3bea241eaec802dbcd7ad4b25550d90246d2816e7c7d23c8f2b453855",
        },
        "manifest": {
            "relative_path": "unitree_g1_seed000/unitree_g1_29dof_anonymous_preview_manifest.json",
            "raw_sha256": "fcb581ac1feb5454bebf7251098548f10648f9f478160adfee0fa764b3405967",
            "payload_sha256": "7d1641f79f1ae72cbddc0e355a4af64d4154da8b5996f8ec3c744d49f0a07f99",
        },
        "xml": {
            "relative_path": "unitree_g1_seed000/unitree_g1_29dof_anonymous_preview.xml",
            "raw_sha256": "35f6e56eb17b018fa1288db6f74eb8c42fc6616c599008c5050a6af8805120f1",
        },
    },
    "unitree_go2": {
        "descriptor": {
            "relative_path": "unitree_go2_seed000/unitree_go2_12dof_structural_descriptor.json",
            "raw_sha256": "795fd0549643cf96ca83385d0c67ba7fb68485b074c16f610a4c197179e82bac",
            "payload_sha256": "09ed1b69922019213d21f6ee8144e64aca688cbee4968952ab58d16e2e016fd1",
        },
        "manifest": {
            "relative_path": "unitree_go2_seed000/unitree_go2_12dof_anonymous_preview_manifest.json",
            "raw_sha256": "a7afd7b32706c27d276c1b71dc527d05ac3c3fede16edde32b6152633169f398",
            "payload_sha256": "c5f1c10b165fd399026f854154f4a012841c75fd90a55aae79a0d002988d35c0",
        },
        "xml": {
            "relative_path": "unitree_go2_seed000/unitree_go2_12dof_anonymous_preview.xml",
            "raw_sha256": "296ad8fb2ae42f1bb1e437c5e722914794676c7c6f0da51b9a60c674d85ebfa9",
        },
    },
}
PARENT_ARTIFACT_NAMES = {
    "physics_overlay": "official_sim_physics_overlay_v1.json",
    "r1": "r1_g1_go2_bound_official_sim_physics_overlay_v1.json",
    "r2": "r2_env_contract_smoke.json",
    "r3": "r3_ppo_update_smoke.json",
}
MJLAB_G1_7CAPSULE_PROFILE_ID = "mjlab_g1_7capsule_v1"
MJLAB_G1_CHECKOUT_COMMIT = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
MJLAB_G1_XML_SHA256 = "56539bc76eadb05dd439c47de94df52130ea8fa243d08bdddd9cbc32dd4c78a0"
TASK071_G1_BOUND_XML_SHA256 = "c622754f2bdd01f68877873f6dfb70e55b37c29c8ba8c1cd052352b41072066d"
MJLAB_G1_XML = ROOT / ".external/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml"
CONTACT_ALIGNMENT_ROOT = DEFAULT_ARTIFACT_DIR / "contact_alignment" / MJLAB_G1_7CAPSULE_PROFILE_ID


@dataclass(frozen=True, slots=True)
class MotorTuple:
    semantic_slot: str
    control_mode: str
    effective_effort: float
    velocity_limit: float | None
    kp: float
    kd: float
    armature: float
    friction: float
    transmission_group: str
    provenance_sha256: str


@dataclass(frozen=True, slots=True)
class Task072BipedRewardConfig:
    version: str = BIPED_REWARD_VERSION
    track_xy: float = 2.0
    track_yaw: float = 0.5
    upright: float = 0.25
    tilt: float = 5.0
    height: float = 0.25
    stand_support: float = 0.30
    phase_gait: float = 0.50
    out_of_phase_double_support: float = 0.35
    clearance: float = 0.50
    touchdown_airtime: float = 0.10
    soft_landing: float = 0.10
    foot_slip: float = 0.20
    nonfoot_contact: float = 0.20
    pose_hip: float = 0.20
    pose_knee: float = 0.30
    pose_ankle: float = 0.20
    pose_waist: float = 0.10
    pose_arm_wrist: float = 0.05
    joint_velocity: float = 0.02
    joint_limit: float = 0.05
    action_magnitude: float = 0.01
    action_rate: float = 0.01
    base_angvel_xy: float = 0.02
    period_s: float = 0.8
    stance_threshold: float = 0.55
    left_phase_offset: float = 0.0
    right_phase_offset: float = 0.5
    moving_command_threshold: float = 0.1


@dataclass(frozen=True, slots=True)
class Task072QuadrupedRewardConfig:
    version: str = QUADRUPED_REWARD_VERSION
    track_xy: float = 1.0
    track_yaw: float = 1.0
    heading: float = 1.0
    upright: float = 1.0
    tilt: float = 1.0
    nonfoot_contact: float = 1.0
    action_rate: float = 1.0


@dataclass(frozen=True, slots=True)
class Task072TrainStageConfig:
    stage: str
    envs: int
    rollout_steps: int
    updates: int
    transitions: int
    seed: int = 72072
    checkpoint_every: int = 200


TRAIN_STAGES = {
    "smoke": Task072TrainStageConfig("smoke", envs=4, rollout_steps=32, updates=2, transitions=256, checkpoint_every=1),
    "pilot": Task072TrainStageConfig("pilot", envs=32, rollout_steps=64, updates=1000, transitions=2_048_000),
    "proof": Task072TrainStageConfig("proof", envs=32, rollout_steps=64, updates=31_200, transitions=63_897_600),
}
E3A_REPAIR_VARIANT = "E3a_mjlab_kl_repair"
E3A_ADAPTIVE_VARIANTS = {"E3a_adaptive_kl", E3A_REPAIR_VARIANT}
R2_PHASE_OBSERVATION_VARIANTS = {"E1_phase", "E2_reward_dt", *E3A_ADAPTIVE_VARIANTS}
R2_REWARD_DT_VARIANTS = {"E2_reward_dt", *E3A_ADAPTIVE_VARIANTS}
R2_NOMINAL_V4_VARIANTS = {*R2_PHASE_OBSERVATION_VARIANTS}
E3A_OPTIMIZER_GATE_THRESHOLDS = {
    "approx_kl_mean": 0.015,
    "approx_kl_p95": 0.03,
    "approx_kl_max": 0.05,
    "clip_fraction_mean": 0.20,
    "clip_fraction_p95": 0.35,
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256_bytes(encoded)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def ppo_reward_from_raw(raw_total_reward: Any, reward_scale: float = 1.0) -> Any:
    """Apply the single final scalar reward scale after component summation."""
    return raw_total_reward * float(reward_scale)


def _yaw_from_quaternion(quaternion_wxyz: Any) -> float:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(torch: Any, path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _runtime_metadata(command: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "command": command,
        "git_head": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--short")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "robot_asset_dataset_or_checkpoint_downloads_performed": False,
    }
    try:
        import mujoco

        metadata["mujoco_version"] = mujoco.__version__
    except ImportError:
        metadata["mujoco_version"] = None
    try:
        import torch

        metadata.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        metadata["torch_version"] = None
    return metadata


def quality_gate(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    """Apply the predeclared per-case Task072 walking gate."""

    reasons: list[str] = []
    thresholds = {
        "zero_fall_ratio": (0.95, "minimum"),
        "planar_velocity_error": (0.35, "maximum"),
        "yaw_error": (0.35, "maximum"),
        "gravity_xy": (0.35, "maximum"),
    }
    for key, (limit, direction) in thresholds.items():
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            reasons.append(f"{key}_not_finite")
        elif direction == "minimum" and float(value) < limit:
            reasons.append(f"{key}_below_threshold")
        elif direction == "maximum" and float(value) > limit:
            reasons.append(f"{key}_above_threshold")

    displacement = metrics.get("nonfall_forward_displacement_mean")
    if not isinstance(displacement, (int, float)) or not math.isfinite(float(displacement)):
        reasons.append("nonfall_forward_displacement_not_finite")
    elif float(displacement) <= 0.0:
        reasons.append("nonfall_forward_displacement_not_positive")

    zero_margin = metrics.get("zero_action_common_prefix_planar_margin")
    update0_margin = metrics.get("update0_common_prefix_planar_margin")
    if not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in (zero_margin, update0_margin)
    ):
        reasons.append("paired_common_prefix_margin_not_finite")
    elif min(float(zero_margin), float(update0_margin)) < 0.05:
        reasons.append("trained_policy_lacks_0p05_learning_margin")

    for key in ("zero_action_forward_displacement_margin", "update0_forward_displacement_margin"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            reasons.append(f"{key}_not_finite")
        elif float(value) < 2.0:
            reasons.append(f"{key}_below_2m")

    for key in (
        "finite",
        "checkpoint_verified",
        "progression_verified",
        "paired_baselines_verified",
        "video_verified",
        "final_eval_configuration_verified",
    ):
        if metrics.get(key) is not True:
            reasons.append(f"{key}_false")
    return not reasons, reasons


# Backwards-compatible name used by the first task-local draft.
gate = quality_gate


def _load_parent_artifacts() -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    artifact_dir = TASK071_DIR / "artifacts"
    for label, name in PARENT_ARTIFACT_NAMES.items():
        path = artifact_dir / name
        _require(path.is_file(), f"missing Task071 parent artifact: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        artifacts[label] = {
            "path": path,
            "payload": payload,
            "raw_sha256": sha256_path(path),
            "payload_sha256": payload_sha256(payload),
        }
    expected_cases = set(CASES)
    for label in ("physics_overlay", "r1", "r2", "r3"):
        payload = artifacts[label]["payload"]
        _require(payload.get("denominator") == 2, f"Task071 {label} denominator drift")
        observed = {record.get("reference_id") for record in payload.get("records", [])}
        _require(observed == expected_cases, f"Task071 {label} case set drift")
    _require(
        artifacts["r1"]["payload"].get("task071_r1_admission_passed") is True,
        "Task071 R1 admission is required",
    )
    _require(
        artifacts["r2"]["payload"].get("task071_r2_admission_passed") is True,
        "Task071 R2 admission is required",
    )
    _require(
        artifacts["r3"]["payload"].get("task071_r3_representative_admission_passed") is True,
        "Task071 R3 admission is required",
    )
    return artifacts


def _task071_modules() -> tuple[Any, Any]:
    for path in (str(TASK071_DIR), str(ROOT / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import task071_env_contract as env_contract
    import task071_physics_overlay as physics

    return physics, env_contract


def _record(records: list[dict[str, Any]], reference_id: str) -> dict[str, Any]:
    matches = [record for record in records if record.get("reference_id") == reference_id]
    _require(len(matches) == 1, f"expected one Task071 record for {reference_id}")
    return matches[0]


def _slot_motor_sources(case: Any) -> dict[str, dict[str, Any]]:
    metadata = case.blueprint.profile_metadata
    motor_configuration = metadata.get("motor_configuration")
    _require(isinstance(motor_configuration, dict), "missing Task070 motor configuration")
    resolved = motor_configuration.get("resolved_anonymous_actuators")
    _require(isinstance(resolved, list), "missing Task070 resolved motor tuples")
    by_slot = {record.get("anonymous_semantic_slot"): record for record in resolved}
    actuation_stack = metadata.get("actuation_stack")
    _require(isinstance(actuation_stack, dict), "missing Task070 actuation stack")
    coherent = actuation_stack.get("coherent_motor_config")
    _require(isinstance(coherent, dict), "missing coherent motor config")
    for family in coherent.get("families", []):
        for slot in family.get("anonymous_semantic_slots", []):
            if slot in by_slot:
                by_slot[slot] = {
                    **by_slot[slot],
                    "family_id": family.get("family_id"),
                    "family_source_motor_class": family.get("source_motor_class"),
                    "family_source_config": family.get("source_config"),
                }
    return by_slot


def resolve_motor_tuples(
    case_id: str,
    parent_artifacts: dict[str, Any] | None = None,
) -> tuple[MotorTuple, ...]:
    parents = parent_artifacts or _load_parent_artifacts()
    physics, _env_contract = _task071_modules()
    case = physics.load_frozen_case(case_id)
    overlay_record = _record(parents["physics_overlay"]["payload"]["records"], case_id)
    motor_records = overlay_record.get("motor_mapping")
    _require(isinstance(motor_records, list), "missing Task071 motor mapping")
    source_by_slot = _slot_motor_sources(case)
    expected_slots = tuple(actuator.semantic_slot for actuator in case.blueprint.actuators)
    _require(
        tuple(record.get("semantic_slot") for record in motor_records) == expected_slots,
        "Task071 motor mapping does not match blueprint actuator order",
    )
    tuples: list[MotorTuple] = []
    for record in motor_records:
        slot = str(record["semantic_slot"])
        source = source_by_slot.get(slot)
        _require(isinstance(source, dict), f"missing Task070 motor source for {slot}")
        raw_or_proxy = source.get("raw_or_proxy")
        final_compiled = source.get("final_compiled")
        _require(isinstance(raw_or_proxy, dict), f"missing raw motor source for {slot}")
        _require(isinstance(final_compiled, dict), f"missing compiled motor source for {slot}")
        force_range = record.get("force_range")
        _require(isinstance(force_range, list) and len(force_range) == 2, f"bad force range for {slot}")
        effective_effort = max(abs(float(value)) for value in force_range)
        kp = float(record["position_kp"])
        kd = float(record["position_kd"])
        armature = float(record["joint_armature"])
        friction = float(record["joint_frictionloss"])
        _require(effective_effort > 0.0 and kp > 0.0, f"non-positive motor tuple for {slot}")
        _require(all(math.isfinite(value) for value in (effective_effort, kp, kd, armature, friction)), f"nonfinite motor tuple for {slot}")
        control_mode = str(
            (
                source.get("family_source_config")
                or {}
            ).get("raw_declared", {}).get("control_mode", "builtin_position_pd")
        )
        velocity_hint = raw_or_proxy.get("velocity_limit_hint")
        velocity_limit = None if velocity_hint is None else float(velocity_hint)
        if velocity_limit is not None:
            _require(math.isfinite(velocity_limit) and velocity_limit > 0.0, f"bad velocity limit for {slot}")
        provenance = {
            "case_id": case_id,
            "semantic_slot": slot,
            "parent_physics_overlay_raw_sha256": parents["physics_overlay"]["raw_sha256"],
            "parent_physics_overlay_payload_sha256": parents["physics_overlay"]["payload_sha256"],
            "overlay_motor_mapping_record": record,
            "task070_resolved_motor_record": source,
        }
        tuples.append(
            MotorTuple(
                semantic_slot=slot,
                control_mode=control_mode,
                effective_effort=effective_effort,
                velocity_limit=velocity_limit,
                kp=kp,
                kd=kd,
                armature=armature,
                friction=friction,
                transmission_group=str(source.get("family_id") or slot),
                provenance_sha256=payload_sha256(provenance),
            )
        )
    _require(len(tuples) == (29 if case_id == "unitree_g1" else 12), f"{case_id} motor tuple count drift")
    return tuple(tuples)


def derive_position_action_amplitudes(tuples: tuple[MotorTuple, ...]) -> dict[str, float]:
    amplitudes: dict[str, float] = {}
    for item in tuples:
        _require(item.control_mode == "builtin_position_pd", f"unsupported control mode: {item.semantic_slot}")
        _require(item.effective_effort > 0.0 and item.kp > 0.0, f"bad quantitative tuple: {item.semantic_slot}")
        amplitude = 0.25 * item.effective_effort / item.kp
        _require(math.isfinite(amplitude) and amplitude > 0.0, f"bad action amplitude: {item.semantic_slot}")
        amplitudes[item.semantic_slot] = amplitude
    _require(len(amplitudes) == len(tuples), "duplicate motor tuple slot")
    return amplitudes


def derive_position_action_bounds(
    tuples: tuple[MotorTuple, ...],
    stance_ctrl: dict[str, float],
    actuator_ranges: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    expected_slots = {item.semantic_slot for item in tuples}
    _require(set(stance_ctrl) == expected_slots, "stance ctrl slots do not match motor tuples")
    _require(set(actuator_ranges) == expected_slots, "actuator range slots do not match motor tuples")
    for item in tuples:
        _require(item.control_mode == "builtin_position_pd", f"unsupported control mode: {item.semantic_slot}")
        _require(item.effective_effort > 0.0 and item.kp > 0.0, f"bad quantitative tuple: {item.semantic_slot}")
        lower, upper = actuator_ranges[item.semantic_slot]
        q0 = float(stance_ctrl[item.semantic_slot])
        _require(
            all(math.isfinite(value) for value in (lower, upper, q0)) and lower < upper,
            f"bad action range: {item.semantic_slot}",
        )
        margin = max(0.05 * (upper - lower), 1e-4)
        neg_headroom = q0 - (lower + margin)
        pos_headroom = (upper - margin) - q0
        _require(neg_headroom > 0.0 and pos_headroom > 0.0, f"non-positive action headroom: {item.semantic_slot}")
        motor_delta = 0.25 * item.effective_effort / item.kp
        delta_neg = min(motor_delta, neg_headroom)
        delta_pos = min(motor_delta, pos_headroom)
        _require(
            all(math.isfinite(value) and value > 0.0 for value in (delta_neg, delta_pos)),
            f"bad action residual bound: {item.semantic_slot}",
        )
        bounds[item.semantic_slot] = (delta_neg, delta_pos)
    _require(len(bounds) == len(tuples), "duplicate motor tuple slot")
    return bounds


def _case_actuator_ranges(overlay_record: dict[str, Any]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for record in overlay_record.get("motor_mapping", []):
        slot = str(record["semantic_slot"])
        values = record.get("range_rad")
        _require(isinstance(values, list) and len(values) == 2, f"bad actuator range for {slot}")
        ranges[slot] = (float(values[0]), float(values[1]))
    return ranges


def _case_stance_ctrl(context: dict[str, Any]) -> dict[str, float]:
    return {
        actuator.semantic_slot: float(context["stance"].actuator_ctrl[actuator.semantic_slot])
        for actuator in context["case"].blueprint.actuators
    }


def _verify_task070_frozen_inputs(case_id: str, overlay_record: dict[str, Any]) -> dict[str, Any]:
    expected = TASK070_FROZEN_INPUTS[case_id]
    frozen_input = overlay_record.get("frozen_input")
    _require(isinstance(frozen_input, dict), f"{case_id} missing frozen_input")
    records: dict[str, Any] = {}
    for label, spec in expected.items():
        path = TASK070_ATTEMPT010_ROOT / spec["relative_path"]
        _require(path.is_file(), f"missing Task070 frozen input: {path}")
        raw_sha = sha256_path(path)
        _require(raw_sha == spec["raw_sha256"], f"{case_id} {label} raw SHA drift")
        _require(str(path) == frozen_input[f"{label}_path"], f"{case_id} {label} path drift")
        _require(frozen_input[f"{label}_sha256"] == spec["raw_sha256"], f"{case_id} Task071 {label} SHA drift")
        item: dict[str, Any] = {
            "path": str(path),
            "relative_path": spec["relative_path"],
            "raw_sha256": raw_sha,
        }
        if label != "xml":
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload_sha = payload_sha256(payload)
            _require(payload_sha == spec["payload_sha256"], f"{case_id} {label} payload SHA drift")
            item["payload_sha256"] = payload_sha
            if label == "manifest":
                _require(payload.get("descriptor_sha256") == expected["descriptor"]["raw_sha256"], f"{case_id} manifest descriptor SHA is not closed")
                _require(payload.get("xml_sha256") == expected["xml"]["raw_sha256"], f"{case_id} manifest XML SHA is not closed")
                blueprint = payload.get("blueprint_manifest", {})
                profile = blueprint.get("profile_metadata", {}) if isinstance(blueprint, dict) else {}
                actuation_stack = profile.get("actuation_stack", {}) if isinstance(profile, dict) else {}
                motor_configuration = profile.get("motor_configuration", {}) if isinstance(profile, dict) else {}
                coherent = actuation_stack.get("coherent_motor_config", {}) if isinstance(actuation_stack, dict) else {}
                item["motor_configuration_payload_sha256"] = payload_sha256(motor_configuration)
                item["coherent_motor_config_payload_sha256"] = payload_sha256(coherent)
        records[label] = item
    return records


def action_contract_payload(cases: Iterable[str]) -> dict[str, Any]:
    parents = _load_parent_artifacts()
    physics, _env_contract = _task071_modules()
    records = []
    for case_id in cases:
        case = physics.load_frozen_case(case_id)
        overlay_record = _record(parents["physics_overlay"]["payload"]["records"], case_id)
        tuples = resolve_motor_tuples(case_id, parents)
        amplitudes = derive_position_action_amplitudes(tuples)
        stance_ctrl = {
            actuator.semantic_slot: float(case.blueprint.profile_metadata.get("visual_audit_nominal_joint_pose", {}).get(actuator.semantic_slot, 0.0))
            for actuator in case.blueprint.actuators
        }
        context = _load_bound_context(case_id)
        stance_ctrl = _case_stance_ctrl(context)
        actuator_ranges = _case_actuator_ranges(overlay_record)
        bounds = derive_position_action_bounds(tuples, stance_ctrl, actuator_ranges)
        record_by_slot = {record["semantic_slot"]: record for record in overlay_record["motor_mapping"]}
        source_by_slot = _slot_motor_sources(case)
        task070_inputs = _verify_task070_frozen_inputs(case_id, overlay_record)
        slots = []
        for item in tuples:
            motor_record = record_by_slot[item.semantic_slot]
            source = source_by_slot[item.semantic_slot]
            lower, upper = actuator_ranges[item.semantic_slot]
            q0 = stance_ctrl[item.semantic_slot]
            margin = max(0.05 * (upper - lower), 1e-4)
            motor_delta = amplitudes[item.semantic_slot]
            neg_headroom = q0 - (lower + margin)
            pos_headroom = (upper - margin) - q0
            neg_bound, pos_bound = bounds[item.semantic_slot]
            slots.append(
                {
                    "semantic_slot": item.semantic_slot,
                    "module": motor_record.get("module"),
                    "source_motor_class": source.get("source_motor_class") or source.get("family_source_motor_class"),
                    "tuple": asdict(item),
                    "motor_delta_rad": motor_delta,
                    "safety_margin_rad": margin,
                    "negative_headroom_rad": neg_headroom,
                    "positive_headroom_rad": pos_headroom,
                    "action_residual_bounds_rad": {
                        "negative_magnitude": neg_bound,
                        "positive_magnitude": pos_bound,
                    },
                    "compiled_actuator_range_rad": [lower, upper],
                    "stance_ctrl_rad": q0,
                    "normal_action_would_clamp": False,
                    "normal_action_actual_clamp": False,
                    "anonymous_joint": motor_record.get("anonymous_joint"),
                    "anonymous_actuator": motor_record.get("anonymous_actuator"),
                    "official_joint": motor_record.get("official_joint"),
                    "official_actuator": motor_record.get("official_actuator"),
                    "source_sha256": (
                        (source.get("family_source_config") or {}).get("source_sha256")
                    ),
                }
            )
        records.append(
            {
                "case_id": case_id,
                "family": case.spec["family"],
                "slot_count": len(slots),
                "active_slots": [slot["semantic_slot"] for slot in slots],
                "action_amplitudes": amplitudes,
                "action_residual_bounds_by_slot": bounds,
                "slots": slots,
                "task070_frozen_inputs": task070_inputs,
                "input_parent_payload_sha256": parents["physics_overlay"]["payload_sha256"],
                "bound_xml_sha256": overlay_record["output_xml_sha256"],
            }
        )
    payload = {
        "schema_version": 1,
        "artifact": ARTIFACT_VERSION,
        "formula_version": ACTION_CONTRACT_VERSION,
        "formula": (
            "motor_delta_i=0.25*effective_effort_i/kp_i; "
            "margin_i=max(0.05*(upper_i-lower_i),1e-4); "
            "delta_neg_i=min(motor_delta_i,q0_i-(lower_i+margin_i)); "
            "delta_pos_i=min(motor_delta_i,(upper_i-margin_i)-q0_i)"
        ),
        "cases": records,
        "parent_artifacts": {
            label: {
                "path": str(item["path"].relative_to(ROOT)),
                "raw_sha256": item["raw_sha256"],
                "payload_sha256": item["payload_sha256"],
            }
            for label, item in parents.items()
        },
        "source_sha256": sha256_path(Path(__file__)),
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def _task072_bound_xml(physics: Any, mujoco: Any, case: Any, overlay_record: dict[str, Any]) -> str:
    regenerated, xml = physics._bind_case(case, mujoco, write_artifact=False)
    actual_sha = sha256_bytes(xml.encode())
    _require(actual_sha == overlay_record["output_xml_sha256"], "Task071 bound XML SHA changed")
    if regenerated != overlay_record:
        relaxed_regenerated = json.loads(json.dumps(regenerated))
        relaxed_overlay = json.loads(json.dumps(overlay_record))
        for payload in (relaxed_regenerated, relaxed_overlay):
            if isinstance(payload.get("compile_evidence"), dict):
                payload["compile_evidence"].pop("mujoco_version", None)
        _require(
            relaxed_regenerated == relaxed_overlay,
            "Task071 in-memory overlay changed beyond MuJoCo compile-version metadata",
        )
    return xml


def _float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split())


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise ValueError(f"missing body {name}")


def _compiled_model_signature(mujoco: Any, model: Any) -> dict[str, Any]:
    def name(obj: Any, index: int) -> str:
        return mujoco.mj_id2name(model, obj, index) or ""

    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "body": [
            {
                "name": name(mujoco.mjtObj.mjOBJ_BODY, index),
                "mass": float(model.body_mass[index]),
                "ipos": [float(value) for value in model.body_ipos[index]],
                "iquat": [float(value) for value in model.body_iquat[index]],
                "inertia": [float(value) for value in model.body_inertia[index]],
            }
            for index in range(model.nbody)
        ],
        "joint": [
            {
                "name": name(mujoco.mjtObj.mjOBJ_JOINT, index),
                "type": int(model.jnt_type[index]),
                "body": int(model.jnt_bodyid[index]),
                "axis": [float(value) for value in model.jnt_axis[index]],
                "range": [float(value) for value in model.jnt_range[index]],
                "limited": int(model.jnt_limited[index]),
                "qposadr": int(model.jnt_qposadr[index]),
                "dofadr": int(model.jnt_dofadr[index]),
            }
            for index in range(model.njnt)
        ],
        "actuator": [
            {
                "name": name(mujoco.mjtObj.mjOBJ_ACTUATOR, index),
                "trntype": int(model.actuator_trntype[index]),
                "trnid": [int(value) for value in model.actuator_trnid[index]],
                "ctrlrange": [float(value) for value in model.actuator_ctrlrange[index]],
                "forcerange": [float(value) for value in model.actuator_forcerange[index]],
                "gainprm": [float(value) for value in model.actuator_gainprm[index]],
                "biasprm": [float(value) for value in model.actuator_biasprm[index]],
                "gear": [float(value) for value in model.actuator_gear[index]],
            }
            for index in range(model.nu)
        ],
    }


def _invariant_diff(parent: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "nq": parent["nq"] == candidate["nq"],
        "nv": parent["nv"] == candidate["nv"],
        "nu": parent["nu"] == candidate["nu"],
        "body_order": [row["name"] for row in parent["body"]] == [row["name"] for row in candidate["body"]],
        "joint_order": [row["name"] for row in parent["joint"]] == [row["name"] for row in candidate["joint"]],
        "actuator_order": [row["name"] for row in parent["actuator"]] == [row["name"] for row in candidate["actuator"]],
        "body_mass_com_inertia": parent["body"] == candidate["body"],
        "joint_axis_range": parent["joint"] == candidate["joint"],
        "actuator_transmission_limits_pd": parent["actuator"] == candidate["actuator"],
    }
    return {"checks": checks, "passed": all(checks.values())}


def _inject_compiled_inertials(mujoco: Any, root: ET.Element, model: Any) -> None:
    for body_id in range(1, model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            continue
        body = _find_body(root, body_name)
        for child in list(body):
            if child.tag == "inertial":
                body.remove(child)
        inertial = ET.Element(
            "inertial",
            {
                "pos": _fmt(model.body_ipos[body_id]),
                "quat": _fmt(model.body_iquat[body_id]),
                "mass": _fmt((float(model.body_mass[body_id]),)),
                "diaginertia": _fmt(model.body_inertia[body_id]),
            },
        )
        body.insert(0, inertial)


def _mjlab_capsule_layout() -> dict[str, list[dict[str, Any]]]:
    _require(MJLAB_G1_XML.is_file(), f"missing MJLab G1 XML: {MJLAB_G1_XML}")
    _require(sha256_path(MJLAB_G1_XML) == MJLAB_G1_XML_SHA256, "MJLab G1 XML SHA mismatch")
    root = ET.parse(MJLAB_G1_XML).getroot()
    layout: dict[str, list[dict[str, Any]]] = {}
    for side in ("left", "right"):
        body = _find_body(root, f"{side}_ankle_roll_link")
        rows = []
        for geom in body.findall("geom"):
            name = geom.get("name", "")
            if name.startswith(f"{side}_foot") and name.endswith("_collision"):
                rows.append({"source_name": name, "fromto": list(_float_tuple(geom.get("fromto", "")))})
        _require(len(rows) == 7, f"MJLab {side} foot capsule count mismatch")
        layout[side] = sorted(rows, key=lambda row: str(row["source_name"]))
    return layout


def _scaled_capsules_for_foot(parent_footpad: ET.Element, source_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pos = _float_tuple(parent_footpad.get("pos", "0 0 0"))
    size = _float_tuple(parent_footpad.get("size", "0 0 0"))
    source_points = [row["fromto"][index:index + 3] for row in source_rows for index in (0, 3)]
    min_x = min(point[0] for point in source_points)
    max_x = max(point[0] for point in source_points)
    max_abs_y = max(abs(point[1]) for point in source_points)
    source_center_x = 0.5 * (min_x + max_x)
    x_scale = float(size[0]) / max(1e-12, 0.5 * (max_x - min_x))
    y_scale = float(size[1]) / max(1e-12, max_abs_y)
    radius = 0.01
    centerline_z = float(pos[2]) - float(size[2]) + radius

    def transform(point: list[float]) -> tuple[float, float, float]:
        return (
            float(pos[0]) + (float(point[0]) - source_center_x) * x_scale,
            float(pos[1]) + float(point[1]) * y_scale,
            centerline_z,
        )

    capsules = []
    for index, row in enumerate(source_rows, start=1):
        p0 = transform(row["fromto"][0:3])
        p1 = transform(row["fromto"][3:6])
        capsules.append(
            {
                "index": index,
                "source_name": row["source_name"],
                "fromto": [*p0, *p1],
                "radius": radius,
            }
        )
    transform_record = {
        "parent_footpad": dict(parent_footpad.attrib),
        "source_x_bounds": [min_x, max_x],
        "source_center_x": source_center_x,
        "source_max_abs_y": max_abs_y,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "centerline_z_rule": "parent_box_bottom_plus_unscaled_mjlab_radius",
        "radius_m": radius,
    }
    return capsules, transform_record


def _make_contact_aligned_asset(context: dict[str, Any]) -> dict[str, Any]:
    import mujoco

    parent_xml = context["xml"]
    _require(sha256_bytes(parent_xml.encode()) == TASK071_G1_BOUND_XML_SHA256, "parent G1 XML SHA mismatch")
    parent_model = mujoco.MjModel.from_xml_string(parent_xml)
    parent_signature = _compiled_model_signature(mujoco, parent_model)
    root = ET.fromstring(parent_xml)
    _inject_compiled_inertials(mujoco, root, parent_model)
    source_layout = _mjlab_capsule_layout()
    side_to_link = {
        "left": "anon_limb0_ankle_roll_link",
        "right": "anon_limb1_ankle_roll_link",
    }
    groups: dict[str, tuple[str, ...]] = {}
    reference_sites: dict[str, str] = {}
    transform_records: dict[str, Any] = {}
    capsule_records: dict[str, Any] = {}
    for side, link_name in side_to_link.items():
        body = _find_body(root, link_name)
        footpad_name = f"{link_name}_footpad"
        footpad = next((geom for geom in body.findall("geom") if geom.get("name") == footpad_name), None)
        _require(footpad is not None and footpad.get("type") == "box", f"missing parent {side} foot box")
        capsules, transform_record = _scaled_capsules_for_foot(footpad, source_layout[side])
        body.remove(footpad)
        logical_name = f"{side}_foot"
        geom_names = []
        for capsule in capsules:
            geom_name = f"{link_name}_mjlab_foot{capsule['index']}_collision"
            geom_names.append(geom_name)
            ET.SubElement(
                body,
                "geom",
                {
                    "name": geom_name,
                    "type": "capsule",
                    "fromto": _fmt(capsule["fromto"]),
                    "size": _fmt((capsule["radius"],)),
                    "density": "0",
                    "friction": footpad.get("friction", "1 0.005 0.0001"),
                    "contype": footpad.get("contype", "1"),
                    "conaffinity": footpad.get("conaffinity", "1"),
                    "condim": "6",
                    "priority": "1",
                    "rgba": footpad.get("rgba", "0.16 0.18 0.22 1"),
                },
            )
        site_name = f"{link_name}_foot"
        reference_sites[logical_name] = site_name
        site = next((item for item in body.findall("site") if item.get("name") == site_name), None)
        if site is not None:
            site.set("pos", _fmt((transform_record["parent_footpad"]["pos"].split()[0], 0.0, transform_record["parent_footpad"]["pos"].split()[2])))
        groups[logical_name] = tuple(geom_names)
        transform_records[logical_name] = transform_record
        capsule_records[logical_name] = capsules
    candidate_xml = ET.tostring(root, encoding="unicode")
    candidate_model = mujoco.MjModel.from_xml_string(candidate_xml)
    candidate_signature = _compiled_model_signature(mujoco, candidate_model)
    return {
        "xml": candidate_xml,
        "xml_sha256": sha256_bytes(candidate_xml.encode()),
        "logical_foot_groups": groups,
        "logical_foot_reference_sites": reference_sites,
        "source_capsules": source_layout,
        "scaled_capsules": capsule_records,
        "scale_transforms": transform_records,
        "invariant_diff": _invariant_diff(parent_signature, candidate_signature),
        "parent_signature_sha256": payload_sha256(parent_signature),
        "candidate_signature_sha256": payload_sha256(candidate_signature),
    }


def _load_contact_alignment_artifacts() -> dict[str, Any]:
    profile_path = CONTACT_ALIGNMENT_ROOT / "contact_profile.json"
    stance_path = CONTACT_ALIGNMENT_ROOT / "stance_solution.json"
    xml_path = CONTACT_ALIGNMENT_ROOT / "unitree_g1_mjlab_g1_7capsule_v1.xml"
    _require(profile_path.is_file(), "003b contact profile is missing")
    _require(stance_path.is_file(), "003b stance solution is missing")
    _require(xml_path.is_file(), "003b bound XML is missing")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    stance_payload = json.loads(stance_path.read_text(encoding="utf-8"))
    xml = xml_path.read_text(encoding="utf-8")
    _require(profile.get("contact_profile_id") == MJLAB_G1_7CAPSULE_PROFILE_ID, "contact profile id mismatch")
    _require(sha256_bytes(xml.encode()) == profile["asset"]["candidate_xml_sha256"], "contact asset SHA drift")
    _require(stance_payload["stance_solution"]["model_xml_sha256"] == profile["asset"]["candidate_xml_sha256"], "stance asset binding mismatch")
    return {"profile": profile, "stance_payload": stance_payload, "xml": xml}


def _load_bound_context(case_id: str, *, contact_profile: str | None = None) -> dict[str, Any]:
    _require(case_id in CASES, f"unsupported Task072 case: {case_id}")
    _require(contact_profile in (None, MJLAB_G1_7CAPSULE_PROFILE_ID), f"unsupported contact profile: {contact_profile}")
    _require(contact_profile is None or case_id == "unitree_g1", "003b contact profile is G1-only")
    import mujoco
    from h200_locomotion_lab.robots.procedural_morphology import morphology_instance_key
    from h200_locomotion_lab.robots.whole_body_stance import StanceSolution

    parents = _load_parent_artifacts()
    physics, env_contract = _task071_modules()
    overlay = parents["physics_overlay"]["payload"]
    r1 = parents["r1"]["payload"]
    case = physics.load_frozen_case(case_id)
    overlay_record = _record(overlay["records"], case_id)
    r1_record = _record(r1["records"], case_id)
    xml = _task072_bound_xml(physics, mujoco, case, overlay_record)
    stance = env_contract._stance_solution(physics, case, overlay, r1)
    contact_profile_payload = None
    if contact_profile == MJLAB_G1_7CAPSULE_PROFILE_ID:
        artifacts = _load_contact_alignment_artifacts()
        profile = artifacts["profile"]
        stance_manifest = artifacts["stance_payload"]["stance_solution"]
        xml = artifacts["xml"]
        overlay_record = dict(overlay_record)
        overlay_record["parent_output_xml_sha256"] = TASK071_G1_BOUND_XML_SHA256
        overlay_record["output_xml_sha256"] = profile["asset"]["candidate_xml_sha256"]
        stance = StanceSolution(
            instance_key=morphology_instance_key(case.blueprint, case.physical),
            base_height=float(stance_manifest["base_height"]),
            joint_qpos={str(key): float(value) for key, value in stance_manifest["joint_qpos"].items()},
            actuator_ctrl={str(key): float(value) for key, value in stance_manifest["actuator_ctrl_eq"].items()},
            root_xy=tuple(float(value) for value in stance_manifest["root_pose_eq"][:2]),
            root_quat=tuple(float(value) for value in stance_manifest["root_pose_eq"][3:7]),
            model_xml_sha256=str(stance_manifest["model_xml_sha256"]),
        )
        contact_profile_payload = profile
    tuples = resolve_motor_tuples(case_id, parents)
    action_amplitudes = derive_position_action_amplitudes(tuples)
    actuator_ranges = _case_actuator_ranges(overlay_record)
    stance_ctrl = {
        actuator.semantic_slot: float(stance.actuator_ctrl[actuator.semantic_slot])
        for actuator in case.blueprint.actuators
    }
    action_bounds = derive_position_action_bounds(tuples, stance_ctrl, actuator_ranges)
    _require(
        sha256_bytes(xml.encode()) == overlay_record["output_xml_sha256"],
        "Task071 bound XML changed",
    )
    if contact_profile is None:
        _require(
            stance.solution_hash == r1_record["stance_solution"]["sha256"],
            "Task071 stance solution changed",
        )
    else:
        _require(
            stance.solution_hash == artifacts["stance_payload"]["stance_solution_sha256"],
            "003b stance solution changed",
        )
    return {
        "case": case,
        "parents": parents,
        "physics": physics,
        "env_contract": env_contract,
        "overlay_record": overlay_record,
        "r1_record": r1_record,
        "xml": xml,
        "stance": stance,
        "motor_tuples": tuples,
        "action_amplitudes": action_amplitudes,
        "action_residual_bounds": action_bounds,
        "actuator_ranges": actuator_ranges,
        "contact_profile": contact_profile_payload,
    }


def _build_shard(
    context: dict[str, Any],
    *,
    num_envs: int,
    trial_seconds: float,
    seed: int,
    action_scale: float,
    action_amplitudes: dict[str, float] | None = None,
    action_residual_bounds: dict[str, tuple[float, float]] | None = None,
    phase_observation: bool = False,
) -> Any:
    from h200_locomotion_lab.envs.whole_body_mujoco import (
        WholeBodyMuJoCoShard,
        WholeBodyMuJoCoShardConfig,
    )
    from h200_locomotion_lab.robots.motor_process import MotorProcessConfig

    if action_amplitudes is not None and action_residual_bounds is not None:
        raise ValueError("action amplitudes and residual bounds are mutually exclusive")
    amplitude_mapping = action_amplitudes
    residual_mapping = (
        context["action_residual_bounds"]
        if action_amplitudes is None and action_residual_bounds is None
        else action_residual_bounds
    )
    config = WholeBodyMuJoCoShardConfig(
        control_hz=CONTROL_HZ,
        physics_hz=PHYSICS_HZ,
        trial_seconds=trial_seconds,
        context_trials=1,
        action_scale=action_scale,
        command_vx_range=(COMMAND[0], COMMAND[0]),
        command_vy_range=(COMMAND[1], COMMAND[1]),
        command_yaw_range=(COMMAND[2], COMMAND[2]),
        seed=seed,
        upright_threshold=math.cos(0.8),
        action_amplitude_by_slot=amplitude_mapping,
        action_residual_bounds_by_slot=residual_mapping,
        observation_joint_reference_by_slot={
            joint.semantic_slot: float(context["stance"].joint_qpos[joint.semantic_slot])
            for joint in context["case"].blueprint.joints
        },
        observation_joint_velocity_scale=0.05,
        observation_base_angular_velocity_scale=0.2,
        observation_phase=phase_observation,
        phase_period_s=0.8,
        logical_foot_groups=(
            context["contact_profile"]["logical_foot_groups"]
            if context.get("contact_profile") is not None
            else None
        ),
        logical_foot_reference_sites=(
            context["contact_profile"]["logical_foot_reference_sites"]
            if context.get("contact_profile") is not None
            else None
        ),
    )
    case = context["case"]
    return WholeBodyMuJoCoShard(
        case.blueprint,
        physical=case.physical,
        num_envs=num_envs,
        config=config,
        motor_config=MotorProcessConfig(
            control_hz=CONTROL_HZ,
            no_event_probability=1.0,
        ),
        model_xml=context["xml"],
        model_xml_sha256=context["overlay_record"]["output_xml_sha256"],
        stance_solution=context["stance"],
    )


class _Task072FixedCommandRewardBase:
    """Task-local fixed-command wrapper around the shared MuJoCo shard."""

    def __init__(self, shard: Any) -> None:
        self.shard = shard
        self.family = str(shard.blueprint.family)
        self._heading_reference = self._current_headings()
        self._height_reference = self._current_heights()

    @property
    def active_action_mask(self) -> Any:
        return self.shard.active_action_mask

    def reset(self) -> Any:
        observation = self.shard.reset()
        self._apply_runtime_command()
        self._write_command_observation(observation)
        self._heading_reference = self._current_headings()
        self._height_reference = self._current_heights()
        return observation

    def set_training_progress(self, progress: float) -> None:
        _require(math.isfinite(float(progress)), "training progress must be finite")

    def target_vx(self) -> float:
        return COMMAND[0]

    def _apply_runtime_command(self) -> None:
        self.shard._commands[:, :] = (self.target_vx(), COMMAND[1], COMMAND[2])

    def _write_command_observation(self, observation: Any) -> None:
        observation[:, 9:12] = (self.target_vx(), COMMAND[1], COMMAND[2])

    def _current_headings(self) -> Any:
        return self.shard.np.asarray(
            [
                _yaw_from_quaternion(self.shard._canonical_state(data).world_quaternion_wxyz)
                for data in self.shard.data
            ],
            dtype=self.shard.np.float64,
        )

    def _current_heights(self) -> Any:
        return self.shard.np.asarray(
            [self.shard._canonical_state(data).world_position[2] for data in self.shard.data],
            dtype=self.shard.np.float64,
        )


class Task072QuadrupedReward(_Task072FixedCommandRewardBase):
    """Frozen quadruped fixed-command shaping used by Go2."""

    def __init__(
        self,
        shard: Any,
        config: Task072QuadrupedRewardConfig | None = None,
        reward_scale: float = 1.0,
    ) -> None:
        super().__init__(shard)
        self.config = config or Task072QuadrupedRewardConfig()
        self.reward_scale = float(reward_scale)

    def step(self, action: Any) -> Any:
        np = self.shard.np
        self._apply_runtime_command()
        result = self.shard.step(action)
        self._apply_runtime_command()
        self._write_command_observation(result.actor_observation)
        self._write_command_observation(result.critic_observation)
        linear = np.asarray(
            result.metrics["post_step_pre_reset_local_linear_velocity"], dtype=np.float64
        )
        angular = np.asarray(
            result.metrics["post_step_pre_reset_local_angular_velocity"], dtype=np.float64
        )
        gravity = np.asarray(
            result.metrics["post_step_pre_reset_projected_gravity"], dtype=np.float64
        )
        heading = np.asarray(
            [
                _yaw_from_quaternion(quaternion)
                for quaternion in result.metrics["post_step_pre_reset_world_quaternion_wxyz"]
            ],
            dtype=np.float64,
        )
        heading_error = np.arctan2(
            np.sin(heading - self._heading_reference),
            np.cos(heading - self._heading_reference),
        )
        planar_error_squared = np.square(linear[:, 0] - self.target_vx()) + np.square(
            linear[:, 1] - COMMAND[1]
        )
        raw = {
            "track_xy": 4.0 * np.exp(-4.0 * planar_error_squared),
            "track_yaw": np.exp(-4.0 * np.square(angular[:, 2] - COMMAND[2])),
            "heading": 1.5 * np.exp(-2.0 * np.square(heading_error)),
            "upright": np.clip(-gravity[:, 2], 0.0, 1.0),
            "tilt": -2.0 * np.square(gravity[:, :2]).sum(axis=1),
            "nonfoot_contact": -np.asarray(
                result.metrics["non_foot_contact_fraction"], dtype=np.float64
            ),
        }
        non_foot_contact = np.asarray(result.metrics["non_foot_contact_fraction"], dtype=np.float64)
        previous_action = np.asarray(result.metrics["previous_action"], dtype=np.float64)
        action_delta = np.asarray(action, dtype=np.float64) - previous_action
        active_mask = np.asarray(result.active_action_mask, dtype=np.float64)
        raw["action_rate"] = (
            -0.05
            * np.square(action_delta * active_mask).sum(axis=1)
            / np.maximum(1.0, active_mask.sum(axis=1))
        )
        weighted = {key: raw_value * getattr(self.config, key) for key, raw_value in raw.items()}
        raw_total_reward = sum(weighted.values())
        reward = ppo_reward_from_raw(raw_total_reward, self.reward_scale)
        metrics = dict(result.metrics)
        metrics["raw_total_reward"] = raw_total_reward
        metrics["ppo_reward"] = reward
        metrics["reward_scale"] = self.reward_scale
        metrics["reward_components"] = {
            key: {"raw": raw[key], "weighted": weighted[key]} for key in raw
        }
        done = np.asarray(result.trial_done, dtype=bool)
        if bool(done.any()):
            self._heading_reference[done] = self._current_headings()[done]
        _ = non_foot_contact
        return replace(result, reward=reward, metrics=metrics)


class Task072BipedReward(_Task072FixedCommandRewardBase):
    """G1 biped pose/contact reward with auditable components."""

    def __init__(
        self,
        shard: Any,
        config: Task072BipedRewardConfig | None = None,
        reward_scale: float = 1.0,
    ) -> None:
        super().__init__(shard)
        self.config = config or Task072BipedRewardConfig()
        self.reward_scale = float(reward_scale)
        mapping = shard.embodiment.mapping
        self._slot_index = {
            slot: index for slot, index in zip(mapping.semantic_slots, mapping.selector)
        }
        self._nominal_pose = shard.np.asarray(
            shard.embodiment.scatter_joint_values(
                tuple(shard.stance_solution.joint_qpos[joint.semantic_slot] for joint in shard.blueprint.joints)
            ),
            dtype=shard.np.float64,
        )
        limits = [shard._joint_limits(joint) for joint in shard.blueprint.joints]
        self._joint_lower = shard.np.asarray(
            shard.embodiment.scatter_joint_values(tuple(lower for lower, _upper in limits)),
            dtype=shard.np.float64,
        )
        self._joint_upper = shard.np.asarray(
            shard.embodiment.scatter_joint_values(tuple(upper for _lower, upper in limits)),
            dtype=shard.np.float64,
        )
        self.semantic_groups = {
            "hip": tuple(index for slot, index in self._slot_index.items() if "_hip_" in slot),
            "knee": tuple(index for slot, index in self._slot_index.items() if "_knee_" in slot),
            "ankle": tuple(index for slot, index in self._slot_index.items() if "_ankle_" in slot),
            "waist": tuple(index for slot, index in self._slot_index.items() if slot.startswith("waist_")),
            "arm_wrist": tuple(index for slot, index in self._slot_index.items() if "arm" in slot or "wrist" in slot),
        }
        for name, indices in self.semantic_groups.items():
            _require(bool(indices), f"empty biped pose group: {name}")

    def _pose_raw(self, joint_position: Any, group: str) -> Any:
        indices = list(self.semantic_groups[group])
        error = joint_position[:, indices] - self._nominal_pose[indices]
        return -self.shard.np.square(error).mean(axis=1)

    def _desired_contact(self) -> Any:
        np = self.shard.np
        phase = phase_from_trial_step(
            self.shard._trial_step, self.shard.config.control_hz, self.config.period_s
        )
        offsets = np.asarray(
            (self.config.left_phase_offset, self.config.right_phase_offset),
            dtype=np.float64,
        )
        leg_phase = (phase[:, None] + offsets[None, :]) % 1.0
        return leg_phase < self.config.stance_threshold

    def step(self, action: Any) -> Any:
        np = self.shard.np
        self._apply_runtime_command()
        result = self.shard.step(action)
        self._apply_runtime_command()
        self._write_command_observation(result.actor_observation)
        self._write_command_observation(result.critic_observation)
        linear = np.asarray(result.metrics["post_step_pre_reset_local_linear_velocity"], dtype=np.float64)
        angular = np.asarray(result.metrics["post_step_pre_reset_local_angular_velocity"], dtype=np.float64)
        gravity = np.asarray(result.metrics["post_step_pre_reset_projected_gravity"], dtype=np.float64)
        height = np.asarray(result.metrics["post_step_pre_reset_world_position"], dtype=np.float64)[:, 2]
        joint_position = np.asarray(result.metrics["joint_position"], dtype=np.float64)
        joint_velocity = np.asarray(result.metrics["joint_velocity"], dtype=np.float64)
        foot_contact = np.asarray(result.metrics["foot_contact"], dtype=bool)
        foot_height = np.asarray(result.metrics["foot_height"], dtype=np.float64)
        foot_planar_speed = np.asarray(result.metrics["foot_planar_speed"], dtype=np.float64)
        foot_vertical_speed = np.asarray(result.metrics["foot_vertical_speed"], dtype=np.float64)
        foot_air_time = np.asarray(result.metrics["foot_air_time"], dtype=np.float64)
        touchdown = np.asarray(result.metrics["touchdown"], dtype=np.float64)
        previous_action = np.asarray(result.metrics["previous_action"], dtype=np.float64)
        active_mask = np.asarray(result.active_action_mask, dtype=np.float64)
        action_array = np.asarray(action, dtype=np.float64)
        non_foot_contact = np.asarray(result.metrics["non_foot_contact_fraction"], dtype=np.float64)
        command_speed = math.sqrt(self.target_vx() * self.target_vx() + COMMAND[1] * COMMAND[1])
        moving = command_speed > self.config.moving_command_threshold
        desired_contact = self._desired_contact()
        if desired_contact.shape != foot_contact.shape:
            raise ValueError("biped phase/contact shape mismatch")
        both_contact = foot_contact.all(axis=1)
        desired_both = desired_contact.all(axis=1)
        desired_swing = ~desired_contact
        lower_inner = self._joint_lower + 0.05 * (self._joint_upper - self._joint_lower)
        upper_inner = self._joint_upper - 0.05 * (self._joint_upper - self._joint_lower)
        low_violation = np.maximum(0.0, lower_inner - joint_position)
        high_violation = np.maximum(0.0, joint_position - upper_inner)
        normalized_violation = (low_violation + high_violation) / np.maximum(
            1e-9, self._joint_upper - self._joint_lower
        )
        clearance_score = np.exp(-np.square((foot_height - 0.10) / 0.05))
        clearance_mask = desired_swing & ~foot_contact
        clearance_denominator = clearance_mask.sum(axis=1)
        clearance = np.divide(
            (clearance_score * clearance_mask).sum(axis=1),
            clearance_denominator,
            out=np.zeros_like(clearance_denominator, dtype=np.float64),
            where=clearance_denominator > 0,
        )
        phase_match = (foot_contact == desired_contact).mean(axis=1)
        raw = {
            "track_xy": np.exp(
                -(
                    np.square(linear[:, 0] - self.target_vx())
                    + np.square(linear[:, 1] - COMMAND[1])
                    + np.square(linear[:, 2])
                )
                / 0.25
            ),
            "track_yaw": np.exp(-np.square(angular[:, 2] - COMMAND[2]) / 0.25),
            "upright": np.clip(-gravity[:, 2], 0.0, 1.0),
            "tilt": -np.square(gravity[:, :2]).sum(axis=1),
            "height": -np.square((height - self._height_reference) / 0.10),
            "stand_support": foot_contact.any(axis=1).astype(np.float64) if not moving else np.zeros(result.reward.shape),
            "phase_gait": phase_match if moving else np.zeros(result.reward.shape),
            "out_of_phase_double_support": (
                -(both_contact & ~desired_both).astype(np.float64)
                if moving
                else np.zeros(result.reward.shape)
            ),
            "clearance": clearance if moving else np.zeros(result.reward.shape),
            "touchdown_airtime": (touchdown * np.clip(foot_air_time / 0.5, 0.0, 1.0)).mean(axis=1) if moving else np.zeros(result.reward.shape),
            "soft_landing": (touchdown * np.exp(-np.square(foot_vertical_speed / 0.5))).mean(axis=1) if moving else np.zeros(result.reward.shape),
            "foot_slip": -(foot_contact * np.square(foot_planar_speed)).mean(axis=1),
            "nonfoot_contact": -non_foot_contact,
            "pose_hip": self._pose_raw(joint_position, "hip"),
            "pose_knee": self._pose_raw(joint_position, "knee"),
            "pose_ankle": self._pose_raw(joint_position, "ankle"),
            "pose_waist": self._pose_raw(joint_position, "waist"),
            "pose_arm_wrist": self._pose_raw(joint_position, "arm_wrist"),
            "joint_velocity": -(
                np.square(joint_velocity * active_mask).sum(axis=1)
                / np.maximum(1.0, active_mask.sum(axis=1))
            ),
            "joint_limit": -(
                np.square(normalized_violation * active_mask).sum(axis=1)
                / np.maximum(1.0, active_mask.sum(axis=1))
            ),
            "action_magnitude": -(
                np.square(action_array * active_mask).sum(axis=1)
                / np.maximum(1.0, active_mask.sum(axis=1))
            ),
            "action_rate": -(
                np.square((action_array - previous_action) * active_mask).sum(axis=1)
                / np.maximum(1.0, active_mask.sum(axis=1))
            ),
            "base_angvel_xy": -(np.square(angular[:, 0]) + np.square(angular[:, 1])),
        }
        weighted = {key: raw_value * getattr(self.config, key) for key, raw_value in raw.items()}
        raw_total_reward = sum(weighted.values())
        reward = ppo_reward_from_raw(raw_total_reward, self.reward_scale)
        metrics = dict(result.metrics)
        metrics["raw_total_reward"] = raw_total_reward
        metrics["ppo_reward"] = reward
        metrics["reward_scale"] = self.reward_scale
        metrics["reward_components"] = {
            key: {"raw": raw[key], "weighted": weighted[key]} for key in raw
        }
        metrics["biped_phase_contact"] = {
            "desired_contact": desired_contact,
            "left_single_support": foot_contact[:, 0] & ~foot_contact[:, 1],
            "right_single_support": foot_contact[:, 1] & ~foot_contact[:, 0],
            "both_contact": both_contact,
            "no_contact": ~foot_contact.any(axis=1),
            "moving_branch": moving,
        }
        return replace(result, reward=reward, metrics=metrics)


Task072LocomotionReward = Task072QuadrupedReward


def _make_task072_reward(shard: Any, *, reward_scale: float = 1.0) -> Any:
    if str(shard.blueprint.family) == "biped":
        return Task072BipedReward(shard, reward_scale=reward_scale)
    if str(shard.blueprint.family) == "quadruped":
        return Task072QuadrupedReward(shard, reward_scale=reward_scale)
    raise ValueError(f"unsupported Task072 family: {shard.blueprint.family}")


def reward_config_payload(case_id: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or _load_bound_context(case_id)
    shard = _build_shard(
        context,
        num_envs=1,
        trial_seconds=1.0,
        seed=72072,
        action_scale=0.35,
    )
    reward = _make_task072_reward(shard)
    if isinstance(reward, Task072BipedReward):
        payload = {
            "schema_version": 1,
            "case_id": case_id,
            "family": "biped",
            "reward_version": reward.config.version,
            "weights": asdict(reward.config),
            "semantic_groups": {
                name: [slot for slot, index in reward._slot_index.items() if index in indices]
                for name, indices in reward.semantic_groups.items()
            },
            "contact_mapping": {
                "foot_geoms": list(sorted(shard._foot_geoms)),
                "foot_geom_ids": [int(value) for value in shard._foot_geom_ids],
            },
            "fall_penalty": None,
        }
    elif isinstance(reward, Task072QuadrupedReward):
        payload = {
            "schema_version": 1,
            "case_id": case_id,
            "family": "quadruped",
            "reward_version": reward.config.version,
            "weights": asdict(reward.config),
            "components": [
                "track_xy",
                "track_yaw",
                "heading",
                "upright",
                "tilt",
                "nonfoot_contact",
                "action_rate",
            ],
            "contact_mapping": {
                "foot_geoms": list(sorted(shard._foot_geoms)),
                "foot_geom_ids": [int(value) for value in shard._foot_geom_ids],
            },
            "fall_penalty": None,
        }
    else:  # pragma: no cover - defensive type narrowing
        raise ValueError("unsupported Task072 reward")
    payload["source_sha256"] = sha256_path(Path(__file__))
    payload["config_sha256"] = payload_sha256(payload)
    return payload


def _static_lineage(
    case_id: str,
    context: dict[str, Any],
    shard: Any,
    *,
    action_scale: float,
    phase_observation: bool = False,
) -> dict[str, Any]:
    from h200_locomotion_lab.robots.whole_body_slots import (
        WHOLE_BODY_ACTION_DIM,
        WHOLE_BODY_ACTOR_OBS_DIM,
        WHOLE_BODY_SCHEMA_HASH,
        WHOLE_BODY_SCHEMA_VERSION,
    )

    case = context["case"]
    r1_record = context["r1_record"]
    sources = {
        "task072_cli": Path(__file__),
        "ppo_kernel": ROOT / "src/h200_locomotion_lab/algorithms/ppo.py",
        "masked_distribution": ROOT / "src/h200_locomotion_lab/masked_distribution.py",
        "environment": ROOT / "src/h200_locomotion_lab/envs/whole_body_mujoco.py",
        "trainer": ROOT / "src/h200_locomotion_lab/training/whole_body_ppo.py",
        "policy": ROOT / "src/h200_locomotion_lab/policies/whole_body_mlp.py",
        "pyproject": ROOT / "pyproject.toml",
        "uv_lock": ROOT / "uv.lock",
    }
    parent_lineage = {
        label: {
            "path": str(item["path"].relative_to(ROOT)),
            "raw_sha256": item["raw_sha256"],
            "payload_sha256": item["payload_sha256"],
        }
        for label, item in context["parents"].items()
    }
    return {
        "artifact": ARTIFACT_VERSION,
        "case": case_id,
        "family": case.spec["family"],
        "asset": {
            "descriptor_sha256": case.spec["descriptor_sha256"],
            "manifest_sha256": case.spec["manifest_sha256"],
            "frozen_xml_sha256": case.spec["frozen_xml_sha256"],
            "structural_descriptor_sha256": case.spec["structural_descriptor_sha256"],
            "blueprint_hash": case.spec["blueprint_hash"],
            "physical_hash": case.spec["physical_hash"],
            "bound_xml_sha256": context["overlay_record"]["output_xml_sha256"],
            "parent_bound_xml_sha256": context["overlay_record"].get("parent_output_xml_sha256"),
            "stance_profile_sha256": r1_record["stance_profile"]["sha256"],
            "stance_solution_sha256": context["stance"].solution_hash,
            "contact_profile_id": (
                context["contact_profile"]["contact_profile_id"]
                if context.get("contact_profile") is not None
                else None
            ),
            "contact_profile_sha256": (
                payload_sha256(context["contact_profile"])
                if context.get("contact_profile") is not None
                else None
            ),
        },
        "schema": {
            "version": WHOLE_BODY_SCHEMA_VERSION,
            "hash": WHOLE_BODY_SCHEMA_HASH,
            "action_dim": WHOLE_BODY_ACTION_DIM,
            "actor_observation_dim": WHOLE_BODY_ACTOR_OBS_DIM + int(shard.config.observation_phase) * 2,
            "active_action_count": int(shard.active_action_mask[0].sum()),
            "active_action_mask_sha256": sha256_bytes(
                bytes(bool(value) for value in shard.active_action_mask[0].tolist())
            ),
        },
        "task": {
            "command": list(COMMAND),
            "control_hz": CONTROL_HZ,
            "physics_hz": PHYSICS_HZ,
            "action_scale": action_scale,
            "action_residual_formula": ACTION_CONTRACT_VERSION,
            "action_residual_bounds_by_slot_sha256": payload_sha256(context["action_residual_bounds"]),
            "observation_transform": {
                "version": OBSERVATION_TRANSFORM_VERSION,
                "joint_reference_sha256": payload_sha256({
                    joint.semantic_slot: float(context["stance"].joint_qpos[joint.semantic_slot])
                    for joint in case.blueprint.joints
                }),
                "joint_velocity_scale": 0.05,
                "base_angular_velocity_scale": 0.2,
            },
            "phase_observation": {
                "enabled": bool(shard.config.observation_phase),
                "clock_source": "WholeBodyMuJoCoShard._trial_step/control_hz",
                "period_s": 0.8,
                "schema": "193->195" if shard.config.observation_phase else "193",
            },
            "runtime_motor_faults": "disabled",
            "training_reward": TRAINING_REWARD_VERSION,
            "case_reward": (
                BIPED_REWARD_VERSION
                if case.spec["family"] == "biped"
                else QUADRUPED_REWARD_VERSION
            ),
            "final_eval_seconds": FINAL_EVAL_SECONDS,
            "final_eval_envs_minimum": FINAL_EVAL_ENVS,
        },
        "parent_artifacts": parent_lineage,
        "sources": {
            label: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_path(path),
            }
            for label, path in sources.items()
        },
    }


def _make_policy(shard: Any, config: dict[str, Any], device: str) -> Any:
    from h200_locomotion_lab.policies.whole_body_mlp import (
        WholeBodyMLPActorCritic,
        WholeBodyMLPConfig,
    )
    from h200_locomotion_lab.robots.whole_body_slots import WHOLE_BODY_ACTOR_OBS_DIM

    policy_config = WholeBodyMLPConfig(
        obs_dim=int(config.get("actor_observation_dim", WHOLE_BODY_ACTOR_OBS_DIM)),
        hidden_dim=int(config["hidden_dim"]),
        hidden_layers=int(config["hidden_layers"]),
        log_std_init=float(config["log_std_init"]),
    )
    return WholeBodyMLPActorCritic(
        policy_config,
        action_mask=shard.active_action_mask,
        device=device,
    )


def _phase_observation_enabled(config: dict[str, Any]) -> bool:
    phase_record = config.get("phase_observation")
    return bool(config.get("variant_id") == "E1_phase" or (
        isinstance(phase_record, dict) and phase_record.get("enabled") is True
    ))


def _checkpoint_lineage(
    *,
    case_id: str,
    static_lineage_sha256: str,
    run_identity_sha256: str,
    update: int,
    env_steps: int,
) -> dict[str, Any]:
    return {
        "artifact": ARTIFACT_VERSION,
        "case": case_id,
        "static_lineage_sha256": static_lineage_sha256,
        "run_identity_sha256": run_identity_sha256,
        "global_update": update,
        "env_steps": env_steps,
    }


def _save_checkpoint(
    torch: Any,
    path: Path,
    *,
    policy: Any,
    optimizer: Any,
    lineage: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "artifact": ARTIFACT_VERSION,
        "lineage": lineage,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    _atomic_torch_save(torch, path, payload)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_path(path),
        "global_update": lineage["global_update"],
        "env_steps": lineage["env_steps"],
    }


def validate_checkpoint_lineage(
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    if observed != expected:
        raise ValueError("Task072 checkpoint lineage mismatch")


def _load_checkpoint(
    torch: Any,
    reference: dict[str, Any],
    expected_lineage: dict[str, Any],
) -> dict[str, Any]:
    path = Path(reference["path"])
    _require(path.is_file(), f"missing Task072 checkpoint: {path}")
    _require(sha256_path(path) == reference["sha256"], "Task072 checkpoint SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(payload.get("artifact") == ARTIFACT_VERSION, "Task072 checkpoint artifact drift")
    validate_checkpoint_lineage(payload.get("lineage"), expected_lineage)
    return payload


def validate_progression(
    progression: dict[str, Any],
    *,
    expected_updates: int,
    expected_env_steps: int,
) -> None:
    reports = progression.get("reports")
    checkpoints = progression.get("checkpoints")
    _require(
        isinstance(reports, list) and len(reports) == expected_updates, "progression report count"
    )
    _require(isinstance(checkpoints, list) and len(checkpoints) >= 2, "progression checkpoints")
    report_updates = [int(report.get("global_update", -1)) for report in reports]
    _require(report_updates == list(range(1, expected_updates + 1)), "progression update order")
    checkpoint_updates = [int(item.get("global_update", -1)) for item in checkpoints]
    _require(checkpoint_updates[0] == 0, "progression lacks update0")
    _require(checkpoint_updates[-1] == expected_updates, "progression lacks final update")
    _require(
        all(right > left for left, right in itertools.pairwise(checkpoint_updates)),
        "progression checkpoint updates are not strictly increasing",
    )
    _require(
        int(checkpoints[-1].get("env_steps", -1)) == expected_env_steps,
        "progression final env-step count mismatch",
    )
    numeric_values = [
        float(value)
        for report in reports
        for key, value in report.items()
        if key not in {"global_update"} and isinstance(value, (int, float))
    ]
    _require(
        all(math.isfinite(value) for value in numeric_values), "progression contains nonfinite data"
    )


def _validate_run_manifest(
    path: Path,
    *,
    expected_case: str,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    _require(path.is_file(), f"missing Task072 run manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest.get("artifact") == ARTIFACT_VERSION, "run manifest artifact drift")
    _require(manifest.get("case") == expected_case, "run manifest case mismatch")
    config = manifest["configuration"]
    contact_profile = config.get("contact_profile_id")
    context = _load_bound_context(expected_case, contact_profile=contact_profile)
    shard = _build_shard(
        context,
        num_envs=1,
        trial_seconds=float(config["training_trial_seconds"]),
        seed=int(config["seed"]),
        action_scale=float(config["action_scale"]),
        phase_observation=_phase_observation_enabled(config),
    )
    current_lineage = _static_lineage(
        expected_case,
        context,
        shard,
        action_scale=float(config["action_scale"]),
    )
    _require(manifest.get("static_lineage") == current_lineage, "run static lineage drift")
    _require(
        manifest.get("static_lineage_sha256") == payload_sha256(current_lineage),
        "run static lineage SHA mismatch",
    )
    _require(
        manifest.get("run_identity_sha256") == payload_sha256(manifest["run_identity"]),
        "run identity SHA mismatch",
    )
    _require(
        manifest["run_identity"]
        == {
            "artifact": ARTIFACT_VERSION,
            "case": expected_case,
            "static_lineage_sha256": manifest["static_lineage_sha256"],
            "configuration": config,
        },
        "run identity content mismatch",
    )
    progression_reference = manifest["progression"]
    progression_path = Path(progression_reference["path"])
    _require(progression_path.is_file(), "run progression artifact is missing")
    _require(
        sha256_path(progression_path) == progression_reference["sha256"],
        "run progression SHA mismatch",
    )
    progression = json.loads(progression_path.read_text(encoding="utf-8"))
    validate_progression(
        progression,
        expected_updates=int(config["updates"]),
        expected_env_steps=int(config["env_steps"]),
    )
    _require(
        progression["checkpoints"][0] == manifest["update0_checkpoint"],
        "progression/update0 checkpoint mismatch",
    )
    _require(
        progression["final_checkpoint"] == manifest["final_checkpoint"],
        "progression/final checkpoint mismatch",
    )
    return manifest, context, shard


def _policy_action_provider(policy: Any, shard: Any, device: str) -> Callable[[Any], Any]:
    torch = policy.torch
    active_mask = torch.as_tensor(shard.active_action_mask, dtype=torch.bool, device=device)

    def provide(observation: Any) -> Any:
        with torch.no_grad():
            action, _log_prob, _value, _entropy = policy.act(
                torch.as_tensor(observation, dtype=torch.float32, device=device),
                deterministic=True,
                active_mask=active_mask,
            )
        return action.cpu().numpy()

    return provide


def evaluate_first_trials(
    shard: Any,
    action_provider: Callable[[Any], Any],
) -> dict[str, Any]:
    """Evaluate each environment's first trial without reset-state contamination."""

    import numpy as np

    observation = shard.reset()
    num_envs = int(shard.num_envs)
    active = np.ones(num_envs, dtype=bool)
    fallen = np.zeros(num_envs, dtype=bool)
    completed_steps = np.zeros(num_envs, dtype=np.int64)
    planar_sum = np.zeros(num_envs, dtype=np.float64)
    yaw_sum = np.zeros(num_envs, dtype=np.float64)
    heading_sum = np.zeros(num_envs, dtype=np.float64)
    gravity_sum = np.zeros(num_envs, dtype=np.float64)
    starts = np.asarray(
        [shard._canonical_state(data).world_position for data in shard.data],
        dtype=np.float64,
    )
    heading_reference = np.asarray(
        [
            _yaw_from_quaternion(shard._canonical_state(data).world_quaternion_wxyz)
            for data in shard.data
        ],
        dtype=np.float64,
    )
    terminals = starts.copy()
    steps_executed = 0
    for _step in range(int(shard.config.trial_steps)):
        action = np.asarray(action_provider(observation), dtype=np.float64)
        _require(action.shape == (num_envs, 45), "Task072 action provider shape mismatch")
        action[~active] = 0.0
        active_before = active.copy()
        result = shard.step(action)
        positions = np.asarray(
            result.metrics["post_step_pre_reset_world_position"], dtype=np.float64
        )
        linear = np.asarray(
            result.metrics["post_step_pre_reset_local_linear_velocity"], dtype=np.float64
        )
        angular = np.asarray(
            result.metrics["post_step_pre_reset_local_angular_velocity"], dtype=np.float64
        )
        projected_gravity = np.asarray(
            result.metrics["post_step_pre_reset_projected_gravity"], dtype=np.float64
        )
        heading = np.asarray(
            [
                _yaw_from_quaternion(quaternion)
                for quaternion in result.metrics["post_step_pre_reset_world_quaternion_wxyz"]
            ],
            dtype=np.float64,
        )
        heading_error = np.abs(
            np.arctan2(
                np.sin(heading - heading_reference),
                np.cos(heading - heading_reference),
            )
        )
        planar_error = np.linalg.norm(linear[:, :2] - np.asarray(COMMAND[:2]), axis=1)
        planar_sum[active_before] += planar_error[active_before]
        yaw_sum[active_before] += np.abs(angular[active_before, 2] - COMMAND[2])
        heading_sum[active_before] += heading_error[active_before]
        gravity_sum[active_before] += np.linalg.norm(projected_gravity[active_before, :2], axis=1)
        completed_steps[active_before] += 1
        done = np.asarray(result.trial_done, dtype=bool) & active_before
        terminals[done] = positions[done]
        fallen[done] = np.asarray(result.metrics["fall"], dtype=bool)[done]
        active[done] = False
        observation = result.actor_observation
        steps_executed += 1
        if not bool(active.any()):
            break
    _require(not bool(active.any()), "Task072 evaluation did not complete every first trial")
    _require(bool((completed_steps > 0).all()), "Task072 evaluation contains empty trials")
    per_trial_planar = planar_sum / completed_steps
    per_trial_yaw = yaw_sum / completed_steps
    per_trial_heading = heading_sum / completed_steps
    per_trial_gravity = gravity_sum / completed_steps
    displacement = terminals[:, 0] - starts[:, 0]
    nonfall_displacement = displacement[~fallen]
    finite_values = np.concatenate(
        (
            per_trial_planar,
            per_trial_yaw,
            per_trial_heading,
            per_trial_gravity,
            displacement,
        )
    )
    return {
        "trials": num_envs,
        "completed_trials": int((~active).sum()),
        "steps_executed": steps_executed,
        "control_hz": float(shard.config.control_hz),
        "trial_seconds": float(shard.config.trial_seconds),
        "zero_fall_ratio": float(np.mean(~fallen)),
        "fall_count": int(fallen.sum()),
        "planar_velocity_error": float(np.mean(per_trial_planar)),
        "yaw_error": float(np.mean(per_trial_yaw)),
        "heading_error": float(np.mean(per_trial_heading)),
        "gravity_xy": float(np.mean(per_trial_gravity)),
        "forward_displacement_mean": float(np.mean(displacement)),
        "nonfall_forward_displacement_mean": (
            float(np.mean(nonfall_displacement)) if nonfall_displacement.size else None
        ),
        "trial_step_counts": completed_steps.tolist(),
        "finite": bool(np.isfinite(finite_values).all()),
    }


def _initial_state_sha(shard: Any, env_id: int) -> str:
    data = shard.data[env_id]
    return payload_sha256(
        {
            "qpos": [float(value) for value in data.qpos],
            "qvel": [float(value) for value in data.qvel],
            "command": [float(value) for value in shard._commands[env_id]],
        }
    )


def evaluate_first_trial_trace(
    shard: Any,
    action_provider: Callable[[Any], Any],
    *,
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    observation = shard.reset()
    num_envs = int(shard.num_envs)
    horizon = int(shard.config.trial_steps)
    active = np.ones(num_envs, dtype=bool)
    terminal_x = np.asarray(
        [shard._canonical_state(data).world_position[0] for data in shard.data],
        dtype=np.float64,
    )
    records = [
        {
            "seed": seed + 1009 * env_id,
            "initial_state_sha256": _initial_state_sha(shard, env_id),
            "alive_before": [],
            "trial_done": [],
            "fall": [],
            "local_linear_velocity_xyz": [],
            "local_yaw_velocity": [],
            "projected_gravity_xy": [],
            "canonical_root_world_x": [],
        }
        for env_id in range(num_envs)
    ]
    for _step in range(horizon):
        active_before = active.copy()
        action = np.asarray(action_provider(observation), dtype=np.float64)
        _require(action.shape == (num_envs, 45), "Task072 action provider shape mismatch")
        action[~active_before] = 0.0
        result = shard.step(action)
        linear = np.asarray(result.metrics["post_step_pre_reset_local_linear_velocity"], dtype=np.float64)
        angular = np.asarray(result.metrics["post_step_pre_reset_local_angular_velocity"], dtype=np.float64)
        gravity = np.asarray(result.metrics["post_step_pre_reset_projected_gravity"], dtype=np.float64)
        world_x = np.asarray(result.metrics["post_step_pre_reset_world_position"], dtype=np.float64)[:, 0]
        done = np.asarray(result.trial_done, dtype=bool) & active_before
        fall = np.asarray(result.metrics["fall"], dtype=bool) & active_before
        terminal_x[done] = world_x[done]
        for env_id, record in enumerate(records):
            record["alive_before"].append(bool(active_before[env_id]))
            record["trial_done"].append(bool(done[env_id]))
            record["fall"].append(bool(fall[env_id]))
            if active_before[env_id]:
                record["local_linear_velocity_xyz"].append([float(value) for value in linear[env_id]])
                record["local_yaw_velocity"].append(float(angular[env_id, 2]))
                record["projected_gravity_xy"].append([float(value) for value in gravity[env_id, :2]])
                record["canonical_root_world_x"].append(float(world_x[env_id]))
            else:
                record["local_linear_velocity_xyz"].append([0.0, 0.0, 0.0])
                record["local_yaw_velocity"].append(0.0)
                record["projected_gravity_xy"].append([0.0, 0.0])
                record["canonical_root_world_x"].append(float(terminal_x[env_id]))
        active[done] = False
        observation = result.actor_observation
    return {"seed": seed, "num_envs": num_envs, "horizon": horizon, "records": records}


def _alive_prefix_count(record: dict[str, Any]) -> int:
    count = 0
    for alive in record["alive_before"]:
        if not alive:
            break
        count += 1
    return max(1, count)


def metrics_from_trace(policy_trace: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    planar_errors = []
    yaw_errors = []
    gravity = []
    displacements = []
    fallen = []
    step_counts = []
    for record in policy_trace["records"]:
        n = _alive_prefix_count(record)
        linear = np.asarray(record["local_linear_velocity_xyz"][:n], dtype=np.float64)
        yaw = np.asarray(record["local_yaw_velocity"][:n], dtype=np.float64)
        gxy = np.asarray(record["projected_gravity_xy"][:n], dtype=np.float64)
        root_x = np.asarray(record["canonical_root_world_x"], dtype=np.float64)
        planar_errors.append(float(np.linalg.norm(linear[:, :2] - np.asarray(COMMAND[:2]), axis=1).mean()))
        yaw_errors.append(float(np.abs(yaw - COMMAND[2]).mean()))
        gravity.append(float(np.linalg.norm(gxy, axis=1).mean()))
        displacements.append(float(root_x[-1] - root_x[0]))
        fall_trace = record.get("fall")
        if fall_trace is None:
            fall_trace = record["trial_done"]
        fallen.append(any(fall_trace[index] for index in range(min(n, len(fall_trace)))))
        step_counts.append(n)
    nonfall_displacement = [value for value, fall in zip(displacements, fallen) if not fall]
    finite_values = [*planar_errors, *yaw_errors, *gravity, *displacements]
    return {
        "trials": int(policy_trace["num_envs"]),
        "completed_trials": int(policy_trace["num_envs"]),
        "steps_executed": int(policy_trace["horizon"]),
        "control_hz": CONTROL_HZ,
        "trial_seconds": float(policy_trace["horizon"]) / CONTROL_HZ,
        "zero_fall_ratio": float(1.0 - sum(fallen) / max(1, len(fallen))),
        "fall_count": int(sum(fallen)),
        "planar_velocity_error": float(np.mean(planar_errors)),
        "yaw_error": float(np.mean(yaw_errors)),
        "heading_error": 0.0,
        "gravity_xy": float(np.mean(gravity)),
        "forward_displacement_mean": float(np.mean(displacements)),
        "nonfall_forward_displacement_mean": (
            float(np.mean(nonfall_displacement)) if nonfall_displacement else None
        ),
        "trial_step_counts": step_counts,
        "finite": bool(np.isfinite(np.asarray(finite_values, dtype=np.float64)).all()),
    }


def common_alive_prefix_planar_margin(
    selected_trace: dict[str, Any],
    baseline_trace: dict[str, Any],
) -> float:
    import numpy as np

    margins = []
    for selected, baseline in zip(selected_trace["records"], baseline_trace["records"]):
        k = min(_alive_prefix_count(selected), _alive_prefix_count(baseline))
        selected_v = np.asarray(selected["local_linear_velocity_xyz"][:k], dtype=np.float64)[:, :2]
        baseline_v = np.asarray(baseline["local_linear_velocity_xyz"][:k], dtype=np.float64)[:, :2]
        target = np.asarray(COMMAND[:2], dtype=np.float64)
        selected_error = np.linalg.norm(selected_v - target, axis=1).mean()
        baseline_error = np.linalg.norm(baseline_v - target, axis=1).mean()
        margins.append(float(baseline_error - selected_error))
    return float(np.mean(margins))


def _zero_action_provider(num_envs: int) -> Callable[[Any], Any]:
    import numpy as np

    def provide(_observation: Any) -> Any:
        return np.zeros((num_envs, 45), dtype=np.float32)

    return provide


def _video_evidence(
    sidecar_path: Path | None,
    *,
    case_id: str,
    checkpoint_sha256: str,
) -> tuple[bool, dict[str, Any] | None]:
    if sidecar_path is None:
        return False, None
    _require(sidecar_path.is_file(), f"video sidecar is missing: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    video_path = Path(sidecar["video_path"])
    midframe_path = Path(sidecar["midframe_path"])
    verified = bool(
        sidecar.get("case") == case_id
        and sidecar.get("checkpoint_sha256") == checkpoint_sha256
        and sidecar.get("fps") == 50
        and sidecar.get("frame_count") == VIDEO_FRAMES
        and math.isclose(float(sidecar.get("duration_seconds", 0.0)), VIDEO_SECONDS)
        and sidecar.get("render_passed") is True
        and sidecar.get("fall_count") == 0
        and sidecar.get("done_count") == 0
        and float(sidecar.get("forward_displacement", 0.0)) > 0.0
        and video_path.is_file()
        and midframe_path.is_file()
        and sha256_path(video_path) == sidecar.get("video_sha256")
        and sha256_path(midframe_path) == sidecar.get("midframe_sha256")
    )
    return verified, sidecar


def _diagnostic_rollout_summary(rows: list[dict[str, Any]], *, active_slot_count: int) -> dict[str, Any]:
    import numpy as np

    _require(len(rows) == VIDEO_FRAMES, "diagnostic rollout must contain 400 rows")
    contacts = np.asarray([row["foot_contact"][:2] for row in rows], dtype=bool)
    touchdowns = np.asarray([row["touchdown"][:2] for row in rows], dtype=bool)
    gravity = np.asarray([row["projected_gravity_xy"] for row in rows], dtype=np.float64)
    root_x = np.asarray([row["canonical_root_world_x"] for row in rows], dtype=np.float64)
    would = np.asarray([row["target_would_clamp_by_slot"] for row in rows], dtype=bool)
    actual = np.asarray([row["actual_clamp_by_slot"] for row in rows], dtype=bool)
    labels = []
    touchdown_counts = [0, 0]
    previous_label = None
    trial_local_step = 0
    for index, row in enumerate(rows):
        if bool(row["trial_done"]):
            previous_label = None
            trial_local_step = 0
            continue
        current = touchdowns[index]
        if trial_local_step <= 0 or int(current.sum()) != 1:
            trial_local_step += 1
            continue
        label = "left" if bool(current[0]) else "right"
        touchdown_counts[0 if label == "left" else 1] += 1
        if label != previous_label:
            labels.append(label)
            previous_label = label
        trial_local_step += 1
    return {
        "frame_count": len(rows),
        "left_noninitial_touchdown_count": touchdown_counts[0],
        "right_noninitial_touchdown_count": touchdown_counts[1],
        "alternating_touchdown_transitions": max(0, len(labels) - 1),
        "left_single_support_fraction": float(np.mean(contacts[:, 0] & ~contacts[:, 1])),
        "right_single_support_fraction": float(np.mean(contacts[:, 1] & ~contacts[:, 0])),
        "both_contact_fraction": float(np.mean(contacts[:, 0] & contacts[:, 1])),
        "no_contact_fraction": float(np.mean(~contacts.any(axis=1))),
        "target_would_clamp_fraction": float(would.sum() / max(1, VIDEO_FRAMES * active_slot_count)),
        "actual_clamp_fraction": float(actual.sum() / max(1, VIDEO_FRAMES * active_slot_count)),
        "max_slot_target_would_clamp_fraction": float(would.mean(axis=0).max()) if would.size else 0.0,
        "max_slot_actual_clamp_fraction": float(actual.mean(axis=0).max()) if actual.size else 0.0,
        "forward_displacement": float(root_x[-1] - root_x[0]),
        "mean_projected_gravity_xy": float(np.linalg.norm(gravity, axis=1).mean()),
        "fall_count": int(sum(bool(row["fall"]) for row in rows)),
        "done_count": int(sum(bool(row["trial_done"]) for row in rows)),
    }


def pilot_gate_payload(report: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, Any]:
    progression_path = Path(report["run_manifest"]["path"]).with_name("progression.json")
    progression = json.loads(progression_path.read_text(encoding="utf-8"))
    reports = progression.get("reports")
    _require(isinstance(reports, list) and len(reports) == 1000, "pilot progression must contain 1000 update rows")
    raw_integrity = True
    raw_integrity_reasons: list[str] = []
    for row in reports:
        records = row.get("minibatches")
        attempted = row.get("minibatches_attempted")
        completed = row.get("minibatches_completed")
        if not isinstance(records, list) or int(attempted) != len(records):
            raw_integrity = False
            raw_integrity_reasons.append("minibatches_attempted_mismatch")
            continue
        applied = sum(record.get("applied") is True for record in records)
        if applied != int(completed):
            raw_integrity = False
            raw_integrity_reasons.append("minibatches_completed_mismatch")
        for record in records:
            if not all(
                isinstance(record.get(key), (int, float)) and math.isfinite(float(record[key]))
                for key in ("approx_kl", "clip_fraction")
            ):
                raw_integrity = False
                raw_integrity_reasons.append("nonfinite_minibatch_sample")
                break
    minibatches = [
        record
        for row in reports
        for record in row.get("minibatches", [])
    ]
    approx_kl = [float(record["approx_kl"]) for record in minibatches]
    clip_fraction = [float(record["clip_fraction"]) for record in minibatches]
    _require(approx_kl and clip_fraction, "pilot progression has no PPO minibatch diagnostics")

    def nearest_rank(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
        return ordered[index]

    early_stop_fraction = sum(bool(row.get("early_stopped")) for row in progression["reports"]) / max(1, len(progression["reports"]))
    selected = report["final_policy"]
    update0 = report["update0_baseline"]
    zero = report["zero_action_baseline"]
    median_selected = sorted(float(value) / CONTROL_HZ for value in selected["trial_step_counts"])[len(selected["trial_step_counts"]) // 2]
    median_update0 = sorted(float(value) / CONTROL_HZ for value in update0["trial_step_counts"])[len(update0["trial_step_counts"]) // 2]
    gates = {
        "touchdowns": diagnostic["left_noninitial_touchdown_count"] >= 4 and diagnostic["right_noninitial_touchdown_count"] >= 4,
        "alternating": diagnostic["alternating_touchdown_transitions"] >= 6,
        "single_support": diagnostic["left_single_support_fraction"] >= 0.05 and diagnostic["right_single_support_fraction"] >= 0.05,
        "contact_fractions": diagnostic["both_contact_fraction"] <= 0.70 and diagnostic["no_contact_fraction"] <= 0.20,
        "clamp": (
            diagnostic["target_would_clamp_fraction"] <= 1e-6
            and diagnostic["actual_clamp_fraction"] <= 1e-6
            and diagnostic["max_slot_target_would_clamp_fraction"] <= 1e-6
            and diagnostic["max_slot_actual_clamp_fraction"] <= 1e-6
        ),
        "kl": (
            raw_integrity
            and sum(approx_kl) / len(approx_kl) <= 0.015
            and nearest_rank(approx_kl, 0.95) <= 0.03
            and max(approx_kl) <= 0.05
            and early_stop_fraction <= 0.50
        ),
        "clip_fraction": (
            sum(clip_fraction) / len(clip_fraction) <= 0.20
            and nearest_rank(clip_fraction, 0.95) <= 0.35
        ),
        "learning_margin": (
            float(report["comparisons"]["update0"]) >= 0.05
            and float(report["comparisons"]["zero_action"]) >= 0.05
        ),
        "survival_improved": median_selected >= 3.0 and median_selected - median_update0 >= 1.0,
        "finite": selected.get("finite") is True,
        "diagnostic_motion": diagnostic["forward_displacement"] >= 0.50 and diagnostic["mean_projected_gravity_xy"] <= 0.45,
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "case_id": report["case"],
        "stage": "pilot",
        "diagnostic": diagnostic,
        "ppo": {
            "raw_integrity": raw_integrity,
            "raw_integrity_reasons": sorted(set(raw_integrity_reasons)),
            "approx_kl_mean": sum(approx_kl) / len(approx_kl),
            "approx_kl_p95": nearest_rank(approx_kl, 0.95),
            "approx_kl_max": max(approx_kl),
            "clip_fraction_mean": sum(clip_fraction) / len(clip_fraction),
            "clip_fraction_p95": nearest_rank(clip_fraction, 0.95),
            "early_stop_fraction": early_stop_fraction,
        },
        "survival": {
            "selected_median_first_trial_s": median_selected,
            "update0_median_first_trial_s": median_update0,
        },
        "gates": gates,
        "pilot_gate_passed": passed,
        "failure_reasons": [name for name, ok in gates.items() if not ok],
    }


def _evaluate_policy_payload(
    payload: dict[str, Any],
    *,
    context: dict[str, Any],
    config: dict[str, Any],
    num_envs: int,
    trial_seconds: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    shard = _build_shard(
        context,
        num_envs=num_envs,
        trial_seconds=trial_seconds,
        seed=seed,
        action_scale=float(config["action_scale"]),
        phase_observation=_phase_observation_enabled(config),
    )
    policy = _make_policy(shard, config, device)
    policy.load_state_dict(payload["policy"])
    return evaluate_first_trials(
        shard,
        _policy_action_provider(policy, shard, device),
    )


def _evaluate_zero_action(
    *,
    context: dict[str, Any],
    config: dict[str, Any],
    num_envs: int,
    trial_seconds: float,
    seed: int,
) -> dict[str, Any]:
    shard = _build_shard(
        context,
        num_envs=num_envs,
        trial_seconds=trial_seconds,
        seed=seed,
        action_scale=float(config["action_scale"]),
        phase_observation=_phase_observation_enabled(config),
    )
    return evaluate_first_trials(shard, _zero_action_provider(num_envs))


def _compose_gate_metrics(
    final_metrics: dict[str, Any],
    update0_metrics: dict[str, Any],
    zero_metrics: dict[str, Any],
    *,
    video_verified: bool,
    eval_envs: int,
    eval_seconds: float,
    pairing_verified: bool = True,
    comparison_margins: dict[str, float] | None = None,
) -> dict[str, Any]:
    comparison_margins = comparison_margins or {}
    metrics = dict(final_metrics)
    metrics.update(
        {
            "zero_baseline_planar_velocity_error": zero_metrics["planar_velocity_error"],
            "update0_baseline_planar_velocity_error": update0_metrics["planar_velocity_error"],
            "zero_action_forward_displacement_margin": (
                _finite_difference(final_metrics.get("forward_displacement_mean"), zero_metrics.get("forward_displacement_mean"))
            ),
            "update0_forward_displacement_margin": (
                _finite_difference(final_metrics.get("forward_displacement_mean"), update0_metrics.get("forward_displacement_mean"))
            ),
            "checkpoint_verified": True,
            "progression_verified": True,
            "paired_baselines_verified": pairing_verified,
            "zero_action_common_prefix_planar_margin": comparison_margins.get("zero_action", float("nan")),
            "update0_common_prefix_planar_margin": comparison_margins.get("update0", float("nan")),
            "video_verified": video_verified,
            "final_eval_configuration_verified": bool(
                eval_envs >= FINAL_EVAL_ENVS
                and math.isclose(eval_seconds, FINAL_EVAL_SECONDS)
                and math.isclose(float(final_metrics["control_hz"]), CONTROL_HZ)
                and int(final_metrics["trials"]) == eval_envs
                and int(final_metrics["completed_trials"]) == eval_envs
            ),
        }
    )
    return metrics


def _finite_difference(left: Any, right: Any) -> float:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return float("nan")
    value = float(left) - float(right)
    return value if math.isfinite(value) else float("nan")


def _parent_config_sha256(variant: str) -> str | None:
    if variant == "E1_phase":
        path = TASK_DIR / "artifacts/nominal_v3/unitree_g1/pilot/run_manifest.json"
        parent_variant = "E0"
    elif variant == "E2_reward_dt":
        path = TASK_DIR / "artifacts/nominal_v4/unitree_g1/E1_phase/run_manifest.json"
        parent_variant = "E1_phase"
    elif variant in E3A_ADAPTIVE_VARIANTS:
        path = TASK_DIR / "artifacts/nominal_v4/unitree_g1/E2_reward_dt/run_manifest.json"
        parent_variant = "E2_reward_dt"
    else:
        return None
    _require(path.is_file(), f"{parent_variant} parent manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    observed_variant = manifest.get("variant_id")
    if parent_variant == "E0" and observed_variant is None:
        observed_variant = "E0"
    _require(observed_variant == parent_variant, "parent variant metadata mismatch")
    return payload_sha256(manifest["configuration"])


E3A_OPTIMIZER_DIFF_ALLOWLIST = {
    "configuration.variant_id",
    "configuration.target_kl",
    "configuration.hard_kl_stop",
    "configuration.adaptive_kl",
    "configuration.desired_kl",
    "configuration.ppo.target_kl",
    "configuration.ppo.hard_kl_stop",
    "configuration.ppo.adaptive_kl",
    "configuration.ppo.desired_kl",
}


def _require_e2_gate_authorizes_e3a(e2_run_dir: Path) -> dict[str, Any]:
    gate_path = e2_run_dir / "e2_gate.json"
    manifest_path = e2_run_dir / "run_manifest.json"
    _require(gate_path.is_file(), "E2 scale gate artifact is missing")
    _require(manifest_path.is_file(), "E2 parent manifest is missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    _require(gate.get("variant_id") == "E2_reward_dt", "E2 scale gate variant mismatch")
    _require(gate.get("r2_e2_scale_gate_passed") is True, "E2 scale gate has not passed")
    _require(
        gate.get("r2_next_variant_allowed") == "E3a_adaptive_kl",
        "E2 scale gate does not authorize E3a",
    )
    _require(
        gate.get("e2_run_manifest_sha256") == sha256_path(manifest_path),
        "E2 scale gate manifest SHA drift",
    )
    return gate


def _verify_e3_optimizer_command(args: argparse.Namespace) -> int:
    e3_manifest_path = args.run_manifest.resolve()
    e3_manifest, _context, _shard = _validate_run_manifest(
        e3_manifest_path, expected_case="unitree_g1"
    )
    _require(e3_manifest.get("variant_id") == "E3a_adaptive_kl", "E3a verifier requires E3a_adaptive_kl")
    e2_path = e3_manifest_path.parent.parent / "E2_reward_dt" / "run_manifest.json"
    e2_gate = _require_e2_gate_authorizes_e3a(e2_path.parent)
    e2_manifest, _e2_progression = _historical_run_manifest_and_progression(
        e2_path,
        expected_case="unitree_g1",
        expected_variant="E2_reward_dt",
    )
    e3_config = e3_manifest["configuration"]
    e2_config = e2_manifest["configuration"]
    diff = _config_diff(e2_config, e3_config)
    _require(set(diff) <= E3A_OPTIMIZER_DIFF_ALLOWLIST, "E3a scientific configuration diff is not allowlisted")
    _require(
        e3_manifest.get("parent_config_sha256") == payload_sha256(e2_config),
        "E3a parent configuration SHA mismatch",
    )
    _require(e3_config.get("variant_id") == "E3a_adaptive_kl", "E3a variant config mismatch")
    _require(e3_config.get("target_kl") is None and e3_config.get("hard_kl_stop") is False, "hard KL stop remains enabled")
    _require(e3_config.get("adaptive_kl") is True and math.isclose(float(e3_config.get("desired_kl")), 0.01, abs_tol=0.0), "adaptive KL config mismatch")
    e3_ppo = e3_config.get("ppo", {})
    _require(e3_ppo.get("target_kl") is None and e3_ppo.get("hard_kl_stop") is False and e3_ppo.get("adaptive_kl") is True and math.isclose(float(e3_ppo.get("desired_kl")), 0.01, abs_tol=0.0), "nested adaptive KL config mismatch")
    _require(math.isclose(float(e3_config["learning_rate"]), 1e-4, abs_tol=0.0), "E3a initial learning rate mismatch")
    progression_path = Path(e3_manifest["progression"]["path"])
    _require(progression_path.is_file(), "E3a progression is missing")
    progression = json.loads(progression_path.read_text(encoding="utf-8"))
    validate_progression(progression, expected_updates=1000, expected_env_steps=2048000)
    rows = progression["reports"]
    minibatches = [record for row in rows for record in row.get("minibatches", [])]
    _require(len(minibatches) == 32000, "E3a progression must contain 32000 minibatch records")
    _require(all(int(row.get("epochs_completed", -1)) == 4 for row in rows), "E3a update did not complete 4 epochs")
    _require(all(int(row.get("minibatches_attempted", -1)) == 32 and int(row.get("minibatches_completed", -1)) == 32 for row in rows), "E3a update did not complete 32 minibatches")
    _require(all(row.get("early_stopped") is False for row in rows), "E3a early stop detected")
    _require(sum(bool(row.get("early_stopped")) for row in rows) == 0, "E3a early_stop_fraction is not exactly 0")
    _require(all(row.get("scheduler_decision") in {"increase", "decrease", "hold"} for row in rows), "E3a scheduler decision is invalid")
    _require(all(float(row.get("desired_kl")) == 0.01 for row in rows), "E3a desired KL telemetry mismatch")
    scheduler_kls = [float(row["scheduler_kl"]) for row in rows]
    _require(all(math.isfinite(value) and value >= 0.0 for value in scheduler_kls), "E3a scheduler KL is invalid")
    learning_rates = [float(row["learning_rate_before"]) for row in rows] + [float(row["learning_rate_after"]) for row in rows]
    _require(all(1e-5 <= value <= 1e-2 for value in learning_rates), "E3a learning rate escaped bounds")
    approx_kl = [float(record["approx_kl"]) for record in minibatches]
    clip_fraction = [float(record["clip_fraction"]) for record in minibatches]
    _require(all(math.isfinite(value) for value in approx_kl + clip_fraction), "E3a PPO diagnostics are nonfinite")
    nearest_rank = lambda values, probability: sorted(values)[max(0, min(len(values) - 1, math.ceil(probability * len(values)) - 1))]
    metrics = {
        "approx_kl_mean": sum(approx_kl) / len(approx_kl),
        "approx_kl_p95": nearest_rank(approx_kl, 0.95),
        "approx_kl_max": max(approx_kl),
        "clip_fraction_mean": sum(clip_fraction) / len(clip_fraction),
        "clip_fraction_p95": nearest_rank(clip_fraction, 0.95),
    }
    passed = (
        metrics["approx_kl_mean"] <= 0.015 and metrics["approx_kl_p95"] <= 0.03
        and metrics["approx_kl_max"] <= 0.05 and metrics["clip_fraction_mean"] <= 0.20
        and metrics["clip_fraction_p95"] <= 0.35
    )
    output = args.output.resolve()
    payload = {
        "schema_version": 1, "variant_id": "E3a_adaptive_kl", "parent_variant": "E2_reward_dt",
        "e2_gate_sha256": sha256_path(e2_path.parent / "e2_gate.json"),
        "e2_gate_schema_version": e2_gate.get("schema_version"),
        "e3a_run_manifest_sha256": sha256_path(e3_manifest_path), "config_diff_paths": diff,
        "metrics": metrics, "early_stop_fraction": 0.0, "progression_verified": True,
        "r2_e3a_optimizer_gate_passed": passed,
        "r2_next_variant_allowed": "E4a_roll_authority/E4b_contact_geometry" if passed else None,
        "claim_boundary": {"mechanistic_gate_only": True, "e3b_started": False, "e4_started": False, "h200_used": False, "task048_checkpoint_used": False, "external_downloads_performed": False},
    }
    _atomic_json(output, payload)
    print(json.dumps({"r2_e3a_optimizer_gate_passed": passed, "output": str(output)}))
    return 0 if passed else 1


def verify_e3_optimizer_command(args: argparse.Namespace) -> int:
    try:
        return _verify_e3_optimizer_command(args)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        _atomic_json(args.output.resolve(), {"schema_version": 1, "variant_id": "E3a_adaptive_kl", "r2_e3a_optimizer_gate_passed": False, "r2_next_variant_allowed": None, "failure_reasons": [str(error)]})
        print(json.dumps({"r2_e3a_optimizer_gate_passed": False, "output": str(args.output.resolve())}))
        return 1


def verify_e3a_kl_repair_command(args: argparse.Namespace) -> int:
    """No-update R0-R3 correctness gate; never constructs a trainer."""
    output = args.output.resolve()
    payload: dict[str, Any] = {
        "schema_version": 2,
        "variant_id": E3A_REPAIR_VARIANT,
        "parent_variant": "E2_reward_dt",
        "training_started": False,
        "smoke_started": False,
        "r0_r3_passed": False,
        "failure_reasons": [],
    }
    try:
        from importlib import metadata, util

        import torch

        from h200_locomotion_lab.algorithms.ppo import (
            _adaptive_lr_factor,
            _joint_gaussian_kl,
            ppo_update,
        )
        from h200_locomotion_lab.masked_distribution import (
            masked_raw_gaussian_log_prob,
            masked_tanh_gaussian_log_prob,
        )
        from h200_locomotion_lab.policies.whole_body_mlp import (
            WholeBodyMLPActorCritic,
            WholeBodyMLPConfig,
        )

        e2_dir = TASK_DIR / "artifacts/nominal_v4/unitree_g1/E2_reward_dt"
        rejected_dir = TASK_DIR / "artifacts/nominal_v4/unitree_g1/E3a_adaptive_kl"
        fixed_artifacts = {
            "e2_run_manifest": (
                e2_dir / "run_manifest.json",
                "25ef89af5c7460701a8b8f5f2b48de876dbcf34b7405e54dac7471e261b35597",
            ),
            "e2_gate": (
                e2_dir / "e2_gate.json",
                "fe41b57a91b715254f09625b73339e6403d02c1caa696ce6b73c0a8571b2910e",
            ),
            "rejected_e3a_run_manifest": (
                rejected_dir / "run_manifest.json",
                "3974dbadfe7d96b234878cc14fe6edfb58def64ea2a154747688bda5f118e33a",
            ),
            "rejected_e3a_progression": (
                rejected_dir / "progression.json",
                "45e9a52ee07a6166fe1ffe2cd1af43a44a4f9bda7fdb3e4f07261782db080d27",
            ),
            "rejected_e3a_gate": (
                rejected_dir / "e3a_gate.json",
                "3cb5594110432cf6bc4ab10bd4d74acec470468e8fe2cb16d05177ceb1706d1e",
            ),
        }
        artifact_records: dict[str, dict[str, str]] = {}
        for label, (path, expected_sha) in fixed_artifacts.items():
            _require(path.is_file(), f"{label} is missing")
            observed_sha = sha256_path(path)
            _require(observed_sha == expected_sha, f"{label} SHA drift")
            artifact_records[label] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": observed_sha,
            }
        e2_gate = _require_e2_gate_authorizes_e3a(e2_dir)
        e2_manifest = json.loads((e2_dir / "run_manifest.json").read_text(encoding="utf-8"))
        rejected_gate = json.loads((rejected_dir / "e3a_gate.json").read_text(encoding="utf-8"))
        _require(
            rejected_gate.get("r2_e3a_optimizer_gate_passed") is False,
            "rejected E3a gate no longer records failure",
        )
        rejected_metrics = rejected_gate.get("metrics", {})
        _require(
            any(
                float(rejected_metrics[key]) > threshold
                for key, threshold in E3A_OPTIMIZER_GATE_THRESHOLDS.items()
            ),
            "rejected E3a no longer fails original optimizer thresholds",
        )

        controlled_sources = {
            "ppo_kernel": ROOT / "src/h200_locomotion_lab/algorithms/ppo.py",
            "masked_distribution": ROOT / "src/h200_locomotion_lab/masked_distribution.py",
            "policy": ROOT / "src/h200_locomotion_lab/policies/whole_body_mlp.py",
            "trainer": ROOT / "src/h200_locomotion_lab/training/whole_body_ppo.py",
            "task072_cli_and_verifier": Path(__file__),
        }
        source_records = {
            label: {"path": str(path.relative_to(ROOT)), "sha256": sha256_path(path)}
            for label, path in controlled_sources.items()
        }

        reference_modules = {
            "rsl_rl.algorithms.ppo": "rsl_rl.algorithms.ppo",
            "rsl_rl.modules.distribution": "rsl_rl.modules.distribution",
            "mjlab": "mjlab",
        }
        reference_sources = {}
        for label, module_name in reference_modules.items():
            spec = util.find_spec(module_name)
            _require(spec is not None and spec.origin is not None, f"{module_name} source not found")
            path = Path(spec.origin)
            reference_sources[label] = {"path": str(path), "sha256": sha256_path(path)}

        action_mask = torch.zeros(45, dtype=torch.bool)
        action_mask[:29] = True
        batch_mask = action_mask.expand(2, -1)
        torch.manual_seed(3072)
        model = WholeBodyMLPActorCritic(
            WholeBodyMLPConfig(obs_dim=3, action_dim=45, hidden_dim=8, hidden_layers=1),
            action_mask=action_mask,
            device="cpu",
        )
        observations = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=torch.float32)
        _action, raw_sample, old_log_prob, _entropy, _value, old_mean, old_log_std = model.act_with_details(
            observations,
            active_mask=batch_mask,
        )
        replay_log_prob, _replay_entropy, _replay_value = model.evaluate_raw_actions(
            observations,
            raw_sample,
            active_action_mask=batch_mask,
        )
        regular_identity_error = float((replay_log_prob - old_log_prob).abs().max().item())

        saturated_raw = torch.zeros_like(raw_sample)
        saturated_raw[:, 0] = 20.0
        saturated_raw[:, 1] = -20.0
        saturated_raw[:, 2] = 8.0
        saturated_log_prob = masked_raw_gaussian_log_prob(
            saturated_raw,
            old_mean,
            old_log_std,
            batch_mask,
            tanh_eps=model.config.tanh_eps,
        )
        saturated_replay, _sat_entropy, _sat_value = model.evaluate_raw_actions(
            observations,
            saturated_raw,
            active_action_mask=batch_mask,
        )
        saturated_identity_error = float((saturated_replay - saturated_log_prob).abs().max().item())
        inverse_reconstructed = masked_tanh_gaussian_log_prob(
            torch.tanh(saturated_raw),
            old_mean,
            old_log_std,
            batch_mask,
            tanh_eps=model.config.tanh_eps,
        )
        saturated_inverse_drift = float((inverse_reconstructed - saturated_log_prob).abs().max().item())

        log_ratio = saturated_replay - saturated_log_prob
        ratio = log_ratio.exp()
        approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
        clip_fraction = float(((ratio - 1.0).abs() > 0.2).float().mean().item())
        _require(regular_identity_error <= 1e-5, "regular raw likelihood identity failed")
        _require(saturated_identity_error <= 1e-5, "saturated raw likelihood identity failed")
        _require(abs(approx_kl) <= 1e-6, "same-policy sampled KL is nonzero")
        _require(clip_fraction == 0.0, "same-policy clip fraction is nonzero")

        def tensor_state() -> tuple[Any, ...]:
            return tuple(parameter.detach().clone() for parameter in model.parameters())

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        before_update = tensor_state()
        ppo_batch = SimpleNamespace(
            observations=observations.reshape(1, 2, 3),
            actions=torch.tanh(saturated_raw).reshape(1, 2, 45),
            raw_actions=saturated_raw.reshape(1, 2, 45),
            old_means=old_mean.reshape(1, 2, 45),
            old_log_stds=old_log_std.reshape(1, 2, 45),
            log_probs=saturated_log_prob.reshape(1, 2),
            rewards=torch.zeros((1, 2)),
            dones=torch.zeros((1, 2), dtype=torch.bool),
            values=_value.detach().reshape(1, 2),
            next_value=torch.zeros(2),
            active_action_mask=batch_mask.reshape(1, 2, 45),
        )
        ppo_config = SimpleNamespace(
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
        ppo_diagnostics = ppo_update(
            model,
            optimizer,
            ppo_batch,
            torch.zeros((1, 2)),
            _value.detach().reshape(1, 2),
            ppo_config,
        )
        first_minibatch = ppo_diagnostics.minibatches[0]
        parameter_delta_max = max(
            float((before - after).abs().max().item())
            for before, after in zip(before_update, model.parameters(), strict=True)
        )
        _require(first_minibatch.same_policy_identity_error is not None, "first minibatch identity telemetry missing")
        _require(first_minibatch.same_policy_identity_error <= 1e-5, "first minibatch raw identity failed")
        _require(abs(first_minibatch.approx_kl) <= 1e-6, "first minibatch sampled KL is nonzero")
        _require(first_minibatch.clip_fraction == 0.0, "first minibatch clip fraction is nonzero")
        _require(
            first_minibatch.scheduler_kl is not None and abs(first_minibatch.scheduler_kl) <= 1e-6,
            "first minibatch scheduler KL is nonzero",
        )
        _require(parameter_delta_max <= 1e-12, "no-update verifier changed policy tensors")

        old_mean = torch.zeros((1, 45))
        old_log_std = torch.zeros((1, 45))

        def joint_fixture(target_kl: float) -> float:
            new_mean = torch.zeros_like(old_mean)
            new_mean[:, :29] = (2.0 * target_kl / 29.0) ** 0.5
            new_mean[:, 29:] = 1000.0
            return float(_joint_gaussian_kl(old_mean, old_log_std, new_mean, old_log_std, action_mask[None, :]).item())

        joint_kls = {str(value): joint_fixture(value) for value in (0.004, 0.01, 0.021)}
        for key, value in joint_kls.items():
            _require(abs(value - float(key)) <= 1e-6, f"joint KL fixture {key} mismatch")
        decisions = {str(kl): _adaptive_lr_factor(kl, 0.01)[1] for kl in (0.004, 0.01, 0.021)}
        _require(decisions == {"0.004": "increase", "0.01": "hold", "0.021": "decrease"}, "scheduler fixture mismatch")

        repaired_config = json.loads(json.dumps(e2_manifest["configuration"]))
        repaired_config.update({
            "variant_id": E3A_REPAIR_VARIANT,
            "target_kl": None,
            "hard_kl_stop": False,
            "adaptive_kl": True,
            "desired_kl": 0.01,
        })
        repaired_config["ppo"].update({
            "target_kl": None,
            "hard_kl_stop": False,
            "adaptive_kl": True,
            "desired_kl": 0.01,
        })
        config_diff = _config_diff(e2_manifest["configuration"], repaired_config)
        _require(
            set(config_diff) <= E3A_OPTIMIZER_DIFF_ALLOWLIST,
            "repaired E3a config diff is not optimizer-only",
        )

        payload.update({
            "r0": {
                "fixed_artifacts": artifact_records,
                "e2_gate_sha256": artifact_records["e2_gate"]["sha256"],
                "e2_gate_schema_version": e2_gate.get("schema_version"),
                "rejected_e3a_remains_rejected": True,
                "controlled_sources": source_records,
                "reference_packages": {
                    "mjlab": {"version": metadata.version("mjlab")},
                    "rsl-rl-lib": {"version": metadata.version("rsl-rl-lib")},
                },
                "reference_sources": reference_sources,
            },
            "r1": {
                "rollout_saves_old_mean_std_and_raw_action": True,
                "environment_action_space": "tanh_squashed_action_unchanged",
                "regular_max_abs_log_prob_delta": regular_identity_error,
                "saturated_max_abs_log_prob_delta": saturated_identity_error,
                "saturated_inverse_reconstruction_drift": saturated_inverse_drift,
                "same_policy_sampled_approx_kl": approx_kl,
                "same_policy_clip_fraction": clip_fraction,
            },
            "r2": {
                "joint_kl_by_target": joint_kls,
                "active_action_dimensions": 29,
                "inactive_slots_excluded": True,
                "scheduler_decisions": decisions,
                "lr_policy": {"desired_kl": 0.01, "decrease_above": 0.02, "increase_below": 0.005,
                              "min_lr": 1e-5, "max_lr": 1e-2},
            },
            "r3": {
                "config_diff_paths": config_diff,
                "config_diff_allowlist": sorted(E3A_OPTIMIZER_DIFF_ALLOWLIST),
                "original_optimizer_gate_thresholds": E3A_OPTIMIZER_GATE_THRESHOLDS,
                "old_e3a_gate_thresholds_unchanged": True,
                "first_minibatch_identity_error": first_minibatch.same_policy_identity_error,
                "first_minibatch_sampled_approx_kl": first_minibatch.approx_kl,
                "first_minibatch_clip_fraction": first_minibatch.clip_fraction,
                "first_minibatch_scheduler_kl": first_minibatch.scheduler_kl,
                "first_minibatch_scheduler_decision": first_minibatch.scheduler_decision,
                "no_update_parameter_delta_max": parameter_delta_max,
            },
        })
        payload["r0_r3_passed"] = True
    except Exception as error:
        payload["failure_reasons"].append(str(error))
    _atomic_json(output, payload)
    print(json.dumps({"r0_r3_passed": payload["r0_r3_passed"], "output": str(output)}))
    return 0 if payload["r0_r3_passed"] else 1


def _config_diff(left: Any, right: Any, prefix: str = "configuration") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                paths.append(f"{prefix}.{key}")
            else:
                paths.extend(_config_diff(left[key], right[key], f"{prefix}.{key}"))
        return paths
    return [] if left == right else [prefix]


def _historical_run_manifest_and_progression(
    path: Path,
    *,
    expected_case: str,
    expected_variant: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(path.is_file(), f"missing historical Task072 run manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest.get("artifact") == ARTIFACT_VERSION, "historical run manifest artifact drift")
    _require(manifest.get("case") == expected_case, "historical run manifest case mismatch")
    _require(manifest.get("variant_id") == expected_variant, "historical run manifest variant mismatch")
    _require(
        manifest.get("run_identity_sha256") == payload_sha256(manifest["run_identity"]),
        "historical run identity SHA mismatch",
    )
    progression_reference = manifest["progression"]
    progression_path = Path(progression_reference["path"])
    _require(progression_path.is_file(), "historical run progression artifact is missing")
    _require(
        sha256_path(progression_path) == progression_reference["sha256"],
        "historical run progression SHA mismatch",
    )
    progression = json.loads(progression_path.read_text(encoding="utf-8"))
    validate_progression(
        progression,
        expected_updates=int(manifest["configuration"]["updates"]),
        expected_env_steps=int(manifest["configuration"]["env_steps"]),
    )
    _require(
        progression["checkpoints"][0] == manifest["update0_checkpoint"],
        "historical progression/update0 checkpoint mismatch",
    )
    _require(
        progression["final_checkpoint"] == manifest["final_checkpoint"],
        "historical progression/final checkpoint mismatch",
    )
    return manifest, progression


def _e2_effect_report(run_dir: Path) -> dict[str, Any]:
    effect: dict[str, Any] = {}
    for name, key in (
        ("eval.json", "eval"),
        ("pilot_gate.json", "pilot_gate"),
        ("agent_visual_observation.json", "agent_visual_observation"),
    ):
        path = run_dir / name
        if path.is_file():
            effect[key] = json.loads(path.read_text(encoding="utf-8"))
    return effect


def _recompute_e2_scale_gate(
    e1_manifest: dict[str, Any],
    e2_manifest: dict[str, Any],
    e1_progression: dict[str, Any],
    e2_progression: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    e1_rows = e1_progression["reports"][:100]
    e2_rows = e2_progression["reports"][:100]
    _require(len(e1_rows) == 100 and len(e2_rows) == 100, "E1/E2 progression must contain first 100 rows")
    scale = float(e2_manifest["configuration"]["reward_scale"])
    ratios: list[float] = []
    raw_total = scaled_total = 0.0
    raw_nonzero = samples = zero_mismatches = 0
    e2_grads: list[float] = []
    e1_grads: list[float] = []
    e1_values: list[float] = []
    e2_values: list[float] = []
    required_e2_numeric_keys = (
        "grad_norm",
        "pre_clip_grad_norm",
        "policy_loss",
        "value_loss",
        "raw_reward_mean",
        "ppo_reward_mean",
        "reward_mean",
        "return_mean",
        "return_std",
        "return_p95",
        "gae_target_mean",
        "gae_target_std",
        "gae_target_p95",
        "value_prediction_mean",
        "value_prediction_std",
        "reward_scale_error_max",
    )

    def numeric_values(payload: Any) -> list[float]:
        if isinstance(payload, dict):
            return [value for item in payload.values() for value in numeric_values(item)]
        if isinstance(payload, list):
            return [value for item in payload for value in numeric_values(item)]
        if isinstance(payload, bool):
            return []
        if isinstance(payload, (int, float)):
            return [float(payload)]
        return []

    for e1, e2 in zip(e1_rows, e2_rows, strict=True):
        e1_grad = float(e1.get("grad_norm", float("nan")))
        e2_grad = float(e2.get("pre_clip_grad_norm", float("nan")))
        e1_value = float(e1.get("value_loss", float("nan")))
        e2_value = float(e2.get("value_loss", float("nan")))
        raw = float(e2.get("raw_reward_mean", float("nan")))
        scaled = float(e2.get("ppo_reward_mean", float("nan")))
        count = int(e2.get("reward_sample_count", 0))
        nonzero = int(e2.get("raw_reward_nonzero_count", 0))
        error = float(e2.get("reward_scale_error_max", float("nan")))
        for value in (e1_grad, e2_grad, e1_value, e2_value, raw, scaled, error):
            _require(math.isfinite(value), "E1/E2 first-100 scale metrics must be finite")
        for key in required_e2_numeric_keys:
            _require(key in e2, f"E2 first-100 row missing {key}")
            _require(math.isfinite(float(e2[key])), f"E2 first-100 {key} is nonfinite")
        _require(
            all(math.isfinite(value) for value in numeric_values(e2)),
            "E2 first-100 row contains nonfinite numeric data",
        )
        _require(e2.get("grad_norm_is_pre_clip") is True, "E2 gradient metric is not marked pre-clip")
        _require(count > 0 and 0 <= nonzero <= count, "E2 reward sample counts are invalid")
        _require(error <= 1e-7, "E2 recorded reward scale error exceeds tolerance")
        if abs(raw) > 1e-12:
            ratios.append(scaled / raw)
        raw_total += raw * count
        scaled_total += scaled * count
        raw_nonzero += nonzero
        samples += count
        zero_mismatches += int(e2.get("reward_zero_mismatch_count", -1))
        e1_grads.append(e1_grad)
        e2_grads.append(e2_grad)
        e1_values.append(e1_value)
        e2_values.append(e2_value)
    mean_e1_grad = sum(e1_grads) / 100
    mean_e2_grad = sum(e2_grads) / 100
    mean_e1_value = sum(e1_values) / 100
    mean_e2_value = sum(e2_values) / 100
    observed_ratio = scaled_total / raw_total if abs(raw_total) > 1e-12 else float("nan")
    details = {
        "e1_grad_norm_source": "grad_norm",
        "e1_pre_clip_grad_norm_mean": mean_e1_grad,
        "e2_pre_clip_grad_norm_mean": mean_e2_grad,
        "e1_value_loss_mean": mean_e1_value,
        "e2_value_loss_mean": mean_e2_value,
        "e2_raw_reward_mean": raw_total / samples,
        "e2_ppo_reward_mean": scaled_total / samples,
        "e2_mean_reward_ratio": observed_ratio,
        "e2_reward_scale_expected": scale,
        "e2_reward_sample_count": samples,
        "e2_raw_reward_nonzero_count": raw_nonzero,
        "e2_reward_zero_mismatch_count": zero_mismatches,
        "e2_reward_scale_error_max": max(float(row["reward_scale_error_max"]) for row in e2_rows),
        "finite_first100": True,
    }
    passed = (
        math.isclose(observed_ratio, scale, rel_tol=0.0, abs_tol=1e-7)
        and zero_mismatches == 0
        and mean_e2_grad <= mean_e1_grad / 5.0
        and mean_e2_grad <= 10.1973994
        and mean_e2_value <= mean_e1_value / 10.0
        and mean_e2_value <= 15.4569912
    )
    details["reward_scale_ratio_exact"] = (
        raw_nonzero > 0
        and math.isclose(observed_ratio, scale, rel_tol=0.0, abs_tol=1e-7)
        and details["e2_reward_scale_error_max"] <= 1e-7
        and zero_mismatches == 0
    )
    details["pre_clip_grad_norm_5x_and_abs"] = mean_e2_grad <= mean_e1_grad / 5.0 and mean_e2_grad <= 10.1973994
    details["value_loss_10x_and_abs"] = mean_e2_value <= mean_e1_value / 10.0 and mean_e2_value <= 15.4569912
    return passed, details


def _verify_e2_scale_command(args: argparse.Namespace) -> int:
    e2_manifest_path = args.run_manifest.resolve()
    e2_manifest, _context, _shard = _validate_run_manifest(e2_manifest_path, expected_case="unitree_g1")
    _require(e2_manifest.get("variant_id") == "E2_reward_dt", "E2 scale verifier requires E2_reward_dt")
    e2_progression = json.loads(Path(e2_manifest["progression"]["path"]).read_text(encoding="utf-8"))
    e1_manifest_path = e2_manifest_path.parent.parent / "E1_phase" / "run_manifest.json"
    e1_manifest, e1_progression = _historical_run_manifest_and_progression(
        e1_manifest_path,
        expected_case="unitree_g1",
        expected_variant="E1_phase",
    )
    diff = _config_diff(e1_manifest["configuration"], e2_manifest["configuration"])
    _require(set(diff) <= {"configuration.reward_scale", "configuration.variant_id"}, "E2 scientific configuration diff is not allowlisted")
    _require(e2_manifest.get("parent_config_sha256") == payload_sha256(e1_manifest["configuration"]), "E2 parent configuration SHA mismatch")
    e2_config = e2_manifest["configuration"]
    phase = e2_config.get("phase_observation", {})
    e1_config = e1_manifest["configuration"]
    e1_phase = e1_config.get("phase_observation", {})
    _require(e1_phase.get("enabled") is True and e1_phase.get("schema") == "193->195", "E1 phase/schema invariant failed")
    _require(math.isclose(float(e1_config.get("reward_scale", 1.0)), 1.0, rel_tol=0.0, abs_tol=0.0), "E1 reward scale invariant failed")
    _require(math.isclose(float(e1_manifest.get("control_dt", CONTROL_DT)), CONTROL_DT, rel_tol=0.0, abs_tol=0.0), "E1 control_dt invariant failed")
    _require(phase.get("enabled") is True and phase.get("schema") == "193->195", "E2 phase/schema invariant failed")
    _require(e2_manifest.get("observation_schema") == "193->195", "E2 observation schema invariant failed")
    _require(math.isclose(float(e2_config.get("reward_scale")), CONTROL_DT, rel_tol=0.0, abs_tol=0.0), "E2 reward scale must equal control_dt")
    _require(math.isclose(float(e2_manifest.get("control_dt")), CONTROL_DT, rel_tol=0.0, abs_tol=0.0), "E2 control_dt invariant failed")
    passed, first100 = _recompute_e2_scale_gate(e1_manifest, e2_manifest, e1_progression, e2_progression)
    output = args.output.resolve()
    payload = {
        "schema_version": 2,
        "variant_id": "E2_reward_dt",
        "parent_variant": "E1_phase",
        "e1_parent_config_sha256": payload_sha256(e1_manifest["configuration"]),
        "e2_run_manifest_sha256": sha256_path(e2_manifest_path),
        "config_diff_paths": diff,
        "scientific_config_diff_paths": [path for path in diff if path != "configuration.variant_id"],
        "metadata_diff_paths": [path for path in diff if path == "configuration.variant_id"],
        "correctness": {"scientific_config_diff_allowlisted": True, "parent_config_sha256_matches_E1": True,
                        "phase_observation_still_enabled": True, "observation_schema_193_to_195": True,
                        "reward_scale_is_control_dt": True, "progression_verified": True},
        "scale_gate": first100,
        "effect_report": _e2_effect_report(e2_manifest_path.parent),
        "r2_e2_scale_gate_passed": passed,
        "r2_next_variant_allowed": "E3a_adaptive_kl" if passed else None,
        "claim_boundary": {"e2_mechanistic_gate_only": True, "full_walking_claimed": False,
                           "e3_started": False, "h200_used": False, "task048_checkpoint_used": False,
                           "external_downloads_performed": False},
    }
    _atomic_json(output, payload)
    print(json.dumps({"r2_e2_scale_gate_passed": passed, "output": str(output)}))
    return 0 if passed else 1


def verify_e2_scale_command(args: argparse.Namespace) -> int:
    try:
        return _verify_e2_scale_command(args)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        payload = {
            "schema_version": 2,
            "variant_id": "E2_reward_dt",
            "r2_e2_scale_gate_passed": False,
            "r2_next_variant_allowed": None,
            "failure_reasons": [str(error)],
            "claim_boundary": {"e2_mechanistic_gate_only": True, "e3_started": False},
        }
        _atomic_json(args.output.resolve(), payload)
        print(json.dumps({"r2_e2_scale_gate_passed": False, "output": str(args.output.resolve())}))
        return 1


def _solve_contact_aligned_stance(context: dict[str, Any], asset: dict[str, Any]) -> Any:
    import mujoco
    import numpy as np
    from h200_locomotion_lab.robots.whole_body_stance import solve_static_stance

    model = mujoco.MjModel.from_xml_string(asset["xml"])
    data = mujoco.MjData(model)
    return solve_static_stance(
        model,
        data,
        context["case"].blueprint,
        context["case"].physical,
        foot_geom_groups=asset["logical_foot_groups"],
        model_xml_sha256=asset["xml_sha256"],
        kinematic_max_nfev=400,
        dynamics_max_nfev=350,
    )


def _contact_aligned_context(parent_context: dict[str, Any], profile: dict[str, Any], stance: Any, xml: str) -> dict[str, Any]:
    overlay_record = dict(parent_context["overlay_record"])
    overlay_record["parent_output_xml_sha256"] = TASK071_G1_BOUND_XML_SHA256
    overlay_record["output_xml_sha256"] = profile["asset"]["candidate_xml_sha256"]
    return {
        **parent_context,
        "xml": xml,
        "stance": stance,
        "overlay_record": overlay_record,
        "contact_profile": profile,
    }


def _contact_no_update_smoke(context: dict[str, Any]) -> dict[str, Any]:
    shard = _build_shard(
        context,
        num_envs=2,
        trial_seconds=0.2,
        seed=72072,
        action_scale=0.35,
    )
    observation = shard.reset()
    step = shard.step(shard.np.zeros((2, 45), dtype=shard.np.float32))
    metrics = step.metrics
    finite_arrays = [
        observation,
        step.actor_observation,
        step.reward,
        metrics["foot_height"],
        metrics["foot_planar_speed"],
        metrics["foot_vertical_speed"],
        metrics["foot_air_time"],
        metrics["foot_normal_force"],
    ]
    return {
        "schema_version": 1,
        "optimizer_step_calls": 0,
        "reset_observation_shape": list(observation.shape),
        "step_observation_shape": list(step.actor_observation.shape),
        "foot_contact_shape": list(metrics["foot_contact"].shape),
        "touchdown_shape": list(metrics["touchdown"].shape),
        "foot_normal_force_shape": list(metrics["foot_normal_force"].shape),
        "non_foot_contact_fraction": [float(value) for value in metrics["non_foot_contact_fraction"]],
        "finite": all(bool(shard.np.isfinite(array).all()) for array in finite_arrays),
        "logical_feet": list(shard._logical_foot_names),
        "foot_geoms": list(sorted(shard._foot_geoms)),
    }


def _compiled_contact_geometry_audit(xml: str, profile: dict[str, Any]) -> dict[str, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_string(xml)
    foot_geom_ids = []
    for geoms in profile["logical_foot_groups"].values():
        for geom in geoms:
            foot_geom_ids.append(int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)))
    legacy_boxes = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").endswith("_footpad")
    ]
    rows = []
    for geom_id in foot_geom_ids:
        rows.append(
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                "type": int(model.geom_type[geom_id]),
                "type_name": "capsule" if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else "other",
                "condim": int(model.geom_condim[geom_id]),
                "priority": int(model.geom_priority[geom_id]),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
                "friction": [float(value) for value in model.geom_friction[geom_id]],
                "size": [float(value) for value in model.geom_size[geom_id]],
            }
        )
    checks = {
        "two_logical_feet": len(profile["logical_foot_groups"]) == 2,
        "fourteen_foot_collision_geoms": len(foot_geom_ids) == 14 and all(geom_id >= 0 for geom_id in foot_geom_ids),
        "zero_legacy_foot_boxes": len(legacy_boxes) == 0,
        "all_capsules": all(row["type_name"] == "capsule" for row in rows),
        "condim_6_priority_1": all(row["condim"] == 6 and row["priority"] == 1 for row in rows),
    }
    return {"checks": checks, "passed": all(checks.values()), "legacy_foot_boxes": legacy_boxes, "foot_geoms": rows}


def align_contact_command(args: argparse.Namespace) -> int:
    output_root = (args.output_root or CONTACT_ALIGNMENT_ROOT).resolve()
    gate_path = output_root / "alignment_gate.json"
    try:
        _require(not (output_root / "contact_profile.json").exists(), "003b contact alignment output already exists")
        parent_context = _load_bound_context("unitree_g1")
        asset = _make_contact_aligned_asset(parent_context)
        stance = _solve_contact_aligned_stance(parent_context, asset)
        profile = {
            "schema_version": 1,
            "contact_profile_id": MJLAB_G1_7CAPSULE_PROFILE_ID,
            "case_id": "unitree_g1",
            "lineage": {
                "parent_bound_xml_sha256": TASK071_G1_BOUND_XML_SHA256,
                "mjlab_checkout_commit": MJLAB_G1_CHECKOUT_COMMIT,
                "mjlab_g1_xml_path": str(MJLAB_G1_XML),
                "mjlab_g1_xml_sha256": MJLAB_G1_XML_SHA256,
            },
            "asset": {
                "candidate_xml_sha256": asset["xml_sha256"],
                "allowed_delta": [
                    "terminal contact geoms",
                    "logical-foot grouping",
                    "foot reference site",
                    "stance root height from new contact geometry",
                ],
            },
            "logical_foot_groups": {key: list(value) for key, value in asset["logical_foot_groups"].items()},
            "logical_foot_reference_sites": asset["logical_foot_reference_sites"],
            "source_capsules": asset["source_capsules"],
            "scaled_capsules": asset["scaled_capsules"],
            "scale_transforms": asset["scale_transforms"],
            "invariant_diff": asset["invariant_diff"],
            "parent_signature_sha256": asset["parent_signature_sha256"],
            "candidate_signature_sha256": asset["candidate_signature_sha256"],
        }
        candidate_context = _contact_aligned_context(parent_context, profile, stance, asset["xml"])
        no_update = _contact_no_update_smoke(candidate_context)
        contact_audit = _compiled_contact_geometry_audit(asset["xml"], profile)
        stance_payload = {
            "schema_version": 1,
            "contact_profile_id": MJLAB_G1_7CAPSULE_PROFILE_ID,
            "stance_solution": stance.manifest(),
            "stance_solution_sha256": stance.solution_hash,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        xml_path = output_root / "unitree_g1_mjlab_g1_7capsule_v1.xml"
        xml_path.write_text(asset["xml"], encoding="utf-8")
        _atomic_json(output_root / "contact_profile.json", profile)
        _atomic_json(output_root / "stance_solution.json", stance_payload)
        _atomic_json(output_root / "compiled_invariant_diff.json", asset["invariant_diff"])
        _atomic_json(output_root / "compiled_contact_geometry.json", contact_audit)
        _atomic_json(output_root / "no_update_smoke.json", no_update)
        gate = {
            "schema_version": 1,
            "contact_profile_id": MJLAB_G1_7CAPSULE_PROFILE_ID,
            "passed": bool(
                asset["invariant_diff"]["passed"]
                and contact_audit["passed"]
                and no_update["finite"]
                and no_update["optimizer_step_calls"] == 0
                and no_update["foot_contact_shape"] == [2, 2]
                and no_update["touchdown_shape"] == [2, 2]
            ),
            "asset_xml": {"path": str(xml_path), "sha256": sha256_path(xml_path)},
            "contact_profile": {"path": str(output_root / "contact_profile.json"), "sha256": sha256_path(output_root / "contact_profile.json")},
            "stance": {"path": str(output_root / "stance_solution.json"), "sha256": sha256_path(output_root / "stance_solution.json")},
            "compiled_invariant_diff": {"path": str(output_root / "compiled_invariant_diff.json"), "sha256": sha256_path(output_root / "compiled_invariant_diff.json")},
            "compiled_contact_geometry": {"path": str(output_root / "compiled_contact_geometry.json"), "sha256": sha256_path(output_root / "compiled_contact_geometry.json")},
            "no_update_smoke": {"path": str(output_root / "no_update_smoke.json"), "sha256": sha256_path(output_root / "no_update_smoke.json")},
            "claim_boundary": {"training_started": False, "optimizer_steps_allowed": False},
        }
        _atomic_json(gate_path, gate)
        print(json.dumps({"003b_passed": gate["passed"], "gate": str(gate_path)}))
        return 0 if gate["passed"] else 1
    except Exception as exc:
        output_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            gate_path,
            {
                "schema_version": 1,
                "contact_profile_id": MJLAB_G1_7CAPSULE_PROFILE_ID,
                "passed": False,
                "failure_reasons": [str(exc)],
                "claim_boundary": {"training_started": False, "optimizer_steps_allowed": False},
            },
        )
        print(json.dumps({"003b_passed": False, "gate": str(gate_path), "error": str(exc)}))
        return 1


def verify_contact_alignment_command(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    gate_path = args.output.resolve()
    try:
        parent_context = _load_bound_context("unitree_g1")
        regenerated = _make_contact_aligned_asset(parent_context)
        artifacts = _load_contact_alignment_artifacts() if output_root == CONTACT_ALIGNMENT_ROOT.resolve() else None
        profile_path = output_root / "contact_profile.json"
        stance_path = output_root / "stance_solution.json"
        xml_path = output_root / "unitree_g1_mjlab_g1_7capsule_v1.xml"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        stance_payload = json.loads(stance_path.read_text(encoding="utf-8"))
        xml = xml_path.read_text(encoding="utf-8")
        _ = artifacts
        contact_audit = _compiled_contact_geometry_audit(xml, profile)
        candidate_context = _load_bound_context("unitree_g1", contact_profile=MJLAB_G1_7CAPSULE_PROFILE_ID) if output_root == CONTACT_ALIGNMENT_ROOT.resolve() else None
        no_update = _contact_no_update_smoke(candidate_context) if candidate_context is not None else json.loads((output_root / "no_update_smoke.json").read_text(encoding="utf-8"))
        checks = {
            "asset_reproducible": xml == regenerated["xml"],
            "asset_sha_bound": sha256_bytes(xml.encode()) == profile["asset"]["candidate_xml_sha256"],
            "invariants": regenerated["invariant_diff"]["passed"],
            "contact_geometry": contact_audit["passed"],
            "stance_bound": stance_payload["stance_solution"]["model_xml_sha256"] == profile["asset"]["candidate_xml_sha256"],
            "stance_hash": payload_sha256(stance_payload["stance_solution"]) == stance_payload["stance_solution_sha256"],
            "no_update": bool(no_update["finite"] and no_update["optimizer_step_calls"] == 0),
        }
        payload = {
            "schema_version": 1,
            "contact_profile_id": MJLAB_G1_7CAPSULE_PROFILE_ID,
            "passed": all(checks.values()),
            "checks": checks,
            "inputs": {
                "contact_profile": {"path": str(profile_path), "sha256": sha256_path(profile_path), "payload_sha256": payload_sha256(profile)},
                "stance": {"path": str(stance_path), "sha256": sha256_path(stance_path), "payload_sha256": payload_sha256(stance_payload)},
                "asset_xml": {"path": str(xml_path), "sha256": sha256_path(xml_path)},
            },
            "regenerated_candidate_xml_sha256": regenerated["xml_sha256"],
            "claim_boundary": {"training_started": False, "optimizer_steps_allowed": False},
        }
        _atomic_json(gate_path, payload)
        print(json.dumps({"003b_verified": payload["passed"], "output": str(gate_path)}))
        return 0 if payload["passed"] else 1
    except Exception as exc:
        _atomic_json(
            gate_path,
            {
                "schema_version": 1,
                "contact_profile_id": MJLAB_G1_7CAPSULE_PROFILE_ID,
                "passed": False,
                "failure_reasons": [str(exc)],
                "claim_boundary": {"training_started": False, "optimizer_steps_allowed": False},
            },
        )
        print(json.dumps({"003b_verified": False, "output": str(gate_path), "error": str(exc)}))
        return 1


def train_command(args: argparse.Namespace) -> int:
    import torch

    from h200_locomotion_lab.training.whole_body_ppo import (
        WholeBodyPPOConfig,
        WholeBodyPPOTrainer,
    )

    variant = args.variant
    _require(variant == "E0" or args.case == "unitree_g1", "E1/E2/E3 variants are G1-only")
    if variant in E3A_ADAPTIVE_VARIANTS:
        _require_e2_gate_authorizes_e3a(
            TASK_DIR / "artifacts/nominal_v4" / args.case / "E2_reward_dt"
        )
    if args.run_dir is None:
        root = "nominal_v4" if variant in R2_NOMINAL_V4_VARIANTS else "nominal_v3"
        args.run_dir = TASK_DIR / "artifacts" / root / args.case / (variant if variant != "E0" else args.stage)
    run_dir = args.run_dir.resolve()
    _require(not (run_dir / "run_manifest.json").exists(), "Task072 run already completed")
    stage = TRAIN_STAGES[args.stage]
    args.envs = stage.envs
    args.rollout_steps = stage.rollout_steps
    args.updates = stage.updates
    args.seed = stage.seed
    args.checkpoint_every = stage.checkpoint_every
    _require(stage.transitions == args.envs * args.rollout_steps * args.updates, "stage transition budget drift")
    _require(args.checkpoint_every > 0, "checkpoint cadence must be positive")
    torch.manual_seed(stage.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(stage.seed)
    context = _load_bound_context(args.case, contact_profile=args.contact_profile)
    shard = _build_shard(
        context,
        num_envs=args.envs,
        trial_seconds=args.trial_seconds,
        seed=args.seed,
        action_scale=args.action_scale,
        phase_observation=variant in R2_PHASE_OBSERVATION_VARIANTS,
    )
    ppo_config = WholeBodyPPOConfig(
        rollout_steps=args.rollout_steps,
        updates=1,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        log_std_init=args.log_std_init,
        entropy_coef=args.entropy_coef,
        target_kl=None if variant in E3A_ADAPTIVE_VARIANTS else args.target_kl,
        hard_kl_stop=variant not in E3A_ADAPTIVE_VARIANTS,
        adaptive_kl=variant in E3A_ADAPTIVE_VARIANTS,
        desired_kl=0.01 if variant in E3A_ADAPTIVE_VARIANTS else None,
        device=args.device,
    )
    reward_scale = CONTROL_DT if variant in R2_REWARD_DT_VARIANTS else 1.0
    training_environment = _make_task072_reward(shard, reward_scale=reward_scale)
    trainer = WholeBodyPPOTrainer(
        training_environment,
        action_mask=shard.active_action_mask,
        config=ppo_config,
    )
    static_lineage = _static_lineage(
        args.case,
        context,
        shard,
        action_scale=args.action_scale,
        phase_observation=variant in R2_PHASE_OBSERVATION_VARIANTS,
    )
    static_sha = payload_sha256(static_lineage)
    configuration = {
        "variant_id": variant,
        "reward_scale": reward_scale,
        "actor_observation_dim": int(trainer.observation.shape[-1]),
        "seed": args.seed,
        "device": args.device,
        "num_envs": args.envs,
        "updates": args.updates,
        "rollout_steps": args.rollout_steps,
        "env_steps": args.envs * args.rollout_steps * args.updates,
        "checkpoint_every": args.checkpoint_every,
        "training_trial_seconds": args.trial_seconds,
        "action_scale": args.action_scale,
        "training_reward": TRAINING_REWARD_VERSION,
        "case_reward": (
            BIPED_REWARD_VERSION
            if context["case"].spec["family"] == "biped"
            else QUADRUPED_REWARD_VERSION
        ),
        "stage": args.stage,
        "contact_profile_id": args.contact_profile,
        "curriculum_enabled": False,
        "velocity_curriculum": False,
        "randomization": {
            "mass": False,
            "com": False,
            "inertia": False,
            "friction": False,
            "motor": False,
            "delay": False,
            "push": False,
            "sensor": False,
            "terrain": False,
            "command": False,
        },
        "motor_fault": False,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "epochs": args.epochs,
        "minibatch_size": args.minibatch_size,
        "learning_rate": args.learning_rate,
        "log_std_init": args.log_std_init,
        "entropy_coef": args.entropy_coef,
        "target_kl": ppo_config.target_kl,
        "hard_kl_stop": ppo_config.hard_kl_stop,
        "adaptive_kl": ppo_config.adaptive_kl,
        "desired_kl": ppo_config.desired_kl,
        "ppo": asdict(ppo_config),
        "observation_transform": {
            "version": OBSERVATION_TRANSFORM_VERSION,
            "joint_velocity_scale": 0.05,
            "base_angular_velocity_scale": 0.2,
        },
        "phase_observation": {
            "enabled": variant in R2_PHASE_OBSERVATION_VARIANTS,
            "clock_source": "WholeBodyMuJoCoShard._trial_step/control_hz",
            "period_s": 0.8,
            "schema": "193->195" if variant in R2_PHASE_OBSERVATION_VARIANTS else "193",
        },
    }
    run_identity = {
        "artifact": ARTIFACT_VERSION,
        "case": args.case,
        "static_lineage_sha256": static_sha,
        "configuration": configuration,
    }
    run_identity_sha = payload_sha256(run_identity)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: list[dict[str, Any]] = []
    update0_lineage = _checkpoint_lineage(
        case_id=args.case,
        static_lineage_sha256=static_sha,
        run_identity_sha256=run_identity_sha,
        update=0,
        env_steps=0,
    )
    update0_reference = _save_checkpoint(
        torch,
        run_dir / "initial.pt",
        policy=trainer.policy,
        optimizer=trainer.optimizer,
        lineage=update0_lineage,
    )
    checkpoints.append(update0_reference)
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    for update in range(1, args.updates + 1):
        training_environment.set_training_progress(update / args.updates)
        trainer.observation[:, 9:12] = torch.as_tensor(
            (training_environment.target_vx(), COMMAND[1], COMMAND[2]),
            dtype=trainer.observation.dtype,
            device=trainer.observation.device,
        )
        report = trainer.train(updates=1)[0]
        report.update(
            {
                "global_update": update,
                "env_steps": update * args.envs * args.rollout_steps,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        reports.append(report)
        if update % args.checkpoint_every == 0 or update == args.updates:
            lineage = _checkpoint_lineage(
                case_id=args.case,
                static_lineage_sha256=static_sha,
                run_identity_sha256=run_identity_sha,
                update=update,
                env_steps=update * args.envs * args.rollout_steps,
            )
            reference = _save_checkpoint(
                torch,
                run_dir / f"update{update:06d}.pt",
                policy=trainer.policy,
                optimizer=trainer.optimizer,
                lineage=lineage,
            )
            checkpoints.append(reference)
        if update == 1 or update % max(1, args.checkpoint_every) == 0:
            print(
                json.dumps(
                    {
                        "case": args.case,
                        "update": update,
                        "updates": args.updates,
                        "reward_mean": report["reward_mean"],
                        "fall_count": report["fall_count"],
                    }
                ),
                flush=True,
            )
    final_lineage = _checkpoint_lineage(
        case_id=args.case,
        static_lineage_sha256=static_sha,
        run_identity_sha256=run_identity_sha,
        update=args.updates,
        env_steps=configuration["env_steps"],
    )
    final_reference = _save_checkpoint(
        torch,
        run_dir / "final.pt",
        policy=trainer.policy,
        optimizer=trainer.optimizer,
        lineage=final_lineage,
    )
    progression = {
        "artifact": ARTIFACT_VERSION,
        "case": args.case,
        "stage": args.stage,
        "reports": reports,
        "checkpoints": checkpoints,
        "final_checkpoint": final_reference,
    }
    validate_progression(
        progression,
        expected_updates=args.updates,
        expected_env_steps=configuration["env_steps"],
    )
    progression_path = run_dir / "progression.json"
    _atomic_json(progression_path, progression)
    manifest = {
        "artifact": ARTIFACT_VERSION,
        "task": "task072-bound-g1-go2-locomotion-proof",
        "case": args.case,
        "static_lineage": static_lineage,
        "static_lineage_sha256": static_sha,
        "run_identity": run_identity,
        "run_identity_sha256": run_identity_sha,
        "configuration": configuration,
        "variant_id": variant,
        "parent_variant": "E0" if variant == "E1_phase" else ("E1_phase" if variant == "E2_reward_dt" else ("E2_reward_dt" if variant in E3A_ADAPTIVE_VARIANTS else None)),
        "parent_config_sha256": _parent_config_sha256(variant),
        "delta_allowlist": (
            ["configuration.reward_scale"]
            if variant == "E2_reward_dt"
            else (sorted(E3A_OPTIMIZER_DIFF_ALLOWLIST - {"configuration.variant_id"})
                  if variant in E3A_ADAPTIVE_VARIANTS
                  else ["configuration.actor_observation_dim", "configuration.phase_observation"])
        ),
        "source_sha256": payload_sha256(static_lineage["sources"]),
        "initial_checkpoint_sha256": update0_reference["sha256"],
        "phase_clock_source": "WholeBodyMuJoCoShard._trial_step/control_hz",
        "phase_source": "Task072BipedReward._desired_contact shared phase helper",
        "phase_period_s": 0.8,
        "control_dt": CONTROL_DT,
        "observation_schema": "193->195" if variant in R2_PHASE_OBSERVATION_VARIANTS else "193",
        "update0_checkpoint": update0_reference,
        "final_checkpoint": final_reference,
        "progression": {
            "path": str(progression_path.resolve()),
            "sha256": sha256_path(progression_path),
        },
        "runtime": _runtime_metadata(" ".join(sys.argv)),
        "claim_boundary": {
            "training_completed": True,
            "evaluation_completed": False,
            "walking_claimed": False,
            "task072_passed": False,
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    _atomic_json(manifest_path, manifest)
    print(json.dumps({"run_manifest": str(manifest_path), "case": args.case}), flush=True)
    return 0


def eval_command(args: argparse.Namespace) -> int:
    import torch

    manifest, context, _validation_shard = _validate_run_manifest(
        args.run_manifest.resolve(), expected_case=args.case
    )
    config = manifest["configuration"]
    static_sha = manifest["static_lineage_sha256"]
    run_identity_sha = manifest["run_identity_sha256"]
    final_lineage = _checkpoint_lineage(
        case_id=args.case,
        static_lineage_sha256=static_sha,
        run_identity_sha256=run_identity_sha,
        update=int(config["updates"]),
        env_steps=int(config["env_steps"]),
    )
    update0_lineage = _checkpoint_lineage(
        case_id=args.case,
        static_lineage_sha256=static_sha,
        run_identity_sha256=run_identity_sha,
        update=0,
        env_steps=0,
    )
    final_payload = _load_checkpoint(torch, manifest["final_checkpoint"], final_lineage)
    update0_payload = _load_checkpoint(torch, manifest["update0_checkpoint"], update0_lineage)

    selected_shard = _build_shard(
        context,
        num_envs=args.eval_envs,
        trial_seconds=args.eval_seconds,
        seed=args.seed,
        action_scale=float(config["action_scale"]),
        phase_observation=_phase_observation_enabled(config),
    )
    selected_policy = _make_policy(selected_shard, config, args.device)
    selected_policy.load_state_dict(final_payload["policy"])
    update0_shard = _build_shard(
        context,
        num_envs=args.eval_envs,
        trial_seconds=args.eval_seconds,
        seed=args.seed,
        action_scale=float(config["action_scale"]),
        phase_observation=_phase_observation_enabled(config),
    )
    update0_policy = _make_policy(update0_shard, config, args.device)
    update0_policy.load_state_dict(update0_payload["policy"])
    zero_shard = _build_shard(
        context,
        num_envs=args.eval_envs,
        trial_seconds=args.eval_seconds,
        seed=args.seed,
        action_scale=float(config["action_scale"]),
        phase_observation=_phase_observation_enabled(config),
    )
    selected_trace = evaluate_first_trial_trace(
        selected_shard,
        _policy_action_provider(selected_policy, selected_shard, args.device),
        seed=args.seed,
    )
    update0_trace = evaluate_first_trial_trace(
        update0_shard,
        _policy_action_provider(update0_policy, update0_shard, args.device),
        seed=args.seed,
    )
    zero_trace = evaluate_first_trial_trace(
        zero_shard,
        _zero_action_provider(args.eval_envs),
        seed=args.seed,
    )
    trace_payload = {
        "schema_version": 1,
        "case_id": args.case,
        "horizon_s": args.eval_seconds,
        "num_envs": args.eval_envs,
        "eval_seed": args.seed,
        "policies": {
            "selected": selected_trace,
            "untrained": update0_trace,
            "zero_action": zero_trace,
        },
    }
    trace_path = args.output.resolve().with_name("eval_trace.json")
    _atomic_json(trace_path, trace_payload)
    final_metrics = metrics_from_trace(selected_trace)
    update0_metrics = metrics_from_trace(update0_trace)
    zero_metrics = metrics_from_trace(zero_trace)
    pairing_verified = all(
        selected["initial_state_sha256"] == update0["initial_state_sha256"] == zero["initial_state_sha256"]
        for selected, update0, zero in zip(
            selected_trace["records"],
            update0_trace["records"],
            zero_trace["records"],
        )
    )
    comparison_margins = {
        "update0": common_alive_prefix_planar_margin(selected_trace, update0_trace),
        "zero_action": common_alive_prefix_planar_margin(selected_trace, zero_trace),
    }
    video_verified, video = _video_evidence(
        args.video_sidecar.resolve() if args.video_sidecar else None,
        case_id=args.case,
        checkpoint_sha256=manifest["final_checkpoint"]["sha256"],
    )
    gate_metrics = _compose_gate_metrics(
        final_metrics,
        update0_metrics,
        zero_metrics,
        video_verified=video_verified,
        eval_envs=args.eval_envs,
        eval_seconds=args.eval_seconds,
        pairing_verified=pairing_verified,
        comparison_margins=comparison_margins,
    )
    if min(comparison_margins.values()) < 0.05:
        gate_metrics["paired_baselines_verified"] = False
    passed, failure_reasons = quality_gate(gate_metrics)
    report = {
        "artifact": ARTIFACT_VERSION,
        "task": "task072-bound-g1-go2-locomotion-proof",
        "case": args.case,
        "run_manifest": {
            "path": str(args.run_manifest.resolve()),
            "sha256": sha256_path(args.run_manifest.resolve()),
        },
        "static_lineage_sha256": static_sha,
        "final_checkpoint": manifest["final_checkpoint"],
        "evaluation": {
            "seed": args.seed,
            "num_envs": args.eval_envs,
            "trial_seconds": args.eval_seconds,
            "command": list(COMMAND),
        },
        "pairing": {
            "verified": pairing_verified,
            "env_initial_state_sha256": [
                record["initial_state_sha256"] for record in selected_trace["records"]
            ],
        },
        "comparisons": comparison_margins,
        "eval_trace": {
            "path": str(trace_path),
            "sha256": sha256_path(trace_path),
            "payload_sha256": payload_sha256(trace_payload),
        },
        "final_policy": final_metrics,
        "update0_baseline": update0_metrics,
        "zero_action_baseline": zero_metrics,
        "gate_metrics": gate_metrics,
        "video": video,
        "video_sidecar": (
            {
                "path": str(args.video_sidecar.resolve()),
                "sha256": sha256_path(args.video_sidecar.resolve()),
            }
            if args.video_sidecar
            else None
        ),
        "task072_case_passed": passed,
        "failure_reasons": failure_reasons,
        "runtime": _runtime_metadata(" ".join(sys.argv)),
        "claim_boundary": {
            "exact_bound_asset_evaluated": True,
            "task072_case_passed": passed,
            "task072_all_cases_passed": False,
            "task073_started": False,
        },
    }
    _atomic_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "case": args.case,
                "passed": passed,
                "failure_reasons": failure_reasons,
                "output": str(args.output.resolve()),
            }
        )
    )
    return 0 if passed else 1


def _write_video_frames(
    imageio: Any,
    output: Path,
    frames: Iterable[Any],
    *,
    expected_frames: int = VIDEO_FRAMES,
) -> tuple[int, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    midframe_path = output.with_name(f"{output.stem}_midframe.png")
    midframe = None
    count = 0
    with imageio.get_writer(
        output,
        fps=int(CONTROL_HZ),
        codec="libx264",
        pixelformat="yuv420p",
    ) as writer:
        for count, frame in enumerate(frames, start=1):
            writer.append_data(frame)
            if count == expected_frames // 2 + 1:
                midframe = frame.copy()
    _require(count == expected_frames, "Task072 video frame count mismatch while writing")
    _require(midframe is not None, "Task072 video midframe was not captured")
    imageio.imwrite(midframe_path, midframe)
    return count, midframe_path


def _decoded_frame_count(imageio: Any, path: Path) -> int:
    reader = imageio.get_reader(path)
    try:
        return sum(1 for _frame in reader)
    finally:
        reader.close()


def render_command(args: argparse.Namespace) -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    import torch

    manifest, context, _validation_shard = _validate_run_manifest(
        args.run_manifest.resolve(), expected_case=args.case
    )
    config = manifest["configuration"]
    lineage = _checkpoint_lineage(
        case_id=args.case,
        static_lineage_sha256=manifest["static_lineage_sha256"],
        run_identity_sha256=manifest["run_identity_sha256"],
        update=int(config["updates"]),
        env_steps=int(config["env_steps"]),
    )
    payload = _load_checkpoint(torch, manifest["final_checkpoint"], lineage)
    shard = _build_shard(
        context,
        num_envs=1,
        # Keep frame 400 inside the first trial.  An exactly 8.0 s trial
        # times out and auto-resets on that same post-step frame.
        trial_seconds=VIDEO_TRIAL_SECONDS,
        seed=args.seed,
        action_scale=float(config["action_scale"]),
        phase_observation=_phase_observation_enabled(config),
    )
    policy = _make_policy(shard, config, args.device)
    policy.load_state_dict(payload["policy"])
    provider = _policy_action_provider(policy, shard, args.device)
    observation = shard.reset()
    start = np.asarray(shard._canonical_state(shard.data[0]).world_position, dtype=float)
    terminal = start.copy()
    fall_count = 0
    done_count = 0
    diagnostic_rows: list[dict[str, Any]] = []
    active_slots = list(shard.embodiment.mapping.semantic_slots)
    active_indices = list(shard.embodiment.mapping.selector)
    renderer = mujoco.Renderer(shard.model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.azimuth = 135.0
    camera.elevation = -20.0
    camera.distance = max(1.6, 2.4 * float(context["case"].blueprint.nominal_height))

    def frames() -> Iterable[Any]:
        nonlocal observation, terminal, fall_count, done_count
        for frame_index in range(VIDEO_FRAMES):
            action = np.asarray(provider(observation), dtype=np.float64)
            result = shard.step(action)
            observation = result.actor_observation
            terminal = np.asarray(
                result.metrics["post_step_pre_reset_world_position"][0], dtype=float
            )
            fall_count += int(bool(result.metrics["fall"][0]))
            done_count += int(bool(result.trial_done[0]))
            diagnostic_rows.append(
                {
                    "step": frame_index,
                    "trial_done": bool(result.trial_done[0]),
                    "fall": bool(result.metrics["fall"][0]),
                    "foot_contact": [bool(value) for value in result.metrics["foot_contact"][0].tolist()],
                    "touchdown": [bool(value) for value in result.metrics["touchdown"][0].tolist()],
                    "foot_air_time": [float(value) for value in result.metrics["foot_air_time"][0].tolist()],
                    "foot_height": [float(value) for value in result.metrics["foot_height"][0].tolist()],
                    "foot_planar_speed": [float(value) for value in result.metrics["foot_planar_speed"][0].tolist()],
                    "canonical_root_world_x": float(terminal[0]),
                    "projected_gravity_xy": [
                        float(value)
                        for value in result.metrics["post_step_pre_reset_projected_gravity"][0, :2].tolist()
                    ],
                    "active_action": [float(action[0, index]) for index in active_indices],
                    "unclamped_target_by_slot": {
                        slot: float(result.metrics["unclamped_target"][0, index])
                        for slot, index in zip(active_slots, active_indices)
                    },
                    "ctrl_target_by_slot": {
                        slot: float(result.metrics["ctrl_target"][0, index])
                        for slot, index in zip(active_slots, active_indices)
                    },
                    "target_would_clamp_by_slot": [
                        bool(result.metrics["target_would_clamp"][0, index])
                        for index in active_indices
                    ],
                    "actual_clamp_by_slot": [
                        bool(result.metrics["actual_clamp"][0, index])
                        for index in active_indices
                    ],
                }
            )
            camera.lookat[:] = (
                float(terminal[0]),
                float(terminal[1]),
                max(0.2, float(terminal[2]) * 0.7),
            )
            renderer.update_scene(shard.data[0], camera=camera)
            yield np.asarray(renderer.render()).copy()

    output = args.output.resolve()
    try:
        written_count, midframe_path = _write_video_frames(imageio, output, frames())
    finally:
        renderer.close()
    decoded_count = _decoded_frame_count(imageio, output)
    displacement = float(terminal[0] - start[0])
    diagnostic_reference = None
    if args.diagnostic_output is not None:
        diagnostic_payload = {
            "schema_version": 1,
            "case_id": args.case,
            "seed": args.seed,
            "checkpoint_sha256": manifest["final_checkpoint"]["sha256"],
            "active_slots": active_slots,
            "rows": diagnostic_rows,
            "summary": _diagnostic_rollout_summary(
                diagnostic_rows, active_slot_count=len(active_slots)
            ),
        }
        diagnostic_output = args.diagnostic_output.resolve()
        _atomic_json(diagnostic_output, diagnostic_payload)
        diagnostic_reference = {
            "path": str(diagnostic_output),
            "sha256": sha256_path(diagnostic_output),
            "payload_sha256": payload_sha256(diagnostic_payload),
        }
    render_passed = bool(
        written_count == decoded_count == VIDEO_FRAMES
        and fall_count == 0
        and done_count == 0
        and displacement > 0.0
    )
    sidecar = {
        "artifact": ARTIFACT_VERSION,
        "case": args.case,
        "seed": args.seed,
        "checkpoint_sha256": manifest["final_checkpoint"]["sha256"],
        "fps": int(CONTROL_HZ),
        "frame_count": decoded_count,
        "duration_seconds": decoded_count / CONTROL_HZ,
        "width": args.width,
        "height": args.height,
        "video_path": str(output),
        "video_sha256": sha256_path(output),
        "video_bytes": output.stat().st_size,
        "midframe_path": str(midframe_path),
        "midframe_sha256": sha256_path(midframe_path),
        "fall_count": fall_count,
        "done_count": done_count,
        "forward_displacement": displacement,
        "diagnostic_rollout": diagnostic_reference,
        "render_passed": render_passed,
        "runtime": _runtime_metadata(" ".join(sys.argv)),
    }
    sidecar_path = output.with_suffix(".json")
    _atomic_json(sidecar_path, sidecar)
    print(json.dumps({"sidecar": str(sidecar_path), "render_passed": render_passed}))
    return 0 if render_passed else 1


def _verify_report(
    path: Path,
    expected_case: str,
    *,
    device: str,
) -> tuple[bool, dict[str, Any]]:
    import torch

    _require(path.is_file(), f"missing Task072 case report: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    _require(report.get("artifact") == ARTIFACT_VERSION, "case report artifact drift")
    _require(report.get("case") == expected_case, "case report identity mismatch")
    manifest_path = Path(report["run_manifest"]["path"])
    _require(sha256_path(manifest_path) == report["run_manifest"]["sha256"], "manifest SHA drift")
    manifest, context, _shard = _validate_run_manifest(manifest_path, expected_case=expected_case)
    _require(
        report["final_checkpoint"] == manifest["final_checkpoint"],
        "case report checkpoint mismatch",
    )
    config = manifest["configuration"]
    final_lineage = _checkpoint_lineage(
        case_id=expected_case,
        static_lineage_sha256=manifest["static_lineage_sha256"],
        run_identity_sha256=manifest["run_identity_sha256"],
        update=int(config["updates"]),
        env_steps=int(config["env_steps"]),
    )
    update0_lineage = _checkpoint_lineage(
        case_id=expected_case,
        static_lineage_sha256=manifest["static_lineage_sha256"],
        run_identity_sha256=manifest["run_identity_sha256"],
        update=0,
        env_steps=0,
    )
    final_payload = _load_checkpoint(torch, manifest["final_checkpoint"], final_lineage)
    update0_payload = _load_checkpoint(torch, manifest["update0_checkpoint"], update0_lineage)
    video_reference = report.get("video_sidecar")
    video_path = Path(video_reference["path"]) if video_reference else None
    if video_path is not None:
        _require(
            video_path.is_file() and sha256_path(video_path) == video_reference["sha256"],
            "video sidecar SHA drift",
        )
    video_verified, sidecar = _video_evidence(
        video_path,
        case_id=expected_case,
        checkpoint_sha256=manifest["final_checkpoint"]["sha256"],
    )
    _require(sidecar == report.get("video"), "case report/video sidecar mismatch")
    evaluation = report["evaluation"]
    _require(evaluation.get("command") == list(COMMAND), "case report command drift")
    evaluation_arguments = {
        "num_envs": int(evaluation["num_envs"]),
        "trial_seconds": float(evaluation["trial_seconds"]),
        "seed": int(evaluation["seed"]),
    }
    selected_shard = _build_shard(
        context,
        action_scale=float(config["action_scale"]),
        phase_observation=_phase_observation_enabled(config),
        **evaluation_arguments,
    )
    selected_policy = _make_policy(selected_shard, config, device)
    selected_policy.load_state_dict(final_payload["policy"])
    update0_shard = _build_shard(
        context,
        action_scale=float(config["action_scale"]),
        phase_observation=_phase_observation_enabled(config),
        **evaluation_arguments,
    )
    update0_policy = _make_policy(update0_shard, config, device)
    update0_policy.load_state_dict(update0_payload["policy"])
    zero_shard = _build_shard(
        context,
        action_scale=float(config["action_scale"]),
        phase_observation=_phase_observation_enabled(config),
        **evaluation_arguments,
    )
    selected_trace = evaluate_first_trial_trace(
        selected_shard,
        _policy_action_provider(selected_policy, selected_shard, device),
        seed=int(evaluation["seed"]),
    )
    update0_trace = evaluate_first_trial_trace(
        update0_shard,
        _policy_action_provider(update0_policy, update0_shard, device),
        seed=int(evaluation["seed"]),
    )
    zero_trace = evaluate_first_trial_trace(
        zero_shard,
        _zero_action_provider(int(evaluation["num_envs"])),
        seed=int(evaluation["seed"]),
    )
    trace_reference = report.get("eval_trace")
    _require(isinstance(trace_reference, dict), "case report missing eval trace")
    trace_path = Path(trace_reference["path"])
    _require(trace_path.is_file() and sha256_path(trace_path) == trace_reference["sha256"], "eval trace SHA drift")
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    _require(payload_sha256(trace_payload) == trace_reference["payload_sha256"], "eval trace payload SHA drift")
    expected_trace = {
        "schema_version": 1,
        "case_id": expected_case,
        "horizon_s": float(evaluation["trial_seconds"]),
        "num_envs": int(evaluation["num_envs"]),
        "eval_seed": int(evaluation["seed"]),
        "policies": {
            "selected": selected_trace,
            "untrained": update0_trace,
            "zero_action": zero_trace,
        },
    }
    _require(trace_payload == expected_trace, "eval trace replay drift")
    final_metrics = metrics_from_trace(selected_trace)
    update0_metrics = metrics_from_trace(update0_trace)
    zero_metrics = metrics_from_trace(zero_trace)
    _require(final_metrics == report["final_policy"], "final-policy evaluation replay drift")
    _require(update0_metrics == report["update0_baseline"], "update0 evaluation replay drift")
    _require(zero_metrics == report["zero_action_baseline"], "zero-action replay drift")
    pairing_verified = all(
        selected["initial_state_sha256"] == update0["initial_state_sha256"] == zero["initial_state_sha256"]
        for selected, update0, zero in zip(
            selected_trace["records"], update0_trace["records"], zero_trace["records"]
        )
    )
    comparison_margins = {
        "update0": common_alive_prefix_planar_margin(selected_trace, update0_trace),
        "zero_action": common_alive_prefix_planar_margin(selected_trace, zero_trace),
    }
    _require(report.get("pairing", {}).get("verified") is pairing_verified, "pairing verification drift")
    _require(report.get("comparisons") == comparison_margins, "comparison margin drift")
    gate_metrics = _compose_gate_metrics(
        final_metrics,
        update0_metrics,
        zero_metrics,
        video_verified=video_verified,
        eval_envs=int(evaluation["num_envs"]),
        eval_seconds=float(evaluation["trial_seconds"]),
        pairing_verified=pairing_verified,
        comparison_margins=comparison_margins,
    )
    if min(comparison_margins.values()) < 0.05:
        gate_metrics["paired_baselines_verified"] = False
    _require(gate_metrics == report["gate_metrics"], "case report gate metrics drift")
    passed, reasons = quality_gate(gate_metrics)
    return passed, {
        "case": expected_case,
        "report_path": str(path.resolve()),
        "report_sha256": sha256_path(path),
        "passed": passed,
        "failure_reasons": reasons,
    }


def verify_command(args: argparse.Namespace) -> int:
    if getattr(args, "freeze_manifest", None) is not None:
        _verify_freeze_manifest(args.freeze_manifest.resolve())
    records = [
        _verify_report(args.g1_report.resolve(), "unitree_g1", device=args.device),
        _verify_report(args.go2_report.resolve(), "unitree_go2", device=args.device),
    ]
    summaries = [summary for _passed, summary in records]
    passed = all(result for result, _summary in records)
    payload = {
        "artifact": ARTIFACT_VERSION,
        "task": "task072-bound-g1-go2-locomotion-proof",
        "denominator": 2,
        "records": summaries,
        "task072_passed": passed,
        "task073_may_start": passed,
        "runtime": _runtime_metadata(" ".join(sys.argv)),
        "failure_reasons": [
            f"{summary['case']}:{reason}"
            for summary in summaries
            for reason in summary["failure_reasons"]
        ],
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps({"task072_passed": passed, "output": str(args.output.resolve())}))
    return 0 if passed else 1


def _controlled_source_paths() -> tuple[Path, ...]:
    return (
        ROOT / "src/h200_locomotion_lab/envs/whole_body_mujoco.py",
        ROOT / "src/h200_locomotion_lab/algorithms/ppo.py",
        ROOT / "src/h200_locomotion_lab/training/whole_body_ppo.py",
        ROOT / "src/h200_locomotion_lab/policies/whole_body_mlp.py",
        ROOT / "src/h200_locomotion_lab/masked_distribution.py",
        ROOT / "src/h200_locomotion_lab/robots/whole_body_adapter.py",
        TASK_DIR / "task072_locomotion_proof.py",
        TASK_DIR / "task.md",
        TASK_DIR / "001-g1-motor-tuple-slot-action-scale.md",
        TASK_DIR / "002-g1-mature-biped-pose-contact-reward.md",
        TASK_DIR / "003-g1-fixed-command-nominal-training.md",
        TASK_DIR / "004-freeze-g1-passing-lineage.md",
        TASK_DIR / "005-go2-same-lineage-rerun.md",
        ROOT / "tests/test_task072_locomotion_proof.py",
        ROOT / "tests/test_whole_body_extended.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    )


def smoke_action_command(args: argparse.Namespace) -> int:
    import numpy as np

    cases = tuple(dict.fromkeys(args.case))
    payload = action_contract_payload(cases)
    smoke: list[dict[str, Any]] = []
    for case_id in cases:
        context = _load_bound_context(case_id)
        shard = _build_shard(
            context,
            num_envs=1,
            trial_seconds=1.0,
            seed=72072,
            action_scale=0.35,
        )
        observation = shard.reset()
        _require(np.isfinite(observation).all(), f"{case_id} reset observation is nonfinite")
        zero = np.zeros((1, 45), dtype=np.float32)
        step = shard.step(zero)
        stance_ctrl = np.asarray(
            [shard.stance_solution.actuator_ctrl[actuator.semantic_slot] for actuator in shard.blueprint.actuators],
            dtype=np.float64,
        )
        actual_ctrl = np.asarray(
            [float(shard.data[0].ctrl[int(actuator_id)]) for actuator_id in shard._actuator_ids],
            dtype=np.float64,
        )
        target_smoke_passed = bool(
            np.isfinite(step.actor_observation).all()
            and np.isfinite(step.reward).all()
            and np.allclose(actual_ctrl, stance_ctrl, rtol=0.0, atol=1e-12)
        )
        smoke.append(
            {
                "case_id": case_id,
                "slot_count": len(context["motor_tuples"]),
                "target_smoke_passed": target_smoke_passed,
                "max_zero_action_ctrl_error": float(np.max(np.abs(actual_ctrl - stance_ctrl))),
            }
        )
    payload["smoke"] = smoke
    payload["payload_sha256"] = payload_sha256({key: value for key, value in payload.items() if key != "payload_sha256"})
    _atomic_json(args.output.resolve(), payload)
    passed = all(record["target_smoke_passed"] for record in smoke)
    print(json.dumps({"smoke_action_passed": passed, "output": str(args.output.resolve())}))
    return 0 if passed else 1


def reward_contract_command(args: argparse.Namespace) -> int:
    records = []
    for case_id in tuple(dict.fromkeys(args.case)):
        payload = reward_config_payload(case_id)
        path = args.output_root.resolve() / case_id / "reward_config.json"
        _atomic_json(path, payload)
        records.append({"case_id": case_id, "path": str(path), "sha256": sha256_path(path)})
    print(json.dumps({"reward_contracts": records}))
    return 0


def verify_case_command(args: argparse.Namespace) -> int:
    passed, payload = _verify_report(args.report.resolve(), args.case, device=args.device)
    if getattr(args, "diagnostic", None) is not None:
        diagnostic_path = args.diagnostic.resolve()
        _require(diagnostic_path.is_file(), f"missing diagnostic rollout: {diagnostic_path}")
        report = json.loads(args.report.resolve().read_text(encoding="utf-8"))
        video = report.get("video") or {}
        diagnostic_reference = video.get("diagnostic_rollout")
        _require(isinstance(diagnostic_reference, dict), "video sidecar missing diagnostic rollout reference")
        _require(str(diagnostic_path) == diagnostic_reference.get("path"), "diagnostic path is not bound to video sidecar")
        _require(sha256_path(diagnostic_path) == diagnostic_reference.get("sha256"), "diagnostic raw SHA drift")
        diagnostic_payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        _require(payload_sha256(diagnostic_payload) == diagnostic_reference.get("payload_sha256"), "diagnostic payload SHA drift")
        _require(diagnostic_payload.get("case_id") == args.case, "diagnostic case mismatch")
        _require(diagnostic_payload.get("seed") == 272072, "diagnostic render seed drift")
        _require(
            diagnostic_payload.get("checkpoint_sha256") == report["final_checkpoint"]["sha256"],
            "diagnostic checkpoint SHA drift",
        )
        summary = _diagnostic_rollout_summary(
            diagnostic_payload["rows"],
            active_slot_count=len(diagnostic_payload["active_slots"]),
        )
        _require(summary == diagnostic_payload.get("summary"), "diagnostic summary drift")
        pilot_payload = pilot_gate_payload(report, summary)
        payload["pilot_gate"] = pilot_payload
        payload["pilot_gate_passed"] = pilot_payload["pilot_gate_passed"]
        if args.pilot_gate_output is not None:
            _atomic_json(args.pilot_gate_output.resolve(), pilot_payload)
        passed = passed and pilot_payload["pilot_gate_passed"]
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps({"case": args.case, "passed": passed, "output": str(args.output.resolve())}))
    return 0 if passed else 1


def _verify_freeze_manifest(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing Task072 freeze manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    commit = _git_output("rev-parse", "HEAD")
    _require(manifest.get("git", {}).get("commit") == commit, "freeze commit mismatch")
    for source in manifest.get("sources", []):
        source_path = ROOT / source["path"]
        _require(source_path.is_file(), f"missing frozen source: {source_path}")
        _require(sha256_path(source_path) == source["sha256"], f"frozen source SHA drift: {source['path']}")
        diff = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", source["path"]],
            cwd=ROOT,
            text=True,
        )
        _require(diff.returncode == 0, f"controlled source dirty: {source['path']}")
    return manifest


def verify_freeze_command(args: argparse.Namespace) -> int:
    manifest = _verify_freeze_manifest(args.manifest.resolve())
    print(json.dumps({"freeze_verified": True, "commit": manifest["git"]["commit"]}))
    return 0


def freeze_command(args: argparse.Namespace) -> int:
    report = json.loads(args.g1_report.resolve().read_text(encoding="utf-8"))
    visual = json.loads(args.g1_visual.resolve().read_text(encoding="utf-8"))
    action_contract = json.loads(args.action_contract.resolve().read_text(encoding="utf-8"))
    _require(report.get("case") == "unitree_g1" and report.get("task072_case_passed") is True, "G1 report has not passed")
    verifier_path = args.g1_report.resolve().with_name("case_verifier.json")
    _require(verifier_path.is_file(), "G1 case verifier is required before freeze")
    verifier = json.loads(verifier_path.read_text(encoding="utf-8"))
    _require(
        verifier.get("case") == "unitree_g1"
        and verifier.get("passed") is True
        and verifier.get("report_sha256") == sha256_path(args.g1_report.resolve()),
        "G1 case verifier has not passed this report",
    )
    _require(visual.get("agent_visual_check_passed") is True, "G1 visual check has not passed")
    go2_smoke = [
        record for record in action_contract.get("smoke", [])
        if record.get("case_id") == "unitree_go2"
    ]
    _require(go2_smoke and go2_smoke[0].get("target_smoke_passed") is True, "Go2 action smoke has not passed")
    controlled_sources = []
    for path in _controlled_source_paths():
        relative = str(path.relative_to(ROOT))
        diff = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative],
            cwd=ROOT,
            text=True,
        )
        _require(diff.returncode == 0, f"controlled source dirty after freeze commit: {relative}")
        controlled_sources.append({"path": relative, "sha256": sha256_path(path)})
    manifest = {
        "schema_version": 1,
        "artifact": ARTIFACT_VERSION,
        "git": {
            "commit": _git_output("rev-parse", "HEAD"),
            "branch": _git_output("branch", "--show-current"),
            "controlled_dirty": False,
        },
        "sources": controlled_sources,
        "parent_artifacts": report.get("run_manifest", {}),
        "action_contracts": {
            "path": str(args.action_contract.resolve()),
            "sha256": sha256_path(args.action_contract.resolve()),
        },
        "g1_checkpoint": report["final_checkpoint"],
        "g1_report": {"path": str(args.g1_report.resolve()), "sha256": sha256_path(args.g1_report.resolve())},
        "g1_visual": {"path": str(args.g1_visual.resolve()), "sha256": sha256_path(args.g1_visual.resolve())},
        "g1_video": report.get("video_sidecar"),
        "go2_action_smoke": go2_smoke[0],
        "go2_checkpoint": {"status": "pending"},
        "go2_report": {"status": "pending"},
        "go2_video": {"status": "pending"},
        "commands": {"freeze": " ".join(sys.argv)},
        "runtime": _runtime_metadata(" ".join(sys.argv)),
    }
    _atomic_json(args.output.resolve(), manifest)
    print(json.dumps({"freeze_manifest": str(args.output.resolve()), "commit": manifest["git"]["commit"]}))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    align_contact = subparsers.add_parser("align-contact")
    align_contact.add_argument("--output-root", type=Path, default=CONTACT_ALIGNMENT_ROOT)
    align_contact.set_defaults(function=align_contact_command)

    verify_contact = subparsers.add_parser("verify-contact-alignment")
    verify_contact.add_argument("--output-root", type=Path, default=CONTACT_ALIGNMENT_ROOT)
    verify_contact.add_argument("--output", type=Path, required=True)
    verify_contact.set_defaults(function=verify_contact_alignment_command)

    train = subparsers.add_parser("train")
    train.add_argument("--case", choices=CASES, required=True)
    train.add_argument("--stage", choices=tuple(TRAIN_STAGES), required=True)
    train.add_argument("--run-dir", type=Path)
    train.add_argument("--variant", choices=("E0", "E1_phase", "E2_reward_dt", "E3a_adaptive_kl", E3A_REPAIR_VARIANT), default="E0")
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--trial-seconds", type=float, default=10.0)
    train.add_argument("--action-scale", type=float, default=0.35)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--hidden-layers", type=int, default=2)
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--minibatch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--log-std-init", type=float, default=-1.0)
    train.add_argument("--entropy-coef", type=float, default=0.01)
    train.add_argument("--target-kl", type=float, default=0.01)
    train.add_argument("--contact-profile", choices=(MJLAB_G1_7CAPSULE_PROFILE_ID,), default=None)
    train.set_defaults(function=train_command)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--case", choices=CASES, required=True)
    evaluate.add_argument("--run-manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--video-sidecar", type=Path)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--seed", type=int, default=172072)
    evaluate.add_argument("--eval-envs", type=int, default=FINAL_EVAL_ENVS)
    evaluate.add_argument("--eval-seconds", type=float, default=FINAL_EVAL_SECONDS)
    evaluate.set_defaults(function=eval_command)

    render = subparsers.add_parser("render")
    render.add_argument("--case", choices=CASES, required=True)
    render.add_argument("--run-manifest", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--device", default="cuda:0")
    render.add_argument("--seed", type=int, default=272072)
    render.add_argument("--width", type=int, default=640)
    render.add_argument("--height", type=int, default=480)
    render.add_argument("--diagnostic-output", type=Path)
    render.set_defaults(function=render_command)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--freeze-manifest", type=Path)
    verify.add_argument("--g1-report", type=Path, required=True)
    verify.add_argument("--go2-report", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--device", default="cuda:0")
    verify.set_defaults(function=verify_command)

    smoke_action = subparsers.add_parser("smoke-action")
    smoke_action.add_argument("--case", choices=CASES, action="append", required=True)
    smoke_action.add_argument("--output", type=Path, required=True)
    smoke_action.set_defaults(function=smoke_action_command)

    reward_contract = subparsers.add_parser("reward-contract")
    reward_contract.add_argument("--case", choices=CASES, action="append", required=True)
    reward_contract.add_argument("--output-root", type=Path, required=True)
    reward_contract.set_defaults(function=reward_contract_command)

    verify_case = subparsers.add_parser("verify-case")
    verify_case.add_argument("--case", choices=CASES, required=True)
    verify_case.add_argument("--report", type=Path, required=True)
    verify_case.add_argument("--output", type=Path, required=True)
    verify_case.add_argument("--diagnostic", type=Path)
    verify_case.add_argument("--pilot-gate-output", type=Path)
    verify_case.add_argument("--device", default="cuda:0")
    verify_case.set_defaults(function=verify_case_command)

    e2_gate = subparsers.add_parser("verify-e2-scale")
    e2_gate.add_argument("--run-manifest", type=Path, required=True)
    e2_gate.add_argument("--output", type=Path, required=True)
    e2_gate.set_defaults(function=verify_e2_scale_command)

    e3_gate = subparsers.add_parser("verify-e3-optimizer")
    e3_gate.add_argument("--run-manifest", type=Path, required=True)
    e3_gate.add_argument("--output", type=Path, required=True)
    e3_gate.set_defaults(function=verify_e3_optimizer_command)

    e3a_repair = subparsers.add_parser("verify-e3a-kl-repair")
    e3a_repair.add_argument("--output", type=Path, required=True)
    e3a_repair.set_defaults(function=verify_e3a_kl_repair_command)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--g1-report", type=Path, required=True)
    freeze.add_argument("--g1-visual", type=Path, required=True)
    freeze.add_argument("--action-contract", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(function=freeze_command)

    verify_freeze = subparsers.add_parser("verify-freeze")
    verify_freeze.add_argument("--manifest", type=Path, required=True)
    verify_freeze.set_defaults(function=verify_freeze_command)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
