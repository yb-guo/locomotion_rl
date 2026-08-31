"""Task072 MJLab/RSL-RL runner for the contact-aligned anonymous G1 asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
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
if not EXTERNAL_MJLAB.exists():
    EXTERNAL_MJLAB = Path("/home/admin1/workspace/proj/locomotion_rl/.external/unitree_rl_mjlab")
GPU_LOCK = Path("/home/admin1/workspace/run/.gpu.lock")
TASK_ID = "Task072-G1-MJLab-7Capsule-SingleGround-Flat"
LINEAGE_ID = "mjlab_g1_7capsule_task_v3_single_ground"
MJLAB_PARENT_TASK = "Unitree-G1-Flat"
MAX_TRANSITIONS = 63_897_600
DEFAULT_SEED = 720301
REQUIRED_CAPACITY_NUM_ENVS = 4096
REQUIRED_ROLLOUT_STEPS = 24
REQUIRED_TRANSITIONS_PER_UPDATE = REQUIRED_CAPACITY_NUM_ENVS * REQUIRED_ROLLOUT_STEPS
RUNTIME_BINDING_ROOT = TASK_DIR / "artifacts/mjlab_runtime_binding/g1" / LINEAGE_ID
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


def action_scale_from_asset_xml() -> dict[str, float]:
    ensure_v2_artifacts()
    root = ET.parse(ASSET_XML).getroot()
    scale: dict[str, float] = {}
    for actuator in root.findall(".//actuator/position"):
        joint = actuator.get("joint")
        kp = float(actuator.get("kp", "nan"))
        force = tuple(float(value) for value in actuator.get("forcerange", "").split())
        if not joint or not force or kp <= 0.0:
            raise ValueError(f"bad position actuator scale inputs: {actuator.attrib}")
        effort = min(abs(value) for value in force)
        scale[joint] = 0.25 * effort / kp
    if len(scale) != 29:
        raise ValueError(f"expected 29 anonymous G1 action scales, got {len(scale)}")
    return scale


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
    import mujoco
    import mjlab.tasks
    import src.tasks
    from mjlab.actuator import XmlPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.sensor import ContactMatch
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls, register_mjlab_task
    from mjlab.utils.spec_config import CollisionCfg

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
    for sensor in env_cfg.scene.sensors or ():
        if sensor.name == "feet_ground_contact":
            sensor.primary = ContactMatch(
                mode="subtree",
                pattern=rf"^({FOOT_BODIES[0]}|{FOOT_BODIES[1]})$",
                entity="robot",
            )
            sensor.secondary = ContactMatch(mode="body", pattern="terrain")
            sensor.track_air_time = True
        elif sensor.name == "self_collision":
            sensor.primary = ContactMatch(mode="subtree", pattern=PELVIS_BODY, entity="robot")
            sensor.secondary = ContactMatch(mode="subtree", pattern=PELVIS_BODY, entity="robot")
    env_cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = FOOT_SITES
    if "foot_friction" in env_cfg.events:
        env_cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
    if "base_com" in env_cfg.events:
        env_cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO_BODY,)
    for name in ("body_orientation_l2", "body_ang_vel"):
        env_cfg.rewards[name].params["asset_cfg"].body_names = (TORSO_BODY,)
    for name in ("foot_clearance", "foot_slip"):
        env_cfg.rewards[name].params["asset_cfg"].site_names = FOOT_SITES
    action_cfg = env_cfg.actions["joint_pos"]
    if not isinstance(action_cfg, JointPositionActionCfg):
        raise TypeError("expected MJLab joint_pos action")
    action_cfg.actuator_names = tuple(semantic_to_joint.values())
    action_cfg.preserve_order = True
    action_cfg.use_default_offset = False
    action_cfg.offset = action_offset
    action_cfg.scale = action_scale_from_asset_xml()
    if fixed_command:
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
        env_cfg.episode_length_s = 10_000.0
        for event_name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
            env_cfg.events.pop(event_name, None)
        if "reset_base" in env_cfg.events:
            env_cfg.events["reset_base"].params["pose_range"] = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        if "reset_robot_joints" in env_cfg.events:
            env_cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
            env_cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
        env_cfg.observations["actor"].enable_corruption = False
        env_cfg.curriculum = {}

    agent_cfg.seed = int(seed)
    agent_cfg.num_steps_per_env = int(rollout_steps)
    agent_cfg.max_iterations = int(max_iterations)
    agent_cfg.save_interval = max(1, int(max_iterations))
    agent_cfg.logger = "tensorboard"
    agent_cfg.upload_model = False
    agent_cfg.resume = False
    agent_cfg.run_name = LINEAGE_ID
    agent_cfg.experiment_name = LINEAGE_ID
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
        "fixed_command": bool(fixed_command),
        "run_name": agent_cfg.run_name,
        "experiment_name": agent_cfg.experiment_name,
        "action_scale_sha256": payload_sha256(action_cfg.scale),
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
        "eval_disabled_events": ["push_robot", "foot_friction", "encoder_bias", "base_com"] if fixed_command else [],
        "eval_curriculum_disabled": bool(fixed_command),
    }
    register_mjlab_task(task_id, env_cfg, env_cfg, agent_cfg, runner_cls)
    return env_cfg, agent_cfg, runner_cls, registration


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
        hold_steps = int(round(2.0 / float(env.unwrapped.step_dt)))
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
    except BaseException as exc:
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
    except BaseException as exc:
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
    except BaseException as exc:
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
        except BaseException as exc:
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


def one_update_train(args: argparse.Namespace) -> int:
    _prepare_external_imports()
    gpu_lock = _require_gpu_lock_for_device(args.device)
    if int(args.num_envs) != REQUIRED_CAPACITY_NUM_ENVS or int(args.rollout_steps) != REQUIRED_ROLLOUT_STEPS:
        raise ValueError("Task072 MJLab training must use 4096 envs x 24 rollout steps")
    capacity_evidence = _load_capacity_evidence(args.capacity_artifact.resolve(), num_envs=args.num_envs, rollout_steps=args.rollout_steps)
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    _env_cfg, _agent_cfg, _runner_cls, registration = build_task_cfg(args.num_envs, args.rollout_steps, args.seed, args.updates)
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
    env = RslRlVecEnvWrapper(outer, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(registration["task_id"]) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), str(log_dir), args.device)
    start = time.time()
    try:
        runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=True)
    finally:
        env.close()
    checkpoints = sorted(log_dir.glob("model_*.pt"))
    payload = {
        **_common_manifest(args),
        "task_id": registration["task_id"],
        "run_dir": str(log_dir),
        "num_envs": args.num_envs,
        "rollout_steps_per_env": args.rollout_steps,
        "updates": args.updates,
        "save_interval": args.save_interval,
        "transitions_per_update": args.num_envs * args.rollout_steps,
        "observed_transitions": args.num_envs * args.rollout_steps * args.updates,
        "checkpoint_paths": [str(path) for path in checkpoints],
        "checkpoint_sha256": {str(path): sha256_path(path) for path in checkpoints},
        "gpu_lock": gpu_lock,
        "capacity_evidence": {
            "path": str(args.capacity_artifact.resolve()),
            "sha256": sha256_path(args.capacity_artifact.resolve()),
            "consumption_checks": capacity_evidence["consumption_checks"],
        },
        "wall_time_s": time.time() - start,
        "passed": len(checkpoints) > 0,
    }
    write_json((log_dir / "task072_mjlab_one_update_smoke.json"), payload)
    print(json.dumps({"passed": payload["passed"], "output": str(log_dir / "task072_mjlab_one_update_smoke.json")}), flush=True)
    return 0 if payload["passed"] else 1


def _force_fixed_command(env: Any) -> None:
    import torch

    term = env.unwrapped.command_manager.get_term("twist")
    term.vel_command_b[:] = torch.tensor((0.5, 0.0, 0.0), device=term.vel_command_b.device)
    term.is_standing_env[:] = False
    if hasattr(term, "is_heading_env"):
        term.is_heading_env[:] = False


def evaluate_checkpoint(args: argparse.Namespace) -> int:
    _prepare_external_imports()
    gpu_lock = _require_gpu_lock_for_device(args.device)
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    checkpoint = args.checkpoint.resolve()
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
        outer = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        env = RslRlVecEnvWrapper(outer, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(registration["task_id"]) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), None, args.device)
        runner.load(str(checkpoint), map_location=args.device)
        policy = runner.get_inference_policy(args.device)
        obs, _extras = env.reset()
        _force_fixed_command(env)

        robot = env.unwrapped.scene["robot"]
        contact_sensor = env.unwrapped.scene["feet_ground_contact"]
        steps = int(round(args.eval_seconds / float(env.unwrapped.step_dt)))
        start_x = robot.data.root_link_pos_w[:, 0].clone()
        fallen = torch.zeros(args.eval_envs, dtype=torch.bool, device=args.device)
        active_counts = torch.zeros(args.eval_envs, dtype=torch.float32, device=args.device)
        planar_error_sum = torch.zeros(args.eval_envs, dtype=torch.float32, device=args.device)
        yaw_error_sum = torch.zeros(args.eval_envs, dtype=torch.float32, device=args.device)
        gravity_xy_sum = torch.zeros(args.eval_envs, dtype=torch.float32, device=args.device)
        touchdown_counts = torch.zeros(2, dtype=torch.int64, device=args.device)
        single_support_counts = torch.zeros(2, dtype=torch.int64, device=args.device)
        alternating = torch.zeros(args.eval_envs, dtype=torch.int64, device=args.device)
        last_touch = torch.full((args.eval_envs,), -1, dtype=torch.int64, device=args.device)
        reward_finite = True
        obs_finite = bool(torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all())
        initial_contact = contact_sensor.data.current_contact_time > 0
        prev_contact = initial_contact.clone()

        for step in range(steps):
            active_before = ~fallen
            action = policy(obs.to(args.device))
            obs, reward, done, _extras = env.step(action.to(args.device))
            _force_fixed_command(env)
            reward_finite = bool(reward_finite and torch.isfinite(reward).all())
            obs_finite = bool(obs_finite and torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all())
            vel = robot.data.root_link_lin_vel_b
            ang = robot.data.root_link_ang_vel_b
            grav = robot.data.projected_gravity_b
            planar = torch.linalg.norm(vel[:, :2] - torch.tensor((0.5, 0.0), device=args.device), dim=1)
            active = active_before.float()
            active_counts += active
            planar_error_sum += planar * active
            yaw_error_sum += torch.abs(ang[:, 2]) * active
            gravity_xy_sum += torch.linalg.norm(grav[:, :2], dim=1) * active
            contact = contact_sensor.data.current_contact_time > 0
            touchdown = (~prev_contact) & contact
            if step > 0:
                touchdown_counts += touchdown.sum(dim=0)
                single = contact.sum(dim=1) == 1
                single_support_counts += (contact & single.unsqueeze(1)).sum(dim=0)
                labels = torch.where(
                    touchdown[:, 0] & ~touchdown[:, 1],
                    torch.zeros(args.eval_envs, dtype=torch.int64, device=args.device),
                    torch.where(
                        touchdown[:, 1] & ~touchdown[:, 0],
                        torch.ones(args.eval_envs, dtype=torch.int64, device=args.device),
                        torch.full((args.eval_envs,), -1, dtype=torch.int64, device=args.device),
                    ),
                )
                valid = labels >= 0
                alternating += (valid & (last_touch >= 0) & (labels != last_touch)).to(torch.int64)
                last_touch = torch.where(valid, labels, last_touch)
            prev_contact = contact.clone()
            fallen |= done.bool()

        final_x = robot.data.root_link_pos_w[:, 0]
        nofall = ~fallen
        displacement = final_x - start_x
        nofall_count = int(nofall.sum().detach().cpu())
        denom = torch.clamp(active_counts, min=1.0)
        mean_displacement = float(displacement[nofall].mean().detach().cpu()) if nofall_count else 0.0
        nofall_missing_value = 1.0e9
        metrics = {
            "eval_seconds": float(args.eval_seconds),
            "eval_envs": int(args.eval_envs),
            "fixed_command": {"vx": 0.5, "vy": 0.0, "yaw": 0.0},
            "zero_fall_ratio": float(nofall.float().mean().detach().cpu()),
            "mean_forward_velocity": float(mean_displacement / float(args.eval_seconds)),
            "mean_x_displacement": mean_displacement,
            "planar_tracking_error": float((planar_error_sum / denom)[nofall].mean().detach().cpu()) if nofall_count else nofall_missing_value,
            "yaw_error": float((yaw_error_sum / denom)[nofall].mean().detach().cpu()) if nofall_count else nofall_missing_value,
            "gravity_xy": float((gravity_xy_sum / denom)[nofall].mean().detach().cpu()) if nofall_count else nofall_missing_value,
            "touchdown_counts": {
                "left": int(touchdown_counts[0].detach().cpu()),
                "right": int(touchdown_counts[1].detach().cpu()),
            },
            "single_support_counts": {
                "left": int(single_support_counts[0].detach().cpu()),
                "right": int(single_support_counts[1].detach().cpu()),
            },
            "alternating_touchdown_transitions": int(alternating.sum().detach().cpu()),
            "reward_finite": reward_finite,
            "obs_finite": obs_finite,
        }
        checks = {
            "zero_fall_ratio": metrics["zero_fall_ratio"] >= 0.95,
            "mean_forward_velocity": metrics["mean_forward_velocity"] >= 0.30,
            "mean_x_displacement": metrics["mean_x_displacement"] >= 6.0,
            "planar_tracking_error": metrics["planar_tracking_error"] <= 0.35,
            "yaw_error": metrics["yaw_error"] <= 0.35,
            "gravity_xy": metrics["gravity_xy"] <= 0.35,
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
            "gpu_lock": gpu_lock,
            "metrics": metrics,
            "checks": checks,
            "passed": all(checks.values()),
            "wall_time_s": time.time() - start,
        }
        env.close()
    except BaseException as exc:
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


def _common_manifest(args: argparse.Namespace) -> dict[str, Any]:
    contact_payload = json.loads(CONTACT_PROFILE.read_text(encoding="utf-8"))
    stance_payload = json.loads(STANCE.read_text(encoding="utf-8"))
    manifest_subtask = "003f" if getattr(args, "command", None) == "verify-runtime-binding" else "003g"
    return {
        "schema_version": 2,
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
        "max_transitions": MAX_TRANSITIONS,
        "seed": getattr(args, "seed", DEFAULT_SEED),
        "runtime": _runtime_metadata(" ".join(sys.argv)),
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
    cap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "r1_capacity_smoke.json")
    cap.add_argument("--candidates", type=int, nargs="+", default=[2048, 4096, 6144])
    cap.add_argument("--rollout-steps", type=int, default=24)
    cap.add_argument("--steps", type=int, default=2)
    cap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    cap.add_argument("--device", default="cuda:0")
    cap.set_defaults(func=capacity_smoke)

    train = sub.add_parser("one-update-train")
    train.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "r1_one_update")
    train.add_argument("--capacity-artifact", type=Path, default=RUNTIME_BINDING_ROOT / "capacity_smoke_2048_4096_6144.json")
    train.add_argument("--num-envs", type=int, default=4096)
    train.add_argument("--rollout-steps", type=int, default=24)
    train.add_argument("--updates", type=int, default=1)
    train.add_argument("--save-interval", type=int, default=1)
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--device", default="cuda:0")
    train.set_defaults(func=one_update_train)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--eval-envs", type=int, default=256)
    evaluate.add_argument("--eval-seconds", type=float, default=20.0)
    evaluate.add_argument("--rollout-steps", type=int, default=24)
    evaluate.add_argument("--seed", type=int, default=DEFAULT_SEED + 99)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.set_defaults(func=evaluate_checkpoint)
    verify = sub.add_parser("verify-runtime-binding")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--seed", type=int, default=DEFAULT_SEED)
    verify.add_argument("--device", default="cpu")
    verify.set_defaults(func=verify_runtime_binding)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
