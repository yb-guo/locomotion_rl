"""Task070 archetype-constrained morphology verification.

This task-scoped tool verifies the new multi-vendor-prior profile without
reusing Task067 stance artifacts or Task069 claim semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from h200_locomotion_lab.robots.archetype_morphology import (
    DISTANCE_BANDS,
    REGION_EXPECTED_PER_FAMILY,
    TASK070_ADDITIONAL_HUMANOID_REFERENCE_IDS,
    TASK070_DISTANCE_CONTRACT_HASH,
    TASK070_PRIOR_SET_ID,
    TASK070_R0_DESIGN_CONTRACT_SHA256,
    TASK070_REFERENCE_REGISTRY_SHA256,
    TASK070_SOURCE_LICENSE_MATRIX_SHA256,
    TASK070_STANCE_CONTRACT_HASH,
    ArchetypeConstrainedMorphologyGenerator,
    MotorDofPreservingArchetypePreviewGenerator,
)
from h200_locomotion_lab.robots.procedural_morphology import (
    ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_HASH,
    ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_VERSION,
    ARCHETYPE_CONSTRAINED_MORPHOLOGY_PROFILE_VERSION,
    LOCOFORMER_MORPHOLOGY_CONTRACT_HASH,
    LOCOFORMER_MORPHOLOGY_CONTRACT_VERSION,
    MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_HASH,
    MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_VERSION,
    MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_PROFILE_VERSION,
    PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
    PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
    LocoFormerMorphologyGenerator,
    MorphologyGenerator,
    PhysicalParams,
    compile_mjcf,
    compile_with_mujoco,
    morphology_blueprint_hash,
    morphology_instance_key,
    physical_params_hash,
)
from h200_locomotion_lab.robots.whole_body_adapter import BoundEmbodiment

TASK070_FAMILIES = (
    "biped",
    "quadruped",
    "wheeled_biped",
    "wheeled_quadruped",
)
TASK070_FINAL_MATRIX_SEEDS = tuple(range(32))
TASK070_FINAL_MATRIX_EXPECTED_DENOMINATOR = len(TASK070_FAMILIES) * len(TASK070_FINAL_MATRIX_SEEDS)
STANCE_THRESHOLDS = {
    "timestep_seconds": 0.002,
    "stance_hold_steps": 1000,
    "max_abs_floor_penetration_m": 0.006,
    "max_self_penetration_m": 1e-7,
    "max_abs_roll_pitch_rad": 0.12,
    "max_qvel_norm": 1.0,
    "base_height_drift_m": 0.08,
    "minimum_terminal_load_n": 0.2,
    "max_solver_warning_count": 0,
    "max_actuator_saturation_fraction": 0.98,
    "minimum_support_margin_m": 0.035,
    "minimum_support_hull_area_m2": 0.006,
    "max_nonterminal_floor_contact_count": 0,
    "max_self_contact_count": 0,
    "max_wheel_wheel_contact_count": 0,
    "max_static_load_residual_fraction": 0.05,
    "max_contact_wrench_force_residual_fraction": 0.05,
    "max_contact_wrench_torque_residual_fraction": 0.02,
    "wheel_max_abs_speed_rad_s": 1.0,
    "wheel_max_effort_fraction": 0.9,
    "wheel_max_passive_damping": 1.0,
    "wheel_max_passive_frictionloss": 0.05,
    "wheel_velocity_hold_gain": 4.0,
    "biped_attitude_hold_hip_pitch_gain": 1.0,
    "biped_attitude_hold_ankle_pitch_gain": 0.5,
    "wheeled_biped_balance_pitch_gain": 400.0,
    "wheeled_biped_balance_pitch_rate_gain": 20.0,
    "zero_control_diagnostic_steps": 100,
    "disturbance_diagnostic_steps": 100,
}
TERMINAL_MISSING_CLEARANCE_M = 0.03
STANCE_TARGET_FLOOR_PENETRATION_M = 0.002

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = REPO_ROOT / ".agent" / "task" / "task070-archetype-constrained-standable-morphology"
DEFAULT_ARTIFACT_ROOT = TASK_ROOT / "artifacts"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_status() -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in completed.stdout.splitlines() if line]


def _runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hardware_assumption": "RTX 5060 Ti-first; local MuJoCo/headless; H200 disabled",
        "git_dirty_files_at_capture": _git_status(),
    }
    try:
        import mujoco

        metadata["mujoco_version"] = str(mujoco.__version__)
    except ImportError:
        metadata["mujoco_version"] = None
    return metadata


def _finite_array(values: Any) -> bool:
    try:
        import numpy as np

        return bool(np.isfinite(values).all())
    except (TypeError, ValueError):
        return False


def _roll_pitch_yaw(quaternion: Any) -> tuple[float, float, float]:
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return 0.0, 0.0, 0.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _object_id(model: Any, mujoco: Any, object_type: Any, name: str) -> int:
    return int(mujoco.mj_name2id(model, object_type, name))


def _terminal_geom_ids(model: Any, blueprint: Any, mujoco: Any) -> tuple[int, ...]:
    names = [f"{link.name}_footpad" for link in blueprint.links if link.foot]
    names.extend(f"{wheel.link_name}_geom" for wheel in blueprint.wheel_specs)
    ids = tuple(_object_id(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names)
    if not ids or any(geom_id < 0 for geom_id in ids):
        raise ValueError("Task070 morphology has no resolvable support terminal geoms")
    return ids


def _wheel_geom_ids(model: Any, blueprint: Any, mujoco: Any) -> tuple[int, ...]:
    ids = tuple(
        _object_id(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, f"{wheel.link_name}_geom")
        for wheel in blueprint.wheel_specs
    )
    if any(geom_id < 0 for geom_id in ids):
        raise ValueError("Task070 morphology has an unresolved wheel geom")
    return ids


def _geom_lower_z(model: Any, data: Any, geom_id: int, mujoco: Any) -> float:
    geom_type = int(model.geom_type[geom_id])
    size = [float(value) for value in model.geom_size[geom_id]]
    vertical_row = [float(value) for value in data.geom_xmat[geom_id][6:9]]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        vertical_extent = sum(abs(vertical_row[index]) * size[index] for index in range(3))
    elif geom_type in {
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    }:
        radial_vertical = math.sqrt(vertical_row[0] ** 2 + vertical_row[1] ** 2)
        vertical_extent = abs(vertical_row[2]) * size[1] + size[0] * radial_vertical
    else:
        vertical_extent = max(size)
    return float(data.geom_xpos[geom_id, 2]) - vertical_extent


def _terminal_floor_distances(
    model: Any,
    data: Any,
    terminal_ids: tuple[int, ...],
    mujoco: Any,
) -> list[dict[str, Any]]:
    return [
        {
            "geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
            "lower_z": _geom_lower_z(model, data, geom_id, mujoco),
        }
        for geom_id in terminal_ids
    ]


def _actuator_and_joint_addresses(model: Any, blueprint: Any, mujoco: Any) -> list[dict[str, Any]]:
    addresses: list[dict[str, Any]] = []
    for actuator, joint in zip(blueprint.actuators, blueprint.joints, strict=True):
        actuator_id = _object_id(model, mujoco, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator.name)
        joint_id = _object_id(model, mujoco, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
        addresses.append(
            {
                "actuator_id": actuator_id,
                "joint": joint,
                "qpos_address": int(model.jnt_qposadr[joint_id]),
                "dof_address": int(model.jnt_dofadr[joint_id]),
            }
        )
    return addresses


def _set_stance_pose(
    model: Any,
    data: Any,
    blueprint: Any,
    addresses: list[dict[str, Any]],
    *,
    root_z: float,
    mujoco: Any,
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = (0.0, 0.0, root_z)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.ctrl[:] = 0.0
    for item in addresses:
        joint = item["joint"]
        data.qpos[item["qpos_address"]] = 0.0 if joint.semantic_slot.endswith("_wheel") else joint.nominal
        if not joint.semantic_slot.endswith("_wheel"):
            data.ctrl[item["actuator_id"]] = joint.nominal
    mujoco.mj_forward(model, data)


def _prepare_stance_pose(
    model: Any,
    blueprint: Any,
    physical: PhysicalParams,
    mujoco: Any,
) -> tuple[Any, dict[str, Any], tuple[int, ...], list[dict[str, Any]]]:
    data = mujoco.MjData(model)
    terminal_ids = _terminal_geom_ids(model, blueprint, mujoco)
    addresses = _actuator_and_joint_addresses(model, blueprint, mujoco)
    _set_stance_pose(model, data, blueprint, addresses, root_z=0.0, mujoco=mujoco)
    origin = _terminal_floor_distances(model, data, terminal_ids, mujoco)
    root_z = max(
        0.001,
        -min(float(item["lower_z"]) for item in origin) - STANCE_TARGET_FLOOR_PENETRATION_M,
    )
    _set_stance_pose(model, data, blueprint, addresses, root_z=root_z, mujoco=mujoco)
    reset_distances = _terminal_floor_distances(model, data, terminal_ids, mujoco)
    floor_contacts, self_contacts = _contact_records(model, data, mujoco, step=0)
    nonterminal_floor_contacts = _nonterminal_floor_contacts(floor_contacts, terminal_ids)
    diagnostics = {
        "root_z": root_z,
        "nominal_root_z": blueprint.nominal_height * physical.global_scale,
        "root_z_source": "all_support_terminal_lower_z_with_fixed_contact_penetration",
        "target_floor_penetration_m": STANCE_TARGET_FLOOR_PENETRATION_M,
        "origin_terminal_floor_distance": origin,
        "terminal_floor_distance": reset_distances,
        "initial_floor_contact_count": len(floor_contacts),
        "initial_nonterminal_floor_contact_count": len(nonterminal_floor_contacts),
        "initial_self_contact_count": len(self_contacts),
        "initial_floor_contacts": floor_contacts,
        "initial_nonterminal_floor_contacts": nonterminal_floor_contacts,
        "initial_self_contacts": self_contacts,
        "reset_terminal_floor_clear": not any(
            float(item["lower_z"]) < -STANCE_THRESHOLDS["max_abs_floor_penetration_m"]
            for item in reset_distances
        ),
        "reset_all_terminals_reachable": not any(
            float(item["lower_z"]) > TERMINAL_MISSING_CLEARANCE_M
            for item in reset_distances
        ),
        "reset_self_collision_free": not self_contacts,
    }
    diagnostics["reset_pose_passed"] = bool(
        diagnostics["reset_terminal_floor_clear"]
        and diagnostics["reset_all_terminals_reachable"]
        and len(nonterminal_floor_contacts) <= STANCE_THRESHOLDS["max_nonterminal_floor_contact_count"]
        and diagnostics["reset_self_collision_free"]
    )
    return data, diagnostics, terminal_ids, addresses


def _contact_records(
    model: Any,
    data: Any,
    mujoco: Any,
    *,
    step: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    floor_id = _object_id(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_contacts: list[dict[str, Any]] = []
    self_contacts: list[dict[str, Any]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        record = {
            "step": step,
            "geom1_id": geom1,
            "geom2_id": geom2,
            "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
            "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
            "distance": float(contact.dist),
        }
        if floor_id in {geom1, geom2}:
            record["kind"] = "floor"
            floor_contacts.append(record)
        elif int(model.geom_bodyid[geom1]) != 0 and int(model.geom_bodyid[geom2]) != 0:
            record["kind"] = "self"
            self_contacts.append(record)
    return floor_contacts, self_contacts


def _nonterminal_floor_contacts(
    floor_contacts: list[dict[str, Any]],
    terminal_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    terminal_set = set(terminal_ids)
    return [
        contact
        for contact in floor_contacts
        if int(contact["geom1_id"]) not in terminal_set and int(contact["geom2_id"]) not in terminal_set
    ]


def _wheel_wheel_contacts(
    self_contacts: list[dict[str, Any]],
    wheel_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    wheel_set = set(wheel_ids)
    return [
        contact
        for contact in self_contacts
        if int(contact["geom1_id"]) in wheel_set and int(contact["geom2_id"]) in wheel_set
    ]


def _terminal_contact_wrench(
    model: Any,
    data: Any,
    terminal_ids: tuple[int, ...],
    mujoco: Any,
) -> dict[str, Any]:
    import numpy as np

    floor_id = _object_id(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    loads = {geom_id: 0.0 for geom_id in terminal_ids}
    terminal_set = set(terminal_ids)
    force = np.zeros(6, dtype=np.float64)
    total_world_force = np.zeros(3, dtype=np.float64)
    total_world_torque = np.zeros(3, dtype=np.float64)
    contact_records: list[dict[str, Any]] = []
    finite = True
    com = np.asarray(data.subtree_com[1] if model.nbody > 1 else data.qpos[0:3], dtype=np.float64)
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if floor_id not in {geom1, geom2}:
            continue
        terminal_id = geom2 if geom1 == floor_id else geom1
        if terminal_id not in terminal_set:
            continue
        mujoco.mj_contactForce(model, data, index, force)
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        world_force = frame.T @ force[:3]
        world_torque = frame.T @ force[3:]
        if terminal_id == geom1:
            world_force = -world_force
            world_torque = -world_torque
        position = np.asarray(contact.pos, dtype=np.float64)
        moment_about_com = np.cross(position - com, world_force) + world_torque
        contact_finite = (
            bool(np.isfinite(force).all())
            and bool(np.isfinite(world_force).all())
            and bool(np.isfinite(moment_about_com).all())
        )
        finite = finite and contact_finite
        normal_load = abs(float(force[0]))
        loads[terminal_id] += normal_load
        total_world_force += world_force
        total_world_torque += moment_about_com
        contact_records.append(
            {
                "contact_index": index,
                "terminal_geom": str(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, terminal_id)
                ),
                "other_geom": str(
                    mujoco.mj_id2name(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        geom1 if terminal_id == geom2 else geom2,
                    )
                ),
                "position": [float(value) for value in position],
                "local_force_torque": [float(value) for value in force],
                "world_force_n": [float(value) for value in world_force],
                "world_torque_about_com_nm": [float(value) for value in moment_about_com],
                "normal_load_n": normal_load,
                "finite": contact_finite,
            }
        )
    robot_weight_n = max(1e-9, float(model.body_mass.sum()) * 9.81)
    gravity_force = np.array([0.0, 0.0, -robot_weight_n], dtype=np.float64)
    net_force = total_world_force + gravity_force
    contact_radius = max(
        (
            0.1,
            *(
            float(np.linalg.norm(np.asarray(record["position"], dtype=np.float64) - com))
            for record in contact_records
            ),
        )
    )
    force_residual_fraction = float(np.linalg.norm(net_force) / robot_weight_n)
    torque_residual_fraction = float(
        np.linalg.norm(total_world_torque) / max(robot_weight_n * contact_radius, 1e-9)
    )
    load_by_name = {
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)): value
        for geom_id, value in loads.items()
    }
    return {
        "finite": finite,
        "terminal_contact_count": len(contact_records),
        "terminal_loads_n": load_by_name,
        "terminal_total_load_n": sum(load_by_name.values()),
        "robot_weight_n": robot_weight_n,
        "center_of_mass": [float(value) for value in com],
        "characteristic_contact_radius_m": contact_radius,
        "sum_world_force_n": [float(value) for value in total_world_force],
        "gravity_force_n": [float(value) for value in gravity_force],
        "net_force_residual_n": [float(value) for value in net_force],
        "force_residual_fraction": force_residual_fraction,
        "sum_world_torque_about_com_nm": [float(value) for value in total_world_torque],
        "torque_residual_fraction": torque_residual_fraction,
        "records": contact_records,
    }


def _warning_count(data: Any) -> int:
    return int(sum(int(data.warning[index].number) for index in range(len(data.warning))))


def _apply_stance_control(
    model: Any,
    data: Any,
    blueprint: Any,
    addresses: list[dict[str, Any]],
    *,
    wheel_velocity_hold: bool,
) -> None:
    gain = STANCE_THRESHOLDS["wheel_velocity_hold_gain"]
    _, pitch, _ = _roll_pitch_yaw(data.qpos[3:7])
    pitch_rate = float(data.qvel[4]) if len(data.qvel) > 4 else 0.0
    biped_attitude_hold = blueprint.family.endswith("biped")
    wheeled_biped_balance = blueprint.family == "wheeled_biped" and wheel_velocity_hold
    for item in addresses:
        joint = item["joint"]
        actuator_id = int(item["actuator_id"])
        if joint.semantic_slot.endswith("_wheel"):
            if not wheel_velocity_hold:
                data.ctrl[actuator_id] = 0.0
                continue
            if wheeled_biped_balance:
                value = (
                    STANCE_THRESHOLDS["wheeled_biped_balance_pitch_gain"] * pitch
                    + STANCE_THRESHOLDS["wheeled_biped_balance_pitch_rate_gain"] * pitch_rate
                )
            else:
                value = -gain * float(data.qvel[int(item["dof_address"])])
        else:
            value = joint.nominal
            if biped_attitude_hold and joint.axis_name == "pitch":
                if joint.semantic_slot.endswith("_hip_pitch"):
                    value += STANCE_THRESHOLDS["biped_attitude_hold_hip_pitch_gain"] * pitch
                elif joint.semantic_slot.endswith("_ankle_pitch"):
                    value += STANCE_THRESHOLDS["biped_attitude_hold_ankle_pitch_gain"] * pitch
        if bool(model.actuator_ctrllimited[actuator_id]):
            low, high = (float(v) for v in model.actuator_ctrlrange[actuator_id])
            value = min(high, max(low, value))
        data.ctrl[actuator_id] = value


def _actuator_saturation(model: Any, data: Any) -> dict[str, float]:
    max_fraction = 0.0
    wheel_fraction = 0.0
    for index in range(model.nu):
        if bool(model.actuator_forcelimited[index]):
            low, high = (float(v) for v in model.actuator_forcerange[index])
            denom = max(abs(low), abs(high), 1e-12)
            fraction = abs(float(data.actuator_force[index])) / denom
            max_fraction = max(max_fraction, fraction)
            name = str(model.actuator(index).name)
            if name.endswith("_wheel_joint_actuator"):
                wheel_fraction = max(wheel_fraction, fraction)
    return {
        "max_actuator_saturation_fraction": max_fraction,
        "wheel_max_effort_fraction": wheel_fraction,
    }


def _wheel_speed(model: Any, data: Any, addresses: list[dict[str, Any]]) -> float:
    speed = 0.0
    for item in addresses:
        joint = item["joint"]
        if joint.semantic_slot.endswith("_wheel"):
            speed = max(speed, abs(float(data.qvel[int(item["dof_address"])])))
    return speed


def _finite_fields(data: Any) -> dict[str, bool]:
    return {
        "qpos": _finite_array(data.qpos),
        "qvel": _finite_array(data.qvel),
        "qacc": _finite_array(data.qacc),
        "ctrl": _finite_array(data.ctrl),
        "qfrc_actuator": _finite_array(data.qfrc_actuator),
        "qfrc_constraint": _finite_array(data.qfrc_constraint),
        "efc_force": _finite_array(data.efc_force),
        "actuator_force": _finite_array(data.actuator_force),
        "contact_wrench": True,
    }


def _support_polygon_diagnostic(
    model: Any,
    data: Any,
    terminal_ids: tuple[int, ...],
    mujoco: Any,
) -> dict[str, Any]:
    points: list[tuple[float, float]] = []
    for geom_id in terminal_ids:
        size = [float(value) for value in model.geom_size[geom_id]]
        center = data.geom_xpos[geom_id]
        geom_type = int(model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            rot = data.geom_xmat[geom_id].reshape(3, 3)
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    point = center + rot @ [sx * size[0], sy * size[1], 0.0]
                    points.append((float(point[0]), float(point[1])))
        else:
            radius = size[0]
            half_width = size[1]
            points.extend(
                [
                    (float(center[0] - radius), float(center[1] - half_width)),
                    (float(center[0] - radius), float(center[1] + half_width)),
                    (float(center[0] + radius), float(center[1] - half_width)),
                    (float(center[0] + radius), float(center[1] + half_width)),
                ]
            )
    com = data.subtree_com[1] if model.nbody > 1 else data.qpos[0:3]
    margin = _support_margin((float(com[0]), float(com[1])), points)
    margin["com_xy"] = [float(com[0]), float(com[1])]
    margin["point_count"] = len(points)
    return margin


def _support_margin(com_xy: tuple[float, float], points: list[tuple[float, float]]) -> dict[str, Any]:
    hull = _convex_hull(points)
    if len(hull) < 3:
        return {
            "degenerate": True,
            "hull_vertices": len(hull),
            "hull_area": 0.0,
            "margin": float("-inf"),
            "inside": False,
        }
    area = 0.0
    for index in range(len(hull)):
        x1, y1 = hull[index]
        x2, y2 = hull[(index + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    margin = float("inf")
    inside = True
    for index in range(len(hull)):
        x1, y1 = hull[index]
        x2, y2 = hull[(index + 1) % len(hull)]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length <= 1e-12:
            continue
        signed = ((com_xy[0] - x1) * dy - (com_xy[1] - y1) * dx) / length
        if signed > 0.0:
            inside = False
        margin = min(margin, -signed)
    return {
        "degenerate": False,
        "hull_vertices": len(hull),
        "hull_area": abs(area) * 0.5,
        "margin": margin,
        "inside": inside,
    }


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _stance_rollout(
    model: Any,
    blueprint: Any,
    physical: PhysicalParams,
    *,
    steps: int,
    wheel_velocity_hold: bool,
    disturbance: bool,
    mujoco: Any,
) -> tuple[Any, dict[str, Any]]:
    data, reset_pose, terminal_ids, addresses = _prepare_stance_pose(
        model,
        blueprint,
        physical,
        mujoco,
    )
    wheel_ids = _wheel_geom_ids(model, blueprint, mujoco)
    if disturbance:
        data.qvel[0] = 0.05
        data.qvel[3] = 0.02
    root_z0 = float(data.qpos[2])
    max_abs_roll_pitch = 0.0
    max_qvel_norm = 0.0
    max_floor_penetration = 0.0
    max_self_penetration = 0.0
    max_terminal_missing_count = 0
    min_terminal_load = float("inf")
    max_warning_count = _warning_count(data)
    max_qacc_norm = 0.0
    max_base_height_drift = 0.0
    max_wheel_speed = 0.0
    max_actuator_saturation = 0.0
    max_wheel_effort_fraction = 0.0
    max_nonterminal_floor_contact_count = 0
    max_self_contact_count = 0
    max_wheel_wheel_contact_count = 0
    max_static_load_residual_fraction = 0.0
    final_static_load_residual_fraction = math.inf
    max_contact_wrench_force_residual_fraction = 0.0
    max_contact_wrench_torque_residual_fraction = 0.0
    final_contact_wrench_force_residual_fraction = math.inf
    final_contact_wrench_torque_residual_fraction = math.inf
    final_contact_wrench: dict[str, Any] | None = None
    robot_weight_n = max(1e-9, float(model.body_mass.sum()) * 9.81)
    finite_fields = _finite_fields(data)
    contact_samples: list[dict[str, Any]] = []

    try:
        for step in range(1, steps + 1):
            _apply_stance_control(
                model,
                data,
                blueprint,
                addresses,
                wheel_velocity_hold=wheel_velocity_hold,
            )
            mujoco.mj_step(model, data)
            roll, pitch, _ = _roll_pitch_yaw(data.qpos[3:7])
            distances = _terminal_floor_distances(model, data, terminal_ids, mujoco)
            floor_contacts, self_contacts = _contact_records(model, data, mujoco, step=step)
            nonterminal_floor_contacts = _nonterminal_floor_contacts(floor_contacts, terminal_ids)
            wheel_wheel_contacts = _wheel_wheel_contacts(self_contacts, wheel_ids)
            contact_wrench = _terminal_contact_wrench(model, data, terminal_ids, mujoco)
            loads = contact_wrench["terminal_loads_n"]
            total_terminal_load = sum(loads.values())
            static_load_residual_fraction = abs(total_terminal_load - robot_weight_n) / robot_weight_n
            contact_wrench_force_residual_fraction = float(
                contact_wrench["force_residual_fraction"]
            )
            contact_wrench_torque_residual_fraction = float(
                contact_wrench["torque_residual_fraction"]
            )
            saturation = _actuator_saturation(model, data)
            max_abs_roll_pitch = max(max_abs_roll_pitch, abs(roll), abs(pitch))
            max_qvel_norm = max(max_qvel_norm, float((data.qvel[:6] ** 2).sum() ** 0.5))
            max_qacc_norm = max(max_qacc_norm, float((data.qacc ** 2).sum() ** 0.5))
            max_base_height_drift = max(max_base_height_drift, abs(float(data.qpos[2]) - root_z0))
            max_floor_penetration = max(
                max_floor_penetration,
                max(0.0, -min(float(item["lower_z"]) for item in distances)),
            )
            max_terminal_missing_count = max(
                max_terminal_missing_count,
                sum(float(item["lower_z"]) > TERMINAL_MISSING_CLEARANCE_M for item in distances),
            )
            for contact in self_contacts:
                max_self_penetration = max(max_self_penetration, max(0.0, -float(contact["distance"])))
            min_terminal_load = min(min_terminal_load, min(loads.values()))
            max_warning_count = max(max_warning_count, _warning_count(data))
            max_wheel_speed = max(max_wheel_speed, _wheel_speed(model, data, addresses))
            max_actuator_saturation = max(
                max_actuator_saturation,
                saturation["max_actuator_saturation_fraction"],
            )
            max_wheel_effort_fraction = max(
                max_wheel_effort_fraction,
                saturation["wheel_max_effort_fraction"],
            )
            max_nonterminal_floor_contact_count = max(
                max_nonterminal_floor_contact_count,
                len(nonterminal_floor_contacts),
            )
            max_self_contact_count = max(max_self_contact_count, len(self_contacts))
            max_wheel_wheel_contact_count = max(
                max_wheel_wheel_contact_count,
                len(wheel_wheel_contacts),
            )
            max_static_load_residual_fraction = max(
                max_static_load_residual_fraction,
                static_load_residual_fraction,
            )
            final_static_load_residual_fraction = static_load_residual_fraction
            max_contact_wrench_force_residual_fraction = max(
                max_contact_wrench_force_residual_fraction,
                contact_wrench_force_residual_fraction,
            )
            max_contact_wrench_torque_residual_fraction = max(
                max_contact_wrench_torque_residual_fraction,
                contact_wrench_torque_residual_fraction,
            )
            final_contact_wrench_force_residual_fraction = contact_wrench_force_residual_fraction
            final_contact_wrench_torque_residual_fraction = contact_wrench_torque_residual_fraction
            final_contact_wrench = contact_wrench
            for key, value in _finite_fields(data).items():
                finite_fields[key] = finite_fields[key] and value
            finite_fields["contact_wrench"] = finite_fields["contact_wrench"] and bool(
                contact_wrench["finite"]
            )
            if step in {1, steps // 2, steps}:
                contact_samples.append(
                    {
                        "step": step,
                        "terminal_floor_distance": distances,
                        "terminal_loads_n": loads,
                        "terminal_total_load_n": total_terminal_load,
                        "robot_weight_n": robot_weight_n,
                        "static_load_residual_fraction": static_load_residual_fraction,
                        "contact_wrench": contact_wrench,
                        "floor_contact_count": len(floor_contacts),
                        "nonterminal_floor_contact_count": len(nonterminal_floor_contacts),
                        "nonterminal_floor_contacts": nonterminal_floor_contacts,
                        "self_contact_count": len(self_contacts),
                        "self_contacts": self_contacts,
                        "wheel_wheel_contact_count": len(wheel_wheel_contacts),
                        "wheel_wheel_contacts": wheel_wheel_contacts,
                        "roll_pitch_yaw": [roll, pitch, _roll_pitch_yaw(data.qpos[3:7])[2]],
                    }
                )
    except Exception as exc:  # noqa: BLE001
        return data, {
            "steps": steps,
            "solver_fatal": True,
            "error": f"{type(exc).__name__}: {exc}",
            "reset_pose": reset_pose,
        }
    if math.isinf(min_terminal_load):
        min_terminal_load = 0.0
    if final_contact_wrench is None:
        final_contact_wrench = _terminal_contact_wrench(model, data, terminal_ids, mujoco)
        final_contact_wrench_force_residual_fraction = float(
            final_contact_wrench["force_residual_fraction"]
        )
        final_contact_wrench_torque_residual_fraction = float(
            final_contact_wrench["torque_residual_fraction"]
        )
    support = _support_polygon_diagnostic(model, data, terminal_ids, mujoco)
    support_gate = (
        not bool(support["degenerate"])
        and bool(support["inside"])
        and float(support["margin"]) >= STANCE_THRESHOLDS["minimum_support_margin_m"]
        and float(support["hull_area"]) >= STANCE_THRESHOLDS["minimum_support_hull_area_m2"]
    )
    passed = (
        all(finite_fields.values())
        and max_warning_count <= STANCE_THRESHOLDS["max_solver_warning_count"]
        and max_floor_penetration <= STANCE_THRESHOLDS["max_abs_floor_penetration_m"]
        and max_self_penetration <= STANCE_THRESHOLDS["max_self_penetration_m"]
        and max_nonterminal_floor_contact_count <= STANCE_THRESHOLDS["max_nonterminal_floor_contact_count"]
        and max_self_contact_count <= STANCE_THRESHOLDS["max_self_contact_count"]
        and max_wheel_wheel_contact_count <= STANCE_THRESHOLDS["max_wheel_wheel_contact_count"]
        and max_terminal_missing_count == 0
        and support_gate
        and final_static_load_residual_fraction <= STANCE_THRESHOLDS["max_static_load_residual_fraction"]
        and final_contact_wrench_force_residual_fraction
        <= STANCE_THRESHOLDS["max_contact_wrench_force_residual_fraction"]
        and final_contact_wrench_torque_residual_fraction
        <= STANCE_THRESHOLDS["max_contact_wrench_torque_residual_fraction"]
        and max_abs_roll_pitch <= STANCE_THRESHOLDS["max_abs_roll_pitch_rad"]
        and max_qvel_norm <= STANCE_THRESHOLDS["max_qvel_norm"]
        and max_base_height_drift <= STANCE_THRESHOLDS["base_height_drift_m"]
        and min_terminal_load >= STANCE_THRESHOLDS["minimum_terminal_load_n"]
        and max_actuator_saturation <= STANCE_THRESHOLDS["max_actuator_saturation_fraction"]
        and (
            not blueprint.family.startswith("wheeled_")
            or (
                max_wheel_speed <= STANCE_THRESHOLDS["wheel_max_abs_speed_rad_s"]
                and max_wheel_effort_fraction <= STANCE_THRESHOLDS["wheel_max_effort_fraction"]
            )
        )
    )
    return data, {
        "steps": steps,
        "timestep_seconds": STANCE_THRESHOLDS["timestep_seconds"],
        "duration_seconds": steps * STANCE_THRESHOLDS["timestep_seconds"],
        "controller": (
            "biped_base_attitude_hold_plus_wheeled_biped_active_wheel_balance"
            if wheel_velocity_hold and blueprint.family == "wheeled_biped"
            else (
                "biped_base_attitude_hold_or_quadruped_position_feedforward_plus_wheel_zero_velocity_hold"
                if wheel_velocity_hold
                else "zero_control_diagnostic"
            )
        ),
        "disturbance": disturbance,
        "solver_fatal": False,
        "error": None,
        "finite_fields": finite_fields,
        "finite": all(finite_fields.values()),
        "warning_count": max_warning_count,
        "warning_free": max_warning_count == 0,
        "max_abs_floor_penetration_m": max_floor_penetration,
        "max_self_penetration_m": max_self_penetration,
        "max_nonterminal_floor_contact_count": max_nonterminal_floor_contact_count,
        "max_self_contact_count": max_self_contact_count,
        "max_wheel_wheel_contact_count": max_wheel_wheel_contact_count,
        "max_terminal_missing_count": max_terminal_missing_count,
        "minimum_terminal_load_n": min_terminal_load,
        "robot_weight_n": robot_weight_n,
        "max_static_load_residual_fraction": max_static_load_residual_fraction,
        "final_static_load_residual_fraction": final_static_load_residual_fraction,
        "max_contact_wrench_force_residual_fraction": max_contact_wrench_force_residual_fraction,
        "max_contact_wrench_torque_residual_fraction": max_contact_wrench_torque_residual_fraction,
        "final_contact_wrench_force_residual_fraction": final_contact_wrench_force_residual_fraction,
        "final_contact_wrench_torque_residual_fraction": final_contact_wrench_torque_residual_fraction,
        "final_contact_wrench": final_contact_wrench,
        "max_abs_roll_pitch_rad": max_abs_roll_pitch,
        "max_qvel_norm": max_qvel_norm,
        "max_qacc_norm": max_qacc_norm,
        "base_height_drift_m": max_base_height_drift,
        "max_actuator_saturation_fraction": max_actuator_saturation,
        "wheel_max_abs_speed_rad_s": max_wheel_speed,
        "wheel_max_effort_fraction": max_wheel_effort_fraction,
        "support_polygon": support,
        "support_gate_passed": support_gate,
        "contact_samples": contact_samples,
        "reset_pose": reset_pose,
        "passed": passed,
    }


def _finite_physics_checks(model: Any) -> dict[str, Any]:
    # MuJoCo body ids 0 and 1 are the world and massless free-root container.
    generated_body_mass = model.body_mass[2:]
    finite = (
        _finite_array(generated_body_mass)
        and _finite_array(model.dof_M0)
        and bool(len(generated_body_mass) and (generated_body_mass > 0.0).all())
        and bool(len(model.dof_M0) and (model.dof_M0 > 0.0).all())
    )
    return {
        "finite": finite,
        "world_and_root_container_excluded": True,
        "positive_body_mass": bool(len(generated_body_mass) and (generated_body_mass > 0.0).all()),
        "positive_dof_inertia": bool(len(model.dof_M0) and (model.dof_M0 > 0.0).all()),
        "minimum_generated_body_mass": float(generated_body_mass.min())
        if len(generated_body_mass)
        else None,
        "minimum_dof_inertia": float(model.dof_M0.min()) if len(model.dof_M0) else None,
    }


def _wheel_xml_checks(blueprint: Any, xml: str) -> dict[str, Any]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    joints = {str(item.get("name")): item for item in root.iter("joint")}
    geoms = {str(item.get("name")): item for item in root.iter("geom")}
    actuators = {str(item.get("name")): item for item in root.find("actuator") or ()}
    checks: list[dict[str, Any]] = []
    for wheel in blueprint.wheel_specs:
        joint = joints.get(wheel.joint_name)
        geom = geoms.get(f"{wheel.link_name}_geom")
        actuator = actuators.get(f"{wheel.joint_name}_actuator")
        checks.append(
            {
                "semantic_slot": wheel.semantic_slot,
                "joint_present": joint is not None,
                "joint_continuous": joint is not None and joint.get("limited") == "false",
                "joint_damping": float(joint.get("damping", "nan")) if joint is not None else None,
                "joint_frictionloss": float(joint.get("frictionloss", "nan")) if joint is not None else None,
                "passive_brake_not_locked": joint is not None
                and float(joint.get("damping", "inf")) <= STANCE_THRESHOLDS["wheel_max_passive_damping"]
                and float(joint.get("frictionloss", "inf"))
                <= STANCE_THRESHOLDS["wheel_max_passive_frictionloss"],
                "geom_present": geom is not None,
                "geom_type": geom.get("type") if geom is not None else None,
                "geom_contact": geom is not None
                and geom.get("contype") == "1"
                and geom.get("conaffinity") == "1",
                "actuator_present": actuator is not None,
                "actuator_type": actuator.tag if actuator is not None else None,
            }
        )
    expected_wheels = 2 if blueprint.family == "wheeled_biped" else 4
    if not blueprint.family.startswith("wheeled_"):
        return {
            "wheel_count": 0,
            "passed": not blueprint.wheel_specs
            and not any(slot.endswith("_wheel") for slot in blueprint.active_slots),
            "checks": checks,
        }
    return {
        "wheel_count": len(blueprint.wheel_specs),
        "expected_wheel_count": expected_wheels,
        "checks": checks,
        "passed": len(blueprint.wheel_specs) == expected_wheels
        and all(
            item["joint_present"]
            and item["joint_continuous"]
            and item["passive_brake_not_locked"]
            and item["geom_present"]
            and item["geom_type"] == "cylinder"
            and item["geom_contact"]
            and item["actuator_present"]
            and item["actuator_type"] == "motor"
            for item in checks
        ),
    }


def _region_checks(blueprint: Any, generator: ArchetypeConstrainedMorphologyGenerator) -> dict[str, Any]:
    metadata = dict(blueprint.profile_metadata)
    region = str(metadata["sampling_region"])
    distance = float(metadata["nearest_prior_distance"])
    lower, upper = DISTANCE_BANDS[region]
    return {
        "expected_region": generator.expected_sampling_region(blueprint.seed),
        "sampling_region": region,
        "nearest_prior": metadata["nearest_prior"],
        "nearest_prior_distance": distance,
        "region_band": [lower, upper],
        "region_band_passed": lower <= distance <= upper,
        "clone_guard_passed": bool(metadata["clone_guard"]["passed"]),
        "prior_contribution": metadata["prior_contribution"],
        "normalized_feature_vector": metadata["normalized_feature_vector"],
        "retry_trace": metadata["retry_trace"],
    }


def _matrix_record(
    generator: ArchetypeConstrainedMorphologyGenerator,
    family: str,
    seed: int,
    *,
    steps: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "family": family,
        "seed": seed,
        "range_fraction": 0.5,
        "status": "failed",
        "built": False,
        "compiled": False,
        "stance_passed": False,
        "error": None,
    }
    try:
        blueprint = generator.generate(family, seed)  # type: ignore[arg-type]
        repeat = generator.generate(family, seed)  # type: ignore[arg-type]
        physical = generator.sample_physical_params(blueprint, seed + 30_000_000, range_fraction=0.5)
        repeat_physical = generator.sample_physical_params(
            repeat,
            seed + 30_000_000,
            range_fraction=0.5,
        )
        xml = compile_mjcf(blueprint, physical)
        repeat_xml = compile_mjcf(repeat, repeat_physical)
        key = morphology_instance_key(blueprint, physical)
        record.update(
            {
                "built": True,
                "profile_version": blueprint.profile_version,
                "contract_version": blueprint.contract_version,
                "contract_hash": blueprint.contract_hash,
                "structural_hash": blueprint.structural_hash,
                "blueprint_hash": morphology_blueprint_hash(blueprint),
                "physical_hash": physical_params_hash(physical),
                "instance_key": key.manifest() | {"cache_key": key.cache_key},
                "xml_sha256": sha256_bytes(xml.encode("utf-8")),
                "active_slots": blueprint.active_slots,
                "active_slot_mask": blueprint.active_slot_mask,
                "joint_count": len(blueprint.joints),
                "actuator_count": len(blueprint.actuators),
                "wheel_count": len(blueprint.wheel_specs),
                "blueprint_manifest": blueprint.manifest(),
                "physical_manifest": physical.manifest(),
                "profile_metadata": dict(blueprint.profile_metadata),
                "region_checks": _region_checks(blueprint, generator),
                "determinism_checks": {
                    "blueprint_manifest": blueprint.manifest() == repeat.manifest(),
                    "physical_manifest": physical.manifest() == repeat_physical.manifest(),
                    "xml_sha256": sha256_bytes(xml.encode("utf-8"))
                    == sha256_bytes(repeat_xml.encode("utf-8")),
                },
                "physical_continuous_identity_separate": physical_params_hash(physical)
                != physical_params_hash(
                    generator.sample_physical_params(
                        blueprint,
                        seed + 30_000_001,
                        range_fraction=0.5,
                    )
                ),
            }
        )
        record["determinism_passed"] = all(record["determinism_checks"].values())
        model = compile_with_mujoco(xml)
        record["compiled"] = True
        record["model_nq"] = int(model.nq)
        record["model_nv"] = int(model.nv)
        record["model_nu"] = int(model.nu)
        import mujoco

        record["finite_physics"] = _finite_physics_checks(model)
        record["finite_physics_passed"] = bool(record["finite_physics"]["finite"])
        mapping = BoundEmbodiment.from_blueprint(blueprint, physical=physical).mapping
        robot_values = tuple(float(index) for index in range(len(blueprint.joints)))
        record["slot_mapping"] = {
            "selector": mapping.selector,
            "mask": mapping.mask,
            "active_count": mapping.active_count,
            "round_trip": mapping.round_trip(robot_values),
        }
        record["slot_mapping_passed"] = mapping.round_trip(robot_values) == robot_values
        record["wheel_xml"] = _wheel_xml_checks(blueprint, xml)
        record["wheel_topology_passed"] = bool(record["wheel_xml"]["passed"])
        _, stance = _stance_rollout(
            model,
            blueprint,
            physical,
            steps=steps,
            wheel_velocity_hold=True,
            disturbance=False,
            mujoco=mujoco,
        )
        record["stance_hold"] = stance
        record["reset_pose"] = stance["reset_pose"]
        record["stance_passed"] = bool(stance["passed"])
        _, zero = _stance_rollout(
            compile_with_mujoco(xml),
            blueprint,
            physical,
            steps=int(STANCE_THRESHOLDS["zero_control_diagnostic_steps"]),
            wheel_velocity_hold=False,
            disturbance=False,
            mujoco=mujoco,
        )
        record["zero_control_diagnostic"] = zero
        _, disturbed = _stance_rollout(
            compile_with_mujoco(xml),
            blueprint,
            physical,
            steps=int(STANCE_THRESHOLDS["disturbance_diagnostic_steps"]),
            wheel_velocity_hold=True,
            disturbance=True,
            mujoco=mujoco,
        )
        record["disturbance_diagnostic"] = disturbed
        record["status"] = "passed" if all(
            (
                record["compiled"],
                record["determinism_passed"],
                record["finite_physics_passed"],
                record["slot_mapping_passed"],
                record["wheel_topology_passed"],
                record["region_checks"]["region_band_passed"],
                record["region_checks"]["clone_guard_passed"],
                record["reset_pose"]["reset_pose_passed"],
                record["stance_passed"],
            )
        ) else "failed"
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _decode_image(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
        return {
            "path": str(path),
            "sha256": sha256_path(path),
            "viewer_decode": True,
            "width": width,
            "height": height,
            "mode": mode,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "path": str(path),
            "sha256": sha256_path(path) if path.exists() else None,
            "viewer_decode": False,
            "width": None,
            "height": None,
            "mode": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _render_frame(
    blueprint: Any,
    physical: PhysicalParams,
    path: Path,
    *,
    stage: str,
    view: str,
    steps: int,
) -> dict[str, Any]:
    try:
        import mujoco
        import numpy as np
        from PIL import Image, ImageDraw

        model = mujoco.MjModel.from_xml_string(compile_mjcf(blueprint, physical))
        data, _, _, _ = _prepare_stance_pose(model, blueprint, physical, mujoco)
        if stage == "verified":
            data, _ = _stance_rollout(
                model,
                blueprint,
                physical,
                steps=steps,
                wheel_velocity_hold=True,
                disturbance=False,
                mujoco=mujoco,
            )
        width, height = 480, 320
        renderer = mujoco.Renderer(model, height=height, width=width)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        base = "biped" if blueprint.family.endswith("biped") else "quadruped"
        camera.lookat[:] = (
            float(data.qpos[0]),
            float(data.qpos[1]),
            float(data.qpos[2]) * (0.48 if base == "biped" else 0.42),
        )
        camera.distance = 3.2 if base == "biped" else 2.3
        if blueprint.profile_version == MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_PROFILE_VERSION:
            camera.lookat[:] = (
                float(data.qpos[0]),
                float(data.qpos[1]),
                float(data.qpos[2]) * 0.54,
            )
            camera.distance = float(
                blueprint.profile_metadata.get("render_camera_distance", 2.35)
            )
        settings = {
            "oblique": (135.0, -10.0),
            "side": (90.0, -8.0),
            "front": (180.0, -8.0),
            "contact": (135.0, -28.0),
            "wheel_axis": (180.0, -4.0),
        }
        camera.azimuth, camera.elevation = settings[view]
        renderer.update_scene(data, camera=camera)
        pixels = np.asarray(renderer.render()).copy()
        renderer.close()
        if pixels.ndim != 3 or pixels.shape[-1] != 3 or int(pixels.max()) <= 5:
            raise RuntimeError("renderer produced a black or malformed frame")
        image = Image.fromarray(pixels.astype("uint8"), mode="RGB")
        draw = ImageDraw.Draw(image)
        label = f"{blueprint.family} seed={blueprint.seed} {stage} {view}"
        draw.rectangle((0, 0, min(330, 7 * len(label)), 20), fill=(0, 0, 0))
        draw.text((4, 4), label, fill=(255, 255, 255))
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")
        return _decode_image(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": str(path),
            "sha256": sha256_path(path) if path.exists() else None,
            "viewer_decode": False,
            "width": None,
            "height": None,
            "mode": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _write_raw_vs_verified(
    blueprint: Any,
    physical: PhysicalParams,
    path: Path,
    *,
    steps: int,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    raw_path = path.with_name(path.stem + "_raw.png")
    verified_path = path.with_name(path.stem + "_verified.png")
    raw = _render_frame(blueprint, physical, raw_path, stage="raw_reset", view="side", steps=steps)
    verified = _render_frame(
        blueprint,
        physical,
        verified_path,
        stage="verified",
        view="side",
        steps=steps,
    )
    if not raw["viewer_decode"] or not verified["viewer_decode"]:
        return {
            "path": str(path),
            "sha256": None,
            "viewer_decode": False,
            "raw": raw,
            "verified": verified,
            "error": "raw or verified frame failed to render",
        }
    with Image.open(raw_path) as left, Image.open(verified_path) as right:
        image = Image.new("RGB", (left.width + right.width, left.height), (24, 24, 24))
        image.paste(left.convert("RGB"), (0, 0))
        image.paste(right.convert("RGB"), (left.width, 0))
    draw = ImageDraw.Draw(image)
    draw.line((left.width, 0, left.width, image.height), fill=(255, 255, 255), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return _decode_image(path) | {"raw": raw, "verified": verified}


def _write_gallery(
    root: Path,
    generator: ArchetypeConstrainedMorphologyGenerator,
    *,
    seeds: range,
    steps: int,
) -> dict[str, Any]:
    from PIL import Image

    gallery: dict[str, Any] = {}
    for family in TASK070_FAMILIES:
        family_root = root / family
        family_root.mkdir(parents=True, exist_ok=True)
        tiles: list[Any | None] = []
        samples: list[dict[str, Any]] = []
        for seed in seeds:
            blueprint = generator.generate(family, seed)  # type: ignore[arg-type]
            physical = generator.sample_physical_params(
                blueprint,
                seed + 30_000_000,
                range_fraction=0.5,
            )
            tile_path = family_root / f"seed_{seed:02d}_verified_oblique.png"
            result = _render_frame(
                blueprint,
                physical,
                tile_path,
                stage="verified",
                view="oblique",
                steps=steps,
            )
            samples.append(result)
            if result["viewer_decode"]:
                with Image.open(tile_path) as image:
                    tiles.append(image.convert("RGB").resize((240, 160)))
            else:
                tiles.append(None)
        montage = Image.new("RGB", (8 * 240, 4 * 160), (24, 24, 24))
        for index, tile in enumerate(tiles):
            if tile is not None:
                montage.paste(tile, ((index % 8) * 240, (index // 8) * 160))
        montage_path = family_root / "montage_verified.png"
        montage.save(montage_path, format="PNG")
        closeups: list[dict[str, Any]] = []
        view_plan = [
            (0, "prior_neighborhood", "oblique"),
            (1, "interpolation_band", "side"),
            (3, "bounded_outward_band", "front"),
            (3, "bounded_outward_band", "contact"),
        ]
        if family.startswith("wheeled_"):
            view_plan.append((3, "bounded_outward_band", "wheel_axis"))
        for seed, region, view in view_plan:
            blueprint = generator.generate(family, seed)  # type: ignore[arg-type]
            physical = generator.sample_physical_params(
                blueprint,
                seed + 30_000_000,
                range_fraction=0.5,
            )
            closeups.append(
                _render_frame(
                    blueprint,
                    physical,
                    family_root / f"closeup_seed_{seed:02d}_{region}_{view}.png",
                    stage="verified",
                    view=view,
                    steps=steps,
                )
            )
        raw_vs = []
        for seed in (0, 3):
            blueprint = generator.generate(family, seed)  # type: ignore[arg-type]
            physical = generator.sample_physical_params(
                blueprint,
                seed + 30_000_000,
                range_fraction=0.5,
            )
            raw_vs.append(
                _write_raw_vs_verified(
                    blueprint,
                    physical,
                    family_root / f"raw_vs_verified_seed_{seed:02d}.png",
                    steps=steps,
                )
            )
        gallery[family] = {
            "expected_samples": len(seeds),
            "rendered_samples": sum(bool(item["viewer_decode"]) for item in samples),
            "montage": _decode_image(montage_path),
            "sample_frames": samples,
            "closeups": closeups,
            "raw_vs_verified": raw_vs,
            }
    return gallery


def write_v2_preview(
    output_dir: Path = DEFAULT_ARTIFACT_ROOT / "preview_task070_v2" / "unitree_g1_seed000",
    *,
    seed: int = 0,
    render: bool = True,
    family: str = "biped",
    reference_id: str = "unitree_g1",
) -> dict[str, Any]:
    """Write one motor-DoF-preserving v2 witness for user inspection."""

    output_dir.mkdir(parents=True, exist_ok=True)
    generator = MotorDofPreservingArchetypePreviewGenerator(reference_id=reference_id)
    blueprint = generator.generate(family, seed)
    physical = generator.sample_physical_params(blueprint, seed + 70_000_000)
    xml = compile_mjcf(blueprint, physical)
    model = compile_with_mujoco(xml)
    dof_count = len(blueprint.joints)
    wheeled_suffix = "_wheeled" if blueprint.is_wheeled else ""
    artifact_stem = f"{reference_id}_{dof_count}dof{wheeled_suffix}"
    xml_path = output_dir / f"{artifact_stem}_anonymous_preview.xml"
    descriptor_path = output_dir / f"{artifact_stem}_structural_descriptor.json"
    manifest_path = output_dir / f"{artifact_stem}_anonymous_preview_manifest.json"
    image_path = output_dir / f"{artifact_stem}_anonymous_preview_sheet.png"
    xml_path.write_text(xml, encoding="utf-8")
    metadata = dict(blueprint.profile_metadata)
    descriptor_payload = dict(metadata["source_tree_descriptor"])
    descriptor_payload["descriptor_sha256"] = metadata["structural_descriptor_sha256"]
    _write_json(descriptor_path, descriptor_payload)
    if render:
        from PIL import Image

        frames = []
        for view in ("front", "side", "oblique", "contact"):
            frames.append(
                _render_frame(
                    blueprint,
                    physical,
                    output_dir / f"{artifact_stem}_anonymous_preview_{view}.png",
                    stage="raw_reset",
                    view=view,
                    steps=1,
                )
            )
        if all(bool(frame["viewer_decode"]) for frame in frames):
            with Image.open(frames[0]["path"]) as first:
                width, height = first.size
            sheet = Image.new("RGB", (width * len(frames), height), (24, 24, 24))
            for index, frame in enumerate(frames):
                with Image.open(frame["path"]) as image:
                    sheet.paste(image.convert("RGB"), (index * width, 0))
            sheet.save(image_path, format="PNG")
            render_result = _decode_image(image_path) | {"frames": frames}
        else:
            render_result = {
                "path": str(image_path),
                "viewer_decode": False,
                "frames": frames,
                "error": "one or more preview frames failed to render",
            }
    else:
        render_result = {"path": str(image_path), "viewer_decode": False, "error": "render_not_requested"}
    payload = {
        "task": "task070-archetype-constrained-standable-morphology",
        "status": "descriptor_driven_preview_pending_agent_visual_check",
        "counts_toward_task070_v2_pass": False,
        "agent_visual_check_passed": False,
        "user_visual_acceptance": False,
        "profile_version": MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_PROFILE_VERSION,
        "contract_version": MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_VERSION,
        "contract_hash": MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_HASH,
        "family": blueprint.family,
        "seed": seed,
        "source_reference_id": metadata["source_reference_id"],
        "structural_center_id": metadata["structural_center_id"],
        "structural_descriptor_sha256": metadata["structural_descriptor_sha256"],
        "descriptor_sha256": sha256_path(descriptor_path),
        "motor_accounting": metadata["motor_accounting"],
        "motor_configuration": metadata["motor_configuration"],
        "actuation_stack": metadata["actuation_stack"],
        "canonical_root_frame": metadata["canonical_root_frame"],
        "source_body_tree_edge_coverage": {
            "selected_motor_edge_count": len(metadata["source_to_anonymous_motor_bijection"]),
            "anonymous_non_wheel_joint_edge_count": metadata["motor_accounting"][
                "anonymous_non_wheel_motor_count"
            ],
            "added_wheel_joint_edge_count": metadata["motor_accounting"][
                "added_wheel_motor_count"
            ],
            "all_selected_motor_edges_preserved": (
                len(metadata["source_to_anonymous_motor_bijection"])
                == metadata["motor_accounting"]["source_non_wheel_motor_count"]
                == metadata["motor_accounting"]["anonymous_non_wheel_motor_count"]
            ),
        },
        "primitive_geometry_only": metadata["primitive_geometry_only"],
        "mesh_texture_logo_copied": metadata["mesh_texture_logo_copied"],
        "stance_claim": metadata["stance_claim"],
        "compiled": True,
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "model_nu": int(model.nu),
        "joint_count": len(blueprint.joints),
        "actuator_count": len(blueprint.actuators),
        "link_count": len(blueprint.links),
        "blueprint_hash": morphology_blueprint_hash(blueprint),
        "physical_hash": physical_params_hash(physical),
        "xml_sha256": sha256_path(xml_path),
        "paths": {
            "xml": str(xml_path),
            "descriptor": str(descriptor_path),
            "manifest": str(manifest_path),
            "image": str(image_path) if render else None,
        },
        "render": render_result,
        "blueprint_manifest": blueprint.manifest(),
        "physical_manifest": physical.manifest(),
        "source_sha256": {
            "archetype_morphology.py": sha256_path(
                REPO_ROOT / "src" / "h200_locomotion_lab" / "robots" / "archetype_morphology.py"
            ),
            "procedural_morphology.py": sha256_path(
                REPO_ROOT / "src" / "h200_locomotion_lab" / "robots" / "procedural_morphology.py"
            ),
            "verification_tool.py": sha256_path(Path(__file__)),
        },
        "runtime": _runtime_metadata(),
    }
    if "candidate_source_evidence" in metadata:
        payload.update(
            {
                "candidate_prior_status": metadata["candidate_prior_status"],
                "policy_adapter_compatible": metadata[
                    "policy_adapter_compatible"
                ],
                "task070_candidate_extra_semantic_slots": metadata[
                    "task070_candidate_extra_semantic_slots"
                ],
                "candidate_source_evidence": metadata[
                    "candidate_source_evidence"
                ],
            }
        )
    _write_json(manifest_path, payload)
    return payload


TASK070_V2_CURRENT_ARENA_CASES: tuple[tuple[str, str], ...] = (
    ("unitree_g1", "biped"),
    ("unitree_g1", "wheeled_biped"),
    ("engineai_pm01", "biped"),
    ("engineai_pm01", "wheeled_biped"),
    ("spot_base", "quadruped"),
    ("spot_base", "wheeled_quadruped"),
    ("unitree_go2", "quadruped"),
    ("unitree_go2", "wheeled_quadruped"),
    ("deeprobotics_lite3", "quadruped"),
    ("deeprobotics_lite3", "wheeled_quadruped"),
)


def _arena_actuator_response(
    model: Any,
    blueprint: Any,
    physical: PhysicalParams,
    *,
    response_steps: int,
    mujoco: Any,
) -> dict[str, Any]:
    if response_steps <= 0:
        raise ValueError("response_steps must be positive")
    records: list[dict[str, Any]] = []
    for selected_index in range(len(blueprint.joints)):
        data, reset_pose, _, addresses = _prepare_stance_pose(
            model,
            blueprint,
            physical,
            mujoco,
        )
        baseline_data, _, _, baseline_addresses = _prepare_stance_pose(
            model,
            blueprint,
            physical,
            mujoco,
        )
        selected = addresses[selected_index]
        joint = selected["joint"]
        qpos_address = int(selected["qpos_address"])
        dof_address = int(selected["dof_address"])
        actuator_id = int(selected["actuator_id"])
        qpos_start = float(data.qpos[qpos_address])
        max_abs_velocity = 0.0
        max_abs_force = 0.0
        max_command_induced_qpos_delta = 0.0
        max_command_induced_qvel_delta = 0.0
        max_command_induced_force_delta = 0.0
        finite = True
        warning_count = max(_warning_count(data), _warning_count(baseline_data))
        wheel = joint.semantic_slot.endswith("_wheel")
        if wheel:
            low, high = (float(value) for value in model.actuator_ctrlrange[actuator_id])
            target = 0.12 * max(abs(low), abs(high))
        else:
            low, high = (float(value) for value in model.actuator_ctrlrange[actuator_id])
            room_positive = high - float(joint.nominal)
            room_negative = float(joint.nominal) - low
            amplitude = min(0.08, 0.16 * max(0.0, high - low))
            if room_positive >= room_negative:
                target = min(high, float(joint.nominal) + amplitude)
            else:
                target = max(low, float(joint.nominal) - amplitude)
        for _ in range(response_steps):
            _apply_stance_control(
                model,
                data,
                blueprint,
                addresses,
                wheel_velocity_hold=True,
            )
            _apply_stance_control(
                model,
                baseline_data,
                blueprint,
                baseline_addresses,
                wheel_velocity_hold=True,
            )
            data.ctrl[actuator_id] = target
            mujoco.mj_step(model, data)
            mujoco.mj_step(model, baseline_data)
            max_abs_velocity = max(
                max_abs_velocity,
                abs(float(data.qvel[dof_address])),
            )
            max_abs_force = max(
                max_abs_force,
                abs(float(data.actuator_force[actuator_id])),
            )
            max_command_induced_qpos_delta = max(
                max_command_induced_qpos_delta,
                abs(
                    float(data.qpos[qpos_address])
                    - float(baseline_data.qpos[qpos_address])
                ),
            )
            max_command_induced_qvel_delta = max(
                max_command_induced_qvel_delta,
                abs(
                    float(data.qvel[dof_address])
                    - float(baseline_data.qvel[dof_address])
                ),
            )
            max_command_induced_force_delta = max(
                max_command_induced_force_delta,
                abs(
                    float(data.actuator_force[actuator_id])
                    - float(baseline_data.actuator_force[actuator_id])
                ),
            )
            finite = (
                finite
                and all(_finite_fields(data).values())
                and all(_finite_fields(baseline_data).values())
            )
            warning_count = max(
                warning_count,
                _warning_count(data),
                _warning_count(baseline_data),
            )
        qpos_delta = abs(float(data.qpos[qpos_address]) - qpos_start)
        response_value = (
            max_command_induced_qvel_delta
            if wheel
            else max_command_induced_qpos_delta
        )
        threshold = 1e-4
        records.append(
            {
                "semantic_slot": joint.semantic_slot,
                "actuator_name": blueprint.actuators[selected_index].name,
                "joint_name": joint.name,
                "control_kind": "continuous_wheel_torque" if wheel else "joint_position",
                "pulse_target": target,
                "response_steps": response_steps,
                "qpos_delta_rad": qpos_delta,
                "max_abs_qvel_rad_s": max_abs_velocity,
                "max_abs_actuator_force": max_abs_force,
                "max_command_induced_qpos_delta_rad": max_command_induced_qpos_delta,
                "max_command_induced_qvel_delta_rad_s": max_command_induced_qvel_delta,
                "max_command_induced_actuator_force_delta": max_command_induced_force_delta,
                "response_metric": (
                    "max_command_induced_qvel_delta_rad_s"
                    if wheel
                    else "max_command_induced_qpos_delta_rad"
                ),
                "response_threshold": threshold,
                "finite": finite,
                "warning_count": warning_count,
                "reset_pose_passed": bool(reset_pose["reset_pose_passed"]),
                "responded": bool(
                    response_value >= threshold and finite and warning_count == 0
                ),
            }
        )
    return {
        "contract": "task070_flat_arena_paired_baseline_actuator_pulse_v2",
        "arena": "compile_mjcf_default_20x20_flat_floor_gravity_free_base",
        "response_steps_per_actuator": response_steps,
        "actuator_count": len(records),
        "responsive_actuator_count": sum(bool(item["responded"]) for item in records),
        "all_actuators_responsive": bool(records)
        and all(bool(item["responded"]) for item in records),
        "records": records,
    }


def run_v2_arena_smoke(
    output: Path,
    *,
    cases: Sequence[tuple[str, str]] | None = None,
    include_candidates: bool = False,
    stance_steps: int = 1000,
    response_steps: int = 32,
    render: bool = False,
) -> dict[str, Any]:
    """Check direct actuator response and stance separately in the flat arena."""

    selected_cases = list(cases or TASK070_V2_CURRENT_ARENA_CASES)
    if cases is None and include_candidates:
        selected_cases.extend(
            (reference_id, "biped")
            for reference_id in TASK070_ADDITIONAL_HUMANOID_REFERENCE_IDS
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    model_dir = output.parent / "models"
    snapshot_dir = output.parent / "snapshots"
    model_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    import mujoco

    for reference_id, family in selected_cases:
        record: dict[str, Any] = {
            "reference_id": reference_id,
            "family": family,
            "compiled": False,
            "accounting_exact": False,
            "reset_pose_passed": False,
            "all_actuators_responsive": False,
            "stance_hold_passed": False,
            "operational_actuator_smoke_passed": False,
            "walking_claimed": False,
            "error": None,
        }
        try:
            generator = MotorDofPreservingArchetypePreviewGenerator(
                reference_id=reference_id
            )
            blueprint = generator.generate(family, 0)  # type: ignore[arg-type]
            physical = generator.sample_physical_params(blueprint, 70_000_000)
            xml = compile_mjcf(blueprint, physical)
            xml_path = model_dir / f"{reference_id}_{family}.xml"
            xml_path.write_text(xml, encoding="utf-8")
            model = compile_with_mujoco(xml)
            accounting = blueprint.profile_metadata["motor_accounting"]
            expected_nu = int(accounting["total_actuator_count"])
            record.update(
                {
                    "compiled": True,
                    "model_nq": int(model.nq),
                    "model_nv": int(model.nv),
                    "model_nu": int(model.nu),
                    "expected_nu": expected_nu,
                    "accounting_exact": bool(
                        accounting["bijection_passed"]
                        and int(model.nu) == expected_nu == len(blueprint.actuators)
                    ),
                    "motor_accounting": accounting,
                    "candidate_prior_status": blueprint.profile_metadata.get(
                        "candidate_prior_status"
                    ),
                    "policy_adapter_compatible": blueprint.profile_metadata.get(
                        "policy_adapter_compatible",
                        True,
                    ),
                    "xml_path": str(xml_path),
                    "xml_sha256": sha256_path(xml_path),
                }
            )
            reset_model = compile_with_mujoco(xml)
            _, reset_pose, _, _ = _prepare_stance_pose(
                reset_model,
                blueprint,
                physical,
                mujoco,
            )
            record["reset_pose"] = reset_pose
            record["reset_pose_passed"] = bool(reset_pose["reset_pose_passed"])
            response = _arena_actuator_response(
                compile_with_mujoco(xml),
                blueprint,
                physical,
                response_steps=response_steps,
                mujoco=mujoco,
            )
            record["actuator_response"] = response
            record["all_actuators_responsive"] = bool(
                response["all_actuators_responsive"]
            )
            _, stance = _stance_rollout(
                compile_with_mujoco(xml),
                blueprint,
                physical,
                steps=stance_steps,
                wheel_velocity_hold=True,
                disturbance=False,
                mujoco=mujoco,
            )
            record["stance_hold"] = stance
            record["stance_hold_passed"] = bool(stance.get("passed", False))
            record["operational_actuator_smoke_passed"] = all(
                (
                    record["compiled"],
                    record["accounting_exact"],
                    record["reset_pose_passed"],
                    record["all_actuators_responsive"],
                )
            )
            if render:
                record["snapshot"] = _render_frame(
                    blueprint,
                    physical,
                    snapshot_dir / f"{reference_id}_{family}_flat_arena.png",
                    stage="raw_reset",
                    view="oblique",
                    steps=1,
                )
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    payload = {
        "task": "task070-archetype-constrained-standable-morphology",
        "status": "flat_arena_actuator_smoke_with_separate_stance_results",
        "counts_toward_task070_v2_pass": False,
        "agent_visual_check_passed": False,
        "user_visual_acceptance": False,
        "walking_claimed": False,
        "dynamic_locomotion_policy_present": False,
        "claim_boundary": (
            "Independent joint/wheel response in a flat MuJoCo arena is not a "
            "walking, gait-policy, dynamic-locomotion, or sim2real claim."
        ),
        "stance_controller": (
            "position feedforward with biped base-attitude hold, wheeled-biped "
            "active balance, or wheeled-quadruped zero-velocity hold"
        ),
        "stance_steps": stance_steps,
        "timestep_seconds": STANCE_THRESHOLDS["timestep_seconds"],
        "records": records,
        "summary": {
            "case_count": len(records),
            "compiled": sum(bool(item["compiled"]) for item in records),
            "accounting_exact": sum(
                bool(item["accounting_exact"]) for item in records
            ),
            "reset_pose_passed": sum(
                bool(item["reset_pose_passed"]) for item in records
            ),
            "all_actuators_responsive": sum(
                bool(item["all_actuators_responsive"]) for item in records
            ),
            "operational_actuator_smoke_passed": sum(
                bool(item["operational_actuator_smoke_passed"])
                for item in records
            ),
            "stance_hold_passed": sum(
                bool(item["stance_hold_passed"]) for item in records
            ),
        },
        "runtime": _runtime_metadata(),
    }
    _write_json(output, payload)
    return payload


def run_archetype_matrix(
    output: Path = DEFAULT_ARTIFACT_ROOT / "r4_archetype_morphology_matrix.json",
    *,
    seeds: range = range(32),
    steps: int = 1000,
    timestep: float = 0.002,
    render: bool = True,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if abs(timestep - STANCE_THRESHOLDS["timestep_seconds"]) > 1e-12:
        raise ValueError("Task070 R0 freezes timestep_seconds at 0.002")
    started = time.monotonic()
    generator = ArchetypeConstrainedMorphologyGenerator()
    records = [
        _matrix_record(generator, family, seed, steps=steps)
        for family in TASK070_FAMILIES
        for seed in seeds
    ]
    summary: dict[str, Any] = {}
    for family in TASK070_FAMILIES:
        family_records = [record for record in records if record["family"] == family]
        region_summary: dict[str, Any] = {}
        for region, expected in REGION_EXPECTED_PER_FAMILY.items():
            subset = [
                item for item in family_records if item.get("region_checks", {}).get("sampling_region") == region
            ]
            region_summary[region] = {
                "expected_denominator": expected,
                "actual_denominator": len(subset),
                "stance_passed": sum(bool(item.get("stance_passed")) for item in subset),
                "support_gate_passed": sum(
                    bool(item.get("stance_hold", {}).get("support_gate_passed")) for item in subset
                ),
                "contact_residual_passed": sum(
                    bool(
                        item.get("stance_hold", {}).get("final_static_load_residual_fraction", math.inf)
                        <= STANCE_THRESHOLDS["max_static_load_residual_fraction"]
                    )
                    for item in subset
                ),
                "contact_wrench_residual_passed": sum(
                    bool(
                        item.get("stance_hold", {}).get(
                            "final_contact_wrench_force_residual_fraction",
                            math.inf,
                        )
                        <= STANCE_THRESHOLDS["max_contact_wrench_force_residual_fraction"]
                        and item.get("stance_hold", {}).get(
                            "final_contact_wrench_torque_residual_fraction",
                            math.inf,
                        )
                        <= STANCE_THRESHOLDS["max_contact_wrench_torque_residual_fraction"]
                    )
                    for item in subset
                ),
                "self_contact_clear": sum(
                    bool(
                        item.get("stance_hold", {}).get("max_self_contact_count", math.inf)
                        <= STANCE_THRESHOLDS["max_self_contact_count"]
                    )
                    for item in subset
                ),
                "wheel_wheel_contact_clear": sum(
                    bool(
                        item.get("stance_hold", {}).get("max_wheel_wheel_contact_count", math.inf)
                        <= STANCE_THRESHOLDS["max_wheel_wheel_contact_count"]
                    )
                    for item in subset
                ),
            }
        summary[family] = {
            "expected_denominator": len(seeds),
            "records_passed": sum(record.get("status") == "passed" for record in family_records),
            "built": sum(bool(record["built"]) for record in family_records),
            "compiled": sum(bool(record["compiled"]) for record in family_records),
            "deterministic": sum(bool(record.get("determinism_passed")) for record in family_records),
            "finite_physics": sum(bool(record.get("finite_physics_passed")) for record in family_records),
            "slot_mapping": sum(bool(record.get("slot_mapping_passed")) for record in family_records),
            "wheel_topology": sum(bool(record.get("wheel_topology_passed")) for record in family_records),
            "reset_pose": sum(
                bool(record.get("reset_pose", {}).get("reset_pose_passed"))
                for record in family_records
            ),
            "stance_hold": sum(bool(record.get("stance_passed")) for record in family_records),
            "support_gate": sum(
                bool(record.get("stance_hold", {}).get("support_gate_passed"))
                for record in family_records
            ),
            "contact_residual": sum(
                bool(
                    record.get("stance_hold", {}).get("final_static_load_residual_fraction", math.inf)
                    <= STANCE_THRESHOLDS["max_static_load_residual_fraction"]
                )
                for record in family_records
            ),
            "contact_wrench_residual": sum(
                bool(
                    record.get("stance_hold", {}).get(
                        "final_contact_wrench_force_residual_fraction",
                        math.inf,
                    )
                    <= STANCE_THRESHOLDS["max_contact_wrench_force_residual_fraction"]
                    and record.get("stance_hold", {}).get(
                        "final_contact_wrench_torque_residual_fraction",
                        math.inf,
                    )
                    <= STANCE_THRESHOLDS["max_contact_wrench_torque_residual_fraction"]
                )
                for record in family_records
            ),
            "nonterminal_support_clear": sum(
                bool(
                    record.get("stance_hold", {}).get("max_nonterminal_floor_contact_count", math.inf)
                    <= STANCE_THRESHOLDS["max_nonterminal_floor_contact_count"]
                )
                for record in family_records
            ),
            "self_contact_clear": sum(
                bool(
                    record.get("stance_hold", {}).get("max_self_contact_count", math.inf)
                    <= STANCE_THRESHOLDS["max_self_contact_count"]
                )
                for record in family_records
            ),
            "wheel_wheel_contact_clear": sum(
                bool(
                    record.get("stance_hold", {}).get("max_wheel_wheel_contact_count", math.inf)
                    <= STANCE_THRESHOLDS["max_wheel_wheel_contact_count"]
                )
                for record in family_records
            ),
            "region_band": sum(
                bool(record.get("region_checks", {}).get("region_band_passed"))
                for record in family_records
            ),
            "clone_guard": sum(
                bool(record.get("region_checks", {}).get("clone_guard_passed"))
                for record in family_records
            ),
            "regions": region_summary,
            "errors": [record["error"] for record in family_records if record.get("error")],
        }
    gallery = _write_gallery(output.parent / "gallery_task070", generator, seeds=seeds, steps=steps) if render else None
    payload: dict[str, Any] = {
        "task": "task070-archetype-constrained-standable-morphology",
        "profile_version": ARCHETYPE_CONSTRAINED_MORPHOLOGY_PROFILE_VERSION,
        "contract_version": ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_VERSION,
        "contract_hash": ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_HASH,
        "reference_registry_sha256": TASK070_REFERENCE_REGISTRY_SHA256,
        "source_license_matrix_sha256": TASK070_SOURCE_LICENSE_MATRIX_SHA256,
        "r0_design_contract_sha256": TASK070_R0_DESIGN_CONTRACT_SHA256,
        "prior_set_id": TASK070_PRIOR_SET_ID,
        "distance_contract_hash": TASK070_DISTANCE_CONTRACT_HASH,
        "stance_contract_hash": TASK070_STANCE_CONTRACT_HASH,
        "stance_thresholds": STANCE_THRESHOLDS,
        "families": TASK070_FAMILIES,
        "expected_denominator": len(TASK070_FAMILIES) * len(seeds),
        "steps": steps,
        "timestep_seconds": timestep,
        "summary": summary,
        "records": records,
        "gallery": gallery,
        "visual_inspection": {
            "agent_viewer_required": True,
            "status": "pending_execution_agent_local_image_viewer_review" if render else "not_requested",
        },
        "runtime": _runtime_metadata(),
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": {
            "archetype_morphology.py": sha256_path(
                REPO_ROOT / "src" / "h200_locomotion_lab" / "robots" / "archetype_morphology.py"
            ),
            "procedural_morphology.py": sha256_path(
                REPO_ROOT / "src" / "h200_locomotion_lab" / "robots" / "procedural_morphology.py"
            ),
            "verification_tool.py": sha256_path(Path(__file__)),
        },
    }
    _write_json(output, payload)
    return payload


VISUAL_REVIEW_STATUS = "passed_execution_agent_full_gallery_local_image_viewer_review"
VISUAL_REQUIRED_OBSERVATION_FIELDS = (
    "family",
    "path",
    "sha256",
    "viewer_decode",
    "manual_viewer_confirmed",
    "seed",
    "region",
    "stage",
    "view",
    "limb_count",
    "attachment_review",
    "proportion_review",
    "terminal_review",
    "wheel_axle_review",
    "contact_review",
    "intersection_review",
    "cropping_review",
    "problems_observed",
)
MATRIX_REQUIRED_SUMMARY_FIELDS = (
    "built",
    "compiled",
    "deterministic",
    "finite_physics",
    "slot_mapping",
    "wheel_topology",
    "reset_pose",
    "stance_hold",
    "support_gate",
    "contact_residual",
    "contact_wrench_residual",
    "nonterminal_support_clear",
    "self_contact_clear",
    "wheel_wheel_contact_clear",
    "region_band",
    "clone_guard",
    "records_passed",
)
MATRIX_REQUIRED_REGION_FIELDS = (
    "stance_passed",
    "support_gate_passed",
    "contact_residual_passed",
    "contact_wrench_residual_passed",
    "self_contact_clear",
    "wheel_wheel_contact_clear",
)


def _relative_to_artifact_root(path: str | Path, artifact_root: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        repo_relative = (REPO_ROOT / candidate).resolve()
        if repo_relative.exists():
            candidate = repo_relative
        else:
            candidate = artifact_root / candidate
    return candidate.resolve().relative_to(artifact_root.resolve()).as_posix()


def _parse_visual_path(relative_path: str, family: str) -> dict[str, Any]:
    name = Path(relative_path).name
    region = "all_regions"
    stage = "verified"
    view = "unknown"
    seed: int | str | None = None
    if name == "montage_verified.png":
        seed = "0:32"
        view = "montage_oblique"
    elif name.startswith("seed_") and name.endswith("_verified_oblique.png"):
        seed = int(name.split("_")[1])
        region = ArchetypeConstrainedMorphologyGenerator().expected_sampling_region(seed)
        view = "oblique"
    elif name.startswith("closeup_seed_"):
        stem = name.removesuffix(".png")
        parts = stem.split("_")
        seed = int(parts[2])
        region = "_".join(parts[3:-1])
        view = parts[-1]
    elif name.startswith("raw_vs_verified_seed_"):
        stem = name.removesuffix(".png")
        parts = stem.split("_")
        seed = int(parts[4])
        region = ArchetypeConstrainedMorphologyGenerator().expected_sampling_region(seed)
        if name.endswith("_raw.png"):
            stage = "raw_reset"
            view = "side"
        elif name.endswith("_verified.png"):
            stage = "verified"
            view = "side"
        else:
            stage = "raw_vs_verified_comparison"
            view = "side_pair"
    return {
        "family": family,
        "seed": seed,
        "region": region,
        "stage": stage,
        "view": view,
    }


def _expected_visual_images(matrix: Mapping[str, Any], artifact_root: Path) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    gallery = matrix.get("gallery") or {}
    for family in TASK070_FAMILIES:
        family_gallery = gallery.get(family) or {}
        candidates: list[tuple[str, Mapping[str, Any]]] = []
        for kind in ("montage",):
            if isinstance(family_gallery.get(kind), dict):
                candidates.append((kind, family_gallery[kind]))
        for kind in ("sample_frames", "closeups", "raw_vs_verified"):
            for item in family_gallery.get(kind) or ():
                if isinstance(item, dict):
                    candidates.append((kind, item))
                    if kind == "raw_vs_verified":
                        for nested in ("raw", "verified"):
                            if isinstance(item.get(nested), dict):
                                candidates.append((f"{kind}_{nested}", item[nested]))
        for kind, item in candidates:
            relative_path = _relative_to_artifact_root(str(item["path"]), artifact_root)
            images.append(
                {
                    "kind": kind,
                    "relative_path": relative_path,
                    **_parse_visual_path(relative_path, family),
                }
            )
    return images


def _visual_denominators(
    images: Sequence[Mapping[str, Any]],
    observations: Mapping[str, Any],
) -> dict[str, Any]:
    image_observations = observations.get("image_observations", {})
    denominators: dict[str, Any] = {}
    for family in TASK070_FAMILIES:
        family_images = [item for item in images if item["family"] == family]
        sample_images = [item for item in family_images if item["kind"] == "sample_frames"]
        regions: dict[str, Any] = {}
        for region, expected in REGION_EXPECTED_PER_FAMILY.items():
            region_samples = [item for item in sample_images if item["region"] == region]
            regions[region] = {
                "expected_denominator": expected,
                "actual_denominator": len(region_samples),
                "visual_passed": sum(
                    bool(
                        image_observations.get(item["relative_path"], {}).get(
                            "manual_viewer_confirmed"
                        )
                    )
                    and not image_observations.get(item["relative_path"], {}).get(
                        "problems_observed"
                    )
                    for item in region_samples
                ),
            }
        denominators[family] = {
            "expected_sample_frames": sum(REGION_EXPECTED_PER_FAMILY.values()),
            "actual_sample_frames": len(sample_images),
            "visual_sample_frames_passed": sum(
                bool(
                    image_observations.get(item["relative_path"], {}).get(
                        "manual_viewer_confirmed"
                    )
                )
                and not image_observations.get(item["relative_path"], {}).get(
                    "problems_observed"
                )
                for item in sample_images
            ),
            "expected_gallery_images": len(family_images),
            "reviewed_gallery_images": sum(
                item["relative_path"] in image_observations for item in family_images
            ),
            "visual_gallery_passed": sum(
                bool(
                    image_observations.get(item["relative_path"], {}).get(
                        "manual_viewer_confirmed"
                    )
                )
                and not image_observations.get(item["relative_path"], {}).get(
                    "problems_observed"
                )
                for item in family_images
            ),
            "regions": regions,
        }
    return denominators


def finalize_visual_inspection(
    path: Path = DEFAULT_ARTIFACT_ROOT / "r4_archetype_morphology_matrix.json",
    observations_path: Path = DEFAULT_ARTIFACT_ROOT / "r4_visual_observations.json",
) -> dict[str, Any]:
    """Validate and bind the execution-agent local viewer manifest to R4."""

    path = path.resolve()
    observations_path = observations_path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    if observations.get("viewer") != "local_image_viewer":
        raise ValueError("visual observations must identify the local image viewer")
    if observations.get("status") != VISUAL_REVIEW_STATUS:
        raise ValueError(f"visual observations must use status {VISUAL_REVIEW_STATUS!r}")
    artifact_root = path.parent
    expected_images = _expected_visual_images(payload, artifact_root)
    image_observations = observations.get("image_observations", {})
    if not isinstance(image_observations, dict):
        raise TypeError("visual observations must contain image_observations")
    for image in expected_images:
        relative = str(image["relative_path"])
        note = image_observations.get(relative)
        if not isinstance(note, dict):
            raise TypeError(f"visual observation missing for {relative}")
        missing = [field for field in VISUAL_REQUIRED_OBSERVATION_FIELDS if field not in note]
        if missing:
            raise ValueError(f"visual observation for {relative} misses {missing}")
        decoded = _decode_image(artifact_root / relative)
        if not decoded["viewer_decode"]:
            raise ValueError(f"final image is not decodable: {relative}")
        if str(note["sha256"]) != str(decoded["sha256"]):
            raise ValueError(f"visual observation SHA mismatch for {relative}")
        for key in ("family", "seed", "region", "stage", "view"):
            if note[key] != image[key]:
                raise ValueError(f"visual observation {relative} has wrong {key}")
        if not bool(note["viewer_decode"]) or not bool(note["manual_viewer_confirmed"]):
            raise ValueError(f"visual observation not viewer-confirmed for {relative}")
        if note["problems_observed"]:
            raise ValueError(f"visual observation records problems for {relative}")
    denominators = _visual_denominators(expected_images, observations)
    for family, counts in denominators.items():
        if int(counts["actual_sample_frames"]) != int(counts["expected_sample_frames"]):
            raise ValueError(f"visual sample denominator mismatch for {family}")
        if int(counts["visual_sample_frames_passed"]) != int(counts["expected_sample_frames"]):
            raise ValueError(f"visual sample pass count mismatch for {family}")
        if int(counts["reviewed_gallery_images"]) != int(counts["expected_gallery_images"]):
            raise ValueError(f"visual gallery review count mismatch for {family}")
        if int(counts["visual_gallery_passed"]) != int(counts["expected_gallery_images"]):
            raise ValueError(f"visual gallery pass count mismatch for {family}")
        for region, region_counts in counts["regions"].items():
            if int(region_counts["actual_denominator"]) != int(
                region_counts["expected_denominator"]
            ):
                raise ValueError(f"visual region denominator mismatch for {family}/{region}")
            if int(region_counts["visual_passed"]) != int(region_counts["expected_denominator"]):
                raise ValueError(f"visual region pass count mismatch for {family}/{region}")
    observations["visual_denominators"] = denominators
    observations["reviewed_image_count"] = len(expected_images)
    _write_json(observations_path, observations)
    payload["visual_inspection"] = {
        "agent_viewer_required": True,
        "status": VISUAL_REVIEW_STATUS,
        "reviewed_image_count": len(expected_images),
        "visual_denominators": denominators,
        "visual_observation_manifest": {
            "path": observations_path.relative_to(artifact_root).as_posix(),
            "sha256": sha256_path(observations_path),
        },
    }
    _write_json(path, payload)
    return payload


def verify_r0_compatibility_baseline(path: Path) -> dict[str, Any]:
    expected = json.loads(path.read_text(encoding="utf-8"))
    legacy_generator = MorphologyGenerator()
    task069_generator = LocoFormerMorphologyGenerator()
    checks: list[dict[str, Any]] = []
    for section, generator, families, seed_offset in (
        ("legacy_v2", legacy_generator, ("biped", "quadruped"), 10_000_000),
        ("task069_profile", task069_generator, TASK070_FAMILIES, 20_000_000),
    ):
        for old in expected[section]["records"]:
            family = old["family"]
            seed = int(old["seed"])
            range_fraction = float(old["range_fraction"])
            if family not in families:
                continue
            blueprint = generator.generate(family, seed)  # type: ignore[arg-type]
            physical = generator.sample_physical_params(
                blueprint,
                seed + seed_offset,
                range_fraction=range_fraction,
            )
            xml = compile_mjcf(blueprint, physical)
            key = morphology_instance_key(blueprint, physical)
            current = {
                "blueprint_hash": morphology_blueprint_hash(blueprint),
                "physical_hash": physical_params_hash(physical),
                "xml_sha256": sha256_bytes(xml.encode("utf-8")),
                "instance_key": key.manifest() | {"cache_key": key.cache_key},
                "contract_version": blueprint.contract_version,
                "contract_hash": blueprint.contract_hash,
            }
            field_checks = {name: current[name] == old[name] for name in current}
            checks.append(
                {
                    "section": section,
                    "family": family,
                    "seed": seed,
                    "range_fraction": range_fraction,
                    "passed": all(field_checks.values()),
                    "checks": field_checks,
                }
            )
    return {
        "expected_denominator": expected["legacy_v2"]["expected_denominator"]
        + expected["task069_profile"]["expected_denominator"],
        "checked_count": len(checks),
        "passed_count": sum(bool(item["passed"]) for item in checks),
        "failed": [item for item in checks if not item["passed"]],
        "passed": bool(checks) and all(item["passed"] for item in checks),
    }


def _visual_review_passed(
    matrix: Mapping[str, Any],
    visual_review: Mapping[str, Any] | None,
    *,
    visual_path: Path,
    visual_status: str,
) -> bool:
    if visual_review is None:
        return False
    matrix_visual = matrix.get("visual_inspection", {})
    manifest = matrix_visual.get("visual_observation_manifest", {}) if isinstance(matrix_visual, dict) else {}
    if visual_status != VISUAL_REVIEW_STATUS:
        return False
    if visual_review.get("status") != VISUAL_REVIEW_STATUS:
        return False
    if matrix_visual.get("status") != VISUAL_REVIEW_STATUS:
        return False
    if manifest.get("sha256") != sha256_path(visual_path):
        return False
    artifact_root = visual_path.parent
    expected_images = _expected_visual_images(matrix, artifact_root)
    image_observations = visual_review.get("image_observations", {})
    if not isinstance(image_observations, dict):
        return False
    if len(image_observations) < len(expected_images):
        return False
    for image in expected_images:
        relative = str(image["relative_path"])
        note = image_observations.get(relative)
        if not isinstance(note, dict):
            return False
        if any(field not in note for field in VISUAL_REQUIRED_OBSERVATION_FIELDS):
            return False
        decoded = _decode_image(artifact_root / relative)
        if (
            not decoded["viewer_decode"]
            or str(note["sha256"]) != str(decoded["sha256"])
            or not bool(note["manual_viewer_confirmed"])
            or bool(note["problems_observed"])
        ):
            return False
        if any(note[key] != image[key] for key in ("family", "seed", "region", "stage", "view")):
            return False
    denominators = _visual_denominators(expected_images, visual_review)
    return all(
        int(counts["actual_sample_frames"]) == int(counts["expected_sample_frames"])
        and int(counts["visual_sample_frames_passed"]) == int(counts["expected_sample_frames"])
        and int(counts["reviewed_gallery_images"]) == int(counts["expected_gallery_images"])
        and int(counts["visual_gallery_passed"]) == int(counts["expected_gallery_images"])
        and all(
            int(region_counts["actual_denominator"])
            == int(region_counts["expected_denominator"])
            and int(region_counts["visual_passed"]) == int(region_counts["expected_denominator"])
            for region_counts in counts["regions"].values()
        )
        for counts in denominators.values()
    )


def _matrix_record_keys(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    for record in records:
        try:
            keys.append((str(record["family"]), int(record["seed"])))
        except (KeyError, TypeError, ValueError):
            keys.append(("<invalid>", -1))
    return keys


def _final_matrix_gate(matrix: Mapping[str, Any]) -> dict[str, Any]:
    expected_family_set = set(TASK070_FAMILIES)
    expected_region_set = set(REGION_EXPECTED_PER_FAMILY)
    expected_record_keys = {
        (family, seed)
        for family in TASK070_FAMILIES
        for seed in TASK070_FINAL_MATRIX_SEEDS
    }
    summary = matrix.get("summary")
    records = matrix.get("records")
    summary_is_dict = isinstance(summary, dict)
    records_is_list = isinstance(records, list)
    summary_dict = summary if summary_is_dict else {}
    records_list = records if records_is_list else []
    record_keys = _matrix_record_keys(records_list)
    checks: dict[str, Any] = {
        "task": matrix.get("task") == "task070-archetype-constrained-standable-morphology",
        "families_exact": list(matrix.get("families", ())) == list(TASK070_FAMILIES),
        "expected_denominator_exact": int(matrix.get("expected_denominator", -1))
        == TASK070_FINAL_MATRIX_EXPECTED_DENOMINATOR,
        "steps_exact": int(matrix.get("steps", -1)) == int(STANCE_THRESHOLDS["stance_hold_steps"]),
        "timestep_exact": abs(
            float(matrix.get("timestep_seconds", float("nan")))
            - STANCE_THRESHOLDS["timestep_seconds"]
        )
        <= 1e-12,
        "summary_is_dict": summary_is_dict,
        "summary_families_exact": summary_is_dict
        and set(summary_dict) == expected_family_set,
        "records_is_list": records_is_list,
        "record_count_exact": records_is_list
        and len(records_list) == TASK070_FINAL_MATRIX_EXPECTED_DENOMINATOR,
        "record_key_set_exact": records_is_list
        and set(record_keys) == expected_record_keys
        and len(record_keys) == len(expected_record_keys),
        "record_keys_unique": records_is_list and len(record_keys) == len(set(record_keys)),
    }
    family_checks: dict[str, Any] = {}
    for family in TASK070_FAMILIES:
        item = summary_dict.get(family)
        item_is_dict = isinstance(item, dict)
        family_records = [
            record
            for record in records_list
            if isinstance(record, dict) and record.get("family") == family
        ]
        region_checks: dict[str, Any] = {}
        regions = item.get("regions", {}) if item_is_dict else {}
        for region, expected in REGION_EXPECTED_PER_FAMILY.items():
            region_item = regions.get(region) if isinstance(regions, dict) else None
            region_is_dict = isinstance(region_item, dict)
            region_records = [
                record
                for record in family_records
                if record.get("region_checks", {}).get("sampling_region") == region
            ]
            region_checks[region] = {
                "present": region_is_dict,
                "expected_denominator": region_is_dict
                and int(region_item.get("expected_denominator", -1)) == expected,
                "actual_denominator": region_is_dict
                and int(region_item.get("actual_denominator", -1)) == expected,
                "record_denominator": len(region_records) == expected,
                "required_fields": region_is_dict
                and all(int(region_item.get(field, -1)) == expected for field in MATRIX_REQUIRED_REGION_FIELDS),
            }
            region_checks[region]["passed"] = all(region_checks[region].values())
        family_checks[family] = {
            "present": item_is_dict,
            "expected_denominator": item_is_dict
            and int(item.get("expected_denominator", -1)) == len(TASK070_FINAL_MATRIX_SEEDS),
            "required_summary_fields": item_is_dict
            and all(
                int(item.get(field, -1)) == len(TASK070_FINAL_MATRIX_SEEDS)
                for field in MATRIX_REQUIRED_SUMMARY_FIELDS
            ),
            "record_denominator": len(family_records) == len(TASK070_FINAL_MATRIX_SEEDS),
            "record_seed_set": {
                int(record.get("seed", -1))
                for record in family_records
                if isinstance(record, dict)
            }
            == set(TASK070_FINAL_MATRIX_SEEDS),
            "record_gates": all(
                record.get("status") == "passed"
                and bool(record.get("region_checks", {}).get("region_band_passed"))
                and bool(record.get("region_checks", {}).get("clone_guard_passed"))
                for record in family_records
            ),
            "regions_exact": isinstance(regions, dict) and set(regions) == expected_region_set,
            "regions": region_checks,
            "errors_empty": item_is_dict and not item.get("errors"),
        }
        family_checks[family]["passed"] = (
            family_checks[family]["present"]
            and family_checks[family]["expected_denominator"]
            and family_checks[family]["required_summary_fields"]
            and family_checks[family]["record_denominator"]
            and family_checks[family]["record_seed_set"]
            and family_checks[family]["record_gates"]
            and family_checks[family]["regions_exact"]
            and family_checks[family]["errors_empty"]
            and all(region["passed"] for region in region_checks.values())
        )
    checks["families"] = family_checks
    checks["passed"] = all(
        value for key, value in checks.items() if key != "families"
    ) and all(item["passed"] for item in family_checks.values())
    return checks


def write_final_verification(
    output: Path = DEFAULT_ARTIFACT_ROOT / "r5_final_verification.json",
    *,
    matrix_path: Path = DEFAULT_ARTIFACT_ROOT / "r4_archetype_morphology_matrix.json",
    compatibility_path: Path = DEFAULT_ARTIFACT_ROOT / "r0_compatibility_baseline.json",
    pytest_status: str = "not_run",
    ruff_status: str = "not_run",
    inspect_status: str = "not_run",
    full_pytest_status: str = "not_run",
    visual_status: str = "not_run",
    pytest_exit_code: int | None = None,
    ruff_exit_code: int | None = None,
    inspect_exit_code: int | None = None,
    full_pytest_exit_code: int | None = None,
    notes: str = "",
) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    visual_path = DEFAULT_ARTIFACT_ROOT / "r4_visual_observations.json"
    visual_review = (
        json.loads(visual_path.read_text(encoding="utf-8")) if visual_path.exists() else None
    )
    compatibility = verify_r0_compatibility_baseline(compatibility_path)
    new_generator = ArchetypeConstrainedMorphologyGenerator()
    legacy_generator = MorphologyGenerator()
    task069_generator = LocoFormerMorphologyGenerator()
    new_blueprint = new_generator.generate("biped", 0)
    new_physical = new_generator.sample_physical_params(new_blueprint, 30_000_000, range_fraction=0.5)
    legacy_blueprint = legacy_generator.generate("biped", 0)
    legacy_physical = legacy_generator.sample_physical_params(
        legacy_blueprint,
        10_000_000,
        range_fraction=0.5,
    )
    task069_blueprint = task069_generator.generate("biped", 0)
    task069_physical = task069_generator.sample_physical_params(
        task069_blueprint,
        20_000_000,
        range_fraction=0.5,
    )
    identity_probe = {
        "legacy_contract": [
            PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
            PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
        ],
        "task069_contract": [
            LOCOFORMER_MORPHOLOGY_CONTRACT_VERSION,
            LOCOFORMER_MORPHOLOGY_CONTRACT_HASH,
        ],
        "task070_contract": [
            ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_VERSION,
            ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_HASH,
        ],
        "cache_keys_distinct": len(
            {
                morphology_instance_key(legacy_blueprint, legacy_physical).cache_key,
                morphology_instance_key(task069_blueprint, task069_physical).cache_key,
                morphology_instance_key(new_blueprint, new_physical).cache_key,
            }
        )
        == 3,
        "registry_hash_in_task070_identity": (
            TASK070_REFERENCE_REGISTRY_SHA256
            in json.dumps(new_blueprint.manifest(), sort_keys=True)
        ),
        "license_hash_in_task070_identity": (
            TASK070_SOURCE_LICENSE_MATRIX_SHA256
            in json.dumps(new_blueprint.manifest(), sort_keys=True)
        ),
        "distance_hash_in_task070_identity": (
            TASK070_DISTANCE_CONTRACT_HASH
            in json.dumps(new_blueprint.manifest(), sort_keys=True)
        ),
    }
    def log_verdict(log_path: Path, kind: str) -> dict[str, Any]:
        if not log_path.exists():
            return {"passed": False, "reason": "missing_log"}
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if kind == "pytest":
            passed = " passed" in text and " failed" not in text and "ERROR" not in text
            return {"passed": passed, "reason": "pytest_summary"}
        if kind == "ruff":
            return {"passed": "All checks passed!" in text, "reason": "ruff_summary"}
        if kind == "inspect":
            passed = "sonic_adapter:" in text and "locoformer_min:" in text
            return {"passed": passed, "reason": "inspect_agent_component_listing"}
        return {"passed": False, "reason": f"unknown_kind:{kind}"}

    def evidence(
        command: str,
        status: str,
        log_name: str,
        *,
        kind: str,
        exit_code: int | None,
    ) -> dict[str, Any]:
        log_path = DEFAULT_ARTIFACT_ROOT / "logs" / log_name
        verdict = log_verdict(log_path, kind)
        return {
            "command": command,
            "status": status,
            "exit_code": exit_code,
            "evidence_log": str(log_path),
            "evidence_sha256": sha256_path(log_path) if log_path.exists() else None,
            "log_verdict": verdict,
            "passed": status.startswith("passed")
            and exit_code == 0
            and bool(verdict["passed"]),
        }

    command_results = [
        evidence(
            (
                ".venv/bin/python -m pytest -q tests/test_whole_body_contract.py "
                "tests/test_whole_body_extended.py tests/test_whole_body_usability_gate.py "
                "tests/test_task069_*.py tests/test_task070_*.py"
            ),
            pytest_status,
            "task070_focused_pytest.log",
            kind="pytest",
            exit_code=pytest_exit_code,
        ),
        evidence(
            (
                ".venv/bin/ruff check src/h200_locomotion_lab/robots "
                "src/h200_locomotion_lab/tools/task070_*.py tests/test_task070_*.py"
            ),
            ruff_status,
            "task070_ruff.log",
            kind="ruff",
            exit_code=ruff_exit_code,
        ),
        evidence(
            ".venv/bin/python -m h200_locomotion_lab.tools.inspect_agent",
            inspect_status,
            "task070_inspect.log",
            kind="inspect",
            exit_code=inspect_exit_code,
        ),
        evidence(
            ".venv/bin/python -m pytest -q",
            full_pytest_status,
            "task070_full_pytest.log",
            kind="pytest",
            exit_code=full_pytest_exit_code,
        ),
    ]
    matrix_gate = _final_matrix_gate(matrix)
    all_matrix_passed = bool(matrix_gate["passed"])
    visual_passed = _visual_review_passed(
        matrix,
        visual_review,
        visual_path=visual_path,
        visual_status=visual_status,
    )
    payload = {
        "task": "task070-archetype-constrained-standable-morphology",
        "status": "execution_verified_pending_independent_readonly_review",
        "claim_boundary": (
            "multi-vendor-audited engineering morphology prior with contact-aware "
            "stance-hold; not passive standing, walking, policy, sim2real, or named-robot parity"
        ),
        "matrix_passed": all_matrix_passed,
        "matrix_gate": matrix_gate,
        "compatibility_baseline": compatibility,
        "identity_probe": identity_probe,
        "command_results": command_results,
        "commands_passed": all(bool(item["passed"]) for item in command_results),
        "visual_status": visual_status,
        "visual_review": {
            "path": str(visual_path),
            "sha256": sha256_path(visual_path) if visual_path.exists() else None,
            "passed": visual_passed,
        },
        "notes": notes,
        "artifact_sha256": {
            "r0_reference_registry.json": sha256_path(
                DEFAULT_ARTIFACT_ROOT / "r0_reference_registry.json"
            ),
            "r0_source_license_matrix.json": sha256_path(
                DEFAULT_ARTIFACT_ROOT / "r0_source_license_matrix.json"
            ),
            "r0_compatibility_baseline.json": sha256_path(compatibility_path),
            "r0_design_contract.json": sha256_path(
                DEFAULT_ARTIFACT_ROOT / "r0_design_contract.json"
            ),
            "r4_archetype_morphology_matrix.json": sha256_path(matrix_path),
            "r4_visual_observations.json": sha256_path(visual_path) if visual_path.exists() else None,
        },
        "source_sha256": {
            "archetype_morphology.py": sha256_path(
                REPO_ROOT / "src" / "h200_locomotion_lab" / "robots" / "archetype_morphology.py"
            ),
            "procedural_morphology.py": sha256_path(
                REPO_ROOT / "src" / "h200_locomotion_lab" / "robots" / "procedural_morphology.py"
            ),
            "verification_tool.py": sha256_path(Path(__file__)),
        },
        "runtime": _runtime_metadata(),
    }
    _write_json(output, payload)
    return payload


def _parse_range(value: str) -> range:
    if ":" not in value:
        count = int(value)
        return range(count)
    start, stop = (int(part) for part in value.split(":", 1))
    return range(start, stop)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    matrix = sub.add_parser("matrix", help="run Task070 matrix and gallery")
    matrix.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_ROOT / "r4_archetype_morphology_matrix.json")
    matrix.add_argument("--seed-range", default="0:32")
    matrix.add_argument("--steps", type=int, default=1000)
    matrix.add_argument("--timestep", type=float, default=0.002)
    matrix.add_argument("--render", action="store_true")
    matrix.add_argument("--no-render", action="store_true")

    preview = sub.add_parser("preview-v2", help="write one Task070 v2 inspection witness")
    preview.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "preview_task070_v2" / "unitree_g1_seed000",
    )
    preview.add_argument("--seed", type=int, default=0)
    preview.add_argument(
        "--family",
        choices=("biped", "quadruped", "wheeled_biped", "wheeled_quadruped"),
        default="biped",
    )
    preview.add_argument(
        "--reference-id",
        choices=(
            "unitree_g1",
            "engineai_pm01",
            "spot_base",
            "unitree_go2",
            "deeprobotics_lite3",
            *TASK070_ADDITIONAL_HUMANOID_REFERENCE_IDS,
        ),
        default="unitree_g1",
    )
    preview.add_argument("--render", action="store_true", default=True)
    preview.add_argument("--no-render", action="store_true")

    arena = sub.add_parser(
        "arena-v2-smoke",
        help="run direct actuator response and separate stance checks in the flat arena",
    )
    arena.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_ARTIFACT_ROOT
            / "arena_task070_v2_attempt006"
            / "flat_arena_smoke.json"
        ),
    )
    arena.add_argument("--include-candidates", action="store_true")
    arena.add_argument("--stance-steps", type=int, default=1000)
    arena.add_argument("--response-steps", type=int, default=32)
    arena.add_argument("--render", action="store_true")

    compat = sub.add_parser("verify-r0-compat", help="verify frozen legacy/Task069 baseline")
    compat.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "r0_compatibility_baseline.json",
    )

    visual = sub.add_parser("finalize-visual", help="validate and bind R4 visual observations")
    visual.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "r4_archetype_morphology_matrix.json",
    )
    visual.add_argument(
        "--observations",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "r4_visual_observations.json",
    )

    final = sub.add_parser("finalize-r5", help="write final verification artifact")
    final.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_ROOT / "r5_final_verification.json")
    final.add_argument("--matrix", type=Path, default=DEFAULT_ARTIFACT_ROOT / "r4_archetype_morphology_matrix.json")
    final.add_argument("--compatibility", type=Path, default=DEFAULT_ARTIFACT_ROOT / "r0_compatibility_baseline.json")
    final.add_argument("--pytest-status", default="not_run")
    final.add_argument("--ruff-status", default="not_run")
    final.add_argument("--inspect-status", default="not_run")
    final.add_argument("--full-pytest-status", default="not_run")
    final.add_argument("--visual-status", default="not_run")
    final.add_argument("--pytest-exit-code", type=int)
    final.add_argument("--ruff-exit-code", type=int)
    final.add_argument("--inspect-exit-code", type=int)
    final.add_argument("--full-pytest-exit-code", type=int)
    final.add_argument("--notes", default="")

    args = parser.parse_args(argv)
    if args.command == "matrix":
        render = bool(args.render and not args.no_render)
        payload = run_archetype_matrix(
            args.output,
            seeds=_parse_range(args.seed_range),
            steps=args.steps,
            timestep=args.timestep,
            render=render,
        )
        print(json.dumps({"summary": payload["summary"], "output": str(args.output)}, indent=2))
        return 0
    if args.command == "preview-v2":
        payload = write_v2_preview(
            args.output_dir,
            seed=args.seed,
            render=bool(args.render and not args.no_render),
            family=args.family,
            reference_id=args.reference_id,
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "motor_accounting": payload["motor_accounting"],
                    "manifest": payload["paths"]["manifest"],
                    "image": payload["paths"]["image"],
                    "xml": payload["paths"]["xml"],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "arena-v2-smoke":
        payload = run_v2_arena_smoke(
            args.output,
            include_candidates=bool(args.include_candidates),
            stance_steps=args.stance_steps,
            response_steps=args.response_steps,
            render=bool(args.render),
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "summary": payload["summary"],
                    "walking_claimed": payload["walking_claimed"],
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "verify-r0-compat":
        print(json.dumps(verify_r0_compatibility_baseline(args.input), indent=2))
        return 0
    if args.command == "finalize-visual":
        payload = finalize_visual_inspection(args.matrix, args.observations)
        print(
            json.dumps(
                {
                    "status": payload["visual_inspection"]["status"],
                    "reviewed_image_count": payload["visual_inspection"]["reviewed_image_count"],
                    "output": str(args.matrix),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "finalize-r5":
        payload = write_final_verification(
            args.output,
            matrix_path=args.matrix,
            compatibility_path=args.compatibility,
            pytest_status=args.pytest_status,
            ruff_status=args.ruff_status,
            inspect_status=args.inspect_status,
            full_pytest_status=args.full_pytest_status,
            visual_status=args.visual_status,
            pytest_exit_code=args.pytest_exit_code,
            ruff_exit_code=args.ruff_exit_code,
            inspect_exit_code=args.inspect_exit_code,
            full_pytest_exit_code=args.full_pytest_exit_code,
            notes=args.notes,
        )
        print(json.dumps({"status": payload["status"], "output": str(args.output)}, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
