"""Task067 R4a dynamic-balance causal diagnosis.

This is a diagnostic-only tool.  It does not change the 45D action, 193D
observation, reward, actuator ``kp/kv``, motor process, or public environment
contract.  The tool asks whether the R1-R3 reset stance has a contact-consistent
free-base equilibrium, and whether that equilibrium is locally stable.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from h200_locomotion_lab.envs.whole_body_mujoco import (
    WholeBodyMuJoCoShard,
    WholeBodyMuJoCoShardConfig,
    _projected_gravity,
)
from h200_locomotion_lab.robots.procedural_morphology import (
    PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
    PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
    MorphologyGenerator,
    morphology_instance_key,
)
from h200_locomotion_lab.robots.whole_body_stance import (
    STANCE_SOLUTION_CONTRACT_HASH,
    STANCE_SOLUTION_CONTRACT_VERSION,
)
from h200_locomotion_lab.tools.whole_body_contact_taxonomy import contact_taxonomy


@dataclass(frozen=True, slots=True)
class RolloutMode:
    name: str
    zero_gravity: bool = False
    root_locked: bool = False
    constant_qfrc_bias_ctrl: bool = False
    dynamic_qfrc_bias_ctrl: bool = False
    contact_equilibrium_ctrl: bool = False


_BIPED_MODES = (
    RolloutMode("baseline"),
    RolloutMode("zero_gravity", zero_gravity=True),
    RolloutMode("root_locked", root_locked=True),
    RolloutMode("constant_qfrc_bias_kp_preload", constant_qfrc_bias_ctrl=True),
    RolloutMode("dynamic_qfrc_bias_kp_preload", dynamic_qfrc_bias_ctrl=True),
    RolloutMode("contact_equilibrium_hold", contact_equilibrium_ctrl=True),
)
_QUADRUPED_MODES = (
    RolloutMode("baseline"),
    RolloutMode("contact_equilibrium_hold", contact_equilibrium_ctrl=True),
)
_EQUILIBRIUM_THRESHOLDS = {
    "root_residual_force_norm": 1e-5,
    "root_residual_torque_norm": 1e-5,
    "contact_force_residual_norm": 1e-5,
    "support_bottom_abs": 0.008,
    "qacc_root_norm": 1.0,
    "qacc_joint_max": 10.0,
    "foot_load_fraction": 0.05,
    "max_joint_adjustment": 0.08,
}


def _stance_qpos(shard: WholeBodyMuJoCoShard) -> Any:
    np = shard.np
    qpos = np.zeros(shard.model.nq, dtype=np.float64)
    qpos[0] = shard.stance_solution.root_xy[0]
    qpos[1] = shard.stance_solution.root_xy[1]
    qpos[2] = shard.stance_solution.base_height
    qpos[3:7] = shard.stance_solution.root_quat
    for joint, address in zip(shard.blueprint.joints, shard._joint_qpos):
        qpos[address] = shard.stance_solution.joint_qpos[joint.semantic_slot]
    return qpos


def _reset_to_qpos(shard: WholeBodyMuJoCoShard, data: Any, qpos: Any) -> None:
    shard.mujoco.mj_resetData(shard.model, data)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    shard.mujoco.mj_forward(shard.model, data)


def _fall_height_threshold(shard: WholeBodyMuJoCoShard) -> float:
    scale = shard.physical.global_scale if shard.physical else 1.0
    return shard.blueprint.nominal_height * scale * shard.config.fall_height_fraction


def _quat_from_roll_pitch(roll: float, pitch: float) -> tuple[float, float, float, float]:
    return _quat_from_roll_pitch_yaw(roll, pitch, 0.0)


def _quat_from_roll_pitch_yaw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _roll_pitch(quat: Any) -> tuple[float, float]:
    roll, pitch, _ = _roll_pitch_yaw(quat)
    return roll, pitch


def _roll_pitch_yaw(quat: Any) -> tuple[float, float, float]:
    w, x, y, z = (float(value) for value in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return 0.0, 0.0, 0.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _set_baseline_ctrl(shard: WholeBodyMuJoCoShard, data: Any) -> None:
    for actuator, actuator_id in zip(shard.blueprint.actuators, shard._actuator_ids):
        lower, upper = (float(value) for value in shard.model.actuator_ctrlrange[actuator_id])
        target = shard.stance_solution.actuator_ctrl[actuator.semantic_slot]
        data.ctrl[actuator_id] = min(upper, max(lower, target))


def _scipy_optimize() -> Any:
    try:
        from scipy import optimize as scipy_optimize  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional diagnostic dependency
        raise RuntimeError("SciPy is required for Task067 R4a.2 equilibrium diagnosis") from exc
    return scipy_optimize


def _ctrl_from_generalized_force(shard: WholeBodyMuJoCoShard, data: Any, qfrc: Any) -> Any:
    np = shard.np
    ctrl = np.array(data.ctrl, copy=True)
    for actuator_id, qpos_address, dof_address in zip(
        shard._actuator_ids,
        shard._joint_qpos,
        shard._joint_dof,
    ):
        kp = max(1e-9, float(shard.model.actuator_gainprm[actuator_id, 0]))
        lower, upper = (float(value) for value in shard.model.actuator_ctrlrange[actuator_id])
        target = float(data.qpos[qpos_address]) + float(qfrc[dof_address]) / kp
        ctrl[actuator_id] = min(upper, max(lower, target))
    return ctrl


def _joint_force_feasibility(shard: WholeBodyMuJoCoShard, data: Any, qfrc: Any) -> dict[str, Any]:
    out_of_ctrl_range = 0
    over_force_limit = 0
    max_abs_tau = 0.0
    max_abs_ctrl_offset = 0.0
    max_ctrl_range_violation = 0.0
    max_force_limit_violation = 0.0
    ctrl_targets = []
    for actuator_id, qpos_address, dof_address in zip(
        shard._actuator_ids,
        shard._joint_qpos,
        shard._joint_dof,
    ):
        tau = float(qfrc[dof_address])
        kp = max(1e-9, float(shard.model.actuator_gainprm[actuator_id, 0]))
        ctrl_target = float(data.qpos[qpos_address]) + tau / kp
        ctrl_low, ctrl_high = (float(value) for value in shard.model.actuator_ctrlrange[actuator_id])
        force_limit = max(abs(float(value)) for value in shard.model.actuator_forcerange[actuator_id])
        ctrl_violation = max(0.0, ctrl_low - ctrl_target, ctrl_target - ctrl_high)
        force_violation = max(0.0, abs(tau) - force_limit)
        out_of_ctrl_range += int(ctrl_violation > 0.0)
        over_force_limit += int(force_violation > 0.0)
        max_abs_tau = max(max_abs_tau, abs(tau))
        max_abs_ctrl_offset = max(max_abs_ctrl_offset, abs(tau / kp))
        max_ctrl_range_violation = max(max_ctrl_range_violation, ctrl_violation)
        max_force_limit_violation = max(max_force_limit_violation, force_violation)
        ctrl_targets.append(ctrl_target)
    return {
        "out_of_ctrl_range": out_of_ctrl_range,
        "over_force_limit": over_force_limit,
        "max_abs_required_tau": max_abs_tau,
        "max_abs_ctrl_offset": max_abs_ctrl_offset,
        "max_ctrl_range_violation": max_ctrl_range_violation,
        "max_force_limit_violation": max_force_limit_violation,
        "ctrl_targets": [float(value) for value in ctrl_targets],
    }


def _foot_geom_ids(shard: WholeBodyMuJoCoShard) -> set[int]:
    return {
        int(shard.mujoco.mj_name2id(shard.model, shard.mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in shard._foot_geoms
    }


def _center_of_mass(shard: WholeBodyMuJoCoShard, data: Any) -> list[float]:
    np = shard.np
    total_mass = float(np.sum(shard.model.body_mass))
    com = np.zeros(3, dtype=np.float64)
    for body_id in range(shard.model.nbody):
        com += float(shard.model.body_mass[body_id]) * np.asarray(data.xipos[body_id])
    com /= max(1e-12, total_mass)
    return [float(value) for value in com]


def _contact_report(shard: WholeBodyMuJoCoShard, data: Any) -> dict[str, Any]:
    taxonomy = contact_taxonomy(shard, data)
    counts = taxonomy["counts"]
    return {
        "foot_contacts": counts["support_foot_floor_contacts"],
        "non_foot_contacts": counts["forbidden_nonfoot_floor_contacts"],
        "self_contacts": counts["self_contacts"],
        "support_foot_floor_contacts": counts["support_foot_floor_contacts"],
        "forbidden_nonfoot_floor_contacts": counts["forbidden_nonfoot_floor_contacts"],
        "foot_normal_force_sum": taxonomy["foot_normal_force_sum"],
        "foot_normal_force_max": taxonomy["foot_normal_force_max"],
        "contacts_by_foot": taxonomy["contacts_by_foot"],
        "normal_force_by_foot": taxonomy["normal_force_by_foot"],
        "center_of_pressure_xy": taxonomy["center_of_pressure_xy"],
        "min_contact_distance": taxonomy["min_floor_contact_distance"],
        "min_self_contact_distance": taxonomy["min_self_contact_distance"],
        "taxonomy": taxonomy,
    }


def _support_points(shard: WholeBodyMuJoCoShard, data: Any) -> list[dict[str, Any]]:
    np = shard.np
    points: list[dict[str, Any]] = []
    for geom_name in sorted(shard._foot_geoms):
        geom_id = int(shard.mujoco.mj_name2id(shard.model, shard.mujoco.mjtObj.mjOBJ_GEOM, geom_name))
        center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        half = np.asarray(shard.model.geom_size[geom_id], dtype=np.float64)[:3]
        corners = [
            center + rot @ np.asarray([sx * half[0], sy * half[1], -half[2]], dtype=np.float64)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
        ]
        lowest = min(float(point[2]) for point in corners)
        for point in corners:
            if float(point[2]) - lowest <= 0.005:
                points.append(
                    {
                        "geom_id": geom_id,
                        "geom_name": geom_name,
                        "body_id": int(shard.model.geom_bodyid[geom_id]),
                        "point": [float(value) for value in point],
                        "bottom_height": float(point[2]),
                    }
                )
    return points


def _foot_bottom_heights(shard: WholeBodyMuJoCoShard, data: Any) -> dict[str, float]:
    heights: dict[str, float] = {}
    for point in _support_points(shard, data):
        name = str(point["geom_name"])
        height = float(point["bottom_height"])
        heights[name] = min(heights.get(name, height), height)
    return heights


def _qfrc_from_vertical_contact_forces(
    shard: WholeBodyMuJoCoShard,
    data: Any,
    support_points: list[dict[str, Any]],
    force_values: Any,
) -> Any:
    np = shard.np
    qfrc_contact = np.zeros(shard.model.nv, dtype=np.float64)
    for item, normal in zip(support_points, force_values):
        if abs(float(normal)) <= 1e-12:
            continue
        jacp = np.zeros((3, shard.model.nv), dtype=np.float64)
        jacr = np.zeros((3, shard.model.nv), dtype=np.float64)
        point = np.asarray(item["point"], dtype=np.float64)
        shard.mujoco.mj_jac(shard.model, data, jacp, jacr, point, int(item["body_id"]))
        qfrc_contact += jacp.T @ np.asarray([0.0, 0.0, float(normal)], dtype=np.float64)
    return qfrc_contact


def _normal_force_by_foot(
    support_points: list[dict[str, Any]],
    force_values: Any,
) -> dict[str, float]:
    by_foot: dict[str, float] = {}
    for item, force in zip(support_points, force_values):
        by_foot[item["geom_name"]] = by_foot.get(item["geom_name"], 0.0) + float(force)
    return dict(sorted(by_foot.items()))


def _solve_joint_bounded_contact_forces(
    shard: WholeBodyMuJoCoShard,
    data: Any,
    support_points: list[dict[str, Any]],
    com: list[float],
) -> dict[str, Any]:
    np = shard.np
    total_mass = float(np.sum(shard.model.body_mass))
    weight = total_mass * abs(float(shard.model.opt.gravity[2]))
    if not support_points or weight <= 0.0:
        return {
            "status": "infeasible",
            "weight": weight,
            "residual_norm": float("inf"),
            "negative_force_violation": float("inf"),
            "forces": [],
            "qfrc_required": np.zeros(shard.model.nv, dtype=np.float64),
            "joint_force": _joint_force_feasibility(shard, data, np.zeros(shard.model.nv, dtype=np.float64)),
            "normal_force_by_foot": {},
            "foot_load_deficit": float("inf"),
        }
    scipy_optimize = _scipy_optimize()
    a_matrix = np.asarray(
        [
            [1.0 for _ in support_points],
            [float(item["point"][0]) for item in support_points],
            [float(item["point"][1]) for item in support_points],
        ],
        dtype=np.float64,
    )
    rhs = np.asarray([weight, weight * com[0], weight * com[1]], dtype=np.float64)
    min_foot_load = _EQUILIBRIUM_THRESHOLDS["foot_load_fraction"] * weight
    foot_indices = {
        name: [index for index, point in enumerate(support_points) if point["geom_name"] == name]
        for name in sorted(shard._foot_geoms)
    }
    if any(not indices for indices in foot_indices.values()):
        force_values = np.zeros(len(support_points), dtype=np.float64)
        qfrc_required = np.asarray(data.qfrc_bias, dtype=np.float64)
        joint_force = _joint_force_feasibility(shard, data, qfrc_required)
        return {
            "status": "infeasible",
            "method": "slsqp_qp",
            "solver_success": False,
            "solver_message": "missing support points for at least one foot",
            "weight": weight,
            "residual_norm": float("inf"),
            "negative_force_violation": 0.0,
            "active_points": 0,
            "normal_force_sum": 0.0,
            "normal_force_max": 0.0,
            "normal_force_by_foot": {},
            "min_foot_load": min_foot_load,
            "foot_load_deficit": float("inf"),
            "joint_force": joint_force,
            "qfrc_required": qfrc_required,
            "forces": [],
        }

    initial = np.full(len(support_points), weight / len(support_points), dtype=np.float64)
    constraints = [
        {
            "type": "ineq",
            "fun": lambda force, indices=indices: float(np.sum(force[indices]) - min_foot_load),
        }
        for indices in foot_indices.values()
    ]

    def objective(force: Any) -> float:
        force_array = np.asarray(force, dtype=np.float64)
        residual = a_matrix @ force_array - rhs
        qfrc_contact = _qfrc_from_vertical_contact_forces(shard, data, support_points, force_array)
        qfrc_required = np.asarray(data.qfrc_bias, dtype=np.float64) - qfrc_contact
        joint_force = _joint_force_feasibility(shard, data, qfrc_required)
        return float(
            np.dot(residual, residual) / max(weight * weight, 1e-12)
            + 1e-8 * np.dot(force_array, force_array) / max(weight * weight, 1e-12)
            + 1e4 * joint_force["max_ctrl_range_violation"] ** 2
            + 1e2 * (joint_force["max_force_limit_violation"] / 100.0) ** 2
        )

    result = scipy_optimize.minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 4.0 * weight) for _ in support_points],
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 100, "disp": False},
    )
    force_values = np.asarray(result.x if hasattr(result, "x") else initial, dtype=np.float64)
    residual = float(np.linalg.norm(a_matrix @ force_values - rhs))
    negative = max(0.0, -float(np.min(force_values))) if len(force_values) else 0.0
    qfrc_contact = _qfrc_from_vertical_contact_forces(shard, data, support_points, force_values)
    qfrc_required = np.asarray(data.qfrc_bias, dtype=np.float64) - qfrc_contact
    joint_force = _joint_force_feasibility(shard, data, qfrc_required)
    normal_by_foot = _normal_force_by_foot(support_points, force_values)
    foot_load_deficit = sum(
        max(0.0, min_foot_load - normal_by_foot.get(name, 0.0))
        for name in shard._foot_geoms
    )
    force_records = []
    for item, force in zip(support_points, force_values):
        force_records.append(
            {
                "geom_name": item["geom_name"],
                "point": item["point"],
                "normal_force": float(force),
            }
        )
    status = (
        "feasible"
        if residual <= _EQUILIBRIUM_THRESHOLDS["contact_force_residual_norm"]
        and negative <= 1e-8 * weight
        and foot_load_deficit <= 1e-8 * weight
        and joint_force["over_force_limit"] == 0
        and joint_force["out_of_ctrl_range"] == 0
        else "infeasible"
    )
    return {
        "status": status,
        "method": "slsqp_qp",
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "weight": weight,
        "residual_norm": residual,
        "negative_force_violation": negative,
        "active_points": len([value for value in force_values if abs(float(value)) > 1e-9]),
        "normal_force_sum": float(np.sum(force_values)),
        "normal_force_max": float(np.max(force_values)) if len(force_values) else 0.0,
        "normal_force_by_foot": normal_by_foot,
        "min_foot_load": min_foot_load,
        "foot_load_deficit": foot_load_deficit,
        "joint_force": joint_force,
        "qfrc_required": qfrc_required,
        "forces": force_records,
    }


def _contact_consistent_snapshot(shard: WholeBodyMuJoCoShard, data: Any) -> dict[str, Any]:
    np = shard.np
    shard.mujoco.mj_forward(shard.model, data)
    com = _center_of_mass(shard, data)
    points = _support_points(shard, data)
    force_solution = _solve_joint_bounded_contact_forces(shard, data, points, com)
    required = force_solution["qfrc_required"]
    joint_force = force_solution["joint_force"]
    bottom_abs_max = max((abs(float(item["bottom_height"])) for item in points), default=float("inf"))
    root_force_norm = float(np.linalg.norm(required[:3]))
    root_torque_norm = float(np.linalg.norm(required[3:6]))
    return {
        "root_residual_force_norm": root_force_norm,
        "root_residual_torque_norm": root_torque_norm,
        "root_residual_wrench": [float(value) for value in required[:6]],
        "joint_force": joint_force,
        "contact_force_solution": {
            key: value for key, value in force_solution.items() if key != "qfrc_required"
        },
        "support_points": points,
        "support_bottom_abs_max": bottom_abs_max,
        "com": com,
        "qfrc_required": required,
    }


def _align_lowest_foot_to_penetration(
    shard: WholeBodyMuJoCoShard,
    data: Any,
    *,
    penetration: float,
) -> None:
    shard.mujoco.mj_forward(shard.model, data)
    points = _support_points(shard, data)
    if not points:
        return
    lowest = min(float(item["bottom_height"]) for item in points)
    data.qpos[2] += -penetration - lowest
    shard.mujoco.mj_forward(shard.model, data)


def _mj_forward_equilibrium_snapshot(shard: WholeBodyMuJoCoShard, data: Any) -> dict[str, Any]:
    np = shard.np
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.qfrc_applied[:] = 0.0
    shard.mujoco.mj_forward(shard.model, data)
    qacc = np.asarray(data.qacc, dtype=np.float64)
    qfrc_constraint = np.asarray(data.qfrc_constraint, dtype=np.float64)
    contact = _contact_report(shard, data)
    joint_errors = [
        abs(float(data.ctrl[actuator_id]) - float(data.qpos[qpos_address]))
        for actuator_id, qpos_address in zip(shard._actuator_ids, shard._joint_qpos)
    ]
    actuator_forces = [abs(float(data.actuator_force[actuator_id])) for actuator_id in shard._actuator_ids]
    saturation_events = 0
    for actuator_id, force in zip(shard._actuator_ids, actuator_forces):
        limit = max(abs(float(value)) for value in shard.model.actuator_forcerange[actuator_id])
        saturation_events += int(force >= 0.995 * limit)
    return {
        "qacc_root_norm": float(np.linalg.norm(qacc[:6])),
        "qacc_base_linear_norm": float(np.linalg.norm(qacc[:3])),
        "qacc_base_angular_norm": float(np.linalg.norm(qacc[3:6])),
        "qacc_joint_max": max((abs(float(qacc[dof])) for dof in shard._joint_dof), default=0.0),
        "qfrc_constraint_root_norm": float(np.linalg.norm(qfrc_constraint[:6])),
        "qfrc_constraint_joint_max": max(
            (abs(float(qfrc_constraint[dof])) for dof in shard._joint_dof),
            default=0.0,
        ),
        "contact": contact,
        "joint_error_max": max(joint_errors, default=0.0),
        "actuator_force_max": max(actuator_forces, default=0.0),
        "actuator_saturation_events": saturation_events,
    }


def _ctrl_range_feasibility(shard: WholeBodyMuJoCoShard, ctrl: Any) -> dict[str, Any]:
    max_violation = 0.0
    violations = 0
    for actuator_id in shard._actuator_ids:
        lower, upper = (float(value) for value in shard.model.actuator_ctrlrange[actuator_id])
        value = float(ctrl[actuator_id])
        violation = max(0.0, lower - value, value - upper)
        max_violation = max(max_violation, violation)
        violations += int(violation > 0.0)
    return {
        "violations": violations,
        "max_violation": max_violation,
    }


def _equilibrium_candidate_feasible(shard: WholeBodyMuJoCoShard, candidate: dict[str, Any]) -> bool:
    static = candidate["static"]
    forward = candidate["mj_forward"]
    contact_solution = static["contact_force_solution"]
    actual_ctrl = candidate.get("actual_ctrl_range", {"violations": 0, "max_violation": 0.0})
    return bool(
        static["root_residual_force_norm"] <= _EQUILIBRIUM_THRESHOLDS["root_residual_force_norm"]
        and static["root_residual_torque_norm"] <= _EQUILIBRIUM_THRESHOLDS["root_residual_torque_norm"]
        and static["support_bottom_abs_max"] <= _EQUILIBRIUM_THRESHOLDS["support_bottom_abs"]
        and contact_solution["status"] == "feasible"
        and static["joint_force"]["over_force_limit"] == 0
        and static["joint_force"]["out_of_ctrl_range"] == 0
        and forward["qacc_root_norm"] <= _EQUILIBRIUM_THRESHOLDS["qacc_root_norm"]
        and forward["qacc_joint_max"] <= _EQUILIBRIUM_THRESHOLDS["qacc_joint_max"]
        and forward["contact"]["non_foot_contacts"] == 0
        and forward["actuator_saturation_events"] == 0
        and actual_ctrl["violations"] == 0
        and all(forward["contact"]["contacts_by_foot"].get(name, 0) > 0 for name in shard._foot_geoms)
        and all(
            contact_solution["normal_force_by_foot"].get(name, 0.0) >= contact_solution["min_foot_load"]
            for name in shard._foot_geoms
        )
    )


def _joint_position_bounds(shard: WholeBodyMuJoCoShard, stance: Any) -> list[tuple[float, float]]:
    bounds = []
    max_adjustment = _EQUILIBRIUM_THRESHOLDS["max_joint_adjustment"]
    for joint, qpos_address in zip(shard.blueprint.joints, shard._joint_qpos):
        joint_id = int(shard.mujoco.mj_name2id(shard.model, shard.mujoco.mjtObj.mjOBJ_JOINT, joint.name))
        lower, upper = (float(value) for value in shard.model.jnt_range[joint_id])
        center = float(stance[qpos_address])
        bounds.append((max(lower, center - max_adjustment), min(upper, center + max_adjustment)))
    return bounds


def _candidate_vector_to_state(
    shard: WholeBodyMuJoCoShard,
    stance: Any,
    vector: Any,
) -> tuple[Any, Any, float]:
    np = shard.np
    joint_count = len(shard._joint_qpos)
    qpos = np.array(stance, copy=True)
    qpos[3:7] = _quat_from_roll_pitch_yaw(float(vector[0]), float(vector[1]), float(vector[2]))
    for qpos_address, value in zip(shard._joint_qpos, vector[4 : 4 + joint_count]):
        qpos[qpos_address] = float(value)
    ctrl = np.asarray(vector[4 + joint_count : 4 + joint_count + len(shard._actuator_ids)], dtype=np.float64)
    return qpos, ctrl, float(vector[3])


def _apply_joint_aware_state(
    shard: WholeBodyMuJoCoShard,
    data: Any,
    stance: Any,
    vector: Any,
) -> tuple[Any, Any]:
    qpos, ctrl, penetration = _candidate_vector_to_state(shard, stance, vector)
    _reset_to_qpos(shard, data, qpos)
    _align_lowest_foot_to_penetration(shard, data, penetration=penetration)
    data.ctrl[:] = ctrl
    shard.mujoco.mj_forward(shard.model, data)
    return shard.np.array(data.qpos, copy=True), shard.np.array(data.ctrl, copy=True)


def _max_joint_adjustment(shard: WholeBodyMuJoCoShard, stance: Any, qpos: Any) -> float:
    return max(
        (abs(float(qpos[address]) - float(stance[address])) for address in shard._joint_qpos),
        default=0.0,
    )


def _com_cop_distance(com: list[float], contact: dict[str, Any]) -> float | None:
    cop = contact["center_of_pressure_xy"]
    if cop is None:
        return None
    return math.hypot(com[0] - float(cop[0]), com[1] - float(cop[1]))


def _fall_reason(shard: WholeBodyMuJoCoShard, data: Any) -> str | None:
    if not all(math.isfinite(float(value)) for value in data.qpos):
        return "non_finite_qpos"
    if float(data.qpos[2]) < _fall_height_threshold(shard):
        return "base_height"
    gravity = _projected_gravity(tuple(float(value) for value in data.qpos[3:7]))
    if -gravity[2] < shard.config.upright_threshold:
        return "upright_tilt"
    return None


def _static_inverse_snapshot(shard: WholeBodyMuJoCoShard, data: Any) -> dict[str, Any]:
    shard.mujoco.mj_forward(shard.model, data)
    forward_qacc = shard.np.array(data.qacc, copy=True)
    data.qacc[:] = 0.0
    shard.mujoco.mj_inverse(shard.model, data)
    inverse_qfrc = shard.np.array(data.qfrc_inverse, copy=True)
    constraint_qfrc = shard.np.array(data.qfrc_constraint, copy=True)
    contact = _contact_report(shard, data)
    com = _center_of_mass(shard, data)
    root_force_norm = float(shard.np.linalg.norm(inverse_qfrc[:3]))
    root_torque_norm = float(shard.np.linalg.norm(inverse_qfrc[3:6]))
    return {
        "forward_root_acc_norm": float(shard.np.linalg.norm(forward_qacc[:6])),
        "forward_base_angular_acc_norm": float(shard.np.linalg.norm(forward_qacc[3:6])),
        "root_required_force_norm": root_force_norm,
        "root_required_torque_norm": root_torque_norm,
        "root_required_wrench": [float(value) for value in inverse_qfrc[:6]],
        "constraint_root_wrench": [float(value) for value in constraint_qfrc[:6]],
        "joint_force": _joint_force_feasibility(shard, data, inverse_qfrc),
        "contact": contact,
        "com": com,
        "com_cop_distance": _com_cop_distance(com, contact),
        "inverse_qfrc": inverse_qfrc,
    }


def solve_contact_consistent_equilibrium(shard: WholeBodyMuJoCoShard) -> dict[str, Any]:
    np = shard.np
    scipy_optimize = _scipy_optimize()
    data = shard.data[0]
    stance = _stance_qpos(shard)
    joint_bounds = _joint_position_bounds(shard, stance)
    lower_bounds = [-0.10, -0.10, -0.15, 0.0]
    upper_bounds = [0.10, 0.10, 0.15, 0.012]
    lower_bounds.extend(lower for lower, _ in joint_bounds)
    upper_bounds.extend(upper for _, upper in joint_bounds)
    for actuator_id in shard._actuator_ids:
        lower, upper = (float(value) for value in shard.model.actuator_ctrlrange[actuator_id])
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    def initial_vector(roll: float, pitch: float, yaw: float, penetration: float) -> Any:
        qpos = np.array(stance, copy=True)
        qpos[3:7] = _quat_from_roll_pitch_yaw(roll, pitch, yaw)
        _reset_to_qpos(shard, data, qpos)
        _align_lowest_foot_to_penetration(shard, data, penetration=penetration)
        _set_baseline_ctrl(shard, data)
        static = _contact_consistent_snapshot(shard, data)
        ctrl = _ctrl_from_generalized_force(shard, data, static["qfrc_required"])
        joint_values = [float(data.qpos[address]) for address in shard._joint_qpos]
        return np.asarray([roll, pitch, yaw, penetration, *joint_values, *ctrl], dtype=np.float64)

    def vector_from_current_data() -> Any:
        roll, pitch, yaw = _roll_pitch_yaw(data.qpos[3:7])
        foot_bottom = _foot_bottom_heights(shard, data)
        lowest = min(foot_bottom.values(), default=0.0)
        penetration = min(0.012, max(0.0, -float(lowest)))
        joint_values = [float(data.qpos[address]) for address in shard._joint_qpos]
        ctrl = [float(data.ctrl[actuator_id]) for actuator_id in shard._actuator_ids]
        return np.asarray([roll, pitch, yaw, penetration, *joint_values, *ctrl], dtype=np.float64)

    def settled_start_vector(control_steps: int) -> Any:
        _reset_to_qpos(shard, data, stance)
        _set_baseline_ctrl(shard, data)
        for _ in range(control_steps):
            for _ in range(shard.config.substeps):
                shard.mujoco.mj_step(shard.model, data)
            if _fall_reason(shard, data) is not None:
                break
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        shard.mujoco.mj_forward(shard.model, data)
        return vector_from_current_data()

    total_mass = float(np.sum(shard.model.body_mass))
    weight = total_mass * abs(float(shard.model.opt.gravity[2]))
    min_foot_load = _EQUILIBRIUM_THRESHOLDS["foot_load_fraction"] * weight

    def residual(vector: Any) -> Any:
        _apply_joint_aware_state(shard, data, stance, vector)
        forward = _mj_forward_equilibrium_snapshot(shard, data)
        support_points = _support_points(shard, data)
        foot_bottom = _foot_bottom_heights(shard, data)
        bottom_abs = max((abs(float(item["bottom_height"])) for item in support_points), default=1e3)
        contact = forward["contact"]
        qacc = np.asarray(data.qacc, dtype=np.float64)
        joint_qpos = np.asarray([data.qpos[address] for address in shard._joint_qpos], dtype=np.float64)
        stance_joint_qpos = np.asarray([stance[address] for address in shard._joint_qpos], dtype=np.float64)
        ctrl = np.asarray(data.ctrl, dtype=np.float64)
        values = []
        values.extend((qacc[:6] / _EQUILIBRIUM_THRESHOLDS["qacc_root_norm"]).tolist())
        values.extend((qacc[list(shard._joint_dof)] / _EQUILIBRIUM_THRESHOLDS["qacc_joint_max"]).tolist())
        values.append(bottom_abs / max(1e-9, 0.5 * _EQUILIBRIUM_THRESHOLDS["support_bottom_abs"]))
        values.extend(
            float(foot_bottom.get(name, 1.0)) / 0.001
            for name in sorted(shard._foot_geoms)
        )
        values.append(10.0 * float(contact["non_foot_contacts"]))
        values.extend(
            10.0
            * max(0.0, min_foot_load - float(contact["normal_force_by_foot"].get(name, 0.0)))
            / max(weight, 1e-9)
            for name in shard._foot_geoms
        )
        values.extend(
            10.0 * float(contact["contacts_by_foot"].get(name, 0) == 0)
            for name in shard._foot_geoms
        )
        values.extend(
            (
                0.05
                * (joint_qpos - stance_joint_qpos)
                / max(1e-9, _EQUILIBRIUM_THRESHOLDS["max_joint_adjustment"])
            ).tolist()
        )
        values.extend((0.01 * (ctrl - joint_qpos)).tolist())
        return np.asarray(values, dtype=np.float64)

    def candidate_snapshot(
        vector: Any,
        *,
        start_index: int,
        optimization: dict[str, Any],
    ) -> dict[str, Any]:
        qpos, ctrl = _apply_joint_aware_state(shard, data, stance, vector)
        static = _contact_consistent_snapshot(shard, data)
        data.ctrl[:] = ctrl
        shard.mujoco.mj_forward(shard.model, data)
        contact = _contact_report(shard, data)
        forward = _mj_forward_equilibrium_snapshot(shard, data)
        joint_force = static["joint_force"]
        foot_load_deficit = static["contact_force_solution"]["foot_load_deficit"]
        foot_contact_deficit = sum(
            int(forward["contact"]["contacts_by_foot"].get(name, 0) == 0)
            for name in shard._foot_geoms
        )
        actual_ctrl_range = _ctrl_range_feasibility(shard, ctrl)
        max_adjustment = _max_joint_adjustment(shard, stance, qpos)
        score = (
            10.0 * forward["qacc_root_norm"]
            + forward["qacc_joint_max"]
            + 1e-3 * forward["qfrc_constraint_root_norm"]
            + 1e-2 * forward["qfrc_constraint_joint_max"]
            + static["root_residual_force_norm"]
            + static["root_residual_torque_norm"]
            + 1000.0 * static["contact_force_solution"]["negative_force_violation"]
            + 1000.0 * static["contact_force_solution"]["residual_norm"]
            + 1000.0 * static["support_bottom_abs_max"]
            + 1000.0 * foot_contact_deficit
            + 100.0 * foot_load_deficit
            + 1000.0 * joint_force["over_force_limit"]
            + 1000.0 * joint_force["out_of_ctrl_range"]
            + 1000.0 * actual_ctrl_range["max_violation"]
            + 1000.0 * forward["contact"]["non_foot_contacts"]
            + 10.0 * max_adjustment
        )
        return {
            "score": float(score),
            "start_index": start_index,
            "optimization": optimization,
            "solver_kind": "joint_aware_least_squares_with_slsqp_contact_qp",
            "penetration": float(vector[3]),
            "roll_deg": math.degrees(float(vector[0])),
            "pitch_deg": math.degrees(float(vector[1])),
            "yaw_deg": math.degrees(float(vector[2])),
            "max_joint_adjustment": max_adjustment,
            "qpos": [float(value) for value in qpos],
            "ctrl": [float(value) for value in ctrl],
            "actual_ctrl_range": actual_ctrl_range,
            "static": {
                key: value
                for key, value in static.items()
                if key != "qfrc_required"
            },
            "post_ctrl_contact": contact,
            "mj_forward": forward,
        }

    starts = [
        initial_vector(0.0, 0.0, 0.0, 0.002),
        initial_vector(0.0, 0.0, 0.0, 0.001),
        initial_vector(0.03, 0.0, 0.0, 0.002),
        initial_vector(-0.03, 0.0, 0.0, 0.002),
        initial_vector(0.0, 0.03, 0.0, 0.002),
        initial_vector(0.0, -0.03, 0.0, 0.002),
        initial_vector(0.0, 0.0, 0.06, 0.002),
        initial_vector(0.0, 0.0, -0.06, 0.002),
        settled_start_vector(20),
        settled_start_vector(60),
    ]
    candidates = []
    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    for start_index, start_vector in enumerate(starts):
        vector0 = np.clip(start_vector, lower, upper)
        result = scipy_optimize.least_squares(
            residual,
            vector0,
            bounds=(lower, upper),
            max_nfev=400,
            xtol=1e-6,
            ftol=1e-6,
            gtol=1e-6,
            x_scale="jac",
        )
        optimization = {
            "method": "scipy_least_squares",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
        }
        candidates.append(candidate_snapshot(result.x, start_index=start_index, optimization=optimization))
    feasible_candidates = [candidate for candidate in candidates if _equilibrium_candidate_feasible(shard, candidate)]
    best = min(feasible_candidates or candidates, key=lambda item: item["score"])
    return {
        "schema": "contact_consistent_joint_aware_dynamic_equilibrium_v3",
        "status": "feasible" if feasible_candidates else "infeasible",
        "feasibility_thresholds": {
            "root_residual_force_norm_max": _EQUILIBRIUM_THRESHOLDS["root_residual_force_norm"],
            "root_residual_torque_norm_max": _EQUILIBRIUM_THRESHOLDS["root_residual_torque_norm"],
            "contact_force_residual_norm_max": _EQUILIBRIUM_THRESHOLDS["contact_force_residual_norm"],
            "support_bottom_abs_max": _EQUILIBRIUM_THRESHOLDS["support_bottom_abs"],
            "qacc_root_norm_max": _EQUILIBRIUM_THRESHOLDS["qacc_root_norm"],
            "qacc_joint_max": _EQUILIBRIUM_THRESHOLDS["qacc_joint_max"],
            "foot_load_fraction_min": _EQUILIBRIUM_THRESHOLDS["foot_load_fraction"],
            "max_joint_adjustment": _EQUILIBRIUM_THRESHOLDS["max_joint_adjustment"],
            "contact_force_solution_status": "feasible",
            "joint_over_force_limit": 0,
            "joint_out_of_ctrl_range": 0,
            "foot_contacts_min": len(shard._foot_geoms),
            "non_foot_contacts": 0,
            "actuator_saturation_events": 0,
        },
        "best": best,
        "candidate_preview": [
            {
                "score": candidate["score"],
                "status": "feasible" if _equilibrium_candidate_feasible(shard, candidate) else "infeasible",
                "start_index": candidate["start_index"],
                "qacc_root_norm": candidate["mj_forward"]["qacc_root_norm"],
                "qacc_joint_max": candidate["mj_forward"]["qacc_joint_max"],
                "max_joint_adjustment": candidate["max_joint_adjustment"],
                "optimization": candidate["optimization"],
            }
            for candidate in sorted(candidates, key=lambda item: item["score"])
        ],
        "candidate_count": len(candidates),
        "feasible_candidate_count": len(feasible_candidates),
    }


def run_rollout(
    shard: WholeBodyMuJoCoShard,
    mode: RolloutMode,
    *,
    equilibrium: dict[str, Any] | None,
    perturbation: dict[str, float] | None = None,
    horizon_steps: int,
) -> dict[str, Any]:
    np = shard.np
    data = shard.data[0]
    if mode.contact_equilibrium_ctrl and (equilibrium is None or equilibrium["status"] != "feasible"):
        return {
            "mode": mode.name,
            "perturbation": perturbation,
            "skipped": True,
            "skip_reason": "no_feasible_equilibrium",
            "equilibrium_feasible": False,
        }
    qpos = np.asarray(
        equilibrium["best"]["qpos"] if mode.contact_equilibrium_ctrl else _stance_qpos(shard),
        dtype=np.float64,
    )
    ctrl = None
    if mode.contact_equilibrium_ctrl and equilibrium is not None:
        ctrl = np.asarray(equilibrium["best"]["ctrl"], dtype=np.float64)
    _reset_to_qpos(shard, data, qpos)
    initial_error = 0.0
    impulse: dict[str, Any] | None = None
    if perturbation is not None:
        if perturbation["kind"] == "velocity":
            data.qvel[int(perturbation["dof"])] += float(perturbation["value"])
            initial_error = abs(float(perturbation["value"]))
        elif perturbation["kind"] == "impulse":
            impulse = perturbation
        else:
            raise ValueError(f"unknown perturbation kind: {perturbation['kind']}")
        shard.mujoco.mj_forward(shard.model, data)
    if ctrl is None:
        _set_baseline_ctrl(shard, data)
        if mode.constant_qfrc_bias_ctrl:
            shard.mujoco.mj_forward(shard.model, data)
            ctrl = _ctrl_from_generalized_force(shard, data, data.qfrc_bias)
        else:
            ctrl = np.array(data.ctrl, copy=True)
    data.ctrl[:] = ctrl
    root_qpos = np.array(data.qpos[:7], copy=True)
    root_qvel = np.array(data.qvel[:6], copy=True)
    saved_gravity = np.array(shard.model.opt.gravity, copy=True)
    if mode.zero_gravity:
        shard.model.opt.gravity[:] = 0.0

    heights: list[float] = []
    tilts: list[float] = []
    base_ang_acc: list[float] = []
    base_lin_acc: list[float] = []
    com_cop: list[float] = []
    foot_contacts: list[int] = []
    non_foot_contacts: list[int] = []
    normal_forces: list[float] = []
    joint_errors: list[float] = []
    actuator_forces: list[float] = []
    saturation_events = 0
    first_fall_step = None
    first_fall_reason = None
    try:
        for step_index in range(horizon_steps):
            for substep_index in range(shard.config.substeps):
                if mode.dynamic_qfrc_bias_ctrl:
                    shard.mujoco.mj_forward(shard.model, data)
                    data.ctrl[:] = _ctrl_from_generalized_force(shard, data, data.qfrc_bias)
                else:
                    data.ctrl[:] = ctrl
                data.qfrc_applied[:] = 0.0
                if impulse is not None and step_index == 0 and substep_index == 0:
                    data.qfrc_applied[int(impulse["dof"])] = float(impulse["impulse"]) / shard.model.opt.timestep
                shard.mujoco.mj_forward(shard.model, data)
                base_lin_acc.append(float(np.linalg.norm(data.qacc[:3])))
                base_ang_acc.append(float(np.linalg.norm(data.qacc[3:6])))
                shard.mujoco.mj_step(shard.model, data)
                if mode.root_locked:
                    data.qpos[:7] = root_qpos
                    data.qvel[:6] = root_qvel
                    shard.mujoco.mj_forward(shard.model, data)
            if impulse is not None and step_index == 0:
                initial_error = abs(float(data.qvel[int(impulse["dof"])]))
            contact = _contact_report(shard, data)
            com = _center_of_mass(shard, data)
            distance = _com_cop_distance(com, contact)
            if distance is not None:
                com_cop.append(distance)
            gravity = _projected_gravity(tuple(float(value) for value in data.qpos[3:7]))
            tilt = math.atan2(math.hypot(gravity[0], gravity[1]), max(1e-9, -gravity[2]))
            heights.append(float(data.qpos[2]))
            tilts.append(tilt)
            foot_contacts.append(int(contact["foot_contacts"]))
            non_foot_contacts.append(int(contact["non_foot_contacts"]))
            normal_forces.append(float(contact["foot_normal_force_sum"]))
            peak_error = 0.0
            peak_force = 0.0
            for actuator_id, qpos_address in zip(shard._actuator_ids, shard._joint_qpos):
                peak_error = max(peak_error, abs(float(data.ctrl[actuator_id]) - float(data.qpos[qpos_address])))
                force = abs(float(data.actuator_force[actuator_id]))
                peak_force = max(peak_force, force)
                limit = max(abs(float(value)) for value in shard.model.actuator_forcerange[actuator_id])
                saturation_events += int(force >= 0.995 * limit)
            joint_errors.append(peak_error)
            actuator_forces.append(peak_force)
            reason = _fall_reason(shard, data)
            if reason is not None:
                first_fall_step = step_index + 1
                first_fall_reason = reason
                break
    finally:
        shard.model.opt.gravity[:] = saved_gravity

    final_roll, final_pitch = _roll_pitch(data.qpos[3:7])
    final_error = 0.0
    response = None
    if perturbation is not None:
        final_error = abs(float(data.qvel[int(perturbation["dof"])]))
        response = _classify_perturbation_response(
            survived=first_fall_step is None,
            initial_error=initial_error,
            final_error=final_error,
        )
    return {
        "mode": mode.name,
        "perturbation": perturbation,
        "skipped": False,
        "equilibrium_feasible": bool(mode.contact_equilibrium_ctrl and equilibrium is not None and equilibrium["status"] == "feasible"),
        "survived": first_fall_step is None,
        "first_fall_step": first_fall_step,
        "first_fall_reason": first_fall_reason,
        "initial_error": initial_error,
        "final_error": final_error,
        "response": response,
        "steps_run": len(heights),
        "base_height_min": min(heights) if heights else float("nan"),
        "base_height_final": heights[-1] if heights else float("nan"),
        "tilt_max_rad": max(tilts) if tilts else float("nan"),
        "base_angular_acc_max": max(base_ang_acc) if base_ang_acc else float("nan"),
        "base_angular_acc_median": statistics.median(base_ang_acc) if base_ang_acc else float("nan"),
        "base_linear_acc_max": max(base_lin_acc) if base_lin_acc else float("nan"),
        "com_cop_distance_max": max(com_cop) if com_cop else None,
        "com_cop_distance_median": statistics.median(com_cop) if com_cop else None,
        "foot_contacts_median": statistics.median(foot_contacts) if foot_contacts else 0,
        "non_foot_contacts_max": max(non_foot_contacts) if non_foot_contacts else 0,
        "foot_normal_force_sum_max": max(normal_forces) if normal_forces else 0.0,
        "joint_error_max": max(joint_errors) if joint_errors else 0.0,
        "actuator_force_max": max(actuator_forces) if actuator_forces else 0.0,
        "actuator_saturation_events": saturation_events,
        "final_root_xy": [float(data.qpos[0]), float(data.qpos[1])],
        "final_roll_pitch": [final_roll, final_pitch],
    }


def _velocity_impulse_perturbations() -> list[dict[str, Any]]:
    return [
        {"kind": "velocity", "axis": "roll_rate", "dof": 3, "value": 0.10},
        {"kind": "velocity", "axis": "roll_rate", "dof": 3, "value": -0.10},
        {"kind": "velocity", "axis": "pitch_rate", "dof": 4, "value": 0.10},
        {"kind": "velocity", "axis": "pitch_rate", "dof": 4, "value": -0.10},
        {"kind": "velocity", "axis": "vx", "dof": 0, "value": 0.03},
        {"kind": "velocity", "axis": "vy", "dof": 1, "value": 0.03},
        {"kind": "impulse", "axis": "roll_impulse", "dof": 3, "impulse": 0.04},
        {"kind": "impulse", "axis": "roll_impulse", "dof": 3, "impulse": -0.04},
        {"kind": "impulse", "axis": "pitch_impulse", "dof": 4, "impulse": 0.04},
        {"kind": "impulse", "axis": "pitch_impulse", "dof": 4, "impulse": -0.04},
    ]


def _classify_perturbation_response(
    *,
    survived: bool,
    initial_error: float,
    final_error: float,
) -> str:
    if not survived:
        return "fell"
    if final_error <= 0.8 * max(initial_error, 1e-12):
        return "decayed"
    if final_error >= 1.2 * max(initial_error, 1e-12):
        return "grew"
    return "neutral"


def diagnose_instance(
    family: str,
    seed: int,
    *,
    range_fraction: float,
    horizon_steps: int,
) -> dict[str, Any]:
    generator = MorphologyGenerator()
    blueprint = generator.generate(family, seed)
    physical = generator.sample_physical_params(
        blueprint,
        seed + 10_000_000,
        range_fraction=range_fraction,
    )
    shard = WholeBodyMuJoCoShard(
        blueprint,
        physical=physical,
        num_envs=1,
        config=WholeBodyMuJoCoShardConfig(seed=seed),
    )
    instance_key = morphology_instance_key(blueprint, physical)
    equilibrium = solve_contact_consistent_equilibrium(shard)
    modes = _BIPED_MODES if family == "biped" else _QUADRUPED_MODES
    rollouts = [
        run_rollout(
            shard,
            mode,
            equilibrium=equilibrium,
            horizon_steps=horizon_steps,
        )
        for mode in modes
    ]
    perturbation_rollouts: list[dict[str, Any]] = []
    if family == "biped" and equilibrium is not None and equilibrium["status"] == "feasible":
        eq_mode = RolloutMode("contact_equilibrium_perturbed", contact_equilibrium_ctrl=True)
        for perturbation in _velocity_impulse_perturbations():
            result = run_rollout(
                shard,
                eq_mode,
                equilibrium=equilibrium,
                perturbation=perturbation,
                horizon_steps=horizon_steps,
            )
            perturbation_rollouts.append(result)
    return {
        "family": family,
        "seed": seed,
        "range_fraction": range_fraction,
        "structural_hash": blueprint.structural_hash,
        "morphology_instance_key": instance_key.manifest(),
        "num_joints": len(blueprint.joints),
        "num_feet": len(shard._foot_geoms),
        "stance_solution_hash": shard.stance_solution.solution_hash,
        "stance_cache_key": shard.stance_solution.cache_key,
        "contact_equilibrium": equilibrium,
        "rollouts": rollouts,
        "perturbation_rollouts": perturbation_rollouts,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for family in sorted({record["family"] for record in records}):
        for range_fraction in sorted({record["range_fraction"] for record in records if record["family"] == family}):
            subset = [
                record for record in records if record["family"] == family and record["range_fraction"] == range_fraction
            ]
            mode_names = sorted({rollout["mode"] for record in subset for rollout in record["rollouts"]})
            for mode in mode_names:
                rollouts = [rollout for record in subset for rollout in record["rollouts"] if rollout["mode"] == mode]
                skipped = [rollout for rollout in rollouts if rollout.get("skipped", False)]
                measured = [rollout for rollout in rollouts if not rollout.get("skipped", False)]
                key_mode = f"{mode}_feasible" if mode.startswith("contact_equilibrium") else mode
                key = f"{family}:rf{range_fraction:g}:{key_mode}"
                if not measured:
                    summary[key] = {
                        "seeds": 0,
                        "skipped": len(skipped),
                        "falls": 0,
                        "fall_ratio": None,
                    }
                    continue
                falls = [rollout for rollout in measured if not rollout["survived"]]
                summary[key] = {
                    "seeds": len(measured),
                    "skipped": len(skipped),
                    "falls": len(falls),
                    "fall_ratio": len(falls) / len(measured),
                    "first_fall_step_min": min((item["first_fall_step"] for item in falls), default=None),
                    "first_fall_step_max": max((item["first_fall_step"] for item in falls), default=None),
                    "tilt_max_rad_max": max(item["tilt_max_rad"] for item in measured),
                    "base_angular_acc_max": max(item["base_angular_acc_max"] for item in measured),
                    "com_cop_distance_median": statistics.median(
                        item["com_cop_distance_median"] for item in measured if item["com_cop_distance_median"] is not None
                    )
                    if any(item["com_cop_distance_median"] is not None for item in measured)
                    else None,
                    "actuator_saturation_events": sum(item["actuator_saturation_events"] for item in measured),
                }
            equilibrium = [
                record["contact_equilibrium"]
                for record in subset
                if record["contact_equilibrium"] is not None
            ]
            if equilibrium:
                eq_key = f"{family}:rf{range_fraction:g}:contact_equilibrium"
                summary[eq_key] = {
                    "seeds": len(equilibrium),
                    "feasible": sum(1 for item in equilibrium if item["status"] == "feasible"),
                    "feasible_ratio": sum(1 for item in equilibrium if item["status"] == "feasible") / len(equilibrium),
                    "feasible_candidate_count_total": sum(item["feasible_candidate_count"] for item in equilibrium),
                    "root_residual_force_norm_min": min(
                        item["best"]["static"]["root_residual_force_norm"] for item in equilibrium
                    ),
                    "root_residual_force_norm_median": statistics.median(
                        item["best"]["static"]["root_residual_force_norm"] for item in equilibrium
                    ),
                    "root_residual_torque_norm_median": statistics.median(
                        item["best"]["static"]["root_residual_torque_norm"] for item in equilibrium
                    ),
                    "qacc_root_norm_median": statistics.median(
                        item["best"]["mj_forward"]["qacc_root_norm"] for item in equilibrium
                    ),
                    "qacc_root_norm_min": min(item["best"]["mj_forward"]["qacc_root_norm"] for item in equilibrium),
                    "qacc_joint_max_median": statistics.median(
                        item["best"]["mj_forward"]["qacc_joint_max"] for item in equilibrium
                    ),
                    "max_joint_adjustment_median": statistics.median(
                        item["best"].get("max_joint_adjustment", 0.0) for item in equilibrium
                    ),
                    "contact_force_residual_norm_median": statistics.median(
                        item["best"]["static"]["contact_force_solution"]["residual_norm"] for item in equilibrium
                    ),
                    "support_bottom_abs_max": max(
                        item["best"]["static"]["support_bottom_abs_max"] for item in equilibrium
                    ),
                    "joint_over_force_limit_seeds": sum(
                        1 for item in equilibrium if item["best"]["static"]["joint_force"]["over_force_limit"] > 0
                    ),
                    "joint_out_of_ctrl_range_seeds": sum(
                        1 for item in equilibrium if item["best"]["static"]["joint_force"]["out_of_ctrl_range"] > 0
                    ),
                    "unloaded_foot_seeds": sum(
                        1 for item in equilibrium if item["best"]["static"]["contact_force_solution"]["foot_load_deficit"] > 0
                    ),
                    "non_foot_contact_seeds": sum(
                        1 for item in equilibrium if item["best"]["mj_forward"]["contact"]["non_foot_contacts"] > 0
                    ),
                    "actual_ctrl_range_violation_seeds": sum(
                        1 for item in equilibrium if item["best"].get("actual_ctrl_range", {}).get("violations", 0) > 0
                    ),
                    "actuator_saturation_seeds": sum(
                        1 for item in equilibrium if item["best"]["mj_forward"]["actuator_saturation_events"] > 0
                    ),
                    "actuator_force_max_median": statistics.median(
                        item["best"]["mj_forward"]["actuator_force_max"] for item in equilibrium
                    ),
                }
            perturbations = [
                item
                for record in subset
                for item in record["perturbation_rollouts"]
                if not item.get("skipped", False) and item.get("equilibrium_feasible", False)
            ]
            if perturbations:
                summary[f"{family}:rf{range_fraction:g}:contact_equilibrium_perturbations_feasible"] = {
                    "probes": len(perturbations),
                    "fell": sum(1 for item in perturbations if item["response"] == "fell"),
                    "grew": sum(1 for item in perturbations if item["response"] == "grew"),
                    "neutral": sum(1 for item in perturbations if item["response"] == "neutral"),
                    "decayed": sum(1 for item in perturbations if item["response"] == "decayed"),
                }
    return summary


def decide(summary: dict[str, Any]) -> dict[str, Any]:
    biped_baseline = [value for key, value in summary.items() if key.startswith("biped:") and key.endswith(":baseline")]
    biped_zero_g = [value for key, value in summary.items() if key.startswith("biped:") and key.endswith(":zero_gravity")]
    biped_root_locked = [value for key, value in summary.items() if key.startswith("biped:") and key.endswith(":root_locked")]
    biped_equilibrium = [
        value for key, value in summary.items() if key.startswith("biped:") and key.endswith(":contact_equilibrium")
    ]
    quadruped_equilibrium = [
        value for key, value in summary.items() if key.startswith("quadruped:") and key.endswith(":contact_equilibrium")
    ]
    biped_equilibrium_hold = [
        value
        for key, value in summary.items()
        if key.startswith("biped:") and key.endswith(":contact_equilibrium_hold_feasible")
    ]
    quadruped_equilibrium_hold = [
        value
        for key, value in summary.items()
        if key.startswith("quadruped:") and key.endswith(":contact_equilibrium_hold_feasible")
    ]
    biped_perturb = [
        value
        for key, value in summary.items()
        if key.startswith("biped:") and key.endswith(":contact_equilibrium_perturbations_feasible")
    ]
    baseline_fails = biped_baseline and min(item["fall_ratio"] for item in biped_baseline) > 0.90
    zero_g_survives = biped_zero_g and max(item["fall_ratio"] for item in biped_zero_g) == 0.0
    root_locked_survives = biped_root_locked and max(item["fall_ratio"] for item in biped_root_locked) == 0.0
    feasible_count = sum(item["feasible"] for item in biped_equilibrium)
    equilibrium_count = sum(item["seeds"] for item in biped_equilibrium)
    feasible_ratio = feasible_count / equilibrium_count if equilibrium_count else 0.0
    majority_equilibrium = bool(equilibrium_count and feasible_ratio >= 0.5)
    quadruped_positive_control = bool(
        quadruped_equilibrium
        and all(item["seeds"] > 0 and item["feasible"] > 0 for item in quadruped_equilibrium)
        and quadruped_equilibrium_hold
        and all(
            item["seeds"] > 0 and item["fall_ratio"] == 0.0
            for item in quadruped_equilibrium_hold
        )
    )
    hold_unstable = biped_equilibrium_hold and any(
        item["fall_ratio"] is not None and item["fall_ratio"] > 0.10 for item in biped_equilibrium_hold
    )
    perturb_unstable = biped_perturb and any(item["fell"] + item["grew"] > 0 for item in biped_perturb)
    if not quadruped_positive_control:
        return {
            "status": "positive_control_failed",
            "decision": "The joint-aware equilibrium solver has not passed the quadruped positive control.",
            "next_allowed_work": "Task067 R4a.2 solver diagnostics only; do not enter R4b or Task061/Task062.",
        }
    if majority_equilibrium and not hold_unstable and not perturb_unstable:
        return {
            "status": "stable_equilibrium_exists",
            "decision": "Most biped instances expose a feasible and stable zero-action equilibrium.",
            "next_allowed_work": "ZeroActionHoldSolution design, with embodiment contract/hash upgrade if hold semantics change.",
        }
    if majority_equilibrium and (hold_unstable or perturb_unstable):
        return {
            "status": "equilibrium_exists_but_perturbation_diverges",
            "decision": (
                "Most biped instances expose a feasible initial equilibrium, but feasible-only hold "
                "or velocity/impulse perturbation probes diverge."
            ),
            "next_allowed_work": "R4b bounded base-attitude/COM feedback diagnosis; do not enter Task061/Task062.",
        }
    if baseline_fails and zero_g_survives and root_locked_survives:
        return {
            "status": "continuous_joint_aware_no_equilibrium",
            "decision": (
                f"Only {feasible_count}/{equilibrium_count} biped instances satisfy the joint-bounded, "
                "contact-consistent MuJoCo-forward equilibrium gate. Gravity/free-base dynamics are causal, "
                "but most joint-aware stance solves still do not expose true zero-action equilibria."
            ),
            "next_allowed_work": "Task067 generator/inertia/contact repair only; Task061/Task062 remain blocked.",
        }
    return {
        "status": "inconclusive",
        "decision": "R4a did not cleanly isolate the causal branch; continue Task067 diagnostics only.",
        "next_allowed_work": "More Task067 diagnostics only.",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biped-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--quadruped-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--range-fractions", nargs="+", type=float, default=[0.0, 0.5])
    parser.add_argument("--horizon-steps", type=int, default=100)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    records = [
        diagnose_instance("biped", seed, range_fraction=range_fraction, horizon_steps=args.horizon_steps)
        for range_fraction in args.range_fractions
        for seed in args.biped_seeds
    ]
    records.extend(
        diagnose_instance("quadruped", seed, range_fraction=range_fraction, horizon_steps=args.horizon_steps)
        for range_fraction in args.range_fractions
        for seed in args.quadruped_seeds
    )
    summary = summarize(records)
    payload = {
        "schema": "task067_r4a2_joint_aware_equilibrium_diagnosis_v3",
        "embodiment_contract_version": PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
        "embodiment_contract_hash": PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
        "stance_solution_contract_version": STANCE_SOLUTION_CONTRACT_VERSION,
        "stance_solution_contract_hash": STANCE_SOLUTION_CONTRACT_HASH,
        "equilibrium_thresholds": _EQUILIBRIUM_THRESHOLDS,
        "horizon_steps": args.horizon_steps,
        "biped_seeds": args.biped_seeds,
        "quadruped_seeds": args.quadruped_seeds,
        "range_fractions": args.range_fractions,
        "summary": summary,
        "decision": decide(summary),
        "records": records,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
