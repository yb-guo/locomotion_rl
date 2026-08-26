"""Actual MuJoCo static-equilibrium stance realization.

This module is intentionally artifact-free: it consumes only the compiled
MuJoCo model/data plus the exact blueprint/physical instance.  Contact-wrench
logic can be used elsewhere as an initializer, but acceptance here is based on
MuJoCo's actual contact EFC, qacc, support loads, collision taxonomy, and
position-actuator control bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from h200_locomotion_lab.robots.procedural_morphology import (
    MorphologyBlueprint,
    PhysicalParams,
)

_STRICT_ROOT_QACC_NORM = 1e-5
_STRICT_JOINT_QACC_MAX = 1e-4
_ROOT_ROLL_PITCH_BOUND = 0.25
_PENETRATION_MAX = 0.012
_KINEMATIC_HEIGHT_SCALE = 5e-4
_DYNAMICS_HEIGHT_SCALE = 8e-4
_CLEARANCE_TARGET = 5e-4
_KINEMATIC_CLEARANCE_TARGET = 2e-3
_SUPPORT_MARGIN_TARGET = 1e-3
_FOOT_LOAD_FRACTION = 0.05
_CTRL_FINITE_DIFFERENCE = 1e-4
_KINEMATIC_REGULARIZATION = 1e-4
_DYNAMICS_REGULARIZATION = 1e-6
_LOAD_DEFICIT_WEIGHT = 12.0
_CLEARANCE_DEFICIT_WEIGHT = 15.0
_PENETRATION_SCHEDULE = (
    0.00025,
    0.00050,
    0.00075,
    0.00100,
    0.00125,
    0.00150,
    0.00200,
    0.00300,
    0.00400,
    0.00600,
    0.00800,
    0.01000,
    0.01200,
)
_FAST_PENETRATION_SCHEDULE = (0.00100, 0.00050, 0.00025, 0.00150, 0.00200, 0.00300)


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
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


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


@dataclass(frozen=True, slots=True)
class _Context:
    mujoco: Any
    np: Any
    model: Any
    data: Any
    blueprint: MorphologyBlueprint
    physical: PhysicalParams | None
    joint_qpos: tuple[int, ...]
    joint_dof: tuple[int, ...]
    actuator_ids: tuple[int, ...]
    foot_geoms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _QposStart:
    name: str
    qpos: Any


def _quat_from_roll_pitch_yaw(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
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


def _reset_to_qpos(ctx: _Context, qpos: Any, ctrl: Any | None = None) -> None:
    ctx.mujoco.mj_resetData(ctx.model, ctx.data)
    ctx.data.qpos[:] = qpos
    ctx.data.qvel[:] = 0.0
    ctx.data.qacc[:] = 0.0
    ctx.data.qfrc_applied[:] = 0.0
    if ctrl is None:
        ctx.data.ctrl[:] = 0.0
    else:
        ctx.data.ctrl[:] = ctrl
    ctx.mujoco.mj_forward(ctx.model, ctx.data)


def _shrink_bounds(lower: float, upper: float, margin: float) -> tuple[float, float]:
    if upper - lower <= 2.0 * margin:
        midpoint = 0.5 * (lower + upper)
        return midpoint, midpoint
    return lower + margin, upper - margin


def _joint_bounds(ctx: _Context, margin: float) -> tuple[Any, Any]:
    lower: list[float] = []
    upper: list[float] = []
    for joint in ctx.blueprint.joints:
        joint_id = int(ctx.mujoco.mj_name2id(ctx.model, ctx.mujoco.mjtObj.mjOBJ_JOINT, joint.name))
        lo, hi = (float(value) for value in ctx.model.jnt_range[joint_id])
        shrunk = _shrink_bounds(lo, hi, margin)
        lower.append(shrunk[0])
        upper.append(shrunk[1])
    return ctx.np.asarray(lower, dtype=ctx.np.float64), ctx.np.asarray(upper, dtype=ctx.np.float64)


def _ctrl_bounds(ctx: _Context, margin: float) -> tuple[Any, Any]:
    lower: list[float] = []
    upper: list[float] = []
    for actuator_id in ctx.actuator_ids:
        lo, hi = (float(value) for value in ctx.model.actuator_ctrlrange[int(actuator_id)])
        shrunk = _shrink_bounds(lo, hi, margin)
        lower.append(shrunk[0])
        upper.append(shrunk[1])
    return ctx.np.asarray(lower, dtype=ctx.np.float64), ctx.np.asarray(upper, dtype=ctx.np.float64)


def _foot_geom_ids(ctx: _Context) -> set[int]:
    return {
        int(ctx.mujoco.mj_name2id(ctx.model, ctx.mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in ctx.foot_geoms
    }


def _floor_geom_id(ctx: _Context) -> int:
    return int(ctx.mujoco.mj_name2id(ctx.model, ctx.mujoco.mjtObj.mjOBJ_GEOM, "floor"))


def _collidable_robot_geom_ids(ctx: _Context) -> list[int]:
    floor = _floor_geom_id(ctx)
    ids: list[int] = []
    for geom_id in range(ctx.model.ngeom):
        if geom_id == floor or int(ctx.model.geom_bodyid[geom_id]) == 0:
            continue
        if int(ctx.model.geom_contype[geom_id]) == 0 and int(ctx.model.geom_conaffinity[geom_id]) == 0:
            continue
        ids.append(int(geom_id))
    return ids


def _collides(ctx: _Context, geom1: int, geom2: int) -> bool:
    contype1 = int(ctx.model.geom_contype[int(geom1)])
    contype2 = int(ctx.model.geom_contype[int(geom2)])
    affinity1 = int(ctx.model.geom_conaffinity[int(geom1)])
    affinity2 = int(ctx.model.geom_conaffinity[int(geom2)])
    return bool((contype1 & affinity2) or (contype2 & affinity1))


def _self_collision_geom_pairs(ctx: _Context) -> list[tuple[int, int]]:
    geoms = _collidable_robot_geom_ids(ctx)
    pairs: list[tuple[int, int]] = []
    for index, geom1 in enumerate(geoms):
        for geom2 in geoms[index + 1 :]:
            if int(ctx.model.geom_bodyid[geom1]) == int(ctx.model.geom_bodyid[geom2]):
                continue
            if _collides(ctx, geom1, geom2):
                pairs.append((geom1, geom2))
    return pairs


def _footpad_corners(ctx: _Context) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for foot in ctx.foot_geoms:
        geom_id = int(ctx.mujoco.mj_name2id(ctx.model, ctx.mujoco.mjtObj.mjOBJ_GEOM, foot))
        center = ctx.np.asarray(ctx.data.geom_xpos[geom_id], dtype=ctx.np.float64)
        rot = ctx.np.asarray(ctx.data.geom_xmat[geom_id], dtype=ctx.np.float64).reshape(3, 3)
        half = ctx.np.asarray(ctx.model.geom_size[geom_id], dtype=ctx.np.float64)[:3]
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                point = center + rot @ ctx.np.asarray(
                    [sx * half[0], sy * half[1], -half[2]],
                    dtype=ctx.np.float64,
                )
                rows.append(
                    {
                        "foot": foot,
                        "geom_id": geom_id,
                        "point": [float(value) for value in point],
                        "height": float(point[2]),
                    }
                )
    return rows


def _flat_patch_report(ctx: _Context, *, penetration: float) -> dict[str, Any]:
    corners = _footpad_corners(ctx)
    heights = [float(corner["height"]) for corner in corners]
    foot_reports: dict[str, dict[str, float | int]] = {}
    for foot in ctx.foot_geoms:
        foot_heights = [float(corner["height"]) for corner in corners if corner["foot"] == foot]
        foot_reports[foot] = {
            "corner_count": len(foot_heights),
            "min_height": min(foot_heights, default=float("inf")),
            "max_height": max(foot_heights, default=float("-inf")),
            "height_spread": max(foot_heights, default=0.0) - min(foot_heights, default=0.0),
            "height_error_to_penetration_max_abs": max(
                (abs(height + penetration) for height in foot_heights),
                default=float("inf"),
            ),
        }
    return {
        "corner_count": len(corners),
        "global_min_height": min(heights, default=float("inf")),
        "global_max_height": max(heights, default=float("-inf")),
        "global_height_spread": max(heights, default=0.0) - min(heights, default=0.0),
        "height_error_to_penetration_max_abs": max(
            (abs(height + penetration) for height in heights),
            default=float("inf"),
        ),
        "feet": foot_reports,
    }


def _flat_geometry_realized(report: dict[str, Any]) -> bool:
    return bool(
        float(report["height_error_to_penetration_max_abs"]) <= 1e-3
        and float(report["global_height_spread"]) <= 1e-3
        and all(float(foot["height_spread"]) <= 1e-3 for foot in report["feet"].values())
    )


def _continuous_clearance(ctx: _Context) -> dict[str, Any]:
    foot_ids = _foot_geom_ids(ctx)
    floor = _floor_geom_id(ctx)
    self_distances = []
    for geom1, geom2 in _self_collision_geom_pairs(ctx):
        fromto = ctx.np.zeros(6, dtype=ctx.np.float64)
        distance = float(ctx.mujoco.mj_geomDistance(ctx.model, ctx.data, geom1, geom2, 0.2, fromto))
        self_distances.append(
            {
                "geom_pair": [int(geom1), int(geom2)],
                "distance": distance,
                "is_foot_foot_pair": bool(geom1 in foot_ids and geom2 in foot_ids),
            }
        )
    nonfoot_floor = []
    for geom_id in _collidable_robot_geom_ids(ctx):
        if geom_id in foot_ids:
            continue
        fromto = ctx.np.zeros(6, dtype=ctx.np.float64)
        distance = float(ctx.mujoco.mj_geomDistance(ctx.model, ctx.data, floor, geom_id, 0.2, fromto))
        nonfoot_floor.append({"geom": int(geom_id), "distance_to_floor": distance})
    return {
        "self_distances": sorted(self_distances, key=lambda row: float(row["distance"])),
        "nonfoot_floor_distances": sorted(
            nonfoot_floor,
            key=lambda row: float(row["distance_to_floor"]),
        ),
        "minimum_self_pair_distance": min(
            (float(row["distance"]) for row in self_distances),
            default=None,
        ),
        "minimum_nonfoot_floor_distance": min(
            (float(row["distance_to_floor"]) for row in nonfoot_floor),
            default=None,
        ),
    }


def _clearance_residuals(ctx: _Context, *, target: float) -> list[float]:
    clearance = _continuous_clearance(ctx)
    values: list[float] = []
    for row in clearance["self_distances"]:
        distance = float(row["distance"])
        values.append(_CLEARANCE_DEFICIT_WEIGHT * max(0.0, target - distance) / max(target, 1e-12))
    for row in clearance["nonfoot_floor_distances"]:
        distance = float(row["distance_to_floor"])
        values.append(_CLEARANCE_DEFICIT_WEIGHT * max(0.0, target - distance) / max(target, 1e-12))
    return values


def _center_of_mass(ctx: _Context) -> Any:
    total_mass = float(ctx.np.sum(ctx.model.body_mass))
    com = ctx.np.zeros(3, dtype=ctx.np.float64)
    for body_id in range(ctx.model.nbody):
        com += float(ctx.model.body_mass[body_id]) * ctx.np.asarray(ctx.data.xipos[body_id])
    return com / max(1e-12, total_mass)


def _support_report(ctx: _Context) -> dict[str, Any]:
    corners = _footpad_corners(ctx)
    com = _center_of_mass(ctx)
    margin = _support_margin(
        (float(com[0]), float(com[1])),
        [(float(corner["point"][0]), float(corner["point"][1])) for corner in corners],
    )
    return {"com": [float(value) for value in com], "support_margin": margin}


def _foot_load_threshold(ctx: _Context) -> float:
    total_mass = float(ctx.np.sum(ctx.model.body_mass))
    return _FOOT_LOAD_FRACTION * total_mass * abs(float(ctx.model.opt.gravity[2]))


def _contact_report(ctx: _Context) -> dict[str, Any]:
    floor_id = _floor_geom_id(ctx)
    foot_by_id = {
        int(ctx.mujoco.mj_name2id(ctx.model, ctx.mujoco.mjtObj.mjOBJ_GEOM, name)): name
        for name in ctx.foot_geoms
    }
    contacts_by_foot = {name: 0 for name in ctx.foot_geoms}
    normal_by_foot = {name: 0.0 for name in ctx.foot_geoms}
    support_contacts = 0
    nonfoot_contacts = 0
    self_contacts = 0
    weighted_xy = ctx.np.zeros(2, dtype=ctx.np.float64)
    foot_normal_sum = 0.0
    for index in range(int(ctx.data.ncon)):
        contact = ctx.data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        has_floor = floor_id in (geom1, geom2)
        other = geom2 if geom1 == floor_id else geom1
        foot_name = foot_by_id.get(other) if has_floor else None
        force = ctx.np.zeros(6, dtype=ctx.np.float64)
        ctx.mujoco.mj_contactForce(ctx.model, ctx.data, index, force)
        normal_force = max(0.0, float(force[0]))
        if has_floor and foot_name is not None:
            support_contacts += 1
            contacts_by_foot[foot_name] += 1
            normal_by_foot[foot_name] += normal_force
            foot_normal_sum += normal_force
            weighted_xy += normal_force * ctx.np.asarray(contact.pos[:2], dtype=ctx.np.float64)
        elif has_floor:
            nonfoot_contacts += 1
        else:
            self_contacts += 1
    center_of_pressure = None
    if foot_normal_sum > 1e-9:
        center_of_pressure = [float(value) for value in weighted_xy / foot_normal_sum]
    return {
        "support_foot_floor_contacts": support_contacts,
        "forbidden_nonfoot_floor_contacts": nonfoot_contacts,
        "self_contacts": self_contacts,
        "contacts_by_foot": contacts_by_foot,
        "normal_force_by_foot": normal_by_foot,
        "foot_normal_force_sum": foot_normal_sum,
        "center_of_pressure_xy": center_of_pressure,
    }


def _snapshot(ctx: _Context, qpos: Any, ctrl: Any) -> dict[str, Any]:
    _reset_to_qpos(ctx, qpos, ctrl)
    qacc = ctx.np.asarray(ctx.data.qacc, dtype=ctx.np.float64)
    contact = _contact_report(ctx)
    min_load = _foot_load_threshold(ctx)
    double_support = all(
        int(contact["contacts_by_foot"].get(name, 0)) > 0
        and float(contact["normal_force_by_foot"].get(name, 0.0)) >= min_load
        for name in ctx.foot_geoms
    )
    actuator_saturation_events = 0
    actuator_force_max = 0.0
    for actuator_id in ctx.actuator_ids:
        force = abs(float(ctx.data.actuator_force[int(actuator_id)]))
        limit = max(abs(float(value)) for value in ctx.model.actuator_forcerange[int(actuator_id)])
        actuator_force_max = max(actuator_force_max, force)
        actuator_saturation_events += int(force >= 0.995 * limit)
    return {
        "root_qacc_norm": float(ctx.np.linalg.norm(qacc[:6])),
        "joint_qacc_max": max((abs(float(qacc[dof])) for dof in ctx.joint_dof), default=0.0),
        "double_support": double_support,
        "contact": contact,
        "minimum_foot_load": min_load,
        "actuator_saturation_events": actuator_saturation_events,
        "actuator_force_max": actuator_force_max,
    }


def _strict_actual_equilibrium(snapshot: dict[str, Any]) -> bool:
    contact = snapshot["contact"]
    return bool(
        float(snapshot["root_qacc_norm"]) <= _STRICT_ROOT_QACC_NORM
        and float(snapshot["joint_qacc_max"]) <= _STRICT_JOINT_QACC_MAX
        and bool(snapshot["double_support"])
        and int(contact["forbidden_nonfoot_floor_contacts"]) == 0
        and int(contact["self_contacts"]) == 0
        and int(snapshot["actuator_saturation_events"]) == 0
    )


def _qpos_vector(ctx: _Context, qpos: Any) -> Any:
    roll, pitch, _ = _roll_pitch_yaw(qpos[3:7])
    joints = [float(qpos[int(address)]) for address in ctx.joint_qpos]
    return ctx.np.asarray([roll, pitch, *joints], dtype=ctx.np.float64)


def _apply_qpos_vector(ctx: _Context, template: Any, vector: Any) -> Any:
    qpos = ctx.np.asarray(template, dtype=ctx.np.float64).copy()
    qpos[0] = 0.0
    qpos[1] = 0.0
    qpos[3:7] = _quat_from_roll_pitch_yaw(float(vector[0]), float(vector[1]), 0.0)
    for address, value in zip(ctx.joint_qpos, vector[2:]):
        qpos[int(address)] = float(value)
    _reset_to_qpos(ctx, qpos)
    corners = _footpad_corners(ctx)
    if corners:
        mean_height = float(ctx.np.mean([float(corner["height"]) for corner in corners]))
        ctx.data.qpos[2] -= mean_height
        ctx.mujoco.mj_forward(ctx.model, ctx.data)
    return ctx.np.asarray(ctx.data.qpos, dtype=ctx.np.float64).copy()


def _kinematic_residual(ctx: _Context, vector: Any, start: Any, lower: Any, upper: Any) -> Any:
    patch = _flat_patch_report(ctx, penetration=0.0)
    heights = [
        float(corner["height"])
        for corner in _footpad_corners(ctx)
    ]
    mean_height = float(ctx.np.mean(heights)) if heights else 0.0
    values = [(height - mean_height) / _KINEMATIC_HEIGHT_SCALE for height in heights]
    values.extend(_clearance_residuals(ctx, target=_KINEMATIC_CLEARANCE_TARGET))
    support = _support_report(ctx)
    margin = float(support["support_margin"]["margin"])
    values.append(
        10.0 * max(0.0, _SUPPORT_MARGIN_TARGET - margin) / max(_SUPPORT_MARGIN_TARGET, 1e-12)
    )
    span = ctx.np.maximum(1e-6, upper - lower)
    values.extend(_KINEMATIC_REGULARIZATION * (ctx.np.asarray(vector) - start) / span)
    if not patch["corner_count"]:
        values.append(100.0)
    return ctx.np.asarray(values, dtype=ctx.np.float64)


def _solve_kinematic(
    ctx: _Context,
    start: _QposStart,
    *,
    joint_margin: float,
    max_nfev: int,
) -> dict[str, Any]:
    from scipy import optimize

    joint_lower, joint_upper = _joint_bounds(ctx, joint_margin)
    lower = ctx.np.asarray([-_ROOT_ROLL_PITCH_BOUND, -_ROOT_ROLL_PITCH_BOUND, *joint_lower])
    upper = ctx.np.asarray([_ROOT_ROLL_PITCH_BOUND, _ROOT_ROLL_PITCH_BOUND, *joint_upper])
    raw_start = _qpos_vector(ctx, start.qpos)
    vector0 = ctx.np.clip(raw_start, lower, upper)

    def residual(vector: Any) -> Any:
        _apply_qpos_vector(ctx, start.qpos, vector)
        return _kinematic_residual(ctx, vector, vector0, lower, upper)

    result = optimize.least_squares(
        residual,
        vector0,
        bounds=(lower, upper),
        method="trf",
        max_nfev=max_nfev,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        x_scale="jac",
        diff_step=1e-5,
    )
    qpos = _apply_qpos_vector(ctx, start.qpos, result.x)
    patch = _flat_patch_report(ctx, penetration=0.0)
    clearance = _continuous_clearance(ctx)
    support = _support_report(ctx)
    min_self = clearance["minimum_self_pair_distance"]
    min_nonfoot = clearance["minimum_nonfoot_floor_distance"]
    passed = bool(
        _flat_geometry_realized(patch)
        and (min_self is None or float(min_self) >= 0.0)
        and (min_nonfoot is None or float(min_nonfoot) >= 0.0)
        and bool(support["support_margin"]["inside"])
        and float(support["support_margin"]["margin"]) >= _SUPPORT_MARGIN_TARGET
    )
    return {
        "phase": "kinematic_qpos_only",
        "start_name": start.name,
        "qpos": [float(value) for value in qpos],
        "solver": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "max_nfev": max_nfev,
        },
        "flat_patch": patch,
        "continuous_clearance": clearance,
        "support": support,
        "passed": passed,
    }


def _kinematic_sort_key(attempt: dict[str, Any]) -> tuple[float, float, float, float]:
    clearance = attempt["continuous_clearance"]
    min_self = clearance["minimum_self_pair_distance"]
    min_nonfoot = clearance["minimum_nonfoot_floor_distance"]
    return (
        0.0 if attempt["passed"] else 1.0,
        max(
            float(attempt["flat_patch"]["height_error_to_penetration_max_abs"]),
            float(attempt["flat_patch"]["global_height_spread"]),
        ),
        max(0.0, -float(min_self)) if min_self is not None else 0.0,
        max(0.0, -float(min_nonfoot)) if min_nonfoot is not None else 0.0,
    )


def _align_to_penetration(ctx: _Context, qpos: Any, *, penetration: float) -> Any:
    _reset_to_qpos(ctx, qpos)
    corners = _footpad_corners(ctx)
    if corners:
        mean_height = float(ctx.np.mean([float(corner["height"]) for corner in corners]))
        ctx.data.qpos[2] += -float(penetration) - mean_height
        ctx.mujoco.mj_forward(ctx.model, ctx.data)
    return ctx.np.asarray(ctx.data.qpos, dtype=ctx.np.float64).copy()


def _contact_entry_steps(
    ctx: _Context,
    kinematic: dict[str, Any],
    *,
    penetration_schedule: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for penetration in penetration_schedule:
        qpos = _align_to_penetration(
            ctx,
            ctx.np.asarray(kinematic["qpos"], dtype=ctx.np.float64),
            penetration=penetration,
        )
        contact = _contact_report(ctx)
        entered = bool(
            all(int(contact["contacts_by_foot"].get(name, 0)) > 0 for name in ctx.foot_geoms)
            and int(contact["forbidden_nonfoot_floor_contacts"]) == 0
            and int(contact["self_contacts"]) == 0
        )
        rows.append(
            {
                "phase": "contact_entry_penetration_continuation",
                "kinematic_start_name": kinematic["start_name"],
                "penetration": penetration,
                "qpos": [float(value) for value in qpos],
                "entered_expected_mode": entered,
                "contact": contact,
                "flat_patch": _flat_patch_report(ctx, penetration=penetration),
                "continuous_clearance": _continuous_clearance(ctx),
            }
        )
    return rows


def _baseline_ctrl(ctx: _Context, qpos: Any, *, ctrl_margin: float) -> Any:
    lower, upper = _ctrl_bounds(ctx, ctrl_margin)
    ctrl = ctx.np.zeros(ctx.model.nu, dtype=ctx.np.float64)
    for local_index, (actuator_id, qpos_address) in enumerate(zip(ctx.actuator_ids, ctx.joint_qpos)):
        value = min(float(upper[local_index]), max(float(lower[local_index]), float(qpos[int(qpos_address)])))
        ctrl[int(actuator_id)] = value
    return ctrl


def _qacc_scaled(ctx: _Context, qpos: Any, ctrl: Any) -> Any:
    _reset_to_qpos(ctx, qpos, ctrl)
    qacc = ctx.np.asarray(ctx.data.qacc, dtype=ctx.np.float64)
    return ctx.np.asarray(
        [
            *list(qacc[:6] / _STRICT_ROOT_QACC_NORM),
            *list(qacc[list(ctx.joint_dof)] / _STRICT_JOINT_QACC_MAX),
        ],
        dtype=ctx.np.float64,
    )


def _bounded_ctrl_subproblem(
    ctx: _Context,
    qpos: Any,
    *,
    ctrl_margin: float,
) -> dict[str, Any]:
    from scipy import optimize

    lower, upper = _ctrl_bounds(ctx, ctrl_margin)
    start_full = _baseline_ctrl(ctx, qpos, ctrl_margin=ctrl_margin)
    start = ctx.np.asarray(
        [float(start_full[int(actuator_id)]) for actuator_id in ctx.actuator_ids],
        dtype=ctx.np.float64,
    )
    y0 = _qacc_scaled(ctx, qpos, start_full)
    columns = []
    for local_index, actuator_id in enumerate(ctx.actuator_ids):
        step = _CTRL_FINITE_DIFFERENCE
        if start[local_index] + step > upper[local_index]:
            step = -_CTRL_FINITE_DIFFERENCE
        if start[local_index] + step < lower[local_index]:
            step = 0.5 * (upper[local_index] - lower[local_index])
        if abs(step) <= 1e-12:
            columns.append(ctx.np.zeros_like(y0))
            continue
        probe = start_full.copy()
        probe[int(actuator_id)] = float(start[local_index] + step)
        columns.append((_qacc_scaled(ctx, qpos, probe) - y0) / step)
    matrix = ctx.np.asarray(columns, dtype=ctx.np.float64).T
    rhs = -y0 + matrix @ start
    result = optimize.lsq_linear(matrix, rhs, bounds=(lower, upper), tol=1e-12, max_iter=200)
    ctrl = ctx.np.zeros(ctx.model.nu, dtype=ctx.np.float64)
    for actuator_id, value in zip(ctx.actuator_ids, result.x):
        ctrl[int(actuator_id)] = float(value)
    predicted = y0 + matrix @ (result.x - start)
    actual = _qacc_scaled(ctx, qpos, ctrl)
    return {
        "phase": "bounded_qacc_ctrl_linear_subproblem",
        "fixed_qpos": True,
        "ctrl": [float(value) for value in ctrl],
        "solver": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "nit": int(result.nit),
        },
        "predicted_scaled_qacc_max_abs": float(max((abs(value) for value in predicted), default=0.0)),
        "actual_scaled_qacc_max_abs": float(max((abs(value) for value in actual), default=0.0)),
        "affine_prediction_error_max_abs": float(
            max((abs(value) for value in actual - predicted), default=0.0)
        ),
    }


def _dynamic_bounds(
    ctx: _Context,
    *,
    joint_margin: float,
    ctrl_margin: float,
) -> tuple[Any, Any]:
    joint_lower, joint_upper = _joint_bounds(ctx, joint_margin)
    ctrl_lower, ctrl_upper = _ctrl_bounds(ctx, ctrl_margin)
    lower = ctx.np.asarray(
        [-_ROOT_ROLL_PITCH_BOUND, -_ROOT_ROLL_PITCH_BOUND, 0.0, *joint_lower, *ctrl_lower],
        dtype=ctx.np.float64,
    )
    upper = ctx.np.asarray(
        [_ROOT_ROLL_PITCH_BOUND, _ROOT_ROLL_PITCH_BOUND, _PENETRATION_MAX, *joint_upper, *ctrl_upper],
        dtype=ctx.np.float64,
    )
    return lower, upper


def _dynamic_vector(ctx: _Context, qpos: Any, ctrl: Any, *, penetration: float) -> Any:
    roll, pitch, _ = _roll_pitch_yaw(qpos[3:7])
    joints = [float(qpos[int(address)]) for address in ctx.joint_qpos]
    ctrls = [float(ctrl[int(actuator_id)]) for actuator_id in ctx.actuator_ids]
    return ctx.np.asarray([roll, pitch, penetration, *joints, *ctrls], dtype=ctx.np.float64)


def _apply_dynamic_vector(ctx: _Context, template: Any, vector: Any) -> tuple[Any, Any]:
    joint_count = len(ctx.joint_qpos)
    qpos = ctx.np.asarray(template, dtype=ctx.np.float64).copy()
    qpos[0] = 0.0
    qpos[1] = 0.0
    qpos[3:7] = _quat_from_roll_pitch_yaw(float(vector[0]), float(vector[1]), 0.0)
    for address, value in zip(ctx.joint_qpos, vector[3 : 3 + joint_count]):
        qpos[int(address)] = float(value)
    _reset_to_qpos(ctx, qpos)
    corners = _footpad_corners(ctx)
    if corners:
        mean_height = float(ctx.np.mean([float(corner["height"]) for corner in corners]))
        ctx.data.qpos[2] += -float(vector[2]) - mean_height
    ctrl = ctx.np.zeros(ctx.model.nu, dtype=ctx.np.float64)
    ctrl_values = vector[3 + joint_count : 3 + joint_count + len(ctx.actuator_ids)]
    for actuator_id, value in zip(ctx.actuator_ids, ctrl_values):
        ctrl[int(actuator_id)] = float(value)
    ctx.data.ctrl[:] = ctrl
    ctx.mujoco.mj_forward(ctx.model, ctx.data)
    return ctx.np.asarray(ctx.data.qpos, dtype=ctx.np.float64).copy(), ctrl.copy()


def _dynamic_residual(ctx: _Context, vector: Any, start: Any, lower: Any, upper: Any) -> Any:
    qacc = ctx.np.asarray(ctx.data.qacc, dtype=ctx.np.float64)
    values = list(qacc[:6] / _STRICT_ROOT_QACC_NORM)
    values.extend(qacc[list(ctx.joint_dof)] / _STRICT_JOINT_QACC_MAX)
    patch = _flat_patch_report(ctx, penetration=float(vector[2]))
    values.extend(
        (float(corner["height"]) + float(vector[2])) / _DYNAMICS_HEIGHT_SCALE
        for corner in _footpad_corners(ctx)
    )
    support = _support_report(ctx)
    margin = float(support["support_margin"]["margin"])
    values.append(
        5.0 * max(0.0, _SUPPORT_MARGIN_TARGET - margin) / max(_SUPPORT_MARGIN_TARGET, 1e-12)
    )
    contact = _contact_report(ctx)
    min_load = _foot_load_threshold(ctx)
    values.extend(
        _LOAD_DEFICIT_WEIGHT
        * max(0.0, min_load - float(contact["normal_force_by_foot"].get(name, 0.0)))
        / max(min_load, 1e-12)
        for name in ctx.foot_geoms
    )
    if not patch["corner_count"]:
        values.append(100.0)
    span = ctx.np.maximum(1e-6, upper - lower)
    values.extend(_DYNAMICS_REGULARIZATION * (ctx.np.asarray(vector) - start) / span)
    return ctx.np.asarray(values, dtype=ctx.np.float64)


def _solve_dynamic(
    ctx: _Context,
    contact_entry: dict[str, Any],
    ctrl_start: dict[str, Any],
    *,
    joint_margin: float,
    ctrl_margin: float,
    max_nfev: int,
) -> dict[str, Any]:
    from scipy import optimize

    template = ctx.np.asarray(contact_entry["qpos"], dtype=ctx.np.float64)
    ctrl0 = ctx.np.asarray(ctrl_start["ctrl"], dtype=ctx.np.float64)
    lower, upper = _dynamic_bounds(ctx, joint_margin=joint_margin, ctrl_margin=ctrl_margin)
    raw_start = _dynamic_vector(ctx, template, ctrl0, penetration=float(contact_entry["penetration"]))
    vector0 = ctx.np.clip(raw_start, lower, upper)

    def residual(vector: Any) -> Any:
        _apply_dynamic_vector(ctx, template, vector)
        return _dynamic_residual(ctx, vector, vector0, lower, upper)

    result = optimize.least_squares(
        residual,
        vector0,
        bounds=(lower, upper),
        method="trf",
        max_nfev=max_nfev,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        x_scale="jac",
        diff_step=1e-5,
    )
    qpos, ctrl = _apply_dynamic_vector(ctx, template, result.x)
    snapshot = _snapshot(ctx, qpos, ctrl)
    patch = _flat_patch_report(ctx, penetration=float(result.x[2]))
    clearance = _continuous_clearance(ctx)
    support = _support_report(ctx)
    strict = _strict_actual_equilibrium(snapshot)
    return {
        "phase": "dynamics_actual_qacc_ctrl_refinement",
        "kinematic_start_name": contact_entry["kinematic_start_name"],
        "contact_entry_penetration": float(contact_entry["penetration"]),
        "qpos": [float(value) for value in qpos],
        "ctrl": [float(value) for value in ctrl],
        "vector": [float(value) for value in result.x],
        "solver": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "max_nfev": max_nfev,
        },
        "bounded_ctrl_subproblem": ctrl_start,
        "actual_snapshot": snapshot,
        "flat_patch": patch,
        "flat_geometry_realized": _flat_geometry_realized(patch),
        "continuous_clearance": clearance,
        "support": support,
        "strict_initial_actual_equilibrium": strict,
    }


def _ctrl_only_attempt(
    ctx: _Context,
    contact_entry: dict[str, Any],
    ctrl_start: dict[str, Any],
) -> dict[str, Any]:
    qpos = ctx.np.asarray(contact_entry["qpos"], dtype=ctx.np.float64)
    ctrl = ctx.np.asarray(ctrl_start["ctrl"], dtype=ctx.np.float64)
    snapshot = _snapshot(ctx, qpos, ctrl)
    patch = _flat_patch_report(ctx, penetration=float(contact_entry["penetration"]))
    clearance = _continuous_clearance(ctx)
    support = _support_report(ctx)
    return {
        "phase": "bounded_ctrl_only_actual_qacc",
        "kinematic_start_name": contact_entry["kinematic_start_name"],
        "contact_entry_penetration": float(contact_entry["penetration"]),
        "qpos": [float(value) for value in qpos],
        "ctrl": [float(value) for value in ctrl],
        "vector": [],
        "solver": {
            "success": bool(ctrl_start["solver"]["success"]),
            "status": int(ctrl_start["solver"]["status"]),
            "message": str(ctrl_start["solver"]["message"]),
            "nfev": 0,
            "cost": float(ctrl_start["solver"]["cost"]),
            "optimality": float(ctrl_start["solver"]["optimality"]),
            "max_nfev": 0,
        },
        "bounded_ctrl_subproblem": ctrl_start,
        "actual_snapshot": snapshot,
        "flat_patch": patch,
        "flat_geometry_realized": _flat_geometry_realized(patch),
        "continuous_clearance": clearance,
        "support": support,
        "strict_initial_actual_equilibrium": _strict_actual_equilibrium(snapshot),
    }


def _dynamic_sort_key(attempt: dict[str, Any]) -> tuple[float, float, float, float, float]:
    snapshot = attempt["actual_snapshot"]
    clearance = attempt["continuous_clearance"]
    min_self = clearance["minimum_self_pair_distance"]
    min_nonfoot = clearance["minimum_nonfoot_floor_distance"]
    return (
        0.0 if attempt["strict_initial_actual_equilibrium"] else 1.0,
        float(snapshot["root_qacc_norm"]) + float(snapshot["joint_qacc_max"]),
        max(
            float(attempt["flat_patch"]["height_error_to_penetration_max_abs"]),
            float(attempt["flat_patch"]["global_height_spread"]),
        ),
        max(0.0, -float(min_self)) if min_self is not None else 0.0,
        max(0.0, -float(min_nonfoot)) if min_nonfoot is not None else 0.0,
    )


def _joint_margin_report(ctx: _Context, qpos: Any, ctrl: Any) -> dict[str, Any]:
    rows = []
    joint_margins = []
    ctrl_margins = []
    ctrl_deltas = []
    for joint, qpos_address, actuator_id in zip(ctx.blueprint.joints, ctx.joint_qpos, ctx.actuator_ids):
        joint_id = int(ctx.mujoco.mj_name2id(ctx.model, ctx.mujoco.mjtObj.mjOBJ_JOINT, joint.name))
        lo, hi = (float(value) for value in ctx.model.jnt_range[joint_id])
        ctrl_lo, ctrl_hi = (float(value) for value in ctx.model.actuator_ctrlrange[int(actuator_id)])
        q = float(qpos[int(qpos_address)])
        c = float(ctrl[int(actuator_id)])
        joint_margin = min(q - lo, hi - q)
        ctrl_margin = min(c - ctrl_lo, ctrl_hi - c)
        delta = c - q
        joint_margins.append(joint_margin)
        ctrl_margins.append(ctrl_margin)
        ctrl_deltas.append(abs(delta))
        rows.append(
            {
                "semantic_slot": joint.semantic_slot,
                "joint_qpos": q,
                "actuator_ctrl": c,
                "ctrl_minus_qpos": delta,
                "joint_margin": joint_margin,
                "ctrl_margin": ctrl_margin,
                "joint_limits": [lo, hi],
                "ctrl_limits": [ctrl_lo, ctrl_hi],
            }
        )
    return {
        "min_joint_margin": min(joint_margins, default=float("inf")),
        "min_ctrl_margin": min(ctrl_margins, default=float("inf")),
        "max_abs_ctrl_minus_qpos": max(ctrl_deltas, default=0.0),
        "per_joint": rows,
    }


def _make_starts(ctx: _Context, start_qpos: Any, *, joint_margin: float) -> list[_QposStart]:
    starts = [_QposStart("geometric_stance", ctx.np.asarray(start_qpos, dtype=ctx.np.float64).copy())]
    upper_slots = ("left_arm_", "right_arm_", "waist_")
    for value in (-0.8, -0.4, 0.4, 0.8):
        qpos = ctx.np.asarray(start_qpos, dtype=ctx.np.float64).copy()
        for joint, address in zip(ctx.blueprint.joints, ctx.joint_qpos):
            if not joint.semantic_slot.startswith(upper_slots):
                continue
            joint_id = int(ctx.mujoco.mj_name2id(ctx.model, ctx.mujoco.mjtObj.mjOBJ_JOINT, joint.name))
            lo, hi = (float(v) for v in ctx.model.jnt_range[joint_id])
            target = -value if joint.semantic_slot.startswith("right_arm_") else value
            qpos[int(address)] = min(hi - joint_margin, max(lo + joint_margin, target))
        starts.append(_QposStart(f"upper_clearance_{value:g}", qpos))
    joint_lower, joint_upper = _joint_bounds(ctx, joint_margin)
    for fraction in (0.35, 0.50, 0.65):
        qpos = ctx.np.asarray(start_qpos, dtype=ctx.np.float64).copy()
        for index, address in enumerate(ctx.joint_qpos):
            qpos[int(address)] = float(
                joint_lower[index] + fraction * (joint_upper[index] - joint_lower[index])
            )
        starts.append(_QposStart(f"all_joint_fraction_{fraction:g}", qpos))
    return starts


def solve_actual_dynamics_stance(
    *,
    mujoco: Any,
    np: Any,
    model: Any,
    data: Any,
    blueprint: MorphologyBlueprint,
    physical: PhysicalParams | None,
    joint_qpos: tuple[int, ...],
    joint_dof: tuple[int, ...],
    actuator_ids: tuple[int, ...],
    foot_geoms: tuple[str, ...],
    start_qpos: Any,
    joint_margin: float = 0.05,
    ctrl_margin: float = 0.01,
    kinematic_max_nfev: int = 800,
    dynamics_max_nfev: int = 900,
    kinematic_branch_limit: int = 4,
    max_dynamic_attempts: int = 40,
    wide_search: bool = False,
) -> dict[str, Any]:
    """Return an actual-dynamics equilibrium stance candidate and diagnostics."""

    ctx = _Context(
        mujoco=mujoco,
        np=np,
        model=model,
        data=data,
        blueprint=blueprint,
        physical=physical,
        joint_qpos=joint_qpos,
        joint_dof=joint_dof,
        actuator_ids=actuator_ids,
        foot_geoms=tuple(sorted(foot_geoms)),
    )
    kinematic_attempts = [
        _solve_kinematic(ctx, start, joint_margin=joint_margin, max_nfev=kinematic_max_nfev)
        for start in _make_starts(ctx, start_qpos, joint_margin=joint_margin)
    ]
    good_kinematics = [
        attempt for attempt in sorted(kinematic_attempts, key=_kinematic_sort_key) if attempt["passed"]
    ]
    contact_entries: list[dict[str, Any]] = []
    dynamic_attempts: list[dict[str, Any]] = []
    schedule = _PENETRATION_SCHEDULE if wide_search else _FAST_PENETRATION_SCHEDULE
    for kinematic in good_kinematics[:kinematic_branch_limit]:
        entries = _contact_entry_steps(ctx, kinematic, penetration_schedule=schedule)
        entered = [entry for entry in entries if entry["entered_expected_mode"]]
        contact_entries.extend(entries)
        for entry in entered:
            ctrl_start = _bounded_ctrl_subproblem(ctx, ctx.np.asarray(entry["qpos"]), ctrl_margin=ctrl_margin)
            ctrl_only = _ctrl_only_attempt(ctx, entry, ctrl_start)
            dynamic_attempts.append(ctrl_only)
            if ctrl_only["strict_initial_actual_equilibrium"]:
                break
            attempt = _solve_dynamic(
                ctx,
                entry,
                ctrl_start,
                joint_margin=joint_margin,
                ctrl_margin=ctrl_margin,
                max_nfev=dynamics_max_nfev,
            )
            dynamic_attempts.append(attempt)
            if attempt["strict_initial_actual_equilibrium"]:
                break
            if len(dynamic_attempts) >= max_dynamic_attempts:
                break
        if dynamic_attempts and dynamic_attempts[-1]["strict_initial_actual_equilibrium"]:
            break
        if len(dynamic_attempts) >= max_dynamic_attempts:
            break
    best = min(dynamic_attempts, key=_dynamic_sort_key) if dynamic_attempts else None
    if best is None:
        raise RuntimeError("actual-dynamics stance solve found no double-foot dynamics attempts")
    qpos = ctx.np.asarray(best["qpos"], dtype=ctx.np.float64)
    ctrl = ctx.np.asarray(best["ctrl"], dtype=ctx.np.float64)
    margin_report = _joint_margin_report(ctx, qpos, ctrl)
    strict = bool(
        best["strict_initial_actual_equilibrium"]
        and float(margin_report["min_joint_margin"]) >= joint_margin - 1e-9
        and float(margin_report["min_ctrl_margin"]) >= ctrl_margin - 1e-9
    )
    return {
        "schema": "whole_body_actual_dynamics_stance_solve_v1",
        "contract": {
            "independent_of_r4a_artifacts": True,
            "root_x_y_yaw_gauge_fixed_zero": True,
            "joint_margin_min": joint_margin,
            "ctrl_margin_min": ctrl_margin,
            "final_truth": "actual_mujoco_efc_qacc_contact_loads",
            "contact_wrench_used_for_acceptance": False,
        },
        "qpos": [float(value) for value in qpos],
        "ctrl": [float(value) for value in ctrl],
        "root_pose_eq": [float(value) for value in qpos[:7]],
        "joint_margin_report": margin_report,
        "strict_initial_actual_equilibrium": strict,
        "best": best,
        "kinematic_phase": {
            "attempt_count": len(kinematic_attempts),
            "passed_attempt_count": len(good_kinematics),
            "attempt_preview": [
                {
                    "start_name": attempt["start_name"],
                    "passed": attempt["passed"],
                    "nfev": attempt["solver"]["nfev"],
                    "flat_height_spread": attempt["flat_patch"]["global_height_spread"],
                    "minimum_self_pair_distance": attempt["continuous_clearance"][
                        "minimum_self_pair_distance"
                    ],
                    "support_margin": attempt["support"]["support_margin"]["margin"],
                }
                for attempt in sorted(kinematic_attempts, key=_kinematic_sort_key)
            ],
        },
        "contact_entry_phase": {
            "steps_tested": len(contact_entries),
            "entered_expected_mode_count": sum(entry["entered_expected_mode"] for entry in contact_entries),
        },
        "dynamics_phase": {
            "attempt_count": len(dynamic_attempts),
            "best_preview": {
                "kinematic_start_name": best["kinematic_start_name"],
                "penetration": best["contact_entry_penetration"],
                "nfev": best["solver"]["nfev"],
                "optimality": best["solver"]["optimality"],
                "root_qacc_norm": best["actual_snapshot"]["root_qacc_norm"],
                "joint_qacc_max": best["actual_snapshot"]["joint_qacc_max"],
                "double_support": best["actual_snapshot"]["double_support"],
                "self_contacts": best["actual_snapshot"]["contact"]["self_contacts"],
                "forbidden_nonfoot_floor_contacts": best["actual_snapshot"]["contact"][
                    "forbidden_nonfoot_floor_contacts"
                ],
            },
        },
    }
