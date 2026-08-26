"""Diagnose procedural whole-body reset stance, static support and PD authority.

This tool is read-only with respect to the training contract: it never changes
the 45D action, the 193D observation, the mask, or the env/task interfaces.  It
only instruments an already-built ``WholeBodyMuJoCoShard`` and reports numbers.

Reported per seed:

* grounded reset base height and per-foot clearance
* initial ``ncon`` and number of feet within a contact band
* COM ground projection versus the convex hull of the candidate foot points
* joint-limit margin at the nominal pose
* actuator force at the nominal target versus the static gravity torque
  (``qfrc_bias``) that would be needed to hold the pose
* zero-action first-fall control step over a bounded horizon
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from h200_locomotion_lab.envs.whole_body_mujoco import (
    WholeBodyMuJoCoShard,
    WholeBodyMuJoCoShardConfig,
)
from h200_locomotion_lab.robots.procedural_morphology import MorphologyGenerator


def _geom_bottom(mujoco: Any, model: Any, data: Any, geom_id: int) -> float:
    size = model.geom_size[geom_id]
    geom_type = int(model.geom_type[geom_id])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        center = tuple(float(data.geom_xpos[geom_id, axis]) for axis in range(3))
        rot = tuple(float(data.geom_xmat[geom_id, index]) for index in range(9))
        half = tuple(float(size[axis]) for axis in range(3))
        return min(
            center[2] + rot[6] * (sx * half[0]) + rot[7] * (sy * half[1]) - rot[8] * half[2]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        extent = float(size[0] + size[1])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        extent = float(size[1])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        extent = float(size[0])
    else:
        extent = float(max(size))
    return float(data.geom_xpos[geom_id, 2]) - extent


def _geom_support_points(
    mujoco: Any,
    model: Any,
    data: Any,
    geom_id: int,
    *,
    floor_corners_only: bool = False,
    floor_corner_band: float = 0.005,
) -> list[tuple[float, float]]:
    """Return XY points for a support geom's actual footprint.

    Capsule tips are still point contacts, but R1 terminal footpads are boxes;
    their support polygon must use the bottom face corners, not the geom center.
    """

    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        return [(float(data.geom_xpos[geom_id, 0]), float(data.geom_xpos[geom_id, 1]))]

    center = tuple(float(data.geom_xpos[geom_id, axis]) for axis in range(3))
    rot = tuple(float(data.geom_xmat[geom_id, index]) for index in range(9))
    half = tuple(float(model.geom_size[geom_id, axis]) for axis in range(3))
    corners: list[tuple[float, float, float]] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            local = (sx * half[0], sy * half[1], -half[2])
            corners.append(
                (
                    center[0] + rot[0] * local[0] + rot[1] * local[1] + rot[2] * local[2],
                    center[1] + rot[3] * local[0] + rot[4] * local[1] + rot[5] * local[2],
                    center[2] + rot[6] * local[0] + rot[7] * local[1] + rot[8] * local[2],
                )
            )
    if floor_corners_only:
        floor_z = min(point[2] for point in corners)
        corners = [point for point in corners if point[2] - floor_z <= floor_corner_band]
    return [(float(point[0]), float(point[1])) for point in corners]


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _support_margin(com_xy: tuple[float, float], points: list[tuple[float, float]]) -> dict[str, Any]:
    """Signed distance from the COM projection to the support-polygon boundary.

    Positive means inside.  A degenerate hull (0, 1 or 2 distinct points) has no
    interior at all, so the best achievable margin is <= 0 by construction; that
    is reported explicitly instead of being smoothed into a small number.
    """

    hull = _convex_hull(points)
    if len(hull) < 3:
        if len(hull) == 2:
            (x1, y1), (x2, y2) = hull
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length <= 1e-12:
                dist = math.hypot(com_xy[0] - x1, com_xy[1] - y1)
            else:
                t = max(0.0, min(1.0, ((com_xy[0] - x1) * dx + (com_xy[1] - y1) * dy) / (length * length)))
                dist = math.hypot(com_xy[0] - (x1 + t * dx), com_xy[1] - (y1 + t * dy))
        elif len(hull) == 1:
            dist = math.hypot(com_xy[0] - hull[0][0], com_xy[1] - hull[0][1])
        else:
            dist = float("nan")
        return {
            "degenerate": True,
            "hull_vertices": len(hull),
            "hull_area": 0.0,
            "margin": -dist,
            "inside": False,
        }
    area = 0.0
    for index in range(len(hull)):
        x1, y1 = hull[index]
        x2, y2 = hull[(index + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    area = abs(area) * 0.5
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
        # Hull is counter-clockwise, so interior points give negative `signed`.
        if signed > 0.0:
            inside = False
        margin = min(margin, -signed)
    return {
        "degenerate": False,
        "hull_vertices": len(hull),
        "hull_area": area,
        "margin": margin,
        "inside": inside,
    }


def diagnose_seed(
    family: str,
    seed: int,
    *,
    range_fraction: float,
    horizon_steps: int,
    contact_band: float = 0.02,
) -> dict[str, Any]:
    import mujoco  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    generator = MorphologyGenerator()
    blueprint = generator.generate(family, seed)
    physical = generator.sample_physical_params(
        blueprint, seed + 10_000_000, range_fraction=range_fraction
    )
    config = WholeBodyMuJoCoShardConfig(seed=seed)
    shard = WholeBodyMuJoCoShard(blueprint, physical=physical, num_envs=1, config=config)
    model, data = shard.model, shard.data[0]

    # --- reset determinism -------------------------------------------------
    qpos_a = np.array(data.qpos, copy=True)
    qvel_a = np.array(data.qvel, copy=True)
    ncon_a = int(data.ncon)
    shard.reset()
    qpos_b = np.array(data.qpos, copy=True)
    qvel_b = np.array(data.qvel, copy=True)
    reset_repeatable = {
        "max_qpos_delta": float(np.max(np.abs(qpos_a - qpos_b))),
        "max_qvel_delta": float(np.max(np.abs(qvel_a - qvel_b))),
        "ncon_before": ncon_a,
        "ncon_after": int(data.ncon),
    }

    # --- reset stance geometry --------------------------------------------
    mujoco.mj_forward(model, data)
    base_height = float(data.qpos[2])
    foot_geom_ids = [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) for name in sorted(shard._foot_geoms)
    ]
    foot_bottoms = [_geom_bottom(mujoco, model, data, gid) for gid in foot_geom_ids]
    foot_point_groups = [
        _geom_support_points(mujoco, model, data, gid) for gid in foot_geom_ids
    ]
    foot_points = [point for group in foot_point_groups for point in group]
    site_heights = []
    for site_name in blueprint.end_sites:
        site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name))
        if site_id >= 0:
            site_heights.append(float(data.site_xpos[site_id, 2]))
    lowest = min(foot_bottoms) if foot_bottoms else float("nan")
    feet_in_band = sum(1 for value in foot_bottoms if value - lowest <= contact_band)
    feet_near_floor = sum(1 for value in foot_bottoms if value <= contact_band)
    # Reset-time contact state, measured before any control step is applied.
    floor_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    reset_ncon = int(data.ncon)
    reset_foot_contacts = 0
    for index in range(reset_ncon):
        contact = data.contact[index]
        if floor_id not in (int(contact.geom1), int(contact.geom2)):
            continue
        other = int(contact.geom2) if int(contact.geom1) == floor_id else int(contact.geom1)
        if other in foot_geom_ids:
            reset_foot_contacts += 1

    # --- COM vs support polygon -------------------------------------------
    total_mass = float(np.sum(model.body_mass))
    com = np.zeros(3)
    for body_id in range(model.nbody):
        com += float(model.body_mass[body_id]) * np.asarray(data.xipos[body_id])
    com /= max(1e-12, total_mass)
    # Only feet that are actually within the contact band can carry load.
    loaded = [
        point
        for gid, bottom in zip(foot_geom_ids, foot_bottoms)
        if bottom <= contact_band
        for point in _geom_support_points(
            mujoco,
            model,
            data,
            gid,
            floor_corners_only=True,
        )
    ]
    support_all = _support_margin((float(com[0]), float(com[1])), foot_points)
    support_loaded = _support_margin((float(com[0]), float(com[1])), loaded)

    # --- joint-limit margin at nominal ------------------------------------
    limit_margins = []
    for joint, address in zip(blueprint.joints, shard._joint_qpos):
        lower, upper = shard._joint_limits(joint)
        value = float(data.qpos[address])
        limit_margins.append(min(value - lower, upper - value))

    # --- PD authority at the nominal target -------------------------------
    # Emulate exactly what the env does on the first zero-action control step.
    shard._set_targets(data, tuple(0.0 for _ in blueprint.joints))
    mujoco.mj_forward(model, data)
    actuator_force = [float(data.actuator_force[aid]) for aid in shard._actuator_ids]
    force_limits = [float(model.actuator_forcerange[aid, 1]) for aid in shard._actuator_ids]
    # qfrc_bias at zero velocity is the generalized gravity torque; holding the
    # pose statically requires the actuators to supply exactly this.
    required = [float(data.qfrc_bias[dof]) for dof in shard._joint_dof]
    ctrl_error = [
        float(data.ctrl[aid]) - float(data.qpos[adr])
        for aid, adr in zip(shard._actuator_ids, shard._joint_qpos)
    ]
    deficit = [abs(r) - abs(f) for r, f in zip(required, actuator_force)]
    saturated = sum(1 for f, lim in zip(actuator_force, force_limits) if abs(f) >= 0.995 * lim)
    over_limit = sum(1 for r, lim in zip(required, force_limits) if abs(r) > lim)

    # --- zero-action rollout ----------------------------------------------
    shard.reset()
    zero_action = np.zeros((1, 45), dtype=np.float64)
    first_fall = None
    heights: list[float] = []
    tilts: list[float] = []
    foot_contacts: list[int] = []
    nan_seen = False
    min_contact_distance = float("inf")
    for step_index in range(horizon_steps):
        result = shard.step(zero_action)
        heights.append(float(data.qpos[2]))
        tilts.append(float(result.metrics["tilt"][0]))
        contacts = 0
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            if floor_id not in (int(contact.geom1), int(contact.geom2)):
                continue
            other = int(contact.geom2) if int(contact.geom1) == floor_id else int(contact.geom1)
            min_contact_distance = min(min_contact_distance, float(contact.dist))
            if other in foot_geom_ids:
                contacts += 1
        foot_contacts.append(contacts)
        if not all(math.isfinite(float(v)) for v in data.qpos):
            nan_seen = True
        if bool(result.trial_done[0]) and first_fall is None:
            first_fall = step_index + 1
            break

    return {
        "family": family,
        "seed": seed,
        "structural_hash": blueprint.structural_hash,
        "range_fraction": range_fraction,
        "has_arms": blueprint.has_arms,
        "num_joints": len(blueprint.joints),
        "num_feet": len(foot_geom_ids),
        "global_scale": physical.global_scale,
        "total_mass": total_mass,
        "reset": reset_repeatable,
        "base_height": base_height,
        "nominal_height_config": blueprint.nominal_height,
        "foot_bottom_heights": foot_bottoms,
        "foot_site_heights": site_heights,
        "foot_height_spread": (max(foot_bottoms) - min(foot_bottoms)) if foot_bottoms else float("nan"),
        "feet_within_band_of_lowest": feet_in_band,
        "feet_near_floor": feet_near_floor,
        "initial_ncon": reset_ncon,
        "initial_foot_contacts": reset_foot_contacts,
        "final_ncon": int(data.ncon),
        "com": [float(v) for v in com],
        "support_all_feet": support_all,
        "support_loaded_feet": support_loaded,
        "min_joint_limit_margin": min(limit_margins) if limit_margins else float("nan"),
        "pd": {
            "max_abs_ctrl_error": max(abs(v) for v in ctrl_error),
            "max_abs_actuator_force": max(abs(v) for v in actuator_force),
            "max_abs_required_static_torque": max(abs(v) for v in required),
            "max_torque_deficit": max(deficit),
            "saturated_actuators": saturated,
            "actuators_over_force_limit": over_limit,
            "force_limit_min": min(force_limits),
            "kp_min": min(float(model.actuator_gainprm[aid, 0]) for aid in shard._actuator_ids),
            "required_sag_rad_max": max(
                abs(r) / max(1e-9, float(model.actuator_gainprm[aid, 0]))
                for r, aid in zip(required, shard._actuator_ids)
            ),
        },
        "zero_action": {
            "first_fall_step": first_fall,
            "survived_steps": len(heights),
            "survived": first_fall is None,
            "final_base_height": heights[-1] if heights else float("nan"),
            "min_base_height": min(heights) if heights else float("nan"),
            "max_tilt_rad": max(tilts) if tilts else float("nan"),
            "mean_foot_contacts": statistics.fmean(foot_contacts) if foot_contacts else 0.0,
            "max_foot_contacts": max(foot_contacts) if foot_contacts else 0,
            "min_contact_distance": None if min_contact_distance == float("inf") else min_contact_distance,
            "nan_seen": nan_seen,
        },
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for family in sorted({record["family"] for record in records}):
        subset = [record for record in records if record["family"] == family]
        falls = [r for r in subset if not r["zero_action"]["survived"]]
        steps = [r["zero_action"]["first_fall_step"] for r in falls]
        summary[family] = {
            "seeds": len(subset),
            "zero_action_falls": len(falls),
            "zero_action_fall_ratio": len(falls) / len(subset),
            "first_fall_step_min": min(steps) if steps else None,
            "first_fall_step_median": statistics.median(steps) if steps else None,
            "first_fall_step_max": max(steps) if steps else None,
            "base_height_median": statistics.median(r["base_height"] for r in subset),
            "initial_ncon_median": statistics.median(r["initial_ncon"] for r in subset),
            "initial_foot_contacts_median": statistics.median(
                r["initial_foot_contacts"] for r in subset
            ),
            "initial_foot_contacts_max": max(r["initial_foot_contacts"] for r in subset),
            "total_mass_median": statistics.median(r["total_mass"] for r in subset),
            "required_sag_rad_median": statistics.median(
                r["pd"]["required_sag_rad_max"] for r in subset
            ),
            "feet_near_floor_median": statistics.median(r["feet_near_floor"] for r in subset),
            "feet_near_floor_min": min(r["feet_near_floor"] for r in subset),
            "foot_height_spread_median": statistics.median(r["foot_height_spread"] for r in subset),
            "foot_height_spread_max": max(r["foot_height_spread"] for r in subset),
            "degenerate_support_all_feet": sum(1 for r in subset if r["support_all_feet"]["degenerate"]),
            "hull_area_median_all_feet": statistics.median(
                r["support_all_feet"]["hull_area"] for r in subset
            ),
            "support_margin_median_all_feet": statistics.median(
                r["support_all_feet"]["margin"] for r in subset
            ),
            "com_inside_support_all_feet": sum(1 for r in subset if r["support_all_feet"]["inside"]),
            "com_inside_support_loaded_feet": sum(
                1 for r in subset if r["support_loaded_feet"]["inside"]
            ),
            "min_joint_limit_margin_min": min(r["min_joint_limit_margin"] for r in subset),
            "max_abs_ctrl_error_max": max(r["pd"]["max_abs_ctrl_error"] for r in subset),
            "max_abs_actuator_force_max": max(r["pd"]["max_abs_actuator_force"] for r in subset),
            "required_static_torque_median": statistics.median(
                r["pd"]["max_abs_required_static_torque"] for r in subset
            ),
            "required_static_torque_max": max(
                r["pd"]["max_abs_required_static_torque"] for r in subset
            ),
            "torque_deficit_median": statistics.median(r["pd"]["max_torque_deficit"] for r in subset),
            "seeds_with_actuator_over_force_limit": sum(
                1 for r in subset if r["pd"]["actuators_over_force_limit"] > 0
            ),
            "reset_max_qpos_delta": max(r["reset"]["max_qpos_delta"] for r in subset),
            "reset_max_qvel_delta": max(r["reset"]["max_qvel_delta"] for r in subset),
            "nan_seeds": sum(1 for r in subset if r["zero_action"]["nan_seen"]),
        }
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", default=["biped", "quadruped"])
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--range-fraction", type=float, default=0.5)
    parser.add_argument("--horizon-steps", type=int, default=100)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    records = [
        diagnose_seed(
            family,
            args.seed_offset + index,
            range_fraction=args.range_fraction,
            horizon_steps=args.horizon_steps,
        )
        for family in args.families
        for index in range(args.seeds)
    ]
    payload = {
        "schema": "whole_body_stance_diagnosis_v1",
        "range_fraction": args.range_fraction,
        "horizon_steps": args.horizon_steps,
        "seeds_per_family": args.seeds,
        "summary": summarize(records),
        "records": records,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
