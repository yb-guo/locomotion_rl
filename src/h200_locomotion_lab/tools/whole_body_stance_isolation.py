"""Isolate why a procedural biped cannot hold a stance, on one fixed topology.

The ladder below changes exactly one thing per rung so the failure can be
attributed.  Nothing here touches the 45D action, the 193D observation, the
mask or the env/task interfaces: each rung builds a private MuJoCo model from
the same blueprint and drives it directly.

Rungs
-----
``nominal_zero_action``      current grounded nominal pose, zero action
``nominal_position_target``  same, target held exactly at the nominal pose
``double_support``           reset pose solved so *both* feet reach the floor
``double_support_contact``   same, grounding margin 0 so contact exists at t=0
``high_pd``                  double-support contact + 10x kp / 5x kv
``gravity_compensation``     double-support contact + joint-space tau_ff = qfrc_bias
``ideal_joint_controller``   high PD *and* gravity compensation (best joint-space case)
``box_feet``                 double-support contact + finite-area feet, current PD
``box_feet_ideal``           finite-area feet + ideal joint controller
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from h200_locomotion_lab.envs.whole_body_mujoco import (
    WholeBodyMuJoCoShardConfig,
    ground_nominal_pose,
)
from h200_locomotion_lab.robots.procedural_morphology import (
    MorphologyGenerator,
    compile_mjcf,
)
from h200_locomotion_lab.tools.whole_body_stance_diagnosis import (
    _geom_bottom,
    _support_margin,
)


def find_simplest_biped(
    generator: MorphologyGenerator, *, search: int = 400
) -> tuple[int, Any]:
    """Pick the lowest-seed arm-free biped with the fewest and most symmetric legs.

    The current grammar always emits three waist joints, so "no waist" is not
    reachable; the tie-break instead prefers left/right legs with the same joint
    count so leg asymmetry is not confounded into this rung.
    """

    best: tuple[tuple[int, int], int, Any] | None = None
    for seed in range(search):
        blueprint = generator.generate("biped", seed)
        if blueprint.has_arms:
            continue
        left = sum(1 for j in blueprint.joints if j.child_link.startswith("left_leg"))
        right = sum(1 for j in blueprint.joints if j.child_link.startswith("right_leg"))
        key = (abs(left - right), len(blueprint.joints))
        if best is None or key < best[0]:
            best = (key, seed, blueprint)
        if key == (0, 7):
            break
    if best is None:
        raise RuntimeError("no arm-free biped topology found in the search range")
    return best[1], best[2]


def _with_box_feet(xml_text: str, blueprint: Any, physical: Any) -> str:
    """Attach a finite-area box foot to every terminal leg link.

    Only the contact geometry changes; the kinematic chain, joints, actuators
    and masses stay identical, so a difference against the capsule-tip variant
    isolates the missing support area.
    """

    root = ET.fromstring(xml_text)
    terminal = {link.name: link for link in blueprint.links if link.end_site}
    scale = physical.global_scale if physical is not None else 1.0
    for body in root.iter("body"):
        link = terminal.get(body.get("name", ""))
        if link is None:
            continue
        length = link.length * scale * (physical.link_scales.get(link.name, 1.0) if physical else 1.0)
        radius = link.size[0] * scale
        # Disable the capsule tip contact so the box is the only ground contact.
        for geom in body.findall("geom"):
            if geom.get("name") == f"{link.name}_geom":
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{link.name}_boxfoot",
                "type": "box",
                "size": f"{0.09 * scale:.6g} {0.05 * scale:.6g} {radius:.6g}",
                "pos": f"0 0 {-(length + radius):.6g}",
                "mass": "0.35",
                "contype": "1",
                "conaffinity": "1",
                "friction": geom.get("friction", "0.8 0.1 0.1"),
            },
        )
    return ET.tostring(root, encoding="unicode")


def _scale_pd(xml_text: str, kp_scale: float, kv_scale: float) -> str:
    root = ET.fromstring(xml_text)
    for actuator in root.iter("position"):
        kp = float(actuator.get("kp", "30")) * kp_scale
        kv = float(actuator.get("kv", "1")) * kv_scale
        actuator.set("kp", f"{kp:.6g}")
        actuator.set("kv", f"{kv:.6g}")
        low, high = (float(value) for value in actuator.get("forcerange", "-80 80").split())
        span = max(abs(low), abs(high)) * max(1.0, kp_scale)
        actuator.set("forcerange", f"{-span:.6g} {span:.6g}")
    return ET.tostring(root, encoding="unicode")


class _Rig:
    """A private model/data pair plus the index bookkeeping the rungs need."""

    def __init__(self, xml_text: str, blueprint: Any, physical: Any) -> None:
        import mujoco  # type: ignore[import-not-found]

        self.mujoco = mujoco
        self.blueprint = blueprint
        self.physical = physical
        self.model = mujoco.MjModel.from_xml_string(xml_text)
        self.data = mujoco.MjData(self.model)
        self.qpos_adr = [
            int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j.name)])
            for j in blueprint.joints
        ]
        self.dof_adr = [
            int(self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j.name)])
            for j in blueprint.joints
        ]
        self.act_ids = [
            int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a.name))
            for a in blueprint.actuators
        ]
        self.floor_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
        self.contact_geoms: dict[str, list[int]] = {}
        for link in blueprint.links:
            if not link.end_site:
                continue
            ids = []
            for suffix in ("_geom", "_boxfoot"):
                gid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{link.name}{suffix}"))
                if gid >= 0 and (
                    int(self.model.geom_contype[gid]) or int(self.model.geom_conaffinity[gid])
                ):
                    ids.append(gid)
            self.contact_geoms[link.name] = ids
        self.foot_geom_ids = [gid for ids in self.contact_geoms.values() for gid in ids]

    def joint_limits(self, joint: Any) -> tuple[float, float]:
        scale = self.physical.joint_limit_scales.get(joint.semantic_slot, 1.0) if self.physical else 1.0
        offset = self.physical.nominal_offsets.get(joint.semantic_slot, 0.0) if self.physical else 0.0
        lower, upper = joint.joint_range
        return lower * scale + offset, upper * scale + offset

    def set_nominal_qpos(self) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = self.blueprint.nominal_height * (
            self.physical.global_scale if self.physical else 1.0
        )
        for joint, adr in zip(self.blueprint.joints, self.qpos_adr):
            offset = self.physical.nominal_offsets.get(joint.semantic_slot, 0.0) if self.physical else 0.0
            lower, upper = self.joint_limits(joint)
            self.data.qpos[adr] = min(upper, max(lower, joint.nominal + offset))
        self.mujoco.mj_forward(self.model, self.data)

    def foot_bottom(self, link_name: str) -> float:
        ids = self.contact_geoms[link_name]
        return min(_geom_bottom(self.mujoco, self.model, self.data, gid) for gid in ids)

    def knee_index(self, link_name: str) -> int | None:
        prefix = link_name.split("_")[0]
        for index, joint in enumerate(self.blueprint.joints):
            if joint.child_link.startswith(prefix) and joint.semantic_slot.endswith("knee_pitch"):
                return index
        return None


def solve_double_support_pose(rig: _Rig, *, margin: float, bend_fraction: float = 0.92) -> dict[str, Any]:
    """Solve a base height and per-leg knee angle so every foot reaches the floor.

    Foot height is monotone in knee flexion, so one scalar bisection per leg is
    enough.  The base height is chosen from the *shortest* leg's reach, which is
    what the current fixed ``nominal_height`` ignores.
    """

    rig.set_nominal_qpos()
    legs = sorted(rig.contact_geoms)
    knees = {leg: rig.knee_index(leg) for leg in legs}
    if any(index is None for index in knees.values()):
        raise RuntimeError("every leg must expose a knee_pitch joint")

    def foot_height(leg: str, knee_angle: float, base_height: float) -> float:
        rig.data.qpos[2] = base_height
        rig.data.qpos[rig.qpos_adr[knees[leg]]] = knee_angle
        rig.mujoco.mj_forward(rig.model, rig.data)
        return rig.foot_bottom(leg)

    reach = {}
    for leg in legs:
        lower, _ = rig.joint_limits(rig.blueprint.joints[knees[leg]])
        reach[leg] = -foot_height(leg, lower, 0.0)
        rig.data.qpos[rig.qpos_adr[knees[leg]]] = rig.blueprint.joints[knees[leg]].nominal
    base_height = bend_fraction * min(reach.values())

    solved = {}
    for leg in legs:
        lower, upper = rig.joint_limits(rig.blueprint.joints[knees[leg]])
        low, high = lower, upper
        for _ in range(60):
            mid = 0.5 * (low + high)
            if foot_height(leg, mid, base_height) + margin > 0.0:
                high = mid
            else:
                low = mid
        solved[leg] = 0.5 * (low + high)
        rig.data.qpos[rig.qpos_adr[knees[leg]]] = solved[leg]
    rig.data.qpos[2] = base_height + margin
    rig.mujoco.mj_forward(rig.model, rig.data)
    return {
        "base_height": float(rig.data.qpos[2]),
        "leg_reach": {leg: float(value) for leg, value in reach.items()},
        "knee_angles": {leg: float(value) for leg, value in solved.items()},
    }


def solve_centered_stance_pose(
    rig: _Rig, *, margin: float, bend_fraction: float = 0.88, iterations: int = 8
) -> dict[str, Any]:
    """Solve knee *and* hip pitch so every foot both reaches the floor and sits
    under its own hip.

    ``solve_double_support_pose`` only fixes height, which leaves the knee bend
    pushing each foot backwards in x with nothing to compensate it; the COM then
    projects outside the foot points no matter how stiff the controller is.  One
    extra scalar bisection per leg on the proximal hip pitch removes that x
    offset, so this rung isolates "COM over support" from "foot has area".
    """

    rig.set_nominal_qpos()
    legs = sorted(rig.contact_geoms)
    knees = {leg: rig.knee_index(leg) for leg in legs}
    hips: dict[str, int | None] = {}
    for leg in legs:
        prefix = leg.split("_")[0]
        hips[leg] = None
        for index, joint in enumerate(rig.blueprint.joints):
            if joint.child_link.startswith(prefix) and joint.semantic_slot.endswith("hip_pitch"):
                hips[leg] = index
                break
    if any(value is None for value in (*knees.values(), *hips.values())):
        raise RuntimeError("every leg must expose a hip_pitch and a knee_pitch joint")

    def forward() -> None:
        rig.mujoco.mj_forward(rig.model, rig.data)

    def foot_xz(leg: str) -> tuple[float, float]:
        gid = rig.contact_geoms[leg][0]
        return float(rig.data.geom_xpos[gid, 0]), rig.foot_bottom(leg)

    reach = {}
    for leg in legs:
        lower, _ = rig.joint_limits(rig.blueprint.joints[knees[leg]])
        rig.data.qpos[2] = 0.0
        rig.data.qpos[rig.qpos_adr[knees[leg]]] = lower
        forward()
        reach[leg] = -foot_xz(leg)[1]
        rig.data.qpos[rig.qpos_adr[knees[leg]]] = rig.blueprint.joints[knees[leg]].nominal
    base_height = bend_fraction * min(reach.values())
    rig.data.qpos[2] = base_height
    forward()

    def bisect(index: int, probe, target: float) -> float:
        lower, upper = rig.joint_limits(rig.blueprint.joints[index])
        adr = rig.qpos_adr[index]
        rig.data.qpos[adr] = lower
        forward()
        low_value = probe()
        rig.data.qpos[adr] = upper
        forward()
        high_value = probe()
        if (low_value - target) * (high_value - target) > 0.0:
            # Target unreachable inside the limits: keep the closer endpoint.
            best = lower if abs(low_value - target) < abs(high_value - target) else upper
            rig.data.qpos[adr] = best
            forward()
            return best
        increasing = high_value > low_value
        low, high = lower, upper
        for _ in range(50):
            mid = 0.5 * (low + high)
            rig.data.qpos[adr] = mid
            forward()
            value = probe()
            if (value > target) == increasing:
                high = mid
            else:
                low = mid
        result = 0.5 * (low + high)
        rig.data.qpos[adr] = result
        forward()
        return result

    # Coordinate-wise alternation can walk both joints to their limits, so scan
    # the hip on a grid and bisect the knee for height at each sample, keeping
    # the hip that best zeroes the foot's x offset.  Two scalar dofs per leg is
    # small enough that this is exact enough and cannot diverge.
    for leg in legs:
        hip_lower, hip_upper = rig.joint_limits(rig.blueprint.joints[hips[leg]])
        best: tuple[float, float, float] | None = None
        samples = 41
        for step in range(samples):
            hip_angle = hip_lower + (hip_upper - hip_lower) * step / (samples - 1)
            rig.data.qpos[rig.qpos_adr[hips[leg]]] = hip_angle
            forward()
            knee_angle = bisect(knees[leg], lambda leg=leg: foot_xz(leg)[1], -margin)
            x_offset, z_offset = foot_xz(leg)
            if abs(z_offset + margin) > 1e-3:
                continue
            score = abs(x_offset)
            if best is None or score < best[0]:
                best = (score, hip_angle, knee_angle)
        if best is None:
            raise RuntimeError(f"no floor-reaching hip/knee pair found for {leg}")
        rig.data.qpos[rig.qpos_adr[hips[leg]]] = best[1]
        rig.data.qpos[rig.qpos_adr[knees[leg]]] = best[2]
        forward()
    rig.data.qpos[2] = base_height + margin
    forward()
    return {
        "base_height": float(rig.data.qpos[2]),
        "leg_reach": {leg: float(value) for leg, value in reach.items()},
        "hip_angles": {leg: float(rig.data.qpos[rig.qpos_adr[hips[leg]]]) for leg in legs},
        "knee_angles": {leg: float(rig.data.qpos[rig.qpos_adr[knees[leg]]]) for leg in legs},
        "foot_x": {leg: float(foot_xz(leg)[0]) for leg in legs},
    }


def solve_flat_stance_pose(
    rig: _Rig, *, margin: float, bend_fraction: float = 0.88
) -> dict[str, Any]:
    """Solve hip pitch, knee and ankle pitch for a flat, centered double stance.

    This is the strongest stance a joint-space solve can produce for the current
    grammar: every foot is level on the floor, at the floor, and directly under
    its hip.  A rung that still falls here cannot be blamed on the reset pose.
    """

    import numpy as np  # type: ignore[import-not-found]

    rig.set_nominal_qpos()
    legs = sorted(rig.contact_geoms)

    def slot_index(leg: str, suffix: str) -> int | None:
        prefix = leg.split("_")[0]
        for index, joint in enumerate(rig.blueprint.joints):
            if joint.child_link.startswith(prefix) and joint.semantic_slot.endswith(suffix):
                return index
        return None

    hips = {leg: slot_index(leg, "hip_pitch") for leg in legs}
    knees = {leg: slot_index(leg, "knee_pitch") for leg in legs}
    ankles = {leg: slot_index(leg, "ankle_pitch") for leg in legs}
    if any(value is None for value in (*hips.values(), *knees.values())):
        raise RuntimeError("every leg must expose a hip_pitch and a knee_pitch joint")

    def forward() -> None:
        rig.mujoco.mj_forward(rig.model, rig.data)

    def corners(leg: str) -> Any:
        gid = rig.contact_geoms[leg][0]
        centre = np.asarray(rig.data.geom_xpos[gid])
        rot = np.asarray(rig.data.geom_xmat[gid]).reshape(3, 3)
        size = np.asarray(rig.model.geom_size[gid])
        if int(rig.model.geom_type[gid]) == int(rig.mujoco.mjtGeom.mjGEOM_BOX):
            half = size[:3]
        else:
            half = np.array([size[1], 0.0, size[0]])
        out = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                out.append(centre + rot @ np.array([sx * half[0], sy * half[1], -half[2]]))
        return np.asarray(out)

    def bottom(leg: str) -> float:
        return float(np.min(corners(leg)[:, 2]))

    def tilt(leg: str) -> float:
        pts = corners(leg)
        front = float(np.mean(pts[2:, 2]))
        rear = float(np.mean(pts[:2, 2]))
        return front - rear

    def foot_x(leg: str) -> float:
        return float(np.mean(corners(leg)[:, 0]))

    def bisect(index: int, probe, target: float) -> float:
        lower, upper = rig.joint_limits(rig.blueprint.joints[index])
        adr = rig.qpos_adr[index]
        rig.data.qpos[adr] = lower
        forward()
        low_value = probe()
        rig.data.qpos[adr] = upper
        forward()
        high_value = probe()
        if (low_value - target) * (high_value - target) > 0.0:
            best = lower if abs(low_value - target) < abs(high_value - target) else upper
            rig.data.qpos[adr] = best
            forward()
            return best
        increasing = high_value > low_value
        low, high = lower, upper
        for _ in range(40):
            mid = 0.5 * (low + high)
            rig.data.qpos[adr] = mid
            forward()
            if (probe() > target) == increasing:
                high = mid
            else:
                low = mid
        result = 0.5 * (low + high)
        rig.data.qpos[adr] = result
        forward()
        return result

    reach = {}
    for leg in legs:
        lower, _ = rig.joint_limits(rig.blueprint.joints[knees[leg]])
        rig.data.qpos[2] = 0.0
        rig.data.qpos[rig.qpos_adr[knees[leg]]] = lower
        forward()
        reach[leg] = -bottom(leg)
        rig.data.qpos[rig.qpos_adr[knees[leg]]] = rig.blueprint.joints[knees[leg]].nominal
    base_height = bend_fraction * min(reach.values())
    rig.data.qpos[2] = base_height + margin
    forward()

    def settle(leg: str) -> None:
        for _ in range(5):
            if ankles[leg] is not None:
                bisect(ankles[leg], lambda leg=leg: tilt(leg), 0.0)
            bisect(knees[leg], lambda leg=leg: bottom(leg), 0.0)

    solved: dict[str, dict[str, float]] = {}
    for leg in legs:
        hip_lower, hip_upper = rig.joint_limits(rig.blueprint.joints[hips[leg]])
        best: tuple[float, dict[str, float]] | None = None
        samples = 25
        for step in range(samples):
            hip_angle = hip_lower + (hip_upper - hip_lower) * step / (samples - 1)
            rig.data.qpos[rig.qpos_adr[hips[leg]]] = hip_angle
            forward()
            settle(leg)
            if abs(bottom(leg)) > 1e-3 or abs(tilt(leg)) > 5e-3:
                continue
            score = abs(foot_x(leg))
            angles = {
                "hip_pitch": hip_angle,
                "knee_pitch": float(rig.data.qpos[rig.qpos_adr[knees[leg]]]),
            }
            if ankles[leg] is not None:
                angles["ankle_pitch"] = float(rig.data.qpos[rig.qpos_adr[ankles[leg]]])
            if best is None or score < best[0]:
                best = (score, angles)
        if best is None:
            raise RuntimeError(f"no flat floor-reaching pose found for {leg}")
        solved[leg] = best[1]
        for suffix, value in best[1].items():
            index = {"hip_pitch": hips, "knee_pitch": knees, "ankle_pitch": ankles}[suffix][leg]
            rig.data.qpos[rig.qpos_adr[index]] = value
        forward()
    rig.data.qpos[2] = base_height + margin
    forward()
    return {
        "base_height": float(rig.data.qpos[2]),
        "leg_reach": {leg: float(value) for leg, value in reach.items()},
        "angles": solved,
        "foot_x": {leg: float(foot_x(leg)) for leg in legs},
        "foot_tilt": {leg: float(tilt(leg)) for leg in legs},
        "foot_bottom": {leg: float(bottom(leg)) for leg in legs},
    }


def solve_optimized_stance_pose(
    rig: _Rig,
    *,
    margin: float,
    restarts: int = 6,
    sweeps: int = 120,
) -> dict[str, Any]:
    """Search *all* leg joints plus base height for a statically balanced stance.

    This answers the first-principles question directly: if a projected
    coordinate descent over every leg dof cannot put the COM ground projection
    inside the foot support polygon while both feet rest flat on the floor, then
    no reset pose exists for this morphology and the deficiency is structural
    rather than a matter of the nominal-pose template.
    """

    import numpy as np  # type: ignore[import-not-found]

    rig.set_nominal_qpos()
    legs = sorted(rig.contact_geoms)
    leg_joints = [
        index
        for index, joint in enumerate(rig.blueprint.joints)
        if not joint.semantic_slot.startswith("waist_")
        and any(joint.child_link.startswith(leg.split("_")[0]) for leg in legs)
    ]
    bounds = [rig.joint_limits(rig.blueprint.joints[index]) for index in leg_joints]

    def corners(leg: str) -> Any:
        gid = rig.contact_geoms[leg][0]
        centre = np.asarray(rig.data.geom_xpos[gid])
        rot = np.asarray(rig.data.geom_xmat[gid]).reshape(3, 3)
        size = np.asarray(rig.model.geom_size[gid])
        if int(rig.model.geom_type[gid]) == int(rig.mujoco.mjtGeom.mjGEOM_BOX):
            half = size[:3]
        else:
            half = np.array([size[1], 0.0, size[0]])
        return np.asarray(
            [
                centre + rot @ np.array([sx * half[0], sy * half[1], -half[2]])
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
            ]
        )

    def com() -> Any:
        total = float(np.sum(rig.model.body_mass))
        acc = np.zeros(3)
        for body_id in range(rig.model.nbody):
            acc += float(rig.model.body_mass[body_id]) * np.asarray(rig.data.xipos[body_id])
        return acc / max(1e-12, total)

    def cost(vector: Any) -> tuple[float, dict[str, float]]:
        rig.data.qpos[2] = vector[0]
        for value, index in zip(vector[1:], leg_joints):
            rig.data.qpos[rig.qpos_adr[index]] = value
        rig.mujoco.mj_forward(rig.model, rig.data)
        contact_terms = 0.0
        tilt_terms = 0.0
        points: list[Any] = []
        for leg in legs:
            pts = corners(leg)
            bottom = float(np.min(pts[:, 2]))
            contact_terms += (bottom - margin) ** 2
            tilt_terms += (float(np.max(pts[:, 2])) - bottom) ** 2
            points.extend(pts)
        centre = np.mean(np.asarray(points)[:, :2], axis=0)
        centre_of_mass = com()
        balance = float(np.sum((centre_of_mass[:2] - centre) ** 2))
        total = 200.0 * contact_terms + 20.0 * tilt_terms + 10.0 * balance
        return total, {
            "contact": contact_terms,
            "tilt": tilt_terms,
            "balance": balance,
            "com_x": float(centre_of_mass[0]),
            "com_y": float(centre_of_mass[1]),
        }

    rng = np.random.default_rng(rig.blueprint.seed)
    best_vector = None
    best_cost = float("inf")
    best_terms: dict[str, float] = {}
    nominal = np.array(
        [rig.blueprint.nominal_height * (rig.physical.global_scale if rig.physical else 1.0)]
        + [float(rig.data.qpos[rig.qpos_adr[index]]) for index in leg_joints]
    )
    for restart in range(restarts):
        vector = nominal.copy()
        if restart:
            for slot, (lower, upper) in enumerate(bounds, start=1):
                vector[slot] = rng.uniform(max(lower, -1.6), min(upper, 1.6))
            vector[0] = nominal[0] * rng.uniform(0.55, 0.95)
        current, terms = cost(vector)
        step = np.array([0.08] + [0.35] * len(leg_joints))
        for _ in range(sweeps):
            improved = False
            for slot in range(len(vector)):
                for direction in (1.0, -1.0):
                    trial = vector.copy()
                    trial[slot] += direction * step[slot]
                    if slot == 0:
                        trial[0] = float(np.clip(trial[0], 0.15, nominal[0] * 1.2))
                    else:
                        lower, upper = bounds[slot - 1]
                        trial[slot] = float(np.clip(trial[slot], lower, upper))
                    value, trial_terms = cost(trial)
                    if value < current - 1e-12:
                        vector, current, terms = trial, value, trial_terms
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if float(np.max(step)) < 1e-4:
                    break
        if current < best_cost:
            best_cost, best_vector, best_terms = current, vector.copy(), terms
    assert best_vector is not None
    cost(best_vector)
    return {
        "base_height": float(rig.data.qpos[2]),
        "cost": best_cost,
        "terms": best_terms,
        "angles": {
            rig.blueprint.joints[index].semantic_slot: float(rig.data.qpos[rig.qpos_adr[index]])
            for index in leg_joints
        },
    }


def run_rung(
    name: str,
    blueprint: Any,
    physical: Any,
    *,
    pose: str,
    margin: float,
    kp_scale: float = 1.0,
    kv_scale: float = 1.0,
    gravity_comp: bool = False,
    box_feet: bool = False,
    seconds: float = 2.0,
    config: WholeBodyMuJoCoShardConfig | None = None,
) -> dict[str, Any]:
    import numpy as np  # type: ignore[import-not-found]

    config = config or WholeBodyMuJoCoShardConfig()
    xml_text = compile_mjcf(blueprint, physical)
    if box_feet:
        xml_text = _with_box_feet(xml_text, blueprint, physical)
    if kp_scale != 1.0 or kv_scale != 1.0:
        xml_text = _scale_pd(xml_text, kp_scale, kv_scale)
    rig = _Rig(xml_text, blueprint, physical)
    mujoco, model, data = rig.mujoco, rig.model, rig.data

    pose_info: dict[str, Any] = {}
    if pose == "nominal":
        rig.set_nominal_qpos()
        ground_nominal_pose(mujoco, model, data, margin=margin)
    elif pose == "double_support":
        pose_info = solve_double_support_pose(rig, margin=margin)
    elif pose == "centered_stance":
        pose_info = solve_centered_stance_pose(rig, margin=margin)
    elif pose == "flat_stance":
        pose_info = solve_flat_stance_pose(rig, margin=margin)
    elif pose == "optimized_stance":
        pose_info = solve_optimized_stance_pose(rig, margin=margin)
    else:
        raise ValueError(f"unknown pose {pose}")

    targets = [float(data.qpos[adr]) for adr in rig.qpos_adr]
    for value, aid in zip(targets, rig.act_ids):
        lower, upper = (float(v) for v in model.actuator_ctrlrange[aid])
        data.ctrl[aid] = min(upper, max(lower, value))

    total_mass = float(np.sum(model.body_mass))
    com = np.zeros(3)
    for body_id in range(model.nbody):
        com += float(model.body_mass[body_id]) * np.asarray(data.xipos[body_id])
    com /= max(1e-12, total_mass)
    support_points: list[tuple[float, float]] = []
    for gid in rig.foot_geom_ids:
        if _geom_bottom(mujoco, model, data, gid) > 0.02:
            continue
        if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            centre = np.asarray(data.geom_xpos[gid])
            rot = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
            half = np.asarray(model.geom_size[gid])[:3]
            local = [
                np.array([sx * half[0], sy * half[1], -half[2]])
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
            ]
            world = [centre + rot @ offset for offset in local]
            floor_z = min(float(point[2]) for point in world)
            # Only corners actually resting on the floor carry load.
            for point in world:
                if float(point[2]) - floor_z <= 0.005:
                    support_points.append((float(point[0]), float(point[1])))
        else:
            support_points.append((float(data.geom_xpos[gid, 0]), float(data.geom_xpos[gid, 1])))
    support = _support_margin((float(com[0]), float(com[1])), support_points)
    reset_state = {
        "base_height": float(data.qpos[2]),
        "foot_bottom_heights": {
            leg: float(rig.foot_bottom(leg)) for leg in sorted(rig.contact_geoms)
        },
        "feet_at_floor": sum(
            1 for leg in rig.contact_geoms if rig.foot_bottom(leg) <= 0.02
        ),
        "ncon": int(data.ncon),
        "support": support,
        "pose": pose_info,
    }

    substeps = config.substeps
    steps = round(config.control_hz * seconds)
    fall_height = blueprint.nominal_height * (physical.global_scale if physical else 1.0) * config.fall_height_fraction
    first_fall: int | None = None
    heights: list[float] = []
    rolls: list[float] = []
    pitches: list[float] = []
    contacts: list[int] = []
    forces: list[float] = []
    saturation = 0
    min_contact_distance = float("inf")
    nan_seen = False
    for step_index in range(steps):
        for _ in range(substeps):
            if gravity_comp:
                mujoco.mj_forward(model, data)
                data.qfrc_applied[:] = 0.0
                for dof in rig.dof_adr:
                    data.qfrc_applied[dof] = float(data.qfrc_bias[dof])
            mujoco.mj_step(model, data)
        heights.append(float(data.qpos[2]))
        w, x, y, z = (float(v) for v in data.qpos[3:7])
        norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        rolls.append(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
        pitches.append(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
        foot_contacts = 0
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            if rig.floor_id not in (int(contact.geom1), int(contact.geom2)):
                continue
            other = int(contact.geom2) if int(contact.geom1) == rig.floor_id else int(contact.geom1)
            min_contact_distance = min(min_contact_distance, float(contact.dist))
            if other in rig.foot_geom_ids:
                foot_contacts += 1
        contacts.append(foot_contacts)
        peak = 0.0
        for aid in rig.act_ids:
            value = abs(float(data.actuator_force[aid]))
            peak = max(peak, value)
            if value >= 0.995 * float(model.actuator_forcerange[aid, 1]):
                saturation += 1
        forces.append(peak)
        if not all(math.isfinite(float(v)) for v in data.qpos):
            nan_seen = True
            break
        upright = 1.0 - 2.0 * (x * x + y * y)
        if (float(data.qpos[2]) < fall_height or upright < config.upright_threshold) and first_fall is None:
            first_fall = step_index + 1
            break
    return {
        "rung": name,
        "reset": reset_state,
        "survived_2s": first_fall is None and not nan_seen,
        "first_fall_step": first_fall,
        "first_fall_seconds": None if first_fall is None else first_fall / config.control_hz,
        "steps_run": len(heights),
        "base_height_final": heights[-1] if heights else float("nan"),
        "base_height_min": min(heights) if heights else float("nan"),
        "abs_roll_max": max(abs(v) for v in rolls) if rolls else float("nan"),
        "abs_pitch_max": max(abs(v) for v in pitches) if pitches else float("nan"),
        "foot_contacts_final": contacts[-1] if contacts else 0,
        "foot_contacts_mean": (sum(contacts) / len(contacts)) if contacts else 0.0,
        "actuator_force_peak": max(forces) if forces else 0.0,
        "actuator_saturation_events": saturation,
        "min_contact_distance": None if min_contact_distance == float("inf") else min_contact_distance,
        "nan_seen": nan_seen,
    }


LADDER: tuple[tuple[str, dict[str, Any]], ...] = (
    ("nominal_zero_action", {"pose": "nominal", "margin": 0.015}),
    ("nominal_position_target", {"pose": "nominal", "margin": 0.015}),
    ("double_support", {"pose": "double_support", "margin": 0.015}),
    ("double_support_contact", {"pose": "double_support", "margin": 0.0}),
    ("high_pd", {"pose": "double_support", "margin": 0.0, "kp_scale": 10.0, "kv_scale": 5.0}),
    ("gravity_compensation", {"pose": "double_support", "margin": 0.0, "gravity_comp": True}),
    (
        "ideal_joint_controller",
        {
            "pose": "double_support",
            "margin": 0.0,
            "kp_scale": 10.0,
            "kv_scale": 5.0,
            "gravity_comp": True,
        },
    ),
    ("box_feet", {"pose": "double_support", "margin": 0.0, "box_feet": True}),
    ("centered_stance_point_feet", {"pose": "centered_stance", "margin": 0.0}),
    ("centered_stance_box_feet", {"pose": "centered_stance", "margin": 0.0, "box_feet": True}),
    (
        "centered_stance_box_feet_gravcomp",
        {"pose": "centered_stance", "margin": 0.0, "box_feet": True, "gravity_comp": True},
    ),
    ("flat_stance_point_feet", {"pose": "flat_stance", "margin": 0.0}),
    ("flat_stance_box_feet", {"pose": "flat_stance", "margin": 0.0, "box_feet": True}),
    (
        "flat_stance_box_feet_gravcomp",
        {"pose": "flat_stance", "margin": 0.0, "box_feet": True, "gravity_comp": True},
    ),
    ("optimized_stance_point_feet", {"pose": "optimized_stance", "margin": 0.0}),
    ("optimized_stance_box_feet", {"pose": "optimized_stance", "margin": 0.0, "box_feet": True}),
    (
        "optimized_stance_box_feet_gravcomp",
        {"pose": "optimized_stance", "margin": 0.0, "box_feet": True, "gravity_comp": True},
    ),
    (
        "box_feet_ideal",
        {
            "pose": "double_support",
            "margin": 0.0,
            "kp_scale": 10.0,
            "kv_scale": 5.0,
            "gravity_comp": True,
            "box_feet": True,
        },
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("biped", "quadruped"), default="biped")
    parser.add_argument("--seed", type=int, default=-1, help="-1 picks the simplest topology")
    parser.add_argument("--range-fraction", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    generator = MorphologyGenerator()
    if args.seed < 0 and args.family == "biped":
        seed, blueprint = find_simplest_biped(generator)
    else:
        seed = max(0, args.seed)
        blueprint = generator.generate(args.family, seed)
    physical = generator.sample_physical_params(
        blueprint, seed + 10_000_000, range_fraction=args.range_fraction
    )
    rungs = [
        run_rung(name, blueprint, physical, seconds=args.seconds, **kwargs)
        for name, kwargs in LADDER
    ]
    payload = {
        "schema": "whole_body_stance_isolation_v1",
        "family": args.family,
        "seed": seed,
        "structural_hash": blueprint.structural_hash,
        "has_arms": blueprint.has_arms,
        "num_joints": len(blueprint.joints),
        "active_slots": list(blueprint.active_slots),
        "range_fraction": args.range_fraction,
        "global_scale": physical.global_scale,
        "seconds": args.seconds,
        "rungs": rungs,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    header = f"{args.family} seed={seed} hash={blueprint.structural_hash} joints={len(blueprint.joints)}"
    print(header)
    print(
        f"{'rung':26s} {'surv':5s} {'fall_s':7s} {'h0':6s} {'hmin':6s} "
        f"{'|roll|':7s} {'|pitch|':7s} {'feet0':5s} {'ncon0':5s} {'hull':8s} "
        f"{'margin':8s} {'Fpk':7s} {'sat':4s}"
    )
    for rung in rungs:
        support = rung["reset"]["support"]
        fall = rung["first_fall_seconds"]
        fall_text = "-" if fall is None else f"{fall:.2f}"
        print(
            f"{rung['rung']:26s} {rung['survived_2s']!s:5s} "
            f"{fall_text:7s} "
            f"{rung['reset']['base_height']:.3f}  {rung['base_height_min']:.3f}  "
            f"{rung['abs_roll_max']:.4f}  {rung['abs_pitch_max']:.4f}  "
            f"{rung['reset']['feet_at_floor']:<5d} {rung['reset']['ncon']:<5d} "
            f"{support['hull_area']:.5f}  {support['margin']:+.4f}  "
            f"{rung['actuator_force_peak']:6.1f} {rung['actuator_saturation_events']:<4d}"
        )


if __name__ == "__main__":
    main()
