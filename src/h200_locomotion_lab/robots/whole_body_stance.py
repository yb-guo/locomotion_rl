"""Contracts for physical-instance-specific whole-body stance solutions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from h200_locomotion_lab.robots.procedural_morphology import (
    MorphologyBlueprint,
    MorphologyInstanceKey,
    PhysicalParams,
    morphology_instance_key,
)
from h200_locomotion_lab.robots.whole_body_actual_stance import (
    solve_actual_dynamics_stance,
)

STANCE_SOLUTION_CONTRACT_VERSION = "whole_body_static_stance_v3_actual_dynamics_feedforward"
_STANCE_SOLUTION_CONTRACT = {
    "version": STANCE_SOLUTION_CONTRACT_VERSION,
    "source": "exact_morphology_instance_key",
    "root_pose_eq": "absolute_free_root_pose_in_compiled_model_with_xy_yaw_gauge_zero",
    "joint_qpos_eq": "absolute_compiled_joint_coordinates",
    "actuator_ctrl_eq": "independent_position_actuator_feedforward_targets",
    "nominal_offsets": "already_applied_by_physical_instance",
    "objective": "flat_collision_free_actual_mujoco_qacc_double_support_hold",
    "minimum_joint_margin_rad": 0.05,
    "minimum_ctrl_margin_rad": 0.01,
    "cache_reuse": "exact_instance_key_only",
}
STANCE_SOLUTION_CONTRACT_HASH = hashlib.sha256(
    json.dumps(_STANCE_SOLUTION_CONTRACT, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


class StaticStanceSolveError(RuntimeError):
    """Raised when a physical morphology instance has no usable static stance."""


_STANCE_CACHE: dict[str, StanceSolution] = {}


def stance_cache_key(instance_key: MorphologyInstanceKey) -> str:
    """Key a stance cache by exact physical instance and solver semantics."""

    encoded = json.dumps(
        {
            "instance_key": instance_key.manifest(),
            "solver_contract_version": STANCE_SOLUTION_CONTRACT_VERSION,
            "solver_contract_hash": STANCE_SOLUTION_CONTRACT_HASH,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class StanceSolution:
    """A solved reset pose and independent feedforward input.

    ``joint_qpos`` and ``actuator_ctrl`` are both absolute compiled-model
    coordinates.  They are intentionally different quantities: reset uses
    ``qpos_eq`` while zero action is centered on ``ctrl_eq``.
    """

    instance_key: MorphologyInstanceKey
    base_height: float
    joint_qpos: Mapping[str, float]
    actuator_ctrl: Mapping[str, float]
    root_xy: tuple[float, float] = (0.0, 0.0)
    root_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    solver_contract_version: str = STANCE_SOLUTION_CONTRACT_VERSION
    solver_contract_hash: str = STANCE_SOLUTION_CONTRACT_HASH
    model_xml_sha256: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.solver_contract_version != STANCE_SOLUTION_CONTRACT_VERSION:
            raise ValueError("stance solver contract version does not match the runtime")
        if self.solver_contract_hash != STANCE_SOLUTION_CONTRACT_HASH:
            raise ValueError("stance solver contract hash does not match the runtime")
        if self.model_xml_sha256 is not None:
            try:
                valid_model_sha = (
                    len(self.model_xml_sha256) == 64
                    and int(self.model_xml_sha256, 16) >= 0
                )
            except ValueError:
                valid_model_sha = False
            if not valid_model_sha:
                raise ValueError("stance model_xml_sha256 must be a full SHA-256")
        if not math.isfinite(self.base_height) or self.base_height <= 0.0:
            raise ValueError("stance base_height must be finite and positive")
        root_xy = tuple(float(value) for value in self.root_xy)
        root_quat = tuple(float(value) for value in self.root_quat)
        if len(root_xy) != 2 or any(not math.isfinite(value) for value in root_xy):
            raise ValueError("stance root_xy must contain two finite coordinates")
        if len(root_quat) != 4 or any(not math.isfinite(value) for value in root_quat):
            raise ValueError("stance root_quat must contain four finite values")
        quat_norm = math.sqrt(sum(value * value for value in root_quat))
        if quat_norm <= 1e-12:
            raise ValueError("stance root_quat must be non-zero")
        root_quat = tuple(value / quat_norm for value in root_quat)
        normalized_qpos = {str(slot): float(value) for slot, value in self.joint_qpos.items()}
        normalized_ctrl = {str(slot): float(value) for slot, value in self.actuator_ctrl.items()}
        if not normalized_qpos or any(not math.isfinite(value) for value in normalized_qpos.values()):
            raise ValueError("stance joint_qpos must contain finite absolute coordinates")
        if normalized_qpos.keys() != normalized_ctrl.keys():
            raise ValueError("stance actuator_ctrl must contain exactly the stance joint slots")
        if any(not math.isfinite(value) for value in normalized_ctrl.values()):
            raise ValueError("stance actuator_ctrl must contain finite absolute targets")
        object.__setattr__(self, "root_xy", root_xy)
        object.__setattr__(self, "root_quat", root_quat)
        object.__setattr__(self, "joint_qpos", MappingProxyType(normalized_qpos))
        object.__setattr__(self, "actuator_ctrl", MappingProxyType(normalized_ctrl))

    def validate_for(
        self,
        blueprint: MorphologyBlueprint,
        physical: PhysicalParams | None = None,
        *,
        expected_model_xml_sha256: str | None = None,
    ) -> None:
        """Reject reuse across topology or continuous-physics realizations."""

        expected = morphology_instance_key(blueprint, physical)
        if self.instance_key != expected:
            raise ValueError("stance solution belongs to a different morphology/physical instance")
        if (
            expected_model_xml_sha256 is not None
            and self.model_xml_sha256 != expected_model_xml_sha256
        ):
            raise ValueError("stance solution belongs to a different model XML")
        expected_slots = set(blueprint.active_slots)
        actual_slots = set(self.joint_qpos)
        if actual_slots != expected_slots:
            missing = sorted(expected_slots - actual_slots)
            unknown = sorted(actual_slots - expected_slots)
            raise ValueError(
                f"stance joint slots do not match blueprint: missing={missing}, unknown={unknown}"
            )
        for joint in blueprint.joints:
            limit_scale = (
                physical.joint_limit_scales.get(joint.semantic_slot, 1.0) if physical else 1.0
            )
            nominal_offset = (
                physical.nominal_offsets.get(joint.semantic_slot, 0.0) if physical else 0.0
            )
            lower = joint.joint_range[0] * limit_scale + nominal_offset
            upper = joint.joint_range[1] * limit_scale + nominal_offset
            value = self.joint_qpos[joint.semantic_slot]
            if value < lower - 1e-9 or value > upper + 1e-9:
                raise ValueError(f"stance joint {joint.semantic_slot} is outside compiled limits")
            ctrl = self.actuator_ctrl[joint.semantic_slot]
            if ctrl < lower - 1e-9 or ctrl > upper + 1e-9:
                raise ValueError(f"stance ctrl {joint.semantic_slot} is outside compiled limits")

    @property
    def root_pose(self) -> tuple[float, float, float, float, float, float, float]:
        return (
            float(self.root_xy[0]),
            float(self.root_xy[1]),
            float(self.base_height),
            *tuple(float(value) for value in self.root_quat),
        )

    def manifest(self) -> dict[str, object]:
        manifest: dict[str, object] = {
            "instance_key": self.instance_key.manifest(),
            "base_height": self.base_height,
            "root_pose_eq": list(self.root_pose),
            "joint_qpos": dict(sorted(self.joint_qpos.items())),
            "joint_qpos_eq": dict(sorted(self.joint_qpos.items())),
            "actuator_ctrl_eq": dict(sorted(self.actuator_ctrl.items())),
            "solver_contract_version": self.solver_contract_version,
            "solver_contract_hash": self.solver_contract_hash,
        }
        if self.model_xml_sha256 is not None:
            manifest["model_xml_sha256"] = self.model_xml_sha256
        return manifest

    @property
    def solution_hash(self) -> str:
        encoded = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def cache_key(self) -> str:
        return stance_cache_key(self.instance_key)


def solve_static_stance(
    model: Any,
    data: Any,
    blueprint: MorphologyBlueprint,
    physical: PhysicalParams | None = None,
    *,
    margin: float = 0.0,
    joint_limit_margin: float = 0.06,
    ctrl_limit_margin: float = 0.01,
    restarts: int = 6,
    sweeps: int = 120,
    kinematic_max_nfev: int = 400,
    dynamics_max_nfev: int = 350,
) -> StanceSolution:
    """Solve a physical-instance-specific static reset pose and input.

    The solver mutates ``data`` to the returned equilibrium.  A deterministic
    projected coordinate descent supplies the geometric start; final acceptance
    is an actual MuJoCo dynamics refinement with independent ``qpos_eq`` and
    ``ctrl_eq``.
    """

    try:
        import mujoco  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional simulator dependency
        raise RuntimeError("MuJoCo and NumPy are required for solve_static_stance") from exc

    if restarts <= 0 or sweeps <= 0:
        raise ValueError("restarts and sweeps must be positive")
    if joint_limit_margin < 0.0:
        raise ValueError("joint_limit_margin must be non-negative")
    if ctrl_limit_margin < 0.0:
        raise ValueError("ctrl_limit_margin must be non-negative")
    instance_key = morphology_instance_key(blueprint, physical)
    cache_key = stance_cache_key(instance_key)
    qpos_adr, joint_ids = _joint_qpos_addresses(mujoco, model, blueprint)
    actuator_ids = tuple(
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator.name))
        for actuator in blueprint.actuators
    )
    cached = _STANCE_CACHE.get(cache_key)
    if cached is not None:
        cached.validate_for(blueprint, physical)
        _apply_solution_to_data(data, blueprint, cached, qpos_adr, actuator_ids)
        mujoco.mj_forward(model, data)
        return cached

    nominal_values = _set_blueprint_nominal_qpos(
        mujoco,
        model,
        data,
        blueprint,
        physical,
        qpos_adr,
        joint_ids,
    )
    foot_geom_ids = _foot_geom_ids(mujoco, model, blueprint)
    leg_joints = _leg_joint_indices(blueprint)
    if not leg_joints:
        raise StaticStanceSolveError("static stance requires at least one generated leg joint")
    bounds = [
        _shrink_joint_bounds(
            tuple(float(value) for value in model.jnt_range[joint_ids[index]]),
            joint_limit_margin,
        )
        for index in leg_joints
    ]

    def corners(geom_id: int) -> Any:
        center = np.asarray(data.geom_xpos[geom_id])
        rot = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
        size = np.asarray(model.geom_size[geom_id])
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            half = size[:3]
        elif int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            half = np.array([size[1], 0.0, size[0]])
        else:
            half = np.array([max(size), max(size), max(size)])
        return np.asarray(
            [
                center + rot @ np.array([sx * half[0], sy * half[1], -half[2]])
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
            ]
        )

    def center_of_mass() -> Any:
        total_mass = float(np.sum(model.body_mass))
        accum = np.zeros(3)
        for body_id in range(model.nbody):
            accum += float(model.body_mass[body_id]) * np.asarray(data.xipos[body_id])
        return accum / max(1e-12, total_mass)

    def evaluate(vector: Any) -> tuple[float, dict[str, float]]:
        data.qpos[2] = float(vector[0])
        for value, index in zip(vector[1:], leg_joints):
            data.qpos[qpos_adr[index]] = float(value)
        mujoco.mj_forward(model, data)
        contact_terms = 0.0
        tilt_terms = 0.0
        support_points = []
        foot_spread_values = []
        for geom_id in foot_geom_ids:
            pts = corners(geom_id)
            bottom = float(np.min(pts[:, 2]))
            top_bottom_face = float(np.max(pts[:, 2]))
            contact_terms += (bottom - margin) ** 2
            tilt_terms += (top_bottom_face - bottom) ** 2
            foot_spread_values.append(bottom)
            support_points.extend(pts[:, :2])
        support_centroid = np.mean(np.asarray(support_points), axis=0)
        com_xy = center_of_mass()[:2]
        balance = float(np.sum((com_xy - support_centroid) ** 2))
        regularization = 0.0
        for value, nominal, (lower, upper) in zip(vector[1:], nominal_values, bounds):
            regularization += ((float(value) - nominal) / max(1e-6, upper - lower)) ** 2
        foot_spread = max(foot_spread_values) - min(foot_spread_values)
        total = (
            1000.0 * contact_terms
            + 1000.0 * tilt_terms
            + 50.0 * balance
            + 1000.0 * foot_spread * foot_spread
            + 1.0 * regularization
        )
        return total, {
            "contact": contact_terms,
            "tilt": tilt_terms,
            "balance": balance,
            "foot_spread": foot_spread,
        }

    nominal = np.asarray(
        [blueprint.nominal_height * (physical.global_scale if physical else 1.0)]
        + [float(data.qpos[qpos_adr[index]]) for index in leg_joints],
        dtype=np.float64,
    )
    rng = np.random.default_rng(blueprint.seed)
    best_vector = None
    best_cost = float("inf")
    best_terms: dict[str, float] = {}
    for restart in range(restarts):
        vector = nominal.copy()
        if restart:
            for slot, (lower, upper) in enumerate(bounds, start=1):
                vector[slot] = rng.uniform(max(lower, -1.6), min(upper, 1.6))
            vector[0] = nominal[0] * rng.uniform(0.55, 0.95)
        current, terms = evaluate(vector)
        step = np.asarray([0.08] + [0.35] * len(leg_joints), dtype=np.float64)
        for _ in range(sweeps):
            improved = False
            for slot in range(len(vector)):
                for direction in (1.0, -1.0):
                    trial = vector.copy()
                    trial[slot] += direction * step[slot]
                    if slot == 0:
                        trial[0] = float(np.clip(trial[0], 0.10, nominal[0] * 1.25))
                    else:
                        lower, upper = bounds[slot - 1]
                        trial[slot] = float(np.clip(trial[slot], lower, upper))
                    value, trial_terms = evaluate(trial)
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
    if best_vector is None:
        raise StaticStanceSolveError("static stance search did not produce a candidate")

    evaluate(best_vector)
    _align_base_to_average_foot_bottom(mujoco, model, data, foot_geom_ids, corners, margin)
    if not all(math.isfinite(float(value)) for value in data.qpos):
        raise StaticStanceSolveError("static stance candidate contains non-finite qpos")
    if best_terms.get("contact", float("inf")) > 0.10 or best_terms.get("tilt", float("inf")) > 0.10:
        raise StaticStanceSolveError(
            "static stance candidate failed contact/tilt objective: "
            f"cost={best_cost:.6g}, terms={best_terms}"
        )

    start_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
    actual = None
    actual_error: Exception | None = None
    if blueprint.family == "biped":
        try:
            actual = solve_actual_dynamics_stance(
                mujoco=mujoco,
                np=np,
                model=model,
                data=data,
                blueprint=blueprint,
                physical=physical,
                joint_qpos=qpos_adr,
                joint_dof=tuple(int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids),
                actuator_ids=actuator_ids,
                foot_geoms=tuple(f"{link.name}_footpad" for link in blueprint.links if link.foot),
                start_qpos=start_qpos,
                joint_margin=max(0.05, joint_limit_margin),
                ctrl_margin=ctrl_limit_margin,
                kinematic_max_nfev=kinematic_max_nfev,
                dynamics_max_nfev=dynamics_max_nfev,
            )
        except RuntimeError as exc:
            actual_error = exc
    if actual is None or not bool(actual["strict_initial_actual_equilibrium"]):
        if blueprint.family == "quadruped":
            actual_qpos = start_qpos.copy()
            actual_qpos[0] = 0.0
            actual_qpos[1] = 0.0
            actual_qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            actual_ctrl = np.zeros(model.nu, dtype=np.float64)
            for actuator_id, qpos_address in zip(actuator_ids, qpos_adr):
                lower, upper = (float(value) for value in model.actuator_ctrlrange[int(actuator_id)])
                actual_ctrl[int(actuator_id)] = min(
                    upper,
                    max(lower, float(actual_qpos[int(qpos_address)])),
                )
        else:
            best = actual["dynamics_phase"]["best_preview"] if actual is not None else str(actual_error)
            raise StaticStanceSolveError(
                "actual-dynamics stance solver failed strict equilibrium: "
                f"best={best}"
            )
    else:
        actual_qpos = np.asarray(actual["qpos"], dtype=np.float64)
        actual_ctrl = np.asarray(actual["ctrl"], dtype=np.float64)
    joint_qpos = {
        joint.semantic_slot: float(actual_qpos[address])
        for joint, address in zip(blueprint.joints, qpos_adr)
    }
    actuator_ctrl = {
        actuator.semantic_slot: float(actual_ctrl[int(actuator_id)])
        for actuator, actuator_id in zip(blueprint.actuators, actuator_ids)
    }
    solution = StanceSolution(
        instance_key=instance_key,
        base_height=float(actual_qpos[2]),
        joint_qpos=joint_qpos,
        actuator_ctrl=actuator_ctrl,
        root_xy=(float(actual_qpos[0]), float(actual_qpos[1])),
        root_quat=tuple(float(value) for value in actual_qpos[3:7]),  # type: ignore[arg-type]
    )
    solution.validate_for(blueprint, physical)
    _STANCE_CACHE[cache_key] = solution
    _apply_solution_to_data(data, blueprint, solution, qpos_adr, actuator_ids)
    mujoco.mj_forward(model, data)
    return solution


def _joint_qpos_addresses(
    mujoco: Any,
    model: Any,
    blueprint: MorphologyBlueprint,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    qpos_addresses = []
    joint_ids = []
    for joint in blueprint.joints:
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.name))
        if joint_id < 0:
            raise StaticStanceSolveError(f"compiled model is missing joint {joint.name}")
        joint_ids.append(joint_id)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
    return tuple(qpos_addresses), tuple(joint_ids)


def _set_blueprint_nominal_qpos(
    mujoco: Any,
    model: Any,
    data: Any,
    blueprint: MorphologyBlueprint,
    physical: PhysicalParams | None,
    qpos_addresses: tuple[int, ...],
    joint_ids: tuple[int, ...],
) -> tuple[float, ...]:
    mujoco.mj_resetData(model, data)
    data.qpos[2] = blueprint.nominal_height * (physical.global_scale if physical else 1.0)
    nominal_values = []
    for joint, qpos_address, joint_id in zip(blueprint.joints, qpos_addresses, joint_ids):
        nominal_offset = physical.nominal_offsets.get(joint.semantic_slot, 0.0) if physical else 0.0
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
        nominal = min(upper, max(lower, joint.nominal + nominal_offset))
        data.qpos[qpos_address] = nominal
        nominal_values.append(float(nominal))
    mujoco.mj_forward(model, data)
    return tuple(nominal_values[index] for index in _leg_joint_indices(blueprint))


def _foot_geom_ids(mujoco: Any, model: Any, blueprint: MorphologyBlueprint) -> tuple[int, ...]:
    geom_ids = []
    for link in blueprint.links:
        if not link.foot:
            continue
        geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{link.name}_footpad"))
        if geom_id < 0:
            raise StaticStanceSolveError(f"compiled model is missing footpad for {link.name}")
        geom_ids.append(geom_id)
    if not geom_ids:
        raise StaticStanceSolveError("static stance requires at least one footpad geom")
    return tuple(geom_ids)


def _shrink_joint_bounds(bounds: tuple[float, float], margin: float) -> tuple[float, float]:
    lower, upper = bounds
    if upper - lower <= 2.0 * margin:
        midpoint = 0.5 * (lower + upper)
        return midpoint, midpoint
    return lower + margin, upper - margin


def _chain_prefix(link_name: str) -> str:
    parts = link_name.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2] == "link":
        return parts[0]
    return link_name


def _leg_joint_indices(blueprint: MorphologyBlueprint) -> tuple[int, ...]:
    semantic_indices = tuple(
        index
        for index, joint in enumerate(blueprint.joints)
        if joint.semantic_slot.startswith("limb") and joint.axis_name != "wheel"
    )
    if semantic_indices:
        return semantic_indices
    prefixes = tuple(_chain_prefix(link.name) for link in blueprint.links if link.foot)
    return tuple(
        index
        for index, joint in enumerate(blueprint.joints)
        if any(joint.child_link.startswith(f"{prefix}_") for prefix in prefixes)
    )


def _align_base_to_average_foot_bottom(
    mujoco: Any,
    model: Any,
    data: Any,
    foot_geom_ids: tuple[int, ...],
    corners: Any,
    margin: float,
) -> None:
    bottoms = [float(min(corners(geom_id)[:, 2])) for geom_id in foot_geom_ids]
    if not bottoms:
        raise StaticStanceSolveError("static stance has no foot bottoms to align")
    data.qpos[2] += margin - (sum(bottoms) / len(bottoms))
    mujoco.mj_forward(model, data)


def _apply_solution_to_data(
    data: Any,
    blueprint: MorphologyBlueprint,
    solution: StanceSolution,
    qpos_addresses: tuple[int, ...],
    actuator_ids: tuple[int, ...] | None = None,
) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[0] = solution.root_xy[0]
    data.qpos[1] = solution.root_xy[1]
    data.qpos[2] = solution.base_height
    data.qpos[3:7] = solution.root_quat
    for joint, qpos_address in zip(blueprint.joints, qpos_addresses):
        data.qpos[qpos_address] = solution.joint_qpos[joint.semantic_slot]
    if actuator_ids is not None:
        for actuator, actuator_id in zip(blueprint.actuators, actuator_ids):
            data.ctrl[int(actuator_id)] = solution.actuator_ctrl[actuator.semantic_slot]
