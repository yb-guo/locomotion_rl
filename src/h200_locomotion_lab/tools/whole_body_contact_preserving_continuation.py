"""Task067 R4a.3.1 contact-preserving continuation diagnosis.

This tool diagnoses the three R4a.3 strict-coverage failures without changing
the public controller, actuator gains, reward, observation/action schema, or
generator grammar.  It follows same-topology range-fraction continuation from
an already strict neighboring solution, then adds a contact-preserving residual
when the qacc-only refinement slides into a single-foot active set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from h200_locomotion_lab.envs.whole_body_mujoco import (
    WholeBodyMuJoCoShard,
    WholeBodyMuJoCoShardConfig,
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
from h200_locomotion_lab.tools.whole_body_dynamic_balance_diagnosis import (
    _align_lowest_foot_to_penetration,
    _center_of_mass,
    _contact_report,
    _foot_bottom_heights,
    _joint_position_bounds,
    _quat_from_roll_pitch_yaw,
    _reset_to_qpos,
    _roll_pitch_yaw,
    _stance_qpos,
)
from h200_locomotion_lab.tools.whole_body_equilibrium_audit import (
    _JOINT_QACC_SCALE,
    _ROOT_QACC_SCALE,
    _STRICT_JOINT_QACC_MAX,
    _STRICT_ROOT_QACC_NORM,
    _sha256_path,
    strict_actual_equilibrium,
)
from h200_locomotion_lab.tools.whole_body_strict_equilibrium_coverage import (
    run_strict_hold_rollout,
)

_DEFAULT_R4A3_INPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a3_strict_equilibrium_coverage_4x2.json"
)
_DEFAULT_OUTPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31_contact_preserving_continuation_3fail.json"
)
_DEFAULT_FAMILY = "biped"
_DEFAULT_STEP = 0.05
_DEFAULT_MIN_STEP = 0.00625
_FOOT_HEIGHT_SCALE = 0.001
_FOOT_LOAD_DEFICIT_SCALE = 20.0
_REGULARIZATION_SCALE = 1e-5
_BOUND_ACTIVE_TOL = 1e-6

STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND = "strict_double_support_equilibrium_found"
SINGLE_SUPPORT_EQUILIBRIUM_ONLY_FOUND = "single_support_equilibrium_only_found"
SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE = "search_exhausted_without_certificate"
KINEMATIC_DOUBLE_SUPPORT_INFEASIBLE = "kinematic_double_support_infeasible"
WRENCH_OR_ACTUATION_INFEASIBLE = "wrench_or_actuation_infeasible"

_FAILURE_CLASSIFICATIONS = (
    STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND,
    SINGLE_SUPPORT_EQUILIBRIUM_ONLY_FOUND,
    SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE,
    KINEMATIC_DOUBLE_SUPPORT_INFEASIBLE,
    WRENCH_OR_ACTUATION_INFEASIBLE,
)


@dataclass(frozen=True, slots=True)
class ContinuationRoute:
    seed: int
    start_range_fraction: float
    end_range_fraction: float

    @property
    def direction(self) -> float:
        return 1.0 if self.end_range_fraction >= self.start_range_fraction else -1.0

    @property
    def endpoint_label(self) -> str:
        return f"{_DEFAULT_FAMILY}:rf{self.end_range_fraction:g}:seed{self.seed}"


_DEFAULT_ROUTES = (
    ContinuationRoute(seed=0, start_range_fraction=0.0, end_range_fraction=0.5),
    ContinuationRoute(seed=3, start_range_fraction=0.0, end_range_fraction=0.5),
    ContinuationRoute(seed=1, start_range_fraction=0.5, end_range_fraction=0.0),
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _same_float(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-12


def _rounded_range_fraction(value: float) -> float:
    return round(float(value), 10)


def _build_shard_for(
    family: str,
    seed: int,
    range_fraction: float,
) -> tuple[WholeBodyMuJoCoShard, dict[str, str]]:
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
    return shard, morphology_instance_key(blueprint, physical).manifest()


def warm_start_mapping_manifest(
    source_shard: WholeBodyMuJoCoShard,
    target_shard: WholeBodyMuJoCoShard,
) -> dict[str, Any]:
    source_joint_names = [joint.name for joint in source_shard.blueprint.joints]
    target_joint_names = [joint.name for joint in target_shard.blueprint.joints]
    source_slots = [joint.semantic_slot for joint in source_shard.blueprint.joints]
    target_slots = [joint.semantic_slot for joint in target_shard.blueprint.joints]
    source_actuators = [actuator.name for actuator in source_shard.blueprint.actuators]
    target_actuators = [actuator.name for actuator in target_shard.blueprint.actuators]
    same = bool(
        source_shard.model.nq == target_shard.model.nq
        and source_shard.model.nv == target_shard.model.nv
        and source_shard.model.nu == target_shard.model.nu
        and source_joint_names == target_joint_names
        and source_slots == target_slots
        and source_actuators == target_actuators
    )
    return {
        "same_topology": same,
        "source_nq": int(source_shard.model.nq),
        "target_nq": int(target_shard.model.nq),
        "source_nv": int(source_shard.model.nv),
        "target_nv": int(target_shard.model.nv),
        "source_nu": int(source_shard.model.nu),
        "target_nu": int(target_shard.model.nu),
        "joint_names_match": source_joint_names == target_joint_names,
        "semantic_slots_match": source_slots == target_slots,
        "actuator_names_match": source_actuators == target_actuators,
    }


def _vector_bounds(shard: WholeBodyMuJoCoShard, stance: Any) -> tuple[Any, Any]:
    np = shard.np
    joint_bounds = _joint_position_bounds(shard, stance)
    lower = [-0.10, -0.10, -0.15, 0.0]
    upper = [0.10, 0.10, 0.15, 0.012]
    lower.extend(value[0] for value in joint_bounds)
    upper.extend(value[1] for value in joint_bounds)
    for actuator_id in shard._actuator_ids:
        ctrl_lower, ctrl_upper = (
            float(value) for value in shard.model.actuator_ctrlrange[int(actuator_id)]
        )
        lower.append(ctrl_lower)
        upper.append(ctrl_upper)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _active_bound_report(
    shard: WholeBodyMuJoCoShard,
    vector: Any,
    lower: Any,
    upper: Any,
) -> dict[str, Any]:
    joint_count = len(shard._joint_qpos)
    joint_start = 4
    ctrl_start = joint_start + joint_count
    active_joints = []
    active_ctrl = []
    joint_margins = []
    ctrl_margins = []
    for local_index, joint in enumerate(shard.blueprint.joints):
        index = joint_start + local_index
        value = float(vector[index])
        lo = float(lower[index])
        hi = float(upper[index])
        margin = min(value - lo, hi - value)
        joint_margins.append(margin)
        side = None
        if abs(value - lo) <= _BOUND_ACTIVE_TOL:
            side = "lower"
        elif abs(value - hi) <= _BOUND_ACTIVE_TOL:
            side = "upper"
        if side is not None:
            active_joints.append(
                {
                    "joint": joint.name,
                    "semantic_slot": joint.semantic_slot,
                    "side": side,
                    "value": value,
                    "lower": lo,
                    "upper": hi,
                }
            )
    for local_index, actuator_id in enumerate(shard._actuator_ids):
        index = ctrl_start + local_index
        actuator = shard.blueprint.actuators[local_index]
        value = float(vector[index])
        lo = float(lower[index])
        hi = float(upper[index])
        margin = min(value - lo, hi - value)
        ctrl_margins.append(margin)
        side = None
        if abs(value - lo) <= _BOUND_ACTIVE_TOL:
            side = "lower"
        elif abs(value - hi) <= _BOUND_ACTIVE_TOL:
            side = "upper"
        if side is not None:
            active_ctrl.append(
                {
                    "actuator": actuator.name,
                    "semantic_slot": actuator.semantic_slot,
                    "actuator_id": int(actuator_id),
                    "side": side,
                    "value": value,
                    "lower": lo,
                    "upper": hi,
                }
            )
    return {
        "active_joint_bounds": active_joints,
        "active_ctrl_bounds": active_ctrl,
        "active_joint_bounds_count": len(active_joints),
        "active_ctrl_bounds_count": len(active_ctrl),
        "min_joint_bound_margin": min(joint_margins) if joint_margins else None,
        "min_ctrl_bound_margin": min(ctrl_margins) if ctrl_margins else None,
    }


def _vector_from_qpos_ctrl(
    shard: WholeBodyMuJoCoShard,
    qpos: Any,
    ctrl: Any,
    lower: Any,
    upper: Any,
) -> tuple[Any, dict[str, Any]]:
    np = shard.np
    qpos_array = np.asarray(qpos, dtype=np.float64)
    ctrl_array = np.asarray(ctrl, dtype=np.float64)
    if qpos_array.shape != (shard.model.nq,):
        raise ValueError(f"warm-start qpos shape {qpos_array.shape} does not match target nq={shard.model.nq}")
    if ctrl_array.shape != (shard.model.nu,):
        raise ValueError(f"warm-start ctrl shape {ctrl_array.shape} does not match target nu={shard.model.nu}")

    stance = _stance_qpos(shard)
    probe_qpos = np.asarray(stance, dtype=np.float64).copy()
    probe_qpos[3:7] = qpos_array[3:7]
    for qpos_address in shard._joint_qpos:
        probe_qpos[int(qpos_address)] = float(qpos_array[int(qpos_address)])
    _reset_to_qpos(shard, shard.data[0], probe_qpos)
    foot_bottoms = _foot_bottom_heights(shard, shard.data[0])
    lowest = min(foot_bottoms.values(), default=0.0)
    penetration = min(0.012, max(0.0, -float(lowest)))
    roll, pitch, yaw = _roll_pitch_yaw(qpos_array[3:7])
    joint_values = [float(qpos_array[int(address)]) for address in shard._joint_qpos]
    ctrl_values = [float(ctrl_array[int(actuator_id)]) for actuator_id in shard._actuator_ids]
    raw = np.asarray([roll, pitch, yaw, penetration, *joint_values, *ctrl_values], dtype=np.float64)
    clipped = np.clip(raw, lower, upper)
    clip_delta = np.abs(clipped - raw)
    return clipped, {
        "raw_penetration_from_target_geometry": penetration,
        "raw_foot_bottom_heights_on_target": dict(sorted(foot_bottoms.items())),
        "clipped_components": int(np.count_nonzero(clip_delta > 0.0)),
        "max_abs_clip": float(np.max(clip_delta)) if clip_delta.size else 0.0,
    }


def _apply_vector(
    shard: WholeBodyMuJoCoShard,
    stance: Any,
    vector: Any,
) -> tuple[Any, Any]:
    np = shard.np
    joint_count = len(shard._joint_qpos)
    qpos = np.asarray(stance, dtype=np.float64).copy()
    qpos[3:7] = _quat_from_roll_pitch_yaw(float(vector[0]), float(vector[1]), float(vector[2]))
    for address, value in zip(shard._joint_qpos, vector[4 : 4 + joint_count]):
        qpos[int(address)] = float(value)
    ctrl = np.zeros(shard.model.nu, dtype=np.float64)
    ctrl_values = vector[4 + joint_count : 4 + joint_count + len(shard._actuator_ids)]
    for actuator_id, value in zip(shard._actuator_ids, ctrl_values):
        ctrl[int(actuator_id)] = float(value)
    _reset_to_qpos(shard, shard.data[0], qpos)
    _align_lowest_foot_to_penetration(
        shard,
        shard.data[0],
        penetration=float(vector[3]),
    )
    shard.data[0].ctrl[:] = ctrl
    shard.mujoco.mj_forward(shard.model, shard.data[0])
    return np.asarray(shard.data[0].qpos, dtype=np.float64).copy(), ctrl.copy()


def _support_mode_from_snapshot(snapshot: dict[str, Any]) -> str:
    contact = snapshot["contact"]
    minimum_load = float(snapshot["minimum_foot_load"])
    foot_names = sorted(contact["normal_force_by_foot"])
    loaded = [
        name
        for name in foot_names
        if int(contact["contacts_by_foot"].get(name, 0)) > 0
        and float(contact["normal_force_by_foot"].get(name, 0.0)) >= minimum_load
    ]
    contacting = [
        name for name in foot_names if int(contact["contacts_by_foot"].get(name, 0)) > 0
    ]
    if foot_names and len(loaded) == len(foot_names):
        return "double_support"
    if len(loaded) == 1 or len(contacting) == 1:
        return "single_support"
    if not contacting:
        return "no_foot_contact"
    return "partial_double_support_unloaded"


def _qacc_strict_ignoring_support(snapshot: dict[str, Any]) -> bool:
    return bool(
        float(snapshot["root_qacc_norm"]) <= _STRICT_ROOT_QACC_NORM
        and float(snapshot["joint_qacc_max"]) <= _STRICT_JOINT_QACC_MAX
        and int(snapshot["contact"]["non_foot_contacts"]) == 0
        and int(snapshot["actuator_saturation_events"]) == 0
    )


def classify_snapshot_without_certificates(snapshot: dict[str, Any]) -> str:
    """Classify an actual MuJoCo snapshot without inventing infeasibility proof."""

    if strict_actual_equilibrium(snapshot):
        return STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND
    support_mode = snapshot.get("support_mode") or _support_mode_from_snapshot(snapshot)
    if _qacc_strict_ignoring_support(snapshot) and support_mode == "single_support":
        return SINGLE_SUPPORT_EQUILIBRIUM_ONLY_FOUND
    return SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE


def _diagnostic_snapshot(
    shard: WholeBodyMuJoCoShard,
    qpos: Any,
    ctrl: Any,
) -> dict[str, Any]:
    np = shard.np
    data = shard.data[0]
    _reset_to_qpos(shard, data, qpos)
    data.ctrl[:] = ctrl
    shard.mujoco.mj_forward(shard.model, data)
    qacc = np.asarray(data.qacc, dtype=np.float64)
    contact = _contact_report(shard, data)
    foot_bottoms = _foot_bottom_heights(shard, data)
    com = _center_of_mass(shard, data)
    cop = contact["center_of_pressure_xy"]
    com_minus_cop = None
    if cop is not None:
        com_minus_cop = [float(com[index]) - float(cop[index]) for index in range(2)]
    total_mass = float(np.sum(shard.model.body_mass))
    weight = total_mass * abs(float(shard.model.opt.gravity[2]))
    minimum_load = 0.05 * weight
    joint_qacc_rows = []
    joint_abs = []
    for joint, dof in zip(shard.blueprint.joints, shard._joint_dof):
        value = float(qacc[int(dof)])
        joint_abs.append(abs(value))
        joint_qacc_rows.append(
            {
                "joint": joint.name,
                "semantic_slot": joint.semantic_slot,
                "dof": int(dof),
                "qacc": value,
            }
        )
    actuator_saturation_events = 0
    actuator_force_max = 0.0
    for actuator_id in shard._actuator_ids:
        force = abs(float(data.actuator_force[int(actuator_id)]))
        actuator_force_max = max(actuator_force_max, force)
        limit = max(abs(float(value)) for value in shard.model.actuator_forcerange[int(actuator_id)])
        actuator_saturation_events += int(force >= 0.995 * limit)
    foot_rows = []
    for name in sorted(shard._foot_geoms):
        contacts = int(contact["contacts_by_foot"].get(name, 0))
        normal_load = float(contact["normal_force_by_foot"].get(name, 0.0))
        foot_rows.append(
            {
                "foot": name,
                "bottom_height": float(foot_bottoms.get(name, float("inf"))),
                "contact_count": contacts,
                "normal_load": normal_load,
                "load_deficit": max(0.0, minimum_load - normal_load),
                "loaded": bool(contacts > 0 and normal_load >= minimum_load),
            }
        )
    snapshot: dict[str, Any] = {
        "root_qacc": [float(value) for value in qacc[:6]],
        "root_qacc_norm": float(np.linalg.norm(qacc[:6])),
        "joint_qacc": joint_qacc_rows,
        "joint_qacc_max": max(joint_abs, default=0.0),
        "com": com,
        "center_of_pressure_xy": cop,
        "signed_com_minus_cop_xy": com_minus_cop,
        "com_minus_cop_xy": com_minus_cop,
        "contact": contact,
        "foot_states": foot_rows,
        "minimum_foot_load": minimum_load,
        "weight": weight,
        "double_support": False,
        "support_mode": "unknown",
        "single_support_active_set": False,
        "actuator_force_max": actuator_force_max,
        "actuator_saturation_events": actuator_saturation_events,
    }
    snapshot["support_mode"] = _support_mode_from_snapshot(snapshot)
    snapshot["single_support_active_set"] = snapshot["support_mode"] == "single_support"
    snapshot["double_support"] = snapshot["support_mode"] == "double_support"
    return snapshot


def _residual_breakdown(
    shard: WholeBodyMuJoCoShard,
    vector: Any,
    warm_start: Any,
    lower: Any,
    upper: Any,
    *,
    contact_preserving: bool,
) -> dict[str, Any]:
    np = shard.np
    data = shard.data[0]
    qacc = np.asarray(data.qacc, dtype=np.float64)
    joint_qacc = qacc[list(shard._joint_dof)] if shard._joint_dof else np.asarray([])
    span = np.maximum(1e-6, upper - lower)
    regularization = _REGULARIZATION_SCALE * (np.asarray(vector) - np.asarray(warm_start)) / span
    breakdown: dict[str, Any] = {
        "root_qacc_scaled_norm": float(np.linalg.norm(qacc[:6] / _ROOT_QACC_SCALE)),
        "joint_qacc_scaled_norm": float(np.linalg.norm(joint_qacc / _JOINT_QACC_SCALE))
        if joint_qacc.size
        else 0.0,
        "regularization_scaled_norm": float(np.linalg.norm(regularization)),
    }
    if contact_preserving:
        contact = _contact_report(shard, data)
        foot_bottom = _foot_bottom_heights(shard, data)
        min_load = 0.05 * float(np.sum(shard.model.body_mass)) * abs(float(shard.model.opt.gravity[2]))
        penetration = float(vector[3])
        breakdown["foot_bottom_error_to_common_penetration"] = {
            name: float(foot_bottom.get(name, float("inf")) + penetration)
            for name in sorted(shard._foot_geoms)
        }
        breakdown["foot_load_deficit"] = {
            name: max(0.0, min_load - float(contact["normal_force_by_foot"].get(name, 0.0)))
            for name in sorted(shard._foot_geoms)
        }
    return breakdown


def refine_from_warm_start(
    shard: WholeBodyMuJoCoShard,
    warm_start_vector: Any,
    *,
    contact_preserving: bool,
    max_nfev: int,
    horizon_steps: int,
) -> dict[str, Any]:
    np = shard.np
    try:
        from scipy import optimize
    except ImportError as exc:  # pragma: no cover - optional diagnostic dependency
        raise RuntimeError("SciPy is required for R4a.3.1 continuation") from exc

    stance = _stance_qpos(shard)
    lower, upper = _vector_bounds(shard, stance)
    start = np.clip(np.asarray(warm_start_vector, dtype=np.float64), lower, upper)
    span = np.maximum(1e-6, upper - lower)

    def residual(vector: Any) -> Any:
        _apply_vector(shard, stance, vector)
        qacc = np.asarray(shard.data[0].qacc, dtype=np.float64)
        values = list(qacc[:6] / _ROOT_QACC_SCALE)
        values.extend(qacc[list(shard._joint_dof)] / _JOINT_QACC_SCALE)
        if contact_preserving:
            contact = _contact_report(shard, shard.data[0])
            foot_bottom = _foot_bottom_heights(shard, shard.data[0])
            weight = float(np.sum(shard.model.body_mass)) * abs(float(shard.model.opt.gravity[2]))
            min_load = 0.05 * weight
            penetration = float(vector[3])
            values.extend(
                float(foot_bottom.get(name, 1.0) + penetration) / _FOOT_HEIGHT_SCALE
                for name in sorted(shard._foot_geoms)
            )
            values.extend(
                _FOOT_LOAD_DEFICIT_SCALE
                * max(0.0, min_load - float(contact["normal_force_by_foot"].get(name, 0.0)))
                / max(min_load, 1e-9)
                for name in sorted(shard._foot_geoms)
            )
        values.extend(_REGULARIZATION_SCALE * (np.asarray(vector) - start) / span)
        return np.asarray(values, dtype=np.float64)

    result = optimize.least_squares(
        residual,
        start,
        bounds=(lower, upper),
        method="trf",
        max_nfev=max_nfev,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        x_scale="jac",
        diff_step=1e-5,
    )
    qpos, ctrl = _apply_vector(shard, stance, result.x)
    snapshot = _diagnostic_snapshot(shard, qpos, ctrl)
    classification = classify_snapshot_without_certificates(snapshot)
    hold_rollout = None
    strict_gate_passed = False
    if classification == STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND:
        hold_rollout = run_strict_hold_rollout(
            shard,
            {"status": "feasible", "best": {"qpos": [float(value) for value in qpos], "ctrl": [float(value) for value in ctrl]}},
            horizon_steps=horizon_steps,
        )
        strict_gate_passed = bool(hold_rollout["passed"])
    return {
        "schema": "task067_r4a31_refinement_attempt_v1",
        "phase": "contact_preserving_refinement" if contact_preserving else "qacc_only_continuation",
        "residual_terms": {
            "actual_root_joint_qacc": True,
            "left_right_bottom_height_to_common_penetration": bool(contact_preserving),
            "per_foot_minimum_load_deficit": bool(contact_preserving),
            "warm_start_regularization": True,
        },
        "solver": {
            "method": "scipy.optimize.least_squares/trf",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "max_nfev": max_nfev,
            "root_qacc_scale": _ROOT_QACC_SCALE,
            "joint_qacc_scale": _JOINT_QACC_SCALE,
            "foot_height_scale": _FOOT_HEIGHT_SCALE if contact_preserving else None,
            "foot_load_deficit_scale": _FOOT_LOAD_DEFICIT_SCALE if contact_preserving else None,
            "regularization_scale": _REGULARIZATION_SCALE,
        },
        "classification": classification,
        "strict_initial_actual_equilibrium": classification == STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND,
        "strict_gate_passed": strict_gate_passed,
        "strict_nominal_hold": hold_rollout,
        "single_support_active_set": bool(snapshot["single_support_active_set"]),
        "qpos": [float(value) for value in qpos],
        "ctrl": [float(value) for value in ctrl],
        "vector": [float(value) for value in result.x],
        "snapshot": snapshot,
        "active_bounds": _active_bound_report(shard, result.x, lower, upper),
        "residual_breakdown": _residual_breakdown(
            shard,
            result.x,
            start,
            lower,
            upper,
            contact_preserving=contact_preserving,
        ),
        "infeasibility_certificate": {
            "classification": None,
            "reason": "No independent kinematic or wrench/actuation infeasibility certificate is produced by this search diagnostic.",
        },
    }


def _step_deltas(current: float, target: float, nominal_step: float, min_step: float) -> list[float]:
    remaining = abs(float(target) - float(current))
    if remaining <= 1e-12:
        return []
    base = min(float(nominal_step), remaining)
    deltas = []
    value = base
    while value + 1e-12 >= min(float(min_step), base):
        deltas.append(round(value, 10))
        value *= 0.5
    if not deltas:
        deltas.append(round(base, 10))
    return deltas


def _choose_attempt_result(
    qacc_only: dict[str, Any],
    contact_preserving: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    if qacc_only["strict_gate_passed"]:
        return qacc_only["phase"], qacc_only
    if contact_preserving is not None and contact_preserving["strict_gate_passed"]:
        return contact_preserving["phase"], contact_preserving
    if qacc_only["classification"] == SINGLE_SUPPORT_EQUILIBRIUM_ONLY_FOUND:
        return qacc_only["phase"], qacc_only
    if (
        contact_preserving is not None
        and contact_preserving["classification"] == SINGLE_SUPPORT_EQUILIBRIUM_ONLY_FOUND
    ):
        return contact_preserving["phase"], contact_preserving
    if contact_preserving is not None:
        return contact_preserving["phase"], contact_preserving
    return qacc_only["phase"], qacc_only


def _strict_source_state(
    records: list[dict[str, Any]],
    *,
    family: str,
    seed: int,
    range_fraction: float,
) -> dict[str, Any]:
    for record in records:
        if (
            record["family"] == family
            and int(record["seed"]) == int(seed)
            and _same_float(float(record["range_fraction"]), range_fraction)
        ):
            if not record.get("strict_contract_passed", False):
                raise ValueError(f"source {family}:rf{range_fraction:g}:seed{seed} is not strict")
            best = record["strict_refinement"]["best"]
            return {
                "record": record,
                "qpos": best["qpos"],
                "ctrl": best["ctrl"],
            }
    raise ValueError(f"missing strict source {family}:rf{range_fraction:g}:seed{seed}")


def _record_for(
    records: list[dict[str, Any]],
    *,
    family: str,
    seed: int,
    range_fraction: float,
) -> dict[str, Any]:
    for record in records:
        if (
            record["family"] == family
            and int(record["seed"]) == int(seed)
            and _same_float(float(record["range_fraction"]), range_fraction)
        ):
            return record
    raise ValueError(f"missing record {family}:rf{range_fraction:g}:seed{seed}")


def _direct_endpoint_diagnostic(
    records: list[dict[str, Any]],
    *,
    family: str,
    seed: int,
    range_fraction: float,
) -> dict[str, Any]:
    record = _record_for(records, family=family, seed=seed, range_fraction=range_fraction)
    shard, key = _build_shard_for(family, seed, range_fraction)
    best = record["strict_refinement"]["best"]
    qpos = best["qpos"]
    ctrl = best["ctrl"]
    snapshot = _diagnostic_snapshot(shard, qpos, ctrl)
    classification = classify_snapshot_without_certificates(snapshot)
    stance = _stance_qpos(shard)
    lower, upper = _vector_bounds(shard, stance)
    vector = shard.np.asarray(best["vector"], dtype=shard.np.float64)
    return {
        "source": "r4a3_direct_strict_refinement_best",
        "range_fraction": range_fraction,
        "morphology_instance_key": key,
        "r4a3_strict_refinement_status": record["strict_refinement"]["status"],
        "r4a3_strict_contract_passed": bool(record["strict_contract_passed"]),
        "classification": classification,
        "single_support_active_set": bool(snapshot["single_support_active_set"]),
        "strict_initial_actual_equilibrium": classification == STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND,
        "solver": {
            "nfev": int(best["attempt"]["nfev"]),
            "optimality": float(best["attempt"]["optimality"]),
            "success": bool(best["attempt"]["solver_success"]),
            "status": int(best["attempt"]["solver_status"]),
            "message": str(best["attempt"]["solver_message"]),
        },
        "snapshot": snapshot,
        "active_bounds": _active_bound_report(shard, vector, lower, upper),
    }


def run_continuation_route(
    *,
    source_records: list[dict[str, Any]],
    family: str,
    route: ContinuationRoute,
    nominal_step: float,
    min_step: float,
    max_nfev: int,
    horizon_steps: int,
) -> dict[str, Any]:
    source_state = _strict_source_state(
        source_records,
        family=family,
        seed=route.seed,
        range_fraction=route.start_range_fraction,
    )
    endpoint_direct = _direct_endpoint_diagnostic(
        source_records,
        family=family,
        seed=route.seed,
        range_fraction=route.end_range_fraction,
    )
    source_shard, source_key = _build_shard_for(family, route.seed, route.start_range_fraction)
    current_rf = float(route.start_range_fraction)
    current_qpos = source_state["qpos"]
    current_ctrl = source_state["ctrl"]
    attempts: list[dict[str, Any]] = []
    accepted_steps: list[dict[str, Any]] = []
    stopped_reason = None
    while route.direction * (route.end_range_fraction - current_rf) > 1e-12:
        accepted = False
        for bisection_level, delta in enumerate(
            _step_deltas(current_rf, route.end_range_fraction, nominal_step, min_step)
        ):
            target_rf = _rounded_range_fraction(current_rf + route.direction * delta)
            target_shard, target_key = _build_shard_for(family, route.seed, target_rf)
            mapping = warm_start_mapping_manifest(source_shard, target_shard)
            if not mapping["same_topology"]:
                raise ValueError(f"warm-start topology mapping changed for seed {route.seed}")
            stance = _stance_qpos(target_shard)
            lower, upper = _vector_bounds(target_shard, stance)
            warm_vector, warm_clip = _vector_from_qpos_ctrl(
                target_shard,
                current_qpos,
                current_ctrl,
                lower,
                upper,
            )
            qacc_only = refine_from_warm_start(
                target_shard,
                warm_vector,
                contact_preserving=False,
                max_nfev=max_nfev,
                horizon_steps=horizon_steps,
            )
            contact_preserving = None
            if not qacc_only["strict_gate_passed"]:
                contact_preserving = refine_from_warm_start(
                    target_shard,
                    warm_vector,
                    contact_preserving=True,
                    max_nfev=max_nfev,
                    horizon_steps=horizon_steps,
                )
            selected_phase, selected = _choose_attempt_result(qacc_only, contact_preserving)
            attempt = {
                "from_range_fraction": current_rf,
                "target_range_fraction": target_rf,
                "attempted_delta": delta,
                "bisection_level": bisection_level,
                "warm_start_source": {
                    "range_fraction": current_rf,
                    "strict_qpos_ctrl": True,
                },
                "target_morphology_instance_key": target_key,
                "model_xml_sha256": _sha256_bytes(target_shard.xml.encode("utf-8")),
                "warm_start_mapping": mapping,
                "warm_start_clip": warm_clip,
                "qacc_only_continuation": qacc_only,
                "contact_preserving_refinement": contact_preserving,
                "selected_phase": selected_phase,
                "classification": selected["classification"],
                "strict_gate_passed": bool(selected["strict_gate_passed"]),
                "single_support_active_set": bool(selected["single_support_active_set"]),
                "accepted": bool(selected["strict_gate_passed"]),
            }
            attempts.append(attempt)
            if selected["strict_gate_passed"]:
                current_rf = target_rf
                current_qpos = selected["qpos"]
                current_ctrl = selected["ctrl"]
                accepted = True
                accepted_steps.append(
                    {
                        "range_fraction": current_rf,
                        "phase": selected_phase,
                        "solver_nfev": selected["solver"]["nfev"],
                        "solver_optimality": selected["solver"]["optimality"],
                    }
                )
                break
        if not accepted:
            stopped_reason = "minimum_bisection_step_exhausted_without_strict_gate"
            break
    endpoint_reached = _same_float(current_rf, route.end_range_fraction)
    final_attempt = attempts[-1] if attempts else None
    if endpoint_reached:
        final_classification = STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND
    elif endpoint_direct["classification"] == SINGLE_SUPPORT_EQUILIBRIUM_ONLY_FOUND:
        final_classification = SINGLE_SUPPORT_EQUILIBRIUM_ONLY_FOUND
    elif final_attempt is not None:
        final_classification = final_attempt["classification"]
    else:
        final_classification = SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE
    return {
        "schema": "task067_r4a31_continuation_route_v1",
        "family": family,
        "seed": route.seed,
        "start_range_fraction": route.start_range_fraction,
        "end_range_fraction": route.end_range_fraction,
        "source_morphology_instance_key": source_key,
        "endpoint_label": route.endpoint_label,
        "direct_endpoint_diagnostic": endpoint_direct,
        "endpoint_reached": endpoint_reached,
        "stopped_at_range_fraction": current_rf,
        "stopped_reason": stopped_reason,
        "accepted_step_count": len(accepted_steps),
        "accepted_steps": accepted_steps,
        "attempt_count": len(attempts),
        "final_classification": final_classification,
        "endpoint_strict_gate_passed": endpoint_reached,
        "attempts": attempts,
    }


def _load_r4a3_records(path: Path, *, family: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(
        [record for record in data["records"] if record["family"] == family],
        key=lambda item: (float(item["range_fraction"]), int(item["seed"])),
    )


def _decide(summary: dict[str, Any]) -> dict[str, str]:
    if int(summary["combined_strict_contract_passed"]) == int(summary["combined_source_records"]):
        return {
            "status": "r4a31_restored_8_of_8",
            "decision": "Continuation recovered strict double-support equilibria for all three prior failures.",
            "next_allowed_work": "Enter R4a.3.2 on the 8 strict equilibria; still do not integrate controller.",
        }
    classifications = set(summary["endpoint_classifications"].values())
    if classifications & {KINEMATIC_DOUBLE_SUPPORT_INFEASIBLE, WRENCH_OR_ACTUATION_INFEASIBLE}:
        return {
            "status": "r4a31_physical_infeasibility_certificate_present",
            "decision": "At least one failed endpoint has an independent infeasibility certificate.",
            "next_allowed_work": "Consider the smallest generator grammar correction shared by the certified failures.",
        }
    if SINGLE_SUPPORT_EQUILIBRIUM_ONLY_FOUND in classifications:
        return {
            "status": "r4a31_single_support_or_search_failure",
            "decision": "At least one endpoint collapses to an exact or near-exact single-support equilibrium; this is not a generator infeasibility certificate.",
            "next_allowed_work": "Continue solver/contact-mode diagnosis; do not modify generator grammar or integrate controller.",
        }
    return {
        "status": "r4a31_search_exhausted_without_certificate",
        "decision": "Continuation did not restore full coverage, and no independent physical infeasibility certificate was produced.",
        "next_allowed_work": "Continue strict solver/contact-mode diagnosis; do not modify generator grammar or integrate controller.",
    }


def run_contact_preserving_continuation(
    *,
    input_json: Path,
    family: str,
    routes: tuple[ContinuationRoute, ...],
    nominal_step: float,
    min_step: float,
    max_nfev: int,
    horizon_steps: int,
) -> dict[str, Any]:
    source_records = _load_r4a3_records(input_json, family=family)
    route_rows = [
        run_continuation_route(
            source_records=source_records,
            family=family,
            route=route,
            nominal_step=nominal_step,
            min_step=min_step,
            max_nfev=max_nfev,
            horizon_steps=horizon_steps,
        )
        for route in routes
    ]
    source_accepted = {
        f"{record['family']}:rf{float(record['range_fraction']):g}:seed{int(record['seed'])}"
        for record in source_records
        if record.get("strict_contract_passed", False)
    }
    endpoint_recovered = {
        row["endpoint_label"] for row in route_rows if row["endpoint_strict_gate_passed"]
    }
    combined_accepted = source_accepted | endpoint_recovered
    summary = {
        "combined_source_records": len(source_records),
        "source_strict_contract_passed": len(source_accepted),
        "failed_endpoints_tested": len(route_rows),
        "endpoints_recovered_by_continuation": len(endpoint_recovered),
        "combined_strict_contract_passed": len(combined_accepted),
        "combined_accepted_labels": sorted(combined_accepted),
        "combined_incomplete_labels": [
            f"{record['family']}:rf{float(record['range_fraction']):g}:seed{int(record['seed'])}"
            for record in source_records
            if f"{record['family']}:rf{float(record['range_fraction']):g}:seed{int(record['seed'])}"
            not in combined_accepted
        ],
        "endpoint_classifications": {
            row["endpoint_label"]: row["final_classification"] for row in route_rows
        },
        "allowed_failure_classifications": list(_FAILURE_CLASSIFICATIONS),
        "infeasibility_certificates": {
            "kinematic_double_support_infeasible": 0,
            "wrench_or_actuation_infeasible": 0,
        },
    }
    return {
        "schema": "task067_r4a31_contact_preserving_continuation_v1",
        "source_artifact": str(input_json.resolve()),
        "provenance": {
            "source_artifact_sha256": _sha256_path(input_json),
            "diagnostic_source_sha256": _sha256_path(Path(__file__)),
            "dependency_source_sha256": {
                "whole_body_dynamic_balance_diagnosis.py": _sha256_path(
                    Path(__file__).with_name("whole_body_dynamic_balance_diagnosis.py")
                ),
                "whole_body_equilibrium_audit.py": _sha256_path(
                    Path(__file__).with_name("whole_body_equilibrium_audit.py")
                ),
                "whole_body_strict_equilibrium_coverage.py": _sha256_path(
                    Path(__file__).with_name("whole_body_strict_equilibrium_coverage.py")
                ),
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "mujoco": _package_version("mujoco"),
            "parameters": {
                "family": family,
                "routes": [
                    {
                        "seed": route.seed,
                        "start_range_fraction": route.start_range_fraction,
                        "end_range_fraction": route.end_range_fraction,
                    }
                    for route in routes
                ],
                "nominal_step": nominal_step,
                "min_step": min_step,
                "max_nfev": max_nfev,
                "horizon_steps": horizon_steps,
                "strict_root_qacc_norm": _STRICT_ROOT_QACC_NORM,
                "strict_joint_qacc_max": _STRICT_JOINT_QACC_MAX,
            },
        },
        "embodiment_contract_version": PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
        "embodiment_contract_hash": PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
        "stance_solution_contract_version": STANCE_SOLUTION_CONTRACT_VERSION,
        "stance_solution_contract_hash": STANCE_SOLUTION_CONTRACT_HASH,
        "summary": summary,
        "decision": _decide(summary),
        "routes": route_rows,
    }


def _routes_from_args(values: list[str] | None) -> tuple[ContinuationRoute, ...]:
    if not values:
        return _DEFAULT_ROUTES
    routes = []
    for value in values:
        seed_text, start_text, end_text = value.split(":", maxsplit=2)
        routes.append(
            ContinuationRoute(
                seed=int(seed_text),
                start_range_fraction=float(start_text),
                end_range_fraction=float(end_text),
            )
        )
    return tuple(routes)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=_DEFAULT_R4A3_INPUT)
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--family", default=_DEFAULT_FAMILY)
    parser.add_argument(
        "--route",
        action="append",
        help="Continuation route as seed:start_rf:end_rf. Defaults to the three R4a.3 failures.",
    )
    parser.add_argument("--range-step", type=float, default=_DEFAULT_STEP)
    parser.add_argument("--min-step", type=float, default=_DEFAULT_MIN_STEP)
    parser.add_argument("--max-nfev", type=int, default=1500)
    parser.add_argument("--horizon-steps", type=int, default=100)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    payload = run_contact_preserving_continuation(
        input_json=args.input_json,
        family=args.family,
        routes=_routes_from_args(args.route),
        nominal_step=args.range_step,
        min_step=args.min_step,
        max_nfev=args.max_nfev,
        horizon_steps=args.horizon_steps,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"decision": payload["decision"], "summary": payload["summary"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
