"""Independent Task067 audit of the R4a.2/R4b-2 causal chain.

This is a diagnostic-only tool.  It does not modify the public environment,
controller, actuator gains, reward, observations, action schema, or motor
fault path.  It checks the actual MuJoCo forward-dynamics equilibrium first,
then measures the existing private feedback mapping around that state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
from pathlib import Path
from typing import Any

from h200_locomotion_lab.tools.whole_body_bounded_feedback_diagnosis import (
    _GAIN_GRID,
    FeedbackGain,
    FeedbackMode,
    _build_shard,
    _controller_delta,
    _load_feasible_records,
)
from h200_locomotion_lab.tools.whole_body_dynamic_balance_diagnosis import (
    RolloutMode,
    _align_lowest_foot_to_penetration,
    _center_of_mass,
    _contact_report,
    _foot_bottom_heights,
    _quat_from_roll_pitch_yaw,
    _reset_to_qpos,
    _roll_pitch_yaw,
    _stance_qpos,
    _velocity_impulse_perturbations,
    run_rollout,
)
from h200_locomotion_lab.tools.whole_body_feedback_authority_diagnosis import (
    AuthorityProbe,
    _apply_probe_state,
    _axis_delta_vector,
    _bounded_ctrl,
    restoring_score,
)

_DEFAULT_INPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a2_joint_aware_equilibrium_diagnosis_4x2.json"
)
_DEFAULT_OUTPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4b2_independent_equilibrium_audit_5eq.json"
)
_STRICT_ROOT_QACC_NORM = 1e-5
_STRICT_JOINT_QACC_MAX = 1e-4
_ROOT_QACC_SCALE = 0.01
_JOINT_QACC_SCALE = 0.1
_REGULARIZATION_SCALE = 1e-6
_LOWER_BODY_JOINT_ADJUSTMENT = 0.08
_UPPER_BODY_JOINT_BOUND_POLICY = "compiled_physical_joint_limits"
_SELF_CONTACT_CLEARANCE = 0.002
_SELF_CONTACT_COUNT_RESIDUAL = 20.0
_SELF_CONTACT_CLEARANCE_RESIDUAL = 10.0
_SMALL_ANGLE_RAD = math.radians(0.1)
_FINITE_DIFFERENCE_DELTA = 1e-4


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _assert_free_root_contract(shard: Any) -> dict[str, int | str]:
    model = shard.model
    mujoco = shard.mujoco
    joint_type = int(model.jnt_type[0])
    expected = int(mujoco.mjtJoint.mjJNT_FREE)
    if joint_type != expected or int(model.jnt_qposadr[0]) != 0 or int(model.jnt_dofadr[0]) != 0:
        raise ValueError("Task067 authority DOFs require a leading MuJoCo free joint")
    return {
        "root_joint_type": "free",
        "root_qpos_address": int(model.jnt_qposadr[0]),
        "root_dof_address": int(model.jnt_dofadr[0]),
        "linear_qvel_qacc_dofs": "0:3",
        "angular_qvel_qacc_dofs": "3:6",
    }


def _state_snapshot(shard: Any, qpos: Any, ctrl: Any) -> dict[str, Any]:
    np = shard.np
    data = shard.data[0]
    _reset_to_qpos(shard, data, qpos)
    data.ctrl[:] = ctrl
    shard.mujoco.mj_forward(shard.model, data)
    qacc = np.asarray(data.qacc, dtype=np.float64)
    contact = _contact_report(shard, data)
    com = _center_of_mass(shard, data)
    cop = contact["center_of_pressure_xy"]
    com_minus_cop = None
    lipm_xy_qacc = None
    if cop is not None:
        com_minus_cop = [float(com[index]) - float(cop[index]) for index in range(2)]
        gravity = abs(float(shard.model.opt.gravity[2]))
        lipm_xy_qacc = [gravity * value / max(1e-12, float(com[2])) for value in com_minus_cop]
    total_mass = float(np.sum(shard.model.body_mass))
    weight = total_mass * abs(float(shard.model.opt.gravity[2]))
    minimum_load = 0.05 * weight
    joint_qacc = [abs(float(qacc[dof])) for dof in shard._joint_dof]
    actuator_saturation_events = 0
    for actuator_id in shard._actuator_ids:
        force = abs(float(data.actuator_force[int(actuator_id)]))
        limit = max(
            abs(float(value)) for value in shard.model.actuator_forcerange[int(actuator_id)]
        )
        actuator_saturation_events += int(force >= 0.995 * limit)
    double_support = all(
        int(contact["contacts_by_foot"].get(name, 0)) > 0
        and float(contact["normal_force_by_foot"].get(name, 0.0)) >= minimum_load
        for name in shard._foot_geoms
    )
    return {
        "root_qacc": [float(value) for value in qacc[:6]],
        "root_qacc_norm": float(np.linalg.norm(qacc[:6])),
        "joint_qacc_max": max(joint_qacc, default=0.0),
        "com": com,
        "center_of_pressure_xy": cop,
        "com_minus_cop_xy": com_minus_cop,
        "lipm_predicted_xy_qacc": lipm_xy_qacc,
        "contact": contact,
        "weight": weight,
        "minimum_foot_load": minimum_load,
        "double_support": double_support,
        "actuator_saturation_events": actuator_saturation_events,
    }


def strict_actual_equilibrium(snapshot: dict[str, Any]) -> bool:
    return bool(
        float(snapshot["root_qacc_norm"]) <= _STRICT_ROOT_QACC_NORM
        and float(snapshot["joint_qacc_max"]) <= _STRICT_JOINT_QACC_MAX
        and snapshot["double_support"]
        and int(snapshot["contact"]["non_foot_contacts"]) == 0
        and int(snapshot["contact"].get("self_contacts", 0)) == 0
        and int(snapshot["actuator_saturation_events"]) == 0
    )


def _diagnostic_joint_adjustment_limit(semantic_slot: str) -> float:
    if semantic_slot.startswith(("left_arm_", "right_arm_")):
        return float("inf")
    if semantic_slot.startswith("waist_"):
        return float("inf")
    return _LOWER_BODY_JOINT_ADJUSTMENT


def _diagnostic_joint_position_bounds(shard: Any, stance: Any) -> list[tuple[float, float]]:
    bounds = []
    for joint, qpos_address in zip(shard.blueprint.joints, shard._joint_qpos):
        joint_id = int(shard.mujoco.mj_name2id(shard.model, shard.mujoco.mjtObj.mjOBJ_JOINT, joint.name))
        lower, upper = (float(value) for value in shard.model.jnt_range[joint_id])
        center = float(stance[qpos_address])
        max_adjustment = _diagnostic_joint_adjustment_limit(joint.semantic_slot)
        if math.isinf(max_adjustment):
            bounds.append((lower, upper))
        else:
            bounds.append((max(lower, center - max_adjustment), min(upper, center + max_adjustment)))
    return bounds


def _upper_body_joint_indices(shard: Any) -> list[int]:
    return [
        index
        for index, joint in enumerate(shard.blueprint.joints)
        if joint.semantic_slot.startswith(("left_arm_", "right_arm_", "waist_"))
    ]


def _self_contact_residuals(contact: dict[str, Any]) -> list[float]:
    self_contacts = int(contact.get("self_contacts", 0))
    min_distance = contact.get("min_self_contact_distance")
    clearance_deficit = 0.0
    if self_contacts > 0:
        distance = 0.0 if min_distance is None else float(min_distance)
        clearance_deficit = max(0.0, _SELF_CONTACT_CLEARANCE - distance)
    return [
        _SELF_CONTACT_COUNT_RESIDUAL * float(self_contacts),
        _SELF_CONTACT_CLEARANCE_RESIDUAL
        * clearance_deficit
        / max(_SELF_CONTACT_CLEARANCE, 1e-12),
    ]


def _source_vector(shard: Any, source_equilibrium: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    np = shard.np
    stance = _stance_qpos(shard)
    qpos = np.asarray(source_equilibrium["best"]["qpos"], dtype=np.float64)
    ctrl = np.asarray(source_equilibrium["best"]["ctrl"], dtype=np.float64)
    _reset_to_qpos(shard, shard.data[0], qpos)
    foot_bottoms = _foot_bottom_heights(shard, shard.data[0])
    penetration = min(0.012, max(0.0, -min(foot_bottoms.values(), default=0.0)))
    roll, pitch, yaw = _roll_pitch_yaw(qpos[3:7])
    joints = [float(qpos[address]) for address in shard._joint_qpos]
    vector = np.asarray([roll, pitch, yaw, penetration, *joints, *ctrl], dtype=np.float64)

    joint_bounds = _diagnostic_joint_position_bounds(shard, stance)
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
    return vector, np.asarray(lower), np.asarray(upper), stance


def _apply_vector(shard: Any, stance: Any, vector: Any) -> tuple[Any, Any]:
    np = shard.np
    joint_count = len(shard._joint_qpos)
    qpos = np.asarray(stance, dtype=np.float64).copy()
    qpos[3:7] = _quat_from_roll_pitch_yaw(float(vector[0]), float(vector[1]), float(vector[2]))
    for address, value in zip(shard._joint_qpos, vector[4 : 4 + joint_count]):
        qpos[address] = float(value)
    ctrl = np.asarray(
        vector[4 + joint_count : 4 + joint_count + len(shard._actuator_ids)],
        dtype=np.float64,
    )
    _reset_to_qpos(shard, shard.data[0], qpos)
    _align_lowest_foot_to_penetration(shard, shard.data[0], penetration=float(vector[3]))
    shard.data[0].ctrl[:] = ctrl
    shard.mujoco.mj_forward(shard.model, shard.data[0])
    return np.asarray(shard.data[0].qpos).copy(), ctrl.copy()


def refine_actual_equilibrium(
    shard: Any,
    source_equilibrium: dict[str, Any],
    *,
    max_nfev: int,
) -> dict[str, Any]:
    """Refine only actual mj_forward qacc inside the original R4a.2 bounds."""

    np = shard.np
    try:
        from scipy import optimize
    except ImportError as exc:  # pragma: no cover - optional diagnostic dependency
        raise RuntimeError("SciPy is required for the Task067 equilibrium audit") from exc

    source, lower, upper, stance = _source_vector(shard, source_equilibrium)
    span = np.maximum(1e-6, upper - lower)
    joint_count = len(shard._joint_qpos)

    def residual(vector: Any) -> Any:
        _apply_vector(shard, stance, vector)
        qacc = np.asarray(shard.data[0].qacc, dtype=np.float64)
        contact = _contact_report(shard, shard.data[0])
        values = list(qacc[:6] / _ROOT_QACC_SCALE)
        values.extend(qacc[list(shard._joint_dof)] / _JOINT_QACC_SCALE)
        values.extend(_self_contact_residuals(contact))
        values.extend(_REGULARIZATION_SCALE * (np.asarray(vector) - source) / span)
        return np.asarray(values, dtype=np.float64)

    starts = [source.copy()]
    for index in range(1, 7):
        start = source.copy()
        start[0] += (-1.0 if index % 2 else 1.0) * 1e-4
        start[1] -= (-1.0 if index % 2 else 1.0) * 1e-4
        start[3] += (-1.0 if index % 2 else 1.0) * (index // 2 + 1) * 5e-5
        starts.append(np.clip(start, lower, upper))
    upper_body_indices = _upper_body_joint_indices(shard)
    for offset in (-0.18, 0.18):
        start = source.copy()
        for local_index in upper_body_indices:
            actuator_id = int(shard._actuator_ids[local_index])
            start[4 + local_index] += offset
            start[4 + joint_count + actuator_id] = start[4 + local_index]
        starts.append(np.clip(start, lower, upper))
    for bound in (lower, upper):
        start = source.copy()
        for local_index in upper_body_indices:
            actuator_id = int(shard._actuator_ids[local_index])
            start[4 + local_index] = float(bound[4 + local_index])
            start[4 + joint_count + actuator_id] = start[4 + local_index]
        starts.append(np.clip(start, lower, upper))
    if upper_body_indices:
        start = source.copy()
        for ordinal, local_index in enumerate(upper_body_indices):
            actuator_id = int(shard._actuator_ids[local_index])
            start[4 + local_index] += 0.18 if ordinal % 2 == 0 else -0.18
            start[4 + joint_count + actuator_id] = start[4 + local_index]
        starts.append(np.clip(start, lower, upper))

    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    for start_index, start in enumerate(starts):
        result = optimize.least_squares(
            residual,
            np.clip(start, lower, upper),
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
        snapshot = _state_snapshot(shard, qpos, ctrl)
        attempt = {
            "start_index": start_index,
            "solver_success": bool(result.success),
            "solver_status": int(result.status),
            "solver_message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "strict_actual_equilibrium": strict_actual_equilibrium(snapshot),
            "root_qacc_norm": snapshot["root_qacc_norm"],
            "joint_qacc_max": snapshot["joint_qacc_max"],
            "double_support": snapshot["double_support"],
            "non_foot_contacts": snapshot["contact"]["non_foot_contacts"],
            "self_contacts": snapshot["contact"].get("self_contacts", 0),
        }
        attempts.append(attempt)
        candidate = {
            "vector": [float(value) for value in result.x],
            "qpos": [float(value) for value in qpos],
            "ctrl": [float(value) for value in ctrl],
            "snapshot": snapshot,
            "attempt": attempt,
        }
        candidate_key = (
            float(snapshot["contact"].get("self_contacts", 0)),
            float(snapshot["contact"]["non_foot_contacts"]),
            0.0 if snapshot["double_support"] else 1.0,
            float(snapshot["root_qacc_norm"]),
            float(snapshot["joint_qacc_max"]),
        )
        if best is None or best_key is None or candidate_key < best_key:
            best = candidate
            best_key = candidate_key
        if attempt["strict_actual_equilibrium"]:
            best = candidate
            break

    assert best is not None
    return {
        "schema": "task067_strict_actual_equilibrium_refinement_v1",
        "strict_thresholds": {
            "root_qacc_norm_max": _STRICT_ROOT_QACC_NORM,
            "joint_qacc_max": _STRICT_JOINT_QACC_MAX,
            "double_support": True,
            "non_foot_contacts": 0,
            "self_contacts": 0,
            "actuator_saturation_events": 0,
        },
        "optimizer": {
            "method": "scipy.optimize.least_squares/trf",
            "max_nfev": max_nfev,
            "root_qacc_scale": _ROOT_QACC_SCALE,
            "joint_qacc_scale": _JOINT_QACC_SCALE,
            "regularization_scale": _REGULARIZATION_SCALE,
            "lower_body_joint_adjustment": _LOWER_BODY_JOINT_ADJUSTMENT,
            "upper_body_joint_bound_policy": _UPPER_BODY_JOINT_BOUND_POLICY,
            "self_contact_clearance": _SELF_CONTACT_CLEARANCE,
            "diff_step": 1e-5,
            "start_count_max": len(starts),
        },
        "attempts": attempts,
        "status": "feasible" if strict_actual_equilibrium(best["snapshot"]) else "infeasible",
        "best": best,
    }


def _root_qacc_for_delta(shard: Any, qpos: Any, ctrl_eq: Any, delta: Any) -> Any:
    np = shard.np
    _reset_to_qpos(shard, shard.data[0], qpos)
    ctrl, _ = _bounded_ctrl(shard, ctrl_eq, delta)
    shard.data[0].ctrl[:] = ctrl
    shard.mujoco.mj_forward(shard.model, shard.data[0])
    return np.asarray(shard.data[0].qacc[:6], dtype=np.float64).copy()


def _mapping_jacobian(shard: Any, equilibrium: dict[str, Any]) -> list[dict[str, Any]]:
    np = shard.np
    qpos = np.asarray(equilibrium["best"]["qpos"], dtype=np.float64)
    ctrl = np.asarray(equilibrium["best"]["ctrl"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for axis, qacc_dof in (("roll", 3), ("pitch", 4)):
        positive = _axis_delta_vector(shard, axis, _FINITE_DIFFERENCE_DELTA)
        plus = _root_qacc_for_delta(shard, qpos, ctrl, positive)
        minus = _root_qacc_for_delta(shard, qpos, ctrl, -positive)
        derivative = (plus - minus) / (2.0 * _FINITE_DIFFERENCE_DELTA)
        unit_mapping = np.asarray(_axis_delta_vector(shard, axis, 1.0))
        component_rows = []
        for joint, actuator_id in zip(shard.blueprint.joints, shard._actuator_ids):
            actuator_id = int(actuator_id)
            coefficient = float(unit_mapping[actuator_id])
            if abs(coefficient) <= 1e-12:
                continue
            component_delta = np.zeros(shard.model.nu, dtype=np.float64)
            component_delta[actuator_id] = _FINITE_DIFFERENCE_DELTA
            component_plus = _root_qacc_for_delta(shard, qpos, ctrl, component_delta)
            component_minus = _root_qacc_for_delta(shard, qpos, ctrl, -component_delta)
            component_derivative = (component_plus - component_minus) / (
                2.0 * _FINITE_DIFFERENCE_DELTA
            )
            component_rows.append(
                {
                    "joint_name": joint.name,
                    "semantic_slot": joint.semantic_slot,
                    "actuator_id": actuator_id,
                    "mapping_coefficient": coefficient,
                    "root_qacc_per_actuator_target_rad": [
                        float(value) for value in component_derivative
                    ],
                    "weighted_root_qacc_contribution": [
                        float(coefficient * value) for value in component_derivative
                    ],
                }
            )
        rows.append(
            {
                "axis": axis,
                "qacc_dof": qacc_dof,
                "root_qacc_per_mapping_command_rad": [float(value) for value in derivative],
                "desired_axis_derivative": float(derivative[qacc_dof]),
                "components": component_rows,
            }
        )
    return rows


def _evaluate_feedback_delta(
    shard: Any,
    equilibrium: dict[str, Any],
    probe: AuthorityProbe,
    delta: Any,
) -> dict[str, Any]:
    np = shard.np
    qpos = np.asarray(equilibrium["best"]["qpos"], dtype=np.float64)
    ctrl_eq = np.asarray(equilibrium["best"]["ctrl"], dtype=np.float64)
    _apply_probe_state(shard, shard.data[0], qpos, probe)
    ctrl, _ = _bounded_ctrl(shard, ctrl_eq, delta)
    shard.data[0].ctrl[:] = ctrl
    shard.mujoco.mj_forward(shard.model, shard.data[0])
    qacc = float(shard.data[0].qacc[probe.qacc_dof])
    contact = _contact_report(shard, shard.data[0])
    minimum_load = (
        0.05 * float(np.sum(shard.model.body_mass)) * abs(float(shard.model.opt.gravity[2]))
    )
    unloaded = [
        name
        for name in sorted(shard._foot_geoms)
        if float(contact["normal_force_by_foot"].get(name, 0.0)) < minimum_load
    ]
    return {
        "qacc": qacc,
        "restoring_score": restoring_score(perturb_sign=probe.sign, qacc_value=qacc),
        "root_qacc": [float(value) for value in shard.data[0].qacc[:6]],
        "unloaded_feet": unloaded,
    }


def _controller_delta_with_equilibrium_ctrl(
    shard: Any,
    qpos: Any,
    ctrl_eq: Any,
    mode: FeedbackMode,
    gain: FeedbackGain,
) -> Any:
    np = shard.np
    _reset_to_qpos(shard, shard.data[0], qpos)
    shard.data[0].ctrl[:] = ctrl_eq
    shard.mujoco.mj_forward(shard.model, shard.data[0])
    delta, _ = _controller_delta(shard, shard.data[0], mode, gain)
    return np.asarray(delta, dtype=np.float64)


def _small_angle_feedback_audit(
    shard: Any,
    equilibrium: dict[str, Any],
) -> list[dict[str, Any]]:
    np = shard.np
    qpos_eq = np.asarray(equilibrium["best"]["qpos"], dtype=np.float64)
    ctrl_eq = np.asarray(equilibrium["best"]["ctrl"], dtype=np.float64)
    high_gain = next(gain for gain in _GAIN_GRID if gain.name == "bounded_high")
    modes = (
        FeedbackMode("attitude_only", attitude=True),
        FeedbackMode("com_cop_oracle", com_cop_oracle=True),
        FeedbackMode("attitude_com_combined", attitude=True, com_cop_oracle=True),
    )
    reference = {
        mode.name: _controller_delta_with_equilibrium_ctrl(shard, qpos_eq, ctrl_eq, mode, high_gain)
        for mode in modes
    }
    rows: list[dict[str, Any]] = []
    for axis, qacc_dof in (("roll", 3), ("pitch", 4)):
        for sign in (-1, 1):
            probe = AuthorityProbe(
                kind="angle",
                axis=axis,
                sign=sign,
                value=_SMALL_ANGLE_RAD,
                qacc_dof=qacc_dof,
                control_axis=axis,
            )
            _apply_probe_state(shard, shard.data[0], qpos_eq, probe)
            shard.data[0].ctrl[:] = ctrl_eq
            shard.mujoco.mj_forward(shard.model, shard.data[0])
            perturbed_qpos = np.asarray(shard.data[0].qpos).copy()
            baseline = _evaluate_feedback_delta(shard, equilibrium, probe, np.zeros(shard.model.nu))
            responses: dict[str, Any] = {}
            for mode in modes:
                perturbed_delta = _controller_delta_with_equilibrium_ctrl(
                    shard, perturbed_qpos, ctrl_eq, mode, high_gain
                )
                absolute = _evaluate_feedback_delta(shard, equilibrium, probe, perturbed_delta)
                incremental = _evaluate_feedback_delta(
                    shard, equilibrium, probe, perturbed_delta - reference[mode.name]
                )
                responses[mode.name] = {
                    "equilibrium_reference_max_abs_delta": float(
                        np.max(np.abs(reference[mode.name]))
                    ),
                    "perturbed_max_abs_delta": float(np.max(np.abs(perturbed_delta))),
                    "absolute_restoring_improvement": float(
                        absolute["restoring_score"] - baseline["restoring_score"]
                    ),
                    "incremental_restoring_improvement": float(
                        incremental["restoring_score"] - baseline["restoring_score"]
                    ),
                    "incremental_root_qacc": incremental["root_qacc"],
                }
            rows.append(
                {
                    "probe": probe.manifest(),
                    "baseline": baseline,
                    "responses": responses,
                }
            )
    return rows


def _quaternion_probe_audit(shard: Any, equilibrium: dict[str, Any]) -> list[dict[str, Any]]:
    np = shard.np
    qpos_eq = np.asarray(equilibrium["best"]["qpos"], dtype=np.float64)
    rows = []
    for axis, dof in (("roll", 3), ("pitch", 4)):
        probe = AuthorityProbe("angle", axis, 1, math.radians(2.0), dof, axis)
        _apply_probe_state(shard, shard.data[0], qpos_eq, probe)
        euler_qpos = np.asarray(shard.data[0].qpos).copy()
        tangent_qpos = qpos_eq.copy()
        velocity = np.zeros(shard.model.nv, dtype=np.float64)
        velocity[dof] = probe.value
        shard.mujoco.mj_integratePos(shard.model, tangent_qpos, velocity, 1.0)
        euler_quat = euler_qpos[3:7] / np.linalg.norm(euler_qpos[3:7])
        tangent_quat = tangent_qpos[3:7] / np.linalg.norm(tangent_qpos[3:7])
        dot = min(1.0, max(-1.0, abs(float(np.dot(euler_quat, tangent_quat)))))
        rows.append(
            {
                "axis": axis,
                "qacc_dof": dof,
                "probe_rad": probe.value,
                "euler_vs_mj_integrate_orientation_error_rad": 2.0 * math.acos(dot),
            }
        )
    return rows


def _equilibrium_for_rollout(refinement: dict[str, Any]) -> dict[str, Any]:
    best = refinement["best"]
    return {
        "status": refinement["status"],
        "best": {"qpos": best["qpos"], "ctrl": best["ctrl"]},
    }


def _summarize_feedback(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for axis in ("roll", "pitch"):
        subset = [row for row in rows if row["probe"]["axis"] == axis]
        for mode in ("attitude_only", "com_cop_oracle", "attitude_com_combined"):
            improvements = [
                float(row["responses"][mode]["incremental_restoring_improvement"]) for row in subset
            ]
            summary[f"{axis}:{mode}"] = {
                "probes": len(improvements),
                "improves": sum(value > 1e-6 for value in improvements),
                "median_incremental_restoring_improvement": statistics.median(improvements),
            }
    return summary


def run_audit(
    *,
    input_json: Path,
    horizon_steps: int,
    long_horizon_steps: int,
    max_nfev: int,
) -> dict[str, Any]:
    records = _load_feasible_records(input_json, family="biped")
    output_rows: list[dict[str, Any]] = []
    all_feedback_rows: list[dict[str, Any]] = []
    for record in records:
        shard = _build_shard(record)
        free_root_contract = _assert_free_root_contract(shard)
        source_equilibrium = record["contact_equilibrium"]
        source_best = source_equilibrium["best"]
        source_snapshot = _state_snapshot(shard, source_best["qpos"], source_best["ctrl"])
        source_rollout = run_rollout(
            shard,
            RolloutMode("source_hold", contact_equilibrium_ctrl=True),
            equilibrium=source_equilibrium,
            horizon_steps=horizon_steps,
        )
        refinement = refine_actual_equilibrium(shard, source_equilibrium, max_nfev=max_nfev)
        refined_equilibrium = _equilibrium_for_rollout(refinement)
        refined_rollout = run_rollout(
            shard,
            RolloutMode("strict_refined_hold", contact_equilibrium_ctrl=True),
            equilibrium=refined_equilibrium,
            horizon_steps=horizon_steps,
        )
        refined_long_rollout = run_rollout(
            shard,
            RolloutMode("strict_refined_hold_long", contact_equilibrium_ctrl=True),
            equilibrium=refined_equilibrium,
            horizon_steps=long_horizon_steps,
        )
        perturbation_rollouts = [
            run_rollout(
                shard,
                RolloutMode("strict_refined_perturbation", contact_equilibrium_ctrl=True),
                equilibrium=refined_equilibrium,
                perturbation=perturbation,
                horizon_steps=horizon_steps,
            )
            for perturbation in _velocity_impulse_perturbations()
        ]
        feedback_rows = _small_angle_feedback_audit(shard, source_equilibrium)
        for item in feedback_rows:
            all_feedback_rows.append(
                {
                    "range_fraction": float(record["range_fraction"]),
                    "seed": int(record["seed"]),
                    **item,
                }
            )
        output_rows.append(
            {
                "family": record["family"],
                "range_fraction": float(record["range_fraction"]),
                "seed": int(record["seed"]),
                "morphology_instance_key": record["morphology_instance_key"],
                "model_xml_sha256": _sha256_bytes(shard.xml.encode("utf-8")),
                "simulation_timing": {
                    "mujoco_timestep": float(shard.model.opt.timestep),
                    "control_substeps": int(shard.config.substeps),
                    "control_timestep": float(shard.model.opt.timestep)
                    * int(shard.config.substeps),
                },
                "free_root_contract": free_root_contract,
                "source": {
                    "r4a2_status": source_equilibrium["status"],
                    "r4a2_qacc_thresholds": source_equilibrium["feasibility_thresholds"],
                    "actual_snapshot": source_snapshot,
                    "strict_actual_equilibrium": strict_actual_equilibrium(source_snapshot),
                    "nominal_rollout": source_rollout,
                },
                "strict_refinement": refinement,
                "strict_refined_nominal_rollout": refined_rollout,
                "strict_refined_long_rollout": refined_long_rollout,
                "strict_refined_perturbation_rollouts": perturbation_rollouts,
                "mapping_jacobian_at_source": _mapping_jacobian(shard, source_equilibrium),
                "small_angle_feedback_at_source": feedback_rows,
                "quaternion_probe_audit": _quaternion_probe_audit(shard, source_equilibrium),
            }
        )

    source_strict = sum(row["source"]["strict_actual_equilibrium"] for row in output_rows)
    refined_strict = sum(row["strict_refinement"]["status"] == "feasible" for row in output_rows)
    source_survived = sum(row["source"]["nominal_rollout"]["survived"] for row in output_rows)
    refined_survived = sum(row["strict_refined_nominal_rollout"]["survived"] for row in output_rows)
    refined_long_survived = sum(
        row["strict_refined_long_rollout"]["survived"] for row in output_rows
    )
    perturbations = [
        rollout for row in output_rows for rollout in row["strict_refined_perturbation_rollouts"]
    ]
    mapping_pitch = [
        mapping
        for row in output_rows
        for mapping in row["mapping_jacobian_at_source"]
        if mapping["axis"] == "pitch"
    ]
    mapping_roll = [
        mapping
        for row in output_rows
        for mapping in row["mapping_jacobian_at_source"]
        if mapping["axis"] == "roll"
    ]
    return {
        "schema": "task067_r4b2_independent_equilibrium_audit_v1",
        "source_artifact": str(input_json.resolve()),
        "provenance": {
            "source_artifact_sha256": _sha256_path(input_json),
            "diagnostic_source_sha256": _sha256_path(Path(__file__)),
            "dependency_source_sha256": {
                "whole_body_dynamic_balance_diagnosis.py": _sha256_path(
                    Path(__file__).with_name("whole_body_dynamic_balance_diagnosis.py")
                ),
                "whole_body_bounded_feedback_diagnosis.py": _sha256_path(
                    Path(__file__).with_name("whole_body_bounded_feedback_diagnosis.py")
                ),
                "whole_body_feedback_authority_diagnosis.py": _sha256_path(
                    Path(__file__).with_name("whole_body_feedback_authority_diagnosis.py")
                ),
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "mujoco": _package_version("mujoco"),
            "parameters": {
                "horizon_steps": horizon_steps,
                "long_horizon_steps": long_horizon_steps,
                "max_nfev": max_nfev,
                "small_angle_probe_rad": _SMALL_ANGLE_RAD,
                "finite_difference_delta": _FINITE_DIFFERENCE_DELTA,
            },
        },
        "summary": {
            "source_records": len(output_rows),
            "source_strict_actual_equilibria": source_strict,
            "strict_refined_equilibria": refined_strict,
            "source_nominal_survived": source_survived,
            "strict_refined_nominal_survived": refined_survived,
            "strict_refined_long_survived": refined_long_survived,
            "strict_refined_perturbations": len(perturbations),
            "strict_refined_perturbations_survived": sum(
                rollout["survived"] for rollout in perturbations
            ),
            "pitch_mapping_positive_desired_axis_derivative": sum(
                mapping["desired_axis_derivative"] > 0.0 for mapping in mapping_pitch
            ),
            "roll_mapping_positive_desired_axis_derivative": sum(
                mapping["desired_axis_derivative"] > 0.0 for mapping in mapping_roll
            ),
            "feedback": _summarize_feedback(all_feedback_rows),
        },
        "records": output_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--horizon-steps", type=int, default=100)
    parser.add_argument("--long-horizon-steps", type=int, default=500)
    parser.add_argument("--max-nfev", type=int, default=1500)
    args = parser.parse_args()
    result = run_audit(
        input_json=args.input_json,
        horizon_steps=args.horizon_steps,
        long_horizon_steps=args.long_horizon_steps,
        max_nfev=args.max_nfev,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
