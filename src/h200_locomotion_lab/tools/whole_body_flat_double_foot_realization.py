"""Task067 R4a.3.1e flat double-foot active-set realization.

This diagnostic removes the selected/unselected sole-corner ambiguity by
searching for a nominal stance where both footpad bottom faces are coplanar and
approximately horizontal.  Final acceptance uses only MuJoCo actual dynamics:
actual EFC/constraint closure, strict qacc, actual per-foot load, no forbidden
contacts, and a 2 second fixed hold.  Rigid wrench calculations are diagnostic
only.

The tool does not modify the public environment, controller, generator,
actuator gains, reward, observation/action schema, or motor process.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from h200_locomotion_lab.robots.procedural_morphology import (
    PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
    PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
)
from h200_locomotion_lab.robots.whole_body_stance import (
    STANCE_SOLUTION_CONTRACT_HASH,
    STANCE_SOLUTION_CONTRACT_VERSION,
)
from h200_locomotion_lab.tools.whole_body_contact_preserving_continuation import (
    STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND,
    _active_bound_report,
    _build_shard_for,
    _diagnostic_snapshot,
    _load_r4a3_records,
    _sha256_bytes,
    _sha256_path,
    classify_snapshot_without_certificates,
)
from h200_locomotion_lab.tools.whole_body_contact_taxonomy import (
    contact_taxonomy,
    efc_qfrc_for_contact_indices,
    full_efc_qfrc,
)
from h200_locomotion_lab.tools.whole_body_dynamic_balance_diagnosis import (
    _center_of_mass,
    _quat_from_roll_pitch_yaw,
    _reset_to_qpos,
    _roll_pitch_yaw,
    _stance_qpos,
)
from h200_locomotion_lab.tools.whole_body_equilibrium_audit import (
    _STRICT_JOINT_QACC_MAX,
    _STRICT_ROOT_QACC_NORM,
    strict_actual_equilibrium,
)
from h200_locomotion_lab.tools.whole_body_fixed_contact_mode_wrench_solve import (
    _foot_load_threshold,
    _input_feasibility,
    _input_tau_from_ctrl,
    _qfrc_from_contact_forces,
    _weight,
)
from h200_locomotion_lab.tools.whole_body_strict_equilibrium_coverage import (
    run_strict_hold_rollout,
)
from h200_locomotion_lab.tools.whole_body_true_continuation_correctness import (
    _compiled_joint_position_bounds,
)

_DEFAULT_INPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31d_contact_taxonomy_collision_free_strict_coverage_4x2.json"
)
_DEFAULT_OUTPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31e_flat_double_foot_active_set_realization.json"
)
_DEFAULT_FAMILY = "biped"
_DEFAULT_POSITIVE_LABEL = "biped:rf0:seed0"
_DEFAULT_INPUT_ONLY_LABEL = "biped:rf0:seed3"
_DEFAULT_LOCAL_MAX_NFEV = 450
_DEFAULT_FULL_MAX_NFEV = 900
_DEFAULT_HORIZON_STEPS = 100

_ROOT_ANGLE_BOUND = 0.15
_ROOT_YAW_BOUND = 0.20
_PENETRATION_MAX = 0.012
_LOCAL_TRUST_RADIUS = 0.08
_HEIGHT_SCALE = 0.001
_LOAD_DEFICIT_SCALE = 20.0
_FORBIDDEN_CONTACT_SCALE = 25.0
_BOUND_MARGIN = 0.01
_BOUND_MARGIN_SCALE = 0.05
_REGULARIZATION_SCALE = 1e-6
_RIGID_RESIDUAL_MAX = 1e-4
_RIGID_JOINT_TAU_MAX = 1e-3
_FLAT_HEIGHT_ABS_MAX = 1e-3
_FLAT_HEIGHT_SPREAD_MAX = 1e-3

COLLISION_FREE_STRICT_FOUND = "collision_free_strict_double_support_equilibrium_found"
ACTUAL_EQUILIBRIUM_FOUND_HOLD_FAILED = "actual_equilibrium_found_but_hold_failed"
GEOMETRY_AND_RIGID_WRENCH_FEASIBLE_ACTUAL_DYNAMICS_FAILED = (
    "geometry_and_rigid_wrench_feasible_actual_dynamics_failed"
)
FLAT_SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE = "flat_search_exhausted_without_certificate"
FLAT_PATCH_GEOMETRY_INFEASIBLE = "flat_patch_geometry_infeasible"

_CORNERS = (
    (-1.0, -1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (1.0, 1.0),
)


@dataclass(frozen=True, slots=True)
class StartState:
    name: str
    qpos: Any
    ctrl: Any


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _label(record: dict[str, Any]) -> str:
    return f"{record['family']}:rf{float(record['range_fraction']):g}:seed{int(record['seed'])}"


def _record_by_label(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for record in records:
        if _label(record) == label:
            return record
    raise ValueError(f"record {label} not found")


def _footpad_corners(shard: Any, data: Any) -> list[dict[str, Any]]:
    np = shard.np
    rows: list[dict[str, Any]] = []
    for foot in sorted(shard._foot_geoms):
        geom_id = int(shard.mujoco.mj_name2id(shard.model, shard.mujoco.mjtObj.mjOBJ_GEOM, foot))
        center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        half = np.asarray(shard.model.geom_size[geom_id], dtype=np.float64)[:3]
        body_id = int(shard.model.geom_bodyid[geom_id])
        for sx, sy in _CORNERS:
            local = np.asarray([sx * half[0], sy * half[1], -half[2]], dtype=np.float64)
            point = center + rot @ local
            rows.append(
                {
                    "foot": foot,
                    "sx": sx,
                    "sy": sy,
                    "key": f"{foot}:sx{sx:g}:sy{sy:g}",
                    "geom_id": geom_id,
                    "body_id": body_id,
                    "geom_name": foot,
                    "point": [float(value) for value in point],
                    "height": float(point[2]),
                }
            )
    return rows


def flat_patch_report(shard: Any, data: Any, *, penetration: float) -> dict[str, Any]:
    corners = _footpad_corners(shard, data)
    heights = [float(corner["height"]) for corner in corners]
    foot_reports = {}
    for foot in sorted(shard._foot_geoms):
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
        "nominal_flat_patch_target": "all footpad bottom corners at -penetration",
        "does_not_require_exact_mujoco_contact_count": True,
        "corner_count": len(corners),
        "corners": corners,
        "global_min_height": min(heights, default=float("inf")),
        "global_max_height": max(heights, default=float("-inf")),
        "global_height_spread": max(heights, default=0.0) - min(heights, default=0.0),
        "height_error_to_penetration_max_abs": max(
            (abs(height + penetration) for height in heights),
            default=float("inf"),
        ),
        "feet": foot_reports,
    }


def _align_flat_patch_to_penetration(shard: Any, data: Any, *, penetration: float) -> None:
    shard.mujoco.mj_forward(shard.model, data)
    corners = _footpad_corners(shard, data)
    if not corners:
        return
    lowest = min(float(corner["height"]) for corner in corners)
    data.qpos[2] += -float(penetration) - lowest
    shard.mujoco.mj_forward(shard.model, data)


def _baseline_ctrl(shard: Any, qpos: Any) -> Any:
    np = shard.np
    ctrl = np.zeros(shard.model.nu, dtype=np.float64)
    for actuator_id, address in zip(shard._actuator_ids, shard._joint_qpos):
        lower, upper = (float(value) for value in shard.model.actuator_ctrlrange[int(actuator_id)])
        ctrl[int(actuator_id)] = min(upper, max(lower, float(qpos[int(address)])))
    return ctrl


def _start_states(shard: Any, record: dict[str, Any]) -> list[StartState]:
    np = shard.np
    best = record["strict_refinement"]["best"]
    qpos_best = np.asarray(best["qpos"], dtype=np.float64)
    ctrl_best = np.asarray(best["ctrl"], dtype=np.float64)
    stance = np.asarray(_stance_qpos(shard), dtype=np.float64)
    return [
        StartState("r4a31d_best", qpos_best, ctrl_best),
        StartState("same_qpos_baseline_ctrl", qpos_best, _baseline_ctrl(shard, qpos_best)),
        StartState("target_r2_stance_baseline_ctrl", stance, _baseline_ctrl(shard, stance)),
    ]


def _vector_bounds(shard: Any, center_qpos: Any | None, *, stage: str) -> tuple[Any, Any]:
    np = shard.np
    lower = [-_ROOT_ANGLE_BOUND, -_ROOT_ANGLE_BOUND, -_ROOT_YAW_BOUND, 0.0]
    upper = [_ROOT_ANGLE_BOUND, _ROOT_ANGLE_BOUND, _ROOT_YAW_BOUND, _PENETRATION_MAX]
    compiled = _compiled_joint_position_bounds(shard)
    for (joint_lower, joint_upper), address in zip(compiled, shard._joint_qpos):
        if stage == "local_trust_region" and center_qpos is not None:
            center = float(center_qpos[int(address)])
            lower.append(max(joint_lower, center - _LOCAL_TRUST_RADIUS))
            upper.append(min(joint_upper, center + _LOCAL_TRUST_RADIUS))
        else:
            lower.append(joint_lower)
            upper.append(joint_upper)
    for actuator_id in shard._actuator_ids:
        ctrl_lower, ctrl_upper = (float(value) for value in shard.model.actuator_ctrlrange[int(actuator_id)])
        lower.append(ctrl_lower)
        upper.append(ctrl_upper)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _vector_from_start(shard: Any, start: StartState) -> Any:
    np = shard.np
    _reset_to_qpos(shard, shard.data[0], start.qpos)
    corners = _footpad_corners(shard, shard.data[0])
    penetration = min(_PENETRATION_MAX, max(0.0, -min(float(corner["height"]) for corner in corners)))
    roll, pitch, yaw = _roll_pitch_yaw(start.qpos[3:7])
    joints = [float(start.qpos[int(address)]) for address in shard._joint_qpos]
    ctrls = [float(start.ctrl[int(actuator_id)]) for actuator_id in shard._actuator_ids]
    return np.asarray([roll, pitch, yaw, penetration, *joints, *ctrls], dtype=np.float64)


def _apply_vector(shard: Any, template_qpos: Any, vector: Any) -> tuple[Any, Any]:
    np = shard.np
    joint_count = len(shard._joint_qpos)
    qpos = np.asarray(template_qpos, dtype=np.float64).copy()
    qpos[3:7] = _quat_from_roll_pitch_yaw(float(vector[0]), float(vector[1]), float(vector[2]))
    for address, value in zip(shard._joint_qpos, vector[4 : 4 + joint_count]):
        qpos[int(address)] = float(value)
    ctrl = np.zeros(shard.model.nu, dtype=np.float64)
    ctrl_values = vector[4 + joint_count : 4 + joint_count + len(shard._actuator_ids)]
    for actuator_id, value in zip(shard._actuator_ids, ctrl_values):
        ctrl[int(actuator_id)] = float(value)
    _reset_to_qpos(shard, shard.data[0], qpos)
    _align_flat_patch_to_penetration(shard, shard.data[0], penetration=float(vector[3]))
    shard.data[0].ctrl[:] = ctrl
    shard.mujoco.mj_forward(shard.model, shard.data[0])
    return np.asarray(shard.data[0].qpos, dtype=np.float64).copy(), ctrl.copy()


def _bound_margin_residuals(vector: Any, lower: Any, upper: Any) -> list[float]:
    values = []
    for index in range(4, len(vector)):
        margin = min(float(vector[index]) - float(lower[index]), float(upper[index]) - float(vector[index]))
        values.append(_BOUND_MARGIN_SCALE * max(0.0, _BOUND_MARGIN - margin) / _BOUND_MARGIN)
    return values


def _actual_efc_report(shard: Any, data: Any) -> dict[str, Any]:
    np = shard.np
    taxonomy = contact_taxonomy(shard, data)
    support_ids = {int(record["contact_index"]) for record in taxonomy["support_foot_floor_contacts"]}
    foot_qfrc = efc_qfrc_for_contact_indices(shard, data, support_ids)
    full_qfrc = full_efc_qfrc(shard, data)
    constraint = np.asarray(data.qfrc_constraint, dtype=np.float64).copy()
    diff = full_qfrc - constraint
    return {
        "support_foot_floor_efc_qfrc": [float(value) for value in foot_qfrc],
        "full_efc_qfrc": [float(value) for value in full_qfrc],
        "qfrc_constraint": [float(value) for value in constraint],
        "full_efc_vs_qfrc_constraint_max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "full_efc_vs_qfrc_constraint_norm": float(np.linalg.norm(diff)),
    }


def _rigid_wrench_diagnostic(shard: Any, data: Any, ctrl: Any) -> dict[str, Any]:
    np = shard.np
    try:
        from scipy import optimize
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for R4a.3.1e flat realization") from exc

    corners = _footpad_corners(shard, data)
    weight = _weight(shard)
    com = _center_of_mass(shard, data)
    min_load = _foot_load_threshold(shard)
    if not corners:
        return {"status": "infeasible", "reason": "no footpad corners", "forces": []}
    a_matrix = np.asarray(
        [
            [1.0 for _ in corners],
            [float(point["point"][0]) for point in corners],
            [float(point["point"][1]) for point in corners],
        ],
        dtype=np.float64,
    )
    rhs = np.asarray([weight, weight * com[0], weight * com[1]], dtype=np.float64)
    foot_indices = {
        foot: [index for index, point in enumerate(corners) if point["foot"] == foot]
        for foot in sorted(shard._foot_geoms)
    }
    initial = np.full(len(corners), weight / len(corners), dtype=np.float64)
    constraints = [
        {"type": "ineq", "fun": lambda force, indices=indices: float(np.sum(force[indices]) - min_load)}
        for indices in foot_indices.values()
    ]

    def objective(force: Any) -> float:
        force_array = np.asarray(force, dtype=np.float64)
        residual = a_matrix @ force_array - rhs
        qfrc_contact = _qfrc_from_contact_forces(shard, data, corners, force_array)
        qfrc_required = np.asarray(data.qfrc_bias, dtype=np.float64) - qfrc_contact
        input_tau = _input_tau_from_ctrl(shard, data, ctrl)
        joint_residual = np.asarray(
            [qfrc_required[int(dof)] - input_tau[int(dof)] for dof in shard._joint_dof],
            dtype=np.float64,
        )
        return float(
            np.dot(residual, residual) / max(weight * weight, 1e-12)
            + 1e-4 * np.dot(joint_residual, joint_residual) / 2500.0
            + 1e-8 * np.dot(force_array, force_array) / max(weight * weight, 1e-12)
        )

    result = optimize.minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 4.0 * weight) for _ in corners],
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 120, "disp": False},
    )
    forces = np.asarray(result.x if hasattr(result, "x") else initial, dtype=np.float64)
    residual = a_matrix @ forces - rhs
    qfrc_contact = _qfrc_from_contact_forces(shard, data, corners, forces)
    qfrc_required = np.asarray(data.qfrc_bias, dtype=np.float64) - qfrc_contact
    input_tau = _input_tau_from_ctrl(shard, data, ctrl)
    joint_residuals = [float(qfrc_required[int(dof)] - input_tau[int(dof)]) for dof in shard._joint_dof]
    normal_by_foot = {
        foot: float(np.sum(forces[indices])) for foot, indices in foot_indices.items()
    }
    load_deficit = {
        foot: max(0.0, min_load - load) for foot, load in normal_by_foot.items()
    }
    residual_norm = float(np.linalg.norm(residual))
    joint_tau_max = max((abs(value) for value in joint_residuals), default=0.0)
    status = (
        "feasible"
        if residual_norm <= _RIGID_RESIDUAL_MAX
        and joint_tau_max <= _RIGID_JOINT_TAU_MAX
        and sum(load_deficit.values()) <= 1e-8 * max(1.0, weight)
        else "incomplete"
    )
    return {
        "status": status,
        "method": "slsqp_full_footpad_vertical_wrench_diagnostic",
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "residual_norm": residual_norm,
        "residual": [float(value) for value in residual],
        "joint_tau_residual_max": joint_tau_max,
        "normal_force_sum": float(np.sum(forces)),
        "normal_force_by_foot": dict(sorted(normal_by_foot.items())),
        "minimum_foot_load": min_load,
        "load_deficit_by_foot": dict(sorted(load_deficit.items())),
        "input": {key: value for key, value in _input_feasibility(shard, data, ctrl).items() if key != "qfrc_actuator_from_ctrl"},
        "forces": [
            {
                "key": point["key"],
                "foot": point["foot"],
                "point": point["point"],
                "normal_force": float(force),
            }
            for point, force in zip(corners, forces)
        ],
    }


def _flat_geometry_realized(report: dict[str, Any]) -> bool:
    return bool(
        float(report["height_error_to_penetration_max_abs"]) <= _FLAT_HEIGHT_ABS_MAX
        and float(report["global_height_spread"]) <= _FLAT_HEIGHT_SPREAD_MAX
        and all(
            float(foot["height_spread"]) <= _FLAT_HEIGHT_SPREAD_MAX
            for foot in report["feet"].values()
        )
    )


def _solve_attempt(
    shard: Any,
    start: StartState,
    *,
    stage: str,
    max_nfev: int,
    horizon_steps: int,
) -> dict[str, Any]:
    np = shard.np
    try:
        from scipy import optimize
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for R4a.3.1e flat realization") from exc

    lower, upper = _vector_bounds(
        shard,
        start.qpos if stage == "local_trust_region" else None,
        stage=stage,
    )
    raw_start = _vector_from_start(shard, start)
    vector0 = np.clip(raw_start, lower, upper)
    span = np.maximum(1e-6, upper - lower)
    min_load = _foot_load_threshold(shard)

    def residual(vector: Any) -> Any:
        _apply_vector(shard, start.qpos, vector)
        data = shard.data[0]
        qacc = np.asarray(data.qacc, dtype=np.float64)
        taxonomy = contact_taxonomy(shard, data)
        patch = flat_patch_report(shard, data, penetration=float(vector[3]))
        values = list(qacc[:6] / _STRICT_ROOT_QACC_NORM)
        values.extend(qacc[list(shard._joint_dof)] / _STRICT_JOINT_QACC_MAX)
        values.extend(
            float(corner["height"] + float(vector[3])) / _HEIGHT_SCALE
            for corner in patch["corners"]
        )
        values.extend(
            _LOAD_DEFICIT_SCALE
            * max(0.0, min_load - float(taxonomy["normal_force_by_foot"].get(foot, 0.0)))
            / max(min_load, 1e-12)
            for foot in sorted(shard._foot_geoms)
        )
        values.append(_FORBIDDEN_CONTACT_SCALE * float(taxonomy["counts"]["forbidden_nonfoot_floor_contacts"]))
        values.append(_FORBIDDEN_CONTACT_SCALE * float(taxonomy["counts"]["self_contacts"]))
        values.extend(_bound_margin_residuals(vector, lower, upper))
        values.extend(_REGULARIZATION_SCALE * (np.asarray(vector) - vector0) / span)
        return np.asarray(values, dtype=np.float64)

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
    qpos, ctrl = _apply_vector(shard, start.qpos, result.x)
    data = shard.data[0]
    snapshot = _diagnostic_snapshot(shard, qpos, ctrl)
    patch = flat_patch_report(shard, data, penetration=float(result.x[3]))
    actual_classification = classify_snapshot_without_certificates(snapshot)
    hold = None
    strict_gate_passed = False
    if actual_classification == STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND:
        hold = run_strict_hold_rollout(
            shard,
            {"status": "feasible", "best": {"qpos": [float(value) for value in qpos], "ctrl": [float(value) for value in ctrl]}},
            horizon_steps=horizon_steps,
        )
        strict_gate_passed = bool(hold["passed"])
    rigid = _rigid_wrench_diagnostic(shard, data, ctrl)
    geometry_realized = _flat_geometry_realized(patch)
    if strict_gate_passed and geometry_realized:
        classification = COLLISION_FREE_STRICT_FOUND
    elif actual_classification == STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND and geometry_realized:
        classification = ACTUAL_EQUILIBRIUM_FOUND_HOLD_FAILED
    elif geometry_realized and rigid["status"] == "feasible":
        classification = GEOMETRY_AND_RIGID_WRENCH_FEASIBLE_ACTUAL_DYNAMICS_FAILED
    else:
        classification = FLAT_SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE
    joint_count = len(shard._joint_qpos)
    ctrl_count = len(shard._actuator_ids)
    return {
        "schema": "task067_r4a31e_flat_double_foot_attempt_v1",
        "stage": stage,
        "start_name": start.name,
        "solver": {
            "method": "scipy.optimize.least_squares/trf",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "max_nfev": max_nfev,
        },
        "variable_contract": {
            "state_variables": ["root_roll", "root_pitch", "root_yaw", "penetration", "actuated_joint_qpos"],
            "input_variables": ["actuator_position_ctrl"],
            "all_joints_use_compiled_physical_limits": stage == "full_limit_fallback",
            "local_trust_region_radius": _LOCAL_TRUST_RADIUS if stage == "local_trust_region" else None,
            "final_acceptance_requires_exactly_8_contacts": False,
        },
        "qpos": [float(value) for value in qpos],
        "ctrl": [float(value) for value in ctrl],
        "vector": [float(value) for value in result.x],
        "flat_patch": patch,
        "flat_geometry_realized": geometry_realized,
        "actual_snapshot": snapshot,
        "actual_efc": _actual_efc_report(shard, data),
        "rigid_wrench_diagnostic": rigid,
        "actual_classification": actual_classification,
        "classification": classification,
        "strict_gate_passed": bool(strict_gate_passed and geometry_realized),
        "actual_strict_gate_and_hold_passed": strict_gate_passed,
        "strict_nominal_hold_2s": hold,
        "active_bounds": _active_bound_report(
            shard,
            result.x[: 4 + joint_count + ctrl_count],
            lower[: 4 + joint_count + ctrl_count],
            upper[: 4 + joint_count + ctrl_count],
        ),
        "infeasibility_certificate": {
            "classification": None,
            "reason": "Nonlinear flat-patch search failure is not promoted to geometry or wrench infeasibility.",
        },
    }


def _attempt_sort_key(attempt: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    snapshot = attempt["actual_snapshot"]
    patch = attempt["flat_patch"]
    return (
        0.0 if attempt["strict_gate_passed"] else 1.0,
        0.0 if attempt["flat_geometry_realized"] else 1.0,
        0.0 if attempt["rigid_wrench_diagnostic"]["status"] == "feasible" else 1.0,
        float(snapshot["root_qacc_norm"]) + float(snapshot["joint_qacc_max"]),
        float(patch["height_error_to_penetration_max_abs"]),
        float(patch["global_height_spread"]),
    )


def _followup_start_from_attempt(shard: Any, attempt: dict[str, Any]) -> StartState:
    np = shard.np
    return StartState(
        f"{attempt['stage']}_{attempt['start_name']}_best",
        np.asarray(attempt["qpos"], dtype=np.float64),
        np.asarray(attempt["ctrl"], dtype=np.float64),
    )


def solve_flat_record(
    record: dict[str, Any],
    *,
    role: str,
    local_max_nfev: int,
    full_max_nfev: int,
    horizon_steps: int,
) -> dict[str, Any]:
    shard, key = _build_shard_for(record["family"], int(record["seed"]), float(record["range_fraction"]))
    starts = _start_states(shard, record)
    local_attempts = [
        _solve_attempt(
            shard,
            start,
            stage="local_trust_region",
            max_nfev=local_max_nfev,
            horizon_steps=horizon_steps,
        )
        for start in starts[:2]
    ]
    if any(attempt["strict_gate_passed"] for attempt in local_attempts):
        full_attempts: list[dict[str, Any]] = []
    else:
        local_best = min(local_attempts, key=_attempt_sort_key)
        full_starts = [_followup_start_from_attempt(shard, local_best), *starts]
        full_attempts = [
            _solve_attempt(
                shard,
                start,
                stage="full_limit_fallback",
                max_nfev=full_max_nfev,
                horizon_steps=horizon_steps,
            )
            for start in full_starts
        ]
    attempts = [*local_attempts, *full_attempts]
    best = min(attempts, key=_attempt_sort_key)
    return {
        "schema": "task067_r4a31e_flat_double_foot_record_v1",
        "role": role,
        "label": _label(record),
        "family": record["family"],
        "seed": int(record["seed"]),
        "range_fraction": float(record["range_fraction"]),
        "morphology_instance_key": key,
        "model_xml_sha256": _sha256_bytes(shard.xml.encode("utf-8")),
        "source_strict_contract_passed": bool(record.get("strict_contract_passed", False)),
        "attempt_count": len(attempts),
        "full_limit_fallback_executed": bool(full_attempts),
        "strict_recovered": bool(best["strict_gate_passed"]),
        "final_classification": best["classification"],
        "best": best,
        "attempt_preview": [
            {
                "stage": attempt["stage"],
                "start_name": attempt["start_name"],
                "classification": attempt["classification"],
                "strict_gate_passed": attempt["strict_gate_passed"],
                "flat_geometry_realized": attempt["flat_geometry_realized"],
                "rigid_wrench_status": attempt["rigid_wrench_diagnostic"]["status"],
                "nfev": attempt["solver"]["nfev"],
                "optimality": attempt["solver"]["optimality"],
                "root_qacc_norm": attempt["actual_snapshot"]["root_qacc_norm"],
                "joint_qacc_max": attempt["actual_snapshot"]["joint_qacc_max"],
                "support_mode": attempt["actual_snapshot"]["support_mode"],
                "self_contacts": attempt["actual_snapshot"]["contact"].get("self_contacts", 0),
                "non_foot_contacts": attempt["actual_snapshot"]["contact"]["non_foot_contacts"],
                "flat_height_error_max_abs": attempt["flat_patch"]["height_error_to_penetration_max_abs"],
                "flat_height_spread": attempt["flat_patch"]["global_height_spread"],
            }
            for attempt in sorted(attempts, key=_attempt_sort_key)
        ],
        "infeasibility_certificate": {
            "classification": None,
            "reason": "No independent flat-patch geometry or wrench/actuation infeasibility certificate is emitted.",
        },
    }


def input_only_probe(
    record: dict[str, Any],
    *,
    max_nfev: int,
    horizon_steps: int,
) -> dict[str, Any]:
    np = __import__("numpy")
    try:
        from scipy import optimize
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for R4a.3.1e input-only probe") from exc

    shard, key = _build_shard_for(record["family"], int(record["seed"]), float(record["range_fraction"]))
    best = record["strict_refinement"]["best"]
    qpos = np.asarray(best["qpos"], dtype=np.float64)
    ctrl0 = np.asarray(best["ctrl"], dtype=np.float64)
    actuator_ids = [int(actuator_id) for actuator_id in shard._actuator_ids]
    lower = np.asarray([float(shard.model.actuator_ctrlrange[aid][0]) for aid in actuator_ids], dtype=np.float64)
    upper = np.asarray([float(shard.model.actuator_ctrlrange[aid][1]) for aid in actuator_ids], dtype=np.float64)
    start = np.clip(np.asarray([float(ctrl0[aid]) for aid in actuator_ids], dtype=np.float64), lower, upper)
    span = np.maximum(1e-6, upper - lower)

    def apply(ctrl_values: Any) -> Any:
        ctrl = np.zeros(shard.model.nu, dtype=np.float64)
        for actuator_id, value in zip(actuator_ids, ctrl_values):
            ctrl[actuator_id] = float(value)
        _reset_to_qpos(shard, shard.data[0], qpos)
        shard.data[0].ctrl[:] = ctrl
        shard.mujoco.mj_forward(shard.model, shard.data[0])
        return ctrl

    before = _diagnostic_snapshot(shard, qpos, ctrl0)

    def residual(ctrl_values: Any) -> Any:
        apply(ctrl_values)
        qacc = np.asarray(shard.data[0].qacc, dtype=np.float64)
        values = list(qacc[:6] / _STRICT_ROOT_QACC_NORM)
        values.extend(qacc[list(shard._joint_dof)] / _STRICT_JOINT_QACC_MAX)
        values.extend(_REGULARIZATION_SCALE * (np.asarray(ctrl_values) - start) / span)
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
    ctrl = apply(result.x)
    after = _diagnostic_snapshot(shard, qpos, ctrl)
    hold = None
    strict_gate_passed = False
    if strict_actual_equilibrium(after):
        hold = run_strict_hold_rollout(
            shard,
            {"status": "feasible", "best": {"qpos": [float(value) for value in qpos], "ctrl": [float(value) for value in ctrl]}},
            horizon_steps=horizon_steps,
        )
        strict_gate_passed = bool(hold["passed"])
    return {
        "schema": "task067_r4a31e_input_only_probe_v1",
        "label": _label(record),
        "morphology_instance_key": key,
        "fixed_qpos": True,
        "solves_ctrl_only": True,
        "before_snapshot": before,
        "after_snapshot": after,
        "strict_gate_passed": strict_gate_passed,
        "strict_nominal_hold_2s": hold,
        "solver": {
            "method": "scipy.optimize.least_squares/trf",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "max_nfev": max_nfev,
        },
        "ctrl": [float(value) for value in ctrl],
        "conclusion": "input_only_recovered_strict" if strict_gate_passed else "input_only_not_sufficient",
    }


def _decide(summary: dict[str, Any]) -> dict[str, str]:
    if int(summary["combined_collision_free_strict_passed"]) == int(summary["source_records"]):
        return {
            "status": "r4a31e_collision_free_strict_coverage_restored_8_of_8",
            "decision": "Flat double-foot realization recovered collision-free strict equilibria for the full 4x2 set.",
            "next_allowed_work": "Design StanceSolutionV3(qpos_eq, ctrl_eq); still do not integrate feedback.",
        }
    if int(summary["geometry_infeasibility_certificates"]) > 0:
        return {
            "status": "r4a31e_flat_patch_geometry_infeasible_certificate_found",
            "decision": "At least one endpoint has an explicit flat-patch geometry infeasibility certificate.",
            "next_allowed_work": "Only now consider minimal generator grammar changes driven by the shared certified feature.",
        }
    if int(summary["actual_equilibrium_found_hold_failed"]) > 0:
        return {
            "status": "r4a31e_actual_equilibrium_found_hold_failed",
            "decision": "At least one endpoint reaches actual equilibrium but fails hold.",
            "next_allowed_work": "Re-enter closed-loop stability/feedback diagnosis only for those collision-free actual equilibria.",
        }
    if int(summary["geometry_and_rigid_wrench_feasible_actual_dynamics_failed"]) > 0:
        return {
            "status": "r4a31e_compliant_contact_realization_blocker",
            "decision": "Flat geometry and rigid wrench are feasible for at least one failed endpoint, but actual MuJoCo dynamics still fail.",
            "next_allowed_work": "Diagnose compliant contact realization; do not modify generator or feedback.",
        }
    return {
        "status": "r4a31e_flat_search_incomplete_no_physical_certificate",
        "decision": "Flat double-foot realization did not restore full coverage and produced no physical infeasibility certificate.",
        "next_allowed_work": "Continue fixed double-foot actual-contact refinement; do not restore R4b or Task061/062.",
    }


def run_flat_double_foot_realization(
    *,
    input_json: Path,
    family: str,
    positive_label: str,
    input_only_label: str,
    local_max_nfev: int,
    full_max_nfev: int,
    horizon_steps: int,
) -> dict[str, Any]:
    records = _load_r4a3_records(input_json, family=family)
    failed_records = [record for record in records if not bool(record.get("strict_contract_passed", False))]
    positive_record = _record_by_label(records, positive_label)
    input_only_record = _record_by_label(records, input_only_label)
    rows = [
        solve_flat_record(
            record,
            role="failed_endpoint",
            local_max_nfev=local_max_nfev,
            full_max_nfev=full_max_nfev,
            horizon_steps=horizon_steps,
        )
        for record in failed_records
    ]
    positive_control = solve_flat_record(
        positive_record,
        role="positive_control",
        local_max_nfev=local_max_nfev,
        full_max_nfev=full_max_nfev,
        horizon_steps=horizon_steps,
    )
    input_probe = input_only_probe(
        input_only_record,
        max_nfev=min(full_max_nfev, 250),
        horizon_steps=horizon_steps,
    )
    recovered = {row["label"] for row in rows if row["strict_recovered"]}
    source_accepted = {
        _label(record) for record in records if bool(record.get("strict_contract_passed", False))
    }
    combined = source_accepted | recovered
    class_counts = {}
    for row in rows:
        class_counts[row["final_classification"]] = class_counts.get(row["final_classification"], 0) + 1
    summary = {
        "source_records": len(records),
        "source_collision_free_strict_passed": len(source_accepted),
        "failed_endpoints_tested": len(rows),
        "failed_endpoint_labels": [_label(record) for record in failed_records],
        "positive_controls_tested": 1,
        "positive_controls_passed_same_path": int(positive_control["strict_recovered"]),
        "endpoints_recovered": len(recovered),
        "combined_collision_free_strict_passed": len(combined),
        "combined_accepted_labels": sorted(combined),
        "combined_incomplete_labels": [_label(record) for record in records if _label(record) not in combined],
        "endpoint_classifications": {row["label"]: row["final_classification"] for row in rows},
        "endpoint_classification_counts": class_counts,
        "full_limit_fallback_executed_count": sum(bool(row["full_limit_fallback_executed"]) for row in rows),
        "geometry_infeasibility_certificates": sum(
            row["final_classification"] == FLAT_PATCH_GEOMETRY_INFEASIBLE for row in rows
        ),
        "geometry_and_rigid_wrench_feasible_actual_dynamics_failed": sum(
            row["final_classification"] == GEOMETRY_AND_RIGID_WRENCH_FEASIBLE_ACTUAL_DYNAMICS_FAILED
            for row in rows
        ),
        "actual_equilibrium_found_hold_failed": sum(
            row["final_classification"] == ACTUAL_EQUILIBRIUM_FOUND_HOLD_FAILED for row in rows
        ),
        "input_only_probe_label": input_only_label,
        "input_only_probe_conclusion": input_probe["conclusion"],
        "infeasibility_certificates": {
            "flat_patch_geometry_infeasible": 0,
            "wrench_or_actuation_infeasible": 0,
        },
    }
    return {
        "schema": "task067_r4a31e_flat_double_foot_active_set_realization_v1",
        "source_artifact": str(input_json.resolve()),
        "provenance": {
            "source_artifact_sha256": _sha256_path(input_json),
            "diagnostic_source_sha256": _sha256_path(Path(__file__)),
            "dependency_source_sha256": {
                "whole_body_contact_taxonomy.py": _sha256_path(Path(__file__).with_name("whole_body_contact_taxonomy.py")),
                "whole_body_strict_equilibrium_coverage.py": _sha256_path(Path(__file__).with_name("whole_body_strict_equilibrium_coverage.py")),
                "whole_body_equilibrium_audit.py": _sha256_path(Path(__file__).with_name("whole_body_equilibrium_audit.py")),
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "mujoco": _package_version("mujoco"),
            "parameters": {
                "family": family,
                "positive_label": positive_label,
                "input_only_label": input_only_label,
                "local_max_nfev": local_max_nfev,
                "full_max_nfev": full_max_nfev,
                "horizon_steps": horizon_steps,
                "hold_seconds": horizon_steps / 50.0,
                "local_trust_radius": _LOCAL_TRUST_RADIUS,
                "full_limit_fallback_required_after_local_failure": True,
                "strict_root_qacc_norm": _STRICT_ROOT_QACC_NORM,
                "strict_joint_qacc_max": _STRICT_JOINT_QACC_MAX,
                "final_acceptance": {
                    "actual_mujoco_efc": True,
                    "actual_qacc": True,
                    "actual_per_foot_load": True,
                    "two_second_hold": True,
                    "exactly_8_contact_points_required": False,
                    "forbidden_nonfoot_floor_contacts": 0,
                    "self_contacts": 0,
                },
            },
            "diagnostic_scope": {
                "flat_double_foot_patch": True,
                "joint_qpos_ctrl_penetration_jointly_optimized": True,
                "rigid_wrench_is_diagnostic_only": True,
                "modifies_public_env_controller_generator_or_gains": False,
            },
        },
        "embodiment_contract_version": PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
        "embodiment_contract_hash": PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
        "stance_solution_contract_version": STANCE_SOLUTION_CONTRACT_VERSION,
        "stance_solution_contract_hash": STANCE_SOLUTION_CONTRACT_HASH,
        "summary": summary,
        "decision": _decide(summary),
        "assertions": {
            "all_four_failed_endpoints_tested": len(rows) == 4,
            "positive_control_same_path_tested": positive_control["role"] == "positive_control",
            "final_acceptance_does_not_require_exactly_8_contacts": True,
            "full_limit_fallback_after_local_failure_available": True,
            "search_failure_not_promoted_to_physical_infeasible": (
                int(summary["infeasibility_certificates"]["flat_patch_geometry_infeasible"]) == 0
                and int(summary["infeasibility_certificates"]["wrench_or_actuation_infeasible"]) == 0
            ),
        },
        "input_only_probe": input_probe,
        "positive_control": positive_control,
        "endpoints": rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--family", default=_DEFAULT_FAMILY)
    parser.add_argument("--positive-label", default=_DEFAULT_POSITIVE_LABEL)
    parser.add_argument("--input-only-label", default=_DEFAULT_INPUT_ONLY_LABEL)
    parser.add_argument("--local-max-nfev", type=int, default=_DEFAULT_LOCAL_MAX_NFEV)
    parser.add_argument("--full-max-nfev", type=int, default=_DEFAULT_FULL_MAX_NFEV)
    parser.add_argument("--horizon-steps", type=int, default=_DEFAULT_HORIZON_STEPS)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    payload = run_flat_double_foot_realization(
        input_json=args.input_json,
        family=args.family,
        positive_label=args.positive_label,
        input_only_label=args.input_only_label,
        local_max_nfev=args.local_max_nfev,
        full_max_nfev=args.full_max_nfev,
        horizon_steps=args.horizon_steps,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "assertions": payload["assertions"],
                "decision": payload["decision"],
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
