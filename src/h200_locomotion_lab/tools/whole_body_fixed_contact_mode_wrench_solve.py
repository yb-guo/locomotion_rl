"""Task067 R4a.3.1b fixed double-foot contact-mode wrench solve.

This diagnostic enters the next branch after true-continuation did not recover
all strict biped equilibria.  It solves a fixed double-foot state-input-wrench
constraint problem for the three failed R4a.3 endpoints.  Contact normal forces
are explicit optimization variables, both feet are kept in the active set, and
the final acceptance still requires the R4a.3 strict actual MuJoCo gate plus a
2 second fixed ``qpos_eq + ctrl_eq`` hold.

The tool is diagnostic-only.  It does not modify the public environment,
controller, generator, actuator gains, reward, observation/action schema, or
motor process.
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
    SINGLE_SUPPORT_EQUILIBRIUM_ONLY_FOUND,
    STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND,
    ContinuationRoute,
    _active_bound_report,
    _build_shard_for,
    _diagnostic_snapshot,
    _direct_endpoint_diagnostic,
    _load_r4a3_records,
    _record_for,
    _sha256_bytes,
    _sha256_path,
    _strict_source_state,
    classify_snapshot_without_certificates,
)
from h200_locomotion_lab.tools.whole_body_dynamic_balance_diagnosis import (
    _center_of_mass,
    _foot_bottom_heights,
    _quat_from_roll_pitch_yaw,
    _reset_to_qpos,
    _roll_pitch_yaw,
    _stance_qpos,
)
from h200_locomotion_lab.tools.whole_body_equilibrium_audit import (
    _STRICT_JOINT_QACC_MAX,
    _STRICT_ROOT_QACC_NORM,
)
from h200_locomotion_lab.tools.whole_body_strict_equilibrium_coverage import (
    run_strict_hold_rollout,
)
from h200_locomotion_lab.tools.whole_body_true_continuation_correctness import (
    _compiled_joint_position_bounds,
    true_warm_start_from_previous_solution,
)

_DEFAULT_R4A3_INPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a3_strict_equilibrium_coverage_4x2.json"
)
_DEFAULT_R4A31A_INPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31a_true_continuation_correctness_3fail.json"
)
_DEFAULT_OUTPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31b_fixed_contact_mode_wrench_solve_3fail.json"
)
_DEFAULT_FAMILY = "biped"
_DEFAULT_MAX_NFEV = 350
_DEFAULT_HORIZON_STEPS = 100
_ROOT_WRENCH_SCALE = 50.0
_JOINT_TAU_SCALE = 50.0
_HEIGHT_SCALE = 0.001
_LOAD_DEFICIT_SCALE = 20.0
_FORCE_SUM_SCALE = 1.0
_REGULARIZATION_SCALE = 1e-5
_RIGID_ROOT_WRENCH_NORM_MAX = 1e-4
_RIGID_JOINT_TAU_RESIDUAL_MAX = 1e-4
_RIGID_SELECTED_HEIGHT_ABS_MAX = 5e-4
_RIGID_FORCE_BALANCE_REL_MAX = 1e-5
_FOOT_LOAD_FRACTION = 0.05

FIXED_CONTACT_WRENCH_SOLUTION_ACTUAL_GATE_FAILED = (
    "fixed_contact_wrench_solution_found_actual_gate_failed"
)
FIXED_CONTACT_MODE_SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE = (
    "fixed_contact_mode_search_exhausted_without_certificate"
)

_DEFAULT_ROUTES = (
    ContinuationRoute(seed=0, start_range_fraction=0.0, end_range_fraction=0.5),
    ContinuationRoute(seed=3, start_range_fraction=0.0, end_range_fraction=0.5),
    ContinuationRoute(seed=1, start_range_fraction=0.5, end_range_fraction=0.0),
)
_CORNERS = (
    (-1.0, -1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (1.0, 1.0),
)


@dataclass(frozen=True, slots=True)
class ContactPointSpec:
    foot: str
    sx: float
    sy: float

    @property
    def key(self) -> str:
        return f"{self.foot}:sx{self.sx:g}:sy{self.sy:g}"

    def manifest(self) -> dict[str, Any]:
        return {"foot": self.foot, "sx": self.sx, "sy": self.sy}


@dataclass(frozen=True, slots=True)
class FixedContactMode:
    name: str
    points: tuple[ContactPointSpec, ...]

    @property
    def foot_names(self) -> tuple[str, ...]:
        return tuple(sorted({point.foot for point in self.points}))

    @property
    def is_double_foot(self) -> bool:
        return len(self.foot_names) == 2

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "fixed_double_foot_contact_mode",
            "point_count": len(self.points),
            "foot_names": list(self.foot_names),
            "points": [point.manifest() for point in self.points],
        }


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


def _foot_load_threshold(shard: Any) -> float:
    total_mass = float(shard.np.sum(shard.model.body_mass))
    return _FOOT_LOAD_FRACTION * total_mass * abs(float(shard.model.opt.gravity[2]))


def _weight(shard: Any) -> float:
    return float(shard.np.sum(shard.model.body_mass)) * abs(float(shard.model.opt.gravity[2]))


def _footpad_corner_points(
    shard: Any,
    data: Any,
    foot: str,
) -> list[dict[str, Any]]:
    np = shard.np
    geom_id = int(shard.mujoco.mj_name2id(shard.model, shard.mujoco.mjtObj.mjOBJ_GEOM, foot))
    center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    half = np.asarray(shard.model.geom_size[geom_id], dtype=np.float64)[:3]
    body_id = int(shard.model.geom_bodyid[geom_id])
    rows = []
    for sx, sy in _CORNERS:
        local = np.asarray([sx * half[0], sy * half[1], -half[2]], dtype=np.float64)
        point = center + rot @ local
        rows.append(
            {
                "foot": foot,
                "sx": sx,
                "sy": sy,
                "key": ContactPointSpec(foot, sx, sy).key,
                "geom_id": geom_id,
                "body_id": body_id,
                "point": [float(value) for value in point],
                "height": float(point[2]),
            }
        )
    return rows


def _all_footpad_corner_points(shard: Any, data: Any) -> list[dict[str, Any]]:
    rows = []
    for foot in sorted(shard._foot_geoms):
        rows.extend(_footpad_corner_points(shard, data, foot))
    return rows


def _selected_contact_points(
    shard: Any,
    data: Any,
    mode: FixedContactMode,
) -> list[dict[str, Any]]:
    by_key = {row["key"]: row for row in _all_footpad_corner_points(shard, data)}
    return [by_key[point.key] for point in mode.points]


def _lowest_corner_mode_from_qpos(
    shard: Any,
    qpos: Any,
    *,
    name: str,
) -> FixedContactMode:
    _reset_to_qpos(shard, shard.data[0], qpos)
    points = _all_footpad_corner_points(shard, shard.data[0])
    specs = []
    for foot in sorted(shard._foot_geoms):
        foot_points = [point for point in points if point["foot"] == foot]
        lowest = min(foot_points, key=lambda point: (float(point["height"]), float(point["sx"]), float(point["sy"])))
        specs.append(ContactPointSpec(foot, float(lowest["sx"]), float(lowest["sy"])))
    return FixedContactMode(name=name, points=tuple(specs))


def _full_patch_mode(shard: Any) -> FixedContactMode:
    return FixedContactMode(
        name="all_footpad_bottom_corners",
        points=tuple(
            ContactPointSpec(foot, sx, sy)
            for foot in sorted(shard._foot_geoms)
            for sx, sy in _CORNERS
        ),
    )


def _dedupe_modes(modes: list[FixedContactMode]) -> list[FixedContactMode]:
    unique: dict[tuple[str, ...], FixedContactMode] = {}
    for mode in modes:
        key = tuple(point.key for point in mode.points)
        unique.setdefault(key, mode)
    return list(unique.values())


def _qfrc_from_contact_forces(
    shard: Any,
    data: Any,
    selected_points: list[dict[str, Any]],
    normal_forces: Any,
) -> Any:
    np = shard.np
    qfrc_contact = np.zeros(shard.model.nv, dtype=np.float64)
    for point, normal in zip(selected_points, normal_forces):
        jacp = np.zeros((3, shard.model.nv), dtype=np.float64)
        jacr = np.zeros((3, shard.model.nv), dtype=np.float64)
        shard.mujoco.mj_jac(
            shard.model,
            data,
            jacp,
            jacr,
            np.asarray(point["point"], dtype=np.float64),
            int(point["body_id"]),
        )
        qfrc_contact += jacp.T @ np.asarray([0.0, 0.0, float(normal)], dtype=np.float64)
    return qfrc_contact


def _normal_force_by_foot(
    mode: FixedContactMode,
    normal_forces: Any,
) -> dict[str, float]:
    rows = {foot: 0.0 for foot in mode.foot_names}
    for point, force in zip(mode.points, normal_forces):
        rows[point.foot] += float(force)
    return dict(sorted(rows.items()))


def _input_tau_from_ctrl(shard: Any, data: Any, ctrl: Any) -> Any:
    np = shard.np
    qfrc = np.zeros(shard.model.nv, dtype=np.float64)
    for actuator_id, qpos_address, dof_address in zip(
        shard._actuator_ids,
        shard._joint_qpos,
        shard._joint_dof,
    ):
        kp = max(1e-9, float(shard.model.actuator_gainprm[int(actuator_id), 0]))
        qfrc[int(dof_address)] = kp * (float(ctrl[int(actuator_id)]) - float(data.qpos[int(qpos_address)]))
    return qfrc


def _input_feasibility(shard: Any, data: Any, ctrl: Any) -> dict[str, Any]:
    ctrl_range_violations = 0
    force_limit_violations = 0
    max_ctrl_range_violation = 0.0
    max_force_limit_violation = 0.0
    max_abs_tau = 0.0
    rows = []
    tau = _input_tau_from_ctrl(shard, data, ctrl)
    for actuator, actuator_id, qpos_address, dof_address in zip(
        shard.blueprint.actuators,
        shard._actuator_ids,
        shard._joint_qpos,
        shard._joint_dof,
    ):
        value = float(ctrl[int(actuator_id)])
        lower, upper = (float(item) for item in shard.model.actuator_ctrlrange[int(actuator_id)])
        force = float(tau[int(dof_address)])
        force_limit = max(abs(float(item)) for item in shard.model.actuator_forcerange[int(actuator_id)])
        ctrl_violation = max(0.0, lower - value, value - upper)
        force_violation = max(0.0, abs(force) - force_limit)
        ctrl_range_violations += int(ctrl_violation > 0.0)
        force_limit_violations += int(force_violation > 0.0)
        max_ctrl_range_violation = max(max_ctrl_range_violation, ctrl_violation)
        max_force_limit_violation = max(max_force_limit_violation, force_violation)
        max_abs_tau = max(max_abs_tau, abs(force))
        rows.append(
            {
                "actuator": actuator.name,
                "semantic_slot": actuator.semantic_slot,
                "actuator_id": int(actuator_id),
                "qpos_address": int(qpos_address),
                "dof_address": int(dof_address),
                "ctrl": value,
                "ctrl_lower": lower,
                "ctrl_upper": upper,
                "tau_from_ctrl": force,
                "force_limit": force_limit,
                "ctrl_range_violation": ctrl_violation,
                "force_limit_violation": force_violation,
            }
        )
    return {
        "ctrl_range_violations": ctrl_range_violations,
        "force_limit_violations": force_limit_violations,
        "max_ctrl_range_violation": max_ctrl_range_violation,
        "max_force_limit_violation": max_force_limit_violation,
        "max_abs_tau_from_ctrl": max_abs_tau,
        "actuator_rows": rows,
        "qfrc_actuator_from_ctrl": tau,
    }


def _clip_ctrl_to_range(shard: Any, ctrl: Any) -> Any:
    np = shard.np
    clipped = np.asarray(ctrl, dtype=np.float64).copy()
    for actuator_id in shard._actuator_ids:
        lower, upper = (float(item) for item in shard.model.actuator_ctrlrange[int(actuator_id)])
        clipped[int(actuator_id)] = min(upper, max(lower, float(clipped[int(actuator_id)])))
    return clipped


def _baseline_stance_start(shard: Any) -> StartState:
    np = shard.np
    qpos = np.asarray(_stance_qpos(shard), dtype=np.float64)
    ctrl = np.zeros(shard.model.nu, dtype=np.float64)
    for joint, actuator_id in zip(shard.blueprint.joints, shard._actuator_ids):
        lower, upper = (float(item) for item in shard.model.actuator_ctrlrange[int(actuator_id)])
        target = float(shard.stance_solution.joint_qpos[joint.semantic_slot])
        ctrl[int(actuator_id)] = min(upper, max(lower, target))
    return StartState("target_r2_stance_baseline_ctrl", qpos, ctrl)


def _attempt_selected_result(attempt: dict[str, Any]) -> dict[str, Any]:
    phase = attempt["selected_phase"]
    if phase == "qacc_only_true_continuation":
        return attempt["qacc_only_true_continuation"]
    if phase == "contact_preserving_refinement":
        return attempt["contact_preserving_refinement"]
    raise ValueError(f"unknown selected phase {phase}")


def _start_states_for_route(
    *,
    source_records: list[dict[str, Any]],
    continuation_payload: dict[str, Any],
    family: str,
    route: ContinuationRoute,
    shard: Any,
) -> list[StartState]:
    np = shard.np
    endpoint_record = _record_for(
        source_records,
        family=family,
        seed=route.seed,
        range_fraction=route.end_range_fraction,
    )
    endpoint_best = endpoint_record["strict_refinement"]["best"]
    starts = [
        StartState(
            "r4a3_direct_endpoint_best",
            np.asarray(endpoint_best["qpos"], dtype=np.float64),
            _clip_ctrl_to_range(shard, endpoint_best["ctrl"]),
        )
    ]

    matching_routes = [
        row
        for row in continuation_payload.get("routes", [])
        if int(row["seed"]) == route.seed
        and abs(float(row["start_range_fraction"]) - route.start_range_fraction) <= 1e-12
        and abs(float(row["end_range_fraction"]) - route.end_range_fraction) <= 1e-12
    ]
    if matching_routes and matching_routes[0].get("attempts"):
        best_attempt = min(
            (_attempt_selected_result(attempt) for attempt in matching_routes[0]["attempts"]),
            key=lambda result: (
                float(result["snapshot"]["root_qacc_norm"]),
                float(result["snapshot"]["joint_qacc_max"]),
            ),
        )
        starts.append(
            StartState(
                "r4a31a_best_true_continuation_attempt",
                np.asarray(best_attempt["qpos"], dtype=np.float64),
                _clip_ctrl_to_range(shard, best_attempt["ctrl"]),
            )
        )

    source_state = _strict_source_state(
        source_records,
        family=family,
        seed=route.seed,
        range_fraction=route.start_range_fraction,
    )
    warm_start, _ = true_warm_start_from_previous_solution(
        shard,
        source_state["qpos"],
        source_state["ctrl"],
    )
    joint_count = len(shard._joint_qpos)
    qpos = np.asarray(source_state["qpos"], dtype=np.float64).copy()
    qpos[3:7] = _quat_from_roll_pitch_yaw(
        float(warm_start.solver_start_vector[0]),
        float(warm_start.solver_start_vector[1]),
        float(warm_start.solver_start_vector[2]),
    )
    for address, value in zip(shard._joint_qpos, warm_start.solver_start_vector[4 : 4 + joint_count]):
        qpos[int(address)] = float(value)
    ctrl = np.zeros(shard.model.nu, dtype=np.float64)
    for actuator_id, value in zip(
        shard._actuator_ids,
        warm_start.solver_start_vector[4 + joint_count : 4 + joint_count + len(shard._actuator_ids)],
    ):
        ctrl[int(actuator_id)] = float(value)
    starts.append(StartState("source_strict_branch_mapped_to_endpoint", qpos, _clip_ctrl_to_range(shard, ctrl)))
    starts.append(_baseline_stance_start(shard))

    unique: dict[str, StartState] = {}
    for start in starts:
        unique.setdefault(start.name, start)
    return list(unique.values())


def fixed_modes_from_start_states(shard: Any, start_states: list[StartState]) -> list[FixedContactMode]:
    modes = [
        _lowest_corner_mode_from_qpos(
            shard,
            start.qpos,
            name=f"lowest_corners_from_{start.name}",
        )
        for start in start_states
    ]
    modes.append(_full_patch_mode(shard))
    modes = _dedupe_modes(modes)
    for mode in modes:
        if not mode.is_double_foot:
            raise ValueError(f"fixed contact mode {mode.name} does not include both feet")
    return modes


def _vector_bounds_for_mode(shard: Any, mode: FixedContactMode) -> tuple[Any, Any]:
    np = shard.np
    lower = [-0.10, -0.10, -0.15, 0.0]
    upper = [0.10, 0.10, 0.15, 0.012]
    for joint_lower, joint_upper in _compiled_joint_position_bounds(shard):
        lower.append(joint_lower)
        upper.append(joint_upper)
    for actuator_id in shard._actuator_ids:
        ctrl_lower, ctrl_upper = (float(item) for item in shard.model.actuator_ctrlrange[int(actuator_id)])
        lower.append(ctrl_lower)
        upper.append(ctrl_upper)
    weight = _weight(shard)
    min_load = _foot_load_threshold(shard)
    for point in mode.points:
        point_count_for_foot = sum(item.foot == point.foot for item in mode.points)
        lower.append(min_load if point_count_for_foot == 1 else 0.0)
        upper.append(4.0 * weight)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _align_selected_points_to_penetration(
    shard: Any,
    data: Any,
    mode: FixedContactMode,
    *,
    penetration: float,
) -> None:
    shard.mujoco.mj_forward(shard.model, data)
    points = _selected_contact_points(shard, data, mode)
    if not points:
        return
    lowest = min(float(point["height"]) for point in points)
    data.qpos[2] += -float(penetration) - lowest
    shard.mujoco.mj_forward(shard.model, data)


def _apply_mode_vector(
    shard: Any,
    template_qpos: Any,
    mode: FixedContactMode,
    vector: Any,
) -> tuple[Any, Any, list[dict[str, Any]], Any]:
    np = shard.np
    joint_count = len(shard._joint_qpos)
    ctrl_count = len(shard._actuator_ids)
    qpos = np.asarray(template_qpos, dtype=np.float64).copy()
    qpos[3:7] = _quat_from_roll_pitch_yaw(float(vector[0]), float(vector[1]), float(vector[2]))
    for address, value in zip(shard._joint_qpos, vector[4 : 4 + joint_count]):
        qpos[int(address)] = float(value)
    ctrl = np.zeros(shard.model.nu, dtype=np.float64)
    ctrl_values = vector[4 + joint_count : 4 + joint_count + ctrl_count]
    for actuator_id, value in zip(shard._actuator_ids, ctrl_values):
        ctrl[int(actuator_id)] = float(value)
    _reset_to_qpos(shard, shard.data[0], qpos)
    _align_selected_points_to_penetration(
        shard,
        shard.data[0],
        mode,
        penetration=float(vector[3]),
    )
    shard.data[0].ctrl[:] = ctrl
    shard.mujoco.mj_forward(shard.model, shard.data[0])
    selected_points = _selected_contact_points(shard, shard.data[0], mode)
    normal_forces = np.asarray(vector[4 + joint_count + ctrl_count :], dtype=np.float64)
    return (
        np.asarray(shard.data[0].qpos, dtype=np.float64).copy(),
        ctrl.copy(),
        selected_points,
        normal_forces,
    )


def _initial_force_values(shard: Any, mode: FixedContactMode) -> list[float]:
    weight = _weight(shard)
    by_foot = {foot: [point for point in mode.points if point.foot == foot] for foot in mode.foot_names}
    values = []
    for point in mode.points:
        values.append(0.5 * weight / max(1, len(by_foot[point.foot])))
    return values


def _vector_from_start_state(shard: Any, start: StartState, mode: FixedContactMode) -> Any:
    np = shard.np
    _reset_to_qpos(shard, shard.data[0], start.qpos)
    selected = _selected_contact_points(shard, shard.data[0], mode)
    penetration = min(0.012, max(0.0, -min(float(point["height"]) for point in selected)))
    roll, pitch, yaw = _roll_pitch_yaw(start.qpos[3:7])
    joints = [float(start.qpos[int(address)]) for address in shard._joint_qpos]
    ctrls = [float(start.ctrl[int(actuator_id)]) for actuator_id in shard._actuator_ids]
    return np.asarray([roll, pitch, yaw, penetration, *joints, *ctrls, *_initial_force_values(shard, mode)])


def _static_constraint_snapshot(
    shard: Any,
    mode: FixedContactMode,
    selected_points: list[dict[str, Any]],
    normal_forces: Any,
    ctrl: Any,
) -> dict[str, Any]:
    np = shard.np
    data = shard.data[0]
    qfrc_contact = _qfrc_from_contact_forces(shard, data, selected_points, normal_forces)
    qfrc_required = np.asarray(data.qfrc_bias, dtype=np.float64) - qfrc_contact
    input_report = _input_feasibility(shard, data, ctrl)
    tau = input_report["qfrc_actuator_from_ctrl"]
    joint_tau_residuals = []
    for joint, dof in zip(shard.blueprint.joints, shard._joint_dof):
        residual = float(qfrc_required[int(dof)] - tau[int(dof)])
        joint_tau_residuals.append(
            {
                "joint": joint.name,
                "semantic_slot": joint.semantic_slot,
                "dof": int(dof),
                "residual": residual,
                "abs_residual": abs(residual),
            }
        )
    normal_by_foot = _normal_force_by_foot(mode, normal_forces)
    min_load = _foot_load_threshold(shard)
    load_deficit_by_foot = {
        foot: max(0.0, min_load - normal_by_foot.get(foot, 0.0))
        for foot in mode.foot_names
    }
    force_sum = float(np.sum(normal_forces))
    weight = _weight(shard)
    selected_heights = [
        {
            "foot": point["foot"],
            "sx": point["sx"],
            "sy": point["sy"],
            "height": float(point["height"]),
        }
        for point in selected_points
    ]
    root_wrench = [float(value) for value in qfrc_required[:6]]
    return {
        "root_wrench": root_wrench,
        "root_wrench_norm": float(np.linalg.norm(qfrc_required[:6])),
        "joint_tau_residual_max": max((row["abs_residual"] for row in joint_tau_residuals), default=0.0),
        "joint_tau_residuals": joint_tau_residuals,
        "normal_force_by_foot": normal_by_foot,
        "normal_force_sum": force_sum,
        "normal_force_sum_error": force_sum - weight,
        "normal_force_sum_relative_error": abs(force_sum - weight) / max(weight, 1e-12),
        "normal_forces": [float(value) for value in normal_forces],
        "minimum_foot_load": min_load,
        "load_deficit_by_foot": load_deficit_by_foot,
        "load_deficit_sum": float(sum(load_deficit_by_foot.values())),
        "selected_contact_heights": selected_heights,
        "selected_contact_height_abs_max": max((abs(item["height"]) for item in selected_heights), default=0.0),
        "input": {key: value for key, value in input_report.items() if key != "qfrc_actuator_from_ctrl"},
        "com": _center_of_mass(shard, data),
        "foot_bottom_heights": _foot_bottom_heights(shard, data),
        "qfrc_contact": [float(value) for value in qfrc_contact],
        "qfrc_required": [float(value) for value in qfrc_required],
    }


def rigid_contact_constraints_feasible(snapshot: dict[str, Any]) -> bool:
    static = snapshot["rigid_static_constraints"]
    return bool(
        float(static["root_wrench_norm"]) <= _RIGID_ROOT_WRENCH_NORM_MAX
        and float(static["joint_tau_residual_max"]) <= _RIGID_JOINT_TAU_RESIDUAL_MAX
        and float(static["selected_contact_height_abs_max"]) <= _RIGID_SELECTED_HEIGHT_ABS_MAX
        and float(static["normal_force_sum_relative_error"]) <= _RIGID_FORCE_BALANCE_REL_MAX
        and float(static["load_deficit_sum"]) <= 1e-8 * max(1.0, float(static["normal_force_sum"]))
        and int(static["input"]["ctrl_range_violations"]) == 0
        and int(static["input"]["force_limit_violations"]) == 0
    )


def _candidate_classification(candidate: dict[str, Any]) -> str:
    if candidate["strict_gate_passed"]:
        return STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND
    if rigid_contact_constraints_feasible(candidate):
        return FIXED_CONTACT_WRENCH_SOLUTION_ACTUAL_GATE_FAILED
    actual_classification = candidate["actual_classification"]
    if actual_classification == SINGLE_SUPPORT_EQUILIBRIUM_ONLY_FOUND:
        return SINGLE_SUPPORT_EQUILIBRIUM_ONLY_FOUND
    return FIXED_CONTACT_MODE_SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE


def solve_fixed_contact_mode(
    shard: Any,
    mode: FixedContactMode,
    start: StartState,
    *,
    max_nfev: int,
    horizon_steps: int,
) -> dict[str, Any]:
    np = shard.np
    try:
        from scipy import optimize
    except ImportError as exc:  # pragma: no cover - optional diagnostic dependency
        raise RuntimeError("SciPy is required for R4a.3.1b fixed contact-mode solve") from exc

    lower, upper = _vector_bounds_for_mode(shard, mode)
    raw_start = _vector_from_start_state(shard, start, mode)
    vector0 = np.clip(raw_start, lower, upper)
    span = np.maximum(1e-6, upper - lower)
    joint_count = len(shard._joint_qpos)
    ctrl_count = len(shard._actuator_ids)
    weight = _weight(shard)
    min_load = _foot_load_threshold(shard)

    def residual(vector: Any) -> Any:
        _, ctrl, selected_points, normal_forces = _apply_mode_vector(shard, start.qpos, mode, vector)
        static = _static_constraint_snapshot(shard, mode, selected_points, normal_forces, ctrl)
        qfrc_required = np.asarray(static["qfrc_required"], dtype=np.float64)
        input_tau = _input_tau_from_ctrl(shard, shard.data[0], ctrl)
        values = list(qfrc_required[:6] / _ROOT_WRENCH_SCALE)
        values.extend(
            (qfrc_required[int(dof)] - input_tau[int(dof)]) / _JOINT_TAU_SCALE
            for dof in shard._joint_dof
        )
        values.extend(float(point["height"]) / _HEIGHT_SCALE for point in selected_points)
        values.append(_FORCE_SUM_SCALE * float(static["normal_force_sum_error"]) / max(weight, 1e-12))
        normal_by_foot = static["normal_force_by_foot"]
        values.extend(
            _LOAD_DEFICIT_SCALE * max(0.0, min_load - float(normal_by_foot.get(foot, 0.0))) / max(min_load, 1e-12)
            for foot in mode.foot_names
        )
        values.extend(
            10.0 * float(static["input"]["max_force_limit_violation"]) / max(_JOINT_TAU_SCALE, 1e-12)
            for _ in (0,)
        )
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
    qpos, ctrl, selected_points, normal_forces = _apply_mode_vector(shard, start.qpos, mode, result.x)
    static = _static_constraint_snapshot(shard, mode, selected_points, normal_forces, ctrl)
    actual_snapshot = _diagnostic_snapshot(shard, qpos, ctrl)
    actual_classification = classify_snapshot_without_certificates(actual_snapshot)
    hold_rollout = None
    strict_gate_passed = False
    if actual_classification == STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND:
        hold_rollout = run_strict_hold_rollout(
            shard,
            {
                "status": "feasible",
                "best": {
                    "qpos": [float(value) for value in qpos],
                    "ctrl": [float(value) for value in ctrl],
                },
            },
            horizon_steps=horizon_steps,
        )
        strict_gate_passed = bool(hold_rollout["passed"])
    candidate = {
        "schema": "task067_r4a31b_fixed_contact_mode_candidate_v1",
        "mode": mode.manifest(),
        "start_name": start.name,
        "variable_contract": {
            "state_variables": ["root_roll", "root_pitch", "root_yaw", "penetration", "actuated_joint_qpos"],
            "input_variables": ["actuator_position_ctrl"],
            "wrench_variables": ["per_selected_contact_point_vertical_normal_force"],
            "force_variable_count": len(mode.points),
            "both_feet_forced_active": mode.is_double_foot,
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
            "root_wrench_scale": _ROOT_WRENCH_SCALE,
            "joint_tau_scale": _JOINT_TAU_SCALE,
            "height_scale": _HEIGHT_SCALE,
            "load_deficit_scale": _LOAD_DEFICIT_SCALE,
            "regularization_scale": _REGULARIZATION_SCALE,
        },
        "qpos": [float(value) for value in qpos],
        "ctrl": [float(value) for value in ctrl],
        "vector": [float(value) for value in result.x],
        "rigid_static_constraints": static,
        "rigid_contact_constraints_feasible": False,
        "actual_snapshot": actual_snapshot,
        "actual_classification": actual_classification,
        "strict_gate_passed": strict_gate_passed,
        "strict_nominal_hold_2s": hold_rollout,
        "active_bounds": _active_bound_report(
            shard,
            result.x[: 4 + joint_count + ctrl_count],
            lower[: 4 + joint_count + ctrl_count],
            upper[: 4 + joint_count + ctrl_count],
        ),
        "infeasibility_certificate": {
            "classification": None,
            "reason": "A failed nonlinear fixed-contact search is not an independent proof of kinematic or wrench/actuation infeasibility.",
        },
    }
    candidate["rigid_contact_constraints_feasible"] = rigid_contact_constraints_feasible(candidate)
    candidate["classification"] = _candidate_classification(candidate)
    return candidate


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float]:
    static = candidate["rigid_static_constraints"]
    actual = candidate["actual_snapshot"]
    return (
        0.0 if candidate["strict_gate_passed"] else 1.0,
        0.0 if candidate["rigid_contact_constraints_feasible"] else 1.0,
        float(static["root_wrench_norm"]),
        float(static["joint_tau_residual_max"]),
        float(actual["root_qacc_norm"]) + float(actual["joint_qacc_max"]),
    )


def solve_endpoint_fixed_contact_modes(
    *,
    source_records: list[dict[str, Any]],
    continuation_payload: dict[str, Any],
    family: str,
    route: ContinuationRoute,
    max_nfev: int,
    horizon_steps: int,
) -> dict[str, Any]:
    shard, key = _build_shard_for(family, route.seed, route.end_range_fraction)
    endpoint_direct = _direct_endpoint_diagnostic(
        source_records,
        family=family,
        seed=route.seed,
        range_fraction=route.end_range_fraction,
    )
    starts = _start_states_for_route(
        source_records=source_records,
        continuation_payload=continuation_payload,
        family=family,
        route=route,
        shard=shard,
    )
    modes = fixed_modes_from_start_states(shard, starts)
    candidates = [
        solve_fixed_contact_mode(
            shard,
            mode,
            start,
            max_nfev=max_nfev,
            horizon_steps=horizon_steps,
        )
        for mode in modes
        for start in starts
    ]
    best = min(candidates, key=_candidate_sort_key)
    strict_candidates = [candidate for candidate in candidates if candidate["strict_gate_passed"]]
    rigid_candidates = [
        candidate for candidate in candidates if candidate["rigid_contact_constraints_feasible"]
    ]
    if strict_candidates:
        final_classification = STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND
        endpoint_strict_gate_passed = True
    elif endpoint_direct["classification"] == SINGLE_SUPPORT_EQUILIBRIUM_ONLY_FOUND:
        final_classification = SINGLE_SUPPORT_EQUILIBRIUM_ONLY_FOUND
        endpoint_strict_gate_passed = False
    elif rigid_candidates:
        final_classification = FIXED_CONTACT_WRENCH_SOLUTION_ACTUAL_GATE_FAILED
        endpoint_strict_gate_passed = False
    else:
        final_classification = FIXED_CONTACT_MODE_SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE
        endpoint_strict_gate_passed = False
    return {
        "schema": "task067_r4a31b_fixed_contact_mode_endpoint_v1",
        "family": family,
        "seed": route.seed,
        "range_fraction": route.end_range_fraction,
        "endpoint_label": f"{family}:rf{route.end_range_fraction:g}:seed{route.seed}",
        "morphology_instance_key": key,
        "model_xml_sha256": _sha256_bytes(shard.xml.encode("utf-8")),
        "direct_endpoint_diagnostic": endpoint_direct,
        "start_names": [start.name for start in starts],
        "fixed_contact_modes": [mode.manifest() for mode in modes],
        "candidate_count": len(candidates),
        "strict_candidate_count": len(strict_candidates),
        "rigid_contact_feasible_candidate_count": len(rigid_candidates),
        "endpoint_strict_gate_passed": endpoint_strict_gate_passed,
        "final_classification": final_classification,
        "best": best,
        "candidate_preview": [
            {
                "classification": candidate["classification"],
                "strict_gate_passed": candidate["strict_gate_passed"],
                "rigid_contact_constraints_feasible": candidate["rigid_contact_constraints_feasible"],
                "mode": candidate["mode"]["name"],
                "start_name": candidate["start_name"],
                "nfev": candidate["solver"]["nfev"],
                "optimality": candidate["solver"]["optimality"],
                "root_wrench_norm": candidate["rigid_static_constraints"]["root_wrench_norm"],
                "joint_tau_residual_max": candidate["rigid_static_constraints"]["joint_tau_residual_max"],
                "selected_contact_height_abs_max": candidate["rigid_static_constraints"][
                    "selected_contact_height_abs_max"
                ],
                "load_deficit_sum": candidate["rigid_static_constraints"]["load_deficit_sum"],
                "actual_root_qacc_norm": candidate["actual_snapshot"]["root_qacc_norm"],
                "actual_joint_qacc_max": candidate["actual_snapshot"]["joint_qacc_max"],
                "actual_support_mode": candidate["actual_snapshot"]["support_mode"],
            }
            for candidate in sorted(candidates, key=_candidate_sort_key)
        ],
        "infeasibility_certificate": {
            "classification": None,
            "reason": "This endpoint solve is diagnostic and does not emit kinematic or wrench/actuation infeasibility certificates.",
        },
    }


def _decide(summary: dict[str, Any]) -> dict[str, str]:
    if int(summary["combined_strict_contract_passed"]) == int(summary["combined_source_records"]):
        return {
            "status": "r4a31b_fixed_contact_solver_restored_8_of_8",
            "decision": "The fixed double-foot contact-mode state-input-wrench solve recovered strict equilibria for every prior failure.",
            "next_allowed_work": "Prepare StanceSolutionV3(qpos_eq, ctrl_eq) from strict solutions; still do not integrate controller.",
        }
    return {
        "status": "r4a31b_fixed_contact_solver_incomplete_no_generator_certificate",
        "decision": "The fixed double-foot contact-mode solve did not restore full strict coverage and produced no independent physical infeasibility certificate.",
        "next_allowed_work": "Continue contact-mode/model diagnosis only; do not modify env, controller, generator, kp/kv, or Task061/062.",
    }


def run_fixed_contact_mode_wrench_solve(
    *,
    input_json: Path,
    continuation_json: Path,
    family: str,
    routes: tuple[ContinuationRoute, ...],
    max_nfev: int,
    horizon_steps: int,
) -> dict[str, Any]:
    source_records = _load_r4a3_records(input_json, family=family)
    continuation_payload = json.loads(continuation_json.read_text(encoding="utf-8"))
    endpoint_rows = [
        solve_endpoint_fixed_contact_modes(
            source_records=source_records,
            continuation_payload=continuation_payload,
            family=family,
            route=route,
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
        row["endpoint_label"] for row in endpoint_rows if row["endpoint_strict_gate_passed"]
    }
    combined_accepted = source_accepted | endpoint_recovered
    summary = {
        "combined_source_records": len(source_records),
        "source_strict_contract_passed": len(source_accepted),
        "failed_endpoints_tested": len(endpoint_rows),
        "endpoints_recovered_by_fixed_contact_solver": len(endpoint_recovered),
        "combined_strict_contract_passed": len(combined_accepted),
        "combined_accepted_labels": sorted(combined_accepted),
        "combined_incomplete_labels": [
            f"{record['family']}:rf{float(record['range_fraction']):g}:seed{int(record['seed'])}"
            for record in source_records
            if f"{record['family']}:rf{float(record['range_fraction']):g}:seed{int(record['seed'])}"
            not in combined_accepted
        ],
        "endpoint_classifications": {
            row["endpoint_label"]: row["final_classification"] for row in endpoint_rows
        },
        "rigid_contact_feasible_candidates": sum(
            int(row["rigid_contact_feasible_candidate_count"]) for row in endpoint_rows
        ),
        "strict_candidates": sum(int(row["strict_candidate_count"]) for row in endpoint_rows),
        "infeasibility_certificates": {
            "kinematic_double_support_infeasible": 0,
            "wrench_or_actuation_infeasible": 0,
        },
    }
    payload = {
        "schema": "task067_r4a31b_fixed_contact_mode_wrench_solve_v1",
        "source_artifact": str(input_json.resolve()),
        "true_continuation_artifact": str(continuation_json.resolve()),
        "provenance": {
            "source_artifact_sha256": _sha256_path(input_json),
            "true_continuation_artifact_sha256": _sha256_path(continuation_json),
            "diagnostic_source_sha256": _sha256_path(Path(__file__)),
            "dependency_source_sha256": {
                "whole_body_contact_preserving_continuation.py": _sha256_path(
                    Path(__file__).with_name("whole_body_contact_preserving_continuation.py")
                ),
                "whole_body_true_continuation_correctness.py": _sha256_path(
                    Path(__file__).with_name("whole_body_true_continuation_correctness.py")
                ),
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
                "max_nfev": max_nfev,
                "horizon_steps": horizon_steps,
                "hold_seconds": horizon_steps / 50.0,
                "strict_root_qacc_norm": _STRICT_ROOT_QACC_NORM,
                "strict_joint_qacc_max": _STRICT_JOINT_QACC_MAX,
                "rigid_root_wrench_norm_max": _RIGID_ROOT_WRENCH_NORM_MAX,
                "rigid_joint_tau_residual_max": _RIGID_JOINT_TAU_RESIDUAL_MAX,
                "rigid_selected_height_abs_max": _RIGID_SELECTED_HEIGHT_ABS_MAX,
                "foot_load_fraction": _FOOT_LOAD_FRACTION,
            },
            "diagnostic_scope": {
                "fixed_double_foot_contact_mode": True,
                "explicit_contact_wrench_variables": True,
                "ordinary_multistart_search": False,
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
            "all_modes_are_double_foot": all(
                len(mode["foot_names"]) == 2 for row in endpoint_rows for mode in row["fixed_contact_modes"]
            ),
            "search_failure_not_promoted_to_physical_infeasible": (
                int(summary["infeasibility_certificates"]["kinematic_double_support_infeasible"]) == 0
                and int(summary["infeasibility_certificates"]["wrench_or_actuation_infeasible"]) == 0
            ),
            "strict_acceptance_requires_actual_gate_and_2s_hold": True,
        },
        "endpoints": endpoint_rows,
    }
    return payload


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
    parser.add_argument("--continuation-json", type=Path, default=_DEFAULT_R4A31A_INPUT)
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--family", default=_DEFAULT_FAMILY)
    parser.add_argument(
        "--route",
        action="append",
        help="Route as seed:start_rf:end_rf. Defaults to the three R4a.3 failures.",
    )
    parser.add_argument("--max-nfev", type=int, default=_DEFAULT_MAX_NFEV)
    parser.add_argument("--horizon-steps", type=int, default=_DEFAULT_HORIZON_STEPS)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    payload = run_fixed_contact_mode_wrench_solve(
        input_json=args.input_json,
        continuation_json=args.continuation_json,
        family=args.family,
        routes=_routes_from_args(args.route),
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
