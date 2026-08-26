"""Task067 R4a.3.1a true-continuation correctness diagnosis.

This closed diagnostic checks whether the R4a.3.1 same-topology continuation
was accidentally cut away from the previous equilibrium branch by re-anchoring
joint bounds around the target R2 stance.  It preserves the prior R4a.3.1
artifact and writes a new artifact.

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
    SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE,
    SINGLE_SUPPORT_EQUILIBRIUM_ONLY_FOUND,
    STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND,
    ContinuationRoute,
    _active_bound_report,
    _build_shard_for,
    _choose_attempt_result,
    _diagnostic_snapshot,
    _direct_endpoint_diagnostic,
    _load_r4a3_records,
    _rounded_range_fraction,
    _sha256_bytes,
    _sha256_path,
    _step_deltas,
    _strict_source_state,
    classify_snapshot_without_certificates,
    warm_start_mapping_manifest,
)
from h200_locomotion_lab.tools.whole_body_dynamic_balance_diagnosis import (
    _align_lowest_foot_to_penetration,
    _contact_report,
    _foot_bottom_heights,
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
)
from h200_locomotion_lab.tools.whole_body_strict_equilibrium_coverage import (
    run_strict_hold_rollout,
)

_DEFAULT_R4A3_INPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a3_strict_equilibrium_coverage_4x2.json"
)
_DEFAULT_R4A31_ARTIFACT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31_contact_preserving_continuation_3fail.json"
)
_DEFAULT_OUTPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31a_true_continuation_correctness_3fail.json"
)
_DEFAULT_FAMILY = "biped"
_DEFAULT_STEP = 0.05
_DEFAULT_MIN_STEP = 0.00625
_FOOT_HEIGHT_SCALE = 0.001
_FOOT_LOAD_DEFICIT_SCALE = 20.0
_REGULARIZATION_SCALE = 1e-5

_DEFAULT_ROUTES = (
    ContinuationRoute(seed=0, start_range_fraction=0.0, end_range_fraction=0.5),
    ContinuationRoute(seed=3, start_range_fraction=0.0, end_range_fraction=0.5),
    ContinuationRoute(seed=1, start_range_fraction=0.5, end_range_fraction=0.0),
)


@dataclass(frozen=True, slots=True)
class TrueContinuationBounds:
    lower: Any
    upper: Any
    physical_lower: Any
    physical_upper: Any
    artificial_trust_region_radius: float | None
    joint_count: int


@dataclass(frozen=True, slots=True)
class TrueWarmStart:
    raw_vector: Any
    physical_clipped_vector: Any
    solver_start_vector: Any
    template_qpos: Any
    report: dict[str, Any]


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _compiled_joint_position_bounds(shard: Any) -> list[tuple[float, float]]:
    bounds = []
    for joint in shard.blueprint.joints:
        joint_id = int(
            shard.mujoco.mj_name2id(
                shard.model,
                shard.mujoco.mjtObj.mjOBJ_JOINT,
                joint.name,
            )
        )
        lower, upper = (float(value) for value in shard.model.jnt_range[joint_id])
        bounds.append((lower, upper))
    return bounds


def _vector_component_names(shard: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"kind": "root", "name": "roll"},
        {"kind": "root", "name": "pitch"},
        {"kind": "root", "name": "yaw"},
        {"kind": "contact", "name": "penetration"},
    ]
    rows.extend(
        {
            "kind": "joint",
            "name": joint.name,
            "semantic_slot": joint.semantic_slot,
        }
        for joint in shard.blueprint.joints
    )
    rows.extend(
        {
            "kind": "ctrl",
            "name": actuator.name,
            "semantic_slot": actuator.semantic_slot,
            "actuator_id": int(actuator_id),
        }
        for actuator, actuator_id in zip(shard.blueprint.actuators, shard._actuator_ids)
    )
    return rows


def _clip_rows(
    *,
    components: list[dict[str, Any]],
    raw: Any,
    clipped: Any,
    lower: Any,
    upper: Any,
    kinds: set[str],
) -> dict[str, Any]:
    import numpy as np

    raw_array = np.asarray(raw, dtype=np.float64)
    clipped_array = np.asarray(clipped, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    rows = []
    deltas = []
    for index, component in enumerate(components):
        if component["kind"] not in kinds:
            continue
        delta = abs(float(clipped_array[index]) - float(raw_array[index]))
        deltas.append(delta)
        if delta <= 0.0:
            continue
        side = "lower" if float(clipped_array[index]) <= float(raw_array[index]) else "upper"
        rows.append(
            {
                **component,
                "index": index,
                "raw": float(raw_array[index]),
                "clipped": float(clipped_array[index]),
                "lower": float(lower_array[index]),
                "upper": float(upper_array[index]),
                "abs_clip": delta,
                "side": side,
            }
        )
    return {
        "component_kinds": sorted(kinds),
        "count": len(rows),
        "max_abs_clip": max(deltas, default=0.0),
        "l2_clip": float(np.linalg.norm([row["abs_clip"] for row in rows])) if rows else 0.0,
        "rows": rows,
    }


def _target_r2_stance_distance_to_previous_solution(
    shard: Any,
    previous_qpos: Any,
) -> dict[str, Any]:
    import numpy as np

    stance = _stance_qpos(shard)
    previous = np.asarray(previous_qpos, dtype=np.float64)
    roll, pitch, yaw = _roll_pitch_yaw(previous[3:7])
    stance_roll, stance_pitch, stance_yaw = _roll_pitch_yaw(stance[3:7])
    rows = []
    values = []
    for joint, qpos_address in zip(shard.blueprint.joints, shard._joint_qpos):
        delta = float(previous[int(qpos_address)] - stance[int(qpos_address)])
        values.append(delta)
        rows.append(
            {
                "joint": joint.name,
                "semantic_slot": joint.semantic_slot,
                "previous_solution_qpos": float(previous[int(qpos_address)]),
                "target_r2_stance_qpos": float(stance[int(qpos_address)]),
                "delta": delta,
                "abs_delta": abs(delta),
            }
        )
    return {
        "joint_abs_max": max((row["abs_delta"] for row in rows), default=0.0),
        "joint_l2": float(np.linalg.norm(values)) if values else 0.0,
        "root_rpy_delta": {
            "roll": float(roll - stance_roll),
            "pitch": float(pitch - stance_pitch),
            "yaw": float(yaw - stance_yaw),
        },
        "max_joint_rows": sorted(rows, key=lambda row: row["abs_delta"], reverse=True)[:5],
    }


def _true_continuation_bounds(
    shard: Any,
    raw_vector: Any,
    *,
    trust_region_radius: float | None,
) -> TrueContinuationBounds:
    np = shard.np
    joint_count = len(shard._joint_qpos)
    physical_lower = [-0.10, -0.10, -0.15, 0.0]
    physical_upper = [0.10, 0.10, 0.15, 0.012]
    for lower, upper in _compiled_joint_position_bounds(shard):
        physical_lower.append(lower)
        physical_upper.append(upper)
    for actuator_id in shard._actuator_ids:
        ctrl_lower, ctrl_upper = (
            float(value) for value in shard.model.actuator_ctrlrange[int(actuator_id)]
        )
        physical_lower.append(ctrl_lower)
        physical_upper.append(ctrl_upper)

    lower = np.asarray(physical_lower, dtype=np.float64)
    upper = np.asarray(physical_upper, dtype=np.float64)
    if trust_region_radius is not None:
        radius = float(trust_region_radius)
        if radius <= 0.0:
            raise ValueError("trust_region_radius must be positive when provided")
        raw = np.asarray(raw_vector, dtype=np.float64)
        joint_slice = slice(4, 4 + joint_count)
        center = np.clip(raw[joint_slice], lower[joint_slice], upper[joint_slice])
        lower[joint_slice] = np.maximum(lower[joint_slice], center - radius)
        upper[joint_slice] = np.minimum(upper[joint_slice], center + radius)
    return TrueContinuationBounds(
        lower=lower,
        upper=upper,
        physical_lower=np.asarray(physical_lower, dtype=np.float64),
        physical_upper=np.asarray(physical_upper, dtype=np.float64),
        artificial_trust_region_radius=trust_region_radius,
        joint_count=joint_count,
    )


def _raw_vector_from_previous_qpos_ctrl(
    shard: Any,
    previous_qpos: Any,
    previous_ctrl: Any,
) -> tuple[Any, dict[str, Any]]:
    np = shard.np
    qpos_array = np.asarray(previous_qpos, dtype=np.float64)
    ctrl_array = np.asarray(previous_ctrl, dtype=np.float64)
    if qpos_array.shape != (shard.model.nq,):
        raise ValueError(f"previous qpos shape {qpos_array.shape} does not match target nq={shard.model.nq}")
    if ctrl_array.shape != (shard.model.nu,):
        raise ValueError(f"previous ctrl shape {ctrl_array.shape} does not match target nu={shard.model.nu}")

    probe_qpos = qpos_array.copy()
    _reset_to_qpos(shard, shard.data[0], probe_qpos)
    foot_bottoms = _foot_bottom_heights(shard, shard.data[0])
    lowest = min(foot_bottoms.values(), default=0.0)
    penetration = min(0.012, max(0.0, -float(lowest)))
    roll, pitch, yaw = _roll_pitch_yaw(qpos_array[3:7])
    joint_values = [float(qpos_array[int(address)]) for address in shard._joint_qpos]
    ctrl_values = [float(ctrl_array[int(actuator_id)]) for actuator_id in shard._actuator_ids]
    return np.asarray([roll, pitch, yaw, penetration, *joint_values, *ctrl_values], dtype=np.float64), {
        "raw_penetration_from_previous_qpos_on_target": penetration,
        "raw_foot_bottom_heights_from_previous_qpos_on_target": dict(sorted(foot_bottoms.items())),
    }


def true_warm_start_from_previous_solution(
    shard: Any,
    previous_qpos: Any,
    previous_ctrl: Any,
    *,
    trust_region_radius: float | None = None,
) -> tuple[TrueWarmStart, TrueContinuationBounds]:
    """Create the continuation start from the previous accepted branch.

    Target R2 stance is measured but never used to center joint bounds or
    replace previous actuated joint values.
    """

    np = shard.np
    raw_vector, raw_geometry = _raw_vector_from_previous_qpos_ctrl(
        shard,
        previous_qpos,
        previous_ctrl,
    )
    components = _vector_component_names(shard)
    bounds = _true_continuation_bounds(
        shard,
        raw_vector,
        trust_region_radius=trust_region_radius,
    )
    physical_clipped = np.clip(raw_vector, bounds.physical_lower, bounds.physical_upper)
    solver_start = np.clip(physical_clipped, bounds.lower, bounds.upper)
    physical_joint_clip = _clip_rows(
        components=components,
        raw=raw_vector,
        clipped=physical_clipped,
        lower=bounds.physical_lower,
        upper=bounds.physical_upper,
        kinds={"joint"},
    )
    ctrl_range_clip = _clip_rows(
        components=components,
        raw=raw_vector,
        clipped=physical_clipped,
        lower=bounds.physical_lower,
        upper=bounds.physical_upper,
        kinds={"ctrl"},
    )
    artificial_trust_clip = _clip_rows(
        components=components,
        raw=physical_clipped,
        clipped=solver_start,
        lower=bounds.lower,
        upper=bounds.upper,
        kinds={"joint"},
    )
    previous = np.asarray(previous_qpos, dtype=np.float64)
    report = {
        "source": "previous_accepted_qpos_ctrl",
        "target_r2_stance_reanchoring_forbidden": True,
        "joint_bounds_policy": {
            "compiled_physical_joint_limits": True,
            "optional_previous_solution_centered_trust_region": trust_region_radius is not None,
            "target_r2_stance_centered_plus_minus_0p08_rad": False,
        },
        **raw_geometry,
        "target_r2_stance_distance_to_previous_solution": _target_r2_stance_distance_to_previous_solution(
            shard,
            previous_qpos,
        ),
        "physical_joint_limit_clip": physical_joint_clip,
        "ctrl_range_clip": ctrl_range_clip,
        "artificial_trust_region_clip": artificial_trust_clip,
        "raw_vector": [float(value) for value in raw_vector],
        "physical_clipped_vector": [float(value) for value in physical_clipped],
        "solver_start_vector": [float(value) for value in solver_start],
    }
    return (
        TrueWarmStart(
            raw_vector=raw_vector,
            physical_clipped_vector=physical_clipped,
            solver_start_vector=solver_start,
            template_qpos=previous.copy(),
            report=report,
        ),
        bounds,
    )


def _apply_true_vector(
    shard: Any,
    template_qpos: Any,
    vector: Any,
) -> tuple[Any, Any]:
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
    _align_lowest_foot_to_penetration(
        shard,
        shard.data[0],
        penetration=float(vector[3]),
    )
    shard.data[0].ctrl[:] = ctrl
    shard.mujoco.mj_forward(shard.model, shard.data[0])
    return np.asarray(shard.data[0].qpos, dtype=np.float64).copy(), ctrl.copy()


def _true_residual_breakdown(
    shard: Any,
    vector: Any,
    start: Any,
    bounds: TrueContinuationBounds,
    *,
    contact_preserving: bool,
) -> dict[str, Any]:
    np = shard.np
    qacc = np.asarray(shard.data[0].qacc, dtype=np.float64)
    joint_qacc = qacc[list(shard._joint_dof)] if shard._joint_dof else np.asarray([])
    span = np.maximum(1e-6, bounds.upper - bounds.lower)
    regularization = _REGULARIZATION_SCALE * (np.asarray(vector) - np.asarray(start)) / span
    breakdown: dict[str, Any] = {
        "root_qacc_scaled_norm": float(np.linalg.norm(qacc[:6] / _ROOT_QACC_SCALE)),
        "joint_qacc_scaled_norm": float(np.linalg.norm(joint_qacc / _JOINT_QACC_SCALE))
        if joint_qacc.size
        else 0.0,
        "regularization_scaled_norm": float(np.linalg.norm(regularization)),
    }
    if contact_preserving:
        contact = _contact_report(shard, shard.data[0])
        foot_bottom = _foot_bottom_heights(shard, shard.data[0])
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


def refine_true_continuation_from_warm_start(
    shard: Any,
    warm_start: TrueWarmStart,
    bounds: TrueContinuationBounds,
    *,
    contact_preserving: bool,
    max_nfev: int,
    horizon_steps: int,
) -> dict[str, Any]:
    np = shard.np
    try:
        from scipy import optimize
    except ImportError as exc:  # pragma: no cover - optional diagnostic dependency
        raise RuntimeError("SciPy is required for R4a.3.1a true-continuation diagnosis") from exc

    start = np.asarray(warm_start.solver_start_vector, dtype=np.float64)
    span = np.maximum(1e-6, bounds.upper - bounds.lower)

    def residual(vector: Any) -> Any:
        _apply_true_vector(shard, warm_start.template_qpos, vector)
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
        bounds=(bounds.lower, bounds.upper),
        method="trf",
        max_nfev=max_nfev,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        x_scale="jac",
        diff_step=1e-5,
    )
    qpos, ctrl = _apply_true_vector(shard, warm_start.template_qpos, result.x)
    snapshot = _diagnostic_snapshot(shard, qpos, ctrl)
    classification = classify_snapshot_without_certificates(snapshot)
    hold_rollout = None
    strict_gate_passed = False
    if classification == STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND:
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
    return {
        "schema": "task067_r4a31a_true_refinement_attempt_v1",
        "phase": "contact_preserving_refinement" if contact_preserving else "qacc_only_true_continuation",
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
        "strict_nominal_hold_2s": hold_rollout,
        "single_support_active_set": bool(snapshot["single_support_active_set"]),
        "qpos": [float(value) for value in qpos],
        "ctrl": [float(value) for value in ctrl],
        "vector": [float(value) for value in result.x],
        "snapshot": snapshot,
        "active_bounds": {
            "solver_bounds": _active_bound_report(shard, result.x, bounds.lower, bounds.upper),
            "compiled_physical_bounds": _active_bound_report(
                shard,
                result.x,
                bounds.physical_lower,
                bounds.physical_upper,
            ),
        },
        "residual_breakdown": _true_residual_breakdown(
            shard,
            result.x,
            start,
            bounds,
            contact_preserving=contact_preserving,
        ),
        "infeasibility_certificate": {
            "classification": None,
            "reason": "This true-continuation search diagnostic does not produce an independent kinematic or wrench/actuation infeasibility certificate.",
        },
    }


def _route_label(family: str, route: ContinuationRoute) -> str:
    return f"{family}:rf{route.end_range_fraction:g}:seed{route.seed}"


def run_true_continuation_route(
    *,
    source_records: list[dict[str, Any]],
    family: str,
    route: ContinuationRoute,
    nominal_step: float,
    min_step: float,
    max_nfev: int,
    horizon_steps: int,
    trust_region_radius: float | None,
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
            warm_start, bounds = true_warm_start_from_previous_solution(
                target_shard,
                current_qpos,
                current_ctrl,
                trust_region_radius=trust_region_radius,
            )
            qacc_only = refine_true_continuation_from_warm_start(
                target_shard,
                warm_start,
                bounds,
                contact_preserving=False,
                max_nfev=max_nfev,
                horizon_steps=horizon_steps,
            )
            contact_preserving = None
            if not qacc_only["strict_gate_passed"]:
                contact_preserving = refine_true_continuation_from_warm_start(
                    target_shard,
                    warm_start,
                    bounds,
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
                    "strict_qpos_ctrl_from_previous_accepted_step": True,
                    "target_r2_stance_used_to_center_joint_bounds": False,
                },
                "target_morphology_instance_key": target_key,
                "model_xml_sha256": _sha256_bytes(target_shard.xml.encode("utf-8")),
                "warm_start_mapping": mapping,
                "true_warm_start": warm_start.report,
                "bounds_contract": {
                    "joint_bounds": "compiled_physical_joint_limits",
                    "trust_region_radius": trust_region_radius,
                    "target_r2_stance_plus_minus_0p08_rad_forbidden": True,
                },
                "qacc_only_true_continuation": qacc_only,
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
    endpoint_reached = abs(float(current_rf) - float(route.end_range_fraction)) <= 1e-12
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
        "schema": "task067_r4a31a_true_continuation_route_v1",
        "family": family,
        "seed": route.seed,
        "start_range_fraction": route.start_range_fraction,
        "end_range_fraction": route.end_range_fraction,
        "source_morphology_instance_key": source_key,
        "endpoint_label": _route_label(family, route),
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


def artificial_warm_start_clip_violations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    violations = []
    for route in payload.get("routes", []):
        for attempt in route.get("attempts", []):
            clip = attempt["true_warm_start"]["artificial_trust_region_clip"]
            if int(clip["count"]) != 0 or float(clip["max_abs_clip"]) != 0.0:
                violations.append(
                    {
                        "endpoint_label": route["endpoint_label"],
                        "from_range_fraction": attempt["from_range_fraction"],
                        "target_range_fraction": attempt["target_range_fraction"],
                        "count": int(clip["count"]),
                        "max_abs_clip": float(clip["max_abs_clip"]),
                    }
                )
    return violations


def _decide(summary: dict[str, Any]) -> dict[str, str]:
    if int(summary["combined_strict_contract_passed"]) == int(summary["combined_source_records"]):
        return {
            "status": "r4a31a_restored_8_of_8_old_continuation_bug_confirmed",
            "decision": "True continuation recovered every prior failed endpoint, so the old continuation/search path cut away the branch.",
            "next_allowed_work": "Build an explicit contact-wrench equilibrium solver and prepare StanceSolutionV3(qpos_eq, ctrl_eq); still do not integrate controller.",
        }
    return {
        "status": "r4a31a_still_incomplete_enter_fixed_contact_mode_solver",
        "decision": "True continuation did not restore full 8/8 strict coverage; do not keep stacking least_squares starts or max_nfev.",
        "next_allowed_work": "Enter a fixed double-foot contact-mode state-input-wrench constraint solve; do not modify env, controller, generator, kp/kv, or Task061/062.",
    }


def run_true_continuation_correctness(
    *,
    input_json: Path,
    previous_r4a31_artifact: Path,
    family: str,
    routes: tuple[ContinuationRoute, ...],
    nominal_step: float,
    min_step: float,
    max_nfev: int,
    horizon_steps: int,
    trust_region_radius: float | None,
) -> dict[str, Any]:
    source_records = _load_r4a3_records(input_json, family=family)
    route_rows = [
        run_true_continuation_route(
            source_records=source_records,
            family=family,
            route=route,
            nominal_step=nominal_step,
            min_step=min_step,
            max_nfev=max_nfev,
            horizon_steps=horizon_steps,
            trust_region_radius=trust_region_radius,
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
        "endpoints_recovered_by_true_continuation": len(endpoint_recovered),
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
        "infeasibility_certificates": {
            "kinematic_double_support_infeasible": 0,
            "wrench_or_actuation_infeasible": 0,
        },
    }
    payload = {
        "schema": "task067_r4a31a_true_continuation_correctness_v1",
        "source_artifact": str(input_json.resolve()),
        "previous_r4a31_artifact_preserved": {
            "path": str(previous_r4a31_artifact.resolve()),
            "exists": previous_r4a31_artifact.exists(),
            "sha256": _sha256_path(previous_r4a31_artifact) if previous_r4a31_artifact.exists() else None,
        },
        "provenance": {
            "source_artifact_sha256": _sha256_path(input_json),
            "diagnostic_source_sha256": _sha256_path(Path(__file__)),
            "dependency_source_sha256": {
                "whole_body_contact_preserving_continuation.py": _sha256_path(
                    Path(__file__).with_name("whole_body_contact_preserving_continuation.py")
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
                "nominal_step": nominal_step,
                "min_step": min_step,
                "max_nfev": max_nfev,
                "horizon_steps": horizon_steps,
                "hold_seconds": horizon_steps / 50.0,
                "trust_region_radius": trust_region_radius,
                "strict_root_qacc_norm": _STRICT_ROOT_QACC_NORM,
                "strict_joint_qacc_max": _STRICT_JOINT_QACC_MAX,
            },
            "joint_bounds_contract": {
                "compiled_physical_joint_limits": True,
                "optional_previous_solution_centered_trust_region": trust_region_radius is not None,
                "target_r2_stance_plus_minus_0p08_rad_forbidden": True,
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
    violations = artificial_warm_start_clip_violations(payload)
    payload["assertions"] = {
        "all_continuation_step_artificial_warm_start_clip_zero": len(violations) == 0,
        "artificial_warm_start_clip_violations": violations,
        "solver_search_failure_not_promoted_to_physical_infeasible": (
            int(summary["infeasibility_certificates"]["kinematic_double_support_infeasible"]) == 0
            and int(summary["infeasibility_certificates"]["wrench_or_actuation_infeasible"]) == 0
        ),
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
    parser.add_argument("--previous-r4a31-artifact", type=Path, default=_DEFAULT_R4A31_ARTIFACT)
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
    parser.add_argument(
        "--trust-region-radius",
        type=float,
        default=None,
        help="Optional joint trust-region radius centered on the previous accepted branch.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    payload = run_true_continuation_correctness(
        input_json=args.input_json,
        previous_r4a31_artifact=args.previous_r4a31_artifact,
        family=args.family,
        routes=_routes_from_args(args.route),
        nominal_step=args.range_step,
        min_step=args.min_step,
        max_nfev=args.max_nfev,
        horizon_steps=args.horizon_steps,
        trust_region_radius=args.trust_region_radius,
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
