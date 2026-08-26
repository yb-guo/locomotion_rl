"""Task067 R4a.3.1f lexicographic collision-free equilibrium realization.

This diagnostic splits the flat double-foot search into three ordered phases:

1. kinematic qpos-only flat-patch and continuous collision-clearance solve;
2. small penetration continuation into the expected double-foot contact mode;
3. actual MuJoCo qacc/contact realization with bounded ctrl refinement.

Final acceptance is still only the strict actual MuJoCo gate: actual EFC/contact
state, strict qacc, per-foot support load, zero forbidden/self contacts, and a
2 second fixed hold.  Search failure is not promoted to physical infeasibility.
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
)
from h200_locomotion_lab.tools.whole_body_flat_double_foot_realization import (
    _actual_efc_report,
    _baseline_ctrl,
    _flat_geometry_realized,
    _footpad_corners,
    flat_patch_report,
)
from h200_locomotion_lab.tools.whole_body_stance_diagnosis import (
    _support_margin,
)
from h200_locomotion_lab.tools.whole_body_strict_equilibrium_coverage import (
    _foot_load_threshold,
    run_strict_hold_rollout,
)
from h200_locomotion_lab.tools.whole_body_true_continuation_correctness import (
    _compiled_joint_position_bounds,
)

_DEFAULT_R4A31D_INPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31d_contact_taxonomy_collision_free_strict_coverage_4x2.json"
)
_DEFAULT_R4A31E_INPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31e_flat_double_foot_active_set_realization.json"
)
_DEFAULT_OUTPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31f_lexicographic_collision_free_equilibrium_realization.json"
)
_DEFAULT_FAMILY = "biped"
_DEFAULT_POSITIVE_LABEL = "biped:rf0:seed0"

_ROOT_ROLL_PITCH_BOUND = 0.25
_ROOT_YAW_BOUND = 0.35
_KINEMATIC_HEIGHT_SCALE = 5e-4
_DYNAMICS_HEIGHT_SCALE = 8e-4
_KINEMATIC_CLEARANCE_TARGET = 2e-3
_DYNAMICS_CLEARANCE_TARGET = 5e-4
_NONFOOT_FLOOR_CLEARANCE_TARGET = 5e-4
_SUPPORT_MARGIN_TARGET = 1e-3
_JOINT_BOUND_MARGIN = 0.01
_TIGHT_JOINT_RADIUS = 0.18
_PENETRATION_MAX = 0.012
_CTRL_FINITE_DIFFERENCE = 1e-4
_KINEMATIC_REGULARIZATION = 1e-4
_DYNAMICS_REGULARIZATION = 1e-6
_LOAD_DEFICIT_WEIGHT = 12.0
_CLEARANCE_DEFICIT_WEIGHT = 15.0
_MAX_KINEMATIC_BRANCHES = 2

_DEFAULT_KINEMATIC_MAX_NFEV = 900
_DEFAULT_DYNAMICS_MAX_NFEV = 900
_DEFAULT_HORIZON_STEPS = 100
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

COLLISION_FREE_STRICT_FOUND = "collision_free_strict_double_support_equilibrium_found"
ACTUAL_EQUILIBRIUM_FOUND_HOLD_FAILED = "actual_equilibrium_found_but_hold_failed"
KINEMATIC_FLAT_COLLISION_FREE_REACHABLE_DYNAMICS_UNRESOLVED = (
    "kinematic_flat_collision_free_reachable_but_dynamics_unresolved"
)
KINEMATIC_SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE = (
    "kinematic_search_exhausted_without_certificate"
)
LEXICOGRAPHIC_SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE = (
    "lexicographic_search_exhausted_without_certificate"
)


@dataclass(frozen=True, slots=True)
class QposStart:
    name: str
    qpos: Any


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


def _parse_label(label: str) -> tuple[str, float, int]:
    family, rest = label.split(":rf", maxsplit=1)
    range_fraction, seed = rest.split(":seed", maxsplit=1)
    return family, float(range_fraction), int(seed)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _endpoint_from_1e(payload: dict[str, Any], label: str) -> dict[str, Any] | None:
    for row in payload.get("endpoints", []):
        if row.get("label") == label:
            return row
    positive = payload.get("positive_control")
    if isinstance(positive, dict) and positive.get("label") == label:
        return positive
    return None


def _collides(shard: Any, geom1: int, geom2: int) -> bool:
    return bool(
        (
            int(shard.model.geom_contype[geom1])
            & int(shard.model.geom_conaffinity[geom2])
        )
        or (
            int(shard.model.geom_contype[geom2])
            & int(shard.model.geom_conaffinity[geom1])
        )
    )


def _geom_name(shard: Any, geom_id: int) -> str:
    return (
        shard.mujoco.mj_id2name(shard.model, shard.mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        or f"geom_{int(geom_id)}"
    )


def _body_name(shard: Any, body_id: int) -> str:
    return (
        shard.mujoco.mj_id2name(shard.model, shard.mujoco.mjtObj.mjOBJ_BODY, int(body_id))
        or f"body_{int(body_id)}"
    )


def _floor_geom_id(shard: Any) -> int:
    return int(
        shard.mujoco.mj_name2id(
            shard.model,
            shard.mujoco.mjtObj.mjOBJ_GEOM,
            "floor",
        )
    )


def _foot_geom_ids(shard: Any) -> set[int]:
    return {
        int(
            shard.mujoco.mj_name2id(
                shard.model,
                shard.mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )
        )
        for name in shard._foot_geoms
    }


def _collidable_robot_geom_ids(shard: Any) -> list[int]:
    floor = _floor_geom_id(shard)
    rows = []
    for geom_id in range(int(shard.model.ngeom)):
        if geom_id == floor:
            continue
        if (
            int(shard.model.geom_contype[geom_id]) == 0
            and int(shard.model.geom_conaffinity[geom_id]) == 0
        ):
            continue
        rows.append(geom_id)
    return rows


def _self_collision_geom_pairs(shard: Any) -> list[tuple[int, int]]:
    geoms = _collidable_robot_geom_ids(shard)
    pairs: list[tuple[int, int]] = []
    for index, geom1 in enumerate(geoms):
        for geom2 in geoms[index + 1 :]:
            if int(shard.model.geom_bodyid[geom1]) == int(shard.model.geom_bodyid[geom2]):
                continue
            if _collides(shard, geom1, geom2):
                pairs.append((geom1, geom2))
    return pairs


def continuous_collision_clearance_report(shard: Any, data: Any) -> dict[str, Any]:
    """Return continuous signed-distance clearance for every relevant pair."""

    np = shard.np
    foot_ids = _foot_geom_ids(shard)
    floor = _floor_geom_id(shard)
    self_pairs = []
    for geom1, geom2 in _self_collision_geom_pairs(shard):
        fromto = np.zeros(6, dtype=np.float64)
        distance = float(shard.mujoco.mj_geomDistance(shard.model, data, geom1, geom2, 0.2, fromto))
        self_pairs.append(
            {
                "geom1": {
                    "id": geom1,
                    "name": _geom_name(shard, geom1),
                    "body": _body_name(shard, int(shard.model.geom_bodyid[geom1])),
                },
                "geom2": {
                    "id": geom2,
                    "name": _geom_name(shard, geom2),
                    "body": _body_name(shard, int(shard.model.geom_bodyid[geom2])),
                },
                "distance": distance,
                "fromto": [float(value) for value in fromto],
                "is_foot_foot_pair": bool(geom1 in foot_ids and geom2 in foot_ids),
            }
        )
    nonfoot_floor = []
    for geom_id in _collidable_robot_geom_ids(shard):
        if geom_id in foot_ids:
            continue
        fromto = np.zeros(6, dtype=np.float64)
        distance = float(shard.mujoco.mj_geomDistance(shard.model, data, floor, geom_id, 0.2, fromto))
        nonfoot_floor.append(
            {
                "geom": {
                    "id": geom_id,
                    "name": _geom_name(shard, geom_id),
                    "body": _body_name(shard, int(shard.model.geom_bodyid[geom_id])),
                },
                "distance_to_floor": distance,
                "fromto": [float(value) for value in fromto],
            }
        )
    self_distances = [float(row["distance"]) for row in self_pairs]
    nonfoot_floor_distances = [float(row["distance_to_floor"]) for row in nonfoot_floor]
    foot_foot_distances = [
        float(row["distance"]) for row in self_pairs if bool(row["is_foot_foot_pair"])
    ]
    return {
        "schema": "task067_r4a31f_continuous_collision_clearance_v1",
        "uses_integer_contact_count_for_clearance": False,
        "self_collision_geom_pair_count": len(self_pairs),
        "self_collision_geom_pairs": sorted(
            self_pairs,
            key=lambda row: (
                float(row["distance"]),
                str(row["geom1"]["name"]),
                str(row["geom2"]["name"]),
            ),
        ),
        "nonfoot_floor_geom_count": len(nonfoot_floor),
        "nonfoot_floor_distances": sorted(
            nonfoot_floor,
            key=lambda row: (float(row["distance_to_floor"]), str(row["geom"]["name"])),
        ),
        "minimum_self_pair_distance": min(self_distances, default=None),
        "minimum_foot_foot_distance": min(foot_foot_distances, default=None),
        "minimum_nonfoot_floor_distance": min(nonfoot_floor_distances, default=None),
        "self_clearance_target": _KINEMATIC_CLEARANCE_TARGET,
        "nonfoot_floor_clearance_target": _NONFOOT_FLOOR_CLEARANCE_TARGET,
    }


def _support_report(shard: Any, data: Any) -> dict[str, Any]:
    corners = _footpad_corners(shard, data)
    com = _center_of_mass(shard, data)
    margin = _support_margin(
        (float(com[0]), float(com[1])),
        [(float(corner["point"][0]), float(corner["point"][1])) for corner in corners],
    )
    return {
        "com": com,
        "support_margin": margin,
        "corner_count": len(corners),
        "support_margin_target": _SUPPORT_MARGIN_TARGET,
    }


def _qpos_bounds(shard: Any) -> tuple[Any, Any]:
    np = shard.np
    lower = [-_ROOT_ROLL_PITCH_BOUND, -_ROOT_ROLL_PITCH_BOUND, -_ROOT_YAW_BOUND]
    upper = [_ROOT_ROLL_PITCH_BOUND, _ROOT_ROLL_PITCH_BOUND, _ROOT_YAW_BOUND]
    for joint_lower, joint_upper in _compiled_joint_position_bounds(shard):
        lower.append(joint_lower)
        upper.append(joint_upper)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _qpos_vector_from_qpos(shard: Any, qpos: Any) -> Any:
    np = shard.np
    roll, pitch, yaw = _roll_pitch_yaw(qpos[3:7])
    joints = [float(qpos[int(address)]) for address in shard._joint_qpos]
    return np.asarray([roll, pitch, yaw, *joints], dtype=np.float64)


def _apply_qpos_vector(shard: Any, template_qpos: Any, vector: Any) -> Any:
    np = shard.np
    qpos = np.asarray(template_qpos, dtype=np.float64).copy()
    qpos[3:7] = _quat_from_roll_pitch_yaw(float(vector[0]), float(vector[1]), float(vector[2]))
    for address, value in zip(shard._joint_qpos, vector[3:]):
        qpos[int(address)] = float(value)
    _reset_to_qpos(shard, shard.data[0], qpos)
    shard.mujoco.mj_forward(shard.model, shard.data[0])
    corners = _footpad_corners(shard, shard.data[0])
    if corners:
        mean_height = float(shard.np.mean([float(corner["height"]) for corner in corners]))
        shard.data[0].qpos[2] -= mean_height
        shard.mujoco.mj_forward(shard.model, shard.data[0])
    return np.asarray(shard.data[0].qpos, dtype=np.float64).copy()


def _clearance_deficit_residuals(
    clearance: dict[str, Any],
    *,
    self_target: float,
    nonfoot_floor_target: float,
) -> list[float]:
    values = []
    for pair in clearance["self_collision_geom_pairs"]:
        distance = float(pair["distance"])
        values.append(
            _CLEARANCE_DEFICIT_WEIGHT
            * max(0.0, self_target - distance)
            / max(self_target, 1e-12)
        )
    for row in clearance["nonfoot_floor_distances"]:
        distance = float(row["distance_to_floor"])
        values.append(
            _CLEARANCE_DEFICIT_WEIGHT
            * max(0.0, nonfoot_floor_target - distance)
            / max(nonfoot_floor_target, 1e-12)
        )
    return values


def _joint_bound_margin_residuals(vector: Any, lower: Any, upper: Any) -> list[float]:
    values = []
    for index in range(3, len(vector)):
        margin = min(float(vector[index]) - float(lower[index]), float(upper[index]) - float(vector[index]))
        values.append(0.1 * max(0.0, _JOINT_BOUND_MARGIN - margin) / _JOINT_BOUND_MARGIN)
    return values


def _kinematic_residual(shard: Any, vector: Any, start: Any, lower: Any, upper: Any) -> Any:
    np = shard.np
    span = np.maximum(1e-6, upper - lower)
    data = shard.data[0]
    patch = flat_patch_report(shard, data, penetration=0.0)
    heights = [float(corner["height"]) for corner in patch["corners"]]
    mean_height = float(np.mean(heights)) if heights else 0.0
    values = [(height - mean_height) / _KINEMATIC_HEIGHT_SCALE for height in heights]
    clearance = continuous_collision_clearance_report(shard, data)
    values.extend(
        _clearance_deficit_residuals(
            clearance,
            self_target=_KINEMATIC_CLEARANCE_TARGET,
            nonfoot_floor_target=_NONFOOT_FLOOR_CLEARANCE_TARGET,
        )
    )
    support = _support_report(shard, data)
    support_margin = float(support["support_margin"]["margin"])
    values.append(
        10.0
        * max(0.0, _SUPPORT_MARGIN_TARGET - support_margin)
        / max(_SUPPORT_MARGIN_TARGET, 1e-12)
    )
    values.extend(_joint_bound_margin_residuals(vector, lower, upper))
    values.extend(_KINEMATIC_REGULARIZATION * (np.asarray(vector) - start) / span)
    return np.asarray(values, dtype=np.float64)


def _kinematic_score(attempt: dict[str, Any]) -> tuple[float, float, float, float, float]:
    patch = attempt["flat_patch"]
    clearance = attempt["continuous_clearance"]
    support_margin = float(attempt["support"]["support_margin"]["margin"])
    min_self = clearance["minimum_self_pair_distance"]
    min_nonfoot_floor = clearance["minimum_nonfoot_floor_distance"]
    return (
        0.0 if attempt["kinematic_constraints_passed"] else 1.0,
        max(
            float(patch["height_error_to_penetration_max_abs"]),
            float(patch["global_height_spread"]),
        ),
        max(0.0, -float(min_self)) if min_self is not None else 0.0,
        max(0.0, -float(min_nonfoot_floor)) if min_nonfoot_floor is not None else 0.0,
        max(0.0, _SUPPORT_MARGIN_TARGET - support_margin),
    )


def solve_kinematic_phase(
    shard: Any,
    start: QposStart,
    *,
    max_nfev: int,
) -> dict[str, Any]:
    try:
        from scipy import optimize
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for R4a.3.1f kinematic phase") from exc

    np = shard.np
    lower, upper = _qpos_bounds(shard)
    raw_start = _qpos_vector_from_qpos(shard, start.qpos)
    vector0 = np.clip(raw_start, lower, upper)
    start_clip = np.abs(vector0 - raw_start)

    def residual(vector: Any) -> Any:
        _apply_qpos_vector(shard, start.qpos, vector)
        return _kinematic_residual(shard, vector, vector0, lower, upper)

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
    qpos = _apply_qpos_vector(shard, start.qpos, result.x)
    data = shard.data[0]
    patch = flat_patch_report(shard, data, penetration=0.0)
    clearance = continuous_collision_clearance_report(shard, data)
    support = _support_report(shard, data)
    min_self = clearance["minimum_self_pair_distance"]
    min_nonfoot_floor = clearance["minimum_nonfoot_floor_distance"]
    kinematic_passed = bool(
        _flat_geometry_realized(patch)
        and (min_self is None or float(min_self) >= 0.0)
        and (min_nonfoot_floor is None or float(min_nonfoot_floor) >= 0.0)
        and bool(support["support_margin"]["inside"])
        and float(support["support_margin"]["margin"]) >= _SUPPORT_MARGIN_TARGET
    )
    return {
        "schema": "task067_r4a31f_kinematic_phase_attempt_v1",
        "phase": "kinematic_qpos_only",
        "start_name": start.name,
        "variable_contract": {
            "qpos_only": True,
            "compiled_physical_joint_limits": True,
            "flat_double_foot_patch_as_hard_acceptance_constraint": True,
            "continuous_signed_distance_clearance": True,
            "uses_integer_contact_count_for_clearance": False,
            "keeps_com_inside_support": True,
            "keeps_feet_separated": True,
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
        },
        "start_clip": {
            "components": int(np.count_nonzero(start_clip > 0.0)),
            "max_abs": float(np.max(start_clip)) if start_clip.size else 0.0,
        },
        "qpos": [float(value) for value in qpos],
        "vector": [float(value) for value in result.x],
        "flat_patch": patch,
        "continuous_clearance": clearance,
        "support": support,
        "kinematic_constraints_passed": kinematic_passed,
        "infeasibility_certificate": {
            "classification": None,
            "reason": "A kinematic solve failure is not promoted to physical infeasibility.",
        },
    }


def _align_flat_patch_mean_to_penetration(shard: Any, qpos: Any, *, penetration: float) -> Any:
    np = shard.np
    _reset_to_qpos(shard, shard.data[0], qpos)
    shard.mujoco.mj_forward(shard.model, shard.data[0])
    corners = _footpad_corners(shard, shard.data[0])
    if corners:
        mean_height = float(np.mean([float(corner["height"]) for corner in corners]))
        shard.data[0].qpos[2] += -float(penetration) - mean_height
        shard.mujoco.mj_forward(shard.model, shard.data[0])
    return np.asarray(shard.data[0].qpos, dtype=np.float64).copy()


def contact_entry_phase(shard: Any, kinematic_attempt: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for penetration in _PENETRATION_SCHEDULE:
        qpos = _align_flat_patch_mean_to_penetration(
            shard,
            kinematic_attempt["qpos"],
            penetration=penetration,
        )
        data = shard.data[0]
        taxonomy = contact_taxonomy(shard, data)
        patch = flat_patch_report(shard, data, penetration=penetration)
        clearance = continuous_collision_clearance_report(shard, data)
        entered = bool(
            all(
                int(taxonomy["contacts_by_foot"].get(name, 0)) > 0
                for name in sorted(shard._foot_geoms)
            )
            and int(taxonomy["counts"]["forbidden_nonfoot_floor_contacts"]) == 0
            and int(taxonomy["counts"]["self_contacts"]) == 0
        )
        rows.append(
            {
                "schema": "task067_r4a31f_contact_entry_step_v1",
                "phase": "contact_entry_penetration_continuation",
                "kinematic_start_name": kinematic_attempt["start_name"],
                "penetration": penetration,
                "qpos": [float(value) for value in qpos],
                "entered_expected_double_foot_contact_mode": entered,
                "does_not_require_exactly_8_contacts": True,
                "flat_patch": patch,
                "continuous_clearance": clearance,
                "contact_taxonomy": taxonomy,
            }
        )
    return rows


def _ctrl_bounds(shard: Any) -> tuple[Any, Any]:
    np = shard.np
    lower = [
        float(shard.model.actuator_ctrlrange[int(actuator_id)][0])
        for actuator_id in shard._actuator_ids
    ]
    upper = [
        float(shard.model.actuator_ctrlrange[int(actuator_id)][1])
        for actuator_id in shard._actuator_ids
    ]
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _qacc_scaled_vector(shard: Any, qpos: Any, ctrl: Any) -> Any:
    np = shard.np
    _reset_to_qpos(shard, shard.data[0], qpos)
    shard.data[0].ctrl[:] = ctrl
    shard.mujoco.mj_forward(shard.model, shard.data[0])
    qacc = np.asarray(shard.data[0].qacc, dtype=np.float64)
    return np.asarray(
        [
            *list(qacc[:6] / _STRICT_ROOT_QACC_NORM),
            *list(qacc[list(shard._joint_dof)] / _STRICT_JOINT_QACC_MAX),
        ],
        dtype=np.float64,
    )


def bounded_ctrl_qacc_subproblem(shard: Any, qpos: Any, ctrl0: Any | None = None) -> dict[str, Any]:
    """Solve the fixed-qpos bounded affine qacc-vs-ctrl subproblem."""

    try:
        from scipy import optimize
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for R4a.3.1f ctrl subproblem") from exc

    np = shard.np
    actuator_ids = [int(actuator_id) for actuator_id in shard._actuator_ids]
    lower, upper = _ctrl_bounds(shard)
    if ctrl0 is None:
        start_full = _baseline_ctrl(shard, qpos)
    else:
        start_full = np.asarray(ctrl0, dtype=np.float64).copy()
    start = np.asarray([float(start_full[actuator_id]) for actuator_id in actuator_ids], dtype=np.float64)
    start = np.clip(start, lower, upper)
    full0 = np.zeros(shard.model.nu, dtype=np.float64)
    for actuator_id, value in zip(actuator_ids, start):
        full0[actuator_id] = float(value)
    y0 = _qacc_scaled_vector(shard, qpos, full0)
    columns = []
    used_steps = []
    for local_index, actuator_id in enumerate(actuator_ids):
        step = _CTRL_FINITE_DIFFERENCE
        if start[local_index] + step > upper[local_index]:
            step = -_CTRL_FINITE_DIFFERENCE
        if start[local_index] + step < lower[local_index]:
            step = 0.5 * (upper[local_index] - lower[local_index])
        if abs(step) <= 1e-12:
            columns.append(np.zeros_like(y0))
            used_steps.append(0.0)
            continue
        probe = full0.copy()
        probe[actuator_id] = float(start[local_index] + step)
        columns.append((_qacc_scaled_vector(shard, qpos, probe) - y0) / step)
        used_steps.append(float(step))
    matrix = np.asarray(columns, dtype=np.float64).T
    rhs = -y0 + matrix @ start
    result = optimize.lsq_linear(
        matrix,
        rhs,
        bounds=(lower, upper),
        tol=1e-12,
        max_iter=200,
    )
    ctrl = np.zeros(shard.model.nu, dtype=np.float64)
    for actuator_id, value in zip(actuator_ids, result.x):
        ctrl[actuator_id] = float(value)
    actual = _qacc_scaled_vector(shard, qpos, ctrl)
    predicted = y0 + matrix @ (result.x - start)
    prediction_error = actual - predicted
    snapshot = _diagnostic_snapshot(shard, qpos, ctrl)
    return {
        "schema": "task067_r4a31f_bounded_ctrl_qacc_subproblem_v1",
        "phase": "bounded_qacc_ctrl_linear_subproblem",
        "fixed_qpos": True,
        "actuator_variables": len(actuator_ids),
        "qacc_rows": int(y0.size),
        "finite_difference_delta_nominal": _CTRL_FINITE_DIFFERENCE,
        "finite_difference_steps": [float(value) for value in used_steps],
        "solver": {
            "method": "scipy.optimize.lsq_linear",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "nit": int(result.nit),
        },
        "ctrl": [float(value) for value in ctrl],
        "predicted_scaled_qacc_max_abs": float(max((abs(value) for value in predicted), default=0.0)),
        "actual_scaled_qacc_max_abs": float(max((abs(value) for value in actual), default=0.0)),
        "affine_prediction_error_max_abs": float(
            max((abs(value) for value in prediction_error), default=0.0)
        ),
        "actual_snapshot": snapshot,
    }


def _dynamic_bounds(
    shard: Any,
    center_qpos: Any,
    *,
    stage: str,
) -> tuple[Any, Any]:
    np = shard.np
    lower = [-_ROOT_ROLL_PITCH_BOUND, -_ROOT_ROLL_PITCH_BOUND, -_ROOT_YAW_BOUND, 0.0]
    upper = [_ROOT_ROLL_PITCH_BOUND, _ROOT_ROLL_PITCH_BOUND, _ROOT_YAW_BOUND, _PENETRATION_MAX]
    compiled = _compiled_joint_position_bounds(shard)
    for (joint_lower, joint_upper), address in zip(compiled, shard._joint_qpos):
        if stage == "tight_kinematic_tube":
            center = float(center_qpos[int(address)])
            lower.append(max(joint_lower, center - _TIGHT_JOINT_RADIUS))
            upper.append(min(joint_upper, center + _TIGHT_JOINT_RADIUS))
        else:
            lower.append(joint_lower)
            upper.append(joint_upper)
    ctrl_lower, ctrl_upper = _ctrl_bounds(shard)
    lower.extend(float(value) for value in ctrl_lower)
    upper.extend(float(value) for value in ctrl_upper)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _dynamic_vector_from_state(shard: Any, qpos: Any, ctrl: Any, *, penetration: float) -> Any:
    np = shard.np
    roll, pitch, yaw = _roll_pitch_yaw(qpos[3:7])
    joints = [float(qpos[int(address)]) for address in shard._joint_qpos]
    ctrls = [float(ctrl[int(actuator_id)]) for actuator_id in shard._actuator_ids]
    return np.asarray([roll, pitch, yaw, penetration, *joints, *ctrls], dtype=np.float64)


def _apply_dynamic_vector(shard: Any, template_qpos: Any, vector: Any) -> tuple[Any, Any]:
    np = shard.np
    joint_count = len(shard._joint_qpos)
    qpos = np.asarray(template_qpos, dtype=np.float64).copy()
    qpos[3:7] = _quat_from_roll_pitch_yaw(float(vector[0]), float(vector[1]), float(vector[2]))
    for address, value in zip(shard._joint_qpos, vector[4 : 4 + joint_count]):
        qpos[int(address)] = float(value)
    _reset_to_qpos(shard, shard.data[0], qpos)
    shard.mujoco.mj_forward(shard.model, shard.data[0])
    corners = _footpad_corners(shard, shard.data[0])
    if corners:
        mean_height = float(np.mean([float(corner["height"]) for corner in corners]))
        shard.data[0].qpos[2] += -float(vector[3]) - mean_height
    ctrl = np.zeros(shard.model.nu, dtype=np.float64)
    ctrl_values = vector[4 + joint_count : 4 + joint_count + len(shard._actuator_ids)]
    for actuator_id, value in zip(shard._actuator_ids, ctrl_values):
        ctrl[int(actuator_id)] = float(value)
    shard.data[0].ctrl[:] = ctrl
    shard.mujoco.mj_forward(shard.model, shard.data[0])
    return np.asarray(shard.data[0].qpos, dtype=np.float64).copy(), ctrl.copy()


def _dynamic_residual(shard: Any, vector: Any, start: Any, lower: Any, upper: Any) -> Any:
    np = shard.np
    data = shard.data[0]
    qacc = np.asarray(data.qacc, dtype=np.float64)
    values = list(qacc[:6] / _STRICT_ROOT_QACC_NORM)
    values.extend(qacc[list(shard._joint_dof)] / _STRICT_JOINT_QACC_MAX)
    patch = flat_patch_report(shard, data, penetration=float(vector[3]))
    values.extend(
        (float(corner["height"]) + float(vector[3])) / _DYNAMICS_HEIGHT_SCALE
        for corner in patch["corners"]
    )
    clearance = continuous_collision_clearance_report(shard, data)
    values.extend(
        _clearance_deficit_residuals(
            clearance,
            self_target=_DYNAMICS_CLEARANCE_TARGET,
            nonfoot_floor_target=_NONFOOT_FLOOR_CLEARANCE_TARGET,
        )
    )
    support = _support_report(shard, data)
    support_margin = float(support["support_margin"]["margin"])
    values.append(
        5.0
        * max(0.0, _SUPPORT_MARGIN_TARGET - support_margin)
        / max(_SUPPORT_MARGIN_TARGET, 1e-12)
    )
    taxonomy = contact_taxonomy(shard, data)
    min_load = _foot_load_threshold(shard)
    values.extend(
        _LOAD_DEFICIT_WEIGHT
        * max(0.0, min_load - float(taxonomy["normal_force_by_foot"].get(name, 0.0)))
        / max(min_load, 1e-12)
        for name in sorted(shard._foot_geoms)
    )
    span = np.maximum(1e-6, upper - lower)
    values.extend(_DYNAMICS_REGULARIZATION * (np.asarray(vector) - start) / span)
    return np.asarray(values, dtype=np.float64)


def _attempt_hold_if_strict(
    shard: Any,
    qpos: Any,
    ctrl: Any,
    snapshot: dict[str, Any],
    *,
    horizon_steps: int,
) -> tuple[bool, dict[str, Any] | None]:
    if classify_snapshot_without_certificates(snapshot) != STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND:
        return False, None
    hold = run_strict_hold_rollout(
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
    return bool(hold["passed"]), hold


def _dynamics_classification(
    *,
    actual_classification: str,
    hold_passed: bool,
    flat_realized: bool,
    kinematic_phase_passed: bool,
) -> str:
    if actual_classification == STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND and hold_passed and flat_realized:
        return COLLISION_FREE_STRICT_FOUND
    if actual_classification == STRICT_DOUBLE_SUPPORT_EQUILIBRIUM_FOUND and flat_realized:
        return ACTUAL_EQUILIBRIUM_FOUND_HOLD_FAILED
    if kinematic_phase_passed:
        return KINEMATIC_FLAT_COLLISION_FREE_REACHABLE_DYNAMICS_UNRESOLVED
    return LEXICOGRAPHIC_SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE


def solve_dynamics_attempt(
    shard: Any,
    contact_entry: dict[str, Any],
    ctrl_subproblem: dict[str, Any],
    *,
    stage: str,
    max_nfev: int,
    horizon_steps: int,
) -> dict[str, Any]:
    try:
        from scipy import optimize
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SciPy is required for R4a.3.1f dynamics phase") from exc

    np = shard.np
    template_qpos = np.asarray(contact_entry["qpos"], dtype=np.float64)
    ctrl0 = np.asarray(ctrl_subproblem["ctrl"], dtype=np.float64)
    lower, upper = _dynamic_bounds(shard, template_qpos, stage=stage)
    raw_start = _dynamic_vector_from_state(
        shard,
        template_qpos,
        ctrl0,
        penetration=float(contact_entry["penetration"]),
    )
    vector0 = np.clip(raw_start, lower, upper)

    def residual(vector: Any) -> Any:
        _apply_dynamic_vector(shard, template_qpos, vector)
        return _dynamic_residual(shard, vector, vector0, lower, upper)

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
    qpos, ctrl = _apply_dynamic_vector(shard, template_qpos, result.x)
    data = shard.data[0]
    patch = flat_patch_report(shard, data, penetration=float(result.x[3]))
    clearance = continuous_collision_clearance_report(shard, data)
    support = _support_report(shard, data)
    snapshot = _diagnostic_snapshot(shard, qpos, ctrl)
    actual_classification = classify_snapshot_without_certificates(snapshot)
    flat_realized = _flat_geometry_realized(patch)
    hold_passed, hold = _attempt_hold_if_strict(
        shard,
        qpos,
        ctrl,
        snapshot,
        horizon_steps=horizon_steps,
    )
    classification = _dynamics_classification(
        actual_classification=actual_classification,
        hold_passed=hold_passed,
        flat_realized=flat_realized,
        kinematic_phase_passed=True,
    )
    joint_count = len(shard._joint_qpos)
    ctrl_count = len(shard._actuator_ids)
    return {
        "schema": "task067_r4a31f_dynamics_phase_attempt_v1",
        "phase": "dynamics_actual_qacc_ctrl_refinement",
        "stage": stage,
        "contact_entry_penetration": float(contact_entry["penetration"]),
        "kinematic_start_name": contact_entry["kinematic_start_name"],
        "bounded_ctrl_subproblem": ctrl_subproblem,
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
            "flat_and_clearance_kept_by_continuous_residuals": True,
            "uses_integer_contact_count_residual": False,
            "compiled_physical_joint_limits": stage == "full_joint_limit_fallback",
            "tight_kinematic_tube_radius": _TIGHT_JOINT_RADIUS
            if stage == "tight_kinematic_tube"
            else None,
            "final_acceptance_requires_actual_mujoco_gate": True,
        },
        "qpos": [float(value) for value in qpos],
        "ctrl": [float(value) for value in ctrl],
        "vector": [float(value) for value in result.x],
        "flat_patch": patch,
        "flat_geometry_realized": flat_realized,
        "continuous_clearance": clearance,
        "support": support,
        "actual_snapshot": snapshot,
        "actual_efc": _actual_efc_report(shard, data),
        "actual_classification": actual_classification,
        "classification": classification,
        "strict_gate_passed": classification == COLLISION_FREE_STRICT_FOUND,
        "actual_strict_gate_and_hold_passed": bool(hold_passed and flat_realized),
        "strict_nominal_hold_2s": hold,
        "active_bounds": _active_bound_report(
            shard,
            result.x[: 4 + joint_count + ctrl_count],
            lower[: 4 + joint_count + ctrl_count],
            upper[: 4 + joint_count + ctrl_count],
        ),
        "infeasibility_certificate": {
            "classification": None,
            "reason": "Dynamics search failure is not an independent physics infeasibility certificate.",
        },
    }


def _dynamic_attempt_sort_key(attempt: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    snapshot = attempt["actual_snapshot"]
    patch = attempt["flat_patch"]
    clearance = attempt["continuous_clearance"]
    min_self = clearance["minimum_self_pair_distance"]
    min_nonfoot_floor = clearance["minimum_nonfoot_floor_distance"]
    return (
        0.0 if attempt["strict_gate_passed"] else 1.0,
        float(snapshot["root_qacc_norm"]) + float(snapshot["joint_qacc_max"]),
        max(
            float(patch["height_error_to_penetration_max_abs"]),
            float(patch["global_height_spread"]),
        ),
        max(0.0, -float(min_self)) if min_self is not None else 0.0,
        max(0.0, -float(min_nonfoot_floor)) if min_nonfoot_floor is not None else 0.0,
        float(attempt["solver"]["cost"]),
    )


def _preview(attempt: dict[str, Any]) -> dict[str, Any]:
    snapshot = attempt["actual_snapshot"]
    patch = attempt["flat_patch"]
    clearance = attempt["continuous_clearance"]
    return {
        "stage": attempt["stage"],
        "kinematic_start_name": attempt["kinematic_start_name"],
        "penetration": attempt["contact_entry_penetration"],
        "classification": attempt["classification"],
        "strict_gate_passed": attempt["strict_gate_passed"],
        "nfev": attempt["solver"]["nfev"],
        "optimality": attempt["solver"]["optimality"],
        "root_qacc_norm": snapshot["root_qacc_norm"],
        "joint_qacc_max": snapshot["joint_qacc_max"],
        "support_mode": snapshot["support_mode"],
        "self_contacts": snapshot["contact"].get("self_contacts", 0),
        "non_foot_contacts": snapshot["contact"]["non_foot_contacts"],
        "flat_height_error_max_abs": patch["height_error_to_penetration_max_abs"],
        "flat_height_spread": patch["global_height_spread"],
        "minimum_self_pair_distance": clearance["minimum_self_pair_distance"],
        "minimum_nonfoot_floor_distance": clearance["minimum_nonfoot_floor_distance"],
    }


def _start_states(
    shard: Any,
    record_1d: dict[str, Any],
    record_1e: dict[str, Any] | None,
) -> list[QposStart]:
    np = shard.np
    starts: list[QposStart] = []
    if record_1e is not None:
        starts.append(
            QposStart(
                "r4a31e_best",
                np.asarray(record_1e["best"]["qpos"], dtype=np.float64),
            )
        )
    starts.append(
        QposStart(
            "r4a31d_best",
            np.asarray(record_1d["strict_refinement"]["best"]["qpos"], dtype=np.float64),
        )
    )
    starts.append(QposStart("target_r2_stance", np.asarray(_stance_qpos(shard), dtype=np.float64)))
    return starts


def solve_record(
    record_1d: dict[str, Any],
    record_1e: dict[str, Any] | None,
    *,
    role: str,
    kinematic_max_nfev: int,
    dynamics_max_nfev: int,
    horizon_steps: int,
) -> dict[str, Any]:
    shard, key = _build_shard_for(
        record_1d["family"],
        int(record_1d["seed"]),
        float(record_1d["range_fraction"]),
    )
    kinematic_attempts = [
        solve_kinematic_phase(shard, start, max_nfev=kinematic_max_nfev)
        for start in _start_states(shard, record_1d, record_1e)
    ]
    good_kinematics = [
        attempt for attempt in sorted(kinematic_attempts, key=_kinematic_score)
        if bool(attempt["kinematic_constraints_passed"])
    ]
    contact_steps: list[dict[str, Any]] = []
    entered_steps: list[dict[str, Any]] = []
    dynamics_attempts: list[dict[str, Any]] = []
    full_fallback_executed = False
    for kinematic_attempt in good_kinematics[:_MAX_KINEMATIC_BRANCHES]:
        steps = contact_entry_phase(shard, kinematic_attempt)
        contact_steps.extend(steps)
        entered_steps.extend(
            step for step in steps if bool(step["entered_expected_double_foot_contact_mode"])
        )
    if entered_steps:
        tight_attempts = []
        for step in entered_steps:
            ctrl = bounded_ctrl_qacc_subproblem(shard, step["qpos"])
            attempt = solve_dynamics_attempt(
                shard,
                step,
                ctrl,
                stage="tight_kinematic_tube",
                max_nfev=dynamics_max_nfev,
                horizon_steps=horizon_steps,
            )
            tight_attempts.append(attempt)
            if attempt["strict_gate_passed"]:
                break
        dynamics_attempts.extend(tight_attempts)
        if not any(attempt["strict_gate_passed"] for attempt in tight_attempts):
            full_fallback_executed = True
            for step in entered_steps:
                ctrl = bounded_ctrl_qacc_subproblem(shard, step["qpos"])
                attempt = solve_dynamics_attempt(
                    shard,
                    step,
                    ctrl,
                    stage="full_joint_limit_fallback",
                    max_nfev=dynamics_max_nfev,
                    horizon_steps=horizon_steps,
                )
                dynamics_attempts.append(attempt)
                if attempt["strict_gate_passed"]:
                    break
    best_dynamics = min(dynamics_attempts, key=_dynamic_attempt_sort_key) if dynamics_attempts else None
    if best_dynamics is None:
        final_classification = (
            KINEMATIC_FLAT_COLLISION_FREE_REACHABLE_DYNAMICS_UNRESOLVED
            if good_kinematics
            else KINEMATIC_SEARCH_EXHAUSTED_WITHOUT_CERTIFICATE
        )
    else:
        final_classification = best_dynamics["classification"]
    return {
        "schema": "task067_r4a31f_lexicographic_record_v1",
        "role": role,
        "label": _label(record_1d),
        "family": record_1d["family"],
        "seed": int(record_1d["seed"]),
        "range_fraction": float(record_1d["range_fraction"]),
        "morphology_instance_key": key,
        "model_xml_sha256": _sha256_bytes(shard.xml.encode("utf-8")),
        "source_r4a31d_strict_contract_passed": bool(record_1d.get("strict_contract_passed", False)),
        "source_r4a31e_strict_recovered": bool(record_1e.get("strict_recovered", False))
        if record_1e is not None
        else None,
        "kinematic_phase": {
            "attempt_count": len(kinematic_attempts),
            "passed_attempt_count": len(good_kinematics),
            "best": min(kinematic_attempts, key=_kinematic_score),
            "attempt_preview": [
                {
                    "start_name": attempt["start_name"],
                    "kinematic_constraints_passed": attempt["kinematic_constraints_passed"],
                    "nfev": attempt["solver"]["nfev"],
                    "optimality": attempt["solver"]["optimality"],
                    "flat_height_error_max_abs": attempt["flat_patch"]["height_error_to_penetration_max_abs"],
                    "flat_height_spread": attempt["flat_patch"]["global_height_spread"],
                    "minimum_self_pair_distance": attempt["continuous_clearance"][
                        "minimum_self_pair_distance"
                    ],
                    "minimum_nonfoot_floor_distance": attempt["continuous_clearance"][
                        "minimum_nonfoot_floor_distance"
                    ],
                    "support_margin": attempt["support"]["support_margin"]["margin"],
                }
                for attempt in sorted(kinematic_attempts, key=_kinematic_score)
            ],
        },
        "contact_entry_phase": {
            "steps_tested": len(contact_steps),
            "entered_expected_mode_count": len(entered_steps),
            "does_not_require_exactly_8_contacts": True,
            "steps": contact_steps,
        },
        "dynamics_phase": {
            "attempt_count": len(dynamics_attempts),
            "full_joint_limit_fallback_executed": full_fallback_executed,
            "best": best_dynamics,
            "attempt_preview": [
                _preview(attempt)
                for attempt in sorted(dynamics_attempts, key=_dynamic_attempt_sort_key)[:8]
            ],
        },
        "strict_recovered": bool(best_dynamics and best_dynamics["strict_gate_passed"]),
        "final_classification": final_classification,
        "infeasibility_certificate": {
            "classification": None,
            "reason": "R4a.3.1f emits no geometry or wrench/actuation infeasibility certificate.",
        },
    }


def _decide(summary: dict[str, Any]) -> dict[str, str]:
    if int(summary["combined_collision_free_strict_passed"]) == int(summary["source_records"]):
        return {
            "status": "r4a31f_collision_free_strict_coverage_restored_8_of_8",
            "decision": (
                "Lexicographic flat/collision-free/contact-entry/dynamics realization restored "
                "strict coverage for the full 4x2 set; the remaining 1e failures were search "
                "formulation failures, not physical infeasibility certificates."
            ),
            "next_allowed_work": (
                "Design the explicit contact-wrench equilibrium solver and prepare "
                "StanceSolutionV3(qpos_eq, ctrl_eq); do not integrate feedback yet."
            ),
        }
    if int(summary["actual_equilibrium_found_hold_failed"]) > 0:
        return {
            "status": "r4a31f_actual_equilibrium_found_hold_failed",
            "decision": "At least one endpoint reaches actual strict equilibrium but fails the 2s hold.",
            "next_allowed_work": "Diagnose hold/contact drift before feedback integration.",
        }
    return {
        "status": "r4a31f_lexicographic_search_incomplete_no_physical_certificate",
        "decision": (
            "Lexicographic realization did not restore full strict coverage and produced no "
            "independent physical infeasibility certificate."
        ),
        "next_allowed_work": (
            "Continue fixed double-foot actual-contact refinement; do not change generator, "
            "controller, feedback authority, or Task061/062."
        ),
    }


def run_lexicographic_realization(
    *,
    r4a31d_json: Path,
    r4a31e_json: Path,
    family: str,
    positive_label: str,
    kinematic_max_nfev: int,
    dynamics_max_nfev: int,
    horizon_steps: int,
) -> dict[str, Any]:
    records_1d = _load_r4a3_records(r4a31d_json, family=family)
    payload_1e = _load_json(r4a31e_json)
    incomplete_labels = list(payload_1e["summary"]["combined_incomplete_labels"])
    endpoint_rows = [
        solve_record(
            _record_by_label(records_1d, label),
            _endpoint_from_1e(payload_1e, label),
            role="failed_endpoint",
            kinematic_max_nfev=kinematic_max_nfev,
            dynamics_max_nfev=dynamics_max_nfev,
            horizon_steps=horizon_steps,
        )
        for label in incomplete_labels
    ]
    positive_control = solve_record(
        _record_by_label(records_1d, positive_label),
        _endpoint_from_1e(payload_1e, positive_label),
        role="positive_control",
        kinematic_max_nfev=kinematic_max_nfev,
        dynamics_max_nfev=dynamics_max_nfev,
        horizon_steps=horizon_steps,
    )
    source_accepted = set(payload_1e["summary"]["combined_accepted_labels"])
    recovered = {row["label"] for row in endpoint_rows if bool(row["strict_recovered"])}
    combined = source_accepted | recovered
    class_counts: dict[str, int] = {}
    for row in endpoint_rows:
        class_counts[row["final_classification"]] = class_counts.get(row["final_classification"], 0) + 1
    summary = {
        "source_records": len(records_1d),
        "source_r4a31e_collision_free_strict_passed": len(source_accepted),
        "failed_endpoints_tested": len(endpoint_rows),
        "failed_endpoint_labels": incomplete_labels,
        "positive_controls_tested": 1,
        "positive_controls_passed_same_path": int(positive_control["strict_recovered"]),
        "endpoints_recovered": len(recovered),
        "combined_collision_free_strict_passed": len(combined),
        "combined_accepted_labels": sorted(combined),
        "combined_incomplete_labels": [
            _label(record) for record in records_1d if _label(record) not in combined
        ],
        "endpoint_classifications": {
            row["label"]: row["final_classification"] for row in endpoint_rows
        },
        "endpoint_classification_counts": class_counts,
        "kinematic_flat_collision_free_reachable": sum(
            row["kinematic_phase"]["passed_attempt_count"] > 0 for row in endpoint_rows
        ),
        "contact_entry_expected_mode_reached": sum(
            row["contact_entry_phase"]["entered_expected_mode_count"] > 0 for row in endpoint_rows
        ),
        "full_joint_limit_dynamics_fallback_executed_count": sum(
            bool(row["dynamics_phase"]["full_joint_limit_fallback_executed"])
            for row in endpoint_rows
        ),
        "actual_equilibrium_found_hold_failed": sum(
            row["final_classification"] == ACTUAL_EQUILIBRIUM_FOUND_HOLD_FAILED
            for row in endpoint_rows
        ),
        "infeasibility_certificates": {
            "kinematic_double_support_infeasible": 0,
            "wrench_or_actuation_infeasible": 0,
        },
    }
    return {
        "schema": "task067_r4a31f_lexicographic_collision_free_equilibrium_realization_v1",
        "source_artifacts": {
            "r4a31d": str(r4a31d_json.resolve()),
            "r4a31e": str(r4a31e_json.resolve()),
        },
        "provenance": {
            "source_artifact_sha256": {
                "r4a31d": _sha256_path(r4a31d_json),
                "r4a31e": _sha256_path(r4a31e_json),
            },
            "diagnostic_source_sha256": _sha256_path(Path(__file__)),
            "dependency_source_sha256": {
                "whole_body_contact_taxonomy.py": _sha256_path(
                    Path(__file__).with_name("whole_body_contact_taxonomy.py")
                ),
                "whole_body_flat_double_foot_realization.py": _sha256_path(
                    Path(__file__).with_name("whole_body_flat_double_foot_realization.py")
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
                "positive_label": positive_label,
                "kinematic_max_nfev": kinematic_max_nfev,
                "dynamics_max_nfev": dynamics_max_nfev,
                "horizon_steps": horizon_steps,
                "hold_seconds": horizon_steps / 50.0,
                "penetration_schedule": list(_PENETRATION_SCHEDULE),
                "kinematic_branches_per_record": _MAX_KINEMATIC_BRANCHES,
                "kinematic_clearance_target": _KINEMATIC_CLEARANCE_TARGET,
                "dynamics_clearance_target": _DYNAMICS_CLEARANCE_TARGET,
                "tight_joint_radius": _TIGHT_JOINT_RADIUS,
                "final_acceptance": {
                    "actual_mujoco_efc": True,
                    "actual_qacc": True,
                    "actual_per_foot_load": True,
                    "two_second_hold": True,
                    "forbidden_nonfoot_floor_contacts": 0,
                    "self_contacts": 0,
                    "exactly_8_contact_points_required": False,
                },
            },
            "diagnostic_scope": {
                "lexicographic_phases": [
                    "kinematic_qpos_only",
                    "contact_entry_penetration_continuation",
                    "bounded_qacc_ctrl_linear_subproblem",
                    "dynamics_actual_qacc_ctrl_refinement",
                ],
                "modifies_public_env_controller_generator_or_gains": False,
                "prepares_stance_solution_v3": False,
                "restores_feedback_or_task061_062": False,
            },
        },
        "embodiment_contract_version": PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
        "embodiment_contract_hash": PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
        "stance_solution_contract_version": STANCE_SOLUTION_CONTRACT_VERSION,
        "stance_solution_contract_hash": STANCE_SOLUTION_CONTRACT_HASH,
        "summary": summary,
        "decision": _decide(summary),
        "assertions": {
            "uses_lexicographic_phases": True,
            "all_remaining_r4a31e_incomplete_endpoints_tested": len(endpoint_rows) == len(incomplete_labels) == 3,
            "positive_control_same_path_tested": positive_control["role"] == "positive_control",
            "kinematic_phase_uses_continuous_signed_distance_not_contact_count": True,
            "dynamics_phase_uses_no_integer_contact_count_residual": True,
            "final_acceptance_does_not_require_exactly_8_contacts": True,
            "search_failure_not_promoted_to_physical_infeasible": (
                int(summary["infeasibility_certificates"]["kinematic_double_support_infeasible"]) == 0
                and int(summary["infeasibility_certificates"]["wrench_or_actuation_infeasible"]) == 0
            ),
        },
        "positive_control": positive_control,
        "endpoints": endpoint_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r4a31d-json", type=Path, default=_DEFAULT_R4A31D_INPUT)
    parser.add_argument("--r4a31e-json", type=Path, default=_DEFAULT_R4A31E_INPUT)
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--family", default=_DEFAULT_FAMILY)
    parser.add_argument("--positive-label", default=_DEFAULT_POSITIVE_LABEL)
    parser.add_argument("--kinematic-max-nfev", type=int, default=_DEFAULT_KINEMATIC_MAX_NFEV)
    parser.add_argument("--dynamics-max-nfev", type=int, default=_DEFAULT_DYNAMICS_MAX_NFEV)
    parser.add_argument("--horizon-steps", type=int, default=_DEFAULT_HORIZON_STEPS)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    payload = run_lexicographic_realization(
        r4a31d_json=args.r4a31d_json,
        r4a31e_json=args.r4a31e_json,
        family=args.family,
        positive_label=args.positive_label,
        kinematic_max_nfev=args.kinematic_max_nfev,
        dynamics_max_nfev=args.dynamics_max_nfev,
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
