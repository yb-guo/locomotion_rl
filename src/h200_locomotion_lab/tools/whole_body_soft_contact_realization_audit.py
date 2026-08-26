"""Task067 R4a.3.1c soft-contact force-closure / realization audit.

This diagnostic calibrates the explicit force/Jacobian equations against
MuJoCo before interpreting the R4a.3.1b rigid contact-wrench candidates.  It
uses the five known strict actual equilibria as positive controls, then audits
the three failed fixed-contact candidates and sweeps root penetration with
qpos joints and ctrl fixed.

The tool is diagnostic-only.  It does not modify the public environment,
controller, generator, actuator gains, reward, observation/action schema, or
motor process.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from itertools import pairwise
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
    _build_shard_for,
    _diagnostic_snapshot,
    _load_r4a3_records,
    _sha256_bytes,
    _sha256_path,
)
from h200_locomotion_lab.tools.whole_body_contact_taxonomy import (
    contact_taxonomy,
    efc_qfrc_for_contact_indices,
    full_efc_qfrc,
)
from h200_locomotion_lab.tools.whole_body_dynamic_balance_diagnosis import (
    _contact_report,
    _reset_to_qpos,
)
from h200_locomotion_lab.tools.whole_body_equilibrium_audit import (
    strict_actual_equilibrium,
)
from h200_locomotion_lab.tools.whole_body_fixed_contact_mode_wrench_solve import (
    ContactPointSpec,
    FixedContactMode,
    _all_footpad_corner_points,
    _foot_load_threshold,
    _normal_force_by_foot,
    _qfrc_from_contact_forces,
    _selected_contact_points,
    _weight,
)

_DEFAULT_R4A3_INPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a3_strict_equilibrium_coverage_4x2.json"
)
_DEFAULT_R4A31B_INPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31b_fixed_contact_mode_wrench_solve_3fail.json"
)
_DEFAULT_OUTPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31c_soft_contact_force_closure_realization_audit.json"
)
_DEFAULT_FAMILY = "biped"
_DYNAMICS_RESIDUAL_MAX = 1e-8
_ACTUATOR_QFRC_MAX_DIFF = 1e-8
_EFC_CONTACT_QFRC_MAX_DIFF = 1e-8
_HAND_CONTACT_QFRC_MAX_DIFF = 1e-6
_CONTACT_HEIGHT_TOL = 1e-7
_PENETRATION_SWEEP_MM = tuple(float(value) for value in range(13))


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _mode_from_manifest(manifest: dict[str, Any]) -> FixedContactMode:
    return FixedContactMode(
        name=str(manifest["name"]),
        points=tuple(
            ContactPointSpec(str(point["foot"]), float(point["sx"]), float(point["sy"]))
            for point in manifest["points"]
        ),
    )


def _set_state(shard: Any, qpos: Any, ctrl: Any) -> None:
    np = shard.np
    _reset_to_qpos(shard, shard.data[0], np.asarray(qpos, dtype=np.float64))
    shard.data[0].ctrl[:] = np.asarray(ctrl, dtype=np.float64)
    shard.mujoco.mj_forward(shard.model, shard.data[0])


def _mass_matrix(shard: Any, data: Any) -> Any:
    np = shard.np
    matrix = np.zeros((shard.model.nv, shard.model.nv), dtype=np.float64)
    shard.mujoco.mj_fullM(shard.model, matrix, data.qM)
    return matrix


def _dof_names(shard: Any) -> list[dict[str, Any]]:
    rows = [
        {"dof": index, "name": f"root_dof_{index}", "kind": "root"}
        for index in range(6)
    ]
    joint_by_dof = {
        int(dof): joint
        for joint, dof in zip(shard.blueprint.joints, shard._joint_dof)
    }
    for dof in range(6, int(shard.model.nv)):
        joint = joint_by_dof.get(dof)
        if joint is None:
            rows.append({"dof": dof, "name": f"dof_{dof}", "kind": "unknown"})
        else:
            rows.append(
                {
                    "dof": dof,
                    "name": joint.name,
                    "kind": "joint",
                    "semantic_slot": joint.semantic_slot,
                }
            )
    return rows


def _qfrc_norm_report(shard: Any, left: Any, right: Any) -> dict[str, Any]:
    np = shard.np
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {
        "norm": float(np.linalg.norm(delta)),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "root_norm": float(np.linalg.norm(delta[:6])),
        "joint_max_abs": max((abs(float(delta[int(dof)])) for dof in shard._joint_dof), default=0.0),
    }


def _actuator_qfrc_from_ctrl_hand(shard: Any, data: Any, ctrl: Any) -> Any:
    np = shard.np
    qfrc = np.zeros(shard.model.nv, dtype=np.float64)
    for actuator_id, qpos_address, dof_address in zip(
        shard._actuator_ids,
        shard._joint_qpos,
        shard._joint_dof,
    ):
        kp = max(1e-9, float(shard.model.actuator_gainprm[int(actuator_id), 0]))
        raw_force = kp * (float(ctrl[int(actuator_id)]) - float(data.qpos[int(qpos_address)]))
        force_lower, force_upper = (
            float(value) for value in shard.model.actuator_forcerange[int(actuator_id)]
        )
        qfrc[int(dof_address)] = min(force_upper, max(force_lower, raw_force))
    return qfrc


def _actuator_closure(shard: Any, ctrl: Any) -> dict[str, Any]:
    data = shard.data[0]
    hand = _actuator_qfrc_from_ctrl_hand(shard, data, ctrl)
    actual = shard.np.asarray(data.qfrc_actuator, dtype=shard.np.float64).copy()
    report = _qfrc_norm_report(shard, hand, actual)
    rows = []
    for joint, actuator_id, dof in zip(
        shard.blueprint.joints,
        shard._actuator_ids,
        shard._joint_dof,
    ):
        rows.append(
            {
                "joint": joint.name,
                "semantic_slot": joint.semantic_slot,
                "actuator_id": int(actuator_id),
                "dof": int(dof),
                "hand_qfrc": float(hand[int(dof)]),
                "data_qfrc_actuator": float(actual[int(dof)]),
                "diff": float(hand[int(dof)] - actual[int(dof)]),
            }
        )
    report["passed"] = report["max_abs"] <= _ACTUATOR_QFRC_MAX_DIFF
    report["rows"] = rows
    return report


def _contact_force_world(frame: Any, force_local: Any) -> tuple[Any, Any]:
    import numpy as np

    frame_matrix = np.asarray(frame, dtype=np.float64).reshape(3, 3)
    return frame_matrix.T @ force_local[:3], frame_matrix.T @ force_local[3:]


def _nearest_corner(contact_pos: Any, corners: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    point = np.asarray(contact_pos, dtype=np.float64)
    return min(
        corners,
        key=lambda row: float(np.linalg.norm(point - np.asarray(row["point"], dtype=np.float64))),
    )


def _actual_contact_records_and_hand_qfrc(shard: Any) -> tuple[list[dict[str, Any]], Any]:
    import numpy as np

    data = shard.data[0]
    floor_id = int(shard.mujoco.mj_name2id(shard.model, shard.mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    foot_by_id = {
        int(shard.mujoco.mj_name2id(shard.model, shard.mujoco.mjtObj.mjOBJ_GEOM, foot)): foot
        for foot in shard._foot_geoms
    }
    corners_by_foot = {
        foot: _footpad_corner_points
        for foot, _footpad_corner_points in (
            (foot, [row for row in _all_footpad_corner_points(shard, data) if row["foot"] == foot])
            for foot in sorted(shard._foot_geoms)
        )
    }
    qfrc = np.zeros(shard.model.nv, dtype=np.float64)
    records = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        if floor_id not in (int(contact.geom1), int(contact.geom2)):
            continue
        other = int(contact.geom2) if int(contact.geom1) == floor_id else int(contact.geom1)
        foot = foot_by_id.get(other)
        if foot is None:
            continue
        force_local = np.zeros(6, dtype=np.float64)
        shard.mujoco.mj_contactForce(shard.model, data, contact_index, force_local)
        force_world, torque_world = _contact_force_world(contact.frame, force_local)
        jacp = np.zeros((3, shard.model.nv), dtype=np.float64)
        jacr = np.zeros((3, shard.model.nv), dtype=np.float64)
        body_id = int(shard.model.geom_bodyid[other])
        position = np.asarray(contact.pos, dtype=np.float64)
        shard.mujoco.mj_jac(shard.model, data, jacp, jacr, position, body_id)
        contribution = jacp.T @ force_world + jacr.T @ torque_world
        qfrc += contribution
        nearest = _nearest_corner(position, corners_by_foot[foot])
        frame_matrix = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        records.append(
            {
                "contact_index": contact_index,
                "geom1": int(contact.geom1),
                "geom2": int(contact.geom2),
                "foot": foot,
                "position": [float(value) for value in position],
                "distance": float(contact.dist),
                "normal_world": [float(value) for value in frame_matrix[0]],
                "force_local": [float(value) for value in force_local],
                "force_world_from_frame_transpose": [float(value) for value in force_world],
                "normal_force": max(0.0, float(force_local[0])),
                "nearest_corner": {
                    "key": nearest["key"],
                    "sx": float(nearest["sx"]),
                    "sy": float(nearest["sy"]),
                    "height": float(nearest["height"]),
                    "distance": float(np.linalg.norm(position - np.asarray(nearest["point"], dtype=np.float64))),
                },
                "qfrc_contribution": [float(value) for value in contribution],
            }
        )
    return records, qfrc


def _contact_closure(shard: Any) -> dict[str, Any]:
    np = shard.np
    data = shard.data[0]
    taxonomy = contact_taxonomy(shard, data)
    actual_records, hand_qfrc = _actual_contact_records_and_hand_qfrc(shard)
    foot_floor_contact_indices = {int(record["contact_index"]) for record in actual_records}
    filtered_foot_floor_efc = efc_qfrc_for_contact_indices(
        shard,
        data,
        foot_floor_contact_indices,
    )
    full_efc = full_efc_qfrc(shard, data)
    constraint = np.asarray(data.qfrc_constraint, dtype=np.float64).copy()
    hand_report = _qfrc_norm_report(shard, hand_qfrc, filtered_foot_floor_efc)
    filtered_report = _qfrc_norm_report(shard, filtered_foot_floor_efc, hand_qfrc)
    full_report = _qfrc_norm_report(shard, full_efc, constraint)
    hand_report["passed"] = hand_report["max_abs"] <= _HAND_CONTACT_QFRC_MAX_DIFF
    filtered_report["passed"] = filtered_report["max_abs"] <= _EFC_CONTACT_QFRC_MAX_DIFF
    full_report["passed"] = full_report["max_abs"] <= _EFC_CONTACT_QFRC_MAX_DIFF
    return {
        "contact_taxonomy": taxonomy,
        "actual_contact_records": actual_records,
        "foot_hand_force_jacobian_vs_filtered_foot_floor_efc": hand_report,
        "filtered_foot_floor_efc_vs_foot_hand_force_jacobian": filtered_report,
        "full_mujoco_efc_jacobian_vs_full_qfrc_constraint": full_report,
        "foot_hand_force_jacobian_qfrc": [float(value) for value in hand_qfrc],
        "filtered_foot_floor_efc_jacobian_qfrc": [float(value) for value in filtered_foot_floor_efc],
        "full_mujoco_efc_jacobian_qfrc": [float(value) for value in full_efc],
        "qfrc_constraint": [float(value) for value in constraint],
    }


def _dynamics_closure(shard: Any) -> dict[str, Any]:
    np = shard.np
    data = shard.data[0]
    lhs = _mass_matrix(shard, data) @ np.asarray(data.qacc, dtype=np.float64)
    rhs = (
        np.asarray(data.qfrc_actuator, dtype=np.float64)
        + np.asarray(data.qfrc_passive, dtype=np.float64)
        - np.asarray(data.qfrc_bias, dtype=np.float64)
        + np.asarray(data.qfrc_constraint, dtype=np.float64)
    )
    report = _qfrc_norm_report(shard, lhs, rhs)
    report["passed"] = report["max_abs"] <= _DYNAMICS_RESIDUAL_MAX
    report["equation"] = "M*qacc = qfrc_actuator + qfrc_passive - qfrc_bias + qfrc_constraint"
    return report


def force_closure_audit_for_state(
    shard: Any,
    qpos: Any,
    ctrl: Any,
) -> dict[str, Any]:
    _set_state(shard, qpos, ctrl)
    actuator = _actuator_closure(shard, ctrl)
    contact = _contact_closure(shard)
    dynamics = _dynamics_closure(shard)
    return {
        "actuator_generalized_force_vs_data_qfrc_actuator": actuator,
        "contact_force_jacobian_and_efc_closure": contact,
        "full_dynamics_closure": dynamics,
        "full_closure_passed": bool(
            actuator["passed"]
            and contact["foot_hand_force_jacobian_vs_filtered_foot_floor_efc"]["passed"]
            and contact["full_mujoco_efc_jacobian_vs_full_qfrc_constraint"]["passed"]
            and dynamics["passed"]
        ),
        "internal_mujoco_contact_closure_passed": bool(
            contact["full_mujoco_efc_jacobian_vs_full_qfrc_constraint"]["passed"]
        ),
    }


def strict_positive_control_audits(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if not record.get("strict_contract_passed", False):
            continue
        family = record["family"]
        seed = int(record["seed"])
        range_fraction = float(record["range_fraction"])
        shard, key = _build_shard_for(family, seed, range_fraction)
        best = record["strict_refinement"]["best"]
        audit = force_closure_audit_for_state(shard, best["qpos"], best["ctrl"])
        rows.append(
            {
                "label": f"{family}:rf{range_fraction:g}:seed{seed}",
                "family": family,
                "seed": seed,
                "range_fraction": range_fraction,
                "morphology_instance_key": key,
                "model_xml_sha256": _sha256_bytes(shard.xml.encode("utf-8")),
                "strict_contract_passed": True,
                "strict_snapshot": _diagnostic_snapshot(shard, best["qpos"], best["ctrl"]),
                **audit,
            }
        )
    return rows


def _corner_height_audit(
    shard: Any,
    mode: FixedContactMode,
) -> dict[str, Any]:
    selected_keys = {point.key for point in mode.points}
    rows = []
    for corner in _all_footpad_corner_points(shard, shard.data[0]):
        selected = corner["key"] in selected_keys
        rows.append(
            {
                "key": corner["key"],
                "foot": corner["foot"],
                "sx": corner["sx"],
                "sy": corner["sy"],
                "height": corner["height"],
                "selected": selected,
            }
        )
    selected_heights = [row["height"] for row in rows if row["selected"]]
    unselected_heights = [row["height"] for row in rows if not row["selected"]]
    min_selected = min(selected_heights, default=float("inf"))
    min_unselected = min(unselected_heights, default=float("inf"))
    return {
        "corners": sorted(rows, key=lambda row: (row["foot"], row["sx"], row["sy"])),
        "selected_min_height": min_selected,
        "unselected_min_height": None if min_unselected == float("inf") else min_unselected,
        "all_min_height": min((row["height"] for row in rows), default=float("inf")),
        "unselected_deeper_than_selected": bool(min_unselected < min_selected - _CONTACT_HEIGHT_TOL),
        "unselected_deeper_margin": max(0.0, min_selected - min_unselected)
        if min_unselected != float("inf")
        else 0.0,
    }


def _cop_from_points(points: list[dict[str, Any]], forces: Any) -> list[float] | None:
    import numpy as np

    total = float(np.sum(forces))
    if total <= 1e-12:
        return None
    xy = np.zeros(2, dtype=np.float64)
    for point, force in zip(points, forces):
        xy += float(force) * np.asarray(point["point"][:2], dtype=np.float64)
    return [float(value) for value in xy / total]


def _dof_difference_rows(shard: Any, values: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    names = _dof_names(shard)
    arrays = {key: shard.np.asarray(value, dtype=shard.np.float64) for key, value in values.items()}
    constraint = arrays["qfrc_constraint"]
    for dof, meta in enumerate(names):
        row = dict(meta)
        for key, value in arrays.items():
            row[key] = float(value[dof])
        row["rigid_minus_constraint"] = float(arrays["qfrc_rigid"][dof] - constraint[dof])
        row["actual_hand_minus_constraint"] = float(arrays["qfrc_actual_hand"][dof] - constraint[dof])
        row["actual_efc_minus_constraint"] = float(arrays["qfrc_actual_efc"][dof] - constraint[dof])
        rows.append(row)
    return rows


def failed_candidate_realization_audits(
    fixed_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for endpoint in fixed_payload["endpoints"]:
        family = endpoint["family"]
        seed = int(endpoint["seed"])
        range_fraction = float(endpoint["range_fraction"])
        shard, key = _build_shard_for(family, seed, range_fraction)
        best = endpoint["best"]
        mode = _mode_from_manifest(best["mode"])
        _set_state(shard, best["qpos"], best["ctrl"])
        selected_points = _selected_contact_points(shard, shard.data[0], mode)
        rigid_forces = shard.np.asarray(
            best["rigid_static_constraints"]["normal_forces"],
            dtype=shard.np.float64,
        )
        qfrc_rigid = _qfrc_from_contact_forces(shard, shard.data[0], selected_points, rigid_forces)
        contact_audit = _contact_closure(shard)
        qfrc_actual_hand = shard.np.asarray(
            contact_audit["foot_hand_force_jacobian_qfrc"],
            dtype=shard.np.float64,
        )
        qfrc_actual_efc = shard.np.asarray(
            contact_audit["filtered_foot_floor_efc_jacobian_qfrc"],
            dtype=shard.np.float64,
        )
        qfrc_full_efc = shard.np.asarray(
            contact_audit["full_mujoco_efc_jacobian_qfrc"],
            dtype=shard.np.float64,
        )
        qfrc_constraint = shard.np.asarray(shard.data[0].qfrc_constraint, dtype=shard.np.float64)
        contact_report = _contact_report(shard, shard.data[0])
        analytic_loads = _normal_force_by_foot(mode, rigid_forces)
        actual_loads = contact_report["normal_force_by_foot"]
        actual_total = float(contact_report["foot_normal_force_sum"])
        analytic_total = float(shard.np.sum(rigid_forces))
        rows.append(
            {
                "endpoint_label": endpoint["endpoint_label"],
                "family": family,
                "seed": seed,
                "range_fraction": range_fraction,
                "morphology_instance_key": key,
                "model_xml_sha256": _sha256_bytes(shard.xml.encode("utf-8")),
                "source_best_classification": best["classification"],
                "mode": mode.manifest(),
                "corner_height_audit": _corner_height_audit(shard, mode),
                "actual_contact_records": contact_audit["actual_contact_records"],
                "contact_taxonomy": contact_audit["contact_taxonomy"],
                "qfrc_maps": {
                    "qfrc_rigid": [float(value) for value in qfrc_rigid],
                    "qfrc_actual_hand": [float(value) for value in qfrc_actual_hand],
                    "qfrc_filtered_foot_floor_efc": [float(value) for value in qfrc_actual_efc],
                    "qfrc_full_efc": [float(value) for value in qfrc_full_efc],
                    "qfrc_constraint": [float(value) for value in qfrc_constraint],
                    "norm_differences": {
                        "rigid_vs_qfrc_constraint": _qfrc_norm_report(shard, qfrc_rigid, qfrc_constraint),
                        "actual_hand_vs_filtered_foot_floor_efc": _qfrc_norm_report(
                            shard,
                            qfrc_actual_hand,
                            qfrc_actual_efc,
                        ),
                        "filtered_foot_floor_efc_vs_qfrc_constraint": _qfrc_norm_report(
                            shard,
                            qfrc_actual_efc,
                            qfrc_constraint,
                        ),
                        "full_efc_vs_qfrc_constraint": _qfrc_norm_report(
                            shard,
                            qfrc_full_efc,
                            qfrc_constraint,
                        ),
                        "rigid_vs_actual_efc": _qfrc_norm_report(shard, qfrc_rigid, qfrc_actual_efc),
                    },
                    "per_dof": _dof_difference_rows(
                        shard,
                        {
                            "qfrc_rigid": qfrc_rigid,
                            "qfrc_actual_hand": qfrc_actual_hand,
                            "qfrc_actual_efc": qfrc_actual_efc,
                            "qfrc_full_efc": qfrc_full_efc,
                            "qfrc_constraint": qfrc_constraint,
                        },
                    ),
                },
                "loads_and_cop": {
                    "weight": _weight(shard),
                    "analytic_normal_force_by_foot": analytic_loads,
                    "actual_normal_force_by_foot": actual_loads,
                    "analytic_total_load": analytic_total,
                    "actual_total_load": actual_total,
                    "actual_over_analytic_total_load": actual_total / max(analytic_total, 1e-12),
                    "actual_over_weight": actual_total / max(_weight(shard), 1e-12),
                    "analytic_cop_xy": _cop_from_points(selected_points, rigid_forces),
                    "actual_cop_xy": contact_report["center_of_pressure_xy"],
                },
                "actual_snapshot": _diagnostic_snapshot(shard, best["qpos"], best["ctrl"]),
            }
        )
    return rows


def _align_global_lowest_corner_to_penetration(
    shard: Any,
    qpos: Any,
    *,
    penetration_m: float,
) -> Any:
    np = shard.np
    qpos_array = np.asarray(qpos, dtype=np.float64).copy()
    _reset_to_qpos(shard, shard.data[0], qpos_array)
    corners = _all_footpad_corner_points(shard, shard.data[0])
    lowest = min(float(corner["height"]) for corner in corners)
    qpos_array[2] += -float(penetration_m) - lowest
    return qpos_array


def _actual_contact_corner_keys(records: list[dict[str, Any]]) -> set[str]:
    return {
        str(record["nearest_corner"]["key"])
        for record in records
        if float(record["normal_force"]) > 1e-9
    }


def _sweep_sample(
    shard: Any,
    mode: FixedContactMode,
    qpos: Any,
    ctrl: Any,
    penetration_m: float,
) -> dict[str, Any]:
    qpos_at_penetration = _align_global_lowest_corner_to_penetration(
        shard,
        qpos,
        penetration_m=penetration_m,
    )
    _set_state(shard, qpos_at_penetration, ctrl)
    contact_records, _ = _actual_contact_records_and_hand_qfrc(shard)
    contact_report = _contact_report(shard, shard.data[0])
    corner_audit = _corner_height_audit(shard, mode)
    snapshot = _diagnostic_snapshot(shard, qpos_at_penetration, ctrl)
    selected_keys = {point.key for point in mode.points}
    actual_keys = _actual_contact_corner_keys(contact_records)
    unexpected_keys = sorted(actual_keys - selected_keys)
    feet_with_load = [
        foot
        for foot, load in contact_report["normal_force_by_foot"].items()
        if float(load) >= _foot_load_threshold(shard)
        and int(contact_report["contacts_by_foot"].get(foot, 0)) > 0
    ]
    return {
        "penetration_m": penetration_m,
        "penetration_mm": penetration_m * 1000.0,
        "actual_total_load": float(contact_report["foot_normal_force_sum"]),
        "actual_over_weight": float(contact_report["foot_normal_force_sum"]) / max(_weight(shard), 1e-12),
        "actual_normal_force_by_foot": contact_report["normal_force_by_foot"],
        "both_feet_load_satisfied": len(feet_with_load) == len(shard._foot_geoms),
        "actual_contact_corner_keys": sorted(actual_keys),
        "unexpected_unselected_actual_contact_corner_keys": unexpected_keys,
        "active_contact_mode_consistent_with_selected": not unexpected_keys
        and set(feet_with_load) == set(mode.foot_names),
        "corner_height_audit": corner_audit,
        "root_qacc_norm": snapshot["root_qacc_norm"],
        "joint_qacc_max": snapshot["joint_qacc_max"],
        "support_mode": snapshot["support_mode"],
        "strict_actual_equilibrium": strict_actual_equilibrium(snapshot),
    }


def _bisect_total_load_crossing(
    shard: Any,
    mode: FixedContactMode,
    qpos: Any,
    ctrl: Any,
    low_m: float,
    high_m: float,
) -> dict[str, Any]:
    low = low_m
    high = high_m
    weight = _weight(shard)
    low_sample = _sweep_sample(shard, mode, qpos, ctrl, low)
    high_sample = _sweep_sample(shard, mode, qpos, ctrl, high)
    for _ in range(24):
        mid = 0.5 * (low + high)
        mid_sample = _sweep_sample(shard, mode, qpos, ctrl, mid)
        if (low_sample["actual_total_load"] - weight) * (mid_sample["actual_total_load"] - weight) <= 0.0:
            high = mid
            high_sample = mid_sample
        else:
            low = mid
            low_sample = mid_sample
    return high_sample


def penetration_sweep_audits(fixed_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for endpoint in fixed_payload["endpoints"]:
        family = endpoint["family"]
        seed = int(endpoint["seed"])
        range_fraction = float(endpoint["range_fraction"])
        shard, _ = _build_shard_for(family, seed, range_fraction)
        best = endpoint["best"]
        mode = _mode_from_manifest(best["mode"])
        qpos = shard.np.asarray(best["qpos"], dtype=shard.np.float64)
        ctrl = shard.np.asarray(best["ctrl"], dtype=shard.np.float64)
        samples = [
            _sweep_sample(shard, mode, qpos, ctrl, value / 1000.0)
            for value in _PENETRATION_SWEEP_MM
        ]
        crossing = None
        weight = _weight(shard)
        for left, right in pairwise(samples):
            left_value = float(left["actual_total_load"]) - weight
            right_value = float(right["actual_total_load"]) - weight
            if left_value == 0.0:
                crossing = left
                break
            if left_value * right_value <= 0.0:
                crossing = _bisect_total_load_crossing(
                    shard,
                    mode,
                    qpos,
                    ctrl,
                    float(left["penetration_m"]),
                    float(right["penetration_m"]),
                )
                break
        strict_samples = [sample for sample in samples if sample["strict_actual_equilibrium"]]
        rows.append(
            {
                "endpoint_label": endpoint["endpoint_label"],
                "mode": mode.manifest(),
                "qpos_joints_and_ctrl_fixed": True,
                "penetration_range_mm": [min(_PENETRATION_SWEEP_MM), max(_PENETRATION_SWEEP_MM)],
                "samples": samples,
                "total_load_crosses_weight": crossing is not None,
                "total_load_weight_crossing_bisection": crossing,
                "both_feet_load_satisfied_any": any(sample["both_feet_load_satisfied"] for sample in samples),
                "strict_actual_equilibrium_any": bool(strict_samples),
                "minimum_root_qacc_norm": min(float(sample["root_qacc_norm"]) for sample in samples),
                "minimum_joint_qacc_max": min(float(sample["joint_qacc_max"]) for sample in samples),
                "active_contact_mode_consistent_all_samples": all(
                    sample["active_contact_mode_consistent_with_selected"] for sample in samples
                ),
            }
        )
    return rows


def _positive_control_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "strict_positive_controls": len(rows),
        "full_closure_passed": sum(bool(row["full_closure_passed"]) for row in rows),
        "actuator_closure_passed": sum(
            bool(row["actuator_generalized_force_vs_data_qfrc_actuator"]["passed"]) for row in rows
        ),
        "hand_contact_jacobian_closure_passed": sum(
            bool(
                row["contact_force_jacobian_and_efc_closure"][
                    "foot_hand_force_jacobian_vs_filtered_foot_floor_efc"
                ]["passed"]
            )
            for row in rows
        ),
        "mujoco_efc_contact_closure_passed": sum(
            bool(
                row["contact_force_jacobian_and_efc_closure"][
                    "full_mujoco_efc_jacobian_vs_full_qfrc_constraint"
                ]["passed"]
            )
            for row in rows
        ),
        "full_dynamics_closure_passed": sum(
            bool(row["full_dynamics_closure"]["passed"]) for row in rows
        ),
        "max_hand_contact_jacobian_qfrc_diff": max(
            (
                float(
                    row["contact_force_jacobian_and_efc_closure"][
                        "foot_hand_force_jacobian_vs_filtered_foot_floor_efc"
                    ]["max_abs"]
                )
                for row in rows
            ),
            default=0.0,
        ),
        "max_mujoco_efc_contact_qfrc_diff": max(
            (
                float(
                    row["contact_force_jacobian_and_efc_closure"][
                        "full_mujoco_efc_jacobian_vs_full_qfrc_constraint"
                    ]["max_abs"]
                )
                for row in rows
            ),
            default=0.0,
        ),
        "max_dynamics_residual": max(
            (float(row["full_dynamics_closure"]["max_abs"]) for row in rows),
            default=0.0,
        ),
    }


def _decide(summary: dict[str, Any]) -> dict[str, str]:
    positive = summary["positive_control_closure"]
    if int(positive["full_closure_passed"]) < int(positive["strict_positive_controls"]):
        return {
            "status": "r4a31c_force_jacobian_mapping_bug_pause_rigid_candidates",
            "decision": (
                f"Only {positive['full_closure_passed']}/{positive['strict_positive_controls']} "
                "strict positive controls close under same-set contact force/Jacobian and "
                "MuJoCo EFC checks, so the R4a.3.1b rigid candidates are paused "
                "as explanatory evidence."
            ),
            "next_allowed_work": (
                "Fix/calibrate the explicit force-frame/Jacobian mapping against filtered MuJoCo EFC; "
                "do not modify env, controller, generator, kp/kv, or Task061/062."
            ),
        }
    if summary["failed_active_set"]["inconsistent_endpoint_count"] > 0:
        return {
            "status": "r4a31c_failed_candidate_active_set_inconsistent",
            "decision": "Positive controls close, but at least one failed candidate realizes a different sole-corner active set.",
            "next_allowed_work": "Fix fixed-contact active-set constraints before interpreting rigid wrench candidates.",
        }
    if summary["penetration_sweep"]["strict_recovered_count"] > 0:
        return {
            "status": "r4a31c_penetration_realization_can_restore_strict_equilibrium",
            "decision": "At least one failed candidate recovers strict equilibrium with qpos joints and ctrl fixed under penetration realization.",
            "next_allowed_work": "Add a soft-contact realization stage to the final equilibrium solver; do not integrate feedback.",
        }
    return {
        "status": "r4a31c_actual_contact_refinement_required",
        "decision": "Positive controls and active sets are consistent, but penetration sweep does not reveal a strict zero.",
        "next_allowed_work": "Run joint actual-contact refinement; do not modify generator or feedback.",
    }


def run_soft_contact_realization_audit(
    *,
    r4a3_json: Path,
    r4a31b_json: Path,
    family: str,
) -> dict[str, Any]:
    source_records = _load_r4a3_records(r4a3_json, family=family)
    fixed_payload = json.loads(r4a31b_json.read_text(encoding="utf-8"))
    positive_controls = strict_positive_control_audits(source_records)
    failed_candidates = failed_candidate_realization_audits(fixed_payload)
    sweeps = penetration_sweep_audits(fixed_payload)
    positive_summary = _positive_control_summary(positive_controls)
    active_set_inconsistent = [
        row
        for row in failed_candidates
        if row["corner_height_audit"]["unselected_deeper_than_selected"]
    ]
    summary = {
        "positive_control_closure": positive_summary,
        "failed_active_set": {
            "failed_endpoint_count": len(failed_candidates),
            "inconsistent_endpoint_count": len(active_set_inconsistent),
            "inconsistent_endpoint_labels": [row["endpoint_label"] for row in active_set_inconsistent],
        },
        "failed_load_realization": {
            "endpoint_labels": [row["endpoint_label"] for row in failed_candidates],
            "actual_over_analytic_total_load": {
                row["endpoint_label"]: row["loads_and_cop"]["actual_over_analytic_total_load"]
                for row in failed_candidates
            },
            "actual_over_weight": {
                row["endpoint_label"]: row["loads_and_cop"]["actual_over_weight"]
                for row in failed_candidates
            },
        },
        "penetration_sweep": {
            "endpoint_count": len(sweeps),
            "load_crossing_count": sum(bool(row["total_load_crosses_weight"]) for row in sweeps),
            "both_feet_load_satisfied_count": sum(bool(row["both_feet_load_satisfied_any"]) for row in sweeps),
            "strict_recovered_count": sum(bool(row["strict_actual_equilibrium_any"]) for row in sweeps),
            "active_contact_mode_consistent_count": sum(
                bool(row["active_contact_mode_consistent_all_samples"]) for row in sweeps
            ),
        },
        "r4a31b_rigid_candidate_interpretation": "paused"
        if int(positive_summary["full_closure_passed"]) < int(positive_summary["strict_positive_controls"])
        else "available",
        "infeasibility_certificates": {
            "kinematic_double_support_infeasible": 0,
            "wrench_or_actuation_infeasible": 0,
        },
    }
    payload = {
        "schema": "task067_r4a31c_soft_contact_force_closure_realization_audit_v2_same_set_efc",
        "source_artifact": str(r4a3_json.resolve()),
        "fixed_contact_artifact": str(r4a31b_json.resolve()),
        "provenance": {
            "source_artifact_sha256": _sha256_path(r4a3_json),
            "fixed_contact_artifact_sha256": _sha256_path(r4a31b_json),
            "diagnostic_source_sha256": _sha256_path(Path(__file__)),
            "dependency_source_sha256": {
                "whole_body_fixed_contact_mode_wrench_solve.py": _sha256_path(
                    Path(__file__).with_name("whole_body_fixed_contact_mode_wrench_solve.py")
                ),
                "whole_body_contact_preserving_continuation.py": _sha256_path(
                    Path(__file__).with_name("whole_body_contact_preserving_continuation.py")
                ),
                "whole_body_dynamic_balance_diagnosis.py": _sha256_path(
                    Path(__file__).with_name("whole_body_dynamic_balance_diagnosis.py")
                ),
                "whole_body_equilibrium_audit.py": _sha256_path(
                    Path(__file__).with_name("whole_body_equilibrium_audit.py")
                ),
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "mujoco": _package_version("mujoco"),
            "parameters": {
                "family": family,
                "penetration_sweep_mm": list(_PENETRATION_SWEEP_MM),
                "dynamics_residual_max": _DYNAMICS_RESIDUAL_MAX,
                "actuator_qfrc_max_diff": _ACTUATOR_QFRC_MAX_DIFF,
                "efc_contact_qfrc_max_diff": _EFC_CONTACT_QFRC_MAX_DIFF,
                "hand_contact_qfrc_max_diff": _HAND_CONTACT_QFRC_MAX_DIFF,
            },
            "diagnostic_scope": {
                "calibrates_on_5_strict_equilibria": True,
                "audits_failed_candidate_active_set": True,
                "sweeps_root_penetration_with_qpos_joints_and_ctrl_fixed": True,
                "foot_hand_reconstruction_compared_only_to_filtered_foot_floor_efc": True,
                "full_efc_compared_to_full_qfrc_constraint": True,
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
            "strict_positive_control_count_is_5": len(positive_controls) == 5,
            "internal_mujoco_efc_contact_closure_passes_all_positive_controls": (
                int(positive_summary["mujoco_efc_contact_closure_passed"]) == len(positive_controls)
            ),
            "full_dynamics_closure_passes_all_positive_controls": (
                int(positive_summary["full_dynamics_closure_passed"]) == len(positive_controls)
            ),
            "search_failure_not_promoted_to_physical_infeasible": (
                int(summary["infeasibility_certificates"]["kinematic_double_support_infeasible"]) == 0
                and int(summary["infeasibility_certificates"]["wrench_or_actuation_infeasible"]) == 0
            ),
        },
        "positive_controls": positive_controls,
        "failed_candidate_realization_audits": failed_candidates,
        "penetration_sweeps": sweeps,
    }
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r4a3-json", type=Path, default=_DEFAULT_R4A3_INPUT)
    parser.add_argument("--r4a31b-json", type=Path, default=_DEFAULT_R4A31B_INPUT)
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--family", default=_DEFAULT_FAMILY)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    payload = run_soft_contact_realization_audit(
        r4a3_json=args.r4a3_json,
        r4a31b_json=args.r4a31b_json,
        family=args.family,
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
