"""Task067 R4a.3 strict actual-equilibrium coverage diagnosis.

This diagnostic supersedes the old R4a.2 branch decision contract.  The
contact-consistent QP result is treated only as a candidate source; acceptance
requires strict actual MuJoCo forward-dynamics qacc, actual per-foot load, no
non-foot contact, no actuator saturation, and a nominal hold rollout.

The tool does not modify the public environment/controller, actuator gains,
reward, observation/action schema, or motor strength/latency/failure path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
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
from h200_locomotion_lab.tools.whole_body_bounded_feedback_diagnosis import (
    _build_shard_with_replay_binding,
    _unique_replay_bindings,
)
from h200_locomotion_lab.tools.whole_body_dynamic_balance_diagnosis import (
    _contact_report,
    _fall_reason,
    _reset_to_qpos,
)
from h200_locomotion_lab.tools.whole_body_equilibrium_audit import (
    _sha256_path,
    _state_snapshot,
    refine_actual_equilibrium,
    strict_actual_equilibrium,
)

_DEFAULT_INPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a2_joint_aware_equilibrium_diagnosis_4x2.json"
)
_DEFAULT_OUTPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a3_strict_equilibrium_coverage_4x2.json"
)
_FOOT_LOAD_FRACTION = 0.05


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_coverage_records(
    path: Path,
    *,
    families: tuple[str, ...],
    seeds: tuple[int, ...] | None = None,
    range_fractions: tuple[float, ...] | None = None,
) -> list[dict[str, Any]]:
    """Load every requested source record, independent of old feasible labels."""

    data = json.loads(path.read_text(encoding="utf-8"))
    records = [
        record
        for record in data["records"]
        if record["family"] in families
        and (seeds is None or int(record["seed"]) in seeds)
        and (
            range_fractions is None
            or any(abs(float(record["range_fraction"]) - value) <= 1e-12 for value in range_fractions)
        )
    ]
    return sorted(records, key=lambda item: (item["family"], float(item["range_fraction"]), int(item["seed"])))


def _foot_load_threshold(shard: Any) -> float:
    total_mass = float(shard.np.sum(shard.model.body_mass))
    return _FOOT_LOAD_FRACTION * total_mass * abs(float(shard.model.opt.gravity[2]))


def run_strict_hold_rollout(
    shard: Any,
    equilibrium: dict[str, Any],
    *,
    horizon_steps: int,
) -> dict[str, Any]:
    """Run fixed qpos_eq/ctrl_eq hold and enforce actual per-foot support."""

    np = shard.np
    data = shard.data[0]
    qpos = np.asarray(equilibrium["best"]["qpos"], dtype=np.float64)
    ctrl = np.asarray(equilibrium["best"]["ctrl"], dtype=np.float64)
    _reset_to_qpos(shard, data, qpos)
    data.ctrl[:] = ctrl
    shard.mujoco.mj_forward(shard.model, data)

    minimum_load = _foot_load_threshold(shard)
    normal_force_min = {name: float("inf") for name in sorted(shard._foot_geoms)}
    contacts_min = {name: 10**9 for name in sorted(shard._foot_geoms)}
    non_foot_contact_steps = 0
    self_contact_steps = 0
    unloaded_foot_steps = 0
    actuator_saturation_events = 0
    root_qacc_norm_max = 0.0
    joint_qacc_max = 0.0
    actuator_force_max = 0.0
    first_fall_step = None
    first_fall_reason = None
    first_self_contact_step = None
    first_self_contact_geom_pair = None
    samples = 0

    def sample(step_index: int) -> None:
        nonlocal non_foot_contact_steps
        nonlocal self_contact_steps
        nonlocal unloaded_foot_steps
        nonlocal actuator_saturation_events
        nonlocal root_qacc_norm_max
        nonlocal joint_qacc_max
        nonlocal actuator_force_max
        nonlocal first_self_contact_step
        nonlocal first_self_contact_geom_pair
        nonlocal samples

        contact = _contact_report(shard, data)
        qacc = np.asarray(data.qacc, dtype=np.float64)
        root_qacc_norm_max = max(root_qacc_norm_max, float(np.linalg.norm(qacc[:6])))
        joint_qacc_max = max(
            joint_qacc_max,
            max((abs(float(qacc[dof])) for dof in shard._joint_dof), default=0.0),
        )
        non_foot_contact_steps += int(contact["non_foot_contacts"] > 0)
        if int(contact.get("self_contacts", 0)) > 0:
            self_contact_steps += 1
            if first_self_contact_step is None:
                first_self_contact_step = step_index
                first_self_contact_geom_pair = contact["taxonomy"]["self_contacts"][0]
        unloaded = False
        for name in sorted(shard._foot_geoms):
            load = float(contact["normal_force_by_foot"].get(name, 0.0))
            normal_force_min[name] = min(normal_force_min[name], load)
            contacts = int(contact["contacts_by_foot"].get(name, 0))
            contacts_min[name] = min(contacts_min[name], contacts)
            unloaded = unloaded or contacts <= 0 or load < minimum_load
        unloaded_foot_steps += int(unloaded)
        for actuator_id in shard._actuator_ids:
            force = abs(float(data.actuator_force[int(actuator_id)]))
            actuator_force_max = max(actuator_force_max, force)
            limit = max(abs(float(value)) for value in shard.model.actuator_forcerange[int(actuator_id)])
            actuator_saturation_events += int(force >= 0.995 * limit)
        samples = max(samples, step_index + 1)

    for step_index in range(horizon_steps):
        for _ in range(shard.config.substeps):
            data.ctrl[:] = ctrl
            data.qfrc_applied[:] = 0.0
            shard.mujoco.mj_forward(shard.model, data)
            sample(step_index)
            shard.mujoco.mj_step(shard.model, data)
        reason = _fall_reason(shard, data)
        if reason is not None:
            first_fall_step = step_index + 1
            first_fall_reason = reason
            break

    sustained_double_support = all(
        contacts_min[name] > 0 and normal_force_min[name] >= minimum_load
        for name in sorted(shard._foot_geoms)
    )
    survived = first_fall_step is None
    passed = bool(
        survived
        and sustained_double_support
        and non_foot_contact_steps == 0
        and self_contact_steps == 0
        and actuator_saturation_events == 0
    )
    return {
        "hold_executed": True,
        "horizon_steps": horizon_steps,
        "samples": samples,
        "survived": survived,
        "first_fall_step": first_fall_step,
        "first_fall_reason": first_fall_reason,
        "passed": passed,
        "minimum_foot_load": minimum_load,
        "normal_force_min_by_foot": normal_force_min,
        "contacts_min_by_foot": contacts_min,
        "sustained_double_support": sustained_double_support,
        "non_foot_contact_steps": non_foot_contact_steps,
        "self_contact_steps": self_contact_steps,
        "first_self_contact_step": first_self_contact_step,
        "first_self_contact_geom_pair": first_self_contact_geom_pair,
        "unloaded_foot_steps": unloaded_foot_steps,
        "actuator_saturation_events": actuator_saturation_events,
        "actuator_force_max": actuator_force_max,
        "root_qacc_norm_max": root_qacc_norm_max,
        "joint_qacc_max": joint_qacc_max,
    }


def strict_equilibrium_contract_passed(
    initial_snapshot: dict[str, Any],
    hold_rollout: dict[str, Any],
) -> bool:
    return bool(strict_actual_equilibrium(initial_snapshot) and hold_rollout["passed"])


def decide_strict_coverage(summary: dict[str, Any]) -> dict[str, str]:
    total = int(summary["source_records"])
    accepted = int(summary["strict_contract_passed"])
    refined = int(summary["strict_refined_actual_equilibria"])
    if total > 0 and accepted == total:
        return {
            "status": "strict_equilibrium_coverage_passed",
            "decision": "All requested biped records have strict actual equilibria that pass nominal hold.",
            "next_allowed_work": "Run strict-equilibrium-only feedback diagnostics; do not integrate controller yet.",
        }
    if total > 0 and refined == total:
        return {
            "status": "strict_equilibrium_nominal_hold_failed",
            "decision": "Strict actual equilibria were found, but nominal hold did not pass for every record.",
            "next_allowed_work": "Diagnose hold/contact drift before any feedback integration.",
        }
    return {
        "status": "strict_equilibrium_coverage_incomplete",
        "decision": (
            f"Only {accepted}/{total} requested records passed strict actual equilibrium plus nominal hold; "
            "full generator biped equilibrium coverage remains unproven."
        ),
        "next_allowed_work": (
            "Continue strict solver/candidate diagnostics, and run feedback diagnostics only on accepted "
            "strict equilibria if needed."
        ),
    }


def run_strict_coverage(
    *,
    input_json: Path,
    families: tuple[str, ...],
    seeds: tuple[int, ...] | None,
    range_fractions: tuple[float, ...] | None,
    horizon_steps: int,
    max_nfev: int,
) -> dict[str, Any]:
    source_records = load_coverage_records(
        input_json,
        families=families,
        seeds=seeds,
        range_fractions=range_fractions,
    )
    rows: list[dict[str, Any]] = []
    replay_bindings = []
    for record in source_records:
        shard, binding = _build_shard_with_replay_binding(record)
        replay_bindings.append(binding)
        source_equilibrium = record["contact_equilibrium"]
        source_best = source_equilibrium["best"]
        source_snapshot = _state_snapshot(shard, source_best["qpos"], source_best["ctrl"])
        refinement = refine_actual_equilibrium(shard, source_equilibrium, max_nfev=max_nfev)
        refined_snapshot = refinement["best"]["snapshot"]
        hold_rollout = (
            run_strict_hold_rollout(
                shard,
                {"status": "feasible", "best": refinement["best"]},
                horizon_steps=horizon_steps,
            )
            if refinement["status"] == "feasible"
            else {
                "horizon_steps": horizon_steps,
                "hold_executed": False,
                "samples": 0,
                "survived": False,
                "first_fall_step": None,
                "first_fall_reason": "no_strict_actual_equilibrium",
                "passed": False,
                "minimum_foot_load": _foot_load_threshold(shard),
                "normal_force_min_by_foot": {},
                "contacts_min_by_foot": {},
                "sustained_double_support": False,
                "non_foot_contact_steps": 0,
                "self_contact_steps": 0,
                "first_self_contact_step": None,
                "first_self_contact_geom_pair": None,
                "unloaded_foot_steps": 0,
                "actuator_saturation_events": 0,
                "actuator_force_max": 0.0,
                "root_qacc_norm_max": 0.0,
                "joint_qacc_max": 0.0,
            }
        )
        rows.append(
            {
                "family": record["family"],
                "range_fraction": float(record["range_fraction"]),
                "seed": int(record["seed"]),
                "morphology_instance_key": record["morphology_instance_key"],
                "replay_contract_binding": binding.manifest(),
                "model_xml_sha256": _sha256_bytes(shard.xml.encode("utf-8")),
                "source_contact_qp_candidate": {
                    "r4a2_status": source_equilibrium["status"],
                    "r4a2_candidate_count": source_equilibrium["candidate_count"],
                    "r4a2_feasible_candidate_count": source_equilibrium["feasible_candidate_count"],
                    "actual_snapshot": source_snapshot,
                    "strict_actual_equilibrium": strict_actual_equilibrium(source_snapshot),
                },
                "strict_refinement": refinement,
                "strict_initial_actual_equilibrium": refinement["status"] == "feasible",
                "strict_nominal_hold": hold_rollout,
                "strict_contract_passed": strict_equilibrium_contract_passed(
                    refined_snapshot,
                    hold_rollout,
                ),
            }
        )

    source_feasible = [
        row for row in rows if row["source_contact_qp_candidate"]["r4a2_status"] == "feasible"
    ]
    source_false_positive = [
        row for row in source_feasible if not row["source_contact_qp_candidate"]["strict_actual_equilibrium"]
    ]
    source_infeasible_refined = [
        row
        for row in rows
        if row["source_contact_qp_candidate"]["r4a2_status"] != "feasible"
        and row["strict_contract_passed"]
    ]
    summary = {
        "source_records": len(rows),
        "source_r4a2_feasible": len(source_feasible),
        "source_strict_actual_equilibria": sum(
            row["source_contact_qp_candidate"]["strict_actual_equilibrium"] for row in rows
        ),
        "source_feasible_false_positive": len(source_false_positive),
        "source_infeasible_refined_to_strict_contract": len(source_infeasible_refined),
        "strict_refined_actual_equilibria": sum(row["strict_initial_actual_equilibrium"] for row in rows),
        "strict_initial_self_collision_free": sum(
            int(row["strict_refinement"]["best"]["snapshot"]["contact"].get("self_contacts", 0)) == 0
            for row in rows
        ),
        "strict_nominal_hold_passed": sum(row["strict_nominal_hold"]["passed"] for row in rows),
        "strict_nominal_hold_self_collision_free": sum(
            int(row["strict_nominal_hold"].get("self_contact_steps", 0)) == 0
            for row in rows
            if bool(row["strict_nominal_hold"].get("hold_executed", False))
        ),
        "strict_nominal_hold_self_collision_free_denominator": sum(
            bool(row["strict_nominal_hold"].get("hold_executed", False)) for row in rows
        ),
        "strict_contract_passed": sum(row["strict_contract_passed"] for row in rows),
        "accepted_labels": [
            f"{row['family']}:rf{row['range_fraction']:g}:seed{row['seed']}"
            for row in rows
            if row["strict_contract_passed"]
        ],
        "incomplete_labels": [
            f"{row['family']}:rf{row['range_fraction']:g}:seed{row['seed']}"
            for row in rows
            if not row["strict_contract_passed"]
        ],
    }
    return {
        "schema": "task067_r4a3_strict_equilibrium_coverage_v2_contact_taxonomy_collision_free",
        "source_artifact": str(input_json.resolve()),
        "source_replay_contracts": _unique_replay_bindings(replay_bindings),
        "runtime_contract": {
            "embodiment_contract_version": PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
            "embodiment_contract_hash": PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
        },
        "provenance": {
            "source_artifact_sha256": _sha256_path(input_json),
            "diagnostic_source_sha256": _sha256_path(Path(__file__)),
            "dependency_source_sha256": {
                "whole_body_equilibrium_audit.py": _sha256_path(
                    Path(__file__).with_name("whole_body_equilibrium_audit.py")
                ),
                "whole_body_dynamic_balance_diagnosis.py": _sha256_path(
                    Path(__file__).with_name("whole_body_dynamic_balance_diagnosis.py")
                ),
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "mujoco": _package_version("mujoco"),
            "parameters": {
                "families": list(families),
                "seeds": list(seeds) if seeds is not None else None,
                "range_fractions": list(range_fractions) if range_fractions is not None else None,
                "horizon_steps": horizon_steps,
                "max_nfev": max_nfev,
                "foot_load_fraction": _FOOT_LOAD_FRACTION,
                "nominal_stance_contract": {
                    "double_support": True,
                    "forbidden_nonfoot_floor_contacts": 0,
                    "self_contacts": 0,
                },
            },
        },
        "embodiment_contract_version": PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
        "embodiment_contract_hash": PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
        "stance_solution_contract_version": STANCE_SOLUTION_CONTRACT_VERSION,
        "stance_solution_contract_hash": STANCE_SOLUTION_CONTRACT_HASH,
        "summary": summary,
        "decision": decide_strict_coverage(summary),
        "records": rows,
    }


def _optional_int_tuple(values: list[int] | None) -> tuple[int, ...] | None:
    return tuple(values) if values is not None else None


def _optional_float_tuple(values: list[float] | None) -> tuple[float, ...] | None:
    return tuple(values) if values is not None else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--families", nargs="+", default=["biped"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--range-fractions", nargs="+", type=float)
    parser.add_argument("--horizon-steps", type=int, default=100)
    parser.add_argument("--max-nfev", type=int, default=1500)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    payload = run_strict_coverage(
        input_json=args.input_json,
        families=tuple(args.families),
        seeds=_optional_int_tuple(args.seeds),
        range_fractions=_optional_float_tuple(args.range_fractions),
        horizon_steps=args.horizon_steps,
        max_nfev=args.max_nfev,
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
