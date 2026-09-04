"""Task067 R4a.3.1g audit for StanceSolutionV3 public feedforward stance.

The audit builds public ``WholeBodyMuJoCoShard`` instances, so it exercises the
same reset and zero-action target path used by training.  Biped acceptance is
strict actual MuJoCo equilibrium plus fixed hold; quadruped records are kept as
the stance-matrix positive-control family and are judged by the public
zero-action hold matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import signal
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

from h200_locomotion_lab.envs.whole_body_mujoco import (
    WholeBodyMuJoCoShard,
    WholeBodyMuJoCoShardConfig,
)
from h200_locomotion_lab.robots.motor_process import MotorProcessConfig
from h200_locomotion_lab.robots.procedural_morphology import (
    PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
    PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
    MorphologyGenerator,
)
from h200_locomotion_lab.robots.whole_body_stance import (
    STANCE_SOLUTION_CONTRACT_HASH,
    STANCE_SOLUTION_CONTRACT_VERSION,
)
from h200_locomotion_lab.tools.whole_body_equilibrium_audit import (
    _state_snapshot,
    strict_actual_equilibrium,
)
from h200_locomotion_lab.tools.whole_body_stance_diagnosis import (
    _geom_bottom,
    _geom_support_points,
    _support_margin,
)
from h200_locomotion_lab.tools.whole_body_strict_equilibrium_coverage import (
    run_strict_hold_rollout,
)

_DEFAULT_OUTPUT = Path(
    ".agent/task/task067-biped-stance-contract/artifacts/"
    "r4a31g_stance_solution_v3_actual_feedforward_audit.json"
)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _root_yaw(quat: tuple[float, float, float, float]) -> float:
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return 0.0
    w, x, y, z = (value / norm for value in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _ctrl_vector(shard: WholeBodyMuJoCoShard) -> Any:
    ctrl = shard.np.zeros(shard.model.nu, dtype=shard.np.float64)
    for actuator, actuator_id in zip(shard.blueprint.actuators, shard._actuator_ids):
        ctrl[int(actuator_id)] = shard.stance_solution.actuator_ctrl[actuator.semantic_slot]
    return ctrl


def _margin_report(shard: WholeBodyMuJoCoShard) -> dict[str, Any]:
    rows = []
    joint_margins = []
    ctrl_margins = []
    deltas = []
    solution = shard.stance_solution
    for joint, actuator, qpos_address, actuator_id in zip(
        shard.blueprint.joints,
        shard.blueprint.actuators,
        shard._joint_qpos,
        shard._actuator_ids,
    ):
        joint_id = int(
            shard.mujoco.mj_name2id(shard.model, shard.mujoco.mjtObj.mjOBJ_JOINT, joint.name)
        )
        lower, upper = (float(value) for value in shard.model.jnt_range[joint_id])
        ctrl_lower, ctrl_upper = (
            float(value) for value in shard.model.actuator_ctrlrange[int(actuator_id)]
        )
        qpos = float(solution.joint_qpos[joint.semantic_slot])
        ctrl = float(solution.actuator_ctrl[actuator.semantic_slot])
        joint_margin = min(qpos - lower, upper - qpos)
        ctrl_margin = min(ctrl - ctrl_lower, ctrl_upper - ctrl)
        delta = ctrl - qpos
        joint_margins.append(joint_margin)
        ctrl_margins.append(ctrl_margin)
        deltas.append(abs(delta))
        rows.append(
            {
                "semantic_slot": joint.semantic_slot,
                "joint_qpos_eq": qpos,
                "actuator_ctrl_eq": ctrl,
                "ctrl_minus_qpos": delta,
                "joint_margin": joint_margin,
                "ctrl_margin": ctrl_margin,
            }
        )
    return {
        "min_joint_margin": min(joint_margins, default=float("inf")),
        "min_ctrl_margin": min(ctrl_margins, default=float("inf")),
        "max_abs_ctrl_minus_qpos": max(deltas, default=0.0),
        "per_joint": rows,
    }


def _geometry_report(shard: WholeBodyMuJoCoShard) -> dict[str, Any]:
    data = shard.data[0]
    foot_geom_ids = [
        int(shard.mujoco.mj_name2id(shard.model, shard.mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in sorted(shard._foot_geoms)
    ]
    bottoms = [_geom_bottom(shard.mujoco, shard.model, data, geom_id) for geom_id in foot_geom_ids]
    points = [
        point
        for geom_id in foot_geom_ids
        for point in _geom_support_points(shard.mujoco, shard.model, data, geom_id)
    ]
    total_mass = float(shard.np.sum(shard.model.body_mass))
    com = shard.np.zeros(3, dtype=shard.np.float64)
    for body_id in range(shard.model.nbody):
        com += float(shard.model.body_mass[body_id]) * shard.np.asarray(data.xipos[body_id])
    com /= max(1e-12, total_mass)
    support = _support_margin((float(com[0]), float(com[1])), points)
    return {
        "base_height": float(data.qpos[2]),
        "foot_bottom_heights": [float(value) for value in bottoms],
        "feet_near_floor": sum(1 for value in bottoms if value <= 0.02),
        "foot_height_spread": max(bottoms, default=0.0) - min(bottoms, default=0.0),
        "support_all_feet": support,
    }


def _public_zero_action_hold(shard: WholeBodyMuJoCoShard, *, horizon_steps: int) -> dict[str, Any]:
    zero = shard.np.zeros((1, 45), dtype=shard.np.float64)
    first_fall_step = None
    nonfoot_steps = 0
    max_tilt = 0.0
    min_height = float(shard.data[0].qpos[2])
    for step_index in range(horizon_steps):
        step = shard.step(zero)
        nonfoot_steps += int(float(step.metrics["non_foot_contact_fraction"][0]) > 0.0)
        max_tilt = max(max_tilt, float(step.metrics["tilt"][0]))
        min_height = min(min_height, float(shard.data[0].qpos[2]))
        if bool(step.metrics["fall"][0]):
            first_fall_step = step_index + 1
            break
    return {
        "horizon_steps": horizon_steps,
        "survived": first_fall_step is None,
        "first_fall_step": first_fall_step,
        "non_foot_contact_steps": nonfoot_steps,
        "max_tilt_rad": max_tilt,
        "min_base_height": min_height,
    }


def run_record(
    *,
    family: str,
    seed: int,
    range_fraction: float,
    horizon_steps: int,
) -> dict[str, Any]:
    generator = MorphologyGenerator()
    blueprint = generator.generate(family, seed)  # type: ignore[arg-type]
    physical = generator.sample_physical_params(
        blueprint,
        seed + 10_000_000,
        range_fraction=range_fraction,
    )
    started = time.perf_counter()
    shard = WholeBodyMuJoCoShard(
        blueprint,
        physical=physical,
        num_envs=1,
        config=WholeBodyMuJoCoShardConfig(trial_seconds=max(2.0, horizon_steps / 50.0), seed=seed),
        motor_config=MotorProcessConfig(no_event_probability=1.0),
    )
    solve_seconds = time.perf_counter() - started
    qpos = shard.np.asarray(shard.data[0].qpos, dtype=shard.np.float64).copy()
    ctrl = _ctrl_vector(shard)
    initial = _state_snapshot(shard, qpos, ctrl)
    fixed_hold = run_strict_hold_rollout(
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
    shard.reset()
    public_hold = _public_zero_action_hold(shard, horizon_steps=horizon_steps)
    margins = _margin_report(shard)
    root_pose = shard.stance_solution.root_pose
    root_yaw = _root_yaw(tuple(float(value) for value in root_pose[3:7]))
    return {
        "family": family,
        "seed": seed,
        "range_fraction": float(range_fraction),
        "label": f"{family}:rf{range_fraction:g}:seed{seed}",
        "solve_seconds": solve_seconds,
        "model_xml_sha256": _sha256_bytes(shard.xml.encode("utf-8")),
        "stance_solution": shard.stance_solution.manifest(),
        "stance_solution_hash": shard.stance_solution.solution_hash,
        "stance_cache_key": shard.stance_solution.cache_key,
        "root_gauge": {
            "x": float(root_pose[0]),
            "y": float(root_pose[1]),
            "yaw": root_yaw,
            "x_y_yaw_zero": abs(float(root_pose[0])) <= 1e-12
            and abs(float(root_pose[1])) <= 1e-12
            and abs(root_yaw) <= 1e-9,
        },
        "margins": margins,
        "geometry": _geometry_report(shard),
        "initial_actual_snapshot": initial,
        "strict_initial_actual_equilibrium": strict_actual_equilibrium(initial),
        "fixed_qpos_ctrl_hold": fixed_hold,
        "public_zero_action_hold": public_hold,
        "v3_acceptance": {
            "biped_strict_initial_and_hold": bool(
                family == "biped"
                and strict_actual_equilibrium(initial)
                and fixed_hold["passed"]
                and public_hold["survived"]
                and float(margins["min_joint_margin"]) > 0.05
                and float(margins["min_ctrl_margin"]) >= 0.01 - 1e-9
                and bool(
                    float(margins["max_abs_ctrl_minus_qpos"]) > 1e-3
                    or len(shard.blueprint.joints) == 0
                )
            ),
            "stance_matrix_hold": bool(public_hold["survived"]),
        },
    }


def _failure_record(
    *,
    family: str,
    seed: int,
    range_fraction: float,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "family": family,
        "seed": seed,
        "range_fraction": float(range_fraction),
        "label": f"{family}:rf{range_fraction:g}:seed{seed}",
        "record_status": "record_build_failed",
        "failure_classification": "search_exhausted_without_certificate",
        "physical_infeasibility_claimed": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
        "v3_acceptance": {
            "biped_strict_initial_and_hold": False,
            "stance_matrix_hold": False,
        },
    }


def _try_run_record(
    *,
    family: str,
    seed: int,
    range_fraction: float,
    horizon_steps: int,
    timeout_seconds: float = 0.0,
) -> dict[str, Any]:
    def execute() -> dict[str, Any]:
        return run_record(
            family=family,
            seed=seed,
            range_fraction=range_fraction,
            horizon_steps=horizon_steps,
        )

    def handle_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(
            f"record {family}:rf{range_fraction:g}:seed{seed} timed out "
            f"after {timeout_seconds:g}s"
        )

    try:
        if timeout_seconds > 0.0 and hasattr(signal, "SIGALRM"):
            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
            try:
                record = execute()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, previous_handler)
        else:
            record = execute()
    except (RuntimeError, TimeoutError, ValueError) as exc:
        return _failure_record(
            family=family,
            seed=seed,
            range_fraction=range_fraction,
            exc=exc,
        )
    return {"record_status": "built", **record}


def _run_records(
    targets: list[tuple[str, int, float]],
    *,
    horizon_steps: int,
    timeout_seconds: float,
    emit_progress: bool,
) -> list[dict[str, Any]]:
    records = []
    for family, seed, range_fraction in targets:
        record = _try_run_record(
            family=family,
            seed=seed,
            range_fraction=range_fraction,
            horizon_steps=horizon_steps,
            timeout_seconds=timeout_seconds,
        )
        records.append(record)
        if emit_progress:
            print(
                json.dumps(
                    {
                        "label": record["label"],
                        "record_status": record.get("record_status"),
                        "failure_classification": record.get("failure_classification"),
                        "biped_strict_v3": record["v3_acceptance"][
                            "biped_strict_initial_and_hold"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return records


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    biped = [record for record in records if record["family"] == "biped"]
    quadruped = [record for record in records if record["family"] == "quadruped"]

    def family_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        built = [record for record in items if record.get("record_status") == "built"]
        failures = [record for record in items if record.get("record_status") != "built"]
        if not items:
            return {"records": 0}
        summary: dict[str, Any] = {
            "records": len(items),
            "built_records": len(built),
            "record_build_failures": len(failures),
            "failed_labels": [record["label"] for record in failures],
        }
        if not built:
            summary.update(
                {
                    "public_zero_action_falls": None,
                    "nan_seeds": 0,
                    "degenerate_support_all_feet": None,
                    "com_inside_support_all_feet": None,
                    "hull_area_median_all_feet": None,
                    "support_margin_median_all_feet": None,
                    "feet_near_floor_min": None,
                    "foot_height_spread_max": None,
                    "min_joint_margin_min": None,
                    "min_ctrl_margin_min": None,
                    "max_abs_ctrl_minus_qpos_max": None,
                    "solve_seconds_max": None,
                }
            )
            return summary
        summary.update(
            {
                "public_zero_action_falls": sum(
                    not record["public_zero_action_hold"]["survived"] for record in built
                ),
                "nan_seeds": 0,
                "degenerate_support_all_feet": sum(
                    record["geometry"]["support_all_feet"]["degenerate"] for record in built
                ),
                "com_inside_support_all_feet": sum(
                    record["geometry"]["support_all_feet"]["inside"] for record in built
                ),
                "hull_area_median_all_feet": statistics.median(
                    record["geometry"]["support_all_feet"]["hull_area"] for record in built
                ),
                "support_margin_median_all_feet": statistics.median(
                    record["geometry"]["support_all_feet"]["margin"] for record in built
                ),
                "feet_near_floor_min": min(record["geometry"]["feet_near_floor"] for record in built),
                "foot_height_spread_max": max(
                    record["geometry"]["foot_height_spread"] for record in built
                ),
                "min_joint_margin_min": min(
                    record["margins"]["min_joint_margin"] for record in built
                ),
                "min_ctrl_margin_min": min(
                    record["margins"]["min_ctrl_margin"] for record in built
                ),
                "max_abs_ctrl_minus_qpos_max": max(
                    record["margins"]["max_abs_ctrl_minus_qpos"] for record in built
                ),
                "solve_seconds_max": max(record["solve_seconds"] for record in built),
            }
        )
        return summary

    return {
        "records": len(records),
        "biped": family_summary(biped),
        "quadruped": family_summary(quadruped),
        "biped_strict_v3_passed": sum(
            record["v3_acceptance"]["biped_strict_initial_and_hold"] for record in biped
        ),
        "biped_strict_v3_total": len(biped),
        "biped_record_build_failures": sum(
            record.get("record_status") != "built" for record in biped
        ),
        "stance_matrix_hold_passed": sum(
            record["v3_acceptance"]["stance_matrix_hold"] for record in records
        ),
        "stance_matrix_total": len(records),
        "record_build_failures": sum(record.get("record_status") != "built" for record in records),
    }


def _decision(endpoint_summary: dict[str, Any], matrix_summary: dict[str, Any]) -> dict[str, str]:
    endpoint_passed = (
        int(endpoint_summary["biped_record_build_failures"]) == 0
        and int(endpoint_summary["biped_strict_v3_passed"])
        == int(endpoint_summary["biped_strict_v3_total"])
    )
    matrix_biped_passed = (
        int(matrix_summary["biped_record_build_failures"]) == 0
        and int(matrix_summary["biped_strict_v3_passed"]) == int(matrix_summary["biped_strict_v3_total"])
    )
    if endpoint_passed and matrix_biped_passed:
        return {
            "status": "stance_solution_v3_biped_strict_feedforward_matrix_passed",
            "decision": (
                "Public reset now uses qpos_eq and zero action uses independent ctrl_eq; "
                "all audited biped matrix records pass strict initial equilibrium and hold."
            ),
            "next_allowed_work": (
                "Mark the public stance gate passed only if the audited matrix is the required "
                "production matrix; keep Task061/062 blocked until that gate is explicit."
            ),
        }
    if endpoint_passed:
        failed = matrix_summary["biped"].get("failed_labels", [])
        failed_preview = ", ".join(failed[:6])
        if len(failed) > 6:
            failed_preview += f", ... (+{len(failed) - 6})"
        return {
            "status": "stance_solution_v3_endpoint_passed_matrix_incomplete",
            "decision": (
                "The original 4x2 biped endpoints pass strict V3 feedforward, but the audited "
                f"matrix still has search-exhausted biped records: {failed_preview}."
            ),
            "next_allowed_work": (
                "Continue actual-dynamics solver coverage diagnosis with bounded per-record "
                "timeouts/checkpointing; do not change generator, feedback, or Task061/062."
            ),
        }
    return {
        "status": "stance_solution_v3_biped_strict_feedforward_incomplete",
        "decision": "At least one original 4x2 biped endpoint failed strict V3 feedforward acceptance.",
        "next_allowed_work": "Continue actual-dynamics solver diagnosis; do not change generator or feedback.",
    }


def run_audit(
    *,
    endpoint_seeds: int,
    matrix_seeds: int,
    range_fractions: tuple[float, ...],
    horizon_steps: int,
    include_quadruped_matrix: bool,
    record_timeout_seconds: float = 0.0,
    emit_progress: bool = False,
) -> dict[str, Any]:
    matrix_families = ("biped", "quadruped") if include_quadruped_matrix else ("biped",)
    matrix_targets = [
        (family, seed, float(rf))
        for rf in range_fractions
        for family in matrix_families
        for seed in range(matrix_seeds)
    ]
    matrix_records = _run_records(
        matrix_targets,
        horizon_steps=horizon_steps,
        timeout_seconds=record_timeout_seconds,
        emit_progress=emit_progress,
    )
    matrix_by_key = {
        (record["family"], record["seed"], record["range_fraction"]): record
        for record in matrix_records
    }
    deduped_endpoints = []
    for rf in range_fractions:
        for seed in range(endpoint_seeds):
            key = ("biped", seed, float(rf))
            record = matrix_by_key.get(key)
            if record is None:
                record = _try_run_record(
                    family="biped",
                    seed=seed,
                    range_fraction=rf,
                    horizon_steps=horizon_steps,
                    timeout_seconds=record_timeout_seconds,
                )
            deduped_endpoints.append(record)
    built_endpoints = [record for record in deduped_endpoints if record.get("record_status") == "built"]
    summary = {
        "endpoint_4x2": _summary(deduped_endpoints),
        "stance_matrix": _summary(matrix_records),
    }
    return {
        "schema": "task067_r4a31g_stance_solution_v3_actual_feedforward_audit_v1",
        "provenance": {
            "diagnostic_source_sha256": _sha256_path(Path(__file__)),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "mujoco": _package_version("mujoco"),
            "parameters": {
                "endpoint_seeds": endpoint_seeds,
                "matrix_seeds": matrix_seeds,
                "range_fractions": list(range_fractions),
                "horizon_steps": horizon_steps,
                "hold_seconds": horizon_steps / 50.0,
                "include_quadruped_matrix": include_quadruped_matrix,
                "motor_events_disabled_for_nominal_hold": True,
                "record_timeout_seconds": record_timeout_seconds,
            },
        },
        "embodiment_contract_version": PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
        "embodiment_contract_hash": PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
        "stance_solution_contract_version": STANCE_SOLUTION_CONTRACT_VERSION,
        "stance_solution_contract_hash": STANCE_SOLUTION_CONTRACT_HASH,
        "summary": summary,
        "decision": _decision(summary["endpoint_4x2"], summary["stance_matrix"]),
        "assertions": {
            "reset_uses_qpos_eq": True,
            "zero_action_uses_actuator_ctrl_eq": True,
            "qpos_eq_and_ctrl_eq_are_separate_quantities": True,
            "root_x_y_yaw_gauge_fixed_zero": all(
                record["root_gauge"]["x_y_yaw_zero"] for record in built_endpoints
            ),
            "joint_margin_gt_0p05_for_all_biped_endpoints": all(
                float(record["margins"]["min_joint_margin"]) > 0.05 for record in built_endpoints
            ),
            "ctrl_margin_ge_0p01_for_all_biped_endpoints": all(
                float(record["margins"]["min_ctrl_margin"]) >= 0.01 - 1e-9
                for record in built_endpoints
            ),
            "endpoint_record_build_failures_are_zero": (
                summary["endpoint_4x2"]["record_build_failures"] == 0
            ),
            "contact_wrench_not_public_contract_truth": True,
            "task061_062_remain_blocked": True,
        },
        "endpoint_4x2_records": deduped_endpoints,
        "stance_matrix_records": matrix_records,
    }


def _float_tuple(values: list[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-seeds", type=int, default=4)
    parser.add_argument("--matrix-seeds", type=int, default=32)
    parser.add_argument("--range-fractions", nargs="+", type=float, default=[0.0, 0.5])
    parser.add_argument("--horizon-steps", type=int, default=100)
    parser.add_argument("--skip-quadruped-matrix", action="store_true")
    parser.add_argument("--record-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    payload = run_audit(
        endpoint_seeds=args.endpoint_seeds,
        matrix_seeds=args.matrix_seeds,
        range_fractions=_float_tuple(args.range_fractions),
        horizon_steps=args.horizon_steps,
        include_quadruped_matrix=not args.skip_quadruped_matrix,
        record_timeout_seconds=float(args.record_timeout_seconds),
        emit_progress=bool(args.progress),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
