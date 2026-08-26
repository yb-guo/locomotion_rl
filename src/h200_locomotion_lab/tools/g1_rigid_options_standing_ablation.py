"""Ablate Genesis rigid solver options against zero-action standing metrics."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from h200_locomotion_lab.envs.vectorized_genesis_backend import (
    GenesisRigidContactSolverConfig,
    VectorizedGenesisBackend,
    VectorizedGenesisConfig,
)
from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.robots import load_g1_27dof_nohand_profile
from h200_locomotion_lab.tools import g1_zero_action_standing_causality as zero_action
from h200_locomotion_lab.tools.path_access import path_exists

DEFAULT_OUTPUT_ROOT = Path("outputs/task021/rigid_options_standing_ablation")
DEFAULT_SCENARIOS = (
    "default_unset",
    "newton_solver_only",
    "newton_mujoco_contact",
    "newton_solver_bundle",
)


@dataclass(frozen=True)
class RigidOptionsScenario:
    name: str
    rigid_contact_solver: GenesisRigidContactSolverConfig


SCENARIOS = {
    "default_unset": RigidOptionsScenario(
        name="default_unset",
        rigid_contact_solver=GenesisRigidContactSolverConfig(),
    ),
    "newton_solver_only": RigidOptionsScenario(
        name="newton_solver_only",
        rigid_contact_solver=GenesisRigidContactSolverConfig(constraint_solver="Newton"),
    ),
    "newton_mujoco_contact": RigidOptionsScenario(
        name="newton_mujoco_contact",
        rigid_contact_solver=GenesisRigidContactSolverConfig(
            constraint_solver="Newton",
            enable_mujoco_compatibility=True,
            enable_multi_contact=True,
        ),
    ),
    "newton_solver_bundle": RigidOptionsScenario(
        name="newton_solver_bundle",
        rigid_contact_solver=GenesisRigidContactSolverConfig(
            constraint_solver="Newton",
            enable_mujoco_compatibility=True,
            enable_multi_contact=True,
            iterations=10,
            tolerance=1e-6,
            ls_iterations=10,
            constraint_timeconst=0.02,
        ),
    ),
}

KEY_METRIC_FIELDS = (
    "status",
    "passed",
    "evaluation_passed",
    "diagnostic_passed",
    "chunks_completed",
    "chunks_expected",
    "first_tilt_chunk",
    "first_tilt_step",
    "max_reset_count",
    "final_reset_count",
    "max_tilt_bad_count",
    "final_tilt_bad_count",
    "final_termination_height_bad_count",
    "final_root_height_mean",
    "final_root_height_min",
    "final_upright_mean",
    "final_joint_position_error_rms",
    "final_joint_velocity_rms",
    "final_foot_or_body_contact_count",
    "max_foot_or_body_contact_count",
    "final_max_contact_force",
    "max_contact_force",
    "min_throughput_env_steps_per_sec",
)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result: dict[str, Any] = {
        "status": "error",
        "blocker": "",
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "scenarios": [],
    }
    exit_code = 0
    try:
        summary = run_ablation(args)
        result.update(summary)
        result["status"] = summary["status"]
    except RECOVERABLE_RUNTIME_ERRORS as exc:  # pragma: no cover - setup failure path.
        result["status"] = "error"
        result["blocker"] = f"{exc.__class__.__name__}:{exc}"
        exit_code = 1
    print(json.dumps(result, sort_keys=True), flush=True)
    if exit_code:
        raise SystemExit(exit_code)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=zero_action.positive_int, default=1024)
    parser.add_argument("--chunks", type=zero_action.positive_int, default=50)
    parser.add_argument("--chunk-steps", type=zero_action.positive_int, default=32)
    parser.add_argument("--warmup-policy-steps", type=zero_action.nonnegative_int, default=0)
    parser.add_argument("--pre-eval-reset", action="store_true")
    parser.add_argument(
        "--pre-eval-reset-scope",
        choices=("full", "all_env_ids"),
        default="full",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--control-mode",
        choices=zero_action.CONTROL_MODES,
        default="genesis_position",
    )
    parser.add_argument("--pose-profile", choices=zero_action.POSE_PROFILES, default="current")
    parser.add_argument("--gain-profile", choices=zero_action.GAIN_PROFILES, default="current")
    parser.add_argument("--root-z", type=zero_action.positive_float, default=zero_action.DEFAULT_ROOT_Z)
    parser.add_argument("--height-min", type=zero_action.positive_float, default=0.45)
    parser.add_argument("--height-max", type=zero_action.positive_float, default=1.20)
    parser.add_argument("--termination-height-min", type=zero_action.positive_float, default=0.20)
    parser.add_argument("--termination-height-max", type=zero_action.positive_float, default=1.20)
    parser.add_argument("--min-upright", type=zero_action.positive_float, default=zero_action.DEFAULT_MIN_UPRIGHT)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--logical-cuda-device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--trace-env-index", type=int, default=-1)
    parser.add_argument("--leg-pose-values", default="")
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenario names to run.",
    )
    return parser.parse_args(argv)


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    zero_action.verify_cuda_isolation(
        backend=args.backend,
        physical_gpu=str(args.physical_gpu),
        logical_cuda_device=args.logical_cuda_device,
    )
    torch = zero_action.require_torch()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = resolve_run_dir(args.output_root, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    selected = parse_scenarios(args.scenarios)
    scenario_results: list[dict[str, Any]] = []
    for scenario in selected:
        try:
            scenario_results.append(run_scenario(args=args, torch=torch, run_dir=run_dir, scenario=scenario))
        except RECOVERABLE_RUNTIME_ERRORS as exc:  # Continue so one bad solver setting does not hide later evidence.
            scenario_dir = run_dir / scenario.name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            error_result = scenario_error_result(scenario=scenario, scenario_dir=scenario_dir, exc=exc)
            zero_action.write_json(
                scenario_dir / "config.json",
                {
                    "task": "task021-rigid-options-standing-ablation",
                    "scenario": scenario_payload(scenario),
                    "status": "error",
                    "blocker": error_result["blocker"],
                },
            )
            (scenario_dir / "metrics.jsonl").touch()
            zero_action.write_json(scenario_dir / "summary.json", error_result)
            scenario_results.append(error_result)

    summary = {
        "status": "completed" if scenario_results else "error",
        "blocker": "" if scenario_results else "no_scenarios_selected",
        "run_dir": str(run_dir),
        "scenario_count": len(scenario_results),
        "scenarios": scenario_results,
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }
    zero_action.write_json(run_dir / "summary.json", summary)
    return summary


def run_scenario(
    *,
    args: argparse.Namespace,
    torch: Any,
    run_dir: Path,
    scenario: RigidOptionsScenario,
) -> dict[str, Any]:
    profile = load_g1_27dof_nohand_profile()
    pose = zero_action.pose_profile_values(args.pose_profile, profile.control.default_angles_rad)
    gains = zero_action.gain_profile_values(args.gain_profile, profile.control)
    scenario_dir = run_dir / scenario.name
    scenario_dir.mkdir(parents=True, exist_ok=False)

    backend = VectorizedGenesisBackend(
        VectorizedGenesisConfig(
            n_envs=args.n_envs,
            backend=args.backend,
            logical_cuda_device=args.logical_cuda_device,
            root_qpos=zero_action.root_qpos(args.root_z),
            default_positions_rad=pose,
            rigid_contact_solver=scenario.rigid_contact_solver,
        ),
        profile=profile,
    )
    rigid_report = backend.contact_solver_config_report()
    config_payload = build_zero_action_run_config(args=args, default_pose=pose, gains=gains)
    config_payload["task"] = "task021-rigid-options-standing-ablation"
    config_payload["scenario"] = scenario_payload(scenario)
    config_payload["rigid_options_report"] = rigid_report
    contact_trace = empty_contact_trace_metadata()
    zero_action.write_json(scenario_dir / "config.json", config_payload)

    zero_action.apply_gain_profile_to_backend(backend, gains)
    backend.reset()
    zero_action_tensor = torch.zeros((args.n_envs, profile.action_dim), device=args.logical_cuda_device)
    warmup = run_zero_action_warmup(
        torch=torch,
        backend=backend,
        zero_action=zero_action_tensor,
        args=args,
        trace_path=None,
        contact_trace=contact_trace,
    )
    if args.pre_eval_reset:
        zero_action.reset_before_eval(torch=torch, backend=backend, args=args)

    rows: list[dict[str, Any]] = []
    metrics_path = scenario_dir / "metrics.jsonl"
    total_policy_steps = 0
    for chunk_index in range(args.chunks):
        row = run_zero_action_chunk(
            torch=torch,
            backend=backend,
            zero_action=zero_action_tensor,
            args=args,
            action_joint_names=profile.actuator_order,
            chunk_index=chunk_index,
            total_policy_steps=total_policy_steps,
            trace_path=None,
            contact_trace=contact_trace,
        )
        total_policy_steps += args.chunk_steps
        rows.append(row)
        zero_action.append_jsonl(metrics_path, row)

    summary = summarize_zero_action_run(
        rows=rows,
        args=args,
        run_dir=scenario_dir,
        warmup_diagnostics=warmup,
        contact_trace=contact_trace,
    )
    summary.update(contact_summary(rows))
    result = scenario_result(
        scenario=scenario,
        scenario_dir=scenario_dir,
        rigid_report=rigid_report,
        summary=summary,
    )
    zero_action.write_json(scenario_dir / "summary.json", result)
    return result


def parse_scenarios(value: str) -> list[RigidOptionsScenario]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    scenarios: list[RigidOptionsScenario] = []
    for name in names:
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name!r}; available: {sorted(SCENARIOS)}")
        scenarios.append(SCENARIOS[name])
    return scenarios


def build_zero_action_run_config(
    *,
    args: argparse.Namespace,
    default_pose: Sequence[float],
    gains: zero_action.DiagnosticGainProfile,
) -> dict[str, Any]:
    ensure_zero_action_arg_defaults(args)
    kwargs: dict[str, Any] = {
        "args": args,
        "default_pose": default_pose,
        "gains": gains,
    }
    parameters = inspect.signature(zero_action.build_run_config).parameters
    add_supported_kwargs(
        kwargs,
        parameters,
        {
            "trace_path": None,
            "contact_trace": empty_contact_trace_metadata(),
        },
    )
    return zero_action.build_run_config(**kwargs)


def ensure_zero_action_arg_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "trace_env_index": -1,
        "leg_pose_values": "",
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)


def run_zero_action_warmup(**kwargs: Any) -> dict[str, Any]:
    return call_zero_action_helper(zero_action.run_warmup, kwargs)


def run_zero_action_chunk(**kwargs: Any) -> dict[str, Any]:
    return call_zero_action_helper(zero_action.run_chunk, kwargs)


def summarize_zero_action_run(**kwargs: Any) -> dict[str, Any]:
    return call_zero_action_helper(zero_action.summarize_run, kwargs)


def call_zero_action_helper(function: Any, kwargs: dict[str, Any]) -> Any:
    parameters = inspect.signature(function).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return function(**kwargs)
    add_supported_kwargs(
        kwargs,
        parameters,
        {
            "trace_path": None,
            "contact_trace": empty_contact_trace_metadata(),
        },
    )
    return function(
        **{
            name: value
            for name, value in kwargs.items()
            if name in parameters
        }
    )


def add_supported_kwargs(
    kwargs: dict[str, Any],
    parameters: Any,
    defaults: dict[str, Any],
) -> None:
    for name, value in defaults.items():
        if name in parameters and name not in kwargs:
            kwargs[name] = value


def empty_contact_trace_metadata() -> dict[str, Any] | None:
    factory = getattr(zero_action, "empty_contact_trace_metadata", None)
    if factory is None:
        return None
    return factory(False)


def scenario_payload(scenario: RigidOptionsScenario) -> dict[str, Any]:
    return {
        "name": scenario.name,
        "rigid_options_request": scenario.rigid_contact_solver.to_genesis_kwargs(),
    }


def scenario_result(
    *,
    scenario: RigidOptionsScenario,
    scenario_dir: Path,
    rigid_report: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        **scenario_payload(scenario),
        "rigid_options_report": rigid_report,
        "status": str(summary["status"]),
        "blocker": "",
        "key_metrics": key_metrics(summary),
        "run_dir": str(scenario_dir),
    }


def scenario_error_result(
    *,
    scenario: RigidOptionsScenario,
    scenario_dir: Path,
    exc: Exception,
) -> dict[str, Any]:
    return {
        **scenario_payload(scenario),
        "rigid_options_report": scenario.rigid_contact_solver.report(),
        "status": "error",
        "blocker": f"{exc.__class__.__name__}:{exc}",
        "key_metrics": {},
        "run_dir": str(scenario_dir),
    }


def key_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: summary[key] for key in KEY_METRIC_FIELDS if key in summary}


def contact_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "final_foot_or_body_contact_count": 0,
            "max_foot_or_body_contact_count": 0,
            "final_max_contact_force": 0.0,
            "max_contact_force": 0.0,
        }
    return {
        "final_foot_or_body_contact_count": int(
            rows[-1].get("foot_or_body_contact_count", 0)
        ),
        "max_foot_or_body_contact_count": max(
            int(row.get("foot_or_body_contact_count", 0)) for row in rows
        ),
        "final_max_contact_force": float(rows[-1].get("max_contact_force", 0.0)),
        "max_contact_force": max(float(row.get("max_contact_force", 0.0)) for row in rows),
    }


def resolve_run_dir(output_root: Path, run_id: str) -> Path:
    root = output_root if output_root.is_absolute() else Path.cwd() / output_root
    run_name = run_id.strip() or time.strftime("%Y%m%d-%H%M%S")
    run_dir = (root / run_name).resolve()
    project_prefix = zero_action.PROJECT_PREFIX.resolve()
    if path_exists(project_prefix) and project_prefix not in (run_dir, *run_dir.parents):
        raise RuntimeError(f"output path must stay under {project_prefix}: {run_dir}")
    return run_dir


def scenario_names(scenarios: Sequence[RigidOptionsScenario]) -> list[str]:
    return [scenario.name for scenario in scenarios]


def scenario_configs() -> dict[str, dict[str, Any]]:
    return {name: asdict(scenario.rigid_contact_solver) for name, scenario in SCENARIOS.items()}


if __name__ == "__main__":
    main()
