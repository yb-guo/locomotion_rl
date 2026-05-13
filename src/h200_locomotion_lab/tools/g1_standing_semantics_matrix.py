"""Run a bounded zero-action standing root-cause matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

from h200_locomotion_lab.tools import g1_zero_action_standing_causality as zero_action


DEFAULT_OUTPUT_ROOT = Path("outputs/task021/standing_semantics_matrix")
DEFAULT_SCENARIOS = (
    "baseline_current",
    "control_resend_physics",
    "control_custom_pd_torque",
    "gain_unitree_leg",
    "gain_global_kv_2x",
    "gain_kp_half_kv_2x",
    "gain_force_limit_2x",
    "pose_unitree_gym",
    "root_z_0_90",
    "root_z_1_00",
    "root_z_1_10",
)
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
    "min_throughput_env_steps_per_sec",
)


@dataclass(frozen=True)
class MatrixScenario:
    name: str
    layer: str
    args: tuple[str, ...] = ()
    hypothesis: str = ""


SCENARIOS: dict[str, MatrixScenario] = {
    "baseline_current": MatrixScenario(
        name="baseline_current",
        layer="baseline",
        hypothesis="Reproduce the current zero-action standing failure.",
    ),
    "control_resend_physics": MatrixScenario(
        name="control_resend_physics",
        layer="control",
        args=("--control-mode", "genesis_position_resend_physics"),
        hypothesis="Resending the same position target every physics step delays failure.",
    ),
    "control_custom_pd_torque": MatrixScenario(
        name="control_custom_pd_torque",
        layer="control",
        args=("--control-mode", "custom_pd_torque"),
        hypothesis="Explicit PD torque control delays failure versus Genesis position control.",
    ),
    "gain_unitree_leg": MatrixScenario(
        name="gain_unitree_leg",
        layer="gain",
        args=("--gain-profile", "unitree_leg_gains"),
        hypothesis="Unitree-like leg gains stabilize the same pose/control path.",
    ),
    "gain_global_kv_2x": MatrixScenario(
        name="gain_global_kv_2x",
        layer="gain",
        args=("--gain-profile", "global_kv_2x"),
        hypothesis="More damping stabilizes the same pose/control path.",
    ),
    "gain_kp_half_kv_2x": MatrixScenario(
        name="gain_kp_half_kv_2x",
        layer="gain",
        args=("--gain-profile", "global_kp_0_5x_kv_2x"),
        hypothesis="Lower stiffness plus more damping stabilizes the same pose/control path.",
    ),
    "gain_force_limit_2x": MatrixScenario(
        name="gain_force_limit_2x",
        layer="gain",
        args=("--gain-profile", "force_limit_2x"),
        hypothesis="Force limits are clipping the stabilizing controller.",
    ),
    "pose_unitree_gym": MatrixScenario(
        name="pose_unitree_gym",
        layer="pose",
        args=("--pose-profile", "unitree_gym"),
        hypothesis="Unitree-style default pose avoids the failure.",
    ),
    "root_z_0_90": MatrixScenario(
        name="root_z_0_90",
        layer="root_z",
        args=("--root-z", "0.90"),
        hypothesis="A higher root reset gives the controller room to settle.",
    ),
    "root_z_1_00": MatrixScenario(
        name="root_z_1_00",
        layer="root_z",
        args=("--root-z", "1.00"),
        hypothesis="A higher root reset gives the controller room to settle.",
    ),
    "root_z_1_10": MatrixScenario(
        name="root_z_1_10",
        layer="root_z",
        args=("--root-z", "1.10"),
        hypothesis="A higher root reset gives the controller room to settle.",
    ),
}


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
        summary = run_matrix(args)
        result.update(summary)
        result["status"] = summary["status"]
    except Exception as exc:  # pragma: no cover - setup failure path.
        result["status"] = "error"
        result["blocker"] = f"{exc.__class__.__name__}:{exc}"
        exit_code = 1
    print(json.dumps(result, sort_keys=True), flush=True)
    if exit_code:
        raise SystemExit(exit_code)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=zero_action.positive_int, default=512)
    parser.add_argument("--chunks", type=zero_action.positive_int, default=8)
    parser.add_argument("--chunk-steps", type=zero_action.positive_int, default=32)
    parser.add_argument("--warmup-policy-steps", type=zero_action.nonnegative_int, default=0)
    parser.add_argument("--pre-eval-reset", action="store_true")
    parser.add_argument(
        "--pre-eval-reset-scope",
        choices=("full", "all_env_ids"),
        default="full",
    )
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--scenario-timeout-s", type=zero_action.positive_float, default=300.0)
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenario names to run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    zero_action.verify_cuda_isolation(
        backend=args.backend,
        physical_gpu=str(args.physical_gpu),
        logical_cuda_device=args.logical_cuda_device,
    )
    run_dir = resolve_run_dir(args.output_root, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    selected = parse_scenarios(args.scenarios)
    scenario_results: list[dict[str, Any]] = []
    for scenario in selected:
        try:
            scenario_results.append(run_scenario(args=args, run_dir=run_dir, scenario=scenario))
        except Exception as exc:
            result = scenario_error_result(args=args, run_dir=run_dir, scenario=scenario, exc=exc)
            write_json(run_dir / f"{scenario.name}.json", result)
            scenario_results.append(result)
    summary = {
        "status": "completed" if scenario_results else "error",
        "blocker": "" if scenario_results else "no_scenarios_selected",
        "run_dir": str(run_dir),
        "scenario_count": len(scenario_results),
        "scenarios": scenario_results,
        "ranked_results": rank_results(scenario_results),
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def run_scenario(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    scenario: MatrixScenario,
) -> dict[str, Any]:
    scenario_run_id = scenario_run_name(run_dir.name, scenario.name)
    command = build_zero_action_command(args=args, scenario=scenario, run_id=scenario_run_id)
    scenario_payload = {
        "name": scenario.name,
        "layer": scenario.layer,
        "hypothesis": scenario.hypothesis,
        "zero_action_args": list(scenario.args),
        "command": command,
        "run_dir": str(args.output_root / scenario_run_id),
    }
    if args.dry_run:
        return {
            **scenario_payload,
            "status": "dry_run",
            "blocker": "",
            "key_metrics": {},
        }
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=float(args.scenario_timeout_s),
        check=False,
    )
    parsed = parse_probe_stdout(completed.stdout)
    scenario_summary = parsed if isinstance(parsed, dict) else {}
    status = str(scenario_summary.get("status", "error"))
    blocker = str(scenario_summary.get("blocker", ""))
    if completed.returncode != 0 and not blocker:
        blocker = last_nonempty_line(completed.stderr) or f"returncode:{completed.returncode}"
    key_metrics = key_metrics_from_summary(scenario_summary)
    result = {
        **scenario_payload,
        "status": status if completed.returncode == 0 else "error",
        "blocker": blocker,
        "returncode": completed.returncode,
        "key_metrics": key_metrics,
        "stdout_tail": tail_lines(completed.stdout),
        "stderr_tail": tail_lines(completed.stderr),
    }
    write_json(run_dir / f"{scenario.name}.json", result)
    return result


def scenario_error_result(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    scenario: MatrixScenario,
    exc: Exception,
) -> dict[str, Any]:
    scenario_run_id = scenario_run_name(run_dir.name, scenario.name)
    try:
        command = build_zero_action_command(args=args, scenario=scenario, run_id=scenario_run_id)
    except Exception:
        command = []
    return {
        "name": scenario.name,
        "layer": scenario.layer,
        "hypothesis": scenario.hypothesis,
        "zero_action_args": list(scenario.args),
        "command": command,
        "run_dir": str(args.output_root / scenario_run_id),
        "status": "error",
        "blocker": f"{exc.__class__.__name__}:{exc}",
        "key_metrics": {},
    }


def build_zero_action_command(
    *,
    args: argparse.Namespace,
    scenario: MatrixScenario,
    run_id: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "h200_locomotion_lab.tools.g1_zero_action_standing_causality",
        "--n-envs",
        str(args.n_envs),
        "--chunks",
        str(args.chunks),
        "--chunk-steps",
        str(args.chunk_steps),
        "--warmup-policy-steps",
        str(args.warmup_policy_steps),
        "--seed",
        str(args.seed),
        "--height-min",
        str(args.height_min),
        "--height-max",
        str(args.height_max),
        "--termination-height-min",
        str(args.termination_height_min),
        "--termination-height-max",
        str(args.termination_height_max),
        "--min-upright",
        str(args.min_upright),
        "--backend",
        args.backend,
        "--physical-gpu",
        str(args.physical_gpu),
        "--logical-cuda-device",
        args.logical_cuda_device,
        "--output-root",
        str(args.output_root),
        "--run-id",
        run_id,
    ]
    if args.pre_eval_reset:
        command.extend(["--pre-eval-reset", "--pre-eval-reset-scope", args.pre_eval_reset_scope])
    command.extend(scenario.args)
    return command


def parse_scenarios(value: str) -> list[MatrixScenario]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    scenarios: list[MatrixScenario] = []
    for name in names:
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name!r}; available: {sorted(SCENARIOS)}")
        scenarios.append(SCENARIOS[name])
    return scenarios


def key_metrics_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = {key: summary[key] for key in KEY_METRIC_FIELDS if key in summary}
    scenario_rows = summary.get("scenarios")
    if isinstance(scenario_rows, list) and scenario_rows:
        nested = scenario_rows[0].get("key_metrics", {})
        if isinstance(nested, dict):
            metrics.update(nested)
    return metrics


def rank_results(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": result["name"],
            "layer": result["layer"],
            "status": result["status"],
            "score": scenario_score(result),
            "first_tilt_step": result["key_metrics"].get("first_tilt_step"),
            "max_reset_count": result["key_metrics"].get("max_reset_count"),
            "final_reset_count": result["key_metrics"].get("final_reset_count"),
            "final_tilt_bad_count": result["key_metrics"].get("final_tilt_bad_count"),
            "final_root_height_min": result["key_metrics"].get("final_root_height_min"),
            "final_upright_mean": result["key_metrics"].get("final_upright_mean"),
        }
        for result in sorted(results, key=scenario_score, reverse=True)
    ]


def scenario_score(result: dict[str, Any]) -> tuple[int, float, float, float]:
    metrics = result.get("key_metrics", {})
    if result.get("status") == "error" or not metrics:
        return (-1, -1.0e9, -1.0e9, -1.0e9)
    passed = 1 if metrics.get("evaluation_passed") else 0
    first_tilt = metrics.get("first_tilt_step")
    first_tilt_value = 1.0e9 if first_tilt is None else float(first_tilt or 0.0)
    max_reset = -float(metrics.get("max_reset_count", 1.0e9) or 0.0)
    root_min = float(metrics.get("final_root_height_min", 0.0) or 0.0)
    return (passed, first_tilt_value, max_reset, root_min)


def parse_probe_stdout(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped or not stripped.startswith("{"):
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def tail_lines(value: str, count: int = 20) -> list[str]:
    return value.splitlines()[-count:]


def last_nonempty_line(value: str) -> str:
    for line in reversed(value.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def scenario_run_name(matrix_run_name: str, scenario_name: str) -> str:
    return f"{matrix_run_name}-{scenario_name}"


def resolve_run_dir(output_root: Path, run_id: str) -> Path:
    root = output_root if output_root.is_absolute() else Path.cwd() / output_root
    run_name = run_id.strip() or time.strftime("%Y%m%d-%H%M%S")
    run_dir = (root / run_name).resolve()
    project_prefix = zero_action.PROJECT_PREFIX.resolve()
    if project_prefix.exists() and project_prefix not in (run_dir, *run_dir.parents):
        raise RuntimeError(f"output path must stay under {project_prefix}: {run_dir}")
    return run_dir


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
