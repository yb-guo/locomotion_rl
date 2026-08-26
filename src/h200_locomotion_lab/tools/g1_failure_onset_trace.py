"""Trace zero-action standing failure onset with one-step metric chunks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.tools import g1_zero_action_standing_causality as zero_action
from h200_locomotion_lab.tools.path_access import path_exists

DEFAULT_OUTPUT_ROOT = Path("outputs/task021/failure_onset_trace")
DEFAULT_SCENARIOS = (
    "baseline_current",
    "control_custom_pd_torque",
    "gain_unitree_leg",
    "combo_custom_pd_unitree_leg",
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
TRACE_ROW_FIELDS = (
    "chunk_index",
    "total_policy_steps",
    "reset_count",
    "height_bad_count",
    "termination_height_bad_count",
    "tilt_bad_count",
    "root_height_mean",
    "root_height_min",
    "upright_mean",
    "upright_min",
    "joint_position_error_rms",
    "joint_position_error_max",
    "joint_velocity_rms",
    "joint_velocity_max",
    "control_kind",
    "control_rms",
    "control_max",
    "force_saturation_ratio",
    "foot_or_body_contact_count",
    "max_contact_force",
    "top_joint_position_error_rms",
)


@dataclass(frozen=True)
class TraceScenario:
    name: str
    layer: str
    args: tuple[str, ...] = ()
    hypothesis: str = ""


SCENARIOS: dict[str, TraceScenario] = {
    "baseline_current": TraceScenario(
        name="baseline_current",
        layer="baseline",
        hypothesis="Current zero-action standing onset trace.",
    ),
    "control_custom_pd_torque": TraceScenario(
        name="control_custom_pd_torque",
        layer="control",
        args=("--control-mode", "custom_pd_torque"),
        hypothesis="Custom PD torque improves onset more than baseline.",
    ),
    "gain_unitree_leg": TraceScenario(
        name="gain_unitree_leg",
        layer="gain",
        args=("--gain-profile", "unitree_leg_gains"),
        hypothesis="Unitree-like leg gains improve onset more than baseline.",
    ),
    "combo_custom_pd_unitree_leg": TraceScenario(
        name="combo_custom_pd_unitree_leg",
        layer="combo",
        args=(
            "--control-mode",
            "custom_pd_torque",
            "--gain-profile",
            "unitree_leg_gains",
        ),
        hypothesis="Combining custom PD torque and Unitree-like leg gains delays failure.",
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
        summary = run_trace(args)
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
    parser.add_argument("--n-envs", type=zero_action.positive_int, default=512)
    parser.add_argument("--chunks", type=zero_action.positive_int, default=96)
    parser.add_argument("--chunk-steps", type=zero_action.positive_int, default=1)
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
    parser.add_argument("--asset-path", type=Path, default=None)
    parser.add_argument("--scenario-timeout-s", type=zero_action.positive_float, default=300.0)
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenario names to run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run_trace(args: argparse.Namespace) -> dict[str, Any]:
    zero_action.verify_cuda_isolation(
        backend=args.backend,
        physical_gpu=str(args.physical_gpu),
        logical_cuda_device=args.logical_cuda_device,
    )
    run_dir = resolve_run_dir(args.output_root, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    scenario_results: list[dict[str, Any]] = []
    for scenario in parse_scenarios(args.scenarios):
        try:
            scenario_results.append(run_scenario(args=args, run_dir=run_dir, scenario=scenario))
        except RECOVERABLE_RUNTIME_ERRORS as exc:
            result = scenario_error_result(args=args, run_dir=run_dir, scenario=scenario, exc=exc)
            write_json(run_dir / f"{scenario.name}.json", result)
            scenario_results.append(result)
    summary = {
        "status": "completed" if scenario_results else "error",
        "blocker": "" if scenario_results else "no_scenarios_selected",
        "run_dir": str(run_dir),
        "scenario_count": len(scenario_results),
        "scenarios": scenario_results,
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "asset_path": None if args.asset_path is None else args.asset_path.as_posix(),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def run_scenario(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    scenario: TraceScenario,
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
        result = {
            **scenario_payload,
            "status": "dry_run",
            "blocker": "",
            "key_metrics": {},
            "timeline_focus": {},
        }
        write_json(run_dir / f"{scenario.name}.json", result)
        return result
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
    metrics_path = args.output_root / scenario_run_id / "metrics.jsonl"
    rows = read_jsonl(metrics_path)
    key_metrics = key_metrics_from_summary(scenario_summary)
    result = {
        **scenario_payload,
        "status": status if completed.returncode == 0 else "error",
        "blocker": blocker,
        "returncode": completed.returncode,
        "key_metrics": key_metrics,
        "timeline_focus": timeline_focus(rows=rows, summary=scenario_summary),
        "stdout_tail": tail_lines(completed.stdout),
        "stderr_tail": tail_lines(completed.stderr),
    }
    write_json(run_dir / f"{scenario.name}.json", result)
    return result


def build_zero_action_command(
    *,
    args: argparse.Namespace,
    scenario: TraceScenario,
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
    if args.asset_path is not None:
        command.extend(["--asset-path", args.asset_path.as_posix()])
    command.extend(scenario.args)
    return command


def parse_scenarios(value: str) -> list[TraceScenario]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    scenarios: list[TraceScenario] = []
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


def timeline_focus(*, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {"rows_around_first_tilt": [], "first_tilt_row": None}
    first_row = first_tilt_row(rows=rows, summary=summary)
    if first_row is None:
        selected_rows = rows[-min(5, len(rows)) :]
    else:
        index = rows.index(first_row)
        selected_rows = rows[max(0, index - 4) : min(len(rows), index + 2)]
    return {
        "first_tilt_row": None if first_row is None else slim_trace_row(first_row),
        "rows_around_first_tilt": [slim_trace_row(row) for row in selected_rows],
        "max_contact_force": max(float(row.get("max_contact_force", 0.0) or 0.0) for row in rows),
        "max_contact_count": max(
            int(row.get("foot_or_body_contact_count", 0) or 0) for row in rows
        ),
        "max_force_saturation_ratio": max(
            float(row.get("force_saturation_ratio", 0.0) or 0.0) for row in rows
        ),
    }


def first_tilt_row(*, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any] | None:
    first_step = summary.get("first_tilt_step")
    if first_step is not None:
        for row in rows:
            chunk_step = int(row.get("chunk_index", -1)) * int(row.get("chunk_steps", 1))
            if chunk_step == int(first_step) and int(row.get("tilt_bad_count", 0)) > 0:
                return row
    return next((row for row in rows if int(row.get("tilt_bad_count", 0)) > 0), None)


def slim_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in TRACE_ROW_FIELDS if key in row}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def scenario_error_result(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    scenario: TraceScenario,
    exc: Exception,
) -> dict[str, Any]:
    scenario_run_id = scenario_run_name(run_dir.name, scenario.name)
    try:
        command = build_zero_action_command(args=args, scenario=scenario, run_id=scenario_run_id)
    except RECOVERABLE_RUNTIME_ERRORS:
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
        "timeline_focus": {},
    }


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


def scenario_run_name(trace_run_name: str, scenario_name: str) -> str:
    return f"{trace_run_name}-{scenario_name}"


def resolve_run_dir(output_root: Path, run_id: str) -> Path:
    root = output_root if output_root.is_absolute() else Path.cwd() / output_root
    run_name = run_id.strip() or time.strftime("%Y%m%d-%H%M%S")
    run_dir = (root / run_name).resolve()
    project_prefix = zero_action.PROJECT_PREFIX.resolve()
    if path_exists(project_prefix) and project_prefix not in (run_dir, *run_dir.parents):
        raise RuntimeError(f"output path must stay under {project_prefix}: {run_dir}")
    return run_dir


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
