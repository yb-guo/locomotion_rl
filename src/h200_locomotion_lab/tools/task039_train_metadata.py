"""Write Task039 training metadata without making a quality claim."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS


class PreflightError(ValueError):
    def __init__(self, reasons: list[str]) -> None:
        super().__init__(", ".join(reasons))
        self.reasons = reasons


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write Task039 train metadata JSON. This records provenance only; "
            "top-level pass is always false unless a separate eval quality gate passes."
        )
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--policy-label", required=True)
    parser.add_argument("--runner-cls", required=True)
    parser.add_argument("--actor-model-class", required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--train-envs", type=int, required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--save-interval", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--command-speed", type=float, default=0.4)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--wall-time-s", type=float)
    parser.add_argument("--steps-per-second", type=float)
    parser.add_argument("--gpu-name", default="")
    return parser.parse_args(argv)


def preflight_args(args: argparse.Namespace) -> None:
    reasons: list[str] = []
    if int(args.action_dim) <= 0:
        reasons.append("action_dim_not_positive")
    if int(args.train_envs) <= 0:
        reasons.append("train_envs_not_positive")
    if int(args.max_iterations) <= 0:
        reasons.append("max_iterations_not_positive")
    if int(args.save_interval) <= 0:
        reasons.append("save_interval_not_positive")
    if int(args.seed) < 0:
        reasons.append("seed_negative")
    if not math.isfinite(float(args.command_speed)):
        reasons.append("command_speed_not_finite")
    if args.wall_time_s is not None and (
        not math.isfinite(float(args.wall_time_s)) or float(args.wall_time_s) < 0.0
    ):
        reasons.append("wall_time_s_invalid")
    if args.steps_per_second is not None and (
        not math.isfinite(float(args.steps_per_second))
        or float(args.steps_per_second) <= 0.0
    ):
        reasons.append("steps_per_second_not_positive")
    if not Path(args.checkpoint).expanduser().exists():
        reasons.append("checkpoint_missing")
    if not Path(args.stdout_log).expanduser().exists():
        reasons.append("stdout_log_missing")
    if not Path(args.log_dir).expanduser().exists():
        reasons.append("log_dir_missing")
    if reasons:
        raise PreflightError(reasons)


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    preflight_args(args)
    return {
        "schema": "task039_train_metadata_v1",
        "task039_train_metadata_only": True,
        "metadata_pass": True,
        "pipeline_pass": True,
        "quality_gate_pass": False,
        "pass": False,
        "task": args.task,
        "policy_label": args.policy_label,
        "runner_cls": args.runner_cls,
        "actor_model_class": args.actor_model_class,
        "action_dim": args.action_dim,
        "train_envs": args.train_envs,
        "max_iterations": args.max_iterations,
        "save_interval": args.save_interval,
        "seed": args.seed,
        "device": args.device,
        "command_speed": args.command_speed,
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "stdout_log": str(Path(args.stdout_log).expanduser()),
        "log_dir": str(Path(args.log_dir).expanduser()),
        "experiment_name": args.experiment_name,
        "run_name": args.run_name,
        "wall_time_s": args.wall_time_s,
        "steps_per_second": args.steps_per_second,
        "gpu_name": args.gpu_name,
        "failure_reasons": [],
        "diagnostic_note": (
            "Training provenance only. This JSON is not a training success, "
            "quality, eval, reproduction, or superiority claim."
        ),
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
    }


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    summary = {
        "schema": "task039_train_metadata_v1",
        "task039_train_metadata_only": True,
        "metadata_pass": False,
        "pipeline_pass": False,
        "quality_gate_pass": False,
        "pass": False,
        "task": getattr(args, "task", ""),
        "policy_label": getattr(args, "policy_label", ""),
        "runner_cls": getattr(args, "runner_cls", ""),
        "actor_model_class": getattr(args, "actor_model_class", ""),
        "action_dim": getattr(args, "action_dim", None),
        "train_envs": getattr(args, "train_envs", None),
        "max_iterations": getattr(args, "max_iterations", None),
        "save_interval": getattr(args, "save_interval", None),
        "seed": getattr(args, "seed", None),
        "device": getattr(args, "device", ""),
        "command_speed": getattr(args, "command_speed", None),
        "checkpoint": str(Path(getattr(args, "checkpoint", "")).expanduser()),
        "stdout_log": str(Path(getattr(args, "stdout_log", "")).expanduser()),
        "log_dir": str(Path(getattr(args, "log_dir", "")).expanduser()),
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "failure_reasons": ["train_metadata_exception"],
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }
    if isinstance(exc, PreflightError):
        summary["preflight_rejected"] = True
        summary["failure_reasons"] = list(exc.reasons)
    return summary


def write_json_summary(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary["json_path"] = str(output)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        summary = build_summary(args)
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        summary = build_failure_summary(args, exc)
    summary["command"] = list(sys.argv)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
