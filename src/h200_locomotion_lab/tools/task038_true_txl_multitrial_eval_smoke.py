"""Run a Task038 true-TXL checkpoint through multi-trial eval pipeline smoke."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.tools import task037_multitrial_eval_checkpoint
from h200_locomotion_lab.tools.task038_true_txl_runner_smoke_probe import (
    DEFAULT_EXPECTED_ACTION_DIM,
    DEFAULT_EXPECTED_ACTOR_MODEL_CLASS,
    DEFAULT_EXPECTED_RUNNER_CLS,
    HELDOUT_TRUE_TXL_RUNNER_SMOKE_TASK_ID,
    TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID,
    _variant_label,
)

DEFAULT_TASK = TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID
ALLOWED_TASKS = (
    TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID,
    HELDOUT_TRUE_TXL_RUNNER_SMOKE_TASK_ID,
)
REQUIRED_TRIAL_NUMERIC_FIELDS = (
    ("trial_index",),
    ("sample_count",),
    ("completion_count",),
    ("completion_ratio",),
    ("fall_count",),
    ("fall_ratio",),
    ("zero_fall_ratio",),
    ("timeout_count",),
    ("reward_mean",),
    ("lin_vel_error", "mean"),
    ("yaw_vel_error", "mean"),
    ("gravity_xy", "mean"),
    ("gravity_xy", "max"),
    ("root_z", "mean"),
    ("root_z", "min"),
)
REQUIRED_AGGREGATE_NUMERIC_FIELDS = (
    ("trial_count",),
    ("sample_count",),
    ("completion_count",),
    ("fall_count",),
    ("fall_ratio",),
    ("zero_fall_ratio",),
    ("lin_vel_error_mean",),
    ("yaw_vel_error_mean",),
    ("gravity_xy_mean",),
    ("gravity_xy_max",),
    ("root_z_min",),
    ("num_envs",),
)


class PreflightError(ValueError):
    def __init__(self, reasons: list[str]) -> None:
        super().__init__(", ".join(reasons))
        self.reasons = reasons


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Task038 true-TXL checkpoint multi-trial eval pipeline smoke. "
            "Top-level pass means pipeline health only, not policy quality."
        )
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=3801601)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trial-length-s", type=float, default=0.5)
    parser.add_argument("--lin-vel-x", type=float, default=0.4)
    parser.add_argument("--lin-vel-y", type=float, default=0.0)
    parser.add_argument("--ang-vel-z", type=float, default=0.0)
    parser.add_argument(
        "--dynamic-case",
        choices=sorted(task037_multitrial_eval_checkpoint.DYNAMIC_CASES),
        default="none",
    )
    parser.add_argument(
        "--dynamic-dead-joint",
        choices=task037_multitrial_eval_checkpoint.DEFAULT_JOINTS,
    )
    parser.add_argument("--dynamic-onset-s", type=float, default=0.2)
    parser.add_argument("--dynamic-recovery-s", type=float, default=0.4)
    parser.add_argument(
        "--force-dead-joint",
        choices=task037_multitrial_eval_checkpoint.DEFAULT_JOINTS,
    )
    parser.add_argument("--dead-scale", type=float, default=0.0)
    parser.add_argument("--min-final-completion-ratio", type=float, default=0.95)
    parser.add_argument("--max-final-fall-ratio", type=float, default=0.50)
    parser.add_argument("--max-final-lin-vel-error", type=float, default=1.20)
    parser.add_argument("--max-final-yaw-vel-error", type=float, default=1.00)
    parser.add_argument("--max-final-gravity-xy", type=float, default=0.90)
    parser.add_argument("--min-final-root-z", type=float, default=0.35)
    parser.add_argument("--expected-action-dim", type=int, default=DEFAULT_EXPECTED_ACTION_DIM)
    parser.add_argument("--expected-runner-cls", default=DEFAULT_EXPECTED_RUNNER_CLS)
    parser.add_argument(
        "--expected-actor-model-class",
        default=DEFAULT_EXPECTED_ACTOR_MODEL_CLASS,
    )
    return parser.parse_args(argv)


def preflight_args(args: argparse.Namespace) -> None:
    reasons: list[str] = []
    if args.task not in ALLOWED_TASKS:
        reasons.append("task_not_true_txl_runner_smoke")
    if int(args.num_envs) <= 0:
        reasons.append("num_envs_not_positive")
    if int(args.steps) <= 0:
        reasons.append("steps_not_positive")
    if float(args.trial_length_s) <= 0.0:
        reasons.append("trial_length_s_not_positive")
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.exists():
        reasons.append("checkpoint_missing")
    if reasons:
        raise PreflightError(reasons)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    preflight_args(args)
    result = task037_multitrial_eval_checkpoint.run_eval(args)
    return wrap_task037_result(args, result)


def wrap_task037_result(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    quality_metric_final_trial_pass = bool(result.get("final_trial_pass", False))
    wrapped = dict(result)
    wrapped["variant_label"] = _variant_label(args.task)
    wrapped["expected_action_dim"] = args.expected_action_dim
    wrapped["expected_runner_cls"] = args.expected_runner_cls
    wrapped["expected_actor_model_class"] = args.expected_actor_model_class
    wrapped["quality_metric_final_trial_pass"] = quality_metric_final_trial_pass
    wrapped["task037_final_trial_pass"] = quality_metric_final_trial_pass
    wrapped["task037_pass"] = bool(result.get("pass", False))
    wrapped["eval_pipeline_smoke_only"] = True
    wrapped["quality_claim"] = False
    wrapped["training_claim"] = False
    wrapped["eval_claim"] = False
    wrapped["reproduction_claim"] = False
    wrapped["superiority_claim"] = False
    wrapped["promotion_gate"] = "pipeline_smoke_only"
    wrapped["pipeline_note"] = (
        "Top-level pass is eval pipeline health only; "
        "quality_metric_final_trial_pass is not a Task038 quality claim."
    )
    wrapped.pop("final_trial_pass", None)
    wrapped["eval_pipeline_smoke_pass"], wrapped["failure_reasons"] = evaluate_pipeline_pass(
        wrapped
    )
    wrapped["pipeline_pass"] = wrapped["eval_pipeline_smoke_pass"]
    wrapped["pass"] = wrapped["eval_pipeline_smoke_pass"]
    return wrapped


def evaluate_pipeline_pass(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary.get("task") not in ALLOWED_TASKS:
        reasons.append("task_not_true_txl_runner_smoke")
    if not summary.get("checkpoint"):
        reasons.append("checkpoint_missing")
    if not isinstance(summary.get("trial_0"), dict):
        reasons.append("trial_0_missing")
    if not isinstance(summary.get("final_trial"), dict):
        reasons.append("final_trial_missing")
    if not isinstance(summary.get("aggregate"), dict):
        reasons.append("aggregate_missing")
    schema_reasons = _required_metric_schema_reasons(summary)
    reasons.extend(reason for reason in schema_reasons if reason not in reasons)
    if not _metrics_all_finite(
        {
            "trial_0": summary.get("trial_0"),
            "final_trial": summary.get("final_trial"),
            "aggregate": summary.get("aggregate"),
        }
    ):
        reasons.append("metrics_not_finite")
    if int(summary.get("num_envs") or 0) <= 0:
        reasons.append("num_envs_not_positive")
    if int(summary.get("steps") or 0) <= 0:
        reasons.append("steps_not_positive")
    if summary.get("error") or summary.get("traceback"):
        reasons.append("top_level_exception")
    if (
        not summary.get("eval_pipeline_smoke_only")
        or summary.get("quality_claim")
        or summary.get("training_claim")
        or summary.get("eval_claim")
        or summary.get("reproduction_claim")
        or summary.get("superiority_claim")
    ):
        reasons.append("claim_boundary_violation")
    return not reasons, reasons


def _required_metric_schema_reasons(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for name in ("trial_0", "final_trial"):
        trial = summary.get(name)
        if not isinstance(trial, dict):
            continue
        if not trial:
            reasons.append(f"{name}_metrics_missing")
            continue
        if not isinstance(trial.get("reset_reason_counts"), dict):
            reasons.append(f"{name}_reset_reason_counts_not_dict")
        if not _required_numeric_fields_finite(trial, REQUIRED_TRIAL_NUMERIC_FIELDS):
            reasons.append(f"{name}_metrics_missing_or_nonfinite")

    aggregate = summary.get("aggregate")
    if isinstance(aggregate, dict):
        if not aggregate:
            reasons.append("aggregate_metrics_missing")
        elif not _required_numeric_fields_finite(
            aggregate, REQUIRED_AGGREGATE_NUMERIC_FIELDS
        ):
            reasons.append("aggregate_metrics_missing_or_nonfinite")
    return reasons


def _required_numeric_fields_finite(
    metrics: dict[str, Any], fields: tuple[tuple[str, ...], ...]
) -> bool:
    for path in fields:
        found, value = _get_path(metrics, path)
        if not found or not _is_finite_number(value):
            return False
    return True


def _get_path(metrics: dict[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    current: Any = metrics
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser()
    summary = {
        "task": args.task,
        "variant_label": _variant_label(args.task),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "command": list(sys.argv),
        "seed": args.seed,
        "device": args.device,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "trial_length_s": args.trial_length_s,
        "expected_action_dim": args.expected_action_dim,
        "expected_runner_cls": args.expected_runner_cls,
        "expected_actor_model_class": args.expected_actor_model_class,
        "quality_metric_final_trial_pass": False,
        "pipeline_pass": False,
        "eval_pipeline_smoke_pass": False,
        "eval_pipeline_smoke_only": True,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "pass": False,
        "failure_reasons": ["probe_exception"],
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


def _metrics_all_finite(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bool)):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_metrics_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_metrics_all_finite(item) for item in value)
    return True


def main() -> None:
    args = parse_args()
    try:
        preflight_args(args)
        summary = run_eval(args)
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        summary = build_failure_summary(args, exc)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
