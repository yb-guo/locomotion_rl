"""Evaluate a Task039 true-TXL clean checkpoint with quality feedback."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.tools import task037_multitrial_eval_checkpoint
from h200_locomotion_lab.tools.task038_true_txl_runner_smoke_probe import (
    DEFAULT_EXPECTED_ACTION_DIM,
    DEFAULT_EXPECTED_ACTOR_MODEL_CLASS,
    DEFAULT_EXPECTED_RUNNER_CLS,
    TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID,
    _variant_label,
)
from h200_locomotion_lab.training.task039_quality_feedback import evaluate_quality_feedback

TASK039_TRUE_TXL_CLEAN_TASK_ID = TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID
ALLOWED_TASKS = (TASK039_TRUE_TXL_CLEAN_TASK_ID,)
MEMORY_DEBUG_FIELDS = ("txl_debug", "memory_debug", "policy_debug")
MEMORY_LENGTH_KEYS = ("memory_lengths", "cache_lengths", "mem_lengths", "memory_length")
INCREMENTAL_STEP_KEYS = (
    "incremental_steps",
    "policy_incremental_steps",
    "memory_incremental_steps",
    "cache_steps",
)
PREVIOUS_MEMORY_KEYS = (
    "last_attended_previous_memory_lengths",
    "previous_memory_lengths",
    "attended_previous_memory_lengths",
    "previous_memory_tokens",
    "cache_hits",
)


class PreflightError(ValueError):
    def __init__(self, reasons: list[str]) -> None:
        super().__init__(", ".join(reasons))
        self.reasons = reasons


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Task039 true-TXL clean-train multi-trial eval. "
            "pipeline_pass and quality_gate_pass are separate diagnostic gates."
        )
    )
    parser.add_argument("--task", default=TASK039_TRUE_TXL_CLEAN_TASK_ID)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--seed", type=int, default=3900301)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trial-length-s", type=float, default=2.0)
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
    parser.add_argument("--dynamic-onset-s", type=float, default=0.5)
    parser.add_argument("--dynamic-recovery-s", type=float, default=1.5)
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
        reasons.append("task_not_task039_true_txl_clean_train")
    if int(args.num_envs) <= 0:
        reasons.append("num_envs_not_positive")
    if int(args.steps) <= 0:
        reasons.append("steps_not_positive")
    if float(args.trial_length_s) <= 0.0:
        reasons.append("trial_length_s_not_positive")
    if int(args.expected_action_dim) <= 0:
        reasons.append("expected_action_dim_not_positive")
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
    wrapped = dict(result)
    wrapped["variant_label"] = _variant_label(args.task)
    wrapped["task039_true_txl_clean_only"] = True
    wrapped["policy_label"] = "True-TXL"
    wrapped["expected_runner_cls"] = args.expected_runner_cls
    wrapped["expected_actor_model_class"] = args.expected_actor_model_class
    wrapped["expected_action_dim"] = args.expected_action_dim
    wrapped["task037_pass"] = bool(result.get("pass", False))
    wrapped["task037_final_trial_pass"] = bool(result.get("final_trial_pass", False))
    wrapped["quality_metric_final_trial_pass"] = bool(result.get("final_trial_pass", False))
    wrapped["quality_claim"] = False
    wrapped["training_claim"] = False
    wrapped["eval_claim"] = False
    wrapped["reproduction_claim"] = False
    wrapped["superiority_claim"] = False
    wrapped["promotion_gate"] = "task039_quality_feedback"
    wrapped["diagnostic_note"] = (
        "Task039 true-TXL clean train only; quality_gate_pass is a diagnostic gate, "
        "not a training, reproduction, eval, or superiority claim."
    )
    memory_contract = evaluate_memory_debug_contract(wrapped)
    wrapped["memory_debug_field"] = memory_contract["field"]
    wrapped["memory_debug_present"] = memory_contract["present"]
    wrapped["memory_debug_active"] = memory_contract["active"]
    wrapped["true_txl_memory_debug_contract"] = memory_contract
    pipeline_pass, pipeline_reasons = evaluate_pipeline_pass(wrapped)
    wrapped["pipeline_pass"] = pipeline_pass
    feedback = evaluate_quality_feedback(wrapped)
    quality_reasons = list(feedback.failure_reasons)
    wrapped["quality_gate_pass"] = feedback.quality_gate_pass
    wrapped["quality_feedback"] = feedback.to_json()
    wrapped["pipeline_failure_reasons"] = pipeline_reasons
    wrapped["quality_failure_reasons"] = quality_reasons
    wrapped["failure_reasons"] = pipeline_reasons + [
        reason for reason in quality_reasons if reason not in pipeline_reasons
    ]
    wrapped["pass"] = pipeline_pass and feedback.quality_gate_pass
    return wrapped


def evaluate_pipeline_pass(summary: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary.get("task") not in ALLOWED_TASKS:
        reasons.append("task_not_task039_true_txl_clean_train")
    if summary.get("runner_cls") != summary.get("expected_runner_cls"):
        reasons.append("runner_cls_mismatch")
    if summary.get("actor_model_class") != summary.get("expected_actor_model_class"):
        reasons.append("actor_model_class_mismatch")
    expected_action_dim = _int_or_none(summary.get("expected_action_dim"))
    if _int_or_none(summary.get("action_dim")) != expected_action_dim:
        reasons.append("action_dim_mismatch")
    if _int_or_none(summary.get("total_action_dim")) != expected_action_dim:
        reasons.append("total_action_dim_mismatch")
    if not isinstance(summary.get("trial_0"), Mapping):
        reasons.append("trial_0_missing")
    if not isinstance(summary.get("final_trial"), Mapping):
        reasons.append("final_trial_missing")
    if not isinstance(summary.get("aggregate"), Mapping):
        reasons.append("aggregate_missing")
    if summary.get("error"):
        reasons.append("error_present")
    if summary.get("traceback"):
        reasons.append("traceback_present")
    memory_contract = evaluate_memory_debug_contract(summary)
    reasons.extend(_memory_debug_failure_reasons(memory_contract))
    if (
        not summary.get("task039_true_txl_clean_only")
        or summary.get("quality_claim") is not False
        or summary.get("training_claim") is not False
        or summary.get("eval_claim") is not False
        or summary.get("reproduction_claim") is not False
        or summary.get("superiority_claim") is not False
    ):
        reasons.append("claim_boundary_violation")
    return not reasons, reasons


def evaluate_memory_debug_contract(summary: Mapping[str, Any]) -> dict[str, Any]:
    field, debug = _memory_debug_mapping(summary)
    if debug is None:
        return {
            "field": None,
            "present": False,
            "shape_valid": False,
            "incremental_steps_seen": False,
            "previous_memory_exposure_seen": False,
            "active": False,
        }
    shape_valid = _memory_debug_shape_valid(debug)
    incremental_steps_seen = _memory_debug_has_incremental_steps(debug)
    previous_memory_exposure_seen = _memory_debug_has_previous_memory_exposure(debug)
    return {
        "field": field,
        "present": True,
        "shape_valid": shape_valid,
        "incremental_steps_seen": incremental_steps_seen,
        "previous_memory_exposure_seen": previous_memory_exposure_seen,
        "active": (
            shape_valid and incremental_steps_seen and previous_memory_exposure_seen
        ),
    }


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
        "fixed_command": {
            "lin_vel_x": args.lin_vel_x,
            "lin_vel_y": args.lin_vel_y,
            "ang_vel_z": args.ang_vel_z,
        },
        "expected_runner_cls": args.expected_runner_cls,
        "expected_actor_model_class": args.expected_actor_model_class,
        "expected_action_dim": args.expected_action_dim,
        "task039_true_txl_clean_only": True,
        "policy_label": "True-TXL",
        "pipeline_pass": False,
        "quality_gate_pass": False,
        "quality_metric_final_trial_pass": False,
        "pass": False,
        "promotion_gate": "task039_quality_feedback",
        "diagnostic_note": "Task039 true-TXL clean preflight/runtime failure; no quality claim.",
        "memory_debug_field": None,
        "memory_debug_present": False,
        "memory_debug_active": False,
        "true_txl_memory_debug_contract": evaluate_memory_debug_contract({}),
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "failure_reasons": ["eval_wrapper_exception"],
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


def _memory_debug_failure_reasons(contract: Mapping[str, Any]) -> list[str]:
    if not contract.get("present"):
        return ["memory_debug_missing"]
    reasons: list[str] = []
    if not contract.get("shape_valid"):
        reasons.append("memory_debug_invalid")
    if not contract.get("incremental_steps_seen"):
        reasons.append("memory_debug_no_incremental_steps")
    if not contract.get("previous_memory_exposure_seen"):
        reasons.append("memory_debug_no_previous_memory_exposure")
    return reasons


def _memory_debug_mapping(
    summary: Mapping[str, Any],
) -> tuple[str | None, Mapping[str, Any] | None]:
    for field in MEMORY_DEBUG_FIELDS:
        debug = summary.get(field)
        if isinstance(debug, Mapping):
            return field, debug
    return None, None


def _memory_debug_shape_valid(debug: Mapping[str, Any]) -> bool:
    envs = debug.get("envs")
    if isinstance(envs, list):
        env_mappings = [env for env in envs if isinstance(env, Mapping)]
        if len(env_mappings) != len(envs) or not env_mappings:
            return False
        return all(_debug_item_has_memory_lengths(env) for env in env_mappings)
    return _debug_item_has_memory_lengths(debug)


def _memory_debug_has_incremental_steps(debug: Mapping[str, Any]) -> bool:
    if _debug_item_has_positive_key(debug, INCREMENTAL_STEP_KEYS):
        return True
    envs = debug.get("envs")
    if not isinstance(envs, list):
        return False
    return any(
        _debug_item_has_positive_key(env, INCREMENTAL_STEP_KEYS)
        for env in envs
        if isinstance(env, Mapping)
    )


def _memory_debug_has_previous_memory_exposure(debug: Mapping[str, Any]) -> bool:
    if _debug_item_has_positive_key(debug, PREVIOUS_MEMORY_KEYS):
        return True
    envs = debug.get("envs")
    if not isinstance(envs, list):
        return False
    return any(
        _debug_item_has_positive_key(env, PREVIOUS_MEMORY_KEYS)
        for env in envs
        if isinstance(env, Mapping)
    )


def _debug_item_has_memory_lengths(item: Mapping[str, Any]) -> bool:
    return any(_has_present_numeric_or_sequence(item.get(key)) for key in MEMORY_LENGTH_KEYS)


def _debug_item_has_positive_key(item: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return any(_has_positive_number(item.get(key)) for key in keys)


def _has_present_numeric_or_sequence(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, (list, tuple)):
        return bool(value)
    return False


def _has_positive_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, (list, tuple)):
        return any(_has_positive_number(item) for item in value)
    return False


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
