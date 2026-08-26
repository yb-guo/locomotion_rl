"""Evaluate Task042 memory ablations on top of the Task041 clean gate."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.tools import (
    task037_multitrial_eval_checkpoint,
    task039_true_txl_clean_eval,
    task041_sequence_txl_clean_eval,
)
from h200_locomotion_lab.training.task039_quality_feedback import evaluate_quality_feedback

TASK042_TRUE_TXL_DYNAMIC_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task042-TrainTrueTxlDynamicMotorFailure-Fast1p6"
)
TASK042_DYNAMIC_EXPECTED_RUNNER_CLS = "Task038TrueTxlMemoryK160Runner"
TASK042_ALLOWED_TASKS = (
    task039_true_txl_clean_eval.TASK039_TRUE_TXL_CLEAN_TASK_ID,
    TASK042_TRUE_TXL_DYNAMIC_TASK_ID,
)


def parse_args(argv: list[str] | None = None, *, description: str | None = None) -> argparse.Namespace:
    return task041_sequence_txl_clean_eval.parse_args(argv, description=description)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    if args.task == task039_true_txl_clean_eval.TASK039_TRUE_TXL_CLEAN_TASK_ID:
        result = task041_sequence_txl_clean_eval.run_eval(args)
    else:
        result = run_dynamic_eval(args)
    return wrap_task041_result(args, result)


def run_dynamic_eval(args: argparse.Namespace) -> dict[str, Any]:
    preflight_dynamic_args(args)
    task041_sequence_txl_clean_eval._install_ipython_display_stub()
    task041_sequence_txl_clean_eval._install_wandb_stub()
    task041_sequence_txl_clean_eval._install_wcwidth_stub()
    result = task037_multitrial_eval_checkpoint.run_eval(args)
    return wrap_dynamic_result(args, result)


def preflight_dynamic_args(args: argparse.Namespace) -> None:
    reasons: list[str] = []
    if args.task not in TASK042_ALLOWED_TASKS:
        reasons.append("task_not_task042_allowed")
    if args.task != TASK042_TRUE_TXL_DYNAMIC_TASK_ID:
        reasons.append("task_not_task042_true_txl_dynamic")
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
        raise task039_true_txl_clean_eval.PreflightError(reasons)


def wrap_dynamic_result(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    wrapped = dict(result)
    wrapped["task042_dynamic_eval"] = True
    wrapped["task039_true_txl_clean_only"] = False
    wrapped["policy_label"] = "SequenceAwareTrueTXL"
    wrapped["expected_runner_cls"] = _expected_runner_cls_for_task(args)
    wrapped["expected_actor_model_class"] = args.expected_actor_model_class
    wrapped["expected_action_dim"] = args.expected_action_dim
    wrapped["quality_claim"] = False
    wrapped["training_claim"] = False
    wrapped["eval_claim"] = False
    wrapped["reproduction_claim"] = False
    wrapped["superiority_claim"] = False
    wrapped.update(
        task041_sequence_txl_clean_eval._memory_ablation_summary(
            wrapped.get("txl_debug"),
            args.memory_ablation_mode,
        )
    )
    memory_contract = task039_true_txl_clean_eval.evaluate_memory_debug_contract(wrapped)
    wrapped["memory_debug_field"] = memory_contract["field"]
    wrapped["memory_debug_present"] = memory_contract["present"]
    wrapped["memory_debug_active"] = memory_contract["active"]
    wrapped["true_txl_memory_debug_contract"] = memory_contract
    feedback = evaluate_quality_feedback(wrapped)
    wrapped["quality_gate_pass"] = feedback.quality_gate_pass
    wrapped["quality_feedback"] = feedback.to_json()
    pipeline_pass, pipeline_reasons = evaluate_dynamic_pipeline_pass(wrapped)
    wrapped["pipeline_pass"] = pipeline_pass
    wrapped["pipeline_failure_reasons"] = pipeline_reasons
    quality_reasons = list(feedback.failure_reasons)
    wrapped["quality_failure_reasons"] = quality_reasons
    wrapped["failure_reasons"] = pipeline_reasons + [
        reason for reason in quality_reasons if reason not in pipeline_reasons
    ]
    wrapped["pass"] = pipeline_pass and feedback.quality_gate_pass
    return wrapped


def evaluate_dynamic_pipeline_pass(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary.get("task") != TASK042_TRUE_TXL_DYNAMIC_TASK_ID:
        reasons.append("task_not_task042_true_txl_dynamic")
    if summary.get("runner_cls") != summary.get("expected_runner_cls"):
        reasons.append("runner_cls_mismatch")
    if summary.get("actor_model_class") != summary.get("expected_actor_model_class"):
        reasons.append("actor_model_class_mismatch")
    expected_action_dim = _int_or_none(summary.get("expected_action_dim"))
    if _int_or_none(summary.get("action_dim")) != expected_action_dim:
        reasons.append("action_dim_mismatch")
    if _int_or_none(summary.get("total_action_dim")) != expected_action_dim:
        reasons.append("total_action_dim_mismatch")
    if summary.get("error"):
        reasons.append("error_present")
    if summary.get("traceback"):
        reasons.append("traceback_present")
    memory_contract = task039_true_txl_clean_eval.evaluate_memory_debug_contract(summary)
    reasons.extend(task039_true_txl_clean_eval._memory_debug_failure_reasons(memory_contract))
    if (
        not summary.get("task042_dynamic_eval")
        or summary.get("quality_claim") is not False
        or summary.get("training_claim") is not False
        or summary.get("eval_claim") is not False
        or summary.get("reproduction_claim") is not False
        or summary.get("superiority_claim") is not False
    ):
        reasons.append("claim_boundary_violation")
    return not reasons, reasons


def wrap_task041_result(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    wrapped = dict(result)
    wrapped["task042_memory_ablation_eval"] = True
    wrapped["task042_policy_label"] = "SequenceAwareTrueTXLMemoryAblation"
    wrapped["task042_primary_gate"] = "record_memory_ablation_and_preserve_task041_contract"
    wrapped["memory_ablation_mode"] = args.memory_ablation_mode
    wrapped["ablation_recorded"] = bool(wrapped.get("memory_ablation_mode_match"))
    wrapped["normal_mode_clean_pass_required"] = args.memory_ablation_mode == "none"
    wrapped["memory_causality_claim"] = False
    wrapped["reproduction_claim"] = False
    wrapped["superiority_claim"] = False
    if args.memory_ablation_mode == "none":
        wrapped["task042_pipeline_pass"] = bool(
            wrapped.get("task041_pipeline_pass")
            if wrapped.get("task041_sequence_txl_clean_eval")
            else wrapped.get("pipeline_pass")
        )
        wrapped["task042_pass"] = bool(wrapped["task042_pipeline_pass"] and wrapped.get("pass"))
    else:
        wrapped["task042_pipeline_pass"] = bool(wrapped.get("ablation_recorded"))
        wrapped["task042_pass"] = bool(wrapped["task042_pipeline_pass"])
    return wrapped


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    summary = task041_sequence_txl_clean_eval.build_failure_summary(args, exc)
    if getattr(args, "task", None) == TASK042_TRUE_TXL_DYNAMIC_TASK_ID:
        summary["task042_dynamic_eval"] = True
        summary["task039_true_txl_clean_only"] = False
    summary["task042_memory_ablation_eval"] = True
    summary["task042_pipeline_pass"] = False
    summary["task042_pass"] = False
    summary["memory_causality_claim"] = False
    summary["error"] = repr(exc)
    summary["traceback"] = traceback.format_exc()
    return summary


def _expected_runner_cls_for_task(args: argparse.Namespace) -> str:
    if (
        args.task == TASK042_TRUE_TXL_DYNAMIC_TASK_ID
        and args.expected_runner_cls
        == task039_true_txl_clean_eval.DEFAULT_EXPECTED_RUNNER_CLS
    ):
        return TASK042_DYNAMIC_EXPECTED_RUNNER_CLS
    return args.expected_runner_cls


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def write_json_summary(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary["json_path"] = str(output)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        summary = run_eval(args)
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        summary = build_failure_summary(args, exc)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
