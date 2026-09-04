"""Evaluate Task043 dynamic-switch checkpoints with memory ablations."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.tools import task042_memory_ablation_eval as base_eval

TASK043_TRUE_TXL_DYNAMIC_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task043-TrainTrueTxlDynamicSwitchMemoryRequired-Fast1p6"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = base_eval.parse_args(
        argv_list,
        description=(
            "Evaluate Task043 true-TXL dynamic-switch checkpoints with memory "
            "ablations. Quality and memory-causality claims require normal "
            "mode to pass and ablations to degrade."
        ),
    )
    if not _flag_present(argv_list, "--task"):
        args.task = TASK043_TRUE_TXL_DYNAMIC_TASK_ID
    return args


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    with _base_task043_context():
        summary = base_eval.run_eval(args)
    return wrap_eval_summary(summary)


def wrap_eval_summary(summary: dict[str, Any]) -> dict[str, Any]:
    wrapped = dict(summary)
    wrapped["task043_dynamic_ablation_eval"] = True
    wrapped["task043_eval_task_id"] = TASK043_TRUE_TXL_DYNAMIC_TASK_ID
    wrapped["task043_eval_pipeline_pass"] = bool(
        wrapped.get("pipeline_pass") or wrapped.get("ablation_recorded")
    )
    wrapped["memory_causality_claim"] = False
    wrapped["dynamic_switch_quality_claim"] = False
    wrapped["reproduction_claim"] = False
    wrapped["superiority_claim"] = False
    wrapped["task043_primary_gate"] = "normal_quality_then_ablation_degradation"
    return wrapped


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    with _base_task043_context():
        summary = base_eval.build_failure_summary(args, exc)
    summary = wrap_eval_summary(summary)
    summary["error"] = repr(exc)
    summary["traceback"] = traceback.format_exc()
    return summary


def write_json_summary(path: str | Path, summary: dict[str, Any]) -> None:
    base_eval.write_json_summary(path, summary)


def main() -> None:
    args = parse_args()
    try:
        summary = run_eval(args)
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        summary = build_failure_summary(args, exc)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


@contextmanager
def _base_task043_context() -> Iterator[None]:
    original_task = base_eval.TASK042_TRUE_TXL_DYNAMIC_TASK_ID
    original_allowed_tasks = base_eval.TASK042_ALLOWED_TASKS
    base_eval.TASK042_TRUE_TXL_DYNAMIC_TASK_ID = TASK043_TRUE_TXL_DYNAMIC_TASK_ID
    base_eval.TASK042_ALLOWED_TASKS = (
        base_eval.task039_true_txl_clean_eval.TASK039_TRUE_TXL_CLEAN_TASK_ID,
        TASK043_TRUE_TXL_DYNAMIC_TASK_ID,
    )
    try:
        yield
    finally:
        base_eval.TASK042_TRUE_TXL_DYNAMIC_TASK_ID = original_task
        base_eval.TASK042_ALLOWED_TASKS = original_allowed_tasks


def _flag_present(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


if __name__ == "__main__":
    main()
