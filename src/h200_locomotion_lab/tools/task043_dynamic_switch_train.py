"""Train the Task043 true-TXL policy on the dynamic-switch task."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from h200_locomotion_lab.tools import task041_sequence_txl_clean_train as base_train


TASK043_TRUE_TXL_DYNAMIC_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task043-TrainTrueTxlDynamicSwitchMemoryRequired-Fast1p6"
)
DEFAULT_OUTPUT_JSON = Path("outputs/task043/dynamic_switch_train/train_summary.json")
DEFAULT_LOG_DIR = Path("outputs/task043/dynamic_switch_train/logs")
DEFAULT_EXPERIMENT_NAME = "task043_dynamic_switch_train"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = base_train.parse_args(
        argv_list,
        description=(
            "Train Task043 sequence-aware true-TXL on the dynamic-switch "
            "motor-failure task. This records train pipeline evidence only; "
            "dynamic-switch quality and memory causality must be proven by "
            "normal-vs-ablation eval."
        ),
    )
    if not _flag_present(argv_list, "--task"):
        args.task = TASK043_TRUE_TXL_DYNAMIC_TASK_ID
    if not _flag_present(argv_list, "--output-json"):
        args.output_json = DEFAULT_OUTPUT_JSON
    if not _flag_present(argv_list, "--log-dir"):
        args.log_dir = DEFAULT_LOG_DIR
    if not _flag_present(argv_list, "--experiment-name"):
        args.experiment_name = DEFAULT_EXPERIMENT_NAME
    if not _flag_present(argv_list, "--run-name"):
        args.run_name = _default_run_name(args)
    return args


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    with _base_expected_task(TASK043_TRUE_TXL_DYNAMIC_TASK_ID):
        summary = base_train.run_train(args)
    return wrap_train_summary(summary)


def preflight_args(args: argparse.Namespace) -> None:
    with _base_expected_task(TASK043_TRUE_TXL_DYNAMIC_TASK_ID):
        base_train.preflight_args(args)


def wrap_train_summary(summary: dict[str, Any]) -> dict[str, Any]:
    wrapped = dict(summary)
    wrapped["task043_dynamic_switch_train"] = True
    wrapped["task043_train_task_id"] = TASK043_TRUE_TXL_DYNAMIC_TASK_ID
    wrapped["task043_train_pipeline_pass"] = bool(wrapped.get("train_pipeline_pass"))
    wrapped["task043_primary_gate"] = "train_pipeline_only_then_dynamic_ablation_eval"
    wrapped["memory_causality_claim"] = False
    wrapped["dynamic_switch_quality_claim"] = False
    wrapped["reproduction_claim"] = False
    wrapped["superiority_claim"] = False
    wrapped["diagnostic_note"] = (
        "Task043 train evidence only. Dynamic-switch quality and memory "
        "causality must be proven by normal-vs-ablation eval."
    )
    return wrapped


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    with _base_expected_task(TASK043_TRUE_TXL_DYNAMIC_TASK_ID):
        summary = base_train.build_failure_summary(args, exc)
    summary = wrap_train_summary(summary)
    summary["error"] = repr(exc)
    summary["traceback"] = traceback.format_exc()
    return summary


def write_json_summary(path: str | Path, summary: dict[str, Any]) -> None:
    base_train.write_json_summary(path, summary)


def main() -> None:
    args = parse_args()
    try:
        preflight_args(args)
        summary = run_train(args)
    except base_train.PreflightError as exc:
        summary = build_failure_summary(args, exc)
    except Exception as exc:
        summary = build_failure_summary(args, exc)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


@contextmanager
def _base_expected_task(task_id: str) -> Iterator[None]:
    original_default_task = base_train.DEFAULT_TASK
    base_train.DEFAULT_TASK = task_id
    try:
        yield
    finally:
        base_train.DEFAULT_TASK = original_default_task


def _flag_present(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _default_run_name(args: argparse.Namespace) -> str:
    return f"seq_txl_dynamic_switch_env{args.num_envs}_iter{args.iterations}_seed{args.seed}"


if __name__ == "__main__":
    main()
