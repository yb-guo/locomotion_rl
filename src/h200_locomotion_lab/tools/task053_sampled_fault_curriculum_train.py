"""Train Task053 one-shot sampled hidden-fault curriculum."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from h200_locomotion_lab.tools import task041_sequence_txl_clean_train as base_train


TASK053_SAMPLED_CURRICULUM_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task053-TrueTxl-SampledCurriculum-Train"
)
TASK053_EXPECTED_RUNNER_CLS = "Task044TrueTxlMemoryK160ClearHistoryRunner"
DEFAULT_OUTPUT_JSON = Path("outputs/task053/sampled_curriculum/train_summary.json")
DEFAULT_LOG_DIR = Path("outputs/task053/sampled_curriculum/logs")
DEFAULT_EXPERIMENT_NAME = "task053_sampled_fault_curriculum_train"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = base_train.parse_args(
        argv_list,
        description=(
            "Train Task053 one-shot sampled curriculum. Each reset samples hidden "
            "left-knee fault scale/onset/vx from distributions that harden with "
            "common_step_counter. Recovery claims still require Task050 eval."
        ),
    )
    if not _flag_present(argv_list, "--task"):
        args.task = TASK053_SAMPLED_CURRICULUM_TASK_ID
    if not _flag_present(argv_list, "--output-json"):
        args.output_json = DEFAULT_OUTPUT_JSON
    if not _flag_present(argv_list, "--log-dir"):
        args.log_dir = DEFAULT_LOG_DIR
    if not _flag_present(argv_list, "--experiment-name"):
        args.experiment_name = DEFAULT_EXPERIMENT_NAME
    if not _flag_present(argv_list, "--run-name"):
        args.run_name = _default_run_name(args)
    if not _flag_present(argv_list, "--expected-runner-cls"):
        args.expected_runner_cls = TASK053_EXPECTED_RUNNER_CLS
    return args


def preflight_args(args: argparse.Namespace) -> None:
    if args.task != TASK053_SAMPLED_CURRICULUM_TASK_ID:
        raise base_train.PreflightError(["task_not_task053_sampled_curriculum_train"])
    with _base_expected_task(args.task):
        base_train.preflight_args(args)


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    preflight_args(args)
    with _base_expected_task(args.task):
        summary = base_train.run_train(args)
    return wrap_train_summary(summary)


def wrap_train_summary(summary: dict[str, Any]) -> dict[str, Any]:
    wrapped = dict(summary)
    wrapped["task053_sampled_fault_curriculum_train"] = True
    wrapped["task053_train_task_id"] = wrapped.get("task", TASK053_SAMPLED_CURRICULUM_TASK_ID)
    wrapped["task053_curriculum_mode"] = "sampled_progressive"
    wrapped["task053_fault_scope"] = "hidden_left_knee_sampled_scale_onset_vx"
    wrapped["task053_train_pipeline_pass"] = bool(wrapped.get("train_pipeline_pass"))
    wrapped["task053_primary_gate"] = "sampled_train_then_stage_matched_and_hard_task050_eval"
    wrapped["quality_claim"] = False
    wrapped["fault_recovery_claim"] = False
    wrapped["hard_dead_motor_recovery_claim"] = False
    wrapped["all_joint_fault_claim"] = False
    wrapped["diagnostic_note"] = (
        "Task053 train evidence only. The sampled curriculum is accepted only "
        "after Task050 continuous and retry eval pass on held-out hard settings."
    )
    return wrapped


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    with _base_expected_task(TASK053_SAMPLED_CURRICULUM_TASK_ID):
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
    return f"task053_sampled_curriculum_env{args.num_envs}_iter{args.iterations}_seed{args.seed}"


if __name__ == "__main__":
    main()
