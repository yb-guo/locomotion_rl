"""Train Task052 fixed-stage hidden-fault curriculum."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from h200_locomotion_lab.tools import task041_sequence_txl_clean_train as base_train


TASK052_STAGE2_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task052-TrueTxl-CurriculumStage2-Train"
)
TASK052_EXPECTED_RUNNER_CLS = "Task044TrueTxlMemoryK160ClearHistoryRunner"
DEFAULT_OUTPUT_JSON = Path("outputs/task052/curriculum_stage2/train_summary.json")
DEFAULT_LOG_DIR = Path("outputs/task052/curriculum_stage2/logs")
DEFAULT_EXPERIMENT_NAME = "task052_fault_curriculum_train"
CURRICULUM_STAGE = "stage2_left_knee_scale0p3_vx1p3_onset1p0"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = base_train.parse_args(
        argv_list,
        description=(
            "Train Task052 fixed-stage curriculum Stage2. This uses a hidden "
            "left-knee partial motor failure at vx=1.3 m/s, onset=1.0 s, "
            "and torque scale=0.3. Recovery claims still require Task050-style eval."
        ),
    )
    if not _flag_present(argv_list, "--task"):
        args.task = TASK052_STAGE2_TASK_ID
    if not _flag_present(argv_list, "--output-json"):
        args.output_json = DEFAULT_OUTPUT_JSON
    if not _flag_present(argv_list, "--log-dir"):
        args.log_dir = DEFAULT_LOG_DIR
    if not _flag_present(argv_list, "--experiment-name"):
        args.experiment_name = DEFAULT_EXPERIMENT_NAME
    if not _flag_present(argv_list, "--run-name"):
        args.run_name = _default_run_name(args)
    if not _flag_present(argv_list, "--expected-runner-cls"):
        args.expected_runner_cls = TASK052_EXPECTED_RUNNER_CLS
    return args


def preflight_args(args: argparse.Namespace) -> None:
    if args.task != TASK052_STAGE2_TASK_ID:
        raise base_train.PreflightError(["task_not_task052_curriculum_stage2_train"])
    with _base_expected_task(args.task):
        base_train.preflight_args(args)


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    preflight_args(args)
    with _base_expected_task(args.task):
        summary = base_train.run_train(args)
    return wrap_train_summary(summary)


def wrap_train_summary(summary: dict[str, Any]) -> dict[str, Any]:
    wrapped = dict(summary)
    wrapped["task052_fault_curriculum_train"] = True
    wrapped["task052_train_task_id"] = wrapped.get("task", TASK052_STAGE2_TASK_ID)
    wrapped["task052_curriculum_mode"] = "fixed_stage"
    wrapped["task052_curriculum_stage"] = CURRICULUM_STAGE
    wrapped["task052_fault_scope"] = "hidden_left_knee_partial_motor_scale0p3"
    wrapped["task052_train_pipeline_pass"] = bool(wrapped.get("train_pipeline_pass"))
    wrapped["task052_primary_gate"] = "stage_matched_eval_then_task050_hard_gate"
    wrapped["quality_claim"] = False
    wrapped["fault_recovery_claim"] = False
    wrapped["hard_dead_motor_recovery_claim"] = False
    wrapped["all_joint_fault_claim"] = False
    wrapped["diagnostic_note"] = (
        "Task052 Stage2 train evidence only. Stage-matched partial-fault eval "
        "and hard Task050 dead-motor eval must be recorded separately."
    )
    return wrapped


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    with _base_expected_task(TASK052_STAGE2_TASK_ID):
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
    return (
        f"task052_stage2_left_knee_scale0p3_env{args.num_envs}"
        f"_iter{args.iterations}_seed{args.seed}"
    )


if __name__ == "__main__":
    main()
