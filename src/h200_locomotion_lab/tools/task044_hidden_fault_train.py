"""Train Task044 true-TXL on hidden dynamic-fault schedules."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from h200_locomotion_lab.tools import task041_sequence_txl_clean_train as base_train


TASK044_HIDDEN_FAULT_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-Fast1p6"
)
TASK044_EVAL_ALIGNED_LEFT_KNEE_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-EvalLeftKnee1p6"
)
TASK044_EVAL_ALIGNED_LEFT_KNEE_VELBOOST_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-EvalLeftKneeVelBoost1p6"
)
TASK044_PERSISTENT_HIDDEN_VELBOOST_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenVelBoost1p6"
)
TASK044_PERSISTENT_IMMEDIATE_VELBOOST_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateVelBoost1p6"
)
TASK044_PERSISTENT_IMMEDIATE_LEFT_KNEE_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneeDeadVelBoost1p6"
)
TASK044_PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneeDeadSpeedPush1p6"
)
TASK044_PERSISTENT_HIDDEN_SPEED_STABILITY_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenSpeedStability1p6"
)
TASK044_PERSISTENT_HIDDEN_HEIGHT_GUARD_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenHeightGuard1p6"
)
TASK044_PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenHeightGuardStrong1p6"
)
TASK044_PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenLowRootTerminate1p6"
)
TASK044_PERSISTENT_HIDDEN_POSE_TERMINATE_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenPoseTerminate1p6"
)
TASK044_PERSISTENT_HIDDEN_POSE_TIGHT_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenPoseTight1p6"
)
TASK044_PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenSpeedPoseBalance1p6"
)
TASK044_PERSISTENT_HIDDEN_FORWARD_FLOOR_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenForwardFloor1p6"
)
TASK044_PERSISTENT_HIDDEN_FORWARD_TARGET_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenForwardTarget1p6"
)
TASK044_PERSISTENT_HIDDEN_SPEED_CURRICULUM_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenSpeedCurriculum1p4To1p6"
)
TASK044_PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneePoseForward1p6"
)
TASK045_PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task045-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneePoseForwardSurvival1p6"
)
TASK045_PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task045-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneeLongSurvival1p6"
)
TASK045_POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task045-TrueTxlHiddenFaultTrain-PoseTightGateLeftKneeLongTail1p6"
)
TASK044_PERSISTENT_HIDDEN_STARTUP_BOOST_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenStartupBoost1p6"
)
TASK044_ALLOWED_TASK_IDS = (
    TASK044_HIDDEN_FAULT_TASK_ID,
    TASK044_EVAL_ALIGNED_LEFT_KNEE_TASK_ID,
    TASK044_EVAL_ALIGNED_LEFT_KNEE_VELBOOST_TASK_ID,
    TASK044_PERSISTENT_HIDDEN_VELBOOST_TASK_ID,
    TASK044_PERSISTENT_IMMEDIATE_VELBOOST_TASK_ID,
    TASK044_PERSISTENT_IMMEDIATE_LEFT_KNEE_TASK_ID,
    TASK044_PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_TASK_ID,
    TASK044_PERSISTENT_HIDDEN_SPEED_STABILITY_TASK_ID,
    TASK044_PERSISTENT_HIDDEN_HEIGHT_GUARD_TASK_ID,
    TASK044_PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_TASK_ID,
    TASK044_PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_TASK_ID,
    TASK044_PERSISTENT_HIDDEN_POSE_TERMINATE_TASK_ID,
    TASK044_PERSISTENT_HIDDEN_POSE_TIGHT_TASK_ID,
    TASK044_PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_TASK_ID,
    TASK044_PERSISTENT_HIDDEN_FORWARD_FLOOR_TASK_ID,
    TASK044_PERSISTENT_HIDDEN_FORWARD_TARGET_TASK_ID,
    TASK044_PERSISTENT_HIDDEN_SPEED_CURRICULUM_TASK_ID,
    TASK044_PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_TASK_ID,
    TASK045_PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_TASK_ID,
    TASK045_PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_TASK_ID,
    TASK045_POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_TASK_ID,
    TASK044_PERSISTENT_HIDDEN_STARTUP_BOOST_TASK_ID,
)
TASK044_EXPECTED_RUNNER_CLS = "Task044TrueTxlMemoryK160ClearHistoryRunner"
DEFAULT_OUTPUT_JSON = Path("outputs/task044/hidden_fault_train/train_summary.json")
DEFAULT_LOG_DIR = Path("outputs/task044/hidden_fault_train/logs")
DEFAULT_EXPERIMENT_NAME = "task044_hidden_fault_train"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = base_train.parse_args(
        argv_list,
        description=(
            "Train Task044 sequence-aware true-TXL on hidden randomized "
            "dynamic-fault schedules. This records train-pipeline evidence "
            "only; memory-required behavior must be proven by triplet eval."
        ),
    )
    if not _flag_present(argv_list, "--task"):
        args.task = TASK044_HIDDEN_FAULT_TASK_ID
    if not _flag_present(argv_list, "--output-json"):
        args.output_json = DEFAULT_OUTPUT_JSON
    if not _flag_present(argv_list, "--log-dir"):
        args.log_dir = DEFAULT_LOG_DIR
    if not _flag_present(argv_list, "--experiment-name"):
        args.experiment_name = DEFAULT_EXPERIMENT_NAME
    if not _flag_present(argv_list, "--run-name"):
        args.run_name = _default_run_name(args)
    if not _flag_present(argv_list, "--expected-runner-cls"):
        args.expected_runner_cls = TASK044_EXPECTED_RUNNER_CLS
    return args


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    with _base_expected_task(args.task):
        summary = base_train.run_train(args)
    return wrap_train_summary(summary)


def preflight_args(args: argparse.Namespace) -> None:
    if args.task not in TASK044_ALLOWED_TASK_IDS:
        raise base_train.PreflightError(["task_not_task044_hidden_fault_allowed"])
    with _base_expected_task(args.task):
        base_train.preflight_args(args)


def wrap_train_summary(summary: dict[str, Any]) -> dict[str, Any]:
    wrapped = dict(summary)
    wrapped["task044_hidden_fault_train"] = True
    wrapped["task044_train_task_id"] = wrapped.get("task", TASK044_HIDDEN_FAULT_TASK_ID)
    wrapped["task044_train_pipeline_pass"] = bool(wrapped.get("train_pipeline_pass"))
    wrapped["task044_primary_gate"] = "train_pipeline_only_then_triplet_eval"
    wrapped["memory_causality_claim"] = False
    wrapped["dynamic_switch_quality_claim"] = False
    wrapped["reproduction_claim"] = False
    wrapped["superiority_claim"] = False
    wrapped["diagnostic_note"] = (
        "Task044 train evidence only. Memory-required behavior must be proven "
        "by normal, zero-residual, and stateless triplet eval."
    )
    return wrapped


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    with _base_expected_task(TASK044_HIDDEN_FAULT_TASK_ID):
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
    return f"seq_txl_hidden_fault_env{args.num_envs}_iter{args.iterations}_seed{args.seed}"


if __name__ == "__main__":
    main()
