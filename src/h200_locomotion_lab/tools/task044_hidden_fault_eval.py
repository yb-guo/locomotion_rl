"""Evaluate Task044 hidden-fault checkpoints with memory ablations."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from h200_locomotion_lab.tools import task042_memory_ablation_eval as base_eval
from h200_locomotion_lab.tools.task044_memory_required_triplet_summary import (
    HIDDEN_FAULT_CONTRACT,
)


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
TASK044_EXPECTED_RUNNER_CLS = "Task044TrueTxlMemoryK160ClearHistoryRunner"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = base_eval.parse_args(
        argv_list,
        description=(
            "Evaluate Task044 hidden-fault true-TXL checkpoints with memory "
            "ablations. Single eval JSONs do not prove success; use the "
            "Task044 triplet summary gate."
        ),
    )
    if not _flag_present(argv_list, "--task"):
        args.task = TASK044_HIDDEN_FAULT_TASK_ID
    if not _flag_present(argv_list, "--lin-vel-x"):
        args.lin_vel_x = 1.6
    if not _flag_present(argv_list, "--dynamic-dead-joint"):
        args.dynamic_dead_joint = "left_knee_joint"
    if not _flag_present(argv_list, "--dynamic-onset-s"):
        args.dynamic_onset_s = 0.0
    if not _flag_present(argv_list, "--dynamic-recovery-s"):
        args.dynamic_recovery_s = 2.0
    if not _flag_present(argv_list, "--final-window-s"):
        args.final_window_s = 0.5
    return args


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    with _base_task044_context(args.task):
        summary = base_eval.run_eval(args)
    return wrap_eval_summary(summary)


def wrap_eval_summary(summary: dict[str, Any]) -> dict[str, Any]:
    wrapped = dict(summary)
    wrapped["task044_hidden_fault_eval"] = True
    wrapped["task044_eval_task_id"] = wrapped.get("task", TASK044_HIDDEN_FAULT_TASK_ID)
    wrapped["task044_eval_pipeline_pass"] = bool(
        wrapped.get("pipeline_pass") or wrapped.get("ablation_recorded")
    )
    wrapped["task044_hidden_fault_contract"] = dict(HIDDEN_FAULT_CONTRACT)
    wrapped["memory_causality_claim"] = False
    wrapped["hidden_fault_quality_claim"] = False
    wrapped["reproduction_claim"] = False
    wrapped["superiority_claim"] = False
    wrapped["task044_primary_gate"] = "triplet_summary_not_single_eval"
    return wrapped


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    with _base_task044_context(args.task):
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
    except Exception as exc:
        summary = build_failure_summary(args, exc)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


@contextmanager
def _base_task044_context(task_id: str = TASK044_HIDDEN_FAULT_TASK_ID) -> Iterator[None]:
    original_task = base_eval.TASK042_TRUE_TXL_DYNAMIC_TASK_ID
    original_allowed_tasks = base_eval.TASK042_ALLOWED_TASKS
    original_expected_runner = base_eval.TASK042_DYNAMIC_EXPECTED_RUNNER_CLS
    base_eval.TASK042_TRUE_TXL_DYNAMIC_TASK_ID = task_id
    base_eval.TASK042_DYNAMIC_EXPECTED_RUNNER_CLS = TASK044_EXPECTED_RUNNER_CLS
    base_eval.TASK042_ALLOWED_TASKS = (
        base_eval.task039_true_txl_clean_eval.TASK039_TRUE_TXL_CLEAN_TASK_ID,
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
    try:
        yield
    finally:
        base_eval.TASK042_TRUE_TXL_DYNAMIC_TASK_ID = original_task
        base_eval.TASK042_DYNAMIC_EXPECTED_RUNNER_CLS = original_expected_runner
        base_eval.TASK042_ALLOWED_TASKS = original_allowed_tasks


def _flag_present(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


if __name__ == "__main__":
    main()
