from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASK052_STAGE2_TASK_ID = "Unitree-G1-Gripper-Flat-Task052-TrueTxl-CurriculumStage2-Train"


def test_task052_stage2_train_defaults_to_stage2_id() -> None:
    module = _load_src_tool("task052_fault_curriculum_train.py")

    args = module.parse_args(["--iterations", "1", "--num-envs", "8", "--num-mini-batches", "4"])

    assert args.task == module.TASK052_STAGE2_TASK_ID
    assert args.output_json == module.DEFAULT_OUTPUT_JSON
    assert args.log_dir == module.DEFAULT_LOG_DIR
    assert args.experiment_name == module.DEFAULT_EXPERIMENT_NAME
    assert args.run_name.startswith("task052_stage2_left_knee_scale0p3_env")
    assert args.expected_runner_cls == module.TASK052_EXPECTED_RUNNER_CLS
    module.preflight_args(args)


def test_task052_stage2_train_rejects_other_tasks() -> None:
    module = _load_src_tool("task052_fault_curriculum_train.py")
    args = module.parse_args(
        [
            "--task",
            "WrongTask",
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )

    with pytest.raises(module.base_train.PreflightError) as exc_info:
        module.preflight_args(args)
    assert exc_info.value.reasons == ["task_not_task052_curriculum_stage2_train"]


def test_task052_registration_uses_stage2_partial_left_knee_fault() -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_runner_cls

    env_cfg = load_env_cfg(TASK052_STAGE2_TASK_ID, play=False)
    runner_cls = load_runner_cls(TASK052_STAGE2_TASK_ID)

    assert runner_cls is not None
    assert runner_cls.__name__ == "Task044TrueTxlMemoryK160ClearHistoryRunner"
    assert env_cfg.observations["actor"].enable_corruption is False
    assert "dynamic_motor_failure" in env_cfg.events
    assert "dynamic_motor_failure_reset" in env_cfg.events
    event = env_cfg.events["dynamic_motor_failure"]
    assert event.mode == "step"
    assert event.params["preserve_schedule_across_inner_resets"] is True
    assert event.params["template"] == (
        (0.0, 1.0, None, "normal", 1.0),
        (1.0, 999.0, "left_knee_joint", "dead", 0.3),
        (999.0, None, None, "normal", 1.0),
    )
    assert env_cfg.commands["twist"].ranges.lin_vel_x == (1.3, 1.3)
    assert env_cfg.commands["twist"].ranges.lin_vel_y == (0.0, 0.0)
    assert env_cfg.commands["twist"].ranges.ang_vel_z == (0.0, 0.0)


def test_task052_wrapper_preserves_no_recovery_claim_boundary() -> None:
    module = _load_src_tool("task052_fault_curriculum_train.py")

    summary = module.wrap_train_summary({"task": TASK052_STAGE2_TASK_ID, "train_pipeline_pass": True})

    assert summary["task052_fault_curriculum_train"] is True
    assert summary["task052_train_pipeline_pass"] is True
    assert summary["task052_curriculum_stage"] == module.CURRICULUM_STAGE
    assert summary["fault_recovery_claim"] is False
    assert summary["hard_dead_motor_recovery_claim"] is False
    assert summary["all_joint_fault_claim"] is False
    assert summary["quality_claim"] is False


def _load_src_tool(filename: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module
