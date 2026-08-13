from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASK051_TRAIN_TASK_ID = "Unitree-G1-Gripper-Flat-Task051-TrueTxl-LeftKneeDead-Train"


def test_task051_fault_train_defaults_to_task051_id() -> None:
    module = _load_src_tool("task051_fault_specialized_train.py")

    args = module.parse_args(["--iterations", "1", "--num-envs", "8", "--num-mini-batches", "4"])

    assert args.task == module.TASK051_LEFT_KNEE_FAULT_TASK_ID
    assert args.output_json == module.DEFAULT_OUTPUT_JSON
    assert args.log_dir == module.DEFAULT_LOG_DIR
    assert args.experiment_name == module.DEFAULT_EXPERIMENT_NAME
    assert args.run_name.startswith("task051_left_knee_fault_env")
    assert args.expected_runner_cls == module.TASK051_EXPECTED_RUNNER_CLS
    module.preflight_args(args)


def test_task051_fault_train_rejects_other_tasks() -> None:
    module = _load_src_tool("task051_fault_specialized_train.py")
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
    assert exc_info.value.reasons == ["task_not_task051_left_knee_fault_train"]


def test_task051_registration_uses_hidden_left_knee_fault_train_runner() -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_runner_cls

    env_cfg = load_env_cfg(TASK051_TRAIN_TASK_ID, play=False)
    runner_cls = load_runner_cls(TASK051_TRAIN_TASK_ID)

    assert runner_cls is not None
    assert runner_cls.__name__ == "Task044TrueTxlMemoryK160ClearHistoryRunner"
    assert env_cfg.episode_length_s == 20.0
    assert env_cfg.observations["actor"].enable_corruption is False
    assert "dynamic_motor_failure" in env_cfg.events
    assert "dynamic_motor_failure_reset" in env_cfg.events
    event = env_cfg.events["dynamic_motor_failure"]
    assert event.mode == "step"
    assert event.params["preserve_schedule_across_inner_resets"] is True
    assert event.params["template"] == (
        (0.0, 0.5, None, "normal", 1.0),
        (0.5, 999.0, "left_knee_joint", "dead", 0.0),
        (999.0, None, None, "normal", 1.0),
    )
    assert env_cfg.commands["twist"].ranges.lin_vel_x == (1.6, 1.6)
    assert env_cfg.commands["twist"].ranges.lin_vel_y == (0.0, 0.0)
    assert env_cfg.commands["twist"].ranges.ang_vel_z == (0.0, 0.0)


def test_task051_wrapper_preserves_no_recovery_claim_boundary() -> None:
    module = _load_src_tool("task051_fault_specialized_train.py")

    summary = module.wrap_train_summary({"task": TASK051_TRAIN_TASK_ID, "train_pipeline_pass": True})

    assert summary["task051_fault_specialized_train"] is True
    assert summary["task051_train_pipeline_pass"] is True
    assert summary["fault_recovery_claim"] is False
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
