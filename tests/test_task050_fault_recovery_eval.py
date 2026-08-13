from __future__ import annotations

import importlib


TASK050_DYNAMIC_EVAL_TASK_ID = "Unitree-G1-Gripper-Flat-Task050-TrueTxl-DynamicFault-Eval"


def test_task050_dynamic_fault_eval_registration_uses_true_txl_runner() -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_runner_cls

    env_cfg = load_env_cfg(TASK050_DYNAMIC_EVAL_TASK_ID, play=True)
    runner_cls = load_runner_cls(TASK050_DYNAMIC_EVAL_TASK_ID)

    assert runner_cls is not None
    assert runner_cls.__name__ == "Task038TrueTxlMemoryK160Runner"
    assert "dynamic_motor_failure" in env_cfg.events
    assert "dynamic_motor_failure_reset" in env_cfg.events
    event = env_cfg.events["dynamic_motor_failure"]
    reset_event = env_cfg.events["dynamic_motor_failure_reset"]
    assert event.mode == "step"
    assert reset_event.mode == "reset"
    assert "actuator_forcerange" in event.func.model_fields
    assert "actuator_forcerange" in reset_event.func.model_fields
    assert event.params["preserve_schedule_across_inner_resets"] is True
    assert "left_knee_joint" in event.params["target_joint_names"]
    assert env_cfg.observations["actor"].enable_corruption is False


def test_task050_dynamic_fault_helper_targets_left_knee_ctrl_id() -> None:
    torch = importlib.import_module("torch")
    env_cfgs = importlib.import_module("src.tasks.velocity.config.g1_gripper.env_cfgs")
    default_forcerange = torch.arange(62, dtype=torch.float32).reshape(31, 2)
    field = default_forcerange.unsqueeze(0).repeat(2, 1, 1).clone()
    active = torch.tensor([False, True])

    env_cfgs._task050_set_forcerange(field, active, 16, default_forcerange[16] * 0.0)

    assert field[0, 16].tolist() == default_forcerange[16].tolist()
    assert field[1, 16].tolist() == [0.0, 0.0]
