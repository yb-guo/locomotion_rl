from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASK053_TRAIN_TASK_ID = "Unitree-G1-Gripper-Flat-Task053-TrueTxl-SampledCurriculum-Train"


def test_task053_sampled_train_defaults_to_task053_id() -> None:
    module = _load_src_tool("task053_sampled_fault_curriculum_train.py")

    args = module.parse_args(["--iterations", "1", "--num-envs", "8", "--num-mini-batches", "4"])

    assert args.task == module.TASK053_SAMPLED_CURRICULUM_TASK_ID
    assert args.output_json == module.DEFAULT_OUTPUT_JSON
    assert args.log_dir == module.DEFAULT_LOG_DIR
    assert args.experiment_name == module.DEFAULT_EXPERIMENT_NAME
    assert args.run_name.startswith("task053_sampled_curriculum_env")
    assert args.expected_runner_cls == module.TASK053_EXPECTED_RUNNER_CLS
    module.preflight_args(args)


def test_task053_sampled_train_rejects_other_tasks() -> None:
    module = _load_src_tool("task053_sampled_fault_curriculum_train.py")
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
    assert exc_info.value.reasons == ["task_not_task053_sampled_curriculum_train"]


def test_task053_registration_uses_sampled_curriculum_runner_and_events() -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_runner_cls

    env_cfg = load_env_cfg(TASK053_TRAIN_TASK_ID, play=False)
    runner_cls = load_runner_cls(TASK053_TRAIN_TASK_ID)

    assert runner_cls is not None
    assert runner_cls.__name__ == "Task044TrueTxlMemoryK160ClearHistoryRunner"
    assert env_cfg.observations["actor"].enable_corruption is False
    assert "dynamic_motor_failure" in env_cfg.events
    assert "dynamic_motor_failure_reset" in env_cfg.events
    step_event = env_cfg.events["dynamic_motor_failure"]
    reset_event = env_cfg.events["dynamic_motor_failure_reset"]
    assert step_event.mode == "step"
    assert reset_event.mode == "reset"
    assert step_event.params["target_joint_name"] == "left_knee_joint"
    assert reset_event.params["target_joint_name"] == "left_knee_joint"
    assert reset_event.params["curriculum_step_count"] == 4320
    assert reset_event.params["easy_scale_range"] == (0.30, 0.60)
    assert reset_event.params["hard_scale_range"] == (0.0, 0.30)
    assert reset_event.params["easy_onset_range_s"] == (0.90, 1.40)
    assert reset_event.params["hard_onset_range_s"] == (0.50, 0.90)
    assert reset_event.params["hard_vx_probability_range"] == (0.10, 0.70)
    assert reset_event.params["dead_probability_range"] == (0.0, 0.35)
    assert env_cfg.commands["twist"].ranges.lin_vel_x == (1.3, 1.6)


def test_task053_sampled_curriculum_progress_and_reset_sampling() -> None:
    torch = pytest.importorskip("torch")
    env_cfgs = importlib.import_module("src.tasks.velocity.config.g1_gripper.env_cfgs")
    env = _FakeTask053Env(torch)

    env_cfgs._task053_reset_sampled_dynamic_motor_curriculum(
        env,
        torch.tensor([0, 1, 2, 3]),
        curriculum_step_count=100,
        dead_probability_range=(1.0, 1.0),
        hard_vx_probability_range=(1.0, 1.0),
        easy_scale_range=(0.3, 0.3),
        hard_scale_range=(0.0, 0.0),
        easy_onset_range_s=(1.0, 1.0),
        hard_onset_range_s=(0.5, 0.5),
    )

    assert env._task053_sampled_curriculum_progress.tolist() == [0.5, 0.5, 0.5, 0.5]
    assert env._task053_sampled_failure_scale.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert env._task053_sampled_failure_onset_s.tolist() == [0.75, 0.75, 0.75, 0.75]
    assert env._task053_sampled_command_vx.tolist() == pytest.approx([1.6, 1.6, 1.6, 1.6])


def test_task053_apply_overrides_command_and_vectorized_force_range() -> None:
    torch = pytest.importorskip("torch")
    env_cfgs = importlib.import_module("src.tasks.velocity.config.g1_gripper.env_cfgs")
    env = _FakeTask053Env(torch)
    env_cfgs._task053_reset_sampled_dynamic_motor_curriculum(
        env,
        torch.tensor([0, 1, 2, 3]),
        curriculum_step_count=100,
        dead_probability_range=(1.0, 1.0),
        hard_vx_probability_range=(1.0, 1.0),
        easy_scale_range=(0.3, 0.3),
        hard_scale_range=(0.0, 0.0),
        easy_onset_range_s=(0.0, 0.0),
        hard_onset_range_s=(0.0, 0.0),
    )

    env_cfgs._task053_apply_sampled_dynamic_motor_curriculum(env, None)

    assert env.command_manager.get_term("twist").vel_command_b[:, 0].tolist() == pytest.approx([1.6] * 4)
    assert env.sim.model.actuator_forcerange[:, 16, :].tolist() == [[0.0, 0.0]] * 4
    assert env._task030_dynamic_failure_target_index.tolist() == [5, 5, 5, 5]
    assert env._task030_dynamic_failure_scale.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_task053_wrapper_preserves_no_recovery_claim_boundary() -> None:
    module = _load_src_tool("task053_sampled_fault_curriculum_train.py")

    summary = module.wrap_train_summary({"task": TASK053_TRAIN_TASK_ID, "train_pipeline_pass": True})

    assert summary["task053_sampled_fault_curriculum_train"] is True
    assert summary["task053_train_pipeline_pass"] is True
    assert summary["task053_curriculum_mode"] == "sampled_progressive"
    assert summary["fault_recovery_claim"] is False
    assert summary["hard_dead_motor_recovery_claim"] is False
    assert summary["all_joint_fault_claim"] is False
    assert summary["quality_claim"] is False


class _FakeTask053Env:
    def __init__(self, torch_module) -> None:
        self.torch = torch_module
        self.num_envs = 4
        self.device = torch_module.device("cpu")
        self.common_step_counter = 50
        self.step_dt = 0.02
        self.episode_length_buf = torch_module.ones(4, dtype=torch_module.long)
        actuator_names = [f"joint_{idx}" for idx in range(31)]
        actuator_names[16] = "left_knee_joint"
        self.scene = {"robot": SimpleNamespace(actuator_names=actuator_names)}
        self.sim = SimpleNamespace(
            model=SimpleNamespace(
                actuator_forcerange=torch_module.arange(62, dtype=torch_module.float32)
                .reshape(31, 2)
                .unsqueeze(0)
                .repeat(4, 1, 1)
                .clone()
            ),
            get_default_field=lambda _name: torch_module.arange(62, dtype=torch_module.float32).reshape(31, 2),
        )
        self.twist_term = SimpleNamespace(
            vel_command_b=torch_module.zeros((4, 3), dtype=torch_module.float32)
        )
        self.command_manager = SimpleNamespace(
            get_term=lambda _name: self.twist_term
        )


def _load_src_tool(filename: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module
