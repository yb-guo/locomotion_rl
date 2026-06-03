import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_task043_dynamic_switch_train_defaults_to_task043_id() -> None:
    module = _load_src_tool("task043_dynamic_switch_train.py")

    args = module.parse_args([])

    assert args.task == module.TASK043_TRUE_TXL_DYNAMIC_TASK_ID
    assert args.output_json == module.DEFAULT_OUTPUT_JSON
    assert args.log_dir == module.DEFAULT_LOG_DIR
    assert args.experiment_name == module.DEFAULT_EXPERIMENT_NAME
    assert args.run_name.startswith("seq_txl_dynamic_switch_env")
    assert args.expected_runner_cls == "Task038TrueTxlMemoryK160Runner"
    assert args.expected_actor_model_class == "Task038TrueTxlMemoryModel"
    assert args.expected_algorithm_class == "Task040SequenceAwareTrueTxlPPO"


def test_task043_preflight_uses_task043_as_expected_task() -> None:
    module = _load_src_tool("task043_dynamic_switch_train.py")
    args = module.parse_args(["--iterations", "1", "--num-envs", "8", "--num-mini-batches", "4"])

    module.preflight_args(args)

    wrong = module.parse_args(
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
        module.preflight_args(wrong)
    assert exc_info.value.reasons == ["task_not_task041_sequence_txl_clean_train"]


def test_task043_wrap_train_summary_keeps_no_overclaim_flags() -> None:
    module = _load_src_tool("task043_dynamic_switch_train.py")
    source = {
        "task": module.TASK043_TRUE_TXL_DYNAMIC_TASK_ID,
        "train_pipeline_pass": True,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
    }

    wrapped = module.wrap_train_summary(source)

    assert wrapped["task043_dynamic_switch_train"] is True
    assert wrapped["task043_train_pipeline_pass"] is True
    assert wrapped["memory_causality_claim"] is False
    assert wrapped["dynamic_switch_quality_claim"] is False
    assert wrapped["reproduction_claim"] is False
    assert wrapped["superiority_claim"] is False


def test_task043_register_dynamic_switch_stage_inserts_task_and_runner() -> None:
    module = _load_task043_script("task043_register_dynamic_switch_train_stage.py")
    source = "from mjlab.rl import MjlabOnPolicyRunner\n"

    patched = module._ensure_runner_import(source)

    assert "Task038TrueTxlMemoryK160Runner" in patched
    assert module.TASK_ID == "Unitree-G1-Gripper-Flat-Task043-TrainTrueTxlDynamicSwitchMemoryRequired-Fast1p6"
    assert "runner_cls=Task038TrueTxlMemoryK160Runner" in module.REGISTER_BLOCK
    assert "_task043_dynamic_failure_fixed1p6_env_cfg()" in module.REGISTER_BLOCK
    assert "twist_cmd.ranges.lin_vel_x = (1.6, 1.6)" in module.FIXED_SPEED_HELPER_BLOCK


def test_task043_register_patch_replaces_existing_task_with_fixed_speed_helper(tmp_path) -> None:
    module = _load_task043_script("task043_register_dynamic_switch_train_stage.py")
    init_path = tmp_path / "__init__.py"
    init_path.write_text(
        "\n".join(
            [
                "from mjlab.rl import MjlabOnPolicyRunner",
                "",
                "register_mjlab_task(",
                f'  task_id="{module.TASK_ID}",',
                "  env_cfg=unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg(),",
                "  play_env_cfg=unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg(play=True),",
                "  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),",
                "  runner_cls=Task038TrueTxlMemoryK160Runner,",
                ")",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    module.patch_init(init_path)
    patched = init_path.read_text(encoding="utf-8")

    assert patched.count(module.TASK_ID) == 1
    assert f"def {module.FIXED_SPEED_HELPER_NAME}" in patched
    assert "twist_cmd.ranges.lin_vel_x = (1.6, 1.6)" in patched
    assert "env_cfg=_task043_dynamic_failure_fixed1p6_env_cfg()" in patched
    assert "play_env_cfg=_task043_dynamic_failure_fixed1p6_env_cfg(play=True)" in patched
    assert "env_cfg=unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg()" not in patched


def test_task043_dynamic_ablation_eval_defaults_to_task043_id() -> None:
    module = _load_src_tool("task043_dynamic_ablation_eval.py")

    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])

    assert args.task == module.TASK043_TRUE_TXL_DYNAMIC_TASK_ID
    assert args.memory_ablation_mode == "none"


def test_task043_dynamic_ablation_eval_wraps_without_memory_claim() -> None:
    module = _load_src_tool("task043_dynamic_ablation_eval.py")
    source = {
        "task": module.TASK043_TRUE_TXL_DYNAMIC_TASK_ID,
        "pipeline_pass": True,
        "quality_gate_pass": True,
        "pass": True,
        "memory_ablation_mode": "none",
    }

    wrapped = module.wrap_eval_summary(source)

    assert wrapped["task043_dynamic_ablation_eval"] is True
    assert wrapped["task043_eval_pipeline_pass"] is True
    assert wrapped["memory_causality_claim"] is False
    assert wrapped["dynamic_switch_quality_claim"] is False
    assert wrapped["reproduction_claim"] is False
    assert wrapped["superiority_claim"] is False


def _load_src_tool(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_task043_script(name: str):
    path = ROOT / ".agent" / "task" / "task043-memory-required-dynamic-switch-training" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
