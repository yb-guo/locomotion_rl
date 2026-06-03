import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_task042_memory_ablation_eval_parse_args_defaults() -> None:
    module = _load_src_tool("task042_memory_ablation_eval.py")

    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])

    assert args.memory_ablation_mode == "none"
    assert args.checkpoint == "model.pt"
    assert args.output_json == "out.json"


def test_task042_memory_ablation_eval_parse_args_accepts_modes() -> None:
    module = _load_src_tool("task042_memory_ablation_eval.py")

    zero = module.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--output-json",
            "out.json",
            "--memory-ablation-mode",
            "zero_txl_residual",
        ]
    )
    stateless = module.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--output-json",
            "out.json",
            "--memory-ablation-mode",
            "stateless_txl_memory",
        ]
    )
    zero_latent = module.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--output-json",
            "out.json",
            "--memory-ablation-mode",
            "zero_memory_latent",
        ]
    )

    assert zero.memory_ablation_mode == "zero_txl_residual"
    assert stateless.memory_ablation_mode == "stateless_txl_memory"
    assert zero_latent.memory_ablation_mode == "zero_memory_latent"


def test_task041_wraps_zero_residual_memory_ablation_fields() -> None:
    module = _load_src_tool("task041_sequence_txl_clean_eval.py")
    args = module.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--output-json",
            "out.json",
            "--memory-ablation-mode",
            "zero_txl_residual",
        ]
    )
    source = _task039_pass_result()
    source["txl_debug"] = {
        "task042_memory_ablation_mode": "zero_txl_residual",
        "memory_residual_enabled": False,
        "stateful_memory_enabled": True,
        "txl_residual_output_norm_last": 0.0,
        "txl_residual_raw_norm_last": 0.25,
        "adaptation_output_norm_last": 0.5,
        "policy_memory_latent_norm_last": 0.5,
    }

    wrapped = module.wrap_task039_result(args, source)

    assert wrapped["memory_ablation_mode"] == "zero_txl_residual"
    assert wrapped["memory_ablation_mode_match"] is True
    assert wrapped["memory_residual_enabled"] is False
    assert wrapped["stateful_memory_enabled"] is True
    assert wrapped["txl_residual_output_norm"] == 0.0
    assert wrapped["txl_residual_raw_norm"] == 0.25
    assert wrapped["task041_pipeline_pass"] is True


def test_task041_wraps_zero_memory_latent_as_full_history_consumer_ablation() -> None:
    module = _load_src_tool("task041_sequence_txl_clean_eval.py")
    args = module.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--output-json",
            "out.json",
            "--memory-ablation-mode",
            "zero_memory_latent",
        ]
    )
    source = _task039_pass_result()
    source["txl_debug"] = {
        "task042_memory_ablation_mode": "zero_memory_latent",
        "memory_residual_enabled": False,
        "memory_latent_enabled": False,
        "stateful_memory_enabled": True,
        "txl_residual_output_norm_last": 0.0,
        "txl_residual_raw_norm_last": 0.25,
        "adaptation_output_norm_last": 0.5,
        "policy_memory_latent_norm_last": 0.0,
    }

    wrapped = module.wrap_task039_result(args, source)

    assert wrapped["memory_ablation_mode"] == "zero_memory_latent"
    assert wrapped["memory_ablation_mode_match"] is True
    assert wrapped["memory_residual_enabled"] is False
    assert wrapped["memory_latent_enabled"] is False
    assert wrapped["stateful_memory_enabled"] is True
    assert wrapped["policy_memory_latent_norm"] == 0.0


def test_task042_ablation_wrapper_records_ablation_without_claiming_causality() -> None:
    module = _load_src_tool("task042_memory_ablation_eval.py")
    args = module.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--output-json",
            "out.json",
            "--memory-ablation-mode",
            "stateless_txl_memory",
        ]
    )
    source = _task041_pass_result()
    source["memory_ablation_mode_match"] = True

    wrapped = module.wrap_task041_result(args, source)

    assert wrapped["task042_memory_ablation_eval"] is True
    assert wrapped["memory_ablation_mode"] == "stateless_txl_memory"
    assert wrapped["ablation_recorded"] is True
    assert wrapped["task042_pipeline_pass"] is True
    assert wrapped["task042_pass"] is True
    assert wrapped["memory_causality_claim"] is False
    assert wrapped["reproduction_claim"] is False
    assert wrapped["superiority_claim"] is False


def test_task042_normal_mode_requires_task041_top_level_pass() -> None:
    module = _load_src_tool("task042_memory_ablation_eval.py")
    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])
    source = _task041_pass_result()
    source["memory_ablation_mode_match"] = True
    source["pass"] = False
    source["quality_gate_pass"] = False

    wrapped = module.wrap_task041_result(args, source)

    assert wrapped["task042_pipeline_pass"] is True
    assert wrapped["task042_pass"] is False


def test_task042_dynamic_task_routes_without_clean_task_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_src_tool("task042_memory_ablation_eval.py")
    args = module.parse_args(
        [
            "--task",
            module.TASK042_TRUE_TXL_DYNAMIC_TASK_ID,
            "--checkpoint",
            "model.pt",
            "--output-json",
            "out.json",
            "--dynamic-dead-joint",
            "right_knee_joint",
        ]
    )

    monkeypatch.setattr(module, "preflight_dynamic_args", lambda _args: None)
    monkeypatch.setattr(module.task041_sequence_txl_clean_eval, "_install_ipython_display_stub", lambda: None)
    monkeypatch.setattr(module.task041_sequence_txl_clean_eval, "_install_wandb_stub", lambda: None)
    monkeypatch.setattr(module.task041_sequence_txl_clean_eval, "_install_wcwidth_stub", lambda: None)
    monkeypatch.setattr(
        module.task037_multitrial_eval_checkpoint,
        "run_eval",
        lambda _args: _task042_dynamic_pass_result(module.TASK042_TRUE_TXL_DYNAMIC_TASK_ID),
    )

    wrapped = module.run_eval(args)

    assert wrapped["task042_dynamic_eval"] is True
    assert wrapped["task042_memory_ablation_eval"] is True
    assert wrapped["task039_true_txl_clean_only"] is False
    assert wrapped["expected_runner_cls"] == "Task038TrueTxlMemoryK160Runner"
    assert wrapped["pipeline_pass"] is True
    assert wrapped["quality_gate_pass"] is True
    assert wrapped["task042_pipeline_pass"] is True
    assert wrapped["task042_pass"] is True
    assert wrapped["memory_debug_active"] is True
    assert wrapped["memory_ablation_mode_match"] is True


def test_task042_dynamic_ablation_records_mode_match(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_src_tool("task042_memory_ablation_eval.py")
    args = module.parse_args(
        [
            "--task",
            module.TASK042_TRUE_TXL_DYNAMIC_TASK_ID,
            "--checkpoint",
            "model.pt",
            "--output-json",
            "out.json",
            "--memory-ablation-mode",
            "zero_txl_residual",
            "--dynamic-dead-joint",
            "right_knee_joint",
        ]
    )

    monkeypatch.setattr(module, "preflight_dynamic_args", lambda _args: None)
    monkeypatch.setattr(module.task041_sequence_txl_clean_eval, "_install_ipython_display_stub", lambda: None)
    monkeypatch.setattr(module.task041_sequence_txl_clean_eval, "_install_wandb_stub", lambda: None)
    monkeypatch.setattr(module.task041_sequence_txl_clean_eval, "_install_wcwidth_stub", lambda: None)
    monkeypatch.setattr(
        module.task037_multitrial_eval_checkpoint,
        "run_eval",
        lambda _args: _task042_dynamic_pass_result(
            module.TASK042_TRUE_TXL_DYNAMIC_TASK_ID,
            ablation_mode="zero_txl_residual",
        ),
    )

    wrapped = module.run_eval(args)

    assert wrapped["memory_ablation_mode"] == "zero_txl_residual"
    assert wrapped["memory_ablation_mode_match"] is True
    assert wrapped["memory_residual_enabled"] is False
    assert wrapped["txl_residual_output_norm"] == 0.0
    assert wrapped["task042_pipeline_pass"] is True
    assert wrapped["task042_pass"] is True


def test_task042_register_true_txl_dynamic_stage_inserts_task_and_runner() -> None:
    module = _load_task042_script("task042_register_true_txl_dynamic_stage.py")
    source = "from mjlab.rl import MjlabOnPolicyRunner\n"

    patched = module._ensure_runner_import(source)

    assert "Task038TrueTxlMemoryK160Runner" in patched
    assert module.TASK_ID == "Unitree-G1-Gripper-Flat-Task042-TrainTrueTxlDynamicMotorFailure-Fast1p6"
    assert module.TASK_ID != "Unitree-G1-Gripper-Flat-Task037-TxlMemoryK160-DynamicMotorFailure-Fast1p6"
    assert "runner_cls=Task038TrueTxlMemoryK160Runner" in module.REGISTER_BLOCK


def _task039_pass_result() -> dict:
    return {
        "pipeline_pass": True,
        "quality_gate_pass": True,
        "pass": True,
        "failure_reasons": [],
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
    }


def _task041_pass_result() -> dict:
    result = _task039_pass_result()
    result["task041_pipeline_pass"] = True
    result["task041_sequence_txl_clean_eval"] = True
    return result


def _task042_dynamic_pass_result(task_id: str, *, ablation_mode: str = "none") -> dict:
    trial = {
        "completion_ratio": 1.0,
        "fall_ratio": 0.0,
        "gravity_xy": {"max": 0.1, "mean": 0.05},
        "root_z": {"min": 0.75, "mean": 0.78},
        "lin_vel_error": {"mean": 0.1},
        "yaw_vel_error": {"mean": 0.1},
        "completion_count": 4,
        "fall_count": 0,
        "sample_count": 16,
    }
    return {
        "task": task_id,
        "runner_cls": "Task038TrueTxlMemoryK160Runner",
        "actor_model_class": "Task038TrueTxlMemoryModel",
        "action_dim": 31,
        "total_action_dim": 31,
        "trial_0": dict(trial),
        "final_trial": dict(trial),
        "aggregate": {
            "completion_ratio_per_trial_mean": 1.0,
            "fall_ratio": 0.0,
            "gravity_xy_max": 0.1,
            "root_z_min": 0.75,
            "lin_vel_error_mean": 0.1,
            "yaw_vel_error_mean": 0.1,
        },
        "pass": True,
        "final_trial_pass": True,
        "txl_debug": {
            "memory_lengths": [64, 64],
            "incremental_steps": 8,
            "last_attended_previous_memory_lengths": [64, 64],
            "task042_memory_ablation_mode": ablation_mode,
            "memory_residual_enabled": ablation_mode
            not in {"zero_txl_residual", "zero_memory_latent"},
            "memory_latent_enabled": ablation_mode != "zero_memory_latent",
            "stateful_memory_enabled": True,
            "txl_residual_output_norm_last": 0.0
            if ablation_mode in {"zero_txl_residual", "zero_memory_latent"}
            else 0.5,
            "txl_residual_raw_norm_last": 0.5,
            "policy_memory_latent_norm_last": 0.0 if ablation_mode == "zero_memory_latent" else 0.5,
        },
    }


def _load_src_tool(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_task042_script(name: str):
    path = ROOT / ".agent" / "task" / "task042-txl-memory-causality-and-residual-training" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
