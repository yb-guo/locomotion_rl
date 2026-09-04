import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK041_DIR = ROOT / ".agent" / "task" / "task041-sequence-aware-txl-clean-gait"


def test_task041_sequence_txl_clean_train_parse_args_defaults() -> None:
    module = _load_src_tool("task041_sequence_txl_clean_train.py")

    args = module.parse_args([])

    assert args.task == "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke"
    assert args.num_envs == 4096
    assert args.rollout_steps == 24
    assert args.iterations == 100
    assert args.save_interval == 50
    assert args.num_mini_batches == 4
    assert args.learning_rate is None
    assert args.desired_kl is None
    assert args.actor_trainable_scope == "all"
    assert args.memory_latent_dim == 32
    assert args.memory_latent_scale == 1.0
    assert args.base_obs_passthrough_scale == 1.0
    assert args.adaptation_warmstart_scale == 1.0
    assert args.task044_fault_aux_loss_weight == 0.0
    assert args.task044_fault_aux_num_classes == 0
    assert args.task044_fault_aux_max_trial_step == -1
    assert args.task044_fault_aux_min_trial_index == 0
    assert args.task046_post_reset_recovery_reward is False
    assert args.task046_final_trial_index == 2
    assert args.task046_recovery_window_steps == 50
    assert args.task046_tail_window_steps == 50
    assert args.task046_early_velocity_weight == 0.0
    assert args.task046_tail_velocity_weight == 0.0
    assert args.task046_orientation_weight == 0.0
    assert args.task046_root_height_weight == 0.0
    assert args.task046_min_root_z == 0.70
    assert args.task046_retry_context is False
    assert args.task046_retry_context_num_trials == 3
    assert args.task046_retry_context_final_trial_index == 2
    assert args.task046_retry_context_step_window_steps == 50
    assert args.base_obs_passthrough is True
    assert args.adaptation_warmstart is True
    assert args.expected_algorithm_class == "Task040SequenceAwareTrueTxlPPO"


def test_task041_sequence_txl_clean_train_preflight_rejects_bad_values() -> None:
    module = _load_src_tool("task041_sequence_txl_clean_train.py")
    args = module.parse_args(
        [
            "--task",
            "WrongTask",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "3",
            "--rollout-steps",
            "0",
            "--iterations",
            "0",
            "--save-interval",
            "0",
            "--num-learning-epochs",
            "0",
            "--learning-rate",
            "0",
            "--desired-kl",
            "0",
            "--base-obs-passthrough-scale",
            "-0.1",
            "--adaptation-warmstart-scale",
            "1.1",
            "--expected-action-dim",
            "0",
            "--task044-fault-aux-loss-weight",
            "-1",
            "--task044-fault-aux-num-classes",
            "-1",
            "--task044-fault-aux-max-trial-step",
            "-2",
            "--task044-fault-aux-min-trial-index",
            "-1",
            "--task046-final-trial-index",
            "-1",
            "--task046-recovery-window-steps",
            "0",
            "--task046-tail-window-steps",
            "-1",
            "--task046-early-velocity-weight",
            "-1",
            "--task046-tail-velocity-weight",
            "-1",
            "--task046-orientation-weight",
            "-1",
            "--task046-root-height-weight",
            "-1",
            "--task046-min-root-z",
            "0",
            "--task046-retry-context-num-trials",
            "0",
            "--task046-retry-context-final-trial-index",
            "-1",
            "--task046-retry-context-step-window-steps",
            "0",
            "--resume-checkpoint",
            "missing.pt",
        ]
    )

    try:
        module.preflight_args(args)
    except module.PreflightError as exc:
        assert exc.reasons == [
            "task_not_task041_sequence_txl_clean_train",
            "rollout_steps_not_positive",
            "iterations_not_positive",
            "save_interval_not_positive",
            "num_envs_not_divisible_by_num_mini_batches",
            "num_learning_epochs_not_positive",
            "learning_rate_not_positive",
            "desired_kl_not_positive",
            "base_obs_passthrough_scale_out_of_range",
            "adaptation_warmstart_scale_out_of_range",
            "task044_fault_aux_loss_weight_negative",
            "task044_fault_aux_num_classes_negative",
            "task044_fault_aux_max_trial_step_less_than_negative_one",
            "task044_fault_aux_min_trial_index_negative",
            "task046_final_trial_index_negative",
            "task046_recovery_window_steps_not_positive",
            "task046_tail_window_steps_negative",
            "task046_early_velocity_weight_negative",
            "task046_tail_velocity_weight_negative",
            "task046_orientation_weight_negative",
            "task046_root_height_weight_negative",
            "task046_min_root_z_not_positive",
            "task046_retry_context_num_trials_not_positive",
            "task046_retry_context_final_trial_index_negative",
            "task046_retry_context_step_window_steps_not_positive",
            "expected_action_dim_not_positive",
            "resume_checkpoint_missing",
        ]
    else:
        raise AssertionError("expected preflight rejection")


def test_task041_sequence_txl_clean_train_mutates_cfg_to_sequence_aware_algorithm() -> None:
    module = _load_src_tool("task041_sequence_txl_clean_train.py")
    cfg = {
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
        }
    }

    mutated = module.mutate_agent_cfg_for_task041_train(
        cfg,
        rollout_steps=24,
        iterations=100,
        save_interval=25,
        seed=4100101,
        num_mini_batches=8,
        num_learning_epochs=2,
        learning_rate=1.0e-4,
        desired_kl=0.005,
        experiment_name="exp",
        run_name="run",
        base_obs_passthrough_scale=0.25,
        adaptation_warmstart_scale=0.5,
        task044_fault_aux_loss_weight=0.05,
        task044_fault_aux_num_classes=16,
        task044_fault_aux_max_trial_step=4,
        task044_fault_aux_min_trial_index=1,
        task046_post_reset_recovery_reward=True,
        task046_final_trial_index=2,
        task046_recovery_window_steps=30,
        task046_tail_window_steps=40,
        task046_early_velocity_weight=0.8,
        task046_tail_velocity_weight=0.4,
        task046_orientation_weight=0.3,
        task046_root_height_weight=1.5,
        task046_min_root_z=0.72,
        task046_retry_context=True,
        task046_retry_context_num_trials=4,
        task046_retry_context_final_trial_index=3,
        task046_retry_context_step_window_steps=60,
    )

    assert mutated["num_steps_per_env"] == 24
    assert mutated["max_iterations"] == 100
    assert mutated["save_interval"] == 25
    assert mutated["logger"] == "tensorboard"
    assert mutated["experiment_name"] == "exp"
    assert mutated["run_name"] == "run"
    assert mutated["seed"] == 4100101
    assert mutated["upload_model"] is False
    assert mutated["resume"] is False
    assert mutated["algorithm"]["class_name"] == module.TASK040_ALGORITHM_CLASS
    assert mutated["algorithm"]["num_mini_batches"] == 8
    assert mutated["algorithm"]["num_learning_epochs"] == 2
    assert mutated["algorithm"]["learning_rate"] == 1.0e-4
    assert mutated["algorithm"]["desired_kl"] == 0.005
    assert mutated["algorithm"]["task044_fault_aux_loss_weight"] == 0.05
    assert mutated["algorithm"]["task044_fault_aux_num_classes"] == 16
    assert mutated["algorithm"]["task044_fault_aux_max_trial_step"] == 4
    assert mutated["algorithm"]["task044_fault_aux_min_trial_index"] == 1
    assert mutated["actor"]["memory_latent_dim"] == 32
    assert mutated["actor"]["memory_latent_scale"] == 1.0
    assert mutated["actor"]["base_obs_passthrough_scale"] == 0.25
    assert mutated["actor"]["adaptation_warmstart_scale"] == 0.5
    assert mutated["actor"]["base_obs_passthrough"] is True
    assert mutated["actor"]["adaptation_warmstart"] is True
    assert mutated["task046_post_reset_recovery_reward"] == {
        "enabled": True,
        "final_trial_index": 2,
        "recovery_window_steps": 30,
        "tail_window_steps": 40,
        "early_velocity_weight": 0.8,
        "tail_velocity_weight": 0.4,
        "orientation_weight": 0.3,
        "root_height_weight": 1.5,
        "min_root_z": 0.72,
    }
    assert mutated["task046_retry_context"] == {
        "enabled": True,
        "num_trials": 4,
        "final_trial_index": 3,
        "step_window_steps": 60,
    }


def test_task041_sequence_txl_clean_train_pipeline_pass_gate() -> None:
    module = _load_src_tool("task041_sequence_txl_clean_train.py")

    passed, reasons = module.evaluate_train_pipeline_pass(_passing_train_summary())

    assert passed is True
    assert reasons == []


def test_task041_sequence_txl_clean_train_rejects_fallback_and_missing_sequence_update() -> None:
    module = _load_src_tool("task041_sequence_txl_clean_train.py")
    summary = _passing_train_summary()
    summary["txl_debug"]["stateless_fallback_forward_batches"] = 1
    summary["txl_debug"]["sequence_update_forward_batches"] = 0
    summary["algorithm_debug"]["sequence_update_batches"] = 0

    passed, reasons = module.evaluate_train_pipeline_pass(summary)

    assert passed is False
    assert "txl_debug_stateless_fallback_seen" in reasons
    assert "txl_debug_no_sequence_update_forward" in reasons
    assert "algorithm_debug_no_sequence_update_batches" in reasons


def test_task041_train_pipeline_requires_fault_aux_updates_when_enabled() -> None:
    module = _load_src_tool("task041_sequence_txl_clean_train.py")
    summary = _passing_train_summary()
    summary["task044_fault_aux_loss_weight"] = 0.05
    summary["algorithm_debug"]["task044_fault_aux_updates"] = 0
    summary["algorithm_debug"]["task044_fault_aux_last_loss"] = None

    passed, reasons = module.evaluate_train_pipeline_pass(summary)

    assert passed is False
    assert "task044_fault_aux_no_updates" in reasons
    assert "task044_fault_aux_missing_loss" in reasons

    summary["algorithm_debug"]["task044_fault_aux_updates"] = 5
    summary["algorithm_debug"]["task044_fault_aux_last_loss"] = 1.25

    passed, reasons = module.evaluate_train_pipeline_pass(summary)

    assert passed is True
    assert reasons == []


def test_task041_sequence_txl_parameter_stats_report_group_deltas() -> None:
    torch = pytest.importorskip("torch")
    module = _load_src_tool("task041_sequence_txl_clean_train.py")
    before = {
        "memory_output_projection.weight": torch.zeros(2, 2),
        "attention_layers.0.in_proj_weight": torch.ones(2, 2),
        "token_projection.weight": torch.ones(1, 2),
        "mlp.0.weight": torch.ones(1, 1),
    }
    after = {
        "memory_output_projection.weight": torch.ones(2, 2),
        "attention_layers.0.in_proj_weight": torch.ones(2, 2) * 2,
        "token_projection.weight": torch.ones(1, 2),
        "mlp.0.weight": torch.ones(1, 1) * 3,
    }

    stats = module._txl_parameter_stats(torch, before, after)

    assert stats["tracked_param_count"] == 10
    assert stats["trainable_param_count"] == 11
    assert stats["memory_output_projection_delta_norm"] == pytest.approx(2.0)
    assert stats["attention_layers_delta_norm"] == pytest.approx(2.0)
    assert stats["token_projection_delta_norm"] == pytest.approx(0.0)
    assert stats["norm_layers_delta_norm"] == pytest.approx(0.0)
    assert stats["position_embedding_delta_norm"] == pytest.approx(0.0)


def test_task041_sequence_txl_actor_trainable_scope_can_freeze_warmstart_path() -> None:
    torch = pytest.importorskip("torch")
    module = _load_src_tool("task041_sequence_txl_clean_train.py")

    class DummyActor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.memory_output_projection = torch.nn.Linear(2, 2)
            self.token_projection = torch.nn.Linear(2, 2)
            self.attention_layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.norm_layers = torch.nn.ModuleList([torch.nn.LayerNorm(2)])
            self.adaptation_encoder = torch.nn.Linear(2, 2)
            self.mlp = torch.nn.Linear(2, 2)
            self.update_normalization = lambda obs: "updated"

    actor = DummyActor()

    report = module._set_actor_trainable_scope(actor, "memory_output_projection_only")

    assert report["scope"] == "memory_output_projection_only"
    assert report["normalization_update_disabled"] is True
    assert report["trainable_parameter_count"] == 6
    assert report["trainable_parameter_names"] == [
        "memory_output_projection.weight",
        "memory_output_projection.bias",
    ]
    assert actor.memory_output_projection.weight.requires_grad is True
    assert actor.adaptation_encoder.weight.requires_grad is False
    assert actor.mlp.weight.requires_grad is False
    assert actor.token_projection.weight.requires_grad is False
    assert actor.update_normalization(None) is None


def test_task041_sequence_txl_actor_trainable_scope_txl_residual_includes_attention() -> None:
    torch = pytest.importorskip("torch")
    module = _load_src_tool("task041_sequence_txl_clean_train.py")

    class DummyActor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.position_embedding = torch.nn.Parameter(torch.zeros(2, 2))
            self.memory_output_projection = torch.nn.Linear(2, 2)
            self.token_projection = torch.nn.Linear(2, 2)
            self.attention_layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.norm_layers = torch.nn.ModuleList([torch.nn.LayerNorm(2)])
            self.adaptation_encoder = torch.nn.Linear(2, 2)
            self.mlp = torch.nn.Linear(2, 2)

    actor = DummyActor()

    report = module._set_actor_trainable_scope(actor, "txl_residual_only")

    assert "position_embedding" in report["trainable_parameter_names"]
    assert "attention_layers.0.weight" in report["trainable_parameter_names"]
    assert "norm_layers.0.weight" in report["trainable_parameter_names"]
    assert "adaptation_encoder.weight" in report["frozen_parameter_names"]
    assert "mlp.weight" in report["frozen_parameter_names"]


def test_task041_sequence_txl_actor_trainable_scope_can_train_memory_input_columns() -> None:
    torch = pytest.importorskip("torch")
    module = _load_src_tool("task041_sequence_txl_clean_train.py")

    class DummyActor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.memory_latent_dim = 2
            self.position_embedding = torch.nn.Parameter(torch.zeros(2, 2))
            self.memory_output_projection = torch.nn.Linear(2, 2)
            self.token_projection = torch.nn.Linear(2, 2)
            self.attention_layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.norm_layers = torch.nn.ModuleList([torch.nn.LayerNorm(2)])
            self.adaptation_encoder = torch.nn.Linear(2, 2)
            self.mlp = torch.nn.Sequential(torch.nn.Linear(5, 2))

    actor = DummyActor()

    report = module._set_actor_trainable_scope(actor, "txl_residual_and_mlp_memory_input")

    assert report["scope"] == "txl_residual_and_mlp_memory_input"
    assert "mlp.0.weight" in report["trainable_parameter_names"]
    assert "mlp.0.bias" in report["frozen_parameter_names"]
    assert report["partial_trainable_parameters"] == [
        {
            "name": "mlp.0.weight",
            "trainable_columns": [3, 5],
            "frozen_columns": [0, 3],
            "reason": "memory_latent_input_columns_only",
        }
    ]
    actor.mlp[0].weight.sum().backward()
    assert torch.count_nonzero(actor.mlp[0].weight.grad[:, :3]).item() == 0
    assert torch.count_nonzero(actor.mlp[0].weight.grad[:, 3:]).item() == 4


def test_task041_sequence_txl_actor_scope_stats_detect_frozen_state_changes() -> None:
    torch = pytest.importorskip("torch")
    module = _load_src_tool("task041_sequence_txl_clean_train.py")
    before = {
        "memory_output_projection.weight": torch.zeros(2, 2),
        "mlp.weight": torch.ones(2, 2),
        "obs_normalizer._mean": torch.zeros(4),
    }
    after = {
        "memory_output_projection.weight": torch.ones(2, 2),
        "mlp.weight": torch.ones(2, 2),
        "obs_normalizer._mean": torch.zeros(4),
    }
    scope_report = {
        "trainable_parameter_names": ["memory_output_projection.weight"],
        "frozen_parameter_names": ["mlp.weight"],
    }

    stats = module._actor_scope_parameter_stats(torch, before, after, scope_report)

    assert stats["trainable_parameter_delta_norm"] == pytest.approx(2.0)
    assert stats["frozen_parameter_delta_norm"] == pytest.approx(0.0)
    assert stats["frozen_obs_normalizer_delta_norm"] == pytest.approx(0.0)
    assert stats["frozen_parameters_unchanged"] is True
    assert stats["frozen_obs_normalizer_unchanged"] is True


def test_task041_sequence_txl_actor_scope_stats_tracks_partial_memory_input_columns() -> None:
    torch = pytest.importorskip("torch")
    module = _load_src_tool("task041_sequence_txl_clean_train.py")
    before = {"mlp.0.weight": torch.zeros(2, 5)}
    after = {"mlp.0.weight": torch.zeros(2, 5)}
    after["mlp.0.weight"][:, 3:] = 1.0
    scope_report = {
        "trainable_parameter_names": ["mlp.0.weight"],
        "frozen_parameter_names": [],
        "partial_trainable_parameters": [
            {
                "name": "mlp.0.weight",
                "trainable_columns": [3, 5],
                "frozen_columns": [0, 3],
            }
        ],
    }

    stats = module._actor_scope_parameter_stats(torch, before, after, scope_report)

    assert stats["partial_trainable_delta_norm"] == pytest.approx(2.0)
    assert stats["partial_frozen_delta_norm"] == pytest.approx(0.0)
    assert stats["partial_frozen_columns_unchanged"] is True


def test_task041_sequence_txl_train_pass_gate_rejects_scope_freeze_drift() -> None:
    module = _load_src_tool("task041_sequence_txl_clean_train.py")
    summary = _passing_train_summary()
    summary["actor_trainable_scope"] = "memory_output_projection_only"
    summary["actor_trainable_scope_report"] = {"trainable_parameter_count": 6}
    summary["actor_scope_parameter_stats"] = {
        "frozen_parameter_delta_norm": 0.01,
        "frozen_obs_normalizer_delta_norm": 0.0,
    }

    passed, reasons = module.evaluate_train_pipeline_pass(summary)

    assert passed is False
    assert "actor_trainable_scope_frozen_parameters_changed" in reasons


def test_task041_sequence_txl_train_pass_gate_rejects_partial_frozen_column_drift() -> None:
    module = _load_src_tool("task041_sequence_txl_clean_train.py")
    summary = _passing_train_summary()
    summary["actor_trainable_scope"] = "txl_residual_and_mlp_memory_input"
    summary["actor_trainable_scope_report"] = {"trainable_parameter_count": 6}
    summary["actor_scope_parameter_stats"] = {
        "frozen_parameter_delta_norm": 0.0,
        "frozen_obs_normalizer_delta_norm": 0.0,
        "partial_frozen_delta_norm": 0.01,
    }

    passed, reasons = module.evaluate_train_pipeline_pass(summary)

    assert passed is False
    assert "actor_trainable_scope_partial_frozen_columns_changed" in reasons


def test_task041_sequence_txl_clean_train_main_preflight_failure() -> None:
    module = _load_src_tool("task041_sequence_txl_clean_train.py")
    argv = ["task041_sequence_txl_clean_train.py", "--iterations", "0"]

    with (
        patch.object(sys, "argv", argv),
        patch.object(module, "run_train") as run_train,
        patch.object(module, "write_json_summary") as write_json_summary,
    ):
        module.main()

    run_train.assert_not_called()
    summary = write_json_summary.call_args.args[1]
    assert summary["preflight_rejected"] is True
    assert summary["train_pipeline_pass"] is False
    assert summary["pass"] is False
    assert summary["failure_reasons"] == ["iterations_not_positive"]


def test_task041_docs_define_eval_pass_as_goal() -> None:
    docs = [
        TASK041_DIR / "task.md",
        TASK041_DIR / "001-sequence-aware-clean-train-entrypoint.md",
        TASK041_DIR / "002-clean-eval-gate.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in docs)

    assert "Task041 goal is eval pass" in combined
    assert "Task040SequenceAwareTrueTxlPPO" in combined
    assert "quality_gate_pass=true" in combined
    assert "training_claim:false" in combined
    assert "reproduction_claim:false" in combined


def _passing_train_summary() -> dict:
    return {
        "task": "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke",
        "runner_cls": "Task038TrueTxlMemoryK160Runner",
        "expected_runner_cls": "Task038TrueTxlMemoryK160Runner",
        "algorithm_class": "Task040SequenceAwareTrueTxlPPO",
        "expected_algorithm_class": "Task040SequenceAwareTrueTxlPPO",
        "actor_model_class": "Task038TrueTxlMemoryModel",
        "expected_actor_model_class": "Task038TrueTxlMemoryModel",
        "action_dim": 31,
        "total_action_dim": 31,
        "expected_action_dim": 31,
        "learn_returned": True,
        "checkpoint_exists": True,
        "log_dir_exists": True,
        "policy_action_shape": [8, 31],
        "policy_action_finite": True,
        "task041_sequence_txl_clean_train_only": True,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "txl_debug": {
            "stateless_fallback_forward_batches": 0,
            "sequence_update_forward_batches": 5,
        },
        "algorithm_debug": {
            "sequence_update_batches": 5,
            "last_loss_dict": {"value": 1.0},
        },
    }


def _load_src_tool(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
