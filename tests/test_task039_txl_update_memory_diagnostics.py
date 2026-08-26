import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def test_task039_txl_update_memory_diagnostics_parse_args_defaults() -> None:
    module = _load_src_tool("task039_txl_update_memory_diagnostics.py")

    args = module.parse_args(["--output-json", "out.json"])

    assert args.task == "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke"
    assert args.output_json == "out.json"
    assert args.log_dir == module.DEFAULT_LOG_DIR
    assert args.num_envs == 8
    assert args.rollout_steps == 2
    assert args.iterations == 1
    assert args.device == "cuda:0"
    assert args.expected_runner_cls == "Task038TrueTxlMemoryK160Runner"
    assert args.expected_actor_model_class == "Task038TrueTxlMemoryModel"
    assert args.expected_action_dim == 31


def test_task039_txl_update_memory_diagnostics_help() -> None:
    module = _load_src_tool("task039_txl_update_memory_diagnostics.py")

    try:
        module.parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("expected argparse help to exit")


def test_task039_txl_update_memory_diagnostics_preflight_rejects_heldout() -> None:
    module = _load_src_tool("task039_txl_update_memory_diagnostics.py")
    args = module.parse_args(
        [
            "--output-json",
            "out.json",
            "--task",
            "Unitree-G1-Gripper-Flat-Task038-HeldoutTrueTxlRunnerSmoke",
        ]
    )

    try:
        module.preflight_args(args)
    except module.PreflightError as exc:
        assert exc.reasons == ["task_not_train_true_txl_runner_smoke"]
    else:
        raise AssertionError("expected heldout task preflight rejection")


def test_task039_txl_update_memory_diagnostics_preflight_rejects_bad_numbers() -> None:
    module = _load_src_tool("task039_txl_update_memory_diagnostics.py")
    args = module.parse_args(
        [
            "--output-json",
            "out.json",
            "--iterations",
            "2",
            "--num-envs",
            "0",
            "--rollout-steps",
            "0",
            "--expected-action-dim",
            "0",
        ]
    )

    try:
        module.preflight_args(args)
    except module.PreflightError as exc:
        assert exc.reasons == [
            "iterations_not_one",
            "num_envs_not_positive",
            "rollout_steps_not_positive",
            "expected_action_dim_not_positive",
        ]
    else:
        raise AssertionError("expected numeric preflight rejection")


def test_task039_txl_update_memory_diagnostics_passes_with_stateless_fallback() -> None:
    module = _load_src_tool("task039_txl_update_memory_diagnostics.py")
    args = module.parse_args(["--output-json", "out.json"])

    wrapped = module.wrap_task038_result(args, _task038_passing_summary())

    assert wrapped["pass"] is True
    assert wrapped["diagnostic_evidence_valid"] is True
    assert wrapped["failure_reasons"] == []
    assert wrapped["stateless_fallback_present"] is True
    assert wrapped["long_memory_training_claim_supported"] is False
    assert wrapped["router_decision"] == "sequence_aware_txl_ppo_update_required_next"
    assert wrapped["minibatches_preserve_temporal_segments"] is False
    assert "High stateless fallback ratio" in wrapped["interpretation"]


def test_task039_txl_update_memory_diagnostics_computes_ratios_and_memory_summary() -> None:
    module = _load_src_tool("task039_txl_update_memory_diagnostics.py")

    diagnostic = module.build_update_memory_diagnostic_summary(_task038_passing_summary())

    assert diagnostic["total_actor_forward_batches"] == 4
    assert diagnostic["total_actor_forward_samples"] == 40
    assert diagnostic["env_cache_stateful_forward_batches"] == 3
    assert diagnostic["env_cache_stateful_forward_samples"] == 24
    assert diagnostic["stateless_fallback_forward_batches"] == 1
    assert diagnostic["stateless_fallback_forward_samples"] == 16
    assert diagnostic["stateless_fallback_ratio_by_batches"] == 0.25
    assert diagnostic["stateless_fallback_ratio_by_samples"] == 0.4
    assert diagnostic["memory_lengths_observable"] is False
    assert diagnostic["memory_lengths_after_update"] == {
        "value_count": 4,
        "min": 2,
        "max": 5,
        "sum": 14,
        "mean": 3.5,
    }
    assert (
        diagnostic["memory_lengths_observable_reason"]
        == "wrapped_task038_ppo_update_smoke_exposes_post_learn_txl_debug_only"
    )


def test_task039_txl_update_memory_diagnostics_rejects_missing_forward_counters() -> None:
    module = _load_src_tool("task039_txl_update_memory_diagnostics.py")
    args = module.parse_args(["--output-json", "out.json"])
    source = _task038_passing_summary()
    source["txl_debug"] = {
        "stateless_forward_batches": 1,
        "stateless_forward_samples": 16,
        "envs": [{"env_id": 0, "memory_lengths": [2, 3]}],
    }

    wrapped = module.wrap_task038_result(args, source)

    assert wrapped["pass"] is False
    assert "forward_counters_missing" in wrapped["failure_reasons"]
    assert wrapped["forward_counter_missing_fields"] == [
        "total_actor_forward_batches",
        "total_actor_forward_samples",
        "env_cache_stateful_forward_batches",
        "env_cache_stateful_forward_samples",
    ]
    assert wrapped["long_memory_training_claim_supported"] is False


def test_task039_txl_update_memory_diagnostics_forces_no_overclaim_flags_false() -> None:
    module = _load_src_tool("task039_txl_update_memory_diagnostics.py")
    args = module.parse_args(["--output-json", "out.json"])
    source = _task038_passing_summary()
    source["quality_claim"] = True
    source["training_claim"] = True
    source["eval_claim"] = True
    source["reproduction_claim"] = True
    source["superiority_claim"] = True

    wrapped = module.wrap_task038_result(args, source)

    assert wrapped["pass"] is True
    assert wrapped["quality_claim"] is False
    assert wrapped["training_claim"] is False
    assert wrapped["eval_claim"] is False
    assert wrapped["reproduction_claim"] is False
    assert wrapped["superiority_claim"] is False


def test_task039_txl_update_memory_diagnostics_run_probe_delegates_train_variant_only() -> None:
    module = _load_src_tool("task039_txl_update_memory_diagnostics.py")
    args = module.parse_args(["--output-json", "out.json"])

    with patch.object(
        module.task038_true_txl_ppo_update_smoke,
        "run_probe",
        return_value=_task038_passing_summary(),
    ) as delegated:
        wrapped = module.run_probe(args)

    delegated.assert_called_once()
    task038_args = delegated.call_args.args[0]
    assert task038_args.task == "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke"
    assert wrapped["train_variant_only"] is True
    assert wrapped["heldout_variant_included"] is False
    assert wrapped["pass"] is True


def test_task039_txl_update_memory_diagnostics_main_rejects_preflight_before_runtime() -> None:
    module = _load_src_tool("task039_txl_update_memory_diagnostics.py")
    argv = [
        "task039_txl_update_memory_diagnostics.py",
        "--output-json",
        "out.json",
        "--task",
        "Unitree-G1-Gripper-Flat-Task038-HeldoutTrueTxlRunnerSmoke",
    ]

    with (
        patch.object(sys, "argv", argv),
        patch.object(module.task038_true_txl_ppo_update_smoke, "run_probe") as run_probe,
        patch.object(module, "write_json_summary") as write_json_summary,
    ):
        module.main()

    run_probe.assert_not_called()
    summary = write_json_summary.call_args.args[1]
    assert summary["preflight_rejected"] is True
    assert summary["pass"] is False
    assert summary["long_memory_training_claim_supported"] is False
    assert summary["failure_reasons"] == ["task_not_train_true_txl_runner_smoke"]


def _load_src_tool(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task038_passing_summary() -> dict:
    return {
        "task": "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke",
        "runner_cls": "Task038TrueTxlMemoryK160Runner",
        "expected_runner_cls": "Task038TrueTxlMemoryK160Runner",
        "actor_model_class": "Task038TrueTxlMemoryModel",
        "expected_actor_model_class": "Task038TrueTxlMemoryModel",
        "expected_action_dim": 31,
        "actual_num_envs": 8,
        "num_envs": 8,
        "rollout_steps": 2,
        "iterations": 1,
        "action_dim": 31,
        "total_action_dim": 31,
        "learn_returned": True,
        "policy_action_shape": [8, 31],
        "policy_action_finite": True,
        "txl_debug": {
            "total_actor_forward_batches": 4,
            "total_actor_forward_samples": 40,
            "env_cache_stateful_forward_batches": 3,
            "env_cache_stateful_forward_samples": 24,
            "stateless_forward_batches": 1,
            "stateless_forward_samples": 16,
            "envs": [
                {"env_id": 0, "memory_lengths": [2, 3]},
                {"env_id": 1, "memory_lengths": [4, 5]},
            ],
        },
        "log_dir_exists": True,
        "wall_time_s": 1.5,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "ppo_update_smoke_only": True,
        "pass": True,
        "failure_reasons": [],
    }
