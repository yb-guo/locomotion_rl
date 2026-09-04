import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK040_DIR = ROOT / ".agent" / "task" / "task040-sequence-aware-txl-ppo-update"


def test_task040_sequence_txl_ppo_update_parse_args_defaults() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")

    args = module.parse_args(["--output-json", "out.json"])

    assert args.task == "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke"
    assert args.output_json == "out.json"
    assert args.log_dir == module.DEFAULT_LOG_DIR
    assert args.num_envs == 8
    assert args.rollout_steps == 2
    assert args.iterations == 1
    assert args.num_mini_batches == 1
    assert args.device == "cuda:0"
    assert args.expected_action_dim == 31
    assert args.expected_runner_cls == "Task038TrueTxlMemoryK160Runner"
    assert args.expected_actor_model_class == "Task038TrueTxlMemoryModel"
    assert args.expected_algorithm_class == "Task040SequenceAwareTrueTxlPPO"


def test_task040_sequence_txl_ppo_update_mutates_agent_cfg_for_sequence_update() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    cfg = {
        "num_steps_per_env": 24,
        "max_iterations": 10001,
        "save_interval": 100,
        "logger": "wandb",
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
        },
    }

    mutated = module.mutate_agent_cfg_for_sequence_smoke(
        cfg,
        rollout_steps=2,
        iterations=1,
        seed=123,
        num_mini_batches=2,
    )

    assert mutated["num_steps_per_env"] == 2
    assert mutated["max_iterations"] == 1
    assert mutated["save_interval"] == 1000000
    assert mutated["logger"] == "tensorboard"
    assert mutated["upload_model"] is False
    assert mutated["resume"] is False
    assert mutated["algorithm"]["class_name"] == module.TASK040_ALGORITHM_CLASS
    assert mutated["algorithm"]["num_learning_epochs"] == 1
    assert mutated["algorithm"]["num_mini_batches"] == 2


def test_task040_sequence_txl_ppo_update_preflight_rejects_bad_batch_split() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    args = module.parse_args(
        [
            "--output-json",
            "out.json",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "3",
        ]
    )

    try:
        module.preflight_args(args)
    except module.PreflightError as exc:
        assert exc.reasons == ["num_envs_not_divisible_by_num_mini_batches"]
    else:
        raise AssertionError("expected bad batch split preflight rejection")


def test_task040_sequence_txl_ppo_update_installs_ipython_display_stub() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    with patch.dict(sys.modules, {}, clear=False):
        sys.modules.pop("IPython.display", None)

        module._install_ipython_display_stub()

        assert "IPython.display" in sys.modules
        display = sys.modules["IPython.display"]
        assert display.display("ignored") is None
        assert display.HTML("ignored")._repr_html_() == ""


def test_task040_sequence_txl_ppo_update_installs_wandb_stub() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    with patch.dict(sys.modules, {}, clear=False):
        sys.modules.pop("wandb", None)

        module._install_wandb_stub()

        assert "wandb" in sys.modules
        wandb = sys.modules["wandb"]
        assert wandb.run is None
        assert wandb.save("ignored") is None


def test_task040_sequence_txl_ppo_update_installs_wcwidth_stub() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    with patch.dict(sys.modules, {}, clear=False):
        sys.modules.pop("wcwidth", None)

        module._install_wcwidth_stub()

        assert "wcwidth" in sys.modules
        wcwidth = sys.modules["wcwidth"]
        assert wcwidth.wcwidth("x") == 1
        assert wcwidth.wcswidth("abc") == 3


def test_task040_sequence_txl_ppo_update_main_rejects_bad_batch_split() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    argv = [
        "task040_sequence_txl_ppo_update_smoke.py",
        "--output-json",
        "out.json",
        "--num-envs",
        "8",
        "--num-mini-batches",
        "3",
    ]

    with (
        patch.object(sys, "argv", argv),
        patch.object(module, "run_probe") as run_probe,
        patch.object(module, "write_json_summary") as write_json_summary,
    ):
        module.main()

    run_probe.assert_not_called()
    summary = write_json_summary.call_args.args[1]
    assert summary["preflight_rejected"] is True
    assert summary["failure_reasons"] == ["num_envs_not_divisible_by_num_mini_batches"]


def test_task040_sequence_txl_ppo_update_positive_pass_gate() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")

    passed, reasons = module.evaluate_probe_pass(_passing_summary())

    assert passed is True
    assert reasons == []


def test_task040_sequence_txl_ppo_update_rejects_stateless_fallback() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    summary = _passing_summary()
    summary["txl_debug"]["stateless_fallback_forward_batches"] = 1
    summary["txl_debug"]["stateless_fallback_forward_samples"] = 16

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "txl_debug_stateless_fallback_seen" in reasons
    assert "txl_debug_stateless_fallback_samples_seen" in reasons


def test_task040_sequence_txl_ppo_update_requires_sequence_counters_and_loss() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    summary = _passing_summary()
    summary["txl_debug"]["sequence_update_forward_batches"] = 0
    summary["algorithm_debug"]["sequence_update_batches"] = 0
    summary["algorithm_debug"]["last_loss_dict"] = {}

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "txl_debug_no_sequence_update_forward" in reasons
    assert "algorithm_debug_no_sequence_update_batches" in reasons
    assert "algorithm_debug_missing_loss_dict" in reasons


def test_task040_sequence_txl_ppo_update_rejects_overclaim_flags() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    summary = _passing_summary()
    summary["quality_claim"] = True
    summary["training_claim"] = True
    summary["eval_claim"] = True
    summary["reproduction_claim"] = True
    summary["superiority_claim"] = True
    summary["sequence_aware_ppo_update_smoke_only"] = False

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "claim_boundary_violation" in reasons


def test_task040_sequence_txl_ppo_update_writer_and_failure_summary() -> None:
    module = _load_src_tool("task040_sequence_txl_ppo_update_smoke.py")
    output = ROOT / ".agent" / "tmp" / "task040_writer_summary.json"
    summary = _passing_summary()

    with (
        patch.object(Path, "mkdir") as mkdir,
        patch.object(Path, "write_text", return_value=None) as write_text,
    ):
        module.write_json_summary(output, summary)

        assert summary["json_path"] == str(output.resolve())
        mkdir.assert_called_once()
        write_text.assert_called_once()

    args = module.parse_args(
        ["--output-json", str(output), "--log-dir", str(ROOT / ".agent" / "tmp" / "logs")]
    )
    failure = module.build_failure_summary(args, RuntimeError("boom"))
    assert failure["pass"] is False
    assert failure["learn_returned"] is False
    assert failure["sequence_aware_ppo_update_smoke_only"] is True
    assert failure["failure_reasons"] == ["probe_exception"]


def test_task040_docs_define_sequence_update_acceptance_without_overclaim() -> None:
    docs = [
        TASK040_DIR / "task.md",
        TASK040_DIR / "001-rsl-storage-and-update-contract.md",
        TASK040_DIR / "002-sequence-aware-txl-actor-forward.md",
        TASK040_DIR / "003-sequence-aware-ppo-update-smoke.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in docs)

    assert "stateless_fallback_forward_batches == 0" in combined
    assert "sequence_update_forward_batches > 0" in combined
    assert "quality_claim:false" in combined
    assert "training_claim:false" in combined
    assert "eval_claim:false" in combined
    assert "reproduction_claim:false" in combined
    assert "superiority_claim:false" in combined
    assert "Status: passed" not in combined


def _load_src_tool(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _passing_summary() -> dict:
    return {
        "task": "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke",
        "runner_cls": "Task038TrueTxlMemoryK160Runner",
        "expected_runner_cls": "Task038TrueTxlMemoryK160Runner",
        "algorithm_class": "Task040SequenceAwareTrueTxlPPO",
        "expected_algorithm_class": "Task040SequenceAwareTrueTxlPPO",
        "actor_model_class": "Task038TrueTxlMemoryModel",
        "expected_actor_model_class": "Task038TrueTxlMemoryModel",
        "expected_action_dim": 31,
        "actual_num_envs": 8,
        "num_envs": 8,
        "rollout_steps": 2,
        "iterations": 1,
        "num_mini_batches": 1,
        "action_dim": 31,
        "total_action_dim": 31,
        "learn_returned": True,
        "policy_action_shape": [8, 31],
        "policy_action_finite": True,
        "txl_debug": {
            "stateless_fallback_forward_batches": 0,
            "stateless_fallback_forward_samples": 0,
            "sequence_update_forward_batches": 1,
            "sequence_update_forward_samples": 16,
        },
        "algorithm_debug": {
            "sequence_update_batches": 1,
            "sequence_update_samples": 16,
            "sequence_update_steps": 2,
            "last_loss_dict": {"value": 1.0, "surrogate": 0.0, "entropy": 0.5},
        },
        "log_dir_exists": True,
        "wall_time_s": 1.5,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "sequence_aware_ppo_update_smoke_only": True,
    }
