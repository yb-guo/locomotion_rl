import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK038_DIR = ROOT / ".agent" / "task" / "task038-locoformer-min-g1like-reproduction"


def test_task038_true_txl_ppo_update_parse_args_defaults() -> None:
    module = _load_src_tool("task038_true_txl_ppo_update_smoke.py")

    args = module.parse_args(["--output-json", "out.json"])

    assert args.task == "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke"
    assert args.output_json == "out.json"
    assert args.log_dir == module.DEFAULT_LOG_DIR
    assert args.num_envs == 8
    assert args.rollout_steps == 2
    assert args.iterations == 1
    assert args.device == "cuda:0"
    assert args.expected_action_dim == 31
    assert args.expected_runner_cls == "Task038TrueTxlMemoryK160Runner"
    assert args.expected_actor_model_class == "Task038TrueTxlMemoryModel"


def test_task038_true_txl_ppo_update_mutates_agent_cfg_for_tiny_smoke() -> None:
    module = _load_src_tool("task038_true_txl_ppo_update_smoke.py")
    cfg = {
        "num_steps_per_env": 24,
        "max_iterations": 10001,
        "save_interval": 100,
        "logger": "wandb",
        "algorithm": {"num_learning_epochs": 5, "num_mini_batches": 4},
    }

    mutated = module.mutate_agent_cfg_for_smoke(
        cfg,
        rollout_steps=2,
        iterations=1,
        seed=123,
    )

    assert mutated["num_steps_per_env"] == 2
    assert mutated["max_iterations"] == 1
    assert mutated["save_interval"] == 1000000
    assert mutated["logger"] == "tensorboard"
    assert mutated["upload_model"] is False
    assert mutated["resume"] is False
    assert mutated["algorithm"]["num_learning_epochs"] == 1
    assert mutated["algorithm"]["num_mini_batches"] == 1


def test_task038_true_txl_ppo_update_preflight_rejects_heldout_task() -> None:
    module = _load_src_tool("task038_true_txl_ppo_update_smoke.py")
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


def test_task038_true_txl_ppo_update_preflight_rejects_multi_iteration() -> None:
    module = _load_src_tool("task038_true_txl_ppo_update_smoke.py")
    args = module.parse_args(["--output-json", "out.json", "--iterations", "100"])

    try:
        module.preflight_args(args)
    except module.PreflightError as exc:
        assert exc.reasons == ["iterations_not_one"]
    else:
        raise AssertionError("expected multi-iteration preflight rejection")


def test_task038_true_txl_ppo_update_main_rejects_heldout_before_runtime_path() -> None:
    module = _load_src_tool("task038_true_txl_ppo_update_smoke.py")
    argv = [
        "task038_true_txl_ppo_update_smoke.py",
        "--output-json",
        "out.json",
        "--task",
        "Unitree-G1-Gripper-Flat-Task038-HeldoutTrueTxlRunnerSmoke",
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
    assert summary["failure_reasons"] == ["task_not_train_true_txl_runner_smoke"]


def test_task038_true_txl_ppo_update_main_rejects_multi_iter_before_runtime_path() -> None:
    module = _load_src_tool("task038_true_txl_ppo_update_smoke.py")
    argv = [
        "task038_true_txl_ppo_update_smoke.py",
        "--output-json",
        "out.json",
        "--iterations",
        "100",
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
    assert summary["failure_reasons"] == ["iterations_not_one"]


def test_task038_true_txl_ppo_update_positive_pass_gate() -> None:
    module = _load_src_tool("task038_true_txl_ppo_update_smoke.py")

    passed, reasons = module.evaluate_probe_pass(_passing_summary())

    assert passed is True
    assert reasons == []


def test_task038_true_txl_ppo_update_rejects_learn_not_returned() -> None:
    module = _load_src_tool("task038_true_txl_ppo_update_smoke.py")
    summary = _passing_summary()
    summary["learn_returned"] = False
    summary["txl_debug"] = {}

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "learn_not_returned" in reasons


def test_task038_true_txl_ppo_update_requires_stateless_minibatch_fallback() -> None:
    module = _load_src_tool("task038_true_txl_ppo_update_smoke.py")
    summary = _passing_summary()
    summary["txl_debug"]["stateless_forward_batches"] = 0
    summary["txl_debug"]["stateless_forward_samples"] = 0

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "txl_debug_no_stateless_minibatch_fallback" in reasons


def test_task038_true_txl_ppo_update_policy_error_after_learn_stays_diagnostic() -> None:
    module = _load_src_tool("task038_true_txl_ppo_update_smoke.py")

    diagnostics = module.collect_post_learn_diagnostics(
        torch=_NoGradTorch(),
        runner=_RunnerWithFailingPolicy(),
        rollout_env=_RolloutEnvWithObs(),
        device="cuda:0",
    )
    summary = _passing_summary()
    summary["actor_model_class"] = diagnostics["actor_model_class"]
    summary["txl_debug"] = diagnostics["txl_debug"]
    summary["policy_action_shape"] = diagnostics["policy_action_shape"]
    summary["policy_action_finite"] = diagnostics["policy_action_finite"]
    summary["policy_error"] = diagnostics["policy_error"]
    summary["obs"] = diagnostics["obs"]
    summary["obs_all_finite"] = diagnostics["obs_all_finite"]

    passed, reasons = module.evaluate_probe_pass(summary)

    assert diagnostics["actor_model_class"] == "Task038TrueTxlMemoryModel"
    assert diagnostics["txl_debug"]["stateless_forward_batches"] == 1
    assert diagnostics["policy_action_shape"] is None
    assert "Inplace update to inference tensor outside InferenceMode" in diagnostics["policy_error"]
    assert passed is True
    assert reasons == []


def test_task038_true_txl_ppo_update_rejects_wrong_runner_model_and_action_dims() -> None:
    module = _load_src_tool("task038_true_txl_ppo_update_smoke.py")
    summary = _passing_summary()
    summary["runner_cls"] = "Task037TxlMemoryK160DeterministicRunner"
    summary["actor_model_class"] = "Task037TxlStyleMemoryModel"
    summary["action_dim"] = 30
    summary["total_action_dim"] = 30

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "runner_cls_mismatch" in reasons
    assert "actor_model_class_mismatch" in reasons
    assert "action_dim_mismatch" in reasons
    assert "total_action_dim_mismatch" in reasons


def test_task038_true_txl_ppo_update_rejects_heldout_or_multi_iter_task() -> None:
    module = _load_src_tool("task038_true_txl_ppo_update_smoke.py")
    summary = _passing_summary()
    summary["task"] = "Unitree-G1-Gripper-Flat-Task038-HeldoutTrueTxlRunnerSmoke"
    summary["iterations"] = 2

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "task_not_train_true_txl_runner_smoke" in reasons
    assert "iterations_not_one" in reasons


def test_task038_true_txl_ppo_update_rejects_overclaim_flags() -> None:
    module = _load_src_tool("task038_true_txl_ppo_update_smoke.py")
    summary = _passing_summary()
    summary["quality_claim"] = True
    summary["training_claim"] = True
    summary["eval_claim"] = True
    summary["reproduction_claim"] = True
    summary["superiority_claim"] = True
    summary["ppo_update_smoke_only"] = False

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "claim_boundary_violation" in reasons


def test_task038_true_txl_ppo_update_rejects_missing_log_dir() -> None:
    module = _load_src_tool("task038_true_txl_ppo_update_smoke.py")
    summary = _passing_summary()
    summary["log_dir_exists"] = False

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "log_dir_missing" in reasons


def test_task038_true_txl_ppo_update_writer_and_failure_summary() -> None:
    module = _load_src_tool("task038_true_txl_ppo_update_smoke.py")
    output = ROOT / ".agent" / "tmp" / "task038_013_writer_summary.json"
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
    assert failure["ppo_update_smoke_only"] is True
    assert failure["failure_reasons"] == ["probe_exception"]


def test_task038_true_txl_ppo_update_doc_does_not_overclaim() -> None:
    doc = (TASK038_DIR / "013-true-txl-ppo-update-smoke.md").read_text(
        encoding="utf-8"
    )
    task_md = (TASK038_DIR / "task.md").read_text(encoding="utf-8")
    combined = doc + "\n" + task_md

    forbidden = (
        "true TXL reproduction passed",
        "LocoFormer reproduced",
        "policy quality passed",
        "eval passed",
        "TXL superiority passed",
        "training passed",
        "Status: passed",
        "heldout training passed",
    )
    for phrase in forbidden:
        assert phrase not in combined
    assert "quality_claim:false" in doc
    assert "training_claim:false" in doc
    assert "eval_claim:false" in doc
    assert "reproduction_claim:false" in doc
    assert "superiority_claim:false" in doc
    assert "ppo_update_smoke_only:true" in doc
    assert "train variant only" in doc
    assert "no heldout training" in doc
    assert "no quality/eval/reproduction/superiority claim" in doc


def _load_src_tool(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _NoGradTorch:
    @staticmethod
    def no_grad():
        return _NoGradContext()


class _NoGradContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class Task038TrueTxlMemoryModel:
    def txl_debug_snapshot(self) -> dict:
        return {
            "stateless_forward_batches": 1,
            "stateless_forward_samples": 16,
        }


class _ActorCritic:
    def __init__(self) -> None:
        self.actor = Task038TrueTxlMemoryModel()


class _Alg:
    def __init__(self) -> None:
        self.actor_critic = _ActorCritic()


class _FailingPolicy:
    def eval(self) -> None:
        return None

    def __call__(self, obs):
        raise RuntimeError("Inplace update to inference tensor outside InferenceMode")


class _RunnerWithFailingPolicy:
    def __init__(self) -> None:
        self.alg = _Alg()

    def get_inference_policy(self, device: str):
        assert device == "cuda:0"
        return _FailingPolicy()


class _RolloutEnvWithObs:
    def get_observations(self):
        return {"policy": object()}


def _passing_summary() -> dict:
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
            "stateless_forward_batches": 1,
            "stateless_forward_samples": 16,
        },
        "log_dir_exists": True,
        "wall_time_s": 1.5,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "ppo_update_smoke_only": True,
    }
