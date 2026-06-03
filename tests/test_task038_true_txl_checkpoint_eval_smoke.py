import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TASK038_DIR = ROOT / ".agent" / "task" / "task038-locoformer-min-g1like-reproduction"


def test_task038_true_txl_checkpoint_eval_parse_args_defaults() -> None:
    module = _load_src_tool("task038_true_txl_checkpoint_eval_smoke.py")

    args = module.parse_args(
        ["--checkpoint", "model.pt", "--output-json", "out.json"]
    )

    assert args.task == "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke"
    assert args.checkpoint == "model.pt"
    assert args.output_json == "out.json"
    assert args.num_envs == 8
    assert args.steps == 10
    assert args.device == "cuda:0"
    assert args.expected_action_dim == 31
    assert args.expected_runner_cls == "Task038TrueTxlMemoryK160Runner"
    assert args.expected_actor_model_class == "Task038TrueTxlMemoryModel"


def test_task038_true_txl_checkpoint_eval_allows_train_and_heldout_tasks() -> None:
    module = _load_src_tool("task038_true_txl_checkpoint_eval_smoke.py")

    for task in module.ALLOWED_TASKS:
        args = module.parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--output-json",
                "out.json",
                "--task",
                task,
            ]
        )
        with patch.object(Path, "exists", return_value=True):
            module.preflight_args(args)


def test_task038_true_txl_checkpoint_eval_rejects_unrelated_task() -> None:
    module = _load_src_tool("task038_true_txl_checkpoint_eval_smoke.py")
    args = module.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--output-json",
            "out.json",
            "--task",
            "Unitree-G1-Gripper-Flat-Task037-AdaptK4",
        ]
    )

    try:
        with patch.object(Path, "exists", return_value=True):
            module.preflight_args(args)
    except module.PreflightError as exc:
        assert exc.reasons == ["task_not_true_txl_runner_smoke"]
    else:
        raise AssertionError("expected unrelated task preflight rejection")


def test_task038_true_txl_checkpoint_eval_rejects_missing_checkpoint() -> None:
    module = _load_src_tool("task038_true_txl_checkpoint_eval_smoke.py")
    args = module.parse_args(
        ["--checkpoint", "missing-model.pt", "--output-json", "out.json"]
    )

    try:
        module.preflight_args(args)
    except module.PreflightError as exc:
        assert exc.reasons == ["checkpoint_missing"]
    else:
        raise AssertionError("expected missing checkpoint preflight rejection")


def test_task038_true_txl_checkpoint_eval_main_rejects_preflight_before_runtime() -> None:
    module = _load_src_tool("task038_true_txl_checkpoint_eval_smoke.py")
    argv = [
        "task038_true_txl_checkpoint_eval_smoke.py",
        "--checkpoint",
        "missing.pt",
        "--output-json",
        "out.json",
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
    assert summary["failure_reasons"] == ["checkpoint_missing"]
    assert summary["checkpoint_eval_load_smoke_only"] is True


def test_task038_true_txl_checkpoint_eval_positive_pass_gate() -> None:
    module = _load_src_tool("task038_true_txl_checkpoint_eval_smoke.py")

    passed, reasons = module.evaluate_probe_pass(_passing_summary())

    assert passed is True
    assert reasons == []


def test_task038_true_txl_checkpoint_eval_snapshot_error_is_optional() -> None:
    module = _load_src_tool("task038_true_txl_checkpoint_eval_smoke.py")

    with patch.object(
        module,
        "_txl_debug_snapshot",
        side_effect=AssertionError("_inner_reset_events is not None"),
    ):
        txl_debug_before, txl_debug_before_error = module._optional_txl_debug_snapshot(
            object()
        )

    summary = _passing_summary()
    summary["txl_debug_before"] = txl_debug_before
    summary["txl_debug_before_error"] = txl_debug_before_error
    summary["txl_debug_after"] = {"envs": []}
    summary["txl_debug_after_error"] = None
    passed, reasons = module.evaluate_probe_pass(summary)

    assert txl_debug_before is None
    assert txl_debug_before_error == (
        "AssertionError('_inner_reset_events is not None')"
    )
    assert passed is True
    assert reasons == []


def test_task038_true_txl_checkpoint_eval_rejects_wrong_task_runner_model_dims() -> None:
    module = _load_src_tool("task038_true_txl_checkpoint_eval_smoke.py")
    summary = _passing_summary()
    summary["task"] = "Unitree-G1-Gripper-Flat-Task037-AdaptK4"
    summary["runner_cls"] = "Task037TxlMemoryK160DeterministicRunner"
    summary["actor_model_class"] = "Task037TxlStyleMemoryModel"
    summary["action_dim"] = 30
    summary["total_action_dim"] = 30

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "task_not_true_txl_runner_smoke" in reasons
    assert "runner_cls_mismatch" in reasons
    assert "actor_model_class_mismatch" in reasons
    assert "action_dim_mismatch" in reasons
    assert "total_action_dim_mismatch" in reasons


def test_task038_true_txl_checkpoint_eval_rejects_load_or_policy_failures() -> None:
    module = _load_src_tool("task038_true_txl_checkpoint_eval_smoke.py")
    summary = _passing_summary()
    summary["checkpoint_exists"] = False
    summary["load_returned"] = False
    summary["policy_error"] = "RuntimeError('boom')"
    summary["step_count"] = 0

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "checkpoint_missing" in reasons
    assert "load_not_returned" in reasons
    assert "policy_error" in reasons
    assert "no_policy_steps_executed" in reasons


def test_task038_true_txl_checkpoint_eval_rejects_nonfinite_or_wrong_shape() -> None:
    module = _load_src_tool("task038_true_txl_checkpoint_eval_smoke.py")
    summary = _passing_summary()
    summary["policy_action_shape"] = [8, 30]
    summary["policy_action_finite"] = False
    summary["obs_all_finite"] = False

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "policy_action_not_finite" in reasons
    assert "policy_action_shape_mismatch" in reasons
    assert "obs_not_finite" in reasons


def test_task038_true_txl_checkpoint_eval_rejects_overclaim_flags() -> None:
    module = _load_src_tool("task038_true_txl_checkpoint_eval_smoke.py")
    summary = _passing_summary()
    summary["quality_claim"] = True
    summary["training_claim"] = True
    summary["eval_claim"] = True
    summary["reproduction_claim"] = True
    summary["superiority_claim"] = True
    summary["checkpoint_eval_load_smoke_only"] = False

    passed, reasons = module.evaluate_probe_pass(summary)

    assert passed is False
    assert "claim_boundary_violation" in reasons


def test_task038_true_txl_checkpoint_eval_writer_and_failure_summary() -> None:
    module = _load_src_tool("task038_true_txl_checkpoint_eval_smoke.py")
    output = ROOT / ".agent" / "tmp" / "task038_015_writer_summary.json"
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
        ["--checkpoint", "missing.pt", "--output-json", str(output)]
    )
    failure = module.build_failure_summary(args, RuntimeError("boom"))
    assert failure["pass"] is False
    assert failure["load_returned"] is False
    assert failure["checkpoint_eval_load_smoke_only"] is True
    assert failure["failure_reasons"] == ["probe_exception"]


def test_task038_true_txl_checkpoint_eval_help() -> None:
    module = _load_src_tool("task038_true_txl_checkpoint_eval_smoke.py")

    try:
        module.parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("expected argparse help to exit")


def test_task038_true_txl_checkpoint_eval_doc_does_not_overclaim() -> None:
    doc = (TASK038_DIR / "015-true-txl-checkpoint-eval-load-smoke.md").read_text(
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
    )
    for phrase in forbidden:
        assert phrase not in combined
    assert "quality_claim:false" in doc
    assert "training_claim:false" in doc
    assert "eval_claim:false" in doc
    assert "reproduction_claim:false" in doc
    assert "superiority_claim:false" in doc
    assert "checkpoint_eval_load_smoke_only:true" in doc
    assert "not quality eval" in doc


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
        "checkpoint_exists": True,
        "runner_cls": "Task038TrueTxlMemoryK160Runner",
        "expected_runner_cls": "Task038TrueTxlMemoryK160Runner",
        "actor_model_class": "Task038TrueTxlMemoryModel",
        "expected_actor_model_class": "Task038TrueTxlMemoryModel",
        "expected_action_dim": 31,
        "actual_num_envs": 8,
        "steps": 10,
        "step_count": 10,
        "action_dim": 31,
        "total_action_dim": 31,
        "load_returned": True,
        "policy_action_shape": [8, 31],
        "policy_action_finite": True,
        "policy_error": None,
        "obs": {"actor_history": {"shape": [8, 16640], "finite": True}},
        "obs_all_finite": True,
        "wall_time_s": 1.5,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "checkpoint_eval_load_smoke_only": True,
    }
