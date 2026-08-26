import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK038_DIR = ROOT / ".agent" / "task" / "task038-locoformer-min-g1like-reproduction"


def test_task038_true_txl_multitrial_eval_parse_args_defaults() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")

    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])

    assert args.task == "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke"
    assert args.checkpoint == "model.pt"
    assert args.output_json == "out.json"
    assert args.num_envs == 8
    assert args.steps == 60
    assert args.trial_length_s == 0.5
    assert args.lin_vel_x == 0.4
    assert args.device == "cuda:0"
    assert args.expected_action_dim == 31
    assert args.expected_runner_cls == "Task038TrueTxlMemoryK160Runner"
    assert args.expected_actor_model_class == "Task038TrueTxlMemoryModel"


def test_task038_true_txl_multitrial_eval_allows_train_and_heldout_tasks() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")

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


def test_task038_true_txl_multitrial_eval_rejects_unrelated_task() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")
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


def test_task038_true_txl_multitrial_eval_rejects_missing_checkpoint() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")
    args = module.parse_args(["--checkpoint", "missing.pt", "--output-json", "out.json"])

    try:
        module.preflight_args(args)
    except module.PreflightError as exc:
        assert exc.reasons == ["checkpoint_missing"]
    else:
        raise AssertionError("expected missing checkpoint preflight rejection")


def test_task038_true_txl_multitrial_eval_rejects_nonpositive_preflight_args() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")
    args = module.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--output-json",
            "out.json",
            "--num-envs",
            "0",
            "--steps",
            "0",
            "--trial-length-s",
            "0",
        ]
    )

    try:
        with patch.object(Path, "exists", return_value=True):
            module.preflight_args(args)
    except module.PreflightError as exc:
        assert exc.reasons == [
            "num_envs_not_positive",
            "steps_not_positive",
            "trial_length_s_not_positive",
        ]
    else:
        raise AssertionError("expected nonpositive preflight rejection")


def test_task038_true_txl_multitrial_eval_main_rejects_preflight_before_runtime() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")
    argv = [
        "task038_true_txl_multitrial_eval_smoke.py",
        "--checkpoint",
        "missing.pt",
        "--output-json",
        "out.json",
    ]

    with (
        patch.object(sys, "argv", argv),
        patch.object(module, "run_eval") as run_eval,
        patch.object(module, "write_json_summary") as write_json_summary,
    ):
        module.main()

    run_eval.assert_not_called()
    summary = write_json_summary.call_args.args[1]
    assert summary["preflight_rejected"] is True
    assert summary["failure_reasons"] == ["checkpoint_missing"]
    assert summary["eval_pipeline_smoke_only"] is True
    assert summary["quality_metric_final_trial_pass"] is False


def test_task038_true_txl_multitrial_eval_wraps_quality_fail_as_pipeline_pass() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")
    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])
    source = _task037_result(final_trial_pass=False)

    wrapped = module.wrap_task037_result(args, source)

    assert wrapped["quality_metric_final_trial_pass"] is False
    assert wrapped["task037_pass"] is False
    assert wrapped["eval_pipeline_smoke_pass"] is True
    assert wrapped["pipeline_pass"] is True
    assert wrapped["pass"] is True
    assert "final_trial_pass" not in wrapped
    assert wrapped["failure_reasons"] == []


def test_task038_true_txl_multitrial_eval_preserves_quality_pass_as_metric_only() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")
    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])
    source = _task037_result(final_trial_pass=True)

    wrapped = module.wrap_task037_result(args, source)

    assert wrapped["quality_metric_final_trial_pass"] is True
    assert wrapped["task037_pass"] is True
    assert wrapped["pass"] is True
    assert wrapped["promotion_gate"] == "pipeline_smoke_only"
    assert wrapped["quality_claim"] is False
    assert wrapped["eval_claim"] is False


def test_task038_true_txl_multitrial_eval_rejects_missing_trial_fields() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")
    summary = _pipeline_summary()
    summary.pop("trial_0")
    summary.pop("final_trial")

    passed, reasons = module.evaluate_pipeline_pass(summary)

    assert passed is False
    assert "trial_0_missing" in reasons
    assert "final_trial_missing" in reasons


def test_task038_true_txl_multitrial_eval_rejects_empty_trial_dict() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")
    summary = _pipeline_summary()
    summary["trial_0"] = {}

    passed, reasons = module.evaluate_pipeline_pass(summary)

    assert passed is False
    assert "trial_0_metrics_missing" in reasons


def test_task038_true_txl_multitrial_eval_rejects_missing_nested_trial_metric() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")
    summary = _pipeline_summary()
    summary["final_trial"]["lin_vel_error"].pop("mean")

    passed, reasons = module.evaluate_pipeline_pass(summary)

    assert passed is False
    assert "final_trial_metrics_missing_or_nonfinite" in reasons


def test_task038_true_txl_multitrial_eval_rejects_none_required_metric() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")
    summary = _pipeline_summary()
    summary["final_trial"]["root_z"]["min"] = None

    passed, reasons = module.evaluate_pipeline_pass(summary)

    assert passed is False
    assert "final_trial_metrics_missing_or_nonfinite" in reasons
    assert "metrics_not_finite" in reasons


def test_task038_true_txl_multitrial_eval_rejects_bad_reset_reason_counts() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")
    summary = _pipeline_summary()
    summary["trial_0"]["reset_reason_counts"] = None

    passed, reasons = module.evaluate_pipeline_pass(summary)

    assert passed is False
    assert "trial_0_reset_reason_counts_not_dict" in reasons
    assert "metrics_not_finite" in reasons


def test_task038_true_txl_multitrial_eval_rejects_missing_or_empty_aggregate() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")
    missing = _pipeline_summary()
    missing.pop("aggregate")
    empty = _pipeline_summary()
    empty["aggregate"] = {}

    missing_passed, missing_reasons = module.evaluate_pipeline_pass(missing)
    empty_passed, empty_reasons = module.evaluate_pipeline_pass(empty)

    assert missing_passed is False
    assert "aggregate_missing" in missing_reasons
    assert empty_passed is False
    assert "aggregate_metrics_missing" in empty_reasons


def test_task038_true_txl_multitrial_eval_rejects_missing_aggregate_metric() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")
    summary = _pipeline_summary()
    summary["aggregate"].pop("root_z_min")

    passed, reasons = module.evaluate_pipeline_pass(summary)

    assert passed is False
    assert "aggregate_metrics_missing_or_nonfinite" in reasons


def test_task038_true_txl_multitrial_eval_rejects_exception_and_nonfinite_metrics() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")
    summary = _pipeline_summary()
    summary["final_trial"]["root_z"]["min"] = float("nan")
    summary["error"] = "RuntimeError('boom')"

    passed, reasons = module.evaluate_pipeline_pass(summary)

    assert passed is False
    assert "metrics_not_finite" in reasons
    assert "top_level_exception" in reasons


def test_task038_true_txl_multitrial_eval_rejects_overclaim_flags() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")
    summary = _pipeline_summary()
    summary["quality_claim"] = True
    summary["training_claim"] = True
    summary["eval_claim"] = True
    summary["reproduction_claim"] = True
    summary["superiority_claim"] = True
    summary["eval_pipeline_smoke_only"] = False

    passed, reasons = module.evaluate_pipeline_pass(summary)

    assert passed is False
    assert "claim_boundary_violation" in reasons


def test_task038_true_txl_multitrial_eval_writer_and_failure_summary() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")
    output = ROOT / ".agent" / "tmp" / "task038_016_writer_summary.json"
    summary = _pipeline_summary()

    with (
        patch.object(Path, "mkdir") as mkdir,
        patch.object(Path, "write_text", return_value=None) as write_text,
    ):
        module.write_json_summary(output, summary)

    assert summary["json_path"] == str(output.resolve())
    mkdir.assert_called_once()
    write_text.assert_called_once()

    args = module.parse_args(["--checkpoint", "missing.pt", "--output-json", str(output)])
    failure = module.build_failure_summary(args, RuntimeError("boom"))
    assert failure["pass"] is False
    assert failure["pipeline_pass"] is False
    assert failure["eval_pipeline_smoke_only"] is True
    assert failure["quality_metric_final_trial_pass"] is False
    assert failure["failure_reasons"] == ["probe_exception"]


def test_task038_true_txl_multitrial_eval_help() -> None:
    module = _load_src_tool("task038_true_txl_multitrial_eval_smoke.py")

    try:
        module.parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("expected argparse help to exit")


def test_task038_true_txl_multitrial_eval_doc_does_not_overclaim() -> None:
    doc = (TASK038_DIR / "016-true-txl-multitrial-metric-eval-smoke.md").read_text(
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
    assert "eval_pipeline_smoke_only:true" in doc
    assert "quality_metric_final_trial_pass" in doc


def _load_src_tool(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task037_result(*, final_trial_pass: bool) -> dict:
    result = _pipeline_summary()
    result["final_trial_pass"] = final_trial_pass
    result["pass"] = final_trial_pass
    result["promotion_gate"] = "final_trial"
    return result


def _pipeline_summary() -> dict:
    return {
        "task": "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke",
        "checkpoint": "model.pt",
        "seed": 3801601,
        "device": "cuda:0",
        "num_envs": 8,
        "steps": 60,
        "trial_length_s": 0.5,
        "trial_0": _trial(0),
        "trial_1": _trial(1),
        "final_trial": _trial(2),
        "aggregate": {
            "trial_count": 3,
            "sample_count": 1440,
            "completion_count": 24,
            "fall_count": 12,
            "fall_ratio": 0.5,
            "zero_fall_ratio": 0.5,
            "lin_vel_error_mean": 0.7,
            "yaw_vel_error_mean": 0.1,
            "gravity_xy_mean": 0.2,
            "gravity_xy_max": 0.7,
            "root_z_min": 0.45,
            "num_envs": 8,
        },
        "eval_pipeline_smoke_only": True,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
    }


def _trial(index: int) -> dict:
    return {
        "trial_index": index,
        "sample_count": 480,
        "completion_count": 8,
        "completion_ratio": 1.0,
        "fall_count": 4,
        "fall_ratio": 0.5,
        "zero_fall_ratio": 0.5,
        "timeout_count": 4,
        "reset_reason_counts": {"1": 4, "2": 4},
        "reward_mean": 1.0,
        "lin_vel_error": {"mean": 0.7},
        "yaw_vel_error": {"mean": 0.1},
        "gravity_xy": {"mean": 0.2, "max": 0.7},
        "root_z": {"mean": 0.6, "min": 0.45},
    }
