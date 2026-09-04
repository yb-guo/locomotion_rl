import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def test_task039_train_metadata_parse_args_defaults() -> None:
    module = _load_src_tool("task039_train_metadata.py")

    args = module.parse_args(_minimal_argv())

    assert args.task == "Task"
    assert args.policy_label == "MLP"
    assert args.runner_cls == "Runner"
    assert args.actor_model_class == "Actor"
    assert args.action_dim == 31
    assert args.train_envs == 4096
    assert args.max_iterations == 30
    assert args.save_interval == 10
    assert args.seed == 3900201
    assert args.device == "cuda:0"
    assert args.command_speed == 0.4


def test_task039_train_metadata_help() -> None:
    module = _load_src_tool("task039_train_metadata.py")

    try:
        module.parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("expected argparse help to exit")


def test_task039_train_metadata_preflight_rejects_missing_paths() -> None:
    module = _load_src_tool("task039_train_metadata.py")
    args = module.parse_args(_minimal_argv())

    try:
        module.preflight_args(args)
    except module.PreflightError as exc:
        assert exc.reasons == [
            "checkpoint_missing",
            "stdout_log_missing",
            "log_dir_missing",
        ]
    else:
        raise AssertionError("expected missing paths to be rejected")


def test_task039_train_metadata_preflight_rejects_bad_numbers() -> None:
    module = _load_src_tool("task039_train_metadata.py")
    args = module.parse_args(
        [
            *_minimal_argv(),
            "--action-dim",
            "0",
            "--train-envs",
            "0",
            "--max-iterations",
            "0",
            "--save-interval",
            "0",
            "--seed",
            "-1",
            "--steps-per-second",
            "0",
            "--wall-time-s",
            "-1",
        ]
    )

    try:
        with patch.object(Path, "exists", return_value=True):
            module.preflight_args(args)
    except module.PreflightError as exc:
        assert exc.reasons == [
            "action_dim_not_positive",
            "train_envs_not_positive",
            "max_iterations_not_positive",
            "save_interval_not_positive",
            "seed_negative",
            "wall_time_s_invalid",
            "steps_per_second_not_positive",
        ]
    else:
        raise AssertionError("expected bad numeric args to be rejected")


def test_task039_train_metadata_build_summary_is_provenance_only() -> None:
    module = _load_src_tool("task039_train_metadata.py")
    args = module.parse_args(_minimal_argv())

    with patch.object(Path, "exists", return_value=True):
        summary = module.build_summary(args)

    assert summary["metadata_pass"] is True
    assert summary["pipeline_pass"] is True
    assert summary["quality_gate_pass"] is False
    assert summary["pass"] is False
    assert summary["task039_train_metadata_only"] is True
    assert summary["quality_claim"] is False
    assert summary["training_claim"] is False
    assert summary["eval_claim"] is False
    assert summary["reproduction_claim"] is False
    assert summary["superiority_claim"] is False


def test_task039_train_metadata_main_writes_structured_preflight_failure() -> None:
    module = _load_src_tool("task039_train_metadata.py")
    argv = ["task039_train_metadata.py", *_minimal_argv()]

    with (
        patch.object(sys, "argv", argv),
        patch.object(module, "write_json_summary") as write_json_summary,
    ):
        module.main()

    summary = write_json_summary.call_args.args[1]
    assert summary["metadata_pass"] is False
    assert summary["pipeline_pass"] is False
    assert summary["quality_gate_pass"] is False
    assert summary["pass"] is False
    assert summary["preflight_rejected"] is True
    assert summary["failure_reasons"] == [
        "checkpoint_missing",
        "stdout_log_missing",
        "log_dir_missing",
    ]


def test_task039_train_metadata_write_json_summary() -> None:
    module = _load_src_tool("task039_train_metadata.py")
    output = ROOT / ".agent" / "tmp" / "task039_train_metadata_writer.json"
    summary = {"metadata_pass": True}

    with (
        patch.object(Path, "mkdir") as mkdir,
        patch.object(Path, "write_text", return_value=None) as write_text,
    ):
        module.write_json_summary(output, summary)

    assert summary["json_path"] == str(output.resolve())
    mkdir.assert_called_once()
    write_text.assert_called_once()


def _minimal_argv() -> list[str]:
    return [
        "--task",
        "Task",
        "--policy-label",
        "MLP",
        "--runner-cls",
        "Runner",
        "--actor-model-class",
        "Actor",
        "--action-dim",
        "31",
        "--train-envs",
        "4096",
        "--max-iterations",
        "30",
        "--save-interval",
        "10",
        "--seed",
        "3900201",
        "--checkpoint",
        "model.pt",
        "--stdout-log",
        "stdout.log",
        "--log-dir",
        "run",
        "--output-json",
        "summary.json",
    ]


def _load_src_tool(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
