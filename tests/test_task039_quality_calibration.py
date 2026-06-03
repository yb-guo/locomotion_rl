import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def test_task039_quality_calibration_parse_args_defaults() -> None:
    module = _load_src_tool("task039_quality_calibration.py")

    args = module.parse_args(["--input-json", "in.json", "--output-json", "out.json"])

    assert args.input_json == "in.json"
    assert args.output_json == "out.json"
    assert args.policy_label == "calibration"
    assert args.checkpoint_label == ""


def test_task039_quality_calibration_help() -> None:
    module = _load_src_tool("task039_quality_calibration.py")

    try:
        module.parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("expected argparse help to exit")


def test_task039_quality_calibration_preflight_rejects_missing_input() -> None:
    module = _load_src_tool("task039_quality_calibration.py")
    args = module.parse_args(["--input-json", "missing.json", "--output-json", "out.json"])

    try:
        module.preflight_args(args)
    except module.PreflightError as exc:
        assert exc.reasons == ["input_json_missing"]
    else:
        raise AssertionError("expected missing input rejection")


def test_task039_quality_calibration_positive_source_passes_gate(tmp_path: Path) -> None:
    module = _load_src_tool("task039_quality_calibration.py")
    input_json = tmp_path / "source.json"
    input_json.write_text(json.dumps(_passing_source()), encoding="utf-8")
    args = module.parse_args(
        [
            "--input-json",
            str(input_json),
            "--output-json",
            str(tmp_path / "out.json"),
            "--policy-label",
            "AdaptK160-positive-calibration",
            "--checkpoint-label",
            "model_5467",
        ]
    )

    summary = module.run_calibration(args)

    assert summary["task039_quality_calibration_only"] is True
    assert summary["policy_label"] == "AdaptK160-positive-calibration"
    assert summary["checkpoint_label"] == "model_5467"
    assert summary["pipeline_pass"] is True
    assert summary["quality_gate_pass"] is True
    assert summary["pass"] is True
    assert summary["failure_reasons"] == []
    assert summary["quality_claim"] is False
    assert summary["training_claim"] is False
    assert summary["eval_claim"] is False
    assert summary["reproduction_claim"] is False
    assert summary["superiority_claim"] is False


def test_task039_quality_calibration_quality_failure_sets_pass_false(tmp_path: Path) -> None:
    module = _load_src_tool("task039_quality_calibration.py")
    source = _passing_source()
    source["final_trial"]["fall_ratio"] = 1.0
    input_json = tmp_path / "source.json"
    input_json.write_text(json.dumps(source), encoding="utf-8")
    args = module.parse_args(["--input-json", str(input_json), "--output-json", "out.json"])

    summary = module.run_calibration(args)

    assert summary["pipeline_pass"] is True
    assert summary["quality_gate_pass"] is False
    assert summary["pass"] is False
    assert "final_fall_ratio_too_high" in summary["failure_reasons"]


def test_task039_quality_calibration_main_writes_structured_preflight_failure() -> None:
    module = _load_src_tool("task039_quality_calibration.py")
    argv = [
        "task039_quality_calibration.py",
        "--input-json",
        "missing.json",
        "--output-json",
        "out.json",
    ]

    with (
        patch.object(sys, "argv", argv),
        patch.object(module, "write_json_summary") as write_json_summary,
    ):
        module.main()

    summary = write_json_summary.call_args.args[1]
    assert summary["preflight_rejected"] is True
    assert summary["pipeline_pass"] is False
    assert summary["quality_gate_pass"] is False
    assert summary["pass"] is False
    assert summary["failure_reasons"] == ["input_json_missing"]


def _load_src_tool(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _passing_source() -> dict:
    return {
        "task": "Unitree-G1-Gripper-Flat-Task037-AdaptK160-CleanUnified-Fast2p0",
        "checkpoint": "model_5467.pt",
        "pipeline_pass": True,
        "pass": True,
        "trial_0": _trial(0),
        "final_trial": _trial(2),
        "aggregate": {
            "completion_ratio_per_trial_mean": 1.0,
            "fall_ratio": 0.0,
            "gravity_xy_max": 0.08,
            "root_z_min": 0.78,
            "lin_vel_error_mean": 0.12,
            "yaw_vel_error_mean": 0.07,
        },
    }


def _trial(index: int) -> dict:
    return {
        "trial_index": index,
        "completion_ratio": 1.0,
        "fall_ratio": 0.0,
        "gravity_xy": {"max": 0.07},
        "root_z": {"min": 0.78},
        "lin_vel_error": {"mean": 0.12},
        "yaw_vel_error": {"mean": 0.07},
    }
