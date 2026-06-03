import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TASK039_DIR = ROOT / ".agent" / "task" / "task039-true-txl-quality-training-diagnosis"


def test_task039_mlp_clean_eval_parse_args_defaults() -> None:
    module = _load_src_tool("task039_mlp_clean_eval.py")

    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])

    assert args.task == "Unitree-G1-Gripper-Flat-Task039-MlpClean-Train"
    assert args.checkpoint == "model.pt"
    assert args.output_json == "out.json"
    assert args.num_envs == 64
    assert args.steps == 360
    assert args.trial_length_s == 2.0
    assert args.lin_vel_x == 0.4
    assert args.expected_runner_cls == "Task037BufferOnlyK4DeterministicInnerResetRunner"
    assert args.expected_actor_model_class == "MLPModel"
    assert args.expected_action_dim == 31


def test_task039_mlp_clean_eval_help() -> None:
    module = _load_src_tool("task039_mlp_clean_eval.py")

    try:
        module.parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("expected argparse help to exit")


def test_task039_mlp_clean_eval_allow_list_accepts_only_clean_task() -> None:
    module = _load_src_tool("task039_mlp_clean_eval.py")
    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])

    with patch.object(Path, "exists", return_value=True):
        module.preflight_args(args)

    bad = module.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--output-json",
            "out.json",
            "--task",
            "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke",
        ]
    )
    try:
        with patch.object(Path, "exists", return_value=True):
            module.preflight_args(bad)
    except module.PreflightError as exc:
        assert exc.reasons == ["task_not_task039_mlp_clean"]
    else:
        raise AssertionError("expected unrelated task preflight rejection")


def test_task039_mlp_clean_eval_rejects_missing_checkpoint() -> None:
    module = _load_src_tool("task039_mlp_clean_eval.py")
    args = module.parse_args(["--checkpoint", "missing.pt", "--output-json", "out.json"])

    try:
        module.preflight_args(args)
    except module.PreflightError as exc:
        assert exc.reasons == ["checkpoint_missing"]
    else:
        raise AssertionError("expected missing checkpoint preflight rejection")


def test_task039_mlp_clean_eval_rejects_nonpositive_args() -> None:
    module = _load_src_tool("task039_mlp_clean_eval.py")
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
            "--expected-action-dim",
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
            "expected_action_dim_not_positive",
        ]
    else:
        raise AssertionError("expected nonpositive preflight rejection")


def test_task039_mlp_clean_eval_main_rejects_preflight_before_runtime() -> None:
    module = _load_src_tool("task039_mlp_clean_eval.py")
    argv = [
        "task039_mlp_clean_eval.py",
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
    assert summary["pipeline_pass"] is False
    assert summary["quality_gate_pass"] is False
    assert summary["failure_reasons"] == ["checkpoint_missing"]
    assert summary["quality_claim"] is False
    assert summary["training_claim"] is False
    assert summary["eval_claim"] is False
    assert summary["reproduction_claim"] is False
    assert summary["superiority_claim"] is False


def test_task039_mlp_clean_eval_wraps_quality_fail_separate_from_pipeline() -> None:
    module = _load_src_tool("task039_mlp_clean_eval.py")
    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])
    source = _task037_result(final_trial=_trial(2, fall_ratio=0.5, lin_error=0.8))

    wrapped = module.wrap_task037_result(args, source)

    assert wrapped["pipeline_pass"] is True
    assert wrapped["quality_gate_pass"] is False
    assert wrapped["pass"] is False
    assert wrapped["quality_feedback"]["pipeline_pass"] is True
    assert wrapped["pipeline_failure_reasons"] == []
    assert "final_fall_ratio_too_high" in wrapped["failure_reasons"]
    assert wrapped["task039_mlp_clean_baseline_only"] is True
    assert wrapped["policy_label"] == "MLP"


def test_task039_mlp_clean_eval_metadata_mismatch_fails_pipeline_only() -> None:
    module = _load_src_tool("task039_mlp_clean_eval.py")
    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])
    source = _task037_result(final_trial=_trial(2, fall_ratio=0.0, lin_error=0.2))
    source["runner_cls"] = "WrongRunner"
    source["actor_model_class"] = "WrongActor"
    source["action_dim"] = 30
    source["total_action_dim"] = 30

    wrapped = module.wrap_task037_result(args, source)

    assert wrapped["pipeline_pass"] is False
    assert wrapped["quality_gate_pass"] is True
    assert wrapped["pass"] is False
    assert wrapped["pipeline_failure_reasons"] == [
        "runner_cls_mismatch",
        "actor_model_class_mismatch",
        "action_dim_mismatch",
        "total_action_dim_mismatch",
    ]
    assert wrapped["quality_failure_reasons"] == []


def test_task039_mlp_clean_eval_missing_metadata_and_eval_fields_fail_pipeline() -> None:
    module = _load_src_tool("task039_mlp_clean_eval.py")
    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])
    source = _task037_result(final_trial=_trial(2, fall_ratio=0.0, lin_error=0.2))
    for key in (
        "runner_cls",
        "actor_model_class",
        "action_dim",
        "total_action_dim",
        "trial_0",
        "final_trial",
        "aggregate",
    ):
        source.pop(key)
    source["error"] = "RuntimeError('boom')"
    source["traceback"] = "trace"

    wrapped = module.wrap_task037_result(args, source)

    assert wrapped["pipeline_pass"] is False
    assert wrapped["quality_gate_pass"] is False
    for reason in (
        "runner_cls_mismatch",
        "actor_model_class_mismatch",
        "action_dim_mismatch",
        "total_action_dim_mismatch",
        "trial_0_missing",
        "final_trial_missing",
        "aggregate_missing",
        "error_present",
        "traceback_present",
    ):
        assert reason in wrapped["pipeline_failure_reasons"]
        assert reason in wrapped["failure_reasons"]


def test_task039_mlp_clean_eval_wraps_quality_pass() -> None:
    module = _load_src_tool("task039_mlp_clean_eval.py")
    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])
    source = _task037_result(final_trial=_trial(2, fall_ratio=0.0, lin_error=0.2))

    wrapped = module.wrap_task037_result(args, source)

    assert wrapped["pipeline_pass"] is True
    assert wrapped["quality_gate_pass"] is True
    assert wrapped["pass"] is True
    assert wrapped["failure_reasons"] == []
    assert wrapped["promotion_gate"] == "task039_quality_feedback"
    assert wrapped["diagnostic_note"]
    assert wrapped["quality_claim"] is False
    assert wrapped["training_claim"] is False
    assert wrapped["eval_claim"] is False
    assert wrapped["reproduction_claim"] is False
    assert wrapped["superiority_claim"] is False


def test_task039_mlp_clean_eval_writer_and_structured_failure() -> None:
    module = _load_src_tool("task039_mlp_clean_eval.py")
    output = ROOT / ".agent" / "tmp" / "task039_002_writer_summary.json"
    summary = module.wrap_task037_result(
        module.parse_args(["--checkpoint", "model.pt", "--output-json", str(output)]),
        _task037_result(final_trial=_trial(2, fall_ratio=0.0, lin_error=0.2)),
    )

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
    assert failure["pipeline_pass"] is False
    assert failure["quality_gate_pass"] is False
    assert failure["task039_mlp_clean_baseline_only"] is True
    assert failure["failure_reasons"] == ["eval_wrapper_exception"]


def test_task039_registration_patch_is_idempotent() -> None:
    module = _load_task_script("task039_register_mlp_clean_baseline.py")
    init_path = MemoryPath(
        "from mjlab.rl import MjlabOnPolicyRunner\n"
        "from mjlab.tasks.registry import register_mjlab_task\n"
    )

    module.patch_init(init_path)
    once = init_path.read_text(encoding="utf-8")
    module.patch_init(init_path)
    twice = init_path.read_text(encoding="utf-8")

    assert once == twice
    assert once.count("Unitree-G1-Gripper-Flat-Task039-MlpClean-Train") == 1
    assert once.count("Task037BufferOnlyK4DeterministicInnerResetRunner") == 2
    assert once.count("unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg") == 3
    assert "runner_cls=Task037BufferOnlyK4DeterministicInnerResetRunner" in once


def _load_src_tool(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_task_script(name: str):
    path = TASK039_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task037_result(*, final_trial: dict) -> dict:
    return {
        "task": "Unitree-G1-Gripper-Flat-Task039-MlpClean-Train",
        "checkpoint": "model.pt",
        "seed": 3900201,
        "device": "cuda:0",
        "num_envs": 64,
        "steps": 360,
        "trial_length_s": 2.0,
        "runner_cls": "Task037BufferOnlyK4DeterministicInnerResetRunner",
        "actor_model_class": "MLPModel",
        "action_dim": 31,
        "total_action_dim": 31,
        "trial_0": _trial(0, fall_ratio=0.0, lin_error=0.2),
        "trial_1": _trial(1, fall_ratio=0.0, lin_error=0.2),
        "final_trial": final_trial,
        "aggregate": {
            "trial_count": 3,
            "sample_count": 69120,
            "completion_count": 192,
            "completion_ratio_per_trial_mean": 1.0,
            "fall_count": 0,
            "fall_ratio": 0.0,
            "zero_fall_ratio": 1.0,
            "lin_vel_error_mean": 0.2,
            "yaw_vel_error_mean": 0.1,
            "gravity_xy_mean": 0.2,
            "gravity_xy_max": 0.4,
            "root_z_min": 0.65,
            "num_envs": 64,
        },
        "final_trial_pass": True,
        "pass": True,
    }


def _trial(index: int, *, fall_ratio: float, lin_error: float) -> dict:
    fall_count = int(64 * fall_ratio)
    return {
        "trial_index": index,
        "sample_count": 23040,
        "completion_count": 64,
        "completion_ratio": 1.0,
        "fall_count": fall_count,
        "fall_ratio": fall_ratio,
        "zero_fall_ratio": 1.0 - fall_ratio,
        "timeout_count": 64 - fall_count,
        "reset_reason_counts": {"2": 64 - fall_count, "1": fall_count},
        "reward_mean": 1.0,
        "lin_vel_error": {"mean": lin_error},
        "yaw_vel_error": {"mean": 0.1},
        "gravity_xy": {"mean": 0.2, "max": 0.4},
        "root_z": {"mean": 0.7, "min": 0.65},
    }


class MemoryPath:
    def __init__(self, text: str) -> None:
        self.text = text

    def read_text(self, encoding: str) -> str:
        assert encoding == "utf-8"
        return self.text

    def write_text(self, text: str, encoding: str) -> None:
        assert encoding == "utf-8"
        self.text = text
