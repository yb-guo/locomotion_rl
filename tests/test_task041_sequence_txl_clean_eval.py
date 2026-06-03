import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_task041_sequence_txl_clean_eval_parse_args_defaults() -> None:
    module = _load_src_tool("task041_sequence_txl_clean_eval.py")

    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])

    assert args.task == "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke"
    assert args.checkpoint == "model.pt"
    assert args.output_json == "out.json"
    assert args.num_envs == 64
    assert args.steps == 360
    assert args.seed == 4100201
    assert args.lin_vel_x == 0.4
    assert args.memory_latent_dim == 32
    assert args.memory_latent_scale == 1.0
    assert args.base_obs_passthrough_scale == 1.0
    assert args.adaptation_warmstart_scale == 1.0
    assert args.base_obs_passthrough is True
    assert args.adaptation_warmstart is True
    assert args.expected_actor_model_class == "Task038TrueTxlMemoryModel"
    assert args.train_summary_json is None


def test_task041_sequence_txl_clean_eval_wraps_quality_pass_with_train_summary(tmp_path: Path) -> None:
    module = _load_src_tool("task041_sequence_txl_clean_eval.py")
    checkpoint = tmp_path / "model_99.pt"
    checkpoint.write_bytes(b"model")
    train_summary = {
        "train_pipeline_pass": True,
        "algorithm_class": "Task040SequenceAwareTrueTxlPPO",
        "checkpoint": str(checkpoint),
    }
    train_json = tmp_path / "train.json"
    train_json.write_text(json.dumps(train_summary), encoding="utf-8")
    args = module.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output-json",
            str(tmp_path / "eval.json"),
            "--train-summary-json",
            str(train_json),
        ]
    )

    wrapped = module.wrap_task039_result(args, _task039_pass_result())

    assert wrapped["task041_sequence_txl_clean_eval"] is True
    assert wrapped["policy_label"] == "SequenceAwareTrueTXL"
    assert wrapped["base_obs_passthrough_scale"] == 1.0
    assert wrapped["adaptation_warmstart_scale"] == 1.0
    assert wrapped["sequence_aware_update_train_pipeline_pass"] is True
    assert wrapped["sequence_aware_checkpoint_match"] is True
    assert wrapped["task041_pipeline_pass"] is True
    assert wrapped["quality_gate_pass"] is True
    assert wrapped["pass"] is True


def test_task041_sequence_txl_clean_eval_rejects_train_summary_mismatch(tmp_path: Path) -> None:
    module = _load_src_tool("task041_sequence_txl_clean_eval.py")
    checkpoint = tmp_path / "model_99.pt"
    other_checkpoint = tmp_path / "model_49.pt"
    checkpoint.write_bytes(b"model")
    other_checkpoint.write_bytes(b"model")
    train_json = tmp_path / "train.json"
    train_json.write_text(
        json.dumps(
            {
                "train_pipeline_pass": False,
                "algorithm_class": "WrongPPO",
                "checkpoint": str(other_checkpoint),
            }
        ),
        encoding="utf-8",
    )
    args = module.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output-json",
            str(tmp_path / "eval.json"),
            "--train-summary-json",
            str(train_json),
        ]
    )

    wrapped = module.wrap_task039_result(args, _task039_pass_result())

    assert wrapped["pass"] is False
    assert "sequence_aware_algorithm_class_mismatch" in wrapped["failure_reasons"]
    assert "train_summary_pipeline_not_passed" in wrapped["failure_reasons"]
    assert "train_summary_checkpoint_mismatch" in wrapped["failure_reasons"]


def test_task041_sequence_txl_clean_eval_allows_no_train_summary_for_direct_eval() -> None:
    module = _load_src_tool("task041_sequence_txl_clean_eval.py")
    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])

    wrapped = module.wrap_task039_result(args, _task039_pass_result())

    assert wrapped["task041_pipeline_pass"] is True
    assert wrapped["pass"] is True
    assert wrapped["train_summary"] is None


def test_task041_sequence_txl_clean_eval_propagates_task039_pipeline_failure() -> None:
    module = _load_src_tool("task041_sequence_txl_clean_eval.py")
    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])
    source = _task039_pass_result()
    source["pipeline_pass"] = False
    source["pass"] = False
    source["failure_reasons"] = ["memory_debug_missing"]

    wrapped = module.wrap_task039_result(args, source)

    assert wrapped["task041_pipeline_pass"] is False
    assert wrapped["pass"] is False
    assert "memory_debug_missing" in wrapped["failure_reasons"]
    assert "task039_pipeline_not_passed" in wrapped["failure_reasons"]


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


def _load_src_tool(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
