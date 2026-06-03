import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_task044_triplet_summary_parse_args_requires_sources() -> None:
    module = _load_tool()

    args = module.parse_args(
        [
            "--normal-json",
            "normal.json",
            "--zero-residual-json",
            "zero.json",
            "--stateless-json",
            "stateless.json",
            "--output-json",
            "summary.json",
        ]
    )

    assert args.normal_json == "normal.json"
    assert args.zero_residual_json == "zero.json"
    assert args.stateless_json == "stateless.json"
    assert args.output_json == "summary.json"
    assert args.confirm_hidden_fault_labels is False
    assert args.metric_scope == "final_trial"


def test_task044_triplet_summary_records_paths_and_no_overclaim_fields(tmp_path: Path) -> None:
    module = _load_tool()
    normal = _write_json(tmp_path / "normal.json", _summary("none", lin_vel_error=0.20))
    zero = _write_json(tmp_path / "zero.json", _summary("zero_txl_residual", lin_vel_error=0.26))
    stateless = _write_json(
        tmp_path / "stateless.json",
        _summary("stateless_txl_memory", lin_vel_error=0.25, pipeline_pass=False),
    )
    output = tmp_path / "summary.json"
    args = module.parse_args(
        [
            "--normal-json",
            str(normal),
            "--zero-residual-json",
            str(zero),
            "--stateless-json",
            str(stateless),
            "--confirm-hidden-fault-labels",
            "--output-json",
            str(output),
        ]
    )

    summary = module.build_triplet_summary(args)
    module.write_json_summary(output, summary)
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written["task044_triplet_summary"] is True
    assert written["task044_memory_required_pass"] is True
    assert written["normal_json"] == str(normal.resolve())
    assert written["zero_residual_json"] == str(zero.resolve())
    assert written["stateless_json"] == str(stateless.resolve())
    assert written["quality_claim"] is False
    assert written["memory_causality_claim"] is False
    assert written["reproduction_claim"] is False
    assert written["superiority_claim"] is False


def test_task044_triplet_summary_without_hidden_confirmation_fails_missing_metadata(
    tmp_path: Path,
) -> None:
    module = _load_tool()
    normal = _write_json(tmp_path / "normal.json", _summary("none", lin_vel_error=0.20))
    zero = _write_json(tmp_path / "zero.json", _summary("zero_txl_residual", lin_vel_error=0.26))
    stateless = _write_json(
        tmp_path / "stateless.json",
        _summary("stateless_txl_memory", lin_vel_error=0.25),
    )
    args = module.parse_args(
        [
            "--normal-json",
            str(normal),
            "--zero-residual-json",
            str(zero),
            "--stateless-json",
            str(stateless),
            "--output-json",
            str(tmp_path / "summary.json"),
        ]
    )

    summary = module.build_triplet_summary(args)

    assert summary["task044_memory_required_pass"] is False
    assert "fault_identity_in_actor_obs_not_false" in summary["failure_reasons"]


def test_task044_triplet_summary_can_use_window_metric_scope(tmp_path: Path) -> None:
    module = _load_tool()
    normal = _write_json(
        tmp_path / "normal.json",
        _summary("none", lin_vel_error=0.20, window_lin_vel_error=0.20),
    )
    zero = _write_json(
        tmp_path / "zero.json",
        _summary("zero_txl_residual", lin_vel_error=0.20, window_lin_vel_error=0.25),
    )
    stateless = _write_json(
        tmp_path / "stateless.json",
        _summary(
            "stateless_txl_memory",
            lin_vel_error=0.20,
            window_lin_vel_error=0.24,
            pipeline_pass=False,
        ),
    )
    args = module.parse_args(
        [
            "--normal-json",
            str(normal),
            "--zero-residual-json",
            str(zero),
            "--stateless-json",
            str(stateless),
            "--confirm-hidden-fault-labels",
            "--metric-scope",
            "final_trial_window",
            "--output-json",
            str(tmp_path / "summary.json"),
        ]
    )

    summary = module.build_triplet_summary(args)

    assert summary["task044_memory_required_pass"] is True
    assert summary["task044_contract"]["thresholds"]["metric_scope"] == "final_trial_window"
    assert summary["task044_contract"]["zero_residual_ablation"]["degradation"]["deltas"][
        "lin_vel_error_delta"
    ] == 0.04999999999999999


def test_task044_triplet_summary_can_use_tail_window_metric_scope(tmp_path: Path) -> None:
    module = _load_tool()
    normal = _write_json(
        tmp_path / "normal.json",
        _summary("none", lin_vel_error=0.20, tail_lin_vel_error=0.18),
    )
    zero = _write_json(
        tmp_path / "zero.json",
        _summary("zero_txl_residual", lin_vel_error=0.20, tail_lin_vel_error=0.23),
    )
    stateless = _write_json(
        tmp_path / "stateless.json",
        _summary(
            "stateless_txl_memory",
            lin_vel_error=0.20,
            tail_lin_vel_error=0.22,
            pipeline_pass=False,
        ),
    )
    args = module.parse_args(
        [
            "--normal-json",
            str(normal),
            "--zero-residual-json",
            str(zero),
            "--stateless-json",
            str(stateless),
            "--confirm-hidden-fault-labels",
            "--metric-scope",
            "final_trial_tail_window",
            "--output-json",
            str(tmp_path / "summary.json"),
        ]
    )

    summary = module.build_triplet_summary(args)

    assert summary["task044_memory_required_pass"] is True
    assert summary["task044_contract"]["thresholds"]["metric_scope"] == "final_trial_tail_window"


def test_task044_triplet_summary_can_use_post_fault_window_metric_scope(tmp_path: Path) -> None:
    module = _load_tool()
    normal = _write_json(
        tmp_path / "normal.json",
        _summary("none", lin_vel_error=0.20, post_fault_lin_vel_error=0.18),
    )
    zero = _write_json(
        tmp_path / "zero.json",
        _summary("zero_memory_latent", lin_vel_error=0.20, post_fault_lin_vel_error=0.24),
    )
    stateless = _write_json(
        tmp_path / "stateless.json",
        _summary(
            "stateless_txl_memory",
            lin_vel_error=0.20,
            post_fault_lin_vel_error=0.22,
            pipeline_pass=False,
        ),
    )
    args = module.parse_args(
        [
            "--normal-json",
            str(normal),
            "--zero-residual-json",
            str(zero),
            "--stateless-json",
            str(stateless),
            "--confirm-hidden-fault-labels",
            "--metric-scope",
            "post_fault_window",
            "--output-json",
            str(tmp_path / "summary.json"),
        ]
    )

    summary = module.build_triplet_summary(args)

    assert summary["task044_memory_required_pass"] is True
    assert summary["task044_contract"]["thresholds"]["metric_scope"] == "post_fault_window"


def _summary(
    mode: str,
    *,
    lin_vel_error: float,
    window_lin_vel_error: float | None = None,
    tail_lin_vel_error: float | None = None,
    post_fault_lin_vel_error: float | None = None,
    pipeline_pass: bool = True,
) -> dict:
    return {
        "pipeline_pass": pipeline_pass,
        "quality_gate_pass": True,
        "pass": pipeline_pass,
        "memory_ablation_mode": mode,
        "memory_ablation_mode_match": True,
        "memory_debug_active": mode == "none",
        "final_trial": {
            "completion_ratio": 1.0,
            "fall_ratio": 0.0,
            "lin_vel_error": {"mean": lin_vel_error},
        },
        "final_trial_window": {
            "completion_ratio": 1.0,
            "fall_ratio": 0.0,
            "lin_vel_error": {
                "mean": window_lin_vel_error if window_lin_vel_error is not None else lin_vel_error
            },
        },
        "final_trial_tail_window": {
            "completion_ratio": 1.0,
            "fall_ratio": 0.0,
            "lin_vel_error": {
                "mean": tail_lin_vel_error if tail_lin_vel_error is not None else lin_vel_error
            },
        },
        "post_fault_window": {
            "completion_ratio": 1.0,
            "fall_ratio": 0.0,
            "lin_vel_error": {
                "mean": post_fault_lin_vel_error
                if post_fault_lin_vel_error is not None
                else lin_vel_error
            },
        },
        "txl_debug": {
            "task042_memory_ablation_mode": mode,
            "stateful_memory_enabled": mode != "stateless_txl_memory",
            "last_attended_previous_memory_lengths": [64, 64],
            "incremental_steps": 8,
        },
    }


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _load_tool():
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / "task044_memory_required_triplet_summary.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module
