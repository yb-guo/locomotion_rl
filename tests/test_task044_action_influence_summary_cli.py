import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_task044_action_influence_summary_records_paths_and_no_overclaim_fields() -> None:
    module = _load_tool()
    fake_json = {
        "normal.json": _summary([0.1, 0.2], mean_l2=0.3),
        "zero.json": _summary([0.3, 0.2], mean_l2=0.5),
        "stateless.json": _summary([0.1, 0.5], mean_l2=0.6),
    }
    module._read_json = lambda path: fake_json[path.name]

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
    summary = module.build_action_influence_summary(args)

    assert summary["task044_action_influence_summary"] is True
    assert summary["action_influence_detected"] is True
    assert summary["normal_json"].endswith("normal.json")
    assert summary["zero_residual_json"].endswith("zero.json")
    assert summary["stateless_json"].endswith("stateless.json")
    assert summary["memory_causality_claim"] is False
    assert summary["reproduction_claim"] is False


def test_task044_action_influence_summary_parse_args_supports_root_scope() -> None:
    module = _load_tool()

    args = module.parse_args(
        [
            "--normal-json",
            "normal.json",
            "--zero-residual-json",
            "zero.json",
            "--stateless-json",
            "stateless.json",
            "--metric-scope",
            "root",
            "--output-json",
            "summary.json",
        ]
    )

    assert args.metric_scope == "root"
    assert args.min_mean_abs_l1_delta > 0.0


def _summary(mean_abs_by_dim: list[float], *, mean_l2: float) -> dict:
    return {
        "final_trial_window": {
            "action_stats": {
                "sample_count": 10,
                "action_dim": len(mean_abs_by_dim),
                "mean_l2": mean_l2,
                "max_abs": max(mean_abs_by_dim),
                "mean_abs_by_dim": mean_abs_by_dim,
                "top_abs_dims": [],
            }
        }
    }


def _load_tool():
    path = (
        ROOT
        / "src"
        / "h200_locomotion_lab"
        / "tools"
        / "task044_action_influence_summary.py"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module
