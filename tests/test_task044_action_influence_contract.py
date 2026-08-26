import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_task044_action_influence_fails_when_actions_are_tied() -> None:
    module = _load_contract()

    result = module.evaluate_task044_action_influence_triplet(
        _summary([0.1, 0.2], mean_l2=0.3),
        _summary([0.101, 0.201], mean_l2=0.301),
        _summary([0.099, 0.199], mean_l2=0.299),
    )

    assert result["action_influence_detected"] is False
    assert "zero_residual_action_stats_tied" in result["failure_reasons"]
    assert "stateless_memory_action_stats_tied" in result["failure_reasons"]
    assert result["memory_causality_claim"] is False


def test_task044_action_influence_records_nontrivial_action_delta() -> None:
    module = _load_contract()

    result = module.evaluate_task044_action_influence_triplet(
        _summary([0.1, 0.2], mean_l2=0.3),
        _summary([0.3, 0.2], mean_l2=0.5),
        _summary([0.1, 0.5], mean_l2=0.6),
    )

    assert result["action_influence_detected"] is True
    assert result["failure_reasons"] == []
    assert result["zero_residual_delta"]["mean_abs_l1_delta"] == pytest.approx(0.1)
    assert result["memory_causality_claim"] is False
    assert result["reproduction_claim"] is False


def test_task044_action_influence_requires_action_stats() -> None:
    module = _load_contract()
    missing = {"final_trial_window": {}}

    result = module.evaluate_task044_action_influence_triplet(missing, missing, missing)

    assert result["action_influence_detected"] is False
    assert "normal_final_trial_window_action_stats_missing" in result["failure_reasons"]


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


def _load_contract():
    path = (
        ROOT
        / "src"
        / "h200_locomotion_lab"
        / "training"
        / "task044_action_influence_contract.py"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module
