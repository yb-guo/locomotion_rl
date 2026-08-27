from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
TASK_DIR = ROOT / ".agent/task/task071-multimorphology-training-readiness"


def _load_module(name: str, path: Path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PHYSICS = _load_module("task071_physics_overlay", TASK_DIR / "task071_physics_overlay.py")
R3 = _load_module("task071_ppo_smoke", TASK_DIR / "task071_ppo_smoke.py")
LOCAL_INPUTS = PHYSICS.FROZEN.is_dir() and PHYSICS.SOURCE.is_dir()


def _cuda_available() -> bool:
    try:
        import torch
    except (ImportError, RuntimeError):
        return False
    return bool(torch.cuda.is_available())


@pytest.mark.skipif(not LOCAL_INPUTS, reason="Task071 ignored frozen/official inputs unavailable")
@pytest.mark.skipif(not _cuda_available(), reason="Task071 R3 requires CUDA")
def test_r3_g1_go2_runs_no_update_rollout_and_one_real_ppo_update() -> None:
    result = R3.run_ppo_smoke(write_artifact=False)

    assert result["denominator"] == 2
    assert result["r2"]["admission"] is True
    assert result["task071_r3_representative_admission_passed"] is True
    assert result["summary"]["no_update_rollout_passed"] == 2
    assert result["summary"]["one_ppo_update_passed"] == 2
    assert result["summary"]["total_completed_ppo_updates"] == 2
    assert result["summary"]["total_env_steps"] == 64
    assert result["runtime"]["gpu"]["torch_device_name"] == "NVIDIA GeForce RTX 5060 Ti"
    assert result["claim_boundary"] == {
        "bounded_one_update_smoke_only": True,
        "training_performed": True,
        "ppo_updates_completed_per_case": 1,
        "checkpoint_written": False,
        "long_training_started": False,
        "walking_or_quality_claimed": False,
        "task071_passed": False,
    }
    assert {record["reference_id"] for record in result["records"]} == {
        "unitree_g1",
        "unitree_go2",
    }
    for record in result["records"]:
        rollout = record["no_update_rollout"]
        update = record["ppo_update"]
        assert record["passed"] is True
        assert all(record["checks"].values())
        assert record["budget"]["env_steps"] == 32
        assert record["budget"]["ppo_updates"] == 1
        assert record["budget"]["runtime_fault_process"] == "disabled_for_nominal_smoke"
        assert rollout["parameter_l1_delta"] == 0.0
        assert rollout["inactive_actions_exactly_zero"] is True
        assert rollout["fall_count"] == 0.0
        assert rollout["trial_done_count_by_step"] == [0, 0, 0, 4, 0, 0, 0, 4]
        assert rollout["context_done_count_by_step"] == [0, 0, 0, 0, 0, 0, 0, 4]
        assert all(rollout["finite_tensors"].values())
        assert update["completed_update_count"] == 1
        assert record["checks"]["exactly_one_optimizer_step_observed"] is True
        assert update["parameter_l1_delta"] > 0.0
        assert update["diagnostics"]["grad_norm"] > 0.0
        assert update["parameters_finite_after_update"] is True
        assert record["gpu_memory"]["peak_allocated_mib"] > 0.0
