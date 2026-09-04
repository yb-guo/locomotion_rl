import importlib.util
from pathlib import Path

import pytest

from h200_locomotion_lab.training.rsl_history_wrapper import (
    migrate_adaptk160_to_task041_true_txl_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def test_task041_adaptk160_to_true_txl_migration_copies_matching_policy_path() -> None:
    torch = pytest.importorskip("torch")
    source = {
        "iter": 5467,
        "infos": {"env_state": {"common_step_counter": 123}},
        "optimizer_state_dict": {"must": "drop"},
        "actor_state_dict": {
            "obs_normalizer._mean": torch.ones(1, 21600),
            "obs_normalizer._var": torch.full((1, 21600), 2.0),
            "obs_normalizer._std": torch.full((1, 21600), 3.0),
            "mlp.0.weight": torch.full((512, 136), 4.0),
            "mlp.0.bias": torch.full((512,), 5.0),
            "adaptation_encoder.0.weight": torch.full((128, 21600), 6.0),
            "adaptation_encoder.0.bias": torch.full((128,), 7.0),
            "adaptation_encoder.2.weight": torch.full((32, 128), 8.0),
            "adaptation_encoder.2.bias": torch.full((32,), 9.0),
            "token_projection.weight": torch.full((128, 135), 10.0),
        },
        "critic_state_dict": {
            "mlp.0.weight": torch.full((512, 119), 11.0),
            "mlp.0.bias": torch.full((512,), 12.0),
        },
    }
    target_actor = {
        "obs_normalizer._mean": torch.zeros(1, 21600),
        "obs_normalizer._var": torch.zeros(1, 21600),
        "obs_normalizer._std": torch.zeros(1, 21600),
        "mlp.0.weight": torch.full((512, 136), 9.0),
        "mlp.0.bias": torch.zeros(512),
        "adaptation_encoder.0.weight": torch.zeros(128, 21600),
        "adaptation_encoder.0.bias": torch.zeros(128),
        "adaptation_encoder.2.weight": torch.zeros(32, 128),
        "adaptation_encoder.2.bias": torch.zeros(32),
        "token_projection.weight": torch.zeros(128, 135),
        "memory_output_projection.weight": torch.zeros(32, 128),
    }
    target_critic = {
        "mlp.0.weight": torch.zeros(512, 119),
        "mlp.0.bias": torch.zeros(512),
    }

    migrated, report = migrate_adaptk160_to_task041_true_txl_checkpoint(
        source,
        target_actor_state=target_actor,
        target_critic_state=target_critic,
    )

    assert migrated["iter"] == 0
    assert "optimizer_state_dict" not in migrated
    assert report["source_iter"] == 5467
    assert report["target_iter"] == 0
    torch.testing.assert_close(migrated["actor_state_dict"]["mlp.0.weight"], source["actor_state_dict"]["mlp.0.weight"])
    torch.testing.assert_close(
        migrated["actor_state_dict"]["adaptation_encoder.2.bias"],
        source["actor_state_dict"]["adaptation_encoder.2.bias"],
    )
    torch.testing.assert_close(
        migrated["actor_state_dict"]["memory_output_projection.weight"],
        target_actor["memory_output_projection.weight"],
    )
    torch.testing.assert_close(
        migrated["critic_state_dict"]["mlp.0.weight"],
        source["critic_state_dict"]["mlp.0.weight"],
    )
    assert "token_projection.weight" in report["actor_copied_keys"]
    assert "memory_output_projection.weight" in report["actor_fresh_keys"]
    assert "task041_adaptk160_true_txl_warmstart" in migrated["infos"]


def test_task041_true_txl_migration_copies_base_obs_prefix_into_memory_actor() -> None:
    torch = pytest.importorskip("torch")
    source = {
        "iter": 5349,
        "infos": {},
        "actor_state_dict": {
            "obs_normalizer._mean": torch.arange(104, dtype=torch.float32).reshape(1, 104),
            "obs_normalizer._var": torch.full((1, 104), 2.0),
            "obs_normalizer._std": torch.full((1, 104), 3.0),
            "obs_normalizer.count": torch.tensor(10.0),
            "distribution.std_param": torch.full((31,), 0.2),
            "mlp.0.weight": torch.full((512, 104), 4.0),
            "mlp.0.bias": torch.full((512,), 5.0),
        },
        "critic_state_dict": {},
    }
    target_actor = {
        "obs_normalizer._mean": torch.zeros(1, 136),
        "obs_normalizer._var": torch.ones(1, 136),
        "obs_normalizer._std": torch.ones(1, 136),
        "obs_normalizer.count": torch.tensor(1.0),
        "distribution.std_param": torch.zeros(31),
        "mlp.0.weight": torch.zeros(512, 136),
        "mlp.0.bias": torch.zeros(512),
        "memory_output_projection.weight": torch.zeros(32, 128),
    }

    migrated, report = migrate_adaptk160_to_task041_true_txl_checkpoint(
        source,
        target_actor_state=target_actor,
        target_critic_state={},
    )

    actor = migrated["actor_state_dict"]
    torch.testing.assert_close(actor["obs_normalizer._mean"][..., :104], source["actor_state_dict"]["obs_normalizer._mean"])
    torch.testing.assert_close(actor["obs_normalizer._mean"][..., 104:], target_actor["obs_normalizer._mean"][..., 104:])
    torch.testing.assert_close(actor["obs_normalizer._var"][..., :104], source["actor_state_dict"]["obs_normalizer._var"])
    torch.testing.assert_close(actor["obs_normalizer._var"][..., 104:], target_actor["obs_normalizer._var"][..., 104:])
    torch.testing.assert_close(actor["mlp.0.weight"][:, :104], source["actor_state_dict"]["mlp.0.weight"])
    torch.testing.assert_close(actor["mlp.0.weight"][:, 104:], torch.zeros(512, 32))
    assert set(report["actor_partial_keys"]) == {
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
        "mlp.0.weight",
    }
    assert "mlp.0.bias" in report["actor_copied_keys"]
    assert "memory_output_projection.weight" in report["actor_fresh_keys"]


def test_task041_true_txl_migration_expands_base_normalizer_across_history() -> None:
    torch = pytest.importorskip("torch")
    source = {
        "iter": 5349,
        "infos": {},
        "actor_state_dict": {
            "obs_normalizer._mean": torch.arange(4, dtype=torch.float32).reshape(1, 4),
            "obs_normalizer._var": torch.full((1, 4), 2.0),
            "obs_normalizer._std": torch.full((1, 4), 3.0),
            "distribution.std_param": torch.zeros(2),
            "mlp.0.weight": torch.full((8, 4), 4.0),
            "mlp.0.bias": torch.ones(8),
        },
        "critic_state_dict": {},
    }
    target_actor = {
        "obs_normalizer._mean": torch.full((1, 18), -1.0),
        "obs_normalizer._var": torch.full((1, 18), -1.0),
        "obs_normalizer._std": torch.full((1, 18), -1.0),
        "distribution.std_param": torch.zeros(2),
        "mlp.0.weight": torch.zeros(8, 6),
        "mlp.0.bias": torch.zeros(8),
    }

    migrated, report = migrate_adaptk160_to_task041_true_txl_checkpoint(
        source,
        target_actor_state=target_actor,
        target_critic_state={},
    )

    actor = migrated["actor_state_dict"]
    expected_mean = torch.tensor([[0, 1, 2, 3, 0, 0, 0, 1, 2, 3, 0, 0, 0, 1, 2, 3, 0, 0]], dtype=torch.float32)
    expected_var = torch.tensor([[2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1]], dtype=torch.float32)
    torch.testing.assert_close(actor["obs_normalizer._mean"], expected_mean)
    torch.testing.assert_close(actor["obs_normalizer._var"], expected_var)
    torch.testing.assert_close(actor["obs_normalizer._std"], torch.full((1, 18), 3.0).masked_fill(torch.tensor([[False, False, False, False, True, True, False, False, False, False, True, True, False, False, False, False, True, True]]), 1.0))
    assert "obs_normalizer._mean" in report["actor_partial_keys"]


def test_task041_warmstart_pipeline_accepts_partial_actor_prefix_copy() -> None:
    module = _load_src_tool("task041_adaptk160_true_txl_warmstart.py")
    summary = {
        "actor_model_class": "Task038TrueTxlMemoryModel",
        "target_checkpoint_exists": True,
        "migration_report": {
            "actor_copied_key_count": 3,
            "actor_copied_keys": ["obs_normalizer.count", "mlp.0.bias"],
            "actor_partial_keys": [
                "obs_normalizer._mean",
                "obs_normalizer._var",
                "obs_normalizer._std",
                "mlp.0.weight",
            ],
        },
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
    }

    passed, reasons = module.evaluate_pipeline_pass(summary)

    assert passed
    assert reasons == []


def test_task041_warmstart_cli_parse_defaults() -> None:
    module = _load_src_tool("task041_adaptk160_true_txl_warmstart.py")

    args = module.parse_args(
        [
            "--source-checkpoint",
            "source.pt",
            "--target-checkpoint",
            "target.pt",
            "--output-json",
            "summary.json",
        ]
    )

    assert args.task == "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke"
    assert args.memory_latent_dim == 32
    assert args.base_obs_passthrough is True
    assert args.adaptation_warmstart is True


def _load_src_tool(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
