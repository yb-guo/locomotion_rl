import importlib.util
from types import SimpleNamespace

import pytest

from h200_locomotion_lab.tools.task033_history_buffer_smoke import run_smoke
from h200_locomotion_lab.training.history_buffer import (
    HistoryBufferConfig,
    TorchHistoryBuffer,
    build_default_actor_history_spec,
    validate_no_actor_fault_leakage,
)
from h200_locomotion_lab.training.history_checkpoint_migration import (
    AdaptationConditioningMigrationConfig,
    StackMlpHistoryMigrationConfig,
    migrate_adaptation_conditioned_checkpoint,
    migrate_stack_mlp_checkpoint,
)


def test_history_buffer_module_imports_without_torch() -> None:
    spec = build_default_actor_history_spec(observation_dim=104, action_dim=31)

    assert spec.frame_dim == 135
    assert "active_failure_joint_id" in spec.debug_field_names
    if importlib.util.find_spec("torch") is None:
        assert True


def test_actor_history_field_validator_rejects_fault_labels() -> None:
    with pytest.raises(ValueError, match="active_failure_joint_id"):
        validate_no_actor_fault_leakage(("obs_0", "active_failure_joint_id"))


def test_torch_history_buffer_appends_and_resets_env_subset() -> None:
    torch = pytest.importorskip("torch")
    buffer = TorchHistoryBuffer(
        HistoryBufferConfig(num_envs=2, history_len=3, frame_dim=2, device="cpu")
    )

    buffer.append(torch.full((2, 2), 1.0))
    buffer.append(torch.full((2, 2), 2.0))
    buffer.append(torch.full((2, 2), 3.0), reset_env_ids=[1])

    latest = buffer.latest_oldest_first()

    assert latest.shape == (2, 3, 2)
    assert latest[0, :, 0].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert latest[1, :, 0].tolist() == pytest.approx([0.0, 0.0, 3.0])
    assert buffer.flatten_latest().shape == (2, 6)
    assert buffer.valid_counts.tolist() == [3, 1]


def test_torch_history_buffer_accepts_done_mask_for_reset() -> None:
    torch = pytest.importorskip("torch")
    buffer = TorchHistoryBuffer(
        HistoryBufferConfig(num_envs=3, history_len=2, frame_dim=1, device="cpu")
    )

    buffer.append(torch.ones((3, 1)))
    buffer.append(torch.full((3, 1), 2.0), done=torch.tensor([False, True, False]))

    latest = buffer.latest_oldest_first()

    torch.testing.assert_close(
        latest[:, :, 0],
        torch.tensor([[1.0, 2.0], [0.0, 2.0], [1.0, 2.0]]),
    )
    assert buffer.valid_counts.tolist() == [2, 1, 2]


def test_task033_history_buffer_smoke_reports_contract() -> None:
    pytest.importorskip("torch")
    result = run_smoke(
        SimpleNamespace(
            num_envs=4,
            history_len=4,
            obs_dim=5,
            action_dim=3,
            residual_dim=2,
            steps=6,
            reset_step=2,
            reset_env=1,
            benchmark_steps=0,
            device="cpu",
            dtype="float32",
            command_label="pytest",
            repo_commit="test",
            h200_checkout="test",
        )
    )

    assert result["status"] == "passed"
    assert result["actor_frame_dim"] == 10
    assert result["actor_input_dim_stack"] == 40
    assert result["actor_fault_leakage_check"] == "passed"
    assert result["reset_env_valid_count"] == 4


def test_stack_mlp_checkpoint_migration_maps_source_actor_to_newest_frame() -> None:
    torch = pytest.importorskip("torch")
    source = {
        "actor_state_dict": {
            "obs_normalizer._mean": torch.arange(4, dtype=torch.float32).reshape(1, 4),
            "obs_normalizer._var": torch.full((1, 4), 2.0),
            "obs_normalizer._std": torch.full((1, 4), 3.0),
            "obs_normalizer.count": torch.tensor(10.0),
            "distribution.std_param": torch.ones(2),
            "mlp.0.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "mlp.0.bias": torch.ones(3),
            "mlp.2.weight": torch.ones(2, 3),
            "mlp.2.bias": torch.ones(2),
        },
        "critic_state_dict": {"mlp.0.weight": torch.ones(1, 5)},
        "iter": 123,
        "infos": {},
    }

    migrated, report = migrate_stack_mlp_checkpoint(
        source,
        StackMlpHistoryMigrationConfig(history_len=2, obs_dim=4, action_dim=2),
    )

    first = migrated["actor_state_dict"]["mlp.0.weight"]
    assert first.shape == (3, 12)
    assert first[:, :6].abs().sum().item() == pytest.approx(0.0)
    torch.testing.assert_close(first[:, 6:10], source["actor_state_dict"]["mlp.0.weight"])
    assert first[:, 10:12].abs().sum().item() == pytest.approx(0.0)
    assert migrated["actor_state_dict"]["obs_normalizer._mean"].shape == (1, 12)
    assert migrated["actor_state_dict"]["obs_normalizer._mean"][0, 0:4].tolist() == pytest.approx([0, 1, 2, 3])
    assert migrated["actor_state_dict"]["obs_normalizer._mean"][0, 6:10].tolist() == pytest.approx([0, 1, 2, 3])
    assert migrated["iter"] == 123
    assert report["target_actor_input_dim"] == 12
    assert report["has_optimizer_state_dict"] is False


def test_adaptation_conditioning_migration_preserves_base_obs_path() -> None:
    torch = pytest.importorskip("torch")
    source = {
        "actor_state_dict": {
            "obs_normalizer._mean": torch.arange(4, dtype=torch.float32).reshape(1, 4),
            "obs_normalizer._var": torch.full((1, 4), 2.0),
            "obs_normalizer._std": torch.full((1, 4), 3.0),
            "obs_normalizer.count": torch.tensor(10.0),
            "distribution.std_param": torch.ones(2),
            "mlp.0.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "mlp.0.bias": torch.ones(3),
            "mlp.2.weight": torch.ones(2, 3),
            "mlp.2.bias": torch.ones(2),
        },
        "critic_state_dict": {"mlp.0.weight": torch.ones(1, 5)},
        "iter": 123,
        "infos": {},
    }

    migrated, report = migrate_adaptation_conditioned_checkpoint(
        source,
        AdaptationConditioningMigrationConfig(obs_dim=4, action_dim=2, history_len=2, latent_dim=2),
    )

    first = migrated["actor_state_dict"]["mlp.0.weight"]
    assert first.shape == (3, 6)
    torch.testing.assert_close(first[:, :4], source["actor_state_dict"]["mlp.0.weight"])
    assert first[:, 4:6].abs().sum().item() == pytest.approx(0.0)
    assert migrated["actor_state_dict"]["obs_normalizer._mean"].shape == (1, 12)
    assert migrated["actor_state_dict"]["obs_normalizer._mean"][0, 0:4].tolist() == pytest.approx([0, 1, 2, 3])
    assert migrated["actor_state_dict"]["obs_normalizer._mean"][0, 4:6].tolist() == pytest.approx([0, 0])
    assert migrated["actor_state_dict"]["obs_normalizer._mean"][0, 6:10].tolist() == pytest.approx([0, 1, 2, 3])
    assert migrated["actor_state_dict"]["obs_normalizer._var"][0, 4:6].tolist() == pytest.approx([1, 1])
    assert migrated["iter"] == 123
    assert report["target_actor_input_dim"] == 6
    assert report["history_actor_input_dim"] == 12
    assert report["has_optimizer_state_dict"] is False
