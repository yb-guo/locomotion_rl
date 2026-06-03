"""Checkpoint migration helpers for Task033 history-input MLP policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StackMlpHistoryMigrationConfig:
    """Map a 104D actor MLP checkpoint into a flattened history-input MLP."""

    history_len: int = 4
    obs_dim: int = 104
    action_dim: int = 31

    @property
    def frame_dim(self) -> int:
        return self.obs_dim + self.action_dim

    @property
    def target_actor_input_dim(self) -> int:
        return self.history_len * self.frame_dim

    @property
    def newest_obs_start(self) -> int:
        return (self.history_len - 1) * self.frame_dim

    @property
    def newest_obs_stop(self) -> int:
        return self.newest_obs_start + self.obs_dim


@dataclass(frozen=True, slots=True)
class AdaptationConditioningMigrationConfig:
    """Map a base actor MLP checkpoint into an adaptation-conditioned actor."""

    obs_dim: int = 104
    action_dim: int = 31
    history_len: int = 4
    latent_dim: int = 32

    @property
    def target_actor_input_dim(self) -> int:
        return self.obs_dim + self.latent_dim

    @property
    def frame_dim(self) -> int:
        return self.obs_dim + self.action_dim

    @property
    def history_actor_input_dim(self) -> int:
        return self.history_len * self.frame_dim

    @property
    def obs_slice(self) -> tuple[int, int]:
        return (0, self.obs_dim)

    @property
    def latent_slice(self) -> tuple[int, int]:
        return (self.obs_dim, self.target_actor_input_dim)


def migrate_stack_mlp_actor_state_dict(
    source_state: dict[str, Any],
    config: StackMlpHistoryMigrationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a StackMLP-compatible actor state dict from a base MLP actor.

    The old policy is preserved at initialization by wiring the source actor's
    first layer only to the newest frame's observation slice. Older frames and
    previous-action columns start with zero weights and can be learned later.
    """

    torch = _require_torch()
    first_weight_key = "mlp.0.weight"
    if first_weight_key not in source_state:
        raise KeyError(f"missing actor first layer: {first_weight_key}")
    source_first = source_state[first_weight_key]
    if tuple(source_first.shape)[1] != config.obs_dim:
        raise ValueError(
            f"source actor input dim {tuple(source_first.shape)[1]} does not match obs_dim={config.obs_dim}"
        )

    migrated: dict[str, Any] = {}
    copied_keys: list[str] = []
    expanded_keys: list[str] = []

    for key, value in source_state.items():
        if key == first_weight_key:
            target = torch.zeros(
                (tuple(value.shape)[0], config.target_actor_input_dim),
                dtype=value.dtype,
                device=value.device,
            )
            target[:, config.newest_obs_start : config.newest_obs_stop] = value
            migrated[key] = target
            expanded_keys.append(key)
        elif key in {"obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"}:
            migrated[key] = _expand_obs_normalizer(key, value, config)
            expanded_keys.append(key)
        else:
            migrated[key] = _clone_value(value)
            copied_keys.append(key)

    report = {
        "migration": "task033_stack_mlp_history",
        "history_len": config.history_len,
        "obs_dim": config.obs_dim,
        "action_dim": config.action_dim,
        "frame_dim": config.frame_dim,
        "source_actor_input_dim": config.obs_dim,
        "target_actor_input_dim": config.target_actor_input_dim,
        "newest_obs_slice": [config.newest_obs_start, config.newest_obs_stop],
        "copied_key_count": len(copied_keys),
        "expanded_keys": expanded_keys,
        "first_layer_policy": "source obs weights copied to newest frame obs slice; all other history/action columns zero",
        "optimizer_policy": "fresh optimizer required after migration",
    }
    return migrated, report


def migrate_adaptation_conditioned_actor_state_dict(
    source_state: dict[str, Any],
    config: AdaptationConditioningMigrationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create an adaptation-conditioned actor state dict from a base MLP actor."""

    torch = _require_torch()
    first_weight_key = "mlp.0.weight"
    if first_weight_key not in source_state:
        raise KeyError(f"missing actor first layer: {first_weight_key}")
    source_first = source_state[first_weight_key]
    if tuple(source_first.shape)[1] != config.obs_dim:
        raise ValueError(
            f"source actor input dim {tuple(source_first.shape)[1]} does not match obs_dim={config.obs_dim}"
        )

    migrated: dict[str, Any] = {}
    copied_keys: list[str] = []
    expanded_keys: list[str] = []
    obs_start, obs_stop = config.obs_slice

    for key, value in source_state.items():
        if key == first_weight_key:
            target = torch.zeros(
                (tuple(value.shape)[0], config.target_actor_input_dim),
                dtype=value.dtype,
                device=value.device,
            )
            target[:, obs_start:obs_stop] = value
            migrated[key] = target
            expanded_keys.append(key)
        elif key in {"obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"}:
            migrated[key] = _expand_adaptation_obs_normalizer(key, value, config)
            expanded_keys.append(key)
        else:
            migrated[key] = _clone_value(value)
            copied_keys.append(key)

    report = {
        "migration": "task036_adaptation_conditioning",
        "obs_dim": config.obs_dim,
        "action_dim": config.action_dim,
        "history_len": config.history_len,
        "latent_dim": config.latent_dim,
        "history_actor_input_dim": config.history_actor_input_dim,
        "source_actor_input_dim": config.obs_dim,
        "target_actor_input_dim": config.target_actor_input_dim,
        "obs_slice": list(config.obs_slice),
        "latent_slice": list(config.latent_slice),
        "copied_key_count": len(copied_keys),
        "expanded_keys": expanded_keys,
        "first_layer_policy": "source obs weights copied to obs slice; adaptation latent columns start at zero",
        "optimizer_policy": "fresh optimizer required after migration",
    }
    return migrated, report


def migrate_adaptation_conditioned_checkpoint(
    source_checkpoint: dict[str, Any],
    config: AdaptationConditioningMigrationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Migrate an RSL-RL checkpoint dict to Task036 adaptation-conditioned actor shape."""

    if "actor_state_dict" not in source_checkpoint:
        raise KeyError("source checkpoint is missing actor_state_dict")
    if "critic_state_dict" not in source_checkpoint:
        raise KeyError("source checkpoint is missing critic_state_dict")

    actor_state, actor_report = migrate_adaptation_conditioned_actor_state_dict(
        source_checkpoint["actor_state_dict"],
        config,
    )
    infos = dict(source_checkpoint.get("infos") or {})
    infos["task036_adaptation_conditioning_migration"] = actor_report
    migrated = {
        "actor_state_dict": actor_state,
        "critic_state_dict": {
            key: _clone_value(value)
            for key, value in source_checkpoint["critic_state_dict"].items()
        },
        "iter": int(source_checkpoint.get("iter", 0)),
        "infos": infos,
    }
    report = {
        **actor_report,
        "source_iteration": migrated["iter"],
        "has_optimizer_state_dict": False,
    }
    return migrated, report


def migrate_stack_mlp_checkpoint(
    source_checkpoint: dict[str, Any],
    config: StackMlpHistoryMigrationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Migrate an RSL-RL checkpoint dict to Task033 StackMLP actor shape."""

    if "actor_state_dict" not in source_checkpoint:
        raise KeyError("source checkpoint is missing actor_state_dict")
    if "critic_state_dict" not in source_checkpoint:
        raise KeyError("source checkpoint is missing critic_state_dict")

    actor_state, actor_report = migrate_stack_mlp_actor_state_dict(
        source_checkpoint["actor_state_dict"],
        config,
    )
    infos = dict(source_checkpoint.get("infos") or {})
    infos["task033_history_migration"] = actor_report
    migrated = {
        "actor_state_dict": actor_state,
        "critic_state_dict": {
            key: _clone_value(value)
            for key, value in source_checkpoint["critic_state_dict"].items()
        },
        "iter": int(source_checkpoint.get("iter", 0)),
        "infos": infos,
    }
    report = {
        **actor_report,
        "source_iteration": migrated["iter"],
        "has_optimizer_state_dict": False,
    }
    return migrated, report


def is_stack_mlp_history_migration_needed(
    actor_state: dict[str, Any],
    target_actor_state: dict[str, Any],
) -> bool:
    """Return true when the checkpoint actor input is smaller than target."""

    key = "mlp.0.weight"
    if key not in actor_state or key not in target_actor_state:
        return False
    return tuple(actor_state[key].shape) != tuple(target_actor_state[key].shape)


def is_adaptation_conditioning_migration_needed(
    actor_state: dict[str, Any],
    target_actor_state: dict[str, Any],
) -> bool:
    """Return true when a base actor checkpoint must expand for adaptation latent input."""

    key = "mlp.0.weight"
    if key not in actor_state or key not in target_actor_state:
        return False
    return tuple(actor_state[key].shape) != tuple(target_actor_state[key].shape)


def _expand_obs_normalizer(
    key: str,
    value: Any,
    config: StackMlpHistoryMigrationConfig,
) -> Any:
    torch = _require_torch()
    fill = 0.0 if key == "obs_normalizer._mean" else 1.0
    if value.shape[-1] != config.obs_dim:
        raise ValueError(
            f"source normalizer dim {value.shape[-1]} does not match obs_dim={config.obs_dim}"
        )
    target = torch.full(
        (value.shape[0], config.target_actor_input_dim),
        fill,
        dtype=value.dtype,
        device=value.device,
    )
    for frame_index in range(config.history_len):
        start = frame_index * config.frame_dim
        target[:, start : start + config.obs_dim] = value
    return target


def _expand_adaptation_obs_normalizer(
    key: str,
    value: Any,
    config: AdaptationConditioningMigrationConfig,
) -> Any:
    torch = _require_torch()
    fill = 0.0 if key == "obs_normalizer._mean" else 1.0
    if value.shape[-1] != config.obs_dim:
        raise ValueError(
            f"source normalizer dim {value.shape[-1]} does not match obs_dim={config.obs_dim}"
        )
    target = torch.full(
        (value.shape[0], config.history_actor_input_dim),
        fill,
        dtype=value.dtype,
        device=value.device,
    )
    for frame_index in range(config.history_len):
        start = frame_index * config.frame_dim
        target[:, start : start + config.obs_dim] = value
    return target


def _clone_value(value: Any) -> Any:
    if hasattr(value, "clone"):
        return value.clone()
    return value


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - local no-torch path.
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch
