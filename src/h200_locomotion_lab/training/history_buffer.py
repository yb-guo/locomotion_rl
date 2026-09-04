"""Shared actor-visible history buffer for memory-policy experiments.

Torch is imported lazily so the package can still be inspected on machines
without the training stack installed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

FORBIDDEN_ACTOR_FIELD_TOKENS = (
    "active_failure",
    "failure",
    "fault",
    "motor_scale",
    "scale_factor",
    "dead_motor",
    "weak_motor",
    "joint_id",
    "failure_mask",
)


@dataclass(frozen=True, slots=True)
class HistoryFrameSpec:
    """Names the actor-visible and debug-only fields in one history frame."""

    actor_field_names: tuple[str, ...]
    debug_field_names: tuple[str, ...] = ()

    @property
    def frame_dim(self) -> int:
        return len(self.actor_field_names)

    def validate_no_actor_fault_leakage(self) -> None:
        validate_no_actor_fault_leakage(self.actor_field_names)


@dataclass(frozen=True, slots=True)
class HistoryBufferConfig:
    num_envs: int
    history_len: int
    frame_dim: int
    device: str = "cuda"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        for name in ("num_envs", "history_len", "frame_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class TorchHistoryBuffer:
    """Batched reset-aware ring buffer for actor-visible history frames."""

    def __init__(self, config: HistoryBufferConfig) -> None:
        torch = require_torch()
        dtype = resolve_torch_dtype(torch, config.dtype)
        self.config = config
        self.storage = torch.zeros(
            (config.num_envs, config.history_len, config.frame_dim),
            device=config.device,
            dtype=dtype,
        )
        self.valid_counts = torch.zeros(config.num_envs, device=config.device, dtype=torch.long)
        self._write_index = 0

    @property
    def device(self) -> Any:
        return self.storage.device

    @property
    def dtype(self) -> Any:
        return self.storage.dtype

    @property
    def write_index(self) -> int:
        return self._write_index

    def reset(self, env_ids: Any | None = None) -> None:
        """Clear all envs or only a batched env-id selection."""

        if env_ids is None:
            self.storage.zero_()
            self.valid_counts.zero_()
            self._write_index = 0
            return
        normalized = self._normalize_env_ids(env_ids)
        if normalized.numel() == 0:
            return
        self.storage[normalized] = 0
        self.valid_counts[normalized] = 0

    def append(self, frames: Any, *, reset_env_ids: Any | None = None, done: Any | None = None) -> None:
        """Append one frame per env, optionally clearing reset envs first.

        `frames` are assumed to describe the post-reset/current observation for
        any env listed in `reset_env_ids` or `done`.
        """

        if reset_env_ids is not None and done is not None:
            raise ValueError("Use reset_env_ids or done, not both")
        if done is not None:
            reset_env_ids = self._done_to_env_ids(done)
        if reset_env_ids is not None:
            self.reset(reset_env_ids)
        self._validate_frames(frames)
        self.storage[:, self._write_index, :] = frames.to(device=self.device, dtype=self.dtype)
        self.valid_counts.add_(1).clamp_(max=self.config.history_len)
        self._write_index = (self._write_index + 1) % self.config.history_len

    def latest_oldest_first(self) -> Any:
        """Return `[num_envs, history_len, frame_dim]`, left-padded with zeros."""

        torch = require_torch()
        indices = (
            torch.arange(self.config.history_len, device=self.device) + self._write_index
        ) % self.config.history_len
        return self.storage.index_select(1, indices)

    def flatten_latest(self) -> Any:
        frames = self.latest_oldest_first()
        return frames.reshape(self.config.num_envs, self.config.history_len * self.config.frame_dim)

    def newest(self) -> Any:
        newest_index = (self._write_index - 1) % self.config.history_len
        return self.storage[:, newest_index, :]

    def _validate_frames(self, frames: Any) -> None:
        if tuple(frames.shape) != (self.config.num_envs, self.config.frame_dim):
            raise ValueError(
                "frames must have shape "
                f"({self.config.num_envs}, {self.config.frame_dim}), got {tuple(frames.shape)}"
            )

    def _normalize_env_ids(self, env_ids: Any) -> Any:
        torch = require_torch()
        if isinstance(env_ids, int):
            env_ids = [env_ids]
        if not hasattr(env_ids, "to"):
            env_ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return env_ids.reshape(0)
        if int(env_ids.min().item()) < 0 or int(env_ids.max().item()) >= self.config.num_envs:
            raise IndexError("env_ids out of range")
        return env_ids.reshape(-1)

    def _done_to_env_ids(self, done: Any) -> Any:
        torch = require_torch()
        if not hasattr(done, "to"):
            done = torch.as_tensor(done, dtype=torch.bool, device=self.device)
        else:
            done = done.to(device=self.device, dtype=torch.bool)
        if tuple(done.shape) != (self.config.num_envs,):
            raise ValueError(f"done must have shape ({self.config.num_envs},), got {tuple(done.shape)}")
        return done.nonzero(as_tuple=False).flatten()


def validate_no_actor_fault_leakage(field_names: Sequence[str]) -> None:
    leaked = [
        name
        for name in field_names
        if any(token in normalized_field_name(name) for token in FORBIDDEN_ACTOR_FIELD_TOKENS)
    ]
    if leaked:
        joined = ", ".join(leaked)
        raise ValueError(f"actor history fields leak fault labels: {joined}")


def normalized_field_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - local no-torch path.
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch


def resolve_torch_dtype(torch: Any, dtype: str) -> Any:
    try:
        return getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype}") from exc


def build_default_actor_history_spec(
    *,
    observation_dim: int,
    action_dim: int,
    residual_dim: int = 0,
) -> HistoryFrameSpec:
    """Build the Task033 default actor-visible frame layout."""

    if observation_dim <= 0 or action_dim <= 0 or residual_dim < 0:
        raise ValueError("observation/action dims must be positive and residual_dim non-negative")
    names = tuple(prefixed_names("obs", observation_dim))
    names += tuple(prefixed_names("prev_action", action_dim))
    names += tuple(prefixed_names("action_response_residual", residual_dim))
    spec = HistoryFrameSpec(
        actor_field_names=names,
        debug_field_names=(
            "active_failure_joint_id",
            "motor_scale",
            "failure_type",
            "segment_case_id",
            "scheduler_state",
        ),
    )
    spec.validate_no_actor_fault_leakage()
    return spec


def prefixed_names(prefix: str, count: int) -> Iterable[str]:
    for index in range(count):
        yield f"{prefix}_{index}"
