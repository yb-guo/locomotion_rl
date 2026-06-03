"""Task038 local Transformer-XL memory/cache contract.

This module is intentionally simulator-free and torch-free. It models the
cache semantics needed by a TXL-style policy consumer before wiring the same
contract into a runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


Number = int | float


@dataclass(frozen=True, slots=True)
class TxlMemoryConfig:
    num_envs: int
    num_layers: int
    memory_len: int
    token_dim: int

    def __post_init__(self) -> None:
        for name, value in (
            ("num_envs", self.num_envs),
            ("num_layers", self.num_layers),
            ("memory_len", self.memory_len),
            ("token_dim", self.token_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True, slots=True)
class FrameToken:
    values: tuple[float, ...]
    obs_dim: int
    action_dim: int


@dataclass(frozen=True, slots=True)
class HiddenStateToken:
    env_id: int
    layer_id: int
    episode_index: int
    token_index: int
    values: tuple[float, ...]


@dataclass(slots=True)
class _EnvState:
    layers: list[list[HiddenStateToken]]
    episode_index: int = 0
    inner_reset_events: int = 0
    outer_reset_events: int = 0
    incremental_steps: int = 0
    segments_appended: int = 0
    tokens_appended: int = 0
    next_token_index: int = 0


def encode_frame_token(
    obs: Sequence[Number] | Mapping[str, Number],
    action: Sequence[Number] | Mapping[str, Number],
    *,
    token_dim: int | None = None,
) -> FrameToken:
    """Encode one obs/action frame into one deterministic token."""

    obs_values = _numeric_values(obs)
    action_values = _numeric_values(action)
    values = (*obs_values, *action_values)
    if token_dim is not None:
        if len(values) > token_dim:
            raise ValueError(
                f"encoded obs/action token dim {len(values)} exceeds token_dim={token_dim}"
            )
        values = (*values, *(0.0 for _ in range(token_dim - len(values))))
    return FrameToken(
        values=tuple(float(value) for value in values),
        obs_dim=len(obs_values),
        action_dim=len(action_values),
    )


@dataclass(slots=True)
class TxlMemoryCache:
    config: TxlMemoryConfig
    _envs: list[_EnvState] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._envs = [
            _EnvState(layers=[[] for _ in range(self.config.num_layers)])
            for _ in range(self.config.num_envs)
        ]

    def memory_lengths(self, env_id: int) -> tuple[int, ...]:
        env = self._state(env_id)
        return tuple(len(layer) for layer in env.layers)

    def debug_snapshot(self) -> dict[str, object]:
        return {
            "num_envs": self.config.num_envs,
            "num_layers": self.config.num_layers,
            "memory_len": self.config.memory_len,
            "envs": [
                {
                    "env_id": env_id,
                    "episode_index": env.episode_index,
                    "memory_lengths": self.memory_lengths(env_id),
                    "inner_reset_events": env.inner_reset_events,
                    "outer_reset_events": env.outer_reset_events,
                    "incremental_steps": env.incremental_steps,
                    "segments_appended": env.segments_appended,
                    "tokens_appended": env.tokens_appended,
                }
                for env_id, env in enumerate(self._envs)
            ],
        }

    def record_inner_reset(self, env_id: int, *, reason: str = "inner_trial") -> dict[str, object]:
        env = self._state(env_id)
        before = self.memory_lengths(env_id)
        env.inner_reset_events += 1
        return {
            "event": "inner_reset",
            "env_id": env_id,
            "reason": reason,
            "decision": "preserve_memory",
            "before_memory_lengths": before,
            "after_memory_lengths": self.memory_lengths(env_id),
            "inner_reset_events": env.inner_reset_events,
        }

    def outer_reset(self, env_ids: Iterable[int], *, reason: str = "outer_episode") -> list[dict[str, object]]:
        return [self._outer_reset_one(env_id, reason=reason) for env_id in env_ids]

    def step(
        self,
        env_id: int,
        obs: Sequence[Number] | Mapping[str, Number],
        action: Sequence[Number] | Mapping[str, Number],
        *,
        inner_reset: bool = False,
        outer_reset: bool = False,
        incremental: bool = True,
    ) -> dict[str, object]:
        return self.append_segment(
            env_id,
            [(obs, action)],
            inner_reset=inner_reset,
            outer_reset=outer_reset,
            incremental=incremental,
        )

    def append_segment(
        self,
        env_id: int,
        frames: Iterable[
            FrameToken
            | tuple[Sequence[Number] | Mapping[str, Number], Sequence[Number] | Mapping[str, Number]]
        ],
        *,
        inner_reset: bool = False,
        outer_reset: bool = False,
        incremental: bool = False,
    ) -> dict[str, object]:
        env = self._state(env_id)
        tokens = tuple(self._coerce_frame(frame) for frame in frames)
        if not tokens:
            raise ValueError("empty TXL segment rejected")

        reset_events: list[dict[str, object]] = []
        if outer_reset:
            reset_events.append(self._outer_reset_one(env_id, reason="outer_episode"))
            env = self._state(env_id)
        if inner_reset:
            reset_events.append(self.record_inner_reset(env_id))

        previous_lengths = self.memory_lengths(env_id)
        self._assert_cache_is_env_local(env_id)

        for token in tokens:
            for layer_id in range(self.config.num_layers):
                env.layers[layer_id].append(
                    HiddenStateToken(
                        env_id=env_id,
                        layer_id=layer_id,
                        episode_index=env.episode_index,
                        token_index=env.next_token_index,
                        values=_layer_hidden_values(token, layer_id),
                    )
                )
                if len(env.layers[layer_id]) > self.config.memory_len:
                    env.layers[layer_id] = env.layers[layer_id][-self.config.memory_len :]
            env.next_token_index += 1

        env.segments_appended += 1
        env.tokens_appended += len(tokens)
        if incremental:
            env.incremental_steps += len(tokens)

        after_lengths = self.memory_lengths(env_id)
        return {
            "event": "append_segment",
            "env_id": env_id,
            "episode_index": env.episode_index,
            "previous_memory_lengths": previous_lengths,
            "attended_previous_memory_lengths": previous_lengths,
            "after_memory_lengths": after_lengths,
            "appended_tokens": len(tokens),
            "incremental": incremental,
            "incremental_steps": env.incremental_steps,
            "segments_appended": env.segments_appended,
            "tokens_appended": env.tokens_appended,
            "inner_reset_events": env.inner_reset_events,
            "outer_reset_events": env.outer_reset_events,
            "reset_events": reset_events,
            "leak_guard_checked": True,
            "cache_env_ids": self._cache_env_ids(env_id),
            "cache_episode_indices": self._cache_episode_indices(env_id),
        }

    def _outer_reset_one(self, env_id: int, *, reason: str) -> dict[str, object]:
        env = self._state(env_id)
        before = self.memory_lengths(env_id)
        env.layers = [[] for _ in range(self.config.num_layers)]
        env.episode_index += 1
        env.outer_reset_events += 1
        env.next_token_index = 0
        return {
            "event": "outer_reset",
            "env_id": env_id,
            "reason": reason,
            "decision": "clear_selected_env_memory",
            "before_memory_lengths": before,
            "after_memory_lengths": self.memory_lengths(env_id),
            "outer_reset_events": env.outer_reset_events,
            "episode_index": env.episode_index,
        }

    def _state(self, env_id: int) -> _EnvState:
        if not 0 <= env_id < self.config.num_envs:
            raise ValueError(f"env_id={env_id} outside [0, {self.config.num_envs})")
        return self._envs[env_id]

    def _coerce_frame(
        self,
        frame: FrameToken
        | tuple[Sequence[Number] | Mapping[str, Number], Sequence[Number] | Mapping[str, Number]],
    ) -> FrameToken:
        if isinstance(frame, FrameToken):
            token = frame
            if len(token.values) != self.config.token_dim:
                raise ValueError(
                    f"FrameToken dim {len(token.values)} does not match token_dim={self.config.token_dim}"
                )
            return token
        obs, action = frame
        return encode_frame_token(obs, action, token_dim=self.config.token_dim)

    def _assert_cache_is_env_local(self, env_id: int) -> None:
        env = self._state(env_id)
        for layer_id, layer in enumerate(env.layers):
            for token in layer:
                if token.env_id != env_id:
                    raise RuntimeError(
                        f"env memory leak detected: env_id={env_id} layer={layer_id} "
                        f"contains token.env_id={token.env_id}"
                    )
                if token.episode_index != env.episode_index:
                    raise RuntimeError(
                        f"stale outer episode memory detected: env_id={env_id} "
                        f"layer={layer_id} token_episode={token.episode_index} "
                        f"current_episode={env.episode_index}"
                    )

    def _cache_env_ids(self, env_id: int) -> tuple[tuple[int, ...], ...]:
        env = self._state(env_id)
        return tuple(tuple(token.env_id for token in layer) for layer in env.layers)

    def _cache_episode_indices(self, env_id: int) -> tuple[tuple[int, ...], ...]:
        env = self._state(env_id)
        return tuple(tuple(token.episode_index for token in layer) for layer in env.layers)


def _numeric_values(values: Sequence[Number] | Mapping[str, Number]) -> tuple[float, ...]:
    if isinstance(values, Mapping):
        ordered = (values[key] for key in sorted(values))
    else:
        ordered = values
    return tuple(float(value) for value in ordered)


def _layer_hidden_values(token: FrameToken, layer_id: int) -> tuple[float, ...]:
    layer_offset = float(layer_id + 1) / 1000.0
    return tuple(value + layer_offset for value in token.values)
