"""Backend-neutral multiplexer for fixed-topology whole-body shards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from h200_locomotion_lab.core.whole_body import WholeBodyStep


class WholeBodyShard(Protocol):
    num_envs: int

    def reset(self) -> Any: ...

    def step(self, action: Any) -> WholeBodyStep: ...


class WholeBodyRolloutMux:
    """Concatenate task shards while keeping topology ownership local.

    A shard owns one compiled MuJoCo model/topology.  The mux only splits and
    concatenates batched tensors; it does not know reward, robot names, or
    simulator internals.
    """

    def __init__(self, shards: Sequence[WholeBodyShard]) -> None:
        self.shards = tuple(shards)
        if not self.shards:
            raise ValueError("at least one whole-body shard is required")
        if any(shard.num_envs <= 0 for shard in self.shards):
            raise ValueError("every shard must contain at least one environment")

    @property
    def num_envs(self) -> int:
        return sum(shard.num_envs for shard in self.shards)

    @property
    def shard_sizes(self) -> tuple[int, ...]:
        return tuple(shard.num_envs for shard in self.shards)

    @property
    def active_action_mask(self) -> Any:
        masks = [getattr(shard, "active_action_mask", None) for shard in self.shards]
        if any(mask is None for mask in masks):
            raise AttributeError("all shards must expose active_action_mask")
        return _concat(masks)

    def reset(self) -> Any:
        return _concat(tuple(shard.reset() for shard in self.shards))

    def step(self, action: Any) -> WholeBodyStep:
        actions = _split_batch(action, self.shard_sizes)
        steps = tuple(shard.step(shard_action) for shard, shard_action in zip(self.shards, actions))
        return WholeBodyStep(
            actor_observation=_concat([step.actor_observation for step in steps]),
            critic_observation=_concat([step.critic_observation for step in steps]),
            reward=_concat([step.reward for step in steps]),
            trial_done=_concat([step.trial_done for step in steps]),
            context_done=_concat([step.context_done for step in steps]),
            active_action_mask=_concat([step.active_action_mask for step in steps]),
            metrics=_concat_metrics([step.metrics for step in steps]),
            final_observation=_concat(
                [step.final_observation for step in steps]
                if all(step.final_observation is not None for step in steps)
                else []
            ),
        )


def _split_batch(batch: Any, sizes: Sequence[int]) -> tuple[Any, ...]:
    if not hasattr(batch, "__getitem__"):
        raise TypeError("batched action must support slicing")
    chunks: list[Any] = []
    start = 0
    for size in sizes:
        chunks.append(batch[start : start + size])
        start += size
    if start != len(batch):
        raise ValueError(f"action batch length must be {start}, got {len(batch)}")
    return tuple(chunks)


def _concat(values: Sequence[Any]) -> Any:
    if not values:
        return None
    first = values[0]
    if hasattr(first, "shape"):
        if hasattr(first, "device") and hasattr(first, "dtype"):
            try:
                import torch  # type: ignore[import-not-found]

                if isinstance(first, torch.Tensor):
                    return torch.cat(tuple(values), dim=0)
            except ImportError:  # pragma: no cover - optional dependency
                pass
        try:
            import numpy as np  # type: ignore[import-not-found]

            return np.concatenate(values, axis=0)
        except ImportError:  # pragma: no cover - optional dependency
            pass
    if isinstance(first, tuple):
        return tuple(item for value in values for item in value)
    if isinstance(first, list):
        return [item for value in values for item in value]
    if len(values) == 1:
        return first
    return tuple(values)


def _concat_metrics(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = set().union(*(item.keys() for item in metrics))
    result: dict[str, Any] = {}
    for key in sorted(keys):
        values = [item[key] for item in metrics if key in item]
        try:
            result[key] = _concat(values)
        except (RuntimeError, TypeError, ValueError):
            # Privileged motor vectors can have one column per active joint;
            # those widths intentionally differ across topology shards.
            result[key] = tuple(values)
    return result
