"""Sequence-aware PPO utilities for recurrent whole-body policies."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from h200_locomotion_lab.policies.recurrent_whole_body import (
    masked_sequence_mean,
    sequence_padding_mask,
)


@dataclass(frozen=True, slots=True)
class SequencePPOMinibatch:
    observations: Any
    actions: Any
    old_log_probs: Any
    advantages: Any
    returns: Any
    padding_mask: Any
    active_action_mask: Any | None = None


def make_sequence_padding_mask(lengths: Any, max_length: int | None = None) -> Any:
    return sequence_padding_mask(lengths, max_length)


def iter_sequence_minibatches(
    batch: SequencePPOMinibatch,
    *,
    sequence_batch_size: int,
    shuffle: bool = True,
) -> Iterator[SequencePPOMinibatch]:
    """Yield whole sequences; never split a recurrent state across minibatches."""

    torch = _require_torch()
    count = batch.observations.shape[0]
    if sequence_batch_size <= 0:
        raise ValueError("sequence_batch_size must be positive")
    indices = torch.randperm(count, device=batch.observations.device) if shuffle else torch.arange(count, device=batch.observations.device)
    for start in range(0, count, sequence_batch_size):
        selected = indices[start : start + sequence_batch_size]
        yield SequencePPOMinibatch(
            observations=batch.observations.index_select(0, selected),
            actions=batch.actions.index_select(0, selected),
            old_log_probs=batch.old_log_probs.index_select(0, selected),
            advantages=batch.advantages.index_select(0, selected),
            returns=batch.returns.index_select(0, selected),
            padding_mask=batch.padding_mask.index_select(0, selected),
            active_action_mask=(
                batch.active_action_mask.index_select(0, selected)
                if batch.active_action_mask is not None
                else None
            ),
        )


def masked_sequence_ppo_loss(
    new_log_prob: Any,
    old_log_prob: Any,
    advantages: Any,
    padding_mask: Any,
    *,
    clip: float = 0.2,
) -> tuple[Any, Any, Any]:
    """Return clipped policy loss, approx-KL, and clip fraction over valid tokens."""

    ratio = (new_log_prob - old_log_prob).exp()
    clipped = ratio.clamp(1.0 - clip, 1.0 + clip)
    policy_terms = -torch_min(ratio * advantages, clipped * advantages)
    policy_loss = masked_sequence_mean(policy_terms, padding_mask)
    approx_kl = masked_sequence_mean((ratio - 1.0) - (new_log_prob - old_log_prob), padding_mask)
    clip_fraction = masked_sequence_mean(((ratio - 1.0).abs() > clip).float(), padding_mask)
    return policy_loss, approx_kl, clip_fraction


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch


def torch_min(left: Any, right: Any) -> Any:
    return _require_torch().minimum(left, right)

