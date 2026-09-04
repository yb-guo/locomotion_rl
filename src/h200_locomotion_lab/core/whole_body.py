"""Framework-neutral whole-body rollout contracts.

This module deliberately has no simulator, Torch, robot, or PPO imports.  It
lets the task and algorithm layers agree on the two reset boundaries and the
active action mask without making either layer own the other.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

WHOLE_BODY_ACTION_DIM = 45
WHOLE_BODY_ACTOR_OBS_DIM = 193


@dataclass(frozen=True, slots=True)
class WholeBodyStep:
    """One vectorized task step with physical and memory reset semantics."""

    actor_observation: Any
    critic_observation: Any
    reward: Any
    trial_done: Any
    context_done: Any
    active_action_mask: Any
    metrics: Mapping[str, Any] = field(default_factory=dict)
    final_observation: Any | None = None


@dataclass(frozen=True, slots=True)
class WholeBodyPolicyOutput:
    """Policy output that carries recurrent state without algorithm coupling."""

    action: Any
    value: Any | None = None
    log_prob: Any | None = None
    next_state: Any | None = None
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WholeBodyRolloutBatch:
    """The sequence-aware PPO boundary for the whole-body task family."""

    actor_observation: Any
    critic_observation: Any
    action: Any
    reward: Any
    value: Any
    log_prob: Any
    trial_done: Any
    context_done: Any
    active_action_mask: Any
    info: Mapping[str, Any] = field(default_factory=dict)


# Public contract name used by the task/algorithm boundary document.
RolloutBatch = WholeBodyRolloutBatch
