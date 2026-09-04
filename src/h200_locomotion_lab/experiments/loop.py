"""Minimal interaction loop shared by training and evaluation compositions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from h200_locomotion_lab.core.rl import (
    LearningAlgorithm,
    Policy,
    TransitionBatch,
    VectorTask,
)


@dataclass(frozen=True, slots=True)
class InteractionSummary:
    """Only universal interaction counts; task and algorithm metrics stay separate."""

    steps: int
    environment_steps: int
    episode_ends: int
    updates: int
    final_observation: Any


def run_interaction(
    task: VectorTask,
    policy: Policy,
    *,
    steps: int,
    algorithm: LearningAlgorithm | None = None,
    deterministic: bool = False,
) -> InteractionSummary:
    """Drive a task with a policy and optionally let an algorithm learn.

    The loop does not interpret reward terms, policy internals, or loss names.
    Algorithms decide their own buffering and update cadence by returning
    ``None`` until an update is due.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")

    observation = task.reset()
    episode_ends = 0
    updates = 0

    for _ in range(steps):
        policy_output = policy.act(observation, deterministic=deterministic)
        task_step = task.step(policy_output.action)
        transition = TransitionBatch(
            observation=observation,
            action=policy_output.action,
            reward=task_step.reward,
            next_observation=task_step.observation,
            terminated=task_step.terminated,
            truncated=task_step.truncated,
            policy_info=policy_output.info,
            task_metrics=task_step.metrics,
            final_observation=task_step.final_observation,
        )
        episode_ends += _count_true(transition.done)
        if algorithm is not None:
            algorithm.observe(transition)
            report = algorithm.update(policy)
            if report is not None and report.updated:
                updates += 1
        observation = task_step.observation

    num_envs = int(getattr(task, "num_envs", 1))
    return InteractionSummary(
        steps=steps,
        environment_steps=steps * num_envs,
        episode_ends=episode_ends,
        updates=updates,
        final_observation=observation,
    )


def _count_true(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if hasattr(value, "sum"):
        total = value.sum()
        if hasattr(total, "item"):
            total = total.item()
        return int(total)
    return sum(_count_true(item) for item in value)
