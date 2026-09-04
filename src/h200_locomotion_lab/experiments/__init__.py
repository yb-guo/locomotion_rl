"""Composition roots for task, policy, algorithm, and runtime components."""

from h200_locomotion_lab.experiments.config import (
    DEFAULT_G1_FLAT_PPO_EXPERIMENT,
    ComponentConfigError,
    load_algorithm,
    load_experiment,
    load_policy,
    load_task,
)
from h200_locomotion_lab.experiments.loop import InteractionSummary, run_interaction

__all__ = [
    "DEFAULT_G1_FLAT_PPO_EXPERIMENT",
    "ComponentConfigError",
    "InteractionSummary",
    "load_algorithm",
    "load_experiment",
    "load_policy",
    "load_task",
    "run_interaction",
]
