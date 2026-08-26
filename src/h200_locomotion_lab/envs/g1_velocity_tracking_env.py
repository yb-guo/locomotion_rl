"""Compatibility imports for the task moved to :mod:`h200_locomotion_lab.tasks`."""

from h200_locomotion_lab.tasks.g1_velocity_tracking import (
    G1_REWARD_COMPONENT_NAMES,
    G1_TASK_METRIC_NAMES,
    G1VelocityTrackingConfig,
    G1VelocityTrackingStep,
    G1VelocityTrackingVectorizedEnv,
)

__all__ = [
    "G1_REWARD_COMPONENT_NAMES",
    "G1_TASK_METRIC_NAMES",
    "G1VelocityTrackingConfig",
    "G1VelocityTrackingStep",
    "G1VelocityTrackingVectorizedEnv",
]
