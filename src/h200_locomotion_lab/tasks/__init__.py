"""MDP definitions: observations, actions, rewards, resets, and termination only."""

from h200_locomotion_lab.core.rl import TaskSpec, TaskStep, TensorSpace, VectorTask
from h200_locomotion_lab.tasks.g1_velocity_tracking import (
    G1_REWARD_COMPONENT_NAMES,
    G1_TASK_METRIC_NAMES,
    G1VelocityTrackingConfig,
    G1VelocityTrackingStep,
    G1VelocityTrackingVectorizedEnv,
)
from h200_locomotion_lab.tasks.whole_body_contract import (
    MUJOCO_MENAGERIE_REVISION,
    UNITREE_MJLAB_REVISION,
    BoundVectorTask,
    WholeBodyTaskConfig,
    make_whole_body_task_spec,
)

__all__ = [
    "G1_REWARD_COMPONENT_NAMES",
    "G1_TASK_METRIC_NAMES",
    "MUJOCO_MENAGERIE_REVISION",
    "UNITREE_MJLAB_REVISION",
    "BoundVectorTask",
    "G1VelocityTrackingConfig",
    "G1VelocityTrackingStep",
    "G1VelocityTrackingVectorizedEnv",
    "TaskSpec",
    "TaskStep",
    "TensorSpace",
    "VectorTask",
    "WholeBodyTaskConfig",
    "make_whole_body_task_spec",
]
