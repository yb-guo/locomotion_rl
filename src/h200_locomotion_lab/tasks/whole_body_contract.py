"""Task-side contract for the whole-body velocity locomotion family."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from h200_locomotion_lab.core.rl import TaskSpec, TensorSpace
from h200_locomotion_lab.core.whole_body import WholeBodyStep
from h200_locomotion_lab.robots.whole_body_slots import (
    WHOLE_BODY_ACTION_DIM,
    WHOLE_BODY_ACTOR_OBS_DIM,
    WHOLE_BODY_SCHEMA_HASH,
    WHOLE_BODY_SCHEMA_VERSION,
)

WHOLE_BODY_TASK_NAME = "procedural_whole_body_velocity"
UNITREE_MJLAB_REVISION = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
MUJOCO_MENAGERIE_REVISION = "da76818e269b82289eba39808e2fb91d679d6994"


@dataclass(frozen=True, slots=True)
class WholeBodyTaskConfig:
    """Algorithm-free task settings shared by all morphology bindings."""

    name: str = WHOLE_BODY_TASK_NAME
    control_hz: float = 50.0
    trial_seconds: float = 10.0
    command_components: tuple[str, ...] = ("vx", "vy", "yaw_rate")
    reward_components: tuple[str, ...] = (
        "linear_velocity_tracking",
        "yaw_velocity_tracking",
        "alive",
        "base_upright",
        "base_height",
        "joint_acceleration",
        "torque",
        "action_rate",
        "undesired_contact",
    )
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.control_hz <= 0 or self.trial_seconds <= 0:
            raise ValueError("control_hz and trial_seconds must be positive")
        if len(self.command_components) != 3:
            raise ValueError("whole-body velocity task expects three command components")

    @property
    def trial_steps(self) -> int:
        return round(self.control_hz * self.trial_seconds)


def make_whole_body_task_spec(config: WholeBodyTaskConfig | None = None) -> TaskSpec:
    """Build the task-owned observation/action contract without a simulator."""

    config = config or WholeBodyTaskConfig()
    return TaskSpec(
        name=config.name,
        observations={
            "policy": TensorSpace((WHOLE_BODY_ACTOR_OBS_DIM,)),
            # The privileged critic may append backend-specific fields at bind
            # time; the neutral contract keeps the actor-compatible baseline
            # shape until that binder is selected.
            "value": TensorSpace((WHOLE_BODY_ACTOR_OBS_DIM,)),
        },
        action=TensorSpace((WHOLE_BODY_ACTION_DIM,), low=-1.0, high=1.0),
        max_episode_steps=config.trial_steps,
        metrics=(
            "survival",
            "normalized_velocity_error",
            "non_foot_contact_fraction",
            "motor_event_active",
        ),
        parameters={
            "schema_version": WHOLE_BODY_SCHEMA_VERSION,
            "schema_hash": WHOLE_BODY_SCHEMA_HASH,
            "unitree_mjlab_revision": UNITREE_MJLAB_REVISION,
            "mujoco_menagerie_revision": MUJOCO_MENAGERIE_REVISION,
            "control_hz": config.control_hz,
            "reward_components": config.reward_components,
            **config.parameters,
        },
    )


@dataclass(frozen=True, slots=True)
class BoundVectorTask:
    """Minimal runtime shape returned by a morphology/task binder.

    Concrete MuJoCo/MJLab adapters can implement ``reset`` and ``step`` while
    preserving this data shape; the task contract itself remains simulator
    independent.
    """

    spec: TaskSpec
    num_envs: int
    schema_version: str = WHOLE_BODY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.schema_version != WHOLE_BODY_SCHEMA_VERSION:
            raise ValueError("bound task schema version mismatch")

    def reset(self) -> Any:  # pragma: no cover - concrete backend owns behavior
        raise NotImplementedError

    def step(self, action: Any) -> WholeBodyStep:  # pragma: no cover
        del action
        raise NotImplementedError
