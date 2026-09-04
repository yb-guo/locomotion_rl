"""Adapters between robot-local state and the versioned 45-slot space."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from h200_locomotion_lab.robots.procedural_morphology import (
    MorphologyBlueprint,
    PhysicalParams,
)
from h200_locomotion_lab.robots.whole_body_slots import (
    WHOLE_BODY_ACTION_DIM,
    WHOLE_BODY_ACTOR_OBS_DIM,
    WholeBodySlotMapping,
    build_whole_body_slot_mapping,
)


@dataclass(frozen=True, slots=True)
class BoundEmbodiment:
    """A compiled morphology plus its stable unified-space mapping."""

    blueprint: MorphologyBlueprint
    mapping: WholeBodySlotMapping
    physical: PhysicalParams | None = None

    def __post_init__(self) -> None:
        if self.mapping.robot_action_dim != len(self.blueprint.joints):
            raise ValueError("mapping actuator count must match blueprint joint count")
        expected = tuple(joint.semantic_slot for joint in self.blueprint.joints)
        if self.mapping.semantic_slots != expected:
            raise ValueError("mapping semantic order must match blueprint joint order")

    @classmethod
    def from_blueprint(
        cls,
        blueprint: MorphologyBlueprint,
        *,
        physical: PhysicalParams | None = None,
    ) -> BoundEmbodiment:
        mapping = build_whole_body_slot_mapping(
            tuple(joint.name for joint in blueprint.joints),
            semantic_slots={joint.name: joint.semantic_slot for joint in blueprint.joints},
        )
        return cls(blueprint=blueprint, mapping=mapping, physical=physical)

    @property
    def action_mask(self) -> tuple[bool, ...]:
        return self.mapping.mask

    def gather_action(self, unified_action: Sequence[float]) -> tuple[float, ...]:
        return self.mapping.project_to_robot_order(unified_action)

    def scatter_joint_values(self, robot_values: Sequence[float]) -> tuple[float, ...]:
        return self.mapping.scatter_to_unified_slots(robot_values)

    def gather_action_batch(self, unified_action: Any) -> Any:
        """Gather ``[..., 45]`` actions into this embodiment's joint order."""

        return gather_action_batch(unified_action, self.mapping)

    def scatter_joint_values_batch(self, robot_values: Any) -> Any:
        """Scatter ``[..., n_active]`` values into ``[..., 45]`` slots."""

        return scatter_joint_values_batch(robot_values, self.mapping)

    def encode_actor_observation(
        self,
        *,
        base_linear_velocity: Sequence[float],
        base_angular_velocity: Sequence[float],
        projected_gravity: Sequence[float],
        command: Sequence[float],
        joint_position: Sequence[float],
        joint_velocity: Sequence[float],
        previous_action: Sequence[float],
        trial_start: float = 0.0,
    ) -> tuple[float, ...]:
        """Encode robot-local proprioception into the 193D actor contract."""

        for label, values in (
            ("base_linear_velocity", base_linear_velocity),
            ("base_angular_velocity", base_angular_velocity),
            ("projected_gravity", projected_gravity),
            ("command", command),
        ):
            if len(values) != 3:
                raise ValueError(f"{label} must contain exactly three values")
        for label, values in (
            ("joint_position", joint_position),
            ("joint_velocity", joint_velocity),
        ):
            if len(values) != self.mapping.robot_action_dim:
                raise ValueError(
                    f"{label} must contain {self.mapping.robot_action_dim} values"
                )
        if len(previous_action) != WHOLE_BODY_ACTION_DIM:
            raise ValueError(f"previous_action must contain {WHOLE_BODY_ACTION_DIM} values")
        position = self.scatter_joint_values(joint_position)
        velocity = self.scatter_joint_values(joint_velocity)
        return tuple(
            float(value)
            for value in (
                *base_linear_velocity,
                *base_angular_velocity,
                *projected_gravity,
                *command,
                *position,
                *velocity,
                *previous_action,
                *self.mapping.mask,
                float(trial_start),
            )
        )

    def validate_observation(self, observation: Sequence[float]) -> None:
        if len(observation) != WHOLE_BODY_ACTOR_OBS_DIM:
            raise ValueError(
                f"actor observation must have {WHOLE_BODY_ACTOR_OBS_DIM} values, got {len(observation)}"
            )


def gather_action_batch(unified_action: Any, mapping: WholeBodySlotMapping) -> Any:
    """Vectorized gather for NumPy, Torch, and array-like backends."""

    _validate_last_dim(unified_action, mapping.action_dim, "unified_action")
    if hasattr(unified_action, "index_select"):
        indices = _torch_indices(mapping.selector, unified_action)
        return unified_action.index_select(-1, indices)
    if hasattr(unified_action, "__array__"):
        return unified_action[..., list(mapping.selector)]
    return tuple(tuple(row[index] for index in mapping.selector) for row in unified_action)


def scatter_joint_values_batch(robot_values: Any, mapping: WholeBodySlotMapping) -> Any:
    """Vectorized scatter into 45 slots; every inactive value is exactly zero."""

    _validate_last_dim(robot_values, mapping.robot_action_dim, "robot_values")
    if hasattr(robot_values, "new_zeros") and hasattr(robot_values, "scatter"):
        output = robot_values.new_zeros((*robot_values.shape[:-1], mapping.action_dim))
        indices = _torch_indices(mapping.selector, robot_values)
        return output.scatter(-1, indices.expand(*robot_values.shape[:-1], len(indices)), robot_values)
    if hasattr(robot_values, "__array__"):
        import numpy as np  # type: ignore[import-not-found]

        output = np.zeros((*robot_values.shape[:-1], mapping.action_dim), dtype=robot_values.dtype)
        output[..., list(mapping.selector)] = robot_values
        return output
    return tuple(
        tuple(next((row[index] for index, slot in enumerate(mapping.selector) if slot == target), 0.0)
               for target in range(mapping.action_dim))
        for row in robot_values
    )


def mask_unified_batch(unified_values: Any, mapping: WholeBodySlotMapping) -> Any:
    """Apply the topology mask to a ``[..., 45]`` tensor."""

    _validate_last_dim(unified_values, mapping.action_dim, "unified_values")
    if hasattr(unified_values, "new_tensor"):
        mask = unified_values.new_tensor(mapping.mask)
        return unified_values * mask
    if hasattr(unified_values, "__array__"):
        import numpy as np  # type: ignore[import-not-found]

        return unified_values * np.asarray(mapping.mask, dtype=unified_values.dtype)
    return tuple(
        tuple(value if active else 0.0 for value, active in zip(row, mapping.mask))
        for row in unified_values
    )


def _validate_last_dim(values: Any, expected: int, label: str) -> None:
    shape = getattr(values, "shape", None)
    if shape is not None:
        if not shape or shape[-1] != expected:
            raise ValueError(f"{label} last dimension must be {expected}, got {tuple(shape)}")
        return
    if not values or len(values[0]) != expected:
        raise ValueError(f"{label} last dimension must be {expected}")


def _torch_indices(selector: Sequence[int], values: Any) -> Any:
    torch = __import__("torch")
    return torch.as_tensor(selector, device=values.device, dtype=torch.long)


def bind_generated_embodiment(
    blueprint: MorphologyBlueprint,
    physical: PhysicalParams | None = None,
) -> BoundEmbodiment:
    """Small functional binder used by tests and future simulator adapters."""

    return BoundEmbodiment.from_blueprint(blueprint, physical=physical)
