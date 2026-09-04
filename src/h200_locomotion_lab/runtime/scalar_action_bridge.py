"""Scalar action bridge from policy order to command target order."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from h200_locomotion_lab.robots import CompiledRobotProfile


@dataclass(frozen=True, slots=True)
class ScalarActionBridge:
    """Map one raw policy action vector to command-order motor targets."""

    action_dim: int
    command_to_policy: tuple[int, ...]
    default_angles_command: tuple[float, ...]
    action_scale_command: tuple[float, ...]

    @classmethod
    def from_profile(cls, profile: CompiledRobotProfile) -> ScalarActionBridge:
        return cls(
            action_dim=profile.action_dim,
            command_to_policy=profile.mapping.command_mujoco_index_to_policy_isaaclab_index,
            default_angles_command=profile.control.default_angles_rad,
            action_scale_command=profile.control.action_scales_rad,
        )

    def policy_action_to_command_targets(
        self,
        raw_action_policy: Sequence[float],
    ) -> tuple[float, ...]:
        """Apply target[i] = default[i] + raw_action[map[i]] * scale[i]."""

        if len(raw_action_policy) != self.action_dim:
            raise ValueError(
                f"Expected raw action length {self.action_dim}, got {len(raw_action_policy)}"
            )

        action = tuple(float(value) for value in raw_action_policy)
        if not all(isfinite(value) for value in action):
            raise ValueError("raw action must contain only finite values")

        return tuple(
            default_angle + action[policy_index] * action_scale
            for default_angle, action_scale, policy_index in zip(
                self.default_angles_command,
                self.action_scale_command,
                self.command_to_policy,
                strict=True,
            )
        )
