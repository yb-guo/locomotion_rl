"""Minimal scalar G1 policy/control runtime.

This module is intentionally Python-object based. It is for one-robot backend
checks, dry runs, and deployment-style loops, not vectorized training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from h200_locomotion_lab.envs.robot_backend import (
    G1MotorCommand,
    G1RobotBackend,
    G1RobotState,
)
from h200_locomotion_lab.runtime.scalar_action_bridge import ScalarActionBridge
from h200_locomotion_lab.sonic.g1_observation import SONIC_ACTION_DIM


class ActionProvider(Protocol):
    """Boundary that turns one backend state into one raw SONIC policy action."""

    def action_for_state(self, state: G1RobotState, step_index: int) -> Sequence[float]:
        """Return one finite raw action in SONIC/IsaacLab policy order."""


@dataclass(frozen=True, slots=True)
class RuntimeStep:
    """One completed scalar policy/control frame."""

    step_index: int
    state: G1RobotState
    raw_action_isaaclab: tuple[float, ...]
    command: G1MotorCommand
    next_state: G1RobotState


@dataclass(frozen=True, slots=True)
class ZeroActionProvider:
    """Provider that always emits a finite zero raw action."""

    action_dim: int = SONIC_ACTION_DIM

    def __post_init__(self) -> None:
        _validate_action_dim(self.action_dim)

    def action_for_state(self, state: G1RobotState, step_index: int) -> tuple[float, ...]:
        return (0.0,) * self.action_dim


@dataclass(frozen=True, slots=True)
class FakeActionProvider:
    """Deterministic finite provider for backend smoke tests."""

    action_dim: int = SONIC_ACTION_DIM
    amplitude: float = 0.05

    def __post_init__(self) -> None:
        _validate_action_dim(self.action_dim)
        if not math.isfinite(self.amplitude) or self.amplitude < 0.0:
            raise ValueError("amplitude must be finite and non-negative")

    def action_for_state(self, state: G1RobotState, step_index: int) -> tuple[float, ...]:
        return tuple(
            self.amplitude * math.sin(float(step_index) + joint_index * 0.37)
            for joint_index in range(self.action_dim)
        )


class SequenceActionProvider:
    """Provider backed by an explicit finite action sequence."""

    def __init__(
        self,
        actions: Sequence[Sequence[float]],
        *,
        repeat_last: bool = False,
        action_dim: int = SONIC_ACTION_DIM,
    ) -> None:
        _validate_action_dim(action_dim)
        if not actions:
            raise ValueError("actions must not be empty")
        self.actions = tuple(
            _coerce_action(row, action_dim, f"actions[{index}]")
            for index, row in enumerate(actions)
        )
        self.repeat_last = bool(repeat_last)
        self.action_dim = action_dim

    def action_for_state(self, state: G1RobotState, step_index: int) -> tuple[float, ...]:
        if step_index < 0:
            raise ValueError("step_index must be non-negative")
        if step_index < len(self.actions):
            return self.actions[step_index]
        if self.repeat_last:
            return self.actions[-1]
        raise ValueError(
            f"sequence provider has {len(self.actions)} actions, cannot serve step {step_index}"
        )


class ScalarG1Runtime:
    """Run one scalar G1 backend with one action provider."""

    def __init__(
        self,
        backend: G1RobotBackend,
        action_provider: ActionProvider,
        *,
        action_bridge: ScalarActionBridge | None = None,
    ) -> None:
        self.backend = backend
        self.action_provider = action_provider
        self.action_bridge = action_bridge
        self._state: G1RobotState | None = None
        self._step_index = 0

    def reset(self) -> G1RobotState:
        self._state = self.backend.reset()
        self._step_index = 0
        return self._state

    def step(self) -> RuntimeStep:
        state = self._state if self._state is not None else self.reset()
        raw_action = _coerce_action(
            self.action_provider.action_for_state(state, self._step_index),
            SONIC_ACTION_DIM,
            "raw_action_isaaclab",
        )
        command = G1MotorCommand.from_raw_sonic_action(
            raw_action,
            action_bridge=self.action_bridge,
        )
        self.backend.write_command(command)
        next_state = self.backend.advance()
        result = RuntimeStep(
            step_index=self._step_index,
            state=state,
            raw_action_isaaclab=raw_action,
            command=command,
            next_state=next_state,
        )
        self._state = next_state
        self._step_index += 1
        return result

    def run(self, policy_steps: int, *, reset: bool = True) -> tuple[RuntimeStep, ...]:
        if policy_steps <= 0:
            raise ValueError("policy_steps must be positive")
        if reset or self._state is None:
            self.reset()
        return tuple(self.step() for _ in range(policy_steps))


def _validate_action_dim(action_dim: int) -> None:
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")


def _coerce_action(values: Sequence[float], action_dim: int, name: str) -> tuple[float, ...]:
    if len(values) != action_dim:
        raise ValueError(f"{name} expected dim={action_dim}, got {len(values)}")
    action = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in action):
        raise ValueError(f"{name} contains a non-finite value")
    return action
