"""Robot backend boundary for SONIC G1 29DoF dry-run and deployment work.

The contract here is deliberately smaller than a simulator API or a real Unitree
SDK wrapper. It only exposes the state and command fields that the current SONIC
body policy needs: root pose, 29 MuJoCo-order motor states, base angular
velocity, and raw 29D SONIC actions mapped to MuJoCo-order motor targets.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from h200_locomotion_lab.runtime import ScalarActionBridge
from h200_locomotion_lab.sonic.g1_observation import (
    SONIC_ACTION_DIM,
    SonicG1HistoryFrame,
    mujoco_motor_state_to_sonic_body_state,
)
from h200_locomotion_lab.sonic.g1_planner_encoder import SONIC_PLANNER_QPOS_DIM
from h200_locomotion_lab.sonic.g1_policy_bridge import get_default_sonic_g1_action_bridge


@dataclass(frozen=True)
class G1RobotState:
    """One backend-neutral G1 body state sample.

    `root_qpos` follows MuJoCo free-joint order:
    `(x, y, z, qw, qx, qy, qz)`.

    Motor positions and velocities are MuJoCo/hardware order for the validated
    29DoF G1 body policy. `last_action_isaaclab` is the previous raw decoder
    action in SONIC policy order; this is what the 994D decoder observation
    expects in its history.
    """

    root_qpos: tuple[float, ...]
    motor_positions_mujoco: tuple[float, ...]
    motor_velocities_mujoco: tuple[float, ...] = (0.0,) * SONIC_ACTION_DIM
    base_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    last_action_isaaclab: tuple[float, ...] = (0.0,) * SONIC_ACTION_DIM

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_qpos", _coerce_vector(self.root_qpos, 7, "root_qpos"))
        object.__setattr__(
            self,
            "motor_positions_mujoco",
            _coerce_vector(
                self.motor_positions_mujoco,
                SONIC_ACTION_DIM,
                "motor_positions_mujoco",
            ),
        )
        object.__setattr__(
            self,
            "motor_velocities_mujoco",
            _coerce_vector(
                self.motor_velocities_mujoco,
                SONIC_ACTION_DIM,
                "motor_velocities_mujoco",
            ),
        )
        object.__setattr__(
            self,
            "base_angular_velocity",
            _coerce_xyz(self.base_angular_velocity, "base_angular_velocity"),
        )
        object.__setattr__(
            self,
            "last_action_isaaclab",
            _coerce_vector(
                self.last_action_isaaclab,
                SONIC_ACTION_DIM,
                "last_action_isaaclab",
            ),
        )

    @property
    def base_quat(self) -> tuple[float, float, float, float]:
        return tuple(self.root_qpos[3:7])  # type: ignore[return-value]

    @property
    def root_z(self) -> float:
        return self.root_qpos[2]


@dataclass(frozen=True)
class G1MotorCommand:
    """One SONIC body command after official action bridge mapping."""

    raw_action_isaaclab: tuple[float, ...]
    motor_position_targets_mujoco: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "raw_action_isaaclab",
            _coerce_vector(self.raw_action_isaaclab, SONIC_ACTION_DIM, "raw_action_isaaclab"),
        )
        object.__setattr__(
            self,
            "motor_position_targets_mujoco",
            _coerce_vector(
                self.motor_position_targets_mujoco,
                SONIC_ACTION_DIM,
                "motor_position_targets_mujoco",
            ),
        )

    @classmethod
    def from_raw_sonic_action(
        cls,
        raw_action_isaaclab: Sequence[float],
        action_bridge: ScalarActionBridge | None = None,
    ) -> G1MotorCommand:
        raw_action = _coerce_vector(raw_action_isaaclab, SONIC_ACTION_DIM, "raw_action_isaaclab")
        bridge = action_bridge or get_default_sonic_g1_action_bridge()
        return cls(
            raw_action_isaaclab=raw_action,
            motor_position_targets_mujoco=bridge.policy_action_to_command_targets(raw_action),
        )


class G1RobotBackend(Protocol):
    """Minimal body backend needed by the SONIC deployment bridge."""

    def reset(self) -> G1RobotState:
        """Reset or rewind the backend and return the first state."""

    def read_state(self) -> G1RobotState:
        """Read the current body state."""

    def write_command(self, command: G1MotorCommand) -> None:
        """Write one motor target command without advancing time."""

    def advance(self) -> G1RobotState:
        """Advance one policy frame and return the new state."""


class LogReplayG1RobotBackend:
    """Hardware-free backend that replays recorded 36D MuJoCo qpos rows."""

    def __init__(self, states: Sequence[G1RobotState]) -> None:
        if not states:
            raise ValueError("states must not be empty")
        self._states = tuple(states)
        self._index = 0
        self._last_action = (0.0,) * SONIC_ACTION_DIM
        self.commands: list[G1MotorCommand] = []

    @classmethod
    def from_mujoco_qpos_rows(
        cls,
        rows: Sequence[Sequence[float]],
        *,
        policy_rate_hz: float = 50.0,
    ) -> LogReplayG1RobotBackend:
        if policy_rate_hz <= 0.0:
            raise ValueError("policy_rate_hz must be positive")
        qpos_rows = tuple(
            _coerce_vector(row, SONIC_PLANNER_QPOS_DIM, "mujoco_qpos_row")
            for row in rows
        )
        states: list[G1RobotState] = []
        previous_motor_positions: tuple[float, ...] | None = None
        for row in qpos_rows:
            motor_positions = row[7:]
            if previous_motor_positions is None:
                motor_velocities = (0.0,) * SONIC_ACTION_DIM
            else:
                motor_velocities = tuple(
                    (position - previous) * policy_rate_hz
                    for position, previous in zip(motor_positions, previous_motor_positions)
                )
            states.append(
                G1RobotState(
                    root_qpos=row[:7],
                    motor_positions_mujoco=motor_positions,
                    motor_velocities_mujoco=motor_velocities,
                )
            )
            previous_motor_positions = motor_positions
        return cls(states)

    def reset(self) -> G1RobotState:
        self._index = 0
        self._last_action = (0.0,) * SONIC_ACTION_DIM
        self.commands.clear()
        return self.read_state()

    def read_state(self) -> G1RobotState:
        return self._state_with_last_action(self._states[self._index])

    def write_command(self, command: G1MotorCommand) -> None:
        self._last_action = command.raw_action_isaaclab
        self.commands.append(command)

    def advance(self) -> G1RobotState:
        if self._index < len(self._states) - 1:
            self._index += 1
        return self.read_state()

    def _state_with_last_action(self, state: G1RobotState) -> G1RobotState:
        return G1RobotState(
            root_qpos=state.root_qpos,
            motor_positions_mujoco=state.motor_positions_mujoco,
            motor_velocities_mujoco=state.motor_velocities_mujoco,
            base_angular_velocity=state.base_angular_velocity,
            last_action_isaaclab=self._last_action,
        )


class GenesisG1SceneRobotBackend:
    """Adapter from the current Genesis scene backend to `G1RobotBackend`."""

    def __init__(self, scene_backend: Any) -> None:
        self.scene_backend = scene_backend
        self._last_command = G1MotorCommand.from_raw_sonic_action((0.0,) * SONIC_ACTION_DIM)

    def reset(self) -> G1RobotState:
        self.scene_backend.reset()
        self._last_command = G1MotorCommand.from_raw_sonic_action((0.0,) * SONIC_ACTION_DIM)
        return self.read_state()

    def read_state(self) -> G1RobotState:
        return read_genesis_g1_robot_state(
            self.scene_backend,
            last_action_isaaclab=self._last_command.raw_action_isaaclab,
        )

    def write_command(self, command: G1MotorCommand) -> None:
        self._last_command = command
        self.scene_backend.robot.control_dofs_position(
            command.motor_position_targets_mujoco,
            dofs_idx_local=self.scene_backend.motor_dof_indices,
        )

    def advance(self) -> G1RobotState:
        for _ in range(self.scene_backend.contract.decimation):
            self.scene_backend.scene.step()
        self.scene_backend.previous_action = self._last_command.raw_action_isaaclab
        self.scene_backend.step_count += 1
        self.scene_backend.record_sonic_history_frame()
        return self.read_state()


def read_genesis_g1_robot_state(
    scene_backend: Any,
    *,
    last_action_isaaclab: Sequence[float] | None = None,
) -> G1RobotState:
    """Read the current Genesis scene backend state through the deployment contract."""

    root_qpos = tuple(float(value) for value in scene_backend._read_root_qpos()[:7])
    motor_positions = tuple(float(value) for value in scene_backend._read_motor_positions())
    motor_velocities = tuple(float(value) for value in scene_backend._read_motor_velocities())
    base_angular_velocity = tuple(
        float(value)
        for value in scene_backend._read_base_angular_velocity()
    )
    return G1RobotState(
        root_qpos=root_qpos,
        motor_positions_mujoco=motor_positions,
        motor_velocities_mujoco=motor_velocities,
        base_angular_velocity=base_angular_velocity,  # type: ignore[arg-type]
        last_action_isaaclab=(
            tuple(float(value) for value in last_action_isaaclab)
            if last_action_isaaclab is not None
            else scene_backend.previous_action
        ),
    )


def robot_state_to_planner_qpos(state: G1RobotState) -> tuple[float, ...]:
    """Convert backend-neutral state into the planner's 36D MuJoCo qpos row."""

    return state.root_qpos + state.motor_positions_mujoco


def robot_state_to_sonic_history_frame(state: G1RobotState) -> SonicG1HistoryFrame:
    """Convert backend-neutral state into one official decoder history frame."""

    body_q, body_dq = mujoco_motor_state_to_sonic_body_state(
        state.motor_positions_mujoco,
        state.motor_velocities_mujoco,
    )
    return SonicG1HistoryFrame(
        base_ang_vel=state.base_angular_velocity,
        body_q=body_q,
        body_dq=body_dq,
        last_action=state.last_action_isaaclab,
        base_quat=state.base_quat,
    )


def _coerce_vector(values: Sequence[float], expected_dim: int, name: str) -> tuple[float, ...]:
    if len(values) != expected_dim:
        raise ValueError(f"{name} expected dim={expected_dim}, got {len(values)}")
    vector = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} contains a non-finite value")
    return vector


def _coerce_xyz(values: Sequence[float], name: str) -> tuple[float, float, float]:
    vector = _coerce_vector(values, 3, name)
    return vector  # type: ignore[return-value]
