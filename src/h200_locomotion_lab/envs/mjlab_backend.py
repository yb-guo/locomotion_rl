"""Adapter from unitree_rl_mjlab ManagerBasedRlEnv to the scalar G1 backend.

The module intentionally avoids importing ``mjlab`` or ``torch`` at import time
so local tests can exercise the contract without the simulator installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from h200_locomotion_lab.envs.genesis_adapter import G1_29DOF_JOINT_ORDER
from h200_locomotion_lab.envs.robot_backend import (
    G1MotorCommand,
    G1RobotState,
)
from h200_locomotion_lab.sonic.g1_observation import SONIC_ACTION_DIM


@dataclass(frozen=True, slots=True)
class MjlabStepResult:
    """Last mjlab transition captured by the backend."""

    reward: float | None
    done: bool | None
    info: Any = None


class MjlabG1RobotBackend:
    """Expose a single-env mjlab G1 environment through `G1RobotBackend`.

    The backend accepts `G1MotorCommand` targets in SONIC MuJoCo command order
    and converts them into mjlab `JointPositionAction` inputs by inverting the
    mjlab action term:

    ``raw_action = (target_position - action_offset) / action_scale``.
    """

    def __init__(
        self,
        raw_env: Any,
        *,
        robot_name: str = "robot",
        action_name: str = "joint_pos",
        sonic_joint_order: Sequence[str] = G1_29DOF_JOINT_ORDER,
    ) -> None:
        self.raw_env = raw_env
        self.robot_name = robot_name
        self.action_name = action_name
        self.sonic_joint_order = tuple(sonic_joint_order)
        if len(self.sonic_joint_order) != SONIC_ACTION_DIM:
            raise ValueError(
                f"sonic_joint_order expected {SONIC_ACTION_DIM} joints, "
                f"got {len(self.sonic_joint_order)}"
            )
        self.robot = self.raw_env.scene[self.robot_name]
        self.action_term = self.raw_env.action_manager.get_term(self.action_name)
        self.robot_joint_names = tuple(self.robot.joint_names)
        self.action_target_names = tuple(self.action_term.target_names)
        self._validate_names()
        self._sonic_to_robot_indices = _indices_for_names(
            self.robot_joint_names,
            self.sonic_joint_order,
            "robot.joint_names",
        )
        self._sonic_to_action_indices = _indices_for_names(
            self.action_target_names,
            self.sonic_joint_order,
            "joint_pos.target_names",
        )
        self.action_scale = _read_action_vector(
            self.action_term,
            "scale",
            SONIC_ACTION_DIM,
        )
        self.action_offset = _read_action_vector(
            self.action_term,
            "offset",
            SONIC_ACTION_DIM,
        )
        self._last_command = G1MotorCommand.from_raw_sonic_action(
            (0.0,) * SONIC_ACTION_DIM,
        )
        self._pending_action: tuple[float, ...] | None = None
        self.last_step_result: MjlabStepResult | None = None

    def reset(self) -> G1RobotState:
        reset = getattr(self.raw_env, "reset", None)
        if callable(reset):
            reset()
        self._last_command = G1MotorCommand.from_raw_sonic_action(
            (0.0,) * SONIC_ACTION_DIM,
        )
        self._pending_action = None
        self.last_step_result = None
        return self.read_state()

    def read_state(self) -> G1RobotState:
        data = self.robot.data
        root_pos = _row_values(data.root_link_pos_w, 3, "root_link_pos_w")
        root_quat = _row_values(data.root_link_quat_w, 4, "root_link_quat_w")
        joint_pos = _select(
            _row_values(data.joint_pos, len(self.robot_joint_names), "joint_pos"),
            self._sonic_to_robot_indices,
        )
        joint_vel = _select(
            _row_values(data.joint_vel, len(self.robot_joint_names), "joint_vel"),
            self._sonic_to_robot_indices,
        )
        base_ang_vel = _row_values(data.root_link_ang_vel_b, 3, "root_link_ang_vel_b")
        return G1RobotState(
            root_qpos=root_pos + root_quat,
            motor_positions_mujoco=joint_pos,
            motor_velocities_mujoco=joint_vel,
            base_angular_velocity=base_ang_vel,  # type: ignore[arg-type]
            last_action_isaaclab=self._last_command.raw_action_isaaclab,
        )

    def write_command(self, command: G1MotorCommand) -> None:
        self._last_command = command
        self._pending_action = self.motor_targets_to_mjlab_action(
            command.motor_position_targets_mujoco,
        )

    def advance(self) -> G1RobotState:
        action = self._pending_action
        if action is None:
            action = (0.0,) * SONIC_ACTION_DIM
        transition = self.raw_env.step(_make_action_batch(action, self.action_term))
        self.last_step_result = _coerce_step_result(transition)
        self._pending_action = None
        return self.read_state()

    def motor_targets_to_mjlab_action(
        self,
        motor_position_targets_mujoco: Sequence[float],
    ) -> tuple[float, ...]:
        """Convert SONIC MuJoCo-order position targets to mjlab action order."""

        targets = _coerce_vector(
            motor_position_targets_mujoco,
            SONIC_ACTION_DIM,
            "motor_position_targets_mujoco",
        )
        action_by_target_index = [0.0] * SONIC_ACTION_DIM
        for sonic_index, action_index in enumerate(self._sonic_to_action_indices):
            scale = self.action_scale[action_index]
            if abs(scale) <= 1.0e-12:
                raise ValueError(f"mjlab action scale is zero for {self.action_target_names[action_index]}")
            action_by_target_index[action_index] = (
                targets[sonic_index] - self.action_offset[action_index]
            ) / scale
        return tuple(action_by_target_index)

    def mjlab_action_to_motor_targets(
        self,
        action: Sequence[float],
    ) -> tuple[float, ...]:
        """Forward action transform, useful for contract tests."""

        action_values = _coerce_vector(action, SONIC_ACTION_DIM, "action")
        targets_by_action_index = tuple(
            offset + scale * value
            for offset, scale, value in zip(self.action_offset, self.action_scale, action_values)
        )
        targets_by_sonic_index = [0.0] * SONIC_ACTION_DIM
        for sonic_index, action_index in enumerate(self._sonic_to_action_indices):
            targets_by_sonic_index[sonic_index] = targets_by_action_index[action_index]
        return tuple(targets_by_sonic_index)

    def _validate_names(self) -> None:
        missing_robot = sorted(set(self.sonic_joint_order) - set(self.robot_joint_names))
        missing_action = sorted(set(self.sonic_joint_order) - set(self.action_target_names))
        if missing_robot:
            raise ValueError(f"mjlab robot is missing SONIC joints: {missing_robot}")
        if missing_action:
            raise ValueError(f"mjlab action term is missing SONIC joints: {missing_action}")


def _indices_for_names(
    available_names: Sequence[str],
    required_names: Sequence[str],
    label: str,
) -> tuple[int, ...]:
    index_by_name = {name: index for index, name in enumerate(available_names)}
    indices = tuple(index_by_name[name] for name in required_names)
    if len(set(indices)) != len(indices):
        raise ValueError(f"{label} produced duplicate indices for required names")
    return indices


def _read_action_vector(action_term: Any, attr: str, expected_dim: int) -> tuple[float, ...]:
    if not hasattr(action_term, attr):
        raise ValueError(f"mjlab action term has no resolved {attr!r} vector")
    values = _flatten_values(getattr(action_term, attr))
    if len(values) == 1 and expected_dim > 1:
        return (values[0],) * expected_dim
    if len(values) != expected_dim:
        raise ValueError(f"mjlab action {attr} expected dim={expected_dim}, got {len(values)}")
    return values


def _row_values(values: Any, expected_dim: int, name: str) -> tuple[float, ...]:
    flat = _flatten_values(values)
    if len(flat) < expected_dim:
        raise ValueError(f"{name} expected at least {expected_dim} values, got {len(flat)}")
    return tuple(flat[:expected_dim])


def _flatten_values(values: Any) -> tuple[float, ...]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "reshape") and hasattr(values, "tolist"):
        try:
            values = values.reshape(-1).tolist()
        except TypeError:
            values = values.tolist()
    elif hasattr(values, "flatten") and hasattr(values, "tolist"):
        values = values.flatten().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        return (float(values),)
    flat: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flat.extend(float(item) for item in _flatten_values(value))
        else:
            flat.append(float(value))
    return tuple(flat)


def _select(values: Sequence[float], indices: Sequence[int]) -> tuple[float, ...]:
    return tuple(float(values[index]) for index in indices)


def _coerce_vector(values: Sequence[float], expected_dim: int, name: str) -> tuple[float, ...]:
    if len(values) != expected_dim:
        raise ValueError(f"{name} expected dim={expected_dim}, got {len(values)}")
    return tuple(float(value) for value in values)


def _make_action_batch(action: Sequence[float], action_term: Any) -> Any:
    raw_action = getattr(action_term, "raw_action", None)
    if hasattr(raw_action, "new_tensor"):
        return raw_action.new_tensor([list(action)])
    return [list(action)]


def _coerce_step_result(transition: Any) -> MjlabStepResult:
    if not isinstance(transition, tuple):
        return MjlabStepResult(reward=None, done=None, info=None)
    if len(transition) >= 4:
        _, reward, done, info = transition[:4]
        return MjlabStepResult(
            reward=_maybe_scalar(reward),
            done=_maybe_bool(done),
            info=info,
        )
    return MjlabStepResult(reward=None, done=None, info=transition)


def _maybe_scalar(value: Any) -> float | None:
    flat = _flatten_values(value)
    return flat[0] if flat else None


def _maybe_bool(value: Any) -> bool | None:
    flat = _flatten_values(value)
    if not flat:
        return None
    return bool(flat[0])
