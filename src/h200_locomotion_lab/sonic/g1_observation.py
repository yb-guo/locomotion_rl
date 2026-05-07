"""GEAR-SONIC G1 decoder observation layout and history helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

from h200_locomotion_lab.sonic.g1_policy_bridge import (
    SONIC_ACTION_DIM,
    SONIC_G1_DEFAULT_ANGLES,
    SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX,
)


SONIC_TOKEN_DIM = 64
SONIC_HISTORY_FRAMES = 10
SONIC_BASE_ANGULAR_VELOCITY_DIM = SONIC_HISTORY_FRAMES * 3
SONIC_BODY_JOINT_POSITIONS_DIM = SONIC_HISTORY_FRAMES * SONIC_ACTION_DIM
SONIC_BODY_JOINT_VELOCITIES_DIM = SONIC_HISTORY_FRAMES * SONIC_ACTION_DIM
SONIC_LAST_ACTIONS_DIM = SONIC_HISTORY_FRAMES * SONIC_ACTION_DIM
SONIC_GRAVITY_DIR_DIM = SONIC_HISTORY_FRAMES * 3
SONIC_DECODER_OBS_DIM = (
    SONIC_TOKEN_DIM
    + SONIC_BASE_ANGULAR_VELOCITY_DIM
    + SONIC_BODY_JOINT_POSITIONS_DIM
    + SONIC_BODY_JOINT_VELOCITIES_DIM
    + SONIC_LAST_ACTIONS_DIM
    + SONIC_GRAVITY_DIR_DIM
)


@dataclass(frozen=True)
class SonicObservationField:
    name: str
    offset: int
    dim: int


SONIC_G1_DECODER_OBSERVATION_FIELDS: tuple[SonicObservationField, ...] = (
    SonicObservationField("token_state", 0, SONIC_TOKEN_DIM),
    SonicObservationField(
        "his_base_angular_velocity_10frame_step1",
        64,
        SONIC_BASE_ANGULAR_VELOCITY_DIM,
    ),
    SonicObservationField(
        "his_body_joint_positions_10frame_step1",
        94,
        SONIC_BODY_JOINT_POSITIONS_DIM,
    ),
    SonicObservationField(
        "his_body_joint_velocities_10frame_step1",
        384,
        SONIC_BODY_JOINT_VELOCITIES_DIM,
    ),
    SonicObservationField("his_last_actions_10frame_step1", 674, SONIC_LAST_ACTIONS_DIM),
    SonicObservationField("his_gravity_dir_10frame_step1", 964, SONIC_GRAVITY_DIR_DIM),
)


@dataclass(frozen=True)
class SonicG1HistoryFrame:
    """One official state-logger frame in SONIC decoder order.

    body_q/body_dq are policy/IsaacLab-order values. body_q is already centered
    by subtracting official MuJoCo-order default_angles before remapping.
    last_action is the raw decoder output in policy order.
    """

    base_ang_vel: tuple[float, float, float]
    body_q: tuple[float, ...]
    body_dq: tuple[float, ...]
    last_action: tuple[float, ...]
    base_quat: tuple[float, float, float, float]
    gravity_dir: tuple[float, float, float] | None = None


class SonicG1HistoryBuffer:
    """StateLogger-compatible fixed history for decoder observations."""

    def __init__(self, maxlen: int = SONIC_HISTORY_FRAMES) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self._frames: deque[SonicG1HistoryFrame] = deque(maxlen=maxlen)

    def append(self, frame: SonicG1HistoryFrame) -> None:
        validate_sonic_g1_history_frame(frame)
        self._frames.append(frame)

    def latest_oldest_first(
        self,
        count: int = SONIC_HISTORY_FRAMES,
    ) -> tuple[SonicG1HistoryFrame, ...]:
        if count <= 0:
            raise ValueError("count must be positive")
        available = tuple(self._frames)[-count:]
        missing = count - len(available)
        if missing <= 0:
            return available
        return (zero_sonic_g1_history_frame(),) * missing + available


def build_sonic_g1_decoder_observation(
    token_state: Sequence[float],
    history_frames_oldest_first: Sequence[SonicG1HistoryFrame],
) -> tuple[float, ...]:
    """Build the official 994D G1 decoder obs from oldest-to-newest history."""

    token = _coerce_vector(token_state, SONIC_TOKEN_DIM, "token_state")
    frames = tuple(history_frames_oldest_first)
    if len(frames) != SONIC_HISTORY_FRAMES:
        raise ValueError(
            f"Expected {SONIC_HISTORY_FRAMES} history frames, got {len(frames)}"
        )
    for frame in frames:
        validate_sonic_g1_history_frame(frame)

    obs = [0.0] * SONIC_DECODER_OBS_DIM
    _assign(obs, "token_state", token)
    _assign(
        obs,
        "his_base_angular_velocity_10frame_step1",
        _flatten(frame.base_ang_vel for frame in frames),
    )
    _assign(
        obs,
        "his_body_joint_positions_10frame_step1",
        _flatten(frame.body_q for frame in frames),
    )
    _assign(
        obs,
        "his_body_joint_velocities_10frame_step1",
        _flatten(frame.body_dq for frame in frames),
    )
    _assign(
        obs,
        "his_last_actions_10frame_step1",
        _flatten(frame.last_action for frame in frames),
    )
    _assign(
        obs,
        "his_gravity_dir_10frame_step1",
        _flatten(_gravity_dir_for_frame(frame) for frame in frames),
    )
    return tuple(obs)


def sonic_g1_history_from_decoder_observation(
    observation: Sequence[float],
) -> tuple[tuple[float, ...], tuple[SonicG1HistoryFrame, ...]]:
    """Parse token and 10 history frames from an official 994D decoder obs."""

    obs = _coerce_vector(observation, SONIC_DECODER_OBS_DIM, "observation")
    token_state = _field_values(obs, "token_state")
    base_ang_vel = _field_values(obs, "his_base_angular_velocity_10frame_step1")
    body_q = _field_values(obs, "his_body_joint_positions_10frame_step1")
    body_dq = _field_values(obs, "his_body_joint_velocities_10frame_step1")
    last_actions = _field_values(obs, "his_last_actions_10frame_step1")
    gravity_dir = _field_values(obs, "his_gravity_dir_10frame_step1")
    frames = tuple(
        SonicG1HistoryFrame(
            base_ang_vel=base_ang_vel[index * 3 : index * 3 + 3],
            body_q=body_q[index * SONIC_ACTION_DIM : (index + 1) * SONIC_ACTION_DIM],
            body_dq=body_dq[index * SONIC_ACTION_DIM : (index + 1) * SONIC_ACTION_DIM],
            last_action=last_actions[index * SONIC_ACTION_DIM : (index + 1) * SONIC_ACTION_DIM],
            base_quat=(0.0, 0.0, 0.0, 0.0),
            gravity_dir=gravity_dir[index * 3 : index * 3 + 3],
        )
        for index in range(SONIC_HISTORY_FRAMES)
    )
    return token_state, frames


def mujoco_motor_state_to_sonic_body_state(
    motor_positions_mujoco: Sequence[float],
    motor_velocities_mujoco: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Convert MuJoCo-order motor state into SONIC policy-order body_q/body_dq."""

    positions = _coerce_vector(
        motor_positions_mujoco,
        SONIC_ACTION_DIM,
        "motor_positions_mujoco",
    )
    velocities = _coerce_vector(
        motor_velocities_mujoco,
        SONIC_ACTION_DIM,
        "motor_velocities_mujoco",
    )
    body_q = tuple(
        positions[mujoco_index] - SONIC_G1_DEFAULT_ANGLES[mujoco_index]
        for mujoco_index in SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX
    )
    body_dq = tuple(
        velocities[mujoco_index]
        for mujoco_index in SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX
    )
    return body_q, body_dq


def gravity_dir_from_base_quat(
    base_quat: Sequence[float],
) -> tuple[float, float, float]:
    """Rotate world gravity into the base frame using official qw/qx/qy/qz convention."""

    qw, qx, qy, qz = _coerce_vector(base_quat, 4, "base_quat")
    if qw == 0.0 and qx == 0.0 and qy == 0.0 and qz == 0.0:
        return (0.0, 0.0, 0.0)
    return _quat_rotate((qw, -qx, -qy, -qz), (0.0, 0.0, -1.0))


def validate_sonic_g1_history_frame(frame: SonicG1HistoryFrame) -> None:
    _coerce_vector(frame.base_ang_vel, 3, "base_ang_vel")
    _coerce_vector(frame.body_q, SONIC_ACTION_DIM, "body_q")
    _coerce_vector(frame.body_dq, SONIC_ACTION_DIM, "body_dq")
    _coerce_vector(frame.last_action, SONIC_ACTION_DIM, "last_action")
    _coerce_vector(frame.base_quat, 4, "base_quat")
    if frame.gravity_dir is not None:
        _coerce_vector(frame.gravity_dir, 3, "gravity_dir")


def zero_sonic_g1_history_frame() -> SonicG1HistoryFrame:
    return SonicG1HistoryFrame(
        base_ang_vel=(0.0, 0.0, 0.0),
        body_q=(0.0,) * SONIC_ACTION_DIM,
        body_dq=(0.0,) * SONIC_ACTION_DIM,
        last_action=(0.0,) * SONIC_ACTION_DIM,
        base_quat=(0.0, 0.0, 0.0, 0.0),
    )


def field_by_name(name: str) -> SonicObservationField:
    for field in SONIC_G1_DECODER_OBSERVATION_FIELDS:
        if field.name == name:
            return field
    raise KeyError(name)


def _assign(obs: list[float], field_name: str, values: Sequence[float]) -> None:
    field = field_by_name(field_name)
    vector = _coerce_vector(values, field.dim, field_name)
    obs[field.offset : field.offset + field.dim] = vector


def _field_values(obs: Sequence[float], field_name: str) -> tuple[float, ...]:
    field = field_by_name(field_name)
    return tuple(obs[field.offset : field.offset + field.dim])


def _gravity_dir_for_frame(frame: SonicG1HistoryFrame) -> tuple[float, float, float]:
    if frame.gravity_dir is not None:
        return frame.gravity_dir
    return gravity_dir_from_base_quat(frame.base_quat)


def _coerce_vector(values: Sequence[float], expected_dim: int, name: str) -> tuple[float, ...]:
    if len(values) != expected_dim:
        raise ValueError(f"{name} expected dim={expected_dim}, got {len(values)}")
    return tuple(float(value) for value in values)


def _flatten(rows: Iterable[Sequence[float]]) -> tuple[float, ...]:
    return tuple(value for row in rows for value in row)


def _quat_rotate(
    quat: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    qw, qx, qy, qz = quat
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )
