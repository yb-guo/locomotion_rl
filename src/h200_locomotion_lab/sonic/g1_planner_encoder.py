"""SONIC G1 planner-to-encoder observation bridge.

The official planner emits MuJoCo qpos frames at 30 Hz. The official deploy
path resamples those frames to a 50 Hz MotionSequence, then gathers the encoder
input from future motion frames. This module implements the G1 encoder mode 0
slice of that path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from h200_locomotion_lab.sonic.g1_policy_bridge import (
    SONIC_ACTION_DIM,
    SONIC_G1_DEFAULT_ANGLES,
    SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX,
)

SONIC_PLANNER_QPOS_DIM = 7 + SONIC_ACTION_DIM
SONIC_PLANNER_CONTEXT_FRAMES = 4
SONIC_PLANNER_PRED_FRAMES = 64
SONIC_PLANNER_ALLOWED_PRED_NUM_TOKENS: tuple[int, ...] = (
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
)
SONIC_PLANNER_DEFAULT_HEIGHT = 0.788740
SONIC_PLANNER_DEFAULT_RANDOM_SEED = 1234

SONIC_ENCODER_OBS_DIM = 1762
SONIC_ENCODER_MODE_G1 = 0


@dataclass(frozen=True)
class SonicEncoderField:
    name: str
    offset: int
    dim: int
    required_for_g1: bool


SONIC_G1_ENCODER_FIELDS: tuple[SonicEncoderField, ...] = (
    SonicEncoderField("encoder_mode_4", 0, 4, True),
    SonicEncoderField("motion_joint_positions_10frame_step5", 4, 290, True),
    SonicEncoderField("motion_joint_velocities_10frame_step5", 294, 290, True),
    SonicEncoderField("motion_root_z_position_10frame_step5", 584, 10, False),
    SonicEncoderField("motion_root_z_position", 594, 1, False),
    SonicEncoderField("motion_anchor_orientation", 595, 6, False),
    SonicEncoderField("motion_anchor_orientation_10frame_step5", 601, 60, True),
    SonicEncoderField("motion_joint_positions_lowerbody_10frame_step5", 661, 120, False),
    SonicEncoderField("motion_joint_velocities_lowerbody_10frame_step5", 781, 120, False),
    SonicEncoderField("vr_3point_local_target", 901, 9, False),
    SonicEncoderField("vr_3point_local_orn_target", 910, 12, False),
    SonicEncoderField("smpl_joints_10frame_step1", 922, 720, False),
    SonicEncoderField("smpl_anchor_orientation_10frame_step1", 1642, 60, False),
    SonicEncoderField("motion_joint_positions_wrists_10frame_step1", 1702, 60, False),
)


@dataclass(frozen=True)
class SonicPlannerInputs:
    context_mujoco_qpos: tuple[tuple[float, ...], ...]
    target_vel: float
    mode: int
    movement_direction: tuple[float, float, float]
    facing_direction: tuple[float, float, float]
    random_seed: int
    has_specific_target: int
    specific_target_positions: tuple[tuple[float, float, float], ...]
    specific_target_headings: tuple[float, ...]
    allowed_pred_num_tokens: tuple[int, ...]
    height: float


@dataclass(frozen=True)
class SonicPlannerMotion50Hz:
    root_positions: tuple[tuple[float, float, float], ...]
    root_quats: tuple[tuple[float, float, float, float], ...]
    joint_positions_policy_order: tuple[tuple[float, ...], ...]
    joint_velocities_policy_order: tuple[tuple[float, ...], ...]

    @property
    def timesteps(self) -> int:
        return len(self.root_positions)


def build_initial_planner_context(
    joint_positions_mujoco: Sequence[float] = SONIC_G1_DEFAULT_ANGLES,
    *,
    root_height: float = SONIC_PLANNER_DEFAULT_HEIGHT,
) -> tuple[tuple[float, ...], ...]:
    """Build official-style 4-frame standing planner context in MuJoCo order."""

    joints = _coerce_vector(joint_positions_mujoco, SONIC_ACTION_DIM, "joint_positions_mujoco")
    frame = (0.0, 0.0, float(root_height), 1.0, 0.0, 0.0, 0.0, *joints)
    return tuple(frame for _ in range(SONIC_PLANNER_CONTEXT_FRAMES))


def build_planner_inputs(
    joint_positions_mujoco: Sequence[float] = SONIC_G1_DEFAULT_ANGLES,
    *,
    mode: int = 2,
    target_vel: float = -1.0,
    movement_direction: Sequence[float] = (1.0, 0.0, 0.0),
    facing_direction: Sequence[float] = (1.0, 0.0, 0.0),
    random_seed: int = SONIC_PLANNER_DEFAULT_RANDOM_SEED,
    height: float = -1.0,
    root_height: float = SONIC_PLANNER_DEFAULT_HEIGHT,
) -> SonicPlannerInputs:
    """Build planner inputs matching official LocalMotionPlannerONNX defaults."""

    return SonicPlannerInputs(
        context_mujoco_qpos=build_initial_planner_context(
            joint_positions_mujoco,
            root_height=root_height,
        ),
        target_vel=float(target_vel),
        mode=int(mode),
        movement_direction=_coerce_xyz(movement_direction, "movement_direction"),
        facing_direction=_coerce_xyz(facing_direction, "facing_direction"),
        random_seed=int(random_seed),
        has_specific_target=0,
        specific_target_positions=((0.0, 0.0, 0.0),) * SONIC_PLANNER_CONTEXT_FRAMES,
        specific_target_headings=(0.0,) * SONIC_PLANNER_CONTEXT_FRAMES,
        allowed_pred_num_tokens=SONIC_PLANNER_ALLOWED_PRED_NUM_TOKENS,
        height=float(height),
    )


def resample_planner_mujoco_qpos_to_50hz(
    mujoco_qpos_frames_30hz: Sequence[Sequence[float]],
    *,
    num_pred_frames: int | None = None,
) -> SonicPlannerMotion50Hz:
    """Resample planner qpos output from 30 Hz MuJoCo order to 50 Hz policy order."""

    frames = tuple(
        _coerce_vector(frame, SONIC_PLANNER_QPOS_DIM, "mujoco_qpos_frame")
        for frame in mujoco_qpos_frames_30hz
    )
    if not frames:
        raise ValueError("mujoco_qpos_frames_30hz must not be empty")
    pred_frames = len(frames) if num_pred_frames is None else int(num_pred_frames)
    if pred_frames <= 0:
        raise ValueError("num_pred_frames must be positive")
    pred_frames = min(pred_frames, len(frames))

    motion_seconds = pred_frames / 30.0
    timesteps_50hz = math.floor(motion_seconds * 50.0)
    if timesteps_50hz < 2:
        raise ValueError("planner output is too short to derive 50 Hz velocities")

    root_positions: list[tuple[float, float, float]] = []
    root_quats: list[tuple[float, float, float, float]] = []
    joint_positions_policy_order: list[tuple[float, ...]] = []

    for frame_50hz in range(timesteps_50hz):
        frame_30hz = (frame_50hz / 50.0) * 30.0
        f0 = math.floor(frame_30hz)
        f0 = min(f0, pred_frames - 1)
        f1 = min(f0 + 1, pred_frames - 1)
        alpha = frame_30hz - f0
        w0 = 1.0 - alpha
        w1 = alpha
        q0 = frames[f0]
        q1 = frames[f1]

        root_positions.append(
            (
                w0 * q0[0] + w1 * q1[0],
                w0 * q0[1] + w1 * q1[1],
                w0 * q0[2] + w1 * q1[2],
            )
        )
        root_quats.append(_quat_slerp(q0[3:7], q1[3:7], alpha))
        joint_positions_policy_order.append(
            tuple(
                w0 * q0[7 + mujoco_index] + w1 * q1[7 + mujoco_index]
                for mujoco_index in SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX
            )
        )

    joint_velocities_policy_order = [
        tuple(
            (
                joint_positions_policy_order[index + 1][joint]
                - joint_positions_policy_order[index][joint]
            )
            * 50.0
            for joint in range(SONIC_ACTION_DIM)
        )
        for index in range(timesteps_50hz - 1)
    ]
    joint_velocities_policy_order.append(joint_velocities_policy_order[-1])

    return SonicPlannerMotion50Hz(
        root_positions=tuple(root_positions),
        root_quats=tuple(root_quats),
        joint_positions_policy_order=tuple(joint_positions_policy_order),
        joint_velocities_policy_order=tuple(joint_velocities_policy_order),
    )


def build_g1_encoder_observation_from_planner_motion(
    motion: SonicPlannerMotion50Hz,
    *,
    current_frame: int = 0,
    robot_base_quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    encoder_mode: int = SONIC_ENCODER_MODE_G1,
) -> tuple[float, ...]:
    """Build the 1762D encoder input for official G1 encoder mode 0."""

    if motion.timesteps <= 0:
        raise ValueError("motion must contain at least one timestep")
    base_quat = _coerce_quat(robot_base_quat, "robot_base_quat")
    obs = [0.0] * SONIC_ENCODER_OBS_DIM
    _assign(obs, "encoder_mode_4", (float(encoder_mode), 0.0, 0.0, 0.0))
    _assign(
        obs,
        "motion_joint_positions_10frame_step5",
        _future_joint_window(motion.joint_positions_policy_order, current_frame, 10, 5),
    )
    _assign(
        obs,
        "motion_joint_velocities_10frame_step5",
        _future_joint_window(motion.joint_velocities_policy_order, current_frame, 10, 5),
    )
    _assign(
        obs,
        "motion_anchor_orientation_10frame_step5",
        _future_anchor_orientation_window(motion.root_quats, current_frame, 10, 5, base_quat),
    )
    return tuple(obs)


def build_planner_context_from_motion(
    motion: SonicPlannerMotion50Hz,
    *,
    gen_frame: int,
    motion_look_ahead_steps: int = 2,
) -> tuple[tuple[float, ...], ...]:
    """Build official-style planner context from the current 50 Hz planner motion."""

    if motion.timesteps <= 0:
        raise ValueError("motion must contain at least one timestep")
    if motion_look_ahead_steps < 0:
        raise ValueError("motion_look_ahead_steps must be non-negative")

    context_rows: list[tuple[float, ...]] = []
    gen_time = (int(gen_frame) + int(motion_look_ahead_steps)) / 50.0
    for context_index in range(SONIC_PLANNER_CONTEXT_FRAMES):
        sample_time = gen_time + context_index / 30.0
        sample = _sample_motion_at_time(motion, sample_time)
        row = [0.0] * SONIC_PLANNER_QPOS_DIM
        row[0:3] = sample[0]
        row[3:7] = sample[1]
        for policy_index, mujoco_index in enumerate(SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX):
            row[7 + mujoco_index] = sample[2][policy_index]
        context_rows.append(tuple(row))
    return tuple(context_rows)


def build_planner_context_from_mujoco_qpos_history(
    qpos_frames_50hz: Sequence[Sequence[float]],
    *,
    current_frame: int | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Build planner context from live MuJoCo-order qpos frames sampled at 50 Hz."""

    frames = tuple(
        _coerce_vector(frame, SONIC_PLANNER_QPOS_DIM, "qpos_frame")
        for frame in qpos_frames_50hz
    )
    if not frames:
        raise ValueError("qpos_frames_50hz must not be empty")
    last_frame = len(frames) - 1 if current_frame is None else int(current_frame)
    last_frame = _clamp_frame(last_frame, len(frames))
    end_time = last_frame / 50.0
    start_time = end_time - (SONIC_PLANNER_CONTEXT_FRAMES - 1) / 30.0
    return tuple(
        _sample_mujoco_qpos_history_at_time(frames, start_time + context_index / 30.0)
        for context_index in range(SONIC_PLANNER_CONTEXT_FRAMES)
    )


def encoder_field_by_name(name: str) -> SonicEncoderField:
    for field in SONIC_G1_ENCODER_FIELDS:
        if field.name == name:
            return field
    raise KeyError(name)


def _sample_motion_at_time(
    motion: SonicPlannerMotion50Hz,
    sample_time_s: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
    tuple[float, ...],
]:
    frame_50hz = max(0.0, sample_time_s * 50.0)
    f0 = _clamp_frame(math.floor(frame_50hz), motion.timesteps)
    f1 = _clamp_frame(f0 + 1, motion.timesteps)
    alpha = max(0.0, min(1.0, frame_50hz - f0))
    w0 = 1.0 - alpha
    w1 = alpha
    root_pos = tuple(
        w0 * motion.root_positions[f0][axis] + w1 * motion.root_positions[f1][axis]
        for axis in range(3)
    )
    root_quat = _quat_slerp(motion.root_quats[f0], motion.root_quats[f1], alpha)
    joint_positions = tuple(
        w0 * motion.joint_positions_policy_order[f0][joint]
        + w1 * motion.joint_positions_policy_order[f1][joint]
        for joint in range(SONIC_ACTION_DIM)
    )
    return root_pos, root_quat, joint_positions


def _sample_mujoco_qpos_history_at_time(
    qpos_frames_50hz: Sequence[tuple[float, ...]],
    sample_time_s: float,
) -> tuple[float, ...]:
    frame_50hz = max(0.0, sample_time_s * 50.0)
    f0 = _clamp_frame(math.floor(frame_50hz), len(qpos_frames_50hz))
    f1 = _clamp_frame(f0 + 1, len(qpos_frames_50hz))
    alpha = max(0.0, min(1.0, frame_50hz - f0))
    w0 = 1.0 - alpha
    w1 = alpha
    q0 = qpos_frames_50hz[f0]
    q1 = qpos_frames_50hz[f1]
    root_position = tuple(w0 * q0[index] + w1 * q1[index] for index in range(3))
    root_quat = _quat_slerp(q0[3:7], q1[3:7], alpha)
    joints = tuple(
        w0 * q0[7 + joint_index] + w1 * q1[7 + joint_index]
        for joint_index in range(SONIC_ACTION_DIM)
    )
    return (*root_position, *root_quat, *joints)


def _future_joint_window(
    rows: Sequence[Sequence[float]],
    current_frame: int,
    num_frames: int,
    step_size: int,
) -> tuple[float, ...]:
    values: list[float] = []
    for frame_index in range(num_frames):
        index = _clamp_frame(current_frame + frame_index * step_size, len(rows))
        values.extend(_coerce_vector(rows[index], SONIC_ACTION_DIM, "joint_row"))
    return tuple(values)


def _future_anchor_orientation_window(
    root_quats: Sequence[Sequence[float]],
    current_frame: int,
    num_frames: int,
    step_size: int,
    robot_base_quat: tuple[float, float, float, float],
) -> tuple[float, ...]:
    values: list[float] = []
    base_inv = _quat_conjugate(robot_base_quat)
    for frame_index in range(num_frames):
        index = _clamp_frame(current_frame + frame_index * step_size, len(root_quats))
        ref_quat = _coerce_quat(root_quats[index], "root_quat")
        values.extend(_quat_to_6d_rows(_quat_mul(base_inv, ref_quat)))
    return tuple(values)


def _assign(obs: list[float], field_name: str, values: Sequence[float]) -> None:
    field = encoder_field_by_name(field_name)
    vector = _coerce_vector(values, field.dim, field_name)
    obs[field.offset : field.offset + field.dim] = vector


def _clamp_frame(index: int, length: int) -> int:
    if length <= 0:
        raise ValueError("length must be positive")
    return max(0, min(int(index), length - 1))


def _coerce_vector(values: Sequence[float], expected_dim: int, name: str) -> tuple[float, ...]:
    if len(values) != expected_dim:
        raise ValueError(f"{name} expected dim={expected_dim}, got {len(values)}")
    return tuple(float(value) for value in values)


def _coerce_xyz(values: Sequence[float], name: str) -> tuple[float, float, float]:
    x, y, z = _coerce_vector(values, 3, name)
    return (x, y, z)


def _coerce_quat(values: Sequence[float], name: str) -> tuple[float, float, float, float]:
    quat = _coerce_vector(values, 4, name)
    return _quat_normalize(quat)


def _quat_normalize(
    quat: Sequence[float],
) -> tuple[float, float, float, float]:
    qw, qx, qy, qz = _coerce_vector(quat, 4, "quat")
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        raise ValueError("zero quaternion")
    return (qw / norm, qx / norm, qy / norm, qz / norm)


def _quat_conjugate(
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    qw, qx, qy, qz = quat
    return (qw, -qx, -qy, -qz)


def _quat_mul(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return _quat_normalize(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _quat_slerp(
    left: Sequence[float],
    right: Sequence[float],
    alpha: float,
) -> tuple[float, float, float, float]:
    q0 = _coerce_quat(left, "left")
    q1 = _coerce_quat(right, "right")
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:
        q1 = tuple(-value for value in q1)
        dot = -dot
    if dot > 0.9995:
        return _quat_normalize(tuple((1.0 - alpha) * a + alpha * b for a, b in zip(q0, q1)))

    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _quat_normalize(tuple(s0 * a + s1 * b for a, b in zip(q0, q1)))


def _quat_to_6d_rows(
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    qw, qx, qy, qz = quat
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    return (r00, r01, r10, r11, r20, r21)
