from __future__ import annotations

import math

import pytest

from h200_locomotion_lab.tools.mjlab_sonic_alignment_trace import (
    SoftLimitClampedMjlabG1RobotBackend,
    clamp_joint_targets,
    joint_limit_margins,
    planner_root_velocity,
    percentile,
    quat_to_rpy,
    summarize_alignment_trace,
    top_joint_fraction_above,
    top_joint_fraction_below,
    zero_fields,
)
from h200_locomotion_lab.envs.genesis_adapter import G1_29DOF_JOINT_ORDER
from h200_locomotion_lab.envs.robot_backend import G1MotorCommand
from h200_locomotion_lab.runtime import ScalarActionBridge


class FakeTraceData:
    def __init__(self) -> None:
        self.root_link_pos_w = [[0.0, 0.0, 0.8]]
        self.root_link_quat_w = [[1.0, 0.0, 0.0, 0.0]]
        self.root_link_ang_vel_b = [[0.0, 0.0, 0.0]]
        self.joint_pos = [[0.0] * 29]
        self.joint_vel = [[0.0] * 29]
        self.soft_joint_pos_limits = [[[-1.0, 1.0] for _ in range(29)]]


class FakeTraceRobot:
    def __init__(self) -> None:
        self.joint_names = G1_29DOF_JOINT_ORDER
        self.data = FakeTraceData()


class FakeTraceActionTerm:
    def __init__(self) -> None:
        self.target_names = list(G1_29DOF_JOINT_ORDER)
        self.scale = [[1.0] * 29]
        self.offset = [[0.0] * 29]
        self.raw_action = None


class FakeTraceActionManager:
    def __init__(self) -> None:
        self.action_term = FakeTraceActionTerm()

    def get_term(self, name: str) -> FakeTraceActionTerm:
        assert name == "joint_pos"
        return self.action_term


class FakeTraceEnv:
    def __init__(self) -> None:
        self.robot = FakeTraceRobot()
        self.scene = {"robot": self.robot}
        self.action_manager = FakeTraceActionManager()


def identity_bridge() -> ScalarActionBridge:
    return ScalarActionBridge(
        action_dim=29,
        command_to_policy=tuple(range(29)),
        default_angles_command=(0.0,) * 29,
        action_scale_command=(1.0,) * 29,
    )


def test_quat_to_rpy_identity() -> None:
    assert quat_to_rpy((1.0, 0.0, 0.0, 0.0)) == pytest.approx((0.0, 0.0, 0.0))


def test_quat_to_rpy_pitch() -> None:
    angle = 0.25
    quat = (math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0)
    roll, pitch, yaw = quat_to_rpy(quat)
    assert roll == pytest.approx(0.0)
    assert pitch == pytest.approx(angle)
    assert yaw == pytest.approx(0.0)


def test_percentile_interpolates() -> None:
    assert percentile([0.0, 10.0, 20.0], 95.0) == pytest.approx(19.0)


def test_summarize_alignment_trace_reports_joint_error_ranking() -> None:
    rows = [
        {
            "root_xyz": [0.0, 0.0, 0.8],
            "root_lin_vel_b": [0.4, 0.1, 0.0],
            "pitch": 0.1,
            "roll": 0.0,
            "joint_error_rms": 0.2,
            "raw_action_absmax": 0.3,
            "mjlab_action_absmax": 0.4,
            "joint_error": [0.0, 0.2, 0.4],
            "raw_action_command_order": [0.0, 1.0, -2.0],
            "effective_action_command_order": [0.0, 0.5, -1.0],
            "effective_action_delta_command_order": [0.0, -0.5, 1.0],
            "effective_action_delta_absmax": 1.0,
            "encoder_field_norms": {"filled": 1.0, "zero": 0.0},
            "planner_root_z": 0.78,
            "planner_root_vel_xyz": [0.5, 0.0, 0.0],
            "mjlab_twist_command": [0.2, 0.0, 0.0],
            "actuator_force": [0.0, 2.0, -4.0],
            "actuator_force_utilization": [0.0, 0.5, 1.0],
            "actual_soft_limit_margin": [0.3, 0.2, 0.1],
            "target_soft_limit_margin": [0.2, 0.1, -0.1],
            "raw_target_soft_limit_margin": [0.2, -0.1, -0.3],
            "target_clip_delta": [0.0, 0.1, 0.3],
            "target_clip_rms": math.sqrt((0.0 + 0.01 + 0.09) / 3.0),
            "target_clip_absmax": 0.3,
            "raw_target": [0.0, 0.9, 1.4],
            "target": [0.0, 0.9, 1.0],
            "soft_joint_pos_limits": [[-1.0, 1.0], [0.0, 1.0], [-1.0, 1.0]],
            "foot_contact_force_norm": [10.0, 20.0],
        },
        {
            "root_xyz": [1.0, 0.0, 0.7],
            "root_lin_vel_b": [0.6, 0.1, 0.0],
            "pitch": -0.2,
            "roll": 0.1,
            "joint_error_rms": 0.3,
            "raw_action_absmax": 0.5,
            "mjlab_action_absmax": 0.6,
            "joint_error": [0.0, 0.1, 0.8],
            "raw_action_command_order": [0.0, 0.5, -3.0],
            "effective_action_command_order": [0.0, 0.5, -1.0],
            "effective_action_delta_command_order": [0.0, 0.0, 2.0],
            "effective_action_delta_absmax": 2.0,
            "encoder_field_norms": {"filled": 2.0, "zero": 0.0},
            "planner_root_z": 0.76,
            "planner_root_vel_xyz": [0.7, 0.0, 0.0],
            "mjlab_twist_command": [0.2, 0.0, 0.0],
            "actuator_force": [0.0, 1.0, -8.0],
            "actuator_force_utilization": [0.0, 0.25, 2.0],
            "actual_soft_limit_margin": [0.4, 0.1, 0.0],
            "target_soft_limit_margin": [0.3, 0.2, -0.2],
            "raw_target_soft_limit_margin": [0.3, 0.2, -0.4],
            "target_clip_delta": [0.0, 0.0, 0.4],
            "target_clip_rms": math.sqrt(0.16 / 3.0),
            "target_clip_absmax": 0.4,
            "raw_target": [0.0, 0.8, 1.5],
            "target": [0.0, 0.8, 1.0],
            "soft_joint_pos_limits": [[-1.0, 1.0], [0.0, 1.0], [-1.0, 1.0]],
            "foot_contact_force_norm": [30.0, 40.0],
        },
    ]

    summary = summarize_alignment_trace(rows, done_steps=[1], joint_names=("a", "b", "c"))

    assert summary["done_steps"] == [1]
    assert summary["root_z_final"] == pytest.approx(0.7)
    assert summary["root_delta_xyz"] == pytest.approx([1.0, 0.0, -0.1])
    assert summary["root_delta_xy_per_s"] == pytest.approx([25.0, 0.0])
    assert summary["planner_root_vel_x_mean"] == pytest.approx(0.6)
    assert summary["root_lin_vel_b_x_mean"] == pytest.approx(0.5)
    assert summary["mjlab_twist_command_x_mean"] == pytest.approx(0.2)
    assert summary["top_joint_error_rms"][0]["joint"] == "c"
    assert summary["top_joint_actuator_force_abs_max"][0] == {"joint": "c", "value": 8.0}
    assert summary["top_joint_force_saturation_fraction"][0] == {
        "joint": "c",
        "value": 1.0,
    }
    assert summary["top_joint_target_soft_limit_margin_min"][0] == {
        "joint": "c",
        "value": -0.2,
    }
    assert summary["top_joint_target_soft_limit_violation_fraction"][0] == {
        "joint": "c",
        "value": 1.0,
    }
    assert summary["top_joint_raw_target_soft_limit_violation_fraction"][0] == {
        "joint": "c",
        "value": 1.0,
    }
    assert summary["target_clip_absmax_max"] == pytest.approx(0.4)
    assert summary["top_joint_target_clip_absmax"][0] == {"joint": "c", "value": 0.4}
    assert summary["effective_action_delta_absmax_max"] == pytest.approx(2.0)
    assert summary["top_joint_effective_action_delta_absmax"][0] == {
        "joint": "c",
        "value": 2.0,
    }
    assert summary["top_joint_target_range_vs_soft_limits"][0] == {
        "joint": "c",
        "soft_low": -1.0,
        "soft_high": 1.0,
        "raw_target_min": 1.4,
        "raw_target_max": 1.5,
        "target_min": 1.0,
        "target_max": 1.0,
        "raw_violation_absmax": 0.5,
        "target_violation_absmax": 0.0,
    }
    assert summary["top_joint_actual_soft_limit_violation_fraction"][0] == {
        "joint": "a",
        "value": 0.0,
    }
    assert summary["foot_contact_force_norm_mean"] == pytest.approx([20.0, 30.0])
    assert summary["encoder_zero_fields_last"] == ["zero"]
    assert summary["root_minus_planner_z_mean"] == pytest.approx(-0.02)


def test_zero_fields_uses_epsilon() -> None:
    assert zero_fields({"a": 0.0, "b": 1.0e-10, "c": 1.0e-4}) == ["a", "b"]


def test_planner_root_velocity_uses_next_frame() -> None:
    velocity = planner_root_velocity(((0.0, 0.0, 0.8), (0.02, 0.04, 0.8)), 0)

    assert velocity == pytest.approx((1.0, 2.0, 0.0))


def test_joint_limit_margins() -> None:
    assert joint_limit_margins((0.0, 0.8), ((-1.0, 1.0), (0.0, 1.0))) == pytest.approx(
        (1.0, 0.2)
    )


def test_clamp_joint_targets() -> None:
    assert clamp_joint_targets(
        (-2.0, 0.5, 2.0),
        ((-1.0, 1.0), (0.0, 1.0), (-1.0, 1.0)),
    ) == pytest.approx((-1.0, 0.5, 1.0))


def test_clamped_backend_preserves_raw_action_history_by_default() -> None:
    env = FakeTraceEnv()
    env.robot.data.soft_joint_pos_limits[0][0] = [-0.5, 0.5]
    backend = SoftLimitClampedMjlabG1RobotBackend(env, action_bridge=identity_bridge())
    raw_action = (2.0,) + (0.0,) * 28
    raw_targets = (2.0,) + (0.0,) * 28

    backend.write_command(
        G1MotorCommand(
            raw_action_isaaclab=raw_action,
            motor_position_targets_mujoco=raw_targets,
        )
    )

    assert backend._last_command.raw_action_isaaclab[0] == pytest.approx(2.0)
    assert backend._last_command.motor_position_targets_mujoco[0] == pytest.approx(0.5)
    assert backend.read_state().last_action_isaaclab[0] == pytest.approx(2.0)


def test_clamped_backend_can_use_effective_action_history() -> None:
    env = FakeTraceEnv()
    env.robot.data.soft_joint_pos_limits[0][0] = [-0.5, 0.5]
    backend = SoftLimitClampedMjlabG1RobotBackend(
        env,
        action_bridge=identity_bridge(),
        history_action_source="effective",
    )
    raw_action = (2.0,) + (0.0,) * 28
    raw_targets = (2.0,) + (0.0,) * 28

    backend.write_command(
        G1MotorCommand(
            raw_action_isaaclab=raw_action,
            motor_position_targets_mujoco=raw_targets,
        )
    )

    assert backend._last_command.raw_action_isaaclab[0] == pytest.approx(0.5)
    assert backend._last_command.motor_position_targets_mujoco[0] == pytest.approx(0.5)
    assert backend.read_state().last_action_isaaclab[0] == pytest.approx(0.5)


def test_clamped_backend_can_use_official_ankle_pitch_hard_limits() -> None:
    env = FakeTraceEnv()
    left_ankle_pitch_index = G1_29DOF_JOINT_ORDER.index("left_ankle_pitch_joint")
    env.robot.data.soft_joint_pos_limits[0][left_ankle_pitch_index] = [-0.5, 0.5]
    backend = SoftLimitClampedMjlabG1RobotBackend(
        env,
        action_bridge=identity_bridge(),
        clamp_limit_source="official-g1-hard-ankle-pitch",
    )
    raw_action = (0.0,) * 29
    raw_targets = [0.0] * 29
    raw_targets[left_ankle_pitch_index] = 1.0

    backend.write_command(
        G1MotorCommand(
            raw_action_isaaclab=raw_action,
            motor_position_targets_mujoco=tuple(raw_targets),
        )
    )

    assert backend._last_command.motor_position_targets_mujoco[
        left_ankle_pitch_index
    ] == pytest.approx(0.5236)


def test_top_joint_fraction_above() -> None:
    scored = top_joint_fraction_above(
        ((0.0, 1.0), (0.5, 0.2)),
        ("a", "b"),
        threshold=0.9,
    )

    assert scored[0] == {"joint": "b", "value": 0.5}


def test_top_joint_fraction_below() -> None:
    scored = top_joint_fraction_below(
        ((0.0, -0.1), (-0.2, 0.2)),
        ("a", "b"),
        threshold=0.0,
    )

    assert scored[0] == {"joint": "a", "value": 0.5}
