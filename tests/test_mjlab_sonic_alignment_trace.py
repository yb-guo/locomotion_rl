from __future__ import annotations

import math

import pytest

from h200_locomotion_lab.tools.mjlab_sonic_alignment_trace import (
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
