from __future__ import annotations

import math

import pytest

from h200_locomotion_lab.tools.mjlab_sonic_alignment_trace import (
    percentile,
    quat_to_rpy,
    summarize_alignment_trace,
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
            "pitch": 0.1,
            "roll": 0.0,
            "joint_error_rms": 0.2,
            "raw_action_absmax": 0.3,
            "mjlab_action_absmax": 0.4,
            "joint_error": [0.0, 0.2, 0.4],
            "encoder_field_norms": {"filled": 1.0, "zero": 0.0},
            "planner_root_z": 0.78,
        },
        {
            "root_xyz": [1.0, 0.0, 0.7],
            "pitch": -0.2,
            "roll": 0.1,
            "joint_error_rms": 0.3,
            "raw_action_absmax": 0.5,
            "mjlab_action_absmax": 0.6,
            "joint_error": [0.0, 0.1, 0.8],
            "encoder_field_norms": {"filled": 2.0, "zero": 0.0},
            "planner_root_z": 0.76,
        },
    ]

    summary = summarize_alignment_trace(rows, done_steps=[1], joint_names=("a", "b", "c"))

    assert summary["done_steps"] == [1]
    assert summary["root_z_final"] == pytest.approx(0.7)
    assert summary["root_delta_xyz"] == pytest.approx([1.0, 0.0, -0.1])
    assert summary["top_joint_error_rms"][0]["joint"] == "c"
    assert summary["encoder_zero_fields_last"] == ["zero"]
    assert summary["root_minus_planner_z_mean"] == pytest.approx(-0.02)


def test_zero_fields_uses_epsilon() -> None:
    assert zero_fields({"a": 0.0, "b": 1.0e-10, "c": 1.0e-4}) == ["a", "b"]
