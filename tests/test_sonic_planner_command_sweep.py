from __future__ import annotations

import pytest

from h200_locomotion_lab.tools.sonic_planner_command_sweep import (
    direction_by_name,
    format_float_tag,
    parse_csv_floats,
    summarize_motion,
)


def test_parse_csv_floats() -> None:
    assert parse_csv_floats("-1.0, 0.5") == pytest.approx((-1.0, 0.5))


def test_direction_by_name() -> None:
    assert direction_by_name("negx") == (-1.0, 0.0, 0.0)


def test_format_float_tag() -> None:
    assert format_float_tag(-0.5) == "neg0p5"


def test_summarize_motion_reports_mean_velocity() -> None:
    summary = summarize_motion(
        (
            (0.0, 0.0, 0.8),
            (0.02, 0.0, 0.82),
            (0.04, 0.0, 0.84),
        )
    )

    assert summary["duration_s"] == pytest.approx(0.04)
    assert summary["root_delta_xyz"] == pytest.approx([0.04, 0.0, 0.04])
    assert summary["root_delta_xy_per_s"] == pytest.approx([1.0, 0.0])
    assert summary["root_z_mean"] == pytest.approx(0.82)
