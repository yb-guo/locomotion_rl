from pathlib import Path

import pytest

from h200_locomotion_lab.tools.genesis_action_replay_smoke import (
    action_range,
    build_action_fixture,
    clip_action,
    count_out_of_range_actions,
    load_action_sequence,
    read_action_csv,
    read_default_joint_positions,
)


def test_build_sine_fixture_has_expected_shape_and_range() -> None:
    rows = build_action_fixture("sine", frames=5, action_dim=29, amplitude=0.2)

    assert len(rows) == 5
    assert all(len(row) == 29 for row in rows)
    assert action_range(rows)[2] <= 0.2


def test_build_zero_fixture_is_deterministic() -> None:
    rows = build_action_fixture("zero", frames=3, action_dim=4, amplitude=0.5)

    assert rows == [(0.0, 0.0, 0.0, 0.0)] * 3


def test_build_fixture_rejects_invalid_frame_count() -> None:
    with pytest.raises(ValueError, match="frames must be positive"):
        build_action_fixture("sine", frames=0, action_dim=29, amplitude=0.1)


def test_read_action_csv_with_header() -> None:
    path = _fixture_path("actions_with_header.csv")

    rows = read_action_csv(path, 29)

    assert rows == [(0.1,) * 29]


def test_read_action_csv_without_header() -> None:
    path = _fixture_path("actions_without_header.csv")

    rows = read_action_csv(path, 29)

    assert rows[0][0] == 0.0
    assert rows[0][-1] == 0.28


def test_read_action_csv_rejects_wrong_width() -> None:
    path = _fixture_path("actions_wrong_width.csv")

    with pytest.raises(ValueError, match="expected 29 action columns"):
        read_action_csv(path, 29)


def test_load_action_sequence_limits_csv_frames() -> None:
    path = _fixture_path("actions_limit_frames.csv")

    rows = load_action_sequence(
        actions_csv=path,
        fixture="sine",
        frames=2,
        action_dim=29,
        amplitude=0.1,
    )

    assert len(rows) == 2


def test_read_default_joint_positions_selects_csv_row() -> None:
    path = _fixture_path("actions_limit_frames.csv")

    row = read_default_joint_positions(path, row_index=1, action_dim=29)

    assert row == (0.1,) * 29


def test_clip_and_out_of_range_count() -> None:
    actions = [(-2.0, -1.0, 0.0, 1.0, 2.0)]

    assert clip_action(actions[0]) == (-1.0, -1.0, 0.0, 1.0, 1.0)
    assert count_out_of_range_actions(actions) == 2


def _fixture_path(name: str) -> Path:
    return Path(__file__).with_name("fixtures") / name
