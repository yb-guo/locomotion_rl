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
from h200_locomotion_lab.tools.sonic_reference_replay_smoke import (
    _horizontal_distance,
    _read_contact_metrics,
    _read_floating_base_position,
    _xy_path_length,
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


def test_read_floating_base_position_prefers_root_qpos() -> None:
    robot = _FakeFloatingBaseRobot()

    assert _read_floating_base_position(robot, motor_dof_indices=(6, 7)) == (0.11, 0.22, 0.79)


def test_read_floating_base_position_falls_back_to_spawn_pose() -> None:
    robot = _FakeFloatingBaseRobot()

    assert _read_floating_base_position(robot, motor_dof_indices=(0, 1)) == (9.0, 8.0, 7.0)


def test_read_contact_metrics_uses_genesis_contact_apis() -> None:
    robot = _FakeContactRobot()

    assert _read_contact_metrics(robot) == (2, 5.0)


def test_xy_root_motion_helpers_report_displacement_and_path() -> None:
    positions = (
        (0.0, 0.0, 0.8),
        (0.03, 0.04, 0.79),
        (0.06, 0.08, 0.78),
    )

    assert _horizontal_distance(positions[0], positions[-1]) == pytest.approx(0.1)
    assert _xy_path_length(positions) == pytest.approx(0.1)


def _fixture_path(name: str) -> Path:
    return Path(__file__).with_name("fixtures") / name


class _FakeFloatingBaseRobot:
    n_dofs = 8

    def get_qpos(self) -> list[float]:
        return [0.11, 0.22, 0.79, 1.0, 0.0, 0.0, 0.0, 0.4, 0.5]

    def get_dofs_position(self, dofs_idx_local: tuple[int, ...] | None = None) -> list[float]:
        values = [0.1, 0.2, 1.3, 0.0, 0.0, 0.0, 0.4, 0.5]
        if dofs_idx_local is None:
            return values
        return [values[index] for index in dofs_idx_local]

    def get_pos(self) -> list[float]:
        return [9.0, 8.0, 7.0]


class _FakeContactRobot:
    def get_contacts(self, exclude_self_contact: bool) -> dict[str, list[bool]]:
        assert exclude_self_contact is True
        return {"valid_mask": [True, False, True]}

    def get_links_net_contact_force(self) -> list[float]:
        return [0.0, 0.0, 0.0, 3.0, 4.0, 0.0]
