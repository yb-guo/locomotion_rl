import math
from pathlib import Path

import pytest

from h200_locomotion_lab.tools.genesis_sonic_policy_locomotion_probe import (
    FootSample,
    RootPose,
    count_contact_switches,
    count_double_support_frames,
    count_no_support_frames,
    count_single_support_frames,
    enforce_wall_time,
    heartbeat,
    read_token_csv_rows,
    read_foot_sample,
    resolve_link_index,
    root_pose_from_qpos,
    summarize_root_motion,
)


def test_root_pose_from_qpos_reads_xyz_and_wxyz_yaw() -> None:
    half_yaw = math.pi / 4.0
    pose = root_pose_from_qpos(
        (
            1.0,
            2.0,
            0.79,
            math.cos(half_yaw),
            0.0,
            0.0,
            math.sin(half_yaw),
        )
    )

    assert pose.x == 1.0
    assert pose.y == 2.0
    assert pose.z == 0.79
    assert pose.yaw == pytest.approx(math.pi / 2.0)


def test_summarize_root_motion_reports_displacement_path_speed_and_yaw() -> None:
    poses = (
        RootPose(0.0, 0.0, 0.8, 0.0),
        RootPose(0.03, 0.04, 0.78, 0.1),
        RootPose(0.06, 0.08, 0.79, 0.2),
    )

    summary = summarize_root_motion(poses, policy_rate_hz=50)

    assert summary["root_z_min"] == 0.78
    assert summary["root_z_max"] == 0.8
    assert summary["horizontal_displacement"] == pytest.approx(0.1)
    assert summary["path_length_xy"] == pytest.approx(0.1)
    assert summary["average_speed_xy"] == pytest.approx(2.5)
    assert summary["yaw_delta"] == pytest.approx(0.2)


def test_contact_summary_counts_alternation_and_support_modes() -> None:
    left = (True, True, False, False, True)
    right = (False, True, True, False, False)

    assert count_contact_switches(left) == 2
    assert count_contact_switches(right) == 2
    assert count_single_support_frames(left, right) == 3
    assert count_double_support_frames(left, right) == 1
    assert count_no_support_frames(left, right) == 1


def test_read_foot_sample_uses_named_link_index_and_contact_force() -> None:
    robot = _FakeLinkRobot()
    link_idx = resolve_link_index(robot, "left_ankle_roll_link")

    sample = read_foot_sample(robot, link_idx, contact_threshold=5.0)

    assert link_idx == 2
    assert sample == FootSample(z=0.12, force=13.0, contact=True)


def test_read_foot_sample_returns_empty_sample_for_missing_link() -> None:
    sample = read_foot_sample(_FakeLinkRobot(), None, contact_threshold=5.0)

    assert sample == FootSample(z=None, force=None, contact=None)


def test_heartbeat_writes_latest_progress() -> None:
    progress_path = _FakeProgressPath()
    args = _Args(heartbeat_every_frame=False)

    heartbeat(args, progress_path, frame=7, stage="step")

    assert "frame=7" in progress_path.content
    assert "stage=step" in progress_path.content


def test_enforce_wall_time_raises_after_budget() -> None:
    with pytest.raises(SystemExit, match="GENESIS_SONIC_POLICY_LOCOMOTION_PROBE_TIMEOUT"):
        enforce_wall_time(started_at=0.0, max_wall_time_s=0.001)


def test_read_token_csv_rows_reads_64d_rows() -> None:
    path = Path(__file__).parent / "fixtures" / "token_state_sample.csv"

    rows = read_token_csv_rows(path)

    assert rows == [tuple(float(index) for index in range(64))]


class _Args:
    def __init__(self, *, heartbeat_every_frame: bool) -> None:
        self.heartbeat_every_frame = heartbeat_every_frame


class _FakeProgressPath:
    content = ""

    @property
    def parent(self) -> "_FakeProgressPath":
        return self

    def mkdir(self, parents: bool, exist_ok: bool) -> None:
        assert parents is True
        assert exist_ok is True

    def write_text(self, content: str) -> None:
        self.content = content


class _FakeLink:
    def __init__(self, idx_local: int) -> None:
        self.idx_local = idx_local


class _FakeLinkRobot:
    def get_link(self, name: str) -> _FakeLink:
        if name != "left_ankle_roll_link":
            raise KeyError(name)
        return _FakeLink(2)

    def get_links_pos(self, links_idx_local: tuple[int, ...] | None = None) -> list[float]:
        all_positions = [
            0.0,
            0.0,
            0.9,
            0.1,
            0.0,
            0.3,
            0.2,
            0.0,
            0.12,
        ]
        if links_idx_local is None:
            return all_positions
        start = links_idx_local[0] * 3
        return all_positions[start : start + 3]

    def get_links_net_contact_force(self) -> list[float]:
        return [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0,
            4.0,
            12.0,
        ]
