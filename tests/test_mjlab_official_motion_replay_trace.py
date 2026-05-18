from __future__ import annotations

from pathlib import Path

import pytest

from h200_locomotion_lab.sonic.g1_policy_bridge import SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX
from h200_locomotion_lab.tools.mjlab_official_motion_replay_trace import (
    motion_from_official_qpos_rows,
    read_official_motion_qpos_csv,
    summarize_motion_csv,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_read_official_motion_qpos_csv_handles_trailing_commas_and_segments():
    path = FIXTURES / "official_motion_segments.csv"

    motion = read_official_motion_qpos_csv(path)

    assert len(motion.rows) == 3
    assert motion.segment_lengths == (2, 1)
    assert motion.rows[0] == pytest.approx([float(index) for index in range(36)])
    assert motion.rows[2] == pytest.approx([float(index) + 1.0 for index in range(36)])


def test_read_official_motion_qpos_csv_rejects_wrong_width():
    with pytest.raises(ValueError, match="expected 36 qpos values"):
        read_official_motion_qpos_csv(FIXTURES / "official_motion_bad_width.csv")


def test_motion_from_official_qpos_rows_reorders_joints_to_policy_order():
    base = [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]
    joints0 = [float(index) for index in range(29)]
    joints1 = [float(index) + 0.1 for index in range(29)]

    motion = motion_from_official_qpos_rows([base + joints0, base + joints1])

    for policy_index, mujoco_index in enumerate(SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX):
        assert motion.joint_positions_policy_order[0][policy_index] == pytest.approx(
            joints0[mujoco_index]
        )
        assert motion.joint_velocities_policy_order[0][policy_index] == pytest.approx(5.0)


def test_summarize_motion_csv_reports_root_and_pitch():
    summary = summarize_motion_csv(
        read_official_motion_qpos_csv(FIXTURES / "official_motion_summary.csv"),
        replay_rows=2,
    )

    assert summary["rows"] == 2
    assert summary["segment_count"] == 1
    assert summary["root_z_final"] == pytest.approx(0.82)
    assert summary["root_delta_xy_per_s"] == pytest.approx([0.5, 0.0])
    assert summary["abs_pitch_p95"] == pytest.approx(0.0)
