import pytest

from h200_locomotion_lab.envs.g1_reset_poses import (
    build_g1_standing_reset_pose_candidates,
    leg_value_summary,
)
from h200_locomotion_lab.robots import load_g1_27dof_nohand_profile
from h200_locomotion_lab.tools import g1_standing_reset_pose_probe as probe


def test_pose_candidates_keep_27dof_shape_and_make_taller_legs() -> None:
    profile = load_g1_27dof_nohand_profile()

    candidates = build_g1_standing_reset_pose_candidates(profile.control.default_angles_rad)

    assert set(candidates) == {
        "profile",
        "half_crouch",
        "mild_crouch",
        "tall_crouch",
        "straight",
    }
    assert all(len(values) == 27 for values in candidates.values())
    assert leg_value_summary(candidates["profile"])["knee"] > leg_value_summary(
        candidates["tall_crouch"]
    )["knee"]
    assert leg_value_summary(candidates["straight"]) == {
        "hip_pitch": 0.0,
        "knee": 0.0,
        "ankle_pitch": 0.0,
    }


def test_choose_best_row_prefers_fewer_resets_then_tilt_then_height() -> None:
    rows = [
        {
            "pose": "tilted-high",
            "reset_count": 0,
            "height_bad_count": 0,
            "tilt_bad_count": 1,
            "root_height_min": 0.9,
            "env_policy_steps_per_sec": 10.0,
        },
        {
            "pose": "height-low",
            "reset_count": 0,
            "height_bad_count": 1,
            "tilt_bad_count": 0,
            "root_height_min": 0.4,
            "env_policy_steps_per_sec": 9.0,
        },
        {
            "pose": "stable-low",
            "reset_count": 0,
            "height_bad_count": 0,
            "tilt_bad_count": 0,
            "root_height_min": 0.6,
            "env_policy_steps_per_sec": 5.0,
        },
        {
            "pose": "stable-high",
            "reset_count": 0,
            "height_bad_count": 0,
            "tilt_bad_count": 0,
            "root_height_min": 0.8,
            "env_policy_steps_per_sec": 4.0,
        },
    ]

    assert probe.choose_best_row(rows)["pose"] == "stable-high"


def test_parse_float_list_rejects_empty() -> None:
    with pytest.raises(Exception, match="at least one float"):
        probe.parse_float_list("")
