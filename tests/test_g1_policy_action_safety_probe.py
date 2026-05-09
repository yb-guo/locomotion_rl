import argparse

import pytest

from h200_locomotion_lab.tools import g1_policy_action_safety_probe as probe


def test_choose_best_row_prefers_fewer_resets_then_height() -> None:
    rows = [
        {
            "name": "large-action",
            "reset_count": 10,
            "height_bad_count": 10,
            "tilt_bad_count": 0,
            "root_height_min": 0.40,
            "env_policy_steps_per_sec": 100.0,
        },
        {
            "name": "small-action-low",
            "reset_count": 0,
            "height_bad_count": 0,
            "tilt_bad_count": 0,
            "root_height_min": 0.46,
            "env_policy_steps_per_sec": 90.0,
        },
        {
            "name": "small-action-high",
            "reset_count": 0,
            "height_bad_count": 0,
            "tilt_bad_count": 0,
            "root_height_min": 0.50,
            "env_policy_steps_per_sec": 80.0,
        },
    ]

    assert probe.choose_best_row(rows)["name"] == "small-action-high"


def test_build_env_config_rejects_unknown_command_mode() -> None:
    args = argparse.Namespace(
        height_min=0.45,
        height_max=1.20,
        base_height_target=0.85,
        base_height_sigma=0.10,
        base_height_reward_scale=0.0,
    )

    with pytest.raises(ValueError, match="unknown command mode"):
        probe.build_env_config(args, "walk_sideways")
