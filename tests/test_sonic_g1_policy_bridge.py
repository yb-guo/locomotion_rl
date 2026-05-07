import pytest

from h200_locomotion_lab.sonic.g1_policy_bridge import (
    EFFORT_LIMIT_7520_22,
    SONIC_G1_ACTION_SCALES,
    SONIC_G1_DEFAULT_ANGLES,
    SONIC_G1_ISAACLAB_TO_MUJOCO,
    STIFFNESS_7520_22,
    sonic_policy_action_to_mujoco_targets,
)


def test_sonic_policy_zero_action_targets_default_angles() -> None:
    targets = sonic_policy_action_to_mujoco_targets([0.0] * 29)

    assert targets == SONIC_G1_DEFAULT_ANGLES


def test_sonic_policy_action_uses_official_order_and_scale() -> None:
    raw_action = tuple(float(index) for index in range(29))

    targets = sonic_policy_action_to_mujoco_targets(raw_action)

    assert SONIC_G1_ISAACLAB_TO_MUJOCO[1] == 3
    assert targets[1] == pytest.approx(
        SONIC_G1_DEFAULT_ANGLES[1] + raw_action[3] * SONIC_G1_ACTION_SCALES[1]
    )
    assert targets[22] == pytest.approx(
        SONIC_G1_DEFAULT_ANGLES[22] + raw_action[12] * SONIC_G1_ACTION_SCALES[22]
    )


def test_sonic_policy_action_scale_matches_official_formula() -> None:
    assert SONIC_G1_ACTION_SCALES[0] == pytest.approx(
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22
    )


def test_sonic_policy_action_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match="Expected SONIC raw action_dim=29"):
        sonic_policy_action_to_mujoco_targets([0.0] * 28)

