import math

import pytest

from h200_locomotion_lab.robots import load_robot_profile
from h200_locomotion_lab.runtime import ScalarActionBridge
from h200_locomotion_lab.sonic.g1_policy_bridge import sonic_policy_action_to_mujoco_targets


@pytest.fixture
def scalar_bridge() -> ScalarActionBridge:
    return ScalarActionBridge.from_profile(load_robot_profile())


@pytest.mark.parametrize(
    "raw_action",
    [
        [0.0] * 29,
        [float(index) for index in range(29)],
        [math.sin(index) for index in range(29)],
        [(-1.0) ** index * (index + 1.0) * 3.0 for index in range(29)],
    ],
)
def test_scalar_bridge_matches_existing_sonic_bridge(
    scalar_bridge: ScalarActionBridge,
    raw_action: list[float],
) -> None:
    assert scalar_bridge.policy_action_to_command_targets(raw_action) == pytest.approx(
        sonic_policy_action_to_mujoco_targets(raw_action)
    )


def test_scalar_bridge_uses_profile_mapping_defaults_and_scales(
    scalar_bridge: ScalarActionBridge,
) -> None:
    raw_action = tuple(float(index) for index in range(29))
    targets = scalar_bridge.policy_action_to_command_targets(raw_action)

    for command_index, policy_index in enumerate(scalar_bridge.command_to_policy):
        assert targets[command_index] == pytest.approx(
            scalar_bridge.default_angles_command[command_index]
            + raw_action[policy_index] * scalar_bridge.action_scale_command[command_index]
        )


def test_scalar_bridge_rejects_wrong_action_length(scalar_bridge: ScalarActionBridge) -> None:
    with pytest.raises(ValueError, match="Expected raw action length 29, got 28"):
        scalar_bridge.policy_action_to_command_targets([0.0] * 28)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_scalar_bridge_rejects_non_finite_actions(
    scalar_bridge: ScalarActionBridge,
    bad_value: float,
) -> None:
    raw_action = [0.0] * 29
    raw_action[3] = bad_value

    with pytest.raises(ValueError, match="raw action must contain only finite values"):
        scalar_bridge.policy_action_to_command_targets(raw_action)


def test_scalar_bridge_does_not_clip_raw_actions(scalar_bridge: ScalarActionBridge) -> None:
    raw_action = [0.0] * 29
    raw_action[0] = 100.0

    targets = scalar_bridge.policy_action_to_command_targets(raw_action)

    assert targets[0] == pytest.approx(
        scalar_bridge.default_angles_command[0] + 100.0 * scalar_bridge.action_scale_command[0]
    )
