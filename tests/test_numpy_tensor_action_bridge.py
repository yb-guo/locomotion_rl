import math

import pytest

np = pytest.importorskip("numpy")

from h200_locomotion_lab.robots import load_robot_profile
from h200_locomotion_lab.runtime import NumpyTensorActionBridge, ScalarActionBridge


@pytest.fixture
def tensor_bridge() -> NumpyTensorActionBridge:
    return NumpyTensorActionBridge.from_profile(load_robot_profile())


@pytest.fixture
def scalar_bridge() -> ScalarActionBridge:
    return ScalarActionBridge.from_profile(load_robot_profile())


def test_numpy_tensor_bridge_maps_batched_actions(tensor_bridge: NumpyTensorActionBridge) -> None:
    raw_actions = np.asarray(
        [
            [0.0] * 29,
            [float(index) for index in range(29)],
            [math.sin(index) for index in range(29)],
            [(-1.0) ** index * (index + 1.0) * 3.0 for index in range(29)],
        ],
        dtype=np.float64,
    )

    targets = tensor_bridge.policy_actions_to_command_targets(raw_actions)

    assert targets.shape == (4, 29)
    expected = (
        tensor_bridge.default_angles_command[None, :]
        + raw_actions[:, tensor_bridge.command_to_policy]
        * tensor_bridge.action_scale_command[None, :]
    )
    np.testing.assert_allclose(targets, expected)


def test_numpy_tensor_bridge_batch_one_matches_scalar_bridge(
    tensor_bridge: NumpyTensorActionBridge,
    scalar_bridge: ScalarActionBridge,
) -> None:
    raw_action = np.asarray([[math.sin(index) for index in range(29)]], dtype=np.float64)

    tensor_targets = tensor_bridge.policy_actions_to_command_targets(raw_action)
    scalar_targets = scalar_bridge.policy_action_to_command_targets(raw_action[0].tolist())

    assert tensor_targets.shape == (1, 29)
    np.testing.assert_allclose(tensor_targets[0], np.asarray(scalar_targets))


def test_numpy_tensor_bridge_batch_rows_match_scalar_bridge(
    tensor_bridge: NumpyTensorActionBridge,
    scalar_bridge: ScalarActionBridge,
) -> None:
    raw_actions = np.asarray(
        [
            [float(index) for index in range(29)],
            [(-1.0) ** index * (index + 1.0) for index in range(29)],
        ],
        dtype=np.float64,
    )

    tensor_targets = tensor_bridge.policy_actions_to_command_targets(raw_actions)

    for row_index, raw_action in enumerate(raw_actions):
        scalar_targets = scalar_bridge.policy_action_to_command_targets(raw_action.tolist())
        np.testing.assert_allclose(tensor_targets[row_index], np.asarray(scalar_targets))


def test_numpy_tensor_bridge_rejects_wrong_rank(tensor_bridge: NumpyTensorActionBridge) -> None:
    with pytest.raises(ValueError, match="Expected raw action rank 2"):
        tensor_bridge.policy_actions_to_command_targets(np.zeros(29))


def test_numpy_tensor_bridge_rejects_wrong_action_width(
    tensor_bridge: NumpyTensorActionBridge,
) -> None:
    with pytest.raises(ValueError, match=r"Expected raw action shape \[N, 29\]"):
        tensor_bridge.policy_actions_to_command_targets(np.zeros((2, 28)))


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_numpy_tensor_bridge_rejects_non_finite_actions(
    tensor_bridge: NumpyTensorActionBridge,
    bad_value: float,
) -> None:
    raw_actions = np.zeros((2, 29), dtype=np.float64)
    raw_actions[1, 3] = bad_value

    with pytest.raises(ValueError, match="raw actions must contain only finite values"):
        tensor_bridge.policy_actions_to_command_targets(raw_actions)


def test_numpy_tensor_bridge_does_not_clip_raw_actions(
    tensor_bridge: NumpyTensorActionBridge,
) -> None:
    raw_actions = np.zeros((1, 29), dtype=np.float64)
    raw_actions[0, 0] = 100.0

    targets = tensor_bridge.policy_actions_to_command_targets(raw_actions)

    expected = (
        tensor_bridge.default_angles_command[0]
        + raw_actions[0, tensor_bridge.command_to_policy[0]]
        * tensor_bridge.action_scale_command[0]
    )
    assert targets[0, 0] == pytest.approx(expected)
