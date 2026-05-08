from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from h200_locomotion_lab.robots import (
    CompiledRobotProfile,
    RobotProfileError,
    load_robot_profile,
    load_robot_profile_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "configs/robots/unitree_g1_29dof_sonic.yaml"


def test_loader_compiles_unitree_g1_29dof_sonic_yaml_to_tuple_profile() -> None:
    profile = load_robot_profile(PROFILE_PATH)

    assert isinstance(profile, CompiledRobotProfile)
    assert profile.name == "unitree_g1_29dof_sonic"
    assert profile.dof_count == 29
    assert profile.action_dim == 29
    assert profile.metadata.source == "official_sonic_deploy_mirror"
    assert profile.source_path == PROFILE_PATH
    assert len(profile.joint_order.command_mujoco) == 29
    assert len(profile.joint_order.policy_isaaclab) == 29
    assert sorted(profile.mapping.command_mujoco_index_to_policy_isaaclab_index) == list(range(29))
    assert sorted(profile.mapping.policy_isaaclab_index_to_command_mujoco_index) == list(range(29))
    assert len(profile.control.default_angles_rad) == 29
    assert len(profile.control.action_scales_rad) == 29
    assert len(profile.control.kp) == 29
    assert len(profile.control.kv) == 29
    assert len(profile.control.force_limits) == 29

    assert isinstance(profile.joint_order.command_mujoco, tuple)
    assert isinstance(profile.joint_order.policy_isaaclab, tuple)
    assert isinstance(profile.mapping.command_mujoco_index_to_policy_isaaclab_index, tuple)
    assert isinstance(profile.mapping.policy_isaaclab_index_to_command_mujoco_index, tuple)
    assert isinstance(profile.control.default_angles_rad, tuple)
    assert isinstance(profile.control.action_scales_rad, tuple)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data["robot"].update({"dof_count": 28}), "robot.dof_count must be 29"),
        (
            lambda data: data["joint_order"]["command_mujoco"].pop(),
            "joint_order.command_mujoco length must be 29",
        ),
        (
            lambda data: data["joint_order"]["policy_isaaclab"].__setitem__(0, "extra_joint"),
            "joint_order command/policy sets must match",
        ),
        (
            lambda data: data["mapping"][
                "command_mujoco_index_to_policy_isaaclab_index"
            ].__setitem__(0, 1),
            "must be a 0..28 permutation",
        ),
        (
            lambda data: data["mapping"][
                "policy_isaaclab_index_to_command_mujoco_index"
            ].__setitem__(0, 1),
            "must be a 0..28 permutation",
        ),
        (
            lambda data: data["control"]["kp"].pop(),
            "control.kp length must be 29",
        ),
        (
            lambda data: data["metadata"].update({"source": ""}),
            "metadata.source must be a non-empty string",
        ),
        (
            lambda data: data["joint_order"]["command_mujoco"].__setitem__(0, "left_hand_joint"),
            "must not include hand/finger joints",
        ),
    ],
)
def test_loader_rejects_invalid_profiles(
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    data = _load_profile_dict()
    mutate(data)

    with pytest.raises(RobotProfileError, match=message):
        load_robot_profile_dict(data)


def test_loader_rejects_mapping_that_is_permutation_but_wrong_for_joint_names() -> None:
    data = _load_profile_dict()
    policy_joints = data["joint_order"]["policy_isaaclab"]
    policy_joints[0], policy_joints[1] = policy_joints[1], policy_joints[0]

    with pytest.raises(
        RobotProfileError,
        match="mapping does not match command/policy joint names",
    ):
        load_robot_profile_dict(data)


def test_loader_rejects_non_inverse_mapping_arrays() -> None:
    data = _load_profile_dict()
    policy_to_command = data["mapping"]["policy_isaaclab_index_to_command_mujoco_index"]
    policy_to_command[0], policy_to_command[1] = policy_to_command[1], policy_to_command[0]

    with pytest.raises(RobotProfileError, match="mapping arrays must be inverse permutations"):
        load_robot_profile_dict(data)


def _load_profile_dict() -> dict[str, object]:
    with PROFILE_PATH.open("r", encoding="utf-8") as stream:
        return deepcopy(yaml.safe_load(stream))
