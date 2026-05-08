from pathlib import Path

import pytest

from h200_locomotion_lab.envs.genesis_adapter import G1_29DOF_JOINT_ORDER
from h200_locomotion_lab.sonic.g1_policy_bridge import (
    SONIC_G1_ACTION_SCALES,
    SONIC_G1_DEFAULT_ANGLES,
    SONIC_G1_ISAACLAB_TO_MUJOCO,
    SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX,
)
from h200_locomotion_lab.tools.sonic_reference_replay_smoke import (
    SONIC_G1_FORCE_LIMITS,
    SONIC_G1_KDS,
    SONIC_G1_KPS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "configs/robots/unitree_g1_29dof_sonic.yaml"


def test_unitree_g1_29dof_sonic_profile_yaml_matches_runtime_constants() -> None:
    yaml = pytest.importorskip("yaml")

    assert PROFILE_PATH.is_file()
    profile = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))

    command_joints = profile["joint_order"]["command_mujoco"]
    policy_joints = profile["joint_order"]["policy_isaaclab"]
    command_to_policy = profile["mapping"]["command_mujoco_index_to_policy_isaaclab_index"]
    policy_to_command = profile["mapping"]["policy_isaaclab_index_to_command_mujoco_index"]
    control = profile["control"]

    assert profile["robot"]["dof_count"] == 29
    assert command_joints == list(G1_29DOF_JOINT_ORDER)
    assert len(policy_joints) == 29
    assert set(policy_joints) == set(command_joints)
    assert command_to_policy == list(SONIC_G1_ISAACLAB_TO_MUJOCO)
    assert policy_to_command == list(SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX)
    assert sorted(command_to_policy) == list(range(29))
    assert sorted(policy_to_command) == list(range(29))
    assert policy_joints == [command_joints[index] for index in policy_to_command]

    assert control["order"] == "command_mujoco"
    assert control["raw_policy_action_order"] == "policy_isaaclab"
    assert control["default_angles_rad"] == list(SONIC_G1_DEFAULT_ANGLES)
    assert control["action_scales_rad"] == pytest.approx(SONIC_G1_ACTION_SCALES)
    assert control["kp"] == pytest.approx(SONIC_G1_KPS)
    assert control["kv"] == pytest.approx(SONIC_G1_KDS)
    assert control["force_limits"] == pytest.approx(SONIC_G1_FORCE_LIMITS)
    for key in ("default_angles_rad", "action_scales_rad", "kp", "kv", "force_limits"):
        assert len(control[key]) == 29

    all_joint_names = command_joints + policy_joints
    assert not any("finger" in joint_name or "hand" in joint_name for joint_name in all_joint_names)
    assert profile["metadata"]["source"] == "official_sonic_deploy_mirror"
