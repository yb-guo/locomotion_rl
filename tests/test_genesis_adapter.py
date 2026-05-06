import sys

import pytest

from h200_locomotion_lab.envs.genesis_adapter import (
    G1_29DOF_JOINT_ORDER,
    GenesisG1Contract,
    GenesisG1Env,
)


def test_genesis_contract_matches_g1_29dof_inventory() -> None:
    contract = GenesisG1Contract()

    assert len(G1_29DOF_JOINT_ORDER) == 29
    assert G1_29DOF_JOINT_ORDER[:6] == (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    )
    assert G1_29DOF_JOINT_ORDER[-3:] == (
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    assert contract.action_dim == 29
    assert contract.observation_dim == 96
    assert contract.policy_rate_hz == 50
    contract.validate()


def test_genesis_adapter_does_not_import_simulator_package() -> None:
    assert "genesis" not in sys.modules


def test_contract_only_backend_resets_and_steps() -> None:
    env = GenesisG1Env.contract_only()

    observation = env.reset(seed=123)
    result = env.step([0.0] * env.contract.action_dim)

    assert len(observation) == env.contract.observation_dim
    assert len(result.observation) == env.contract.observation_dim
    assert result.reward == 0.0
    assert result.terminated is False
    assert result.truncated is False
    assert result.info["step_count"] == 1
    assert "96D obs" in env.describe()


def test_step_rejects_wrong_action_shape() -> None:
    env = GenesisG1Env.contract_only()
    env.reset()

    with pytest.raises(ValueError, match="Expected action_dim=29"):
        env.step([0.0] * 28)
