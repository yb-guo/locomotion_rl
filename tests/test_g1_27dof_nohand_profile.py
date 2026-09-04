from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from h200_locomotion_lab.robots import (
    G1_27DOF_NOHAND_ACTUATOR_ORDER,
    REMOVED_FROM_G1_29DOF_COMMAND_ORDER,
    G1NoHandGenesisTrainingProfile,
    G1NoHandProfileError,
    RobotProfileError,
    load_g1_27dof_nohand_profile,
    load_g1_27dof_nohand_profile_dict,
    load_robot_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_27DOF_PATH = REPO_ROOT / "configs/robots/unitree_g1_27dof_nohand_genesis.yaml"
PROFILE_29DOF_PATH = REPO_ROOT / "configs/robots/unitree_g1_29dof_sonic.yaml"
PREPARED_ASSET_PATH = (
    "/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/"
    "GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_27dof_nohand.xml"
)


def test_loads_g1_27dof_nohand_genesis_training_profile() -> None:
    profile = load_g1_27dof_nohand_profile(PROFILE_27DOF_PATH)

    assert isinstance(profile, G1NoHandGenesisTrainingProfile)
    assert profile.name == "unitree_g1_27dof_nohand_genesis"
    assert profile.route == "VectorizedGenesisBackend"
    assert profile.dof_count == 27
    assert profile.action_dim == 27
    assert profile.asset.format == "mjcf"
    assert profile.asset.genesis_morph == "MJCF"
    assert profile.asset.path == PREPARED_ASSET_PATH
    assert profile.asset.usage == "prepared_path_metadata_only"
    assert profile.source_path == PROFILE_27DOF_PATH
    assert profile.removed_from_29dof_command_order == REMOVED_FROM_G1_29DOF_COMMAND_ORDER
    assert profile.excludes_floating_base_dofs is True
    assert profile.training_contract.sim_dt_s == pytest.approx(0.005)
    assert profile.training_contract.decimation == 4
    assert profile.training_contract.policy_rate_hz == 50
    assert profile.training_contract.action_size == 27
    assert profile.training_contract.observation_dim == 90
    assert sum(segment.size for segment in profile.training_contract.observation_segments) == 90

    assert len(profile.actuator_order) == 27
    assert profile.actuator_order == G1_27DOF_NOHAND_ACTUATOR_ORDER
    assert len(set(profile.actuator_order)) == 27
    assert "waist_yaw_joint" in profile.actuator_order
    assert "waist_roll_joint" not in profile.actuator_order
    assert "waist_pitch_joint" not in profile.actuator_order
    assert not any("hand" in joint or "finger" in joint for joint in profile.actuator_order)

    assert len(profile.control.default_angles_rad) == 27
    assert len(profile.control.action_scales_rad) == 27
    assert len(profile.control.kp) == 27
    assert len(profile.control.kv) == 27
    assert len(profile.control.force_limits) == 27


def test_g1_27dof_actuator_order_is_29dof_command_order_without_waist_roll_pitch() -> None:
    with PROFILE_29DOF_PATH.open("r", encoding="utf-8") as stream:
        sonic_profile = yaml.safe_load(stream)
    profile = load_g1_27dof_nohand_profile(PROFILE_27DOF_PATH)

    expected_order = tuple(
        joint
        for joint in sonic_profile["joint_order"]["command_mujoco"]
        if joint not in REMOVED_FROM_G1_29DOF_COMMAND_ORDER
    )
    assert profile.actuator_order == expected_order


def test_g1_27dof_profile_keeps_h200_training_evidence() -> None:
    profile = load_g1_27dof_nohand_profile(PROFILE_27DOF_PATH)

    assert profile.h200_evidence.cuda_visible_devices == "1"
    assert profile.h200_evidence.physical_gpu == 1
    assert profile.h200_evidence.logical_cuda_device == "cuda:0"
    assert profile.h200_evidence.output.endswith(
        "/outputs/task011/g1_simpler/g1_27dof_nohand_xml_n1024.txt"
    )
    assert profile.h200_evidence.n_envs == 1024
    assert profile.h200_evidence.env_policy_steps_per_sec == pytest.approx(45827.527990)
    assert profile.h200_evidence.env_sim_steps_per_sec == pytest.approx(183310.111961)
    assert profile.h200_evidence.build_time_s == pytest.approx(50.844098)
    assert profile.h200_evidence.measure_time_s == pytest.approx(2.234465)
    assert profile.h200_evidence.tensor_device_ok is True
    assert profile.h200_evidence.selected_reset_target_only is True
    assert profile.component_evidence.action_write_time_s == pytest.approx(0.018051)
    assert profile.component_evidence.state_read_time_s == pytest.approx(0.024633)
    assert profile.component_evidence.scene_step_time_s == pytest.approx(2.134225)
    assert profile.component_evidence.scene_steps_per_sec == pytest.approx(187.421643)
    assert profile.component_evidence.env_scene_steps_per_sec == pytest.approx(191919.762451)
    assert profile.component_evidence.combined_env_policy_steps_per_sec == pytest.approx(
        29547.329979
    )
    assert profile.component_evidence.combined_env_sim_steps_per_sec == pytest.approx(
        118189.319917
    )


def test_29dof_sonic_loader_remains_strict_and_rejects_27dof_profile() -> None:
    with pytest.raises(RobotProfileError):
        load_robot_profile(PROFILE_27DOF_PATH)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data["robot"].update({"dof_count": 29}), "robot.dof_count must be 27"),
        (
            lambda data: data["robot"].update({"route": "GenesisG1SceneBackend"}),
            "robot.route must be VectorizedGenesisBackend",
        ),
        (
            lambda data: data["joint_order"]["actuator"].append("waist_roll_joint"),
            "joint_order.actuator length must be 27",
        ),
        (
            lambda data: data["joint_order"]["actuator"].reverse(),
            "joint_order.actuator must match the canonical 29DoF command order",
        ),
        (
            lambda data: data["joint_order"].update({"order": "policy"}),
            "joint_order.order must be actuator",
        ),
        (
            lambda data: data["joint_order"].update({"derived_from": "manual"}),
            "joint_order.derived_from must be unitree_g1_29dof_sonic.command_mujoco",
        ),
        (
            lambda data: data["joint_order"]["actuator"].__setitem__(0, "left_hand_joint"),
            "must not include hand/finger joints",
        ),
        (
            lambda data: data["joint_order"].update({"excludes_floating_base_dofs": False}),
            "excludes_floating_base_dofs must be true",
        ),
        (lambda data: data["asset"].update({"format": "mujoco_xml"}), "asset.format must be mjcf"),
        (
            lambda data: data["training_contract"].update({"observation_dim": 91}),
            "observation_dim must equal observation segment total",
        ),
        (
            lambda data: data["h200_evidence"].update({"tensor_device_ok": False}),
            "tensor_device_ok must be true",
        ),
        (lambda data: data["control"]["kp"].pop(), "control.kp length must be 27"),
    ],
)
def test_rejects_invalid_g1_27dof_profiles(mutate, message: str) -> None:
    data = _load_27dof_profile_dict()
    mutate(data)

    with pytest.raises(G1NoHandProfileError, match=message):
        load_g1_27dof_nohand_profile_dict(data)


def _load_27dof_profile_dict() -> dict[str, object]:
    with PROFILE_27DOF_PATH.open("r", encoding="utf-8") as stream:
        return deepcopy(yaml.safe_load(stream))
