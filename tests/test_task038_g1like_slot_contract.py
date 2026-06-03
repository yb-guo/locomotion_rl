from pathlib import Path

import pytest

from h200_locomotion_lab.robots.g1_27dof_nohand import G1_27DOF_NOHAND_ACTUATOR_ORDER
from h200_locomotion_lab.robots.g1like_slots import (
    G1LIKE_ACTION_DIM,
    G1LIKE_ACTION_SLOT_NAMES,
    G1LIKE_MISSING_FROM_27DOF_NOHAND,
    G1_27DOF_NOHAND_SLOT_MAPPING,
    G1_29DOF_COMMAND_MUJOCO_ACTUATOR_ORDER,
    G1_29DOF_COMMAND_MUJOCO_SLOT_MAPPING,
    G1LikeSlotError,
    G1LikeSlotSchema,
    build_g1like_slot_mapping,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_29DOF_PATH = REPO_ROOT / "configs/robots/unitree_g1_29dof_sonic.yaml"


def _load_unitree_g1_29dof_command_order() -> tuple[str, ...]:
    yaml = pytest.importorskip("yaml")
    profile = yaml.safe_load(PROFILE_29DOF_PATH.read_text(encoding="utf-8"))
    return tuple(profile["joint_order"]["command_mujoco"])


def test_slot_names_are_unique_and_action_dim_is_fixed_29() -> None:
    schema = G1LikeSlotSchema()

    assert schema.action_dim == G1LIKE_ACTION_DIM == 29
    assert len(schema.slot_names) == 29
    assert len(set(schema.slot_names)) == 29
    assert not any(slot_name.endswith("_joint") for slot_name in schema.slot_names)
    assert not any("finger" in slot_name or "hand" in slot_name for slot_name in schema.slot_names)


def test_29dof_mapping_round_trips_existing_command_mujoco_order() -> None:
    command_order = _load_unitree_g1_29dof_command_order()
    mapping = build_g1like_slot_mapping(command_order)
    unified_action = tuple(float(index) for index in range(G1LIKE_ACTION_DIM))

    assert command_order == G1_29DOF_COMMAND_MUJOCO_ACTUATOR_ORDER
    assert mapping.selector == tuple(range(G1LIKE_ACTION_DIM))
    assert mapping.mask == tuple(True for _ in range(G1LIKE_ACTION_DIM))
    assert mapping.project_to_robot_order(unified_action) == unified_action
    assert mapping.scatter_to_unified_slots(unified_action) == unified_action
    assert G1_29DOF_COMMAND_MUJOCO_SLOT_MAPPING.project_to_robot_order(unified_action) == unified_action


def test_27dof_nohand_mapping_masks_waist_roll_pitch_and_selects_present_values() -> None:
    mapping = G1_27DOF_NOHAND_SLOT_MAPPING
    unified_action = tuple(float(index) for index in range(G1LIKE_ACTION_DIM))
    expected_robot_action = tuple(
        unified_action[G1LIKE_ACTION_SLOT_NAMES.index(joint_name.removesuffix("_joint"))]
        for joint_name in G1_27DOF_NOHAND_ACTUATOR_ORDER
    )

    assert mapping.action_dim == 29
    assert mapping.robot_action_dim == 27
    assert mapping.robot_joint_order == G1_27DOF_NOHAND_ACTUATOR_ORDER
    assert mapping.missing_slot_names == G1LIKE_MISSING_FROM_27DOF_NOHAND
    for missing_slot in G1LIKE_MISSING_FROM_27DOF_NOHAND:
        assert mapping.mask[G1LIKE_ACTION_SLOT_NAMES.index(missing_slot)] is False
    assert mapping.project_to_robot_order(unified_action) == expected_robot_action


def test_missing_slot_cannot_be_selected_accidentally() -> None:
    mapping = G1_27DOF_NOHAND_SLOT_MAPPING
    unified_action = [0.0 for _ in range(G1LIKE_ACTION_DIM)]
    unified_action[G1LIKE_ACTION_SLOT_NAMES.index("waist_roll")] = 111.0
    unified_action[G1LIKE_ACTION_SLOT_NAMES.index("waist_pitch")] = 222.0

    robot_action = mapping.project_to_robot_order(unified_action)
    scattered = mapping.scatter_to_unified_slots(robot_action, fill=-7.0)

    assert 111.0 not in robot_action
    assert 222.0 not in robot_action
    assert scattered[G1LIKE_ACTION_SLOT_NAMES.index("waist_roll")] == -7.0
    assert scattered[G1LIKE_ACTION_SLOT_NAMES.index("waist_pitch")] == -7.0


def test_duplicate_semantic_slot_is_rejected() -> None:
    duplicate_schema_names = G1LIKE_ACTION_SLOT_NAMES[:-1] + (G1LIKE_ACTION_SLOT_NAMES[0],)
    with pytest.raises(G1LikeSlotError, match="duplicate semantic slots"):
        G1LikeSlotSchema(duplicate_schema_names)

    with pytest.raises(G1LikeSlotError, match="duplicate joints"):
        build_g1like_slot_mapping(("left_knee_joint", "left_knee_joint"))


def test_malformed_or_bare_semantic_joint_name_is_rejected() -> None:
    with pytest.raises(
        G1LikeSlotError,
        match="robot joint name must be a non-empty canonical name ending with _joint",
    ):
        build_g1like_slot_mapping(("left_knee",))

    with pytest.raises(
        G1LikeSlotError,
        match="robot joint name must be a non-empty canonical name ending with _joint",
    ):
        build_g1like_slot_mapping(("",))


def test_unknown_joint_name_is_rejected() -> None:
    with pytest.raises(G1LikeSlotError, match="unknown G1-like joint name"):
        build_g1like_slot_mapping(("left_hip_pitch_joint", "left_gripper_joint"))
