"""YAML robot profile loader with initialization-time validation."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from h200_locomotion_lab.robots.spec import (
    CompiledRobotProfile,
    ControlProfileSpec,
    JointMappingSpec,
    JointOrderSpec,
    RobotProfileMetadata,
)


EXPECTED_G1_29DOF_ACTION_DIM = 29
DEFAULT_UNITREE_G1_29DOF_SONIC_PROFILE = (
    Path(__file__).resolve().parents[3] / "configs" / "robots" / "unitree_g1_29dof_sonic.yaml"
)
_BANNED_BODY_JOINT_TOKENS = ("hand", "finger")


class RobotProfileError(ValueError):
    """Raised when a robot YAML profile is structurally invalid."""


def load_robot_profile(
    path: str | Path = DEFAULT_UNITREE_G1_29DOF_SONIC_PROFILE,
) -> CompiledRobotProfile:
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return load_robot_profile_dict(data, source_path=profile_path)


def load_robot_profile_dict(
    data: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> CompiledRobotProfile:
    if not isinstance(data, Mapping):
        raise RobotProfileError("robot profile must be a mapping")

    robot = _required_mapping(data, "robot")
    metadata_data = _required_mapping(data, "metadata")
    joint_order_data = _required_mapping(data, "joint_order")
    mapping_data = _required_mapping(data, "mapping")
    control_data = _required_mapping(data, "control")

    name = _required_str(robot, "robot.name")
    family = _required_str(robot, "robot.family")
    dof_count = _required_int(robot, "robot.dof_count")
    body_profile = _required_str(robot, "robot.body_profile")
    if dof_count != EXPECTED_G1_29DOF_ACTION_DIM:
        raise RobotProfileError(f"robot.dof_count must be 29, got {dof_count}")

    metadata = RobotProfileMetadata(
        source=_required_str(metadata_data, "metadata.source"),
        mirrors=_tuple_of_str(metadata_data.get("mirrors", ()), "metadata.mirrors"),
        note=_optional_str(metadata_data.get("note"), "metadata.note"),
    )

    joint_order = JointOrderSpec(
        command_mujoco=_required_tuple_of_str(joint_order_data, "joint_order.command_mujoco"),
        policy_isaaclab=_required_tuple_of_str(joint_order_data, "joint_order.policy_isaaclab"),
    )
    _validate_joint_orders(joint_order, dof_count)

    mapping = JointMappingSpec(
        command_mujoco_index_to_policy_isaaclab_index=_required_tuple_of_int(
            mapping_data, "mapping.command_mujoco_index_to_policy_isaaclab_index"
        ),
        policy_isaaclab_index_to_command_mujoco_index=_required_tuple_of_int(
            mapping_data, "mapping.policy_isaaclab_index_to_command_mujoco_index"
        ),
    )
    _validate_mappings(mapping, joint_order, dof_count)

    control = ControlProfileSpec(
        order=_required_str(control_data, "control.order"),
        raw_policy_action_order=_required_str(control_data, "control.raw_policy_action_order"),
        default_angles_rad=_required_tuple_of_float(control_data, "control.default_angles_rad"),
        action_scales_rad=_required_tuple_of_float(control_data, "control.action_scales_rad"),
        kp=_required_tuple_of_float(control_data, "control.kp"),
        kv=_required_tuple_of_float(control_data, "control.kv"),
        force_limits=_required_tuple_of_float(control_data, "control.force_limits"),
    )
    _validate_control(control, dof_count)

    return CompiledRobotProfile(
        name=name,
        family=family,
        dof_count=dof_count,
        body_profile=body_profile,
        metadata=metadata,
        joint_order=joint_order,
        mapping=mapping,
        control=control,
        source_path=Path(source_path) if source_path is not None else None,
    )


def _required_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise RobotProfileError(f"{key} must be a mapping")
    return value


def _required_str(data: Mapping[str, Any], key_path: str) -> str:
    value = _get_key_path(data, key_path)
    if not isinstance(value, str) or not value.strip():
        raise RobotProfileError(f"{key_path} must be a non-empty string")
    return value


def _optional_str(value: Any, key_path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RobotProfileError(f"{key_path} must be a string when provided")
    return value


def _required_int(data: Mapping[str, Any], key_path: str) -> int:
    value = _get_key_path(data, key_path)
    if not isinstance(value, int):
        raise RobotProfileError(f"{key_path} must be an integer")
    return value


def _required_tuple_of_str(data: Mapping[str, Any], key_path: str) -> tuple[str, ...]:
    return _tuple_of_str(_get_key_path(data, key_path), key_path)


def _tuple_of_str(value: Any, key_path: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RobotProfileError(f"{key_path} must be a sequence of strings")
    values = tuple(value)
    if any(not isinstance(item, str) or not item for item in values):
        raise RobotProfileError(f"{key_path} must contain only non-empty strings")
    return values


def _required_tuple_of_int(data: Mapping[str, Any], key_path: str) -> tuple[int, ...]:
    value = _get_key_path(data, key_path)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RobotProfileError(f"{key_path} must be a sequence of integers")
    values = tuple(value)
    if any(not isinstance(item, int) for item in values):
        raise RobotProfileError(f"{key_path} must contain only integers")
    return values


def _required_tuple_of_float(data: Mapping[str, Any], key_path: str) -> tuple[float, ...]:
    value = _get_key_path(data, key_path)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RobotProfileError(f"{key_path} must be a sequence of numbers")
    values = tuple(value)
    if any(not isinstance(item, int | float) for item in values):
        raise RobotProfileError(f"{key_path} must contain only numbers")
    return tuple(float(item) for item in values)


def _get_key_path(data: Mapping[str, Any], key_path: str) -> Any:
    key = key_path.split(".")[-1]
    if key not in data:
        raise RobotProfileError(f"{key_path} is required")
    return data[key]


def _validate_joint_orders(joint_order: JointOrderSpec, dof_count: int) -> None:
    _expect_length(joint_order.command_mujoco, dof_count, "joint_order.command_mujoco")
    _expect_length(joint_order.policy_isaaclab, dof_count, "joint_order.policy_isaaclab")
    banned = [
        joint_name
        for joint_name in joint_order.command_mujoco + joint_order.policy_isaaclab
        if any(token in joint_name.lower() for token in _BANNED_BODY_JOINT_TOKENS)
    ]
    if banned:
        raise RobotProfileError(f"body profile must not include hand/finger joints: {banned[0]}")
    if set(joint_order.command_mujoco) != set(joint_order.policy_isaaclab):
        raise RobotProfileError("joint_order command/policy sets must match")
    if len(set(joint_order.command_mujoco)) != dof_count:
        raise RobotProfileError("joint_order.command_mujoco must not contain duplicate joints")
    if len(set(joint_order.policy_isaaclab)) != dof_count:
        raise RobotProfileError("joint_order.policy_isaaclab must not contain duplicate joints")


def _validate_mappings(
    mapping: JointMappingSpec,
    joint_order: JointOrderSpec,
    dof_count: int,
) -> None:
    command_to_policy = mapping.command_mujoco_index_to_policy_isaaclab_index
    policy_to_command = mapping.policy_isaaclab_index_to_command_mujoco_index
    _expect_permutation(command_to_policy, dof_count, "mapping.command_mujoco_index_to_policy")
    _expect_permutation(policy_to_command, dof_count, "mapping.policy_isaaclab_index_to_command")
    for command_index, policy_index in enumerate(command_to_policy):
        if policy_to_command[policy_index] != command_index:
            raise RobotProfileError("mapping arrays must be inverse permutations")
        if joint_order.command_mujoco[command_index] != joint_order.policy_isaaclab[policy_index]:
            raise RobotProfileError("mapping does not match command/policy joint names")
    for policy_index, command_index in enumerate(policy_to_command):
        if joint_order.policy_isaaclab[policy_index] != joint_order.command_mujoco[command_index]:
            raise RobotProfileError("mapping does not match policy/command joint names")


def _validate_control(control: ControlProfileSpec, dof_count: int) -> None:
    if control.order != "command_mujoco":
        raise RobotProfileError("control.order must be command_mujoco")
    if control.raw_policy_action_order != "policy_isaaclab":
        raise RobotProfileError("control.raw_policy_action_order must be policy_isaaclab")
    for key, values in (
        ("control.default_angles_rad", control.default_angles_rad),
        ("control.action_scales_rad", control.action_scales_rad),
        ("control.kp", control.kp),
        ("control.kv", control.kv),
        ("control.force_limits", control.force_limits),
    ):
        _expect_length(values, dof_count, key)


def _expect_length(values: Sequence[object], expected: int, key_path: str) -> None:
    if len(values) != expected:
        raise RobotProfileError(f"{key_path} length must be {expected}, got {len(values)}")


def _expect_permutation(values: Sequence[int], expected: int, key_path: str) -> None:
    _expect_length(values, expected, key_path)
    if sorted(values) != list(range(expected)):
        raise RobotProfileError(f"{key_path} must be a 0..{expected - 1} permutation")
