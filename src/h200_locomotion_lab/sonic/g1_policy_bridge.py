"""GEAR-SONIC G1 policy output bridge.

The official C++ deploy path emits raw policy actions in IsaacLab order. It then
maps them to MuJoCo/hardware order and applies per-joint position scaling before
sending PD targets.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from h200_locomotion_lab.robots import CompiledRobotProfile, load_robot_profile
from h200_locomotion_lab.runtime import ScalarActionBridge

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10 * 2.0 * math.pi

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ * NATURAL_FREQ

EFFORT_LIMIT_5020 = 25.0
EFFORT_LIMIT_7520_14 = 88.0
EFFORT_LIMIT_7520_22 = 139.0
EFFORT_LIMIT_4010 = 5.0

_DEFAULT_SONIC_G1_PROFILE = load_robot_profile()
_DEFAULT_SONIC_G1_ACTION_BRIDGE = ScalarActionBridge.from_profile(_DEFAULT_SONIC_G1_PROFILE)

SONIC_G1_ISAACLAB_TO_MUJOCO: tuple[int, ...] = (
    _DEFAULT_SONIC_G1_PROFILE.mapping.command_mujoco_index_to_policy_isaaclab_index
)
SONIC_G1_MUJOCO_INDEX_TO_POLICY_INDEX = SONIC_G1_ISAACLAB_TO_MUJOCO
SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX: tuple[int, ...] = (
    _DEFAULT_SONIC_G1_PROFILE.mapping.policy_isaaclab_index_to_command_mujoco_index
)
SONIC_G1_DEFAULT_ANGLES: tuple[float, ...] = (
    _DEFAULT_SONIC_G1_ACTION_BRIDGE.default_angles_command
)
SONIC_G1_ACTION_SCALES: tuple[float, ...] = _DEFAULT_SONIC_G1_ACTION_BRIDGE.action_scale_command
SONIC_ACTION_DIM = _DEFAULT_SONIC_G1_ACTION_BRIDGE.action_dim


def get_default_sonic_g1_profile() -> CompiledRobotProfile:
    """Return the cached default Unitree G1 SONIC runtime profile."""

    return _DEFAULT_SONIC_G1_PROFILE


def get_default_sonic_g1_action_bridge() -> ScalarActionBridge:
    """Return the cached profile-backed scalar bridge for SONIC G1 actions."""

    return _DEFAULT_SONIC_G1_ACTION_BRIDGE


def sonic_policy_action_to_mujoco_targets(
    raw_action_isaaclab: Sequence[float],
) -> tuple[float, ...]:
    """Map raw SONIC policy action to MuJoCo-order motor position targets.

    Compatibility facade for older imports. Runtime authority lives in the
    compiled robot profile and cached ``ScalarActionBridge`` above.
    """

    if len(raw_action_isaaclab) != SONIC_ACTION_DIM:
        raise ValueError(
            f"Expected SONIC raw action_dim={SONIC_ACTION_DIM}, got {len(raw_action_isaaclab)}"
        )
    return _DEFAULT_SONIC_G1_ACTION_BRIDGE.policy_action_to_command_targets(raw_action_isaaclab)
