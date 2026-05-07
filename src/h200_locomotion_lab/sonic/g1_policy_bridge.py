"""GEAR-SONIC G1 policy output bridge.

The official C++ deploy path emits raw policy actions in IsaacLab order. It then
maps them to MuJoCo/hardware order and applies per-joint position scaling before
sending PD targets.
"""

from __future__ import annotations

import math
from typing import Sequence


SONIC_G1_ISAACLAB_TO_MUJOCO: tuple[int, ...] = (
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
)

SONIC_G1_MUJOCO_INDEX_TO_POLICY_INDEX = SONIC_G1_ISAACLAB_TO_MUJOCO
SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX: tuple[int, ...] = tuple(
    SONIC_G1_MUJOCO_INDEX_TO_POLICY_INDEX.index(policy_index)
    for policy_index in range(len(SONIC_G1_MUJOCO_INDEX_TO_POLICY_INDEX))
)

SONIC_G1_DEFAULT_ANGLES: tuple[float, ...] = (
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
)

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

SONIC_G1_ACTION_SCALES: tuple[float, ...] = (
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
    0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
    0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,
    0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,
    0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,
)

SONIC_ACTION_DIM = 29


def sonic_policy_action_to_mujoco_targets(
    raw_action_isaaclab: Sequence[float],
) -> tuple[float, ...]:
    """Map raw SONIC policy action to MuJoCo-order motor position targets."""

    if len(raw_action_isaaclab) != SONIC_ACTION_DIM:
        raise ValueError(
            f"Expected SONIC raw action_dim={SONIC_ACTION_DIM}, got {len(raw_action_isaaclab)}"
        )
    action = tuple(float(value) for value in raw_action_isaaclab)
    return tuple(
        SONIC_G1_DEFAULT_ANGLES[mujoco_index]
        + action[isaaclab_index] * SONIC_G1_ACTION_SCALES[mujoco_index]
        for mujoco_index, isaaclab_index in enumerate(SONIC_G1_ISAACLAB_TO_MUJOCO)
    )
