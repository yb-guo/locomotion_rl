"""Whole-body semantic joint slots shared by procedural and named robots.

The schema deliberately separates semantic slot order from a robot's XML joint
order.  It is the stable tensor contract used by both training and evaluation;
individual robot adapters own the mapping into this space.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from h200_locomotion_lab.core.checkpoint import (
    WHOLE_BODY_SCHEMA_HASH as CORE_WHOLE_BODY_SCHEMA_HASH,
)
from h200_locomotion_lab.core.checkpoint import (
    WHOLE_BODY_SCHEMA_VERSION as CORE_WHOLE_BODY_SCHEMA_VERSION,
)

WHOLE_BODY_SCHEMA_VERSION = CORE_WHOLE_BODY_SCHEMA_VERSION

_LIMB_JOINTS = (
    "hip_yaw",
    "hip_roll",
    "hip_pitch",
    "knee_pitch",
    "ankle_roll",
    "ankle_pitch",
    "wheel",
)
_WAIST_JOINTS = ("waist_yaw", "waist_roll", "waist_pitch")
_ARM_JOINTS = (
    "shoulder_yaw",
    "shoulder_roll",
    "shoulder_pitch",
    "elbow_pitch",
    "wrist_yaw",
    "wrist_roll",
    "wrist_pitch",
)

WHOLE_BODY_SLOT_NAMES = tuple(
    f"limb{limb}_{joint}"
    for limb in range(4)
    for joint in _LIMB_JOINTS
) + _WAIST_JOINTS + tuple(
    f"{side}_{joint}"
    for side in ("left_arm", "right_arm")
    for joint in _ARM_JOINTS
)

WHOLE_BODY_ACTION_DIM = len(WHOLE_BODY_SLOT_NAMES)
WHOLE_BODY_ACTOR_OBS_DIM = 3 + 3 + 3 + 3 + 4 * WHOLE_BODY_ACTION_DIM + 1
WHOLE_BODY_OBSERVATION_LAYOUT = (
    ("base_linear_velocity", 3),
    ("base_angular_velocity", 3),
    ("projected_gravity", 3),
    ("velocity_command", 3),
    ("joint_position", WHOLE_BODY_ACTION_DIM),
    ("joint_velocity", WHOLE_BODY_ACTION_DIM),
    ("previous_action", WHOLE_BODY_ACTION_DIM),
    ("active_joint_mask", WHOLE_BODY_ACTION_DIM),
    ("trial_start", 1),
)
_COMPUTED_WHOLE_BODY_SCHEMA_HASH = hashlib.sha256(
    json.dumps(
        {
            "version": WHOLE_BODY_SCHEMA_VERSION,
            "slots": WHOLE_BODY_SLOT_NAMES,
            "observation": WHOLE_BODY_OBSERVATION_LAYOUT,
        },
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
if _COMPUTED_WHOLE_BODY_SCHEMA_HASH != CORE_WHOLE_BODY_SCHEMA_HASH:
    raise RuntimeError("whole-body core/schema hash drifted; update the versioned contract")
WHOLE_BODY_SCHEMA_HASH = CORE_WHOLE_BODY_SCHEMA_HASH


class WholeBodySlotError(ValueError):
    """Raised when a whole-body schema or mapping is invalid."""


@dataclass(frozen=True, slots=True)
class WholeBodySlotSchema:
    """Immutable semantic slot schema and its versioned tensor dimensions."""

    version: str = WHOLE_BODY_SCHEMA_VERSION
    slot_names: tuple[str, ...] = WHOLE_BODY_SLOT_NAMES

    def __post_init__(self) -> None:
        names = tuple(self.slot_names)
        object.__setattr__(self, "slot_names", names)
        if self.version != WHOLE_BODY_SCHEMA_VERSION:
            raise WholeBodySlotError(
                f"unsupported schema version {self.version!r}; "
                f"expected {WHOLE_BODY_SCHEMA_VERSION!r}"
            )
        if len(names) != WHOLE_BODY_ACTION_DIM:
            raise WholeBodySlotError(f"schema must contain {WHOLE_BODY_ACTION_DIM} slots")
        if len(set(names)) != len(names):
            raise WholeBodySlotError("schema slot names must be unique")

    @property
    def action_dim(self) -> int:
        return len(self.slot_names)

    @property
    def actor_obs_dim(self) -> int:
        return WHOLE_BODY_ACTOR_OBS_DIM

    def slot_index(self, slot_name: str) -> int:
        try:
            return self.slot_names.index(slot_name)
        except ValueError as exc:
            raise WholeBodySlotError(f"unknown whole-body slot {slot_name!r}") from exc


@dataclass(frozen=True, slots=True)
class WholeBodySlotMapping:
    """Mapping between a robot actuator order and the unified 45-slot order."""

    schema: WholeBodySlotSchema
    robot_joint_order: tuple[str, ...]
    selector: tuple[int, ...]
    mask: tuple[bool, ...]
    semantic_slots: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.robot_joint_order) != len(self.selector):
            raise WholeBodySlotError("robot order and selector must have equal lengths")
        if len(self.mask) != self.schema.action_dim:
            raise WholeBodySlotError("mask must have schema action dimension")
        if len(self.semantic_slots) != len(self.robot_joint_order):
            raise WholeBodySlotError("semantic slot count must match robot joint count")
        if any(index < 0 or index >= self.schema.action_dim for index in self.selector):
            raise WholeBodySlotError("selector contains an out-of-range slot index")
        if len(set(self.selector)) != len(self.selector):
            raise WholeBodySlotError("a robot actuator cannot map to duplicate slots")

    @property
    def action_dim(self) -> int:
        return self.schema.action_dim

    @property
    def robot_action_dim(self) -> int:
        return len(self.robot_joint_order)

    @property
    def active_count(self) -> int:
        return sum(self.mask)

    def project_to_robot_order(self, unified_action: Sequence[float]) -> tuple[float, ...]:
        """Gather active unified action values in robot actuator order."""

        if len(unified_action) != self.action_dim:
            raise WholeBodySlotError(
                f"unified action must have length {self.action_dim}, got {len(unified_action)}"
            )
        return tuple(float(unified_action[index]) for index in self.selector)

    def scatter_to_unified_slots(
        self, robot_values: Sequence[float], *, fill: float = 0.0
    ) -> tuple[float, ...]:
        """Scatter robot-order values into the unified space."""

        if len(robot_values) != self.robot_action_dim:
            raise WholeBodySlotError(
                f"robot values must have length {self.robot_action_dim}, got {len(robot_values)}"
            )
        unified = [float(fill)] * self.action_dim
        for value, slot_index in zip(robot_values, self.selector):
            unified[slot_index] = float(value)
        return tuple(unified)

    def mask_values(self, values: Sequence[float], *, fill: float = 0.0) -> tuple[float, ...]:
        """Apply the active mask to a unified vector."""

        if len(values) != self.action_dim:
            raise WholeBodySlotError(
                f"values must have length {self.action_dim}, got {len(values)}"
            )
        return tuple(float(value) if active else float(fill) for value, active in zip(values, self.mask))

    def round_trip(self, robot_values: Sequence[float]) -> tuple[float, ...]:
        """Convenience contract used by tests and adapter smoke probes."""

        return self.project_to_robot_order(self.scatter_to_unified_slots(robot_values))


def build_whole_body_slot_mapping(
    robot_joint_order: Sequence[str],
    *,
    semantic_slots: Mapping[str, str] | None = None,
    schema: WholeBodySlotSchema | None = None,
) -> WholeBodySlotMapping:
    """Build an explicit mapping from robot names to semantic slots.

    ``semantic_slots`` is required for non-canonical names such as the Berkeley
    Humanoid's ``LL_HR``.  Canonical names are accepted as a convenience for
    generated robots and Unitree-style XML files.
    """

    schema = schema or WHOLE_BODY_SCHEMA
    robot_order = tuple(robot_joint_order)
    if len(set(robot_order)) != len(robot_order):
        raise WholeBodySlotError("robot joint names must be unique")
    aliases = semantic_slots or {}
    slot_to_index = {name: index for index, name in enumerate(schema.slot_names)}
    selector: list[int] = []
    resolved: list[str] = []
    seen: set[str] = set()
    for joint_name in robot_order:
        slot_name = (
            aliases[joint_name]
            if joint_name in aliases
            else canonical_semantic_slot(joint_name)
        )
        if slot_name not in slot_to_index:
            raise WholeBodySlotError(
                f"joint {joint_name!r} resolved to unknown whole-body slot {slot_name!r}"
            )
        if slot_name in seen:
            raise WholeBodySlotError(f"duplicate semantic slot {slot_name!r}")
        seen.add(slot_name)
        resolved.append(slot_name)
        selector.append(slot_to_index[slot_name])
    mask = tuple(index in set(selector) for index in range(schema.action_dim))
    return WholeBodySlotMapping(
        schema=schema,
        robot_joint_order=robot_order,
        selector=tuple(selector),
        mask=mask,
        semantic_slots=tuple(resolved),
    )


def canonical_semantic_slot(joint_name: str) -> str:
    """Resolve common Unitree/procedural names to a 45-slot semantic name."""

    name = joint_name.strip().lower().removesuffix("_joint")
    side_prefix = {"left": 0, "right": 1, "front_left": 0, "front_right": 1,
                   "rear_left": 2, "rear_right": 3}
    for prefix, limb in side_prefix.items():
        if name.startswith(prefix + "_"):
            suffix = name[len(prefix) + 1 :]
            if suffix in _LIMB_JOINTS:
                return f"limb{limb}_{suffix}"
            if prefix in {"left", "right"} and suffix in _ARM_JOINTS:
                return f"{prefix}_arm_{suffix}"
    if name in _WAIST_JOINTS:
        return name
    if name in WHOLE_BODY_SLOT_NAMES:
        return name
    raise WholeBodySlotError(
        f"cannot infer semantic slot for {joint_name!r}; pass an explicit semantic_slots map"
    )


WHOLE_BODY_SCHEMA = WholeBodySlotSchema()


G1_WHOLE_BODY_JOINT_ORDER = tuple(
    f"{slot}_joint"
    for slot in (
        "left_hip_pitch",
        "left_hip_roll",
        "left_hip_yaw",
        "left_knee_pitch",
        "left_ankle_pitch",
        "left_ankle_roll",
        "right_hip_pitch",
        "right_hip_roll",
        "right_hip_yaw",
        "right_knee_pitch",
        "right_ankle_pitch",
        "right_ankle_roll",
        "waist_yaw",
        "waist_roll",
        "waist_pitch",
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow_pitch",
        "left_wrist_roll",
        "left_wrist_pitch",
        "left_wrist_yaw",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow_pitch",
        "right_wrist_roll",
        "right_wrist_pitch",
        "right_wrist_yaw",
    )
)

BERKELEY_HUMANOID_JOINT_ORDER = (
    "LL_HR",
    "LL_HAA",
    "LL_HFE",
    "LL_KFE",
    "LL_FFE",
    "LL_FAA",
    "LR_HR",
    "LR_HAA",
    "LR_HFE",
    "LR_KFE",
    "LR_FFE",
    "LR_FAA",
)
BERKELEY_HUMANOID_SEMANTIC_SLOTS = {
    "LL_HR": "limb0_hip_yaw",
    "LL_HAA": "limb0_hip_roll",
    "LL_HFE": "limb0_hip_pitch",
    "LL_KFE": "limb0_knee_pitch",
    "LL_FFE": "limb0_ankle_pitch",
    "LL_FAA": "limb0_ankle_roll",
    "LR_HR": "limb1_hip_yaw",
    "LR_HAA": "limb1_hip_roll",
    "LR_HFE": "limb1_hip_pitch",
    "LR_KFE": "limb1_knee_pitch",
    "LR_FFE": "limb1_ankle_pitch",
    "LR_FAA": "limb1_ankle_roll",
}

ANYMAL_C_JOINT_ORDER = (
    "LF_HAA", "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE",
)
ANYMAL_C_SEMANTIC_SLOTS = {
    **{f"LF_{suffix}": f"limb0_{semantic}" for suffix, semantic in
       (("HAA", "hip_roll"), ("HFE", "hip_pitch"), ("KFE", "knee_pitch"))},
    **{f"RF_{suffix}": f"limb1_{semantic}" for suffix, semantic in
       (("HAA", "hip_roll"), ("HFE", "hip_pitch"), ("KFE", "knee_pitch"))},
    **{f"LH_{suffix}": f"limb2_{semantic}" for suffix, semantic in
       (("HAA", "hip_roll"), ("HFE", "hip_pitch"), ("KFE", "knee_pitch"))},
    **{f"RH_{suffix}": f"limb3_{semantic}" for suffix, semantic in
       (("HAA", "hip_roll"), ("HFE", "hip_pitch"), ("KFE", "knee_pitch"))},
}
GO2_JOINT_ORDER = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
GO2_SEMANTIC_SLOTS = {
    **{f"FL_{suffix}_joint": f"limb0_{semantic}" for suffix, semantic in
       (("hip", "hip_roll"), ("thigh", "hip_pitch"), ("calf", "knee_pitch"))},
    **{f"FR_{suffix}_joint": f"limb1_{semantic}" for suffix, semantic in
       (("hip", "hip_roll"), ("thigh", "hip_pitch"), ("calf", "knee_pitch"))},
    **{f"RL_{suffix}_joint": f"limb2_{semantic}" for suffix, semantic in
       (("hip", "hip_roll"), ("thigh", "hip_pitch"), ("calf", "knee_pitch"))},
    **{f"RR_{suffix}_joint": f"limb3_{semantic}" for suffix, semantic in
       (("hip", "hip_roll"), ("thigh", "hip_pitch"), ("calf", "knee_pitch"))},
}


def build_g1_whole_body_mapping() -> WholeBodySlotMapping:
    return build_whole_body_slot_mapping(G1_WHOLE_BODY_JOINT_ORDER)


def build_berkeley_humanoid_mapping() -> WholeBodySlotMapping:
    return build_whole_body_slot_mapping(
        BERKELEY_HUMANOID_JOINT_ORDER,
        semantic_slots=BERKELEY_HUMANOID_SEMANTIC_SLOTS,
    )


def build_anymal_c_mapping() -> WholeBodySlotMapping:
    return build_whole_body_slot_mapping(
        ANYMAL_C_JOINT_ORDER,
        semantic_slots=ANYMAL_C_SEMANTIC_SLOTS,
    )


def build_go2_mapping() -> WholeBodySlotMapping:
    return build_whole_body_slot_mapping(
        GO2_JOINT_ORDER,
        semantic_slots=GO2_SEMANTIC_SLOTS,
    )
