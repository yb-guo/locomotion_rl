"""Fixed G1-like semantic action slots and robot-order mappings."""

from collections.abc import Sequence
from dataclasses import dataclass

from h200_locomotion_lab.robots.g1_27dof_nohand import G1_27DOF_NOHAND_ACTUATOR_ORDER

G1LIKE_ACTION_SLOT_NAMES = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)
G1LIKE_ACTION_DIM = 29
G1LIKE_MISSING_FROM_27DOF_NOHAND = ("waist_roll", "waist_pitch")
G1_29DOF_COMMAND_MUJOCO_ACTUATOR_ORDER = tuple(
    f"{slot_name}_joint" for slot_name in G1LIKE_ACTION_SLOT_NAMES
)


class G1LikeSlotError(ValueError):
    """Raised when a G1-like slot schema or mapping is invalid."""


@dataclass(frozen=True, slots=True)
class G1LikeSlotSchema:
    """Fixed semantic slot names for G1-like action vectors."""

    slot_names: tuple[str, ...] = G1LIKE_ACTION_SLOT_NAMES

    def __post_init__(self) -> None:
        slot_names = tuple(self.slot_names)
        object.__setattr__(self, "slot_names", slot_names)
        if len(slot_names) != G1LIKE_ACTION_DIM:
            raise G1LikeSlotError(f"slot schema must contain {G1LIKE_ACTION_DIM} slots")
        if len(set(slot_names)) != len(slot_names):
            raise G1LikeSlotError("slot schema must not contain duplicate semantic slots")
        for slot_name in slot_names:
            if not slot_name or slot_name.endswith("_joint"):
                raise G1LikeSlotError(
                    "slot names must be non-empty semantic names without _joint suffix"
                )

    @property
    def action_dim(self) -> int:
        return len(self.slot_names)

    def slot_index(self, slot_name: str) -> int:
        try:
            return self.slot_names.index(slot_name)
        except ValueError as error:
            raise G1LikeSlotError(f"unknown G1-like slot: {slot_name}") from error


_DEFAULT_SLOT_SCHEMA = G1LikeSlotSchema()


@dataclass(frozen=True, slots=True)
class G1LikeSlotMapping:
    """Mapping from fixed G1-like slots to a robot actuator order."""

    schema: G1LikeSlotSchema
    robot_joint_order: tuple[str, ...]
    selector: tuple[int, ...]
    mask: tuple[bool, ...]

    @property
    def action_dim(self) -> int:
        return self.schema.action_dim

    @property
    def robot_action_dim(self) -> int:
        return len(self.robot_joint_order)

    @property
    def missing_slot_names(self) -> tuple[str, ...]:
        return tuple(
            slot_name
            for slot_name, is_present in zip(self.schema.slot_names, self.mask)
            if not is_present
        )

    def project_to_robot_order(self, unified_action: Sequence[float]) -> tuple[float, ...]:
        """Select present unified slots in robot actuator order."""

        if len(unified_action) != self.action_dim:
            raise G1LikeSlotError(
                f"unified action length must be {self.action_dim}, got {len(unified_action)}"
            )
        return tuple(float(unified_action[slot_index]) for slot_index in self.selector)

    def scatter_to_unified_slots(
        self, robot_action: Sequence[float], *, fill: float = 0.0
    ) -> tuple[float, ...]:
        """Scatter a robot-order action back into unified slots."""

        if len(robot_action) != self.robot_action_dim:
            raise G1LikeSlotError(
                f"robot action length must be {self.robot_action_dim}, got {len(robot_action)}"
            )
        unified = [float(fill) for _ in range(self.action_dim)]
        for robot_index, slot_index in enumerate(self.selector):
            unified[slot_index] = float(robot_action[robot_index])
        return tuple(unified)


def semantic_slot_name(joint_name: str) -> str:
    """Convert a robot joint name to a semantic slot name."""

    if not joint_name or not joint_name.endswith("_joint"):
        raise G1LikeSlotError(
            f"robot joint name must be a non-empty canonical name ending with _joint: {joint_name}"
        )
    return joint_name.removesuffix("_joint")


def build_g1like_slot_mapping(
    robot_joint_order: Sequence[str],
    *,
    schema: G1LikeSlotSchema = _DEFAULT_SLOT_SCHEMA,
) -> G1LikeSlotMapping:
    """Build a selector/mask from a robot actuator order into fixed G1-like slots."""

    joint_order = tuple(robot_joint_order)
    if len(set(joint_order)) != len(joint_order):
        raise G1LikeSlotError("robot joint order must not contain duplicate joints")

    slot_to_index = {slot_name: index for index, slot_name in enumerate(schema.slot_names)}
    selector: list[int] = []
    seen_slots: set[str] = set()
    for joint_name in joint_order:
        slot_name = semantic_slot_name(joint_name)
        if slot_name not in slot_to_index:
            raise G1LikeSlotError(f"unknown G1-like joint name: {joint_name}")
        if slot_name in seen_slots:
            raise G1LikeSlotError(f"duplicate semantic slot in robot order: {slot_name}")
        seen_slots.add(slot_name)
        selector.append(slot_to_index[slot_name])

    present_indices = set(selector)
    mask = tuple(index in present_indices for index in range(schema.action_dim))
    return G1LikeSlotMapping(
        schema=schema,
        robot_joint_order=joint_order,
        selector=tuple(selector),
        mask=mask,
    )


G1_29DOF_COMMAND_MUJOCO_SLOT_MAPPING = build_g1like_slot_mapping(
    G1_29DOF_COMMAND_MUJOCO_ACTUATOR_ORDER
)
G1_27DOF_NOHAND_SLOT_MAPPING = build_g1like_slot_mapping(G1_27DOF_NOHAND_ACTUATOR_ORDER)
