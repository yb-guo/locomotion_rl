"""Typed robot profile objects compiled from YAML."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RobotProfileMetadata:
    source: str
    mirrors: tuple[str, ...]
    note: str | None = None


@dataclass(frozen=True, slots=True)
class JointOrderSpec:
    command_mujoco: tuple[str, ...]
    policy_isaaclab: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JointMappingSpec:
    command_mujoco_index_to_policy_isaaclab_index: tuple[int, ...]
    policy_isaaclab_index_to_command_mujoco_index: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ControlProfileSpec:
    order: str
    raw_policy_action_order: str
    default_angles_rad: tuple[float, ...]
    action_scales_rad: tuple[float, ...]
    kp: tuple[float, ...]
    kv: tuple[float, ...]
    force_limits: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CompiledRobotProfile:
    name: str
    family: str
    dof_count: int
    body_profile: str
    metadata: RobotProfileMetadata
    joint_order: JointOrderSpec
    mapping: JointMappingSpec
    control: ControlProfileSpec
    source_path: Path | None = None

    @property
    def action_dim(self) -> int:
        return self.dof_count
