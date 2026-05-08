"""Robot profile loading and validation."""

from h200_locomotion_lab.robots.loader import (
    DEFAULT_UNITREE_G1_29DOF_SONIC_PROFILE,
    RobotProfileError,
    load_robot_profile,
    load_robot_profile_dict,
)
from h200_locomotion_lab.robots.spec import (
    CompiledRobotProfile,
    ControlProfileSpec,
    JointMappingSpec,
    JointOrderSpec,
    RobotProfileMetadata,
)

__all__ = [
    "CompiledRobotProfile",
    "ControlProfileSpec",
    "DEFAULT_UNITREE_G1_29DOF_SONIC_PROFILE",
    "JointMappingSpec",
    "JointOrderSpec",
    "RobotProfileError",
    "RobotProfileMetadata",
    "load_robot_profile",
    "load_robot_profile_dict",
]
