"""Robot profile loading and validation."""

from h200_locomotion_lab.robots.g1_27dof_nohand import (
    DEFAULT_UNITREE_G1_27DOF_NOHAND_GENESIS_PROFILE,
    EXPECTED_G1_27DOF_ACTION_DIM,
    G1_27DOF_NOHAND_ACTUATOR_ORDER,
    REMOVED_FROM_G1_29DOF_COMMAND_ORDER,
    G1NoHandAssetMetadata,
    G1NoHandComponentEvidence,
    G1NoHandControlProfile,
    G1NoHandGenesisTrainingProfile,
    G1NoHandH200Evidence,
    G1NoHandObservationSegment,
    G1NoHandProfileError,
    G1NoHandTrainingContract,
    load_g1_27dof_nohand_profile,
    load_g1_27dof_nohand_profile_dict,
)
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
    "DEFAULT_UNITREE_G1_27DOF_NOHAND_GENESIS_PROFILE",
    "DEFAULT_UNITREE_G1_29DOF_SONIC_PROFILE",
    "EXPECTED_G1_27DOF_ACTION_DIM",
    "G1_27DOF_NOHAND_ACTUATOR_ORDER",
    "G1NoHandAssetMetadata",
    "G1NoHandComponentEvidence",
    "G1NoHandControlProfile",
    "G1NoHandGenesisTrainingProfile",
    "G1NoHandH200Evidence",
    "G1NoHandObservationSegment",
    "G1NoHandProfileError",
    "G1NoHandTrainingContract",
    "JointMappingSpec",
    "JointOrderSpec",
    "REMOVED_FROM_G1_29DOF_COMMAND_ORDER",
    "RobotProfileError",
    "RobotProfileMetadata",
    "load_g1_27dof_nohand_profile",
    "load_g1_27dof_nohand_profile_dict",
    "load_robot_profile",
    "load_robot_profile_dict",
]
