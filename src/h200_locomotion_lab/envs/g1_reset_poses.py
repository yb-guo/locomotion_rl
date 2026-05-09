"""Standing reset pose candidates for the G1 27DoF no-hand env."""

from __future__ import annotations

from h200_locomotion_lab.robots import G1_27DOF_NOHAND_ACTUATOR_ORDER


G1_STANDING_RESET_POSE_NAMES = (
    "profile",
    "half_crouch",
    "mild_crouch",
    "tall_crouch",
    "straight",
)


def build_g1_standing_reset_pose_candidates(
    base_pose: tuple[float, ...],
) -> dict[str, tuple[float, ...]]:
    return {
        "profile": tuple(base_pose),
        "half_crouch": with_leg_pose(
            base_pose,
            hip_pitch=-0.20,
            knee=0.45,
            ankle_pitch=-0.25,
        ),
        "mild_crouch": with_leg_pose(
            base_pose,
            hip_pitch=-0.12,
            knee=0.25,
            ankle_pitch=-0.14,
        ),
        "tall_crouch": with_leg_pose(
            base_pose,
            hip_pitch=-0.06,
            knee=0.12,
            ankle_pitch=-0.07,
        ),
        "straight": with_leg_pose(
            base_pose,
            hip_pitch=0.0,
            knee=0.0,
            ankle_pitch=0.0,
        ),
    }


def with_leg_pose(
    base_pose: tuple[float, ...],
    *,
    hip_pitch: float,
    knee: float,
    ankle_pitch: float,
) -> tuple[float, ...]:
    values = list(base_pose)
    for joint_name, value in (
        ("left_hip_pitch_joint", hip_pitch),
        ("right_hip_pitch_joint", hip_pitch),
        ("left_knee_joint", knee),
        ("right_knee_joint", knee),
        ("left_ankle_pitch_joint", ankle_pitch),
        ("right_ankle_pitch_joint", ankle_pitch),
    ):
        values[G1_27DOF_NOHAND_ACTUATOR_ORDER.index(joint_name)] = value
    return tuple(values)


def leg_value_summary(pose: tuple[float, ...]) -> dict[str, float]:
    return {
        "hip_pitch": pose[G1_27DOF_NOHAND_ACTUATOR_ORDER.index("left_hip_pitch_joint")],
        "knee": pose[G1_27DOF_NOHAND_ACTUATOR_ORDER.index("left_knee_joint")],
        "ankle_pitch": pose[
            G1_27DOF_NOHAND_ACTUATOR_ORDER.index("left_ankle_pitch_joint")
        ],
    }
