from pathlib import Path

import pytest

from h200_locomotion_lab.sonic.g1_planner_encoder import (
    SONIC_ENCODER_OBS_DIM,
    SONIC_G1_ENCODER_FIELDS,
    SONIC_PLANNER_ALLOWED_PRED_NUM_TOKENS,
    SONIC_PLANNER_DEFAULT_HEIGHT,
    build_g1_encoder_observation_from_planner_motion,
    build_initial_planner_context,
    build_planner_inputs,
    encoder_field_by_name,
    resample_planner_mujoco_qpos_to_50hz,
)
from h200_locomotion_lab.sonic.g1_policy_bridge import SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX
from h200_locomotion_lab.tools.sonic_planner_encoder_decoder_forward import read_planner_qpos_csv


def test_build_initial_planner_context_repeats_official_standing_qpos() -> None:
    joints = tuple(float(index) for index in range(29))

    context = build_initial_planner_context(joints)

    assert len(context) == 4
    assert all(len(frame) == 36 for frame in context)
    assert context[0][:7] == (0.0, 0.0, SONIC_PLANNER_DEFAULT_HEIGHT, 1.0, 0.0, 0.0, 0.0)
    assert context[0][7:] == joints
    assert context[1] == context[0]


def test_build_planner_inputs_matches_official_defaults() -> None:
    inputs = build_planner_inputs(mode=2)

    assert inputs.mode == 2
    assert inputs.target_vel == -1.0
    assert inputs.movement_direction == (1.0, 0.0, 0.0)
    assert inputs.facing_direction == (1.0, 0.0, 0.0)
    assert inputs.random_seed == 1234
    assert inputs.has_specific_target == 0
    assert inputs.allowed_pred_num_tokens == SONIC_PLANNER_ALLOWED_PRED_NUM_TOKENS


def test_resample_planner_qpos_maps_mujoco_joints_to_policy_order() -> None:
    frames = []
    for frame_index in range(64):
        joints_mujoco = tuple(1000.0 * frame_index + joint for joint in range(29))
        frames.append((0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, *joints_mujoco))

    motion = resample_planner_mujoco_qpos_to_50hz(frames, num_pred_frames=64)

    assert motion.timesteps == 106
    first_policy_joints = motion.joint_positions_policy_order[0]
    for policy_index, mujoco_index in enumerate(SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX):
        assert first_policy_joints[policy_index] == pytest.approx(float(mujoco_index))
    assert motion.joint_velocities_policy_order[0][0] == pytest.approx(30000.0)


def test_build_g1_encoder_observation_fills_only_mode0_required_fields() -> None:
    frames = []
    for frame_index in range(64):
        joints_mujoco = tuple(0.01 * frame_index + 0.001 * joint for joint in range(29))
        frames.append((0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, *joints_mujoco))
    motion = resample_planner_mujoco_qpos_to_50hz(frames, num_pred_frames=64)

    obs = build_g1_encoder_observation_from_planner_motion(motion)

    assert len(obs) == SONIC_ENCODER_OBS_DIM
    assert obs[encoder_field_by_name("encoder_mode_4").offset] == 0.0
    for field in SONIC_G1_ENCODER_FIELDS:
        values = obs[field.offset : field.offset + field.dim]
        if field.required_for_g1 and field.name != "encoder_mode_4":
            assert any(abs(value) > 0.0 for value in values)
        elif not field.required_for_g1:
            assert all(value == 0.0 for value in values)


def test_read_planner_qpos_csv_reads_36d_rows() -> None:
    path = Path(__file__).parent / "fixtures" / "planner_qpos_sample.csv"
    rows = read_planner_qpos_csv(path)

    assert rows == (tuple(float(index) for index in range(36)),)
