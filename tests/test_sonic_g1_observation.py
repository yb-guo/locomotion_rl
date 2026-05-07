import pytest

from h200_locomotion_lab.sonic.g1_observation import (
    SONIC_DECODER_OBS_DIM,
    SONIC_G1_DECODER_OBSERVATION_FIELDS,
    SONIC_HISTORY_FRAMES,
    SonicG1HistoryBuffer,
    SonicG1HistoryFrame,
    build_sonic_g1_decoder_observation,
    field_by_name,
    gravity_dir_from_base_quat,
    mujoco_motor_state_to_sonic_body_state,
    sonic_g1_history_from_decoder_observation,
)
from h200_locomotion_lab.sonic.g1_policy_bridge import (
    SONIC_ACTION_DIM,
    SONIC_G1_DEFAULT_ANGLES,
    SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX,
)


def test_sonic_g1_decoder_layout_matches_official_release_config() -> None:
    fields = SONIC_G1_DECODER_OBSERVATION_FIELDS

    assert [(field.name, field.offset, field.dim) for field in fields] == [
        ("token_state", 0, 64),
        ("his_base_angular_velocity_10frame_step1", 64, 30),
        ("his_body_joint_positions_10frame_step1", 94, 290),
        ("his_body_joint_velocities_10frame_step1", 384, 290),
        ("his_last_actions_10frame_step1", 674, 290),
        ("his_gravity_dir_10frame_step1", 964, 30),
    ]
    assert fields[-1].offset + fields[-1].dim == SONIC_DECODER_OBS_DIM == 994


def test_build_sonic_decoder_observation_places_oldest_to_newest_history() -> None:
    token = tuple(float(index) for index in range(64))
    frames = tuple(_history_frame(frame_index) for frame_index in range(SONIC_HISTORY_FRAMES))

    obs = build_sonic_g1_decoder_observation(token, frames)

    assert len(obs) == SONIC_DECODER_OBS_DIM
    assert obs[:64] == token
    assert _field_slice(obs, "his_base_angular_velocity_10frame_step1")[:6] == (
        0.1,
        0.2,
        0.3,
        1.1,
        1.2,
        1.3,
    )
    assert _field_slice(obs, "his_body_joint_positions_10frame_step1")[:30] == (
        tuple(float(joint_index) for joint_index in range(SONIC_ACTION_DIM))
        + (100.0,)
    )
    assert _field_slice(obs, "his_body_joint_velocities_10frame_step1")[:30] == (
        tuple(float(joint_index) for joint_index in range(SONIC_ACTION_DIM))
        + (200.0,)
    )
    assert _field_slice(obs, "his_last_actions_10frame_step1")[:30] == (
        tuple(float(joint_index) for joint_index in range(SONIC_ACTION_DIM))
        + (300.0,)
    )
    assert _field_slice(obs, "his_gravity_dir_10frame_step1") == (0.0, 0.0, -1.0) * 10


def test_history_buffer_left_pads_like_official_state_logger() -> None:
    history = SonicG1HistoryBuffer()
    history.append(_history_frame(7))

    frames = history.latest_oldest_first()
    obs = build_sonic_g1_decoder_observation((0.0,) * 64, frames)

    assert len(frames) == 10
    assert _field_slice(obs, "his_body_joint_positions_10frame_step1")[: 9 * 29] == (
        0.0,
    ) * (9 * 29)
    assert _field_slice(obs, "his_body_joint_positions_10frame_step1")[9 * 29] == 700.0
    assert _field_slice(obs, "his_gravity_dir_10frame_step1")[: 9 * 3] == (0.0,) * (9 * 3)
    assert _field_slice(obs, "his_gravity_dir_10frame_step1")[-3:] == (0.0, 0.0, -1.0)


def test_decoder_observation_history_parse_round_trips_official_fields() -> None:
    token = tuple(float(index) for index in range(64))
    frames = tuple(_history_frame(frame_index) for frame_index in range(SONIC_HISTORY_FRAMES))
    observation = build_sonic_g1_decoder_observation(token, frames)

    parsed_token, parsed_frames = sonic_g1_history_from_decoder_observation(observation)
    rebuilt = build_sonic_g1_decoder_observation(parsed_token, parsed_frames)

    assert parsed_token == token
    assert rebuilt == observation


def test_mujoco_motor_state_to_sonic_body_state_uses_policy_order_and_default_centering() -> None:
    positions = tuple(
        SONIC_G1_DEFAULT_ANGLES[mujoco_index] + 10.0 + mujoco_index
        for mujoco_index in range(SONIC_ACTION_DIM)
    )
    velocities = tuple(100.0 + mujoco_index for mujoco_index in range(SONIC_ACTION_DIM))

    body_q, body_dq = mujoco_motor_state_to_sonic_body_state(positions, velocities)

    policy_index = 1
    mujoco_index = SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX[policy_index]
    assert body_q[policy_index] == pytest.approx(10.0 + mujoco_index)
    assert body_dq[policy_index] == pytest.approx(100.0 + mujoco_index)


def test_gravity_dir_uses_inverse_base_quaternion() -> None:
    assert gravity_dir_from_base_quat((1.0, 0.0, 0.0, 0.0)) == (0.0, 0.0, -1.0)
    assert gravity_dir_from_base_quat((0.0, 1.0, 0.0, 0.0)) == pytest.approx(
        (0.0, 0.0, 1.0)
    )
    assert gravity_dir_from_base_quat((0.0, 0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)


def test_build_sonic_decoder_observation_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError, match="token_state expected dim=64"):
        build_sonic_g1_decoder_observation((0.0,) * 63, tuple(_history_frame(i) for i in range(10)))

    with pytest.raises(ValueError, match="Expected 10 history frames"):
        build_sonic_g1_decoder_observation((0.0,) * 64, tuple(_history_frame(i) for i in range(9)))

    bad_frame = SonicG1HistoryFrame(
        base_ang_vel=(0.0, 0.0, 0.0),
        body_q=(0.0,) * 28,
        body_dq=(0.0,) * 29,
        last_action=(0.0,) * 29,
        base_quat=(1.0, 0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="body_q expected dim=29"):
        build_sonic_g1_decoder_observation((0.0,) * 64, (bad_frame,) * 10)


def _history_frame(frame_index: int) -> SonicG1HistoryFrame:
    return SonicG1HistoryFrame(
        base_ang_vel=(frame_index + 0.1, frame_index + 0.2, frame_index + 0.3),
        body_q=tuple(float(frame_index * 100 + joint_index) for joint_index in range(29)),
        body_dq=tuple(float(frame_index * 200 + joint_index) for joint_index in range(29)),
        last_action=tuple(float(frame_index * 300 + joint_index) for joint_index in range(29)),
        base_quat=(1.0, 0.0, 0.0, 0.0),
    )


def _field_slice(obs: tuple[float, ...], field_name: str) -> tuple[float, ...]:
    field = field_by_name(field_name)
    return obs[field.offset : field.offset + field.dim]
