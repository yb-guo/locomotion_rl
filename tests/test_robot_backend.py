from h200_locomotion_lab.envs.robot_backend import (
    G1MotorCommand,
    G1RobotState,
    LogReplayG1RobotBackend,
    robot_state_to_planner_qpos,
    robot_state_to_sonic_history_frame,
)
from h200_locomotion_lab.sonic.g1_observation import SONIC_ACTION_DIM
from h200_locomotion_lab.sonic.g1_planner_encoder import SONIC_PLANNER_QPOS_DIM


def test_robot_state_to_planner_qpos_returns_36d_root_plus_motors() -> None:
    state = G1RobotState(
        root_qpos=(0.1, 0.2, 0.79, 1.0, 0.0, 0.0, 0.0),
        motor_positions_mujoco=tuple(float(index) for index in range(SONIC_ACTION_DIM)),
    )

    qpos = robot_state_to_planner_qpos(state)

    assert len(qpos) == SONIC_PLANNER_QPOS_DIM
    assert qpos[:7] == (0.1, 0.2, 0.79, 1.0, 0.0, 0.0, 0.0)
    assert qpos[7:] == tuple(float(index) for index in range(SONIC_ACTION_DIM))


def test_robot_state_to_sonic_history_frame_keeps_last_raw_action() -> None:
    action = tuple(0.01 * index for index in range(SONIC_ACTION_DIM))
    state = G1RobotState(
        root_qpos=(0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0),
        motor_positions_mujoco=(0.0,) * SONIC_ACTION_DIM,
        motor_velocities_mujoco=(0.1,) * SONIC_ACTION_DIM,
        base_angular_velocity=(1.0, 2.0, 3.0),
        last_action_isaaclab=action,
    )

    frame = robot_state_to_sonic_history_frame(state)

    assert frame.base_ang_vel == (1.0, 2.0, 3.0)
    assert len(frame.body_q) == SONIC_ACTION_DIM
    assert len(frame.body_dq) == SONIC_ACTION_DIM
    assert frame.last_action == action
    assert frame.base_quat == (1.0, 0.0, 0.0, 0.0)


def test_log_replay_backend_records_commands_and_advances_state() -> None:
    qpos_rows = (
        (0.0, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0) + (0.0,) * SONIC_ACTION_DIM,
        (0.1, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0) + (0.01,) * SONIC_ACTION_DIM,
    )
    backend = LogReplayG1RobotBackend.from_mujoco_qpos_rows(qpos_rows)
    first = backend.reset()

    command = G1MotorCommand.from_raw_sonic_action((0.5,) * SONIC_ACTION_DIM)
    backend.write_command(command)
    second = backend.advance()

    assert first.root_z == 0.79
    assert second.root_qpos[0] == 0.1
    assert second.last_action_isaaclab == (0.5,) * SONIC_ACTION_DIM
    assert second.motor_velocities_mujoco == (0.5,) * SONIC_ACTION_DIM
    assert backend.commands == [command]


def test_log_replay_backend_rejects_wrong_qpos_width() -> None:
    try:
        LogReplayG1RobotBackend.from_mujoco_qpos_rows(((0.0,) * 35,))
    except ValueError as exc:
        assert "expected dim=36" in str(exc)
    else:
        raise AssertionError("expected ValueError")

