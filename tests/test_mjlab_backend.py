from __future__ import annotations

import pytest

from h200_locomotion_lab.envs.genesis_adapter import G1_29DOF_JOINT_ORDER
from h200_locomotion_lab.envs.mjlab_backend import MjlabG1RobotBackend
from h200_locomotion_lab.runtime.scalar_g1_runtime import ScalarG1Runtime, ZeroActionProvider
from h200_locomotion_lab.sonic.g1_policy_bridge import SONIC_G1_DEFAULT_ANGLES


class FakeData:
    def __init__(self, joint_names: tuple[str, ...]) -> None:
        self.root_link_pos_w = [[0.1, 0.2, 0.8]]
        self.root_link_quat_w = [[1.0, 0.0, 0.0, 0.0]]
        self.root_link_ang_vel_b = [[0.01, 0.02, 0.03]]
        self.joint_pos = [[0.0 for _ in joint_names]]
        self.joint_vel = [[0.0 for _ in joint_names]]


class FakeRobot:
    def __init__(self, joint_names: tuple[str, ...]) -> None:
        self.joint_names = joint_names
        self.data = FakeData(joint_names)


class FakeActionTerm:
    def __init__(
        self,
        target_names: tuple[str, ...],
        *,
        scale: tuple[float, ...],
        offset: tuple[float, ...],
    ) -> None:
        self.target_names = list(target_names)
        self.scale = [list(scale)]
        self.offset = [list(offset)]
        self.raw_action = None


class FakeActionManager:
    def __init__(self, action_term: FakeActionTerm) -> None:
        self.action_term = action_term

    def get_term(self, name: str) -> FakeActionTerm:
        assert name == "joint_pos"
        return self.action_term


class FakeMjlabEnv:
    def __init__(self, action_term: FakeActionTerm) -> None:
        self.robot = FakeRobot(G1_29DOF_JOINT_ORDER)
        self.scene = {"robot": self.robot}
        self.action_manager = FakeActionManager(action_term)
        self.last_action = None
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def step(self, action_batch):
        self.last_action = tuple(float(value) for value in action_batch[0])
        target_by_action_name = {
            name: offset + scale * action
            for name, offset, scale, action in zip(
                self.action_manager.action_term.target_names,
                self.action_manager.action_term.offset[0],
                self.action_manager.action_term.scale[0],
                self.last_action,
            )
        }
        self.robot.data.joint_pos = [
            [target_by_action_name[name] for name in self.robot.joint_names]
        ]
        return None, [0.0], [False], {"fake": True}


def test_mjlab_backend_inverts_joint_position_action_by_name() -> None:
    target_names = tuple(reversed(G1_29DOF_JOINT_ORDER))
    scale = tuple(0.5 + index * 0.01 for index in range(29))
    offset = tuple(-0.2 + index * 0.02 for index in range(29))
    env = FakeMjlabEnv(FakeActionTerm(target_names, scale=scale, offset=offset))
    backend = MjlabG1RobotBackend(env)

    targets = tuple(index * 0.1 for index in range(29))
    action = backend.motor_targets_to_mjlab_action(targets)
    round_trip = backend.mjlab_action_to_motor_targets(action)

    assert round_trip == pytest.approx(targets)


def test_mjlab_backend_reads_state_in_sonic_order() -> None:
    target_names = G1_29DOF_JOINT_ORDER
    env = FakeMjlabEnv(
        FakeActionTerm(
            target_names,
            scale=(1.0,) * 29,
            offset=(0.0,) * 29,
        )
    )
    env.robot.data.joint_pos = [[float(index) for index in range(29)]]
    env.robot.data.joint_vel = [[float(index + 100) for index in range(29)]]

    state = MjlabG1RobotBackend(env).read_state()

    assert state.root_qpos == pytest.approx((0.1, 0.2, 0.8, 1.0, 0.0, 0.0, 0.0))
    assert state.motor_positions_mujoco == pytest.approx(tuple(float(i) for i in range(29)))
    assert state.motor_velocities_mujoco == pytest.approx(tuple(float(i + 100) for i in range(29)))
    assert state.base_angular_velocity == pytest.approx((0.01, 0.02, 0.03))


def test_scalar_runtime_drives_mjlab_backend_with_zero_sonic_action() -> None:
    env = FakeMjlabEnv(
        FakeActionTerm(
            G1_29DOF_JOINT_ORDER,
            scale=(0.5,) * 29,
            offset=(0.1,) * 29,
        )
    )
    backend = MjlabG1RobotBackend(env)
    runtime = ScalarG1Runtime(backend, ZeroActionProvider())

    step = runtime.step()

    assert env.reset_count == 1
    assert step.raw_action_isaaclab == (0.0,) * 29
    assert step.command.motor_position_targets_mujoco == pytest.approx(SONIC_G1_DEFAULT_ANGLES)
    assert step.next_state.motor_positions_mujoco == pytest.approx(SONIC_G1_DEFAULT_ANGLES)
