import sys

import pytest

from h200_locomotion_lab.envs.g1_velocity_tracking_env import (
    G1VelocityTrackingConfig,
    G1VelocityTrackingVectorizedEnv,
)
from h200_locomotion_lab.envs.vectorized_genesis_backend import (
    VectorizedGenesisBackend,
    VectorizedGenesisConfig,
    tensor_shape,
)
from h200_locomotion_lab.robots import (
    G1_27DOF_NOHAND_ACTUATOR_ORDER,
    load_g1_27dof_nohand_profile,
)


def test_g1_velocity_tracking_env_does_not_import_genesis_at_module_import() -> None:
    assert "genesis" not in sys.modules


def test_g1_velocity_tracking_env_reset_returns_training_contract_shapes() -> None:
    env = _make_env(n_envs=4)

    observation = env.reset()

    assert tensor_shape(observation) == (4, 90)
    assert tensor_shape(env.commands) == (4, 3)
    assert env.commands == [[0.4, 0.0, 0.0]] * 4
    assert env.episode_lengths == [0, 0, 0, 0]


def test_g1_velocity_tracking_env_step_returns_reward_done_and_components() -> None:
    env = _make_env(n_envs=3)
    env.reset()

    step = env.step([[0.0] * env.action_dim for _ in range(env.n_envs)])

    assert tensor_shape(step.observation) == (3, 90)
    assert tensor_shape(step.reward) == (3,)
    assert tensor_shape(step.terminated) == (3,)
    assert tensor_shape(step.truncated) == (3,)
    assert tensor_shape(step.done) == (3,)
    assert all(value > 0.0 for value in step.reward)
    assert step.done == [False, False, False]
    assert set(step.info["components"]) == {
        "tracking_lin_vel",
        "tracking_yaw_rate",
        "tracking_base_height",
        "root_height",
        "upright",
        "action_rate_penalty",
        "joint_velocity_penalty",
        "joint_deviation_penalty",
        "termination_penalty",
        "height_bad",
        "termination_height_bad",
        "tilt_bad",
        "timeout",
    }


def test_g1_velocity_tracking_env_joint_velocity_penalty_is_opt_in() -> None:
    unscaled_env = _make_env(n_envs=1)
    scaled_env = _make_env(
        n_envs=1,
        config=G1VelocityTrackingConfig(joint_velocity_penalty_scale=0.5),
    )
    unscaled_env.reset()
    scaled_env.reset()
    for env in (unscaled_env, scaled_env):
        for index in env.backend.motor_dof_indices:
            env.backend.robot.velocities[0][index] = 2.0

    unscaled = unscaled_env.step([[0.0] * unscaled_env.action_dim])
    scaled = scaled_env.step([[0.0] * scaled_env.action_dim])

    assert unscaled.info["components"]["joint_velocity_penalty"] == pytest.approx([4.0])
    assert scaled.info["components"]["joint_velocity_penalty"] == pytest.approx([4.0])
    assert scaled.reward[0] == pytest.approx(unscaled.reward[0] - 2.0)


def test_g1_velocity_tracking_env_timeout_resets_only_done_env() -> None:
    env = _make_env(n_envs=3, max_episode_steps=2)
    env.reset()
    env.episode_lengths = [1, 0, 0]

    step = env.step([[1.0] * env.action_dim for _ in range(env.n_envs)])

    assert step.truncated == [True, False, False]
    assert step.done == [True, False, False]
    assert step.info["episode_lengths"] == [2, 1, 1]
    assert step.info["completed_episode_lengths"] == [2]
    assert not step.info["full_env_reset_wave"]
    assert env.episode_lengths == [0, 1, 1]
    assert env.backend.previous_action[0] == [0.0] * env.action_dim
    assert env.backend.previous_action[1] == [1.0] * env.action_dim
    assert env.last_action[0] == [0.0] * env.action_dim
    assert env.last_action[1] == [1.0] * env.action_dim
    assert env.backend.robot.qpos[0] == pytest.approx(env.backend.config.root_qpos)


def test_g1_velocity_tracking_env_height_done_resets_fallen_env() -> None:
    env = _make_env(n_envs=3)
    env.reset()
    env.backend.robot.qpos[1][2] = 0.4
    env.backend.robot.qpos[2][2] = 0.1

    step = env.step([[0.0] * env.action_dim for _ in range(env.n_envs)])

    assert step.info["components"]["height_bad"] == [False, True, True]
    assert step.info["components"]["termination_height_bad"] == [False, False, True]
    assert step.terminated == [False, False, True]
    assert step.done == [False, False, True]
    assert step.info["episode_lengths"] == [1, 1, 1]
    assert step.info["completed_episode_lengths"] == [1]
    assert not step.info["full_env_reset_wave"]
    assert env.episode_lengths == [1, 1, 0]
    assert env.backend.robot.qpos[2] == pytest.approx(env.backend.config.root_qpos)


def test_g1_velocity_tracking_env_reports_full_reset_wave() -> None:
    env = _make_env(n_envs=3, max_episode_steps=1)
    env.reset()

    step = env.step([[0.0] * env.action_dim for _ in range(env.n_envs)])

    assert step.done == [True, True, True]
    assert step.info["reset_count"] == 3
    assert step.info["episode_lengths"] == [1, 1, 1]
    assert step.info["completed_episode_lengths"] == [1, 1, 1]
    assert step.info["full_env_reset_wave"]
    assert env.episode_lengths == [0, 0, 0]


def test_g1_velocity_tracking_env_backend_state_and_step_physics_helpers() -> None:
    env = _make_env(n_envs=2)
    backend = env.backend
    backend.reset()

    state = backend.state()
    assert tensor_shape(state.root_ang_vel) == (2, 3)
    assert tensor_shape(state.dof_pos) == (2, 27)

    clipped = backend.step_physics([[2.0] * backend.action_dim for _ in range(backend.n_envs)])

    assert backend.scene.step_count == backend.decimation
    assert clipped == [[1.0] * backend.action_dim for _ in range(backend.n_envs)]
    assert backend.previous_action == clipped


def test_g1_velocity_tracking_env_device_report_is_non_cuda_for_fake_backend() -> None:
    env = _make_env(n_envs=2)
    env.reset()
    step = env.step([[0.0] * env.action_dim for _ in range(env.n_envs)])

    assert env.tensor_device_ok(step)
    assert env.tensor_device_report(step) == {
        "command_device": "not_tensor",
        "episode_length_device": "not_tensor",
        "last_action_device": "not_tensor",
        "qpos_device": "not_tensor",
        "root_pos_device": "not_tensor",
        "root_quat_device": "not_tensor",
        "root_vel_device": "not_tensor",
        "root_ang_vel_device": "not_tensor",
        "dofs_pos_device": "not_tensor",
        "dofs_vel_device": "not_tensor",
        "observation_device": "not_tensor",
        "reward_device": "not_tensor",
        "terminated_device": "not_tensor",
        "truncated_device": "not_tensor",
        "done_device": "not_tensor",
    }


def _make_env(
    n_envs: int,
    max_episode_steps: int = 1000,
    config: G1VelocityTrackingConfig | None = None,
) -> G1VelocityTrackingVectorizedEnv:
    backend = VectorizedGenesisBackend(
        VectorizedGenesisConfig(
            n_envs=n_envs,
            backend="cpu",
            logical_cuda_device="cpu",
            require_asset_path=False,
        ),
        genesis_module=_FakeGenesisModule(),
        profile=load_g1_27dof_nohand_profile(),
    )
    return G1VelocityTrackingVectorizedEnv(
        backend,
        config or G1VelocityTrackingConfig(max_episode_steps=max_episode_steps),
    )


class _FakeGenesisModule:
    cpu = "fake-cpu"

    class options:
        class SimOptions:
            def __init__(self, dt: float) -> None:
                self.dt = dt

    class morphs:
        class Plane:
            pass

        class MJCF:
            def __init__(self, file: str) -> None:
                self.file = file

    def __init__(self) -> None:
        self.init_calls: list[tuple[str, str]] = []

    def init(self, backend: str, logging_level: str) -> None:
        self.init_calls.append((backend, logging_level))

    def Scene(self, show_viewer: bool, sim_options: object) -> "_FakeScene":
        return _FakeScene(show_viewer=show_viewer, sim_options=sim_options)


class _FakeScene:
    def __init__(self, show_viewer: bool, sim_options: object) -> None:
        self.show_viewer = show_viewer
        self.sim_options = sim_options
        self.step_count = 0
        self.n_envs = 0
        self.robot = _FakeRobot()

    def add_entity(self, morph: object) -> object:
        if isinstance(morph, _FakeGenesisModule.morphs.MJCF):
            self.robot.morph_file = morph.file
            return self.robot
        return object()

    def build(self, n_envs: int) -> None:
        self.n_envs = n_envs
        self.robot.configure_envs(n_envs)

    def step(self) -> None:
        self.step_count += 1


class _FakeJoint:
    def __init__(self, index: int) -> None:
        self.dofs_idx_local = [index]


class _FakeRobot:
    def __init__(self) -> None:
        self.n_envs = 0
        self.morph_file = ""
        self.joint_indices = {
            joint_name: index
            for index, joint_name in enumerate(G1_27DOF_NOHAND_ACTUATOR_ORDER, start=6)
        }
        self.qpos: list[list[float]] = []
        self.positions: list[list[float]] = []
        self.velocities: list[list[float]] = []
        self.root_velocities: list[list[float]] = []
        self.root_ang_velocities: list[list[float]] = []

    def configure_envs(self, n_envs: int) -> None:
        self.n_envs = n_envs
        self.qpos = [[0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0] for _ in range(n_envs)]
        self.positions = [[0.0] * 33 for _ in range(n_envs)]
        self.velocities = [[0.0] * 33 for _ in range(n_envs)]
        self.root_velocities = [[0.0, 0.0, 0.0] for _ in range(n_envs)]
        self.root_ang_velocities = [[0.0, 0.0, 0.0] for _ in range(n_envs)]

    def get_joint(self, name: str) -> _FakeJoint:
        return _FakeJoint(self.joint_indices[name])

    def control_dofs_position(
        self,
        position: list[list[float]],
        dofs_idx_local: tuple[int, ...],
    ) -> None:
        self._set_rows(self.positions, position, dofs_idx_local, None)

    def set_qpos(
        self,
        qpos: list[list[float]],
        qs_idx_local: tuple[int, ...],
        envs_idx: tuple[int, ...] | None = None,
        zero_velocity: bool = False,
    ) -> None:
        rows = range(self.n_envs) if envs_idx is None else envs_idx
        for source_index, env_index in enumerate(rows):
            for offset, q_index in enumerate(qs_idx_local):
                self.qpos[env_index][q_index] = qpos[source_index][offset]

    def set_dofs_position(
        self,
        position: list[list[float]],
        dofs_idx_local: tuple[int, ...],
        envs_idx: tuple[int, ...] | None = None,
        zero_velocity: bool = False,
    ) -> None:
        self._set_rows(self.positions, position, dofs_idx_local, envs_idx)

    def set_dofs_velocity(
        self,
        velocity: list[list[float]],
        dofs_idx_local: tuple[int, ...],
        envs_idx: tuple[int, ...] | None = None,
    ) -> None:
        self._set_rows(self.velocities, velocity, dofs_idx_local, envs_idx)

    def get_qpos(self) -> list[list[float]]:
        return self.qpos

    def get_pos(self) -> list[list[float]]:
        return [row[:3] for row in self.qpos]

    def get_quat(self) -> list[list[float]]:
        return [row[3:7] for row in self.qpos]

    def get_vel(self) -> list[list[float]]:
        return self.root_velocities

    def get_ang(self) -> list[list[float]]:
        return self.root_ang_velocities

    def get_dofs_position(self, dofs_idx_local: tuple[int, ...]) -> list[list[float]]:
        return [[row[index] for index in dofs_idx_local] for row in self.positions]

    def get_dofs_velocity(self, dofs_idx_local: tuple[int, ...]) -> list[list[float]]:
        return [[row[index] for index in dofs_idx_local] for row in self.velocities]

    def _set_rows(
        self,
        target: list[list[float]],
        values: list[list[float]],
        indices: tuple[int, ...],
        envs_idx: tuple[int, ...] | None,
    ) -> None:
        rows = range(self.n_envs) if envs_idx is None else envs_idx
        for source_index, env_index in enumerate(rows):
            for offset, dof_index in enumerate(indices):
                target[env_index][dof_index] = values[source_index][offset]
