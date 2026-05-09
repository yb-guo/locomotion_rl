import sys

import pytest

from h200_locomotion_lab.envs.vectorized_genesis_backend import (
    VectorizedGenesisBackend,
    VectorizedGenesisConfig,
    tensor_shape,
)
from h200_locomotion_lab.robots import (
    G1_27DOF_NOHAND_ACTUATOR_ORDER,
    load_g1_27dof_nohand_profile,
)


def test_vectorized_genesis_backend_does_not_import_genesis_at_module_import() -> None:
    assert "genesis" not in sys.modules


def test_vectorized_genesis_backend_builds_and_resets_27dof_profile() -> None:
    backend = _make_backend(n_envs=4)

    observation = backend.reset()

    assert backend.action_dim == 27
    assert backend.observation_dim == 90
    assert backend.motor_dof_indices == tuple(range(6, 33))
    assert tensor_shape(observation) == (4, 90)
    assert backend.scene.n_envs == 4
    assert backend.gs.init_calls == [("fake-cpu", "warning")]
    assert backend.robot.morph_file.endswith("g1_27dof_nohand.xml")


def test_vectorized_genesis_backend_steps_action_batch_and_reports_info() -> None:
    backend = _make_backend(n_envs=3)
    backend.reset()

    action = [[0.5] * backend.action_dim for _ in range(backend.n_envs)]
    result = backend.step(action)

    assert tensor_shape(result.observation) == (3, 90)
    assert result.info["backend"] == "vectorized_genesis"
    assert result.info["n_envs"] == 3
    assert result.info["action_dim"] == 27
    assert result.info["observation_dim"] == 90
    assert backend.scene.step_count == backend.decimation
    expected_first_target = (
        backend.profile.control.default_angles_rad[0]
        + 0.5 * backend.profile.control.action_scales_rad[0]
    )
    assert backend.robot.positions[0][6] == pytest.approx(expected_first_target)


def test_vectorized_genesis_backend_rejects_wrong_action_shape() -> None:
    backend = _make_backend(n_envs=2)

    with pytest.raises(ValueError, match="action expected shape=\\(2, 27\\)"):
        backend.step([[0.0] * 26 for _ in range(2)])


def test_vectorized_genesis_backend_selected_reset_changes_only_target_env() -> None:
    backend = _make_backend(n_envs=3)
    backend.reset()
    backend.step([[1.0] * backend.action_dim for _ in range(backend.n_envs)])
    before_positions = [row[:] for row in backend.robot.positions]
    for env_index, row in enumerate(backend.robot.qpos):
        row[0] = float(env_index + 10)
    before_qpos = [row[:] for row in backend.robot.qpos]

    observation = backend.reset(env_ids=[1])

    assert tensor_shape(observation) == (3, 90)
    assert backend.robot.positions[0] == before_positions[0]
    assert backend.robot.positions[2] == before_positions[2]
    assert backend.robot.qpos[0] == before_qpos[0]
    assert backend.robot.qpos[2] == before_qpos[2]
    assert backend.robot.positions[1][6:33] == pytest.approx(
        backend.profile.control.default_angles_rad
    )
    assert backend.robot.qpos[1] == pytest.approx(backend.reset_root_qpos)
    assert backend.previous_action[0] == [1.0] * backend.action_dim
    assert backend.previous_action[1] == [0.0] * backend.action_dim
    assert backend.previous_action[2] == [1.0] * backend.action_dim


def test_vectorized_genesis_backend_can_override_reset_pose() -> None:
    backend = _make_backend(n_envs=2)
    custom_defaults = tuple(0.01 * index for index in range(backend.action_dim))
    custom_root = (0.0, 0.0, 1.1, 1.0, 0.0, 0.0, 0.0)

    backend.set_reset_pose(
        root_qpos=custom_root,
        default_positions_rad=custom_defaults,
    )
    backend.reset()
    backend.step([[0.0] * backend.action_dim for _ in range(backend.n_envs)])

    assert backend.robot.qpos[0] == pytest.approx(custom_root)
    assert backend.robot.positions[0][6:33] == pytest.approx(custom_defaults)


def test_vectorized_genesis_backend_validates_env_ids() -> None:
    backend = _make_backend(n_envs=2)

    with pytest.raises(ValueError, match="out-of-range"):
        backend.reset(env_ids=[2])

    with pytest.raises(ValueError, match="duplicates"):
        backend.reset(env_ids=[1, 1])


def test_vectorized_genesis_backend_device_report_is_non_cuda_for_fake_backend() -> None:
    backend = _make_backend(n_envs=2)
    backend.reset()

    assert backend.tensor_device_report() == {
        "action_device": "not_tensor",
        "qpos_device": "not_tensor",
        "dofs_pos_device": "not_tensor",
        "dofs_vel_device": "not_tensor",
        "root_pos_device": "not_tensor",
        "root_quat_device": "not_tensor",
        "root_vel_device": "not_tensor",
    }
    assert backend.tensor_device_ok()


def test_27dof_profile_order_matches_fake_robot_inventory() -> None:
    backend = _make_backend(n_envs=1)

    assert tuple(backend.robot.joint_indices) == G1_27DOF_NOHAND_ACTUATOR_ORDER


def _make_backend(n_envs: int) -> VectorizedGenesisBackend:
    return VectorizedGenesisBackend(
        VectorizedGenesisConfig(
            n_envs=n_envs,
            backend="cpu",
            logical_cuda_device="cpu",
            require_asset_path=False,
        ),
        genesis_module=_FakeGenesisModule(),
        profile=load_g1_27dof_nohand_profile(),
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

    def configure_envs(self, n_envs: int) -> None:
        self.n_envs = n_envs
        self.qpos = [[0.0] * 7 for _ in range(n_envs)]
        self.positions = [[0.0] * 33 for _ in range(n_envs)]
        self.velocities = [[0.0] * 33 for _ in range(n_envs)]

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
        return [[0.0, 0.0, 0.0] for _ in range(self.n_envs)]

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
