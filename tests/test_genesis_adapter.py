import sys
from pathlib import Path

import pytest

from h200_locomotion_lab.envs.genesis_adapter import (
    G1_29DOF_JOINT_ORDER,
    GenesisG1Contract,
    GenesisG1Env,
    GenesisG1SceneBackend,
    GenesisSceneConfig,
)


def test_genesis_contract_matches_g1_29dof_inventory() -> None:
    contract = GenesisG1Contract()

    assert len(G1_29DOF_JOINT_ORDER) == 29
    assert G1_29DOF_JOINT_ORDER[:6] == (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    )
    assert G1_29DOF_JOINT_ORDER[-3:] == (
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    assert contract.action_dim == 29
    assert contract.observation_dim == 96
    assert contract.policy_rate_hz == 50
    contract.validate()


def test_genesis_adapter_does_not_import_simulator_package() -> None:
    assert "genesis" not in sys.modules


def test_contract_only_backend_resets_and_steps() -> None:
    env = GenesisG1Env.contract_only()

    observation = env.reset(seed=123)
    result = env.step([0.0] * env.contract.action_dim)

    assert len(observation) == env.contract.observation_dim
    assert len(result.observation) == env.contract.observation_dim
    assert result.reward == 0.0
    assert result.terminated is False
    assert result.truncated is False
    assert result.info["step_count"] == 1
    assert "96D obs" in env.describe()


def test_step_rejects_wrong_action_shape() -> None:
    env = GenesisG1Env.contract_only()
    env.reset()

    with pytest.raises(ValueError, match="Expected action_dim=29"):
        env.step([0.0] * 28)


def test_genesis_scene_backend_maps_29_motor_dofs_and_steps() -> None:
    asset = Path(__file__)
    fake_genesis = _FakeGenesisModule()
    backend = GenesisG1SceneBackend(
        GenesisSceneConfig(asset_path=str(asset), backend="cuda"),
        genesis_module=fake_genesis,
    )

    observation = backend.reset()
    result = backend.step([0.5] * backend.contract.action_dim)

    assert fake_genesis.init_calls == [("fake-cuda", "warning")]
    assert backend.motor_dof_indices == tuple(range(6, 35))
    assert len(observation) == backend.contract.observation_dim
    assert len(result.observation) == backend.contract.observation_dim
    assert result.info["backend"] == "genesis"
    assert result.info["motor_dof_count"] == 29
    assert result.info["robot_n_dofs"] == 35
    assert backend.scene.step_count == backend.contract.decimation
    assert backend.robot.last_position_target == (0.125,) * backend.contract.action_dim


def test_genesis_scene_backend_requires_existing_asset() -> None:
    missing_asset = Path(__file__).with_name("missing_g1_29dof.xml")

    with pytest.raises(FileNotFoundError, match="Genesis G1 asset not found"):
        GenesisG1SceneBackend(
            GenesisSceneConfig(asset_path=str(missing_asset)),
            genesis_module=_FakeGenesisModule(),
        )


class _FakeGenesisModule:
    cuda = "fake-cuda"

    class options:
        class SimOptions:
            def __init__(self, dt: float) -> None:
                self.dt = dt

    class morphs:
        class Plane:
            pass

        class MJCF:
            def __init__(
                self,
                file: str,
                pos: tuple[float, float, float],
                convexify: bool,
                decimate: bool,
            ) -> None:
                self.file = file
                self.pos = pos
                self.convexify = convexify
                self.decimate = decimate

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
        self.robot = _FakeRobot()

    def add_entity(self, morph: object) -> object:
        if isinstance(morph, _FakeGenesisModule.morphs.MJCF):
            return self.robot
        return object()

    def build(self, n_envs: int) -> None:
        self.n_envs = n_envs

    def step(self) -> None:
        self.step_count += 1


class _FakeJoint:
    def __init__(self, index: int) -> None:
        self.dofs_idx_local = [index]


class _FakeRobot:
    n_dofs = 35
    n_links = 31

    def __init__(self) -> None:
        self.positions = [0.0] * self.n_dofs
        self.velocities = [0.0] * self.n_dofs
        self.last_position_target: tuple[float, ...] | None = None
        self.joint_indices = {
            joint_name: index for index, joint_name in enumerate(G1_29DOF_JOINT_ORDER, start=6)
        }

    def get_joint(self, name: str) -> _FakeJoint:
        return _FakeJoint(self.joint_indices[name])

    def get_dofs_position(self, dofs_idx_local: tuple[int, ...]) -> list[float]:
        return [self.positions[index] for index in dofs_idx_local]

    def get_dofs_velocity(self, dofs_idx_local: tuple[int, ...]) -> list[float]:
        return [self.velocities[index] for index in dofs_idx_local]

    def set_dofs_position(
        self,
        position: tuple[float, ...],
        dofs_idx_local: tuple[int, ...],
        zero_velocity: bool,
    ) -> None:
        for value, index in zip(position, dofs_idx_local):
            self.positions[index] = value
            if zero_velocity:
                self.velocities[index] = 0.0

    def set_dofs_velocity(self, velocity: object = None) -> None:
        self.velocities = [0.0] * self.n_dofs

    def control_dofs_position(
        self,
        position: tuple[float, ...],
        dofs_idx_local: tuple[int, ...],
    ) -> None:
        self.last_position_target = position
        for value, index in zip(position, dofs_idx_local):
            self.positions[index] = value
