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
from h200_locomotion_lab.sonic.g1_policy_bridge import (
    SONIC_G1_ACTION_SCALES,
    SONIC_G1_DEFAULT_ANGLES,
    SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX,
    get_default_sonic_g1_action_bridge,
)
from h200_locomotion_lab.sonic.g1_observation import field_by_name


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


def test_genesis_scene_backend_accepts_default_motor_position_override() -> None:
    asset = Path(__file__)
    backend = GenesisG1SceneBackend(
        GenesisSceneConfig(
            asset_path=str(asset),
            backend="cuda",
            default_motor_positions=(0.2,) * GenesisG1Contract().action_dim,
        ),
        genesis_module=_FakeGenesisModule(),
    )

    backend.reset()
    result = backend.step([0.5] * backend.contract.action_dim)

    assert result.info["motor_dof_count"] == 29
    assert backend.default_motor_positions == (0.2,) * backend.contract.action_dim
    assert backend.robot.last_position_target == (0.325,) * backend.contract.action_dim


def test_genesis_scene_backend_maps_raw_sonic_policy_action() -> None:
    asset = Path(__file__)
    backend = GenesisG1SceneBackend(
        GenesisSceneConfig(asset_path=str(asset), backend="cuda", action_mode="sonic_policy_raw"),
        genesis_module=_FakeGenesisModule(),
    )

    raw_action = tuple(float(index) for index in range(backend.contract.action_dim))
    result = backend.step(raw_action)

    assert result.info["action_mode"] == "sonic_policy_raw"
    assert backend.sonic_action_bridge is get_default_sonic_g1_action_bridge()
    assert backend.default_motor_positions == SONIC_G1_DEFAULT_ANGLES
    assert backend.default_motor_positions == (
        get_default_sonic_g1_action_bridge().default_angles_command
    )
    assert backend.previous_action == raw_action
    assert backend.robot.last_position_target is not None
    assert backend.robot.last_position_target == pytest.approx(
        get_default_sonic_g1_action_bridge().policy_action_to_command_targets(raw_action)
    )
    assert backend.robot.last_position_target[0] == pytest.approx(
        SONIC_G1_DEFAULT_ANGLES[0] + 0.0 * SONIC_G1_ACTION_SCALES[0]
    )
    assert backend.robot.last_position_target[1] == pytest.approx(
        SONIC_G1_DEFAULT_ANGLES[1] + 3.0 * SONIC_G1_ACTION_SCALES[1]
    )


def test_genesis_scene_backend_keeps_sonic_default_targets_separate_from_initial_pose() -> None:
    asset = Path(__file__)
    initial_pose = tuple(0.01 * index for index in range(GenesisG1Contract().action_dim))
    backend = GenesisG1SceneBackend(
        GenesisSceneConfig(
            asset_path=str(asset),
            backend="cuda",
            action_mode="sonic_policy_raw",
            initial_motor_positions=initial_pose,
        ),
        genesis_module=_FakeGenesisModule(),
    )

    backend.reset()
    assert backend.robot.get_dofs_position(backend.motor_dof_indices) == list(initial_pose)
    backend.step([0.0] * backend.contract.action_dim)

    assert backend.default_motor_positions == SONIC_G1_DEFAULT_ANGLES
    assert backend.initial_motor_positions == initial_pose
    assert backend.robot.last_position_target == SONIC_G1_DEFAULT_ANGLES


def test_genesis_scene_backend_rejects_default_override_in_sonic_policy_mode() -> None:
    asset = Path(__file__)

    with pytest.raises(ValueError, match="default_motor_positions must not be provided"):
        GenesisG1SceneBackend(
            GenesisSceneConfig(
                asset_path=str(asset),
                backend="cuda",
                action_mode="sonic_policy_raw",
                default_motor_positions=(0.0,) * GenesisG1Contract().action_dim,
            ),
            genesis_module=_FakeGenesisModule(),
        )


def test_genesis_scene_backend_rejects_wrong_initial_motor_position_shape() -> None:
    asset = Path(__file__)

    with pytest.raises(ValueError, match="Expected 29 initial motor positions"):
        GenesisG1SceneBackend(
            GenesisSceneConfig(
                asset_path=str(asset),
                backend="cuda",
                initial_motor_positions=(0.0,) * 28,
            ),
            genesis_module=_FakeGenesisModule(),
        )


def test_genesis_scene_backend_resets_root_qpos_when_configured() -> None:
    asset = Path(__file__)
    root_qpos = (0.1, 0.2, 0.79, 0.7, 0.0, 0.0, -0.7)
    backend = GenesisG1SceneBackend(
        GenesisSceneConfig(asset_path=str(asset), backend="cuda", root_qpos=root_qpos),
        genesis_module=_FakeGenesisModule(),
    )

    backend.robot.qpos = [9.0] * 36
    backend.reset()

    assert backend.robot.qpos[:7] == list(root_qpos)


def test_genesis_scene_backend_builds_sonic_decoder_observation_from_history() -> None:
    asset = Path(__file__)
    backend = GenesisG1SceneBackend(
        GenesisSceneConfig(asset_path=str(asset), backend="cuda", action_mode="sonic_policy_raw"),
        genesis_module=_FakeGenesisModule(),
    )
    backend.reset()
    backend.robot.qpos[:7] = [0.0, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0]
    backend.robot.velocities[3:6] = [0.1, 0.2, 0.3]
    for mujoco_index, dof_index in enumerate(backend.motor_dof_indices):
        backend.robot.positions[dof_index] = (
            SONIC_G1_DEFAULT_ANGLES[mujoco_index] + 10.0 + mujoco_index
        )
        backend.robot.velocities[dof_index] = 100.0 + mujoco_index
    backend.previous_action = tuple(float(1000 + index) for index in range(29))

    backend.record_sonic_history_frame()
    observation = backend.sonic_decoder_observation((0.0,) * 64)

    assert len(observation) == 994
    assert _field_slice(observation, "his_base_angular_velocity_10frame_step1")[
        -3:
    ] == pytest.approx((0.1, 0.2, 0.3))
    policy_index = 1
    mujoco_index = SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX[policy_index]
    assert _field_slice(observation, "his_body_joint_positions_10frame_step1")[
        9 * 29 + policy_index
    ] == pytest.approx(10.0 + mujoco_index)
    assert _field_slice(observation, "his_body_joint_velocities_10frame_step1")[
        9 * 29 + policy_index
    ] == pytest.approx(100.0 + mujoco_index)
    assert _field_slice(observation, "his_last_actions_10frame_step1")[-29:] == tuple(
        float(1000 + index) for index in range(29)
    )
    assert _field_slice(observation, "his_gravity_dir_10frame_step1")[-3:] == (
        0.0,
        0.0,
        -1.0,
    )


def test_genesis_scene_backend_requires_existing_asset() -> None:
    missing_asset = Path(__file__).with_name("missing_g1_29dof.xml")

    with pytest.raises(FileNotFoundError, match="Genesis G1 asset not found"):
        GenesisG1SceneBackend(
            GenesisSceneConfig(asset_path=str(missing_asset)),
            genesis_module=_FakeGenesisModule(),
        )


def _field_slice(observation: tuple[float, ...], field_name: str) -> tuple[float, ...]:
    field = field_by_name(field_name)
    return observation[field.offset : field.offset + field.dim]


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
                quat: tuple[float, float, float, float],
                convexify: bool,
                decimate: bool,
            ) -> None:
                self.file = file
                self.pos = pos
                self.quat = quat
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
        self.qpos = [0.0] * 36
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

    def set_qpos(
        self,
        qpos: tuple[float, ...],
        qs_idx_local: tuple[int, ...],
        zero_velocity: bool,
    ) -> None:
        for value, index in zip(qpos, qs_idx_local):
            self.qpos[index] = value
        if zero_velocity:
            self.velocities = [0.0] * self.n_dofs

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
