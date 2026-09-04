"""Genesis G1 environment boundary.

This module intentionally does not import ``genesis`` at module import time.
Local tests exercise the reset/step boundary with a contract-only backend while
the real H200 path can provide a Genesis-backed implementation later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.runtime import ScalarActionBridge
from h200_locomotion_lab.sonic.g1_observation import (
    SonicG1HistoryBuffer,
    SonicG1HistoryFrame,
    build_sonic_g1_decoder_observation,
    mujoco_motor_state_to_sonic_body_state,
)
from h200_locomotion_lab.sonic.g1_policy_bridge import (
    get_default_sonic_g1_action_bridge,
)

G1_29DOF_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class ObservationField:
    """One contiguous observation segment in policy order."""

    name: str
    size: int
    units: str


@dataclass(frozen=True)
class GenesisG1Contract:
    """Simulator-independent G1 observation/action/control contract."""

    joint_order: tuple[str, ...] = G1_29DOF_JOINT_ORDER
    observation_fields: tuple[ObservationField, ...] = (
        ObservationField("base_angular_velocity", 3, "rad_per_s"),
        ObservationField("projected_gravity", 3, "unit_vector"),
        ObservationField("command_velocity", 3, "m_per_s_and_rad_per_s"),
        ObservationField("joint_position_error", 29, "rad"),
        ObservationField("joint_velocity", 29, "rad_per_s"),
        ObservationField("previous_action", 29, "normalized"),
    )
    sim_dt_s: float = 0.005
    decimation: int = 4
    policy_rate_hz: int = 50
    action_scale_rad: float = 0.25

    @property
    def action_dim(self) -> int:
        return len(self.joint_order)

    @property
    def observation_dim(self) -> int:
        return sum(field.size for field in self.observation_fields)

    def validate(self) -> None:
        if self.action_dim != 29:
            raise ValueError(f"Expected 29 G1 joints, got {self.action_dim}")
        if self.observation_dim != 96:
            raise ValueError(f"Expected 96D observation, got {self.observation_dim}")
        if self.decimation <= 0:
            raise ValueError("decimation must be positive")
        derived_rate = round(1.0 / (self.sim_dt_s * self.decimation))
        if derived_rate != self.policy_rate_hz:
            raise ValueError(
                f"policy_rate_hz={self.policy_rate_hz} does not match "
                f"sim_dt_s={self.sim_dt_s} and decimation={self.decimation}"
            )


@dataclass(frozen=True)
class StepResult:
    """Minimal Gymnasium-style step result without a hard Gym dependency."""

    observation: tuple[float, ...]
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)


class GenesisBackend(Protocol):
    """Backend boundary implemented by local stubs or a real Genesis wrapper."""

    def reset(self, seed: int | None = None) -> tuple[float, ...]:
        """Reset simulation and return the first observation."""

    def step(self, action: Sequence[float]) -> StepResult:
        """Apply one policy action and return the next transition."""


@dataclass(frozen=True)
class GenesisCameraConfig:
    """Camera options that must be attached before Genesis scene build."""

    res: tuple[int, int] = (420, 320)
    pos: tuple[float, float, float] = (3.4, -4.2, 2.2)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.85)
    fov: float = 42.0
    gui: bool = False


@dataclass(frozen=True)
class GenesisSceneConfig:
    """Runtime options for the single-env Genesis G1 smoke backend."""

    asset_path: str
    backend: str = "cuda"
    show_viewer: bool = False
    n_envs: int = 1
    add_plane: bool = True
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    root_qpos: tuple[float, float, float, float, float, float, float] | None = None
    default_motor_positions: tuple[float, ...] | None = None
    initial_motor_positions: tuple[float, ...] | None = None
    action_mode: str = "normalized_delta"
    convexify: bool = False
    decimate: bool = False
    logging_level: str = "warning"
    camera: GenesisCameraConfig | None = None


class ContractOnlyBackend:
    """Deterministic backend for local boundary tests.

    It validates shape and timing semantics but does not claim physics fidelity.
    """

    def __init__(self, contract: GenesisG1Contract | None = None) -> None:
        self.contract = contract or GenesisG1Contract()
        self.contract.validate()
        self.step_count = 0

    def reset(self, seed: int | None = None) -> tuple[float, ...]:
        self.step_count = 0
        return (0.0,) * self.contract.observation_dim

    def step(self, action: Sequence[float]) -> StepResult:
        if len(action) != self.contract.action_dim:
            raise ValueError(f"Expected action_dim={self.contract.action_dim}, got {len(action)}")
        self.step_count += 1
        return StepResult(
            observation=(0.0,) * self.contract.observation_dim,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"step_count": self.step_count, "backend": "contract_only"},
        )


class GenesisG1SceneBackend:
    """Single-environment Genesis backend for the validated 29-motor G1 asset."""

    def __init__(
        self,
        config: GenesisSceneConfig,
        contract: GenesisG1Contract | None = None,
        genesis_module: Any | None = None,
    ) -> None:
        self.config = config
        self.contract = contract or GenesisG1Contract()
        self.contract.validate()
        if self.config.n_envs != 1:
            raise NotImplementedError("GenesisG1SceneBackend currently supports n_envs=1 only")
        if not Path(self.config.asset_path).is_file():
            raise FileNotFoundError(f"Genesis G1 asset not found: {self.config.asset_path}")
        if self.config.action_mode not in {"normalized_delta", "sonic_policy_raw"}:
            raise ValueError(f"Unknown Genesis action_mode: {self.config.action_mode}")
        self.sonic_action_bridge: ScalarActionBridge | None = (
            get_default_sonic_g1_action_bridge()
            if self.config.action_mode == "sonic_policy_raw"
            else None
        )

        self.gs = genesis_module or import_genesis_module()
        self._init_genesis()
        self.camera = None
        self.scene = self._build_scene()
        self._reset_root_qpos()
        self.motor_dof_indices = self._resolve_motor_dof_indices()
        self.default_motor_positions = self._resolve_default_motor_positions()
        self.initial_motor_positions = self._resolve_initial_motor_positions()
        self.previous_action = (0.0,) * self.contract.action_dim
        self.sonic_history = SonicG1HistoryBuffer()
        self.step_count = 0

    def reset(self, seed: int | None = None) -> tuple[float, ...]:
        self._reset_root_qpos()
        self.robot.set_dofs_position(
            self.initial_motor_positions,
            dofs_idx_local=self.motor_dof_indices,
            zero_velocity=True,
        )
        self.robot.set_dofs_velocity(None)
        self.previous_action = (0.0,) * self.contract.action_dim
        self.step_count = 0
        self.sonic_history = SonicG1HistoryBuffer()
        self.record_sonic_history_frame()
        return self._observation()

    def step(self, action: Sequence[float]) -> StepResult:
        if len(action) != self.contract.action_dim:
            raise ValueError(f"Expected action_dim={self.contract.action_dim}, got {len(action)}")
        action_values = tuple(float(value) for value in action)
        target = self._motor_targets_from_action(action_values)
        self.robot.control_dofs_position(target, dofs_idx_local=self.motor_dof_indices)
        for _ in range(self.contract.decimation):
            self.scene.step()
        self.previous_action = self._observation_action(action_values)
        self.step_count += 1
        self.record_sonic_history_frame()
        return StepResult(
            observation=self._observation(),
            reward=0.0,
            terminated=False,
            truncated=False,
            info={
                "backend": "genesis",
                "step_count": self.step_count,
                "asset_path": self.config.asset_path,
                "robot_n_dofs": int(getattr(self.robot, "n_dofs", -1)),
                "motor_dof_count": len(self.motor_dof_indices),
                "action_mode": self.config.action_mode,
            },
        )

    def record_sonic_history_frame(self) -> SonicG1HistoryFrame:
        """Append current Genesis state in official SONIC decoder history format."""

        frame = self._sonic_history_frame()
        self.sonic_history.append(frame)
        return frame

    def sonic_decoder_observation(self, token_state: Sequence[float]) -> tuple[float, ...]:
        """Build the official 994D decoder observation from recorded Genesis history."""

        return build_sonic_g1_decoder_observation(
            token_state,
            self.sonic_history.latest_oldest_first(),
        )

    def _init_genesis(self) -> None:
        backend = getattr(self.gs, self.config.backend, self.config.backend)
        self.gs.init(backend=backend, logging_level=self.config.logging_level)

    def _build_scene(self) -> Any:
        scene = self.gs.Scene(
            show_viewer=self.config.show_viewer,
            sim_options=self.gs.options.SimOptions(dt=self.contract.sim_dt_s),
        )
        if self.config.add_plane:
            scene.add_entity(self.gs.morphs.Plane())
        self.robot = scene.add_entity(
            self.gs.morphs.MJCF(
                file=self.config.asset_path,
                pos=self.config.base_pos,
                quat=self.config.base_quat,
                convexify=self.config.convexify,
                decimate=self.config.decimate,
            )
        )
        if self.config.camera is not None:
            self.camera = scene.add_camera(
                res=self.config.camera.res,
                pos=self.config.camera.pos,
                lookat=self.config.camera.lookat,
                fov=self.config.camera.fov,
                GUI=self.config.camera.gui,
            )
        scene.build(n_envs=self.config.n_envs)
        return scene

    def _reset_root_qpos(self) -> None:
        if self.config.root_qpos is None:
            return
        root_qpos = tuple(float(value) for value in self.config.root_qpos)
        if len(root_qpos) != 7:
            raise ValueError(f"Expected root_qpos length 7, got {len(root_qpos)}")
        if hasattr(self.robot, "set_qpos"):
            self.robot.set_qpos(
                root_qpos,
                qs_idx_local=tuple(range(7)),
                zero_velocity=True,
            )
            return
        self.robot.set_pos(root_qpos[:3], zero_velocity=True)
        self.robot.set_quat(root_qpos[3:], zero_velocity=True)

    def _resolve_motor_dof_indices(self) -> tuple[int, ...]:
        indices: list[int] = []
        for joint_name in self.contract.joint_order:
            joint = self.robot.get_joint(joint_name)
            joint_indices = joint.dofs_idx_local
            if len(joint_indices) != 1:
                raise ValueError(f"Expected single-DoF joint {joint_name}, got {joint_indices}")
            indices.append(int(joint_indices[0]))
        if len(indices) != self.contract.action_dim:
            raise ValueError(
                f"Expected {self.contract.action_dim} motor DOFs, got {len(indices)}"
            )
        if len(set(indices)) != len(indices):
            raise ValueError(f"Duplicate Genesis motor DOF indices: {indices}")
        return tuple(indices)

    def _read_motor_positions(self) -> tuple[float, ...]:
        return _flatten_numeric(self.robot.get_dofs_position(dofs_idx_local=self.motor_dof_indices))

    def _resolve_default_motor_positions(self) -> tuple[float, ...]:
        if self.config.action_mode == "sonic_policy_raw":
            if self.config.default_motor_positions is not None:
                raise ValueError(
                    "default_motor_positions must not be provided in sonic_policy_raw mode; "
                    "SONIC uses profile control default_angles"
                )
            return self._require_sonic_action_bridge().default_angles_command
        if self.config.default_motor_positions is None:
            return self._read_motor_positions()
        if len(self.config.default_motor_positions) != self.contract.action_dim:
            raise ValueError(
                f"Expected {self.contract.action_dim} default motor positions, "
                f"got {len(self.config.default_motor_positions)}"
            )
        return tuple(float(value) for value in self.config.default_motor_positions)

    def _resolve_initial_motor_positions(self) -> tuple[float, ...]:
        if self.config.initial_motor_positions is None:
            return self.default_motor_positions
        if len(self.config.initial_motor_positions) != self.contract.action_dim:
            raise ValueError(
                f"Expected {self.contract.action_dim} initial motor positions, "
                f"got {len(self.config.initial_motor_positions)}"
            )
        return tuple(float(value) for value in self.config.initial_motor_positions)

    def _read_motor_velocities(self) -> tuple[float, ...]:
        return _flatten_numeric(self.robot.get_dofs_velocity(dofs_idx_local=self.motor_dof_indices))

    def _read_root_qpos(self) -> tuple[float, ...]:
        if hasattr(self.robot, "get_qpos"):
            return _flatten_numeric(self.robot.get_qpos())
        if hasattr(self.robot, "qpos"):
            return _flatten_numeric(self.robot.qpos)
        return self.config.base_pos + self.config.base_quat

    def _read_base_quat(self) -> tuple[float, float, float, float]:
        qpos = self._read_root_qpos()
        if len(qpos) >= 7:
            return tuple(qpos[3:7])  # type: ignore[return-value]
        return self.config.base_quat

    def _read_base_angular_velocity(self) -> tuple[float, float, float]:
        try:
            root_velocity = _flatten_numeric(
                self.robot.get_dofs_velocity(dofs_idx_local=tuple(range(6)))
            )
        except RECOVERABLE_RUNTIME_ERRORS:
            return (0.0, 0.0, 0.0)
        if len(root_velocity) < 6:
            return (0.0, 0.0, 0.0)
        return tuple(root_velocity[3:6])  # type: ignore[return-value]

    def _sonic_history_frame(self) -> SonicG1HistoryFrame:
        body_q, body_dq = mujoco_motor_state_to_sonic_body_state(
            self._read_motor_positions(),
            self._read_motor_velocities(),
        )
        return SonicG1HistoryFrame(
            base_ang_vel=self._read_base_angular_velocity(),
            body_q=body_q,
            body_dq=body_dq,
            last_action=self.previous_action,
            base_quat=self._read_base_quat(),
        )

    def _motor_targets_from_action(self, action: Sequence[float]) -> tuple[float, ...]:
        if self.config.action_mode == "sonic_policy_raw":
            return self._require_sonic_action_bridge().policy_action_to_command_targets(action)
        clipped_action = tuple(max(-1.0, min(1.0, float(value))) for value in action)
        return tuple(
            default + self.contract.action_scale_rad * delta
            for default, delta in zip(self.default_motor_positions, clipped_action)
        )

    def _require_sonic_action_bridge(self) -> ScalarActionBridge:
        if self.sonic_action_bridge is None:
            raise RuntimeError("SONIC action bridge is only available in sonic_policy_raw mode")
        return self.sonic_action_bridge

    def _observation_action(self, action: Sequence[float]) -> tuple[float, ...]:
        if self.config.action_mode == "sonic_policy_raw":
            return tuple(float(value) for value in action)
        return tuple(max(-1.0, min(1.0, float(value))) for value in action)

    def _observation(self) -> tuple[float, ...]:
        motor_positions = self._read_motor_positions()
        motor_velocities = self._read_motor_velocities()
        position_error = tuple(
            position - default
            for position, default in zip(motor_positions, self.default_motor_positions)
        )
        observation = (
            (0.0, 0.0, 0.0)
            + (0.0, 0.0, -1.0)
            + (0.0, 0.0, 0.0)
            + position_error
            + motor_velocities
            + self.previous_action
        )
        if len(observation) != self.contract.observation_dim:
            raise ValueError(
                f"Expected observation_dim={self.contract.observation_dim}, got {len(observation)}"
            )
        return observation


class GenesisG1Env:
    """Thin environment shell around a Genesis-compatible backend."""

    simulator = "genesis"
    robot = "unitree_g1"

    def __init__(
        self,
        contract: GenesisG1Contract | None = None,
        backend: GenesisBackend | None = None,
    ) -> None:
        self.contract = contract or GenesisG1Contract()
        self.contract.validate()
        self.backend = backend
        self._last_observation: tuple[float, ...] | None = None

    @classmethod
    def contract_only(cls) -> GenesisG1Env:
        contract = GenesisG1Contract()
        return cls(contract=contract, backend=ContractOnlyBackend(contract))

    @classmethod
    def from_genesis_asset(
        cls,
        asset_path: str,
        *,
        backend: str = "cuda",
        show_viewer: bool = False,
        base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        root_qpos: tuple[float, float, float, float, float, float, float] | None = None,
        default_motor_positions: tuple[float, ...] | None = None,
        initial_motor_positions: tuple[float, ...] | None = None,
        action_mode: str = "normalized_delta",
        convexify: bool = False,
        decimate: bool = False,
        logging_level: str = "warning",
    ) -> GenesisG1Env:
        contract = GenesisG1Contract()
        scene_config = GenesisSceneConfig(
            asset_path=asset_path,
            backend=backend,
            show_viewer=show_viewer,
            base_pos=base_pos,
            base_quat=base_quat,
            root_qpos=root_qpos,
            default_motor_positions=default_motor_positions,
            initial_motor_positions=initial_motor_positions,
            action_mode=action_mode,
            convexify=convexify,
            decimate=decimate,
            logging_level=logging_level,
        )
        return cls(contract=contract, backend=GenesisG1SceneBackend(scene_config, contract))

    def describe(self) -> str:
        return (
            "Genesis G1 boundary: "
            f"{self.contract.observation_dim}D obs, "
            f"{self.contract.action_dim}D action, "
            f"{self.contract.policy_rate_hz}Hz policy."
        )

    def reset(self, seed: int | None = None) -> tuple[float, ...]:
        backend = self._require_backend()
        observation = tuple(float(value) for value in backend.reset(seed=seed))
        self._validate_observation(observation)
        self._last_observation = observation
        return observation

    def step(self, action: Sequence[float]) -> StepResult:
        if len(action) != self.contract.action_dim:
            raise ValueError(f"Expected action_dim={self.contract.action_dim}, got {len(action)}")
        backend = self._require_backend()
        result = backend.step(action)
        observation = tuple(float(value) for value in result.observation)
        self._validate_observation(observation)
        typed_result = StepResult(
            observation=observation,
            reward=float(result.reward),
            terminated=bool(result.terminated),
            truncated=bool(result.truncated),
            info=dict(result.info),
        )
        self._last_observation = observation
        return typed_result

    def _require_backend(self) -> GenesisBackend:
        if self.backend is None:
            raise RuntimeError(
                "GenesisG1Env needs an explicit backend. Use GenesisG1Env.contract_only() "
                "for local shape tests or provide a real Genesis backend on H200."
            )
        return self.backend

    def _validate_observation(self, observation: Sequence[float]) -> None:
        if len(observation) != self.contract.observation_dim:
            raise ValueError(
                f"Expected observation_dim={self.contract.observation_dim}, got {len(observation)}"
            )


def import_genesis_module() -> Any:
    """Import Genesis lazily so local tooling can run without the simulator."""

    try:
        import genesis as gs  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Genesis is not installed. On the H200 target install the optional "
            "genesis dependency, then provide a real backend for GenesisG1Env."
        ) from exc
    return gs


def _flatten_numeric(values: Any) -> tuple[float, ...]:
    """Convert tensor-like Genesis results to a flat tuple of Python floats."""

    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "reshape"):
        values = values.reshape(-1)
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        return (float(values),)
    return tuple(float(value) for value in values)
