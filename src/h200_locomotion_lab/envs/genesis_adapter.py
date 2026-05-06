"""Genesis G1 environment boundary.

This module intentionally does not import ``genesis`` at module import time.
Local tests exercise the reset/step boundary with a contract-only backend while
the real H200 path can provide a Genesis-backed implementation later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


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
    def contract_only(cls) -> "GenesisG1Env":
        contract = GenesisG1Contract()
        return cls(contract=contract, backend=ContractOnlyBackend(contract))

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
