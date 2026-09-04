"""Small, tensor-library-neutral contracts for reinforcement learning.

The boundary follows the MDP rather than any framework:

* a task owns observations, actions, rewards, and termination;
* a policy maps observations and internal state to actions;
* an algorithm updates a compatible policy from transitions;
* an experiment is the only place that composes all three.

These types intentionally contain no Torch, simulator, robot, or PPO concepts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

InteractionMode = Literal["supervised", "on_policy", "off_policy"]


class CompositionError(ValueError):
    """Raised when independently valid components cannot be composed."""


@dataclass(frozen=True, slots=True)
class TensorSpace:
    """Shape and numeric contract for one observation or action tensor."""

    shape: tuple[int, ...]
    dtype: str = "float32"
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        if not self.shape or any(not isinstance(size, int) or size <= 0 for size in self.shape):
            raise ValueError("shape must contain positive integers")
        if not self.dtype:
            raise ValueError("dtype must be non-empty")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("low must be <= high")

    @property
    def flat_dim(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Algorithm-free MDP contract."""

    name: str
    observations: Mapping[str, TensorSpace]
    action: TensorSpace
    max_episode_steps: int
    metrics: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.name, "task.name")
        if "policy" not in self.observations:
            raise ValueError("task observations must define the 'policy' role")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("task metrics must be unique")

    def observation(self, role: str = "policy") -> TensorSpace:
        try:
            return self.observations[role]
        except KeyError as exc:
            raise KeyError(f"task {self.name!r} has no observation role {role!r}") from exc


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """Task-free action-generator contract."""

    name: str
    family: str
    capabilities: frozenset[str]
    action_horizon: int = 1
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.name, "policy.name")
        _require_name(self.family, "policy.family")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if any(not capability for capability in self.capabilities):
            raise ValueError("policy capabilities must be non-empty strings")


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """Task-free policy-update contract."""

    name: str
    family: str
    interaction: InteractionMode
    required_policy_capabilities: frozenset[str]
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.name, "algorithm.name")
        _require_name(self.family, "algorithm.family")
        if self.interaction not in {"supervised", "on_policy", "off_policy"}:
            raise ValueError(f"unsupported interaction mode: {self.interaction}")
        if any(not capability for capability in self.required_policy_capabilities):
            raise ValueError("algorithm requirements must be non-empty strings")


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    """Execution resources, deliberately outside the task definition."""

    backend: str
    device: str
    num_envs: int
    headless: bool = True
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.backend, "runtime.backend")
        _require_name(self.device, "runtime.device")
        if self.num_envs <= 0:
            raise ValueError("runtime.num_envs must be positive")


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """The composition root; no component may refer back to this object."""

    name: str
    task: TaskSpec
    policy: PolicySpec
    algorithm: AlgorithmSpec
    runtime: RuntimeSpec

    def __post_init__(self) -> None:
        _require_name(self.name, "experiment.name")
        validate_composition(self.task, self.policy, self.algorithm)


def validate_composition(
    task: TaskSpec,
    policy: PolicySpec,
    algorithm: AlgorithmSpec,
) -> None:
    """Check only real cross-component requirements; avoid name-based coupling."""

    missing = algorithm.required_policy_capabilities - policy.capabilities
    if missing:
        formatted = ", ".join(sorted(missing))
        raise CompositionError(
            f"algorithm {algorithm.name!r} requires missing policy capabilities: {formatted}"
        )
    if policy.action_horizon > 1 and "action_chunk" not in policy.capabilities:
        raise CompositionError("multi-step policies must declare the action_chunk capability")
    # Force validation of the standard policy-facing task roles at composition time.
    task.observation("policy")


@dataclass(frozen=True, slots=True)
class TaskStep:
    """Result of applying one action batch to a task."""

    observation: Any
    reward: Any
    terminated: Any
    truncated: Any
    metrics: Mapping[str, Any] = field(default_factory=dict)
    final_observation: Any | None = None

    @property
    def done(self) -> Any:
        return logical_or(self.terminated, self.truncated)


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    """Action plus optional policy-owned state and training information."""

    action: Any
    state: Any | None = None
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    """The sole data boundary between interaction and learning."""

    observation: Any
    action: Any
    reward: Any
    next_observation: Any
    terminated: Any
    truncated: Any
    policy_info: Mapping[str, Any] = field(default_factory=dict)
    task_metrics: Mapping[str, Any] = field(default_factory=dict)
    final_observation: Any | None = None

    @property
    def done(self) -> Any:
        return logical_or(self.terminated, self.truncated)


@dataclass(frozen=True, slots=True)
class UpdateReport:
    """Algorithm-owned update result with framework-neutral scalar metrics."""

    updated: bool
    samples: int = 0
    metrics: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class VectorTask(Protocol):
    spec: TaskSpec
    num_envs: int

    def reset(self) -> Any: ...

    def step(self, action: Any) -> TaskStep: ...


@runtime_checkable
class Policy(Protocol):
    spec: PolicySpec

    def act(self, observation: Any, *, deterministic: bool = False) -> PolicyOutput: ...


@runtime_checkable
class LearningAlgorithm(Protocol):
    spec: AlgorithmSpec

    def observe(self, transition: TransitionBatch) -> None: ...

    def update(self, policy: Policy) -> UpdateReport | None: ...


def logical_or(left: Any, right: Any) -> Any:
    """Boolean OR for scalars, tensor-like values, and simple Python sequences."""

    if isinstance(left, bool) and isinstance(right, bool):
        return left or right
    try:
        return left | right
    except TypeError:
        if _is_sequence(left) and _is_sequence(right):
            if len(left) != len(right):
                raise ValueError("termination arrays must have equal length")
            return [bool(a) or bool(b) for a, b in zip(left, right)]
        raise


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _require_name(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
