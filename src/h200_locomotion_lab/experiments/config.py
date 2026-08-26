"""Strict loader for separately owned task, policy, and algorithm YAML files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from h200_locomotion_lab.core.rl import (
    AlgorithmSpec,
    ExperimentSpec,
    InteractionMode,
    PolicySpec,
    RuntimeSpec,
    TaskSpec,
    TensorSpace,
)

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"
DEFAULT_G1_FLAT_TASK = CONFIG_ROOT / "tasks" / "unitree_g1_flat_walk.yaml"
DEFAULT_MLP_GAUSSIAN_POLICY = CONFIG_ROOT / "policies" / "mlp_gaussian_actor_critic.yaml"
DEFAULT_PPO_ALGORITHM = CONFIG_ROOT / "algorithms" / "ppo.yaml"
DEFAULT_G1_FLAT_PPO_EXPERIMENT = CONFIG_ROOT / "experiments" / "unitree_g1_flat_ppo.yaml"


class ComponentConfigError(ValueError):
    """Raised when a component file violates its ownership schema."""


def load_task(
    path: str | Path = DEFAULT_G1_FLAT_TASK,
    *,
    config_root: str | Path = CONFIG_ROOT,
) -> TaskSpec:
    """Load one task contract without importing a policy or algorithm."""

    root = Path(config_root).expanduser().resolve()
    return _parse_task(_load_component(_component_path(root, path), "task"))


def load_policy(
    path: str | Path = DEFAULT_MLP_GAUSSIAN_POLICY,
    *,
    config_root: str | Path = CONFIG_ROOT,
) -> PolicySpec:
    """Load one policy contract without importing a task or algorithm."""

    root = Path(config_root).expanduser().resolve()
    return _parse_policy(_load_component(_component_path(root, path), "policy"))


def load_algorithm(
    path: str | Path = DEFAULT_PPO_ALGORITHM,
    *,
    config_root: str | Path = CONFIG_ROOT,
) -> AlgorithmSpec:
    """Load one algorithm contract without importing a task or policy."""

    root = Path(config_root).expanduser().resolve()
    return _parse_algorithm(_load_component(_component_path(root, path), "algorithm"))


def load_experiment(
    path: str | Path = DEFAULT_G1_FLAT_PPO_EXPERIMENT,
    *,
    config_root: str | Path = CONFIG_ROOT,
) -> ExperimentSpec:
    """Load and validate one composition without a global registry."""

    root = Path(config_root).expanduser().resolve()
    experiment_path = Path(path).expanduser()
    if not experiment_path.is_absolute():
        experiment_path = (root / experiment_path).resolve()
    else:
        experiment_path = experiment_path.resolve()

    data = _load_component(experiment_path, "experiment")
    _strict_keys(
        data,
        required={"name", "components", "runtime"},
        optional=set(),
        label="experiment",
    )
    components = _mapping(data, "components", "experiment.components")
    _strict_keys(
        components,
        required={"task", "policy", "algorithm"},
        optional=set(),
        label="experiment.components",
    )

    task_path = _component_path(root, _string(components, "task", "experiment.components.task"))
    policy_path = _component_path(
        root,
        _string(components, "policy", "experiment.components.policy"),
    )
    algorithm_path = _component_path(
        root,
        _string(components, "algorithm", "experiment.components.algorithm"),
    )

    return ExperimentSpec(
        name=_string(data, "name", "experiment.name"),
        task=_parse_task(_load_component(task_path, "task")),
        policy=_parse_policy(_load_component(policy_path, "policy")),
        algorithm=_parse_algorithm(_load_component(algorithm_path, "algorithm")),
        runtime=_parse_runtime(_mapping(data, "runtime", "experiment.runtime")),
    )


def _parse_task(data: Mapping[str, Any]) -> TaskSpec:
    _strict_keys(
        data,
        required={"name", "observations", "action", "max_episode_steps"},
        optional={"metrics", "parameters"},
        label="task",
    )
    observations_data = _mapping(data, "observations", "task.observations")
    observations = {
        role: _parse_tensor_space(value, f"task.observations.{role}")
        for role, value in observations_data.items()
        if _validate_role(role, "task observation role")
    }
    return TaskSpec(
        name=_string(data, "name", "task.name"),
        observations=observations,
        action=_parse_tensor_space(_mapping(data, "action", "task.action"), "task.action"),
        max_episode_steps=_integer(data, "max_episode_steps", "task.max_episode_steps"),
        metrics=_strings(data.get("metrics", ()), "task.metrics"),
        parameters=_optional_mapping(data.get("parameters"), "task.parameters"),
    )


def _parse_policy(data: Mapping[str, Any]) -> PolicySpec:
    _strict_keys(
        data,
        required={"name", "family", "capabilities"},
        optional={"action_horizon", "parameters"},
        label="policy",
    )
    return PolicySpec(
        name=_string(data, "name", "policy.name"),
        family=_string(data, "family", "policy.family"),
        capabilities=frozenset(_strings(data["capabilities"], "policy.capabilities")),
        action_horizon=_optional_integer(data.get("action_horizon"), 1, "policy.action_horizon"),
        parameters=_optional_mapping(data.get("parameters"), "policy.parameters"),
    )


def _parse_algorithm(data: Mapping[str, Any]) -> AlgorithmSpec:
    _strict_keys(
        data,
        required={"name", "family", "interaction", "required_policy_capabilities"},
        optional={"parameters"},
        label="algorithm",
    )
    interaction = _string(data, "interaction", "algorithm.interaction")
    return AlgorithmSpec(
        name=_string(data, "name", "algorithm.name"),
        family=_string(data, "family", "algorithm.family"),
        interaction=cast(InteractionMode, interaction),
        required_policy_capabilities=frozenset(
            _strings(
                data["required_policy_capabilities"],
                "algorithm.required_policy_capabilities",
            )
        ),
        parameters=_optional_mapping(data.get("parameters"), "algorithm.parameters"),
    )


def _parse_runtime(data: Mapping[str, Any]) -> RuntimeSpec:
    _strict_keys(
        data,
        required={"backend", "device", "num_envs"},
        optional={"headless", "parameters"},
        label="experiment.runtime",
    )
    headless = data.get("headless", True)
    if not isinstance(headless, bool):
        raise ComponentConfigError("experiment.runtime.headless must be a boolean")
    return RuntimeSpec(
        backend=_string(data, "backend", "experiment.runtime.backend"),
        device=_string(data, "device", "experiment.runtime.device"),
        num_envs=_integer(data, "num_envs", "experiment.runtime.num_envs"),
        headless=headless,
        parameters=_optional_mapping(data.get("parameters"), "experiment.runtime.parameters"),
    )


def _parse_tensor_space(data: Mapping[str, Any], label: str) -> TensorSpace:
    _strict_keys(
        data,
        required={"shape"},
        optional={"dtype", "low", "high"},
        label=label,
    )
    shape_values = data["shape"]
    if isinstance(shape_values, str) or not isinstance(shape_values, Sequence):
        raise ComponentConfigError(f"{label}.shape must be a sequence of integers")
    shape = tuple(shape_values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in shape):
        raise ComponentConfigError(f"{label}.shape must contain only integers")
    dtype = data.get("dtype", "float32")
    if not isinstance(dtype, str):
        raise ComponentConfigError(f"{label}.dtype must be a string")
    return TensorSpace(
        shape=shape,
        dtype=dtype,
        low=_optional_number(data.get("low"), f"{label}.low"),
        high=_optional_number(data.get("high"), f"{label}.high"),
    )


def _load_component(path: Path, kind: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except OSError as exc:
        raise ComponentConfigError(f"cannot read {kind} config {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ComponentConfigError(f"{kind} config must contain a mapping")
    _strict_keys(document, required={kind}, optional=set(), label=f"{kind} document")
    return _mapping(document, kind, kind)


def _component_path(root: Path, relative: str | Path) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ComponentConfigError(f"component path escapes config root: {relative}")
    return candidate


def _strict_keys(
    data: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    label: str,
) -> None:
    keys = set(data)
    missing = required - keys
    if missing:
        raise ComponentConfigError(f"{label} missing keys: {', '.join(sorted(missing))}")
    unknown = keys - required - optional
    if unknown:
        raise ComponentConfigError(f"{label} has unknown keys: {', '.join(sorted(unknown))}")


def _mapping(data: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ComponentConfigError(f"{label} must be a mapping")
    return value


def _optional_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ComponentConfigError(f"{label} must be a mapping")
    return value


def _string(data: Mapping[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ComponentConfigError(f"{label} must be a non-empty string")
    return value


def _integer(data: Mapping[str, Any], key: str, label: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComponentConfigError(f"{label} must be an integer")
    return value


def _optional_integer(value: Any, default: int, label: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComponentConfigError(f"{label} must be an integer")
    return value


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ComponentConfigError(f"{label} must be a sequence of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ComponentConfigError(f"{label} must contain only non-empty strings")
    return result


def _optional_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ComponentConfigError(f"{label} must be numeric")
    return float(value)


def _validate_role(value: Any, label: str) -> bool:
    if not isinstance(value, str) or not value:
        raise ComponentConfigError(f"{label} must be a non-empty string")
    return True
