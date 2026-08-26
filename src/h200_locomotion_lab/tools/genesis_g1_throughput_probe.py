"""Physics-only Genesis G1 throughput probe.

The probe excludes SONIC, ONNX, planner subprocesses, rendering, and GIF/video
work. It imports Genesis only through ``GenesisG1SceneBackend`` construction.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from h200_locomotion_lab.envs.genesis_adapter import (
    GenesisG1Contract,
    GenesisG1SceneBackend,
    GenesisSceneConfig,
)

ACTION_PATTERNS = ("zero", "random", "sine")


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    asset_path: str
    backend: str = "cuda"
    n_envs: int = 1
    warmup_policy_steps: int = 50
    measure_policy_steps: int = 500
    sim_dt_s: float = 0.005
    decimation: int = 4
    action_pattern: str = "zero"
    amplitude: float = 0.1
    seed: int = 0
    logging_level: str = "warning"


@dataclass(frozen=True, slots=True)
class ThroughputMetrics:
    build_time_s: float
    warmup_time_s: float
    measure_time_s: float
    warmup_policy_steps: int
    measure_policy_steps: int
    decimation: int
    n_envs: int
    backend: str
    action_pattern: str
    capability_flags: Mapping[str, bool]
    capability_failure: str = "none"

    @property
    def policy_steps_per_sec(self) -> float:
        return self.measure_policy_steps / self.measure_time_s

    @property
    def sim_steps_per_sec(self) -> float:
        return self.measure_policy_steps * self.decimation * self.n_envs / self.measure_time_s

    @property
    def env_steps_per_sec(self) -> float:
        return self.measure_policy_steps * self.n_envs / self.measure_time_s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, help="Path to SONIC-compatible G1 MJCF asset.")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--warmup-policy-steps", type=int, default=50)
    parser.add_argument("--measure-policy-steps", type=int, default=500)
    parser.add_argument("--sim-dt", type=float, default=0.005)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--action-pattern", choices=ACTION_PATTERNS, default="zero")
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logging-level", default="warning")
    args = parser.parse_args()

    config = ProbeConfig(
        asset_path=args.asset,
        backend=args.backend,
        n_envs=args.n_envs,
        warmup_policy_steps=args.warmup_policy_steps,
        measure_policy_steps=args.measure_policy_steps,
        sim_dt_s=args.sim_dt,
        decimation=args.decimation,
        action_pattern=args.action_pattern,
        amplitude=args.amplitude,
        seed=args.seed,
        logging_level=args.logging_level,
    )
    try:
        metrics = run_probe(config)
    except CapabilityFailure as exc:
        print_kv(
            {
                "probe": "genesis_g1_physics_throughput",
                "status": "capability_failure",
                "backend": config.backend,
                "n_envs": config.n_envs,
                "action_pattern": config.action_pattern,
                "render_enabled": False,
                "sonic_enabled": False,
                "onnx_enabled": False,
                "planner_enabled": False,
                "gif_enabled": False,
                "capability_failure": str(exc),
                "gpu_backend": is_gpu_backend(config.backend),
                **default_capability_flags(config.n_envs, batched_build_supported=False),
            }
        )
        return

    print_kv(format_metrics(metrics))


def run_probe(config: ProbeConfig) -> ThroughputMetrics:
    validate_probe_config(config)
    contract = GenesisG1Contract(sim_dt_s=config.sim_dt_s, decimation=config.decimation)
    contract.validate()

    start = time.perf_counter()
    try:
        backend = GenesisG1SceneBackend(
            GenesisSceneConfig(
                asset_path=config.asset_path,
                backend=config.backend,
                n_envs=config.n_envs,
                show_viewer=False,
                action_mode="normalized_delta",
                logging_level=config.logging_level,
                camera=None,
            ),
            contract=contract,
        )
    except NotImplementedError as exc:
        raise CapabilityFailure(str(exc)) from exc
    build_time_s = time.perf_counter() - start

    rng = random.Random(config.seed)
    backend.reset(seed=config.seed)

    warmup_start = time.perf_counter()
    for step_index in range(config.warmup_policy_steps):
        step_physics_only(
            backend,
            action_for_step(
                config.action_pattern,
                step_index,
                contract.action_dim,
                amplitude=config.amplitude,
                rng=rng,
            )
        )
    warmup_time_s = time.perf_counter() - warmup_start

    measure_start = time.perf_counter()
    for step_index in range(config.measure_policy_steps):
        step_physics_only(
            backend,
            action_for_step(
                config.action_pattern,
                config.warmup_policy_steps + step_index,
                contract.action_dim,
                amplitude=config.amplitude,
                rng=rng,
            )
        )
    measure_time_s = time.perf_counter() - measure_start

    return calculate_metrics(
        build_time_s=build_time_s,
        warmup_time_s=warmup_time_s,
        measure_time_s=measure_time_s,
        warmup_policy_steps=config.warmup_policy_steps,
        measure_policy_steps=config.measure_policy_steps,
        decimation=config.decimation,
        n_envs=config.n_envs,
        backend=config.backend,
        action_pattern=config.action_pattern,
        capability_flags=default_capability_flags(
            config.n_envs,
            batched_build_supported=config.n_envs > 1,
        ),
    )


def step_physics_only(backend: GenesisG1SceneBackend, action: Sequence[float]) -> None:
    """Apply one normalized action and advance physics without obs/history work."""

    target = backend._motor_targets_from_action(action)
    backend.robot.control_dofs_position(target, dofs_idx_local=backend.motor_dof_indices)
    for _ in range(backend.contract.decimation):
        backend.scene.step()


def validate_probe_config(config: ProbeConfig) -> None:
    if not config.asset_path:
        raise ValueError("asset_path must not be empty")
    if config.n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if config.warmup_policy_steps < 0:
        raise ValueError("warmup_policy_steps must be non-negative")
    if config.measure_policy_steps <= 0:
        raise ValueError("measure_policy_steps must be positive")
    if config.decimation <= 0:
        raise ValueError("decimation must be positive")
    if not math.isfinite(config.sim_dt_s) or config.sim_dt_s <= 0.0:
        raise ValueError("sim_dt_s must be finite and positive")
    if config.action_pattern not in ACTION_PATTERNS:
        raise ValueError(f"action_pattern must be one of {ACTION_PATTERNS}")
    if not math.isfinite(config.amplitude) or config.amplitude < 0.0:
        raise ValueError("amplitude must be finite and non-negative")


def calculate_metrics(
    *,
    build_time_s: float,
    warmup_time_s: float,
    measure_time_s: float,
    warmup_policy_steps: int,
    measure_policy_steps: int,
    decimation: int,
    n_envs: int,
    backend: str,
    action_pattern: str,
    capability_flags: Mapping[str, bool],
    capability_failure: str = "none",
) -> ThroughputMetrics:
    if measure_time_s <= 0.0:
        raise ValueError("measure_time_s must be positive")
    if build_time_s < 0.0:
        raise ValueError("build_time_s must be non-negative")
    if warmup_time_s < 0.0:
        raise ValueError("warmup_time_s must be non-negative")
    if warmup_policy_steps < 0:
        raise ValueError("warmup_policy_steps must be non-negative")
    if measure_policy_steps <= 0:
        raise ValueError("measure_policy_steps must be positive")
    if decimation <= 0:
        raise ValueError("decimation must be positive")
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    return ThroughputMetrics(
        build_time_s=build_time_s,
        warmup_time_s=warmup_time_s,
        measure_time_s=measure_time_s,
        warmup_policy_steps=warmup_policy_steps,
        measure_policy_steps=measure_policy_steps,
        decimation=decimation,
        n_envs=n_envs,
        backend=backend,
        action_pattern=action_pattern,
        capability_flags=dict(capability_flags),
        capability_failure=capability_failure,
    )


def action_for_step(
    pattern: str,
    step_index: int,
    action_dim: int,
    *,
    amplitude: float,
    rng: random.Random,
) -> tuple[float, ...]:
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    if pattern == "zero":
        return (0.0,) * action_dim
    if pattern == "random":
        return tuple(rng.uniform(-amplitude, amplitude) for _ in range(action_dim))
    if pattern == "sine":
        return tuple(
            amplitude * math.sin(step_index * 0.13 + joint_index * 0.37)
            for joint_index in range(action_dim)
        )
    raise ValueError(f"Unknown action pattern: {pattern}")


def default_capability_flags(
    n_envs: int,
    *,
    batched_build_supported: bool,
) -> dict[str, bool]:
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    return {
        "batched_build_supported": batched_build_supported,
        "batched_action_write_supported": False,
        "batched_state_read_supported": False,
        "selected_reset_supported": False,
        "cpu_readback_per_step": True,
        "per_env_python_loop": False,
    }


def format_metrics(metrics: ThroughputMetrics) -> dict[str, object]:
    rows: dict[str, object] = {
        "probe": "genesis_g1_physics_throughput",
        "status": "ok",
        "backend": metrics.backend,
        "gpu_backend": is_gpu_backend(metrics.backend),
        "n_envs": metrics.n_envs,
        "action_pattern": metrics.action_pattern,
        "render_enabled": False,
        "sonic_enabled": False,
        "onnx_enabled": False,
        "planner_enabled": False,
        "gif_enabled": False,
        "build_time_s": metrics.build_time_s,
        "warmup_time_s": metrics.warmup_time_s,
        "measure_time_s": metrics.measure_time_s,
        "warmup_policy_steps": metrics.warmup_policy_steps,
        "measure_policy_steps": metrics.measure_policy_steps,
        "decimation": metrics.decimation,
        "policy_steps_per_sec": metrics.policy_steps_per_sec,
        "sim_steps_per_sec": metrics.sim_steps_per_sec,
        "env_steps_per_sec": metrics.env_steps_per_sec,
        "capability_failure": metrics.capability_failure,
    }
    rows.update(metrics.capability_flags)
    return rows


def print_kv(values: Mapping[str, object]) -> None:
    for key, value in values.items():
        print(f"{key}={format_value(value)}")


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.9g}"
    if isinstance(value, Path):
        return str(value)
    return str(value)


def is_gpu_backend(backend: str) -> bool:
    return backend.lower() == "cuda"


class CapabilityFailure(RuntimeError):
    """Raised when the current backend explicitly lacks a requested capability."""


if __name__ == "__main__":
    main()
