"""Smoke/profile the minimal VectorizedGenesisBackend on the H200 target."""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from h200_locomotion_lab.envs.vectorized_genesis_backend import (
    VectorizedGenesisBackend,
    VectorizedGenesisConfig,
    as_rows,
    tensor_shape,
)


METRIC_KEYS = (
    "status",
    "cuda_visible_devices",
    "physical_gpu",
    "logical_cuda_device",
    "backend",
    "n_envs",
    "action_shape",
    "observation_shape",
    "build_time_s",
    "warmup_time_s",
    "measure_time_s",
    "policy_steps_per_sec",
    "sim_steps_per_sec",
    "env_policy_steps_per_sec",
    "env_sim_steps_per_sec",
    "action_device",
    "qpos_device",
    "dofs_pos_device",
    "dofs_vel_device",
    "root_pos_device",
    "root_quat_device",
    "root_vel_device",
    "tensor_device_ok",
    "selected_reset_changes_only_target_envs",
    "blocker",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=positive_int, required=True)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--logical-cuda-device", default="cuda:0")
    parser.add_argument("--warmup-policy-steps", type=non_negative_int, default=20)
    parser.add_argument("--measure-policy-steps", type=positive_int, default=100)
    args = parser.parse_args()

    metrics = initial_metrics(args)
    try:
        metrics.update(run_smoke(args))
        metrics["status"] = "ok"
    except Exception as exc:  # pragma: no cover - target-only failure path.
        metrics["status"] = "failed"
        metrics["blocker"] = f"{exc.__class__.__name__}:{exc}"
    emit_metrics(metrics)
    if metrics["status"] != "ok":
        raise SystemExit(1)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend == "cuda":
        verify_cuda_isolation(
            physical_gpu=str(args.physical_gpu),
            logical_cuda_device=args.logical_cuda_device,
        )

    started = time.perf_counter()
    backend = VectorizedGenesisBackend(
        VectorizedGenesisConfig(
            n_envs=args.n_envs,
            backend=args.backend,
            logical_cuda_device=args.logical_cuda_device,
        )
    )
    build_time_s = elapsed_since(started)

    observation = backend.reset()
    action = make_action(backend, value=0.0)

    warmup_started = time.perf_counter()
    for _ in range(args.warmup_policy_steps):
        backend.step(action)
    warmup_time_s = elapsed_since(warmup_started)

    selected_reset_ok = exercise_selected_reset(backend)

    measure_started = time.perf_counter()
    for _ in range(args.measure_policy_steps):
        transition = backend.step(action)
    measure_time_s = elapsed_since(measure_started)

    device_report = backend.tensor_device_report()
    final_observation = transition.observation if args.measure_policy_steps else observation
    return {
        "n_envs": args.n_envs,
        "build_time_s": build_time_s,
        "warmup_time_s": warmup_time_s,
        "measure_time_s": measure_time_s,
        "action_shape": shape_string(action),
        "observation_shape": shape_string(final_observation),
        "selected_reset_changes_only_target_envs": selected_reset_ok,
        "tensor_device_ok": backend.tensor_device_ok(),
        **throughput(
            policy_steps=args.measure_policy_steps,
            decimation=backend.decimation,
            n_envs=args.n_envs,
            elapsed_s=measure_time_s,
        ),
        **device_report,
    }


def exercise_selected_reset(backend: VectorizedGenesisBackend) -> bool:
    if backend.n_envs < 2:
        return False
    action = make_action(backend, value=1.0)
    backend.step(action)
    before = reset_state_rows(backend)
    backend.reset(env_ids=[0])
    after = reset_state_rows(backend)
    changed = [
        index
        for index, (before_row, after_row) in enumerate(zip(before, after))
        if rows_differ(before_row, after_row)
    ]
    return changed == [0]


def reset_state_rows(backend: VectorizedGenesisBackend) -> list[list[float]]:
    qpos_rows = as_rows(backend.robot.get_qpos())
    dof_rows = as_rows(backend.robot.get_dofs_position(dofs_idx_local=backend.motor_dof_indices))
    return [qpos_row + dof_row for qpos_row, dof_row in zip(qpos_rows, dof_rows)]


def make_action(backend: VectorizedGenesisBackend, *, value: float) -> Any:
    if backend.torch is not None:
        return backend.torch.full(
            (backend.n_envs, backend.action_dim),
            float(value),
            dtype=backend.torch.float32,
            device=backend.config.logical_cuda_device,
        )
    return [[float(value)] * backend.action_dim for _ in range(backend.n_envs)]


def verify_cuda_isolation(*, physical_gpu: str, logical_cuda_device: str) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if tokens != [physical_gpu]:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES expected {physical_gpu}, got {visible}")
    if logical_cuda_device != "cuda:0":
        raise RuntimeError("logical_cuda_device must be cuda:0")
    import torch  # type: ignore[import-not-found]

    if not torch.cuda.is_available():
        raise RuntimeError("torch CUDA unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible CUDA device, got {torch.cuda.device_count()}")


def throughput(
    *,
    policy_steps: int,
    decimation: int,
    n_envs: int,
    elapsed_s: float,
) -> dict[str, float]:
    if elapsed_s <= 0:
        return {
            "policy_steps_per_sec": 0.0,
            "sim_steps_per_sec": 0.0,
            "env_policy_steps_per_sec": 0.0,
            "env_sim_steps_per_sec": 0.0,
        }
    return {
        "policy_steps_per_sec": policy_steps / elapsed_s,
        "sim_steps_per_sec": policy_steps * decimation / elapsed_s,
        "env_policy_steps_per_sec": policy_steps * n_envs / elapsed_s,
        "env_sim_steps_per_sec": policy_steps * decimation * n_envs / elapsed_s,
    }


def rows_differ(left: list[float], right: list[float], *, atol: float = 1e-8) -> bool:
    return len(left) != len(right) or any(abs(a - b) > atol for a, b in zip(left, right))


def initial_metrics(args: argparse.Namespace) -> dict[str, Any]:
    metrics = {key: "unavailable" for key in METRIC_KEYS}
    metrics.update(
        {
            "status": "failed",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
            "physical_gpu": str(args.physical_gpu),
            "logical_cuda_device": args.logical_cuda_device,
            "backend": args.backend,
            "n_envs": args.n_envs,
            "tensor_device_ok": False,
            "selected_reset_changes_only_target_envs": False,
            "blocker": "",
        }
    )
    return metrics


def emit_metrics(metrics: dict[str, Any]) -> None:
    for key in METRIC_KEYS:
        print(format_key_value(key, metrics.get(key, "unavailable")), flush=True)


def format_key_value(key: str, value: Any) -> str:
    if isinstance(value, bool):
        formatted = "true" if value else "false"
    elif isinstance(value, float):
        formatted = f"{value:.6f}"
    else:
        formatted = str(value)
    escaped = formatted.replace("\n", "\\n")
    return f"{key}={escaped}"


def shape_string(value: Any) -> str:
    return "x".join(str(item) for item in tensor_shape(value))


def elapsed_since(started: float) -> float:
    return time.perf_counter() - started


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


if __name__ == "__main__":
    main()
