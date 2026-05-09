"""Smoke/profile G1 velocity-tracking vectorized env on H200."""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from h200_locomotion_lab.envs.g1_velocity_tracking_env import (
    G1VelocityTrackingConfig,
    G1VelocityTrackingVectorizedEnv,
)
from h200_locomotion_lab.envs.vectorized_genesis_backend import (
    VectorizedGenesisBackend,
    VectorizedGenesisConfig,
    as_rows,
    is_tensor_like,
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
    "reward_shape",
    "terminated_shape",
    "truncated_shape",
    "done_shape",
    "command_shape",
    "build_time_s",
    "warmup_time_s",
    "measure_time_s",
    "policy_steps_per_sec",
    "sim_steps_per_sec",
    "env_policy_steps_per_sec",
    "env_sim_steps_per_sec",
    "observation_device",
    "reward_device",
    "terminated_device",
    "truncated_device",
    "done_device",
    "command_device",
    "episode_length_device",
    "last_action_device",
    "qpos_device",
    "root_pos_device",
    "root_quat_device",
    "root_vel_device",
    "root_ang_vel_device",
    "dofs_pos_device",
    "dofs_vel_device",
    "tensor_device_ok",
    "selected_reset_changes_only_target_envs",
    "done_reset_resets_only_done_envs",
    "reward_mean",
    "done_count",
    "timeout_count",
    "fallen_count",
    "tracking_lin_vel_mean",
    "tracking_yaw_rate_mean",
    "upright_mean",
    "action_rate_penalty_mean",
    "joint_deviation_penalty_mean",
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
    parser.add_argument("--max-episode-steps", type=positive_int, default=1000)
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
    env = G1VelocityTrackingVectorizedEnv(
        backend,
        G1VelocityTrackingConfig(max_episode_steps=args.max_episode_steps),
    )
    build_time_s = elapsed_since(started)

    observation = env.reset()
    action = make_action(env, value=0.0)

    warmup_started = time.perf_counter()
    for _ in range(args.warmup_policy_steps):
        env.step(action)
    warmup_time_s = elapsed_since(warmup_started)

    selected_reset_ok = exercise_selected_reset(env)
    done_reset_ok = exercise_done_reset(env)

    measure_started = time.perf_counter()
    transition = None
    for _ in range(args.measure_policy_steps):
        transition = env.step(action)
    measure_time_s = elapsed_since(measure_started)
    if transition is None:
        raise RuntimeError("measure-policy-steps must be positive")

    device_report = env.tensor_device_report(transition)
    components = transition.info["components"]
    return {
        "n_envs": args.n_envs,
        "build_time_s": build_time_s,
        "warmup_time_s": warmup_time_s,
        "measure_time_s": measure_time_s,
        "action_shape": shape_string(action),
        "observation_shape": shape_string(transition.observation),
        "reward_shape": shape_string(transition.reward),
        "terminated_shape": shape_string(transition.terminated),
        "truncated_shape": shape_string(transition.truncated),
        "done_shape": shape_string(transition.done),
        "command_shape": shape_string(env.commands),
        "selected_reset_changes_only_target_envs": selected_reset_ok,
        "done_reset_resets_only_done_envs": done_reset_ok,
        "tensor_device_ok": env.tensor_device_ok(transition),
        "reward_mean": scalar_mean(transition.reward),
        "done_count": true_count(transition.done),
        "timeout_count": true_count(transition.truncated),
        "fallen_count": true_count(transition.terminated),
        "tracking_lin_vel_mean": scalar_mean(components["tracking_lin_vel"]),
        "tracking_yaw_rate_mean": scalar_mean(components["tracking_yaw_rate"]),
        "upright_mean": scalar_mean(components["upright"]),
        "action_rate_penalty_mean": scalar_mean(components["action_rate_penalty"]),
        "joint_deviation_penalty_mean": scalar_mean(components["joint_deviation_penalty"]),
        **throughput(
            policy_steps=args.measure_policy_steps,
            decimation=backend.decimation,
            n_envs=args.n_envs,
            elapsed_s=measure_time_s,
        ),
        **device_report,
    }


def exercise_selected_reset(env: G1VelocityTrackingVectorizedEnv) -> bool:
    if env.n_envs < 2:
        return False
    env.reset()
    env.step(make_action(env, value=1.0))
    before = reset_state_rows(env.backend)
    env.reset(env_ids=[0])
    after = reset_state_rows(env.backend)
    changed = [
        index
        for index, (before_row, after_row) in enumerate(zip(before, after))
        if rows_differ(before_row, after_row)
    ]
    return changed == [0]


def exercise_done_reset(env: G1VelocityTrackingVectorizedEnv) -> bool:
    if env.n_envs < 2:
        return False
    env.reset()
    if env.torch is not None:
        env.episode_lengths[:] = 0
        env.episode_lengths[0] = env.config.max_episode_steps - 1
    else:
        env.episode_lengths = [0 for _ in range(env.n_envs)]
        env.episode_lengths[0] = env.config.max_episode_steps - 1
    transition = env.step(make_action(env, value=1.0))
    action_rows = as_rows(env.backend.previous_action)
    episode_lengths = env.episode_lengths.detach().cpu().tolist() if is_tensor_like(
        env.episode_lengths
    ) else env.episode_lengths
    done = transition.done.detach().cpu().tolist() if is_tensor_like(
        transition.done
    ) else transition.done
    target_reset = all(abs(value) <= 1e-8 for value in action_rows[0])
    non_target_kept = all(abs(value - 1.0) <= 1e-8 for value in action_rows[1])
    return bool(done[0]) and target_reset and non_target_kept and episode_lengths[0] == 0


def reset_state_rows(backend: VectorizedGenesisBackend) -> list[list[float]]:
    state = backend.state()
    root_rows = as_rows(state.root_pos)
    dof_rows = as_rows(state.dof_pos)
    return [root_row + dof_row for root_row, dof_row in zip(root_rows, dof_rows)]


def make_action(env: G1VelocityTrackingVectorizedEnv, *, value: float) -> Any:
    if env.torch is not None:
        return env.torch.full(
            (env.n_envs, env.action_dim),
            float(value),
            dtype=env.torch.float32,
            device=env.backend.config.logical_cuda_device,
        )
    return [[float(value)] * env.action_dim for _ in range(env.n_envs)]


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


def scalar_mean(value: Any) -> float:
    if is_tensor_like(value):
        return float(value.float().mean().item())
    rows = as_rows(value)
    flat = [item for row in rows for item in row]
    return float(sum(flat) / len(flat)) if flat else 0.0


def true_count(value: Any) -> int:
    if is_tensor_like(value):
        return int(value.sum().item())
    rows = as_rows(value)
    return int(sum(1 for row in rows for item in row if bool(item)))


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
            "done_reset_resets_only_done_envs": False,
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
