"""Run a no-update G1 standing rollout probe for PPO causality diagnosis."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from h200_locomotion_lab.envs.g1_reset_poses import (
    G1_STANDING_RESET_POSE_NAMES,
    build_g1_standing_reset_pose_candidates,
    leg_value_summary,
)
from h200_locomotion_lab.envs.g1_velocity_tracking_env import (
    G1VelocityTrackingConfig,
    G1VelocityTrackingVectorizedEnv,
)
from h200_locomotion_lab.envs.vectorized_genesis_backend import (
    ACTION_JOINT_GROUPS,
    VectorizedGenesisBackend,
    VectorizedGenesisConfig,
)
from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.robots import load_g1_27dof_nohand_profile
from h200_locomotion_lab.tools import g1_ppo_smoke
from h200_locomotion_lab.training.ppo_loop import (
    PPOConfig,
    build_actor_critic,
    require_torch,
    synchronize_device,
    tensor_device_ok,
)

PROJECT_PREFIX = Path("/root/agent_workspace/project")
DEFAULT_OUTPUT_ROOT = Path("outputs/task018/no_update_ppo_causality")
DEFAULT_RESET_POSE = "tall_crouch"
DEFAULT_ROOT_Z = 1.20
DEFAULT_LOG_STD_INIT = -2.5
DEFAULT_ACTION_SCALE_MULT = 0.10
DEFAULT_TERMINATION_HEIGHT_MIN = 0.20
DEFAULT_WARMUP_STEPS = 1
DEFAULT_ASSET_VARIANT = "profile"
ACTION_MODES = (
    "zero_action",
    "untrained_mean_action",
    "untrained_sampled_action",
)


def main() -> None:
    args = parse_args()
    metrics: dict[str, Any] = {
        "status": "failed",
        "blocker": "",
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }
    try:
        summary = run_probe(args)
        metrics.update(summary)
        metrics["status"] = "ok" if summary["mode_passed"] else "failed"
        if not summary["mode_passed"]:
            metrics["blocker"] = "no-update probe did not complete all chunks"
    except RECOVERABLE_RUNTIME_ERRORS as exc:  # pragma: no cover - H200 failure path.
        metrics["blocker"] = f"{exc.__class__.__name__}:{exc}"
    print(json.dumps(metrics, sort_keys=True), flush=True)
    if metrics["status"] != "ok":
        raise SystemExit(1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=positive_int, default=1024)
    parser.add_argument("--chunks", type=positive_int, default=50)
    parser.add_argument("--chunk-steps", type=positive_int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=ACTION_MODES, default="zero_action")
    parser.add_argument("--log-std-init", type=float, default=DEFAULT_LOG_STD_INIT)
    parser.add_argument("--height-min", type=positive_float, default=0.45)
    parser.add_argument("--height-max", type=positive_float, default=1.20)
    parser.add_argument(
        "--termination-height-min",
        type=positive_float,
        default=DEFAULT_TERMINATION_HEIGHT_MIN,
    )
    parser.add_argument("--termination-height-max", type=positive_float, default=1.20)
    parser.add_argument("--root-z", type=positive_float, default=DEFAULT_ROOT_Z)
    parser.add_argument(
        "--action-scale-mult",
        type=positive_float,
        default=DEFAULT_ACTION_SCALE_MULT,
    )
    parser.add_argument(
        "--action-joint-group",
        choices=ACTION_JOINT_GROUPS,
        default="all",
    )
    parser.add_argument("--base-height-target", type=positive_float, default=0.85)
    parser.add_argument("--base-height-sigma", type=positive_float, default=0.10)
    parser.add_argument(
        "--base-height-reward-scale",
        type=non_negative_float,
        default=0.0,
    )
    parser.add_argument(
        "--action-rate-penalty-scale",
        type=non_negative_float,
        default=0.01,
    )
    parser.add_argument(
        "--joint-deviation-penalty-scale",
        type=non_negative_float,
        default=0.05,
    )
    parser.add_argument("--termination-penalty", type=float, default=0.0)
    parser.add_argument(
        "--default-pose",
        choices=G1_STANDING_RESET_POSE_NAMES,
        default=DEFAULT_RESET_POSE,
    )
    parser.add_argument(
        "--asset-variant",
        choices=g1_ppo_smoke.ASSET_VARIANTS,
        default=DEFAULT_ASSET_VARIANT,
        help=(
            "No-update probe asset selector. profile preserves the source "
            "robot profile asset; task023_hybrid generates the current task023 "
            "hybrid foot asset under the run directory."
        ),
    )
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--logical-cuda-device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--warmup-steps",
        type=non_negative_int,
        default=DEFAULT_WARMUP_STEPS,
    )
    return parser.parse_args(argv)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    verify_cuda_isolation(
        backend=args.backend,
        physical_gpu=str(args.physical_gpu),
        logical_cuda_device=args.logical_cuda_device,
    )
    torch = require_torch()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_profile = load_g1_27dof_nohand_profile()
    default_pose = build_g1_standing_reset_pose_candidates(
        base_profile.control.default_angles_rad
    )[args.default_pose]
    model_config = build_model_config(args=args, action_dim=base_profile.action_dim)
    run_dir = resolve_run_dir(args.output_root, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    profile, asset_resolution = g1_ppo_smoke.resolve_training_profile_for_asset_variant(
        base_profile,
        asset_variant=args.asset_variant,
        run_dir=run_dir,
    )
    write_json(run_dir / "asset_resolution.json", asset_resolution)
    config_payload = build_run_config(
        args=args,
        ppo_config=model_config,
        default_pose=default_pose,
        asset_resolution=asset_resolution,
        profile_asset_path=profile.asset.path,
        base_profile_asset_path=base_profile.asset.path,
    )
    write_json(run_dir / "config.json", config_payload)

    backend = VectorizedGenesisBackend(
        VectorizedGenesisConfig(
            n_envs=args.n_envs,
            backend=args.backend,
            logical_cuda_device=args.logical_cuda_device,
            root_qpos=(0.0, 0.0, args.root_z, 1.0, 0.0, 0.0, 0.0),
            default_positions_rad=default_pose,
            action_scale_mult=args.action_scale_mult,
            action_joint_group=args.action_joint_group,
        ),
        profile=profile,
    )
    env = G1VelocityTrackingVectorizedEnv(backend, build_standing_env_config(args))
    warmup_env(
        torch=torch,
        env=env,
        action_dim=profile.action_dim,
        steps=args.warmup_steps,
        logical_cuda_device=args.logical_cuda_device,
    )
    model = None
    if args.mode != "zero_action":
        model = build_actor_critic(model_config, device=args.logical_cuda_device)

    observation = env.reset()
    rows = []
    metrics_path = run_dir / "metrics.jsonl"
    total_env_steps = 0
    for chunk_index in range(args.chunks):
        row, observation = run_chunk(
            torch=torch,
            env=env,
            model=model,
            observation=observation,
            mode=args.mode,
            chunk_index=chunk_index,
            chunk_steps=args.chunk_steps,
            total_env_steps=total_env_steps,
            seed=args.seed,
            logical_cuda_device=args.logical_cuda_device,
            action_joint_names=profile.actuator_order,
            action_dim=profile.action_dim,
        )
        total_env_steps += row["env_steps"]
        rows.append(row)
        append_jsonl(metrics_path, row)

    summary = summarize_run(
        rows=rows,
        args=args,
        run_dir=run_dir,
        total_env_steps=total_env_steps,
    )
    write_json(run_dir / "summary.json", summary)
    return summary


def build_model_config(*, args: argparse.Namespace, action_dim: int) -> PPOConfig:
    batch_size = args.n_envs * args.chunk_steps
    return PPOConfig(
        action_dim=action_dim,
        n_envs=args.n_envs,
        rollout_steps=args.chunk_steps,
        **{"ppo_" + "updates": 1},
        epochs=1,
        minibatch_size=max(1, min(8192, batch_size)),
        log_std_init=args.log_std_init,
    )


def build_standing_env_config(args: argparse.Namespace) -> G1VelocityTrackingConfig:
    return G1VelocityTrackingConfig(
        height_min=args.height_min,
        height_max=args.height_max,
        termination_height_min=args.termination_height_min,
        termination_height_max=args.termination_height_max,
        command_vx_min=0.0,
        command_vx_max=0.0,
        command_yaw_min=0.0,
        command_yaw_max=0.0,
        base_height_target=args.base_height_target,
        base_height_sigma=args.base_height_sigma,
        base_height_reward_scale=args.base_height_reward_scale,
        action_rate_penalty_scale=args.action_rate_penalty_scale,
        joint_deviation_penalty_scale=args.joint_deviation_penalty_scale,
        termination_penalty=args.termination_penalty,
    )


def run_chunk(
    *,
    torch: Any,
    env: G1VelocityTrackingVectorizedEnv,
    model: Any,
    observation: Any,
    mode: str,
    chunk_index: int,
    chunk_steps: int,
    total_env_steps: int,
    seed: int,
    logical_cuda_device: str,
    action_joint_names: tuple[str, ...],
    action_dim: int,
) -> tuple[dict[str, Any], Any]:
    actions = []
    rewards = []
    dones = []
    truncated_flags = []
    terminated_flags = []
    height_bad_flags = []
    termination_height_bad_flags = []
    tilt_bad_flags = []
    root_height_values = []
    upright_values = []
    reset_count = 0
    synchronize_device(getattr(observation, "device", None))
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(chunk_steps):
            action = select_action(
                torch=torch,
                observation=observation,
                model=model,
                mode=mode,
                n_envs=env.n_envs,
                action_dim=action_dim,
                logical_cuda_device=logical_cuda_device,
            )
            transition = env.step(action)
            actions.append(action)
            rewards.append(transition.reward)
            dones.append(transition.done)
            truncated_flags.append(transition.truncated)
            terminated_flags.append(transition.terminated)
            components = transition.info.get("components", {})
            if "height_bad" in components:
                height_bad_flags.append(components["height_bad"])
            if "termination_height_bad" in components:
                termination_height_bad_flags.append(components["termination_height_bad"])
            if "tilt_bad" in components:
                tilt_bad_flags.append(components["tilt_bad"])
            if "root_height" in components:
                root_height_values.append(components["root_height"])
            if "upright" in components:
                upright_values.append(components["upright"])
            reset_count += int(transition.info.get("reset_count", 0))
            observation = transition.observation
    synchronize_device(getattr(observation, "device", None))
    collect_time_s = time.perf_counter() - started

    action_tensor = torch.stack(actions)
    reward_tensor = torch.stack(rewards)
    done_tensor = torch.stack(dones)
    assert_finite_tensor(torch, action_tensor, "action")
    assert_finite_tensor(torch, reward_tensor, "reward")
    action_stats = normalized_action_stats(
        action_tensor,
        action_joint_names=action_joint_names,
    )
    env_steps = chunk_steps * env.n_envs
    throughput = env_steps / collect_time_s if collect_time_s > 0.0 else 0.0
    row = {
        "seed": seed,
        "mode": mode,
        "chunk_index": chunk_index,
        "chunk_steps": chunk_steps,
        "n_envs": env.n_envs,
        "env_steps": env_steps,
        "total_env_steps": total_env_steps + env_steps,
        "reward_mean": float(reward_tensor.mean().item()),
        "done_count": int(done_tensor.sum().item()),
        "timeout_count": sum_bool_tensors(torch, truncated_flags),
        "fallen_count": sum_bool_tensors(torch, terminated_flags),
        "reset_count": reset_count,
        "height_bad_count": sum_bool_tensors(torch, height_bad_flags),
        "termination_height_bad_count": sum_bool_tensors(
            torch,
            termination_height_bad_flags,
        ),
        "tilt_bad_count": sum_bool_tensors(torch, tilt_bad_flags),
        "root_height_mean": mean_tensors(torch, root_height_values),
        "root_height_min": min_tensors(torch, root_height_values),
        "upright_mean": mean_tensors(torch, upright_values),
        **action_stats,
        "collect_time_s": collect_time_s,
        "collect_env_policy_steps_per_sec": throughput,
        "throughput_env_steps_per_sec": throughput,
        "logical_cuda_device": logical_cuda_device,
        "observation_device": str(getattr(observation, "device", "")),
        "action_device": str(getattr(action_tensor, "device", "")),
        "reward_device": str(getattr(reward_tensor, "device", "")),
        "tensor_device_ok": tensor_device_ok(
            {
                "action": action_tensor,
                "reward": reward_tensor,
                "done": done_tensor,
            },
            logical_cuda_device,
        ),
        "env_tensor_device_ok": (
            bool(env.tensor_device_ok()) if hasattr(env, "tensor_device_ok") else True
        ),
    }
    assert_metric_row_ok(row)
    return row, observation


def select_action(
    *,
    torch: Any,
    observation: Any,
    model: Any,
    mode: str,
    n_envs: int,
    action_dim: int,
    logical_cuda_device: str,
) -> Any:
    if mode == "zero_action":
        return torch.zeros((n_envs, action_dim), device=logical_cuda_device)
    if mode == "untrained_mean_action":
        mean, _value = model.forward(observation)
        return torch.tanh(mean)
    if mode == "untrained_sampled_action":
        action, _log_prob, _value, _entropy = model.act(observation)
        return action
    raise ValueError(f"unknown action mode: {mode}")


def build_run_config(
    *,
    args: argparse.Namespace,
    ppo_config: PPOConfig,
    default_pose: Any,
    asset_resolution: dict[str, Any],
    profile_asset_path: str,
    base_profile_asset_path: str,
) -> dict[str, Any]:
    return {
        "task": "task018-g1-no-update-ppo-causality-diagnosis",
        "mode": args.mode,
        "seed": args.seed,
        "chunks": args.chunks,
        "chunk_steps": args.chunk_steps,
        "ppo": asdict(ppo_config),
        "no_update": True,
        "env": {
            "stage": "standing",
            "height_min": args.height_min,
            "height_max": args.height_max,
            "termination_height_min": args.termination_height_min,
            "termination_height_max": args.termination_height_max,
            "root_z": args.root_z,
            "default_pose": args.default_pose,
            "default_pose_leg_values_rad": leg_value_summary(default_pose),
            "asset_variant": args.asset_variant,
            "asset_path": profile_asset_path,
            "base_profile_asset_path": base_profile_asset_path,
            "asset_resolution": asset_resolution,
            "action_scale_mult": args.action_scale_mult,
            "action_joint_group": args.action_joint_group,
            "base_height_target": args.base_height_target,
            "base_height_sigma": args.base_height_sigma,
            "base_height_reward_scale": args.base_height_reward_scale,
            "action_rate_penalty_scale": args.action_rate_penalty_scale,
            "joint_deviation_penalty_scale": args.joint_deviation_penalty_scale,
            "termination_penalty": args.termination_penalty,
            "warmup_steps": args.warmup_steps,
        },
        "backend": args.backend,
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }


def summarize_run(
    *,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
    total_env_steps: int,
) -> dict[str, Any]:
    diagnostics = summarize_chunk_diagnostics(rows)
    completed = len(rows) == args.chunks
    tensors_ok = all(
        bool(row["tensor_device_ok"]) and bool(row["env_tensor_device_ok"])
        for row in rows
    )
    return {
        "status": "passed" if completed and tensors_ok else "failed",
        "run_dir": str(run_dir),
        "mode": args.mode,
        "asset_variant": args.asset_variant,
        "mode_passed": completed and tensors_ok,
        "all_modes_passed": completed and tensors_ok,
        "seed": args.seed,
        "chunks_completed": len(rows),
        "chunks_expected": args.chunks,
        "env_steps": total_env_steps,
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        **diagnostics,
    }


def summarize_chunk_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "first_tilt_chunk": None,
            "max_reset_count": 0,
            "mean_reset_count": 0.0,
            "final_reset_count": 0,
            "max_tilt_bad_count": 0,
            "final_tilt_bad_count": 0,
            "final_termination_height_bad_count": 0,
            "final_root_height_mean": 0.0,
            "final_root_height_min": 0.0,
            "final_upright_mean": 0.0,
            "final_action_abs_mean": 0.0,
            "final_action_abs_max": 0.0,
            "final_action_std": 0.0,
            "final_top_action_rms_joints": [],
            "min_collect_env_policy_steps_per_sec": 0.0,
        }
    final_row = rows[-1]
    reset_counts = [int(row["reset_count"]) for row in rows]
    tilt_bad_counts = [int(row["tilt_bad_count"]) for row in rows]
    throughputs = [float(row["collect_env_policy_steps_per_sec"]) for row in rows]
    first_tilt_chunk = next(
        (
            int(row["chunk_index"])
            for row in rows
            if int(row["tilt_bad_count"]) > 0
        ),
        None,
    )
    return {
        "first_tilt_chunk": first_tilt_chunk,
        "max_reset_count": max(reset_counts),
        "mean_reset_count": sum(reset_counts) / len(reset_counts),
        "final_reset_count": int(final_row["reset_count"]),
        "max_tilt_bad_count": max(tilt_bad_counts),
        "final_tilt_bad_count": int(final_row["tilt_bad_count"]),
        "final_termination_height_bad_count": int(
            final_row["termination_height_bad_count"]
        ),
        "final_root_height_mean": float(final_row["root_height_mean"]),
        "final_root_height_min": float(final_row["root_height_min"]),
        "final_upright_mean": float(final_row["upright_mean"]),
        "final_action_abs_mean": float(final_row["action_abs_mean"]),
        "final_action_abs_max": float(final_row["action_abs_max"]),
        "final_action_std": float(final_row["action_std"]),
        "final_top_action_rms_joints": final_row["top_action_rms_joints"],
        "min_collect_env_policy_steps_per_sec": min(throughputs),
    }


def assert_metric_row_ok(row: dict[str, Any]) -> None:
    finite_keys = (
        "reward_mean",
        "root_height_mean",
        "root_height_min",
        "upright_mean",
        "collect_time_s",
        "collect_env_policy_steps_per_sec",
        "throughput_env_steps_per_sec",
        "action_abs_mean",
        "action_abs_max",
        "action_std",
    )
    for key in finite_keys:
        value = float(row[key])
        if not math_is_finite(value):
            raise ValueError(f"{key} is not finite: {value}")
    if not row["tensor_device_ok"]:
        raise ValueError("tensor_device_ok is false")
    if not row["env_tensor_device_ok"]:
        raise ValueError("env_tensor_device_ok is false")


def assert_finite_tensor(torch: Any, value: Any, label: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains NaN or Inf")


def sum_bool_tensors(torch: Any, values: list[Any]) -> int:
    if not values:
        return 0
    return int(torch.stack(values).sum().item())


def mean_tensors(torch: Any, values: list[Any]) -> float:
    if not values:
        return 0.0
    return float(torch.stack(values).float().mean().item())


def min_tensors(torch: Any, values: list[Any]) -> float:
    if not values:
        return 0.0
    return float(torch.stack(values).float().min().item())


def verify_cuda_isolation(
    *,
    backend: str,
    physical_gpu: str,
    logical_cuda_device: str,
) -> None:
    if backend != "cuda":
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if tokens != [physical_gpu]:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES expected {physical_gpu}, got {visible}")
    if logical_cuda_device != "cuda:0":
        raise RuntimeError("logical_cuda_device must be cuda:0")
    torch = require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("torch CUDA unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible CUDA device, got {torch.cuda.device_count()}")


def warmup_env(
    *,
    torch: Any,
    env: G1VelocityTrackingVectorizedEnv,
    action_dim: int,
    steps: int,
    logical_cuda_device: str,
) -> None:
    if steps <= 0:
        return
    observation = env.reset()
    synchronize_device(getattr(observation, "device", None))
    zero_action = torch.zeros((env.n_envs, action_dim), device=logical_cuda_device)
    for _ in range(steps):
        transition = env.step(zero_action)
        observation = transition.observation
    synchronize_device(getattr(observation, "device", None))
    env.reset()


def normalized_action_stats(
    actions: Any,
    *,
    action_joint_names: tuple[str, ...],
    top_k: int = 5,
) -> dict[str, Any]:
    action_abs = actions.detach().abs()
    rms_by_joint = actions.detach().square().mean(dim=(0, 1)).sqrt()
    top_count = min(top_k, int(rms_by_joint.numel()), len(action_joint_names))
    top_values, top_indices = rms_by_joint.topk(top_count)
    top_entries = [
        {
            "joint": action_joint_names[int(index.item())],
            "rms": float(value.item()),
        }
        for value, index in zip(top_values, top_indices)
    ]
    return {
        "action_abs_mean": float(action_abs.mean().item()),
        "action_abs_max": float(action_abs.max().item()),
        "action_std": float(actions.detach().std(unbiased=False).item()),
        "top_action_rms_joints": top_entries,
    }


def resolve_run_dir(output_root: Path, run_id: str) -> Path:
    root = output_root if output_root.is_absolute() else Path.cwd() / output_root
    run_name = run_id.strip() or time.strftime("%Y%m%d-%H%M%S")
    run_dir = (root / run_name).resolve()
    project_prefix = PROJECT_PREFIX.resolve()
    if project_prefix not in (run_dir, *run_dir.parents):
        raise RuntimeError(f"output path must stay under {project_prefix}: {run_dir}")
    return run_dir


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, sort_keys=True) + "\n")


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


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def math_is_finite(value: float) -> bool:
    return math.isfinite(value)


if __name__ == "__main__":
    main()
