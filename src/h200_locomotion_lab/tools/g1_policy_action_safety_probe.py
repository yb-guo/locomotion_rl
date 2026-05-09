"""Sweep initial policy action safety for the G1 vectorized Genesis env."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from h200_locomotion_lab.envs.g1_reset_poses import (
    build_g1_standing_reset_pose_candidates,
    leg_value_summary,
)
from h200_locomotion_lab.envs.g1_velocity_tracking_env import (
    G1VelocityTrackingConfig,
    G1VelocityTrackingVectorizedEnv,
)
from h200_locomotion_lab.envs.vectorized_genesis_backend import (
    VectorizedGenesisBackend,
    VectorizedGenesisConfig,
)
from h200_locomotion_lab.robots import load_g1_27dof_nohand_profile
from h200_locomotion_lab.training.ppo_loop import (
    PPOConfig,
    build_actor_critic,
    collect_rollout,
    require_torch,
)
from h200_locomotion_lab.tools.g1_ppo_smoke import (
    COMMAND_MODES,
    DEFAULT_RESET_POSE,
    DEFAULT_ROOT_Z,
    command_ranges_for_mode,
    non_negative_float,
    positive_float,
    positive_int,
    resolve_run_dir,
    verify_cuda_isolation,
)
from h200_locomotion_lab.tools.g1_standing_reset_pose_probe import (
    parse_float_list,
    parse_name_list,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/task014/policy_action_safety_probe")
DEFAULT_ACTION_SCALE_MULTS = "1.0,0.5,0.25"
DEFAULT_LOG_STD_INITS = "-0.5,-1.5,-2.5"
DEFAULT_COMMAND_MODES = "vx_yaw,standing"


def main() -> None:
    args = parse_args()
    result: dict[str, Any] = {
        "status": "failed",
        "blocker": "",
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }
    try:
        result.update(run_probe(args))
        result["status"] = "ok"
    except Exception as exc:  # pragma: no cover - H200 failure path.
        result["blocker"] = f"{exc.__class__.__name__}:{exc}"
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "ok":
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=positive_int, default=1024)
    parser.add_argument("--rollout-steps", type=positive_int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-scale-mults", default=DEFAULT_ACTION_SCALE_MULTS)
    parser.add_argument("--log-std-inits", default=DEFAULT_LOG_STD_INITS)
    parser.add_argument("--command-modes", default=DEFAULT_COMMAND_MODES)
    parser.add_argument("--height-min", type=positive_float, default=0.45)
    parser.add_argument("--height-max", type=positive_float, default=1.20)
    parser.add_argument("--root-z", type=positive_float, default=DEFAULT_ROOT_Z)
    parser.add_argument("--default-pose", default=DEFAULT_RESET_POSE)
    parser.add_argument("--base-height-target", type=positive_float, default=0.85)
    parser.add_argument("--base-height-sigma", type=positive_float, default=0.10)
    parser.add_argument("--base-height-reward-scale", type=non_negative_float, default=0.0)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--logical-cuda-device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    verify_cuda_isolation(
        backend=args.backend,
        physical_gpu=str(args.physical_gpu),
        logical_cuda_device=args.logical_cuda_device,
    )
    torch = require_torch()
    action_scale_mults = parse_float_list(args.action_scale_mults)
    log_std_inits = parse_float_list(args.log_std_inits)
    command_modes = parse_name_list(args.command_modes)
    unknown_modes = [mode for mode in command_modes if mode not in COMMAND_MODES]
    if unknown_modes:
        raise ValueError(f"unknown command mode: {unknown_modes[0]}")

    profile = load_g1_27dof_nohand_profile()
    pose_candidates = build_g1_standing_reset_pose_candidates(
        profile.control.default_angles_rad
    )
    if args.default_pose not in pose_candidates:
        raise ValueError(f"unknown default pose: {args.default_pose}")
    default_pose = pose_candidates[args.default_pose]
    config = PPOConfig(
        n_envs=args.n_envs,
        rollout_steps=args.rollout_steps,
        ppo_updates=1,
        log_std_init=0.0,
    )

    run_dir = resolve_run_dir(args.output_root, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        run_dir / "config.json",
        {
            "ppo": asdict(config),
            "seed": args.seed,
            "action_scale_mults": action_scale_mults,
            "log_std_inits": log_std_inits,
            "command_modes": command_modes,
            "height_min": args.height_min,
            "height_max": args.height_max,
            "root_z": args.root_z,
            "default_pose": args.default_pose,
            "default_pose_leg_values_rad": leg_value_summary(default_pose),
            "base_height_target": args.base_height_target,
            "base_height_sigma": args.base_height_sigma,
            "base_height_reward_scale": args.base_height_reward_scale,
            "backend": args.backend,
            "physical_gpu": str(args.physical_gpu),
            "logical_cuda_device": args.logical_cuda_device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        },
    )

    rows: list[dict[str, Any]] = []
    rows_path = run_dir / "rows.jsonl"
    backend = VectorizedGenesisBackend(
        VectorizedGenesisConfig(
            n_envs=args.n_envs,
            backend=args.backend,
            logical_cuda_device=args.logical_cuda_device,
            root_qpos=root_qpos(args.root_z),
            default_positions_rad=default_pose,
            action_scale_mult=action_scale_mults[0],
        ),
        profile=profile,
    )
    for action_scale_mult in action_scale_mults:
        backend.set_action_scale_mult(action_scale_mult)
        for command_mode in command_modes:
            env_config = build_env_config(args, command_mode)
            for log_std_init in log_std_inits:
                row = run_candidate(
                    torch=torch,
                    backend=backend,
                    env_config=env_config,
                    config=config,
                    seed=args.seed,
                    command_mode=command_mode,
                    action_scale_mult=action_scale_mult,
                    log_std_init=log_std_init,
                    logical_cuda_device=args.logical_cuda_device,
                )
                rows.append(row)
                append_jsonl(rows_path, row)

    summary = {
        "run_dir": str(run_dir),
        "row_count": len(rows),
        "best": choose_best_row(rows),
        "rows": rows,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def run_candidate(
    *,
    torch: Any,
    backend: VectorizedGenesisBackend,
    env_config: G1VelocityTrackingConfig,
    config: PPOConfig,
    seed: int,
    command_mode: str,
    action_scale_mult: float,
    log_std_init: float,
    logical_cuda_device: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model_config = PPOConfig(
        n_envs=config.n_envs,
        rollout_steps=config.rollout_steps,
        ppo_updates=1,
        log_std_init=log_std_init,
    )
    model = build_actor_critic(model_config, device=logical_cuda_device)
    env = G1VelocityTrackingVectorizedEnv(backend, env_config)
    observation = env.reset()
    started = time.perf_counter()
    batch = collect_rollout(env, model, observation, model_config)
    elapsed = time.perf_counter() - started
    actions_abs = batch.actions.abs()
    return {
        "seed": seed,
        "command_mode": command_mode,
        "action_scale_mult": action_scale_mult,
        "log_std_init": log_std_init,
        "rollout_steps": config.rollout_steps,
        "n_envs": config.n_envs,
        "env_steps": batch.env_steps,
        "time_s": elapsed,
        "collect_time_s": batch.collect_time_s,
        "env_policy_steps_per_sec": (
            batch.env_steps / batch.collect_time_s if batch.collect_time_s > 0.0 else 0.0
        ),
        "reset_count": batch.reset_count,
        "done_count": batch.done_count,
        "height_bad_count": batch.height_bad_count,
        "tilt_bad_count": batch.tilt_bad_count,
        "root_height_mean": batch.root_height_mean,
        "root_height_min": batch.root_height_min,
        "upright_mean": batch.upright_mean,
        "reward_mean": batch.reward_mean,
        "action_abs_mean": float(actions_abs.mean().item()),
        "action_abs_max": float(actions_abs.max().item()),
    }


def build_env_config(args: argparse.Namespace, command_mode: str) -> G1VelocityTrackingConfig:
    ranges = command_ranges_for_mode(command_mode)
    return G1VelocityTrackingConfig(
        command_vx_min=ranges["command_vx_min"],
        command_vx_max=ranges["command_vx_max"],
        command_yaw_min=ranges["command_yaw_min"],
        command_yaw_max=ranges["command_yaw_max"],
        height_min=args.height_min,
        height_max=args.height_max,
        base_height_target=args.base_height_target,
        base_height_sigma=args.base_height_sigma,
        base_height_reward_scale=args.base_height_reward_scale,
    )


def choose_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no rows to choose from")
    return min(
        rows,
        key=lambda row: (
            row["reset_count"],
            row["height_bad_count"],
            row["tilt_bad_count"],
            -row["root_height_min"],
            -row["env_policy_steps_per_sec"],
        ),
    )


def root_qpos(root_z: float) -> tuple[float, float, float, float, float, float, float]:
    return (0.0, 0.0, root_z, 1.0, 0.0, 0.0, 0.0)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
