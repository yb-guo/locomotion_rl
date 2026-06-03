"""Evaluate Task033 history-policy checkpoints with the runner-owned env wrapper."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any


DYNAMIC_CASES = {
    "switch": {
        "template": (
            (0.0, 2.0, None, "normal", 1.0),
            (2.0, 4.0, "left_knee_joint", "dead", 0.0),
            (4.0, 5.0, None, "normal", 1.0),
            (5.0, 7.0, "right_hip_yaw_joint", "dead", 0.0),
            (7.0, None, None, "normal", 1.0),
        ),
        "switch_start_s": 5.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Task033 history-policy checkpoint under canonical dynamic switch."
    )
    parser.add_argument("--task", default="Unitree-G1-Gripper-Flat-Task033-StackMlpK4-Fast2p0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dynamic-case", choices=sorted(DYNAMIC_CASES), default="switch")
    parser.add_argument("--lin-vel-x", type=float, default=2.0)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3303400)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-zero-fall-ratio", type=float, default=0.90)
    parser.add_argument("--min-recovery-success-ratio", type=float, default=0.75)
    parser.add_argument("--max-post-recovery-lin-vel-error", type=float, default=0.80)
    parser.add_argument("--max-post-recovery-yaw-vel-error", type=float, default=0.80)
    parser.add_argument("--max-gravity-xy-after-onset", type=float, default=0.80)
    return parser.parse_args()


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import torch
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    torch.set_grad_enabled(False)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    _configure_fixed_command(env_cfg, args.lin_vel_x)
    _configure_dynamic_case(env_cfg, args.dynamic_case)

    start = time.time()
    outer_env = None
    try:
        base = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        outer_env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        runner = runner_cls(outer_env, asdict(agent_cfg), device=args.device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=args.device)
        policy.eval()

        rollout_env = runner.env
        obs, _ = rollout_env.reset()
        base_env = rollout_env.unwrapped
        robot = base_env.scene["robot"]
        dt = float(base_env.step_dt)
        num_envs = args.num_envs

        done_counts = torch.zeros(num_envs, dtype=torch.long, device=args.device)
        done_after_onset = torch.zeros(num_envs, dtype=torch.bool, device=args.device)
        lin_error_sum = torch.zeros(num_envs, dtype=torch.float32, device=args.device)
        yaw_error_sum = torch.zeros(num_envs, dtype=torch.float32, device=args.device)
        gravity_xy_sum = torch.zeros(num_envs, dtype=torch.float32, device=args.device)
        gravity_xy_max_after_onset = torch.zeros(num_envs, dtype=torch.float32, device=args.device)
        root_z_min = torch.full((num_envs,), float("inf"), device=args.device)
        root_z_min_after_onset = torch.full((num_envs,), float("inf"), device=args.device)
        post_lin_sum = torch.zeros((), dtype=torch.float32, device=args.device)
        post_yaw_sum = torch.zeros((), dtype=torch.float32, device=args.device)
        post_count = torch.zeros((), dtype=torch.float32, device=args.device)
        recovery_success = torch.zeros(num_envs, dtype=torch.bool, device=args.device)

        for _ in range(args.steps):
            action = policy(obs)
            obs, _reward, dones = _step_env(rollout_env, action)
            dones_bool = dones.to(dtype=torch.bool)
            time_s = base_env.episode_length_buf.to(dtype=torch.float32) * dt
            command = base_env.command_manager.get_command("twist")
            lin_vel = robot.data.root_link_lin_vel_b[:, :2]
            yaw_vel = robot.data.root_link_ang_vel_b[:, 2]
            lin_error = torch.linalg.norm(command[:, :2] - lin_vel, dim=-1)
            yaw_error = torch.abs(command[:, 2] - yaw_vel)
            gravity_xy = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
            root_z = robot.data.root_link_pos_w[:, 2]

            done_counts += dones_bool.to(dtype=torch.long)
            lin_error_sum += lin_error
            yaw_error_sum += yaw_error
            gravity_xy_sum += gravity_xy
            root_z_min = torch.minimum(root_z_min, root_z)

            after_onset = time_s >= 2.0
            post_recovery = time_s >= 7.0
            done_after_onset |= after_onset & dones_bool
            root_z_min_after_onset[after_onset] = torch.minimum(
                root_z_min_after_onset[after_onset],
                root_z[after_onset],
            )
            gravity_xy_max_after_onset[after_onset] = torch.maximum(
                gravity_xy_max_after_onset[after_onset],
                gravity_xy[after_onset],
            )
            recovery_success |= (
                post_recovery
                & (lin_error <= args.max_post_recovery_lin_vel_error)
                & (yaw_error <= args.max_post_recovery_yaw_vel_error)
                & (gravity_xy <= args.max_gravity_xy_after_onset)
            )
            post_lin_sum += lin_error[post_recovery].sum()
            post_yaw_sum += yaw_error[post_recovery].sum()
            post_count += post_recovery.float().sum()

        steps = float(args.steps)
        zero_fall_ratio = float((done_counts == 0).float().mean().item())
        post_den = max(float(post_count.item()), 1.0)
        result = {
            "task": args.task,
            "checkpoint": str(checkpoint),
            "command": list(sys.argv),
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "eval_time_s": args.steps * dt,
            "fixed_lin_vel_x": args.lin_vel_x,
            "zero_fall_ratio": zero_fall_ratio,
            "done_count_total": int(done_counts.sum().item()),
            "done_after_onset_ratio": float(done_after_onset.float().mean().item()),
            "recovery_success_ratio": float(recovery_success.float().mean().item()),
            "lin_vel_error_mean": float((lin_error_sum / steps).mean().item()),
            "yaw_vel_error_mean": float((yaw_error_sum / steps).mean().item()),
            "gravity_xy_mean": float((gravity_xy_sum / steps).mean().item()),
            "post_recovery_lin_vel_error_mean": float((post_lin_sum / post_den).item()),
            "post_recovery_yaw_vel_error_mean": float((post_yaw_sum / post_den).item()),
            "max_gravity_xy_after_onset": float(gravity_xy_max_after_onset.max().item()),
            "root_z_min": _tensor_stats(torch, root_z_min),
            "root_z_min_after_onset": _tensor_stats(torch, root_z_min_after_onset),
            "wall_time_s": time.time() - start,
        }
        result["pass"] = (
            result["zero_fall_ratio"] >= args.min_zero_fall_ratio
            and result["recovery_success_ratio"] >= args.min_recovery_success_ratio
            and result["post_recovery_lin_vel_error_mean"] <= args.max_post_recovery_lin_vel_error
            and result["post_recovery_yaw_vel_error_mean"] <= args.max_post_recovery_yaw_vel_error
            and result["max_gravity_xy_after_onset"] <= args.max_gravity_xy_after_onset
        )
        return result
    finally:
        if outer_env is not None:
            outer_env.close()


def _configure_fixed_command(env_cfg: Any, lin_vel_x: float) -> None:
    twist_cmd = env_cfg.commands["twist"]
    twist_cmd.heading_command = False
    twist_cmd.rel_heading_envs = 0.0
    twist_cmd.rel_standing_envs = 0.0
    twist_cmd.ranges.lin_vel_x = (lin_vel_x, lin_vel_x)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
    twist_cmd.ranges.heading = None


def _configure_dynamic_case(env_cfg: Any, dynamic_case: str) -> None:
    if "dynamic_motor_failure" not in env_cfg.events:
        raise RuntimeError("dynamic_motor_failure event is absent")
    params = env_cfg.events["dynamic_motor_failure"].params
    if "template" in params:
        params["template"] = DYNAMIC_CASES[dynamic_case]["template"]


def _step_env(env: Any, action: Any) -> tuple[Any, Any, Any]:
    result = env.step(action)
    if len(result) == 4:
        obs, reward, dones, _extras = result
    else:
        obs, reward, terminated, truncated, _extras = result
        dones = terminated | truncated
    return obs, reward, dones


def _tensor_stats(torch: Any, x: Any) -> dict[str, float]:
    finite = x[torch.isfinite(x)].detach().float()
    if finite.numel() == 0:
        return {"mean": float("inf"), "min": float("inf"), "max": float("inf")}
    return {
        "mean": float(finite.mean().item()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
    }


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_eval(args)
    except Exception as exc:
        result = {
            "task": args.task,
            "checkpoint": args.checkpoint,
            "command": list(sys.argv),
            "seed": args.seed,
            "pass": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    result["json_path"] = str(output_json)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
