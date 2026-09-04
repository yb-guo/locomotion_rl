"""Closed-loop fixed-command evaluation for the Task048 normal-walking policy."""

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

from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS

DEFAULT_TASK = "Unitree-G1-Flat"
DEFAULT_LIN_VEL_X = 0.5
DEFAULT_MIN_ZERO_FALL_RATIO = 0.95
DEFAULT_MAX_LIN_VEL_ERROR = 0.35
DEFAULT_MAX_YAW_VEL_ERROR = 0.35
DEFAULT_MAX_GRAVITY_XY = 0.35

_RANDOMIZATION_EVENTS = (
    "push_robot",
    "foot_friction",
    "encoder_bias",
    "base_com",
    "body_com_offset",
    "pseudo_inertia",
    "pd_gains",
    "effort_limits",
    "joint_damping",
    "joint_friction",
    "randomize_terrain",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4800301)
    parser.add_argument("--lin-vel-x", type=float, default=DEFAULT_LIN_VEL_X)
    parser.add_argument("--min-zero-fall-ratio", type=float, default=DEFAULT_MIN_ZERO_FALL_RATIO)
    parser.add_argument("--max-lin-vel-error", type=float, default=DEFAULT_MAX_LIN_VEL_ERROR)
    parser.add_argument("--max-yaw-vel-error", type=float, default=DEFAULT_MAX_YAW_VEL_ERROR)
    parser.add_argument("--max-gravity-xy", type=float, default=DEFAULT_MAX_GRAVITY_XY)
    return parser.parse_args(argv)


def configure_clean_fixed_command(env_cfg: Any, lin_vel_x: float) -> None:
    """Make evaluation deterministic and command straight-ahead walking."""

    twist_cmd = env_cfg.commands["twist"]
    twist_cmd.heading_command = False
    twist_cmd.rel_heading_envs = 0.0
    twist_cmd.rel_standing_envs = 0.0
    twist_cmd.ranges.lin_vel_x = (lin_vel_x, lin_vel_x)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
    twist_cmd.ranges.heading = None

    for event_name in _RANDOMIZATION_EVENTS:
        env_cfg.events.pop(event_name, None)
    reset_base = env_cfg.events.get("reset_base")
    reset_params = getattr(reset_base, "params", None)
    if isinstance(reset_params, dict):
        pose_range = reset_params.get("pose_range")
        if isinstance(pose_range, dict):
            pose_range["yaw"] = (0.0, 0.0)
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.curriculum = {}
    terrain = getattr(env_cfg.scene, "terrain", None)
    generator = getattr(terrain, "terrain_generator", None)
    if generator is not None:
        generator.curriculum = False


def evaluate_gate(summary: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if float(summary["zero_fall_ratio"]) < float(args.min_zero_fall_ratio):
        reasons.append("zero_fall_ratio_below_threshold")
    if float(summary["lin_vel_error_mean"]) > float(args.max_lin_vel_error):
        reasons.append("lin_vel_error_above_threshold")
    if float(summary["yaw_vel_error_mean"]) > float(args.max_yaw_vel_error):
        reasons.append("yaw_vel_error_above_threshold")
    if float(summary["gravity_xy_mean"]) > float(args.max_gravity_xy):
        reasons.append("gravity_xy_above_threshold")
    return not reasons, reasons


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError("num_envs and steps must be positive")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import mjlab.tasks as _mjlab_tasks
    import src.tasks as _project_tasks

    del _mjlab_tasks, _project_tasks  # Imports register task packages by side effect.
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    torch.set_grad_enabled(False)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    configure_clean_fixed_command(env_cfg, args.lin_vel_x)

    start = time.time()
    outer_env = None
    try:
        base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        outer_env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
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
        unwrapped = rollout_env.unwrapped
        robot = unwrapped.scene["robot"]
        num_envs = int(args.num_envs)
        initial_x = robot.data.root_link_pos_w[:, 0].clone()

        done_counts = torch.zeros(num_envs, dtype=torch.long, device=args.device)
        lin_error_sum = torch.zeros((), dtype=torch.float64, device=args.device)
        yaw_error_sum = torch.zeros((), dtype=torch.float64, device=args.device)
        gravity_xy_sum = torch.zeros((), dtype=torch.float64, device=args.device)
        forward_vel_sum = torch.zeros((), dtype=torch.float64, device=args.device)
        reward_sum = torch.zeros((), dtype=torch.float64, device=args.device)
        gravity_xy_max = torch.zeros((), dtype=torch.float32, device=args.device)
        root_z_min = torch.full((num_envs,), float("inf"), device=args.device)
        action_abs_sum = torch.zeros((), dtype=torch.float64, device=args.device)
        action_abs_max = torch.zeros((), dtype=torch.float32, device=args.device)
        action_dim = 0

        for _ in range(args.steps):
            action = policy(obs)
            action_dim = int(action.shape[-1])
            obs, reward, dones = _step_env(rollout_env, action)
            command = unwrapped.command_manager.get_command("twist")
            lin_vel = robot.data.root_link_lin_vel_b[:, :2]
            yaw_vel = robot.data.root_link_ang_vel_b[:, 2]
            gravity_xy = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
            lin_error = torch.linalg.norm(command[:, :2] - lin_vel, dim=-1)
            yaw_error = torch.abs(command[:, 2] - yaw_vel)

            done_counts += dones.to(dtype=torch.long)
            lin_error_sum += lin_error.double().sum()
            yaw_error_sum += yaw_error.double().sum()
            gravity_xy_sum += gravity_xy.double().sum()
            forward_vel_sum += lin_vel[:, 0].double().sum()
            reward_sum += reward.double().sum()
            gravity_xy_max = torch.maximum(gravity_xy_max, gravity_xy.max())
            root_z_min = torch.minimum(root_z_min, robot.data.root_link_pos_w[:, 2])
            action_abs = action.abs()
            action_abs_sum += action_abs.double().sum()
            action_abs_max = torch.maximum(action_abs_max, action_abs.max())

        sample_count = float(args.steps * num_envs)
        action_count = float(args.steps * num_envs * max(action_dim, 1))
        final_x = robot.data.root_link_pos_w[:, 0]
        no_reset = done_counts == 0
        displacement = final_x - initial_x
        no_reset_displacement = displacement[no_reset]
        summary: dict[str, Any] = {
            "task": args.task,
            "checkpoint": str(checkpoint),
            "device": args.device,
            "seed": args.seed,
            "num_envs": num_envs,
            "steps": args.steps,
            "eval_time_s": args.steps * float(unwrapped.step_dt),
            "fixed_command": [args.lin_vel_x, 0.0, 0.0],
            "action_dim": action_dim,
            "zero_fall_ratio": float(no_reset.float().mean().item()),
            "done_count_total": int(done_counts.sum().item()),
            "lin_vel_error_mean": float((lin_error_sum / sample_count).item()),
            "yaw_vel_error_mean": float((yaw_error_sum / sample_count).item()),
            "gravity_xy_mean": float((gravity_xy_sum / sample_count).item()),
            "gravity_xy_max": float(gravity_xy_max.item()),
            "forward_vel_x_mean": float((forward_vel_sum / sample_count).item()),
            "reward_mean": float((reward_sum / sample_count).item()),
            "root_z_min": _tensor_stats(torch, root_z_min),
            "no_reset_world_x_displacement": _tensor_stats(torch, no_reset_displacement),
            "action_abs_mean": float((action_abs_sum / action_count).item()),
            "action_abs_max": float(action_abs_max.item()),
            "thresholds": {
                "min_zero_fall_ratio": args.min_zero_fall_ratio,
                "max_lin_vel_error": args.max_lin_vel_error,
                "max_yaw_vel_error": args.max_yaw_vel_error,
                "max_gravity_xy": args.max_gravity_xy,
            },
            "wall_time_s": time.time() - start,
        }
        summary["pass"], summary["failure_reasons"] = evaluate_gate(summary, args)
        return summary
    finally:
        if outer_env is not None:
            outer_env.close()


def _step_env(env: Any, action: Any) -> tuple[Any, Any, Any]:
    result = env.step(action)
    if len(result) == 4:
        obs, reward, dones, _extras = result
    else:
        obs, reward, terminated, truncated, _extras = result
        dones = terminated | truncated
    return obs, reward, dones


def _tensor_stats(torch: Any, values: Any) -> dict[str, float | int]:
    finite = values[torch.isfinite(values)].detach().float()
    if finite.numel() == 0:
        return {"count": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary = run_eval(args)
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        summary = {
            "task": args.task,
            "checkpoint": args.checkpoint,
            "command": list(sys.argv),
            "pass": False,
            "failure_reasons": ["eval_exception"],
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    summary["json_path"] = str(output)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
