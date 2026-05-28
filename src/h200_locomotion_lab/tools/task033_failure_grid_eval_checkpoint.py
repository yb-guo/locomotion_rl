"""Evaluate Task033 history-policy checkpoints on forced persistent dead motors."""

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


DEFAULT_TASK = "Unitree-G1-Gripper-Flat-Task033-StackMlpK4-FocusedDeadGrid-Fast2p0"
DEFAULT_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_knee_joint",
    "right_hip_roll_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
THRESHOLDS = {
    "min_zero_fall_ratio": 0.50,
    "max_lin_vel_error": 1.00,
    "max_yaw_vel_error": 1.00,
    "max_gravity_xy": 0.75,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Task033 history-policy checkpoint on forced dead-motor grid cases."
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--joints", nargs="*", default=list(DEFAULT_JOINTS))
    parser.add_argument("--lin-vel-x", type=float, default=2.0)
    parser.add_argument("--lin-vel-y", type=float, default=0.0)
    parser.add_argument("--ang-vel-z", type=float, default=0.0)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3303500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dead-scale", type=float, default=0.0)
    return parser.parse_args()


def run_case(args: argparse.Namespace, joint_name: str, index: int) -> dict[str, Any]:
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
    env_cfg.seed = args.seed + index
    _configure_fixed_command(env_cfg, args)
    _force_dead_motor(env_cfg, joint_name, args.dead_scale)

    outer_env = None
    start = time.time()
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

        done_counts = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
        reward_sum = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)
        lin_error_sum = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)
        yaw_error_sum = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)
        gravity_xy_sum = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)
        root_z_min = torch.full((args.num_envs,), float("inf"), device=args.device)

        initial_failure_report = _compact_failure_report(base_env)
        for _ in range(args.steps):
            action = policy(obs)
            obs, reward, dones = _step_env(rollout_env, action)
            command = base_env.command_manager.get_command("twist")
            lin_vel = robot.data.root_link_lin_vel_b[:, :2]
            yaw_vel = robot.data.root_link_ang_vel_b[:, 2]
            gravity_xy = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
            root_z = robot.data.root_link_pos_w[:, 2]

            done_counts += dones.to(dtype=torch.long)
            reward_sum += reward
            lin_error_sum += torch.linalg.norm(command[:, :2] - lin_vel, dim=-1)
            yaw_error_sum += torch.abs(command[:, 2] - yaw_vel)
            gravity_xy_sum += gravity_xy
            root_z_min = torch.minimum(root_z_min, root_z)

        lin_error = lin_error_sum / args.steps
        yaw_error = yaw_error_sum / args.steps
        gravity_xy = gravity_xy_sum / args.steps
        zero_fall_ratio = float((done_counts == 0).float().mean().item())
        result = {
            "case": f"dead_motor_grid_{index:02d}_{joint_name}",
            "task": args.task,
            "checkpoint": str(checkpoint),
            "command": list(sys.argv),
            "fixed_command": {
                "lin_vel_x": args.lin_vel_x,
                "lin_vel_y": args.lin_vel_y,
                "ang_vel_z": args.ang_vel_z,
            },
            "seed": args.seed + index,
            "device": args.device,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "control_dt_s": dt,
            "eval_time_s": args.steps * dt,
            "joint_name": joint_name,
            "dead_scale": args.dead_scale,
            "thresholds": THRESHOLDS,
            "zero_fall_ratio": zero_fall_ratio,
            "mean_done_count": float(done_counts.float().mean().item()),
            "max_done_count": int(done_counts.max().item()),
            "mean_reward": _tensor_stats(torch, reward_sum / args.steps),
            "lin_vel_error": _tensor_stats(torch, lin_error),
            "yaw_vel_error": _tensor_stats(torch, yaw_error),
            "gravity_xy": _tensor_stats(torch, gravity_xy),
            "root_z_min": _tensor_stats(torch, root_z_min),
            "failure_settings_initial": initial_failure_report,
            "failure_settings_final": _compact_failure_report(base_env),
            "wall_time_s": time.time() - start,
        }
        result["pass"] = _pass_thresholds(result)
        return result
    finally:
        if outer_env is not None:
            outer_env.close()


def safe_run_case(args: argparse.Namespace, output_dir: Path, joint_name: str, index: int) -> dict[str, Any]:
    case_name = f"dead_motor_grid_{index:02d}_{joint_name}"
    try:
        result = run_case(args, joint_name, index)
    except Exception as exc:
        result = {
            "case": case_name,
            "task": args.task,
            "checkpoint": args.checkpoint,
            "command": list(sys.argv),
            "fixed_command": {
                "lin_vel_x": args.lin_vel_x,
                "lin_vel_y": args.lin_vel_y,
                "ang_vel_z": args.ang_vel_z,
            },
            "seed": args.seed + index,
            "joint_name": joint_name,
            "thresholds": THRESHOLDS,
            "pass": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    path = output_dir / f"{case_name}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["json_path"] = str(path)
    print(json.dumps({"case": case_name, "pass": result.get("pass"), "path": str(path)}))
    return result


def _configure_fixed_command(env_cfg: Any, args: argparse.Namespace) -> None:
    twist_cmd = env_cfg.commands["twist"]
    twist_cmd.heading_command = False
    twist_cmd.rel_heading_envs = 0.0
    twist_cmd.rel_standing_envs = 0.0
    twist_cmd.ranges.lin_vel_x = (args.lin_vel_x, args.lin_vel_x)
    twist_cmd.ranges.lin_vel_y = (args.lin_vel_y, args.lin_vel_y)
    twist_cmd.ranges.ang_vel_z = (args.ang_vel_z, args.ang_vel_z)
    twist_cmd.ranges.heading = None


def _force_dead_motor(env_cfg: Any, joint_name: str, scale: float) -> None:
    if "motor_failure" not in env_cfg.events:
        raise RuntimeError("cannot force dead motor because motor_failure event is absent")
    params = env_cfg.events["motor_failure"].params
    params["max_failed_motors"] = 1
    params["forced_joint_name"] = joint_name
    params["forced_failure_type"] = "dead"
    params["forced_scale"] = scale


def _step_env(env: Any, action: Any) -> tuple[Any, Any, Any]:
    result = env.step(action)
    if len(result) == 4:
        obs, reward, dones, _extras = result
    else:
        obs, reward, terminated, truncated, _extras = result
        dones = terminated | truncated
    return obs, reward, dones


def _compact_failure_report(env: Any) -> dict[str, Any]:
    targets = list(getattr(env, "_task029_motor_failure_targets", []))
    records = list(getattr(env, "_task029_motor_failure_last_records", []))
    return {
        "resolved_failure_targets": targets,
        "last_reset_records_sample": records[: min(8, len(records))],
        "reset_log_records": len(getattr(env, "_task029_motor_failure_log", [])),
    }


def _tensor_stats(torch: Any, x: Any) -> dict[str, float]:
    x = x.detach().float()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def _pass_thresholds(result: dict[str, Any]) -> bool:
    return (
        float(result["zero_fall_ratio"]) >= THRESHOLDS["min_zero_fall_ratio"]
        and float(result["lin_vel_error"]["mean"]) <= THRESHOLDS["max_lin_vel_error"]
        and float(result["yaw_vel_error"]["mean"]) <= THRESHOLDS["max_yaw_vel_error"]
        and float(result["gravity_xy"]["mean"]) <= THRESHOLDS["max_gravity_xy"]
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        safe_run_case(args, output_dir, joint_name, index)
        for index, joint_name in enumerate(args.joints)
    ]
    aggregate = {
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "command": list(sys.argv),
        "fixed_command": {
            "lin_vel_x": args.lin_vel_x,
            "lin_vel_y": args.lin_vel_y,
            "ang_vel_z": args.ang_vel_z,
        },
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "thresholds": THRESHOLDS,
        "output_dir": str(output_dir),
        "grid_case_count": len(summaries),
        "complete_grid": len(summaries) == len(args.joints),
        "pass_count": sum(1 for item in summaries if item.get("pass")),
        "failed": [
            {
                "case": item.get("case"),
                "joint_name": item.get("joint_name"),
                "zero_fall_ratio": item.get("zero_fall_ratio"),
                "lin_vel_error_mean": item.get("lin_vel_error", {}).get("mean")
                if isinstance(item.get("lin_vel_error"), dict)
                else None,
                "yaw_vel_error_mean": item.get("yaw_vel_error", {}).get("mean")
                if isinstance(item.get("yaw_vel_error"), dict)
                else None,
                "gravity_xy_mean": item.get("gravity_xy", {}).get("mean")
                if isinstance(item.get("gravity_xy"), dict)
                else None,
                "json_path": item.get("json_path"),
                "error": item.get("error"),
            }
            for item in summaries
            if not item.get("pass")
        ],
        "case_json": {item["case"]: item.get("json_path") for item in summaries},
    }
    aggregate["pass"] = aggregate["complete_grid"] and not aggregate["failed"]
    aggregate_path = output_dir / "task033_failure_grid_eval_aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
