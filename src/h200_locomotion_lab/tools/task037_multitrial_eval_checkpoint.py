"""Evaluate a checkpoint with Task037 per-trial JSON semantics."""

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


DEFAULT_TASK = "Unitree-G1-Gripper-Flat-Task037-AdaptK4-DeterministicInnerReset-Fast2p0"
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
DYNAMIC_CASES = {
    "none": None,
    "switch": (
        (0.0, 2.0, None, "normal", 1.0),
        (2.0, 4.0, "left_knee_joint", "dead", 0.0),
        (4.0, 5.0, None, "normal", 1.0),
        (5.0, 7.0, "right_hip_yaw_joint", "dead", 0.0),
        (7.0, None, None, "normal", 1.0),
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Task037 checkpoint and emit per-trial multi-trial JSON."
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--seed", type=int, default=3700401)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trial-length-s", type=float, default=2.0)
    parser.add_argument("--lin-vel-x", type=float, default=2.0)
    parser.add_argument("--lin-vel-y", type=float, default=0.0)
    parser.add_argument("--ang-vel-z", type=float, default=0.0)
    parser.add_argument("--dynamic-case", choices=sorted(DYNAMIC_CASES), default="none")
    parser.add_argument("--force-dead-joint", choices=DEFAULT_JOINTS)
    parser.add_argument("--dead-scale", type=float, default=0.0)
    parser.add_argument("--min-final-completion-ratio", type=float, default=0.95)
    parser.add_argument("--max-final-fall-ratio", type=float, default=0.50)
    parser.add_argument("--max-final-lin-vel-error", type=float, default=1.20)
    parser.add_argument("--max-final-yaw-vel-error", type=float, default=1.00)
    parser.add_argument("--max-final-gravity-xy", type=float, default=0.90)
    parser.add_argument("--min-final-root-z", type=float, default=0.35)
    return parser.parse_args(argv)


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
    env_cfg.episode_length_s = args.trial_length_s
    _configure_fixed_command(env_cfg, args)
    if args.dynamic_case != "none":
        _configure_dynamic_case(env_cfg, args.dynamic_case)
    if args.force_dead_joint:
        _force_dead_motor(env_cfg, args.force_dead_joint, args.dead_scale)

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
        num_trials = int(getattr(getattr(rollout_env.env, "config", None), "num_trials", 3))
        current_trial = torch.zeros(args.num_envs, device=args.device, dtype=torch.long)
        trial_stats = [_TrialAccumulator(torch, args.num_envs, args.device) for _ in range(num_trials)]

        for _step in range(args.steps):
            trial_before = current_trial.clone()
            action = policy(obs)
            obs, reward, done, extras = _step_env(rollout_env, action)
            command = base_env.command_manager.get_command("twist")
            lin_vel = robot.data.root_link_lin_vel_b[:, :2]
            yaw_vel = robot.data.root_link_ang_vel_b[:, 2]
            lin_error = torch.linalg.norm(command[:, :2] - lin_vel, dim=-1)
            yaw_error = torch.abs(command[:, 2] - yaw_vel)
            gravity_xy = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
            root_z = robot.data.root_link_pos_w[:, 2]

            trial_done = _bool_tensor(torch, extras.get("trial_done", done), args.device)
            episode_done = _bool_tensor(torch, extras.get("episode_done", done), args.device)
            reset_reason = _long_tensor(
                torch,
                extras.get("reset_reason", torch.zeros(args.num_envs, device=args.device)),
                args.device,
            )

            for trial_idx, accumulator in enumerate(trial_stats):
                mask = trial_before == trial_idx
                accumulator.add_sample(
                    mask,
                    reward=reward,
                    lin_error=lin_error,
                    yaw_error=yaw_error,
                    gravity_xy=gravity_xy,
                    root_z=root_z,
                )
                accumulator.add_reset_events(mask & trial_done, reset_reason)

            current_trial[trial_done & ~episode_done] += 1
            current_trial[episode_done] = 0

        per_trial = [stats.to_json(trial_idx=i, num_envs=args.num_envs) for i, stats in enumerate(trial_stats)]
        final_trial = per_trial[-1]
        aggregate = _aggregate_trials(per_trial, num_envs=args.num_envs)
        thresholds = _thresholds(args)
        final_trial_pass = _final_trial_pass(final_trial, thresholds)
        gpu_name = (
            torch.cuda.get_device_name(torch.device(args.device))
            if str(args.device).startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        result = {
            "task": args.task,
            "checkpoint": str(checkpoint),
            "command": list(sys.argv),
            "seed": args.seed,
            "device": args.device,
            "gpu_name": gpu_name,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "control_dt_s": dt,
            "trial_length_s": args.trial_length_s,
            "eval_time_s": args.steps * dt,
            "fixed_command": {
                "lin_vel_x": args.lin_vel_x,
                "lin_vel_y": args.lin_vel_y,
                "ang_vel_z": args.ang_vel_z,
            },
            "eval_mode": _eval_mode(args),
            "dynamic_case": args.dynamic_case,
            "force_dead_joint": args.force_dead_joint,
            "dead_scale": args.dead_scale,
            "thresholds": thresholds,
            "trial_0": per_trial[0],
            "trial_1": per_trial[1] if len(per_trial) > 1 else None,
            "final_trial": final_trial,
            "aggregate": aggregate,
            "final_trial_pass": final_trial_pass,
            "pass": final_trial_pass,
            "promotion_gate": "final_trial",
            "quality_claim": False,
            "wall_time_s": time.time() - start,
        }
        return result
    finally:
        if outer_env is not None:
            outer_env.close()


class _TrialAccumulator:
    def __init__(self, torch: Any, num_envs: int, device: str) -> None:
        self.torch = torch
        self.num_envs = num_envs
        self.device = device
        self.sample_count = torch.zeros((), device=device, dtype=torch.float32)
        self.reward_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.lin_error_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.yaw_error_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.gravity_xy_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.gravity_xy_max = torch.zeros((), device=device, dtype=torch.float32)
        self.root_z_min = torch.full((), float("inf"), device=device, dtype=torch.float32)
        self.root_z_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.reset_reason_counts: dict[str, int] = {}

    def add_sample(
        self,
        mask: Any,
        *,
        reward: Any,
        lin_error: Any,
        yaw_error: Any,
        gravity_xy: Any,
        root_z: Any,
    ) -> None:
        if not bool(mask.any().item()):
            return
        self.sample_count += mask.float().sum()
        self.reward_sum += reward[mask].float().sum()
        self.lin_error_sum += lin_error[mask].float().sum()
        self.yaw_error_sum += yaw_error[mask].float().sum()
        self.gravity_xy_sum += gravity_xy[mask].float().sum()
        self.gravity_xy_max = self.torch.maximum(self.gravity_xy_max, gravity_xy[mask].float().max())
        self.root_z_min = self.torch.minimum(self.root_z_min, root_z[mask].float().min())
        self.root_z_sum += root_z[mask].float().sum()

    def add_reset_events(self, mask: Any, reset_reason: Any) -> None:
        if not bool(mask.any().item()):
            return
        for reason in reset_reason[mask].detach().cpu().tolist():
            key = str(int(reason))
            self.reset_reason_counts[key] = self.reset_reason_counts.get(key, 0) + 1

    def to_json(self, *, trial_idx: int, num_envs: int) -> dict[str, Any]:
        sample_count = float(self.sample_count.item())
        den = max(sample_count, 1.0)
        trial_done_count = sum(self.reset_reason_counts.values())
        fall_count = self.reset_reason_counts.get("1", 0) + self.reset_reason_counts.get("3", 0)
        timeout_count = self.reset_reason_counts.get("2", 0)
        return {
            "trial_index": trial_idx,
            "sample_count": int(sample_count),
            "completion_count": int(trial_done_count),
            "completion_ratio": float(trial_done_count / max(num_envs, 1)),
            "fall_count": int(fall_count),
            "fall_ratio": float(fall_count / max(trial_done_count, 1)),
            "zero_fall_ratio": float(1.0 - (fall_count / max(trial_done_count, 1))),
            "timeout_count": int(timeout_count),
            "reset_reason_counts": dict(sorted(self.reset_reason_counts.items())),
            "reward_mean": float((self.reward_sum / den).item()),
            "lin_vel_error": {
                "mean": float((self.lin_error_sum / den).item()),
            },
            "yaw_vel_error": {
                "mean": float((self.yaw_error_sum / den).item()),
            },
            "gravity_xy": {
                "mean": float((self.gravity_xy_sum / den).item()),
                "max": float(self.gravity_xy_max.item()),
            },
            "root_z": {
                "mean": float((self.root_z_sum / den).item()),
                "min": _finite_or_none(self.root_z_min.item()),
            },
        }


def _configure_fixed_command(env_cfg: Any, args: argparse.Namespace) -> None:
    twist_cmd = env_cfg.commands["twist"]
    twist_cmd.heading_command = False
    twist_cmd.rel_heading_envs = 0.0
    twist_cmd.rel_standing_envs = 0.0
    twist_cmd.ranges.lin_vel_x = (args.lin_vel_x, args.lin_vel_x)
    twist_cmd.ranges.lin_vel_y = (args.lin_vel_y, args.lin_vel_y)
    twist_cmd.ranges.ang_vel_z = (args.ang_vel_z, args.ang_vel_z)
    twist_cmd.ranges.heading = None


def _configure_dynamic_case(env_cfg: Any, dynamic_case: str) -> None:
    if "dynamic_motor_failure" not in env_cfg.events:
        raise RuntimeError("dynamic_motor_failure event is absent")
    params = env_cfg.events["dynamic_motor_failure"].params
    template = DYNAMIC_CASES[dynamic_case]
    if template is not None and "template" in params:
        params["template"] = template


def _force_dead_motor(env_cfg: Any, joint_name: str, scale: float) -> None:
    if "motor_failure" not in env_cfg.events:
        raise RuntimeError("motor_failure event is absent")
    params = env_cfg.events["motor_failure"].params
    params["max_failed_motors"] = 1
    params["forced_joint_name"] = joint_name
    params["forced_failure_type"] = "dead"
    params["forced_scale"] = scale


def _eval_mode(args: argparse.Namespace) -> str:
    if args.force_dead_joint:
        return "forced_deadgrid"
    if args.dynamic_case != "none":
        return "dynamic_switch"
    return "clean_multitrial"


def _step_env(env: Any, action: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    result = env.step(action)
    if len(result) == 4:
        obs, reward, dones, extras = result
        return obs, reward, dones, extras
    if len(result) == 5:
        obs, reward, terminated, truncated, extras = result
        return obs, reward, terminated | truncated, extras
    if len(result) == 3:
        obs, reward, dones = result
        return obs, reward, dones, {}
    raise ValueError(f"Unsupported env.step result length: {len(result)}")


def _bool_tensor(torch: Any, value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device=device, dtype=torch.bool)
    return torch.as_tensor(value, device=device, dtype=torch.bool)


def _long_tensor(torch: Any, value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device=device, dtype=torch.long)
    return torch.as_tensor(value, device=device, dtype=torch.long)


def _thresholds(args: argparse.Namespace) -> dict[str, float]:
    return {
        "min_final_completion_ratio": args.min_final_completion_ratio,
        "max_final_fall_ratio": args.max_final_fall_ratio,
        "max_final_lin_vel_error": args.max_final_lin_vel_error,
        "max_final_yaw_vel_error": args.max_final_yaw_vel_error,
        "max_final_gravity_xy": args.max_final_gravity_xy,
        "min_final_root_z": args.min_final_root_z,
    }


def _final_trial_pass(final_trial: dict[str, Any], thresholds: dict[str, float]) -> bool:
    root_z_min = final_trial["root_z"]["min"]
    if root_z_min is None:
        return False
    return (
        final_trial["completion_ratio"] >= thresholds["min_final_completion_ratio"]
        and final_trial["fall_ratio"] <= thresholds["max_final_fall_ratio"]
        and final_trial["lin_vel_error"]["mean"] <= thresholds["max_final_lin_vel_error"]
        and final_trial["yaw_vel_error"]["mean"] <= thresholds["max_final_yaw_vel_error"]
        and final_trial["gravity_xy"]["max"] <= thresholds["max_final_gravity_xy"]
        and root_z_min >= thresholds["min_final_root_z"]
    )


def _aggregate_trials(per_trial: list[dict[str, Any]], *, num_envs: int) -> dict[str, Any]:
    completions = sum(int(item["completion_count"]) for item in per_trial)
    falls = sum(int(item["fall_count"]) for item in per_trial)
    samples = sum(int(item["sample_count"]) for item in per_trial)
    if samples <= 0:
        samples = 1
    return {
        "trial_count": len(per_trial),
        "sample_count": sum(int(item["sample_count"]) for item in per_trial),
        "completion_count": completions,
        "completion_ratio_per_trial_mean": float(
            sum(float(item["completion_ratio"]) for item in per_trial) / max(len(per_trial), 1)
        ),
        "fall_count": falls,
        "fall_ratio": float(falls / max(completions, 1)),
        "zero_fall_ratio": float(1.0 - (falls / max(completions, 1))),
        "lin_vel_error_mean": _weighted_mean(per_trial, "lin_vel_error", "mean", samples),
        "yaw_vel_error_mean": _weighted_mean(per_trial, "yaw_vel_error", "mean", samples),
        "gravity_xy_mean": _weighted_mean(per_trial, "gravity_xy", "mean", samples),
        "gravity_xy_max": max(float(item["gravity_xy"]["max"]) for item in per_trial),
        "root_z_min": min(
            float(item["root_z"]["min"])
            for item in per_trial
            if item["root_z"]["min"] is not None
        ),
        "num_envs": num_envs,
        "promotion_note": "auxiliary only; final_trial_pass is the promotion gate",
    }


def _weighted_mean(
    per_trial: list[dict[str, Any]],
    metric: str,
    field: str,
    total_samples: int,
) -> float:
    return float(
        sum(
            float(item[metric][field]) * int(item["sample_count"])
            for item in per_trial
        )
        / max(total_samples, 1)
    )


def _finite_or_none(value: float) -> float | None:
    if value == float("inf") or value == float("-inf"):
        return None
    return float(value)


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
            "final_trial_pass": False,
            "promotion_gate": "final_trial",
            "quality_claim": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    result["json_path"] = str(output_json)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
