"""Evaluate a checkpoint with Task037 per-trial JSON semantics."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping


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


def _install_ipython_display_stub() -> None:
    """Provide the tiny optional display API that mediapy imports on H200."""

    if "IPython.display" in sys.modules:
        return
    ipython_module = sys.modules.get("IPython") or types.ModuleType("IPython")
    display_module = types.ModuleType("IPython.display")

    class _DisplayObject:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def _repr_html_(self) -> str:
            return ""

    def _display(*args: Any, **kwargs: Any) -> None:
        return None

    display_module.display = _display
    display_module.HTML = _DisplayObject
    display_module.Image = _DisplayObject
    display_module.Video = _DisplayObject
    display_module.clear_output = _display
    ipython_module.display = display_module
    sys.modules.setdefault("IPython", ipython_module)
    sys.modules["IPython.display"] = display_module


def _install_wandb_stub() -> None:
    """Avoid importing optional W&B transitive deps during task registration."""

    if "wandb" in sys.modules:
        return
    wandb_module = types.ModuleType("wandb")

    class _Api:
        pass

    def _noop(*args: Any, **kwargs: Any) -> None:
        return None

    wandb_module.run = None
    wandb_module.Api = _Api
    wandb_module.init = _noop
    wandb_module.log = _noop
    wandb_module.finish = _noop
    wandb_module.save = _noop
    wandb_module.login = _noop
    sys.modules["wandb"] = wandb_module


def _install_wcwidth_stub() -> None:
    """Provide width helpers for prettytable in the slim H200 conda env."""

    if "wcwidth" in sys.modules:
        return
    wcwidth_module = types.ModuleType("wcwidth")

    def _wcwidth(char: str) -> int:
        return 0 if not char else 1

    def _wcswidth(text: str) -> int:
        return len(str(text))

    wcwidth_module.wcwidth = _wcwidth
    wcwidth_module.wcswidth = _wcswidth
    sys.modules["wcwidth"] = wcwidth_module


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
    parser.add_argument("--dynamic-dead-joint", choices=DEFAULT_JOINTS)
    parser.add_argument("--dynamic-onset-s", type=float, default=0.5)
    parser.add_argument("--dynamic-recovery-s", type=float, default=1.5)
    parser.add_argument("--force-dead-joint", choices=DEFAULT_JOINTS)
    parser.add_argument("--dead-scale", type=float, default=0.0)
    parser.add_argument(
        "--final-window-s",
        type=float,
        default=0.0,
        help=(
            "Optional first-N-seconds diagnostic window inside the final trial. "
            "When positive, the JSON includes final_trial_window in addition "
            "to full final_trial metrics."
        ),
    )
    parser.add_argument("--min-final-completion-ratio", type=float, default=0.95)
    parser.add_argument("--max-final-fall-ratio", type=float, default=0.50)
    parser.add_argument("--max-final-lin-vel-error", type=float, default=1.20)
    parser.add_argument("--max-final-yaw-vel-error", type=float, default=1.00)
    parser.add_argument("--max-final-gravity-xy", type=float, default=0.90)
    parser.add_argument("--min-final-root-z", type=float, default=0.35)
    parser.add_argument("--memory-latent-dim", type=int)
    parser.add_argument("--memory-latent-scale", type=float)
    parser.add_argument("--base-obs-passthrough-scale", type=float)
    parser.add_argument("--adaptation-warmstart-scale", type=float)
    parser.add_argument("--action-dim", type=int)
    parser.add_argument("--adaptation-hidden-dim", type=int)
    parser.set_defaults(base_obs_passthrough=None, adaptation_warmstart=None)
    parser.add_argument("--base-obs-passthrough", dest="base_obs_passthrough", action="store_true")
    parser.add_argument("--no-base-obs-passthrough", dest="base_obs_passthrough", action="store_false")
    parser.add_argument("--adaptation-warmstart", dest="adaptation_warmstart", action="store_true")
    parser.add_argument("--no-adaptation-warmstart", dest="adaptation_warmstart", action="store_false")
    return parser.parse_args(argv)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    _install_ipython_display_stub()
    _install_wandb_stub()
    _install_wcwidth_stub()

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
    if args.dynamic_case != "none" or args.dynamic_dead_joint:
        _configure_dynamic_case(env_cfg, args)
    if args.force_dead_joint:
        _force_dead_motor(env_cfg, args.force_dead_joint, args.dead_scale)

    start = time.time()
    outer_env = None
    try:
        base = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        outer_env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        train_cfg = asdict(agent_cfg)
        _apply_optional_txl_actor_cfg(args, train_cfg)
        runner = runner_cls(outer_env, train_cfg, device=args.device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        actor = _find_actor(runner)
        _apply_optional_memory_ablation(actor, args)
        action_dim = _action_dim(runner.env, base)
        total_action_dim = _total_action_dim(base) or action_dim
        policy = runner.get_inference_policy(device=args.device)
        policy.eval()

        rollout_env = runner.env
        obs, _ = rollout_env.reset()
        base_env = rollout_env.unwrapped
        robot = base_env.scene["robot"]
        dt = float(base_env.step_dt)
        num_trials = int(getattr(getattr(rollout_env.env, "config", None), "num_trials", 3))
        final_trial_idx = max(num_trials - 1, 0)
        final_window_steps = _final_window_steps(args.final_window_s, dt)
        trial_length_steps = max(1, int(math.ceil(args.trial_length_s / dt)))
        current_trial = torch.zeros(args.num_envs, device=args.device, dtype=torch.long)
        trial_step_index = torch.zeros(args.num_envs, device=args.device, dtype=torch.long)
        trial_stats = [_TrialAccumulator(torch, args.num_envs, args.device) for _ in range(num_trials)]
        final_window_stats = (
            _TrialAccumulator(torch, args.num_envs, args.device)
            if final_window_steps is not None
            else None
        )
        final_tail_window_stats = (
            _TrialAccumulator(torch, args.num_envs, args.device)
            if final_window_steps is not None
            else None
        )
        trial_action_stats = [
            _ActionAccumulator(torch, args.device) for _ in range(num_trials)
        ]
        final_window_action_stats = (
            _ActionAccumulator(torch, args.device) if final_window_steps is not None else None
        )
        final_tail_window_action_stats = (
            _ActionAccumulator(torch, args.device) if final_window_steps is not None else None
        )

        for _step in range(args.steps):
            trial_before = current_trial.clone()
            trial_step_before = trial_step_index.clone()
            action = policy(obs)
            for trial_idx, accumulator in enumerate(trial_action_stats):
                accumulator.add_sample(trial_before == trial_idx, action)
            if final_window_action_stats is not None:
                window_mask = (trial_before == final_trial_idx) & (
                    trial_step_before < final_window_steps
                )
                final_window_action_stats.add_sample(window_mask, action)
            if final_tail_window_action_stats is not None:
                tail_window_mask = (trial_before == final_trial_idx) & (
                    trial_step_before >= max(0, trial_length_steps - final_window_steps)
                )
                final_tail_window_action_stats.add_sample(tail_window_mask, action)
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
                    command=command,
                    lin_vel=lin_vel,
                    lin_error=lin_error,
                    yaw_error=yaw_error,
                    gravity_xy=gravity_xy,
                    root_z=root_z,
                )
                accumulator.add_reset_events(mask & trial_done, reset_reason)

            if final_window_stats is not None:
                window_mask = (trial_before == final_trial_idx) & (
                    trial_step_before < final_window_steps
                )
                final_window_stats.add_sample(
                    window_mask,
                    reward=reward,
                    command=command,
                    lin_vel=lin_vel,
                    lin_error=lin_error,
                    yaw_error=yaw_error,
                    gravity_xy=gravity_xy,
                    root_z=root_z,
                )
                final_window_stats.add_reset_events(window_mask & trial_done, reset_reason)
            if final_tail_window_stats is not None:
                tail_window_mask = (trial_before == final_trial_idx) & (
                    trial_step_before >= max(0, trial_length_steps - final_window_steps)
                )
                final_tail_window_stats.add_sample(
                    tail_window_mask,
                    reward=reward,
                    command=command,
                    lin_vel=lin_vel,
                    lin_error=lin_error,
                    yaw_error=yaw_error,
                    gravity_xy=gravity_xy,
                    root_z=root_z,
                )
                final_tail_window_stats.add_reset_events(
                    tail_window_mask & trial_done,
                    reset_reason,
                )

            trial_step_index += 1
            trial_step_index[trial_done] = 0
            current_trial[trial_done & ~episode_done] += 1
            current_trial[episode_done] = 0

        per_trial = []
        for i, stats in enumerate(trial_stats):
            trial_json = stats.to_json(trial_idx=i, num_envs=args.num_envs)
            trial_json["action_stats"] = trial_action_stats[i].to_json()
            per_trial.append(trial_json)
        final_trial = per_trial[-1]
        final_trial_window = (
            {
                **final_window_stats.to_json(trial_idx=final_trial_idx, num_envs=args.num_envs),
                "action_stats": final_window_action_stats.to_json()
                if final_window_action_stats is not None
                else None,
                "window_s": args.final_window_s,
                "window_steps": final_window_steps,
                "metric_scope": "first_final_trial_window",
            }
            if final_window_stats is not None
            else None
        )
        final_trial_tail_window = (
            {
                **final_tail_window_stats.to_json(
                    trial_idx=final_trial_idx,
                    num_envs=args.num_envs,
                ),
                "action_stats": final_tail_window_action_stats.to_json()
                if final_tail_window_action_stats is not None
                else None,
                "window_s": args.final_window_s,
                "window_steps": final_window_steps,
                "metric_scope": "last_final_trial_window",
            }
            if final_tail_window_stats is not None
            else None
        )
        aggregate = _aggregate_trials(per_trial, num_envs=args.num_envs)
        thresholds = _thresholds(args)
        final_trial_pass = _final_trial_pass(final_trial, thresholds)
        txl_debug = _txl_debug_snapshot(actor)
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
            "runner_cls": runner_cls.__name__,
            "actor_model_class": type(actor).__name__ if actor is not None else None,
            "action_dim": action_dim,
            "total_action_dim": total_action_dim,
            "eval_mode": _eval_mode(args),
            "dynamic_case": args.dynamic_case,
            "dynamic_dead_joint": args.dynamic_dead_joint,
            "dynamic_onset_s": args.dynamic_onset_s,
            "dynamic_recovery_s": args.dynamic_recovery_s,
            "force_dead_joint": args.force_dead_joint,
            "dead_scale": args.dead_scale,
            "thresholds": thresholds,
            "trial_0": per_trial[0],
            "trial_1": per_trial[1] if len(per_trial) > 1 else None,
            "final_trial": final_trial,
            "final_trial_window": final_trial_window,
            "final_trial_tail_window": final_trial_tail_window,
            "aggregate": aggregate,
            "final_trial_pass": final_trial_pass,
            "pass": final_trial_pass,
            "promotion_gate": "final_trial",
            "quality_claim": False,
            "txl_debug": txl_debug,
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
        self.command_x_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.command_y_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.lin_vel_x_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.lin_vel_y_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.lin_vel_error_x_abs_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.lin_vel_error_y_abs_sum = torch.zeros((), device=device, dtype=torch.float32)
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
        command: Any,
        lin_vel: Any,
        lin_error: Any,
        yaw_error: Any,
        gravity_xy: Any,
        root_z: Any,
    ) -> None:
        if not bool(mask.any().item()):
            return
        self.sample_count += mask.float().sum()
        self.reward_sum += reward[mask].float().sum()
        selected_command = command[mask].float()
        selected_lin_vel = lin_vel[mask].float()
        self.command_x_sum += selected_command[:, 0].sum()
        self.command_y_sum += selected_command[:, 1].sum()
        self.lin_vel_x_sum += selected_lin_vel[:, 0].sum()
        self.lin_vel_y_sum += selected_lin_vel[:, 1].sum()
        self.lin_vel_error_x_abs_sum += self.torch.abs(
            selected_command[:, 0] - selected_lin_vel[:, 0]
        ).sum()
        self.lin_vel_error_y_abs_sum += self.torch.abs(
            selected_command[:, 1] - selected_lin_vel[:, 1]
        ).sum()
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
            "completion_ratio": float(min(trial_done_count / max(num_envs, 1), 1.0)),
            "fall_count": int(fall_count),
            "fall_ratio": float(fall_count / max(trial_done_count, 1)),
            "zero_fall_ratio": float(1.0 - (fall_count / max(trial_done_count, 1))),
            "timeout_count": int(timeout_count),
            "reset_reason_counts": dict(sorted(self.reset_reason_counts.items())),
            "reward_mean": float((self.reward_sum / den).item()),
            "lin_vel_command": {
                "mean_x": float((self.command_x_sum / den).item()),
                "mean_y": float((self.command_y_sum / den).item()),
            },
            "lin_vel_actual": {
                "mean_x": float((self.lin_vel_x_sum / den).item()),
                "mean_y": float((self.lin_vel_y_sum / den).item()),
            },
            "lin_vel_error_components": {
                "mean_abs_x": float((self.lin_vel_error_x_abs_sum / den).item()),
                "mean_abs_y": float((self.lin_vel_error_y_abs_sum / den).item()),
            },
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


class _ActionAccumulator:
    def __init__(self, torch: Any, device: str) -> None:
        self.torch = torch
        self.device = device
        self.sample_count = torch.zeros((), device=device, dtype=torch.float32)
        self.l2_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.max_abs = torch.zeros((), device=device, dtype=torch.float32)
        self.abs_sum = None
        self.abs_max = None

    def add_sample(self, mask: Any, action: Any) -> None:
        if action is None:
            return
        action_2d = action.reshape(int(action.shape[0]), -1)
        selected = action_2d[mask]
        count = int(selected.shape[0])
        if count <= 0:
            return
        abs_action = self.torch.abs(selected)
        if self.abs_sum is None:
            action_dim = int(abs_action.shape[-1])
            self.abs_sum = self.torch.zeros(action_dim, device=self.device, dtype=abs_action.dtype)
            self.abs_max = self.torch.zeros(action_dim, device=self.device, dtype=abs_action.dtype)
        self.sample_count += float(count)
        self.l2_sum += self.torch.sum(self.torch.linalg.norm(selected, dim=-1))
        self.max_abs = self.torch.maximum(self.max_abs, self.torch.max(abs_action))
        self.abs_sum += self.torch.sum(abs_action, dim=0)
        self.abs_max = self.torch.maximum(self.abs_max, self.torch.max(abs_action, dim=0).values)

    def to_json(self, *, top_k: int = 8) -> dict[str, Any]:
        count = int(self.sample_count.item())
        if count <= 0 or self.abs_sum is None or self.abs_max is None:
            return {
                "sample_count": 0,
                "action_dim": None,
                "mean_l2": None,
                "max_abs": None,
                "mean_abs_by_dim": [],
                "max_abs_by_dim": [],
                "top_abs_dims": [],
            }

        mean_abs = (self.abs_sum / float(count)).detach().cpu()
        max_abs_by_dim = self.abs_max.detach().cpu()
        action_dim = int(mean_abs.shape[0])
        top_count = min(top_k, action_dim)
        top_values, top_indices = self.torch.topk(mean_abs, k=top_count)
        return {
            "sample_count": count,
            "action_dim": action_dim,
            "mean_l2": float((self.l2_sum / float(count)).item()),
            "max_abs": float(self.max_abs.item()),
            "mean_abs_by_dim": [float(value) for value in mean_abs.tolist()],
            "max_abs_by_dim": [float(value) for value in max_abs_by_dim.tolist()],
            "top_abs_dims": [
                {"dim": int(index), "mean_abs": float(value)}
                for index, value in zip(top_indices.tolist(), top_values.tolist())
            ],
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


def _configure_dynamic_case(env_cfg: Any, args: argparse.Namespace) -> None:
    if "dynamic_motor_failure" not in env_cfg.events:
        raise RuntimeError("dynamic_motor_failure event is absent")
    params = env_cfg.events["dynamic_motor_failure"].params
    template = _dynamic_template(args)
    if template is not None and "template" in params:
        params["template"] = template


def _dynamic_template(args: argparse.Namespace) -> Any:
    if args.dynamic_dead_joint:
        if args.dynamic_recovery_s <= args.dynamic_onset_s:
            raise ValueError("--dynamic-recovery-s must be greater than --dynamic-onset-s")
        return (
            (0.0, args.dynamic_onset_s, None, "normal", 1.0),
            (args.dynamic_onset_s, args.dynamic_recovery_s, args.dynamic_dead_joint, "dead", 0.0),
            (args.dynamic_recovery_s, None, None, "normal", 1.0),
        )
    return DYNAMIC_CASES[args.dynamic_case]


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
    if args.dynamic_dead_joint:
        return "dynamic_single_onset"
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


def _final_window_steps(window_s: float, dt: float) -> int | None:
    if window_s <= 0.0:
        return None
    if dt <= 0.0:
        raise ValueError("step dt must be positive when --final-window-s is used")
    return max(1, int(math.ceil(window_s / dt)))


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


def _find_actor(runner: Any) -> Any | None:
    for path in (
        ("alg", "actor_critic", "actor"),
        ("alg", "actor_critic"),
        ("actor_critic", "actor"),
        ("actor_critic",),
        ("policy",),
        ("alg", "actor"),
    ):
        obj = runner
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _apply_optional_txl_actor_cfg(args: argparse.Namespace, train_cfg: dict[str, Any]) -> None:
    actor = train_cfg.get("actor")
    if not isinstance(actor, dict):
        return
    for attr in (
        "memory_latent_dim",
        "action_dim",
        "base_obs_passthrough",
        "adaptation_warmstart",
        "adaptation_hidden_dim",
        "memory_latent_scale",
        "base_obs_passthrough_scale",
        "adaptation_warmstart_scale",
    ):
        if hasattr(args, attr):
            value = getattr(args, attr)
            if value is not None:
                actor[attr] = value


def _txl_debug_snapshot(actor: Any | None) -> dict[str, Any]:
    if actor is None:
        return {}
    snapshot = getattr(actor, "txl_debug_snapshot", None)
    if not callable(snapshot):
        return {}
    data = snapshot()
    return dict(data) if isinstance(data, Mapping) else {}


def _apply_optional_memory_ablation(actor: Any | None, args: argparse.Namespace) -> None:
    if actor is None or not hasattr(args, "memory_ablation_mode"):
        return
    setter = getattr(actor, "task042_set_memory_ablation_mode", None)
    if callable(setter):
        setter(getattr(args, "memory_ablation_mode"))


def _action_dim(env: Any, base: Any) -> int | None:
    for source in (env, base):
        for name in ("num_actions", "action_dim", "total_action_dim"):
            value = getattr(source, name, None)
            if value is not None:
                return int(value)
    return None


def _total_action_dim(base: Any) -> int | None:
    action_manager = getattr(base, "action_manager", None)
    for source in (base, action_manager):
        if source is None:
            continue
        for name in ("total_action_dim", "action_dim", "num_actions"):
            value = getattr(source, name, None)
            if value is not None:
                return int(value)
    return None


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
