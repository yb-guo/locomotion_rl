"""Evaluate Task044 checkpoints without physical inner resets."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.tools import task037_multitrial_eval_checkpoint as metrics_eval
from h200_locomotion_lab.tools.task041_sequence_txl_clean_eval import (
    _install_ipython_display_stub,
    _install_wandb_stub,
    _install_wcwidth_stub,
)
from h200_locomotion_lab.tools.task044_hidden_fault_eval import (
    HIDDEN_FAULT_CONTRACT,
    TASK044_PERSISTENT_HIDDEN_POSE_TIGHT_TASK_ID,
)

TASK044_CONTINUOUS_EXPECTED_RUNNER_CLS = "Task044TrueTxlMemoryK160ContinuousRunner"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Task044 hidden-fault checkpoints in one continuous episode. "
            "This avoids Task037 inner physical resets and reports post-fault "
            "window metrics for memory-required triplet diagnostics."
        )
    )
    parser.add_argument("--task", default=TASK044_PERSISTENT_HIDDEN_POSE_TIGHT_TASK_ID)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--seed", type=int, default=4417001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episode-length-s", type=float, default=30.0)
    parser.add_argument("--lin-vel-x", type=float, default=1.6)
    parser.add_argument("--lin-vel-y", type=float, default=0.0)
    parser.add_argument("--ang-vel-z", type=float, default=0.0)
    parser.add_argument(
        "--dynamic-dead-joint",
        choices=metrics_eval.DEFAULT_JOINTS,
        default="left_knee_joint",
    )
    parser.add_argument("--dynamic-onset-s", type=float, default=2.0)
    parser.add_argument("--dynamic-recovery-s", type=float, default=999.0)
    parser.add_argument("--startup-excluded-s", type=float, default=0.5)
    parser.add_argument("--post-fault-window-s", type=float, default=2.0)
    parser.add_argument("--reset-time-bin-s", type=float, default=0.5)
    parser.add_argument("--min-window-coverage-ratio", type=float, default=0.95)
    parser.add_argument("--max-post-fault-fall-ratio", type=float, default=0.05)
    parser.add_argument("--max-post-fault-lin-vel-error", type=float, default=0.45)
    parser.add_argument("--max-post-fault-yaw-vel-error", type=float, default=1.00)
    parser.add_argument("--max-post-fault-gravity-xy", type=float, default=0.90)
    parser.add_argument("--min-post-fault-root-z", type=float, default=0.35)
    parser.add_argument("--memory-latent-dim", type=int, default=32)
    parser.add_argument("--memory-latent-scale", type=float, default=1.0)
    parser.add_argument("--base-obs-passthrough-scale", type=float, default=1.0)
    parser.add_argument("--adaptation-warmstart-scale", type=float, default=1.0)
    parser.add_argument("--action-dim", type=int, default=31)
    parser.add_argument("--adaptation-hidden-dim", type=int, default=128)
    parser.set_defaults(base_obs_passthrough=True, adaptation_warmstart=True)
    parser.add_argument("--base-obs-passthrough", dest="base_obs_passthrough", action="store_true")
    parser.add_argument("--no-base-obs-passthrough", dest="base_obs_passthrough", action="store_false")
    parser.add_argument("--adaptation-warmstart", dest="adaptation_warmstart", action="store_true")
    parser.add_argument("--no-adaptation-warmstart", dest="adaptation_warmstart", action="store_false")
    parser.add_argument(
        "--memory-ablation-mode",
        choices=("none", "zero_txl_residual", "stateless_txl_memory", "zero_memory_latent"),
        default="none",
    )
    parser.add_argument("--expected-action-dim", type=int, default=31)
    parser.add_argument("--expected-runner-cls", default=TASK044_CONTINUOUS_EXPECTED_RUNNER_CLS)
    parser.add_argument("--expected-actor-model-class", default="Task038TrueTxlMemoryModel")
    return parser.parse_args(argv)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    _install_ipython_display_stub()
    _install_wandb_stub()
    _install_wcwidth_stub()

    import mjlab.tasks as _mjlab_tasks
    import src.tasks as _project_tasks

    del _mjlab_tasks, _project_tasks  # Imports register task packages by side effect.
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.utils.torch import configure_torch_backends

    from h200_locomotion_lab.training.rsl_history_wrapper import (
        Task044TrueTxlMemoryK160ContinuousRunner,
    )

    configure_torch_backends()
    torch.set_grad_enabled(False)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if args.dynamic_recovery_s <= args.dynamic_onset_s:
        raise ValueError("--dynamic-recovery-s must be greater than --dynamic-onset-s")
    if args.startup_excluded_s < 0.0:
        raise ValueError("--startup-excluded-s must be non-negative")
    if args.post_fault_window_s <= 0.0:
        raise ValueError("--post-fault-window-s must be positive")
    if args.reset_time_bin_s <= 0.0:
        raise ValueError("--reset-time-bin-s must be positive")

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = max(
        float(args.episode_length_s),
        float(args.steps) * 0.02 + 1.0,
        args.dynamic_recovery_s + 1.0 if args.dynamic_recovery_s < 900.0 else 0.0,
    )
    metrics_eval._configure_fixed_command(env_cfg, args)
    metrics_eval._configure_dynamic_case(env_cfg, args)

    start = time.time()
    outer_env = None
    try:
        base = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        outer_env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
        train_cfg = asdict(agent_cfg)
        metrics_eval._apply_optional_txl_actor_cfg(args, train_cfg)
        runner = Task044TrueTxlMemoryK160ContinuousRunner(
            outer_env,
            train_cfg,
            device=args.device,
        )
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        actor = metrics_eval._find_actor(runner)
        metrics_eval._apply_optional_memory_ablation(actor, args)
        action_dim = metrics_eval._action_dim(runner.env, base)
        total_action_dim = metrics_eval._total_action_dim(base) or action_dim
        policy = runner.get_inference_policy(device=args.device)
        policy.eval()

        rollout_env = runner.env
        obs, _ = rollout_env.reset()
        base_env = rollout_env.unwrapped
        robot = base_env.scene["robot"]
        dt = float(base_env.step_dt)
        post_start_s = args.dynamic_onset_s + args.startup_excluded_s
        post_end_s = post_start_s + args.post_fault_window_s
        expected_post_samples = _expected_window_samples(
            num_envs=args.num_envs,
            steps=args.steps,
            dt=dt,
            start_s=post_start_s,
            end_s=post_end_s,
        )
        all_stats = metrics_eval._TrialAccumulator(torch, args.num_envs, args.device)
        pre_fault_stats = metrics_eval._TrialAccumulator(torch, args.num_envs, args.device)
        post_fault_stats = metrics_eval._TrialAccumulator(torch, args.num_envs, args.device)
        action_stats = metrics_eval._ActionAccumulator(torch, args.device)
        post_action_stats = metrics_eval._ActionAccumulator(torch, args.device)
        physical_reset_events = 0
        physical_fall_events = 0
        physical_timeout_events = 0
        reset_time_diagnostic = _make_physical_reset_time_diagnostic(
            num_envs=args.num_envs,
            steps=args.steps,
            dt=dt,
            bin_s=args.reset_time_bin_s,
            dynamic_onset_s=args.dynamic_onset_s,
            post_start_s=post_start_s,
            post_end_s=post_end_s,
        )
        first_reset_step = torch.full(
            (args.num_envs,),
            -1,
            device=args.device,
            dtype=torch.long,
        )
        first_fall_step = torch.full(
            (args.num_envs,),
            -1,
            device=args.device,
            dtype=torch.long,
        )

        for step in range(args.steps):
            time_before = step * dt
            action = policy(obs)
            all_mask = torch.ones(args.num_envs, device=args.device, dtype=torch.bool)
            pre_mask = all_mask & (time_before < args.dynamic_onset_s)
            post_mask = all_mask & (time_before >= post_start_s) & (time_before < post_end_s)
            action_stats.add_sample(all_mask, action)
            post_action_stats.add_sample(post_mask, action)

            obs, reward, done, extras = metrics_eval._step_env(rollout_env, action)
            command = base_env.command_manager.get_command("twist")
            lin_vel = robot.data.root_link_lin_vel_b[:, :2]
            yaw_vel = robot.data.root_link_ang_vel_b[:, 2]
            lin_error = torch.linalg.norm(command[:, :2] - lin_vel, dim=-1)
            yaw_error = torch.abs(command[:, 2] - yaw_vel)
            gravity_xy = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
            root_z = robot.data.root_link_pos_w[:, 2]

            done_bool = metrics_eval._bool_tensor(torch, done, args.device)
            timeout_bool = metrics_eval._bool_tensor(
                torch,
                extras.get("time_outs", torch.zeros(args.num_envs, device=args.device)),
                args.device,
            )
            timeout_bool = timeout_bool & done_bool
            fall_bool = done_bool & ~timeout_bool
            physical_reset_events += int(done_bool.sum().detach().cpu().item())
            physical_fall_events += int(fall_bool.sum().detach().cpu().item())
            physical_timeout_events += int(timeout_bool.sum().detach().cpu().item())
            new_reset = done_bool & (first_reset_step < 0)
            new_fall = fall_bool & (first_fall_step < 0)
            first_reset_step[new_reset] = step
            first_fall_step[new_fall] = step
            _add_physical_reset_time_events(
                reset_time_diagnostic,
                step=step,
                reset_count=int(done_bool.sum().detach().cpu().item()),
                fall_count=int(fall_bool.sum().detach().cpu().item()),
                timeout_count=int(timeout_bool.sum().detach().cpu().item()),
            )
            reset_reason = torch.where(
                fall_bool,
                torch.ones(args.num_envs, device=args.device, dtype=torch.long),
                torch.where(
                    timeout_bool,
                    torch.full((args.num_envs,), 2, device=args.device, dtype=torch.long),
                    torch.zeros(args.num_envs, device=args.device, dtype=torch.long),
                ),
            )

            for stats, mask in (
                (all_stats, all_mask),
                (pre_fault_stats, pre_mask),
                (post_fault_stats, post_mask),
            ):
                stats.add_sample(
                    mask,
                    reward=reward,
                    command=command,
                    lin_vel=lin_vel,
                    lin_error=lin_error,
                    yaw_error=yaw_error,
                    gravity_xy=gravity_xy,
                    root_z=root_z,
                )
                stats.add_reset_events(mask & done_bool, reset_reason)

        txl_debug = metrics_eval._txl_debug_snapshot(actor)
        memory_debug_active = _memory_debug_active(txl_debug)
        inner_reset_events_total = _inner_reset_events_total(txl_debug)
        post_fault_window = _continuous_window_json(
            post_fault_stats,
            trial_idx=0,
            num_envs=args.num_envs,
            expected_samples=expected_post_samples,
        )
        post_fault_window["action_stats"] = post_action_stats.to_json()
        post_fault_window["window_start_s"] = post_start_s
        post_fault_window["window_end_s"] = post_end_s
        post_fault_window["window_s"] = args.post_fault_window_s
        post_fault_window["startup_excluded_s"] = args.startup_excluded_s
        post_fault_window["metric_scope"] = "post_fault_window"

        thresholds = _thresholds(args)
        physical_continuity_pass = physical_reset_events == 0 and inner_reset_events_total == 0
        post_fault_window_pass = _post_fault_window_pass(post_fault_window, thresholds)
        pipeline_pass = (
            action_dim == args.expected_action_dim
            and total_action_dim == args.expected_action_dim
            and type(actor).__name__ == args.expected_actor_model_class
            and physical_continuity_pass
        )
        quality_gate_pass = pipeline_pass and post_fault_window_pass
        failure_reasons = _failure_reasons(
            args=args,
            action_dim=action_dim,
            total_action_dim=total_action_dim,
            actor_model_class=type(actor).__name__ if actor is not None else None,
            physical_continuity_pass=physical_continuity_pass,
            post_fault_window=post_fault_window,
            post_fault_window_pass=post_fault_window_pass,
            thresholds=thresholds,
        )
        gpu_name = (
            torch.cuda.get_device_name(torch.device(args.device))
            if str(args.device).startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        return {
            "task044_continuous_fault_eval": True,
            "task044_hidden_fault_contract": dict(HIDDEN_FAULT_CONTRACT),
            "task": args.task,
            "checkpoint": str(checkpoint),
            "command": list(sys.argv),
            "seed": args.seed,
            "device": args.device,
            "gpu_name": gpu_name,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "control_dt_s": dt,
            "episode_length_s": env_cfg.episode_length_s,
            "eval_time_s": args.steps * dt,
            "fixed_command": {
                "lin_vel_x": args.lin_vel_x,
                "lin_vel_y": args.lin_vel_y,
                "ang_vel_z": args.ang_vel_z,
            },
            "dynamic_dead_joint": args.dynamic_dead_joint,
            "dynamic_onset_s": args.dynamic_onset_s,
            "dynamic_recovery_s": args.dynamic_recovery_s,
            "startup_excluded_s": args.startup_excluded_s,
            "runner_cls": TASK044_CONTINUOUS_EXPECTED_RUNNER_CLS,
            "actor_model_class": type(actor).__name__ if actor is not None else None,
            "expected_runner_cls": args.expected_runner_cls,
            "expected_actor_model_class": args.expected_actor_model_class,
            "expected_action_dim": args.expected_action_dim,
            "action_dim": action_dim,
            "total_action_dim": total_action_dim,
            "memory_ablation_mode": args.memory_ablation_mode,
            "memory_ablation_mode_match": (
                txl_debug.get("task042_memory_ablation_mode") == args.memory_ablation_mode
            ),
            "memory_debug_active": memory_debug_active,
            "memory_residual_enabled": txl_debug.get("memory_residual_enabled"),
            "memory_latent_enabled": txl_debug.get("memory_latent_enabled"),
            "stateful_memory_enabled": txl_debug.get("stateful_memory_enabled"),
            "txl_residual_output_norm": txl_debug.get("txl_residual_output_norm_last"),
            "txl_residual_raw_norm": txl_debug.get("txl_residual_raw_norm_last"),
            "adaptation_output_norm": txl_debug.get("adaptation_output_norm_last"),
            "policy_memory_latent_norm": txl_debug.get("policy_memory_latent_norm_last"),
            "physical_continuity_pass": physical_continuity_pass,
            "physical_reset_events": physical_reset_events,
            "physical_fall_events": physical_fall_events,
            "physical_timeout_events": physical_timeout_events,
            "physical_reset_time_diagnostic": _finalize_physical_reset_time_diagnostic(
                reset_time_diagnostic,
                num_envs=args.num_envs,
                dt=dt,
                first_reset_steps=first_reset_step.detach().cpu().tolist(),
                first_fall_steps=first_fall_step.detach().cpu().tolist(),
            ),
            "inner_reset_events_total": inner_reset_events_total,
            "pre_fault_window": _continuous_window_json(
                pre_fault_stats,
                trial_idx=0,
                num_envs=args.num_envs,
                expected_samples=_expected_window_samples(
                    num_envs=args.num_envs,
                    steps=args.steps,
                    dt=dt,
                    start_s=0.0,
                    end_s=args.dynamic_onset_s,
                ),
            ),
            "post_fault_window": post_fault_window,
            "aggregate": _continuous_window_json(
                all_stats,
                trial_idx=0,
                num_envs=args.num_envs,
                expected_samples=args.num_envs * args.steps,
            ),
            "action_stats": action_stats.to_json(),
            "thresholds": thresholds,
            "pipeline_pass": pipeline_pass,
            "quality_gate_pass": quality_gate_pass,
            "pass": quality_gate_pass,
            "quality_claim": False,
            "training_claim": False,
            "eval_claim": False,
            "memory_causality_claim": False,
            "reproduction_claim": False,
            "superiority_claim": False,
            "failure_reasons": failure_reasons,
            "txl_debug": txl_debug,
            "wall_time_s": time.time() - start,
        }
    finally:
        if outer_env is not None:
            outer_env.close()


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    return {
        "task044_continuous_fault_eval": True,
        "task044_hidden_fault_contract": dict(HIDDEN_FAULT_CONTRACT),
        "task": getattr(args, "task", None),
        "checkpoint": getattr(args, "checkpoint", None),
        "memory_ablation_mode": getattr(args, "memory_ablation_mode", None),
        "pipeline_pass": False,
        "quality_gate_pass": False,
        "pass": False,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "memory_causality_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "error": repr(exc),
        "traceback": traceback.format_exc(),
        "failure_reasons": ["task044_continuous_eval_error"],
    }


def write_json_summary(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary["json_path"] = str(output)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_physical_reset_time_diagnostic(
    *,
    num_envs: int,
    steps: int,
    dt: float,
    bin_s: float,
    dynamic_onset_s: float,
    post_start_s: float,
    post_end_s: float,
) -> dict[str, Any]:
    if bin_s <= 0.0:
        raise ValueError("bin_s must be positive")
    eval_time_s = float(steps) * float(dt)
    bin_count = max(math.ceil(eval_time_s / bin_s), 1)
    phases = {
        "pre_fault": _empty_reset_time_bucket(),
        "fault_onset_to_post_window_start": _empty_reset_time_bucket(),
        "post_fault_window": _empty_reset_time_bucket(),
        "post_fault_after_window": _empty_reset_time_bucket(),
    }
    return {
        "schema": "task044_physical_reset_time_diagnostic_v1",
        "diagnostic_only": True,
        "event_scope": "physical_env_done",
        "time_axis": "pre_step_time_s",
        "bin_s": float(bin_s),
        "num_envs": int(num_envs),
        "steps": int(steps),
        "dt_s": float(dt),
        "eval_time_s": eval_time_s,
        "phase_boundaries_s": {
            "dynamic_onset_s": float(dynamic_onset_s),
            "post_fault_window_start_s": float(post_start_s),
            "post_fault_window_end_s": float(post_end_s),
        },
        "segments": phases,
        "bins": [
            {
                "bin_index": idx,
                "start_s": float(idx * bin_s),
                "end_s": float(min((idx + 1) * bin_s, eval_time_s)),
                **_empty_reset_time_bucket(),
            }
            for idx in range(bin_count)
        ],
        "totals": {
            "reset_count": 0,
            "fall_count": 0,
            "timeout_count": 0,
        },
    }


def _add_physical_reset_time_events(
    diagnostic: dict[str, Any],
    *,
    step: int,
    reset_count: int,
    fall_count: int,
    timeout_count: int,
) -> None:
    dt = float(diagnostic["dt_s"])
    bin_s = float(diagnostic["bin_s"])
    time_s = float(step) * dt
    bin_index = min(int(time_s // bin_s), len(diagnostic["bins"]) - 1)
    phase = _reset_time_phase(
        time_s=time_s,
        dynamic_onset_s=diagnostic["phase_boundaries_s"]["dynamic_onset_s"],
        post_start_s=diagnostic["phase_boundaries_s"]["post_fault_window_start_s"],
        post_end_s=diagnostic["phase_boundaries_s"]["post_fault_window_end_s"],
    )
    _add_reset_counts(
        diagnostic["bins"][bin_index],
        step_count=1,
        reset_count=reset_count,
        fall_count=fall_count,
        timeout_count=timeout_count,
    )
    _add_reset_counts(
        diagnostic["segments"][phase],
        step_count=1,
        reset_count=reset_count,
        fall_count=fall_count,
        timeout_count=timeout_count,
    )
    diagnostic["totals"]["reset_count"] += int(reset_count)
    diagnostic["totals"]["fall_count"] += int(fall_count)
    diagnostic["totals"]["timeout_count"] += int(timeout_count)


def _finalize_physical_reset_time_diagnostic(
    diagnostic: dict[str, Any],
    *,
    num_envs: int,
    dt: float,
    first_reset_steps: list[int],
    first_fall_steps: list[int],
) -> dict[str, Any]:
    for bucket in diagnostic["bins"]:
        _finalize_reset_time_bucket(bucket, num_envs=num_envs)
    for bucket in diagnostic["segments"].values():
        _finalize_reset_time_bucket(bucket, num_envs=num_envs)
    diagnostic["first_reset"] = _first_event_json(first_reset_steps, dt=dt, num_envs=num_envs)
    diagnostic["first_fall"] = _first_event_json(first_fall_steps, dt=dt, num_envs=num_envs)
    diagnostic["reset_events_per_env"] = float(
        int(diagnostic["totals"]["reset_count"]) / max(int(num_envs), 1)
    )
    diagnostic["fall_events_per_env"] = float(
        int(diagnostic["totals"]["fall_count"]) / max(int(num_envs), 1)
    )
    return diagnostic


def _empty_reset_time_bucket() -> dict[str, Any]:
    return {
        "step_count": 0,
        "reset_count": 0,
        "fall_count": 0,
        "timeout_count": 0,
    }


def _add_reset_counts(
    bucket: dict[str, Any],
    *,
    step_count: int,
    reset_count: int,
    fall_count: int,
    timeout_count: int,
) -> None:
    bucket["step_count"] += int(step_count)
    bucket["reset_count"] += int(reset_count)
    bucket["fall_count"] += int(fall_count)
    bucket["timeout_count"] += int(timeout_count)


def _finalize_reset_time_bucket(bucket: dict[str, Any], *, num_envs: int) -> None:
    bucket["reset_events_per_env"] = float(int(bucket["reset_count"]) / max(int(num_envs), 1))
    bucket["fall_events_per_env"] = float(int(bucket["fall_count"]) / max(int(num_envs), 1))
    bucket["timeout_events_per_env"] = float(
        int(bucket["timeout_count"]) / max(int(num_envs), 1)
    )


def _first_event_json(steps: list[int], *, dt: float, num_envs: int) -> dict[str, Any]:
    valid_steps = [int(step) for step in steps if int(step) >= 0]
    if not valid_steps:
        return {
            "env_count": 0,
            "env_ratio": 0.0,
            "min_time_s": None,
            "mean_time_s": None,
            "max_time_s": None,
        }
    times = [float(step) * float(dt) for step in valid_steps]
    return {
        "env_count": len(valid_steps),
        "env_ratio": float(len(valid_steps) / max(int(num_envs), 1)),
        "min_time_s": min(times),
        "mean_time_s": float(sum(times) / len(times)),
        "max_time_s": max(times),
    }


def _reset_time_phase(
    *,
    time_s: float,
    dynamic_onset_s: float,
    post_start_s: float,
    post_end_s: float,
) -> str:
    if time_s < dynamic_onset_s:
        return "pre_fault"
    if time_s < post_start_s:
        return "fault_onset_to_post_window_start"
    if time_s < post_end_s:
        return "post_fault_window"
    return "post_fault_after_window"


def _continuous_window_json(
    stats: Any,
    *,
    trial_idx: int,
    num_envs: int,
    expected_samples: int,
) -> dict[str, Any]:
    data = stats.to_json(trial_idx=trial_idx, num_envs=num_envs)
    sample_count = int(data["sample_count"])
    data["expected_sample_count"] = int(expected_samples)
    data["completion_count"] = sample_count
    data["completion_ratio"] = float(sample_count / max(int(expected_samples), 1))
    data["coverage_ratio"] = data["completion_ratio"]
    data["physical_reset_count"] = sum(int(value) for value in data["reset_reason_counts"].values())
    data["fall_ratio"] = float(int(data["fall_count"]) / max(int(num_envs), 1))
    data["zero_fall_ratio"] = float(1.0 - data["fall_ratio"])
    return data


def _expected_window_samples(
    *,
    num_envs: int,
    steps: int,
    dt: float,
    start_s: float,
    end_s: float,
) -> int:
    count = 0
    for step in range(steps):
        time_before = step * dt
        if start_s <= time_before < end_s:
            count += 1
    return int(count * num_envs)


def _thresholds(args: argparse.Namespace) -> dict[str, float]:
    return {
        "min_window_coverage_ratio": args.min_window_coverage_ratio,
        "max_post_fault_fall_ratio": args.max_post_fault_fall_ratio,
        "max_post_fault_lin_vel_error": args.max_post_fault_lin_vel_error,
        "max_post_fault_yaw_vel_error": args.max_post_fault_yaw_vel_error,
        "max_post_fault_gravity_xy": args.max_post_fault_gravity_xy,
        "min_post_fault_root_z": args.min_post_fault_root_z,
    }


def _post_fault_window_pass(window: dict[str, Any], thresholds: dict[str, float]) -> bool:
    root_z_min = window["root_z"]["min"]
    if root_z_min is None:
        return False
    return (
        window["coverage_ratio"] >= thresholds["min_window_coverage_ratio"]
        and window["fall_ratio"] <= thresholds["max_post_fault_fall_ratio"]
        and window["lin_vel_error"]["mean"] <= thresholds["max_post_fault_lin_vel_error"]
        and window["yaw_vel_error"]["mean"] <= thresholds["max_post_fault_yaw_vel_error"]
        and window["gravity_xy"]["max"] <= thresholds["max_post_fault_gravity_xy"]
        and root_z_min >= thresholds["min_post_fault_root_z"]
    )


def _failure_reasons(
    *,
    args: argparse.Namespace,
    action_dim: int | None,
    total_action_dim: int | None,
    actor_model_class: str | None,
    physical_continuity_pass: bool,
    post_fault_window: dict[str, Any],
    post_fault_window_pass: bool,
    thresholds: dict[str, float],
) -> list[str]:
    reasons: list[str] = []
    if args.expected_runner_cls != TASK044_CONTINUOUS_EXPECTED_RUNNER_CLS:
        reasons.append("expected_runner_cls_not_continuous")
    if actor_model_class != args.expected_actor_model_class:
        reasons.append("actor_model_class_mismatch")
    if action_dim != args.expected_action_dim:
        reasons.append("action_dim_mismatch")
    if total_action_dim != args.expected_action_dim:
        reasons.append("total_action_dim_mismatch")
    if not physical_continuity_pass:
        reasons.append("physical_continuity_not_preserved")
    if not post_fault_window_pass:
        reasons.append("post_fault_window_quality_not_passed")
    if post_fault_window["coverage_ratio"] < thresholds["min_window_coverage_ratio"]:
        reasons.append("post_fault_window_coverage_too_low")
    if post_fault_window["lin_vel_error"]["mean"] > thresholds["max_post_fault_lin_vel_error"]:
        reasons.append("post_fault_lin_vel_error_too_high")
    if post_fault_window["gravity_xy"]["max"] > thresholds["max_post_fault_gravity_xy"]:
        reasons.append("post_fault_gravity_xy_too_high")
    root_z_min = post_fault_window["root_z"]["min"]
    if root_z_min is None or root_z_min < thresholds["min_post_fault_root_z"]:
        reasons.append("post_fault_root_z_too_low")
    return list(dict.fromkeys(reasons))


def _inner_reset_events_total(txl_debug: dict[str, Any]) -> int:
    total = 0
    envs = txl_debug.get("envs")
    if isinstance(envs, list):
        for row in envs:
            if isinstance(row, dict):
                total += int(row.get("inner_reset_events") or 0)
    return total


def _memory_debug_active(txl_debug: dict[str, Any]) -> bool:
    if not isinstance(txl_debug, dict):
        return False
    return (
        int(txl_debug.get("total_actor_forward_batches") or 0) > 0
        and int(txl_debug.get("total_actor_forward_samples") or 0) > 0
        and bool(txl_debug.get("last_attended_previous_memory_lengths"))
    )


def main() -> None:
    args = parse_args()
    try:
        summary = run_eval(args)
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        summary = build_failure_summary(args, exc)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
