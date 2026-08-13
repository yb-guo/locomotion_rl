"""Render a Task041 checkpoint rollout to video for Task047 visual inspection.

This is a visual diagnostic helper. It reuses the Task037/039/041 MJLab runner
and actor construction path, records checkpoint/seed/camera provenance, and
does not promote policy-quality or reproduction claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from h200_locomotion_lab.tools import task039_true_txl_clean_eval
from h200_locomotion_lab.tools.task037_multitrial_eval_checkpoint import (
    _action_dim,
    _apply_optional_memory_ablation,
    _apply_optional_txl_actor_cfg,
    _configure_dynamic_case,
    _configure_fixed_command,
    _find_actor,
    _force_dead_motor,
    _step_env,
    _total_action_dim,
    _txl_debug_snapshot,
)
from h200_locomotion_lab.tools.task038_true_txl_runner_smoke_probe import (
    DEFAULT_EXPECTED_ACTION_DIM,
    DEFAULT_EXPECTED_ACTOR_MODEL_CLASS,
    DEFAULT_EXPECTED_RUNNER_CLS,
)
from h200_locomotion_lab.tools.task040_sequence_txl_ppo_update_smoke import (
    _install_ipython_display_stub,
    _install_wandb_stub,
    _install_wcwidth_stub,
)

TASK042_MEMORY_ABLATION_MODES = (
    "none",
    "zero_txl_residual",
    "stateless_txl_memory",
    "zero_memory_latent",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a Task041 sequence-aware true-TXL checkpoint rollout to MP4. "
            "This is visual diagnostic evidence only."
        )
    )
    parser.add_argument("--task", default=task039_true_txl_clean_eval.TASK039_TRUE_TXL_CLEAN_TASK_ID)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-summary-json")
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--env-idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=4102205)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trial-length-s", type=float, default=5.0)
    parser.add_argument("--lin-vel-x", type=float, default=0.4)
    parser.add_argument("--lin-vel-y", type=float, default=0.0)
    parser.add_argument("--ang-vel-z", type=float, default=0.0)
    parser.add_argument(
        "--dynamic-case",
        choices=sorted(task039_true_txl_clean_eval.task037_multitrial_eval_checkpoint.DYNAMIC_CASES),
        default="none",
    )
    parser.add_argument(
        "--dynamic-dead-joint",
        choices=task039_true_txl_clean_eval.task037_multitrial_eval_checkpoint.DEFAULT_JOINTS,
    )
    parser.add_argument("--dynamic-onset-s", type=float, default=0.5)
    parser.add_argument("--dynamic-recovery-s", type=float, default=1.5)
    parser.add_argument(
        "--force-dead-joint",
        choices=task039_true_txl_clean_eval.task037_multitrial_eval_checkpoint.DEFAULT_JOINTS,
    )
    parser.add_argument("--dead-scale", type=float, default=0.0)
    parser.add_argument("--memory-latent-dim", type=int, default=32)
    parser.add_argument("--memory-latent-scale", type=float, default=1.0)
    parser.add_argument("--base-obs-passthrough-scale", type=float, default=1.0)
    parser.add_argument("--adaptation-warmstart-scale", type=float, default=1.0)
    parser.add_argument("--action-dim", type=int, default=DEFAULT_EXPECTED_ACTION_DIM)
    parser.add_argument("--adaptation-hidden-dim", type=int, default=128)
    parser.set_defaults(base_obs_passthrough=True, adaptation_warmstart=True)
    parser.add_argument("--base-obs-passthrough", dest="base_obs_passthrough", action="store_true")
    parser.add_argument("--no-base-obs-passthrough", dest="base_obs_passthrough", action="store_false")
    parser.add_argument("--adaptation-warmstart", dest="adaptation_warmstart", action="store_true")
    parser.add_argument("--no-adaptation-warmstart", dest="adaptation_warmstart", action="store_false")
    parser.add_argument(
        "--memory-ablation-mode",
        choices=TASK042_MEMORY_ABLATION_MODES,
        default="none",
    )
    parser.add_argument("--expected-action-dim", type=int, default=DEFAULT_EXPECTED_ACTION_DIM)
    parser.add_argument("--expected-runner-cls", default=DEFAULT_EXPECTED_RUNNER_CLS)
    parser.add_argument(
        "--expected-actor-model-class",
        default=DEFAULT_EXPECTED_ACTOR_MODEL_CLASS,
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.5)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-20.0)
    parser.add_argument("--camera-lookat", nargs=3, type=float, default=(0.0, 0.0, 0.8))
    parser.add_argument(
        "--camera-origin",
        choices=("auto", "world", "asset_root", "asset_body"),
        default="asset_root",
    )
    parser.add_argument("--camera-entity-name", default="robot")
    parser.add_argument("--camera-body-name")
    parser.add_argument("--max-extra-envs", type=int, default=0)
    parser.add_argument("--no-shadows", action="store_true")
    parser.add_argument("--no-reflections", action="store_true")
    return parser.parse_args(argv)


def run_video(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    _install_ipython_display_stub()
    _install_wandb_stub()
    _install_wcwidth_stub()

    import imageio
    import mjlab.tasks  # noqa: F401
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    import src.tasks  # noqa: F401

    configure_torch_backends()
    torch.set_grad_enabled(False)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    train_summary = _load_json(args.train_summary_json)
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
    _configure_viewer(env_cfg, args)

    start = time.time()
    output_video = Path(args.output_video).expanduser().resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    outer_env = None
    frames: list[np.ndarray] = []
    try:
        base = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
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
        policy = runner.get_inference_policy(device=args.device)
        policy.eval()
        action_dim = _action_dim(runner.env, base)
        total_action_dim = _total_action_dim(base) or action_dim
        rollout_env = runner.env
        obs, _ = rollout_env.reset()
        base_env = rollout_env.unwrapped
        robot = base_env.scene["robot"]
        dt = float(base_env.step_dt)
        metrics = _MetricAccumulator(torch, args.num_envs, args.device)
        frames.append(_coerce_frame(base.render()))
        reset_reason_counts: dict[str, int] = {}
        last_done = None
        last_extras: dict[str, Any] = {}

        for step in range(args.steps):
            action = policy(obs)
            obs, reward, done, extras = _step_env(rollout_env, action)
            last_done = done
            last_extras = extras
            command = base_env.command_manager.get_command("twist")
            lin_vel = robot.data.root_link_lin_vel_b[:, :2]
            yaw_vel = robot.data.root_link_ang_vel_b[:, 2]
            gravity_xy = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
            root_z = robot.data.root_link_pos_w[:, 2]
            metrics.add(
                reward=reward,
                command=command,
                lin_vel=lin_vel,
                yaw_vel=yaw_vel,
                gravity_xy=gravity_xy,
                root_z=root_z,
            )
            reset_reason = extras.get(
                "reset_reason",
                torch.zeros(args.num_envs, device=args.device, dtype=torch.long),
            )
            trial_done = extras.get("trial_done", done)
            _add_reset_counts(torch, reset_reason_counts, trial_done, reset_reason, args.device)
            if (step + 1) % args.render_every == 0:
                frames.append(_coerce_frame(base.render()))

        imageio.mimsave(output_video, frames, fps=float(args.fps), macro_block_size=1)
        video_bytes = output_video.stat().st_size
        gpu_name = (
            torch.cuda.get_device_name(torch.device(args.device))
            if str(args.device).startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        summary = {
            "task": args.task,
            "task047_task041_eval_video": True,
            "visual_diagnostic_only": True,
            "quality_claim": False,
            "training_claim": False,
            "eval_claim": False,
            "reproduction_claim": False,
            "superiority_claim": False,
            "checkpoint": str(checkpoint),
            "checkpoint_exists": checkpoint.exists(),
            "train_summary_json": args.train_summary_json,
            "train_summary_checkpoint_match": _checkpoint_matches_train_summary(
                checkpoint,
                train_summary,
            ),
            "command": list(sys.argv),
            "seed": args.seed,
            "device": args.device,
            "gpu_name": gpu_name,
            "num_envs": args.num_envs,
            "env_idx": args.env_idx,
            "steps": args.steps,
            "render_every": args.render_every,
            "frame_count": len(frames),
            "fps": float(args.fps),
            "control_dt_s": dt,
            "video_duration_s": len(frames) / float(args.fps),
            "rollout_time_s": args.steps * dt,
            "trial_length_s": args.trial_length_s,
            "fixed_command": {
                "lin_vel_x": args.lin_vel_x,
                "lin_vel_y": args.lin_vel_y,
                "ang_vel_z": args.ang_vel_z,
            },
            "camera": _camera_summary(args),
            "runner_cls": runner_cls.__name__,
            "actor_model_class": type(actor).__name__ if actor is not None else None,
            "expected_runner_cls": args.expected_runner_cls,
            "expected_actor_model_class": args.expected_actor_model_class,
            "action_dim": action_dim,
            "total_action_dim": total_action_dim,
            "expected_action_dim": args.expected_action_dim,
            "reset_reason_counts": dict(sorted(reset_reason_counts.items())),
            "last_done_any": _bool_any(last_done),
            "last_extra_keys": sorted(last_extras),
            "metrics": metrics.to_json(),
            "txl_debug": _small_txl_debug(_txl_debug_snapshot(actor)),
            "output_video": str(output_video),
            "video_bytes": video_bytes,
            "wall_time_s": time.time() - start,
        }
        summary["pass"], summary["failure_reasons"] = evaluate_video_pass(summary)
        return summary
    finally:
        if outer_env is not None:
            outer_env.close()


def evaluate_video_pass(summary: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not summary.get("checkpoint_exists"):
        reasons.append("checkpoint_missing")
    if int(summary.get("frame_count") or 0) <= 0:
        reasons.append("no_frames_rendered")
    if int(summary.get("video_bytes") or 0) <= 0:
        reasons.append("video_empty")
    if summary.get("runner_cls") != summary.get("expected_runner_cls"):
        reasons.append("runner_cls_mismatch")
    if summary.get("actor_model_class") != summary.get("expected_actor_model_class"):
        reasons.append("actor_model_class_mismatch")
    if int(summary.get("action_dim") or -1) != int(summary.get("expected_action_dim") or -2):
        reasons.append("action_dim_mismatch")
    if int(summary.get("total_action_dim") or -1) != int(summary.get("expected_action_dim") or -2):
        reasons.append("total_action_dim_mismatch")
    if summary.get("quality_claim") is not False or summary.get("reproduction_claim") is not False:
        reasons.append("claim_boundary_violation")
    return not reasons, reasons


def write_json_summary(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary["json_path"] = str(output)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser()
    summary = {
        "task": getattr(args, "task", None),
        "task047_task041_eval_video": True,
        "visual_diagnostic_only": True,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "output_video": str(Path(args.output_video).expanduser()),
        "pass": False,
        "failure_reasons": ["video_exception"],
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }
    if isinstance(exc, ValueError):
        summary["failure_reasons"] = ["video_preflight_rejected"]
    return summary


class _MetricAccumulator:
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
        self.root_z_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.root_z_min = torch.full((), float("inf"), device=device, dtype=torch.float32)
        self.command_x_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.lin_vel_x_sum = torch.zeros((), device=device, dtype=torch.float32)

    def add(
        self,
        *,
        reward: Any,
        command: Any,
        lin_vel: Any,
        yaw_vel: Any,
        gravity_xy: Any,
        root_z: Any,
    ) -> None:
        count = int(root_z.shape[0])
        if count <= 0:
            return
        self.sample_count += float(count)
        lin_error = self.torch.linalg.norm(command[:, :2] - lin_vel, dim=-1)
        yaw_error = self.torch.abs(command[:, 2] - yaw_vel)
        self.reward_sum += reward.float().sum()
        self.lin_error_sum += lin_error.float().sum()
        self.yaw_error_sum += yaw_error.float().sum()
        self.gravity_xy_sum += gravity_xy.float().sum()
        self.gravity_xy_max = self.torch.maximum(self.gravity_xy_max, gravity_xy.float().max())
        self.root_z_sum += root_z.float().sum()
        self.root_z_min = self.torch.minimum(self.root_z_min, root_z.float().min())
        self.command_x_sum += command[:, 0].float().sum()
        self.lin_vel_x_sum += lin_vel[:, 0].float().sum()

    def to_json(self) -> dict[str, Any]:
        count = float(self.sample_count.item())
        den = max(count, 1.0)
        root_z_min = float(self.root_z_min.item())
        if root_z_min == float("inf"):
            root_z_min_json = None
        else:
            root_z_min_json = root_z_min
        return {
            "sample_count": int(count),
            "reward_mean": float((self.reward_sum / den).item()),
            "lin_vel_error_mean": float((self.lin_error_sum / den).item()),
            "yaw_vel_error_mean": float((self.yaw_error_sum / den).item()),
            "gravity_xy_mean": float((self.gravity_xy_sum / den).item()),
            "gravity_xy_max": float(self.gravity_xy_max.item()),
            "root_z_mean": float((self.root_z_sum / den).item()),
            "root_z_min": root_z_min_json,
            "command_x_mean": float((self.command_x_sum / den).item()),
            "lin_vel_x_mean": float((self.lin_vel_x_sum / den).item()),
        }


def _validate_args(args: argparse.Namespace) -> None:
    reasons: list[str] = []
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.exists():
        reasons.append("checkpoint_missing")
    if int(args.num_envs) <= 0:
        reasons.append("num_envs_not_positive")
    if int(args.env_idx) < 0 or int(args.env_idx) >= int(args.num_envs):
        reasons.append("env_idx_out_of_range")
    if int(args.steps) <= 0:
        reasons.append("steps_not_positive")
    if int(args.render_every) <= 0:
        reasons.append("render_every_not_positive")
    if float(args.fps) <= 0:
        reasons.append("fps_not_positive")
    if int(args.width) <= 0 or int(args.height) <= 0:
        reasons.append("invalid_resolution")
    if reasons:
        raise ValueError(", ".join(reasons))


def _configure_viewer(env_cfg: Any, args: argparse.Namespace) -> None:
    viewer = env_cfg.viewer
    viewer.width = int(args.width)
    viewer.height = int(args.height)
    viewer.env_idx = int(args.env_idx)
    viewer.max_extra_envs = int(args.max_extra_envs)
    viewer.distance = float(args.camera_distance)
    viewer.azimuth = float(args.camera_azimuth)
    viewer.elevation = float(args.camera_elevation)
    viewer.lookat = tuple(float(value) for value in args.camera_lookat)
    viewer.entity_name = args.camera_entity_name
    viewer.body_name = args.camera_body_name
    viewer.enable_shadows = not bool(args.no_shadows)
    viewer.enable_reflections = not bool(args.no_reflections)
    origin_type = type(viewer).OriginType
    viewer.origin_type = {
        "auto": origin_type.AUTO,
        "world": origin_type.WORLD,
        "asset_root": origin_type.ASSET_ROOT,
        "asset_body": origin_type.ASSET_BODY,
    }[args.camera_origin]


def _camera_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "width": args.width,
        "height": args.height,
        "origin": args.camera_origin,
        "env_idx": args.env_idx,
        "entity_name": args.camera_entity_name,
        "body_name": args.camera_body_name,
        "lookat": [float(value) for value in args.camera_lookat],
        "distance": args.camera_distance,
        "azimuth": args.camera_azimuth,
        "elevation": args.camera_elevation,
        "max_extra_envs": args.max_extra_envs,
        "shadows": not args.no_shadows,
        "reflections": not args.no_reflections,
    }


def _coerce_frame(frame: Any) -> np.ndarray:
    if frame is None:
        raise RuntimeError("render returned None")
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise RuntimeError(f"render frame must be HxWx3/4, got {array.shape}")
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0.0, 1.0)
            array = (array * 255.0).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _add_reset_counts(
    torch: Any,
    counts: dict[str, int],
    trial_done: Any,
    reset_reason: Any,
    device: str,
) -> None:
    done = _to_bool_tensor(torch, trial_done, device)
    if not bool(done.any().item()):
        return
    reason = _to_long_tensor(torch, reset_reason, device)
    for value in reason[done].detach().cpu().tolist():
        key = str(int(value))
        counts[key] = counts.get(key, 0) + 1


def _to_bool_tensor(torch: Any, value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device=device, dtype=torch.bool)
    return torch.as_tensor(value, device=device, dtype=torch.bool)


def _to_long_tensor(torch: Any, value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device=device, dtype=torch.long)
    return torch.as_tensor(value, device=device, dtype=torch.long)


def _bool_any(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        return bool(value.any().item())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _checkpoint_matches_train_summary(
    checkpoint: Path,
    train_summary: Mapping[str, Any] | None,
) -> bool:
    if not train_summary:
        return False
    return checkpoint.resolve() == Path(str(train_summary.get("checkpoint", ""))).expanduser().resolve()


def _small_txl_debug(debug: Mapping[str, Any]) -> dict[str, Any]:
    envs = debug.get("envs")
    env0 = envs[0] if isinstance(envs, list) and envs else None
    return {
        "stateful_memory_enabled": debug.get("stateful_memory_enabled"),
        "memory_residual_enabled": debug.get("memory_residual_enabled"),
        "memory_latent_enabled": debug.get("memory_latent_enabled"),
        "task042_memory_ablation_mode": debug.get("task042_memory_ablation_mode"),
        "total_actor_forward_batches": debug.get("total_actor_forward_batches"),
        "env_cache_stateful_forward_batches": debug.get("env_cache_stateful_forward_batches"),
        "stateless_fallback_forward_batches": debug.get("stateless_fallback_forward_batches"),
        "env0": env0,
    }


def main() -> None:
    args = parse_args()
    try:
        summary = run_video(args)
    except Exception as exc:  # noqa: BLE001 - CLI must emit JSON failure provenance.
        summary = build_failure_summary(args, exc)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
