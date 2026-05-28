#!/usr/bin/env python3
"""Render a task028 Unitree MJLab checkpoint headlessly."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mediapy as media
import numpy as np
import torch


def _configure_forward_0p5(env_cfg: Any, clean: bool) -> None:
  twist_cmd = env_cfg.commands["twist"]
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.ranges.lin_vel_x = (0.5, 0.5)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None

  if clean:
    for name in (
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
    ):
      env_cfg.events.pop(name, None)
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.curriculum = {}


def _stats(values: list[float]) -> dict[str, float]:
  arr = np.asarray(values, dtype=np.float32)
  return {
    "mean": float(arr.mean()) if arr.size else 0.0,
    "min": float(arr.min()) if arr.size else 0.0,
    "max": float(arr.max()) if arr.size else 0.0,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="Unitree-G1-Gripper-Flat-Control")
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--prefix", default="task028-render")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--steps", type=int, default=400)
  parser.add_argument("--width", type=int, default=960)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--seed", type=int, default=28005)
  parser.add_argument("--clean", action="store_true")
  args = parser.parse_args()

  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  torch.set_grad_enabled(False)

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  checkpoint = Path(args.checkpoint).expanduser().resolve()
  if not checkpoint.exists():
    raise FileNotFoundError(checkpoint)

  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.seed
  env_cfg.viewer.width = args.width
  env_cfg.viewer.height = args.height
  _configure_forward_0p5(env_cfg, clean=args.clean)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped_env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)
  policy.eval()

  obs, _ = wrapped_env.reset()
  frames: list[np.ndarray] = []
  done_count = 0
  left_gripper_actions: list[float] = []
  right_gripper_actions: list[float] = []
  start = time.time()

  for _ in range(args.steps):
    action = policy(obs)
    if action.shape[1] >= 2:
      left_gripper_actions.append(float(action[0, -2].detach().cpu().item()))
      right_gripper_actions.append(float(action[0, -1].detach().cpu().item()))

    step_result = wrapped_env.step(action)
    if len(step_result) == 4:
      obs, _, dones, _ = step_result
    else:
      obs, _, terminated, truncated, _ = step_result
      dones = terminated | truncated
    done_count += int(dones[0].detach().cpu().item())

    frame = env.render()
    if frame is not None:
      frame0 = frame[0] if frame.ndim == 4 else frame
      if frame0.dtype != np.uint8:
        frame0 = (np.clip(frame0, 0, 1) * 255).astype(np.uint8)
      frames.append(frame0)

  if not frames:
    raise RuntimeError("renderer produced zero frames")

  fps = int(round(1.0 / float(wrapped_env.unwrapped.step_dt)))
  video_path = output_dir / f"{args.prefix}.mp4"
  midframe_path = output_dir / f"{args.prefix}-midframe.png"
  summary_path = output_dir / f"{args.prefix}.json"
  media.write_video(str(video_path), frames, fps=fps)
  media.write_image(str(midframe_path), frames[len(frames) // 2])

  summary = {
    "task": args.task,
    "checkpoint": str(checkpoint),
    "clean": bool(args.clean),
    "device": args.device,
    "steps": args.steps,
    "frames": len(frames),
    "fps": fps,
    "duration_s": len(frames) / fps,
    "width": int(frames[0].shape[1]),
    "height": int(frames[0].shape[0]),
    "done_count": done_count,
    "wall_time_s": time.time() - start,
    "video_path": str(video_path),
    "midframe_path": str(midframe_path),
    "video_bytes": video_path.stat().st_size,
    "midframe_bytes": midframe_path.stat().st_size,
    "gripper_action": {
      "left": _stats(left_gripper_actions),
      "right": _stats(right_gripper_actions),
    },
  }
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  print(json.dumps(summary, indent=2, sort_keys=True))
  wrapped_env.close()


if __name__ == "__main__":
  main()
