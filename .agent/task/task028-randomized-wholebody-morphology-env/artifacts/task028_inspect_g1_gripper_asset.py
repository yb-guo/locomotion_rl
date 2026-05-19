#!/usr/bin/env python3
"""Inspect and smoke-step the task028 Unitree G1 gripper environment."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch


def tensor_summary(x: torch.Tensor) -> dict[str, Any]:
  return {
    "shape": list(x.shape),
    "finite": bool(torch.isfinite(x).all().item()),
    "min": float(x.min().item()) if x.numel() else 0.0,
    "max": float(x.max().item()) if x.numel() else 0.0,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="Unitree-G1-Gripper-Flat")
  parser.add_argument("--output", required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--steps", type=int, default=10)
  parser.add_argument("--seed", type=int, default=28002)
  args = parser.parse_args()

  os.environ.setdefault("MUJOCO_GL", "egl")

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.entity.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import list_tasks, load_env_cfg
  from mjlab.utils.torch import configure_torch_backends
  from src.assets.robots.unitree_g1_gripper import get_g1_gripper_robot_cfg

  configure_torch_backends()
  torch.set_grad_enabled(False)

  tasks = list(list_tasks())
  if args.task not in tasks:
    raise RuntimeError(f"{args.task} not registered")

  robot_entity = Entity(get_g1_gripper_robot_cfg())
  model = robot_entity.spec.compile()
  joint_names = [
    model.joint(i).name
    for i in range(model.njnt)
  ]

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  for name in (
    "push_robot",
    "foot_friction",
    "encoder_bias",
    "base_com",
    "randomize_terrain",
  ):
    env_cfg.events.pop(name, None)

  start = time.time()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  obs, _ = env.reset()
  reset_obs = {
    name: tensor_summary(value)
    for name, value in obs.items()
    if isinstance(value, torch.Tensor)
  }

  action_terms = {}
  for name in env.action_manager.active_terms:
    term = env.action_manager.get_term(name)
    action_terms[name] = {
      "dim": int(term.action_dim),
      "target_names": list(term.target_names),
      "target_ids": [int(x) for x in term.target_ids.detach().cpu().tolist()],
    }

  action = torch.zeros(
    args.num_envs,
    env.action_manager.total_action_dim,
    device=args.device,
  )

  done_count = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
  finite_obs = True
  for _ in range(args.steps):
    step_result = env.step(action)
    if len(step_result) == 4:
      obs, _, dones, _ = step_result
    else:
      obs, _, terminated, truncated, _ = step_result
      dones = terminated | truncated
    done_count += dones.to(dtype=torch.long)
    finite_obs = finite_obs and all(
      bool(torch.isfinite(value).all().item())
      for value in obs.values()
      if isinstance(value, torch.Tensor)
    )

  robot = env.scene["robot"]
  summary = {
    "task": args.task,
    "registered": True,
    "device": args.device,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "wall_time_s": time.time() - start,
    "model": {
      "nq": int(model.nq),
      "nv": int(model.nv),
      "nu": int(model.nu),
      "njnt": int(model.njnt),
      "joint_names": joint_names,
      "gripper_joints": [name for name in joint_names if "gripper" in name],
    },
    "action_manager": {
      "active_terms": list(env.action_manager.active_terms),
      "total_action_dim": int(env.action_manager.total_action_dim),
      "terms": action_terms,
    },
    "reset_observation": reset_obs,
    "post_step": {
      "finite_obs": bool(finite_obs),
      "done_count": [int(x) for x in done_count.detach().cpu().tolist()],
      "root_z": tensor_summary(robot.data.root_link_pos_w[:, 2]),
      "joint_pos": tensor_summary(robot.data.joint_pos),
      "joint_vel": tensor_summary(robot.data.joint_vel),
    },
    "pass": (
      int(model.nu) == 31
      and int(env.action_manager.total_action_dim) == 31
      and list(env.action_manager.active_terms) == [
        "body_joint_pos",
        "gripper_joint_pos",
      ]
      and action_terms.get("body_joint_pos", {}).get("dim") == 29
      and action_terms.get("gripper_joint_pos", {}).get("dim") == 2
      and bool(finite_obs)
    ),
  }

  env.close()

  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
