#!/usr/bin/env python3
"""Patch H200 MJLab with Task032 focused dead-grid curriculum envs."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path("/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab")


FUNCTIONS = r'''

def _add_task032_focused_curriculum_stage(
  cfg: ManagerBasedRlEnvCfg,
  *,
  stage: str,
) -> None:
  """Task032 focused persistent motor-failure curriculum for MLP ceiling test."""
  _add_motor_failure_stage(cfg)
  params = cfg.events["motor_failure"].params
  params["max_failed_motors"] = 1
  params["single_dead_joint_names"] = (
    "left_hip_yaw_joint",
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_knee_joint",
    "left_hip_pitch_joint",
  )
  _add_task029_phase_randomization(cfg)
  if stage == "weak":
    params["single_dead_probability"] = 0.75
    params["dead_probability"] = 0.20
    params["weak_scale_range"] = (0.55, 0.85)
    params["dead_scale_range"] = (0.20, 0.40)
  elif stage == "mixed":
    params["single_dead_probability"] = 0.90
    params["dead_probability"] = 0.50
    params["weak_scale_range"] = (0.40, 0.75)
    params["dead_scale_range"] = (0.05, 0.25)
  elif stage == "hard":
    params["single_dead_probability"] = 0.95
    params["dead_probability"] = 0.80
    params["weak_scale_range"] = (0.30, 0.70)
    params["dead_scale_range"] = (0.0, 0.10)
  else:
    raise ValueError(f"unknown task032 stage: {stage}")


def unitree_g1_gripper_flat_task032_weak_focused_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Task032 weak-focused curriculum env over 0.4..2.0 m/s."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _add_task032_focused_curriculum_stage(cfg, stage="weak")
  _configure_task029_forward_speed_command(cfg, min_speed=0.4, max_speed=2.0)
  return cfg


def unitree_g1_gripper_flat_task032_mixed_focused_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Task032 mixed-focused curriculum env over 0.4..2.0 m/s."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _add_task032_focused_curriculum_stage(cfg, stage="mixed")
  _configure_task029_forward_speed_command(cfg, min_speed=0.4, max_speed=2.0)
  return cfg


def unitree_g1_gripper_flat_task032_hard_focused_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Task032 hard-focused curriculum env over 0.4..2.0 m/s."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _add_task032_focused_curriculum_stage(cfg, stage="hard")
  _configure_task029_forward_speed_command(cfg, min_speed=0.4, max_speed=2.0)
  return cfg
'''


IMPORT_LINES = (
  "  unitree_g1_gripper_flat_task032_weak_focused_env_cfg,\n"
  "  unitree_g1_gripper_flat_task032_mixed_focused_env_cfg,\n"
  "  unitree_g1_gripper_flat_task032_hard_focused_env_cfg,\n"
)


REGISTERS = r'''
register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task032-WeakFocused-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task032_weak_focused_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task032_weak_focused_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task032-MixedFocused-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task032_mixed_focused_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task032_mixed_focused_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task032-HardFocused-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task032_hard_focused_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task032_hard_focused_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
'''


def replace_once(text: str, old: str, new: str) -> str:
  if old not in text:
    raise RuntimeError(f"anchor not found: {old[:80]!r}")
  return text.replace(old, new, 1)


def patch_env_cfgs(path: Path) -> None:
  text = path.read_text(encoding="utf-8")
  if "def unitree_g1_gripper_flat_task032_weak_focused_env_cfg(" not in text:
    anchor = "\n\ndef _add_motor_failure_stage(cfg: ManagerBasedRlEnvCfg) -> None:\n"
    text = replace_once(text, anchor, FUNCTIONS + anchor)
  path.write_text(text, encoding="utf-8")


def patch_init(path: Path) -> None:
  text = path.read_text(encoding="utf-8")
  import_anchor = "  unitree_g1_gripper_flat_task031_focused_deadgrid_env_cfg,\n"
  if IMPORT_LINES not in text:
    text = replace_once(text, import_anchor, import_anchor + IMPORT_LINES)
  task_id = "Unitree-G1-Gripper-Flat-Task032-WeakFocused-Fast2p0"
  if task_id not in text:
    text = text.rstrip() + "\n\n" + REGISTERS.strip() + "\n"
  path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=ROOT)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  cfg_path = args.root / "src/tasks/velocity/config/g1_gripper/env_cfgs.py"
  init_path = args.root / "src/tasks/velocity/config/g1_gripper/__init__.py"
  patch_env_cfgs(cfg_path)
  patch_init(init_path)
  print(cfg_path)
  print(init_path)


if __name__ == "__main__":
  main()
