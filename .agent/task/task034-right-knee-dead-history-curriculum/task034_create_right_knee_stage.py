#!/usr/bin/env python3
"""Patch H200 MJLab with Task034 right-knee focused frozen-base stages."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path("/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab")


FUNCTIONS = r'''

def _add_task034_right_knee_focused_stage(
  cfg: ManagerBasedRlEnvCfg,
  *,
  stage: str,
) -> None:
  """Task034 right-knee focused persistent motor-failure curriculum."""
  _add_motor_failure_stage(cfg)
  params = cfg.events["motor_failure"].params
  params["max_failed_motors"] = 1
  params["single_dead_joint_names"] = (
    "right_knee_joint",
    "right_knee_joint",
    "right_knee_joint",
    "right_knee_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "left_knee_joint",
    "left_hip_roll_joint",
  )
  _add_task029_phase_randomization(cfg)
  if stage == "weak":
    params["single_dead_probability"] = 0.95
    params["dead_probability"] = 0.35
    params["weak_scale_range"] = (0.45, 0.80)
    params["dead_scale_range"] = (0.10, 0.35)
  elif stage == "mixed":
    params["single_dead_probability"] = 0.98
    params["dead_probability"] = 0.65
    params["weak_scale_range"] = (0.35, 0.70)
    params["dead_scale_range"] = (0.02, 0.20)
  elif stage == "hard":
    params["single_dead_probability"] = 1.0
    params["dead_probability"] = 0.90
    params["weak_scale_range"] = (0.30, 0.65)
    params["dead_scale_range"] = (0.0, 0.08)
  else:
    raise ValueError(f"unknown task034 stage: {stage}")


def unitree_g1_gripper_flat_task034_rightknee_weak_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Task034 weak right-knee focused env at 2.0 m/s."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _add_task034_right_knee_focused_stage(cfg, stage="weak")
  _configure_task029_forward_speed_command(cfg, min_speed=2.0, max_speed=2.0)
  return cfg


def unitree_g1_gripper_flat_task034_rightknee_mixed_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Task034 mixed right-knee focused env at 2.0 m/s."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _add_task034_right_knee_focused_stage(cfg, stage="mixed")
  _configure_task029_forward_speed_command(cfg, min_speed=2.0, max_speed=2.0)
  return cfg


def unitree_g1_gripper_flat_task034_rightknee_hard_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Task034 hard right-knee focused env at 2.0 m/s."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _add_task034_right_knee_focused_stage(cfg, stage="hard")
  _configure_task029_forward_speed_command(cfg, min_speed=2.0, max_speed=2.0)
  return cfg
'''


IMPORT_LINES = (
  "  unitree_g1_gripper_flat_task034_rightknee_weak_env_cfg,\n"
  "  unitree_g1_gripper_flat_task034_rightknee_mixed_env_cfg,\n"
  "  unitree_g1_gripper_flat_task034_rightknee_hard_env_cfg,\n"
)


REGISTERS = r'''
register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task034-RightKneeWeak-FrozenBase-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task034_rightknee_weak_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task034_rightknee_weak_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task033StackMlpK4FrozenBaseRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task034-RightKneeMixed-FrozenBase-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task034_rightknee_mixed_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task034_rightknee_mixed_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task033StackMlpK4FrozenBaseRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task034-RightKneeHard-FrozenBase-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task034_rightknee_hard_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task034_rightknee_hard_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task033StackMlpK4FrozenBaseRunner,
)
'''


def replace_once(text: str, old: str, new: str) -> str:
  if old not in text:
    raise RuntimeError(f"anchor not found: {old[:80]!r}")
  return text.replace(old, new, 1)


def patch_env_cfgs(path: Path) -> None:
  text = path.read_text(encoding="utf-8")
  if "def unitree_g1_gripper_flat_task034_rightknee_weak_env_cfg(" not in text:
    anchor = "\n\ndef _add_motor_failure_stage(cfg: ManagerBasedRlEnvCfg) -> None:\n"
    text = replace_once(text, anchor, FUNCTIONS + anchor)
  path.write_text(text, encoding="utf-8")


def patch_init(path: Path) -> None:
  text = path.read_text(encoding="utf-8")
  import_anchor = "  unitree_g1_gripper_flat_task032_hard_focused_env_cfg,\n"
  fallback_anchor = "  unitree_g1_gripper_flat_task031_focused_deadgrid_env_cfg,\n"
  if IMPORT_LINES not in text:
    if import_anchor in text:
      text = replace_once(text, import_anchor, import_anchor + IMPORT_LINES)
    else:
      text = replace_once(text, fallback_anchor, fallback_anchor + IMPORT_LINES)
  task_id = "Unitree-G1-Gripper-Flat-Task034-RightKneeWeak-FrozenBase-Fast2p0"
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
