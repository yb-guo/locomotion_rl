#!/usr/bin/env python3
"""Patch H200 MJLab with Task035 eval-gated curriculum stages."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path("/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab")


FUNCTIONS = r'''

TASK035_BALANCED_MOTOR_FAILURE_JOINTS = (
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


def _add_task035_balanced_persistent_stage(
  cfg: ManagerBasedRlEnvCfg,
  *,
  stage: str,
) -> None:
  """Task035 balanced persistent motor-failure curriculum."""
  _add_motor_failure_stage(cfg)
  params = cfg.events["motor_failure"].params
  params["max_failed_motors"] = 1
  params["single_dead_joint_names"] = TASK035_BALANCED_MOTOR_FAILURE_JOINTS
  _add_task029_phase_randomization(cfg)
  if stage == "weak":
    params["single_dead_probability"] = 0.80
    params["dead_probability"] = 0.0
    params["weak_scale_range"] = (0.55, 0.85)
    params["dead_scale_range"] = (0.20, 0.40)
  elif stage == "mixed":
    params["single_dead_probability"] = 0.90
    params["dead_probability"] = 0.35
    params["weak_scale_range"] = (0.45, 0.80)
    params["dead_scale_range"] = (0.08, 0.30)
  elif stage == "deadgrid":
    params["single_dead_probability"] = 0.98
    params["dead_probability"] = 0.85
    params["weak_scale_range"] = (0.35, 0.70)
    params["dead_scale_range"] = (0.0, 0.10)
  else:
    raise ValueError(f"unknown task035 stage: {stage}")


def unitree_g1_gripper_flat_task035_clean_unified_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Task035 clean unified-speed rehearsal env over 0.4..2.0 m/s."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _configure_task029_forward_speed_command(cfg, min_speed=0.4, max_speed=2.0)
  return cfg


def unitree_g1_gripper_flat_task035_weak_persistent_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Task035 weak persistent motor-failure env over 0.4..2.0 m/s."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _add_task035_balanced_persistent_stage(cfg, stage="weak")
  _configure_task029_forward_speed_command(cfg, min_speed=0.4, max_speed=2.0)
  return cfg


def unitree_g1_gripper_flat_task035_mixed_persistent_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Task035 mixed weak/dead persistent env over 0.4..2.0 m/s."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _add_task035_balanced_persistent_stage(cfg, stage="mixed")
  _configure_task029_forward_speed_command(cfg, min_speed=0.4, max_speed=2.0)
  return cfg


def unitree_g1_gripper_flat_task035_forced_deadgrid_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Task035 hard balanced persistent dead-grid rehearsal env."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _add_task035_balanced_persistent_stage(cfg, stage="deadgrid")
  _configure_task029_forward_speed_command(cfg, min_speed=0.4, max_speed=2.0)
  return cfg
'''


IMPORT_LINES = (
  "  unitree_g1_gripper_flat_task035_clean_unified_env_cfg,\n"
  "  unitree_g1_gripper_flat_task035_weak_persistent_env_cfg,\n"
  "  unitree_g1_gripper_flat_task035_mixed_persistent_env_cfg,\n"
  "  unitree_g1_gripper_flat_task035_forced_deadgrid_env_cfg,\n"
)


RUNNER_IMPORT = (
  "from h200_locomotion_lab.training.rsl_history_wrapper import "
  "Task033StackMlpK4FrozenBaseRunner\n"
)


REGISTERS = r'''
register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task035-CleanUnified-FrozenBase-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task035_clean_unified_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task035_clean_unified_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task033StackMlpK4FrozenBaseRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task035-WeakPersistent-FrozenBase-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task035_weak_persistent_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task035_weak_persistent_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task033StackMlpK4FrozenBaseRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task035-MixedPersistent-FrozenBase-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task035_mixed_persistent_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task035_mixed_persistent_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task033StackMlpK4FrozenBaseRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task035-ForcedDeadGrid-FrozenBase-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task035_forced_deadgrid_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task035_forced_deadgrid_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task033StackMlpK4FrozenBaseRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task035-DynamicSwitch-FrozenBase-Fast1p6",
  env_cfg=unitree_g1_gripper_flat_dynamic_failure_train_fast1p6_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_dynamic_failure_train_fast1p6_env_cfg(play=True),
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
  if "def unitree_g1_gripper_flat_task035_clean_unified_env_cfg(" not in text:
    anchor = "\n\ndef _add_motor_failure_stage(cfg: ManagerBasedRlEnvCfg) -> None:\n"
    text = replace_once(text, anchor, FUNCTIONS + anchor)
  path.write_text(text, encoding="utf-8")


def patch_init(path: Path) -> None:
  text = path.read_text(encoding="utf-8")
  import_anchor_candidates = (
    "  unitree_g1_gripper_flat_task034_rightknee_hard_env_cfg,\n",
    "  unitree_g1_gripper_flat_task032_hard_focused_env_cfg,\n",
    "  unitree_g1_gripper_flat_task031_focused_deadgrid_env_cfg,\n",
  )
  if IMPORT_LINES not in text:
    for anchor in import_anchor_candidates:
      if anchor in text:
        text = replace_once(text, anchor, anchor + "".join(IMPORT_LINES))
        break
    else:
      raise RuntimeError("no env cfg import anchor found")
  if "Task033StackMlpK4FrozenBaseRunner" not in text:
    runner_anchor = "from mjlab.rl import MjlabOnPolicyRunner\n"
    text = replace_once(text, runner_anchor, runner_anchor + RUNNER_IMPORT)
  task_id = "Unitree-G1-Gripper-Flat-Task035-CleanUnified-FrozenBase-Fast2p0"
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
