#!/usr/bin/env python3
"""Patch H200 MJLab with Task035 eval-gated curriculum stages."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path("/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab")


FUNCTIONS = r'''

TASK035_SPEED_BINS = (0.4, 1.2, 2.0)
TASK035_SPEED_BIN_WEIGHTS = (0.34, 0.33, 0.33)
TASK035_HARDCASE_SPEED_BIN_WEIGHTS = (0.45, 0.10, 0.45)

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


TASK035_HARDCASE_MOTOR_FAILURE_JOINTS = (
  "left_hip_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_pitch_joint",
  "right_knee_joint",
)


def _configure_task035_speed_bin_command(
  cfg: ManagerBasedRlEnvCfg,
  *,
  weights: tuple[float, float, float] = TASK035_SPEED_BIN_WEIGHTS,
) -> None:
  """Use explicit low/mid/high forward-speed bins for eval-gated training."""
  _configure_task029_forward_speed_command(cfg, min_speed=0.4, max_speed=2.0)
  twist_cmd = cfg.commands["twist"]
  twist_cmd.lin_vel_x_choices = TASK035_SPEED_BINS
  twist_cmd.lin_vel_x_choice_weights = weights


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


def _add_task035_hardcase_persistent_stage(cfg: ManagerBasedRlEnvCfg) -> None:
  """Task035 focused hard-case persistent motor-failure curriculum."""
  _add_motor_failure_stage(cfg)
  params = cfg.events["motor_failure"].params
  params["max_failed_motors"] = 1
  params["single_dead_joint_names"] = TASK035_HARDCASE_MOTOR_FAILURE_JOINTS
  params["single_dead_probability"] = 0.95
  params["dead_probability"] = 0.55
  params["weak_scale_range"] = (0.45, 0.80)
  params["dead_scale_range"] = (0.05, 0.20)
  _add_task029_phase_randomization(cfg)


def unitree_g1_gripper_flat_task035_clean_unified_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Task035 clean unified-speed rehearsal env over 0.4..2.0 m/s."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _configure_task035_speed_bin_command(cfg)
  return cfg


def unitree_g1_gripper_flat_task035_weak_persistent_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Task035 weak persistent motor-failure env over 0.4..2.0 m/s."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _add_task035_balanced_persistent_stage(cfg, stage="weak")
  _configure_task035_speed_bin_command(cfg)
  return cfg


def unitree_g1_gripper_flat_task035_mixed_persistent_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Task035 mixed weak/dead persistent env over 0.4..2.0 m/s."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _add_task035_balanced_persistent_stage(cfg, stage="mixed")
  _configure_task035_speed_bin_command(cfg)
  return cfg


def unitree_g1_gripper_flat_task035_forced_deadgrid_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Task035 hard balanced persistent dead-grid rehearsal env."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _add_task035_balanced_persistent_stage(cfg, stage="deadgrid")
  _configure_task035_speed_bin_command(cfg)
  return cfg


def unitree_g1_gripper_flat_task035_hardcase_persistent_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Task035 focused low/high hard-case failure rehearsal env."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _add_task035_hardcase_persistent_stage(cfg)
  _configure_task035_speed_bin_command(cfg, weights=TASK035_HARDCASE_SPEED_BIN_WEIGHTS)
  return cfg
'''


VELOCITY_COMMAND_RESAMPLE_OLD = (
  "    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)\n"
)


VELOCITY_COMMAND_RESAMPLE_NEW = '''\
    lin_vel_x_choices = getattr(self.cfg, "lin_vel_x_choices", None)
    if lin_vel_x_choices is None:
      self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    else:
      choices = torch.as_tensor(
        lin_vel_x_choices,
        device=self.device,
        dtype=self.vel_command_b.dtype,
      )
      weights = getattr(self.cfg, "lin_vel_x_choice_weights", None)
      if weights is None:
        choice_ids = torch.randint(
          len(lin_vel_x_choices),
          (len(env_ids),),
          device=self.device,
        )
      else:
        probabilities = torch.as_tensor(weights, device=self.device, dtype=torch.float32)
        probabilities = probabilities / probabilities.sum()
        choice_ids = torch.multinomial(
          probabilities,
          len(env_ids),
          replacement=True,
        )
      self.vel_command_b[env_ids, 0] = choices[choice_ids]
'''


VELOCITY_COMMAND_CFG_OLD = "  init_velocity_prob: float = 0.0\n"


VELOCITY_COMMAND_CFG_NEW = '''\
  init_velocity_prob: float = 0.0
  lin_vel_x_choices: tuple[float, ...] | None = None
  lin_vel_x_choice_weights: tuple[float, ...] | None = None
'''


IMPORT_LINES = (
  "  unitree_g1_gripper_flat_task035_clean_unified_env_cfg,\n"
  "  unitree_g1_gripper_flat_task035_weak_persistent_env_cfg,\n"
  "  unitree_g1_gripper_flat_task035_mixed_persistent_env_cfg,\n"
  "  unitree_g1_gripper_flat_task035_forced_deadgrid_env_cfg,\n"
  "  unitree_g1_gripper_flat_task035_hardcase_persistent_env_cfg,\n"
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
  task_id="Unitree-G1-Gripper-Flat-Task035-HardCasePersistent-FrozenBase-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task035_hardcase_persistent_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task035_hardcase_persistent_env_cfg(play=True),
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
  if "TASK035_SPEED_BINS" not in text:
    anchor = "TASK035_BALANCED_MOTOR_FAILURE_JOINTS = (\n"
    text = replace_once(text, anchor, "TASK035_SPEED_BINS = (0.4, 1.2, 2.0)\nTASK035_SPEED_BIN_WEIGHTS = (0.34, 0.33, 0.33)\nTASK035_HARDCASE_SPEED_BIN_WEIGHTS = (0.45, 0.10, 0.45)\n\n" + anchor)
  if "TASK035_HARDCASE_SPEED_BIN_WEIGHTS" not in text:
    text = replace_once(
      text,
      "TASK035_SPEED_BIN_WEIGHTS = (0.34, 0.33, 0.33)\n\n",
      "TASK035_SPEED_BIN_WEIGHTS = (0.34, 0.33, 0.33)\nTASK035_HARDCASE_SPEED_BIN_WEIGHTS = (0.45, 0.10, 0.45)\n\n",
    )
  if "TASK035_HARDCASE_MOTOR_FAILURE_JOINTS" not in text:
    anchor = "  \"right_ankle_roll_joint\",\n)\n\n"
    hardcase = '''\
  "right_ankle_roll_joint",
)

TASK035_HARDCASE_MOTOR_FAILURE_JOINTS = (
  "left_hip_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_pitch_joint",
  "right_knee_joint",
)

'''
    text = replace_once(text, anchor, hardcase)
  if "def _configure_task035_speed_bin_command(" not in text:
    anchor = "\n\ndef _add_task035_balanced_persistent_stage(\n"
    helper = '''\

def _configure_task035_speed_bin_command(
  cfg: ManagerBasedRlEnvCfg,
  *,
  weights: tuple[float, float, float] = TASK035_SPEED_BIN_WEIGHTS,
) -> None:
  """Use explicit low/mid/high forward-speed bins for eval-gated training."""
  _configure_task029_forward_speed_command(cfg, min_speed=0.4, max_speed=2.0)
  twist_cmd = cfg.commands["twist"]
  twist_cmd.lin_vel_x_choices = TASK035_SPEED_BINS
  twist_cmd.lin_vel_x_choice_weights = weights
'''
    text = replace_once(text, anchor, helper + anchor)
  elif "weights: tuple[float, float, float]" not in text:
    old_helper = '''\
def _configure_task035_speed_bin_command(cfg: ManagerBasedRlEnvCfg) -> None:
  """Use explicit low/mid/high forward-speed bins for eval-gated training."""
  _configure_task029_forward_speed_command(cfg, min_speed=0.4, max_speed=2.0)
  twist_cmd = cfg.commands["twist"]
  twist_cmd.lin_vel_x_choices = TASK035_SPEED_BINS
  twist_cmd.lin_vel_x_choice_weights = TASK035_SPEED_BIN_WEIGHTS
'''
    new_helper = '''\
def _configure_task035_speed_bin_command(
  cfg: ManagerBasedRlEnvCfg,
  *,
  weights: tuple[float, float, float] = TASK035_SPEED_BIN_WEIGHTS,
) -> None:
  """Use explicit low/mid/high forward-speed bins for eval-gated training."""
  _configure_task029_forward_speed_command(cfg, min_speed=0.4, max_speed=2.0)
  twist_cmd = cfg.commands["twist"]
  twist_cmd.lin_vel_x_choices = TASK035_SPEED_BINS
  twist_cmd.lin_vel_x_choice_weights = weights
'''
    text = replace_once(text, old_helper, new_helper)
  if "def _add_task035_hardcase_persistent_stage(" not in text:
    anchor = "\n\ndef unitree_g1_gripper_flat_task035_clean_unified_env_cfg(\n"
    stage = '''\

def _add_task035_hardcase_persistent_stage(cfg: ManagerBasedRlEnvCfg) -> None:
  """Task035 focused hard-case persistent motor-failure curriculum."""
  _add_motor_failure_stage(cfg)
  params = cfg.events["motor_failure"].params
  params["max_failed_motors"] = 1
  params["single_dead_joint_names"] = TASK035_HARDCASE_MOTOR_FAILURE_JOINTS
  params["single_dead_probability"] = 0.95
  params["dead_probability"] = 0.55
  params["weak_scale_range"] = (0.45, 0.80)
  params["dead_scale_range"] = (0.05, 0.20)
  _add_task029_phase_randomization(cfg)
'''
    text = replace_once(text, anchor, stage + anchor)
  if "def unitree_g1_gripper_flat_task035_hardcase_persistent_env_cfg(" not in text:
    anchor = "\n\n\ndef _add_motor_failure_stage(cfg: ManagerBasedRlEnvCfg) -> None:\n"
    if anchor not in text:
      anchor = "\n\nTASK036_LOW_SPEED_FAILURE_JOINTS = (\n"
    env_fn = '''\

def unitree_g1_gripper_flat_task035_hardcase_persistent_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Task035 focused low/high hard-case failure rehearsal env."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _add_task035_hardcase_persistent_stage(cfg)
  _configure_task035_speed_bin_command(cfg, weights=TASK035_HARDCASE_SPEED_BIN_WEIGHTS)
  return cfg
'''
    text = replace_once(text, anchor, env_fn + anchor)
  task035_start = text.find("def unitree_g1_gripper_flat_task035_clean_unified_env_cfg(")
  task035_end = text.find("\n\ndef _add_motor_failure_stage(", task035_start)
  if task035_start >= 0 and task035_end > task035_start:
    task035_block = text[task035_start:task035_end].replace(
      "  _configure_task029_forward_speed_command(cfg, min_speed=0.4, max_speed=2.0)\n  return cfg\n",
      "  _configure_task035_speed_bin_command(cfg)\n  return cfg\n",
    )
    text = text[:task035_start] + task035_block + text[task035_end:]
  path.write_text(text, encoding="utf-8")


def patch_velocity_command(path: Path) -> None:
  text = path.read_text(encoding="utf-8")
  if 'getattr(self.cfg, "lin_vel_x_choices", None)' not in text:
    text = replace_once(
      text,
      VELOCITY_COMMAND_RESAMPLE_OLD,
      VELOCITY_COMMAND_RESAMPLE_NEW,
    )
  if "lin_vel_x_choice_weights" not in text:
    text = replace_once(text, VELOCITY_COMMAND_CFG_OLD, VELOCITY_COMMAND_CFG_NEW)
  path.write_text(text, encoding="utf-8")


def patch_init(path: Path) -> None:
  text = path.read_text(encoding="utf-8")
  import_anchor_candidates = (
    "  unitree_g1_gripper_flat_task034_rightknee_hard_env_cfg,\n",
    "  unitree_g1_gripper_flat_task032_hard_focused_env_cfg,\n",
    "  unitree_g1_gripper_flat_task031_focused_deadgrid_env_cfg,\n",
  )
  import_block = "".join(IMPORT_LINES)
  if import_block not in text:
    for anchor in import_anchor_candidates:
      if anchor in text:
        text = replace_once(text, anchor, anchor + import_block)
        break
    else:
      raise RuntimeError("no env cfg import anchor found")
  hardcase_import = "  unitree_g1_gripper_flat_task035_hardcase_persistent_env_cfg,\n"
  if hardcase_import not in text:
    anchor = "  unitree_g1_gripper_flat_task035_forced_deadgrid_env_cfg,\n"
    text = replace_once(text, anchor, anchor + hardcase_import)
  if "Task033StackMlpK4FrozenBaseRunner" not in text:
    runner_anchor = "from mjlab.rl import MjlabOnPolicyRunner\n"
    text = replace_once(text, runner_anchor, runner_anchor + RUNNER_IMPORT)
  task_id = "Unitree-G1-Gripper-Flat-Task035-CleanUnified-FrozenBase-Fast2p0"
  if task_id not in text:
    text = text.rstrip() + "\n\n" + REGISTERS.strip() + "\n"
  hardcase_task_id = "Unitree-G1-Gripper-Flat-Task035-HardCasePersistent-FrozenBase-Fast2p0"
  if task_id in text and hardcase_task_id not in text:
    anchor = '''\
register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task035-DynamicSwitch-FrozenBase-Fast1p6",
'''
    hardcase_register = '''\
register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task035-HardCasePersistent-FrozenBase-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task035_hardcase_persistent_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task035_hardcase_persistent_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task033StackMlpK4FrozenBaseRunner,
)

'''
    text = replace_once(text, anchor, hardcase_register + anchor)
  path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=ROOT)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  cfg_path = args.root / "src/tasks/velocity/config/g1_gripper/env_cfgs.py"
  init_path = args.root / "src/tasks/velocity/config/g1_gripper/__init__.py"
  command_path = args.root / "src/tasks/velocity/mdp/velocity_command.py"
  patch_velocity_command(command_path)
  patch_env_cfgs(cfg_path)
  patch_init(init_path)
  print(command_path)
  print(cfg_path)
  print(init_path)


if __name__ == "__main__":
  main()
