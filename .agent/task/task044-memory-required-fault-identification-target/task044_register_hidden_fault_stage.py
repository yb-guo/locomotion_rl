#!/usr/bin/env python3
"""Patch H200 MJLab with the Task044 hidden-fault true-TXL task."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path("/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab")
TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-Fast1p6"
EVAL_ALIGNED_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-EvalLeftKnee1p6"
EVAL_ALIGNED_VELBOOST_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-EvalLeftKneeVelBoost1p6"
PERSISTENT_HIDDEN_VELBOOST_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenVelBoost1p6"
PERSISTENT_IMMEDIATE_VELBOOST_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateVelBoost1p6"
PERSISTENT_IMMEDIATE_LEFT_KNEE_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneeDeadVelBoost1p6"
PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneeDeadSpeedPush1p6"
PERSISTENT_HIDDEN_SPEED_STABILITY_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenSpeedStability1p6"
PERSISTENT_HIDDEN_HEIGHT_GUARD_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenHeightGuard1p6"
PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenHeightGuardStrong1p6"
PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenLowRootTerminate1p6"
PERSISTENT_HIDDEN_POSE_TERMINATE_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenPoseTerminate1p6"
PERSISTENT_HIDDEN_POSE_TIGHT_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenPoseTight1p6"
PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenSpeedPoseBalance1p6"
PERSISTENT_HIDDEN_FORWARD_FLOOR_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenForwardFloor1p6"
PERSISTENT_HIDDEN_FORWARD_TARGET_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenForwardTarget1p6"
PERSISTENT_HIDDEN_SPEED_CURRICULUM_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenSpeedCurriculum1p4To1p6"
PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneePoseForward1p6"
PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_TASK_ID = "Unitree-G1-Gripper-Flat-Task045-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneePoseForwardSurvival1p6"
PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_TASK_ID = "Unitree-G1-Gripper-Flat-Task045-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneeLongSurvival1p6"
POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_TASK_ID = "Unitree-G1-Gripper-Flat-Task045-TrueTxlHiddenFaultTrain-PoseTightGateLeftKneeLongTail1p6"
PERSISTENT_HIDDEN_STARTUP_BOOST_TASK_ID = "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenStartupBoost1p6"
RUNNER_NAME = "Task044TrueTxlMemoryK160ClearHistoryRunner"
FIXED_SPEED_HELPER_NAME = "_task044_hidden_fault_fixed1p6_env_cfg"
EVAL_ALIGNED_HELPER_NAME = "_task044_eval_left_knee_fixed1p6_env_cfg"
EVAL_ALIGNED_VELBOOST_HELPER_NAME = "_task044_eval_left_knee_velboost_fixed1p6_env_cfg"
PERSISTENT_HIDDEN_VELBOOST_HELPER_NAME = "_task044_persistent_hidden_velboost_fixed1p6_env_cfg"
PERSISTENT_IMMEDIATE_VELBOOST_HELPER_NAME = "_task044_persistent_immediate_velboost_fixed1p6_env_cfg"
PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_NAME = "_task044_persistent_immediate_left_knee_dead_velboost_fixed1p6_env_cfg"
PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_HELPER_NAME = "_task044_persistent_immediate_left_knee_dead_speedpush_fixed1p6_env_cfg"
PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_NAME = "_task044_persistent_hidden_speed_stability_fixed1p6_env_cfg"
PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_NAME = "_task044_persistent_hidden_height_guard_fixed1p6_env_cfg"
PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_HELPER_NAME = "_task044_persistent_hidden_height_guard_strong_fixed1p6_env_cfg"
PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_HELPER_NAME = "_task044_persistent_hidden_low_root_terminate_fixed1p6_env_cfg"
PERSISTENT_HIDDEN_POSE_TERMINATE_HELPER_NAME = "_task044_persistent_hidden_pose_terminate_fixed1p6_env_cfg"
PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_NAME = "_task044_persistent_hidden_pose_tight_fixed1p6_env_cfg"
PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_NAME = "_task044_persistent_hidden_speed_pose_balance_fixed1p6_env_cfg"
PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_NAME = "_task044_persistent_hidden_forward_floor_fixed1p6_env_cfg"
PERSISTENT_HIDDEN_FORWARD_TARGET_HELPER_NAME = "_task044_persistent_hidden_forward_target_fixed1p6_env_cfg"
PERSISTENT_HIDDEN_SPEED_CURRICULUM_HELPER_NAME = "_task044_persistent_hidden_speed_curriculum_1p4_to_1p6_env_cfg"
PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_HELPER_NAME = "_task044_persistent_immediate_left_knee_pose_forward_fixed1p6_env_cfg"
PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_NAME = "_task045_persistent_immediate_left_knee_pose_forward_survival_fixed1p6_env_cfg"
PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_HELPER_NAME = "_task045_persistent_immediate_left_knee_long_survival_fixed1p6_env_cfg"
POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_NAME = "_task045_pose_tight_gate_left_knee_long_tail_fixed1p6_env_cfg"
PERSISTENT_HIDDEN_STARTUP_BOOST_HELPER_NAME = "_task044_persistent_hidden_startup_boost_fixed1p6_env_cfg"
TRAIN_ENV_CFG_NAME = "unitree_g1_gripper_flat_dynamic_failure_train_fast1p6_env_cfg"
DETERMINISTIC_ENV_CFG_NAME = "unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg"

RUNNER_IMPORT_BLOCK = (
    "from h200_locomotion_lab.training.rsl_history_wrapper import (\n"
    f"  {RUNNER_NAME},\n"
    ")\n"
)

RUNNER_IMPORT_RE = re.compile(
    r"from\s+h200_locomotion_lab\.training\.rsl_history_wrapper\s+import\s*\(\s*\n"
    r"(?P<body>.*?)"
    r"(?P<footer>\n\s*\))",
    re.DOTALL,
)

TASK_REGISTER_RE = re.compile(
    r"\n*register_mjlab_task\(\n"
    rf'\s*task_id="(?:{re.escape(TASK_ID)}|{re.escape(EVAL_ALIGNED_TASK_ID)}|{re.escape(EVAL_ALIGNED_VELBOOST_TASK_ID)}|{re.escape(PERSISTENT_HIDDEN_VELBOOST_TASK_ID)}|{re.escape(PERSISTENT_IMMEDIATE_VELBOOST_TASK_ID)}|{re.escape(PERSISTENT_IMMEDIATE_LEFT_KNEE_TASK_ID)}|{re.escape(PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_TASK_ID)}|{re.escape(PERSISTENT_HIDDEN_SPEED_STABILITY_TASK_ID)}|{re.escape(PERSISTENT_HIDDEN_HEIGHT_GUARD_TASK_ID)}|{re.escape(PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_TASK_ID)}|{re.escape(PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_TASK_ID)}|{re.escape(PERSISTENT_HIDDEN_POSE_TERMINATE_TASK_ID)}|{re.escape(PERSISTENT_HIDDEN_POSE_TIGHT_TASK_ID)}|{re.escape(PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_TASK_ID)}|{re.escape(PERSISTENT_HIDDEN_FORWARD_FLOOR_TASK_ID)}|{re.escape(PERSISTENT_HIDDEN_FORWARD_TARGET_TASK_ID)}|{re.escape(PERSISTENT_HIDDEN_SPEED_CURRICULUM_TASK_ID)}|{re.escape(PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_TASK_ID)}|{re.escape(PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_TASK_ID)}|{re.escape(PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_TASK_ID)}|{re.escape(POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_TASK_ID)}|{re.escape(PERSISTENT_HIDDEN_STARTUP_BOOST_TASK_ID)})",'
    r".*?\n\)\n*",
    re.DOTALL,
)

FIXED_SPEED_HELPER_RE = re.compile(
    rf"\n*def {FIXED_SPEED_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

EVAL_ALIGNED_HELPER_RE = re.compile(
    rf"\n*def {EVAL_ALIGNED_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

EVAL_ALIGNED_VELBOOST_HELPER_RE = re.compile(
    rf"\n*def {EVAL_ALIGNED_VELBOOST_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_HIDDEN_VELBOOST_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_HIDDEN_VELBOOST_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_IMMEDIATE_VELBOOST_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_IMMEDIATE_VELBOOST_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_HIDDEN_POSE_TERMINATE_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_HIDDEN_POSE_TERMINATE_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_HIDDEN_FORWARD_TARGET_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_HIDDEN_FORWARD_TARGET_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_HIDDEN_SPEED_CURRICULUM_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_HIDDEN_SPEED_CURRICULUM_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_RE = re.compile(
    rf"\n*def {POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

PERSISTENT_HIDDEN_STARTUP_BOOST_HELPER_RE = re.compile(
    rf"\n*def {PERSISTENT_HIDDEN_STARTUP_BOOST_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

BASE_HEIGHT_REWARD_RE = re.compile(
    r"\n*def base_height_below_l2\(\n"
    r".*?\n  return torch\.square\(below\)\n*",
    re.DOTALL,
)

BASE_HEIGHT_REWARD_BLOCK = '''
def base_height_below_l2(
  env: ManagerBasedRlEnv,
  min_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize only root heights below a minimum world-z threshold."""
  asset: Entity = env.scene[asset_cfg.name]
  root_z = asset.data.root_link_pos_w[:, 2]
  below = torch.clamp(float(min_height) - root_z, min=0.0)
  return torch.square(below)
'''

FORWARD_VELOCITY_BELOW_L1_RE = re.compile(
    r"\n*def forward_velocity_below_l1\(\n"
    r".*?\n  return torch\.clamp\(float\(target_x\) - forward_vel, min=0\.0\)\n*",
    re.DOTALL,
)

FORWARD_VELOCITY_BELOW_L1_BLOCK = '''
def forward_velocity_below_l1(
  env: ManagerBasedRlEnv,
  target_x: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize only under-speed along the body-frame forward x direction."""
  asset: Entity = env.scene[asset_cfg.name]
  forward_vel = asset.data.root_link_lin_vel_b[:, 0]
  return torch.clamp(float(target_x) - forward_vel, min=0.0)
'''

FORWARD_VELOCITY_BELOW_COMMAND_L1_RE = re.compile(
    r"\n*def forward_velocity_below_command_l1\(\n"
    r".*?\n  return torch\.clamp\(command\[:, 0\] - forward_vel, min=0\.0\)\n*",
    re.DOTALL,
)

FORWARD_VELOCITY_BELOW_COMMAND_L1_BLOCK = '''
def forward_velocity_below_command_l1(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize only under-speed relative to the sampled forward command."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  forward_vel = asset.data.root_link_lin_vel_b[:, 0]
  return torch.clamp(command[:, 0] - forward_vel, min=0.0)
'''

FORWARD_VELOCITY_BELOW_COMMAND_EARLY_L1_RE = re.compile(
    r"\n*def forward_velocity_below_command_early_l1\(\n"
    r".*?\n  return below \* early\n*",
    re.DOTALL,
)

FORWARD_VELOCITY_BELOW_COMMAND_EARLY_L1_BLOCK = '''
def forward_velocity_below_command_early_l1(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_step: int,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize under-speed only during the first steps after a trial reset."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  forward_vel = asset.data.root_link_lin_vel_b[:, 0]
  below = torch.clamp(command[:, 0] - forward_vel, min=0.0)
  early = (env.episode_length_buf < int(max_step)).to(dtype=below.dtype)
  return below * early
'''

ROOT_HEIGHT_TERMINATION_RE = re.compile(
    r"\n*def root_height_below\(\n"
    r".*?\n  return root_z < float\(min_height\)\n*",
    re.DOTALL,
)

PROJECTED_GRAVITY_TERMINATION_RE = re.compile(
    r"\n*def projected_gravity_xy_above\(\n"
    r".*?\n  return gravity_xy > float\(max_xy\)\n*",
    re.DOTALL,
)

ROOT_HEIGHT_TERMINATION_BLOCK = '''
def root_height_below(
  env: ManagerBasedRlEnv,
  min_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate envs whose root height falls below a world-z threshold."""
  asset = env.scene[asset_cfg.name]
  root_z = asset.data.root_link_pos_w[:, 2]
  return root_z < float(min_height)
'''

PROJECTED_GRAVITY_TERMINATION_BLOCK = '''
def projected_gravity_xy_above(
  env: ManagerBasedRlEnv,
  max_xy: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate envs whose projected gravity xy norm exceeds a threshold."""
  asset = env.scene[asset_cfg.name]
  gravity_xy = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=-1)
  return gravity_xy > float(max_xy)
'''

REWARD_CFG_IMPORT_BLOCK = (
    "from mjlab.managers.reward_manager import RewardTermCfg\n"
    "from mjlab.managers.scene_entity_config import SceneEntityCfg\n"
    "from mjlab.managers.termination_manager import TerminationTermCfg\n"
    "import src.tasks.velocity.mdp as mdp\n"
)

TERMINATION_IMPORT_BLOCK = (
    "from mjlab.managers.scene_entity_config import SceneEntityCfg\n"
)

FIXED_SPEED_HELPER_BLOCK = f'''
def {FIXED_SPEED_HELPER_NAME}(play: bool = False):
  cfg = {TRAIN_ENV_CFG_NAME}(play=play)
  twist_cmd = cfg.commands["twist"]
  twist_cmd.ranges.lin_vel_x = (1.6, 1.6)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  params = cfg.events["dynamic_motor_failure"].params
  params["clean_probability"] = 0.05
  params["persistent_probability"] = 0.10
  params["dynamic_single_probability"] = 0.45
  params["left_knee_probability"] = 0.25
  params["dynamic_dead_probability"] = 0.65
  params["weak_scale_range"] = (0.25, 0.75)
  params["dead_scale_range"] = (0.0, 0.1)
  return cfg
'''

EVAL_ALIGNED_HELPER_BLOCK = f'''
def {EVAL_ALIGNED_HELPER_NAME}(play: bool = False):
  cfg = {DETERMINISTIC_ENV_CFG_NAME}(play=play)
  cfg.episode_length_s = 2.0
  twist_cmd = cfg.commands["twist"]
  twist_cmd.ranges.lin_vel_x = (1.6, 1.6)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  params = cfg.events["dynamic_motor_failure"].params
  params["template"] = (
    (0.0, 2.0, "left_knee_joint", "dead", 0.0),
    (2.0, None, None, "normal", 1.0),
  )
  params["transient_window_s"] = 0.3
  return cfg
'''

EVAL_ALIGNED_VELBOOST_HELPER_BLOCK = f'''
def {EVAL_ALIGNED_VELBOOST_HELPER_NAME}(play: bool = False):
  cfg = {EVAL_ALIGNED_HELPER_NAME}(play=play)
  lin_vel_reward = cfg.rewards["track_linear_velocity"]
  lin_vel_reward.weight = 3.0
  lin_vel_reward.params["std"] = 1.0
  return cfg
'''

PERSISTENT_HIDDEN_VELBOOST_HELPER_BLOCK = f'''
def {PERSISTENT_HIDDEN_VELBOOST_HELPER_NAME}(play: bool = False):
  cfg = {TRAIN_ENV_CFG_NAME}(play=play)
  cfg.episode_length_s = 2.0
  twist_cmd = cfg.commands["twist"]
  twist_cmd.ranges.lin_vel_x = (1.6, 1.6)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  params = cfg.events["dynamic_motor_failure"].params
  params["clean_probability"] = 0.02
  params["persistent_probability"] = 0.08
  params["dynamic_single_probability"] = 0.70
  params["left_knee_probability"] = 0.65
  params["dynamic_dead_probability"] = 0.85
  params["weak_scale_range"] = (0.25, 0.70)
  params["dead_scale_range"] = (0.0, 0.05)
  params["transient_window_s"] = 0.3
  params["preserve_schedule_across_inner_resets"] = True
  lin_vel_reward = cfg.rewards["track_linear_velocity"]
  lin_vel_reward.weight = 3.0
  lin_vel_reward.params["std"] = 1.0
  return cfg
'''

PERSISTENT_IMMEDIATE_VELBOOST_HELPER_BLOCK = f'''
def {PERSISTENT_IMMEDIATE_VELBOOST_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_HIDDEN_VELBOOST_HELPER_NAME}(play=play)
  params = cfg.events["dynamic_motor_failure"].params
  params["clean_probability"] = 0.0
  params["persistent_probability"] = 0.0
  params["dynamic_single_probability"] = 1.0
  params["left_knee_probability"] = 0.75
  params["dynamic_dead_probability"] = 0.90
  params["dynamic_single_onset_range_s"] = (0.0, 0.0)
  params["dynamic_single_duration_range_s"] = (2.0, 2.0)
  return cfg
'''

PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_BLOCK = f'''
def {PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_IMMEDIATE_VELBOOST_HELPER_NAME}(play=play)
  params = cfg.events["dynamic_motor_failure"].params
  params["left_knee_probability"] = 1.0
  params["dynamic_dead_probability"] = 1.0
  params["dead_scale_range"] = (0.0, 0.0)
  return cfg
'''

PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_HELPER_BLOCK = f'''
def {PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_NAME}(play=play)
  lin_vel_reward = cfg.rewards["track_linear_velocity"]
  lin_vel_reward.weight = 6.0
  lin_vel_reward.params["std"] = 0.5
  return cfg
'''

PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_BLOCK = f'''
def {PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_HIDDEN_VELBOOST_HELPER_NAME}(play=play)
  lin_vel_reward = cfg.rewards["track_linear_velocity"]
  lin_vel_reward.weight = 4.0
  lin_vel_reward.params["std"] = 0.8
  cfg.rewards["body_orientation_l2"].weight = -2.0
  cfg.rewards["is_terminated"].weight = -300.0
  return cfg
'''

PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_BLOCK = f'''
def {PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_NAME}(play=play)
  cfg.rewards["base_height_below_l2"] = RewardTermCfg(
    func=mdp.base_height_below_l2,
    weight=-8.0,
    params={{"min_height": 0.70, "asset_cfg": SceneEntityCfg("robot")}},
  )
  return cfg
'''

PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_HELPER_BLOCK = f'''
def {PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_NAME}(play=play)
  lin_vel_reward = cfg.rewards["track_linear_velocity"]
  lin_vel_reward.weight = 5.0
  lin_vel_reward.params["std"] = 0.7
  cfg.rewards["base_height_below_l2"].weight = -24.0
  cfg.rewards["base_height_below_l2"].params["min_height"] = 0.72
  return cfg
'''

PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_HELPER_BLOCK = f'''
def {PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_NAME}(play=play)
  lin_vel_reward = cfg.rewards["track_linear_velocity"]
  lin_vel_reward.weight = 4.5
  lin_vel_reward.params["std"] = 0.75
  cfg.terminations["root_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below,
    params={{"min_height": 0.58, "asset_cfg": SceneEntityCfg("robot")}},
  )
  return cfg
'''

PERSISTENT_HIDDEN_POSE_TERMINATE_HELPER_BLOCK = f'''
def {PERSISTENT_HIDDEN_POSE_TERMINATE_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_HELPER_NAME}(play=play)
  cfg.rewards["body_orientation_l2"].weight = -3.0
  cfg.terminations["gravity_xy_too_high"] = TerminationTermCfg(
    func=mdp.projected_gravity_xy_above,
    params={{"max_xy": 0.78, "asset_cfg": SceneEntityCfg("robot")}},
  )
  return cfg
'''

PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_BLOCK = f'''
def {PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_HIDDEN_POSE_TERMINATE_HELPER_NAME}(play=play)
  lin_vel_reward = cfg.rewards["track_linear_velocity"]
  lin_vel_reward.weight = 5.0
  lin_vel_reward.params["std"] = 0.65
  cfg.terminations["gravity_xy_too_high"].params["max_xy"] = 0.74
  return cfg
'''

PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_BLOCK = f'''
def {PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_NAME}(play=play)
  lin_vel_reward = cfg.rewards["track_linear_velocity"]
  lin_vel_reward.weight = 6.5
  lin_vel_reward.params["std"] = 0.55
  cfg.rewards["body_orientation_l2"].weight = -4.0
  cfg.rewards["is_terminated"].weight = -400.0
  cfg.terminations["gravity_xy_too_high"].params["max_xy"] = 0.72
  return cfg
'''

PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_BLOCK = f'''
def {PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_NAME}(play=play)
  lin_vel_reward = cfg.rewards["track_linear_velocity"]
  lin_vel_reward.weight = 5.5
  lin_vel_reward.params["std"] = 0.60
  cfg.rewards["forward_velocity_below_l1"] = RewardTermCfg(
    func=mdp.forward_velocity_below_l1,
    weight=-3.0,
    params={{"target_x": 1.45, "asset_cfg": SceneEntityCfg("robot")}},
  )
  return cfg
'''

PERSISTENT_HIDDEN_FORWARD_TARGET_HELPER_BLOCK = f'''
def {PERSISTENT_HIDDEN_FORWARD_TARGET_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_NAME}(play=play)
  lin_vel_reward = cfg.rewards["track_linear_velocity"]
  lin_vel_reward.weight = 5.5
  lin_vel_reward.params["std"] = 0.60
  cfg.rewards["forward_velocity_below_l1"] = RewardTermCfg(
    func=mdp.forward_velocity_below_l1,
    weight=-5.0,
    params={{"target_x": 1.55, "asset_cfg": SceneEntityCfg("robot")}},
  )
  return cfg
'''

PERSISTENT_HIDDEN_SPEED_CURRICULUM_HELPER_BLOCK = f'''
def {PERSISTENT_HIDDEN_SPEED_CURRICULUM_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_NAME}(play=play)
  twist_cmd = cfg.commands["twist"]
  twist_cmd.ranges.lin_vel_x = (1.4, 1.6)
  lin_vel_reward = cfg.rewards["track_linear_velocity"]
  lin_vel_reward.weight = 5.5
  lin_vel_reward.params["std"] = 0.60
  cfg.rewards["forward_velocity_below_command_l1"] = RewardTermCfg(
    func=mdp.forward_velocity_below_command_l1,
    weight=-4.0,
    params={{"command_name": "twist", "asset_cfg": SceneEntityCfg("robot")}},
  )
  return cfg
'''

PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_HELPER_BLOCK = f'''
def {PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_NAME}(play=play)
  lin_vel_reward = cfg.rewards["track_linear_velocity"]
  lin_vel_reward.weight = 5.0
  lin_vel_reward.params["std"] = 0.65
  cfg.rewards["body_orientation_l2"].weight = -3.0
  cfg.rewards["base_height_below_l2"] = RewardTermCfg(
    func=mdp.base_height_below_l2,
    weight=-8.0,
    params={{"min_height": 0.70, "asset_cfg": SceneEntityCfg("robot")}},
  )
  cfg.rewards["forward_velocity_below_command_l1"] = RewardTermCfg(
    func=mdp.forward_velocity_below_command_l1,
    weight=-4.0,
    params={{"command_name": "twist", "asset_cfg": SceneEntityCfg("robot")}},
  )
  cfg.terminations["root_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below,
    params={{"min_height": 0.58, "asset_cfg": SceneEntityCfg("robot")}},
  )
  cfg.terminations["gravity_xy_too_high"] = TerminationTermCfg(
    func=mdp.projected_gravity_xy_above,
    params={{"max_xy": 0.74, "asset_cfg": SceneEntityCfg("robot")}},
  )
  return cfg
'''

PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_BLOCK = f'''
def {PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_HELPER_NAME}(play=play)
  cfg.rewards["body_orientation_l2"].weight = -5.0
  cfg.rewards["is_terminated"].weight = -700.0
  cfg.terminations["gravity_xy_too_high"].params["max_xy"] = 0.70
  return cfg
'''

PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_HELPER_BLOCK = f'''
def {PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_NAME}(play=play)
  cfg.episode_length_s = 8.0
  params = cfg.events["dynamic_motor_failure"].params
  params["dynamic_single_duration_range_s"] = (8.0, 8.0)
  return cfg
'''

POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_BLOCK = f'''
def {POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_NAME}(play=play)
  cfg.episode_length_s = 8.0
  cfg.rewards["body_orientation_l2"].weight = -5.0
  cfg.rewards["is_terminated"].weight = -700.0
  cfg.terminations["gravity_xy_too_high"].params["max_xy"] = 0.70
  params = cfg.events["dynamic_motor_failure"].params
  params["clean_probability"] = 0.0
  params["persistent_probability"] = 0.0
  params["dynamic_single_probability"] = 1.0
  params["left_knee_probability"] = 1.0
  params["dynamic_dead_probability"] = 1.0
  params["dead_scale_range"] = (0.0, 0.0)
  params["dynamic_single_onset_range_s"] = (2.0, 2.0)
  params["dynamic_single_duration_range_s"] = (8.0, 8.0)
  params["preserve_schedule_across_inner_resets"] = True
  return cfg
'''

PERSISTENT_HIDDEN_STARTUP_BOOST_HELPER_BLOCK = f'''
def {PERSISTENT_HIDDEN_STARTUP_BOOST_HELPER_NAME}(play: bool = False):
  cfg = {PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_NAME}(play=play)
  cfg.rewards["forward_velocity_below_command_early_l1"] = RewardTermCfg(
    func=mdp.forward_velocity_below_command_early_l1,
    weight=-6.0,
    params={{"command_name": "twist", "max_step": 25, "asset_cfg": SceneEntityCfg("robot")}},
  )
  return cfg
'''

REGISTER_BLOCK = f'''
register_mjlab_task(
  task_id="{TASK_ID}",
  env_cfg={FIXED_SPEED_HELPER_NAME}(),
  play_env_cfg={FIXED_SPEED_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{EVAL_ALIGNED_TASK_ID}",
  env_cfg={EVAL_ALIGNED_HELPER_NAME}(),
  play_env_cfg={EVAL_ALIGNED_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{EVAL_ALIGNED_VELBOOST_TASK_ID}",
  env_cfg={EVAL_ALIGNED_VELBOOST_HELPER_NAME}(),
  play_env_cfg={EVAL_ALIGNED_VELBOOST_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_HIDDEN_VELBOOST_TASK_ID}",
  env_cfg={PERSISTENT_HIDDEN_VELBOOST_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_HIDDEN_VELBOOST_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_IMMEDIATE_VELBOOST_TASK_ID}",
  env_cfg={PERSISTENT_IMMEDIATE_VELBOOST_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_IMMEDIATE_VELBOOST_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_IMMEDIATE_LEFT_KNEE_TASK_ID}",
  env_cfg={PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_TASK_ID}",
  env_cfg={PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_HIDDEN_SPEED_STABILITY_TASK_ID}",
  env_cfg={PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_HIDDEN_HEIGHT_GUARD_TASK_ID}",
  env_cfg={PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_TASK_ID}",
  env_cfg={PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_TASK_ID}",
  env_cfg={PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_HIDDEN_POSE_TERMINATE_TASK_ID}",
  env_cfg={PERSISTENT_HIDDEN_POSE_TERMINATE_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_HIDDEN_POSE_TERMINATE_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_HIDDEN_POSE_TIGHT_TASK_ID}",
  env_cfg={PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_TASK_ID}",
  env_cfg={PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_HIDDEN_FORWARD_FLOOR_TASK_ID}",
  env_cfg={PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_HIDDEN_FORWARD_TARGET_TASK_ID}",
  env_cfg={PERSISTENT_HIDDEN_FORWARD_TARGET_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_HIDDEN_FORWARD_TARGET_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_HIDDEN_SPEED_CURRICULUM_TASK_ID}",
  env_cfg={PERSISTENT_HIDDEN_SPEED_CURRICULUM_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_HIDDEN_SPEED_CURRICULUM_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_TASK_ID}",
  env_cfg={PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_TASK_ID}",
  env_cfg={PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_TASK_ID}",
  env_cfg={PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_TASK_ID}",
  env_cfg={POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_NAME}(),
  play_env_cfg={POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)

register_mjlab_task(
  task_id="{PERSISTENT_HIDDEN_STARTUP_BOOST_TASK_ID}",
  env_cfg={PERSISTENT_HIDDEN_STARTUP_BOOST_HELPER_NAME}(),
  play_env_cfg={PERSISTENT_HIDDEN_STARTUP_BOOST_HELPER_NAME}(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls={RUNNER_NAME},
)
'''


def patch_init(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = _ensure_reward_cfg_imports(text)
    text = _ensure_train_env_cfg_import(text)
    text = _ensure_runner_import(text)
    text = _ensure_fixed_speed_helper(text)
    text = _ensure_eval_aligned_helper(text)
    text = _ensure_eval_aligned_velboost_helper(text)
    text = _ensure_persistent_hidden_velboost_helper(text)
    text = _ensure_persistent_immediate_velboost_helper(text)
    text = _ensure_persistent_immediate_left_knee_helper(text)
    text = _ensure_persistent_immediate_left_knee_speed_push_helper(text)
    text = _ensure_persistent_hidden_speed_stability_helper(text)
    text = _ensure_persistent_hidden_height_guard_helper(text)
    text = _ensure_persistent_hidden_height_guard_strong_helper(text)
    text = _ensure_persistent_hidden_low_root_terminate_helper(text)
    text = _ensure_persistent_hidden_pose_terminate_helper(text)
    text = _ensure_persistent_hidden_pose_tight_helper(text)
    text = _ensure_persistent_hidden_speed_pose_balance_helper(text)
    text = _ensure_persistent_hidden_forward_floor_helper(text)
    text = _ensure_persistent_hidden_forward_target_helper(text)
    text = _ensure_persistent_hidden_speed_curriculum_helper(text)
    text = _ensure_persistent_immediate_left_knee_pose_forward_helper(text)
    text = _ensure_persistent_immediate_left_knee_survival_helper(text)
    text = _ensure_persistent_immediate_left_knee_long_survival_helper(text)
    text = _ensure_pose_tight_gate_left_knee_long_tail_helper(text)
    text = _ensure_persistent_hidden_startup_boost_helper(text)
    text = _remove_existing_task_registration(text)
    text = text.rstrip() + "\n\n" + REGISTER_BLOCK.strip() + "\n"
    path.write_text(text, encoding="utf-8")


def patch_rewards(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "def base_height_below_l2(" in text:
        text = BASE_HEIGHT_REWARD_RE.sub(
            "\n\n" + BASE_HEIGHT_REWARD_BLOCK.strip() + "\n",
            text,
            count=1,
        )
    else:
        text = text.rstrip() + "\n\n" + BASE_HEIGHT_REWARD_BLOCK.strip() + "\n"
    if "def forward_velocity_below_l1(" in text:
        text = FORWARD_VELOCITY_BELOW_L1_RE.sub(
            "\n\n" + FORWARD_VELOCITY_BELOW_L1_BLOCK.strip() + "\n",
            text,
            count=1,
        )
    else:
        text = text.rstrip() + "\n\n" + FORWARD_VELOCITY_BELOW_L1_BLOCK.strip() + "\n"
    if "def forward_velocity_below_command_l1(" in text:
        text = FORWARD_VELOCITY_BELOW_COMMAND_L1_RE.sub(
            "\n\n" + FORWARD_VELOCITY_BELOW_COMMAND_L1_BLOCK.strip() + "\n",
            text,
            count=1,
        )
    else:
        text = text.rstrip() + "\n\n" + FORWARD_VELOCITY_BELOW_COMMAND_L1_BLOCK.strip() + "\n"
    if "def forward_velocity_below_command_early_l1(" in text:
        text = FORWARD_VELOCITY_BELOW_COMMAND_EARLY_L1_RE.sub(
            "\n\n" + FORWARD_VELOCITY_BELOW_COMMAND_EARLY_L1_BLOCK.strip() + "\n",
            text,
            count=1,
        )
    else:
        text = text.rstrip() + "\n\n" + FORWARD_VELOCITY_BELOW_COMMAND_EARLY_L1_BLOCK.strip() + "\n"
    path.write_text(text, encoding="utf-8")


def patch_terminations(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = _ensure_termination_imports(text)
    if "_DEFAULT_ASSET_CFG = SceneEntityCfg(\"robot\")" not in text:
        anchor = "if TYPE_CHECKING:\n  from mjlab.envs import ManagerBasedRlEnv\n"
        if anchor not in text:
            raise RuntimeError("termination TYPE_CHECKING anchor not found")
        text = text.replace(anchor, anchor + "\n\n_DEFAULT_ASSET_CFG = SceneEntityCfg(\"robot\")\n", 1)
    if "def root_height_below(" in text:
        text = ROOT_HEIGHT_TERMINATION_RE.sub(
            "\n\n" + ROOT_HEIGHT_TERMINATION_BLOCK.strip() + "\n",
            text,
            count=1,
        )
    else:
        text = text.rstrip() + "\n\n" + ROOT_HEIGHT_TERMINATION_BLOCK.strip() + "\n"
    if "def projected_gravity_xy_above(" in text:
        text = PROJECTED_GRAVITY_TERMINATION_RE.sub(
            "\n\n" + PROJECTED_GRAVITY_TERMINATION_BLOCK.strip() + "\n",
            text,
            count=1,
        )
    else:
        text = text.rstrip() + "\n\n" + PROJECTED_GRAVITY_TERMINATION_BLOCK.strip() + "\n"
    path.write_text(text, encoding="utf-8")


def _ensure_reward_cfg_imports(text: str) -> str:
    for line in REWARD_CFG_IMPORT_BLOCK.splitlines():
        if line not in text:
            anchor = "from mjlab.rl import MjlabOnPolicyRunner\n"
            if anchor not in text:
                raise RuntimeError(f"reward cfg import anchor not found: {anchor!r}")
            text = text.replace(anchor, anchor + line + "\n", 1)
    return text


def _ensure_termination_imports(text: str) -> str:
    for line in TERMINATION_IMPORT_BLOCK.splitlines():
        if line not in text:
            anchor = "from mjlab.sensor import ContactSensor\n"
            if anchor not in text:
                raise RuntimeError(f"termination import anchor not found: {anchor!r}")
            text = text.replace(anchor, anchor + line + "\n", 1)
    return text


def _ensure_train_env_cfg_import(text: str) -> str:
    import_line = f"  {TRAIN_ENV_CFG_NAME},\n"
    if import_line in text:
        return text
    anchor = "  unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg,\n"
    if anchor not in text:
        raise RuntimeError(f"env cfg import anchor not found: {anchor!r}")
    return text.replace(anchor, anchor + import_line, 1)


def _ensure_runner_import(text: str) -> str:
    match = RUNNER_IMPORT_RE.search(text)
    if match is None:
        anchor = "from mjlab.rl import MjlabOnPolicyRunner\n"
        if anchor not in text:
            raise RuntimeError(f"runner import anchor not found: {anchor!r}")
        return text.replace(anchor, anchor + RUNNER_IMPORT_BLOCK, 1)
    body = match.group("body")
    if RUNNER_NAME in body:
        return text
    insert = f"  {RUNNER_NAME},\n"
    return text[: match.start("body")] + insert + text[match.start("body") :]


def _ensure_fixed_speed_helper(text: str) -> str:
    if FIXED_SPEED_HELPER_NAME in text:
        return FIXED_SPEED_HELPER_RE.sub(
            "\n\n" + FIXED_SPEED_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + FIXED_SPEED_HELPER_BLOCK.strip() + "\n"


def _ensure_eval_aligned_helper(text: str) -> str:
    if EVAL_ALIGNED_HELPER_NAME in text:
        return EVAL_ALIGNED_HELPER_RE.sub(
            "\n\n" + EVAL_ALIGNED_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + EVAL_ALIGNED_HELPER_BLOCK.strip() + "\n"


def _ensure_eval_aligned_velboost_helper(text: str) -> str:
    if EVAL_ALIGNED_VELBOOST_HELPER_NAME in text:
        return EVAL_ALIGNED_VELBOOST_HELPER_RE.sub(
            "\n\n" + EVAL_ALIGNED_VELBOOST_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + EVAL_ALIGNED_VELBOOST_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_hidden_velboost_helper(text: str) -> str:
    if PERSISTENT_HIDDEN_VELBOOST_HELPER_NAME in text:
        return PERSISTENT_HIDDEN_VELBOOST_HELPER_RE.sub(
            "\n\n" + PERSISTENT_HIDDEN_VELBOOST_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_HIDDEN_VELBOOST_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_immediate_velboost_helper(text: str) -> str:
    if PERSISTENT_IMMEDIATE_VELBOOST_HELPER_NAME in text:
        return PERSISTENT_IMMEDIATE_VELBOOST_HELPER_RE.sub(
            "\n\n" + PERSISTENT_IMMEDIATE_VELBOOST_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_IMMEDIATE_VELBOOST_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_immediate_left_knee_helper(text: str) -> str:
    if PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_NAME in text:
        return PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_RE.sub(
            "\n\n" + PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_immediate_left_knee_speed_push_helper(text: str) -> str:
    if PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_HELPER_NAME in text:
        return PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_HELPER_RE.sub(
            "\n\n" + PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_hidden_speed_stability_helper(text: str) -> str:
    if PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_NAME in text:
        return PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_RE.sub(
            "\n\n" + PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_hidden_height_guard_helper(text: str) -> str:
    if PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_NAME in text:
        return PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_RE.sub(
            "\n\n" + PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_hidden_height_guard_strong_helper(text: str) -> str:
    if PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_HELPER_NAME in text:
        return PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_HELPER_RE.sub(
            "\n\n" + PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_hidden_low_root_terminate_helper(text: str) -> str:
    if PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_HELPER_NAME in text:
        return PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_HELPER_RE.sub(
            "\n\n" + PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_hidden_pose_terminate_helper(text: str) -> str:
    if PERSISTENT_HIDDEN_POSE_TERMINATE_HELPER_NAME in text:
        return PERSISTENT_HIDDEN_POSE_TERMINATE_HELPER_RE.sub(
            "\n\n" + PERSISTENT_HIDDEN_POSE_TERMINATE_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_HIDDEN_POSE_TERMINATE_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_hidden_pose_tight_helper(text: str) -> str:
    if PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_NAME in text:
        return PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_RE.sub(
            "\n\n" + PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_hidden_speed_pose_balance_helper(text: str) -> str:
    if PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_NAME in text:
        return PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_RE.sub(
            "\n\n" + PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_hidden_forward_floor_helper(text: str) -> str:
    if PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_NAME in text:
        return PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_RE.sub(
            "\n\n" + PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_hidden_forward_target_helper(text: str) -> str:
    if PERSISTENT_HIDDEN_FORWARD_TARGET_HELPER_NAME in text:
        return PERSISTENT_HIDDEN_FORWARD_TARGET_HELPER_RE.sub(
            "\n\n" + PERSISTENT_HIDDEN_FORWARD_TARGET_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_HIDDEN_FORWARD_TARGET_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_hidden_speed_curriculum_helper(text: str) -> str:
    if PERSISTENT_HIDDEN_SPEED_CURRICULUM_HELPER_NAME in text:
        return PERSISTENT_HIDDEN_SPEED_CURRICULUM_HELPER_RE.sub(
            "\n\n" + PERSISTENT_HIDDEN_SPEED_CURRICULUM_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_HIDDEN_SPEED_CURRICULUM_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_immediate_left_knee_pose_forward_helper(text: str) -> str:
    if PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_HELPER_NAME in text:
        return PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_HELPER_RE.sub(
            "\n\n" + PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_immediate_left_knee_survival_helper(text: str) -> str:
    if PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_NAME in text:
        return PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_RE.sub(
            "\n\n" + PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_immediate_left_knee_long_survival_helper(text: str) -> str:
    if PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_HELPER_NAME in text:
        return PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_HELPER_RE.sub(
            "\n\n" + PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_HELPER_BLOCK.strip() + "\n"


def _ensure_pose_tight_gate_left_knee_long_tail_helper(text: str) -> str:
    if POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_NAME in text:
        return POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_RE.sub(
            "\n\n" + POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_BLOCK.strip() + "\n"


def _ensure_persistent_hidden_startup_boost_helper(text: str) -> str:
    if PERSISTENT_HIDDEN_STARTUP_BOOST_HELPER_NAME in text:
        return PERSISTENT_HIDDEN_STARTUP_BOOST_HELPER_RE.sub(
            "\n\n" + PERSISTENT_HIDDEN_STARTUP_BOOST_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + PERSISTENT_HIDDEN_STARTUP_BOOST_HELPER_BLOCK.strip() + "\n"


def _remove_existing_task_registration(text: str) -> str:
    return TASK_REGISTER_RE.sub("\n\n", text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    init_path = args.root / "src/tasks/velocity/config/g1_gripper/__init__.py"
    rewards_path = args.root / "src/tasks/velocity/mdp/rewards.py"
    terminations_path = args.root / "src/tasks/velocity/mdp/terminations.py"
    patch_init(init_path)
    patch_rewards(rewards_path)
    patch_terminations(terminations_path)
    print(init_path)
    print(rewards_path)
    print(terminations_path)


if __name__ == "__main__":
    main()
