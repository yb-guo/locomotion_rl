#!/usr/bin/env python3
"""Register self-contained Task048 clean gait reproduction stages."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path("/home/xyzl/yubo/locomotion_rl/external/unitree_rl_mjlab")

MLP_TRAIN_TASK_ID = "Unitree-G1-Gripper-Flat-Task048-Mlp-CleanBins-Train"
MLP_CURRICULUM_TRAIN_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task048-Mlp-OfficialCurriculum-Train"
)
MLP_EVAL_TASK_ID = "Unitree-G1-Gripper-Flat-Task048-Mlp-CleanBins-Eval"
ADAPTK4_TRAIN_TASK_ID = "Unitree-G1-Gripper-Flat-Task048-AdaptK4-CleanBins-Train"
ADAPTK4_EVAL_TASK_ID = "Unitree-G1-Gripper-Flat-Task048-AdaptK4-CleanBins-Eval"
ADAPTK160_TASK_ID = "Unitree-G1-Gripper-Flat-Task048-AdaptK160-CleanBins"
TRUE_TXL_EVAL_TASK_ID = "Unitree-G1-Gripper-Flat-Task048-TrueTxl-CleanBins-Eval"

ENV_CFGS_REL = Path("src/tasks/velocity/config/g1_gripper/env_cfgs.py")
INIT_REL = Path("src/tasks/velocity/config/g1_gripper/__init__.py")
VELOCITY_COMMAND_REL = Path("src/tasks/velocity/mdp/velocity_command.py")

VELOCITY_COMMAND_RESAMPLE_OLD = (
    "    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)\n"
)
VELOCITY_COMMAND_RESAMPLE_NEW = '''\
    lin_vel_x_choices = self.cfg.lin_vel_x_choices
    if lin_vel_x_choices is None:
      self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    else:
      choices = torch.as_tensor(
        lin_vel_x_choices,
        device=self.device,
        dtype=self.vel_command_b.dtype,
      )
      weights = self.cfg.lin_vel_x_choice_weights
      if weights is None:
        choice_ids = torch.randint(
          len(lin_vel_x_choices),
          (len(env_ids),),
          device=self.device,
        )
      else:
        probabilities = torch.as_tensor(
          weights,
          device=self.device,
          dtype=torch.float32,
        )
        probabilities = probabilities / probabilities.sum()
        choice_ids = torch.multinomial(probabilities, len(env_ids), replacement=True)
      self.vel_command_b[env_ids, 0] = choices[choice_ids]
'''

VELOCITY_COMMAND_CFG_OLD = "  init_velocity_prob: float = 0.0\n"
VELOCITY_COMMAND_CFG_NEW = '''\
  init_velocity_prob: float = 0.0
  lin_vel_x_choices: tuple[float, ...] | None = None
  lin_vel_x_choice_weights: tuple[float, ...] | None = None
'''

ENV_CFG_HELPER_BLOCK = '''

TASK048_SPEED_BINS = (0.4, 1.2, 2.0)
TASK048_SPEED_BIN_WEIGHTS = (0.34, 0.33, 0.33)


def _configure_task048_clean_speed_bins(cfg: ManagerBasedRlEnvCfg) -> None:
  """Use the historical clean low/mid/high forward command matrix."""
  twist_cmd = cfg.commands["twist"]
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.ranges.lin_vel_x = (0.4, 2.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None
  twist_cmd.lin_vel_x_choices = TASK048_SPEED_BINS
  twist_cmd.lin_vel_x_choice_weights = TASK048_SPEED_BIN_WEIGHTS


def unitree_g1_gripper_flat_task048_clean_bins_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a clean G1 gripper env over explicit 0.4/1.2/2.0 m/s bins."""
  cfg = unitree_g1_gripper_flat_env_cfg(play=play)
  _strip_randomization(cfg)
  _configure_task048_clean_speed_bins(cfg)
  return cfg
'''

ENV_CFG_IMPORT_BLOCK = '''\
from .env_cfgs import unitree_g1_gripper_flat_task048_clean_bins_env_cfg
'''

RUNNER_NAMES = (
    "Task038TrueTxlMemoryK160Runner",
    "Task036AdaptK4Runner",
    "Task037AdaptK4DeterministicInnerResetRunner",
    "Task037AdaptK160DeterministicInnerResetRunner",
    "Task037BufferOnlyK4DeterministicInnerResetRunner",
)
RUNNER_IMPORT_BLOCK = '''\
from h200_locomotion_lab.training.rsl_history_wrapper import (
  Task036AdaptK4Runner,
  Task037AdaptK4DeterministicInnerResetRunner,
  Task037AdaptK160DeterministicInnerResetRunner,
  Task037BufferOnlyK4DeterministicInnerResetRunner,
)
'''
RUNNER_IMPORT_RE = re.compile(
    r"from\s+h200_locomotion_lab\.training\.rsl_history_wrapper\s+import\s*\(\s*\n"
    r"(?P<body>.*?)"
    r"(?P<footer>\n\s*\))",
    re.DOTALL,
)

REGISTER_BLOCKS = {
    MLP_CURRICULUM_TRAIN_TASK_ID: f'''
register_mjlab_task(
  task_id="{MLP_CURRICULUM_TRAIN_TASK_ID}",
  env_cfg=unitree_g1_gripper_flat_env_cfg(play=False),
  play_env_cfg=unitree_g1_gripper_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
''',
    MLP_TRAIN_TASK_ID: f'''
register_mjlab_task(
  task_id="{MLP_TRAIN_TASK_ID}",
  env_cfg=unitree_g1_gripper_flat_task048_clean_bins_env_cfg(play=False),
  play_env_cfg=unitree_g1_gripper_flat_task048_clean_bins_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
''',
    MLP_EVAL_TASK_ID: f'''
register_mjlab_task(
  task_id="{MLP_EVAL_TASK_ID}",
  env_cfg=unitree_g1_gripper_flat_task048_clean_bins_env_cfg(play=False),
  play_env_cfg=unitree_g1_gripper_flat_task048_clean_bins_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task037BufferOnlyK4DeterministicInnerResetRunner,
)
''',
    ADAPTK4_TRAIN_TASK_ID: f'''
register_mjlab_task(
  task_id="{ADAPTK4_TRAIN_TASK_ID}",
  env_cfg=unitree_g1_gripper_flat_task048_clean_bins_env_cfg(play=False),
  play_env_cfg=unitree_g1_gripper_flat_task048_clean_bins_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task036AdaptK4Runner,
)
''',
    ADAPTK4_EVAL_TASK_ID: f'''
register_mjlab_task(
  task_id="{ADAPTK4_EVAL_TASK_ID}",
  env_cfg=unitree_g1_gripper_flat_task048_clean_bins_env_cfg(play=False),
  play_env_cfg=unitree_g1_gripper_flat_task048_clean_bins_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task037AdaptK4DeterministicInnerResetRunner,
)
''',
    ADAPTK160_TASK_ID: f'''
register_mjlab_task(
  task_id="{ADAPTK160_TASK_ID}",
  env_cfg=unitree_g1_gripper_flat_task048_clean_bins_env_cfg(play=False),
  play_env_cfg=unitree_g1_gripper_flat_task048_clean_bins_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task037AdaptK160DeterministicInnerResetRunner,
)
''',
    TRUE_TXL_EVAL_TASK_ID: f'''
register_mjlab_task(
  task_id="{TRUE_TXL_EVAL_TASK_ID}",
  env_cfg=unitree_g1_gripper_flat_task048_clean_bins_env_cfg(play=False),
  play_env_cfg=unitree_g1_gripper_flat_task048_clean_bins_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task038TrueTxlMemoryK160Runner,
)
''',
}


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"anchor not found: {old[:80]!r}")
    return text.replace(old, new, 1)


def patch_velocity_command(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "lin_vel_x_choices = self.cfg.lin_vel_x_choices" not in text:
        text = replace_once(text, VELOCITY_COMMAND_RESAMPLE_OLD, VELOCITY_COMMAND_RESAMPLE_NEW)
    if "lin_vel_x_choice_weights: tuple[float, ...] | None" not in text:
        text = replace_once(text, VELOCITY_COMMAND_CFG_OLD, VELOCITY_COMMAND_CFG_NEW)
    path.write_text(text, encoding="utf-8")


def patch_env_cfgs(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "def unitree_g1_gripper_flat_task048_clean_bins_env_cfg(" not in text:
        text = text.rstrip() + ENV_CFG_HELPER_BLOCK + "\n"
    path.write_text(text, encoding="utf-8")


def _ensure_runner_import(text: str) -> str:
    match = RUNNER_IMPORT_RE.search(text)
    if match is None:
        return RUNNER_IMPORT_BLOCK + text
    for name in RUNNER_NAMES:
        match = RUNNER_IMPORT_RE.search(text)
        assert match is not None
        if name in match.group(0):
            continue
        footer_offset = match.start("footer") - match.start()
        replacement = (
            match.group(0)[:footer_offset]
            + f"\n  {name},"
            + match.group("footer")
        )
        text = text[: match.start()] + replacement + text[match.end() :]
    return text


def patch_init(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = _ensure_runner_import(text)
    if "unitree_g1_gripper_flat_task048_clean_bins_env_cfg" not in text:
        text = ENV_CFG_IMPORT_BLOCK + text
    for task_id, block in REGISTER_BLOCKS.items():
        if task_id not in text:
            text = text.rstrip() + "\n\n" + block.strip() + "\n"
    path.write_text(text, encoding="utf-8")


def patch_root(root: Path) -> list[Path]:
    targets = [root / VELOCITY_COMMAND_REL, root / ENV_CFGS_REL, root / INIT_REL]
    for target in targets:
        if not target.exists():
            raise FileNotFoundError(target)
    patch_velocity_command(targets[0])
    patch_env_cfgs(targets[1])
    patch_init(targets[2])
    return targets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    for target in patch_root(args.root):
        print(target)


if __name__ == "__main__":
    main()
