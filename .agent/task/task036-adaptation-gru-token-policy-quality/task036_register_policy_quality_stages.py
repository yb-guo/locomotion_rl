#!/usr/bin/env python3
"""Patch H200 MJLab with Task036 policy-quality task ids."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path("/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab")


RUNNER_IMPORT = (
    "from h200_locomotion_lab.training.rsl_history_wrapper import (\n"
    "  Task033GruK4Runner,\n"
    "  Task033TokenK4Runner,\n"
    "  Task036AdaptK4Runner,\n"
    ")\n"
)


REGISTERS = {
    "Unitree-G1-Gripper-Flat-Task036-AdaptK4-Fast2p0": r'''
register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task036-AdaptK4-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task031_unified_dynamic_switch_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task031_unified_dynamic_switch_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task036AdaptK4Runner,
)
''',
    "Unitree-G1-Gripper-Flat-Task036-AdaptK4-FocusedDeadGrid-Fast2p0": r'''
register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task036-AdaptK4-FocusedDeadGrid-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task031_focused_deadgrid_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task031_focused_deadgrid_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task036AdaptK4Runner,
)
''',
    "Unitree-G1-Gripper-Flat-Task036-AdaptK4-DynamicMotorFailure-Fast1p6": r'''
register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task036-AdaptK4-DynamicMotorFailure-Fast1p6",
  env_cfg=unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task036AdaptK4Runner,
)
''',
    "Unitree-G1-Gripper-Flat-Task036-GruK4-Fast2p0": r'''
register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task036-GruK4-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task031_unified_dynamic_switch_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task031_unified_dynamic_switch_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task033GruK4Runner,
)
''',
    "Unitree-G1-Gripper-Flat-Task036-GruK4-FocusedDeadGrid-Fast2p0": r'''
register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task036-GruK4-FocusedDeadGrid-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task031_focused_deadgrid_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task031_focused_deadgrid_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task033GruK4Runner,
)
''',
    "Unitree-G1-Gripper-Flat-Task036-GruK4-DynamicMotorFailure-Fast1p6": r'''
register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task036-GruK4-DynamicMotorFailure-Fast1p6",
  env_cfg=unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task033GruK4Runner,
)
''',
    "Unitree-G1-Gripper-Flat-Task036-TokenK4-Fast2p0": r'''
register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task036-TokenK4-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task031_unified_dynamic_switch_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task031_unified_dynamic_switch_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task033TokenK4Runner,
)
''',
    "Unitree-G1-Gripper-Flat-Task036-TokenK4-FocusedDeadGrid-Fast2p0": r'''
register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task036-TokenK4-FocusedDeadGrid-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task031_focused_deadgrid_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task031_focused_deadgrid_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task033TokenK4Runner,
)
''',
    "Unitree-G1-Gripper-Flat-Task036-TokenK4-DynamicMotorFailure-Fast1p6": r'''
register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task036-TokenK4-DynamicMotorFailure-Fast1p6",
  env_cfg=unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task033TokenK4Runner,
)
''',
}


def patch_init(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "Task036AdaptK4Runner" not in text:
        anchor = "from mjlab.rl import MjlabOnPolicyRunner\n"
        if anchor not in text:
            raise RuntimeError(f"anchor not found in {path}: {anchor!r}")
        text = text.replace(anchor, anchor + RUNNER_IMPORT, 1)
    for task_id, block in REGISTERS.items():
        if task_id not in text:
            text = text.rstrip() + "\n\n" + block.strip() + "\n"
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_path = args.root / "src/tasks/velocity/config/g1_gripper/__init__.py"
    patch_init(init_path)
    print(init_path)


if __name__ == "__main__":
    main()
