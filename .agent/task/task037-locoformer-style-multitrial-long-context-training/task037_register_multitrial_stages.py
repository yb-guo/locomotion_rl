#!/usr/bin/env python3
"""Patch H200 MJLab with Task037 multi-trial smoke task ids."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path("/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab")


RUNNER_IMPORT = (
    "from h200_locomotion_lab.training.rsl_history_wrapper import (\n"
    "  Task037BufferOnlyK4AutoResetRunner,\n"
    "  Task037BufferOnlyK4DeterministicInnerResetRunner,\n"
    ")\n"
)


REGISTERS = {
    "Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-AutoReset-Fast2p0": r'''
register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-AutoReset-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task031_unified_dynamic_switch_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task031_unified_dynamic_switch_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task037BufferOnlyK4AutoResetRunner,
)
''',
    "Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-DeterministicInnerReset-Fast2p0": r'''
register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-DeterministicInnerReset-Fast2p0",
  env_cfg=unitree_g1_gripper_flat_task031_unified_dynamic_switch_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task031_unified_dynamic_switch_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task037BufferOnlyK4DeterministicInnerResetRunner,
)
''',
}


def patch_init(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "Task037BufferOnlyK4DeterministicInnerResetRunner" not in text:
        anchor = "from mjlab.rl import MjlabOnPolicyRunner\n"
        if anchor not in text:
            raise RuntimeError(f"anchor not found in {path}: {anchor!r}")
        text = text.replace(anchor, anchor + RUNNER_IMPORT, 1)
    for task_id, block in REGISTERS.items():
        if task_id not in text:
            text = text.rstrip() + "\n\n" + block.strip() + "\n"
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    init_path = args.root / "src/tasks/velocity/config/g1_gripper/__init__.py"
    patch_init(init_path)
    print(init_path)


if __name__ == "__main__":
    main()
