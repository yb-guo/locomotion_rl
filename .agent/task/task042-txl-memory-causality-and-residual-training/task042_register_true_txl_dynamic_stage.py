#!/usr/bin/env python3
"""Patch H200 MJLab with the Task042 true-TXL dynamic-failure eval task."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path("/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab")
TASK_ID = "Unitree-G1-Gripper-Flat-Task042-TrainTrueTxlDynamicMotorFailure-Fast1p6"
RUNNER_NAME = "Task038TrueTxlMemoryK160Runner"

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

REGISTER_BLOCK = f'''
register_mjlab_task(
  task_id="{TASK_ID}",
  env_cfg=unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(
  ),
  runner_cls={RUNNER_NAME},
)
'''


def patch_init(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = _ensure_runner_import(text)
    if TASK_ID not in text:
        text = text.rstrip() + "\n\n" + REGISTER_BLOCK.strip() + "\n"
    path.write_text(text, encoding="utf-8")


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
