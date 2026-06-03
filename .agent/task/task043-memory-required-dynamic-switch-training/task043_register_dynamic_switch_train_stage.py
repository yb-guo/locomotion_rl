#!/usr/bin/env python3
"""Patch H200 MJLab with the Task043 true-TXL dynamic-switch train task."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path("/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab")
TASK_ID = "Unitree-G1-Gripper-Flat-Task043-TrainTrueTxlDynamicSwitchMemoryRequired-Fast1p6"
RUNNER_NAME = "Task038TrueTxlMemoryK160Runner"
FIXED_SPEED_HELPER_NAME = "_task043_dynamic_failure_fixed1p6_env_cfg"

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
    rf'\s*task_id="{re.escape(TASK_ID)}",'
    r".*?\n\)\n*",
    re.DOTALL,
)

FIXED_SPEED_HELPER_RE = re.compile(
    rf"\n*def {FIXED_SPEED_HELPER_NAME}\(play: bool = False\):\n"
    r".*?\n  return cfg\n*",
    re.DOTALL,
)

FIXED_SPEED_HELPER_BLOCK = f'''
def {FIXED_SPEED_HELPER_NAME}(play: bool = False):
  cfg = unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg(play=play)
  twist_cmd = cfg.commands["twist"]
  twist_cmd.ranges.lin_vel_x = (1.6, 1.6)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
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
'''


def patch_init(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = _ensure_runner_import(text)
    text = _ensure_fixed_speed_helper(text)
    text = _remove_existing_task_registration(text)
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


def _ensure_fixed_speed_helper(text: str) -> str:
    if FIXED_SPEED_HELPER_NAME in text:
        return FIXED_SPEED_HELPER_RE.sub(
            "\n\n" + FIXED_SPEED_HELPER_BLOCK.strip() + "\n\n",
            text,
            count=1,
        )
    return text.rstrip() + "\n\n" + FIXED_SPEED_HELPER_BLOCK.strip() + "\n"


def _remove_existing_task_registration(text: str) -> str:
    return TASK_REGISTER_RE.sub("\n\n", text)


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
