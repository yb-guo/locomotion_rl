#!/usr/bin/env python3
"""Patch H200 MJLab with the Task039 MLP clean-baseline task id."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path("/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab")

TASK039_MLP_CLEAN_TASK_ID = "Unitree-G1-Gripper-Flat-Task039-MlpClean-Train"
INIT_REL = Path("src/tasks/velocity/config/g1_gripper/__init__.py")

ENV_CFG_IMPORT_BLOCK = """\
from .env_cfgs import (
  unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg,
)
"""

RUNNER_IMPORT_BLOCK = """\
from h200_locomotion_lab.training.rsl_history_wrapper import (
  Task037BufferOnlyK4DeterministicInnerResetRunner,
)
"""

RUNNER_IMPORT_RE = re.compile(
    r"from\s+h200_locomotion_lab\.training\.rsl_history_wrapper\s+import\s*\(\s*\n"
    r"(?P<body>.*?)"
    r"(?P<footer>\n\s*\))",
    re.DOTALL,
)

REGISTER_BLOCK = f'''
register_mjlab_task(
  task_id="{TASK039_MLP_CLEAN_TASK_ID}",
  env_cfg=unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg(play=False),
  play_env_cfg=unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=Task037BufferOnlyK4DeterministicInnerResetRunner,
)
'''


def _ensure_runner_import(text: str) -> str:
    match = RUNNER_IMPORT_RE.search(text)
    if match is None:
        return RUNNER_IMPORT_BLOCK + text
    if "Task037BufferOnlyK4DeterministicInnerResetRunner" in match.group(0):
        return text
    footer_offset = match.start("footer") - match.start()
    insertion = (
        match.group(0)[:footer_offset]
        + "\n  Task037BufferOnlyK4DeterministicInnerResetRunner,"
        + match.group("footer")
    )
    return text[: match.start()] + insertion + text[match.end() :]


def patch_init(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = _ensure_runner_import(text)
    if "unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg" not in text:
        text = ENV_CFG_IMPORT_BLOCK + text
    if TASK039_MLP_CLEAN_TASK_ID not in text:
        text = text.rstrip() + "\n\n" + REGISTER_BLOCK.strip() + "\n"
    path.write_text(text, encoding="utf-8")


def patch_root(root: Path) -> list[Path]:
    init_path = root / INIT_REL
    if not init_path.exists():
        raise FileNotFoundError(init_path)
    patch_init(init_path)
    return [init_path]


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
