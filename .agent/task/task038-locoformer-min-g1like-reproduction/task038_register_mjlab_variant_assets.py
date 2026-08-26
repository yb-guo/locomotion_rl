#!/usr/bin/env python3
"""Patch external Unitree MJLab with Task038 XML-variant load-smoke tasks."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path("/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab")

TRAIN_TASK_ID = "Unitree-G1-Gripper-Flat-Task038-TrainAssetSmoke"
HELDOUT_TASK_ID = "Unitree-G1-Gripper-Flat-Task038-HeldoutAssetSmoke"
TRAIN_RUNNER_SMOKE_TASK_ID = "Unitree-G1-Gripper-Flat-Task038-TrainRunnerSmoke"
HELDOUT_RUNNER_SMOKE_TASK_ID = "Unitree-G1-Gripper-Flat-Task038-HeldoutRunnerSmoke"
TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke"
)
HELDOUT_TRUE_TXL_RUNNER_SMOKE_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task038-HeldoutTrueTxlRunnerSmoke"
)

TASK038_TRAIN_XML = (
    "/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/"
    "outputs/task038/g1like_compile_smoke/variants/g1like-train-none-e6ba46370d.xml"
)
TASK038_HELDOUT_XML = (
    "/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/"
    "outputs/task038/g1like_compile_smoke/variants/g1like-heldout-combined-6ac730c265.xml"
)

CONSTANTS_REL = Path("src/assets/robots/unitree_g1_gripper/g1_gripper_constants.py")
ENV_CFGS_REL = Path("src/tasks/velocity/config/g1_gripper/env_cfgs.py")
INIT_REL = Path("src/tasks/velocity/config/g1_gripper/__init__.py")


CONSTANTS_BLOCK = f'''

##
# Task038 external XML variant smoke assets.
##

TASK038_TRAIN_XML: Path = Path("{TASK038_TRAIN_XML}")
TASK038_HELDOUT_XML: Path = Path("{TASK038_HELDOUT_XML}")


def get_task038_variant_spec(xml_path: str | Path) -> mujoco.MjSpec:
  """Load a Task038 patched XML while preserving the G1 gripper articulation."""
  variant_xml = Path(xml_path).expanduser()
  if not variant_xml.exists():
    raise FileNotFoundError(f"Task038 variant XML does not exist: {{variant_xml}}")
  spec = mujoco.MjSpec.from_file(str(variant_xml))
  meshdir = Path(spec.meshdir) if spec.meshdir else None
  if meshdir is not None:
    asset_dir = meshdir if meshdir.is_absolute() else variant_xml.parent / meshdir
    if asset_dir.exists():
      spec.assets = {{}}
      update_assets(spec.assets, asset_dir, spec.meshdir)
  return spec


def get_g1_gripper_robot_cfg_for_xml(xml_path: str | Path) -> EntityCfg:
  """Get a fresh G1 gripper config whose spec_fn loads the given XML path."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=lambda: get_task038_variant_spec(xml_path),
    articulation=G1_GRIPPER_ARTICULATION,
  )
'''


ENV_CFG_IMPORT_BLOCK = """\
from src.assets.robots.unitree_g1_gripper.g1_gripper_constants import (
  TASK038_HELDOUT_XML,
  TASK038_TRAIN_XML,
  get_g1_gripper_robot_cfg_for_xml,
)
"""

ENV_CFG_HELPER_BLOCK = '''

def _apply_task038_variant_xml(
  cfg: ManagerBasedRlEnvCfg, xml_path: str | Path
) -> ManagerBasedRlEnvCfg:
  cfg.scene.entities = {"robot": get_g1_gripper_robot_cfg_for_xml(xml_path)}
  return cfg


def unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a flat G1 gripper env that loads the Task038 train patched XML."""
  return _apply_task038_variant_xml(
    unitree_g1_gripper_flat_env_cfg(play=play), TASK038_TRAIN_XML
  )


def unitree_g1_gripper_flat_task038_heldout_asset_smoke_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a flat G1 gripper env that loads the Task038 held-out patched XML."""
  return _apply_task038_variant_xml(
    unitree_g1_gripper_flat_env_cfg(play=play), TASK038_HELDOUT_XML
  )
'''

INIT_IMPORT_BLOCK = """\
from .env_cfgs import (
  unitree_g1_gripper_flat_task038_heldout_asset_smoke_env_cfg,
  unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg,
)
"""

INIT_RUNNER_IMPORT_BLOCK = """\
from h200_locomotion_lab.training.rsl_history_wrapper import (
  Task037TxlMemoryK160DeterministicRunner,
  Task038TrueTxlMemoryK160Runner,
)
"""

RUNNER_IMPORT_RE = re.compile(
    r"from\s+h200_locomotion_lab\.training\.rsl_history_wrapper\s+import\s*\(\s*\n"
    r"(?P<body>.*?)"
    r"(?P<footer>\n\s*\))",
    re.DOTALL,
)

REGISTER_BLOCKS = {
    TRAIN_TASK_ID: f'''
register_mjlab_task(
  task_id="{TRAIN_TASK_ID}",
  env_cfg=unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
''',
    HELDOUT_TASK_ID: f'''
register_mjlab_task(
  task_id="{HELDOUT_TASK_ID}",
  env_cfg=unitree_g1_gripper_flat_task038_heldout_asset_smoke_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task038_heldout_asset_smoke_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
''',
    TRAIN_RUNNER_SMOKE_TASK_ID: f'''
register_mjlab_task(
  task_id="{TRAIN_RUNNER_SMOKE_TASK_ID}",
  env_cfg=unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(
  ),
  runner_cls=Task037TxlMemoryK160DeterministicRunner,
)
''',
    HELDOUT_RUNNER_SMOKE_TASK_ID: f'''
register_mjlab_task(
  task_id="{HELDOUT_RUNNER_SMOKE_TASK_ID}",
  env_cfg=unitree_g1_gripper_flat_task038_heldout_asset_smoke_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_task038_heldout_asset_smoke_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(
  ),
  runner_cls=Task037TxlMemoryK160DeterministicRunner,
)
''',
    TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID: f'''
register_mjlab_task(
  task_id="{TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID}",
  env_cfg=unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg(play=False),
  play_env_cfg=unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(
  ),
  runner_cls=Task038TrueTxlMemoryK160Runner,
)
''',
    HELDOUT_TRUE_TXL_RUNNER_SMOKE_TASK_ID: f'''
register_mjlab_task(
  task_id="{HELDOUT_TRUE_TXL_RUNNER_SMOKE_TASK_ID}",
  env_cfg=unitree_g1_gripper_flat_task038_heldout_asset_smoke_env_cfg(play=False),
  play_env_cfg=unitree_g1_gripper_flat_task038_heldout_asset_smoke_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(
  ),
  runner_cls=Task038TrueTxlMemoryK160Runner,
)
''',
}


def patch_constants(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "TASK038_TRAIN_XML" not in text:
        text = text.rstrip() + CONSTANTS_BLOCK + "\n"
    path.write_text(text, encoding="utf-8")


def patch_env_cfgs(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "from pathlib import Path" not in text:
        text = 'from pathlib import Path\n' + text
    if "get_g1_gripper_robot_cfg_for_xml" not in text:
        text = ENV_CFG_IMPORT_BLOCK + text
    if "def unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg" not in text:
        text = text.rstrip() + ENV_CFG_HELPER_BLOCK + "\n"
    path.write_text(text, encoding="utf-8")


def _ensure_runner_import(text: str) -> str:
    match = RUNNER_IMPORT_RE.search(text)
    if match is None:
        return INIT_RUNNER_IMPORT_BLOCK + text
    if "Task038TrueTxlMemoryK160Runner" in match.group(0):
        return text
    footer_offset = match.start("footer") - match.start()
    insertion = (
        match.group(0)[:footer_offset]
        + "\n  Task038TrueTxlMemoryK160Runner,"
        + match.group("footer")
    )
    return text[: match.start()] + insertion + text[match.end() :]


def patch_init(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = _ensure_runner_import(text)
    if "unitree_g1_gripper_flat_task038_train_asset_smoke_env_cfg" not in text:
        text = INIT_IMPORT_BLOCK + text
    for task_id, block in REGISTER_BLOCKS.items():
        if task_id not in text:
            text = text.rstrip() + "\n\n" + block.strip() + "\n"
    path.write_text(text, encoding="utf-8")


def patch_root(root: Path) -> list[Path]:
    targets = [root / CONSTANTS_REL, root / ENV_CFGS_REL, root / INIT_REL]
    for target in targets:
        if not target.exists():
            raise FileNotFoundError(target)
    patch_constants(targets[0])
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
