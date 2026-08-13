#!/usr/bin/env python3
"""Create the task028 G1 gripper prototype inside upstream Unitree MJLab."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path("/home/xyzl/yubo/locomotion_rl/external/unitree_rl_mjlab")


GRIPPER_CONSTANTS = '''"""G1-like whole-body robot with two simple gripper joints."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from src.assets.robots.unitree_g1.g1_constants import (
  FULL_COLLISION,
  G1_ACTION_SCALE,
  G1_ACTUATOR_4010,
  G1_ACTUATOR_5020,
  G1_ACTUATOR_7520_14,
  G1_ACTUATOR_7520_22,
  G1_ACTUATOR_ANKLE,
  G1_ACTUATOR_WAIST,
  HOME_KEYFRAME,
)

##
# MJCF and assets.
##

G1_GRIPPER_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1_gripper" / "xmls" / "g1_gripper.xml"
)
assert G1_GRIPPER_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, G1_GRIPPER_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_GRIPPER_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

G1_GRIPPER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_gripper_joint",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=5.0,
  armature=1.0e-4,
)

G1_GRIPPER_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
    G1_GRIPPER_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_gripper_robot_cfg() -> EntityCfg:
  """Get a fresh G1 gripper robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_GRIPPER_ARTICULATION,
  )


G1_GRIPPER_BODY_ACTION_SCALE: dict[str, float] = dict(G1_ACTION_SCALE)
G1_GRIPPER_ACTION_SCALE: dict[str, float] = {".*_gripper_joint": 0.018}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_gripper_robot_cfg())

  viewer.launch(robot.spec.compile())
'''


GRIPPER_ASSET_INIT = '''from .g1_gripper_constants import (
  G1_GRIPPER_ACTION_SCALE as G1_GRIPPER_ACTION_SCALE,
  G1_GRIPPER_BODY_ACTION_SCALE as G1_GRIPPER_BODY_ACTION_SCALE,
)
from .g1_gripper_constants import (
  get_g1_gripper_robot_cfg as get_g1_gripper_robot_cfg,
)
'''


ENV_CFGS = '''"""Unitree G1 gripper velocity environment configurations."""

from src.assets.robots.unitree_g1_gripper.g1_gripper_constants import (
  G1_GRIPPER_ACTION_SCALE,
  G1_GRIPPER_BODY_ACTION_SCALE,
  get_g1_gripper_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg as _unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg as _unitree_g1_rough_env_cfg,
)


_BODY_ACTION_PATTERNS = tuple(G1_GRIPPER_BODY_ACTION_SCALE.keys())
_GRIPPER_ACTION_PATTERNS = tuple(G1_GRIPPER_ACTION_SCALE.keys())
_FOOT_GEOM_NAMES = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)
_BASE_BODY_NAMES = ("torso_link",)
_ALL_BODY_NAMES = (".*",)
_ALL_JOINT_NAMES = (".*",)
_ALL_ACTUATOR_IDS = list(range(7))

_CONTROL_EVENTS = ("reset_base", "reset_robot_joints")
_RANDOMIZATION_EVENTS = (
  "push_robot",
  "foot_friction",
  "encoder_bias",
  "base_com",
  "body_com_offset",
  "pseudo_inertia",
  "pd_gains",
  "effort_limits",
  "joint_damping",
  "joint_friction",
)


def _apply_gripper_overrides(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  cfg.scene.entities = {"robot": get_g1_gripper_robot_cfg()}

  cfg.actions = {
    "body_joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=_BODY_ACTION_PATTERNS,
      scale=G1_GRIPPER_BODY_ACTION_SCALE,
      use_default_offset=True,
    ),
    "gripper_joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=_GRIPPER_ACTION_PATTERNS,
      scale=G1_GRIPPER_ACTION_SCALE,
      use_default_offset=True,
    ),
  }

  # Keep the grippers near their neutral opening during locomotion without
  # introducing object or ground-contact manipulation objectives.
  for key in ("std_walking", "std_running"):
    cfg.rewards["pose"].params[key][r".*_gripper_joint"] = 0.03

  return cfg


def _strip_randomization(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  cfg.events = {
    name: event for name, event in cfg.events.items() if name in _CONTROL_EVENTS
  }
  cfg.observations["actor"].enable_corruption = False
  cfg.curriculum = {}
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = False
  return cfg


def _add_contact_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.events["foot_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=_FOOT_GEOM_NAMES),
      "operation": "abs",
      "ranges": (0.3, 1.6),
      "shared_random": True,
    },
  )


def _add_encoder_noise_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.observations["actor"].enable_corruption = True
  cfg.events["encoder_bias"] = EventTermCfg(
    mode="startup",
    func=dr.encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "bias_range": (-0.015, 0.015),
    },
  )


def _add_mass_com_inertia_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.events["body_com_offset"] = EventTermCfg(
    mode="startup",
    func=dr.body_com_offset,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=_BASE_BODY_NAMES),
      "operation": "add",
      "ranges": {
        0: (-0.04, 0.04),
        1: (-0.03, 0.03),
        2: (-0.03, 0.03),
      },
    },
  )
  cfg.events["pseudo_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=_ALL_BODY_NAMES),
      "alpha_range": (-0.05, 0.05),
      "d_range": (-0.03, 0.03),
      "t_range": (-0.02, 0.02),
    },
  )


def _add_motor_pd_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_ids=_ALL_ACTUATOR_IDS),
      "kp_range": (0.85, 1.15),
      "kd_range": (0.85, 1.15),
      "operation": "scale",
    },
  )
  cfg.events["effort_limits"] = EventTermCfg(
    mode="startup",
    func=dr.effort_limits,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_ids=_ALL_ACTUATOR_IDS),
      "effort_limit_range": (0.9, 1.1),
      "operation": "scale",
    },
  )
  cfg.events["joint_damping"] = EventTermCfg(
    mode="startup",
    func=dr.joint_damping,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=_ALL_JOINT_NAMES),
      "operation": "scale",
      "ranges": (0.8, 1.2),
    },
  )
  cfg.events["joint_friction"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=_ALL_JOINT_NAMES),
      "operation": "abs",
      "ranges": (0.0, 0.08),
    },
  )


def apply_randomization_stage(
  cfg: ManagerBasedRlEnvCfg, stage: str
) -> ManagerBasedRlEnvCfg:
  """Apply an explicit flat-terrain randomization stage."""
  _strip_randomization(cfg)

  if stage == "control":
    return cfg
  if stage == "contact":
    _add_contact_randomization(cfg)
    return cfg
  if stage == "encoder_noise":
    _add_encoder_noise_randomization(cfg)
    return cfg
  if stage == "mass_com_inertia":
    _add_mass_com_inertia_randomization(cfg)
    return cfg
  if stage == "motor_pd":
    _add_motor_pd_randomization(cfg)
    return cfg
  if stage == "combined":
    _add_contact_randomization(cfg)
    _add_encoder_noise_randomization(cfg)
    _add_mass_com_inertia_randomization(cfg)
    _add_motor_pd_randomization(cfg)
    return cfg

  raise ValueError(f"unknown G1 gripper randomization stage: {stage}")


def unitree_g1_gripper_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 gripper rough terrain velocity configuration."""
  return _apply_gripper_overrides(_unitree_g1_rough_env_cfg(play=play))


def unitree_g1_gripper_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 gripper flat terrain velocity configuration."""
  return _apply_gripper_overrides(_unitree_g1_flat_env_cfg(play=play))


def unitree_g1_gripper_flat_stage_env_cfg(
  stage: str, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 gripper flat configuration for a randomization stage."""
  return apply_randomization_stage(unitree_g1_gripper_flat_env_cfg(play=play), stage)
'''


RL_CFG = '''"""RL configuration for Unitree G1 gripper velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_gripper_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 gripper velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_gripper_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
'''


TASK_INIT = '''from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl import MjlabOnPolicyRunner

from .env_cfgs import (
  unitree_g1_gripper_flat_env_cfg,
  unitree_g1_gripper_flat_stage_env_cfg,
  unitree_g1_gripper_rough_env_cfg,
)
from .rl_cfg import unitree_g1_gripper_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-Gripper-Rough",
  env_cfg=unitree_g1_gripper_rough_env_cfg(),
  play_env_cfg=unitree_g1_gripper_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Gripper-Flat",
  env_cfg=unitree_g1_gripper_flat_env_cfg(),
  play_env_cfg=unitree_g1_gripper_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)


_FLAT_RANDOMIZATION_STAGES = {
  "Control": "control",
  "Contact": "contact",
  "EncoderNoise": "encoder_noise",
  "MassComInertia": "mass_com_inertia",
  "MotorPd": "motor_pd",
  "Combined": "combined",
}

for _task_suffix, _stage in _FLAT_RANDOMIZATION_STAGES.items():
  register_mjlab_task(
    task_id=f"Unitree-G1-Gripper-Flat-{_task_suffix}",
    env_cfg=unitree_g1_gripper_flat_stage_env_cfg(_stage),
    play_env_cfg=unitree_g1_gripper_flat_stage_env_cfg(_stage, play=True),
    rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
  )
'''


LEFT_ANCHOR = '''                          <geom name="left_hand_collision" class="collision" size="0.035" rgba=".2 .6 .2 .2" fromto="0.07 0 0 0.15 -0.02 0"/>'''

LEFT_GRIPPER = '''                          <body name="left_gripper_base_link" pos="0.145 0 0">
                            <inertial pos="0 0 0" mass="0.04" diaginertia="1e-05 1e-05 1e-05"/>
                            <geom name="left_gripper_base_visual" type="box" size="0.02 0.015 0.018" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0" group="2"/>
                            <geom name="left_gripper_fixed_finger_visual" type="box" pos="0.035 -0.026 0" size="0.045 0.006 0.012" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0" group="2"/>
                            <body name="left_gripper_finger_link" pos="0.035 0.026 0">
                              <inertial pos="0 0 0" mass="0.02" diaginertia="5e-06 5e-06 5e-06"/>
                              <joint name="left_gripper_joint" type="slide" axis="0 1 0" range="-0.018 0.018" damping="0.1"/>
                              <geom name="left_gripper_moving_finger_visual" type="box" size="0.045 0.006 0.012" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0" group="2"/>
                            </body>
                          </body>'''

RIGHT_ANCHOR = '''                          <geom name="right_hand_collision" class="collision" size="0.035" rgba=".2 .6 .2 .2" fromto="0.07 0 0 0.15 0.02 0"/>'''

RIGHT_GRIPPER = '''                          <body name="right_gripper_base_link" pos="0.145 0 0">
                            <inertial pos="0 0 0" mass="0.04" diaginertia="1e-05 1e-05 1e-05"/>
                            <geom name="right_gripper_base_visual" type="box" size="0.02 0.015 0.018" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0" group="2"/>
                            <geom name="right_gripper_fixed_finger_visual" type="box" pos="0.035 0.026 0" size="0.045 0.006 0.012" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0" group="2"/>
                            <body name="right_gripper_finger_link" pos="0.035 -0.026 0">
                              <inertial pos="0 0 0" mass="0.02" diaginertia="5e-06 5e-06 5e-06"/>
                              <joint name="right_gripper_joint" type="slide" axis="0 -1 0" range="-0.018 0.018" damping="0.1"/>
                              <geom name="right_gripper_moving_finger_visual" type="box" size="0.045 0.006 0.012" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0" group="2"/>
                            </body>
                          </body>'''


def write(path: Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def build_xml(source_xml: Path) -> str:
  text = source_xml.read_text(encoding="utf-8")
  text = text.replace('model="g1_29dof_rev_1_0"', 'model="g1_gripper_31dof"', 1)
  if "left_gripper_joint" not in text:
    text = text.replace(LEFT_ANCHOR, LEFT_ANCHOR + "\n" + LEFT_GRIPPER, 1)
  if "right_gripper_joint" not in text:
    text = text.replace(RIGHT_ANCHOR, RIGHT_ANCHOR + "\n" + RIGHT_GRIPPER, 1)
  if "left_gripper_joint" not in text or "right_gripper_joint" not in text:
    raise RuntimeError("failed to insert gripper joints into G1 MJCF")
  return text


def main() -> None:
  source_asset = ROOT / "src" / "assets" / "robots" / "unitree_g1"
  target_asset = ROOT / "src" / "assets" / "robots" / "unitree_g1_gripper"
  source_xml = source_asset / "xmls" / "g1.xml"
  source_meshes = source_asset / "xmls" / "assets"
  target_meshes = target_asset / "xmls" / "assets"

  if not source_xml.exists():
    raise FileNotFoundError(source_xml)

  target_asset.mkdir(parents=True, exist_ok=True)
  shutil.copytree(source_meshes, target_meshes, dirs_exist_ok=True)
  write(target_asset / "__init__.py", GRIPPER_ASSET_INIT)
  write(target_asset / "g1_gripper_constants.py", GRIPPER_CONSTANTS)
  write(target_asset / "xmls" / "g1_gripper.xml", build_xml(source_xml))

  task_dir = ROOT / "src" / "tasks" / "velocity" / "config" / "g1_gripper"
  write(task_dir / "__init__.py", TASK_INIT)
  write(task_dir / "env_cfgs.py", ENV_CFGS)
  write(task_dir / "rl_cfg.py", RL_CFG)

  print(target_asset / "xmls" / "g1_gripper.xml")
  print(task_dir / "__init__.py")


if __name__ == "__main__":
  main()
