"""Task070 archetype-constrained morphology profile.

This profile is intentionally independent from the legacy whole-body generator
and the Task069 paper-faithful envelope.  It uses anonymous primitive geometry
whose ratios are sampled around the R0-audited multi-vendor prior set.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from h200_locomotion_lab.robots.procedural_morphology import (
    ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_HASH,
    ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_VERSION,
    ARCHETYPE_CONSTRAINED_MORPHOLOGY_PROFILE_VERSION,
    MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_HASH,
    MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_VERSION,
    MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_PROFILE_VERSION,
    ActuatorBlueprint,
    JointBlueprint,
    LinkBlueprint,
    MorphologyBlueprint,
    MorphologyFamily,
    PhysicalParams,
    WheelBlueprint,
)
from h200_locomotion_lab.robots.whole_body_slots import WHOLE_BODY_SLOT_NAMES

TASK070_REFERENCE_REGISTRY_SHA256 = (
    "931bf5346fe379b3fd1c25e91d39007704e06208a4b2c93b38b96293a58922f1"
)
TASK070_SOURCE_LICENSE_MATRIX_SHA256 = (
    "1d1c243797cb721942a88445a2ea8c4f9ee30b4237eb4c6e9214850fa1c3465a"
)
TASK070_R0_DESIGN_CONTRACT_SHA256 = (
    "5d9bc169681984d5c9682cec6bbaa2e031e82c9eda4439ee716f95d28ae2cf7d"
)
TASK070_PRIOR_SET_ID = "de203b60c4e3f0abd4f3880196efe7e589879b4006996b340319209411d3bf79"
TASK070_DISTANCE_CONTRACT_HASH = (
    "a488cde1c0bd86e1fdcf851b69c691121e114424d3cdbb526660ec65354442a1"
)
TASK070_STANCE_CONTRACT_HASH = hashlib.sha256(
    json.dumps(
        {
            "version": "task070_contact_aware_stance_v1",
            "timestep_seconds": 0.002,
            "stance_hold_steps": 1000,
            "controller": "biped_base_attitude_hold_or_quadruped_position_feedforward_with_wheel_balance",
            "wheel_velocity_hold_gain": 4.0,
            "biped_attitude_hold_hip_pitch_gain": 1.0,
            "biped_attitude_hold_ankle_pitch_gain": 0.5,
            "wheeled_biped_balance_pitch_gain": 400.0,
            "wheeled_biped_balance_pitch_rate_gain": 20.0,
            "max_retry_attempts": 6,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()
REPO_ROOT = Path(__file__).resolve().parents[3]
TASK070_G1_SOURCE_PATH = (
    REPO_ROOT / ".external" / "unitree_rl_mjlab" / "src" / "assets" / "robots"
    / "unitree_g1" / "xmls" / "g1.xml"
)
TASK070_G1_MOTOR_CONFIG_PATH = (
    REPO_ROOT / ".external" / "unitree_rl_mjlab" / "src" / "assets" / "robots"
    / "unitree_g1" / "g1_constants.py"
)
TASK070_G1_MOTOR_CONFIG_SHA256 = (
    "136b59af97082a74fd3a2a4250bc2e290b3ba6a26533b1492be52114dd844c5d"
)
TASK070_PM01_SOURCE_PATH = (
    REPO_ROOT
    / ".external"
    / "task070_reference_sources"
    / "engineai_amp"
    / "serial_pm01.urdf"
)
TASK070_PM01_NATIVE_CONFIG_ROOT = (
    REPO_ROOT
    / ".external"
    / "task070_reference_sources"
    / "engineai_robotics_native_sdk"
    / "assets"
    / "config"
    / "pm01_edu"
)
TASK070_PM01_NATIVE_MOTOR_CONFIG_PATH = (
    TASK070_PM01_NATIVE_CONFIG_ROOT / "motor" / "default.yaml"
)
TASK070_PM01_NATIVE_MOTOR_CONFIG_SHA256 = (
    "64c9616773f9ffa4730dede808a01a609c6dd8e53c50b50c7fa1f32bd906cc5e"
)
TASK070_PM01_NATIVE_TRANSFORM_CONFIG_PATH = (
    TASK070_PM01_NATIVE_CONFIG_ROOT
    / "joint_motor_transform_runner"
    / "default.yaml"
)
TASK070_PM01_NATIVE_TRANSFORM_CONFIG_SHA256 = (
    "5392901182bf0efbf8ae1d835a407bc014ceac49dec428f3fee2395499a33cb4"
)
TASK070_PM01_NATIVE_PARALLEL_ANKLE_CONFIG_PATH = (
    TASK070_PM01_NATIVE_CONFIG_ROOT / "parallel_ankle" / "default.yaml"
)
TASK070_PM01_NATIVE_PARALLEL_ANKLE_CONFIG_SHA256 = (
    "08bfacd58fa933b19546f8b161012034d1c76a8a4773a1f4eccc9e5720eb0d8e"
)
TASK070_PM01_NATIVE_RL_CONFIG_PATH = (
    TASK070_PM01_NATIVE_CONFIG_ROOT / "rl_lab" / "default.yaml"
)
TASK070_PM01_NATIVE_RL_CONFIG_SHA256 = (
    "cc44d1694b0b0c014d4b4fab9b70cbcd8e10f4c8634257791ad7bde64b969ff3"
)
TASK070_QUADRUPED_SOURCE_PATHS: Mapping[str, Path] = {
    "spot_base": (
        REPO_ROOT
        / ".external"
        / "task070_reference_sources"
        / "boston_dynamics_spot_sdk"
        / "spot_base_model.urdf"
    ),
    "unitree_go2": (
        REPO_ROOT
        / ".external"
        / "unitree_rl_mjlab"
        / "src"
        / "assets"
        / "robots"
        / "unitree_go2"
        / "xmls"
        / "go2.xml"
    ),
    "deeprobotics_lite3": (
        REPO_ROOT
        / ".external"
        / "task070_reference_sources"
        / "deep_robotics_model"
        / "Lite3.urdf"
    ),
}
TASK070_GO2_MOTOR_CONFIG_PATH = (
    REPO_ROOT / ".external" / "unitree_rl_mjlab" / "src" / "assets" / "robots"
    / "unitree_go2" / "go2_constants.py"
)
TASK070_GO2_MOTOR_CONFIG_SHA256 = (
    "c1406f6e469b8b6bb24bb7f9b122612c74cec412a8b9b5d9cddf64e37fb19a0f"
)

TASK070_ADDITIONAL_HUMANOID_SOURCES: Mapping[str, Mapping[str, object]] = {
    "agibot_x1_serial": {
        "source_path": (
            REPO_ROOT
            / ".external/task070_reference_sources/agibot_x1_infer/src/module/"
            "sim_module/model/mjcf/robot/xyber_x1/xyber_x1_serial.xml"
        ),
        "source_format": "MJCF",
        "expected_motor_count": 29,
        "configured_physical_motor_count": 31,
        "evidence_paths": (
            REPO_ROOT
            / ".external/task070_reference_sources/agibot_x1_infer/src/module/"
            "dcu_driver_module/cfg/dcu_x1.yaml",
            REPO_ROOT
            / ".external/task070_reference_sources/agibot_x1_infer/src/module/"
            "dcu_driver_module/src/ankle_transmission.cc",
            REPO_ROOT
            / ".external/task070_reference_sources/agibot_x1_infer/src/module/"
            "dcu_driver_module/src/lumbar_transmission.cc",
            REPO_ROOT
            / ".external/task070_reference_sources/agibot_x1_infer/src/module/"
            "dcu_driver_module/src/wrist_transmission.cc",
        ),
        "missing_for_promotion": (
            "serial MJCF omits the two configured claw joints",
            "nonlinear ankle lookup table was not retained",
            "L28 current-to-torque constant and claw limits are unavailable",
        ),
    },
    "agibot_x2_ultra": {
        "source_path": (
            REPO_ROOT
            / ".external/task070_reference_sources/agibot_x2_urdf/"
            "X2_URDF-v1.4.0/X2-Ultra.urdf"
        ),
        "source_format": "URDF",
        "expected_motor_count": 31,
        "configured_physical_motor_count": 31,
        "evidence_paths": (
            REPO_ROOT
            / ".external/task070_reference_sources/agibot_x2_urdf/"
            "X2_URDF-v1.4.0/X2-Ultra.xml",
        ),
        "missing_for_promotion": (
            "physical transmission mapping is not published in the selected source",
            "motor family, rotor inertia, and controller gains are unavailable",
        ),
    },
    "engineai_t800": {
        "source_path": (
            REPO_ROOT
            / ".external/task070_reference_sources/engineai_robotics_native_sdk/"
            "assets/resource/robot/t800/urdf/serial_t800.urdf"
        ),
        "source_format": "URDF",
        "expected_motor_count": 25,
        "configured_physical_motor_count": 25,
        "evidence_paths": tuple(
            REPO_ROOT
            / ".external/task070_reference_sources/engineai_robotics_native_sdk/"
            f"assets/config/t800/{relative}"
            for relative in (
                "motor/default.yaml",
                "joint_motor_transform_runner/default.yaml",
                "parallel_ankle/default.yaml",
                "pd_stand/default.yaml",
            )
        ),
        "missing_for_promotion": (
            "physical motor family names and rotor inertia are unavailable",
            "joint-order/config alignment remains candidate-only",
        ),
    },
    "engineai_t800pro": {
        "source_path": (
            REPO_ROOT
            / ".external/task070_reference_sources/engineai_robotics_native_sdk/"
            "assets/resource/robot/t800pro/urdf/serial_t800pro.urdf"
        ),
        "source_format": "URDF",
        "expected_motor_count": 43,
        "configured_physical_motor_count": 43,
        "evidence_paths": tuple(
            REPO_ROOT
            / ".external/task070_reference_sources/engineai_robotics_native_sdk/"
            f"assets/config/t800pro/{relative}"
            for relative in (
                "motor/default.yaml",
                "joint_motor_transform_runner/default.yaml",
                "parallel_ankle/default.yaml",
                "parallel_palms/default.yaml",
                "pd_stand/default.yaml",
            )
        ),
        "missing_for_promotion": (
            "parallel-palm forward_net.mnn was not downloaded",
            "physical motor family names and rotor inertia are unavailable",
            "joint-order/config alignment remains candidate-only",
        ),
    },
    "limx_hu_d04": {
        "source_path": (
            REPO_ROOT
            / ".external/task070_reference_sources/limx_humanoid_description/"
            "HU_D04_description/urdf/HU_D04_01.urdf"
        ),
        "source_format": "URDF",
        "expected_motor_count": 31,
        "configured_physical_motor_count": 31,
        "evidence_paths": (
            REPO_ROOT
            / ".external/task070_reference_sources/limx_humanoid_description/"
            "HU_D04_description/urdf/HU_D04_01.srdf",
            REPO_ROOT
            / ".external/task070_reference_sources/limx_humanoid_description/"
            "HU_D04_description/xml/HU_D04_01.xml",
            REPO_ROOT
            / ".external/task070_reference_sources/limx_humanoid_rl_deploy_python/"
            "doc/parallel_joint_mapping_en.md",
            REPO_ROOT
            / ".external/task070_reference_sources/limx_humanoid_rl_deploy_python/"
            "controllers/HU_D04_01/walk_controller/walk_param.yaml",
        ),
        "missing_for_promotion": (
            "SDK-internal exact PR-to-AB runtime mapping code is unavailable",
            "descriptor and config alignment remains candidate-only",
        ),
    },
    "booster_t1_23": {
        "source_path": (
            REPO_ROOT
            / ".external/task070_reference_sources/booster_assets/robots/T1/"
            "T1_23dof.urdf"
        ),
        "source_format": "URDF",
        "expected_motor_count": 23,
        "configured_physical_motor_count": 23,
        "evidence_paths": (
            REPO_ROOT
            / ".external/task070_reference_sources/booster_assets/robots/T1/"
            "T1_23dof.xml",
        ),
        "missing_for_promotion": (
            "physical transmission and motor-family evidence are unavailable",
            "controller gains are unavailable",
        ),
    },
    "booster_t1_29": {
        "source_path": (
            REPO_ROOT
            / ".external/task070_reference_sources/booster_assets/robots/T1/"
            "T1_29dof.urdf"
        ),
        "source_format": "URDF",
        "expected_motor_count": 29,
        "configured_physical_motor_count": 29,
        "evidence_paths": (),
        "missing_for_promotion": (
            "physical transmission and motor-family evidence are unavailable",
            "MJCF actuator config and controller gains are unavailable",
        ),
    },
    "robotera_star1": {
        "source_path": (
            REPO_ROOT
            / ".external/task070_reference_sources/robotera_models/star1/urdf/"
            "l3_with_hand_fixedpin_xml.urdf"
        ),
        "source_format": "URDF",
        "expected_motor_count": 55,
        "configured_physical_motor_count": 55,
        "evidence_paths": (),
        "missing_for_promotion": (
            "MJCF actuator config and physical transmission are unavailable",
            "motor family, rotor inertia, and controller gains are unavailable",
        ),
    },
}
TASK070_ADDITIONAL_HUMANOID_REFERENCE_IDS = tuple(
    TASK070_ADDITIONAL_HUMANOID_SOURCES
)

Task070SamplingRegion = Literal[
    "prior_neighborhood",
    "interpolation_band",
    "bounded_outward_band",
]

FEATURE_NAMES = (
    "trunk_half_x_norm",
    "trunk_half_y_norm",
    "trunk_half_z_norm",
    "front_attachment_x_norm",
    "rear_attachment_x_norm",
    "lateral_attachment_y_norm",
    "leg_length_norm",
    "upper_link_fraction",
    "lower_link_fraction",
    "terminal_radius_or_foot_norm",
    "mean_joint_count_norm",
    "wheel_radius_norm",
)
FEATURE_WEIGHTS = {
    "trunk_half_x_norm": 1.0,
    "trunk_half_y_norm": 1.0,
    "trunk_half_z_norm": 0.75,
    "front_attachment_x_norm": 1.0,
    "rear_attachment_x_norm": 1.0,
    "lateral_attachment_y_norm": 1.0,
    "leg_length_norm": 1.5,
    "upper_link_fraction": 0.75,
    "lower_link_fraction": 0.75,
    "terminal_radius_or_foot_norm": 0.75,
    "mean_joint_count_norm": 1.0,
    "wheel_radius_norm": 0.5,
}
DISTANCE_BANDS = {
    "clone_guard_minimum": (0.035, math.inf),
    "prior_neighborhood": (0.04, 0.18),
    "interpolation_band": (0.12, 0.36),
    "bounded_outward_band": (0.24, 0.55),
}
REGION_EXPECTED_PER_FAMILY = {
    "prior_neighborhood": 8,
    "interpolation_band": 16,
    "bounded_outward_band": 8,
}
MAX_PHYSICAL_LINK_SCALE = 1.08
ATTACHMENT_CLEARANCE = 0.02
LEG_RADIUS = 0.045
WHEEL_WIDTH_FRACTION = 1.20
BIPED_FOOT_HALF_LENGTH_MIN = 0.13
BIPED_FOOT_HALF_LENGTH_MAX = 0.28
BIPED_FOOT_MAX_FULL_LENGTH_TO_LEG_RATIO = 0.78
BIPED_FOOT_HALF_WIDTH_MIN = 0.055
BIPED_FOOT_HALF_WIDTH_MAX = 0.115


@dataclass(frozen=True, slots=True)
class Task070ArchetypeConfig:
    """Versioned identity knobs for the Task070 profile."""

    reference_registry_sha256: str = TASK070_REFERENCE_REGISTRY_SHA256
    source_license_matrix_sha256: str = TASK070_SOURCE_LICENSE_MATRIX_SHA256
    r0_design_contract_sha256: str = TASK070_R0_DESIGN_CONTRACT_SHA256
    prior_set_id: str = TASK070_PRIOR_SET_ID
    distance_contract_hash: str = TASK070_DISTANCE_CONTRACT_HASH
    stance_contract_hash: str = TASK070_STANCE_CONTRACT_HASH
    clone_guard_minimum: float = 0.035
    max_retry_attempts: int = 6
    wheel_velocity_hold_gain: float = 4.0

    def __post_init__(self) -> None:
        for label, value in (
            ("reference_registry_sha256", self.reference_registry_sha256),
            ("source_license_matrix_sha256", self.source_license_matrix_sha256),
            ("r0_design_contract_sha256", self.r0_design_contract_sha256),
            ("distance_contract_hash", self.distance_contract_hash),
            ("stance_contract_hash", self.stance_contract_hash),
        ):
            if len(value) != 64:
                raise ValueError(f"{label} must be a SHA-256 digest")
        if self.max_retry_attempts <= 0:
            raise ValueError("max_retry_attempts must be positive")

    @property
    def contract_identity_hash(self) -> str:
        payload = {
            "profile": ARCHETYPE_CONSTRAINED_MORPHOLOGY_PROFILE_VERSION,
            "reference_registry_sha256": self.reference_registry_sha256,
            "source_license_matrix_sha256": self.source_license_matrix_sha256,
            "r0_design_contract_sha256": self.r0_design_contract_sha256,
            "prior_set_id": self.prior_set_id,
            "distance_contract_hash": self.distance_contract_hash,
            "stance_contract_hash": self.stance_contract_hash,
            "clone_guard_minimum": self.clone_guard_minimum,
            "max_retry_attempts": self.max_retry_attempts,
        }
        return _hash_payload(payload)


@dataclass(frozen=True, slots=True)
class PriorCenter:
    prior_id: str
    family: Literal["biped", "quadruped"]
    features: Mapping[str, float]


PRIOR_CENTERS: tuple[PriorCenter, ...] = (
    PriorCenter(
        prior_id="unitree_g1",
        family="biped",
        features={
            "trunk_half_x_norm": 0.22,
            "trunk_half_y_norm": 0.13,
            "trunk_half_z_norm": 0.21,
            "front_attachment_x_norm": 0.0,
            "rear_attachment_x_norm": 0.0,
            "lateral_attachment_y_norm": 0.25,
            "leg_length_norm": 0.80,
            "upper_link_fraction": 0.45,
            "lower_link_fraction": 0.43,
            "terminal_radius_or_foot_norm": 0.095,
            "mean_joint_count_norm": 1.0,
            "wheel_radius_norm": 0.0,
        },
    ),
    PriorCenter(
        prior_id="engineai_pm01",
        family="biped",
        features={
            "trunk_half_x_norm": 0.20,
            "trunk_half_y_norm": 0.12,
            "trunk_half_z_norm": 0.19,
            "front_attachment_x_norm": 0.0,
            "rear_attachment_x_norm": 0.0,
            "lateral_attachment_y_norm": 0.27,
            "leg_length_norm": 0.72,
            "upper_link_fraction": 0.43,
            "lower_link_fraction": 0.44,
            "terminal_radius_or_foot_norm": 0.10,
            "mean_joint_count_norm": 1.0,
            "wheel_radius_norm": 0.0,
        },
    ),
    PriorCenter(
        prior_id="spot_base",
        family="quadruped",
        features={
            "trunk_half_x_norm": 0.34,
            "trunk_half_y_norm": 0.12,
            "trunk_half_z_norm": 0.09,
            "front_attachment_x_norm": 0.42,
            "rear_attachment_x_norm": -0.42,
            "lateral_attachment_y_norm": 0.40,
            "leg_length_norm": 0.48,
            "upper_link_fraction": 0.47,
            "lower_link_fraction": 0.45,
            "terminal_radius_or_foot_norm": 0.055,
            "mean_joint_count_norm": 0.5,
            "wheel_radius_norm": 0.0,
        },
    ),
    PriorCenter(
        prior_id="unitree_go2",
        family="quadruped",
        features={
            "trunk_half_x_norm": 0.30,
            "trunk_half_y_norm": 0.11,
            "trunk_half_z_norm": 0.08,
            "front_attachment_x_norm": 0.43,
            "rear_attachment_x_norm": -0.43,
            "lateral_attachment_y_norm": 0.42,
            "leg_length_norm": 0.43,
            "upper_link_fraction": 0.49,
            "lower_link_fraction": 0.49,
            "terminal_radius_or_foot_norm": 0.05,
            "mean_joint_count_norm": 0.5,
            "wheel_radius_norm": 0.0,
        },
    ),
    PriorCenter(
        prior_id="deeprobotics_lite3",
        family="quadruped",
        features={
            "trunk_half_x_norm": 0.31,
            "trunk_half_y_norm": 0.13,
            "trunk_half_z_norm": 0.09,
            "front_attachment_x_norm": 0.38,
            "rear_attachment_x_norm": -0.38,
            "lateral_attachment_y_norm": 0.44,
            "leg_length_norm": 0.46,
            "upper_link_fraction": 0.44,
            "lower_link_fraction": 0.48,
            "terminal_radius_or_foot_norm": 0.052,
            "mean_joint_count_norm": 0.5,
            "wheel_radius_norm": 0.0,
        },
    ),
)

FEATURE_LIMITS = {
    "biped": {
        "trunk_half_x_norm": (0.16, 0.34),
        "trunk_half_y_norm": (0.09, 0.23),
        "trunk_half_z_norm": (0.14, 0.31),
        "front_attachment_x_norm": (-0.08, 0.08),
        "rear_attachment_x_norm": (-0.08, 0.08),
        "lateral_attachment_y_norm": (0.18, 0.55),
        "leg_length_norm": (0.60, 1.00),
        "upper_link_fraction": (0.34, 0.60),
        "lower_link_fraction": (0.32, 0.60),
        "terminal_radius_or_foot_norm": (0.065, 0.17),
        "mean_joint_count_norm": (1.0, 1.0),
        "wheel_radius_norm": (0.0, 0.45),
    },
    "quadruped": {
        "trunk_half_x_norm": (0.24, 0.45),
        "trunk_half_y_norm": (0.08, 0.22),
        "trunk_half_z_norm": (0.06, 0.16),
        "front_attachment_x_norm": (0.28, 0.60),
        "rear_attachment_x_norm": (-0.60, -0.28),
        "lateral_attachment_y_norm": (0.17, 0.50),
        "leg_length_norm": (0.34, 0.66),
        "upper_link_fraction": (0.34, 0.60),
        "lower_link_fraction": (0.34, 0.60),
        "terminal_radius_or_foot_norm": (0.035, 0.105),
        "mean_joint_count_norm": (0.5, 0.5),
        "wheel_radius_norm": (0.0, 0.17),
    },
}


class ArchetypeConstrainedMorphologyGenerator:
    """Deterministic Task070 multi-vendor prior sampler."""

    profile_version = ARCHETYPE_CONSTRAINED_MORPHOLOGY_PROFILE_VERSION
    contract_version = ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_VERSION
    contract_hash = ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_HASH

    def __init__(self, config: Task070ArchetypeConfig | None = None) -> None:
        self.config = config or Task070ArchetypeConfig()

    def generate(self, family: MorphologyFamily, seed: int) -> MorphologyBlueprint:
        if family not in {"biped", "quadruped", "wheeled_biped", "wheeled_quadruped"}:
            raise ValueError(f"unsupported Task070 family: {family!r}")
        sample = self._sample_features(family, seed)
        return self._build_blueprint(family, seed, sample)

    def sample_physical_params(
        self,
        blueprint: MorphologyBlueprint,
        seed: int,
        *,
        range_fraction: float = 1.0,
    ) -> PhysicalParams:
        if blueprint.profile_version != self.profile_version:
            raise ValueError("Task070 physical sampler requires a Task070 blueprint")
        if not 0.0 <= range_fraction <= 1.0:
            raise ValueError("range_fraction must be between zero and one")
        rng = random.Random(_stable_seed("task070_physical", blueprint.family, seed))

        def centered(low: float, high: float) -> float:
            midpoint = 0.5 * (low + high)
            radius = 0.5 * (high - low) * range_fraction
            return rng.uniform(midpoint - radius, midpoint + radius)

        global_scale = centered(0.94, 1.06)
        link_scales = {link.name: 1.0 for link in blueprint.links}
        mass_scales = {link.name: centered(0.9, 1.1) for link in blueprint.links}
        joint_limit_scales = {joint.semantic_slot: centered(0.95, 1.05) for joint in blueprint.joints}
        raw_motor_strength = {joint.semantic_slot: centered(1.15, 1.45) for joint in blueprint.joints}
        raw_kp_scales = {joint.semantic_slot: centered(0.95, 1.1) for joint in blueprint.joints}
        raw_kd_scales = {joint.semantic_slot: centered(1.0, 1.15) for joint in blueprint.joints}
        scaling = _actuator_scaling_profile(
            blueprint,
            mass_scales=mass_scales,
            global_scale=global_scale,
        )
        motor_strength = {
            slot: raw_motor_strength[slot] * float(values["motor_factor"])
            for slot, values in scaling["slots"].items()
        }
        kp_scales = {
            slot: raw_kp_scales[slot] * float(values["kp_factor"])
            for slot, values in scaling["slots"].items()
        }
        kd_scales = {
            slot: raw_kd_scales[slot] * float(values["kd_factor"])
            for slot, values in scaling["slots"].items()
        }
        scaling_audit = {
            "contract": "task070_morphology_aware_sampled_mass_scale_lever_arm_scaling_v2",
            "raw_seed_policy": (
                "stable_seed(task070_physical, family, seed); sampled mass scales, "
                "global geometry scale, and blueprint lever arms enter deterministic factors"
            ),
            "family_nominal_mass_kg": scaling["family_nominal_mass_kg"],
            "total_blueprint_mass_kg": scaling["total_blueprint_mass_kg"],
            "total_sampled_mass_kg": scaling["total_sampled_mass_kg"],
            "global_scale": scaling["global_scale"],
            "slots": {
                slot: {
                    **values,
                    "raw_motor_strength": raw_motor_strength[slot],
                    "raw_kp_scale": raw_kp_scales[slot],
                    "raw_kd_scale": raw_kd_scales[slot],
                    "final_motor_strength": motor_strength[slot],
                    "final_kp_scale": kp_scales[slot],
                    "final_kd_scale": kd_scales[slot],
                }
                for slot, values in scaling["slots"].items()
            },
        }
        return PhysicalParams(
            global_scale=global_scale,
            link_scales=link_scales,
            mass_scales=mass_scales,
            com_offsets={
                link.name: tuple(
                    float(base + centered(-0.008, 0.008)) for base in link.com
                )
                for link in blueprint.links
            },
            joint_limit_scales=joint_limit_scales,
            nominal_offsets={joint.semantic_slot: 0.0 for joint in blueprint.joints},
            friction=centered(0.95, 1.2),
            motor_strength=motor_strength,
            kp_scales=kp_scales,
            kd_scales=kd_scales,
            delay_ms=0.0,
            ema_alpha=1.0,
            payload_mass=0.0,
            metadata={"task070_actuator_scaling": scaling_audit},
        )

    def expected_sampling_region(self, seed: int) -> Task070SamplingRegion:
        modulo = seed % 4
        if modulo == 0:
            return "prior_neighborhood"
        if modulo in {1, 2}:
            return "interpolation_band"
        return "bounded_outward_band"

    def _sample_features(self, family: MorphologyFamily, seed: int) -> dict[str, object]:
        base_family = _base_family(family)
        region = self.expected_sampling_region(seed)
        centers = [center for center in PRIOR_CENTERS if center.family == base_family]
        retry_trace: list[dict[str, object]] = []
        last: dict[str, object] | None = None
        for attempt in range(self.config.max_retry_attempts):
            rng = random.Random(_stable_seed("task070", family, seed, attempt))
            features, contribution = _region_features(base_family, region, centers, seed, attempt, rng)
            if family.startswith("wheeled_"):
                wheel_base = 0.36 if base_family == "biped" else 0.10
                features["wheel_radius_norm"] = _clamp(
                    wheel_base + rng.uniform(-0.015, 0.015),
                    *FEATURE_LIMITS[base_family]["wheel_radius_norm"],
                )
            else:
                features["wheel_radius_norm"] = 0.0
            features = _apply_realized_feature_envelope(base_family, features)
            lower, upper = DISTANCE_BANDS[region]
            features = _project_into_distance_band(
                features,
                centers,
                base_family=base_family,
                lower=lower,
                upper=upper,
                rng=rng,
            )
            features = _apply_realized_feature_envelope(base_family, features)
            nearest, distance = _nearest_prior(features, centers)
            passed = (
                distance >= self.config.clone_guard_minimum
                and distance >= lower
                and distance <= upper
            )
            trace = {
                "attempt": attempt,
                "region": region,
                "prior_contribution": contribution,
                "nearest_prior": nearest.prior_id,
                "nearest_prior_distance": distance,
                "clone_guard_passed": distance >= self.config.clone_guard_minimum,
                "region_band_passed": lower <= distance <= upper,
            }
            retry_trace.append(trace)
            last = {
                "features": features,
                "contribution": contribution,
                "nearest": nearest,
                "distance": distance,
                "trace": retry_trace,
            }
            if passed:
                return last
        assert last is not None
        raise ValueError(
            "Task070 constrained sampler exhausted deterministic retries without "
            f"a valid region-band sample: family={family!r} seed={seed} "
            f"region={region!r} attempts={json.dumps(retry_trace, sort_keys=True)}"
        )

    def _build_blueprint(
        self,
        family: MorphologyFamily,
        seed: int,
        sample: Mapping[str, object],
    ) -> MorphologyBlueprint:
        features = dict(sample["features"])  # type: ignore[arg-type]
        base_family = _base_family(family)
        is_wheeled = family.startswith("wheeled_")
        links: list[LinkBlueprint] = []
        joints: list[JointBlueprint] = []
        actuators: list[ActuatorBlueprint] = []
        wheels: list[WheelBlueprint] = []
        end_sites: list[str] = []
        realized_attachments: list[dict[str, object]] = []

        trunk_half_x = float(features["trunk_half_x_norm"])
        trunk_half_y = float(features["trunk_half_y_norm"])
        trunk_half_z = float(features["trunk_half_z_norm"])
        leg_length = float(features["leg_length_norm"])
        trunk_mass = 14.0 if base_family == "biped" else 9.0
        links.append(
            LinkBlueprint(
                name="trunk",
                parent="root",
                geom_type="box",
                size=(trunk_half_x, trunk_half_y),
                pos=(0.0, 0.0, 0.0),
                length=2.0 * trunk_half_z,
                mass=trunk_mass,
                com=(
                    0.0,
                    0.0,
                    -0.72 * trunk_half_z if base_family == "biped" else -0.39 * trunk_half_z,
                ),
                contact=True,
            )
        )

        if base_family == "biped":
            sampled_sagittal = 0.5 * (
                float(features["front_attachment_x_norm"])
                + float(features["rear_attachment_x_norm"])
            ) * leg_length
            sagittal_limit = max(
                0.0,
                trunk_half_x - LEG_RADIUS * MAX_PHYSICAL_LINK_SCALE - ATTACHMENT_CLEARANCE,
            )
            sagittal = _clamp(sampled_sagittal, -sagittal_limit, sagittal_limit)
            lateral = max(
                float(features["lateral_attachment_y_norm"]) * leg_length,
                trunk_half_y + LEG_RADIUS * MAX_PHYSICAL_LINK_SCALE + ATTACHMENT_CLEARANCE,
            )
            z_attach = -trunk_half_z - LEG_RADIUS * MAX_PHYSICAL_LINK_SCALE - ATTACHMENT_CLEARANCE
            for limb, side in enumerate(("left", "right")):
                attachment = (sagittal, lateral if side == "left" else -lateral, z_attach)
                realized_attachments.append(
                    {
                        "limb": side,
                        "parent": "trunk",
                        "attachment": attachment,
                        "source": "biped_pelvis_bottom_surface_with_mirrored_lateral_pair",
                    }
                )
                terminal, terminal_length = self._append_limb(
                    links,
                    joints,
                    actuators,
                    end_sites,
                    prefix=f"{side}_leg",
                    semantic_prefix=f"limb{limb}",
                    parent="trunk",
                    attachment=attachment,
                    joint_suffixes=("hip_yaw", "hip_roll", "hip_pitch", "knee_pitch", "ankle_pitch", "ankle_roll"),
                    total_length=leg_length,
                    upper_fraction=float(features["upper_link_fraction"]),
                    lower_fraction=float(features["lower_link_fraction"]),
                    mirror_axes=side == "right",
                    terminal_foot=not is_wheeled,
                    foot_fraction=float(features["terminal_radius_or_foot_norm"]),
                    limb_kind="biped_leg",
                )
                if is_wheeled:
                    wheels.append(
                        self._append_wheel(
                            links,
                            joints,
                            actuators,
                            terminal_link=terminal,
                            terminal_length=terminal_length,
                            semantic_prefix=f"limb{limb}",
                            prefix=f"{side}_leg",
                            radius=max(0.06, float(features["wheel_radius_norm"])),
                        )
                    )
            nominal_height = trunk_half_z + leg_length + (max((w.radius for w in wheels), default=0.0)) + 0.035
            has_arms = False
            archetype = "generic_humanoid_biped"
        else:
            front_x = float(features["front_attachment_x_norm"]) * leg_length
            rear_x = float(features["rear_attachment_x_norm"]) * leg_length
            if is_wheeled:
                wheel_radius = max(0.055, float(features["wheel_radius_norm"]))
                center_x = 0.5 * (front_x + rear_x)
                half_gap = max(0.5 * (front_x - rear_x), wheel_radius * 1.25 + 0.03)
                front_x = center_x + half_gap
                rear_x = center_x - half_gap
            lateral = max(
                float(features["lateral_attachment_y_norm"]) * leg_length,
                trunk_half_y + LEG_RADIUS * MAX_PHYSICAL_LINK_SCALE + ATTACHMENT_CLEARANCE,
            )
            z_attach = -trunk_half_z - LEG_RADIUS * MAX_PHYSICAL_LINK_SCALE - ATTACHMENT_CLEARANCE
            for limb, (name, x, y, mirror) in enumerate(
                (
                    ("front_left", front_x, lateral, False),
                    ("front_right", front_x, -lateral, True),
                    ("rear_left", rear_x, lateral, False),
                    ("rear_right", rear_x, -lateral, True),
                )
            ):
                attachment = (x, y, z_attach)
                realized_attachments.append(
                    {
                        "limb": name,
                        "parent": "trunk",
                        "attachment": attachment,
                        "source": "quadruped_trunk_bottom_surface_with_front_rear_ratios",
                    }
                )
                terminal, terminal_length = self._append_limb(
                    links,
                    joints,
                    actuators,
                    end_sites,
                    prefix=f"{name}_leg",
                    semantic_prefix=f"limb{limb}",
                    parent="trunk",
                    attachment=attachment,
                    joint_suffixes=("hip_roll", "hip_pitch", "knee_pitch"),
                    total_length=leg_length,
                    upper_fraction=float(features["upper_link_fraction"]),
                    lower_fraction=float(features["lower_link_fraction"]),
                    mirror_axes=mirror,
                    terminal_foot=not is_wheeled,
                    foot_fraction=float(features["terminal_radius_or_foot_norm"]),
                    limb_kind="quadruped_leg",
                )
                if is_wheeled:
                    wheels.append(
                        self._append_wheel(
                            links,
                            joints,
                            actuators,
                            terminal_link=terminal,
                            terminal_length=terminal_length,
                            semantic_prefix=f"limb{limb}",
                            prefix=f"{name}_leg",
                            radius=max(0.055, float(features["wheel_radius_norm"])),
                        )
                    )
            nominal_height = trunk_half_z + leg_length + (max((w.radius for w in wheels), default=0.0)) + 0.03
            has_arms = False
            archetype = "generic_mammal_quadruped"

        topology_payload = {
            "profile_version": self.profile_version,
            "family": family,
            "archetype": archetype,
            "joint_order": [
                (joint.semantic_slot, joint.axis_name, tuple(round(value, 6) for value in joint.axis))
                for joint in joints
            ],
            "wheel_topology": [
                (wheel.semantic_slot, wheel.axis_name, tuple(round(value, 6) for value in wheel.axis))
                for wheel in wheels
            ],
        }
        structural_hash = _hash_payload(topology_payload)[:16]
        nearest = sample["nearest"]  # type: ignore[assignment]
        profile_metadata = {
            "task": "task070-archetype-constrained-standable-morphology",
            "profile_version": self.profile_version,
            "reference_registry_sha256": self.config.reference_registry_sha256,
            "source_license_matrix_sha256": self.config.source_license_matrix_sha256,
            "r0_design_contract_sha256": self.config.r0_design_contract_sha256,
            "prior_set_id": self.config.prior_set_id,
            "distance_contract_hash": self.config.distance_contract_hash,
            "stance_contract_hash": self.config.stance_contract_hash,
            "contract_identity_hash": self.config.contract_identity_hash,
            "archetype": archetype,
            "topology_variant": "mirrored_symmetric_v1",
            "sampling_region": self.expected_sampling_region(seed),
            "region_expected_denominator": REGION_EXPECTED_PER_FAMILY[self.expected_sampling_region(seed)],
            "prior_contribution": sample["contribution"],
            "nearest_prior": nearest.prior_id,
            "nearest_prior_distance": sample["distance"],
            "clone_guard": {
                "passed": float(sample["distance"]) >= self.config.clone_guard_minimum,
                "minimum_distance": self.config.clone_guard_minimum,
                "legal_conclusion": False,
            },
            "retry_trace": sample["trace"],
            "normalized_feature_vector": features,
            "attachment_derivation": {
                "trunk_half_extents": {
                    "x": trunk_half_x,
                    "y": trunk_half_y,
                    "z": trunk_half_z,
                },
                "link_radius": LEG_RADIUS,
                "max_physical_link_scale": MAX_PHYSICAL_LINK_SCALE,
                "clearance": ATTACHMENT_CLEARANCE,
                "policy": "derived_from_trunk_surface_link_radius_scale_and_clearance",
                "realized_attachments": realized_attachments,
            },
            "reference_role": "all_prior_centers_are_seen_reference_not_heldout",
            "primitive_geometry_only": True,
        }
        return MorphologyBlueprint(
            family=family,
            seed=seed,
            links=tuple(links),
            joints=tuple(joints),
            actuators=tuple(actuators),
            nominal_height=nominal_height,
            has_arms=has_arms,
            structural_hash=structural_hash,
            end_sites=tuple(end_sites),
            profile_version=self.profile_version,
            contract_version=self.contract_version,
            contract_hash=self.contract_hash,
            wheel_specs=tuple(wheels),
            profile_metadata=profile_metadata,
        )

    def _append_limb(
        self,
        links: list[LinkBlueprint],
        joints: list[JointBlueprint],
        actuators: list[ActuatorBlueprint],
        end_sites: list[str],
        *,
        prefix: str,
        semantic_prefix: str,
        parent: str,
        attachment: tuple[float, float, float],
        joint_suffixes: Sequence[str],
        total_length: float,
        upper_fraction: float,
        lower_fraction: float,
        mirror_axes: bool,
        terminal_foot: bool,
        foot_fraction: float,
        limb_kind: Literal["biped_leg", "quadruped_leg"],
    ) -> tuple[str, float]:
        lengths = _segment_lengths(joint_suffixes, total_length, upper_fraction, lower_fraction)
        current_parent = parent
        previous_length = 0.0
        for index, (suffix, length) in enumerate(zip(joint_suffixes, lengths, strict=True)):
            axis_name = "pitch" if suffix.endswith("pitch") else "roll" if suffix.endswith("roll") else "yaw"
            semantic_slot = f"{semantic_prefix}_{suffix}"
            child_link = f"{prefix}_{index}_link"
            joint_name = f"{child_link}_joint"
            pos = attachment if index == 0 else (0.0, 0.0, -previous_length)
            is_terminal = index == len(joint_suffixes) - 1
            foot_size = None
            if terminal_foot and is_terminal:
                if limb_kind == "biped_leg":
                    foot_size = (
                        min(
                            BIPED_FOOT_HALF_LENGTH_MAX,
                            min(
                                0.5 * BIPED_FOOT_MAX_FULL_LENGTH_TO_LEG_RATIO * total_length,
                                max(
                                    BIPED_FOOT_HALF_LENGTH_MIN,
                                    total_length * foot_fraction * 2.35,
                                ),
                            ),
                        ),
                        min(
                            BIPED_FOOT_HALF_WIDTH_MAX,
                            min(
                                0.18 * total_length,
                                max(
                                    BIPED_FOOT_HALF_WIDTH_MIN,
                                    total_length * foot_fraction * 0.82,
                                ),
                            ),
                        ),
                    )
                else:
                    foot_size = (
                        max(0.055, total_length * foot_fraction * 1.2),
                        max(0.04, total_length * foot_fraction * 0.85),
                    )
            links.append(
                LinkBlueprint(
                    name=child_link,
                    parent=current_parent,
                    geom_type="capsule",
                    size=(LEG_RADIUS, length * 0.5),
                    pos=pos,
                    length=length,
                    mass=2.4 if limb_kind == "biped_leg" else 0.95,
                    contact=terminal_foot and is_terminal,
                    end_site=terminal_foot and is_terminal,
                    foot=terminal_foot and is_terminal,
                    foot_size=foot_size,
                )
            )
            axis = _axis(axis_name, mirrored=mirror_axes)
            joints.append(
                JointBlueprint(
                    name=joint_name,
                    parent_link=current_parent,
                    child_link=child_link,
                    semantic_slot=semantic_slot,
                    axis_name=axis_name,
                    axis=axis,
                    joint_range=_joint_range(suffix),
                    nominal=0.0,
                damping=3.0 if limb_kind == "biped_leg" else 1.2,
                friction=0.08 if limb_kind == "biped_leg" else 0.04,
                armature=0.03 if limb_kind == "biped_leg" else 0.015,
                )
            )
            effort = 900.0 if limb_kind == "biped_leg" else 180.0
            actuators.append(
                ActuatorBlueprint(
                    name=f"{joint_name}_actuator",
                    joint_name=joint_name,
                    semantic_slot=semantic_slot,
                    kp=280.0 if limb_kind == "biped_leg" else 90.0,
                    kd=14.0 if limb_kind == "biped_leg" else 3.5,
                    effort_limit=effort,
                )
            )
            current_parent = child_link
            previous_length = length
            if terminal_foot and is_terminal:
                end_sites.append(f"{child_link}_foot")
        return current_parent, previous_length

    def _append_wheel(
        self,
        links: list[LinkBlueprint],
        joints: list[JointBlueprint],
        actuators: list[ActuatorBlueprint],
        *,
        terminal_link: str,
        terminal_length: float,
        semantic_prefix: str,
        prefix: str,
        radius: float,
    ) -> WheelBlueprint:
        width = max(0.045, radius * WHEEL_WIDTH_FRACTION)
        link_name = f"{prefix}_wheel"
        joint_name = f"{link_name}_joint"
        semantic_slot = f"{semantic_prefix}_wheel"
        links.append(
            LinkBlueprint(
                name=link_name,
                parent=terminal_link,
                geom_type="cylinder",
                size=(radius, width * 0.5),
                pos=(0.0, 0.0, -terminal_length),
                length=width,
                mass=max(0.5, 14.0 * radius * width),
                contact=True,
            )
        )
        joints.append(
            JointBlueprint(
                name=joint_name,
                parent_link=terminal_link,
                child_link=link_name,
                semantic_slot=semantic_slot,
                axis_name="wheel",
                axis=(0.0, 1.0, 0.0),
                joint_range=(-math.pi, math.pi),
                nominal=0.0,
                damping=0.3,
                friction=0.02,
                armature=0.02,
            )
        )
        effort = 120.0
        actuators.append(
            ActuatorBlueprint(
                name=f"{joint_name}_actuator",
                joint_name=joint_name,
                semantic_slot=semantic_slot,
                kp=0.0,
                kd=5.0,
                effort_limit=effort,
            )
        )
        return WheelBlueprint(
            link_name=link_name,
            joint_name=joint_name,
            semantic_slot=semantic_slot,
            radius=radius,
            width=width,
            axis_name="terminal_link_local_lateral",
            axis=(0.0, 1.0, 0.0),
            friction=1.2,
            effort_limit=effort,
            kp=0.0,
            kd=12.0,
        )


def _segment_lengths(
    suffixes: Sequence[str],
    total: float,
    upper_fraction: float,
    lower_fraction: float,
) -> tuple[float, ...]:
    if len(suffixes) == 3:
        hip = max(0.04, total * 0.12)
        upper = max(0.08, total * upper_fraction)
        lower = max(0.08, total - hip - upper)
        return (hip, upper, lower)
    hip_budget = min(total * 0.2, 0.16)
    upper = total * upper_fraction
    lower = total * lower_fraction
    ankle_budget = max(0.04, total - hip_budget - upper - lower)
    lengths = (
        hip_budget * 0.28,
        hip_budget * 0.32,
        max(0.05, upper),
        max(0.06, lower * 0.72),
        max(0.035, lower * 0.18),
        max(0.03, ankle_budget + lower * 0.10),
    )
    scale = total / sum(lengths)
    return tuple(max(0.025, value * scale) for value in lengths)


def _actuator_scaling_profile(
    blueprint: MorphologyBlueprint,
    *,
    mass_scales: Mapping[str, float] | None = None,
    global_scale: float = 1.0,
) -> dict[str, object]:
    links = {link.name: link for link in blueprint.links}
    children: dict[str, list[str]] = {}
    for link in blueprint.links:
        children.setdefault(link.parent, []).append(link.name)
    total_mass = sum(link.mass for link in blueprint.links)
    resolved_mass_scales = mass_scales or {link.name: 1.0 for link in blueprint.links}
    total_sampled_mass = sum(
        link.mass * float(resolved_mass_scales.get(link.name, 1.0))
        for link in blueprint.links
    )
    family_nominal_mass = 23.5 if blueprint.family.endswith("biped") else 12.8
    nominal_lever = 0.62 if blueprint.family.endswith("biped") else 0.36
    wheel_radius_by_slot = {wheel.semantic_slot: wheel.radius for wheel in blueprint.wheel_specs}
    slots: dict[str, dict[str, float]] = {}
    for joint in blueprint.joints:
        lever = _downstream_length(joint.child_link, links, children) * global_scale
        mass_factor = math.sqrt(max(total_sampled_mass, 1e-6) / family_nominal_mass)
        lever_factor = math.sqrt(max(lever, 0.04) / nominal_lever)
        if joint.semantic_slot.endswith("_wheel"):
            radius = wheel_radius_by_slot[joint.semantic_slot] * global_scale
            radius_factor = math.sqrt(radius / 0.11)
            motor_factor = _clamp(mass_factor * radius_factor, 0.70, 1.65)
            kp_factor = 1.0
            kd_factor = _clamp(math.sqrt(motor_factor), 0.85, 1.35)
            lever_for_audit = radius
        else:
            motor_factor = _clamp(mass_factor * lever_factor, 0.72, 1.55)
            kp_factor = _clamp(motor_factor * math.sqrt(max(lever_factor, 1e-6)), 0.70, 1.65)
            kd_factor = _clamp(math.sqrt(motor_factor), 0.80, 1.35)
            lever_for_audit = lever
        slots[joint.semantic_slot] = {
            "estimated_lever_arm_m": lever_for_audit,
            "mass_factor": mass_factor,
            "lever_factor": lever_factor,
            "motor_factor": motor_factor,
            "kp_factor": kp_factor,
            "kd_factor": kd_factor,
        }
    return {
        "family_nominal_mass_kg": family_nominal_mass,
        "total_blueprint_mass_kg": total_mass,
        "total_sampled_mass_kg": total_sampled_mass,
        "global_scale": global_scale,
        "slots": slots,
    }


def _downstream_length(
    link_name: str,
    links: Mapping[str, LinkBlueprint],
    children: Mapping[str, Sequence[str]],
) -> float:
    link = links[link_name]
    child_lengths = [
        _downstream_length(child, links, children)
        for child in children.get(link_name, ())
        if child in links
    ]
    return link.length + (max(child_lengths) if child_lengths else 0.0)


def _axis(axis_name: str, *, mirrored: bool) -> tuple[float, float, float]:
    if axis_name == "yaw":
        return (0.0, 0.0, -1.0 if mirrored else 1.0)
    if axis_name == "roll":
        return (-1.0 if mirrored else 1.0, 0.0, 0.0)
    return (0.0, 1.0, 0.0)


def _joint_range(suffix: str) -> tuple[float, float]:
    if suffix == "knee_pitch":
        return (-0.2, 2.2)
    if suffix.startswith("ankle"):
        return (-0.9, 0.9)
    if suffix == "hip_roll":
        return (-0.75, 0.75)
    if suffix == "hip_yaw":
        return (-1.2, 1.2)
    return (-1.4, 1.4)


def _base_family(family: MorphologyFamily) -> Literal["biped", "quadruped"]:
    return "biped" if family.endswith("biped") else "quadruped"


def _region_features(
    base_family: Literal["biped", "quadruped"],
    region: Task070SamplingRegion,
    centers: Sequence[PriorCenter],
    seed: int,
    attempt: int,
    rng: random.Random,
) -> tuple[dict[str, float], dict[str, float]]:
    if region == "prior_neighborhood":
        center = centers[(seed // 4 + attempt) % len(centers)]
        features = {
            name: float(center.features[name]) + rng.choice((-1.0, 1.0)) * rng.uniform(0.018, 0.035)
            for name in FEATURE_NAMES
        }
        contribution = {item.prior_id: (1.0 if item == center else 0.0) for item in centers}
    elif region == "interpolation_band":
        first = centers[(seed + attempt) % len(centers)]
        second = centers[(seed + attempt + 1) % len(centers)]
        alpha = rng.uniform(0.25, 0.75)
        features = {
            name: alpha * float(first.features[name])
            + (1.0 - alpha) * float(second.features[name])
            + rng.uniform(-0.012, 0.012)
            for name in FEATURE_NAMES
        }
        contribution = {
            item.prior_id: (
                alpha if item == first else (1.0 - alpha if item == second else 0.0)
            )
            for item in centers
        }
    else:
        mean = {
            name: sum(float(center.features[name]) for center in centers) / len(centers)
            for name in FEATURE_NAMES
        }
        direction = {name: rng.uniform(-1.0, 1.0) for name in FEATURE_NAMES}
        magnitude = math.sqrt(sum(direction[name] ** 2 for name in FEATURE_NAMES)) or 1.0
        outward = 0.16 + 0.025 * attempt
        features = {
            name: mean[name] + outward * direction[name] / magnitude
            for name in FEATURE_NAMES
        }
        contribution = {item.prior_id: 1.0 / len(centers) for item in centers}

    limits = FEATURE_LIMITS[base_family]
    clamped = {name: _clamp(float(features[name]), *limits[name]) for name in FEATURE_NAMES}
    return clamped, contribution


def _apply_realized_feature_envelope(
    base_family: Literal["biped", "quadruped"],
    features: Mapping[str, float],
) -> dict[str, float]:
    limits = FEATURE_LIMITS[base_family]
    realized = {name: _clamp(float(features[name]), *limits[name]) for name in FEATURE_NAMES}
    leg_length = max(realized["leg_length_norm"], 1e-6)
    lateral_min = (
        realized["trunk_half_y_norm"]
        + LEG_RADIUS * MAX_PHYSICAL_LINK_SCALE
        + ATTACHMENT_CLEARANCE
        + (
            realized["wheel_radius_norm"] * WHEEL_WIDTH_FRACTION * 0.5
            if realized["wheel_radius_norm"] > 0.0
            else 0.0
        )
    ) / leg_length
    realized["lateral_attachment_y_norm"] = _clamp(
        max(realized["lateral_attachment_y_norm"], lateral_min),
        *limits["lateral_attachment_y_norm"],
    )
    if base_family == "biped":
        sagittal = 0.5 * (
            realized["front_attachment_x_norm"] + realized["rear_attachment_x_norm"]
        )
        sagittal_limit = max(
            0.0,
            (
                realized["trunk_half_x_norm"]
                - LEG_RADIUS * MAX_PHYSICAL_LINK_SCALE
                - ATTACHMENT_CLEARANCE
            )
            / leg_length,
        )
        sagittal = _clamp(sagittal, -sagittal_limit, sagittal_limit)
        realized["front_attachment_x_norm"] = sagittal
        realized["rear_attachment_x_norm"] = sagittal
    else:
        realized["front_attachment_x_norm"] = max(0.0, realized["front_attachment_x_norm"])
        realized["rear_attachment_x_norm"] = min(0.0, realized["rear_attachment_x_norm"])
    return realized


def _project_into_distance_band(
    features: Mapping[str, float],
    centers: Sequence[PriorCenter],
    *,
    base_family: Literal["biped", "quadruped"],
    lower: float,
    upper: float,
    rng: random.Random,
) -> dict[str, float]:
    limits = FEATURE_LIMITS[base_family]
    projected = {name: float(features[name]) for name in FEATURE_NAMES}
    for _ in range(8):
        nearest, distance = _nearest_prior(projected, centers)
        if lower <= distance <= upper:
            return projected
        if distance < lower:
            target = lower + 0.02
            direction = {
                name: projected[name] - float(nearest.features[name])
                for name in FEATURE_NAMES
            }
            weighted = math.sqrt(
                sum(FEATURE_WEIGHTS[name] * direction[name] ** 2 for name in FEATURE_NAMES)
            )
            if weighted < 1e-12:
                direction = {name: rng.choice((-1.0, 1.0)) for name in FEATURE_NAMES}
                weighted = math.sqrt(
                    sum(FEATURE_WEIGHTS[name] * direction[name] ** 2 for name in FEATURE_NAMES)
                )
            scale = target / max(weighted, 1e-12)
            projected = {
                name: _clamp(
                    float(nearest.features[name]) + direction[name] * scale,
                    *limits[name],
                )
                for name in FEATURE_NAMES
            }
        else:
            target = upper - 0.02
            scale = target / max(distance, 1e-12)
            projected = {
                name: _clamp(
                    float(nearest.features[name])
                    + (projected[name] - float(nearest.features[name])) * scale,
                    *limits[name],
                )
                for name in FEATURE_NAMES
            }
    return projected


def _nearest_prior(
    features: Mapping[str, float],
    centers: Sequence[PriorCenter],
) -> tuple[PriorCenter, float]:
    distances = [
        (
            center,
            math.sqrt(
                sum(
                    FEATURE_WEIGHTS[name]
                    * (float(features[name]) - float(center.features[name])) ** 2
                    for name in FEATURE_NAMES
                )
            ),
        )
        for center in centers
    ]
    return min(distances, key=lambda item: item[1])


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return int(digest[:16], 16)


def _hash_payload(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceMotorConfig:
    """Auditable source-side motor hints, separate from anonymous final values."""

    provenance: str
    confidence: str
    source_path: str
    source_sha256: str
    source_motor_class: str
    declared_effort_limit: float | None
    declared_velocity_limit: float | None
    stiffness: float | None = None
    damping: float | None = None
    armature: float | None = None
    rotor_inertias: tuple[float, ...] | None = None
    gear_ratios: tuple[float, ...] | None = None
    control_mode: str | None = None
    usable_as_quantitative_prior: bool = True
    rejection_reason: str | None = None
    note: str | None = None

    def manifest(self) -> dict[str, object]:
        return {
            "provenance": self.provenance,
            "confidence": self.confidence,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_motor_class": self.source_motor_class,
            "raw_declared": {
                "effort_limit": self.declared_effort_limit,
                "velocity_limit": self.declared_velocity_limit,
                "stiffness": self.stiffness,
                "damping": self.damping,
                "armature": self.armature,
                "rotor_inertias": self.rotor_inertias,
                "gear_ratios": self.gear_ratios,
                "control_mode": self.control_mode,
            },
            "usable_as_quantitative_prior": self.usable_as_quantitative_prior,
            "rejection_reason": self.rejection_reason,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class MotorDofSourceMotor:
    source_joint_name: str
    anonymous_semantic_slot: str
    module: str
    source_joint_type: str
    source_parent_body: str
    source_child_body: str
    source_body_local_pos: tuple[float, float, float]
    source_body_local_quat: tuple[float, float, float, float]
    source_joint_local_pos: tuple[float, float, float]
    normalized_local_axis: tuple[float, float, float]
    joint_range: tuple[float, float]
    anonymous_parent_link: str
    anonymous_child_link: str
    source_tree_depth: int
    source_motor_config: SourceMotorConfig | None = None

    def manifest(self) -> dict[str, object]:
        return {
            "source_joint_name": self.source_joint_name,
            "anonymous_semantic_slot": self.anonymous_semantic_slot,
            "module": self.module,
            "source_joint_type": self.source_joint_type,
            "source_parent_body": self.source_parent_body,
            "source_child_body": self.source_child_body,
            "source_body_local_pos": self.source_body_local_pos,
            "source_body_local_quat": self.source_body_local_quat,
            "source_joint_local_pos": self.source_joint_local_pos,
            "normalized_local_axis": self.normalized_local_axis,
            "joint_range": self.joint_range,
            "anonymous_parent_link": self.anonymous_parent_link,
            "anonymous_child_link": self.anonymous_child_link,
            "source_tree_depth": self.source_tree_depth,
            "source_motor_config": (
                self.source_motor_config.manifest()
                if self.source_motor_config is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class G1SourceBodyNode:
    source_body_name: str
    source_parent_body: str
    source_body_local_pos: tuple[float, float, float]
    source_body_local_quat: tuple[float, float, float, float]
    source_tree_depth: int
    selected_motor_joint: str | None
    anonymous_link: str
    child_bodies: tuple[str, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "source_body_name": self.source_body_name,
            "source_parent_body": self.source_parent_body,
            "source_body_local_pos": self.source_body_local_pos,
            "source_body_local_quat": self.source_body_local_quat,
            "source_tree_depth": self.source_tree_depth,
            "selected_motor_joint": self.selected_motor_joint,
            "anonymous_link": self.anonymous_link,
            "child_bodies": self.child_bodies,
        }


@dataclass(frozen=True, slots=True)
class G1StructuralDescriptor:
    center_id: str
    reference_id: str
    source_path: str
    source_sha256: str
    source_root_body: str
    source_to_anonymous_frame_transform: Mapping[str, object]
    body_tree: tuple[G1SourceBodyNode, ...]
    motors: tuple[MotorDofSourceMotor, ...]

    @property
    def module_dof_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for motor in self.motors:
            counts[motor.module] = counts.get(motor.module, 0) + 1
        return counts

    @property
    def descriptor_hash(self) -> str:
        return _hash_payload(self.manifest())

    def manifest(self) -> dict[str, object]:
        return {
            "center_id": self.center_id,
            "reference_id": self.reference_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_root_body": self.source_root_body,
            "source_to_anonymous_frame_transform": dict(
                self.source_to_anonymous_frame_transform
            ),
            "source_motor_count": len(self.motors),
            "anonymous_motor_count": len(self.motors),
            "module_dof_counts": self.module_dof_counts,
            "source_body_tree": [node.manifest() for node in self.body_tree],
            "source_to_anonymous_motor_bijection": [
                motor.manifest() for motor in self.motors
            ],
            "source_body_tree_edges": [
                {
                    "source_parent_body": motor.source_parent_body,
                    "source_child_body": motor.source_child_body,
                    "anonymous_parent_link": motor.anonymous_parent_link,
                    "anonymous_child_link": motor.anonymous_child_link,
                    "anonymous_semantic_slot": motor.anonymous_semantic_slot,
                }
                for motor in self.motors
            ],
        }


G1_29DOF_MOTOR_SEMANTICS: tuple[tuple[str, str, str], ...] = (
    ("left_hip_pitch_joint", "limb0_hip_pitch", "left_leg"),
    ("left_hip_roll_joint", "limb0_hip_roll", "left_leg"),
    ("left_hip_yaw_joint", "limb0_hip_yaw", "left_leg"),
    ("left_knee_joint", "limb0_knee_pitch", "left_leg"),
    ("left_ankle_pitch_joint", "limb0_ankle_pitch", "left_leg"),
    ("left_ankle_roll_joint", "limb0_ankle_roll", "left_leg"),
    ("right_hip_pitch_joint", "limb1_hip_pitch", "right_leg"),
    ("right_hip_roll_joint", "limb1_hip_roll", "right_leg"),
    ("right_hip_yaw_joint", "limb1_hip_yaw", "right_leg"),
    ("right_knee_joint", "limb1_knee_pitch", "right_leg"),
    ("right_ankle_pitch_joint", "limb1_ankle_pitch", "right_leg"),
    ("right_ankle_roll_joint", "limb1_ankle_roll", "right_leg"),
    ("waist_yaw_joint", "waist_yaw", "waist"),
    ("waist_roll_joint", "waist_roll", "waist"),
    ("waist_pitch_joint", "waist_pitch", "waist"),
    ("left_shoulder_pitch_joint", "left_arm_shoulder_pitch", "left_arm"),
    ("left_shoulder_roll_joint", "left_arm_shoulder_roll", "left_arm"),
    ("left_shoulder_yaw_joint", "left_arm_shoulder_yaw", "left_arm"),
    ("left_elbow_joint", "left_arm_elbow_pitch", "left_arm"),
    ("left_wrist_roll_joint", "left_arm_wrist_roll", "left_arm"),
    ("left_wrist_pitch_joint", "left_arm_wrist_pitch", "left_arm"),
    ("left_wrist_yaw_joint", "left_arm_wrist_yaw", "left_arm"),
    ("right_shoulder_pitch_joint", "right_arm_shoulder_pitch", "right_arm"),
    ("right_shoulder_roll_joint", "right_arm_shoulder_roll", "right_arm"),
    ("right_shoulder_yaw_joint", "right_arm_shoulder_yaw", "right_arm"),
    ("right_elbow_joint", "right_arm_elbow_pitch", "right_arm"),
    ("right_wrist_roll_joint", "right_arm_wrist_roll", "right_arm"),
    ("right_wrist_pitch_joint", "right_arm_wrist_pitch", "right_arm"),
    ("right_wrist_yaw_joint", "right_arm_wrist_yaw", "right_arm"),
)

PM01_23DOF_MOTOR_SEMANTICS: tuple[tuple[str, str, str], ...] = (
    ("J00_HIP_PITCH_L", "limb0_hip_pitch", "left_leg"),
    ("J01_HIP_ROLL_L", "limb0_hip_roll", "left_leg"),
    ("J02_HIP_YAW_L", "limb0_hip_yaw", "left_leg"),
    ("J03_KNEE_PITCH_L", "limb0_knee_pitch", "left_leg"),
    ("J04_ANKLE_PITCH_L", "limb0_ankle_pitch", "left_leg"),
    ("J05_ANKLE_ROLL_L", "limb0_ankle_roll", "left_leg"),
    ("J06_HIP_PITCH_R", "limb1_hip_pitch", "right_leg"),
    ("J07_HIP_ROLL_R", "limb1_hip_roll", "right_leg"),
    ("J08_HIP_YAW_R", "limb1_hip_yaw", "right_leg"),
    ("J09_KNEE_PITCH_R", "limb1_knee_pitch", "right_leg"),
    ("J10_ANKLE_PITCH_R", "limb1_ankle_pitch", "right_leg"),
    ("J11_ANKLE_ROLL_R", "limb1_ankle_roll", "right_leg"),
    ("J12_WAIST_YAW", "waist_yaw", "waist"),
    ("J13_SHOULDER_PITCH_L", "left_arm_shoulder_pitch", "left_arm"),
    ("J14_SHOULDER_ROLL_L", "left_arm_shoulder_roll", "left_arm"),
    ("J15_SHOULDER_YAW_L", "left_arm_shoulder_yaw", "left_arm"),
    ("J16_ELBOW_PITCH_L", "left_arm_elbow_pitch", "left_arm"),
    ("J17_ELBOW_YAW_L", "left_arm_wrist_yaw", "left_arm"),
    ("J18_SHOULDER_PITCH_R", "right_arm_shoulder_pitch", "right_arm"),
    ("J19_SHOULDER_ROLL_R", "right_arm_shoulder_roll", "right_arm"),
    ("J20_SHOULDER_YAW_R", "right_arm_shoulder_yaw", "right_arm"),
    ("J21_ELBOW_PITCH_R", "right_arm_elbow_pitch", "right_arm"),
    ("J22_ELBOW_YAW_R", "right_arm_wrist_yaw", "right_arm"),
)

QUADRUPED_12DOF_MOTOR_SEMANTICS: Mapping[
    str, tuple[tuple[str, str, str], ...]
] = {
    "spot_base": (
        ("fl.hx", "limb0_hip_roll", "front_left_leg"),
        ("fl.hy", "limb0_hip_pitch", "front_left_leg"),
        ("fl.kn", "limb0_knee_pitch", "front_left_leg"),
        ("fr.hx", "limb1_hip_roll", "front_right_leg"),
        ("fr.hy", "limb1_hip_pitch", "front_right_leg"),
        ("fr.kn", "limb1_knee_pitch", "front_right_leg"),
        ("hl.hx", "limb2_hip_roll", "rear_left_leg"),
        ("hl.hy", "limb2_hip_pitch", "rear_left_leg"),
        ("hl.kn", "limb2_knee_pitch", "rear_left_leg"),
        ("hr.hx", "limb3_hip_roll", "rear_right_leg"),
        ("hr.hy", "limb3_hip_pitch", "rear_right_leg"),
        ("hr.kn", "limb3_knee_pitch", "rear_right_leg"),
    ),
    "unitree_go2": (
        ("FL_hip_joint", "limb0_hip_roll", "front_left_leg"),
        ("FL_thigh_joint", "limb0_hip_pitch", "front_left_leg"),
        ("FL_calf_joint", "limb0_knee_pitch", "front_left_leg"),
        ("FR_hip_joint", "limb1_hip_roll", "front_right_leg"),
        ("FR_thigh_joint", "limb1_hip_pitch", "front_right_leg"),
        ("FR_calf_joint", "limb1_knee_pitch", "front_right_leg"),
        ("RL_hip_joint", "limb2_hip_roll", "rear_left_leg"),
        ("RL_thigh_joint", "limb2_hip_pitch", "rear_left_leg"),
        ("RL_calf_joint", "limb2_knee_pitch", "rear_left_leg"),
        ("RR_hip_joint", "limb3_hip_roll", "rear_right_leg"),
        ("RR_thigh_joint", "limb3_hip_pitch", "rear_right_leg"),
        ("RR_calf_joint", "limb3_knee_pitch", "rear_right_leg"),
    ),
    "deeprobotics_lite3": (
        ("FL_HipX_joint", "limb0_hip_roll", "front_left_leg"),
        ("FL_HipY_joint", "limb0_hip_pitch", "front_left_leg"),
        ("FL_Knee_joint", "limb0_knee_pitch", "front_left_leg"),
        ("FR_HipX_joint", "limb1_hip_roll", "front_right_leg"),
        ("FR_HipY_joint", "limb1_hip_pitch", "front_right_leg"),
        ("FR_Knee_joint", "limb1_knee_pitch", "front_right_leg"),
        ("HL_HipX_joint", "limb2_hip_roll", "rear_left_leg"),
        ("HL_HipY_joint", "limb2_hip_pitch", "rear_left_leg"),
        ("HL_Knee_joint", "limb2_knee_pitch", "rear_left_leg"),
        ("HR_HipX_joint", "limb3_hip_roll", "rear_right_leg"),
        ("HR_HipY_joint", "limb3_hip_pitch", "rear_right_leg"),
        ("HR_Knee_joint", "limb3_knee_pitch", "rear_right_leg"),
    ),
}


def _two_stage_planetary_reflected_inertia(
    rotor_inertias: tuple[float, float, float],
    gear_ratios: tuple[float, float, float],
) -> float:
    """Match the local MJLab helper used by the audited G1 companion config."""

    return (
        rotor_inertias[0] * (gear_ratios[1] * gear_ratios[2]) ** 2
        + rotor_inertias[1] * gear_ratios[2] ** 2
        + rotor_inertias[2]
    )


def _audited_companion_config_sha256(path: Path, expected_sha256: str) -> str:
    actual_sha256 = _sha256_path(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"motor companion config changed without re-audit: {path}; "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    return actual_sha256


def _g1_source_motor_config(source_joint_name: str) -> SourceMotorConfig:
    specs = {
        "5020": (
            (0.139e-4, 0.017e-4, 0.169e-4),
            (1.0, 1.0 + 46.0 / 18.0, 1.0 + 56.0 / 16.0),
            25.0,
            37.0,
        ),
        "7520_14": (
            (0.489e-4, 0.098e-4, 0.533e-4),
            (1.0, 4.5, 1.0 + 48.0 / 22.0),
            88.0,
            32.0,
        ),
        "7520_22": (
            (0.489e-4, 0.109e-4, 0.738e-4),
            (1.0, 4.5, 5.0),
            139.0,
            20.0,
        ),
        "4010": (
            (0.068e-4, 0.0, 0.0),
            (1.0, 5.0, 5.0),
            5.0,
            22.0,
        ),
    }
    parallel_5020 = "ankle_" in source_joint_name or source_joint_name in {
        "waist_pitch_joint",
        "waist_roll_joint",
    }
    if parallel_5020:
        source_class = "parallel_5020_x2"
        base_class = "5020"
        count = 2.0
    elif "wrist_pitch" in source_joint_name or "wrist_yaw" in source_joint_name:
        source_class = base_class = "4010"
        count = 1.0
    elif "hip_roll" in source_joint_name or "knee" in source_joint_name:
        source_class = base_class = "7520_22"
        count = 1.0
    elif (
        "hip_pitch" in source_joint_name
        or "hip_yaw" in source_joint_name
        or source_joint_name == "waist_yaw_joint"
    ):
        source_class = base_class = "7520_14"
        count = 1.0
    else:
        source_class = base_class = "5020"
        count = 1.0
    rotor_inertias, gear_ratios, effort, velocity = specs[base_class]
    armature = _two_stage_planetary_reflected_inertia(rotor_inertias, gear_ratios) * count
    natural_frequency = 10.0 * 2.0 * 3.1415926535
    stiffness = armature * natural_frequency**2
    damping = 2.0 * 2.0 * armature * natural_frequency
    return SourceMotorConfig(
        provenance="upstream_local_companion_config",
        confidence=(
            "source_reported_with_documented_nominal_parallel_linkage_assumption"
            if parallel_5020
            else "source_reported"
        ),
        source_path=_repo_relative(TASK070_G1_MOTOR_CONFIG_PATH),
        source_sha256=_audited_companion_config_sha256(
            TASK070_G1_MOTOR_CONFIG_PATH,
            TASK070_G1_MOTOR_CONFIG_SHA256,
        ),
        source_motor_class=source_class,
        declared_effort_limit=effort * count,
        declared_velocity_limit=velocity,
        stiffness=stiffness,
        damping=damping,
        armature=armature,
        rotor_inertias=rotor_inertias,
        gear_ratios=gear_ratios,
        control_mode="builtin_position_pd",
        note=(
            "The companion config models two parallel 5020 actuators with a nominal "
            "1:1 linkage; velocity remains the base-motor hint."
            if parallel_5020
            else "PD gains are derived in the companion config at 10 Hz and damping ratio 2."
        ),
    )


def _go2_source_motor_config(source_joint_name: str) -> SourceMotorConfig:
    if source_joint_name.endswith("_hip_joint"):
        source_class, stiffness, damping, effort, armature = (
            "hip",
            20.0,
            1.0,
            23.5,
            0.01,
        )
    elif source_joint_name.endswith("_thigh_joint"):
        source_class, stiffness, damping, effort, armature = (
            "thigh",
            20.0,
            1.0,
            23.5,
            0.01,
        )
    elif source_joint_name.endswith("_calf_joint"):
        source_class, stiffness, damping, effort, armature = (
            "calf",
            40.0,
            2.0,
            45.0,
            0.02,
        )
    else:
        raise ValueError(f"unsupported Go2 source motor: {source_joint_name}")
    return SourceMotorConfig(
        provenance="upstream_local_companion_config",
        confidence="source_reported",
        source_path=_repo_relative(TASK070_GO2_MOTOR_CONFIG_PATH),
        source_sha256=_audited_companion_config_sha256(
            TASK070_GO2_MOTOR_CONFIG_PATH,
            TASK070_GO2_MOTOR_CONFIG_SHA256,
        ),
        source_motor_class=source_class,
        declared_effort_limit=effort,
        declared_velocity_limit=None,
        stiffness=stiffness,
        damping=damping,
        armature=armature,
        control_mode="builtin_position_pd",
        note="The companion config does not declare a velocity limit.",
    )


def _urdf_source_motor_config(
    *,
    reference_id: str,
    source_joint_name: str,
    item: Mapping[str, object],
    source_path: Path,
) -> SourceMotorConfig:
    effort = item.get("declared_effort_limit")
    velocity = item.get("declared_velocity_limit")
    declared_effort = float(effort) if effort is not None else None
    declared_velocity = float(velocity) if velocity is not None else None
    placeholder = (
        reference_id == "spot_base"
        and declared_effort == 1000.0
        and declared_velocity == 1000.0
    )
    candidate_only = reference_id in TASK070_ADDITIONAL_HUMANOID_SOURCES
    if reference_id == "engineai_pm01":
        source_class = "high_torque" if declared_effort == 164.0 else "standard"
    elif reference_id == "deeprobotics_lite3":
        source_class = "knee" if "Knee" in source_joint_name else "hip"
    elif candidate_only:
        source_class = "candidate_source_declared_limit"
    else:
        source_class = "uniform_placeholder" if placeholder else "urdf_declared"
    return SourceMotorConfig(
        provenance="selected_source_urdf_joint_limit",
        confidence=(
            "placeholder_rejected"
            if placeholder
            else (
                "candidate_source_declared_limit_only"
                if candidate_only
                else "source_declared_limit_only"
            )
        ),
        source_path=_repo_relative(source_path),
        source_sha256=_sha256_path(source_path),
        source_motor_class=source_class,
        declared_effort_limit=declared_effort,
        declared_velocity_limit=declared_velocity,
        usable_as_quantitative_prior=(
            not placeholder and not candidate_only and declared_effort is not None
        ),
        rejection_reason=(
            "All 12 joints declare uniform 1000/1000 limits; treat them as URDF "
            "placeholders rather than physical motor evidence."
            if placeholder
            else (
                "Additional humanoid source remains candidate/fail-closed and is not "
                "a promoted quantitative motor prior."
                if candidate_only
                else None
            )
        ),
        note=(
            None
            if placeholder
            else "The selected URDF declares effort and velocity only; it has no PD or armature config."
        ),
    )


class MotorDofPreservingArchetypePreviewGenerator:
    """Task070 v2 primitive witness built from an audited source joint tree."""

    profile_version = MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_PROFILE_VERSION
    contract_version = MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_VERSION
    contract_hash = MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_HASH

    def __init__(
        self,
        source_path: Path | None = None,
        *,
        reference_id: str = "unitree_g1",
    ) -> None:
        self.reference_id = reference_id
        if source_path is not None:
            self.source_path = source_path
        elif reference_id == "unitree_g1":
            self.source_path = TASK070_G1_SOURCE_PATH
        elif reference_id == "engineai_pm01":
            self.source_path = TASK070_PM01_SOURCE_PATH
        elif reference_id in TASK070_ADDITIONAL_HUMANOID_SOURCES:
            candidate_source = TASK070_ADDITIONAL_HUMANOID_SOURCES[reference_id][
                "source_path"
            ]
            if not isinstance(candidate_source, Path):
                raise TypeError("additional humanoid source path must be a Path")
            self.source_path = candidate_source
        else:
            try:
                self.source_path = TASK070_QUADRUPED_SOURCE_PATHS[reference_id]
            except KeyError as exc:
                raise ValueError(f"unsupported Task070 v2 reference: {reference_id}") from exc

    def generate(self, family: MorphologyFamily = "biped", seed: int = 0) -> MorphologyBlueprint:
        if self.reference_id == "unitree_g1":
            if family not in {"biped", "wheeled_biped"}:
                raise ValueError("the G1 source supports biped and wheeled_biped witnesses")
            descriptor = load_g1_motor_dof_preserving_descriptor(self.source_path)
            blueprint = _build_g1_descriptor_blueprint(
                descriptor,
                family="biped",
                seed=seed,
                profile_version=self.profile_version,
                contract_version=self.contract_version,
                contract_hash=self.contract_hash,
            )
            if family == "wheeled_biped":
                return _compose_terminal_wheels(blueprint, family=family)
            return blueprint
        if self.reference_id in QUADRUPED_12DOF_MOTOR_SEMANTICS:
            if family not in {"quadruped", "wheeled_quadruped"}:
                raise ValueError("quadruped sources support quadruped family witnesses")
            descriptor = load_quadruped_motor_dof_preserving_descriptor(
                self.reference_id,
                self.source_path,
            )
            blueprint = _build_quadruped_descriptor_blueprint(
                descriptor,
                seed=seed,
                profile_version=self.profile_version,
                contract_version=self.contract_version,
                contract_hash=self.contract_hash,
            )
            if family == "wheeled_quadruped":
                return _compose_terminal_wheels(blueprint, family=family)
            return blueprint
        if self.reference_id == "engineai_pm01":
            if family not in {"biped", "wheeled_biped"}:
                raise ValueError("the PM01 source supports biped family witnesses")
            descriptor = load_pm01_motor_dof_preserving_descriptor(self.source_path)
            blueprint = _build_g1_descriptor_blueprint(
                descriptor,
                family="biped",
                seed=seed,
                profile_version=self.profile_version,
                contract_version=self.contract_version,
                contract_hash=self.contract_hash,
            )
            if family == "wheeled_biped":
                return _compose_terminal_wheels(blueprint, family=family)
            return blueprint
        if self.reference_id in TASK070_ADDITIONAL_HUMANOID_SOURCES:
            if family not in {"biped", "wheeled_biped"}:
                raise ValueError(
                    "additional humanoid sources support biped family witnesses"
                )
            descriptor = load_additional_humanoid_motor_dof_preserving_descriptor(
                self.reference_id,
                self.source_path,
            )
            blueprint = _build_additional_humanoid_descriptor_blueprint(
                descriptor,
                seed=seed,
                profile_version=self.profile_version,
                contract_version=self.contract_version,
                contract_hash=self.contract_hash,
            )
            if family == "wheeled_biped":
                return _compose_terminal_wheels(blueprint, family=family)
            return blueprint
        raise ValueError(f"unsupported Task070 v2 reference: {self.reference_id}")

    def sample_physical_params(
        self,
        blueprint: MorphologyBlueprint,
        seed: int,
        *,
        range_fraction: float = 0.0,
    ) -> PhysicalParams:
        if blueprint.profile_version != self.profile_version:
            raise ValueError("v2 preview physical sampler requires a v2 preview blueprint")
        if not 0.0 <= range_fraction <= 1.0:
            raise ValueError("range_fraction must be between zero and one")
        rng = random.Random(_stable_seed("task070_v2_preview_physical", seed))

        def centered(low: float, high: float) -> float:
            midpoint = 0.5 * (low + high)
            radius = 0.5 * (high - low) * range_fraction
            return rng.uniform(midpoint - radius, midpoint + radius)

        global_scale = centered(0.98, 1.02)
        link_scales = {link.name: centered(0.98, 1.02) for link in blueprint.links}
        slots = [joint.semantic_slot for joint in blueprint.joints]
        stack = blueprint.profile_metadata.get("actuation_stack", {})
        transmission = stack.get("transmission_model", {})
        coherent = stack.get("coherent_motor_config", {})
        coverage = blueprint.profile_metadata.get("motor_configuration", {}).get(
            "source_config_coverage", {}
        )
        source_count = len(slots)
        coverage_complete = (
            coverage.get("source_motor_count") == source_count
            and coverage.get("config_record_count") == source_count
            and coverage.get("usable_quantitative_prior_count") == source_count
            and coverage.get("rejected_placeholder_count") == 0
        )
        families = coherent.get("families", [])
        groups = transmission.get("groups", [])
        family_slots = [
            slot for family in families for slot in family.get("anonymous_semantic_slots", [])
        ]
        group_slots = [
            slot for group in groups for slot in group.get("anonymous_semantic_slots", [])
        ]
        topology_coverage_exact = (
            len(family_slots) == source_count
            and len(group_slots) == source_count
            and len(set(family_slots)) == source_count
            and len(set(group_slots)) == source_count
            and set(family_slots) == set(slots)
            and set(group_slots) == set(slots)
        )
        companion_config = bool(families) and all(
            isinstance(family.get("source_config"), Mapping)
            and family["source_config"].get("provenance")
            == "upstream_local_companion_config"
            for family in families
        )
        candidate_fail_closed = blueprint.profile_metadata.get(
            "candidate_prior_status"
        ) == "candidate_fail_closed"
        eligible = bool(
            coverage_complete
            and topology_coverage_exact
            and companion_config
            and not candidate_fail_closed
        )
        reason = "eligible_complete_companion_source" if eligible else (
            "source_config_or_transmission_evidence_incomplete"
        )
        family_latents: dict[str, dict[str, float]] = {}
        group_latents: dict[str, float] = {}
        if eligible and range_fraction:
            for family in families:
                family_id = str(family["family_id"])
                family_latents[family_id] = {
                    "effort": 1.0 + rng.uniform(-0.10, 0.10) * range_fraction,
                    "bandwidth": 1.0 + rng.uniform(-0.10, 0.10) * range_fraction,
                }
            for group in groups:
                group_latents[str(group["group_id"])] = (
                    1.0 + rng.uniform(-0.05, 0.0) * range_fraction
                )
        family_by_slot = {
            str(slot): family
            for family in families
            for slot in family.get("anonymous_semantic_slots", [])
        }
        group_by_slot = {
            str(slot): group
            for group in groups
            for slot in group.get("anonymous_semantic_slots", [])
        }
        motor_strength: dict[str, float] = {}
        kp_scales: dict[str, float] = {}
        kd_scales: dict[str, float] = {}
        for slot in slots:
            family = family_by_slot.get(slot)
            group = group_by_slot.get(slot)
            family_id = str(family["family_id"]) if family else ""
            group_id = str(group["group_id"]) if group else ""
            effort = family_latents.get(family_id, {}).get("effort", 1.0)
            bandwidth = family_latents.get(family_id, {}).get("bandwidth", 1.0)
            efficiency = group_latents.get(group_id, 1.0)
            motor_strength[slot] = effort * efficiency
            kp_scales[slot] = bandwidth * (0.5 + 0.5 * effort)
            kd_scales[slot] = bandwidth * (0.75 + 0.25 * effort)
        delay_ms = rng.uniform(0.0, 40.0) * range_fraction if eligible else 0.0
        correlation_metadata = {
            "contract": "task071_v2_correlated_actuation_randomization_v1",
            "eligible": eligible,
            "reason": reason,
            "source_config_coverage_complete": coverage_complete,
            "candidate_fail_closed": candidate_fail_closed,
            "topology_coverage_exact": topology_coverage_exact,
            "formula": {
                "motor_strength": "family_effort * group_efficiency",
                "kp_scale": "family_bandwidth * (0.5 + 0.5 * family_effort)",
                "kd_scale": "family_bandwidth * (0.75 + 0.25 * family_effort)",
                "family_effort_bandwidth_range": "[0.90, 1.10]",
                "group_efficiency_range": "[0.95, 1.00]",
                "delay_ms_range": "[0, 40] milliseconds",
            },
            "family_latents": family_latents,
            "group_efficiency_latents": group_latents,
            "slot_composition": {
                slot: {
                    "family_id": str(family_by_slot[slot]["family_id"])
                    if slot in family_by_slot else None,
                    "group_id": str(group_by_slot[slot]["group_id"])
                    if slot in group_by_slot else None,
                    "motor_strength": motor_strength[slot],
                    "kp_scale": kp_scales[slot],
                    "kd_scale": kd_scales[slot],
                }
                for slot in slots
            },
            "independent_per_slot_noise": False,
            "delay_runtime_owner": "WholeBodyMuJoCoShard→MotorProcess",
            "nominal_strength_owner": "compile_mjcf.actuator_forcerange",
            "runtime_fault_strength_owner": "MotorProcess_with_identity_nominal_baseline",
            "exact_physical_transmission_mapping_claimed": False,
        }

        return PhysicalParams(
            global_scale=global_scale,
            link_scales=link_scales,
            mass_scales={link.name: centered(0.96, 1.04) for link in blueprint.links},
            com_offsets={
                link.name: tuple(
                    value * global_scale * link_scales[link.name] for value in link.com
                )
                for link in blueprint.links
            },
            joint_limit_scales={joint.semantic_slot: 1.0 for joint in blueprint.joints},
            nominal_offsets={joint.semantic_slot: 0.0 for joint in blueprint.joints},
            friction=centered(0.85, 1.05),
            motor_strength=motor_strength,
            kp_scales=kp_scales,
            kd_scales=kd_scales,
            delay_ms=delay_ms,
            ema_alpha=1.0,
            payload_mass=0.0,
            metadata={"task070_v2_preview": True, "task071_correlated_actuation": correlation_metadata},
        )


def _canonical_root_frame(
    family: MorphologyFamily,
    links: Sequence[LinkBlueprint],
    joints: Sequence[JointBlueprint],
) -> dict[str, object]:
    """Build the stable, virtual root pose contract for a v2 preview."""
    if family.startswith("wheeled_"):
        base_family = family.removeprefix("wheeled_")
    else:
        base_family = family
    first: list[tuple[str, LinkBlueprint]] = []
    by_name = {link.name: link for link in links}
    for index in range(4 if base_family == "quadruped" else 2):
        limb_joints = [
            joint for joint in joints if joint.semantic_slot.startswith(f"limb{index}_")
        ]
        joint = next(
            joint
            for joint in limb_joints
            if not joint.parent_link.startswith(f"anon_limb{index}_")
        )
        link = by_name[joint.child_link]
        first.append((joint.parent_link, link))
    parents = {parent for parent, _ in first}
    if len(parents) != 1:
        raise ValueError("canonical root anchors must share one parent link")
    body = first[0][0]
    points = [tuple(float(value) for value in link.pos) for _, link in first]
    if base_family == "biped":
        origin = tuple((points[0][axis] + points[1][axis]) * 0.5 for axis in range(3))
        semantics = {
            "origin": "midpoint(limb0,limb1 first joint attachment)",
            "x": "audited anchor body local +X forward",
            "y": "audited anchor body local +Y left",
            "z": "audited anchor body local +Z up",
        }
    else:
        origin = _mul(_add(_add(points[0], points[1]), _add(points[2], points[3])), 0.25)
        semantics = {
            "origin": "centroid(limb0..limb3 first joint attachments)",
            "x": "audited anchor body local +X forward",
            "y": "audited anchor body local +Y left",
            "z": "audited anchor body local +Z up",
        }
    x = (1.0, 0.0, 0.0)
    y = (0.0, 1.0, 0.0)
    z = (0.0, 0.0, 1.0)
    quat = _quat_from_axes(x, y, z)
    inv_quat = (quat[0], -quat[1], -quat[2], -quat[3])
    inverse_translation = _mul(_quat_rotate(inv_quat, origin), -1.0)
    return {
        "contract_version": "canonical_root_frame_v1",
        "site_name": "canonical_root",
        "site_body_link": body,
        "coordinate_convention": {
            "handedness": "right_handed",
            "x": "forward",
            "y": "left",
            "z": "up",
            "quaternion_order": "wxyz",
            "pose": "site_to_world",
            "twist": "expressed_in_canonical_frame/angular_then_linear",
            "projected_gravity": "world_minus_z_expressed_in_canonical",
            "transform_units": "translation_is_blueprint_length_scaled_by_physical.global_scale_at_compile",
        },
        "origin": origin,
        "axis_semantics": semantics,
        "anchor_body_from_canonical": {
            "translation": origin,
            "quaternion_wxyz": quat,
            "point_equation": "p_anchor = translation + R(quaternion) @ p_canonical",
        },
        "canonical_from_anchor_body": {
            "translation": inverse_translation,
            "quaternion_wxyz": inv_quat,
            "point_equation": "p_canonical = R(inverse_quaternion) @ (p_anchor - translation)",
        },
        "native_free_root_qpos_is_canonical": False,
        "downstream_query_guidance": "query site canonical_root pose and mj_objectVelocity(site, flg_local=1)",
    }


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(x + y) for x, y in zip(a, b))  # type: ignore[return-value]


def _mul(a: Sequence[float], scalar: float) -> tuple[float, float, float]:
    return tuple(float(value * scalar) for value in a)  # type: ignore[return-value]


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _quat_from_axes(x: Sequence[float], y: Sequence[float], z: Sequence[float]) -> tuple[float, float, float, float]:
    trace = x[0] + y[1] + z[2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * s, (y[2] - z[1]) / s, (z[0] - x[2]) / s, (x[1] - y[0]) / s)
    diagonal = (x[0], y[1], z[2])
    index = max(range(3), key=lambda i: diagonal[i])
    if index == 0:
        s = math.sqrt(1.0 + x[0] - y[1] - z[2]) * 2.0
        return ((y[2] - z[1]) / s, 0.25 * s, (y[0] + x[1]) / s, (z[0] + x[2]) / s)
    if index == 1:
        s = math.sqrt(1.0 + y[1] - x[0] - z[2]) * 2.0
        return ((z[0] - x[2]) / s, (y[0] + x[1]) / s, 0.25 * s, (z[1] + y[2]) / s)
    s = math.sqrt(1.0 + z[2] - x[0] - y[1]) * 2.0
    return ((x[1] - y[0]) / s, (z[0] + x[2]) / s, (z[1] + y[2]) / s, 0.25 * s)


def _quat_rotate(quat: Sequence[float], vector: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = quat
    qv = (x, y, z)
    t = _mul(_cross(qv, vector), 2.0)
    return _add(vector, _add(_mul(t, w), _cross(qv, t)))


def _source_motor_config_coverage(
    motors: Sequence[MotorDofSourceMotor],
) -> dict[str, object]:
    configs = [motor.source_motor_config for motor in motors]
    available = [config for config in configs if config is not None]
    usable = [config for config in available if config.usable_as_quantitative_prior]
    sources = sorted(
        {
            (config.provenance, config.source_path, config.source_sha256)
            for config in available
        }
    )
    return {
        "source_motor_count": len(motors),
        "config_record_count": len(available),
        "usable_quantitative_prior_count": len(usable),
        "rejected_placeholder_count": sum(
            not config.usable_as_quantitative_prior for config in available
        ),
        "declared_effort_count": sum(
            config.declared_effort_limit is not None for config in available
        ),
        "declared_velocity_count": sum(
            config.declared_velocity_limit is not None for config in available
        ),
        "source_pd_gain_count": sum(
            config.stiffness is not None and config.damping is not None
            for config in available
        ),
        "source_armature_count": sum(
            config.armature is not None for config in available
        ),
        "source_rotor_inertia_count": sum(
            config.rotor_inertias is not None for config in available
        ),
        "source_gear_ratio_count": sum(
            config.gear_ratios is not None for config in available
        ),
        "source_control_mode_count": sum(
            config.control_mode is not None for config in available
        ),
        "config_sources": [
            {
                "provenance": provenance,
                "source_path": source_path,
                "source_sha256": source_sha256,
            }
            for provenance, source_path, source_sha256 in sources
        ],
        "rejected_source_motor_slots": [
            motor.anonymous_semantic_slot
            for motor in motors
            if motor.source_motor_config is not None
            and not motor.source_motor_config.usable_as_quantitative_prior
        ],
    }


_G1_PARALLEL_TRANSMISSION_GROUPS: tuple[
    tuple[str, tuple[str, str]], ...
] = (
    (
        "left_ankle_pitch_roll",
        ("limb0_ankle_pitch", "limb0_ankle_roll"),
    ),
    (
        "right_ankle_pitch_roll",
        ("limb1_ankle_pitch", "limb1_ankle_roll"),
    ),
    ("waist_pitch_roll", ("waist_roll", "waist_pitch")),
)
_PM01_PARALLEL_TRANSMISSION_GROUPS: tuple[
    tuple[str, tuple[str, str]], ...
] = (
    (
        "left_ankle_pitch_roll",
        ("limb0_ankle_pitch", "limb0_ankle_roll"),
    ),
    (
        "right_ankle_pitch_roll",
        ("limb1_ankle_pitch", "limb1_ankle_roll"),
    ),
)
_CANDIDATE_PARALLEL_TRANSMISSION_GROUPS: Mapping[
    str, tuple[tuple[str, tuple[str, ...]], ...]
] = {
    "agibot_x1_serial": (
        ("left_ankle_pitch_roll", ("limb0_ankle_pitch", "limb0_ankle_roll")),
        ("right_ankle_pitch_roll", ("limb1_ankle_pitch", "limb1_ankle_roll")),
        ("lumbar_pitch_roll", ("waist_roll", "waist_pitch")),
        (
            "left_wrist_pitch_roll",
            ("left_arm_wrist_pitch", "left_arm_wrist_roll"),
        ),
        (
            "right_wrist_pitch_roll",
            ("right_arm_wrist_pitch", "right_arm_wrist_roll"),
        ),
    ),
    "engineai_t800": (
        ("left_ankle_pitch_roll", ("limb0_ankle_pitch", "limb0_ankle_roll")),
        ("right_ankle_pitch_roll", ("limb1_ankle_pitch", "limb1_ankle_roll")),
    ),
    "engineai_t800pro": (
        ("left_ankle_pitch_roll", ("limb0_ankle_pitch", "limb0_ankle_roll")),
        ("right_ankle_pitch_roll", ("limb1_ankle_pitch", "limb1_ankle_roll")),
        (
            "left_parallel_palm",
            (
                "left_hand_thumb_base",
                "left_hand_thumb_mid",
                "left_hand_thumb_tip",
                "left_hand_index_base",
                "left_hand_index_tip",
                "left_hand_middle_base",
                "left_hand_middle_tip",
            ),
        ),
        (
            "right_parallel_palm",
            (
                "right_hand_thumb_base",
                "right_hand_thumb_mid",
                "right_hand_thumb_tip",
                "right_hand_index_base",
                "right_hand_index_tip",
                "right_hand_middle_base",
                "right_hand_middle_tip",
            ),
        ),
    ),
    "limx_hu_d04": (
        ("left_ankle_pitch_roll", ("limb0_ankle_pitch", "limb0_ankle_roll")),
        ("right_ankle_pitch_roll", ("limb1_ankle_pitch", "limb1_ankle_roll")),
        ("waist_pitch_roll", ("waist_roll", "waist_pitch")),
    ),
}


def _transmission_groups(
    descriptor: G1StructuralDescriptor,
) -> list[dict[str, object]]:
    motors_by_slot = {
        motor.anonymous_semantic_slot: motor for motor in descriptor.motors
    }
    parallel_specs = (
        _G1_PARALLEL_TRANSMISSION_GROUPS
        if descriptor.reference_id == "unitree_g1"
        else (
            _PM01_PARALLEL_TRANSMISSION_GROUPS
            if descriptor.reference_id == "engineai_pm01"
            else _CANDIDATE_PARALLEL_TRANSMISSION_GROUPS.get(
                descriptor.reference_id,
                (),
            )
        )
    )
    parallel_by_slot = {
        slot: (group_name, slots)
        for group_name, slots in parallel_specs
        for slot in slots
    }
    groups: list[dict[str, object]] = []
    consumed: set[str] = set()
    for motor in descriptor.motors:
        slot = motor.anonymous_semantic_slot
        if slot in consumed:
            continue
        if slot in parallel_by_slot:
            group_name, slots = parallel_by_slot[slot]
            group_motors = [motors_by_slot[item] for item in slots]
            if descriptor.reference_id == "unitree_g1":
                configs = [item.source_motor_config for item in group_motors]
                if any(config is None for config in configs):
                    raise ValueError(
                        f"G1 parallel group {group_name!r} lacks motor evidence"
                    )
                config = configs[0]
                assert config is not None
                evidence = {
                    "path": config.source_path,
                    "sha256": config.source_sha256,
                }
                kind = "parallel_two_axis_two_motor_nominal_aggregate"
                fail_closed_reason = (
                    "The audited companion config reports a four-bar linkage and "
                    "two 5020 actuators, but explicitly says its exact geometry is "
                    "unknown and uses a nominal 1:1 aggregate approximation."
                )
            elif descriptor.reference_id == "engineai_pm01":
                evidence = {
                    "paths": [
                        _repo_relative(TASK070_PM01_NATIVE_TRANSFORM_CONFIG_PATH),
                        _repo_relative(
                            TASK070_PM01_NATIVE_PARALLEL_ANKLE_CONFIG_PATH
                        ),
                    ],
                    "sha256": [
                        _audited_companion_config_sha256(
                            TASK070_PM01_NATIVE_TRANSFORM_CONFIG_PATH,
                            TASK070_PM01_NATIVE_TRANSFORM_CONFIG_SHA256,
                        ),
                        _audited_companion_config_sha256(
                            TASK070_PM01_NATIVE_PARALLEL_ANKLE_CONFIG_PATH,
                            TASK070_PM01_NATIVE_PARALLEL_ANKLE_CONFIG_SHA256,
                        ),
                    ],
                }
                kind = "parallel_two_axis_two_motor_related_variant_mapping"
                fail_closed_reason = (
                    "The official PM01-Edu native SDK publishes the parallel-ankle "
                    "converter and geometry, but it has 24 enabled slots including a "
                    "head motor while this selected PM01 descriptor has 23. The shared "
                    "ankle names support topology only until variant alignment is audited."
                )
            else:
                candidate = _additional_humanoid_candidate_evidence(
                    descriptor.reference_id
                )
                evidence = {
                    "paths": [
                        item["path"]
                        for item in candidate[
                            "transmission_and_motor_config_evidence"
                        ]
                    ],
                    "sha256": [
                        item["sha256"]
                        for item in candidate[
                            "transmission_and_motor_config_evidence"
                        ]
                    ],
                }
                kind = "source_parallel_group_candidate_fail_closed"
                fail_closed_reason = (
                    "Published candidate evidence identifies this coupled group, but "
                    "the exact runtime mapping and quantitative-prior promotion gate "
                    "remain incomplete."
                )
            groups.append(
                {
                    "group_id": f"source_{group_name}",
                    "kind": kind,
                    "source_joint_names": [
                        item.source_joint_name for item in group_motors
                    ],
                    "anonymous_semantic_slots": list(slots),
                    "modeled_physical_actuator_count": len(slots),
                    "source_evidence": evidence,
                    "exact_kinematic_mapping_available": False,
                    "mapping_usable_as_quantitative_prior": False,
                    "fail_closed_reason": fail_closed_reason,
                }
            )
            consumed.update(slots)
            continue

        config = motor.source_motor_config
        has_companion_config = bool(
            config is not None
            and config.provenance == "upstream_local_companion_config"
        )
        groups.append(
            {
                "group_id": f"source_{slot}",
                "kind": (
                    "one_to_one_joint_space_companion_proxy"
                    if has_companion_config
                    else "source_unspecified_joint_space_proxy"
                ),
                "source_joint_names": [motor.source_joint_name],
                "anonymous_semantic_slots": [slot],
                "modeled_physical_actuator_count": 1,
                "source_evidence": (
                    {
                        "path": config.source_path,
                        "sha256": config.source_sha256,
                    }
                    if config is not None
                    else None
                ),
                "exact_kinematic_mapping_available": False,
                "mapping_usable_as_quantitative_prior": False,
                "fail_closed_reason": (
                    "The selected source exposes a joint-space actuator/config hint, "
                    "not a cleared physical transmission ratio or Jacobian."
                ),
            }
        )
        consumed.add(slot)
    return groups


def _validate_transmission_coverage(
    groups: Sequence[Mapping[str, object]],
    expected_slots: Sequence[str],
) -> None:
    covered = [
        str(slot)
        for group in groups
        for slot in group["anonymous_semantic_slots"]  # type: ignore[index]
    ]
    if len(covered) != len(set(covered)):
        raise ValueError("a motor slot belongs to more than one transmission group")
    if set(covered) != set(expected_slots) or len(covered) != len(expected_slots):
        raise ValueError("transmission groups do not exactly cover generalized motor slots")


def _coherent_motor_families(
    descriptor: G1StructuralDescriptor,
    resolved_motor_configs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    configs_by_slot = {
        motor.anonymous_semantic_slot: motor.source_motor_config
        for motor in descriptor.motors
    }
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in resolved_motor_configs:
        slot = str(record["anonymous_semantic_slot"])
        source = configs_by_slot[slot]
        source_identity = (
            source.manifest()
            if source is not None
            else {
                "source_motor_class": None,
                "raw_declared": None,
                "usable_as_quantitative_prior": False,
            }
        )
        key = json.dumps(source_identity, sort_keys=True, separators=(",", ":"))
        grouped.setdefault(key, []).append(record)

    families: list[dict[str, object]] = []
    for index, key in enumerate(sorted(grouped)):
        records = grouped[key]
        first = records[0]
        source = configs_by_slot[str(first["anonymous_semantic_slot"])]
        families.append(
            {
                "family_id": f"motor_family_{index:02d}",
                "source_motor_class": (
                    source.source_motor_class if source is not None else None
                ),
                "anonymous_semantic_slots": [
                    str(record["anonymous_semantic_slot"]) for record in records
                ],
                "source_config": source.manifest() if source is not None else None,
                "randomization_unit": "shared_motor_family_latent",
                "independent_per_slot_scalar_randomization_allowed": False,
            }
        )
    return families


def _actuation_stack_manifest(
    descriptor: G1StructuralDescriptor,
    resolved_motor_configs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    slots = [motor.anonymous_semantic_slot for motor in descriptor.motors]
    groups = _transmission_groups(descriptor)
    _validate_transmission_coverage(groups, slots)
    families = _coherent_motor_families(descriptor, resolved_motor_configs)
    related_config_evidence: list[dict[str, object]] = []
    if descriptor.reference_id == "engineai_pm01":
        related_config_evidence.append(
            {
                "source_variant": "official_pm01_edu_native_sdk",
                "paths": [
                    _repo_relative(TASK070_PM01_NATIVE_MOTOR_CONFIG_PATH),
                    _repo_relative(TASK070_PM01_NATIVE_RL_CONFIG_PATH),
                ],
                "sha256": [
                    _audited_companion_config_sha256(
                        TASK070_PM01_NATIVE_MOTOR_CONFIG_PATH,
                        TASK070_PM01_NATIVE_MOTOR_CONFIG_SHA256,
                    ),
                    _audited_companion_config_sha256(
                        TASK070_PM01_NATIVE_RL_CONFIG_PATH,
                        TASK070_PM01_NATIVE_RL_CONFIG_SHA256,
                    ),
                ],
                "published_enabled_motor_count": 24,
                "selected_descriptor_shared_named_motor_count": 23,
                "extra_related_variant_motor": "J23_HEAD_YAW",
                "published_fields": [
                    "motor_enable",
                    "motor_sign",
                    "default_position_offset",
                    "joint_kp",
                    "joint_kd",
                    "action_scale",
                    "control_dt",
                ],
                "applied_to_resolved_anonymous_actuators": False,
                "fail_closed_reason": (
                    "The related PM01-Edu config is not silently merged into the "
                    "23-DoF selected PM01 source until variant alignment is audited."
                ),
            }
        )
    return {
        "contract": "task070_actuation_stack_v1",
        "structural_descriptor": {
            "reference_id": descriptor.reference_id,
            "descriptor_sha256": descriptor.descriptor_hash,
            "source_actuated_motor_count": len(descriptor.motors),
            "anonymous_generalized_motor_slots": slots,
            "source_parent_child_axis_range_order_preserved": True,
            "added_local_wheel_slots": [],
        },
        "transmission_model": {
            "groups": groups,
            "generalized_joint_coverage_exact": True,
            "generalized_joint_slot_count": len(slots),
            "modeled_physical_actuator_count": sum(
                int(group["modeled_physical_actuator_count"]) for group in groups
            ),
            "exact_physical_mapping_claimed": False,
        },
        "coherent_motor_config": {
            "families": families,
            "resolved_anonymous_actuators": list(resolved_motor_configs),
            "related_official_config_evidence": related_config_evidence,
            "independent_scalar_randomization_allowed": False,
            "required_correlated_latents": [
                "motor_family_strength_and_effort",
                "motor_family_speed_bandwidth",
                "motor_family_kp_kd_armature",
                "transmission_group_efficiency_backlash_and_coupling",
            ],
            "exact_named_robot_parameter_parity_claimed": False,
        },
        "runtime_fault_process": {
            "status": "declared_not_applied_in_task070_preview",
            "current_implementation": (
                "h200_locomotion_lab.robots.motor_process.MotorProcess"
            ),
            "current_process_coordinate": "generalized_joint_action_slot",
            "current_supported_events": ["weak", "dead", "latency"],
            "preview_faults_sampled": False,
            "physical_parallel_motor_fault_projection": "not_implemented",
            "required_future_rule": (
                "sample physical-motor events, then project through the active "
                "transmission Jacobian before generalized-joint control"
            ),
        },
    }


def _compose_wheel_actuation_stack(
    base_stack: Mapping[str, object],
    wheel_records: Sequence[Mapping[str, object]],
    wheel_motor_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    structural = dict(base_stack["structural_descriptor"])  # type: ignore[arg-type]
    transmission = dict(base_stack["transmission_model"])  # type: ignore[arg-type]
    coherent = dict(base_stack["coherent_motor_config"])  # type: ignore[arg-type]
    groups = list(transmission["groups"])  # type: ignore[arg-type]
    wheel_slots = [str(record["semantic_slot"]) for record in wheel_records]
    for record in wheel_records:
        slot = str(record["semantic_slot"])
        groups.append(
            {
                "group_id": f"local_{slot}",
                "kind": "continuous_wheel_direct",
                "source_joint_names": [],
                "anonymous_semantic_slots": [slot],
                "modeled_physical_actuator_count": 1,
                "source_evidence": {
                    "provenance": "local_engineering_module",
                    "parent_non_wheel_link": record["parent_non_wheel_link"],
                },
                "exact_kinematic_mapping_available": True,
                "mapping_usable_as_quantitative_prior": True,
                "fail_closed_reason": None,
            }
        )
    expected_slots = [
        *structural["anonymous_generalized_motor_slots"],  # type: ignore[misc]
        *wheel_slots,
    ]
    _validate_transmission_coverage(groups, expected_slots)
    transmission.update(
        {
            "groups": groups,
            "generalized_joint_slot_count": len(expected_slots),
            "modeled_physical_actuator_count": int(
                transmission["modeled_physical_actuator_count"]
            )
            + len(wheel_slots),
            "local_direct_wheel_mapping_count": len(wheel_slots),
        }
    )
    families = list(coherent["families"])  # type: ignore[arg-type]
    families.append(
        {
            "family_id": "local_continuous_wheel_motor_family",
            "source_motor_class": "local_continuous_wheel",
            "anonymous_semantic_slots": wheel_slots,
            "source_config": {
                "provenance": "local_engineering_module",
                "raw_declared": {
                    "effort_limit": 45.0,
                    "velocity_limit": None,
                    "stiffness": 0.0,
                    "damping": 0.15,
                    "armature": 0.01,
                },
                "usable_as_quantitative_prior": True,
            },
            "randomization_unit": "shared_motor_family_latent",
            "independent_per_slot_scalar_randomization_allowed": False,
        }
    )
    coherent.update(
        {
            "families": families,
            "resolved_anonymous_actuators": [
                *coherent["resolved_anonymous_actuators"],  # type: ignore[misc]
                *wheel_motor_records,
            ],
        }
    )
    structural.update(
        {
            "added_local_wheel_slots": wheel_slots,
            "anonymous_generalized_motor_slot_count": len(expected_slots),
        }
    )
    return {
        **base_stack,
        "structural_descriptor": structural,
        "transmission_model": transmission,
        "coherent_motor_config": coherent,
    }


def _anonymous_motor_control_profile(
    motor: MotorDofSourceMotor,
    *,
    family: Literal["biped", "quadruped"],
    lever_scale: float,
) -> tuple[float, float, float, float, float, float, dict[str, object]]:
    if family == "quadruped":
        fallback_kp, fallback_kd, fallback_effort, fallback_armature = (
            35.0,
            1.6,
            45.0,
            0.02,
        )
        proxy_kp_per_effort = 0.85
        proxy_kd_per_kp = 0.05
        joint_friction = 0.015
    elif motor.module.endswith("_leg"):
        fallback_kp, fallback_kd, fallback_effort, fallback_armature = (
            55.0,
            2.5,
            90.0,
            0.02,
        )
        proxy_kp_per_effort = 0.60
        proxy_kd_per_kp = 0.06
        joint_friction = 0.02
    elif motor.module.endswith("_arm"):
        wrist = "_wrist_" in motor.anonymous_semantic_slot
        fallback_kp, fallback_kd, fallback_effort, fallback_armature = (
            (18.0, 0.8, 18.0, 0.006)
            if wrist
            else (25.0, 1.0, 30.0, 0.008)
        )
        proxy_kp_per_effort = 0.60
        proxy_kd_per_kp = 0.06
        joint_friction = 0.015
    else:
        fallback_kp, fallback_kd, fallback_effort, fallback_armature = (
            45.0,
            2.0,
            60.0,
            0.015,
        )
        proxy_kp_per_effort = 0.60
        proxy_kd_per_kp = 0.06
        joint_friction = 0.02

    source = motor.source_motor_config
    source_hint_used = bool(
        source is not None
        and source.usable_as_quantitative_prior
        and source.declared_effort_limit is not None
    )
    if source_hint_used and source is not None:
        raw_effort = float(source.declared_effort_limit)
        if source.stiffness is not None and source.damping is not None:
            raw_kp = source.stiffness
            raw_kd = source.damping
            gain_derivation = "source_companion_config"
        else:
            raw_kp = raw_effort * proxy_kp_per_effort
            raw_kd = raw_kp * proxy_kd_per_kp
            gain_derivation = "generic_pd_proxy_from_source_effort_class"
        raw_armature = (
            source.armature if source.armature is not None else fallback_armature
        )
        fallback_reason = None
    else:
        raw_kp = fallback_kp
        raw_kd = fallback_kd
        raw_effort = fallback_effort
        raw_armature = fallback_armature
        gain_derivation = "anonymous_family_fallback"
        fallback_reason = (
            source.rejection_reason
            if source is not None and source.rejection_reason is not None
            else "no usable source motor config"
        )

    inertia_gain_scale = lever_scale**2
    final_kp = raw_kp * inertia_gain_scale
    final_kd = raw_kd * inertia_gain_scale
    final_effort = raw_effort * lever_scale
    final_armature = raw_armature * inertia_gain_scale
    joint_damping = min(0.25, max(0.03, 0.08 * final_kd))
    record = {
        "source_joint_name": motor.source_joint_name,
        "anonymous_semantic_slot": motor.anonymous_semantic_slot,
        "source_motor_class": (
            source.source_motor_class if source is not None else None
        ),
        "source_hint_used": source_hint_used,
        "source_config_usable_as_quantitative_prior": (
            source.usable_as_quantitative_prior if source is not None else False
        ),
        "raw_or_proxy": {
            "kp": raw_kp,
            "kd": raw_kd,
            "effort_limit": raw_effort,
            "armature": raw_armature,
            "velocity_limit_hint": (
                source.declared_velocity_limit if source is not None else None
            ),
            "gain_derivation": gain_derivation,
        },
        "anonymous_transform": {
            "mass_scale": 1.0,
            "lever_scale": lever_scale,
            "inertia_and_gain_scale": inertia_gain_scale,
            "effort_scale": lever_scale,
            "policy": "fixed anonymous link mass with lever-aware inertia/gain scaling",
        },
        "final_compiled": {
            "kp": final_kp,
            "kd": final_kd,
            "effort_limit": final_effort,
            "armature": final_armature,
            "joint_damping": joint_damping,
            "joint_friction": joint_friction,
        },
        "velocity_limit_runtime_enforced": False,
        "velocity_limit_note": (
            "Recorded as a controller-side speed hint; MuJoCo position actuators "
            "do not provide a direct joint-speed limit attribute."
        ),
        "fallback_reason": fallback_reason,
    }
    return (
        final_kp,
        final_kd,
        final_effort,
        final_armature,
        joint_damping,
        joint_friction,
        record,
    )


def _build_g1_descriptor_blueprint(
    descriptor: G1StructuralDescriptor,
    *,
    family: MorphologyFamily,
    seed: int,
    profile_version: str,
    contract_version: str,
    contract_hash: str,
) -> MorphologyBlueprint:
    rng = random.Random(_stable_seed("task070_v2_g1_descriptor_preview", seed))
    root_link = next(
        motor.anonymous_parent_link
        for motor in descriptor.motors
        if motor.source_parent_body == descriptor.source_root_body
    )
    torso_slot = [
        motor.anonymous_semantic_slot
        for motor in descriptor.motors
        if motor.module == "waist"
    ][-1]
    module_scales = {
        "left_leg": rng.uniform(1.00, 1.04),
        "right_leg": rng.uniform(1.00, 1.04),
        "waist": rng.uniform(0.98, 1.02),
        "left_arm": rng.uniform(1.08, 1.16),
        "right_arm": rng.uniform(1.08, 1.16),
    }
    pelvis_half_x = 0.125 * module_scales["waist"]
    pelvis_half_y = 0.090 * module_scales["waist"]
    pelvis_height = 0.13 * module_scales["waist"]
    pelvis_geom_center_z = -0.065 * module_scales["waist"]
    torso_half_x = 0.135 * module_scales["waist"]
    torso_half_y = 0.112 * module_scales["waist"]
    torso_height = 0.35 * module_scales["waist"]
    torso_geom_center_z = 0.19 * module_scales["waist"]
    links: list[LinkBlueprint] = [
        LinkBlueprint(
            name=root_link,
            parent="root",
            geom_type="box",
            size=(pelvis_half_x, pelvis_half_y),
            pos=(0.0, 0.0, 0.0),
            length=pelvis_height,
            mass=4.1,
            contact=False,
            com=(0.0, 0.0, -0.06),
        )
    ]
    joints: list[JointBlueprint] = []
    actuators: list[ActuatorBlueprint] = []
    end_sites: list[str] = []
    realized_positions: list[dict[str, object]] = []
    resolved_motor_configs: list[dict[str, object]] = []
    link_visual_rgba: dict[str, tuple[float, float, float, float]] = {
        root_link: (0.84, 0.08, 0.06, 1.0)
    }
    capsule_local_fromto: dict[str, tuple[float, float, float, float, float, float]] = {}
    body_local_quat: dict[str, tuple[float, float, float, float]] = {
        root_link: (1.0, 0.0, 0.0, 0.0)
    }
    geom_local_pos: dict[str, tuple[float, float, float]] = {
        root_link: (0.0, 0.0, pelvis_geom_center_z)
    }
    footpad_local_pos: dict[str, tuple[float, float, float]] = {}
    auxiliary_capsule_visuals: dict[str, list[dict[str, object]]] = {}

    for motor in descriptor.motors:
        scale = module_scales[motor.module]
        radius = _g1_link_radius(motor)
        foot = motor.anonymous_semantic_slot in {"limb0_ankle_roll", "limb1_ankle_roll"}
        geom_type = "box" if motor.anonymous_semantic_slot == torso_slot else "capsule"
        length = (
            torso_height
            if geom_type == "box"
            else _g1_link_visual_length(motor, descriptor, module_scales)
        )
        pos, adjustment = _g1_anonymous_body_pos(
            motor,
            scale=scale,
            pelvis_half_y=pelvis_half_y,
            torso_half_y=torso_half_y,
            leg_radius=_g1_link_radius_for_module("left_leg"),
            arm_radius=_g1_link_radius_for_module("left_arm"),
        )
        mass = _g1_link_mass(motor)
        (
            kp,
            kd,
            effort,
            armature,
            joint_damping,
            joint_friction,
            resolved_motor_config,
        ) = _anonymous_motor_control_profile(
            motor,
            family="biped",
            lever_scale=scale,
        )
        if motor.anonymous_semantic_slot.endswith("hip_pitch"):
            attachment_start = (
                pos[0],
                math.copysign(pelvis_half_y, pos[1]),
                pos[2],
            )
            auxiliary_capsule_visuals.setdefault(motor.anonymous_parent_link, []).append(
                {
                    "name": f"{motor.anonymous_semantic_slot}_attachment",
                    "fromto": (*attachment_start, *pos),
                    "radius": 0.028,
                    "rgba": _g1_link_rgba(motor),
                }
            )
        elif motor.anonymous_semantic_slot.endswith("shoulder_pitch"):
            attachment_start = (
                pos[0],
                math.copysign(torso_half_y, pos[1]),
                pos[2],
            )
            auxiliary_capsule_visuals.setdefault(motor.anonymous_parent_link, []).append(
                {
                    "name": f"{motor.anonymous_semantic_slot}_attachment",
                    "fromto": (*attachment_start, *pos),
                    "radius": 0.024,
                    "rgba": _g1_link_rgba(motor),
                }
            )
        links.append(
            LinkBlueprint(
                name=motor.anonymous_child_link,
                parent=motor.anonymous_parent_link,
                geom_type=geom_type,
                size=(
                    (torso_half_x, torso_half_y)
                    if geom_type == "box"
                    else (radius, 0.5 * length)
                ),
                pos=pos,
                length=length,
                mass=mass,
                contact=foot,
                end_site=foot,
                foot=foot,
                foot_size=_g1_foot_size(scale) if foot else None,
                com=(0.0, 0.0, 0.08 if geom_type == "box" else 0.0),
            )
        )
        link_visual_rgba[motor.anonymous_child_link] = (
            (0.05, 0.15, 0.86, 1.0)
            if motor.anonymous_semantic_slot == torso_slot
            else _g1_link_rgba(motor)
        )
        body_local_quat[motor.anonymous_child_link] = motor.source_body_local_quat
        if geom_type == "box":
            geom_local_pos[motor.anonymous_child_link] = (0.0, 0.0, torso_geom_center_z)
        if geom_type == "capsule":
            capsule_local_fromto[motor.anonymous_child_link] = _g1_capsule_local_fromto(
                motor,
                descriptor,
                module_scales,
                length=length,
            )
        if foot:
            footpad_local_pos[motor.anonymous_child_link] = (
                0.035 * scale,
                0.0,
                -0.058 * scale,
            )
        joints.append(
            JointBlueprint(
                name=f"{motor.anonymous_child_link}_joint",
                parent_link=motor.anonymous_parent_link,
                child_link=motor.anonymous_child_link,
                semantic_slot=motor.anonymous_semantic_slot,
                axis_name=_axis_name_from_semantic_slot(motor.anonymous_semantic_slot),
                axis=motor.normalized_local_axis,
                joint_range=motor.joint_range,
                nominal=_g1_preview_joint_nominal(motor),
                damping=joint_damping,
                friction=joint_friction,
                armature=armature,
            )
        )
        actuators.append(
            ActuatorBlueprint(
                name=f"{motor.anonymous_child_link}_joint_actuator",
                joint_name=f"{motor.anonymous_child_link}_joint",
                semantic_slot=motor.anonymous_semantic_slot,
                kp=kp,
                kd=kd,
                effort_limit=effort,
            )
        )
        if foot:
            end_sites.append(f"{motor.anonymous_child_link}_foot")
        resolved_motor_configs.append(resolved_motor_config)
        realized_positions.append(
            {
                "source_joint_name": motor.source_joint_name,
                "anonymous_semantic_slot": motor.anonymous_semantic_slot,
                "source_body_local_pos": motor.source_body_local_pos,
                "source_body_local_quat": motor.source_body_local_quat,
                "anonymous_body_local_pos": pos,
                "anonymous_body_local_quat": motor.source_body_local_quat,
                "module_scale": scale,
                "surface_clearance_adjustment": adjustment,
            }
        )

    descriptor_hash = descriptor.descriptor_hash
    topology_payload = {
        "profile_version": profile_version,
        "descriptor_hash": descriptor_hash,
        "family": family,
        "anonymous_links": [(link.name, link.parent, link.geom_type) for link in links],
        "anonymous_joints": [
            (
                joint.semantic_slot,
                joint.parent_link,
                joint.child_link,
                joint.axis_name,
                tuple(round(value, 6) for value in joint.axis),
            )
            for joint in joints
        ],
    }
    profile_metadata = {
        "task": "task070-archetype-constrained-standable-morphology",
        "profile_version": profile_version,
        "preview_status": "descriptor_driven_preview_pending_agent_visual_check",
        "counts_toward_task070_v2_pass": False,
        "reference_registry_sha256": TASK070_REFERENCE_REGISTRY_SHA256,
        "source_license_matrix_sha256": TASK070_SOURCE_LICENSE_MATRIX_SHA256,
        "source_reference_id": descriptor.reference_id,
        "structural_center_id": descriptor.center_id,
        "structural_descriptor_sha256": descriptor_hash,
        "source_sha256": descriptor.source_sha256,
        "source_path": descriptor.source_path,
        "source_tree_descriptor": descriptor.manifest(),
        "source_to_anonymous_frame_transform": dict(
            descriptor.source_to_anonymous_frame_transform
        ),
        "motor_accounting": {
            "source_actuated_motor_count": len(descriptor.motors),
            "anonymous_motor_count": len(joints),
            "source_non_wheel_motor_count": len(descriptor.motors),
            "anonymous_non_wheel_motor_count": len(joints),
            "added_wheel_motor_count": 0,
            "total_actuator_count": len(actuators),
            "module_dof_counts": descriptor.module_dof_counts,
            "bijection_passed": (
                len(descriptor.motors) == len(joints) == len(actuators)
            ),
        },
        "source_to_anonymous_motor_bijection": [
            motor.manifest() for motor in descriptor.motors
        ],
        "motor_configuration": {
            "contract": "task070_v2_source_hint_anonymous_scaling_v1",
            "source_config_coverage": _source_motor_config_coverage(descriptor.motors),
            "resolved_anonymous_actuators": resolved_motor_configs,
            "policy_action": "target_joint_position",
            "exact_named_robot_parameter_parity_claimed": False,
            "source_values_are_prior_hints_not_identity": True,
            "control_gain_randomization_required_for_training_distribution": True,
        },
        "actuation_stack": _actuation_stack_manifest(
            descriptor,
            resolved_motor_configs,
        ),
        "anonymous_body_tree": {
            "links": [
                {"name": link.name, "parent": link.parent, "geom_type": link.geom_type}
                for link in links
            ],
            "joint_parent_child_edges": [
                {
                    "semantic_slot": joint.semantic_slot,
                    "parent_link": joint.parent_link,
                    "child_link": joint.child_link,
                    "axis": joint.axis,
                }
                for joint in joints
            ],
        },
        "geometry_randomization": {
            "seed": seed,
            "module_scales": module_scales,
            "realized_source_tree_positions": realized_positions,
        },
        "capsule_local_fromto": capsule_local_fromto,
        "body_local_quat": body_local_quat,
        "geom_local_pos": geom_local_pos,
        "footpad_local_pos": footpad_local_pos,
        "footpad_visual_rgba": (0.16, 0.18, 0.22, 1.0),
        "auxiliary_capsule_visuals": auxiliary_capsule_visuals,
        "link_visual_rgba": link_visual_rgba,
        "joint_marker_sites": "source_motor_origins",
        "joint_marker_rgba": (1.0, 0.88, 0.10, 1.0),
        "visual_audit_nominal_joint_pose": {
            motor.anonymous_semantic_slot: _g1_preview_joint_nominal(motor)
            for motor in descriptor.motors
        },
        "primitive_link_visual_contract": {
            "segment_rule": "current motor body origin to next descriptor joint origin",
            "body_frame_rule": "preserve parsed source body-local quaternion",
            "torso_pelvis_rule": "separate locally offset primitive boxes with visible waist stack",
            "terminal_rule": "ankle terminal capsule and footpad share an explicit local endpoint",
        },
        "primitive_geometry_only": True,
        "mesh_texture_logo_copied": False,
        "stance_claim": "not_run_preview_only",
    }
    profile_metadata["canonical_root_frame"] = _canonical_root_frame(
        family,
        links,
        joints,
    )
    return MorphologyBlueprint(
        family=family,
        seed=seed,
        links=tuple(links),
        joints=tuple(joints),
        actuators=tuple(actuators),
        nominal_height=_g1_nominal_height(links),
        has_arms=True,
        structural_hash=_hash_payload(topology_payload)[:16],
        end_sites=tuple(end_sites),
        profile_version=profile_version,
        contract_version=contract_version,
        contract_hash=contract_hash,
        profile_metadata=profile_metadata,
    )


def _candidate_humanoid_joint_nominal(
    motor: MotorDofSourceMotor,
    *,
    reference_id: str,
) -> float:
    slot = motor.anonymous_semantic_slot
    preferred = 0.0
    # T1's arm source frames place the shoulder-to-elbow chain almost entirely
    # on the lateral axis.  A mirrored shoulder-roll audit pose makes the
    # primitive witness legible without changing the parsed source geometry or
    # generalized-joint contract.
    if reference_id in {"booster_t1_23", "booster_t1_29"}:
        if slot == "left_arm_shoulder_roll":
            preferred = -0.42
        elif slot == "right_arm_shoulder_roll":
            preferred = 0.42
        elif slot == "left_arm_elbow_yaw":
            preferred = 0.58
        elif slot == "right_arm_elbow_yaw":
            preferred = -0.58
    if slot.endswith("_hip_pitch"):
        preferred = -0.24
        if reference_id == "agibot_x1_serial" and slot == "limb1_hip_pitch":
            preferred = 0.24
    elif slot.endswith("_knee_pitch"):
        preferred = 0.56
    elif slot.endswith("_ankle_pitch"):
        preferred = -0.20
    elif slot.endswith("_elbow_pitch"):
        preferred = 0.62
    lower, upper = motor.joint_range
    margin = min(0.02, 0.04 * (upper - lower))
    if lower + margin <= preferred <= upper - margin:
        return preferred
    if lower + margin <= -preferred <= upper - margin:
        return -preferred
    return min(upper - margin, max(lower + margin, 0.0))


def _candidate_humanoid_link_rgba(
    module: str,
) -> tuple[float, float, float, float]:
    return {
        "left_leg": (0.04, 0.68, 0.18, 1.0),
        "right_leg": (0.78, 0.08, 0.78, 1.0),
        "waist": (0.94, 0.78, 0.06, 1.0),
        "left_arm": (0.02, 0.72, 0.88, 1.0),
        "right_arm": (0.94, 0.18, 0.08, 1.0),
        "left_hand": (0.08, 0.28, 0.92, 1.0),
        "right_hand": (0.12, 0.78, 0.22, 1.0),
        "head": (0.72, 0.72, 0.76, 1.0),
    }.get(module, (0.42, 0.44, 0.48, 1.0))


def _candidate_humanoid_endpoint(
    node: G1SourceBodyNode,
    *,
    nodes_by_source: Mapping[str, G1SourceBodyNode],
    motor: MotorDofSourceMotor | None,
    visual_scale: float,
    terminal: bool,
) -> tuple[float, float, float]:
    candidates = [
        tuple(float(value) * visual_scale for value in nodes_by_source[child].source_body_local_pos)
        for child in node.child_bodies
        if child in nodes_by_source
    ]
    nonzero = [item for item in candidates if _g1_vector_norm(item) > 0.012]
    if nonzero:
        return max(nonzero, key=_g1_vector_norm)  # type: ignore[return-value]
    module = motor.module if motor is not None else "passive"
    if terminal:
        return (0.075 * visual_scale, 0.0, -0.055 * visual_scale)
    if module.endswith("_hand"):
        return (0.032 * visual_scale, 0.0, 0.0)
    if module.endswith("_arm"):
        return (0.065 * visual_scale, 0.0, 0.0)
    if module == "head":
        return (0.0, 0.0, 0.070 * visual_scale)
    if module == "waist":
        return (0.0, 0.0, 0.080 * visual_scale)
    return (0.0, 0.0, 0.026 * visual_scale)


def _candidate_humanoid_source_world_pose(
    node: G1SourceBodyNode,
    nodes_by_source: Mapping[str, G1SourceBodyNode],
) -> tuple[float, tuple[float, float, float, float]]:
    position_z = 0.0
    source_quat = (1.0, 0.0, 0.0, 0.0)
    source_root = next(
        item.source_body_name
        for item in nodes_by_source.values()
        if item.source_parent_body == "root"
    )
    chain: list[G1SourceBodyNode] = [node]
    current = node
    while current.source_parent_body != "root" and current.source_body_name != source_root:
        current = nodes_by_source[current.source_parent_body]
        if current.source_body_name != source_root:
            chain.append(current)
    for item in reversed(chain):
        w, x, y, z = source_quat
        vx, vy, vz = (float(value) for value in item.source_body_local_pos)
        if item.source_body_name == source_root:
            vx = vy = vz = 0.0
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        position_z += vz + w * tz + (x * ty - y * tx)
        bw, bx, by, bz = item.source_body_local_quat
        source_quat = (
            w * bw - x * bx - y * by - z * bz,
            w * bx + x * bw + y * bz - z * by,
            w * by - x * bz + y * bw + z * bx,
            w * bz + x * by - y * bx + z * bw,
        )
    return position_z, source_quat


def _candidate_humanoid_box_visual_layout(
    node: G1SourceBodyNode,
    *,
    is_root: bool,
    paired_hub: bool,
    nodes_by_source: Mapping[str, G1SourceBodyNode],
    visual_scale: float,
    root_shift_world: float,
    hub_shift_world: float,
) -> tuple[tuple[float, float, float], tuple[float, float], float]:
    """Place candidate torso/pelvis boxes around their source attachment envelope."""

    _, source_quat = _candidate_humanoid_source_world_pose(node, nodes_by_source)
    if is_root:
        w, x, y, z = source_quat
        qx, qy, qz = -float(x), -float(y), -float(z)
        vx, vy, vz = (0.0, 0.0, root_shift_world)
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        return (
            (
                vx + float(w) * tx + (qy * tz - qz * ty),
                vy + float(w) * ty + (qz * tx - qx * tz),
                vz + float(w) * tz + (qx * ty - qy * tx),
            ),
            (0.135 * visual_scale, 0.105 * visual_scale),
            0.180 * visual_scale,
        )
    if not paired_hub:
        return (
            (0.0, 0.0, 0.0),
            (0.080 * visual_scale, 0.080 * visual_scale),
            0.220 * visual_scale,
        )

    world_offset = (0.0, 0.0, hub_shift_world)
    w, x, y, z = source_quat
    # Convert the desired world-Z separation into this source body's local frame.
    qx, qy, qz = -float(x), -float(y), -float(z)
    vx, vy, vz = world_offset
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    local_offset = (
        vx + float(w) * tx + (qy * tz - qz * ty),
        vy + float(w) * ty + (qz * tx - qx * tz),
        vz + float(w) * tz + (qx * ty - qy * tx),
    )
    return (local_offset, (0.080 * visual_scale, 0.080 * visual_scale), 0.140 * visual_scale)


def _build_additional_humanoid_descriptor_blueprint(
    descriptor: G1StructuralDescriptor,
    *,
    seed: int,
    profile_version: str,
    contract_version: str,
    contract_hash: str,
) -> MorphologyBlueprint:
    """Build a primitive-only candidate witness, including passive branches."""

    rng = random.Random(
        _stable_seed("task070_v2_additional_humanoid_candidate_preview", seed)
    )
    visual_scale = rng.uniform(0.995, 1.015)
    nodes_by_source = {node.source_body_name: node for node in descriptor.body_tree}
    if descriptor.source_root_body not in nodes_by_source:
        raise ValueError("candidate descriptor root body is missing from its body tree")
    motor_by_child = {motor.source_child_body: motor for motor in descriptor.motors}
    terminal_motors = {
        max(
            (motor for motor in descriptor.motors if motor.module == module),
            key=lambda motor: motor.source_tree_depth,
        ).source_joint_name
        for module in ("left_leg", "right_leg")
    }
    terminal_links: list[str] = []
    links: list[LinkBlueprint] = []
    joints: list[JointBlueprint] = []
    actuators: list[ActuatorBlueprint] = []
    resolved_motor_configs: list[dict[str, object]] = []
    body_local_quat: dict[str, tuple[float, float, float, float]] = {}
    capsule_local_fromto: dict[
        str, tuple[float, float, float, float, float, float]
    ] = {}
    geom_local_pos: dict[str, tuple[float, float, float]] = {}
    footpad_local_pos: dict[str, tuple[float, float, float]] = {}
    link_visual_rgba: dict[str, tuple[float, float, float, float]] = {}
    auxiliary_capsule_visuals: dict[str, list[dict[str, object]]] = {}
    emitted_body_records: list[dict[str, object]] = []

    def is_candidate_hub(node: G1SourceBodyNode) -> bool:
        child_modules = {
            motor_by_child[child].module
            for child in node.child_bodies
            if child in motor_by_child
        }
        return node.source_body_name == descriptor.source_root_body or len(child_modules) >= 2 or (
            node.source_body_name in motor_by_child
            and motor_by_child[node.source_body_name].module == "waist"
            and len(node.child_bodies) >= 2
        )

    hub_nodes = [node for node in descriptor.body_tree if is_candidate_hub(node)]
    paired_candidates = [
        node
        for node in hub_nodes
        if node.source_body_name != descriptor.source_root_body
        and node.source_body_name in motor_by_child
        and motor_by_child[node.source_body_name].module == "waist"
    ]
    paired_hub_name = (
        min(paired_candidates, key=lambda item: item.source_tree_depth).source_body_name
        if paired_candidates
        else None
    )
    root_shift_world = 0.0
    hub_shift_world = 0.0
    if paired_hub_name is not None:
        paired_node = nodes_by_source[paired_hub_name]
        source_distance, _ = _candidate_humanoid_source_world_pose(
            paired_node, nodes_by_source
        )
        direction = -1.0 if source_distance < -1e-9 else 1.0
        total_shift = max(
            0.0,
            (0.09 + 0.07 + 0.02) * visual_scale
            - abs(source_distance) * visual_scale,
        )
        root_shift_world = -direction * 0.40 * total_shift
        hub_shift_world = direction * 0.60 * total_shift

    for node in sorted(descriptor.body_tree, key=lambda item: item.source_tree_depth):
        motor = motor_by_child.get(node.source_body_name)
        is_root = node.source_body_name == descriptor.source_root_body
        terminal = bool(motor and motor.source_joint_name in terminal_motors)
        parent = (
            "root"
            if is_root
            else nodes_by_source[node.source_parent_body].anonymous_link
        )
        pos = (
            (0.0, 0.0, 0.0)
            if is_root
            else tuple(
                float(value) * visual_scale for value in node.source_body_local_pos
            )
        )
        hub = is_candidate_hub(node)
        endpoint = _candidate_humanoid_endpoint(
            node,
            nodes_by_source=nodes_by_source,
            motor=motor,
            visual_scale=visual_scale,
            terminal=terminal,
        )
        module = motor.module if motor is not None else "passive"
        if hub:
            geom_type = "box"
            geom_local_pos[node.anonymous_link], size, length = (
                _candidate_humanoid_box_visual_layout(
                    node,
                    is_root=is_root,
                    paired_hub=node.source_body_name == paired_hub_name,
                    nodes_by_source=nodes_by_source,
                    visual_scale=visual_scale,
                    root_shift_world=root_shift_world,
                    hub_shift_world=(
                        hub_shift_world
                        if node.source_body_name == paired_hub_name
                        else 0.0
                    ),
                )
            )
        else:
            geom_type = "capsule"
            radius = (
                0.040
                if module.endswith("_leg")
                else (
                    0.027
                    if module.endswith("_arm")
                    else (0.012 if module.endswith("_hand") else 0.022)
                )
            ) * visual_scale
            length = max(0.018, _g1_vector_norm(endpoint))
            size = (radius, 0.5 * length)
            capsule_local_fromto[node.anonymous_link] = (
                0.0,
                0.0,
                0.0,
                *endpoint,
            )
        mass = (
            4.2
            if is_root
            else (
                1.15
                if module.endswith("_leg")
                else (
                    0.42
                    if module.endswith("_arm")
                    else (
                        0.055
                        if module.endswith("_hand")
                        else (0.35 if module in {"waist", "head"} else 0.08)
                    )
                )
            )
        )
        links.append(
            LinkBlueprint(
                name=node.anonymous_link,
                parent=parent,
                geom_type=geom_type,
                size=size,
                pos=pos,  # type: ignore[arg-type]
                length=length,
                mass=mass,
                contact=terminal,
                foot=terminal,
                foot_size=(0.10 * visual_scale, 0.052 * visual_scale)
                if terminal
                else None,
            )
        )
        body_local_quat[node.anonymous_link] = node.source_body_local_quat
        link_visual_rgba[node.anonymous_link] = (
            (0.82, 0.08, 0.06, 1.0)
            if is_root
            else _candidate_humanoid_link_rgba(module)
        )
        if terminal:
            footpad_local_pos[node.anonymous_link] = endpoint
            terminal_links.append(node.anonymous_link)

        child_offsets = [
            (
                child,
                tuple(
                    float(value) * visual_scale
                    for value in nodes_by_source[child].source_body_local_pos
                ),
            )
            for child in node.child_bodies
            if child in nodes_by_source
        ]
        visible_offsets = [
            (child, offset)
            for child, offset in child_offsets
            if _g1_vector_norm(offset) > 0.012
        ]
        if hub or len(visible_offsets) > 1:
            for child, offset in visible_offsets:
                auxiliary_capsule_visuals.setdefault(node.anonymous_link, []).append(
                    {
                        "name": f"branch_{nodes_by_source[child].anonymous_link}",
                        "fromto": (0.0, 0.0, 0.0, *offset),
                        "radius": 0.017 if is_root else 0.013,
                        "rgba": link_visual_rgba[node.anonymous_link],
                    }
                )

        if motor is not None:
            (
                kp,
                kd,
                effort,
                armature,
                joint_damping,
                joint_friction,
                resolved_motor_config,
            ) = _anonymous_motor_control_profile(
                motor,
                family="biped",
                lever_scale=visual_scale,
            )
            joint_name = f"{node.anonymous_link}_joint"
            joints.append(
                JointBlueprint(
                    name=joint_name,
                    parent_link=parent,
                    child_link=node.anonymous_link,
                    semantic_slot=motor.anonymous_semantic_slot,
                    axis_name=_axis_name_from_semantic_slot(
                        motor.anonymous_semantic_slot
                    ),
                    axis=motor.normalized_local_axis,
                    joint_range=motor.joint_range,
                    nominal=_candidate_humanoid_joint_nominal(
                        motor,
                        reference_id=descriptor.reference_id,
                    ),
                    damping=joint_damping,
                    friction=joint_friction,
                    armature=armature,
                )
            )
            actuators.append(
                ActuatorBlueprint(
                    name=f"{joint_name}_actuator",
                    joint_name=joint_name,
                    semantic_slot=motor.anonymous_semantic_slot,
                    kp=kp,
                    kd=kd,
                    effort_limit=effort,
                )
            )
            resolved_motor_configs.append(resolved_motor_config)
        emitted_body_records.append(
            {
                "source_body": node.source_body_name,
                "source_parent_body": node.source_parent_body,
                "anonymous_link": node.anonymous_link,
                "anonymous_parent_link": parent,
                "selected_motor_joint": node.selected_motor_joint,
                "source_body_local_pos": node.source_body_local_pos,
                "source_body_local_quat": node.source_body_local_quat,
            }
        )

    if len(terminal_links) != 2:
        raise ValueError(
            f"candidate humanoid must expose two load-bearing terminals, got {terminal_links}"
        )
    candidate_evidence = _additional_humanoid_candidate_evidence(
        descriptor.reference_id
    )
    descriptor_hash = descriptor.descriptor_hash
    source_slots = [motor.anonymous_semantic_slot for motor in descriptor.motors]
    extra_slots = sorted(set(source_slots) - set(WHOLE_BODY_SLOT_NAMES))
    actuation_stack = _actuation_stack_manifest(
        descriptor,
        resolved_motor_configs,
    )
    actuation_stack["candidate_clearance"] = candidate_evidence
    motor_count = len(descriptor.motors)
    profile_metadata = {
        "task": "task070-archetype-constrained-standable-morphology",
        "profile_version": profile_version,
        "preview_status": "additional_humanoid_candidate_pending_agent_visual_check",
        "counts_toward_task070_v2_pass": False,
        "candidate_prior_status": "candidate_fail_closed",
        "policy_adapter_compatible": False,
        "policy_adapter_reason": (
            "The frozen 45-slot policy schema does not encode every source head, "
            "hand, finger, or axial joint; arena smoke uses direct robot actuator order."
        ),
        "task070_candidate_extra_semantic_slots": extra_slots,
        "candidate_source_evidence": candidate_evidence,
        "source_reference_id": descriptor.reference_id,
        "structural_center_id": descriptor.center_id,
        "structural_descriptor_sha256": descriptor_hash,
        "source_sha256": descriptor.source_sha256,
        "source_path": descriptor.source_path,
        "source_tree_descriptor": descriptor.manifest(),
        "source_to_anonymous_frame_transform": dict(
            descriptor.source_to_anonymous_frame_transform
        ),
        "motor_accounting": {
            "source_actuated_motor_count": motor_count,
            "source_non_wheel_motor_count": motor_count,
            "anonymous_non_wheel_motor_count": len(joints),
            "anonymous_motor_count": len(joints),
            "added_wheel_motor_count": 0,
            "total_actuator_count": len(actuators),
            "configured_physical_motor_count": candidate_evidence[
                "configured_physical_motor_count"
            ],
            "source_model_config_motor_count_gap": candidate_evidence[
                "source_model_config_motor_count_gap"
            ],
            "module_dof_counts": descriptor.module_dof_counts,
            "bijection_passed": motor_count == len(joints) == len(actuators),
        },
        "source_to_anonymous_motor_bijection": [
            motor.manifest() for motor in descriptor.motors
        ],
        "motor_configuration": {
            "contract": "task070_v2_candidate_source_hint_fail_closed_v1",
            "source_config_coverage": _source_motor_config_coverage(
                descriptor.motors
            ),
            "resolved_anonymous_actuators": resolved_motor_configs,
            "candidate_source_evidence": candidate_evidence,
            "policy_action": "target_joint_position_direct_robot_order_preview",
            "exact_named_robot_parameter_parity_claimed": False,
            "source_values_are_prior_hints_not_identity": True,
            "control_gain_randomization_required_for_training_distribution": True,
        },
        "actuation_stack": actuation_stack,
        "anonymous_body_tree": _anonymous_body_tree_manifest(links, joints),
        "candidate_passive_body_emission": emitted_body_records,
        "geometry_randomization": {
            "seed": seed,
            "uniform_visual_scale": visual_scale,
        },
        "capsule_local_fromto": capsule_local_fromto,
        "body_local_quat": body_local_quat,
        "geom_local_pos": geom_local_pos,
        "footpad_local_pos": footpad_local_pos,
        "load_bearing_terminal_links": terminal_links,
        "footpad_visual_rgba": (0.16, 0.18, 0.22, 1.0),
        "auxiliary_capsule_visuals": auxiliary_capsule_visuals,
        "link_visual_rgba": link_visual_rgba,
        "joint_marker_sites": "source_motor_origins",
        "joint_marker_rgba": (1.0, 0.88, 0.10, 1.0),
        "visual_audit_nominal_joint_pose": {
            motor.anonymous_semantic_slot: _candidate_humanoid_joint_nominal(
                motor,
                reference_id=descriptor.reference_id,
            )
            for motor in descriptor.motors
        },
        "primitive_link_visual_contract": {
            "segment_rule": "current source body origin to its next source-tree child",
            "passive_body_rule": "emit anonymous primitive bodies for fixed attachment branches",
            "body_frame_rule": "preserve parsed source body-local quaternion and motor axis",
            "terminal_rule": "last motor in each complete leg owns one explicit footpad",
        },
        "render_camera_distance": 2.55,
        "primitive_geometry_only": True,
        "mesh_texture_logo_copied": False,
        "stance_claim": "not_run_candidate_preview_only",
    }
    profile_metadata["canonical_root_frame"] = _canonical_root_frame(
        "biped",
        links,
        joints,
    )
    topology_payload = {
        "profile_version": profile_version,
        "descriptor_hash": descriptor_hash,
        "family": "biped",
        "anonymous_links": [(link.name, link.parent, link.geom_type) for link in links],
        "anonymous_joints": [
            (joint.semantic_slot, joint.parent_link, joint.child_link, joint.axis)
            for joint in joints
        ],
    }
    return MorphologyBlueprint(
        family="biped",
        seed=seed,
        links=tuple(links),
        joints=tuple(joints),
        actuators=tuple(actuators),
        nominal_height=max(0.55, _g1_nominal_height(links)),
        has_arms=True,
        structural_hash=_hash_payload(topology_payload)[:16],
        end_sites=(),
        profile_version=profile_version,
        contract_version=contract_version,
        contract_hash=contract_hash,
        profile_metadata=profile_metadata,
    )


def _build_quadruped_descriptor_blueprint(
    descriptor: G1StructuralDescriptor,
    *,
    seed: int,
    profile_version: str,
    contract_version: str,
    contract_hash: str,
) -> MorphologyBlueprint:
    rng = random.Random(_stable_seed("task070_v2_quadruped_descriptor_preview", seed))
    visual_scale = rng.uniform(1.0, 1.035)
    root_motors = [
        motor
        for motor in descriptor.motors
        if motor.source_parent_body == descriptor.source_root_body
    ]
    if len(root_motors) != 4:
        raise ValueError("quadruped descriptor must expose four root hip attachments")
    attachment_x = max(abs(motor.source_body_local_pos[0]) for motor in root_motors)
    attachment_y = max(abs(motor.source_body_local_pos[1]) for motor in root_motors)
    trunk_half_x = max(0.12, 0.80 * attachment_x) * visual_scale
    trunk_half_y = max(0.04, 0.88 * attachment_y) * visual_scale
    trunk_height = max(0.09, 0.38 * attachment_x) * visual_scale
    root_link = "anon_trunk_core"
    links: list[LinkBlueprint] = [
        LinkBlueprint(
            name=root_link,
            parent="root",
            geom_type="box",
            size=(trunk_half_x, trunk_half_y),
            pos=(0.0, 0.0, 0.0),
            length=trunk_height,
            mass=7.0,
            contact=False,
        )
    ]
    joints: list[JointBlueprint] = []
    actuators: list[ActuatorBlueprint] = []
    capsule_local_fromto: dict[
        str, tuple[float, float, float, float, float, float]
    ] = {}
    body_local_quat: dict[str, tuple[float, float, float, float]] = {
        root_link: (1.0, 0.0, 0.0, 0.0)
    }
    footpad_local_pos: dict[str, tuple[float, float, float]] = {}
    auxiliary_capsule_visuals: dict[str, list[dict[str, object]]] = {root_link: []}
    link_visual_rgba: dict[str, tuple[float, float, float, float]] = {
        root_link: (0.84, 0.08, 0.06, 1.0)
    }
    terminal_records = descriptor.source_to_anonymous_frame_transform.get(
        "terminal_local_offsets"
    )
    if not isinstance(terminal_records, Mapping):
        raise TypeError("quadruped descriptor is missing terminal local offsets")
    realized_positions: list[dict[str, object]] = []
    resolved_motor_configs: list[dict[str, object]] = []

    for motor in descriptor.motors:
        pos = tuple(float(value) * visual_scale for value in motor.source_body_local_pos)
        child_motors = [
            child
            for child in descriptor.motors
            if child.source_parent_body == motor.source_child_body
        ]
        terminal = motor.anonymous_semantic_slot.endswith("knee_pitch")
        if child_motors:
            if len(child_motors) != 1:
                raise ValueError(
                    f"quadruped motor {motor.source_joint_name!r} must have one motor child"
                )
            endpoint = tuple(
                float(value) * visual_scale
                for value in child_motors[0].source_body_local_pos
            )
            endpoint_source = f"next_motor:{child_motors[0].source_joint_name}"
        else:
            record = terminal_records.get(motor.anonymous_semantic_slot)
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"missing terminal offset for {motor.anonymous_semantic_slot!r}"
                )
            offset = record.get("offset")
            if not isinstance(offset, Sequence) or len(offset) != 3:
                raise ValueError("terminal offset must contain three values")
            endpoint = tuple(float(value) * visual_scale for value in offset)
            endpoint_source = str(record["source_element"])
        length = _g1_vector_norm(endpoint)
        if length <= 1e-9:
            raise ValueError(f"zero-length visual edge for {motor.source_joint_name!r}")
        radius = 0.027 * visual_scale if terminal else 0.032 * visual_scale
        links.append(
            LinkBlueprint(
                name=motor.anonymous_child_link,
                parent=motor.anonymous_parent_link,
                geom_type="capsule",
                size=(radius, 0.5 * length),
                pos=pos,  # type: ignore[arg-type]
                length=length,
                mass=0.42 if terminal else 0.82,
                contact=terminal,
                end_site=False,
                foot=terminal,
                foot_size=(0.038 * visual_scale, 0.030 * visual_scale)
                if terminal
                else None,
            )
        )
        capsule_local_fromto[motor.anonymous_child_link] = (
            0.0,
            0.0,
            0.0,
            *endpoint,
        )
        body_local_quat[motor.anonymous_child_link] = motor.source_body_local_quat
        if terminal:
            footpad_local_pos[motor.anonymous_child_link] = endpoint  # type: ignore[assignment]
        if motor.source_parent_body == descriptor.source_root_body:
            x, y, z = pos
            attachment_start = (
                math.copysign(trunk_half_x, x),
                math.copysign(trunk_half_y, y),
                z,
            )
            auxiliary_capsule_visuals[root_link].append(
                {
                    "name": f"{motor.anonymous_semantic_slot}_attachment",
                    "fromto": (*attachment_start, *pos),
                    "radius": 0.022,
                    "rgba": _quadruped_link_rgba(motor),
                }
            )
        nominal = _quadruped_preview_joint_nominal(motor)
        joint_name = f"{motor.anonymous_child_link}_joint"
        (
            kp,
            kd,
            effort,
            armature,
            joint_damping,
            joint_friction,
            resolved_motor_config,
        ) = _anonymous_motor_control_profile(
            motor,
            family="quadruped",
            lever_scale=visual_scale,
        )
        joints.append(
            JointBlueprint(
                name=joint_name,
                parent_link=motor.anonymous_parent_link,
                child_link=motor.anonymous_child_link,
                semantic_slot=motor.anonymous_semantic_slot,
                axis_name=_axis_name_from_semantic_slot(motor.anonymous_semantic_slot),
                axis=motor.normalized_local_axis,
                joint_range=motor.joint_range,
                nominal=nominal,
                damping=joint_damping,
                friction=joint_friction,
                armature=armature,
            )
        )
        actuators.append(
            ActuatorBlueprint(
                name=f"{joint_name}_actuator",
                joint_name=joint_name,
                semantic_slot=motor.anonymous_semantic_slot,
                kp=kp,
                kd=kd,
                effort_limit=effort,
            )
        )
        resolved_motor_configs.append(resolved_motor_config)
        link_visual_rgba[motor.anonymous_child_link] = _quadruped_link_rgba(motor)
        realized_positions.append(
            {
                "source_joint_name": motor.source_joint_name,
                "anonymous_semantic_slot": motor.anonymous_semantic_slot,
                "source_body_local_pos": motor.source_body_local_pos,
                "source_body_local_quat": motor.source_body_local_quat,
                "anonymous_body_local_pos": pos,
                "anonymous_body_local_quat": motor.source_body_local_quat,
                "outgoing_edge_local_endpoint": endpoint,
                "outgoing_edge_source": endpoint_source,
                "uniform_visual_scale": visual_scale,
            }
        )

    descriptor_hash = descriptor.descriptor_hash
    profile_metadata = {
        "task": "task070-archetype-constrained-standable-morphology",
        "profile_version": profile_version,
        "preview_status": "descriptor_driven_preview_pending_agent_visual_check",
        "counts_toward_task070_v2_pass": False,
        "reference_registry_sha256": TASK070_REFERENCE_REGISTRY_SHA256,
        "source_license_matrix_sha256": TASK070_SOURCE_LICENSE_MATRIX_SHA256,
        "source_reference_id": descriptor.reference_id,
        "structural_center_id": descriptor.center_id,
        "structural_descriptor_sha256": descriptor_hash,
        "source_sha256": descriptor.source_sha256,
        "source_path": descriptor.source_path,
        "source_tree_descriptor": descriptor.manifest(),
        "source_to_anonymous_frame_transform": dict(
            descriptor.source_to_anonymous_frame_transform
        ),
        "motor_accounting": {
            "source_actuated_motor_count": 12,
            "source_non_wheel_motor_count": 12,
            "anonymous_non_wheel_motor_count": 12,
            "anonymous_motor_count": 12,
            "added_wheel_motor_count": 0,
            "total_actuator_count": 12,
            "module_dof_counts": descriptor.module_dof_counts,
            "bijection_passed": len(joints) == len(actuators) == len(descriptor.motors) == 12,
        },
        "source_to_anonymous_motor_bijection": [
            motor.manifest() for motor in descriptor.motors
        ],
        "motor_configuration": {
            "contract": "task070_v2_source_hint_anonymous_scaling_v1",
            "source_config_coverage": _source_motor_config_coverage(descriptor.motors),
            "resolved_anonymous_actuators": resolved_motor_configs,
            "policy_action": "target_joint_position",
            "exact_named_robot_parameter_parity_claimed": False,
            "source_values_are_prior_hints_not_identity": True,
            "control_gain_randomization_required_for_training_distribution": True,
        },
        "actuation_stack": _actuation_stack_manifest(
            descriptor,
            resolved_motor_configs,
        ),
        "anonymous_body_tree": _anonymous_body_tree_manifest(links, joints),
        "geometry_randomization": {
            "seed": seed,
            "uniform_visual_scale": visual_scale,
            "realized_source_tree_positions": realized_positions,
        },
        "capsule_local_fromto": capsule_local_fromto,
        "body_local_quat": body_local_quat,
        "footpad_local_pos": footpad_local_pos,
        "footpad_visual_rgba": (0.16, 0.18, 0.22, 1.0),
        "auxiliary_capsule_visuals": auxiliary_capsule_visuals,
        "link_visual_rgba": link_visual_rgba,
        "joint_marker_sites": "source_motor_origins",
        "joint_marker_rgba": (1.0, 0.88, 0.10, 1.0),
        "visual_audit_nominal_joint_pose": {
            motor.anonymous_semantic_slot: _quadruped_preview_joint_nominal(motor)
            for motor in descriptor.motors
        },
        "primitive_link_visual_contract": {
            "segment_rule": "current motor origin to next source joint or recorded terminal",
            "body_frame_rule": "preserve parsed child-frame quaternion and local joint axis",
            "attachment_rule": "trunk surface to each first hip origin is explicitly visible",
            "terminal_rule": "knee shank ends at recorded local foot/contact attachment",
        },
        "render_camera_distance": 1.45,
        "primitive_geometry_only": True,
        "mesh_texture_logo_copied": False,
        "stance_claim": "not_run_preview_only",
    }
    profile_metadata["canonical_root_frame"] = _canonical_root_frame(
        "quadruped",
        links,
        joints,
    )
    topology_payload = {
        "profile_version": profile_version,
        "descriptor_hash": descriptor_hash,
        "family": "quadruped",
        "anonymous_links": [(link.name, link.parent, link.geom_type) for link in links],
        "anonymous_joints": [
            (joint.semantic_slot, joint.parent_link, joint.child_link, joint.axis)
            for joint in joints
        ],
    }
    return MorphologyBlueprint(
        family="quadruped",
        seed=seed,
        links=tuple(links),
        joints=tuple(joints),
        actuators=tuple(actuators),
        nominal_height=0.52 * visual_scale,
        has_arms=False,
        structural_hash=_hash_payload(topology_payload)[:16],
        end_sites=(),
        profile_version=profile_version,
        contract_version=contract_version,
        contract_hash=contract_hash,
        profile_metadata=profile_metadata,
    )


def _compose_terminal_wheels(
    blueprint: MorphologyBlueprint,
    *,
    family: Literal["wheeled_biped", "wheeled_quadruped"],
) -> MorphologyBlueprint:
    limb_count = 2 if family == "wheeled_biped" else 4
    terminal_kind = "ankle_roll" if family == "wheeled_biped" else "knee_pitch"
    radius = 0.105 if family == "wheeled_biped" else 0.068
    width = 0.055 if family == "wheeled_biped" else 0.045
    metadata = dict(blueprint.profile_metadata)
    footpad_positions = metadata.get("footpad_local_pos")
    if not isinstance(footpad_positions, Mapping):
        raise TypeError("wheel composition requires descriptor terminal positions")
    declared_terminal_links = metadata.get("load_bearing_terminal_links")
    if (
        isinstance(declared_terminal_links, Sequence)
        and not isinstance(declared_terminal_links, (str, bytes))
        and len(declared_terminal_links) == limb_count
    ):
        ordered_terminal_links = [str(name) for name in declared_terminal_links]
    else:
        ordered_terminal_links = [
            f"anon_limb{index}_{terminal_kind}_link"
            for index in range(limb_count)
        ]
    terminal_links = set(ordered_terminal_links)
    replaced_links = [
        replace(link, contact=False, end_site=False, foot=False, foot_size=None)
        if link.name in terminal_links
        else link
        for link in blueprint.links
    ]
    links = list(replaced_links)
    joints = list(blueprint.joints)
    actuators = list(blueprint.actuators)
    wheels: list[WheelBlueprint] = []
    body_local_quat = dict(metadata.get("body_local_quat", {}))
    link_visual_rgba = dict(metadata.get("link_visual_rgba", {}))
    wheel_records: list[dict[str, object]] = []
    wheel_motor_records: list[dict[str, object]] = []
    for limb_index, parent in enumerate(ordered_terminal_links):
        terminal_pos = footpad_positions.get(parent)
        if not isinstance(terminal_pos, Sequence) or len(terminal_pos) != 3:
            raise ValueError(f"wheel parent {parent!r} has no local terminal position")
        link_name = f"anon_limb{limb_index}_wheel_link"
        joint_name = f"{link_name}_joint"
        semantic_slot = f"limb{limb_index}_wheel"
        local_axis = (0.0, 1.0, 0.0)
        links.append(
            LinkBlueprint(
                name=link_name,
                parent=parent,
                geom_type="cylinder",
                size=(radius, 0.5 * width),
                pos=tuple(float(value) for value in terminal_pos),  # type: ignore[arg-type]
                length=width,
                mass=0.38 if limb_count == 2 else 0.22,
                contact=True,
                end_site=False,
            )
        )
        joints.append(
            JointBlueprint(
                name=joint_name,
                parent_link=parent,
                child_link=link_name,
                semantic_slot=semantic_slot,
                axis_name="wheel",
                axis=local_axis,
                joint_range=(-math.pi, math.pi),
                damping=0.15,
                friction=0.01,
                armature=0.01,
            )
        )
        actuators.append(
            ActuatorBlueprint(
                name=f"{joint_name}_actuator",
                joint_name=joint_name,
                semantic_slot=semantic_slot,
                kp=0.0,
                kd=0.15,
                effort_limit=45.0,
            )
        )
        wheels.append(
            WheelBlueprint(
                link_name=link_name,
                joint_name=joint_name,
                semantic_slot=semantic_slot,
                radius=radius,
                width=width,
                axis_name="source_terminal_local_lateral",
                axis=local_axis,
            )
        )
        body_local_quat[link_name] = (1.0, 0.0, 0.0, 0.0)
        link_visual_rgba[link_name] = (0.10, 0.12, 0.15, 1.0)
        wheel_records.append(
            {
                "semantic_slot": semantic_slot,
                "parent_non_wheel_link": parent,
                "local_position": tuple(float(value) for value in terminal_pos),
                "local_axis": local_axis,
                "axis_derivation": (
                    "terminal child frame local lateral basis; inherited source frame, not world axis"
                ),
                "motor_config_provenance": "local_engineering_module",
            }
        )
        wheel_motor_records.append(
            {
                "source_joint_name": None,
                "anonymous_semantic_slot": semantic_slot,
                "source_motor_class": None,
                "source_hint_used": False,
                "source_config_usable_as_quantitative_prior": False,
                "raw_or_proxy": {
                    "kp": 0.0,
                    "kd": 0.15,
                    "effort_limit": 45.0,
                    "armature": 0.01,
                    "velocity_limit_hint": None,
                    "gain_derivation": "local_continuous_torque_wheel_module",
                },
                "anonymous_transform": {
                    "mass_scale": 1.0,
                    "lever_scale": 1.0,
                    "inertia_and_gain_scale": 1.0,
                    "effort_scale": 1.0,
                    "policy": "local engineering module; no named wheel motor source",
                },
                "final_compiled": {
                    "kp": 0.0,
                    "kd": 0.0,
                    "effort_limit": 45.0,
                    "armature": 0.01,
                    "joint_damping": 0.15,
                    "joint_friction": 0.01,
                },
                "velocity_limit_runtime_enforced": False,
                "velocity_limit_note": "No source speed limit; wheel torque is controller-bounded.",
                "fallback_reason": "local engineering module has no mature wheel-motor prior",
            }
        )
    source_count = len(blueprint.joints)
    total_count = len(joints)
    accounting = dict(metadata["motor_accounting"])
    accounting.update(
        {
            "source_actuated_motor_count": source_count,
            "source_non_wheel_motor_count": source_count,
            "anonymous_non_wheel_motor_count": source_count,
            "anonymous_motor_count": total_count,
            "added_wheel_motor_count": limb_count,
            "total_actuator_count": total_count,
            "bijection_passed": (
                len(blueprint.joints) == len(blueprint.actuators) == source_count
                and total_count == source_count + limb_count == len(actuators)
            ),
        }
    )
    motor_configuration = dict(metadata["motor_configuration"])
    motor_configuration.update(
        {
            "resolved_anonymous_actuators": [
                *motor_configuration["resolved_anonymous_actuators"],
                *wheel_motor_records,
            ],
            "local_wheel_motor_config_count": limb_count,
            "wheel_motor_config_provenance": "local_engineering_module",
            "policy_action": "mixed_joint_position_and_continuous_wheel_torque",
        }
    )
    actuation_stack = _compose_wheel_actuation_stack(
        metadata["actuation_stack"],  # type: ignore[arg-type]
        wheel_records,
        wheel_motor_records,
    )
    metadata.update(
        {
            "motor_accounting": accounting,
            "motor_configuration": motor_configuration,
            "actuation_stack": actuation_stack,
            "anonymous_body_tree": _anonymous_body_tree_manifest(links, joints),
            "body_local_quat": body_local_quat,
            "link_visual_rgba": link_visual_rgba,
            "terminal_wheel_composition": wheel_records,
            "visual_audit_nominal_joint_pose": {
                **dict(metadata["visual_audit_nominal_joint_pose"]),
                **{wheel.semantic_slot: 0.0 for wheel in wheels},
            },
            "primitive_link_visual_contract": {
                **dict(metadata["primitive_link_visual_contract"]),
                "wheel_rule": (
                    "append one continuous local-lateral wheel hinge after every complete source limb"
                ),
            },
        }
    )
    topology_payload = {
        "source_structural_hash": blueprint.structural_hash,
        "family": family,
        "wheel_records": wheel_records,
    }
    return MorphologyBlueprint(
        family=family,
        seed=blueprint.seed,
        links=tuple(links),
        joints=tuple(joints),
        actuators=tuple(actuators),
        nominal_height=blueprint.nominal_height,
        has_arms=blueprint.has_arms,
        structural_hash=_hash_payload(topology_payload)[:16],
        end_sites=(),
        profile_version=blueprint.profile_version,
        contract_version=blueprint.contract_version,
        contract_hash=blueprint.contract_hash,
        wheel_specs=tuple(wheels),
        profile_metadata=metadata,
    )


def _anonymous_body_tree_manifest(
    links: Sequence[LinkBlueprint],
    joints: Sequence[JointBlueprint],
) -> dict[str, object]:
    return {
        "links": [
            {"name": link.name, "parent": link.parent, "geom_type": link.geom_type}
            for link in links
        ],
        "joint_parent_child_edges": [
            {
                "semantic_slot": joint.semantic_slot,
                "parent_link": joint.parent_link,
                "child_link": joint.child_link,
                "axis": joint.axis,
            }
            for joint in joints
        ],
    }


def _quadruped_preview_joint_nominal(motor: MotorDofSourceMotor) -> float:
    slot = motor.anonymous_semantic_slot
    if slot.endswith("hip_pitch"):
        desired_local_y_rotation = 0.58
    elif slot.endswith("knee_pitch"):
        desired_local_y_rotation = -1.18
    else:
        return 0.0
    axis_y = motor.normalized_local_axis[1]
    if abs(axis_y) < 0.5:
        raise ValueError(f"quadruped pitch motor has a non-lateral axis: {motor.source_joint_name}")
    nominal = desired_local_y_rotation / axis_y
    lower, upper = motor.joint_range
    margin = 0.04 * (upper - lower)
    return min(upper - margin, max(lower + margin, nominal))


def _quadruped_link_rgba(
    motor: MotorDofSourceMotor,
) -> tuple[float, float, float, float]:
    limb_index = int(motor.anonymous_semantic_slot[4])
    stage = motor.anonymous_semantic_slot.rsplit("_", 2)[-2]
    palette = (
        (0.05, 0.68, 0.16, 1.0),
        (0.02, 0.55, 0.95, 1.0),
        (0.80, 0.65, 0.02, 1.0),
        (0.75, 0.05, 0.80, 1.0),
    )
    base = palette[limb_index]
    if stage == "knee":
        return tuple(min(1.0, value * 1.18) for value in base[:3]) + (1.0,)
    return base


def load_g1_motor_dof_preserving_descriptor(
    source_path: Path | None = None,
) -> G1StructuralDescriptor:
    resolved_source = source_path or TASK070_G1_SOURCE_PATH
    parsed = _parse_g1_mjcf_source_tree(resolved_source)
    joint_table = parsed["joints"]
    body_table = parsed["bodies"]
    root_bodies = parsed["root_bodies"]
    expected = {name for name, _, _ in G1_29DOF_MOTOR_SEMANTICS}
    if set(joint_table) != expected:
        missing = sorted(expected - set(joint_table))
        extra = sorted(set(joint_table) - expected)
        raise ValueError(f"G1 source motor inventory mismatch; missing={missing} extra={extra}")
    if len(root_bodies) != 1:
        raise ValueError(f"G1 source must have exactly one root body, got {root_bodies!r}")
    source_root_body = str(root_bodies[0])
    source_joint_to_slot = {
        source_joint_name: semantic_slot
        for source_joint_name, semantic_slot, _ in G1_29DOF_MOTOR_SEMANTICS
    }
    source_body_to_joint = {
        str(joint_table[source_joint_name]["child_body"]): source_joint_name
        for source_joint_name in source_joint_to_slot
    }
    body_to_anonymous_link = {source_root_body: _g1_anonymous_root_link()}
    for source_body, source_joint_name in source_body_to_joint.items():
        body_to_anonymous_link[source_body] = _g1_anonymous_motor_link(
            source_joint_to_slot[source_joint_name]
        )
    for source_body in body_table:
        body_to_anonymous_link.setdefault(
            source_body,
            f"anon_passive_{len(body_to_anonymous_link):02d}",
        )

    motors: list[MotorDofSourceMotor] = []
    for source_joint_name, semantic_slot, module in G1_29DOF_MOTOR_SEMANTICS:
        item = joint_table[source_joint_name]
        source_parent_body = str(item["parent_body"])
        source_child_body = str(item["child_body"])
        motors.append(
            MotorDofSourceMotor(
                source_joint_name=source_joint_name,
                anonymous_semantic_slot=semantic_slot,
                module=module,
                source_joint_type=str(item["joint_type"]),
                source_parent_body=source_parent_body,
                source_child_body=source_child_body,
                source_body_local_pos=item["body_local_pos"],  # type: ignore[arg-type]
                source_body_local_quat=item["body_local_quat"],  # type: ignore[arg-type]
                source_joint_local_pos=item["joint_local_pos"],  # type: ignore[arg-type]
                normalized_local_axis=item["axis"],  # type: ignore[arg-type]
                joint_range=item["joint_range"],  # type: ignore[arg-type]
                anonymous_parent_link=body_to_anonymous_link[source_parent_body],
                anonymous_child_link=body_to_anonymous_link[source_child_body],
                source_tree_depth=int(item["source_tree_depth"]),
                source_motor_config=_g1_source_motor_config(source_joint_name),
            )
        )

    selected_motor_by_body = {
        motor.source_child_body: motor.source_joint_name for motor in motors
    }
    body_nodes = tuple(
        G1SourceBodyNode(
            source_body_name=source_body_name,
            source_parent_body=str(item["parent_body"]),
            source_body_local_pos=item["body_local_pos"],  # type: ignore[arg-type]
            source_body_local_quat=item["body_local_quat"],  # type: ignore[arg-type]
            source_tree_depth=int(item["source_tree_depth"]),
            selected_motor_joint=selected_motor_by_body.get(source_body_name),
            anonymous_link=body_to_anonymous_link[source_body_name],
            child_bodies=tuple(str(child) for child in item["child_bodies"]),  # type: ignore[index]
        )
        for source_body_name, item in body_table.items()
    )
    return G1StructuralDescriptor(
        center_id="unitree_g1_29dof_motor_preserving_descriptor_v2",
        reference_id="unitree_g1",
        source_path=_repo_relative(resolved_source),
        source_sha256=_sha256_path(resolved_source),
        source_root_body=source_root_body,
        source_to_anonymous_frame_transform={
            "policy": (
                "source MJCF child-body local pos and quaternion are preserved as "
                "descriptor coordinates; primitive witness applies recorded module "
                "scale and surface-clearance adjustment without changing motor order"
            ),
            "root_recenter": f"{source_root_body}->anon_pelvis_core",
            "joint_axis_frame": (
                "source child-body local joint axis, normalized; source body-local "
                "quaternion is emitted on the matching anonymous body"
            ),
            "body_identity": "source body names kept only in descriptor; MJCF link names are anonymous",
        },
        body_tree=body_nodes,
        motors=tuple(motors),
    )


def load_pm01_motor_dof_preserving_descriptor(
    source_path: Path | None = None,
) -> G1StructuralDescriptor:
    """Parse the complete local PM01 motor tree, including elbow axial motors."""

    resolved_source = source_path or TASK070_PM01_SOURCE_PATH
    return _load_urdf_motor_dof_preserving_descriptor(
        reference_id="engineai_pm01",
        center_id="engineai_pm01_23dof_motor_preserving_descriptor_v2",
        source_path=resolved_source,
        semantics=PM01_23DOF_MOTOR_SEMANTICS,
        anonymous_root_link="anon_pelvis_core",
    )


def load_quadruped_motor_dof_preserving_descriptor(
    reference_id: str,
    source_path: Path | None = None,
) -> G1StructuralDescriptor:
    """Parse one authorized 12-motor quadruped source into anonymous slots."""

    try:
        semantics = QUADRUPED_12DOF_MOTOR_SEMANTICS[reference_id]
        resolved_source = source_path or TASK070_QUADRUPED_SOURCE_PATHS[reference_id]
    except KeyError as exc:
        raise ValueError(f"unsupported Task070 quadruped reference: {reference_id}") from exc
    if reference_id == "unitree_go2":
        parsed = _parse_g1_mjcf_source_tree(resolved_source)
        descriptor = _descriptor_from_parsed_source_tree(
            reference_id=reference_id,
            center_id=f"{reference_id}_12dof_motor_preserving_descriptor_v2",
            source_path=resolved_source,
            parsed=parsed,
            semantics=semantics,
            source_format="MJCF",
        )
    else:
        descriptor = _load_urdf_motor_dof_preserving_descriptor(
            reference_id=reference_id,
            center_id=f"{reference_id}_12dof_motor_preserving_descriptor_v2",
            source_path=resolved_source,
            semantics=semantics,
        )
    terminal_records = _quadruped_terminal_offset_records(descriptor, resolved_source)
    frame_transform = dict(descriptor.source_to_anonymous_frame_transform)
    frame_transform["terminal_local_offsets"] = terminal_records
    return G1StructuralDescriptor(
        center_id=descriptor.center_id,
        reference_id=descriptor.reference_id,
        source_path=descriptor.source_path,
        source_sha256=descriptor.source_sha256,
        source_root_body=descriptor.source_root_body,
        source_to_anonymous_frame_transform=frame_transform,
        body_tree=descriptor.body_tree,
        motors=descriptor.motors,
    )


def _additional_humanoid_semantic(
    source_joint_name: str,
) -> tuple[str, str]:
    """Map audited humanoid joint names to anonymous, source-faithful semantics."""

    normalized = source_joint_name.strip().lower().replace("-", "_")
    normalized = re.sub(r"^j\d+_", "", normalized)
    normalized = normalized.removesuffix("_joint")
    normalized = re.sub(r"_+", "_", normalized)
    if normalized.startswith("aahead_"):
        normalized = normalized[2:]

    side: str | None = None
    core = normalized
    for prefix, resolved in (("left_", "left"), ("right_", "right")):
        if core.startswith(prefix):
            side = resolved
            core = core[len(prefix) :]
            break
    if side is None:
        for suffix, resolved in (
            ("_left", "left"),
            ("_right", "right"),
            ("_l", "left"),
            ("_r", "right"),
        ):
            if core.endswith(suffix):
                side = resolved
                core = core[: -len(suffix)]
                break

    if "head" in core or "neck" in core:
        if "yaw" in core:
            return "head_yaw", "head"
        if "pitch" in core:
            return "head_pitch", "head"
        raise ValueError(f"head joint has no pitch/yaw semantic: {source_joint_name!r}")

    if any(token in core for token in ("hip", "knee", "ankle")):
        if side is None:
            raise ValueError(f"leg joint has no side semantic: {source_joint_name!r}")
        limb = 0 if side == "left" else 1
        if "hip" in core:
            axis = next(
                (item for item in ("pitch", "roll", "yaw") if item in core),
                None,
            )
            if axis is None:
                raise ValueError(f"hip joint has no axis semantic: {source_joint_name!r}")
            kind = f"hip_{axis}"
        elif "knee" in core:
            kind = "knee_pitch"
        else:
            axis = next((item for item in ("pitch", "roll") if item in core), None)
            if axis is None:
                raise ValueError(f"ankle joint has no axis semantic: {source_joint_name!r}")
            kind = f"ankle_{axis}"
        return f"limb{limb}_{kind}", f"{side}_leg"

    if any(token in core for token in ("waist", "lumbar", "torso")):
        axis = next((item for item in ("yaw", "roll", "pitch") if item in core), "yaw")
        return f"waist_{axis}", "waist"

    hand_tokens = ("hand", "thumb", "index", "middle", "ring", "pinky")
    if any(token in core for token in hand_tokens) or core.startswith("mid_"):
        if side is None:
            raise ValueError(f"hand joint has no side semantic: {source_joint_name!r}")
        hand_core = core.removeprefix("hand_")
        if hand_core.startswith("mid_"):
            hand_core = f"middle_{hand_core[4:]}"
        hand_core = hand_core.replace("rota", "rotation")
        hand_core = re.sub(r"joint([12])$", r"segment\1", hand_core)
        return f"{side}_hand_{hand_core}", f"{side}_hand"

    if any(token in core for token in ("shoulder", "elbow", "wrist", "arm_yaw")):
        if side is None:
            raise ValueError(f"arm joint has no side semantic: {source_joint_name!r}")
        if "shoulder" in core:
            axis = next(
                (item for item in ("pitch", "roll", "yaw") if item in core),
                None,
            )
            if axis is None:
                raise ValueError(
                    f"shoulder joint has no axis semantic: {source_joint_name!r}"
                )
            kind = f"shoulder_{axis}"
        elif "arm_yaw" in core:
            kind = "upper_yaw"
        elif "elbow" in core:
            kind = "elbow_yaw" if "yaw" in core else "elbow_pitch"
        else:
            axis = next(
                (item for item in ("yaw", "roll", "pitch") if item in core),
                None,
            )
            if axis is None:
                raise ValueError(f"wrist joint has no axis semantic: {source_joint_name!r}")
            kind = f"wrist_{axis}"
        return f"{side}_arm_{kind}", f"{side}_arm"

    raise ValueError(f"unsupported additional humanoid motor semantic: {source_joint_name!r}")


def _additional_humanoid_candidate_evidence(reference_id: str) -> dict[str, object]:
    spec = TASK070_ADDITIONAL_HUMANOID_SOURCES[reference_id]
    source_path = spec["source_path"]
    evidence_paths = spec["evidence_paths"]
    if not isinstance(source_path, Path) or not isinstance(evidence_paths, Sequence):
        raise TypeError("additional humanoid source specification is malformed")
    resolved_evidence: list[dict[str, object]] = []
    for path in evidence_paths:
        if not isinstance(path, Path):
            raise TypeError("candidate evidence path must be a Path")
        resolved_evidence.append(
            {
                "path": _repo_relative(path),
                "sha256": _sha256_path(path),
            }
        )
    expected = int(spec["expected_motor_count"])
    configured = int(spec["configured_physical_motor_count"])
    return {
        "status": "candidate_fail_closed",
        "source_model_actuated_joint_count": expected,
        "configured_physical_motor_count": configured,
        "source_model_config_motor_count_aligned": expected == configured,
        "source_model_config_motor_count_gap": configured - expected,
        "source_description": {
            "path": _repo_relative(source_path),
            "sha256": _sha256_path(source_path),
            "format": spec["source_format"],
        },
        "transmission_and_motor_config_evidence": resolved_evidence,
        "missing_for_promotion": list(spec["missing_for_promotion"]),
        "quantitative_training_prior_promoted": False,
    }


def load_additional_humanoid_motor_dof_preserving_descriptor(
    reference_id: str,
    source_path: Path | None = None,
) -> G1StructuralDescriptor:
    """Parse one authorized additional humanoid as a fail-closed candidate."""

    try:
        spec = TASK070_ADDITIONAL_HUMANOID_SOURCES[reference_id]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Task070 additional humanoid reference: {reference_id}"
        ) from exc
    configured_path = spec["source_path"]
    if not isinstance(configured_path, Path):
        raise TypeError("additional humanoid source path must be a Path")
    resolved_source = source_path or configured_path
    source_format = str(spec["source_format"])
    parsed = (
        _parse_g1_mjcf_source_tree(resolved_source)
        if source_format == "MJCF"
        else _parse_urdf_source_tree(resolved_source)
    )
    joint_table = parsed["joints"]
    if not isinstance(joint_table, Mapping):
        raise TypeError("parsed candidate joint table must be a mapping")
    actuated_names = tuple(
        str(name)
        for name, item in joint_table.items()
        if isinstance(item, Mapping)
        and str(item["joint_type"]) not in {"fixed", "free", "floating"}
    )
    expected_count = int(spec["expected_motor_count"])
    if len(actuated_names) != expected_count:
        raise ValueError(
            f"{reference_id} source motor count mismatch; "
            f"expected={expected_count} parsed={len(actuated_names)}"
        )
    semantics = tuple(
        (source_name, *_additional_humanoid_semantic(source_name))
        for source_name in actuated_names
    )
    semantic_slots = [slot for _, slot, _ in semantics]
    if len(semantic_slots) != len(set(semantic_slots)):
        duplicates = sorted(
            slot for slot in set(semantic_slots) if semantic_slots.count(slot) > 1
        )
        raise ValueError(f"candidate semantic slots are not unique: {duplicates}")
    descriptor = _descriptor_from_parsed_source_tree(
        reference_id=reference_id,
        center_id=f"{reference_id}_{expected_count}dof_candidate_descriptor_v2",
        source_path=resolved_source,
        parsed=parsed,
        semantics=semantics,
        source_format=source_format,
        anonymous_root_link="anon_humanoid_core",
    )
    frame_transform = dict(descriptor.source_to_anonymous_frame_transform)
    frame_transform.update(
        {
            "candidate_prior_status": "candidate_fail_closed",
            "source_model_config_accounting": {
                "source_model_actuated_joint_count": expected_count,
                "configured_physical_motor_count": int(
                    spec["configured_physical_motor_count"]
                ),
            },
        }
    )
    return G1StructuralDescriptor(
        center_id=descriptor.center_id,
        reference_id=descriptor.reference_id,
        source_path=descriptor.source_path,
        source_sha256=descriptor.source_sha256,
        source_root_body=descriptor.source_root_body,
        source_to_anonymous_frame_transform=frame_transform,
        body_tree=descriptor.body_tree,
        motors=descriptor.motors,
    )


def _load_urdf_motor_dof_preserving_descriptor(
    *,
    reference_id: str,
    center_id: str,
    source_path: Path,
    semantics: Sequence[tuple[str, str, str]],
    anonymous_root_link: str = "anon_trunk_core",
) -> G1StructuralDescriptor:
    return _descriptor_from_parsed_source_tree(
        reference_id=reference_id,
        center_id=center_id,
        source_path=source_path,
        parsed=_parse_urdf_source_tree(source_path),
        semantics=semantics,
        source_format="URDF",
        anonymous_root_link=anonymous_root_link,
    )


def _descriptor_from_parsed_source_tree(
    *,
    reference_id: str,
    center_id: str,
    source_path: Path,
    parsed: Mapping[str, object],
    semantics: Sequence[tuple[str, str, str]],
    source_format: str,
    anonymous_root_link: str = "anon_trunk_core",
) -> G1StructuralDescriptor:
    joint_table = parsed["joints"]
    body_table = parsed["bodies"]
    root_bodies = parsed["root_bodies"]
    if not isinstance(joint_table, Mapping) or not isinstance(body_table, Mapping):
        raise TypeError("parsed source tree tables must be mappings")
    if not isinstance(root_bodies, Sequence) or len(root_bodies) != 1:
        raise ValueError(f"source must have exactly one root body, got {root_bodies!r}")
    source_root_body = str(root_bodies[0])
    expected_order = tuple(name for name, _, _ in semantics)
    missing = [name for name in expected_order if name not in joint_table]
    if missing:
        raise ValueError(f"source motor inventory is missing {missing}")
    parsed_actuated_order = tuple(
        str(name)
        for name, item in joint_table.items()
        if isinstance(item, Mapping) and str(item["joint_type"]) not in {"fixed", "free"}
    )
    if parsed_actuated_order != expected_order:
        raise ValueError(
            "source actuated joint order mismatch; "
            f"expected={expected_order!r} parsed={parsed_actuated_order!r}"
        )

    joint_to_slot = {name: slot for name, slot, _ in semantics}
    body_to_joint = {
        str(joint_table[name]["child_body"]): name  # type: ignore[index]
        for name in expected_order
    }
    body_to_anonymous_link = {source_root_body: anonymous_root_link}
    for source_body, source_joint_name in body_to_joint.items():
        body_to_anonymous_link[source_body] = f"anon_{joint_to_slot[source_joint_name]}_link"
    for source_body in body_table:
        body_to_anonymous_link.setdefault(
            str(source_body),
            f"anon_passive_{len(body_to_anonymous_link):02d}",
        )

    motors: list[MotorDofSourceMotor] = []
    for source_joint_name, semantic_slot, module in semantics:
        item = joint_table[source_joint_name]
        if not isinstance(item, Mapping):
            raise TypeError("parsed source joint record must be a mapping")
        parent = str(item["parent_body"])
        child = str(item["child_body"])
        source_motor_config = (
            _go2_source_motor_config(source_joint_name)
            if reference_id == "unitree_go2"
            else _urdf_source_motor_config(
                reference_id=reference_id,
                source_joint_name=source_joint_name,
                item=item,
                source_path=source_path,
            )
        )
        motors.append(
            MotorDofSourceMotor(
                source_joint_name=source_joint_name,
                anonymous_semantic_slot=semantic_slot,
                module=module,
                source_joint_type=str(item["joint_type"]),
                source_parent_body=parent,
                source_child_body=child,
                source_body_local_pos=item["body_local_pos"],  # type: ignore[arg-type]
                source_body_local_quat=item["body_local_quat"],  # type: ignore[arg-type]
                source_joint_local_pos=item["joint_local_pos"],  # type: ignore[arg-type]
                normalized_local_axis=item["axis"],  # type: ignore[arg-type]
                joint_range=item["joint_range"],  # type: ignore[arg-type]
                anonymous_parent_link=body_to_anonymous_link[parent],
                anonymous_child_link=body_to_anonymous_link[child],
                source_tree_depth=int(item["source_tree_depth"]),
                source_motor_config=source_motor_config,
            )
        )
    selected_motor_by_body = {
        motor.source_child_body: motor.source_joint_name for motor in motors
    }
    body_nodes = tuple(
        G1SourceBodyNode(
            source_body_name=str(source_body_name),
            source_parent_body=str(item["parent_body"]),
            source_body_local_pos=item["body_local_pos"],  # type: ignore[arg-type]
            source_body_local_quat=item["body_local_quat"],  # type: ignore[arg-type]
            source_tree_depth=int(item["source_tree_depth"]),
            selected_motor_joint=selected_motor_by_body.get(str(source_body_name)),
            anonymous_link=body_to_anonymous_link[str(source_body_name)],
            child_bodies=tuple(str(child) for child in item["child_bodies"]),  # type: ignore[index]
        )
        for source_body_name, item in body_table.items()
        if isinstance(item, Mapping)
    )
    return G1StructuralDescriptor(
        center_id=center_id,
        reference_id=reference_id,
        source_path=_repo_relative(source_path),
        source_sha256=_sha256_path(source_path),
        source_root_body=source_root_body,
        source_to_anonymous_frame_transform={
            "policy": (
                f"preserve parsed {source_format} child-frame position, quaternion, "
                "joint order, local axis, and range; apply only a recorded uniform visual scale"
            ),
            "root_recenter": f"{source_root_body}->{anonymous_root_link}",
            "joint_axis_frame": "normalized source child-link joint axis",
            "body_identity": "source names remain descriptor-only; emitted links are anonymous",
        },
        body_tree=body_nodes,
        motors=tuple(motors),
    )


def _parse_urdf_source_tree(source_path: Path) -> dict[str, object]:
    root = ET.parse(source_path).getroot()
    link_names = tuple(
        str(link.get("name")) for link in root.findall("link") if link.get("name")
    )
    if not link_names:
        raise ValueError(f"URDF source has no links: {source_path}")
    joint_elements = tuple(root.findall("joint"))
    parent_by_child: dict[str, str] = {}
    children_by_parent: dict[str, list[str]] = {name: [] for name in link_names}
    joint_rows: list[tuple[str, ET.Element, str, str]] = []
    for joint in joint_elements:
        name = joint.get("name")
        parent_node = joint.find("parent")
        child_node = joint.find("child")
        if not name or parent_node is None or child_node is None:
            raise ValueError("URDF joint is missing name, parent, or child")
        parent = parent_node.get("link")
        child = child_node.get("link")
        if not parent or not child:
            raise ValueError(f"URDF joint {name!r} has an empty parent or child")
        if child in parent_by_child:
            raise ValueError(f"URDF link {child!r} has multiple parents")
        parent_by_child[child] = parent
        children_by_parent.setdefault(parent, []).append(child)
        joint_rows.append((name, joint, parent, child))
    roots = tuple(name for name in link_names if name not in parent_by_child)
    if len(roots) != 1:
        raise ValueError(f"URDF must have exactly one root link, got {roots!r}")
    depth_by_body: dict[str, int] = {}

    def assign_depth(body: str, depth: int) -> None:
        if body in depth_by_body:
            raise ValueError(f"URDF source tree contains a cycle at {body!r}")
        depth_by_body[body] = depth
        for child in children_by_parent.get(body, ()):
            assign_depth(child, depth + 1)

    assign_depth(roots[0], 0)
    if set(depth_by_body) != set(link_names):
        raise ValueError("URDF source tree contains disconnected links")
    body_local: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {
        roots[0]: ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    }
    joints: dict[str, dict[str, object]] = {}
    for name, joint, parent, child in joint_rows:
        origin = joint.find("origin")
        xyz = _parse_float_tuple(
            origin.get("xyz", "0 0 0") if origin is not None else "0 0 0",
            expected=3,
        )
        rpy = _parse_float_tuple(
            origin.get("rpy", "0 0 0") if origin is not None else "0 0 0",
            expected=3,
        )
        quat = _rpy_to_quat(rpy)
        body_local[child] = (xyz, quat)  # type: ignore[assignment]
        joint_type = str(joint.get("type", "fixed"))
        axis_node = joint.find("axis")
        raw_axis = _parse_float_tuple(
            axis_node.get("xyz", "0 0 1") if axis_node is not None else "0 0 1",
            expected=3,
        )
        axis = (
            (0.0, 0.0, 1.0)
            if joint_type in {"fixed", "floating"}
            and _g1_vector_norm(raw_axis) <= 1e-12
            else _normalize_axis(raw_axis)
        )
        limit = joint.find("limit")
        declared_effort_limit = (
            float(limit.get("effort"))
            if limit is not None and limit.get("effort") is not None
            else None
        )
        declared_velocity_limit = (
            float(limit.get("velocity"))
            if limit is not None and limit.get("velocity") is not None
            else None
        )
        joint_range = (
            (-math.pi, math.pi)
            if joint_type in {"continuous", "fixed"} or limit is None
            else (float(limit.get("lower", "-3.14159265")), float(limit.get("upper", "3.14159265")))
        )
        joints[name] = {
            "parent_body": parent,
            "child_body": child,
            "body_local_pos": xyz,
            "body_local_quat": quat,
            "joint_local_pos": (0.0, 0.0, 0.0),
            "joint_type": joint_type,
            "axis": axis,
            "joint_range": joint_range,
            "declared_effort_limit": declared_effort_limit,
            "declared_velocity_limit": declared_velocity_limit,
            "source_tree_depth": depth_by_body[child],
        }
    bodies = {
        name: {
            "parent_body": parent_by_child.get(name, "root"),
            "body_local_pos": body_local[name][0],
            "body_local_quat": body_local[name][1],
            "source_tree_depth": depth_by_body[name],
            "child_bodies": tuple(children_by_parent.get(name, ())),
        }
        for name in link_names
    }
    return {"bodies": bodies, "joints": joints, "root_bodies": roots}


def _rpy_to_quat(rpy: Sequence[float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return _normalize_quat(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _quadruped_terminal_offset_records(
    descriptor: G1StructuralDescriptor,
    source_path: Path,
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    bodies = {node.source_body_name: node for node in descriptor.body_tree}
    if descriptor.reference_id == "unitree_go2":
        source_root = ET.parse(source_path).getroot()
        body_elements = {
            str(body.get("name")): body for body in source_root.iter("body") if body.get("name")
        }
        for motor in descriptor.motors:
            if not motor.anonymous_semantic_slot.endswith("knee_pitch"):
                continue
            body = body_elements[motor.source_child_body]
            site = next((item for item in body.findall("site") if item.get("pos")), None)
            if site is None:
                raise ValueError(f"Go2 terminal body {motor.source_child_body!r} has no site offset")
            records[motor.anonymous_semantic_slot] = {
                "offset": _parse_float_tuple(site.get("pos", ""), expected=3),
                "source_element": f"body[{motor.source_child_body}]/site[{site.get('name')}]",
                "policy": "exact_source_terminal_site_local_offset",
            }
    elif descriptor.reference_id == "deeprobotics_lite3":
        for motor in descriptor.motors:
            if not motor.anonymous_semantic_slot.endswith("knee_pitch"):
                continue
            passive_children = [
                bodies[name]
                for name in bodies[motor.source_child_body].child_bodies
                if bodies[name].selected_motor_joint is None
            ]
            if len(passive_children) != 1:
                raise ValueError(f"Lite3 terminal body {motor.source_child_body!r} is ambiguous")
            child = passive_children[0]
            records[motor.anonymous_semantic_slot] = {
                "offset": child.source_body_local_pos,
                "source_element": f"fixed_child[{child.source_body_name}]",
                "policy": "exact_source_fixed_terminal_child_local_offset",
            }
    else:
        for motor in descriptor.motors:
            if not motor.anonymous_semantic_slot.endswith("knee_pitch"):
                continue
            records[motor.anonymous_semantic_slot] = {
                "offset": motor.source_body_local_pos,
                "source_element": f"joint[{motor.source_joint_name}]/origin",
                "policy": "same_source_upper_span_fallback_source_omits_terminal_frame",
            }
    return records


def _load_g1_source_motors(source_path: Path) -> tuple[MotorDofSourceMotor, ...]:
    return load_g1_motor_dof_preserving_descriptor(source_path).motors


def _parse_mjcf_joint_table(source_path: Path) -> dict[str, dict[str, object]]:
    return _parse_g1_mjcf_source_tree(source_path)["joints"]  # type: ignore[return-value]


def _parse_g1_mjcf_source_tree(source_path: Path) -> dict[str, object]:
    root = ET.parse(source_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF source has no worldbody: {source_path}")
    bodies: dict[str, dict[str, object]] = {}
    joints: dict[str, dict[str, object]] = {}
    root_bodies: list[str] = []

    def walk(body: ET.Element, parent_body: str, depth: int) -> None:
        body_name = body.get("name")
        if not body_name:
            raise ValueError("MJCF body is missing a name")
        if body_name in bodies:
            raise ValueError(f"duplicate MJCF body name: {body_name}")
        child_names = tuple(
            str(child.get("name")) for child in body.findall("body") if child.get("name")
        )
        body_local_pos = _parse_float_tuple(body.get("pos", "0 0 0"), expected=3)
        body_local_quat = _normalize_quat(
            _parse_float_tuple(body.get("quat", "1 0 0 0"), expected=4)
        )
        bodies[body_name] = {
            "parent_body": parent_body,
            "body_local_pos": body_local_pos,
            "body_local_quat": body_local_quat,
            "source_tree_depth": depth,
            "child_bodies": child_names,
        }
        for joint in body.findall("joint"):
            name = joint.get("name")
            if not name or name in {"floating_base_joint", "root_free"}:
                continue
            if name in joints:
                raise ValueError(f"duplicate MJCF joint name: {name}")
            axis = _normalize_axis(_parse_float_tuple(joint.get("axis", "0 0 1"), expected=3))
            joint_range = _parse_float_tuple(joint.get("range", "-3.14159265 3.14159265"), expected=2)
            joints[name] = {
                "parent_body": parent_body,
                "child_body": body_name,
                "body_local_pos": body_local_pos,
                "body_local_quat": body_local_quat,
                "joint_local_pos": _parse_float_tuple(joint.get("pos", "0 0 0"), expected=3),
                "joint_type": joint.get("type", "hinge"),
                "axis": axis,
                "joint_range": joint_range,
                "source_tree_depth": depth,
            }
        for child in body.findall("body"):
            walk(child, body_name, depth + 1)

    for child in worldbody.findall("body"):
        if child.get("name"):
            root_bodies.append(str(child.get("name")))
        walk(child, "root", 0)
    return {"bodies": bodies, "joints": joints, "root_bodies": tuple(root_bodies)}


def _parse_float_tuple(value: str, *, expected: int) -> tuple[float, ...]:
    parts = tuple(float(part) for part in value.split())
    if len(parts) != expected:
        raise ValueError(f"expected {expected} floats, got {value!r}")
    return parts


def _normalize_axis(axis: Sequence[float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in axis))
    if norm <= 1e-12:
        raise ValueError("joint axis must be non-zero")
    return tuple(float(value) / norm for value in axis)  # type: ignore[return-value]


def _normalize_quat(quat: Sequence[float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in quat))
    if norm <= 1e-12:
        raise ValueError("body quaternion must be non-zero")
    return tuple(float(value) / norm for value in quat)  # type: ignore[return-value]


def _g1_anonymous_root_link() -> str:
    return "anon_pelvis_core"


def _g1_anonymous_motor_link(semantic_slot: str) -> str:
    return f"anon_{semantic_slot}_link"


def _g1_link_radius_for_module(module: str) -> float:
    if module.endswith("_leg"):
        return 0.042
    if module.endswith("_arm"):
        return 0.032
    return 0.046


def _g1_link_radius(motor: MotorDofSourceMotor) -> float:
    slot = motor.anonymous_semantic_slot
    if "_wrist_" in slot:
        return 0.024
    if "_ankle_" in slot:
        return 0.033
    return _g1_link_radius_for_module(motor.module)


def _g1_link_mass(motor: MotorDofSourceMotor) -> float:
    if motor.module.endswith("_leg"):
        return 1.65
    if motor.module.endswith("_arm"):
        return 0.24 if "_wrist_" in motor.anonymous_semantic_slot else 0.58
    return 0.70


def _g1_preview_joint_nominal(motor: MotorDofSourceMotor) -> float:
    slot = motor.anonymous_semantic_slot
    requested = 0.0
    if slot.endswith("hip_pitch"):
        requested = -0.28
    elif slot.endswith("knee_pitch"):
        requested = 0.68
    elif slot.endswith("ankle_pitch"):
        requested = -0.24
    elif slot.endswith("elbow_pitch"):
        requested = 0.75
    lower, upper = motor.joint_range
    margin = 0.02 * (upper - lower)
    return min(upper - margin, max(lower + margin, requested))


def _g1_link_rgba(motor: MotorDofSourceMotor) -> tuple[float, float, float, float]:
    slot = motor.anonymous_semantic_slot
    if slot == "waist_pitch":
        return (0.05, 0.15, 0.86, 1.0)
    if motor.module == "waist":
        return (0.95, 0.82, 0.08, 1.0)
    if motor.module == "left_leg":
        return (0.05, 0.68, 0.16, 1.0) if "hip" in slot else (0.02, 0.55, 0.95, 1.0)
    if motor.module == "right_leg":
        return (0.75, 0.05, 0.80, 1.0) if "hip" in slot else (0.80, 0.65, 0.02, 1.0)
    if motor.module == "left_arm":
        return (0.08, 0.85, 0.85, 1.0) if "wrist" not in slot else (0.12, 0.12, 0.90, 1.0)
    return (0.95, 0.18, 0.10, 1.0) if "wrist" not in slot else (0.18, 0.78, 0.18, 1.0)


def _g1_vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _g1_capsule_local_fromto(
    motor: MotorDofSourceMotor,
    descriptor: G1StructuralDescriptor,
    module_scales: Mapping[str, float],
    *,
    length: float,
) -> tuple[float, float, float, float, float, float]:
    children = [
        child
        for child in descriptor.motors
        if child.source_parent_body == motor.source_child_body
    ]
    if children:
        child = children[0]
        direction = tuple(
            float(value) * float(module_scales[child.module])
            for value in child.source_body_local_pos
        )
        if _g1_vector_norm(direction) > 1e-12:
            return (0.0, 0.0, 0.0, *direction)
    elif motor.module.endswith("_arm"):
        return (0.0, 0.0, 0.0, 0.075, 0.0, 0.0)
    elif motor.module.endswith("_leg"):
        return (0.0, 0.0, 0.0, 0.075, 0.0, -0.04)

    half_axis = tuple(0.5 * length * value for value in motor.normalized_local_axis)
    return (*(-value for value in half_axis), *half_axis)


def _g1_link_visual_length(
    motor: MotorDofSourceMotor,
    descriptor: G1StructuralDescriptor,
    module_scales: Mapping[str, float],
) -> float:
    child_spans = [
        _g1_vector_norm(child.source_body_local_pos) * float(module_scales[child.module])
        for child in descriptor.motors
        if child.source_parent_body == motor.source_child_body
    ]
    outgoing_span = max(child_spans, default=0.0)
    if outgoing_span > 1e-12:
        return outgoing_span
    if motor.module.endswith("_leg"):
        return _g1_vector_norm((0.075, 0.0, -0.04))
    if motor.module.endswith("_arm"):
        return 0.075
    return 0.055


def _g1_anonymous_body_pos(
    motor: MotorDofSourceMotor,
    *,
    scale: float,
    pelvis_half_y: float,
    torso_half_y: float,
    leg_radius: float,
    arm_radius: float,
) -> tuple[tuple[float, float, float], dict[str, object] | None]:
    x, y, z = (float(value) * scale for value in motor.source_body_local_pos)
    adjustment: dict[str, object] | None = None
    if motor.anonymous_semantic_slot.endswith("hip_pitch"):
        minimum_abs_y = pelvis_half_y + leg_radius + ATTACHMENT_CLEARANCE
        if abs(y) < minimum_abs_y:
            source_y = y
            y = math.copysign(minimum_abs_y, y if abs(y) > 1e-12 else 1.0)
            adjustment = {
                "axis": "y",
                "source_value": source_y,
                "minimum_abs_value": minimum_abs_y,
                "reason": "primitive pelvis surface clearance",
            }
    elif motor.anonymous_semantic_slot.endswith("shoulder_pitch"):
        minimum_abs_y = torso_half_y + arm_radius + ATTACHMENT_CLEARANCE
        if abs(y) < minimum_abs_y:
            source_y = y
            y = math.copysign(minimum_abs_y, y if abs(y) > 1e-12 else 1.0)
            adjustment = {
                "axis": "y",
                "source_value": source_y,
                "minimum_abs_value": minimum_abs_y,
                "reason": "primitive torso surface clearance",
            }
    return (x, y, z), adjustment


def _g1_foot_size(scale: float) -> tuple[float, float]:
    return (0.105 * scale, 0.052 * scale)


def _g1_nominal_height(links: Sequence[LinkBlueprint]) -> float:
    by_parent: dict[str, list[LinkBlueprint]] = {}
    for link in links:
        by_parent.setdefault(link.parent, []).append(link)
    world_positions: dict[str, tuple[float, float, float]] = {}

    def walk(parent_name: str, parent_pos: tuple[float, float, float]) -> None:
        for link in by_parent.get(parent_name, ()):
            pos = tuple(parent_pos[index] + link.pos[index] for index in range(3))
            world_positions[link.name] = pos  # type: ignore[assignment]
            walk(link.name, pos)  # type: ignore[arg-type]

    walk("root", (0.0, 0.0, 0.0))
    terminal_lowers = []
    for link in links:
        if not link.foot:
            continue
        z = world_positions[link.name][2]
        terminal_lowers.append(z - link.length - 2.0 * link.size[0])
    return max(0.45, -min(terminal_lowers, default=-0.79) + 0.015)


def _axis_name_from_semantic_slot(slot: str) -> Literal["yaw", "roll", "pitch", "wheel"]:
    if slot.endswith("_wheel"):
        return "wheel"
    if slot.endswith("_yaw"):
        return "yaw"
    if slot.endswith("_roll"):
        return "roll"
    return "pitch"


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


Task070MorphologyGenerator = ArchetypeConstrainedMorphologyGenerator
