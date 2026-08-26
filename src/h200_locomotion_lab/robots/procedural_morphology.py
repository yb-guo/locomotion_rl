"""Deterministic primitive-link morphology generation for whole-body training.

The generator is intentionally small: it produces a bounded biped/quadruped
grammar using primitive MuJoCo geoms.  Discrete structure is represented by a
blueprint and compiled once per topology shard; continuous physical parameters
are sampled separately and can be applied at context/reset boundaries.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

from h200_locomotion_lab.robots.whole_body_slots import (
    WHOLE_BODY_SLOT_NAMES,
)

MorphologyFamily = Literal["biped", "quadruped", "wheeled_biped", "wheeled_quadruped"]
LegacyMorphologyFamily = Literal["biped", "quadruped"]

LEGACY_MORPHOLOGY_PROFILE_VERSION = "legacy_v2"
LOCOFORMER_MORPHOLOGY_PROFILE_VERSION = "locoformer_paper_faithful_morphology_v1"
LOCOFORMER_MORPHOLOGY_CONTRACT_VERSION = "procedural_locoformer_paper_faithful_v1"
ARCHETYPE_CONSTRAINED_MORPHOLOGY_PROFILE_VERSION = "archetype_constrained_morphology_v1"
ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_VERSION = "procedural_archetype_constrained_morphology_v1"
MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_PROFILE_VERSION = (
    "motor_dof_preserving_archetype_morphology_v2"
)
MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_VERSION = (
    "procedural_motor_dof_preserving_archetype_morphology_v2_actuation_stack_v1_canonical_root_v1"
)


FOOTPAD_HALF_LENGTH_FRACTION = 0.16
FOOTPAD_HALF_WIDTH_FRACTION = 0.09
ARM_LINK_RADIUS = 0.035
MAX_PHYSICAL_LINK_SCALE = 1.25
LOCOFORMER_ARM_TRUNK_CLEARANCE = 0.03


PROCEDURAL_EMBODIMENT_CONTRACT_VERSION = "procedural_whole_body_v2_footpad_actual_stance_feedforward"
_PROCEDURAL_EMBODIMENT_CONTRACT = {
    "version": PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
    "contact_geometry": "terminal_leg_box_footpad_only",
    "footpad_size_from_leg_length": {
        "half_length": FOOTPAD_HALF_LENGTH_FRACTION,
        "half_width": FOOTPAD_HALF_WIDTH_FRACTION,
    },
    "reset_pose": "exact_physical_actual_equilibrium_qpos",
    "action_midpoint": "exact_physical_actual_equilibrium_actuator_ctrl",
    "instance_identity": "full_blueprint_manifest_plus_full_physical_manifest",
    "stance_solver": "whole_body_static_stance_v3_actual_dynamics_feedforward",
}
PROCEDURAL_EMBODIMENT_CONTRACT_HASH = hashlib.sha256(
    json.dumps(_PROCEDURAL_EMBODIMENT_CONTRACT, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

_LOCOFORMER_MORPHOLOGY_CONTRACT = {
    "version": LOCOFORMER_MORPHOLOGY_CONTRACT_VERSION,
    "profile": LOCOFORMER_MORPHOLOGY_PROFILE_VERSION,
    "families": ("biped", "quadruped", "wheeled_biped", "wheeled_quadruped"),
    "training_morphology": "procedural_primitive_link_envelope_not_named_robot_parameters",
    "wheel_topology": "one_terminal_dynamical_wheel_per_load_bearing_leg",
    "wheel_joint": "continuous_hinge_with_lateral_axle",
    "unified_slots": "whole_body_v1_45_with_per_limb_wheel_slot",
    "legacy_isolation": PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
}
LOCOFORMER_MORPHOLOGY_CONTRACT_HASH = hashlib.sha256(
    json.dumps(_LOCOFORMER_MORPHOLOGY_CONTRACT, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
_ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT = {
    "version": ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_VERSION,
    "profile": ARCHETYPE_CONSTRAINED_MORPHOLOGY_PROFILE_VERSION,
    "families": ("biped", "quadruped", "wheeled_biped", "wheeled_quadruped"),
    "reference_registry_sha256": "931bf5346fe379b3fd1c25e91d39007704e06208a4b2c93b38b96293a58922f1",
    "source_license_matrix_sha256": "1d1c243797cb721942a88445a2ea8c4f9ee30b4237eb4c6e9214850fa1c3465a",
    "r0_design_contract_sha256": "5d9bc169681984d5c9682cec6bbaa2e031e82c9eda4439ee716f95d28ae2cf7d",
    "prior_set_id": "de203b60c4e3f0abd4f3880196efe7e589879b4006996b340319209411d3bf79",
    "distance_contract_hash": "a488cde1c0bd86e1fdcf851b69c691121e114424d3cdbb526660ec65354442a1",
    "topology": "anonymous_generic_humanoid_biped_and_mammal_quadruped_archetypes",
    "sampling": "multi_center_prior_interpolation_bounded_outward_with_clone_guard",
    "stance": "task070_contact_aware_position_feedforward_or_wheel_velocity_hold",
    "geometry": "repository_primitive_geometry_no_vendor_mesh_logo_texture",
    "legacy_isolation": (
        PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
        LOCOFORMER_MORPHOLOGY_CONTRACT_VERSION,
    ),
}
ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_HASH = hashlib.sha256(
    json.dumps(
        _ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()
_MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT = {
    "version": MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_VERSION,
    "profile": MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_PROFILE_VERSION,
    "families": ("biped", "quadruped", "wheeled_biped", "wheeled_quadruped"),
    "task": "task070-archetype-constrained-standable-morphology",
    "source_motor_preservation": (
        "selected source actuated motors must map one-to-one into anonymous primitive joints"
    ),
    "required_descriptor_fields": (
        "source_joint_name",
        "anonymous_semantic_slot",
        "source_parent_body",
        "source_child_body",
        "normalized_local_axis",
        "joint_range",
        "module",
    ),
    "geometry": "repository_primitive_links_without_vendor_mesh_texture_logo_or_model_identity",
    "wheel_composition": "terminal local wheel module adds motors without deleting source motors",
    "actuation_stack": (
        "structural_descriptor_to_transmission_model_to_coherent_motor_config_to_runtime_fault_process"
    ),
    "transmission_coverage": "every generalized motor slot belongs to exactly one transmission group",
    "motor_randomization": "correlated_by_transmission_group_and_motor_family_not_independent_scalar_noise",
    "runtime_fault_coordinate": (
        "current_runtime_is_generalized_joint_action_slot; physical_parallel_motor_fault_mapping_is_not_claimed"
    ),
    "canonical_root": {
        "site_name": "canonical_root",
        "contract_version": "canonical_root_frame_v1",
        "axes": "right_handed_x_forward_y_left_z_up",
        "native_free_root_qpos_is_canonical": False,
        "downstream_query": "site_pose_and_local_site_twist",
    },
    "legacy_isolation": (
        PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
        LOCOFORMER_MORPHOLOGY_CONTRACT_VERSION,
        ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_VERSION,
    ),
}
MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_HASH = hashlib.sha256(
    json.dumps(
        _MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()
# These aliases make the boundary explicit to callers without making the
# paper-faithful profile look like an official upstream implementation.
LOCOFORMER_EMBODIMENT_CONTRACT_VERSION = LOCOFORMER_MORPHOLOGY_CONTRACT_VERSION
LOCOFORMER_EMBODIMENT_CONTRACT_HASH = LOCOFORMER_MORPHOLOGY_CONTRACT_HASH
ARCHETYPE_CONSTRAINED_EMBODIMENT_CONTRACT_VERSION = (
    ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_VERSION
)
ARCHETYPE_CONSTRAINED_EMBODIMENT_CONTRACT_HASH = (
    ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_HASH
)
MOTOR_DOF_PRESERVING_ARCHETYPE_EMBODIMENT_CONTRACT_VERSION = (
    MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_VERSION
)
MOTOR_DOF_PRESERVING_ARCHETYPE_EMBODIMENT_CONTRACT_HASH = (
    MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_HASH
)


def _stable_manifest_hash(manifest: Mapping[str, object]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _vec3(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("a 3D vector must contain exactly three values")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _fmt(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)


def _axis_for(name: str) -> tuple[float, float, float]:
    return {
        "yaw": (0.0, 0.0, 1.0),
        "roll": (1.0, 0.0, 0.0),
        "pitch": (0.0, 1.0, 0.0),
    }[name]


def _mirrored_axis(name: str) -> tuple[float, float, float]:
    if name == "roll":
        return (-1.0, 0.0, 0.0)
    if name == "yaw":
        return (0.0, 0.0, -1.0)
    return _axis_for(name)


def _joint_range(kind: str) -> tuple[float, float]:
    if kind in {"knee_pitch", "elbow_pitch"}:
        return (-0.15, 2.5)
    if kind.startswith(("ankle", "wrist")):
        return (-1.2, 1.2)
    return (-2.6, 2.6)


@dataclass(frozen=True, slots=True)
class LinkBlueprint:
    name: str
    parent: str
    geom_type: str
    size: tuple[float, float]
    pos: tuple[float, float, float]
    length: float
    mass: float
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)
    contact: bool = True
    end_site: bool = False
    foot: bool = False
    foot_size: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.geom_type not in {"capsule", "box", "cylinder"}:
            raise ValueError(f"unsupported primitive geom type: {self.geom_type}")
        if self.length <= 0 or self.mass <= 0:
            raise ValueError("link length and mass must be positive")
        if self.foot and self.geom_type != "capsule":
            raise ValueError("foot links currently require capsule shanks")
        if self.foot and self.foot_size is None:
            raise ValueError("foot links must define foot_size")
        if self.foot_size is not None and any(value <= 0.0 for value in self.foot_size):
            raise ValueError("foot_size entries must be positive")


@dataclass(frozen=True, slots=True)
class JointBlueprint:
    name: str
    parent_link: str
    child_link: str
    semantic_slot: str
    axis_name: str
    axis: tuple[float, float, float]
    joint_range: tuple[float, float]
    nominal: float = 0.0
    damping: float = 0.5
    friction: float = 0.02
    armature: float = 0.01

    def __post_init__(self) -> None:
        if self.axis_name not in {"yaw", "roll", "pitch", "wheel"}:
            raise ValueError(f"unsupported joint axis: {self.axis_name}")
        if self.joint_range[0] >= self.joint_range[1]:
            raise ValueError("joint range must be increasing")


@dataclass(frozen=True, slots=True)
class ActuatorBlueprint:
    name: str
    joint_name: str
    semantic_slot: str
    kp: float = 30.0
    kd: float = 1.0
    effort_limit: float = 80.0


@dataclass(frozen=True, slots=True)
class WheelBlueprint:
    """Physical and semantic metadata for one terminal rolling wheel.

    Wheel dimensions live in the blueprint because they are morphology
    parameters.  ``PhysicalParams`` still randomizes the usual per-instance
    scales around this geometry, while this record keeps the topology and
    actuator contract visible in manifests.
    """

    link_name: str
    joint_name: str
    semantic_slot: str
    radius: float
    width: float
    axis_name: str = "lateral"
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    joint_range: tuple[float, float] = (-math.pi, math.pi)
    continuous: bool = True
    friction: float = 0.9
    effort_limit: float = 45.0
    kp: float = 0.0
    kd: float = 0.15
    geom_quat: tuple[float, float, float, float] = (
        0.7071067811865476,
        0.7071067811865476,
        0.0,
        0.0,
    )

    def __post_init__(self) -> None:
        if self.radius <= 0.0 or self.width <= 0.0:
            raise ValueError("wheel radius and width must be positive")
        if len(self.axis) != 3 or not any(abs(value) > 1e-12 for value in self.axis):
            raise ValueError("wheel axis must be a non-zero 3D vector")
        if self.joint_range[0] >= self.joint_range[1]:
            raise ValueError("wheel joint range must be increasing")
        if not self.continuous:
            raise ValueError("task069 wheels must use continuous hinge joints")
        if self.friction <= 0.0 or self.effort_limit <= 0.0:
            raise ValueError("wheel friction and effort limit must be positive")
        if len(self.geom_quat) != 4:
            raise ValueError("wheel geom_quat must contain four values")


@dataclass(frozen=True, slots=True)
class MorphologyBlueprint:
    family: MorphologyFamily
    seed: int
    links: tuple[LinkBlueprint, ...]
    joints: tuple[JointBlueprint, ...]
    actuators: tuple[ActuatorBlueprint, ...]
    nominal_height: float
    has_arms: bool
    structural_hash: str
    end_sites: tuple[str, ...] = ()
    profile_version: str = LEGACY_MORPHOLOGY_PROFILE_VERSION
    contract_version: str = PROCEDURAL_EMBODIMENT_CONTRACT_VERSION
    contract_hash: str = PROCEDURAL_EMBODIMENT_CONTRACT_HASH
    wheel_specs: tuple[WheelBlueprint, ...] = ()
    profile_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in {"biped", "quadruped", "wheeled_biped", "wheeled_quadruped"}:
            raise ValueError(f"unsupported morphology family: {self.family}")
        if self.profile_version == LEGACY_MORPHOLOGY_PROFILE_VERSION:
            if self.family not in {"biped", "quadruped"} or self.wheel_specs:
                raise ValueError("legacy_v2 only supports non-wheeled biped/quadruped blueprints")
            if self.contract_version != PROCEDURAL_EMBODIMENT_CONTRACT_VERSION:
                raise ValueError("legacy_v2 contract version does not match the frozen contract")
            if self.contract_hash != PROCEDURAL_EMBODIMENT_CONTRACT_HASH:
                raise ValueError("legacy_v2 contract hash does not match the frozen contract")
        elif self.profile_version == LOCOFORMER_MORPHOLOGY_PROFILE_VERSION:
            if self.contract_version != LOCOFORMER_MORPHOLOGY_CONTRACT_VERSION:
                raise ValueError("paper-faithful profile has an unexpected contract version")
            if self.contract_hash != LOCOFORMER_MORPHOLOGY_CONTRACT_HASH:
                raise ValueError("paper-faithful profile has an unexpected contract hash")
        elif self.profile_version == ARCHETYPE_CONSTRAINED_MORPHOLOGY_PROFILE_VERSION:
            if self.contract_version != ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_VERSION:
                raise ValueError("archetype-constrained profile has an unexpected contract version")
            if self.contract_hash != ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_HASH:
                raise ValueError("archetype-constrained profile has an unexpected contract hash")
        elif self.profile_version == MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_PROFILE_VERSION:
            if self.contract_version != MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_VERSION:
                raise ValueError("motor-DoF-preserving profile has an unexpected contract version")
            if self.contract_hash != MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_HASH:
                raise ValueError("motor-DoF-preserving profile has an unexpected contract hash")
        else:
            raise ValueError(f"unsupported morphology profile: {self.profile_version}")
        if not self.links or not self.joints or not self.actuators:
            raise ValueError("a morphology must contain links, joints, and actuators")
        if self.nominal_height <= 0:
            raise ValueError("nominal height must be positive")
        link_names = {link.name for link in self.links}
        if len(link_names) != len(self.links):
            raise ValueError("link names must be unique")
        joint_names = {joint.name for joint in self.joints}
        if len(joint_names) != len(self.joints):
            raise ValueError("joint names must be unique")
        if any(joint.child_link not in link_names for joint in self.joints):
            raise ValueError("joint child links must exist")
        if any(actuator.joint_name not in joint_names for actuator in self.actuators):
            raise ValueError("actuators must reference existing joints")
        if len({actuator.semantic_slot for actuator in self.actuators}) != len(self.actuators):
            raise ValueError("a blueprint cannot use an actuator semantic slot twice")
        if len({joint.semantic_slot for joint in self.joints}) != len(self.joints):
            raise ValueError("a blueprint cannot use a semantic slot twice")
        unknown_slots = set(self.active_slots) - set(WHOLE_BODY_SLOT_NAMES)
        if unknown_slots:
            declared_candidate_slots = self.profile_metadata.get(
                "task070_candidate_extra_semantic_slots",
                (),
            )
            declared = (
                {str(slot) for slot in declared_candidate_slots}
                if isinstance(declared_candidate_slots, Sequence)
                and not isinstance(declared_candidate_slots, (str, bytes))
                else set()
            )
            candidate_only = (
                self.profile_version
                == MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_PROFILE_VERSION
                and self.profile_metadata.get("candidate_prior_status")
                == "candidate_fail_closed"
                and self.profile_metadata.get("policy_adapter_compatible") is False
                and self.profile_metadata.get("counts_toward_task070_v2_pass") is False
            )
            if not candidate_only or declared != unknown_slots:
                raise ValueError(
                    f"blueprint contains unknown whole-body slots: {sorted(unknown_slots)}"
                )
        link_by_name = {link.name: link for link in self.links}
        for wheel in self.wheel_specs:
            if wheel.link_name not in link_by_name:
                raise ValueError(f"wheel references unknown link {wheel.link_name!r}")
            if wheel.joint_name not in joint_names:
                raise ValueError(f"wheel references unknown joint {wheel.joint_name!r}")
            if wheel.semantic_slot not in self.active_slots:
                raise ValueError("wheel semantic slot must be an active joint slot")
            if not wheel.semantic_slot.endswith("_wheel"):
                raise ValueError("wheel semantic slot must end with '_wheel'")
        wheel_slots = {wheel.semantic_slot for wheel in self.wheel_specs}
        if self.family.startswith("wheeled_"):
            expected_limb_count = 2 if self.family == "wheeled_biped" else 4
            expected_slots = {f"limb{index}_wheel" for index in range(expected_limb_count)}
            if wheel_slots != expected_slots:
                raise ValueError(
                    f"wheeled family must have one wheel per load-bearing limb: {sorted(expected_slots)}"
                )
        elif wheel_slots:
            raise ValueError("non-wheeled families must not contain active wheel topology")

    @property
    def active_slots(self) -> tuple[str, ...]:
        return tuple(joint.semantic_slot for joint in self.joints)

    @property
    def active_slot_mask(self) -> tuple[bool, ...]:
        active = set(self.active_slots)
        return tuple(slot in active for slot in WHOLE_BODY_SLOT_NAMES)

    @property
    def is_wheeled(self) -> bool:
        return bool(self.wheel_specs)

    @property
    def wheels(self) -> tuple[WheelBlueprint, ...]:
        """Alias used by inspection tools and kept separate from link grammar."""

        return self.wheel_specs

    @property
    def embodiment_contract_version(self) -> str:
        return self.contract_version

    @property
    def embodiment_contract_hash(self) -> str:
        return self.contract_hash

    def manifest(self) -> dict[str, object]:
        """Return a JSON-safe structural manifest for train/heldout splits."""

        manifest: dict[str, object] = {
            "family": self.family,
            "seed": self.seed,
            "structural_hash": self.structural_hash,
            "has_arms": self.has_arms,
            "nominal_height": self.nominal_height,
            "active_slots": self.active_slots,
            "links": [asdict(link) for link in self.links],
            "joints": [asdict(joint) for joint in self.joints],
            "actuators": [asdict(actuator) for actuator in self.actuators],
            "end_sites": self.end_sites,
        }
        # Keep the legacy v2 manifest byte-for-byte compatible.  New profile
        # metadata is deliberately opt-in so old cache/checkpoint identities
        # cannot drift merely because this dataclass gained wheel support.
        if self.profile_version != LEGACY_MORPHOLOGY_PROFILE_VERSION or self.wheel_specs:
            manifest.update(
                {
                    "profile_version": self.profile_version,
                    "contract_version": self.contract_version,
                    "contract_hash": self.contract_hash,
                    "wheel_specs": [asdict(wheel) for wheel in self.wheel_specs],
                }
            )
        if self.profile_metadata:
            manifest["profile_metadata"] = dict(self.profile_metadata)
        return manifest


@dataclass(frozen=True, slots=True)
class PhysicalParams:
    """Continuous per-context parameters sampled around one blueprint."""

    global_scale: float
    link_scales: Mapping[str, float]
    mass_scales: Mapping[str, float]
    com_offsets: Mapping[str, tuple[float, float, float]]
    joint_limit_scales: Mapping[str, float]
    nominal_offsets: Mapping[str, float]
    friction: float
    motor_strength: Mapping[str, float]
    kp_scales: Mapping[str, float]
    kd_scales: Mapping[str, float]
    delay_ms: float
    ema_alpha: float
    payload_mass: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def density_scales(self) -> Mapping[str, float]:
        """Mass/density factor alias for callers that randomize density."""

        return self.mass_scales

    def __post_init__(self) -> None:
        if self.global_scale <= 0:
            raise ValueError("global scale must be positive")
        if not 0.0 < self.friction:
            raise ValueError("friction must be positive")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("EMA alpha must be in (0, 1]")
        for values in (
            self.link_scales,
            self.mass_scales,
            self.joint_limit_scales,
            self.motor_strength,
            self.kp_scales,
            self.kd_scales,
        ):
            if any(value <= 0 for value in values.values()):
                raise ValueError("physical scales must be positive")


        if any(abs(value) > 0.15 + 1e-9 for value in self.nominal_offsets.values()):
            raise ValueError("nominal offsets must be within +/-0.15 rad")

    def manifest(self) -> dict[str, object]:
        manifest: dict[str, object] = {
            "global_scale": self.global_scale,
            "link_scales": dict(self.link_scales),
            "mass_scales": dict(self.mass_scales),
            "com_offsets": {key: list(value) for key, value in self.com_offsets.items()},
            "joint_limit_scales": dict(self.joint_limit_scales),
            "nominal_offsets": dict(self.nominal_offsets),
            "friction": self.friction,
            "motor_strength": dict(self.motor_strength),
            "kp_scales": dict(self.kp_scales),
            "kd_scales": dict(self.kd_scales),
            "delay_ms": self.delay_ms,
            "ema_alpha": self.ema_alpha,
            "payload_mass": self.payload_mass,
        }
        if self.metadata:
            manifest["metadata"] = dict(self.metadata)
        return manifest


CANONICAL_ROOT_SITE_NAME = "canonical_root"


@dataclass(frozen=True, slots=True)
class CanonicalRootState:
    world_position: tuple[float, float, float]
    world_quaternion_wxyz: tuple[float, float, float, float]
    local_angular_velocity: tuple[float, float, float]
    local_linear_velocity: tuple[float, float, float]
    projected_gravity: tuple[float, float, float]


def read_canonical_root_state(model: object, data: object, site_id: int | None = None) -> CanonicalRootState:
    """Read the canonical site pose and local velocity without name lookups per step."""
    try:
        import mujoco  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MuJoCo and NumPy are required") from exc
    if site_id is None:
        site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, CANONICAL_ROOT_SITE_NAME))
    if site_id < 0:
        raise ValueError("model has no canonical_root site")
    position = tuple(float(value) for value in data.site_xpos[site_id])
    matrix = np.asarray(data.site_xmat[site_id], dtype=float).reshape(9)
    quaternion = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, matrix)
    velocity = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, velocity, 1)
    projected_gravity = tuple(float(-matrix[6 + column]) for column in range(3))
    return CanonicalRootState(
        world_position=position,
        world_quaternion_wxyz=tuple(float(value) for value in quaternion),
        local_angular_velocity=tuple(float(value) for value in velocity[:3]),
        local_linear_velocity=tuple(float(value) for value in velocity[3:]),
        projected_gravity=projected_gravity,
    )


@dataclass(frozen=True, slots=True)
class MorphologyInstanceKey:
    """Exact compiled embodiment identity used by stance caches and checkpoints.

    A topology hash alone is insufficient because link scales, mass/COM samples,
    shifted joint ranges, and nominal offsets all affect a valid static stance.
    """

    blueprint_hash: str
    physical_hash: str
    embodiment_contract_version: str = PROCEDURAL_EMBODIMENT_CONTRACT_VERSION
    embodiment_contract_hash: str = PROCEDURAL_EMBODIMENT_CONTRACT_HASH

    def __post_init__(self) -> None:
        for label, value in (
            ("blueprint_hash", self.blueprint_hash),
            ("physical_hash", self.physical_hash),
            ("embodiment_contract_hash", self.embodiment_contract_hash),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{label} must be a full SHA-256 digest")
        if not self.embodiment_contract_version:
            raise ValueError("embodiment_contract_version must be non-empty")

    def manifest(self) -> dict[str, str]:
        return {
            "blueprint_hash": self.blueprint_hash,
            "physical_hash": self.physical_hash,
            "embodiment_contract_version": self.embodiment_contract_version,
            "embodiment_contract_hash": self.embodiment_contract_hash,
        }

    @property
    def cache_key(self) -> str:
        return _stable_manifest_hash(self.manifest())


def morphology_blueprint_hash(blueprint: MorphologyBlueprint) -> str:
    """Hash the complete blueprint, not only its topology split identifier."""

    return _stable_manifest_hash(blueprint.manifest())


def physical_params_hash(physical: PhysicalParams) -> str:
    """Hash every continuous geometry/dynamics value that can affect stance."""

    return _stable_manifest_hash(physical.manifest())


def morphology_instance_key(
    blueprint: MorphologyBlueprint,
    physical: PhysicalParams | None = None,
) -> MorphologyInstanceKey:
    """Return the exact cache/checkpoint key for one compiled realization."""

    resolved_physical = physical or _identity_physical_params(blueprint)
    return MorphologyInstanceKey(
        blueprint_hash=morphology_blueprint_hash(blueprint),
        physical_hash=physical_params_hash(resolved_physical),
        embodiment_contract_version=blueprint.embodiment_contract_version,
        embodiment_contract_hash=blueprint.embodiment_contract_hash,
    )


@dataclass(frozen=True, slots=True)
class MorphologyGeneratorConfig:
    arm_probability: float = 0.75
    waist_max_joints: int = 3
    min_link_length: float = 0.12
    max_link_length: float = 0.42
    require_biped_ankle: bool = True
    mirror_biped_legs: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.arm_probability <= 1.0:
            raise ValueError("arm probability must be in [0, 1]")
        if not 0 <= self.waist_max_joints <= 3:
            raise ValueError("waist joints must be in [0, 3]")
        if self.min_link_length <= 0 or self.min_link_length >= self.max_link_length:
            raise ValueError("link length bounds must be positive and ordered")
        if not isinstance(self.require_biped_ankle, bool) or not isinstance(self.mirror_biped_legs, bool):
            raise TypeError("biped grammar toggles must be booleans")


@dataclass(frozen=True, slots=True)
class MorphologySplitManifest:
    """Deterministic train/validation/heldout topology split."""

    train: tuple[MorphologyBlueprint, ...]
    validation: tuple[MorphologyBlueprint, ...]
    heldout: tuple[MorphologyBlueprint, ...]
    parameter_seeds: Mapping[str, int]

    def __post_init__(self) -> None:
        all_items = self.train + self.validation + self.heldout
        hashes = [item.structural_hash for item in all_items]
        if len(hashes) != len(set(hashes)):
            raise ValueError("topology structural hashes must not leak across splits")
        if any(item.family not in {"biped", "quadruped"} for item in all_items):
            raise ValueError("manifest contains an unsupported family")

    def as_dict(self) -> dict[str, object]:
        return {
            "train": [item.manifest() for item in self.train],
            "validation": [item.manifest() for item in self.validation],
            "heldout": [item.manifest() for item in self.heldout],
            "parameter_seeds": dict(self.parameter_seeds),
        }

    @property
    def structural_hash(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class MorphologyGenerator:
    """Seeded primitive morphology generator with a bounded grammar."""

    def __init__(self, config: MorphologyGeneratorConfig | None = None) -> None:
        self.config = config or MorphologyGeneratorConfig()

    def generate(self, family: MorphologyFamily, seed: int) -> MorphologyBlueprint:
        rng = random.Random(seed)
        links: list[LinkBlueprint] = [
            LinkBlueprint(
                name="trunk",
                parent="root",
                geom_type="box",
                size=(0.24 if family == "biped" else 0.42, 0.16 if family == "biped" else 0.22),
                pos=(0.0, 0.0, 0.0),
                length=0.45 if family == "biped" else 0.28,
                mass=12.0 if family == "biped" else 8.0,
                contact=True,
            )
        ]
        joints: list[JointBlueprint] = []
        actuators: list[ActuatorBlueprint] = []
        end_sites: list[str] = []

        if family == "biped":
            nominal_height = 1.05
            mirrored_leg_spec = (
                self._sample_chain_spec(rng, prefix="left_leg", family="leg")
                if self.config.mirror_biped_legs
                else None
            )
            for limb, side in enumerate(("left", "right")):
                self._append_chain(
                    rng,
                    links,
                    joints,
                    actuators,
                    end_sites,
                    prefix=f"{side}_leg",
                    semantic_prefix=f"limb{limb}",
                    parent="trunk",
                    attachment=(0.0, 0.13 if side == "left" else -0.13, -0.23),
                    family="leg",
                    chain_spec=mirrored_leg_spec,
                    mirror_axes=side == "right" and self.config.mirror_biped_legs,
                )
            waist_axes = self._sample_axes(rng, self.config.waist_max_joints, required=None)
            self._append_waist(rng, links, joints, actuators, waist_axes)
            has_arms = rng.random() < self.config.arm_probability
            if has_arms:
                for side, y in (("left", 0.19), ("right", -0.19)):
                    self._append_chain(
                        rng,
                        links,
                        joints,
                        actuators,
                        end_sites,
                        prefix=f"{side}_arm",
                        semantic_prefix=f"{side}_arm",
                        parent="trunk",
                        attachment=(0.0, y, 0.16),
                        family="arm",
                    )
        else:
            nominal_height = 0.52
            for limb, (name, x, y) in enumerate(
                (
                    ("front_left", 0.27, 0.16),
                    ("front_right", 0.27, -0.16),
                    ("rear_left", -0.27, 0.16),
                    ("rear_right", -0.27, -0.16),
                )
            ):
                self._append_chain(
                    rng,
                    links,
                    joints,
                    actuators,
                    end_sites,
                    prefix=f"{name}_leg",
                    semantic_prefix=f"limb{limb}",
                    parent="trunk",
                    attachment=(x, y, -0.13),
                    family="leg",
                )
            has_arms = False

        structural_payload = {
            "family": family,
            "has_arms": has_arms,
            "links": [(link.name, link.parent, link.geom_type, round(link.length, 6)) for link in links],
            "joints": [
                (
                    joint.name,
                    joint.parent_link,
                    joint.child_link,
                    joint.semantic_slot,
                    joint.axis_name,
                    tuple(round(value, 6) for value in joint.axis),
                )
                for joint in joints
            ],
        }
        structural_hash = hashlib.sha256(
            json.dumps(structural_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
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
        )

    def sample_physical_params(
        self,
        blueprint: MorphologyBlueprint,
        seed: int,
        *,
        range_fraction: float = 1.0,
    ) -> PhysicalParams:
        """Sample continuous physics inside a centered fraction of each range.

        ``range_fraction=1`` preserves the full Task051 randomization range;
        ``0.5`` is the specialist curriculum's half-width range and ``0`` is
        the nominal center case.  The initial specialist stage also keeps
        action delay at zero so online motor faults remain a later task.
        """

        if not 0.0 <= range_fraction <= 1.0:
            raise ValueError("range_fraction must be between zero and one")
        rng = random.Random(seed)
        centered = lambda low, high: rng.uniform(
            (low + high) * 0.5 - (high - low) * 0.5 * range_fraction,
            (low + high) * 0.5 + (high - low) * 0.5 * range_fraction,
        )
        global_scale = centered(0.7, 1.3)
        link_scales = {
            link.name: centered(0.75, 1.25) for link in blueprint.links
        }
        mass_scales = {link.name: centered(0.5, 1.5) for link in blueprint.links}
        com_offsets = {
            link.name: tuple(centered(-0.03, 0.03) for _ in range(3))
            for link in blueprint.links
        }
        joint_limit_scales = {
            joint.semantic_slot: centered(0.75, 1.25) for joint in blueprint.joints
        }
        nominal_offsets = {
            joint.semantic_slot: centered(-0.15, 0.15) for joint in blueprint.joints
        }
        motor_strength = {
            joint.semantic_slot: centered(0.5, 2.0) for joint in blueprint.joints
        }
        kp_scales = {joint.semantic_slot: centered(0.7, 1.3) for joint in blueprint.joints}
        kd_scales = {joint.semantic_slot: centered(0.7, 1.3) for joint in blueprint.joints}
        total_mass = sum(link.mass for link in blueprint.links)
        return PhysicalParams(
            global_scale=global_scale,
            link_scales=link_scales,
            mass_scales=mass_scales,
            com_offsets=com_offsets,
            joint_limit_scales=joint_limit_scales,
            nominal_offsets=nominal_offsets,
            friction=centered(0.4, 1.6),
            motor_strength=motor_strength,
            kp_scales=kp_scales,
            kd_scales=kd_scales,
            delay_ms=0.0 if range_fraction < 1.0 else rng.choice((0.0, 20.0)),
            ema_alpha=centered(0.5, 1.0),
            payload_mass=rng.uniform(0.0, min(10.0, 0.3 * total_mass) * range_fraction),
        )

    def _sample_axes(
        self,
        rng: random.Random,
        count: int,
        *,
        required: str | None,
    ) -> tuple[str, ...]:
        count = max(0, min(3, count))
        if count == 0:
            return ()
        axes = ["yaw", "roll", "pitch"]
        if required and required not in axes:
            raise ValueError(f"unsupported required axis {required}")
        if required:
            chosen = [required]
            chosen.extend(axis for axis in rng.sample([a for a in axes if a != required], count - 1))
            rng.shuffle(chosen)
            return tuple(chosen)
        return tuple(rng.sample(axes, count))

    def _append_chain(
        self,
        rng: random.Random,
        links: list[LinkBlueprint],
        joints: list[JointBlueprint],
        actuators: list[ActuatorBlueprint],
        end_sites: list[str],
        *,
        prefix: str,
        semantic_prefix: str,
        parent: str,
        attachment: tuple[float, float, float],
        family: Literal["leg", "arm"],
        chain_spec: tuple[tuple[str, ...], float] | None = None,
        mirror_axes: bool = False,
        terminal_foot: bool = True,
    ) -> tuple[str, float, float]:
        if chain_spec is None:
            required_joints, total_length = self._sample_chain_spec(rng, prefix=prefix, family=family)
        else:
            required_joints, total_length = chain_spec
        segment_length = total_length / len(required_joints)
        current_parent = parent
        previous_length = 0.0
        for index, semantic_suffix in enumerate(required_joints):
            if semantic_suffix.startswith(("shoulder_", "hip_")):
                axis_name = semantic_suffix.rsplit("_", 1)[-1]
            elif semantic_suffix in {"knee_pitch", "elbow_pitch"}:
                axis_name = "pitch"
            else:
                axis_name = semantic_suffix.rsplit("_", 1)[-1]
            semantic_slot = f"{semantic_prefix}_{semantic_suffix}"
            child_link = f"{prefix}_{index}_link"
            joint_name = f"{child_link}_joint"
            pos = attachment if index == 0 else (0.0, 0.0, -previous_length)
            geom_type = "capsule"
            radius = 0.045 if family == "leg" else 0.035
            link_mass = 3.0 if family == "leg" else 1.2
            is_terminal_leg = family == "leg" and index == len(required_joints) - 1
            has_terminal_foot = is_terminal_leg and terminal_foot
            links.append(
                LinkBlueprint(
                    name=child_link,
                    parent=current_parent,
                    geom_type=geom_type,
                    size=(radius, segment_length * 0.5),
                    pos=pos,
                    length=segment_length,
                    mass=link_mass,
                    # Terminal arm links remain collidable so a hand/forearm
                    # touch can be charged as an undesired (non-foot) contact.
                    # For a wheeled leg the terminal capsule is non-contact;
                    # the wheel is the sole load-bearing terminal geom.
                    contact=(
                        has_terminal_foot
                        if family == "leg" and is_terminal_leg
                        else index == len(required_joints) - 1
                    ),
                    end_site=has_terminal_foot,
                    foot=has_terminal_foot,
                    foot_size=(
                        FOOTPAD_HALF_LENGTH_FRACTION * total_length,
                        FOOTPAD_HALF_WIDTH_FRACTION * total_length,
                    )
                    if has_terminal_foot
                    else None,
                )
            )
            joint = JointBlueprint(
                name=joint_name,
                parent_link=current_parent,
                child_link=child_link,
                semantic_slot=semantic_slot,
                axis_name=axis_name,
                axis=_mirrored_axis(axis_name) if mirror_axes else _axis_for(axis_name),
                joint_range=_joint_range(semantic_suffix),
                nominal=0.55 if semantic_suffix in {"knee_pitch", "elbow_pitch"} else 0.0,
            )
            joints.append(joint)
            actuators.append(
                ActuatorBlueprint(
                    name=f"{joint_name}_actuator",
                    joint_name=joint_name,
                    semantic_slot=semantic_slot,
                )
            )
            current_parent = child_link
            previous_length = segment_length
            if links[-1].end_site:
                end_sites.append(f"{child_link}_foot")
        return current_parent, previous_length, total_length

    def _sample_chain_spec(
        self,
        rng: random.Random,
        *,
        prefix: str,
        family: Literal["leg", "arm"],
    ) -> tuple[tuple[str, ...], float]:
        if family == "leg":
            is_biped_leg = prefix in {"left_leg", "right_leg"}
            if is_biped_leg and self.config.require_biped_ankle:
                proximal_count = rng.randint(2, 3)
                proximal = ["pitch", "roll"]
                if proximal_count > 2:
                    proximal.append("yaw")
                rng.shuffle(proximal)
                distal_count = rng.randint(1, 2)
                distal = ["pitch"]
                if distal_count > 1:
                    distal.append("roll")
                rng.shuffle(distal)
            else:
                proximal_count = rng.randint(1 if is_biped_leg else 2, 3)
                proximal = list(self._sample_axes(rng, proximal_count, required="pitch"))
                distal_count = rng.randint(0, 2 if is_biped_leg else 1)
                distal = list(rng.sample(("roll", "pitch"), distal_count))
            required_joints = tuple(f"hip_{axis}" for axis in proximal) + ("knee_pitch",) + tuple(
                f"ankle_{axis}" for axis in distal
            )
            total_length = rng.uniform(0.58, 0.78) if is_biped_leg else rng.uniform(0.28, 0.42)
            return required_joints, total_length

        # Shoulder axes are a genuinely random non-empty subset; unlike a leg
        # there is no mandatory pitch axis for an arm topology.
        proximal = self._sample_axes(rng, rng.randint(1, 3), required=None)
        distal = self._sample_axes(rng, rng.randint(0, 3), required=None)
        required_joints = tuple(f"shoulder_{axis}" for axis in proximal) + ("elbow_pitch",) + tuple(
            f"wrist_{axis}" for axis in distal
        )
        return required_joints, rng.uniform(0.42, 0.62)

    def _append_waist(
        self,
        rng: random.Random,
        links: list[LinkBlueprint],
        joints: list[JointBlueprint],
        actuators: list[ActuatorBlueprint],
        axes: Sequence[str],
    ) -> None:
        parent = "trunk"
        for index, axis_name in enumerate(axes):
            child_link = f"waist_{axis_name}_link"
            joint_name = f"{child_link}_joint"
            links.append(
                LinkBlueprint(
                    name=child_link,
                    parent=parent,
                    geom_type="box",
                    size=(0.16, 0.1),
                    pos=(0.0, 0.0, 0.28 if index == 0 else 0.0),
                    length=0.12,
                    mass=1.0,
                    contact=False,
                )
            )
            joints.append(
                JointBlueprint(
                    name=joint_name,
                    parent_link=parent,
                    child_link=child_link,
                    semantic_slot=f"waist_{axis_name}",
                    axis_name=axis_name,
                    axis=_axis_for(axis_name),
                    joint_range=(-1.2, 1.2),
                )
            )
            actuators.append(
                ActuatorBlueprint(
                    name=f"{joint_name}_actuator",
                    joint_name=joint_name,
                    semantic_slot=f"waist_{axis_name}",
                )
            )
            parent = child_link


def compile_mjcf(
    blueprint: MorphologyBlueprint,
    physical: PhysicalParams | None = None,
    *,
    include_floor: bool = True,
) -> str:
    """Compile one primitive blueprint into a self-contained MJCF string."""

    physical = physical or _identity_physical_params(blueprint)
    root = ET.Element("mujoco", {"model": f"procedural_{blueprint.structural_hash}"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})
    worldbody = ET.SubElement(root, "worldbody")
    if include_floor:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "floor",
                "type": "plane",
                "size": "20 20 0.1",
                "pos": "0 0 0",
                "friction": f"{physical.friction:.6g} 0.1 0.1",
            },
        )
    root_body = ET.SubElement(
        worldbody,
        "body",
        {"name": "root", "pos": f"0 0 {blueprint.nominal_height * physical.global_scale:.6g}"},
    )
    ET.SubElement(root_body, "joint", {"name": "root_free", "type": "free"})
    links_by_parent: dict[str, list[LinkBlueprint]] = defaultdict(list)
    for link in blueprint.links:
        links_by_parent[link.parent].append(link)
    joints_by_child = {joint.child_link: joint for joint in blueprint.joints}
    wheels_by_link = {wheel.link_name: wheel for wheel in blueprint.wheel_specs}
    wheels_by_joint = {wheel.joint_name: wheel for wheel in blueprint.wheel_specs}
    show_joint_markers = (
        blueprint.profile_metadata.get("joint_marker_sites") == "source_motor_origins"
    )
    link_visual_rgba = blueprint.profile_metadata.get("link_visual_rgba", {})
    capsule_local_fromto = blueprint.profile_metadata.get("capsule_local_fromto", {})
    body_local_quat = blueprint.profile_metadata.get("body_local_quat", {})
    geom_local_pos = blueprint.profile_metadata.get("geom_local_pos", {})
    footpad_local_pos = blueprint.profile_metadata.get("footpad_local_pos", {})
    footpad_visual_rgba = blueprint.profile_metadata.get("footpad_visual_rgba")
    joint_marker_rgba = blueprint.profile_metadata.get("joint_marker_rgba")
    auxiliary_capsule_visuals = blueprint.profile_metadata.get(
        "auxiliary_capsule_visuals",
        {},
    )
    canonical = blueprint.profile_metadata.get("canonical_root_frame")
    if canonical is not None:
        if not isinstance(canonical, Mapping):
            raise ValueError("canonical_root_frame metadata must be a mapping")
        required = {
            "contract_version", "site_name", "site_body_link", "origin",
            "anchor_body_from_canonical", "canonical_from_anchor_body",
            "native_free_root_qpos_is_canonical", "axis_semantics",
            "downstream_query_guidance",
            "coordinate_convention",
        }
        if set(canonical) != required or canonical["contract_version"] != "canonical_root_frame_v1":
            raise ValueError("invalid canonical_root_frame metadata")
        if canonical["site_name"] != "canonical_root" or canonical["native_free_root_qpos_is_canonical"] is not False:
            raise ValueError("canonical_root_frame has an invalid site contract")
        convention = canonical["coordinate_convention"]
        expected_convention = {
            "handedness": "right_handed",
            "x": "forward",
            "y": "left",
            "z": "up",
            "quaternion_order": "wxyz",
            "pose": "site_to_world",
            "twist": "expressed_in_canonical_frame/angular_then_linear",
            "projected_gravity": "world_minus_z_expressed_in_canonical",
            "transform_units": (
                "translation_is_blueprint_length_scaled_by_physical.global_scale_at_compile"
            ),
        }
        if not isinstance(convention, Mapping) or dict(convention) != expected_convention:
            raise ValueError("canonical_root_frame coordinate convention is invalid")
        if canonical["site_body_link"] not in {link.name for link in blueprint.links}:
            raise ValueError("canonical_root_frame references an unknown body")
        origin = canonical["origin"]
        transform = canonical["anchor_body_from_canonical"]
        if not isinstance(origin, Sequence) or len(origin) != 3 or not isinstance(transform, Mapping):
            raise ValueError("canonical_root_frame transform is malformed")
        quat = transform.get("quaternion_wxyz")
        if not isinstance(quat, Sequence) or len(quat) != 4:
            raise ValueError("canonical_root_frame quaternion is malformed")

    def append_children(parent_xml: ET.Element, parent_name: str) -> None:
        for link in links_by_parent.get(parent_name, []):
            scale = physical.global_scale * physical.link_scales.get(link.name, 1.0)
            attrs = {
                "name": link.name,
                "pos": _fmt(tuple(value * physical.global_scale for value in link.pos)),
            }
            if isinstance(body_local_quat, Mapping):
                quat = body_local_quat.get(link.name)
                if isinstance(quat, Sequence) and len(quat) == 4:
                    attrs["quat"] = _fmt(tuple(float(value) for value in quat))
            body_xml = ET.SubElement(parent_xml, "body", attrs)
            if canonical is not None and link.name == canonical["site_body_link"]:
                transform = canonical["anchor_body_from_canonical"]
                ET.SubElement(
                    body_xml,
                    "site",
                    {
                        "name": "canonical_root",
                        "type": "sphere",
                        "pos": _fmt(tuple(float(value) * physical.global_scale for value in canonical["origin"])),
                        "quat": _fmt(tuple(float(value) for value in transform["quaternion_wxyz"])),
                        "size": "0.012",
                        "rgba": "0 0 0 0",
                    },
                )
            joint = joints_by_child.get(link.name)
            if joint is not None:
                wheel = wheels_by_joint.get(joint.name)
                joint_attrs = {
                    "name": joint.name,
                    "type": "hinge",
                    "axis": _fmt(joint.axis),
                }
                if wheel is None:
                    joint_attrs["range"] = _fmt(_scaled_joint_range(joint, physical))
                else:
                    # A wheel rotates without a wraparound limit.  The
                    # representative +/-pi range remains in the JSON
                    # manifest for auditability, but is not a joint limit.
                    joint_attrs["limited"] = "false"
                joint_attrs.update(
                    {
                        "damping": f"{joint.damping:.6g}",
                        "frictionloss": f"{joint.friction:.6g}",
                        "armature": f"{joint.armature:.6g}",
                    }
                )
                ET.SubElement(
                    body_xml,
                    "joint",
                    joint_attrs,
                )
            mass = link.mass * physical.mass_scales.get(link.name, 1.0)
            wheel = wheels_by_link.get(link.name)
            geom_attrs = {
                "name": f"{link.name}_geom",
                "type": link.geom_type,
                "mass": f"{mass:.6g}",
                "friction": f"{physical.friction:.6g} 0.1 0.1",
                "contype": "1" if link.contact and not link.foot else "0",
                "conaffinity": "1" if link.contact and not link.foot else "0",
            }
            if isinstance(link_visual_rgba, Mapping):
                rgba = link_visual_rgba.get(link.name)
                if isinstance(rgba, Sequence) and len(rgba) == 4:
                    geom_attrs["rgba"] = _fmt(tuple(float(value) for value in rgba))
            if isinstance(geom_local_pos, Mapping):
                local_pos = geom_local_pos.get(link.name)
                if isinstance(local_pos, Sequence) and len(local_pos) == 3:
                    geom_attrs["pos"] = _fmt(tuple(float(value) * scale for value in local_pos))
            if wheel is not None:
                geom_attrs["friction"] = (
                    f"{physical.friction * wheel.friction:.6g} 0.1 0.1"
                )
                geom_attrs["quat"] = _fmt(wheel.geom_quat)
            if link.geom_type == "capsule":
                fromto = None
                if isinstance(capsule_local_fromto, Mapping):
                    fromto = capsule_local_fromto.get(link.name)
                if isinstance(fromto, Sequence) and len(fromto) == 6:
                    geom_attrs["fromto"] = _fmt(tuple(float(value) * scale for value in fromto))
                else:
                    geom_attrs["fromto"] = _fmt((0.0, 0.0, 0.0, 0.0, 0.0, -link.length * scale))
                geom_attrs["size"] = f"{link.size[0] * scale:.6g}"
            elif link.geom_type == "cylinder":
                geom_attrs["size"] = _fmt((link.size[0] * scale, link.length * scale * 0.5))
            else:
                geom_attrs["size"] = _fmt((link.size[0] * scale, link.size[1] * scale, link.length * scale * 0.5))
            ET.SubElement(body_xml, "geom", geom_attrs)
            if isinstance(auxiliary_capsule_visuals, Mapping):
                visual_specs = auxiliary_capsule_visuals.get(link.name, ())
                if isinstance(visual_specs, Sequence):
                    for visual_spec in visual_specs:
                        if not isinstance(visual_spec, Mapping):
                            raise TypeError("auxiliary capsule visual must be a mapping")
                        fromto = visual_spec.get("fromto")
                        rgba = visual_spec.get("rgba")
                        if not isinstance(fromto, Sequence) or len(fromto) != 6:
                            raise ValueError("auxiliary capsule fromto must contain six values")
                        if not isinstance(rgba, Sequence) or len(rgba) != 4:
                            raise ValueError("auxiliary capsule rgba must contain four values")
                        ET.SubElement(
                            body_xml,
                            "geom",
                            {
                                "name": f"{link.name}_{visual_spec['name']}_visual",
                                "type": "capsule",
                                "fromto": _fmt(
                                    tuple(float(value) * scale for value in fromto)
                                ),
                                "size": f"{float(visual_spec['radius']) * scale:.6g}",
                                "density": "0",
                                "contype": "0",
                                "conaffinity": "0",
                                "rgba": _fmt(tuple(float(value) for value in rgba)),
                            },
                        )
            if link.foot:
                if link.foot_size is None:
                    raise ValueError(f"foot link {link.name} is missing foot_size")
                radius = link.size[0] * scale
                footpad_pos = (0.0, 0.0, -(link.length * scale + radius))
                if isinstance(footpad_local_pos, Mapping):
                    local_pos = footpad_local_pos.get(link.name)
                    if isinstance(local_pos, Sequence) and len(local_pos) == 3:
                        footpad_pos = tuple(float(value) * scale for value in local_pos)
                footpad_attrs = {
                    "name": f"{link.name}_footpad",
                    "type": "box",
                    "pos": _fmt(footpad_pos),
                    "size": _fmt((link.foot_size[0] * scale, link.foot_size[1] * scale, radius)),
                    "friction": f"{physical.friction:.6g} 0.1 0.1",
                    "contype": "1",
                    "conaffinity": "1",
                }
                if isinstance(footpad_visual_rgba, Sequence) and len(footpad_visual_rgba) == 4:
                    footpad_attrs["rgba"] = _fmt(
                        tuple(float(value) for value in footpad_visual_rgba)
                    )
                ET.SubElement(
                    body_xml,
                    "geom",
                    footpad_attrs,
                )
            if link.mass > 0:
                com = physical.com_offsets.get(link.name, link.com)
                inertial = _primitive_diaginertia(link, mass=mass, scale=scale)
                ET.SubElement(
                    body_xml,
                    "inertial",
                    {
                        "pos": _fmt(com),
                        "mass": f"{mass:.6g}",
                        "diaginertia": _fmt(inertial),
                    },
                )
            if show_joint_markers and joint is not None:
                marker_rgba = "0.1 0.45 1 1"
                if isinstance(joint_marker_rgba, Sequence) and len(joint_marker_rgba) == 4:
                    marker_rgba = _fmt(tuple(float(value) for value in joint_marker_rgba))
                ET.SubElement(
                    body_xml,
                    "site",
                    {
                        "name": f"{link.name}_joint_marker",
                        "type": "sphere",
                        "pos": "0 0 0",
                        "size": f"{0.025 * scale:.6g}",
                        "rgba": marker_rgba,
                    },
                )
            if link.end_site:
                ET.SubElement(
                    body_xml,
                    "site",
                    {
                        "name": f"{link.name}_foot",
                        "type": "sphere",
                        "pos": _fmt((0.0, 0.0, -link.length * scale)),
                        "size": f"{0.05 * scale:.6g}",
                    },
                )
            append_children(body_xml, link.name)

    append_children(root_body, "root")
    if physical.payload_mass > 0.0:
        ET.SubElement(
            root_body,
            "inertial",
            {
                "pos": "0 0 0",
                "mass": f"{physical.payload_mass:.6g}",
                "diaginertia": _fmt((physical.payload_mass * 0.01,) * 3),
            },
        )
    actuator_xml = ET.SubElement(root, "actuator")
    for actuator in blueprint.actuators:
        strength = physical.motor_strength.get(actuator.semantic_slot, 1.0)
        kp_scale = physical.kp_scales.get(actuator.semantic_slot, 1.0)
        kd_scale = physical.kd_scales.get(actuator.semantic_slot, 1.0)
        joint = next(item for item in blueprint.joints if item.name == actuator.joint_name)
        wheel = wheels_by_joint.get(actuator.joint_name)
        if wheel is None:
            ET.SubElement(
                actuator_xml,
                "position",
                {
                    "name": actuator.name,
                    "joint": actuator.joint_name,
                    "kp": f"{actuator.kp * kp_scale:.6g}",
                    "kv": f"{actuator.kd * kd_scale:.6g}",
                    "forcerange": f"{-actuator.effort_limit * strength:.6g} {actuator.effort_limit * strength:.6g}",
                    "ctrlrange": _fmt(_scaled_joint_range(joint, physical)),
                },
            )
        else:
            effort = wheel.effort_limit * strength
            ET.SubElement(
                actuator_xml,
                "motor",
                {
                    "name": actuator.name,
                    "joint": actuator.joint_name,
                    "gear": "1",
                    "ctrllimited": "true",
                    "ctrlrange": _fmt((-effort, effort)),
                    "forcelimited": "true",
                    "forcerange": _fmt((-effort, effort)),
                },
            )
    ET.indent(ET.ElementTree(root), space="  ")
    return ET.tostring(root, encoding="unicode")


def validate_mjcf_text(xml_text: str) -> None:
    """Validate XML syntax without requiring MuJoCo to be installed."""

    root = ET.fromstring(xml_text)
    if root.tag != "mujoco":
        raise ValueError("MJCF root must be <mujoco>")
    if root.find("worldbody") is None or root.find("actuator") is None:
        raise ValueError("MJCF must contain worldbody and actuator sections")


def compile_with_mujoco(xml_text: str) -> object:
    """Compile using the optional MuJoCo dependency for smoke tests."""

    try:
        import mujoco  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise RuntimeError("MuJoCo is required for compile_with_mujoco") from exc
    return mujoco.MjModel.from_xml_string(xml_text)


def _scaled_joint_range(joint: JointBlueprint, physical: PhysicalParams) -> tuple[float, float]:
    scale = physical.joint_limit_scales.get(joint.semantic_slot, 1.0)
    offset = physical.nominal_offsets.get(joint.semantic_slot, 0.0)
    lower, upper = joint.joint_range
    return (lower * scale + offset, upper * scale + offset)


def _primitive_diaginertia(
    link: LinkBlueprint,
    *,
    mass: float,
    scale: float,
) -> tuple[float, float, float]:
    """Compute a positive diagonal inertia from the primitive dimensions.

    The generator uses body-local z as the link's long axis.  These closed-form
    approximations are sufficient for the primitive training distribution and,
    importantly, make mass/inertia changes physically coupled instead of
    sampling unrelated inertial scalars.
    """

    radius = link.size[0] * scale
    length = link.length * scale
    if link.geom_type == "box":
        half_x = link.size[0] * scale
        half_y = link.size[1] * scale
        half_z = 0.5 * length
        return (
            mass * (half_y * half_y + half_z * half_z) / 3.0,
            mass * (half_x * half_x + half_z * half_z) / 3.0,
            mass * (half_x * half_x + half_y * half_y) / 3.0,
        )
    if link.geom_type == "cylinder":
        return (
            mass * (3.0 * radius * radius + length * length) / 12.0,
            mass * (3.0 * radius * radius + length * length) / 12.0,
            mass * radius * radius / 2.0,
        )
    # Capsule: model the swept volume as a cylinder plus two hemispheres; the
    # conservative cylinder approximation keeps all three values positive.
    return (
        mass * (length * length / 12.0 + radius * radius / 4.0),
        mass * (length * length / 12.0 + radius * radius / 4.0),
        mass * radius * radius / 2.0,
    )


def _identity_physical_params(blueprint: MorphologyBlueprint) -> PhysicalParams:
    return PhysicalParams(
        global_scale=1.0,
        link_scales={link.name: 1.0 for link in blueprint.links},
        mass_scales={link.name: 1.0 for link in blueprint.links},
        com_offsets={link.name: link.com for link in blueprint.links},
        joint_limit_scales={joint.semantic_slot: 1.0 for joint in blueprint.joints},
        nominal_offsets={joint.semantic_slot: 0.0 for joint in blueprint.joints},
        friction=0.8,
        motor_strength={joint.semantic_slot: 1.0 for joint in blueprint.joints},
        kp_scales={joint.semantic_slot: 1.0 for joint in blueprint.joints},
        kd_scales={joint.semantic_slot: 1.0 for joint in blueprint.joints},
        delay_ms=0.0,
        ema_alpha=1.0,
        payload_mass=0.0,
    )


def build_morphology_split_manifest(
    generator: MorphologyGenerator | None = None,
    *,
    train_per_family: int = 32,
    validation_per_family: int = 8,
    heldout_per_family: int = 8,
    seed_stride: int = 1_000_003,
) -> MorphologySplitManifest:
    """Create the fixed 32/8/8 biped/quadruped split without named robots.

    Seeds are searched deterministically until structural hashes are unique,
    so changing the random grammar cannot silently introduce train/heldout
    topology leakage.
    """

    if min(train_per_family, validation_per_family, heldout_per_family) <= 0:
        raise ValueError("every split must contain at least one topology")
    generator = generator or MorphologyGenerator()
    requested = train_per_family + validation_per_family + heldout_per_family
    train: list[MorphologyBlueprint] = []
    validation: list[MorphologyBlueprint] = []
    heldout: list[MorphologyBlueprint] = []
    parameter_seeds: dict[str, int] = {}
    for family_index, family in enumerate(("biped", "quadruped")):
        seen: set[str] = set()
        candidates: list[MorphologyBlueprint] = []
        candidate_seed = family_index * seed_stride
        while len(candidates) < requested:
            blueprint = generator.generate(family, candidate_seed)
            candidate_seed += 1
            if blueprint.structural_hash in seen:
                continue
            seen.add(blueprint.structural_hash)
            candidates.append(blueprint)
        train.extend(candidates[:train_per_family])
        validation.extend(candidates[train_per_family : train_per_family + validation_per_family])
        heldout.extend(candidates[train_per_family + validation_per_family :])
        for blueprint in candidates:
            parameter_seeds[blueprint.structural_hash] = blueprint.seed + 10_000_000
    return MorphologySplitManifest(
        train=tuple(train),
        validation=tuple(validation),
        heldout=tuple(heldout),
        parameter_seeds=parameter_seeds,
    )


@dataclass(frozen=True, slots=True)
class LocoFormerMorphologyGeneratorConfig:
    """Local implementation choices for the paper-faithful envelope.

    The public paper identifies the four family categories and procedural
    training bodies, but does not publish the exact generator ranges.  These
    values are therefore versioned repository choices, not claimed paper
    parameters.
    """

    arm_probability: float = 0.65
    waist_max_joints: int = 3
    require_biped_ankle: bool = False
    mirror_biped_legs: bool = False
    biped_ankle_probability: float = 0.6
    biped_extra_hip_yaw_probability: float = 0.45
    quadruped_three_axis_probability: float = 0.55
    wheel_radius_range: tuple[float, float] = (0.09, 0.18)
    wheel_width_range: tuple[float, float] = (0.05, 0.11)
    wheel_friction_range: tuple[float, float] = (0.7, 1.2)
    wheel_effort_range: tuple[float, float] = (25.0, 60.0)

    def __post_init__(self) -> None:
        for label, value in (
            ("arm_probability", self.arm_probability),
            ("biped_ankle_probability", self.biped_ankle_probability),
            ("biped_extra_hip_yaw_probability", self.biped_extra_hip_yaw_probability),
            ("quadruped_three_axis_probability", self.quadruped_three_axis_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1]")
        if not 0 <= self.waist_max_joints <= 3:
            raise ValueError("waist_max_joints must be in [0, 3]")
        if not isinstance(self.require_biped_ankle, bool) or not isinstance(
            self.mirror_biped_legs, bool
        ):
            raise TypeError("biped grammar toggles must be booleans")
        for label, bounds in (
            ("wheel_radius_range", self.wheel_radius_range),
            ("wheel_width_range", self.wheel_width_range),
            ("wheel_friction_range", self.wheel_friction_range),
            ("wheel_effort_range", self.wheel_effort_range),
        ):
            if len(bounds) != 2 or bounds[0] <= 0.0 or bounds[0] >= bounds[1]:
                raise ValueError(f"{label} must contain two positive ordered values")


class LocoFormerMorphologyGenerator(MorphologyGenerator):
    """Versioned, paper-faithful local four-family morphology generator.

    This class deliberately sits beside the frozen ``MorphologyGenerator``.
    The latter remains the v2 Task067 profile; callers must opt into this
    profile explicitly so old XML, manifest, and cache identities cannot drift.
    """

    profile_version = LOCOFORMER_MORPHOLOGY_PROFILE_VERSION
    contract_version = LOCOFORMER_MORPHOLOGY_CONTRACT_VERSION
    contract_hash = LOCOFORMER_MORPHOLOGY_CONTRACT_HASH

    def __init__(
        self,
        config: LocoFormerMorphologyGeneratorConfig | None = None,
    ) -> None:
        self.config = config or LocoFormerMorphologyGeneratorConfig()

    def generate(self, family: MorphologyFamily, seed: int) -> MorphologyBlueprint:
        if family not in {"biped", "quadruped", "wheeled_biped", "wheeled_quadruped"}:
            raise ValueError(f"unsupported paper-faithful morphology family: {family!r}")
        rng = random.Random(seed)
        base_family = "biped" if family.endswith("biped") else "quadruped"
        wheeled = family.startswith("wheeled_")
        if base_family == "biped":
            trunk_size = (rng.uniform(0.20, 0.32), rng.uniform(0.12, 0.20))
            trunk_length = rng.uniform(0.38, 0.56)
            trunk_mass = rng.uniform(8.0, 16.0)
        else:
            trunk_size = (rng.uniform(0.34, 0.52), rng.uniform(0.18, 0.28))
            trunk_length = rng.uniform(0.22, 0.36)
            trunk_mass = rng.uniform(6.0, 12.0)
        links: list[LinkBlueprint] = [
            LinkBlueprint(
                name="trunk",
                parent="root",
                geom_type="box",
                size=trunk_size,
                pos=(0.0, 0.0, 0.0),
                length=trunk_length,
                mass=trunk_mass,
                contact=True,
            )
        ]
        joints: list[JointBlueprint] = []
        actuators: list[ActuatorBlueprint] = []
        wheels: list[WheelBlueprint] = []
        end_sites: list[str] = []
        leg_lengths: list[float] = []

        if base_family == "biped":
            shared_spec = self._sample_leg_spec(rng, biped=True)
            leg_specs = (
                (shared_spec, shared_spec)
                if self.config.mirror_biped_legs
                else (shared_spec, self._sample_leg_spec(rng, biped=True))
            )
            for limb, (side, spec) in enumerate(zip(("left", "right"), leg_specs)):
                terminal_link, terminal_length, total_length = self._append_chain(
                    rng,
                    links,
                    joints,
                    actuators,
                    end_sites,
                    prefix=f"{side}_leg",
                    semantic_prefix=f"limb{limb}",
                    parent="trunk",
                    attachment=(0.0, 0.13 if side == "left" else -0.13, -0.23),
                    family="leg",
                    chain_spec=spec,
                    mirror_axes=side == "right" and self.config.mirror_biped_legs,
                    terminal_foot=not wheeled,
                )
                leg_lengths.append(total_length)
                if wheeled:
                    wheels.append(
                        self._append_wheel(
                            rng,
                            links,
                            joints,
                            actuators,
                            terminal_link=terminal_link,
                            terminal_length=terminal_length,
                            semantic_prefix=f"limb{limb}",
                            prefix=f"{side}_leg",
                        )
                    )
            waist_axes = self._sample_axes(rng, self.config.waist_max_joints, required=None)
            self._append_waist(rng, links, joints, actuators, waist_axes)
            has_arms = rng.random() < self.config.arm_probability
            if has_arms:
                arm_spec_left = self._sample_arm_spec(rng)
                arm_specs = (
                    (arm_spec_left, arm_spec_left)
                    if self.config.mirror_biped_legs
                    else (arm_spec_left, self._sample_arm_spec(rng))
                )
                arm_lateral_offset = (
                    (trunk_size[1] + ARM_LINK_RADIUS) * MAX_PHYSICAL_LINK_SCALE
                    + LOCOFORMER_ARM_TRUNK_CLEARANCE
                )
                for side, y, spec in zip(
                    ("left", "right"), (arm_lateral_offset, -arm_lateral_offset), arm_specs
                ):
                    self._append_chain(
                        rng,
                        links,
                        joints,
                        actuators,
                        end_sites,
                        prefix=f"{side}_arm",
                        semantic_prefix=f"{side}_arm",
                        parent="trunk",
                        attachment=(0.0, y, 0.16),
                        family="arm",
                        chain_spec=spec,
                        mirror_axes=side == "right" and self.config.mirror_biped_legs,
                        terminal_foot=False,
                    )
        else:
            has_arms = False
            for limb, (name, x, y) in enumerate(
                (
                    ("front_left", 0.27, 0.16),
                    ("front_right", 0.27, -0.16),
                    ("rear_left", -0.27, 0.16),
                    ("rear_right", -0.27, -0.16),
                )
            ):
                spec = self._sample_leg_spec(rng, biped=False)
                terminal_link, terminal_length, total_length = self._append_chain(
                    rng,
                    links,
                    joints,
                    actuators,
                    end_sites,
                    prefix=f"{name}_leg",
                    semantic_prefix=f"limb{limb}",
                    parent="trunk",
                    attachment=(x, y, -0.13),
                    family="leg",
                    chain_spec=spec,
                    terminal_foot=not wheeled,
                )
                leg_lengths.append(total_length)
                if wheeled:
                    wheels.append(
                        self._append_wheel(
                            rng,
                            links,
                            joints,
                            actuators,
                            terminal_link=terminal_link,
                            terminal_length=terminal_length,
                            semantic_prefix=f"limb{limb}",
                            prefix=f"{name}_leg",
                        )
                    )

        wheel_radius = max((wheel.radius for wheel in wheels), default=0.05)
        nominal_height = (
            (0.23 if base_family == "biped" else 0.14)
            + max(leg_lengths)
            + wheel_radius
            + 0.03
        )
        structural_payload = {
            "profile_version": self.profile_version,
            "family": family,
            "has_arms": has_arms,
            "links": [(link.name, link.parent, link.geom_type) for link in links],
            "joints": [
                (
                    joint.name,
                    joint.parent_link,
                    joint.child_link,
                    joint.semantic_slot,
                    joint.axis_name,
                    tuple(round(value, 6) for value in joint.axis),
                )
                for joint in joints
            ],
            "wheel_topology": [
                (
                    wheel.link_name,
                    wheel.joint_name,
                    wheel.semantic_slot,
                    wheel.axis_name,
                    tuple(round(value, 6) for value in wheel.axis),
                    wheel.continuous,
                )
                for wheel in wheels
            ],
        }
        structural_hash = hashlib.sha256(
            json.dumps(structural_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
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
        )

    def _sample_leg_spec(
        self,
        rng: random.Random,
        *,
        biped: bool,
    ) -> tuple[tuple[str, ...], float]:
        if biped:
            proximal = ["pitch", "roll"]
            if rng.random() < self.config.biped_extra_hip_yaw_probability:
                proximal.append("yaw")
            rng.shuffle(proximal)
            has_ankle = self.config.require_biped_ankle or (
                rng.random() < self.config.biped_ankle_probability
            )
            distal = ["pitch"] if has_ankle else []
            if has_ankle and rng.random() < 0.45:
                distal.append("roll")
            rng.shuffle(distal)
            total_length = rng.uniform(0.60, 0.92)
        else:
            count = 3 if rng.random() < self.config.quadruped_three_axis_probability else 2
            proximal = list(self._sample_axes(rng, count, required="pitch"))
            distal = list(rng.sample(("roll", "pitch"), rng.randint(0, 1)))
            total_length = rng.uniform(0.26, 0.52)
        required_joints = tuple(f"hip_{axis}" for axis in proximal) + ("knee_pitch",) + tuple(
            f"ankle_{axis}" for axis in distal
        )
        return required_joints, total_length

    def _sample_arm_spec(self, rng: random.Random) -> tuple[tuple[str, ...], float]:
        proximal = self._sample_axes(rng, rng.randint(1, 3), required=None)
        distal = self._sample_axes(rng, rng.randint(0, 2), required=None)
        return (
            tuple(f"shoulder_{axis}" for axis in proximal)
            + ("elbow_pitch",)
            + tuple(f"wrist_{axis}" for axis in distal),
            rng.uniform(0.38, 0.62),
        )

    def _append_wheel(
        self,
        rng: random.Random,
        links: list[LinkBlueprint],
        joints: list[JointBlueprint],
        actuators: list[ActuatorBlueprint],
        *,
        terminal_link: str,
        terminal_length: float,
        semantic_prefix: str,
        prefix: str,
    ) -> WheelBlueprint:
        config = self.config
        radius = rng.uniform(*config.wheel_radius_range)
        width = rng.uniform(*config.wheel_width_range)
        friction = rng.uniform(*config.wheel_friction_range)
        effort_limit = rng.uniform(*config.wheel_effort_range)
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
                mass=max(0.35, 8.0 * radius * width),
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
                damping=0.05,
                friction=0.01,
                armature=0.005,
            )
        )
        actuators.append(
            ActuatorBlueprint(
                name=f"{joint_name}_actuator",
                joint_name=joint_name,
                semantic_slot=semantic_slot,
                kp=0.0,
                kd=0.15,
                effort_limit=effort_limit,
            )
        )
        return WheelBlueprint(
            link_name=link_name,
            joint_name=joint_name,
            semantic_slot=semantic_slot,
            radius=radius,
            width=width,
            friction=friction,
            effort_limit=effort_limit,
        )


# Descriptive aliases for callers that do not want to depend on the class name
# used in the task title.
PaperFaithfulMorphologyGenerator = LocoFormerMorphologyGenerator
PaperFaithfulMorphologyGeneratorConfig = LocoFormerMorphologyGeneratorConfig
