"""Small vectorized MuJoCo backend for the procedural whole-body task.

This backend is intentionally a reference implementation: one compiled
topology per shard and one ``MjData`` per environment.  MJLab can replace the
shard internals later without changing the 45D/193D rollout contract.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from h200_locomotion_lab.core.whole_body import WholeBodyStep
from h200_locomotion_lab.robots.motor_process import MotorProcess, MotorProcessConfig
from h200_locomotion_lab.robots.procedural_morphology import (
    CANONICAL_ROOT_SITE_NAME,
    MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_PROFILE_VERSION,
    MorphologyBlueprint,
    PhysicalParams,
    compile_mjcf,
    read_canonical_root_state,
)
from h200_locomotion_lab.robots.whole_body_adapter import BoundEmbodiment
from h200_locomotion_lab.robots.whole_body_stance import StanceSolution, solve_static_stance
from h200_locomotion_lab.tasks.whole_body_contract import (
    WholeBodyTaskConfig,
    make_whole_body_task_spec,
)


def phase_from_trial_step(trial_step: Any, control_hz: float, period_s: float) -> Any:
    """Return the shared per-trial gait phase used by reward and observation."""
    return ((trial_step / control_hz) % period_s) / period_s


@dataclass(frozen=True, slots=True)
class WholeBodyMuJoCoShardConfig:
    control_hz: float = 50.0
    physics_hz: float = 500.0
    trial_seconds: float = 10.0
    context_trials: int = 3
    action_scale: float = 0.65
    action_amplitude_by_slot: Mapping[str, float] | None = None
    action_residual_bounds_by_slot: Mapping[str, tuple[float, float]] | None = None
    observation_joint_reference_by_slot: Mapping[str, float] | None = None
    observation_joint_velocity_scale: float = 1.0
    observation_base_angular_velocity_scale: float = 1.0
    observation_phase: bool = False
    phase_period_s: float = 0.8
    logical_foot_groups: Mapping[str, tuple[str, ...]] | None = None
    logical_foot_reference_sites: Mapping[str, str] | None = None
    command_vx_range: tuple[float, float] = (0.2, 0.8)
    command_vy_range: tuple[float, float] = (-0.2, 0.2)
    command_yaw_range: tuple[float, float] = (-0.5, 0.5)
    fall_height_fraction: float = 0.35
    upright_threshold: float = 0.35
    seed: int = 0

    def __post_init__(self) -> None:
        if self.control_hz <= 0.0 or self.physics_hz <= 0.0:
            raise ValueError("control_hz and physics_hz must be positive")
        ratio = self.physics_hz / self.control_hz
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("physics_hz/control_hz must be an integer")
        if self.trial_seconds <= 0.0 or self.context_trials <= 0:
            raise ValueError("trial_seconds and context_trials must be positive")
        if self.action_scale <= 0.0:
            raise ValueError("action_scale must be positive")
        if self.action_amplitude_by_slot is not None:
            for slot, amplitude in self.action_amplitude_by_slot.items():
                if not isinstance(slot, str) or not slot:
                    raise ValueError("action amplitude slots must be non-empty strings")
                if not math.isfinite(float(amplitude)) or float(amplitude) <= 0.0:
                    raise ValueError("action amplitudes must be finite positive radians")
        if self.action_amplitude_by_slot is not None and self.action_residual_bounds_by_slot is not None:
            raise ValueError("action amplitude and residual bounds are mutually exclusive")
        if self.action_residual_bounds_by_slot is not None:
            for slot, bounds in self.action_residual_bounds_by_slot.items():
                if not isinstance(slot, str) or not slot or len(bounds) != 2:
                    raise ValueError("action residual bound slots must be non-empty pairs")
                if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in bounds):
                    raise ValueError("action residual bounds must be finite positive radians")
        if self.observation_joint_reference_by_slot is not None and any(
            not isinstance(slot, str) or not slot or not math.isfinite(float(value))
            for slot, value in self.observation_joint_reference_by_slot.items()
        ):
            raise ValueError("observation joint references must be finite values")
        if not math.isfinite(self.observation_joint_velocity_scale) or self.observation_joint_velocity_scale <= 0.0:
            raise ValueError("observation_joint_velocity_scale must be finite positive")
        if not math.isfinite(self.observation_base_angular_velocity_scale) or self.observation_base_angular_velocity_scale <= 0.0:
            raise ValueError("observation_base_angular_velocity_scale must be finite positive")
        if not math.isfinite(self.phase_period_s) or self.phase_period_s <= 0.0:
            raise ValueError("phase_period_s must be finite positive")
        if self.logical_foot_groups is not None:
            if not self.logical_foot_groups:
                raise ValueError("logical_foot_groups must not be empty")
            for foot, geoms in self.logical_foot_groups.items():
                if not isinstance(foot, str) or not foot or not geoms:
                    raise ValueError("logical foot groups require non-empty names and geoms")
                if any(not isinstance(geom, str) or not geom for geom in geoms):
                    raise ValueError("logical foot geom names must be non-empty strings")
        if self.logical_foot_reference_sites is not None:
            if self.logical_foot_groups is None:
                raise ValueError("logical foot reference sites require logical foot groups")
            if set(self.logical_foot_reference_sites) != set(self.logical_foot_groups):
                raise ValueError("logical foot reference sites must exactly cover logical feet")
            if any(
                not isinstance(site, str) or not site
                for site in self.logical_foot_reference_sites.values()
            ):
                raise ValueError("logical foot reference site names must be non-empty strings")

    @property
    def substeps(self) -> int:
        return round(self.physics_hz / self.control_hz)

    @property
    def trial_steps(self) -> int:
        return round(self.control_hz * self.trial_seconds)


def _motor_process_baselines(
    blueprint: MorphologyBlueprint,
    physical: PhysicalParams | None,
    control_hz: float,
) -> tuple[tuple[float, ...], tuple[int, ...], tuple[float, ...]]:
    """Return runtime baselines, keeping compiled v2 effort authoritative."""
    v2 = blueprint.profile_version == MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_PROFILE_VERSION
    if v2:
        strength = (1.0,) * len(blueprint.joints)
    else:
        strength = tuple(
            physical.motor_strength.get(joint.semantic_slot, 1.0)
            if physical
            else 1.0
            for joint in blueprint.joints
        )
    delay_steps = round((physical.delay_ms if physical else 0.0) * control_hz / 1000.0)
    latency = (delay_steps,) * len(blueprint.joints)
    ema = ((physical.ema_alpha if physical else 1.0),) * len(blueprint.joints)
    return strength, latency, ema


def _validate_precompiled_model(
    mujoco: Any,
    np: Any,
    model: Any,
    blueprint: MorphologyBlueprint,
    physical: PhysicalParams | None,
    joint_ids: tuple[int, ...],
    actuator_ids: tuple[int, ...],
    canonical_root_site_id: int,
) -> None:
    """Reject precompiled XML that does not implement the bound blueprint."""
    if model.nu != len(blueprint.actuators) or model.njnt != len(blueprint.joints) + 1:
        raise ValueError("model_xml joint or actuator accounting does not match blueprint")
    if int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("model_xml must have one free root joint")

    global_scale = physical.global_scale if physical else 1.0
    limit_scales = physical.joint_limit_scales if physical else {}
    nominal_offsets = physical.nominal_offsets if physical else {}
    kp_scales = physical.kp_scales if physical else {}
    kd_scales = physical.kd_scales if physical else {}
    joints_by_slot = {joint.semantic_slot: joint for joint in blueprint.joints}
    wheels_by_joint = {wheel.joint_name: wheel for wheel in blueprint.wheel_specs}
    joint_ids_by_slot = {
        joint.semantic_slot: joint_id for joint, joint_id in zip(blueprint.joints, joint_ids)
    }
    for joint, joint_id in zip(blueprint.joints, joint_ids):
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"model_xml joint type mismatch: {joint.name}")
        if not np.allclose(model.jnt_axis[joint_id], joint.axis, rtol=0.0, atol=1e-12):
            raise ValueError(f"model_xml joint axis mismatch: {joint.name}")
        if joint.name not in wheels_by_joint:
            scale = limit_scales.get(joint.semantic_slot, 1.0)
            offset = nominal_offsets.get(joint.semantic_slot, 0.0)
            expected_range = np.asarray(
                (
                    joint.joint_range[0] * scale + offset,
                    joint.joint_range[1] * scale + offset,
                )
            )
            if not bool(model.jnt_limited[joint_id]) or not np.allclose(
                model.jnt_range[joint_id], expected_range, rtol=0.0, atol=1e-8
            ):
                raise ValueError(f"model_xml joint range mismatch: {joint.name}")

    for actuator, actuator_id in zip(blueprint.actuators, actuator_ids):
        joint = joints_by_slot[actuator.semantic_slot]
        expected_joint_id = joint_ids_by_slot[actuator.semantic_slot]
        if (
            int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT)
            or int(model.actuator_trnid[actuator_id, 0]) != expected_joint_id
        ):
            raise ValueError(f"model_xml actuator transmission mismatch: {actuator.name}")
        if not np.allclose(
            model.actuator_gear[actuator_id],
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"model_xml actuator gear mismatch: {actuator.name}")
        if joint.name in wheels_by_joint:
            continue
        expected_kp = actuator.kp * kp_scales.get(actuator.semantic_slot, 1.0)
        expected_kd = actuator.kd * kd_scales.get(actuator.semantic_slot, 1.0)
        expected_ctrlrange = model.jnt_range[expected_joint_id]
        position_semantics = bool(
            int(model.actuator_gaintype[actuator_id]) == int(mujoco.mjtGain.mjGAIN_FIXED)
            and int(model.actuator_biastype[actuator_id]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
            and bool(model.actuator_ctrllimited[actuator_id])
            and bool(model.actuator_forcelimited[actuator_id])
            and math.isclose(
                float(model.actuator_gainprm[actuator_id, 0]),
                expected_kp,
                rel_tol=1e-5,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(model.actuator_biasprm[actuator_id, 1]),
                -expected_kp,
                rel_tol=1e-5,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(model.actuator_biasprm[actuator_id, 2]),
                -expected_kd,
                rel_tol=1e-5,
                abs_tol=1e-8,
            )
            and np.allclose(
                model.actuator_ctrlrange[actuator_id],
                expected_ctrlrange,
                rtol=0.0,
                atol=1e-8,
            )
            and np.isfinite(model.actuator_forcerange[actuator_id]).all()
            and float(model.actuator_forcerange[actuator_id, 0]) < 0.0
            and float(model.actuator_forcerange[actuator_id, 1]) > 0.0
        )
        if not position_semantics:
            raise ValueError(f"model_xml position actuator semantics mismatch: {actuator.name}")

    canonical = blueprint.profile_metadata.get("canonical_root_frame")
    if not isinstance(canonical, Mapping):
        raise TypeError("model_xml blueprint is missing canonical root metadata")
    expected_body_id = int(
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            str(canonical["site_body_link"]),
        )
    )
    transform = canonical["anchor_body_from_canonical"]
    expected_position = np.asarray(canonical["origin"], dtype=float) * global_scale
    expected_quaternion = np.asarray(transform["quaternion_wxyz"], dtype=float)
    if (
        expected_body_id < 0
        or int(model.site_bodyid[canonical_root_site_id]) != expected_body_id
        or not np.allclose(
            model.site_pos[canonical_root_site_id], expected_position, rtol=0.0, atol=1e-8
        )
        or not np.allclose(
            model.site_quat[canonical_root_site_id],
            expected_quaternion,
            rtol=0.0,
            atol=1e-10,
        )
    ):
        raise ValueError("model_xml canonical root frame mismatch")


class WholeBodyMuJoCoShard:
    """A fixed-topology shard implementing ``WholeBodyShard``."""

    def __init__(
        self,
        blueprint: MorphologyBlueprint,
        *,
        physical: PhysicalParams | None = None,
        num_envs: int = 1,
        config: WholeBodyMuJoCoShardConfig | None = None,
        motor_config: MotorProcessConfig | None = None,
        model_xml: str | None = None,
        model_xml_sha256: str | None = None,
        stance_solution: StanceSolution | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        try:
            import mujoco  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional simulator dependency
            raise RuntimeError("MuJoCo and NumPy are required for WholeBodyMuJoCoShard") from exc
        self.mujoco = mujoco
        self.np = np
        self.blueprint = blueprint
        self.physical = physical
        self.config = config or WholeBodyMuJoCoShardConfig()
        self.num_envs = num_envs
        self.spec = make_whole_body_task_spec(
            WholeBodyTaskConfig(
                control_hz=self.config.control_hz,
                trial_seconds=self.config.trial_seconds,
            )
        )
        if model_xml is None and model_xml_sha256 is not None:
            raise ValueError("model_xml SHA bindings require model_xml")
        if (
            model_xml is None
            and stance_solution is not None
            and stance_solution.model_xml_sha256 is not None
        ):
            raise ValueError("XML-bound stance solution requires model_xml")
        if model_xml is not None:
            actual_xml_sha256 = hashlib.sha256(model_xml.encode("utf-8")).hexdigest()
            if model_xml_sha256 != actual_xml_sha256:
                raise ValueError("model_xml SHA mismatch")
            if stance_solution is None:
                raise ValueError("precompiled model requires an XML-bound stance solution")
        else:
            actual_xml_sha256 = None
        self.xml = compile_mjcf(blueprint, physical) if model_xml is None else model_xml
        self.model_xml_sha256 = actual_xml_sha256
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        if (
            model_xml is not None
            and abs(float(self.model.opt.timestep) - 1.0 / self.config.physics_hz) > 1e-12
        ):
            raise ValueError("model_xml timestep does not match physics_hz")
        self.data = tuple(mujoco.MjData(self.model) for _ in range(num_envs))
        self.embodiment = BoundEmbodiment.from_blueprint(blueprint, physical=physical)
        joint_ids = tuple(
            int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint.name))
            for joint in blueprint.joints
        )
        actuator_ids = tuple(
            int(
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    actuator.name,
                )
            )
            for actuator in blueprint.actuators
        )
        if any(index < 0 for index in (*joint_ids, *actuator_ids)):
            raise ValueError("compiled model is missing a generated joint or actuator")
        self._joint_qpos = tuple(
            int(self.model.jnt_qposadr[joint_id]) for joint_id in joint_ids
        )
        self._joint_dof = tuple(
            int(self.model.jnt_dofadr[joint_id]) for joint_id in joint_ids
        )
        self._actuator_ids = actuator_ids
        self._action_amplitude_by_slot: dict[str, float] | None = None
        self._action_residual_bounds_by_slot: dict[str, tuple[float, float]] | None = None
        configured_slots = {actuator.semantic_slot for actuator in blueprint.actuators}
        for name, mapping in (("action_amplitude_by_slot", self.config.action_amplitude_by_slot),
                              ("action_residual_bounds_by_slot", self.config.action_residual_bounds_by_slot)):
            if mapping is not None and set(mapping) != configured_slots:
                raise ValueError(f"{name} must exactly cover active actuator slots")
        if self.config.action_amplitude_by_slot is not None:
            self._action_amplitude_by_slot = {
                slot: float(amplitude)
                for slot, amplitude in self.config.action_amplitude_by_slot.items()
            }
        if self.config.action_residual_bounds_by_slot is not None:
            self._action_residual_bounds_by_slot = {
                slot: (float(bounds[0]), float(bounds[1]))
                for slot, bounds in self.config.action_residual_bounds_by_slot.items()
            }
        if self.config.observation_joint_reference_by_slot is not None:
            if set(self.config.observation_joint_reference_by_slot) != configured_slots:
                raise ValueError("observation_joint_reference_by_slot must exactly cover active joint slots")
            self._observation_joint_reference_by_slot = {
                slot: float(value) for slot, value in self.config.observation_joint_reference_by_slot.items()
            }
        else:
            self._observation_joint_reference_by_slot = None
        self._canonical_root_site_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, CANONICAL_ROOT_SITE_NAME)
        )
        self._canonical_root_site_id = (
            self._canonical_root_site_id if self._canonical_root_site_id >= 0 else None
        )
        if model_xml is not None and self._canonical_root_site_id is None:
            raise ValueError("model_xml is missing the canonical root site")
        if model_xml is not None:
            _validate_precompiled_model(
                mujoco,
                np,
                self.model,
                blueprint,
                physical,
                joint_ids,
                actuator_ids,
                self._canonical_root_site_id,
            )
        self._canonical_stance_height: float | None = None
        if self.config.logical_foot_groups is None:
            foot_groups = {
                link.name: (f"{link.name}_footpad",) for link in blueprint.links if link.foot
            }
        else:
            foot_groups = {
                str(foot): tuple(str(geom) for geom in geoms)
                for foot, geoms in self.config.logical_foot_groups.items()
            }
        self._logical_foot_names = tuple(sorted(foot_groups))
        self._logical_foot_groups = {
            foot: tuple(foot_groups[foot]) for foot in self._logical_foot_names
        }
        self._foot_geoms = {
            geom for geoms in self._logical_foot_groups.values() for geom in geoms
        }
        self._foot_geom_ids_by_foot = tuple(
            tuple(
                int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name))
                for name in self._logical_foot_groups[foot]
            )
            for foot in self._logical_foot_names
        )
        self._foot_geom_ids = tuple(
            geom_id for group in self._foot_geom_ids_by_foot for geom_id in group
        )
        if any(geom_id < 0 for geom_id in self._foot_geom_ids):
            raise ValueError("compiled model is missing a generated footpad geom")
        site_names = self.config.logical_foot_reference_sites or {}
        self._foot_reference_site_ids = tuple(
            int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_names[foot]))
            if foot in site_names
            else -1
            for foot in self._logical_foot_names
        )
        if site_names and any(site_id < 0 for site_id in self._foot_reference_site_ids):
            raise ValueError("compiled model is missing a logical foot reference site")
        self._floor_geom_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        )
        if self._floor_geom_id < 0:
            raise ValueError("compiled model is missing floor geom")
        self.stance_solution: StanceSolution = (
            solve_static_stance(self.model, self.data[0], blueprint, physical)
            if stance_solution is None
            else stance_solution
        )
        self.stance_solution.validate_for(
            blueprint,
            physical,
            expected_model_xml_sha256=actual_xml_sha256,
        )
        self._commands = np.zeros((num_envs, 3), dtype=np.float64)
        self._last_action = np.zeros((num_envs, 45), dtype=np.float64)
        self._foot_air_time = np.zeros((num_envs, len(self._logical_foot_names)), dtype=np.float64)
        self._foot_contact_previous = np.zeros(
            (num_envs, len(self._logical_foot_names)), dtype=bool
        )
        self._trial_step = np.zeros(num_envs, dtype=np.int64)
        self._trial_index = np.zeros(num_envs, dtype=np.int64)
        self._context_index = np.zeros(num_envs, dtype=np.int64)
        self._rngs = tuple(random.Random(self.config.seed + 1009 * index) for index in range(num_envs))
        baseline_strength, baseline_latency, baseline_ema = _motor_process_baselines(
            blueprint, physical, self.config.control_hz
        )
        process_config = motor_config or MotorProcessConfig(control_hz=self.config.control_hz)
        self._motor = tuple(
            MotorProcess(
                tuple(joint.semantic_slot for joint in blueprint.joints),
                config=process_config,
                baseline_strength=baseline_strength,
                baseline_latency_steps=baseline_latency,
                baseline_ema_alpha=baseline_ema,
            )
            for _ in range(num_envs)
        )
        self.reset()

    @property
    def active_action_mask(self) -> Any:
        return self.np.broadcast_to(
            self.np.asarray(self.embodiment.action_mask, dtype=bool),
            (self.num_envs, 45),
        ).copy()

    def reset(self) -> Any:
        for env_id in range(self.num_envs):
            self._reset_env(env_id, context=True)
        return self._batch_observation(trial_start=True)

    def step(self, action: Any) -> WholeBodyStep:
        action_array = self.np.asarray(action, dtype=self.np.float64)
        if action_array.shape != (self.num_envs, 45):
            raise ValueError(f"action must have shape ({self.num_envs}, 45), got {action_array.shape}")
        action_array = action_array.clip(-1.0, 1.0)
        action_array *= self.np.asarray(self.embodiment.action_mask, dtype=self.np.float64)
        previous_action = self._last_action.copy()
        rewards = self.np.zeros(self.num_envs, dtype=self.np.float64)
        trial_done = self.np.zeros(self.num_envs, dtype=bool)
        context_done = self.np.zeros(self.num_envs, dtype=bool)
        non_foot_contact = self.np.zeros(self.num_envs, dtype=self.np.float64)
        normalized_error = self.np.zeros(self.num_envs, dtype=self.np.float64)
        motor_active = self.np.zeros(self.num_envs, dtype=bool)
        fall_flags = self.np.zeros(self.num_envs, dtype=bool)
        tilt = self.np.zeros(self.num_envs, dtype=self.np.float64)
        motor_strength = self.np.ones((self.num_envs, len(self.blueprint.joints)), dtype=self.np.float64)
        motor_latency = self.np.zeros_like(motor_strength, dtype=self.np.int64)
        post_root_position = self.np.zeros((self.num_envs, 3), dtype=self.np.float64)
        post_root_quaternion = self.np.zeros((self.num_envs, 4), dtype=self.np.float64)
        post_root_linear_velocity = self.np.zeros((self.num_envs, 3), dtype=self.np.float64)
        post_root_angular_velocity = self.np.zeros((self.num_envs, 3), dtype=self.np.float64)
        post_projected_gravity = self.np.zeros((self.num_envs, 3), dtype=self.np.float64)
        joint_position = self.np.zeros((self.num_envs, 45), dtype=self.np.float64)
        joint_velocity = self.np.zeros((self.num_envs, 45), dtype=self.np.float64)
        foot_count = len(self._logical_foot_names)
        foot_contact = self.np.zeros((self.num_envs, foot_count), dtype=bool)
        foot_normal_force = self.np.zeros((self.num_envs, foot_count), dtype=self.np.float64)
        foot_height = self.np.zeros((self.num_envs, foot_count), dtype=self.np.float64)
        foot_planar_speed = self.np.zeros((self.num_envs, foot_count), dtype=self.np.float64)
        foot_vertical_speed = self.np.zeros((self.num_envs, foot_count), dtype=self.np.float64)
        foot_air_time = self.np.zeros((self.num_envs, foot_count), dtype=self.np.float64)
        touchdown = self.np.zeros((self.num_envs, foot_count), dtype=bool)
        target_would_clamp = self.np.zeros((self.num_envs, 45), dtype=bool)
        actual_clamp = self.np.zeros((self.num_envs, 45), dtype=bool)
        unclamped_target = self.np.zeros((self.num_envs, 45), dtype=self.np.float64)
        ctrl_target = self.np.zeros((self.num_envs, 45), dtype=self.np.float64)
        for env_id, data in enumerate(self.data):
            robot_action = self.embodiment.gather_action(action_array[env_id].tolist())
            processed = self._motor[env_id].process_action(tuple(robot_action), int(self._trial_step[env_id]))
            clamp_diagnostics = self._set_targets(data, processed)
            for name, values in clamp_diagnostics.items():
                if name == "target_would_clamp":
                    target_would_clamp[env_id, self.embodiment.mapping.selector] = values
                elif name == "actual_clamp":
                    actual_clamp[env_id, self.embodiment.mapping.selector] = values
                elif name == "unclamped_target":
                    unclamped_target[env_id, self.embodiment.mapping.selector] = values
                elif name == "ctrl_target":
                    ctrl_target[env_id, self.embodiment.mapping.selector] = values
            for _ in range(self.config.substeps):
                self.mujoco.mj_step(self.model, data)
            self._trial_step[env_id] += 1
            post_root = self._canonical_state(data)
            post_root_position[env_id] = post_root.world_position
            post_root_quaternion[env_id] = post_root.world_quaternion_wxyz
            post_root_linear_velocity[env_id] = post_root.local_linear_velocity
            post_root_angular_velocity[env_id] = post_root.local_angular_velocity
            post_projected_gravity[env_id] = post_root.projected_gravity
            joint_position[env_id] = self.embodiment.scatter_joint_values(
                tuple(float(data.qpos[address]) for address in self._joint_qpos)
            )
            joint_velocity[env_id] = self.embodiment.scatter_joint_values(
                tuple(float(data.qvel[address]) for address in self._joint_dof)
            )
            (
                foot_contact[env_id],
                foot_normal_force[env_id],
                foot_height[env_id],
                foot_planar_speed[env_id],
                foot_vertical_speed[env_id],
                foot_air_time[env_id],
                touchdown[env_id],
            ) = self._post_step_foot_metrics(data, env_id)
            state = self._motor[env_id].state_at(int(self._trial_step[env_id]))
            motor_strength[env_id] = state.strength
            motor_latency[env_id] = state.extra_latency_steps
            motor_active[env_id] = any(event.active_at(int(self._trial_step[env_id])) for event in state.events)
            rewards[env_id], normalized_error[env_id], non_foot_contact[env_id] = self._reward(data, env_id)
            fall = self._is_fallen(data)
            fall_flags[env_id] = fall
            gravity = post_root.projected_gravity
            tilt[env_id] = math.atan2(math.sqrt(gravity[0] ** 2 + gravity[1] ** 2), max(1e-9, -gravity[2]))
            timeout = self._trial_step[env_id] >= self.config.trial_steps
            trial_done[env_id] = fall or timeout
            context_done[env_id] = trial_done[env_id] and (
                self._trial_index[env_id] + 1 >= self.config.context_trials
            )
        self._last_action = action_array
        # Preserve the post-action, pre-reset state for bootstrapping.  The
        # regular observation below is allowed to contain a fresh reset row.
        final_observation = self._batch_observation(trial_start=False)
        for env_id in range(self.num_envs):
            if trial_done[env_id]:
                if context_done[env_id]:
                    self._trial_index[env_id] = 0
                    self._context_index[env_id] += 1
                    self._reset_env(env_id, context=True)
                else:
                    self._trial_index[env_id] += 1
                    self._reset_env(env_id, context=False)
        observation = self._batch_observation(trial_start=trial_done)
        return WholeBodyStep(
            actor_observation=observation,
            critic_observation=observation.copy(),
            reward=rewards,
            trial_done=trial_done,
            context_done=context_done,
            active_action_mask=self.np.broadcast_to(
                self.np.asarray(self.embodiment.action_mask, dtype=bool), (self.num_envs, 45)
            ).copy(),
            metrics={
                "normalized_velocity_error": normalized_error,
                "non_foot_contact_fraction": non_foot_contact,
                "motor_event_active": motor_active,
                "fall": fall_flags,
                "tilt": tilt,
                "motor_strength": motor_strength,
                "motor_latency_steps": motor_latency,
                "previous_action": previous_action,
                "joint_position": joint_position,
                "joint_velocity": joint_velocity,
                "foot_contact": foot_contact,
                "foot_normal_force": foot_normal_force,
                "foot_height": foot_height,
                "foot_planar_speed": foot_planar_speed,
                "foot_vertical_speed": foot_vertical_speed,
                "foot_air_time": foot_air_time,
                "touchdown": touchdown,
                "target_would_clamp": target_would_clamp,
                "actual_clamp": actual_clamp,
                "unclamped_target": unclamped_target,
                "ctrl_target": ctrl_target,
                "post_step_pre_reset_world_position": post_root_position,
                "post_step_pre_reset_world_quaternion_wxyz": post_root_quaternion,
                "post_step_pre_reset_local_linear_velocity": post_root_linear_velocity,
                "post_step_pre_reset_local_angular_velocity": post_root_angular_velocity,
                "post_step_pre_reset_projected_gravity": post_projected_gravity,
            },
            final_observation=final_observation,
        )

    def _reset_env(self, env_id: int, *, context: bool) -> None:
        data = self.data[env_id]
        self.mujoco.mj_resetData(self.model, data)
        data.qpos[0] = self.stance_solution.root_xy[0]
        data.qpos[1] = self.stance_solution.root_xy[1]
        data.qpos[2] = self.stance_solution.base_height
        data.qpos[3:7] = self.stance_solution.root_quat
        for joint, qpos_address in zip(self.blueprint.joints, self._joint_qpos):
            data.qpos[qpos_address] = self.stance_solution.joint_qpos[joint.semantic_slot]
        for actuator, actuator_id in zip(self.blueprint.actuators, self._actuator_ids):
            data.ctrl[int(actuator_id)] = self.stance_solution.actuator_ctrl[actuator.semantic_slot]
        self.mujoco.mj_forward(self.model, data)
        if self._canonical_root_site_id is not None and self._canonical_stance_height is None:
            self._canonical_stance_height = float(data.site_xpos[self._canonical_root_site_id, 2])
        self._commands[env_id] = self._sample_command(self._rngs[env_id])
        self._trial_step[env_id] = 0
        if context:
            self._motor[env_id].reset_context(
                self.config.seed + env_id + int(self._context_index[env_id]) * 7919
            )
        else:
            self._motor[env_id].reset_trial()
        self._last_action[env_id] = 0.0
        self._foot_air_time[env_id] = 0.0
        self._foot_contact_previous[env_id] = False

    def _set_targets(self, data: Any, robot_action: tuple[float, ...]) -> dict[str, Any]:
        would_clamp = self.np.zeros(len(self.blueprint.actuators), dtype=bool)
        actual_clamp = self.np.zeros(len(self.blueprint.actuators), dtype=bool)
        unclamped_target = self.np.zeros(len(self.blueprint.actuators), dtype=self.np.float64)
        ctrl_target = self.np.zeros(len(self.blueprint.actuators), dtype=self.np.float64)
        for index, (value, actuator, actuator_id) in enumerate(
            zip(robot_action, self.blueprint.actuators, self._actuator_ids)
        ):
            midpoint = self.stance_solution.actuator_ctrl[actuator.semantic_slot]
            lower, upper = (
                float(self.model.actuator_ctrlrange[int(actuator_id), 0]),
                float(self.model.actuator_ctrlrange[int(actuator_id), 1]),
            )
            if self._action_residual_bounds_by_slot is not None:
                negative, positive = self._action_residual_bounds_by_slot[actuator.semantic_slot]
                target = midpoint + float(value) * (negative if float(value) < 0.0 else positive)
            elif self._action_amplitude_by_slot is None:
                half_span = 0.5 * (upper - lower)
                target = midpoint + self.config.action_scale * float(value) * half_span
            else:
                target = (
                    midpoint
                    + float(value) * self._action_amplitude_by_slot[actuator.semantic_slot]
                )
            would_clamp[index] = target < lower or target > upper
            clamped = min(upper, max(lower, target))
            actual_clamp[index] = clamped != target
            data.ctrl[actuator_id] = clamped
            unclamped_target[index] = target
            ctrl_target[index] = clamped
        return {
            "target_would_clamp": would_clamp,
            "actual_clamp": actual_clamp,
            "unclamped_target": unclamped_target,
            "ctrl_target": ctrl_target,
        }

    def _joint_limits(self, joint: Any) -> tuple[float, float]:
        scale = self.physical.joint_limit_scales.get(joint.semantic_slot, 1.0) if self.physical else 1.0
        offset = self.physical.nominal_offsets.get(joint.semantic_slot, 0.0) if self.physical else 0.0
        lower, upper = joint.joint_range
        return lower * scale + offset, upper * scale + offset

    def _batch_observation(self, *, trial_start: Any) -> Any:
        rows = []
        for env_id, data in enumerate(self.data):
            rows.append(self._observation(data, env_id, bool(trial_start[env_id]) if hasattr(trial_start, "__len__") else trial_start))
        return self.np.asarray(rows, dtype=self.np.float32)

    def _observation(self, data: Any, env_id: int, trial_start: bool) -> tuple[float, ...]:
        qpos = tuple(float(data.qpos[address]) for address in self._joint_qpos)
        qvel = tuple(float(data.qvel[address]) for address in self._joint_dof)
        if self._observation_joint_reference_by_slot is not None:
            qpos = tuple(
                value - self._observation_joint_reference_by_slot[joint.semantic_slot]
                for value, joint in zip(qpos, self.blueprint.joints)
            )
        qvel = tuple(value * self.config.observation_joint_velocity_scale for value in qvel)
        root = self._canonical_state(data)
        observation = self.embodiment.encode_actor_observation(
            base_linear_velocity=root.local_linear_velocity,
            base_angular_velocity=tuple(
                value * self.config.observation_base_angular_velocity_scale
                for value in root.local_angular_velocity
            ),
            projected_gravity=root.projected_gravity,
            command=tuple(float(value) for value in self._commands[env_id]),
            joint_position=qpos,
            joint_velocity=qvel,
            previous_action=tuple(float(value) for value in self._last_action[env_id]),
            trial_start=float(trial_start),
        )
        if self.config.observation_phase:
            phase = phase_from_trial_step(
                self._trial_step[env_id], self.config.control_hz, self.config.phase_period_s
            )
            observation += (math.sin(2.0 * math.pi * float(phase)), math.cos(2.0 * math.pi * float(phase)))
        return observation

    def _reward(self, data: Any, env_id: int) -> tuple[float, float, float]:
        command = self._commands[env_id]
        root = self._canonical_state(data)
        error = math.sqrt(float((root.local_linear_velocity[0] - command[0]) ** 2 + (root.local_linear_velocity[1] - command[1]) ** 2))
        yaw_error = abs(float(root.local_angular_velocity[2] - command[2]))
        gravity = root.projected_gravity
        upright = max(0.0, min(1.0, -gravity[2]))
        non_foot = self._non_foot_contact(data)
        reward = math.exp(-2.0 * error) + 0.25 * math.exp(-yaw_error) + 0.25 * upright
        reward -= 0.10 * non_foot
        return reward, error / (1.0 + abs(float(command[0]))), non_foot

    def _is_fallen(self, data: Any) -> bool:
        root = self._canonical_state(data)
        gravity = root.projected_gravity
        height = root.world_position[2] if root is not None else float(data.qpos[2])
        return bool(
            height < self._fall_height_threshold()
            or -gravity[2] < self.config.upright_threshold
            or not all(math.isfinite(float(value)) for value in data.qpos)
        )

    def _fall_height_threshold(self) -> float:
        if self._canonical_stance_height is not None:
            return self._canonical_stance_height * self.config.fall_height_fraction
        scale = self.physical.global_scale if self.physical else 1.0
        return self.blueprint.nominal_height * scale * self.config.fall_height_fraction

    def _canonical_state(self, data: Any) -> Any:
        if self._canonical_root_site_id is None:
            return type("LegacyRoot", (), {
                "world_position": (float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])),
                "world_quaternion_wxyz": tuple(float(value) for value in data.qpos[3:7]),
                "local_linear_velocity": tuple(float(value) for value in data.qvel[:3]),
                "local_angular_velocity": tuple(float(value) for value in data.qvel[3:6]),
                "projected_gravity": _projected_gravity(tuple(float(value) for value in data.qpos[3:7])),
            })()
        return read_canonical_root_state(self.model, data, self._canonical_root_site_id)

    def _non_foot_contact(self, data: Any) -> float:
        contacts = 0
        for index in range(data.ncon):
            contact = data.contact[index]
            if contact.geom1 != self._floor_geom_id and contact.geom2 != self._floor_geom_id:
                continue
            other = contact.geom2 if contact.geom1 == self._floor_geom_id else contact.geom1
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            if name not in self._foot_geoms:
                contacts += 1
        return float(contacts > 0)

    def _foot_contacts(self, data: Any) -> tuple[Any, Any]:
        foot_index_by_id = {
            int(geom_id): foot_index
            for foot_index, group in enumerate(self._foot_geom_ids_by_foot)
            for geom_id in group
        }
        contact = self.np.zeros(len(self._logical_foot_names), dtype=bool)
        normal_force = self.np.zeros(len(self._logical_foot_names), dtype=self.np.float64)
        for index in range(data.ncon):
            pair = data.contact[index]
            if pair.geom1 == self._floor_geom_id and pair.geom2 in foot_index_by_id:
                foot_index = foot_index_by_id[int(pair.geom2)]
            elif pair.geom2 == self._floor_geom_id and pair.geom1 in foot_index_by_id:
                foot_index = foot_index_by_id[int(pair.geom1)]
            else:
                continue
            force = self.np.zeros(6, dtype=self.np.float64)
            self.mujoco.mj_contactForce(self.model, data, index, force)
            contact[foot_index] = True
            normal_force[foot_index] += max(0.0, float(force[0]))
        return contact, normal_force

    def _post_step_foot_metrics(self, data: Any, env_id: int) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        contact, normal_force = self._foot_contacts(data)
        height = self.np.asarray(
            [
                float(data.site_xpos[site_id, 2])
                if site_id >= 0
                else float(data.geom_xpos[self._foot_geom_ids_by_foot[index][0], 2])
                for index, site_id in enumerate(self._foot_reference_site_ids)
            ],
            dtype=self.np.float64,
        )
        velocity = self.np.zeros((len(self._logical_foot_names), 6), dtype=self.np.float64)
        for index, site_id in enumerate(self._foot_reference_site_ids):
            obj_type = self.mujoco.mjtObj.mjOBJ_SITE if site_id >= 0 else self.mujoco.mjtObj.mjOBJ_GEOM
            obj_id = site_id if site_id >= 0 else self._foot_geom_ids_by_foot[index][0]
            self.mujoco.mj_objectVelocity(
                self.model,
                data,
                obj_type,
                int(obj_id),
                velocity[index],
                0,
            )
        linear_velocity = velocity[:, 3:6]
        planar_speed = self.np.linalg.norm(linear_velocity[:, :2], axis=1)
        vertical_speed = linear_velocity[:, 2]
        touchdown = contact & ~self._foot_contact_previous[env_id]
        previous_air_time = self._foot_air_time[env_id].copy()
        next_air_time = self.np.where(
            contact,
            0.0,
            previous_air_time + 1.0 / self.config.control_hz,
        )
        metric_air_time = self.np.where(touchdown, previous_air_time, next_air_time)
        self._foot_air_time[env_id] = next_air_time
        self._foot_contact_previous[env_id] = contact
        return contact, normal_force, height, planar_speed, vertical_speed, metric_air_time, touchdown

    def _sample_command(self, rng: random.Random) -> tuple[float, float, float]:
        return (
            rng.uniform(*self.config.command_vx_range),
            rng.uniform(*self.config.command_vy_range),
            rng.uniform(*self.config.command_yaw_range),
        )


def _projected_gravity(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Rotate world gravity into the base frame for MuJoCo ``wxyz`` quaternions."""

    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-9:
        return (0.0, 0.0, -1.0)
    w, x, y, z = (value / norm for value in quaternion)
    return (
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    )


def ground_nominal_pose(mujoco: Any, model: Any, data: Any, *, margin: float = 0.0) -> float:
    """Translate the free base so the lowest collidable generated geom clears the floor.

    Primitive chains have variable lengths and per-link scale randomization, so
    one fixed nominal height leaves some feet floating and creates an avoidable
    impact before the first policy action.  This helper only changes the free
    root translation; joint targets and topology remain untouched.
    """

    minimum_bottom = float("inf")
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) == 0:
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            continue
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if geom_name == "floor":
            continue
        geom_type = int(model.geom_type[geom_id])
        size = model.geom_size[geom_id]
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            center_z = float(data.geom_xpos[geom_id, 2])
            rot = tuple(float(data.geom_xmat[geom_id, index]) for index in range(9))
            half = tuple(float(size[axis]) for axis in range(3))
            bottom = min(
                center_z + rot[6] * (sx * half[0]) + rot[7] * (sy * half[1]) - rot[8] * half[2]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
            )
            minimum_bottom = min(minimum_bottom, bottom)
            continue
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            extent = float(size[0] + size[1])
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            extent = float(size[1])
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            extent = float(size[0])
        else:
            extent = float(max(size))
        minimum_bottom = min(minimum_bottom, float(data.geom_xpos[geom_id, 2]) - extent)
    if minimum_bottom == float("inf"):
        return float(data.qpos[2])
    data.qpos[2] += margin - minimum_bottom
    mujoco.mj_forward(model, data)
    return float(data.qpos[2])
