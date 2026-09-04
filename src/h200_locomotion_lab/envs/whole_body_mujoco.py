"""Small vectorized MuJoCo backend for the procedural whole-body task.

This backend is intentionally a reference implementation: one compiled
topology per shard and one ``MjData`` per environment.  MJLab can replace the
shard internals later without changing the 45D/193D rollout contract.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from h200_locomotion_lab.core.whole_body import WholeBodyStep
from h200_locomotion_lab.robots.motor_process import MotorProcess, MotorProcessConfig
from h200_locomotion_lab.robots.procedural_morphology import (
    CANONICAL_ROOT_SITE_NAME,
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


@dataclass(frozen=True, slots=True)
class WholeBodyMuJoCoShardConfig:
    control_hz: float = 50.0
    physics_hz: float = 500.0
    trial_seconds: float = 10.0
    context_trials: int = 3
    action_scale: float = 0.65
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

    @property
    def substeps(self) -> int:
        return round(self.physics_hz / self.control_hz)

    @property
    def trial_steps(self) -> int:
        return round(self.control_hz * self.trial_seconds)


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
        self.xml = compile_mjcf(blueprint, physical)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = tuple(mujoco.MjData(self.model) for _ in range(num_envs))
        self.embodiment = BoundEmbodiment.from_blueprint(blueprint, physical=physical)
        self._joint_qpos = tuple(
            int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)])
            for joint in blueprint.joints
        )
        self._joint_dof = tuple(
            int(self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)])
            for joint in blueprint.joints
        )
        self._actuator_ids = tuple(
            int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator.name))
            for actuator in blueprint.actuators
        )
        if any(index < 0 for index in (*self._joint_qpos, *self._joint_dof, *self._actuator_ids)):
            raise ValueError("compiled model is missing a generated joint or actuator")
        self._canonical_root_site_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, CANONICAL_ROOT_SITE_NAME)
        )
        self._canonical_root_site_id = (
            self._canonical_root_site_id if self._canonical_root_site_id >= 0 else None
        )
        self._canonical_stance_height: float | None = None
        self._foot_geoms = {
            f"{link.name}_footpad" for link in blueprint.links if link.foot
        }
        self.stance_solution: StanceSolution = solve_static_stance(
            self.model,
            self.data[0],
            blueprint,
            physical,
        )
        self.stance_solution.validate_for(blueprint, physical)
        self._commands = np.zeros((num_envs, 3), dtype=np.float64)
        self._last_action = np.zeros((num_envs, 45), dtype=np.float64)
        self._trial_step = np.zeros(num_envs, dtype=np.int64)
        self._trial_index = np.zeros(num_envs, dtype=np.int64)
        self._rngs = tuple(random.Random(self.config.seed + 1009 * index) for index in range(num_envs))
        baseline_strength = tuple(
            (physical.motor_strength.get(joint.semantic_slot, 1.0) if physical else 1.0)
            for joint in blueprint.joints
        )
        baseline_latency = tuple(
            (round((physical.delay_ms if physical else 0.0) * self.config.control_hz / 1000.0),)
            * len(blueprint.joints)
        )
        baseline_ema = tuple((physical.ema_alpha if physical else 1.0,) * len(blueprint.joints))
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
        pre_observation = self._batch_observation(trial_start=False)
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
        for env_id, data in enumerate(self.data):
            robot_action = self.embodiment.gather_action(action_array[env_id].tolist())
            processed = self._motor[env_id].process_action(tuple(robot_action), int(self._trial_step[env_id]))
            self._set_targets(data, processed)
            for _ in range(self.config.substeps):
                self.mujoco.mj_step(self.model, data)
            self._trial_step[env_id] += 1
            state = self._motor[env_id].state_at(int(self._trial_step[env_id]))
            motor_strength[env_id] = state.strength
            motor_latency[env_id] = state.extra_latency_steps
            motor_active[env_id] = any(event.active_at(int(self._trial_step[env_id])) for event in state.events)
            rewards[env_id], normalized_error[env_id], non_foot_contact[env_id] = self._reward(data, env_id)
            fall = self._is_fallen(data)
            fall_flags[env_id] = fall
            gravity = self._canonical_state(data).projected_gravity
            tilt[env_id] = math.atan2(math.sqrt(gravity[0] ** 2 + gravity[1] ** 2), max(1e-9, -gravity[2]))
            timeout = self._trial_step[env_id] >= self.config.trial_steps
            trial_done[env_id] = fall or timeout
            context_done[env_id] = trial_done[env_id] and (
                self._trial_index[env_id] + 1 >= self.config.context_trials
            )
        self._last_action = action_array
        # Keep the pre-reset observation for every vector element; consumers
        # can select it with ``trial_done`` and mux shards without losing
        # partial-done rows.
        final_observation = pre_observation
        for env_id in range(self.num_envs):
            if trial_done[env_id]:
                if context_done[env_id]:
                    self._trial_index[env_id] = 0
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
            self._motor[env_id].reset_context(self.config.seed + env_id + int(self._trial_index[env_id]) * 7919)
        else:
            self._motor[env_id].reset_trial()
        self._last_action[env_id] = 0.0

    def _set_targets(self, data: Any, robot_action: tuple[float, ...]) -> None:
        for value, actuator, actuator_id in zip(robot_action, self.blueprint.actuators, self._actuator_ids):
            midpoint = self.stance_solution.actuator_ctrl[actuator.semantic_slot]
            lower, upper = (
                float(self.model.actuator_ctrlrange[int(actuator_id), 0]),
                float(self.model.actuator_ctrlrange[int(actuator_id), 1]),
            )
            half_span = 0.5 * (upper - lower)
            target = midpoint + self.config.action_scale * float(value) * half_span
            data.ctrl[actuator_id] = min(upper, max(lower, target))

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
        root = self._canonical_state(data)
        return self.embodiment.encode_actor_observation(
            base_linear_velocity=root.local_linear_velocity,
            base_angular_velocity=root.local_angular_velocity,
            projected_gravity=root.projected_gravity,
            command=tuple(float(value) for value in self._commands[env_id]),
            joint_position=qpos,
            joint_velocity=qvel,
            previous_action=tuple(float(value) for value in self._last_action[env_id]),
            trial_start=float(trial_start),
        )

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
                "local_linear_velocity": tuple(float(value) for value in data.qvel[:3]),
                "local_angular_velocity": tuple(float(value) for value in data.qvel[3:6]),
                "projected_gravity": _projected_gravity(tuple(float(value) for value in data.qpos[3:7])),
            })()
        return read_canonical_root_state(self.model, data, self._canonical_root_site_id)

    def _non_foot_contact(self, data: Any) -> float:
        floor_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, "floor")
        contacts = 0
        for index in range(data.ncon):
            contact = data.contact[index]
            if contact.geom1 != floor_id and contact.geom2 != floor_id:
                continue
            other = contact.geom2 if contact.geom1 == floor_id else contact.geom1
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            if name not in self._foot_geoms:
                contacts += 1
        return float(contacts > 0)

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
