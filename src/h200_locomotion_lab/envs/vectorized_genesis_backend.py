"""Vectorized Genesis backend for the G1 27DoF no-hand training asset.

This module intentionally does not import ``genesis`` or ``torch`` at module
import time. Local tests inject fake Genesis/Torch-like objects; H200 smoke runs
use the real packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from h200_locomotion_lab.robots import (
    G1_27DOF_NOHAND_ACTUATOR_ORDER,
    G1NoHandGenesisTrainingProfile,
    load_g1_27dof_nohand_profile,
)


ACTION_JOINT_GROUPS = ("all", "legs", "legs_waist")
LEG_JOINT_SUFFIXES = (
    "_hip_pitch_joint",
    "_hip_roll_joint",
    "_hip_yaw_joint",
    "_knee_joint",
    "_ankle_pitch_joint",
    "_ankle_roll_joint",
)


@dataclass(frozen=True, slots=True)
class VectorizedGenesisConfig:
    """Runtime options for the official Genesis batched backend."""

    n_envs: int
    profile_path: str | Path | None = None
    backend: str = "cuda"
    logical_cuda_device: str = "cuda:0"
    show_viewer: bool = False
    add_plane: bool = True
    logging_level: str = "warning"
    root_qpos: tuple[float, float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.78,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    default_positions_rad: tuple[float, ...] | None = None
    action_scale_mult: float = 1.0
    action_joint_group: str = "all"
    motor_kp_mult: float = 1.0
    motor_kv_mult: float = 1.0
    motor_force_limit_mult: float = 1.0
    require_asset_path: bool = True

    def __post_init__(self) -> None:
        if self.n_envs <= 0:
            raise ValueError("n_envs must be positive")
        if self.backend == "cuda" and self.logical_cuda_device != "cuda:0":
            raise ValueError("CUDA backend expects logical_cuda_device=cuda:0")
        if self.default_positions_rad is not None and len(self.default_positions_rad) != 27:
            raise ValueError("default_positions_rad must have length 27")
        if self.action_scale_mult <= 0.0:
            raise ValueError("action_scale_mult must be positive")
        if self.action_joint_group not in ACTION_JOINT_GROUPS:
            raise ValueError(f"action_joint_group must be one of {ACTION_JOINT_GROUPS}")
        if self.motor_kp_mult <= 0.0:
            raise ValueError("motor_kp_mult must be positive")
        if self.motor_kv_mult <= 0.0:
            raise ValueError("motor_kv_mult must be positive")
        if self.motor_force_limit_mult <= 0.0:
            raise ValueError("motor_force_limit_mult must be positive")


@dataclass(frozen=True, slots=True)
class VectorizedGenesisStep:
    """One vectorized transition returned by ``VectorizedGenesisBackend.step``."""

    observation: Any
    reward: Any
    terminated: Any
    truncated: Any
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VectorizedGenesisState:
    """Batched state snapshot read once from the Genesis robot."""

    qpos: Any
    root_pos: Any
    root_quat: Any
    root_vel: Any
    root_ang_vel: Any
    dof_pos: Any
    dof_vel: Any
    previous_action: Any


class VectorizedGenesisBackend:
    """Minimal batched Genesis backend for training smoke tests."""

    def __init__(
        self,
        config: VectorizedGenesisConfig,
        *,
        genesis_module: Any | None = None,
        torch_module: Any | None = None,
        profile: G1NoHandGenesisTrainingProfile | None = None,
    ) -> None:
        self.config = config
        if profile is not None:
            self.profile = profile
        elif config.profile_path is None:
            self.profile = load_g1_27dof_nohand_profile()
        else:
            self.profile = load_g1_27dof_nohand_profile(config.profile_path)
        if self.profile.route != "VectorizedGenesisBackend":
            raise ValueError("profile.route must be VectorizedGenesisBackend")
        if (
            config.require_asset_path
            and genesis_module is None
            and not Path(self.profile.asset.path).is_file()
        ):
            raise FileNotFoundError(f"Genesis asset not found: {self.profile.asset.path}")
        self.gs = genesis_module or import_genesis_module()
        self.torch = torch_module if torch_module is not None else self._maybe_import_torch()
        self.action_dim = self.profile.action_dim
        self.observation_dim = self.profile.training_contract.observation_dim
        self.decimation = self.profile.training_contract.decimation
        self.scene, self.robot = self._build_scene()
        self.motor_dof_indices = self._resolve_motor_dof_indices()
        self.motor_kp_mult = float(self.config.motor_kp_mult)
        self.motor_kv_mult = float(self.config.motor_kv_mult)
        self.motor_force_limit_mult = float(self.config.motor_force_limit_mult)
        self._apply_motor_config()
        self.reset_root_qpos = tuple(float(value) for value in self.config.root_qpos)
        self.default_positions_values = tuple(
            float(value)
            for value in (
                self.config.default_positions_rad
                if self.config.default_positions_rad is not None
                else self.profile.control.default_angles_rad
            )
        )
        self.default_positions = self._vector(self.default_positions_values)
        self.action_scale_mult = float(self.config.action_scale_mult)
        self.action_joint_group = self.config.action_joint_group
        self.action_scale_values = self._scaled_action_values(
            self.action_scale_mult,
            self.action_joint_group,
        )
        self.action_scales = self._vector(self.action_scale_values)
        self.previous_action = self._zeros((self.config.n_envs, self.action_dim))
        self.step_count = 0

    def set_motor_config_multipliers(
        self,
        *,
        kp_mult: float | None = None,
        kv_mult: float | None = None,
        force_limit_mult: float | None = None,
    ) -> None:
        """Reapply profile motor config with diagnostic scale multipliers."""

        next_kp = self.motor_kp_mult if kp_mult is None else float(kp_mult)
        next_kv = self.motor_kv_mult if kv_mult is None else float(kv_mult)
        next_force = (
            self.motor_force_limit_mult
            if force_limit_mult is None
            else float(force_limit_mult)
        )
        if next_kp <= 0.0:
            raise ValueError("kp_mult must be positive")
        if next_kv <= 0.0:
            raise ValueError("kv_mult must be positive")
        if next_force <= 0.0:
            raise ValueError("force_limit_mult must be positive")
        self.motor_kp_mult = next_kp
        self.motor_kv_mult = next_kv
        self.motor_force_limit_mult = next_force
        self._apply_motor_config()

    @property
    def n_envs(self) -> int:
        return self.config.n_envs

    def reset(self, env_ids: Any | None = None) -> Any:
        """Reset all envs or a selected env subset and return observations."""

        selected_envs = self._normalize_env_ids(env_ids)
        target_count = self.n_envs if selected_envs is None else self._env_ids_count(selected_envs)
        root_target = self._repeat(self.reset_root_qpos, target_count)
        motor_target = self._repeat(self.default_positions_values, target_count)
        self._set_root_qpos(root_target, selected_envs)
        self._set_dofs_position(motor_target, selected_envs)
        self._zero_dofs_velocity(selected_envs)
        self._reset_previous_action(selected_envs)
        if selected_envs is None:
            self.step_count = 0
        return self.observation()

    def set_reset_pose(
        self,
        *,
        root_qpos: Sequence[float] | None = None,
        default_positions_rad: Sequence[float] | None = None,
    ) -> None:
        """Update reset targets for standing-pose probes without rebuilding Genesis."""

        if root_qpos is not None:
            root_values = tuple(float(value) for value in root_qpos)
            if len(root_values) != 7:
                raise ValueError("root_qpos must have length 7")
            self.reset_root_qpos = root_values
        if default_positions_rad is not None:
            default_values = tuple(float(value) for value in default_positions_rad)
            if len(default_values) != self.action_dim:
                raise ValueError(f"default_positions_rad must have length {self.action_dim}")
            self.default_positions_values = default_values
            self.default_positions = self._vector(default_values)

    def set_action_scale_mult(
        self,
        action_scale_mult: float,
        *,
        action_joint_group: str | None = None,
    ) -> None:
        """Update normalized-action-to-joint-target scale without rebuilding Genesis."""

        if action_scale_mult <= 0.0:
            raise ValueError("action_scale_mult must be positive")
        next_group = action_joint_group or self.action_joint_group
        if next_group not in ACTION_JOINT_GROUPS:
            raise ValueError(f"action_joint_group must be one of {ACTION_JOINT_GROUPS}")
        self.action_scale_mult = float(action_scale_mult)
        self.action_joint_group = next_group
        self.action_scale_values = self._scaled_action_values(
            self.action_scale_mult,
            self.action_joint_group,
        )
        self.action_scales = self._vector(self.action_scale_values)

    def step_physics(self, action: Any) -> Any:
        """Apply one action batch and advance one policy frame without building obs."""

        clipped_action = self._clip_action(self._coerce_action(action))
        targets = self._action_targets(clipped_action)
        self._control_dofs_position(targets)
        for _ in range(self.decimation):
            self.scene.step()
        self.previous_action = clipped_action
        self.step_count += 1
        return clipped_action

    def step(self, action: Any) -> VectorizedGenesisStep:
        """Apply one normalized action batch and advance one policy frame."""

        self.step_physics(action)
        observation = self.observation()
        return VectorizedGenesisStep(
            observation=observation,
            reward=self._zeros((self.n_envs,)),
            terminated=self._false_vector(),
            truncated=self._false_vector(),
            info={
                "backend": "vectorized_genesis",
                "step_count": self.step_count,
                "n_envs": self.n_envs,
                "action_dim": self.action_dim,
                "observation_dim": self.observation_dim,
                "motor_dof_count": len(self.motor_dof_indices),
                "asset_path": self.profile.asset.path,
            },
        )

    def state(self) -> VectorizedGenesisState:
        """Read all state tensors needed by training envs in one place."""

        dof_pos = self._ensure_matrix(self._read_dofs_position(), self.action_dim)
        dof_vel = self._ensure_matrix(self._read_dofs_velocity(), self.action_dim)
        return VectorizedGenesisState(
            qpos=self.robot.get_qpos(),
            root_pos=self._ensure_matrix(self.robot.get_pos(), 3),
            root_quat=self._ensure_matrix(self.robot.get_quat(), 4),
            root_vel=self._ensure_matrix(self.robot.get_vel(), 3),
            root_ang_vel=self._ensure_matrix(self._read_root_ang_velocity(), 3),
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            previous_action=self.previous_action,
        )

    def observation(self) -> Any:
        state = self.state()
        motor_positions = state.dof_pos
        motor_velocities = state.dof_vel
        position_error = self._sub(motor_positions, self.default_positions)
        observation = self._concat_columns(
            (
                self._zeros((self.n_envs, 3)),
                self._repeat((0.0, 0.0, -1.0), self.n_envs),
                self._zeros((self.n_envs, 3)),
                position_error,
                motor_velocities,
                self.previous_action,
            )
        )
        self._expect_shape(observation, (self.n_envs, self.observation_dim), "observation")
        return observation

    def tensor_device_report(self) -> dict[str, str]:
        state = self._read_state_for_devices()
        return {
            "action_device": tensor_device_name(self.previous_action),
            "qpos_device": tensor_device_name(state.get("qpos")),
            "dofs_pos_device": tensor_device_name(state.get("dofs_pos")),
            "dofs_vel_device": tensor_device_name(state.get("dofs_vel")),
            "root_pos_device": tensor_device_name(state.get("root_pos")),
            "root_quat_device": tensor_device_name(state.get("root_quat")),
            "root_vel_device": tensor_device_name(state.get("root_vel")),
        }

    def tensor_device_ok(self) -> bool:
        if self.config.backend != "cuda":
            return True
        return all(
            value == self.config.logical_cuda_device
            for value in self.tensor_device_report().values()
        )

    def _build_scene(self) -> tuple[Any, Any]:
        self._init_genesis()
        sim_options = self._make_sim_options()
        scene_kwargs = {"show_viewer": self.config.show_viewer}
        if sim_options is not None:
            scene_kwargs["sim_options"] = sim_options
        scene = self.gs.Scene(**scene_kwargs)
        if self.config.add_plane and hasattr(getattr(self.gs, "morphs", None), "Plane"):
            scene.add_entity(self.gs.morphs.Plane())
        morph_type = getattr(self.gs.morphs, self.profile.asset.genesis_morph)
        robot = scene.add_entity(morph_type(file=self.profile.asset.path))
        scene.build(n_envs=self.n_envs)
        return scene, robot

    def _init_genesis(self) -> None:
        backend_value = getattr(self.gs, self.config.backend, self.config.backend)
        try:
            self.gs.init(backend=backend_value, logging_level=self.config.logging_level)
        except TypeError:
            self.gs.init(backend=backend_value)

    def _make_sim_options(self) -> Any | None:
        options = getattr(self.gs, "options", None)
        if options is None or not hasattr(options, "SimOptions"):
            return None
        try:
            return options.SimOptions(dt=self.profile.training_contract.sim_dt_s)
        except TypeError:
            return options.SimOptions()

    def _maybe_import_torch(self) -> Any | None:
        if self.config.backend != "cuda":
            return None
        try:
            import torch  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - H200 environment path.
            raise RuntimeError(f"torch import failed for CUDA backend: {exc}") from exc
        return torch

    def _resolve_motor_dof_indices(self) -> tuple[int, ...]:
        indices: list[int] = []
        for joint_name in self.profile.actuator_order:
            joint = self.robot.get_joint(joint_name)
            joint_indices = getattr(joint, "dofs_idx_local")
            if len(joint_indices) != 1:
                raise ValueError(f"Expected single-DoF joint {joint_name}, got {joint_indices}")
            indices.append(int(joint_indices[0]))
        if len(set(indices)) != self.action_dim:
            raise ValueError(f"Duplicate motor DOF indices: {indices}")
        return tuple(indices)

    def _apply_motor_config(self) -> None:
        self._set_dofs_kp(
            self._scale_values(self.profile.control.kp, self.motor_kp_mult)
        )
        self._set_dofs_kv(
            self._scale_values(self.profile.control.kv, self.motor_kv_mult)
        )
        self._set_dofs_force_range(
            self._scale_values(
                self.profile.control.force_limits,
                self.motor_force_limit_mult,
            )
        )

    def _scale_values(self, values: Sequence[float], multiplier: float) -> tuple[float, ...]:
        return tuple(float(value) * multiplier for value in values)

    def _set_dofs_kp(self, values: Sequence[float]) -> None:
        if not hasattr(self.robot, "set_dofs_kp"):
            return
        gains = tuple(float(value) for value in values)
        try:
            self.robot.set_dofs_kp(gains, dofs_idx_local=self.motor_dof_indices)
        except TypeError:
            self.robot.set_dofs_kp(gains, self.motor_dof_indices)

    def _set_dofs_kv(self, values: Sequence[float]) -> None:
        if not hasattr(self.robot, "set_dofs_kv"):
            return
        gains = tuple(float(value) for value in values)
        try:
            self.robot.set_dofs_kv(gains, dofs_idx_local=self.motor_dof_indices)
        except TypeError:
            self.robot.set_dofs_kv(gains, self.motor_dof_indices)

    def _set_dofs_force_range(self, limits: Sequence[float]) -> None:
        if not hasattr(self.robot, "set_dofs_force_range"):
            return
        lower = tuple(-float(limit) for limit in limits)
        upper = tuple(float(limit) for limit in limits)
        try:
            self.robot.set_dofs_force_range(
                lower,
                upper,
                dofs_idx_local=self.motor_dof_indices,
            )
        except TypeError:
            self.robot.set_dofs_force_range(lower, upper, self.motor_dof_indices)

    def _coerce_action(self, action: Any) -> Any:
        if is_tensor_like(action):
            self._expect_shape(action, (self.n_envs, self.action_dim), "action")
            return action.to(self.config.logical_cuda_device) if self.torch is not None else action
        rows = as_rows(action)
        if len(rows) != self.n_envs or any(len(row) != self.action_dim for row in rows):
            raise ValueError(f"action expected shape=({self.n_envs}, {self.action_dim})")
        return self._matrix(rows)

    def _clip_action(self, action: Any) -> Any:
        if is_tensor_like(action):
            return action.clamp(-1.0, 1.0)
        return [[max(-1.0, min(1.0, value)) for value in row] for row in action]

    def _action_targets(self, action: Any) -> Any:
        if is_tensor_like(action):
            return self.default_positions.unsqueeze(0) + action * self.action_scales.unsqueeze(0)
        return [
            [
                default + delta * scale
                for default, delta, scale in zip(
                    self.default_positions_values,
                    row,
                    self.action_scale_values,
                )
            ]
            for row in action
        ]

    def _scaled_action_values(
        self,
        action_scale_mult: float,
        action_joint_group: str,
    ) -> tuple[float, ...]:
        mask = self._action_group_mask(action_joint_group)
        return tuple(
            float(value) * action_scale_mult * group_scale
            for value, group_scale in zip(self.profile.control.action_scales_rad, mask)
        )

    def _action_group_mask(self, action_joint_group: str) -> tuple[float, ...]:
        if action_joint_group == "all":
            return (1.0,) * self.action_dim
        values: list[float] = []
        for joint_name in self.profile.actuator_order:
            is_leg = joint_name.startswith(("left_", "right_")) and joint_name.endswith(
                LEG_JOINT_SUFFIXES
            )
            is_waist = joint_name == "waist_yaw_joint"
            if is_leg or (action_joint_group == "legs_waist" and is_waist):
                values.append(1.0)
            else:
                values.append(0.0)
        if tuple(self.profile.actuator_order) != G1_27DOF_NOHAND_ACTUATOR_ORDER:
            raise ValueError("unexpected G1 27DoF actuator order")
        return tuple(values)

    def _control_dofs_position(self, targets: Any) -> None:
        try:
            self.robot.control_dofs_position(
                targets,
                dofs_idx_local=self.motor_dof_indices,
            )
        except TypeError:
            self.robot.control_dofs_position(targets, self.motor_dof_indices)

    def _set_root_qpos(self, root_target: Any, env_ids: Any | None) -> None:
        if not hasattr(self.robot, "set_qpos"):
            return
        envs_idx = self._env_ids_tensor(env_ids)
        kwargs: dict[str, Any] = {"qs_idx_local": tuple(range(7)), "zero_velocity": True}
        if envs_idx is not None:
            kwargs["envs_idx"] = envs_idx
        try:
            self.robot.set_qpos(root_target, **kwargs)
        except TypeError:
            kwargs.pop("zero_velocity", None)
            self.robot.set_qpos(root_target, **kwargs)

    def _set_dofs_position(self, target: Any, env_ids: Any | None) -> None:
        envs_idx = self._env_ids_tensor(env_ids)
        kwargs: dict[str, Any] = {
            "position": target,
            "dofs_idx_local": self.motor_dof_indices,
            "zero_velocity": True,
        }
        if envs_idx is not None:
            kwargs["envs_idx"] = envs_idx
        try:
            self.robot.set_dofs_position(**kwargs)
        except TypeError:
            kwargs.pop("zero_velocity", None)
            self.robot.set_dofs_position(**kwargs)

    def _zero_dofs_velocity(self, env_ids: Any | None) -> None:
        if not hasattr(self.robot, "set_dofs_velocity"):
            return
        target = self._zeros(
            (self.n_envs if env_ids is None else self._env_ids_count(env_ids), self.action_dim)
        )
        kwargs: dict[str, Any] = {"velocity": target, "dofs_idx_local": self.motor_dof_indices}
        envs_idx = self._env_ids_tensor(env_ids)
        if envs_idx is not None:
            kwargs["envs_idx"] = envs_idx
        try:
            self.robot.set_dofs_velocity(**kwargs)
        except TypeError:
            self.robot.set_dofs_velocity(None)

    def _reset_previous_action(self, env_ids: Any | None) -> None:
        if env_ids is None:
            self.previous_action = self._zeros((self.n_envs, self.action_dim))
            return
        if is_tensor_like(env_ids) and is_tensor_like(self.previous_action):
            self.previous_action[env_ids] = 0.0
            return
        rows = as_rows(self.previous_action)
        for env_id in env_ids:
            rows[env_id] = [0.0] * self.action_dim
        self.previous_action = self._matrix(rows)

    def _read_dofs_position(self) -> Any:
        return self.robot.get_dofs_position(dofs_idx_local=self.motor_dof_indices)

    def _read_dofs_velocity(self) -> Any:
        return self.robot.get_dofs_velocity(dofs_idx_local=self.motor_dof_indices)

    def _read_root_ang_velocity(self) -> Any:
        for method_name in (
            "get_ang",
            "get_ang_vel",
            "get_angular_velocity",
            "get_base_ang_vel",
        ):
            if hasattr(self.robot, method_name):
                return getattr(self.robot, method_name)()
        return self._zeros((self.n_envs, 3))

    def _read_state_for_devices(self) -> dict[str, Any]:
        return {
            "qpos": read_optional(lambda: self.robot.get_qpos()),
            "dofs_pos": read_optional(self._read_dofs_position),
            "dofs_vel": read_optional(self._read_dofs_velocity),
            "root_pos": read_optional(lambda: self.robot.get_pos()),
            "root_quat": read_optional(lambda: self.robot.get_quat()),
            "root_vel": read_optional(lambda: self.robot.get_vel()),
        }

    def _ensure_matrix(self, value: Any, width: int) -> Any:
        if is_tensor_like(value):
            if len(value.shape) == 1:
                value = value.reshape(1, width).expand(self.n_envs, width)
            self._expect_shape(value, (self.n_envs, width), "state tensor")
            return value.to(self.config.logical_cuda_device) if self.torch is not None else value
        rows = as_rows(value)
        if len(rows) == 1 and self.n_envs > 1:
            rows = rows * self.n_envs
        if len(rows) != self.n_envs or any(len(row) != width for row in rows):
            raise ValueError(f"state tensor expected shape=({self.n_envs}, {width})")
        return self._matrix(rows)

    def _sub(self, left: Any, right: Any) -> Any:
        if is_tensor_like(left):
            return left - right.unsqueeze(0)
        return [[value - baseline for value, baseline in zip(row, right)] for row in left]

    def _concat_columns(self, values: Sequence[Any]) -> Any:
        if self.torch is not None and any(is_tensor_like(value) for value in values):
            return self.torch.cat(
                [value if is_tensor_like(value) else self._matrix(value) for value in values],
                dim=1,
            )
        rows = [as_rows(value) for value in values]
        return [sum((group[index] for group in rows), []) for index in range(self.n_envs)]

    def _zeros(self, shape: tuple[int, ...]) -> Any:
        if self.torch is not None:
            return self.torch.zeros(
                shape,
                dtype=self.torch.float32,
                device=self.config.logical_cuda_device,
            )
        if len(shape) == 1:
            return [0.0 for _ in range(shape[0])]
        rows, columns = shape
        return [[0.0 for _ in range(columns)] for _ in range(rows)]

    def _false_vector(self) -> Any:
        if self.torch is not None:
            return self.torch.zeros(
                (self.n_envs,),
                dtype=self.torch.bool,
                device=self.config.logical_cuda_device,
            )
        return [False for _ in range(self.n_envs)]

    def _vector(self, values: Sequence[float]) -> Any:
        if self.torch is not None:
            return self.torch.tensor(
                tuple(float(value) for value in values),
                dtype=self.torch.float32,
                device=self.config.logical_cuda_device,
            )
        return [float(value) for value in values]

    def _matrix(self, values: Sequence[Sequence[float]]) -> Any:
        if self.torch is not None:
            return self.torch.tensor(
                values,
                dtype=self.torch.float32,
                device=self.config.logical_cuda_device,
            )
        return [[float(value) for value in row] for row in values]

    def _repeat(self, values: Sequence[float], rows: int) -> Any:
        return self._matrix([tuple(float(value) for value in values)] * rows)

    def _env_ids_tensor(self, env_ids: Any | None) -> Any | None:
        if env_ids is None:
            return None
        if is_tensor_like(env_ids):
            if self.torch is not None:
                return env_ids.to(self.config.logical_cuda_device)
            return env_ids
        if self.torch is not None:
            return self.torch.tensor(env_ids, device=self.config.logical_cuda_device)
        return env_ids

    def _normalize_env_ids(self, env_ids: Any | None) -> Any | None:
        if env_ids is None:
            return None
        if is_tensor_like(env_ids):
            shape = tensor_shape(env_ids)
            if len(shape) != 1:
                raise ValueError("env_ids must be a 1D tensor or a sequence of integers")
            if int(env_ids.numel()) == 0:
                raise ValueError("env_ids must not be empty")
            if self.torch is not None:
                return env_ids.to(self.config.logical_cuda_device)
            return env_ids
        normalized = tuple(int(env_id) for env_id in env_ids)
        if not normalized:
            raise ValueError("env_ids must not be empty")
        if any(env_id < 0 or env_id >= self.n_envs for env_id in normalized):
            raise ValueError("env_ids contains an out-of-range environment index")
        if len(set(normalized)) != len(normalized):
            raise ValueError("env_ids must not contain duplicates")
        return normalized

    def _env_ids_count(self, env_ids: Any) -> int:
        if is_tensor_like(env_ids):
            return int(env_ids.numel())
        return len(env_ids)

    def _expect_shape(self, value: Any, expected: tuple[int, ...], label: str) -> None:
        shape = tensor_shape(value)
        if shape != expected:
            raise ValueError(f"{label} expected shape={expected}, got {shape}")


def import_genesis_module() -> Any:
    try:
        import genesis as gs  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - H200 environment path.
        raise RuntimeError(f"genesis import failed: {exc}") from exc
    return gs


def tensor_device_name(value: Any) -> str:
    if value is None:
        return "unavailable"
    device = getattr(value, "device", None)
    if device is None:
        return "not_tensor"
    return str(device)


def tensor_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(item) for item in shape)
    if isinstance(value, list) and all(isinstance(item, (int, float, bool)) for item in value):
        return (len(value),)
    rows = as_rows(value)
    if not rows:
        return (0,)
    if len(rows) == 1:
        return (1, len(rows[0]))
    return (len(rows), len(rows[0]))


def is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "device")


def as_rows(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [[float(value)]]
    if not isinstance(value, list):
        value = list(value)
    if not value:
        return []
    if all(isinstance(item, (int, float, bool)) for item in value):
        return [[float(item) for item in value]]
    return [[float(item) for item in row] for row in value]


def read_optional(reader: Any) -> Any | None:
    try:
        return reader()
    except Exception:
        return None
