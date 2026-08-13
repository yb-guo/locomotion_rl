"""MJLab reset hooks for Task037 deterministic inner-trial reset."""

from __future__ import annotations

from types import MethodType
from typing import Any

from h200_locomotion_lab.training.multitrial_wrapper import (
    TASK037_TERMINAL_COMMAND_KEY,
    TASK037_TERMINAL_GRAVITY_XY_KEY,
    TASK037_TERMINAL_LIN_VEL_KEY,
    TASK037_TERMINAL_METRIC_MASK_KEY,
    TASK037_TERMINAL_METRIC_SCHEMA,
    TASK037_TERMINAL_METRIC_SCHEMA_KEY,
    TASK037_TERMINAL_ROOT_Z_KEY,
    TASK037_TERMINAL_YAW_VEL_KEY,
)


class Task037MjlabInnerResetController:
    """Patch a MJLab ManagerBasedRlEnv to split inner and outer resets."""

    def __init__(self, env: Any, *, num_trials: int = 3) -> None:
        if num_trials <= 0:
            raise ValueError("num_trials must be positive")
        self.env = env
        self.num_trials = num_trials
        self.torch = _require_torch()
        self.trial_index = self.torch.zeros(env.num_envs, device=env.device, dtype=self.torch.long)
        self.inner_reset_count = self.torch.zeros(env.num_envs, device=env.device, dtype=self.torch.long)
        self.outer_reset_count = self.torch.zeros(env.num_envs, device=env.device, dtype=self.torch.long)
        self._full_reset_active = False
        self._pending_post_step_restore: tuple[_ConditionSnapshot, Any] | None = None
        self._last_inner_reset_ids = self.torch.empty(0, device=env.device, dtype=self.torch.long)
        self._original_reset = env.reset
        self._original_reset_idx = env._reset_idx
        self._original_step = env.step

    def install(self) -> None:
        configure_deterministic_reset_pose(self.env.cfg)
        controller = self

        def reset(env_self: Any, *args: Any, **kwargs: Any) -> Any:
            controller._full_reset_active = True
            try:
                result = controller._original_reset(*args, **kwargs)
            finally:
                controller._full_reset_active = False
            env_ids = kwargs.get("env_ids")
            if env_ids is None:
                controller.trial_index.zero_()
            else:
                controller.trial_index[env_ids] = 0
            return result

        def reset_idx(env_self: Any, env_ids: Any | None = None) -> None:
            if env_ids is None:
                env_ids = controller.torch.arange(
                    controller.env.num_envs,
                    dtype=controller.torch.long,
                    device=controller.env.device,
                )
            if controller._full_reset_active:
                controller._original_reset_idx(env_ids)
                return
            controller._split_reset_idx(env_ids)

        def step(env_self: Any, action: Any) -> Any:
            result = controller._original_step(action)
            if controller._pending_post_step_restore is not None:
                snapshot, inner_ids = controller._pending_post_step_restore
                snapshot.restore(controller.env, inner_ids)
                controller._pending_post_step_restore = None
            return result

        self.env.reset = MethodType(reset, self.env)
        self.env._reset_idx = MethodType(reset_idx, self.env)
        self.env.step = MethodType(step, self.env)
        self.env._task037_inner_reset_controller = self

    def _split_reset_idx(self, env_ids: Any) -> None:
        env_ids = env_ids.to(device=self.env.device, dtype=self.torch.long).reshape(-1)
        if env_ids.numel() == 0:
            return
        final_mask = self.trial_index[env_ids] >= self.num_trials - 1
        outer_ids = env_ids[final_mask]
        inner_ids = env_ids[~final_mask]
        self._store_terminal_metrics(env_ids)
        snapshot = _ConditionSnapshot.capture(self.env, inner_ids)
        self._original_reset_idx(env_ids)
        if inner_ids.numel() > 0:
            snapshot.restore(self.env, inner_ids)
            self._pending_post_step_restore = (snapshot, inner_ids.clone())
            self._last_inner_reset_ids = inner_ids.clone()
            self.trial_index[inner_ids] += 1
            self.inner_reset_count[inner_ids] += 1
        if outer_ids.numel() > 0:
            self.trial_index[outer_ids] = 0
            self.outer_reset_count[outer_ids] += 1
        self.env.extras["task037_inner_reset_count"] = self.inner_reset_count.clone()
        self.env.extras["task037_outer_reset_count"] = self.outer_reset_count.clone()

    def _store_terminal_metrics(self, env_ids: Any) -> None:
        metrics = _capture_terminal_metrics(self.env, env_ids, self.torch)
        if metrics:
            self.env.extras.update(metrics)


class _ConditionSnapshot:
    def __init__(
        self,
        *,
        command_tensors: dict[str, Any],
        env_tensors: dict[str, Any],
    ) -> None:
        self.command_tensors = command_tensors
        self.env_tensors = env_tensors

    @classmethod
    def capture(cls, env: Any, env_ids: Any) -> "_ConditionSnapshot":
        command_tensors: dict[str, Any] = {}
        if env_ids.numel() > 0:
            command_term = _command_term(env, "twist")
            if command_term is not None:
                for name in (
                    "vel_command_b",
                    "heading_target",
                    "heading_error",
                    "is_heading_env",
                    "is_standing_env",
                    "time_left",
                    "command_counter",
                ):
                    value = getattr(command_term, name, None)
                    if _is_env_tensor(value, env):
                        command_tensors[name] = value.index_select(0, env_ids).clone()
        env_tensors = _capture_env_condition_tensors(env, env_ids)
        return cls(command_tensors=command_tensors, env_tensors=env_tensors)

    def restore(self, env: Any, env_ids: Any) -> None:
        command_term = _command_term(env, "twist")
        if command_term is not None:
            for name, value in self.command_tensors.items():
                target = getattr(command_term, name, None)
                if _is_env_tensor(target, env):
                    target[env_ids] = value.to(device=target.device, dtype=target.dtype)
        for name, value in self.env_tensors.items():
            target = getattr(env, name, None)
            if _is_env_tensor(target, env):
                target[env_ids] = value.to(device=target.device, dtype=target.dtype)


def install_task037_inner_reset_controller(
    env: Any,
    *,
    num_trials: int = 3,
) -> Task037MjlabInnerResetController:
    existing = getattr(env, "_task037_inner_reset_controller", None)
    if existing is not None:
        return existing
    controller = Task037MjlabInnerResetController(env, num_trials=num_trials)
    controller.install()
    return controller


def configure_deterministic_reset_pose(cfg: Any) -> None:
    reset_base = cfg.events.get("reset_base") if hasattr(cfg, "events") else None
    if reset_base is not None:
        params = reset_base.params
        pose_range = dict(params.get("pose_range", {}))
        pose_range.update(
            {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        )
        params["pose_range"] = pose_range
        params["velocity_range"] = {}
    reset_joints = cfg.events.get("reset_robot_joints") if hasattr(cfg, "events") else None
    if reset_joints is not None:
        reset_joints.params["position_range"] = (0.0, 0.0)
        reset_joints.params["velocity_range"] = (0.0, 0.0)


def _command_term(env: Any, name: str) -> Any | None:
    terms = getattr(env.command_manager, "_terms", None)
    if isinstance(terms, dict):
        return terms.get(name)
    get_term = getattr(env.command_manager, "get_term", None)
    if get_term is not None:
        try:
            return get_term(name)
        except Exception:
            return None
    return None


def _capture_env_condition_tensors(env: Any, env_ids: Any) -> dict[str, Any]:
    tensors = {}
    for name in dir(env):
        if not (
            name.startswith("_task029_motor_failure_")
            or name.startswith("_task030_dynamic")
            or name.startswith("_task031_dynamic")
        ):
            continue
        value = getattr(env, name)
        if _is_env_tensor(value, env):
            tensors[name] = value.index_select(0, env_ids).clone()
    return tensors


def _capture_terminal_metrics(env: Any, env_ids: Any, torch: Any) -> dict[str, Any]:
    env_ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if env_ids.numel() <= 0:
        return {
            TASK037_TERMINAL_METRIC_SCHEMA_KEY: TASK037_TERMINAL_METRIC_SCHEMA,
            TASK037_TERMINAL_METRIC_MASK_KEY: mask,
        }
    scene = getattr(env, "scene", None)
    command_manager = getattr(env, "command_manager", None)
    if not isinstance(scene, dict) or "robot" not in scene or command_manager is None:
        return {}
    robot = scene["robot"]
    data = getattr(robot, "data", None)
    get_command = getattr(command_manager, "get_command", None)
    if data is None or not callable(get_command):
        return {}
    try:
        command = get_command("twist")
        lin_vel = data.root_link_lin_vel_b[:, :2]
        yaw_vel = data.root_link_ang_vel_b[:, 2]
        gravity_xy = torch.linalg.norm(data.projected_gravity_b[:, :2], dim=-1)
        root_z = data.root_link_pos_w[:, 2]
    except Exception:
        return {}
    mask[env_ids] = True
    return {
        TASK037_TERMINAL_METRIC_SCHEMA_KEY: TASK037_TERMINAL_METRIC_SCHEMA,
        TASK037_TERMINAL_METRIC_MASK_KEY: mask,
        TASK037_TERMINAL_COMMAND_KEY: command.to(device=env.device).clone(),
        TASK037_TERMINAL_LIN_VEL_KEY: lin_vel.to(device=env.device).clone(),
        TASK037_TERMINAL_YAW_VEL_KEY: yaw_vel.to(device=env.device).clone(),
        TASK037_TERMINAL_GRAVITY_XY_KEY: gravity_xy.to(device=env.device).clone(),
        TASK037_TERMINAL_ROOT_Z_KEY: root_z.to(device=env.device).clone(),
    }


def _is_env_tensor(value: Any, env: Any) -> bool:
    return hasattr(value, "shape") and value.shape[:1] == (env.num_envs,)


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - H200-only path.
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch
