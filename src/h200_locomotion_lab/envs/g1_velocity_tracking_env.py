"""Training-oriented vectorized velocity tracking env for G1 27DoF Genesis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from h200_locomotion_lab.envs.vectorized_genesis_backend import (
    VectorizedGenesisBackend,
    VectorizedGenesisState,
    as_rows,
    is_tensor_like,
    tensor_device_name,
    tensor_shape,
)


@dataclass(frozen=True, slots=True)
class G1VelocityTrackingConfig:
    """Task13 training-env semantics, kept policy-agnostic."""

    max_episode_steps: int = 1000
    command_vx_min: float = 0.0
    command_vx_max: float = 0.8
    command_yaw_min: float = -0.5
    command_yaw_max: float = 0.5
    height_min: float = 0.45
    height_max: float = 1.20
    min_upright: float = 0.30
    lin_vel_sigma: float = 0.25
    yaw_rate_sigma: float = 0.25
    alive_reward: float = 0.05
    lin_vel_reward_scale: float = 1.00
    yaw_rate_reward_scale: float = 0.50
    upright_reward_scale: float = 0.50
    action_rate_penalty_scale: float = 0.01
    joint_deviation_penalty_scale: float = 0.05

    def __post_init__(self) -> None:
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.command_vx_min > self.command_vx_max:
            raise ValueError("command_vx_min must be <= command_vx_max")
        if self.command_yaw_min > self.command_yaw_max:
            raise ValueError("command_yaw_min must be <= command_yaw_max")
        if self.height_min >= self.height_max:
            raise ValueError("height_min must be < height_max")
        if self.lin_vel_sigma <= 0 or self.yaw_rate_sigma <= 0:
            raise ValueError("tracking sigmas must be positive")


@dataclass(frozen=True, slots=True)
class G1VelocityTrackingStep:
    """One vectorized env step."""

    observation: Any
    reward: Any
    terminated: Any
    truncated: Any
    done: Any
    info: Mapping[str, Any] = field(default_factory=dict)


class G1VelocityTrackingVectorizedEnv:
    """Policy-agnostic training env wrapper over VectorizedGenesisBackend."""

    def __init__(
        self,
        backend: VectorizedGenesisBackend,
        config: G1VelocityTrackingConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or G1VelocityTrackingConfig()
        self.torch = backend.torch
        self.n_envs = backend.n_envs
        self.action_dim = backend.action_dim
        self.observation_dim = backend.observation_dim
        self.commands = self._zeros((self.n_envs, 3))
        self.episode_lengths = self._zeros_int((self.n_envs,))
        self.last_action = self._zeros((self.n_envs, self.action_dim))
        self.last_components: dict[str, Any] = {}

    def reset(self, env_ids: Any | None = None) -> Any:
        selected_envs = self.backend._normalize_env_ids(env_ids)
        self.backend.reset(selected_envs)
        self._reset_bookkeeping(selected_envs)
        return self._observation(self.backend.state())

    def step(self, action: Any) -> G1VelocityTrackingStep:
        previous_action = self.last_action
        clipped_action = self.backend.step_physics(action)
        state = self.backend.state()
        self._increment_episode_lengths()
        reward, terminated, truncated, done, components = self._reward_done(
            state=state,
            action=clipped_action,
            previous_action=previous_action,
        )
        self.last_action = clipped_action
        done_env_ids = self._done_env_ids(done)
        reset_count = self._env_ids_count(done_env_ids)
        if reset_count:
            self.backend.reset(done_env_ids)
            self._reset_bookkeeping(done_env_ids)
            state = self.backend.state()
        observation = self._observation(state)
        self.last_components = components
        info = {
            "backend": "vectorized_genesis",
            "task": "g1_velocity_tracking",
            "n_envs": self.n_envs,
            "action_dim": self.action_dim,
            "observation_dim": self.observation_dim,
            "reset_count": reset_count,
            "components": components,
        }
        return G1VelocityTrackingStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            done=done,
            info=info,
        )

    def tensor_device_report(self, sample: G1VelocityTrackingStep | None = None) -> dict[str, str]:
        state = self.backend.state()
        report = {
            "command_device": tensor_device_name(self.commands),
            "episode_length_device": tensor_device_name(self.episode_lengths),
            "last_action_device": tensor_device_name(self.last_action),
            "qpos_device": tensor_device_name(state.qpos),
            "root_pos_device": tensor_device_name(state.root_pos),
            "root_quat_device": tensor_device_name(state.root_quat),
            "root_vel_device": tensor_device_name(state.root_vel),
            "root_ang_vel_device": tensor_device_name(state.root_ang_vel),
            "dofs_pos_device": tensor_device_name(state.dof_pos),
            "dofs_vel_device": tensor_device_name(state.dof_vel),
        }
        if sample is not None:
            report.update(
                {
                    "observation_device": tensor_device_name(sample.observation),
                    "reward_device": tensor_device_name(sample.reward),
                    "terminated_device": tensor_device_name(sample.terminated),
                    "truncated_device": tensor_device_name(sample.truncated),
                    "done_device": tensor_device_name(sample.done),
                }
            )
        return report

    def tensor_device_ok(self, sample: G1VelocityTrackingStep | None = None) -> bool:
        if self.backend.config.backend != "cuda":
            return True
        return all(
            value == self.backend.config.logical_cuda_device
            for value in self.tensor_device_report(sample).values()
        )

    def _observation(self, state: VectorizedGenesisState) -> Any:
        position_error = self._sub(state.dof_pos, self.backend.default_positions)
        observation = self._concat_columns(
            (
                state.root_ang_vel,
                self._projected_gravity(state.root_quat),
                self.commands,
                position_error,
                state.dof_vel,
                self.last_action,
            )
        )
        if tensor_shape(observation) != (self.n_envs, self.observation_dim):
            raise ValueError(
                f"observation expected shape=({self.n_envs}, {self.observation_dim}), "
                f"got {tensor_shape(observation)}"
            )
        return observation

    def _reward_done(
        self,
        *,
        state: VectorizedGenesisState,
        action: Any,
        previous_action: Any,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        if self.torch is not None and is_tensor_like(state.root_pos):
            return self._reward_done_torch(
                state=state,
                action=action,
                previous_action=previous_action,
            )
        return self._reward_done_lists(
            state=state,
            action=action,
            previous_action=previous_action,
        )

    def _reward_done_torch(
        self,
        *,
        state: VectorizedGenesisState,
        action: Any,
        previous_action: Any,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        root_height = state.root_pos[:, 2]
        projected_gravity = self._projected_gravity(state.root_quat)
        upright = (-projected_gravity[:, 2]).clamp(0.0, 1.0)
        lin_vel_error = (state.root_vel[:, 0] - self.commands[:, 0]).square()
        yaw_rate_error = (state.root_ang_vel[:, 2] - self.commands[:, 2]).square()
        tracking_lin_vel = self.torch.exp(-lin_vel_error / self.config.lin_vel_sigma)
        tracking_yaw_rate = self.torch.exp(-yaw_rate_error / self.config.yaw_rate_sigma)
        action_rate_penalty = (action - previous_action).square().mean(dim=1)
        joint_error = self._sub(state.dof_pos, self.backend.default_positions)
        joint_deviation_penalty = joint_error.square().mean(dim=1)
        reward = (
            self.config.alive_reward
            + self.config.lin_vel_reward_scale * tracking_lin_vel
            + self.config.yaw_rate_reward_scale * tracking_yaw_rate
            + self.config.upright_reward_scale * upright
            - self.config.action_rate_penalty_scale * action_rate_penalty
            - self.config.joint_deviation_penalty_scale * joint_deviation_penalty
        )
        height_bad = (root_height < self.config.height_min) | (
            root_height > self.config.height_max
        )
        tilt_bad = upright < self.config.min_upright
        terminated = height_bad | tilt_bad
        truncated = self.episode_lengths >= self.config.max_episode_steps
        done = terminated | truncated
        components = {
            "tracking_lin_vel": tracking_lin_vel,
            "tracking_yaw_rate": tracking_yaw_rate,
            "upright": upright,
            "action_rate_penalty": action_rate_penalty,
            "joint_deviation_penalty": joint_deviation_penalty,
            "height_bad": height_bad,
            "tilt_bad": tilt_bad,
            "timeout": truncated,
        }
        return reward, terminated, truncated, done, components

    def _reward_done_lists(
        self,
        *,
        state: VectorizedGenesisState,
        action: Any,
        previous_action: Any,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        root_pos = as_rows(state.root_pos)
        root_vel = as_rows(state.root_vel)
        root_ang = as_rows(state.root_ang_vel)
        dof_pos = as_rows(state.dof_pos)
        commands = as_rows(self.commands)
        actions = as_rows(action)
        previous_actions = as_rows(previous_action)
        projected = as_rows(self._projected_gravity(state.root_quat))
        default = list(self.backend.profile.control.default_angles_rad)
        reward: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        tracking_lin_vel: list[float] = []
        tracking_yaw_rate: list[float] = []
        upright_values: list[float] = []
        action_rate_penalty: list[float] = []
        joint_deviation_penalty: list[float] = []
        height_bad_values: list[bool] = []
        tilt_bad_values: list[bool] = []
        for index in range(self.n_envs):
            upright = max(0.0, min(1.0, -projected[index][2]))
            lin_error = (root_vel[index][0] - commands[index][0]) ** 2
            yaw_error = (root_ang[index][2] - commands[index][2]) ** 2
            tracking_lin = math.exp(-lin_error / self.config.lin_vel_sigma)
            tracking_yaw = math.exp(-yaw_error / self.config.yaw_rate_sigma)
            action_penalty = sum(
                (value - previous) ** 2
                for value, previous in zip(actions[index], previous_actions[index])
            ) / self.action_dim
            joint_penalty = sum(
                (value - baseline) ** 2 for value, baseline in zip(dof_pos[index], default)
            ) / self.action_dim
            item_reward = (
                self.config.alive_reward
                + self.config.lin_vel_reward_scale * tracking_lin
                + self.config.yaw_rate_reward_scale * tracking_yaw
                + self.config.upright_reward_scale * upright
                - self.config.action_rate_penalty_scale * action_penalty
                - self.config.joint_deviation_penalty_scale * joint_penalty
            )
            height_bad = (
                root_pos[index][2] < self.config.height_min
                or root_pos[index][2] > self.config.height_max
            )
            tilt_bad = upright < self.config.min_upright
            timeout = self.episode_lengths[index] >= self.config.max_episode_steps
            reward.append(item_reward)
            terminated.append(height_bad or tilt_bad)
            truncated.append(timeout)
            tracking_lin_vel.append(tracking_lin)
            tracking_yaw_rate.append(tracking_yaw)
            upright_values.append(upright)
            action_rate_penalty.append(action_penalty)
            joint_deviation_penalty.append(joint_penalty)
            height_bad_values.append(height_bad)
            tilt_bad_values.append(tilt_bad)
        done = [left or right for left, right in zip(terminated, truncated)]
        components = {
            "tracking_lin_vel": tracking_lin_vel,
            "tracking_yaw_rate": tracking_yaw_rate,
            "upright": upright_values,
            "action_rate_penalty": action_rate_penalty,
            "joint_deviation_penalty": joint_deviation_penalty,
            "height_bad": height_bad_values,
            "tilt_bad": tilt_bad_values,
            "timeout": truncated,
        }
        return reward, terminated, truncated, done, components

    def _projected_gravity(self, root_quat: Any) -> Any:
        if self.torch is not None and is_tensor_like(root_quat):
            quat = root_quat / root_quat.norm(dim=1, keepdim=True).clamp_min(1e-6)
            w = quat[:, 0]
            x = quat[:, 1]
            y = quat[:, 2]
            z = quat[:, 3]
            return self.torch.stack(
                (
                    2.0 * (w * y - x * z),
                    -2.0 * (w * x + y * z),
                    -1.0 + 2.0 * (x.square() + y.square()),
                ),
                dim=1,
            )
        rows = as_rows(root_quat)
        projected: list[list[float]] = []
        for row in rows:
            norm = math.sqrt(sum(value * value for value in row)) or 1.0
            w, x, y, z = (value / norm for value in row)
            projected.append(
                [
                    2.0 * (w * y - x * z),
                    -2.0 * (w * x + y * z),
                    -1.0 + 2.0 * (x * x + y * y),
                ]
            )
        return projected

    def _reset_bookkeeping(self, env_ids: Any | None) -> None:
        if env_ids is None:
            self.commands = self._sample_commands(self.n_envs)
            self.episode_lengths = self._zeros_int((self.n_envs,))
            self.last_action = self._zeros((self.n_envs, self.action_dim))
            return
        count = self._env_ids_count(env_ids)
        commands = self._sample_commands(count)
        if self.torch is not None and is_tensor_like(env_ids):
            self.commands[env_ids] = commands
            self.episode_lengths[env_ids] = 0
            self.last_action[env_ids] = 0.0
            return
        command_rows = as_rows(self.commands)
        sampled_rows = as_rows(commands)
        action_rows = as_rows(self.last_action)
        for source_index, env_index in enumerate(env_ids):
            command_rows[env_index] = sampled_rows[source_index]
            self.episode_lengths[env_index] = 0
            action_rows[env_index] = [0.0] * self.action_dim
        self.commands = command_rows
        self.last_action = action_rows

    def _sample_commands(self, rows: int) -> Any:
        if self.torch is not None:
            commands = self._zeros((rows, 3))
            vx_rand = self.torch.rand((rows,), device=self.backend.config.logical_cuda_device)
            yaw_rand = self.torch.rand((rows,), device=self.backend.config.logical_cuda_device)
            commands[:, 0] = self.config.command_vx_min + vx_rand * (
                self.config.command_vx_max - self.config.command_vx_min
            )
            commands[:, 1] = 0.0
            commands[:, 2] = self.config.command_yaw_min + yaw_rand * (
                self.config.command_yaw_max - self.config.command_yaw_min
            )
            return commands
        vx = 0.5 * (self.config.command_vx_min + self.config.command_vx_max)
        yaw = 0.5 * (self.config.command_yaw_min + self.config.command_yaw_max)
        return [[vx, 0.0, yaw] for _ in range(rows)]

    def _increment_episode_lengths(self) -> None:
        if self.torch is not None and is_tensor_like(self.episode_lengths):
            self.episode_lengths += 1
            return
        self.episode_lengths = [value + 1 for value in self.episode_lengths]

    def _done_env_ids(self, done: Any) -> Any:
        if self.torch is not None and is_tensor_like(done):
            return self.torch.nonzero(done, as_tuple=False).flatten()
        return tuple(index for index, value in enumerate(done) if value)

    def _env_ids_count(self, env_ids: Any) -> int:
        if is_tensor_like(env_ids):
            return int(env_ids.numel())
        return len(env_ids)

    def _zeros(self, shape: tuple[int, ...]) -> Any:
        return self.backend._zeros(shape)

    def _zeros_int(self, shape: tuple[int, ...]) -> Any:
        if self.torch is not None:
            return self.torch.zeros(
                shape,
                dtype=self.torch.long,
                device=self.backend.config.logical_cuda_device,
            )
        return [0 for _ in range(shape[0])]

    def _sub(self, left: Any, right: Any) -> Any:
        return self.backend._sub(left, right)

    def _concat_columns(self, values: tuple[Any, ...]) -> Any:
        return self.backend._concat_columns(values)
