"""Run a tiny local CUDA PPO-core smoke for Task047 evidence.

This command intentionally avoids simulator/runtime assets. It only verifies
that the repository's PPO rollout, timeout-aware GAE, and optimizer update
path can execute on the local CUDA device.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from h200_locomotion_lab.training.ppo_loop import (
    PPOConfig,
    build_actor_critic,
    collect_rollout,
    compute_gae,
    parameter_l1_sum,
    ppo_update,
    require_torch,
    synchronize_device,
)

DEFAULT_OUTPUT_JSON = Path(
    ".agent/task/task047-training-eval-correctness-repair/"
    "task047_local_4090_ppo_core_smoke.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--seed", type=int, default=4704090)
    parser.add_argument("--obs-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--ppo-updates", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow a CPU fallback for parser/debug use. Task047 evidence should not set this.",
    )
    return parser.parse_args(argv)


class SyntheticTimeoutEnv:
    """Small tensor-only env with timeout rows and terminal observations."""

    def __init__(self, *, torch: Any, config: PPOConfig, device: str) -> None:
        self.torch = torch
        self.config = config
        self.device = torch.device(device)
        self.step_index = 0
        self.obs = torch.zeros((config.n_envs, config.obs_dim), device=self.device)
        self.env_offsets = torch.arange(config.n_envs, device=self.device, dtype=torch.float32)

    def reset(self) -> Any:
        self.step_index = 0
        self.obs.zero_()
        return self.obs.clone()

    def step(self, action: Any) -> Any:
        self.step_index += 1
        action_features = self.torch.zeros_like(self.obs)
        action_features[:, : self.config.action_dim] = action
        drift = (self.env_offsets.reshape(-1, 1) + float(self.step_index)).sin() * 0.01
        next_obs = 0.82 * self.obs + 0.12 * action_features + drift
        terminal_observation = next_obs.clone()
        truncated = (self.step_index % 3 == 0) & ((self.env_offsets.long() % 4) == 0)
        terminated = self.torch.zeros(
            (self.config.n_envs,),
            dtype=self.torch.bool,
            device=self.device,
        )
        done = terminated | truncated
        reward = 1.0 - 0.05 * action.square().mean(dim=1) - 0.01 * next_obs.square().mean(dim=1)
        episode_lengths = self.torch.full(
            (self.config.n_envs,),
            self.step_index,
            dtype=self.torch.float32,
            device=self.device,
        )
        completed_episode_lengths = episode_lengths[done]
        if bool(done.any().item()):
            next_obs = next_obs.clone()
            next_obs[done] = 0.0
        self.obs = next_obs
        info = {
            "reset_count": int(done.sum().item()),
            "full_env_reset_wave": bool(done.all().item()),
            "episode_lengths": episode_lengths,
            "completed_episode_lengths": completed_episode_lengths,
            "terminal_observation": terminal_observation,
            "components": {
                "root_height": 0.82 + 0.01 * terminal_observation[:, 0].tanh(),
                "upright": self.torch.ones((self.config.n_envs,), device=self.device),
                "tracking_lin_vel": reward,
                "tracking_yaw_rate": reward * 0.5,
                "action_rate_penalty": -action.square().mean(dim=1),
                "joint_velocity_penalty": next_obs.square().mean(dim=1),
                "joint_deviation_penalty": action.abs().mean(dim=1),
                "height_bad": self.torch.zeros(
                    (self.config.n_envs,),
                    dtype=self.torch.bool,
                    device=self.device,
                ),
                "termination_height_bad": self.torch.zeros(
                    (self.config.n_envs,),
                    dtype=self.torch.bool,
                    device=self.device,
                ),
                "tilt_bad": terminated,
            },
        }
        return SimpleNamespace(
            observation=next_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            done=done,
            terminal_observation=terminal_observation,
            info=info,
        )


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not args.allow_cpu and not args.device.startswith("cuda"):
        raise RuntimeError("Task047 local evidence requires a CUDA device")

    torch.manual_seed(args.seed)
    config = PPOConfig(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        hidden_layers=2,
        n_envs=args.n_envs,
        rollout_steps=args.rollout_steps,
        ppo_updates=args.ppo_updates,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
    )
    model = build_actor_critic(config, device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    env = SyntheticTimeoutEnv(torch=torch, config=config, device=args.device)
    observation = env.reset()
    actor_before = parameter_l1_sum(model.actor)
    value_before = parameter_l1_sum(model.value)
    started = time.perf_counter()
    updates: list[dict[str, Any]] = []
    last_batch = None
    for update_idx in range(config.ppo_updates):
        batch = collect_rollout(env, model, observation, config)
        advantages, returns = compute_gae(batch, config)
        diagnostics = ppo_update(model, optimizer, batch, advantages, returns, config)
        observation = batch.next_observation
        last_batch = batch
        updates.append(
            {
                "update": update_idx,
                "env_steps": batch.env_steps,
                "reward_mean": batch.reward_mean,
                "done_count": batch.done_count,
                "timeout_count": batch.timeout_count,
                "collect_time_s": batch.collect_time_s,
                "policy_loss": diagnostics.policy_loss,
                "value_loss": diagnostics.value_loss,
                "entropy": diagnostics.entropy,
                "approx_kl": diagnostics.approx_kl,
                "clip_fraction": diagnostics.clip_fraction,
                "update_samples_per_sec": diagnostics.update_samples_per_sec,
                "observation_device": _tensor_device_name(batch.observations),
                "action_device": _tensor_device_name(batch.actions),
                "terminal_values_device": _tensor_device_name(batch.terminal_values),
            }
        )
    synchronize_device(torch.device(args.device))
    wall_time_s = time.perf_counter() - started
    actor_after = parameter_l1_sum(model.actor)
    value_after = parameter_l1_sum(model.value)
    all_finite = _all_numeric_update_values_finite(updates)
    pass_status = bool(
        all_finite
        and abs(actor_after - actor_before) > 0.0
        and abs(value_after - value_before) > 0.0
        and last_batch is not None
        and _tensor_device_name(last_batch.observations).startswith(args.device.split(":")[0])
        and _tensor_device_name(last_batch.actions).startswith(args.device.split(":")[0])
        and _tensor_device_name(last_batch.terminal_values).startswith(args.device.split(":")[0])
    )
    cuda_device = torch.device(args.device)
    return {
        "pass": pass_status,
        "failure_reasons": [] if pass_status else _failure_reasons(
            all_finite=all_finite,
            actor_before=actor_before,
            actor_after=actor_after,
            value_before=value_before,
            value_after=value_after,
            last_batch=last_batch,
            expected_device_prefix=args.device.split(":")[0],
        ),
        "command": list(sys.argv),
        "host": platform.node(),
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(cuda_device) if args.device.startswith("cuda") else "none",
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_memory_allocated_bytes": (
            int(torch.cuda.memory_allocated(cuda_device)) if args.device.startswith("cuda") else 0
        ),
        "cuda_memory_reserved_bytes": (
            int(torch.cuda.memory_reserved(cuda_device)) if args.device.startswith("cuda") else 0
        ),
        "config": {
            "obs_dim": config.obs_dim,
            "action_dim": config.action_dim,
            "hidden_dim": config.hidden_dim,
            "n_envs": config.n_envs,
            "rollout_steps": config.rollout_steps,
            "ppo_updates": config.ppo_updates,
            "epochs": config.epochs,
            "minibatch_size": config.minibatch_size,
        },
        "wall_time_s": wall_time_s,
        "actor_l1_before": actor_before,
        "actor_l1_after": actor_after,
        "value_l1_before": value_before,
        "value_l1_after": value_after,
        "updates": updates,
        "scope": "local CUDA PPO core only; not a robot locomotion quality claim",
    }


def _tensor_device_name(tensor: Any) -> str:
    return str(getattr(tensor, "device", "none"))


def _all_numeric_update_values_finite(updates: list[dict[str, Any]]) -> bool:
    return all(
        math.isfinite(float(value))
        for row in updates
        for key, value in row.items()
        if isinstance(value, (float, int)) and key != "update"
    )


def _failure_reasons(
    *,
    all_finite: bool,
    actor_before: float,
    actor_after: float,
    value_before: float,
    value_after: float,
    last_batch: Any,
    expected_device_prefix: str,
) -> list[str]:
    reasons: list[str] = []
    if not all_finite:
        reasons.append("nonfinite_update_metric")
    if actor_after == actor_before:
        reasons.append("actor_parameters_unchanged")
    if value_after == value_before:
        reasons.append("value_parameters_unchanged")
    if last_batch is None:
        reasons.append("missing_rollout_batch")
        return reasons
    if not _tensor_device_name(last_batch.observations).startswith(expected_device_prefix):
        reasons.append("rollout_observations_not_on_expected_device")
    if not _tensor_device_name(last_batch.actions).startswith(expected_device_prefix):
        reasons.append("rollout_actions_not_on_expected_device")
    if not _tensor_device_name(last_batch.terminal_values).startswith(expected_device_prefix):
        reasons.append("terminal_values_not_on_expected_device")
    return reasons


def main() -> None:
    args = parse_args()
    result = run_smoke(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
