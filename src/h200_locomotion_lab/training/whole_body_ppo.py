"""Masked PPO loops for procedural whole-body specialist/shared baselines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from h200_locomotion_lab.algorithms.ppo import compute_gae, ppo_update
from h200_locomotion_lab.core.whole_body import WholeBodyStep
from h200_locomotion_lab.policies.whole_body_mlp import WholeBodyMLPActorCritic, WholeBodyMLPConfig
from h200_locomotion_lab.robots.whole_body_slots import (
    WHOLE_BODY_ACTION_DIM,
    WHOLE_BODY_ACTOR_OBS_DIM,
)


@dataclass(frozen=True, slots=True)
class WholeBodyPPOConfig:
    rollout_steps: int = 128
    updates: int = 1
    hidden_dim: int = 256
    hidden_layers: int = 2
    epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    log_std_init: float = -1.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if min(self.rollout_steps, self.updates, self.epochs, self.minibatch_size) <= 0:
            raise ValueError("rollout/update sizes must be positive")


@dataclass(frozen=True, slots=True)
class SpecialistQualityThresholds:
    zero_fall_ratio: float = 0.95
    normalized_velocity_error: float = 0.25
    non_foot_contact_fraction: float = 0.05
    roll_pitch_p95: float = 0.45


def evaluate_specialist_gate(
    metrics: dict[str, float],
    thresholds: SpecialistQualityThresholds | None = None,
) -> tuple[bool, list[str]]:
    """Apply the predeclared Task053 specialist acceptance gate."""

    thresholds = thresholds or SpecialistQualityThresholds()
    reasons: list[str] = []
    if metrics.get("zero_fall_ratio", 0.0) < thresholds.zero_fall_ratio:
        reasons.append("zero_fall_ratio_below_threshold")
    if metrics.get("normalized_velocity_error", float("inf")) > thresholds.normalized_velocity_error:
        reasons.append("normalized_velocity_error_above_threshold")
    if metrics.get("non_foot_contact_fraction", float("inf")) > thresholds.non_foot_contact_fraction:
        reasons.append("non_foot_contact_fraction_above_threshold")
    if metrics.get("roll_pitch_p95", float("inf")) > thresholds.roll_pitch_p95:
        reasons.append("roll_pitch_p95_above_threshold")
    return not reasons, reasons


class WholeBodyPPOTrainer:
    """Collect fixed-schema rollouts and update one active-mask-aware MLP."""

    def __init__(self, env: Any, *, action_mask: Any, config: WholeBodyPPOConfig | None = None) -> None:
        torch = _require_torch()
        self.torch = torch
        self.env = env
        self.config = config or WholeBodyPPOConfig()
        self.policy = WholeBodyMLPActorCritic(
            WholeBodyMLPConfig(
                obs_dim=WHOLE_BODY_ACTOR_OBS_DIM,
                action_dim=WHOLE_BODY_ACTION_DIM,
                hidden_dim=self.config.hidden_dim,
                hidden_layers=self.config.hidden_layers,
                log_std_init=self.config.log_std_init,
            ),
            action_mask=action_mask,
            device=self.config.device,
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.observation = torch.as_tensor(env.reset(), dtype=torch.float32, device=self.config.device)
        self.active_mask = torch.as_tensor(
            getattr(env, "active_action_mask", action_mask),
            dtype=torch.bool,
            device=self.config.device,
        )
        if self.observation.ndim != 2 or self.observation.shape[-1] != WHOLE_BODY_ACTOR_OBS_DIM:
            raise ValueError("whole-body env reset must return [num_envs, 193]")

    def train(self, *, updates: int | None = None) -> list[dict[str, float]]:
        reports: list[dict[str, float]] = []
        for _ in range(updates if updates is not None else self.config.updates):
            batch, reward_mean, fall_count = self._collect_rollout()
            advantages, returns = compute_gae(batch, self.config)
            diagnostics = ppo_update(
                self.policy,
                self.optimizer,
                batch,
                advantages,
                returns,
                self.config,
            )
            reports.append(
                {
                    "reward_mean": reward_mean,
                    "fall_count": fall_count,
                    "policy_loss": diagnostics.policy_loss,
                    "value_loss": diagnostics.value_loss,
                    "entropy": diagnostics.entropy,
                    "approx_kl": diagnostics.approx_kl,
                    "clip_fraction": diagnostics.clip_fraction,
                }
            )
        return reports

    def _collect_rollout(self) -> tuple[Any, float, float]:
        torch = self.torch
        observations: list[Any] = []
        actions: list[Any] = []
        rewards: list[Any] = []
        values: list[Any] = []
        log_probs: list[Any] = []
        dones: list[Any] = []
        active_masks: list[Any] = []
        reward_values: list[float] = []
        falls = 0.0
        for _ in range(self.config.rollout_steps):
            action, log_prob, value, _entropy = self.policy.act(
                self.observation,
                active_mask=self.active_mask,
            )
            step = self.env.step(action.detach().cpu().numpy())
            if not isinstance(step, WholeBodyStep):
                raise TypeError("whole-body trainer requires WholeBodyStep from env.step")
            next_observation = torch.as_tensor(
                step.actor_observation,
                dtype=torch.float32,
                device=self.config.device,
            )
            reward = torch.as_tensor(step.reward, dtype=torch.float32, device=self.config.device)
            done = torch.as_tensor(step.trial_done, dtype=torch.bool, device=self.config.device)
            observations.append(self.observation)
            actions.append(action.detach())
            rewards.append(reward)
            values.append(value.detach())
            log_probs.append(log_prob.detach())
            dones.append(done)
            active_masks.append(self.active_mask)
            reward_values.append(float(reward.mean().item()))
            if "fall" in step.metrics:
                falls += float(torch.as_tensor(step.metrics["fall"]).sum().item())
            self.observation = next_observation
            self.active_mask = torch.as_tensor(
                step.active_action_mask,
                dtype=torch.bool,
                device=self.config.device,
            )
        with torch.no_grad():
            _, next_value = self.policy.forward(self.observation)
        batch = SimpleNamespace(
            observations=torch.stack(observations),
            actions=torch.stack(actions),
            rewards=torch.stack(rewards),
            dones=torch.stack(dones),
            values=torch.stack(values),
            log_probs=torch.stack(log_probs),
            active_action_mask=torch.stack(active_masks),
            next_value=next_value,
        )
        return batch, sum(reward_values) / max(1, len(reward_values)), falls


def evaluate_whole_body_policy(
    env_factory: Callable[[], Any],
    policy: WholeBodyMLPActorCritic,
    *,
    trials: int = 100,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate fixed-length trials with the plan's quality gates."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    env = env_factory()
    observation = policy.torch.as_tensor(env.reset(), dtype=policy.torch.float32, device=device)
    completed = 0
    falls = 0.0
    errors: list[float] = []
    contacts: list[float] = []
    tilts: list[float] = []
    while completed < trials:
        active_mask = getattr(env, "active_action_mask", None)
        if active_mask is not None:
            active_mask = policy.torch.as_tensor(active_mask, dtype=policy.torch.bool, device=device)
        with policy.torch.no_grad():
            action, _log_prob, _value, _entropy = policy.act(
                observation,
                deterministic=True,
                active_mask=active_mask,
            )
        step = env.step(action.cpu().numpy())
        observation = policy.torch.as_tensor(step.actor_observation, dtype=policy.torch.float32, device=device)
        if "normalized_velocity_error" in step.metrics:
            errors.extend(float(value) for value in step.metrics["normalized_velocity_error"])
        if "non_foot_contact_fraction" in step.metrics:
            contacts.extend(float(value) for value in step.metrics["non_foot_contact_fraction"])
        if "tilt" in step.metrics:
            tilts.extend(float(value) for value in step.metrics["tilt"])
        if "fall" in step.metrics:
            falls += float(policy.torch.as_tensor(step.metrics["fall"]).sum().item())
        done_count = int(policy.torch.as_tensor(step.trial_done).sum().item())
        completed += done_count
    return {
        "trials": float(trials),
        "zero_fall_ratio": max(0.0, 1.0 - falls / max(1.0, trials)),
        "normalized_velocity_error": sum(errors) / max(1, len(errors)),
        "non_foot_contact_fraction": sum(contacts) / max(1, len(contacts)),
        "roll_pitch_p95": _percentile(tilts, 0.95),
    }


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))
    return ordered[index]


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch
