"""Masked PPO loops for procedural whole-body specialist/shared baselines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

from h200_locomotion_lab.algorithms.ppo import compute_gae, ppo_update
from h200_locomotion_lab.core.whole_body import WholeBodyStep
from h200_locomotion_lab.policies.whole_body_mlp import WholeBodyMLPActorCritic, WholeBodyMLPConfig
from h200_locomotion_lab.robots.whole_body_slots import (
    WHOLE_BODY_ACTION_DIM,
    WHOLE_BODY_ACTOR_OBS_DIM,
)


def _nearest_rank_p95(value: Any) -> float:
    flat = value.detach().reshape(-1).sort().values
    index = max(0, min(flat.numel() - 1, int((0.95 * flat.numel() + 0.9999999999)) - 1))
    return float(flat[index].item())


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
    target_kl: float | None = None
    hard_kl_stop: bool = True
    adaptive_kl: bool = False
    desired_kl: float | None = None
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    log_std_init: float = -1.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if min(self.rollout_steps, self.updates, self.epochs, self.minibatch_size) <= 0:
            raise ValueError("rollout/update sizes must be positive")
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ValueError("target_kl must be positive when provided")


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
        self.observation = torch.as_tensor(env.reset(), dtype=torch.float32, device=self.config.device)
        if self.observation.ndim != 2:
            raise ValueError("whole-body env reset must return [num_envs, obs_dim]")
        obs_dim = int(self.observation.shape[-1])
        self.policy = WholeBodyMLPActorCritic(
            WholeBodyMLPConfig(
                obs_dim=obs_dim,
                action_dim=WHOLE_BODY_ACTION_DIM,
                hidden_dim=self.config.hidden_dim,
                hidden_layers=self.config.hidden_layers,
                log_std_init=self.config.log_std_init,
            ),
            action_mask=action_mask,
            device=self.config.device,
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.active_mask = torch.as_tensor(
            getattr(env, "active_action_mask", action_mask),
            dtype=torch.bool,
            device=self.config.device,
        )
        if self.observation.shape[-1] != obs_dim:
            raise ValueError("whole-body env observation dimension changed during reset")

    def train(self, *, updates: int | None = None) -> list[dict[str, float]]:
        reports: list[dict[str, float]] = []
        for _ in range(updates if updates is not None else self.config.updates):
            batch, rollout_diagnostics = self._collect_rollout()
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
                    **rollout_diagnostics,
                    "policy_loss": diagnostics.policy_loss,
                    "value_loss": diagnostics.value_loss,
                    "entropy": diagnostics.entropy,
                    "approx_kl": diagnostics.approx_kl,
                    "clip_fraction": diagnostics.clip_fraction,
                    "grad_norm": diagnostics.grad_norm,
                    "pre_clip_grad_norm": diagnostics.grad_norm,
                    "grad_norm_is_pre_clip": True,
                    "return_mean": float(returns.mean().item()),
                    "return_std": float(returns.std(unbiased=False).item()),
                    "return_p95": _nearest_rank_p95(returns),
                    "gae_target_mean": float(advantages.mean().item()),
                    "gae_target_std": float(advantages.std(unbiased=False).item()),
                    "gae_target_p95": _nearest_rank_p95(advantages),
                    "value_prediction_mean": float(batch.values.mean().item()),
                    "value_prediction_std": float(batch.values.std(unbiased=False).item()),
                    "early_stopped": diagnostics.early_stopped,
                    "minibatches_attempted": diagnostics.minibatches_attempted,
                    "minibatches_completed": diagnostics.minibatches_completed,
                    "epochs_completed": diagnostics.epochs_completed,
                    "learning_rate": diagnostics.learning_rate,
                    "scheduler_decision": diagnostics.scheduler_decision,
                    "learning_rate_before": diagnostics.learning_rate_before,
                    "learning_rate_after": diagnostics.learning_rate_after,
                    "scheduler_kl": diagnostics.scheduler_kl,
                    "desired_kl": diagnostics.desired_kl,
                    "minibatches": [asdict(record) for record in diagnostics.minibatches],
                }
            )
        return reports

    def _collect_rollout(self) -> tuple[Any, dict[str, Any]]:
        torch = self.torch
        observations: list[Any] = []
        actions: list[Any] = []
        rewards: list[Any] = []
        values: list[Any] = []
        log_probs: list[Any] = []
        raw_actions: list[Any] = []
        old_means: list[Any] = []
        old_log_stds: list[Any] = []
        dones: list[Any] = []
        active_masks: list[Any] = []
        reward_values: list[float] = []
        raw_reward_values: list[float] = []
        ppo_reward_values: list[float] = []
        reward_scale_error_max = 0.0
        reward_zero_mismatch_count = 0
        reward_nonzero_count = 0
        reward_sample_count = 0
        falls = 0.0
        contact_counts = {"left_single": 0, "right_single": 0, "both": 0, "none": 0}
        contact_samples = 0
        touchdown_count = 0.0
        airtime_values: list[float] = []
        slip_values: list[float] = []
        target_would_clamp = 0.0
        actual_clamp = 0.0
        clamp_samples = 0
        action_abs_values: list[float] = []
        reward_component_sums: dict[str, float] = {}
        reward_component_samples = 0
        for _ in range(self.config.rollout_steps):
            action, raw_action, log_prob, _entropy, value, mean, log_std = self.policy.act_with_details(
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
            raw_actions.append(raw_action.detach())
            old_means.append(mean.detach())
            old_log_stds.append(log_std.detach())
            rewards.append(reward)
            values.append(value.detach())
            log_probs.append(log_prob.detach())
            dones.append(done)
            active_masks.append(self.active_mask)
            reward_values.append(float(reward.mean().item()))
            raw_reward = step.metrics.get("raw_total_reward", step.reward)
            raw_reward_tensor = torch.as_tensor(raw_reward, dtype=torch.float32)
            raw_reward_values.append(float(raw_reward_tensor.mean().item()))
            ppo_reward = step.metrics.get("ppo_reward", step.reward)
            ppo_reward_tensor = torch.as_tensor(ppo_reward, dtype=torch.float32)
            ppo_reward_values.append(float(ppo_reward_tensor.mean().item()))
            reward_scale = float(step.metrics.get("reward_scale", 1.0))
            nonzero = raw_reward_tensor != 0.0
            reward_nonzero_count += int(nonzero.sum().item())
            reward_sample_count += int(raw_reward_tensor.numel())
            if bool(nonzero.any()):
                error = (ppo_reward_tensor[nonzero] / raw_reward_tensor[nonzero] - reward_scale).abs()
                reward_scale_error_max = max(reward_scale_error_max, float(error.max().item()))
            reward_zero_mismatch_count += int(((~nonzero) & (ppo_reward_tensor != 0.0)).sum().item())
            if "fall" in step.metrics:
                falls += float(torch.as_tensor(step.metrics["fall"]).sum().item())
            if "foot_contact" in step.metrics:
                foot_contact = torch.as_tensor(step.metrics["foot_contact"], dtype=torch.bool)
                if foot_contact.ndim == 2 and foot_contact.shape[1] >= 2:
                    left = foot_contact[:, 0]
                    right = foot_contact[:, 1]
                    contact_counts["left_single"] += int((left & ~right).sum().item())
                    contact_counts["right_single"] += int((right & ~left).sum().item())
                    contact_counts["both"] += int((left & right).sum().item())
                    contact_counts["none"] += int((~left & ~right).sum().item())
                    contact_samples += int(foot_contact.shape[0])
            if "touchdown" in step.metrics:
                touchdown_count += float(torch.as_tensor(step.metrics["touchdown"]).sum().item())
            if "foot_air_time" in step.metrics:
                airtime = torch.as_tensor(step.metrics["foot_air_time"], dtype=torch.float32)
                airtime_values.append(float(airtime.mean().item()))
            if "foot_planar_speed" in step.metrics and "foot_contact" in step.metrics:
                contact = torch.as_tensor(step.metrics["foot_contact"], dtype=torch.float32)
                speed = torch.as_tensor(step.metrics["foot_planar_speed"], dtype=torch.float32)
                slip_values.append(float((contact * speed.square()).mean().item()))
            if "target_would_clamp" in step.metrics:
                would = torch.as_tensor(step.metrics["target_would_clamp"], dtype=torch.bool)
                target_would_clamp += float(would.sum().item())
                clamp_samples += int(would.numel())
            if "actual_clamp" in step.metrics:
                actual_clamp += float(torch.as_tensor(step.metrics["actual_clamp"], dtype=torch.bool).sum().item())
            active = torch.as_tensor(step.active_action_mask, dtype=torch.bool)
            action_abs_values.append(float(action.detach().cpu()[active].abs().mean().item()))
            components = step.metrics.get("reward_components")
            if isinstance(components, dict):
                reward_component_samples += 1
                for name, component_values in components.items():
                    weighted = component_values.get("weighted") if isinstance(component_values, dict) else None
                    if weighted is not None:
                        reward_component_sums[name] = reward_component_sums.get(name, 0.0) + float(
                            torch.as_tensor(weighted, dtype=torch.float32).mean().item()
                        )
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
            raw_actions=torch.stack(raw_actions),
            old_means=torch.stack(old_means),
            old_log_stds=torch.stack(old_log_stds),
            rewards=torch.stack(rewards),
            dones=torch.stack(dones),
            values=torch.stack(values),
            log_probs=torch.stack(log_probs),
            active_action_mask=torch.stack(active_masks),
            next_value=next_value,
        )
        rollout_diagnostics = {
            "reward_mean": sum(reward_values) / max(1, len(reward_values)),
            "raw_reward_mean": sum(raw_reward_values) / max(1, len(raw_reward_values)),
            "ppo_reward_mean": sum(ppo_reward_values) / max(1, len(ppo_reward_values)),
            "reward_scale_ratio": (
                sum(ppo_reward_values) / sum(raw_reward_values)
                if any(raw_reward_values) else 0.0
            ),
            "reward_scale_error_max": reward_scale_error_max,
            "reward_zero_mismatch_count": reward_zero_mismatch_count,
            "raw_reward_nonzero_count": reward_nonzero_count,
            "reward_sample_count": reward_sample_count,
            "fall_count": falls,
            "contact_counts": contact_counts,
            "contact_samples": contact_samples,
            "touchdown_count": touchdown_count,
            "foot_airtime_mean": sum(airtime_values) / max(1, len(airtime_values)),
            "foot_slip_mean": sum(slip_values) / max(1, len(slip_values)),
            "target_would_clamp_fraction": target_would_clamp / max(1, clamp_samples),
            "actual_clamp_fraction": actual_clamp / max(1, clamp_samples),
            "active_action_abs_mean": sum(action_abs_values) / max(1, len(action_abs_values)),
            "reward_components": {
                name: value / max(1, reward_component_samples)
                for name, value in reward_component_sums.items()
            },
        }
        return batch, rollout_diagnostics


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
