"""Task-independent tanh-Gaussian actor-critic policy implementation."""

from __future__ import annotations

import math
from typing import Any, Protocol


class TanhGaussianSettings(Protocol):
    """Construction values supplied after task and policy specs are composed."""

    obs_dim: int
    action_dim: int
    hidden_dim: int
    hidden_layers: int
    log_std_init: float
    tanh_eps: float


def build_tanh_gaussian_actor_critic(
    config: TanhGaussianSettings,
    *,
    device: str,
) -> Any:
    """Build a policy/value module without importing a task or PPO."""

    torch = _require_torch()
    nn = torch.nn

    class ActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.actor = make_mlp(nn, config.obs_dim, config.action_dim, config)
            self.value = make_mlp(nn, config.obs_dim, 1, config)
            self.log_std = nn.Parameter(
                torch.full((config.action_dim,), config.log_std_init, device=device)
            )

        def forward(self, observation: Any) -> tuple[Any, Any]:
            mean = self.actor(observation)
            value = self.value(observation).squeeze(-1)
            return mean, value

        def act(self, observation: Any) -> tuple[Any, Any, Any, Any]:
            mean, value = self.forward(observation)
            log_std = self.log_std.expand_as(mean)
            std = log_std.exp()
            raw_action = mean + std * torch.randn_like(mean)
            action = torch.tanh(raw_action)
            log_prob = tanh_gaussian_log_prob_from_raw(raw_action, mean, log_std, config)
            entropy = gaussian_entropy(log_std)
            return action, log_prob, value, entropy

        def evaluate_actions(self, observation: Any, action: Any) -> tuple[Any, Any, Any]:
            mean, value = self.forward(observation)
            log_std = self.log_std.expand_as(mean)
            log_prob = tanh_gaussian_log_prob_from_action(action, mean, log_std, config)
            entropy = gaussian_entropy(log_std)
            return log_prob, entropy, value

    return ActorCritic().to(device)


def make_mlp(
    nn: Any,
    input_dim: int,
    output_dim: int,
    config: TanhGaussianSettings,
) -> Any:
    layers: list[Any] = []
    current_dim = input_dim
    for _ in range(config.hidden_layers):
        layer = nn.Linear(current_dim, config.hidden_dim)
        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        layers.append(nn.Tanh())
        current_dim = config.hidden_dim
    output = nn.Linear(current_dim, output_dim)
    nn.init.orthogonal_(output.weight, gain=0.01)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


def tanh_gaussian_log_prob_from_raw(
    raw_action: Any,
    mean: Any,
    log_std: Any,
    config: TanhGaussianSettings,
) -> Any:
    torch = _require_torch()
    std = log_std.exp()
    normal = torch.distributions.Normal(mean, std)
    action = torch.tanh(raw_action)
    log_prob = normal.log_prob(raw_action) - torch.log(
        1.0 - action.square() + config.tanh_eps
    )
    return log_prob.sum(dim=-1)


def tanh_gaussian_log_prob_from_action(
    action: Any,
    mean: Any,
    log_std: Any,
    config: TanhGaussianSettings,
) -> Any:
    torch = _require_torch()
    clipped = action.clamp(-1.0 + config.tanh_eps, 1.0 - config.tanh_eps)
    raw_action = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))
    return tanh_gaussian_log_prob_from_raw(raw_action, mean, log_std, config)


def gaussian_entropy(log_std: Any) -> Any:
    return (log_std + 0.5 * math.log(2.0 * math.pi * math.e)).sum(dim=-1)


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - local no-torch path.
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch
