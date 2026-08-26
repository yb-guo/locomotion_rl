"""Masked 45-slot MLP actor-critic used by Task053/054 PPO baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from h200_locomotion_lab.masked_distribution import (
    masked_entropy,
    masked_tanh_gaussian_log_prob,
    sample_masked_tanh_gaussian,
)


@dataclass(frozen=True, slots=True)
class WholeBodyMLPConfig:
    obs_dim: int = 193
    action_dim: int = 45
    hidden_dim: int = 256
    hidden_layers: int = 2
    log_std_init: float = -1.0
    tanh_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.obs_dim <= 0 or self.action_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("network dimensions must be positive")
        if self.hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")


class WholeBodyMLPActorCritic:
    """Torch module with active-slot likelihoods and zeroed inactive actions."""

    def __init__(self, config: WholeBodyMLPConfig | None = None, *, action_mask: Any, device: str = "cpu") -> None:
        torch = _require_torch()
        self.torch = torch
        self.config = config or WholeBodyMLPConfig()
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device)
        if mask.ndim == 1 and tuple(mask.shape) != (self.config.action_dim,):
            raise ValueError("action_mask must have shape (action_dim,)")
        if mask.ndim not in {1, 2} or mask.shape[-1] != self.config.action_dim:
            raise ValueError("action_mask must have shape (action_dim,) or (batch, action_dim)")
        self.action_mask = mask if mask.ndim == 1 else torch.ones(self.config.action_dim, dtype=torch.bool, device=device)
        self.module = _ActorCriticModule(self.config, mask=self.action_mask, device=device)
        # ``_ActorCriticModule`` owns a real ``nn.Module`` internally; move its
        # registered actor/value submodules as well as the mask before the
        # first forward pass.  Without this, CUDA observations hit CPU Linear
        # weights even though ``log_std`` is already on the requested device.
        self.module.to(device)

    def __getattr__(self, name: str) -> Any:
        if name in {"module", "torch", "config", "action_mask"}:
            return object.__getattribute__(self, name)
        return getattr(self.module, name)

    def parameters(self) -> Any:
        return self.module.parameters()

    def to(self, device: str) -> WholeBodyMLPActorCritic:
        self.module.to(device)
        self.action_mask = self.action_mask.to(device)
        return self

    def act(
        self,
        observation: Any,
        *,
        deterministic: bool = False,
        active_mask: Any | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        return self.module.act(observation, deterministic=deterministic, active_mask=active_mask)

    def evaluate_actions(
        self,
        observation: Any,
        action: Any,
        *,
        active_action_mask: Any | None = None,
    ) -> tuple[Any, Any, Any]:
        return self.module.evaluate_actions(observation, action, active_mask=active_action_mask)

    def forward(self, observation: Any) -> tuple[Any, Any]:
        return self.module(observation)

    def state_dict(self) -> Any:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: Any) -> Any:
        return self.module.load_state_dict(state_dict)


class _ActorCriticModule:
    def __init__(self, config: WholeBodyMLPConfig, *, mask: Any, device: str) -> None:
        torch = _require_torch()
        nn = torch.nn
        self._module = nn.Module()
        # Register submodules on a real nn.Module and proxy the module methods
        # through this lightweight wrapper.
        self.actor = _make_mlp(nn, config.obs_dim, config.action_dim, config)
        self.value = _make_mlp(nn, config.obs_dim, 1, config)
        self.log_std = nn.Parameter(torch.full((config.action_dim,), config.log_std_init, device=device))
        self._module.add_module("actor", self.actor)
        self._module.add_module("value", self.value)
        self._module.register_parameter("log_std", self.log_std)
        self.mask = mask
        self.config = config

    def __call__(self, observation: Any) -> tuple[Any, Any]:
        mean = self.actor(observation)
        value = self.value(observation).squeeze(-1)
        return mean, value

    def act(
        self,
        observation: Any,
        *,
        deterministic: bool = False,
        active_mask: Any | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        mean, value = self(observation)
        log_std = self.log_std.expand_as(mean)
        active_mask = self.mask if active_mask is None else active_mask
        action, log_prob, entropy = sample_masked_tanh_gaussian(
            mean,
            log_std,
            active_mask,
            deterministic=deterministic,
            tanh_eps=self.config.tanh_eps,
        )
        return action, log_prob, value, entropy

    def evaluate_actions(
        self,
        observation: Any,
        action: Any,
        *,
        active_mask: Any | None = None,
    ) -> tuple[Any, Any, Any]:
        mean, value = self(observation)
        log_std = self.log_std.expand_as(mean)
        active_mask = self.mask if active_mask is None else active_mask
        log_prob = masked_tanh_gaussian_log_prob(
            action,
            mean,
            log_std,
            active_mask,
            tanh_eps=self.config.tanh_eps,
        )
        entropy = masked_entropy(
            self._normal_entropy(log_std),
            active_mask,
        )
        return log_prob, entropy, value

    def _normal_entropy(self, log_std: Any) -> Any:
        return log_std + 0.5 * math.log(2.0 * math.pi * math.e)

    def parameters(self) -> Any:
        return self._module.parameters()

    def state_dict(self) -> Any:
        return self._module.state_dict()

    def load_state_dict(self, state_dict: Any) -> Any:
        return self._module.load_state_dict(state_dict)

    def to(self, device: str) -> _ActorCriticModule:
        self._module.to(device)
        self.actor = self._module.actor
        self.value = self._module.value
        self.log_std = self._module.log_std
        self.mask = self.mask.to(device)
        return self


def _make_mlp(nn: Any, input_dim: int, output_dim: int, config: WholeBodyMLPConfig) -> Any:
    layers: list[Any] = []
    current = input_dim
    for _ in range(config.hidden_layers):
        layer = nn.Linear(current, config.hidden_dim)
        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(layer.bias)
        layers.extend((layer, nn.Tanh()))
        current = config.hidden_dim
    output = nn.Linear(current, output_dim)
    nn.init.orthogonal_(output.weight, gain=0.01)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch
