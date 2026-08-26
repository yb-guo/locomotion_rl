"""GRU and Transformer-XL policy cores with explicit trial/context resets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from h200_locomotion_lab.core.whole_body import (
    WHOLE_BODY_ACTION_DIM,
    WHOLE_BODY_ACTOR_OBS_DIM,
)
from h200_locomotion_lab.masked_distribution import sample_masked_tanh_gaussian


@dataclass(frozen=True, slots=True)
class RecurrentPolicyOutput:
    action: Any
    log_prob: Any
    value: Any
    entropy: Any
    state: Any


@dataclass(frozen=True, slots=True)
class GRUState:
    hidden: Any


@dataclass(frozen=True, slots=True)
class TXLState:
    memory: Any

    @property
    def kv_cache(self) -> Any:
        """Compatibility name for rollout-side cached key/value memory."""

        return self.memory


@dataclass(frozen=True, slots=True)
class WholeBodyGRUConfig:
    obs_dim: int = WHOLE_BODY_ACTOR_OBS_DIM
    action_dim: int = WHOLE_BODY_ACTION_DIM
    hidden_dim: int = 256
    log_std_init: float = -1.0
    tanh_eps: float = 1e-6
    reset_memory_every_trial: bool = False


@dataclass(frozen=True, slots=True)
class WholeBodyTXLConfig:
    obs_dim: int = WHOLE_BODY_ACTOR_OBS_DIM
    action_dim: int = WHOLE_BODY_ACTION_DIM
    layers: int = 6
    hidden_dim: int = 256
    attention_heads: int = 8
    segment_length: int = 128
    memory_length: int = 128
    log_std_init: float = -1.0
    tanh_eps: float = 1e-6
    reset_memory_every_trial: bool = False

    def __post_init__(self) -> None:
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.layers <= 0 or self.segment_length <= 0 or self.memory_length < 0:
            raise ValueError("TXL layers and lengths must be positive")


class WholeBodyGRUPolicy:
    """Single-step GRU policy; only context_done clears its hidden state."""

    def __init__(self, *, action_mask: Any, config: WholeBodyGRUConfig | None = None, device: str = "cpu") -> None:
        torch = _require_torch()
        nn = torch.nn
        self.torch = torch
        self.config = config or WholeBodyGRUConfig()
        self.action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device)
        if self.action_mask.ndim != 1 or self.action_mask.shape[0] != self.config.action_dim:
            raise ValueError("GRU action_mask must have shape (action_dim,)")
        self.gru = nn.GRUCell(self.config.obs_dim, self.config.hidden_dim).to(device)
        self.actor = nn.Linear(self.config.hidden_dim, self.config.action_dim).to(device)
        self.value = nn.Linear(self.config.hidden_dim, 1).to(device)
        self.log_std = nn.Parameter(torch.full((self.config.action_dim,), self.config.log_std_init, device=device))

    def parameters(self) -> Any:
        return (*self.gru.parameters(), *self.actor.parameters(), *self.value.parameters(), self.log_std)

    def initial_state(self, batch_size: int, *, device: str | None = None) -> GRUState:
        return GRUState(
            self.torch.zeros(
                batch_size,
                self.config.hidden_dim,
                device=device or self.log_std.device,
            )
        )

    def step(
        self,
        observation: Any,
        state: GRUState | None = None,
        *,
        trial_done: Any | None = None,
        context_done: Any | None = None,
        active_mask: Any | None = None,
        deterministic: bool = False,
    ) -> RecurrentPolicyOutput:
        state = state or self.initial_state(observation.shape[0], device=str(observation.device))
        hidden = reset_recurrent_state(
            state.hidden,
            trial_done=trial_done,
            context_done=context_done,
            reset_on_trial=self.config.reset_memory_every_trial,
        )
        hidden = self.gru(observation, hidden)
        return self._head(hidden, state=GRUState(hidden), active_mask=active_mask, deterministic=deterministic)

    def _head(self, hidden: Any, *, state: Any, active_mask: Any | None, deterministic: bool) -> RecurrentPolicyOutput:
        mean = self.actor(hidden)
        value = self.value(hidden).squeeze(-1)
        log_std = self.log_std.expand_as(mean)
        mask = self.action_mask if active_mask is None else active_mask
        action, log_prob, entropy = sample_masked_tanh_gaussian(
            mean,
            log_std,
            mask,
            deterministic=deterministic,
        )
        return RecurrentPolicyOutput(action, log_prob, value, entropy, state)


class WholeBodyTransformerXLPolicy:
    """Canonical six-layer TXL core with a bounded detached memory cache."""

    def __init__(self, *, action_mask: Any, config: WholeBodyTXLConfig | None = None, device: str = "cpu") -> None:
        torch = _require_torch()
        nn = torch.nn
        self.torch = torch
        self.config = config or WholeBodyTXLConfig()
        self.action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device)
        if self.action_mask.ndim != 1 or self.action_mask.shape[0] != self.config.action_dim:
            raise ValueError("TXL action_mask must have shape (action_dim,)")
        self.input_projection = nn.Linear(self.config.obs_dim, self.config.hidden_dim).to(device)
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.attention_heads,
            dim_feedforward=4 * self.config.hidden_dim,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.config.layers).to(device)
        self.actor = nn.Linear(self.config.hidden_dim, self.config.action_dim).to(device)
        self.value = nn.Linear(self.config.hidden_dim, 1).to(device)
        self.log_std = nn.Parameter(torch.full((self.config.action_dim,), self.config.log_std_init, device=device))

    def parameters(self) -> Any:
        return (
            *self.input_projection.parameters(),
            *self.encoder.parameters(),
            *self.actor.parameters(),
            *self.value.parameters(),
            self.log_std,
        )

    def initial_state(self, batch_size: int, *, device: str | None = None) -> TXLState:
        return TXLState(
            self.torch.zeros(
                0,
                batch_size,
                self.config.hidden_dim,
                device=device or self.log_std.device,
            )
        )

    def step(
        self,
        observation: Any,
        state: TXLState | None = None,
        *,
        trial_done: Any | None = None,
        context_done: Any | None = None,
        active_mask: Any | None = None,
        deterministic: bool = False,
    ) -> RecurrentPolicyOutput:
        state = state or self.initial_state(observation.shape[0], device=str(observation.device))
        memory = reset_txl_memory(
            state.memory,
            trial_done=trial_done,
            context_done=context_done,
            reset_on_trial=self.config.reset_memory_every_trial,
        )
        token = self.input_projection(observation).unsqueeze(1)
        if memory.shape[0] > 0:
            prefix = memory.permute(1, 0, 2)
            sequence = self.encoder(self.torch.cat((prefix, token), dim=1), mask=_causal_mask(prefix.shape[1] + 1, observation.device))
            hidden = sequence[:, -1]
            updated = _last_memory(sequence, self.config.memory_length)
        else:
            sequence = self.encoder(token, mask=_causal_mask(1, observation.device))
            hidden = sequence[:, -1]
            updated = _last_memory(sequence, self.config.memory_length)
        return self._head(hidden, state=TXLState(updated), active_mask=active_mask, deterministic=deterministic)

    def _head(self, hidden: Any, *, state: TXLState, active_mask: Any | None, deterministic: bool) -> RecurrentPolicyOutput:
        mean = self.actor(hidden)
        value = self.value(hidden).squeeze(-1)
        log_std = self.log_std.expand_as(mean)
        mask = self.action_mask if active_mask is None else active_mask
        action, log_prob, entropy = sample_masked_tanh_gaussian(
            mean,
            log_std,
            mask,
            deterministic=deterministic,
        )
        return RecurrentPolicyOutput(action, log_prob, value, entropy, state)


def reset_recurrent_state(
    hidden: Any,
    *,
    trial_done: Any | None = None,
    context_done: Any | None = None,
    reset_on_trial: bool = False,
) -> Any:
    """Clear at context boundaries; optionally expose the reset-trial ablation."""

    if context_done is None and (not reset_on_trial or trial_done is None):
        return hidden
    if context_done is None:
        mask = trial_done.to(device=hidden.device, dtype=hidden.dtype)
    elif trial_done is None or not reset_on_trial:
        mask = context_done.to(device=hidden.device, dtype=hidden.dtype)
    else:
        mask = context_done.to(device=hidden.device, dtype=hidden.dtype) | trial_done.to(
            device=hidden.device, dtype=hidden.dtype
        )
    while mask.ndim < hidden.ndim:
        mask = mask.unsqueeze(-1)
    return hidden * (1.0 - mask)


def reset_txl_memory(
    memory: Any,
    *,
    trial_done: Any | None = None,
    context_done: Any | None = None,
    reset_on_trial: bool = False,
) -> Any:
    if memory.shape[0] == 0 or (context_done is None and (not reset_on_trial or trial_done is None)):
        return memory
    if context_done is None:
        mask = trial_done.to(device=memory.device, dtype=memory.dtype)
    elif trial_done is None or not reset_on_trial:
        mask = context_done.to(device=memory.device, dtype=memory.dtype)
    else:
        mask = context_done.to(device=memory.device, dtype=memory.dtype) | trial_done.to(
            device=memory.device, dtype=memory.dtype
        )
    while mask.ndim < memory.ndim - 1:
        mask = mask.unsqueeze(-1)
    return memory * (1.0 - mask.unsqueeze(0))


def sequence_padding_mask(lengths: Any, max_length: int | None = None) -> Any:
    """Return ``[batch, time]`` true-for-padding mask for sequence PPO."""

    torch = _require_torch()
    lengths = torch.as_tensor(lengths, dtype=torch.long)
    max_length = int(max_length or lengths.max().item())
    steps = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return steps >= lengths.unsqueeze(1)


def masked_sequence_mean(loss: Any, padding_mask: Any) -> Any:
    """Average a per-token loss while excluding padded sequence positions."""

    valid = (~padding_mask).to(dtype=loss.dtype, device=loss.device)
    while valid.ndim < loss.ndim:
        valid = valid.unsqueeze(-1)
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def _causal_mask(length: int, device: Any) -> Any:
    torch = _require_torch()
    return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)


def _last_memory(sequence: Any, memory_length: int) -> Any:
    if memory_length <= 0:
        return sequence[:, :0].detach().permute(1, 0, 2)
    return sequence[:, -memory_length:].detach().permute(1, 0, 2)


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch
