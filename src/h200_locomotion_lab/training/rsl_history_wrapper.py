"""RSL-RL wrappers for Task033 shared-history policy consumers."""

from __future__ import annotations

from typing import Any, Mapping

from h200_locomotion_lab.training.history_buffer import (
    HistoryBufferConfig,
    TorchHistoryBuffer,
)
from h200_locomotion_lab.training.history_checkpoint_migration import (
    AdaptationConditioningMigrationConfig,
    StackMlpHistoryMigrationConfig,
    is_adaptation_conditioning_migration_needed,
    is_stack_mlp_history_migration_needed,
    migrate_adaptation_conditioned_checkpoint,
    migrate_stack_mlp_checkpoint,
)
from h200_locomotion_lab.training.mjlab_inner_reset import install_task037_inner_reset_controller
from h200_locomotion_lab.training.multitrial_wrapper import (
    EPISODE_DONE_KEY,
    INNER_RESET_KEY,
    OUTER_RESET_KEY,
    RESET_REASON_KEY,
    TASK037_EPISODE_DONE_KEY,
    TASK037_INNER_RESET_KEY,
    TASK037_OUTER_RESET_KEY,
    TASK037_RESET_REASON_KEY,
    TASK037_TRIAL_INDEX_KEY,
    TASK037_ZERO_ACTION_RESET_KEY,
    TRIAL_INDEX_KEY,
    Task037MultiTrialVecEnvWrapper,
)


TASK042_MEMORY_ABLATION_MODES = (
    "none",
    "zero_txl_residual",
    "stateless_txl_memory",
    "zero_memory_latent",
)


def _base_runner() -> type[Any]:
    try:
        from mjlab.rl.runner import MjlabOnPolicyRunner  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - local no-training-stack path.
        return _missing_dependency_base("mjlab.rl.runner.MjlabOnPolicyRunner", exc)
    else:
        return MjlabOnPolicyRunner


def _base_mlp_model() -> type[Any]:
    try:
        from rsl_rl.models.mlp_model import MLPModel  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - local no-training-stack path.
        return _missing_dependency_base("rsl_rl.models.mlp_model.MLPModel", exc)
    else:
        return MLPModel


def _base_ppo() -> type[Any]:
    try:
        from rsl_rl.algorithms import PPO  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - local no-training-stack path.
        return _missing_dependency_base("rsl_rl.algorithms.PPO", exc)
    else:
        return PPO


def _missing_dependency_base(dependency: str, exc: Exception) -> type[Any]:
    class MissingDependencyBase:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(f"{dependency} import failed: {exc}") from exc

    return MissingDependencyBase


class Task033HistoryTokenMlpModel(_base_mlp_model()):
    """Minimal tokenized history model for the LocoFormer-style smoke."""

    is_recurrent: bool = False

    def __init__(
        self,
        *args: Any,
        history_len: int = 4,
        token_dim: int = 128,
        **kwargs: Any,
    ) -> None:
        self.history_len = history_len
        self.token_dim = token_dim
        super().__init__(*args, **kwargs)
        torch = _require_torch()
        nn = torch.nn
        if self.obs_dim % self.history_len != 0:
            raise ValueError(
                f"history actor obs dim {self.obs_dim} is not divisible by history_len={self.history_len}"
            )
        self.frame_dim = self.obs_dim // self.history_len
        self.token_projection = nn.Linear(self.frame_dim, self.token_dim)
        self.time_embedding = nn.Parameter(torch.zeros(self.history_len, self.token_dim))
        nn.init.orthogonal_(self.token_projection.weight)
        nn.init.zeros_(self.token_projection.bias)

    def get_latent(
        self,
        obs: Any,
        masks: Any | None = None,
        hidden_state: Any = None,
    ) -> Any:
        latent = super().get_latent(obs, masks, hidden_state)
        batch = latent.shape[0]
        tokens = latent.reshape(batch, self.history_len, self.frame_dim)
        tokens = self.token_projection(tokens) + self.time_embedding.unsqueeze(0)
        return tokens.mean(dim=1)

    def _get_latent_dim(self) -> int:
        return self.token_dim


class Task036AdaptationConditionedMlpModel(_base_mlp_model()):
    """Actor that conditions the base observation on a learned history latent."""

    is_recurrent: bool = False

    def __init__(
        self,
        *args: Any,
        history_len: int = 4,
        action_dim: int = 31,
        adaptation_latent_dim: int = 32,
        adaptation_hidden_dim: int = 128,
        **kwargs: Any,
    ) -> None:
        self.history_len = history_len
        self.action_dim = action_dim
        self.adaptation_latent_dim = adaptation_latent_dim
        self.adaptation_hidden_dim = adaptation_hidden_dim
        super().__init__(*args, **kwargs)
        torch = _require_torch()
        nn = torch.nn
        if self.obs_dim % self.history_len != 0:
            raise ValueError(
                f"history actor obs dim {self.obs_dim} is not divisible by history_len={self.history_len}"
            )
        self.frame_dim = self.obs_dim // self.history_len
        self.base_obs_dim = self.frame_dim - self.action_dim
        if self.base_obs_dim <= 0:
            raise ValueError(
                f"frame_dim={self.frame_dim} must be larger than action_dim={self.action_dim}"
            )
        self.adaptation_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.adaptation_hidden_dim),
            nn.ELU(),
            nn.Linear(self.adaptation_hidden_dim, self.adaptation_latent_dim),
            nn.Tanh(),
        )
        for module in self.adaptation_encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)

    def get_latent(
        self,
        obs: Any,
        masks: Any | None = None,
        hidden_state: Any = None,
    ) -> Any:
        normalized_history = super().get_latent(obs, masks, hidden_state)
        frames = normalized_history.reshape(normalized_history.shape[0], self.history_len, self.frame_dim)
        newest_base_obs = frames[:, -1, : self.base_obs_dim]
        adaptation_latent = self.adaptation_encoder(normalized_history)
        return _require_torch().cat((newest_base_obs, adaptation_latent), dim=-1)

    def _get_latent_dim(self) -> int:
        frame_dim = self.obs_dim // self.history_len
        return (frame_dim - self.action_dim) + self.adaptation_latent_dim


class Task037TxlStyleMemoryModel(_base_mlp_model()):
    """Segment-token long-context actor for Task037 construction smokes."""

    is_recurrent: bool = False

    def __init__(
        self,
        *args: Any,
        history_len: int = 160,
        segment_len: int = 16,
        token_dim: int = 128,
        **kwargs: Any,
    ) -> None:
        self.history_len = history_len
        self.segment_len = segment_len
        self.token_dim = token_dim
        super().__init__(*args, **kwargs)
        torch = _require_torch()
        nn = torch.nn
        if self.obs_dim % self.history_len != 0:
            raise ValueError(
                f"history actor obs dim {self.obs_dim} is not divisible by history_len={self.history_len}"
            )
        if self.history_len % self.segment_len != 0:
            raise ValueError(
                f"history_len={self.history_len} must be divisible by segment_len={self.segment_len}"
            )
        self.frame_dim = self.obs_dim // self.history_len
        self.segment_count = self.history_len // self.segment_len
        self.token_projection = nn.Linear(self.frame_dim, self.token_dim)
        self.segment_embedding = nn.Parameter(torch.zeros(self.segment_count, self.token_dim))
        self.memory_query = nn.Parameter(torch.zeros(self.token_dim))
        nn.init.orthogonal_(self.token_projection.weight)
        nn.init.zeros_(self.token_projection.bias)
        nn.init.normal_(self.memory_query, mean=0.0, std=0.02)

    def get_latent(
        self,
        obs: Any,
        masks: Any | None = None,
        hidden_state: Any = None,
    ) -> Any:
        torch = _require_torch()
        normalized_history = super().get_latent(obs, masks, hidden_state)
        batch = normalized_history.shape[0]
        frames = normalized_history.reshape(batch, self.history_len, self.frame_dim)
        segments = frames.reshape(batch, self.segment_count, self.segment_len, self.frame_dim)
        segment_means = segments.mean(dim=2)
        tokens = self.token_projection(segment_means) + self.segment_embedding.unsqueeze(0)
        scores = (tokens * self.memory_query.reshape(1, 1, -1)).sum(dim=-1) / (self.token_dim**0.5)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (tokens * weights).sum(dim=1)

    def _get_latent_dim(self) -> int:
        return self.token_dim


class Task038TrueTxlMemoryModel(_base_mlp_model()):
    """Stateful TXL-cache actor for the Task038 runner-consumer smoke."""

    is_recurrent: bool = False

    def __init__(
        self,
        *args: Any,
        history_len: int = 160,
        segment_len: int = 16,
        token_dim: int = 128,
        memory_len: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        action_dim: int = 31,
        memory_latent_dim: int | None = None,
        base_obs_passthrough: bool = False,
        adaptation_warmstart: bool = False,
        adaptation_hidden_dim: int = 128,
        memory_latent_scale: float = 1.0,
        base_obs_passthrough_scale: float = 1.0,
        adaptation_warmstart_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.history_len = history_len
        self.segment_len = segment_len
        self.token_dim = token_dim
        self.memory_len = memory_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.action_dim = action_dim
        self.memory_latent_dim = memory_latent_dim if memory_latent_dim is not None else token_dim
        self.base_obs_passthrough = base_obs_passthrough
        self.adaptation_warmstart = adaptation_warmstart
        self.adaptation_hidden_dim = adaptation_hidden_dim
        self.memory_latent_scale = float(memory_latent_scale)
        self.base_obs_passthrough_scale = float(base_obs_passthrough_scale)
        self.adaptation_warmstart_scale = float(adaptation_warmstart_scale)
        super().__init__(*args, **kwargs)
        torch = _require_torch()
        nn = torch.nn
        if self.obs_dim % self.history_len != 0:
            raise ValueError(
                f"history actor obs dim {self.obs_dim} is not divisible by history_len={self.history_len}"
            )
        if not 0 < self.segment_len <= self.history_len:
            raise ValueError(
                f"segment_len={self.segment_len} must be in (0, history_len={self.history_len}]"
            )
        if self.memory_len <= 0:
            raise ValueError(f"memory_len must be positive, got {self.memory_len}")
        self.frame_dim = self.obs_dim // self.history_len
        self.base_obs_dim = self.frame_dim - self.action_dim
        if self.base_obs_passthrough and self.base_obs_dim <= 0:
            raise ValueError(
                f"frame_dim={self.frame_dim} must be larger than action_dim={self.action_dim}"
            )
        self.token_projection = nn.Linear(self.frame_dim, self.token_dim)
        self.position_embedding = nn.Parameter(torch.zeros(self.segment_len, self.token_dim))
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.token_dim,
                    self.num_heads,
                    batch_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(self.token_dim) for _ in range(self.num_layers)])
        self.memory_output_projection = (
            nn.Identity()
            if self.memory_latent_dim == self.token_dim
            else nn.Linear(self.token_dim, self.memory_latent_dim)
        )
        self.adaptation_encoder = None
        if self.adaptation_warmstart:
            self.adaptation_encoder = nn.Sequential(
                nn.Linear(self.obs_dim, self.adaptation_hidden_dim),
                nn.ELU(),
                nn.Linear(self.adaptation_hidden_dim, self.memory_latent_dim),
                nn.Tanh(),
            )
            for module in self.adaptation_encoder:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight)
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.token_projection.weight)
        nn.init.zeros_(self.token_projection.bias)
        if isinstance(self.memory_output_projection, nn.Linear):
            if self.adaptation_warmstart:
                nn.init.zeros_(self.memory_output_projection.weight)
            else:
                nn.init.orthogonal_(self.memory_output_projection.weight)
            nn.init.zeros_(self.memory_output_projection.bias)
        self._memory_tensors: list[Any] = []
        self._memory_lengths: list[Any] = []
        self._inner_reset_events: Any | None = None
        self._outer_reset_events: Any | None = None
        self._incremental_steps: Any | None = None
        self._segments_appended: Any | None = None
        self._tokens_appended: Any | None = None
        self._last_attended_previous_memory_lengths: list[list[int]] = []
        self._total_actor_forward_batches = 0
        self._total_actor_forward_samples = 0
        self._env_cache_stateful_forward_batches = 0
        self._env_cache_stateful_forward_samples = 0
        self._stateless_forward_batches = 0
        self._stateless_forward_samples = 0
        self._sequence_update_forward_batches = 0
        self._sequence_update_forward_samples = 0
        self._sequence_update_forward_steps = 0
        self._sequence_update_reset_events = 0
        self._task042_memory_ablation_mode = "none"
        self._last_txl_residual_output_norm: float | None = None
        self._last_txl_residual_raw_norm: float | None = None
        self._last_adaptation_output_norm: float | None = None
        self._last_policy_memory_latent_norm: float | None = None
        self._task040_last_sequence_latents: Any | None = None
        self._task040_last_sequence_memory_latents: Any | None = None

    def get_latent(
        self,
        obs: Any,
        masks: Any | None = None,
        hidden_state: Any = None,
    ) -> Any:
        torch = _require_torch()
        normalized_history = super().get_latent(obs, masks, hidden_state)
        batch = int(normalized_history.shape[0])
        if self._task042_memory_ablation_mode == "stateless_txl_memory":
            self._ensure_cache(batch, normalized_history.device, normalized_history.dtype)
            return self._get_stateless_latent(normalized_history)
        if self._memory_tensors and batch != int(self._memory_tensors[0].shape[0]):
            return self._get_stateless_latent(normalized_history)
        self._ensure_cache(batch, normalized_history.device, normalized_history.dtype)
        frames = normalized_history.reshape(batch, self.history_len, self.frame_dim)
        segment = frames[:, -self.segment_len :, :]
        tokens = self.token_projection(segment) + self.position_embedding.unsqueeze(0)

        previous_lengths_per_layer: list[Any] = []
        hidden = tokens
        for layer_id, (attention, norm) in enumerate(zip(self.attention_layers, self.norm_layers)):
            memory = self._memory_tensors[layer_id]
            lengths = self._memory_lengths[layer_id].clone()
            previous_lengths_per_layer.append(lengths)
            key_value = torch.cat((memory.detach(), hidden), dim=1)
            valid_memory = (
                torch.arange(self.memory_len, device=hidden.device).unsqueeze(0)
                < lengths.unsqueeze(1)
            )
            valid_segment = torch.ones(
                (batch, self.segment_len),
                device=hidden.device,
                dtype=torch.bool,
            )
            key_padding_mask = ~torch.cat((valid_memory, valid_segment), dim=1)
            attended, _ = attention(
                hidden,
                key_value,
                key_value,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            hidden = norm(hidden + attended)
            self._append_layer_memory(layer_id, hidden.detach())

        assert self._incremental_steps is not None
        assert self._segments_appended is not None
        assert self._tokens_appended is not None
        self._incremental_steps += 1
        self._segments_appended += 1
        self._tokens_appended += self.segment_len
        stacked_lengths = torch.stack(previous_lengths_per_layer, dim=1)
        self._last_attended_previous_memory_lengths = [
            [int(value) for value in row] for row in stacked_lengths.detach().cpu().tolist()
        ]
        self._total_actor_forward_batches += 1
        self._total_actor_forward_samples += batch
        self._env_cache_stateful_forward_batches += 1
        self._env_cache_stateful_forward_samples += batch
        return self._compose_policy_latent(frames, hidden.mean(dim=1), normalized_history)

    def txl_debug_snapshot(self) -> dict[str, Any]:
        self._ensure_cache(num_envs=0, device=None, dtype=None)
        assert self._inner_reset_events is not None
        assert self._outer_reset_events is not None
        assert self._incremental_steps is not None
        assert self._segments_appended is not None
        assert self._tokens_appended is not None
        envs = []
        num_envs = int(self._incremental_steps.shape[0])
        for env_id in range(num_envs):
            envs.append(
                {
                    "env_id": env_id,
                    "memory_lengths": [
                        int(lengths[env_id].detach().cpu().item())
                        for lengths in self._memory_lengths
                    ],
                    "inner_reset_events": int(self._inner_reset_events[env_id].detach().cpu().item()),
                    "outer_reset_events": int(self._outer_reset_events[env_id].detach().cpu().item()),
                    "incremental_steps": int(self._incremental_steps[env_id].detach().cpu().item()),
                    "segments_appended": int(self._segments_appended[env_id].detach().cpu().item()),
                    "tokens_appended": int(self._tokens_appended[env_id].detach().cpu().item()),
                    "last_attended_previous_memory_lengths": (
                        self._last_attended_previous_memory_lengths[env_id]
                        if env_id < len(self._last_attended_previous_memory_lengths)
                        else [0 for _ in range(self.num_layers)]
                    ),
                }
            )
        return {
            "num_layers": self.num_layers,
            "memory_len": self.memory_len,
            "segment_len": self.segment_len,
            "envs": envs,
            "last_attended_previous_memory_lengths": self._last_attended_previous_memory_lengths,
            "segments_appended": int(self._segments_appended.sum().detach().cpu().item()),
            "tokens_appended": int(self._tokens_appended.sum().detach().cpu().item()),
            "total_actor_forward_batches": int(self._total_actor_forward_batches),
            "total_actor_forward_samples": int(self._total_actor_forward_samples),
            "env_cache_stateful_forward_batches": int(
                self._env_cache_stateful_forward_batches
            ),
            "env_cache_stateful_forward_samples": int(
                self._env_cache_stateful_forward_samples
            ),
            "stateless_forward_batches": int(self._stateless_forward_batches),
            "stateless_forward_samples": int(self._stateless_forward_samples),
            "stateless_fallback_forward_batches": int(self._stateless_forward_batches),
            "stateless_fallback_forward_samples": int(self._stateless_forward_samples),
            "sequence_update_forward_batches": int(
                getattr(self, "_sequence_update_forward_batches", 0)
            ),
            "sequence_update_forward_samples": int(
                getattr(self, "_sequence_update_forward_samples", 0)
            ),
            "sequence_update_forward_steps": int(
                getattr(self, "_sequence_update_forward_steps", 0)
            ),
            "sequence_update_reset_events": int(
                getattr(self, "_sequence_update_reset_events", 0)
            ),
            "task042_memory_ablation_mode": self._task042_memory_ablation_mode,
            "memory_residual_enabled": self._task042_memory_ablation_mode
            not in {"zero_txl_residual", "zero_memory_latent"},
            "memory_latent_enabled": self._task042_memory_ablation_mode != "zero_memory_latent",
            "stateful_memory_enabled": self._task042_memory_ablation_mode != "stateless_txl_memory",
            "base_obs_passthrough_scale": self.base_obs_passthrough_scale,
            "adaptation_warmstart_scale": self.adaptation_warmstart_scale,
            "txl_residual_output_norm_last": self._last_txl_residual_output_norm,
            "txl_residual_raw_norm_last": self._last_txl_residual_raw_norm,
            "adaptation_output_norm_last": self._last_adaptation_output_norm,
            "policy_memory_latent_norm_last": self._last_policy_memory_latent_norm,
        }

    def task042_set_memory_ablation_mode(self, mode: str) -> None:
        if mode not in TASK042_MEMORY_ABLATION_MODES:
            raise ValueError(
                f"unsupported Task042 memory ablation mode {mode!r}; "
                f"expected one of {TASK042_MEMORY_ABLATION_MODES}"
            )
        self._task042_memory_ablation_mode = mode

    def task038_txl_record_inner_reset(self, env_ids: Any) -> None:
        env_ids_tensor = self._env_ids_tensor(env_ids)
        if env_ids_tensor is None:
            return
        assert self._inner_reset_events is not None
        self._prepare_cache_tensors_for_mutation()
        self._inner_reset_events[env_ids_tensor] += 1

    def task038_txl_outer_reset(self, env_ids: Any) -> None:
        env_ids_tensor = self._env_ids_tensor(env_ids)
        if env_ids_tensor is None:
            return
        assert self._outer_reset_events is not None
        self._prepare_cache_tensors_for_mutation()
        self._outer_reset_events[env_ids_tensor] += 1
        for lengths in self._memory_lengths:
            lengths[env_ids_tensor] = 0

    def task038_txl_clear_memory(self, env_ids: Any | None = None) -> None:
        env_ids_tensor = self._all_env_ids_tensor() if env_ids is None else self._env_ids_tensor(env_ids)
        if env_ids_tensor is None:
            return
        self._prepare_cache_tensors_for_mutation()
        for lengths in self._memory_lengths:
            lengths[env_ids_tensor] = 0

    def task040_forward_sequence(
        self,
        obs: Any,
        *,
        reset_mask: Any | None = None,
        stochastic_output: bool = False,
    ) -> Any:
        """Forward a rollout sequence without using or mutating the inference cache.

        PPO update batches arrive as rollout tensors with time and environment
        axes. Feeding them through ``get_latent`` flattens the rollout and loses
        the temporal cache contract, so Task040 provides an explicit sequence
        path used only by the update algorithm.
        """

        latents = self.task040_get_sequence_latent(obs, reset_mask=reset_mask)
        self._task040_last_sequence_latents = latents
        mlp_output = self.mlp(latents)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def task040_get_sequence_latent(self, obs: Any, *, reset_mask: Any | None = None) -> Any:
        torch = _require_torch()
        normalized_history = self._task040_normalized_sequence_history(obs)
        if len(normalized_history.shape) != 3:
            raise ValueError(
                "Task040 sequence history must have shape [time, batch, obs_dim], "
                f"got {tuple(normalized_history.shape)}"
            )
        steps = int(normalized_history.shape[0])
        batch = int(normalized_history.shape[1])
        if int(normalized_history.shape[2]) != self.obs_dim:
            raise ValueError(
                f"Task040 sequence obs dim mismatch: got {int(normalized_history.shape[2])}, "
                f"expected {self.obs_dim}"
            )
        reset_mask = _task040_sequence_reset_mask(
            torch,
            reset_mask,
            steps=steps,
            batch=batch,
            device=normalized_history.device,
        )

        memory_tensors = [
            torch.zeros(
                (batch, self.memory_len, self.token_dim),
                device=normalized_history.device,
                dtype=normalized_history.dtype,
            )
            for _ in range(self.num_layers)
        ]
        memory_lengths = [
            torch.zeros(batch, device=normalized_history.device, dtype=torch.long)
            for _ in range(self.num_layers)
        ]

        latents = []
        memory_latents = []
        for step in range(steps):
            step_reset_mask = reset_mask[step]
            if bool(step_reset_mask.any().item()):
                reset_ids = step_reset_mask.nonzero(as_tuple=False).flatten()
                self._sequence_update_reset_events += int(reset_ids.shape[0])
                for lengths in memory_lengths:
                    lengths[reset_ids] = 0

            frames = normalized_history[step].reshape(batch, self.history_len, self.frame_dim)
            segment = frames[:, -self.segment_len :, :]
            hidden = self.token_projection(segment) + self.position_embedding.unsqueeze(0)

            for layer_id, (attention, norm) in enumerate(zip(self.attention_layers, self.norm_layers)):
                memory = memory_tensors[layer_id]
                lengths = memory_lengths[layer_id]
                key_value = torch.cat((memory.detach(), hidden), dim=1)
                valid_memory = (
                    torch.arange(self.memory_len, device=hidden.device).unsqueeze(0)
                    < lengths.unsqueeze(1)
                )
                valid_segment = torch.ones(
                    (batch, self.segment_len),
                    device=hidden.device,
                    dtype=torch.bool,
                )
                key_padding_mask = ~torch.cat((valid_memory, valid_segment), dim=1)
                attended, _ = attention(
                    hidden,
                    key_value,
                    key_value,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                hidden = norm(hidden + attended)
                memory_tensors[layer_id], memory_lengths[layer_id] = _task040_append_sequence_memory(
                    torch,
                    memory,
                    lengths,
                    hidden.detach(),
                    memory_len=self.memory_len,
                )

            memory_latent = self._compose_memory_latent(
                memory_summary=hidden.mean(dim=1),
                normalized_history=normalized_history[step],
            )
            memory_latents.append(memory_latent)
            latents.append(self._compose_policy_latent_from_memory(frames, memory_latent))

        self._sequence_update_forward_batches += 1
        self._sequence_update_forward_steps += steps
        self._sequence_update_forward_samples += steps * batch
        self._task040_last_sequence_memory_latents = torch.stack(memory_latents, dim=0).reshape(
            steps * batch,
            self.memory_latent_dim,
        )
        return torch.stack(latents, dim=0).reshape(steps * batch, self._get_latent_dim())

    def task044_memory_latents_from_last_sequence(self) -> Any | None:
        return self._task040_last_sequence_memory_latents

    def _get_latent_dim(self) -> int:
        base_dim = 0
        if self.base_obs_passthrough:
            frame_dim = self.obs_dim // self.history_len
            base_dim = frame_dim - self.action_dim
        return base_dim + self.memory_latent_dim

    def _get_stateless_latent(self, normalized_history: Any) -> Any:
        batch = int(normalized_history.shape[0])
        frames = normalized_history.reshape(batch, self.history_len, self.frame_dim)
        segment = frames[:, -self.segment_len :, :]
        hidden = self.token_projection(segment) + self.position_embedding.unsqueeze(0)
        for attention, norm in zip(self.attention_layers, self.norm_layers):
            attended, _ = attention(
                hidden,
                hidden,
                hidden,
                need_weights=False,
            )
            hidden = norm(hidden + attended)
        self._stateless_forward_batches += 1
        self._stateless_forward_samples += batch
        self._total_actor_forward_batches += 1
        self._total_actor_forward_samples += batch
        return self._compose_policy_latent(frames, hidden.mean(dim=1), normalized_history)

    def _compose_policy_latent(self, frames: Any, memory_summary: Any, normalized_history: Any) -> Any:
        memory_latent = self._compose_memory_latent(
            memory_summary=memory_summary,
            normalized_history=normalized_history,
        )
        return self._compose_policy_latent_from_memory(frames, memory_latent)

    def _compose_memory_latent(self, *, memory_summary: Any, normalized_history: Any) -> Any:
        torch = _require_torch()
        txl_residual = self.memory_output_projection(memory_summary)
        self._last_txl_residual_raw_norm = _task042_tensor_mean_l2_norm(torch, txl_residual)
        if self._task042_memory_ablation_mode in {"zero_txl_residual", "zero_memory_latent"}:
            txl_residual = torch.zeros_like(txl_residual)
        self._last_txl_residual_output_norm = _task042_tensor_mean_l2_norm(torch, txl_residual)
        memory_latent = txl_residual
        if self.adaptation_encoder is not None:
            adaptation_latent = self.adaptation_encoder(normalized_history)
            if self.adaptation_warmstart_scale != 1.0:
                adaptation_latent = adaptation_latent * self.adaptation_warmstart_scale
            self._last_adaptation_output_norm = _task042_tensor_mean_l2_norm(torch, adaptation_latent)
            memory_latent = adaptation_latent + memory_latent
        else:
            self._last_adaptation_output_norm = None
        if self._task042_memory_ablation_mode == "zero_memory_latent":
            memory_latent = torch.zeros_like(memory_latent)
        if self.memory_latent_scale != 1.0:
            memory_latent = memory_latent * self.memory_latent_scale
        self._last_policy_memory_latent_norm = _task042_tensor_mean_l2_norm(torch, memory_latent)
        return memory_latent

    def _compose_policy_latent_from_memory(self, frames: Any, memory_latent: Any) -> Any:
        torch = _require_torch()
        if int(memory_latent.shape[-1]) != self.memory_latent_dim:
            raise ValueError(
                "memory latent dim mismatch: "
                f"got {int(memory_latent.shape[-1])}, expected {self.memory_latent_dim}"
            )
        if not self.base_obs_passthrough:
            return memory_latent
        newest_base_obs = frames[:, -1, : self.base_obs_dim]
        if self.base_obs_passthrough_scale != 1.0:
            newest_base_obs = newest_base_obs * self.base_obs_passthrough_scale
        return torch.cat((newest_base_obs, memory_latent), dim=-1)

    def _task040_normalized_sequence_history(self, obs: Any) -> Any:
        torch = _require_torch()
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        history = torch.cat(obs_list, dim=-1)
        if len(history.shape) == 2:
            history = history.unsqueeze(0)
        if len(history.shape) != 3:
            raise ValueError(
                "Task040 sequence obs groups must concatenate to [time, batch, obs_dim], "
                f"got {tuple(history.shape)}"
            )
        steps = int(history.shape[0])
        batch = int(history.shape[1])
        flattened = history.reshape(steps * batch, int(history.shape[-1]))
        normalized = self.obs_normalizer(flattened)
        return normalized.reshape(steps, batch, int(normalized.shape[-1]))

    def _ensure_cache(self, num_envs: int, device: Any, dtype: Any) -> None:
        if self._memory_tensors:
            return
        if num_envs <= 0:
            return
        torch = _require_torch()
        self._memory_tensors = [
            torch.zeros(
                (num_envs, self.memory_len, self.token_dim),
                device=device,
                dtype=dtype,
            )
            for _ in range(self.num_layers)
        ]
        self._memory_lengths = [
            torch.zeros(num_envs, device=device, dtype=torch.long) for _ in range(self.num_layers)
        ]
        self._inner_reset_events = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._outer_reset_events = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._incremental_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._segments_appended = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._tokens_appended = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._last_attended_previous_memory_lengths = [
            [0 for _ in range(self.num_layers)] for _ in range(num_envs)
        ]

    def _prepare_cache_tensors_for_mutation(self) -> None:
        self._memory_tensors = [
            _task038_clone_inference_tensor(tensor) for tensor in self._memory_tensors
        ]
        self._memory_lengths = [
            _task038_clone_inference_tensor(lengths) for lengths in self._memory_lengths
        ]
        self._inner_reset_events = _task038_clone_inference_tensor(self._inner_reset_events)
        self._outer_reset_events = _task038_clone_inference_tensor(self._outer_reset_events)
        self._incremental_steps = _task038_clone_inference_tensor(self._incremental_steps)
        self._segments_appended = _task038_clone_inference_tensor(self._segments_appended)
        self._tokens_appended = _task038_clone_inference_tensor(self._tokens_appended)

    def _append_layer_memory(self, layer_id: int, segment_tokens: Any) -> None:
        torch = _require_torch()
        self._prepare_cache_tensors_for_mutation()
        memory = self._memory_tensors[layer_id]
        lengths = self._memory_lengths[layer_id]
        updated = torch.zeros_like(memory)
        for env_id in range(int(segment_tokens.shape[0])):
            length = int(lengths[env_id].detach().cpu().item())
            combined = torch.cat((memory[env_id, :length], segment_tokens[env_id]), dim=0)
            if int(combined.shape[0]) > self.memory_len:
                combined = combined[-self.memory_len :]
            new_len = int(combined.shape[0])
            updated[env_id, :new_len] = combined
            lengths[env_id] = new_len
        self._memory_tensors[layer_id] = updated

    def _env_ids_tensor(self, env_ids: Any) -> Any | None:
        if not self._memory_lengths:
            return None
        torch = _require_torch()
        device = self._memory_lengths[0].device
        if hasattr(env_ids, "nonzero") and getattr(env_ids, "dtype", None) == torch.bool:
            env_ids = env_ids.to(device=device).nonzero(as_tuple=False).flatten()
        elif hasattr(env_ids, "to"):
            env_ids = env_ids.to(device=device, dtype=torch.long).flatten()
        elif isinstance(env_ids, int):
            env_ids = torch.as_tensor([env_ids], device=device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(list(env_ids), device=device, dtype=torch.long).flatten()
        return env_ids

    def _all_env_ids_tensor(self) -> Any | None:
        if not self._memory_lengths:
            return None
        torch = _require_torch()
        return torch.arange(
            int(self._memory_lengths[0].shape[0]),
            device=self._memory_lengths[0].device,
            dtype=torch.long,
        )


class Task033HistoryVecEnvWrapper:
    """Add an actor-visible history observation group without changing the env."""

    def __init__(
        self,
        env: Any,
        *,
        history_len: int,
        actor_group: str = "actor",
        history_group: str = "actor_history",
        clear_history_on_inner_reset: bool = False,
    ) -> None:
        self.env = env
        self.history_len = history_len
        self.actor_group = actor_group
        self.history_group = history_group
        self.clear_history_on_inner_reset = clear_history_on_inner_reset
        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.num_actions = env.num_actions
        self._buffer: TorchHistoryBuffer | None = None
        self._initialized = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def cfg(self) -> Any:
        return self.env.cfg

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> Any:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: Any) -> None:
        self.env.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.env.seed(seed)

    def reset(self) -> tuple[Any, dict[str, Any]]:
        obs, extras = self.env.reset()
        self._reset_buffer()
        self._append_from_obs(obs, self._zero_actions())
        self._initialized = True
        return self._with_history(obs), extras

    def get_observations(self) -> Any:
        obs = self.env.get_observations()
        if not self._initialized:
            self._reset_buffer()
            self._append_from_obs(obs, self._zero_actions())
            self._initialized = True
        return self._with_history(obs)

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        obs, rewards, dones, extras = self.env.step(actions)
        history_done = dones
        if self.clear_history_on_inner_reset:
            history_done = _combined_reset_mask(
                dones,
                extras.get(TASK037_INNER_RESET_KEY),
                actions.device,
                self.num_envs,
            )
        self._append_from_obs(
            obs,
            actions,
            done=history_done,
            zero_action_mask=extras.get(TASK037_ZERO_ACTION_RESET_KEY),
        )
        return self._with_history(obs), rewards, dones, extras

    def close(self) -> None:
        return self.env.close()

    def _reset_buffer(self) -> None:
        torch = _require_torch()
        actor = self.env.get_observations()[self.actor_group]
        frame_dim = actor.shape[-1] + self.num_actions
        self._buffer = TorchHistoryBuffer(
            HistoryBufferConfig(
                num_envs=self.num_envs,
                history_len=self.history_len,
                frame_dim=frame_dim,
                device=str(actor.device),
                dtype=_dtype_name(torch, actor.dtype),
            )
        )

    def _append_from_obs(
        self,
        obs: Any,
        actions: Any,
        *,
        done: Any | None = None,
        zero_action_mask: Any | None = None,
    ) -> None:
        if self._buffer is None:
            self._reset_buffer()
        history_actions = actions.to(obs[self.actor_group].device)
        reset_action_mask = _combined_reset_mask(
            done,
            zero_action_mask,
            history_actions.device,
            self.num_envs,
        )
        if reset_action_mask is not None:
            history_actions = history_actions.clone()
            history_actions[reset_action_mask] = 0.0
        frame = _require_torch().cat((obs[self.actor_group], history_actions), dim=-1)
        assert self._buffer is not None
        self._buffer.append(frame, done=done)

    def _with_history(self, obs: Any) -> Any:
        assert self._buffer is not None
        obs[self.history_group] = self._buffer.flatten_latest()
        return obs

    def _zero_actions(self) -> Any:
        torch = _require_torch()
        return torch.zeros((self.num_envs, self.num_actions), device=self.device)


class Task038TrueTxlResetHookVecEnvWrapper:
    """Dispatch Task037 reset extras to a Task038 true-TXL actor cache."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.num_actions = env.num_actions
        self.actor: Any | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def cfg(self) -> Any:
        return self.env.cfg

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> Any:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: Any) -> None:
        self.env.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.env.seed(seed)

    def reset(self) -> tuple[Any, dict[str, Any]]:
        obs, extras = self.env.reset()
        self._clear_all_actor_memory()
        return obs, extras

    def get_observations(self) -> Any:
        return self.env.get_observations()

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        obs, rewards, dones, extras = self.env.step(actions)
        self._dispatch_reset_hooks(extras)
        return obs, rewards, dones, extras

    def close(self) -> None:
        return self.env.close()

    def attach_task038_txl_actor(self, actor: Any) -> None:
        self.actor = actor

    def _dispatch_reset_hooks(self, extras: dict[str, Any]) -> None:
        actor = self.actor
        if actor is None:
            return
        outer_mask = _any_extras_mask(
            extras,
            (OUTER_RESET_KEY, EPISODE_DONE_KEY, TASK037_OUTER_RESET_KEY, TASK037_EPISODE_DONE_KEY),
            self.device,
            self.num_envs,
        )
        if outer_mask is not None and bool(outer_mask.any().item()):
            actor.task038_txl_outer_reset(_mask_to_env_ids(outer_mask))

        inner_mask = _any_extras_mask(
            extras,
            (INNER_RESET_KEY, TASK037_INNER_RESET_KEY),
            self.device,
            self.num_envs,
        )
        if inner_mask is not None and bool(inner_mask.any().item()):
            actor.task038_txl_record_inner_reset(_mask_to_env_ids(inner_mask))

    def _clear_all_actor_memory(self) -> None:
        actor = self.actor
        if actor is None:
            return
        clear = getattr(actor, "task038_txl_clear_memory", None)
        if callable(clear):
            clear()
            return
        actor.task038_txl_outer_reset(_all_env_ids(self.num_envs, self.device))


class Task044FaultLabelVecEnvWrapper:
    """Expose hidden fault labels as a privileged non-actor observation group."""

    label_group = "task044_fault_label"
    trial_step_group = "task044_trial_step"
    trial_index_group = "task044_trial_index"

    def __init__(self, env: Any) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.num_actions = env.num_actions

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def cfg(self) -> Any:
        return self.env.cfg

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> Any:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: Any) -> None:
        self.env.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.env.seed(seed)

    def reset(self) -> tuple[Any, dict[str, Any]]:
        obs, extras = self.env.reset()
        return self._with_fault_label(obs), extras

    def get_observations(self) -> Any:
        return self._with_fault_label(self.env.get_observations())

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        obs, rewards, dones, extras = self.env.step(actions)
        return self._with_fault_label(obs), rewards, dones, extras

    def close(self) -> None:
        return self.env.close()

    def _with_fault_label(self, obs: Any) -> Any:
        obs[self.label_group] = self._fault_label()
        obs[self.trial_step_group] = self._trial_tensor("trial_step")
        obs[self.trial_index_group] = self._trial_tensor("trial_index")
        return obs

    def _fault_label(self) -> Any:
        torch = _require_torch()
        target_index = getattr(self.unwrapped, "_task030_dynamic_failure_target_index", None)
        if target_index is None:
            return torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        labels = target_index.to(device=self.device, dtype=torch.long) + 1
        labels = torch.clamp(labels, min=0)
        return labels.to(dtype=torch.float32).reshape(self.num_envs, 1)

    def _trial_tensor(self, name: str) -> Any:
        torch = _require_torch()
        value = getattr(self.env, name, None)
        if value is None:
            return torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        return value.to(device=self.device, dtype=torch.float32).reshape(self.num_envs, 1)


TASK046_RETRY_CONTEXT_FEATURE_NAMES = (
    "retry_trial_index_norm",
    "retry_is_final_trial",
    "retry_trial_step_norm",
    "retry_just_inner_reset",
    "retry_last_reset_was_fall",
    "retry_last_reset_was_timeout",
)


class Task046RetryContextVecEnvWrapper:
    """Append retry/reset context to actor observations without exposing fault identity."""

    context_group = "task046_retry_context"

    def __init__(
        self,
        env: Any,
        *,
        num_trials: int = 3,
        final_trial_index: int = 2,
        step_window_steps: int = 50,
    ) -> None:
        if num_trials <= 0:
            raise ValueError("num_trials must be positive")
        if final_trial_index < 0:
            raise ValueError("final_trial_index must be non-negative")
        if step_window_steps <= 0:
            raise ValueError("step_window_steps must be positive")
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.num_actions = env.num_actions
        self.num_trials = int(num_trials)
        self.final_trial_index = int(final_trial_index)
        self.step_window_steps = int(step_window_steps)
        torch = _require_torch()
        self._last_inner_reset_reason = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self._last_inner_reset_mask = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        self._base_actor_dim: int | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def cfg(self) -> Any:
        return self.env.cfg

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> Any:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: Any) -> None:
        self.env.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.env.seed(seed)

    def reset(self) -> tuple[Any, dict[str, Any]]:
        obs, extras = self.env.reset()
        self._last_inner_reset_reason.zero_()
        self._last_inner_reset_mask.zero_()
        return self._with_retry_context(obs, extras or {}), extras

    def get_observations(self) -> Any:
        return self._with_retry_context(self.env.get_observations(), {})

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        obs, rewards, dones, extras = self.env.step(actions)
        extras = dict(extras or {})
        self._update_reset_reason_state(extras)
        return self._with_retry_context(obs, extras), rewards, dones, extras

    def close(self) -> None:
        return self.env.close()

    def task046_retry_context_debug_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "feature_dim": len(TASK046_RETRY_CONTEXT_FEATURE_NAMES),
            "feature_names": list(TASK046_RETRY_CONTEXT_FEATURE_NAMES),
            "num_trials": int(self.num_trials),
            "final_trial_index": int(self.final_trial_index),
            "step_window_steps": int(self.step_window_steps),
            "base_actor_dim": self._base_actor_dim,
        }

    def _update_reset_reason_state(self, extras: dict[str, Any]) -> None:
        torch = _require_torch()
        inner_reset = _any_extras_mask(
            extras,
            (INNER_RESET_KEY, TASK037_INNER_RESET_KEY),
            self.device,
            self.num_envs,
        )
        outer_reset = _any_extras_mask(
            extras,
            (OUTER_RESET_KEY, EPISODE_DONE_KEY, TASK037_OUTER_RESET_KEY, TASK037_EPISODE_DONE_KEY),
            self.device,
            self.num_envs,
        )
        if inner_reset is None:
            inner_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if outer_reset is None:
            outer_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        reset_reason = extras.get(TASK037_RESET_REASON_KEY)
        if reset_reason is None:
            reset_reason = extras.get(RESET_REASON_KEY)
        if reset_reason is None:
            reset_reason = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        else:
            reset_reason = reset_reason.to(device=self.device, dtype=torch.long)
        self._last_inner_reset_mask = inner_reset.to(device=self.device, dtype=torch.bool)
        if bool(inner_reset.any().item()):
            self._last_inner_reset_reason[inner_reset] = reset_reason[inner_reset]
        if bool(outer_reset.any().item()):
            self._last_inner_reset_reason[outer_reset] = 0
            self._last_inner_reset_mask[outer_reset] = False

    def _with_retry_context(self, obs: Any, extras: dict[str, Any]) -> Any:
        context = self._retry_context(extras)
        actor = obs["actor"]
        context_dim = int(context.shape[-1])
        actor_dim = int(actor.shape[-1])
        if self._base_actor_dim is None:
            self._base_actor_dim = actor_dim
        elif actor_dim == self._base_actor_dim + context_dim:
            actor = actor[..., : self._base_actor_dim]
        elif actor_dim != self._base_actor_dim:
            raise ValueError(
                "Task046 retry context actor dim drift: "
                f"expected base dim {self._base_actor_dim} or appended dim "
                f"{self._base_actor_dim + context_dim}, got {actor_dim}"
            )
        updated = obs if hasattr(obs, "to") else dict(obs)
        updated["actor"] = _require_torch().cat((actor, context), dim=-1)
        updated[self.context_group] = context
        return updated

    def _retry_context(self, extras: dict[str, Any]) -> Any:
        torch = _require_torch()
        trial_index = extras.get(TASK037_TRIAL_INDEX_KEY)
        if trial_index is None:
            trial_index = extras.get(TRIAL_INDEX_KEY)
        if trial_index is None:
            trial_index = getattr(self.env, "trial_index", None)
        if trial_index is None:
            trial_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        trial_index = trial_index.to(device=self.device, dtype=torch.float32).reshape(self.num_envs)

        trial_step = extras.get("task037_trial_step")
        if trial_step is None:
            trial_step = getattr(self.env, "trial_step", None)
        if trial_step is None:
            trial_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        trial_step = trial_step.to(device=self.device, dtype=torch.float32).reshape(self.num_envs)

        trial_index_den = float(max(self.num_trials - 1, 1))
        trial_index_norm = torch.clamp(trial_index / trial_index_den, 0.0, 1.0)
        is_final = (trial_index >= float(self.final_trial_index)).to(dtype=torch.float32)
        trial_step_norm = torch.clamp(trial_step / float(self.step_window_steps), 0.0, 1.0)
        just_inner_reset = self._last_inner_reset_mask.to(device=self.device, dtype=torch.float32)
        last_was_fall = (self._last_inner_reset_reason == 1).to(dtype=torch.float32)
        last_was_timeout = (self._last_inner_reset_reason == 2).to(dtype=torch.float32)
        return torch.stack(
            (
                trial_index_norm,
                is_final,
                trial_step_norm,
                just_inner_reset,
                last_was_fall,
                last_was_timeout,
            ),
            dim=-1,
        )


class Task046PostResetRecoveryRewardVecEnvWrapper:
    """Shape final-trial recovery immediately after an inner reset."""

    def __init__(
        self,
        env: Any,
        *,
        final_trial_index: int = 2,
        recovery_window_steps: int = 50,
        tail_window_steps: int = 50,
        early_velocity_weight: float = 0.0,
        tail_velocity_weight: float = 0.0,
        orientation_weight: float = 0.0,
        root_height_weight: float = 0.0,
        min_root_z: float = 0.70,
    ) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.num_actions = env.num_actions
        self.final_trial_index = int(final_trial_index)
        self.recovery_window_steps = int(recovery_window_steps)
        self.tail_window_steps = int(tail_window_steps)
        self.early_velocity_weight = float(early_velocity_weight)
        self.tail_velocity_weight = float(tail_velocity_weight)
        self.orientation_weight = float(orientation_weight)
        self.root_height_weight = float(root_height_weight)
        self.min_root_z = float(min_root_z)
        torch = _require_torch()
        self._sample_count = torch.zeros((), device=self.device, dtype=torch.float32)
        self._recovery_sample_count = torch.zeros((), device=self.device, dtype=torch.float32)
        self._tail_sample_count = torch.zeros((), device=self.device, dtype=torch.float32)
        self._reward_delta_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        self._early_velocity_penalty_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        self._tail_velocity_penalty_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        self._orientation_penalty_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        self._root_height_penalty_sum = torch.zeros((), device=self.device, dtype=torch.float32)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def cfg(self) -> Any:
        return self.env.cfg

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> Any:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: Any) -> None:
        self.env.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.env.seed(seed)

    def reset(self) -> tuple[Any, dict[str, Any]]:
        return self.env.reset()

    def get_observations(self) -> Any:
        return self.env.get_observations()

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        obs, rewards, dones, extras = self.env.step(actions)
        shaped_rewards, debug = self._shape_rewards(rewards, extras)
        if debug:
            extras = dict(extras)
            extras["task046_post_reset_recovery_reward"] = debug
        return obs, shaped_rewards, dones, extras

    def close(self) -> None:
        return self.env.close()

    def task046_recovery_debug_snapshot(self) -> dict[str, Any]:
        sample_count = float(self._sample_count.detach().cpu().item())
        den = max(sample_count, 1.0)
        return {
            "enabled": True,
            "final_trial_index": int(self.final_trial_index),
            "recovery_window_steps": int(self.recovery_window_steps),
            "tail_window_steps": int(self.tail_window_steps),
            "early_velocity_weight": float(self.early_velocity_weight),
            "tail_velocity_weight": float(self.tail_velocity_weight),
            "orientation_weight": float(self.orientation_weight),
            "root_height_weight": float(self.root_height_weight),
            "min_root_z": float(self.min_root_z),
            "sample_count": int(sample_count),
            "recovery_sample_count": int(self._recovery_sample_count.detach().cpu().item()),
            "tail_sample_count": int(self._tail_sample_count.detach().cpu().item()),
            "reward_delta_mean": float((self._reward_delta_sum / den).detach().cpu().item()),
            "early_velocity_penalty_mean": float(
                (self._early_velocity_penalty_sum / den).detach().cpu().item()
            ),
            "tail_velocity_penalty_mean": float(
                (self._tail_velocity_penalty_sum / den).detach().cpu().item()
            ),
            "orientation_penalty_mean": float(
                (self._orientation_penalty_sum / den).detach().cpu().item()
            ),
            "root_height_penalty_mean": float(
                (self._root_height_penalty_sum / den).detach().cpu().item()
            ),
        }

    def _shape_rewards(self, rewards: Any, extras: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        torch = _require_torch()
        trial_index = extras.get(TASK037_TRIAL_INDEX_KEY)
        if trial_index is None:
            trial_index = extras.get(TRIAL_INDEX_KEY)
        trial_step = extras.get("task037_trial_step")
        if trial_index is None or trial_step is None:
            return rewards, {}
        trial_index = trial_index.to(device=self.device, dtype=torch.long)
        trial_step = trial_step.to(device=self.device, dtype=torch.long)
        final_mask = trial_index >= self.final_trial_index
        recovery_mask = final_mask & (trial_step > 0) & (trial_step <= self.recovery_window_steps)
        tail_mask = final_mask & (trial_step > 0) & (trial_step > self.recovery_window_steps)
        if self.tail_window_steps > 0:
            tail_mask = tail_mask & (trial_step <= self.recovery_window_steps + self.tail_window_steps)
        if not bool((recovery_mask | tail_mask).any().item()):
            return rewards, {}

        robot = self.unwrapped.scene["robot"]
        command = self.unwrapped.command_manager.get_command("twist")
        forward_vel = robot.data.root_link_lin_vel_b[:, 0].to(device=self.device)
        command_x = command[:, 0].to(device=self.device)
        velocity_shortfall = torch.clamp(command_x - forward_vel, min=0.0)
        gravity_xy = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=-1).to(
            device=self.device
        )
        root_z = robot.data.root_link_pos_w[:, 2].to(device=self.device)
        root_shortfall = torch.clamp(self.min_root_z - root_z, min=0.0)

        early_velocity_penalty = self.early_velocity_weight * velocity_shortfall * recovery_mask.float()
        tail_velocity_penalty = self.tail_velocity_weight * velocity_shortfall * tail_mask.float()
        orientation_penalty = self.orientation_weight * gravity_xy * recovery_mask.float()
        root_height_penalty = self.root_height_weight * root_shortfall * recovery_mask.float()
        total_penalty = (
            early_velocity_penalty
            + tail_velocity_penalty
            + orientation_penalty
            + root_height_penalty
        )
        shaped_rewards = rewards - total_penalty.to(device=rewards.device, dtype=rewards.dtype)

        active = recovery_mask | tail_mask
        self._sample_count += active.float().sum()
        self._recovery_sample_count += recovery_mask.float().sum()
        self._tail_sample_count += tail_mask.float().sum()
        self._reward_delta_sum += (-total_penalty[active]).float().sum()
        self._early_velocity_penalty_sum += early_velocity_penalty[active].float().sum()
        self._tail_velocity_penalty_sum += tail_velocity_penalty[active].float().sum()
        self._orientation_penalty_sum += orientation_penalty[active].float().sum()
        self._root_height_penalty_sum += root_height_penalty[active].float().sum()

        return shaped_rewards, {
            "active_count": int(active.float().sum().detach().cpu().item()),
            "recovery_count": int(recovery_mask.float().sum().detach().cpu().item()),
            "tail_count": int(tail_mask.float().sum().detach().cpu().item()),
            "reward_delta_mean": float((-total_penalty[active]).mean().detach().cpu().item()),
        }


def _task046_post_reset_recovery_reward_cfg(train_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    cfg = train_cfg.get("task046_post_reset_recovery_reward")
    if not isinstance(cfg, Mapping) or not bool(cfg.get("enabled", False)):
        return None
    return {
        "final_trial_index": int(cfg.get("final_trial_index", 2)),
        "recovery_window_steps": int(cfg.get("recovery_window_steps", 50)),
        "tail_window_steps": int(cfg.get("tail_window_steps", 50)),
        "early_velocity_weight": float(cfg.get("early_velocity_weight", 0.0)),
        "tail_velocity_weight": float(cfg.get("tail_velocity_weight", 0.0)),
        "orientation_weight": float(cfg.get("orientation_weight", 0.0)),
        "root_height_weight": float(cfg.get("root_height_weight", 0.0)),
        "min_root_z": float(cfg.get("min_root_z", 0.70)),
    }


def _task046_retry_context_cfg(train_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    cfg = train_cfg.get("task046_retry_context")
    if not isinstance(cfg, Mapping) or not bool(cfg.get("enabled", False)):
        return None
    return {
        "num_trials": int(cfg.get("num_trials", 3)),
        "final_trial_index": int(cfg.get("final_trial_index", 2)),
        "step_window_steps": int(cfg.get("step_window_steps", 50)),
    }


class _Task033StackMlpWarmstartMixin:
    """Load base MLP or migrated StackMLP checkpoints with a fresh optimizer."""

    task033_history_len: int

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        torch = _require_torch()
        loaded = torch.load(path, map_location=map_location, weights_only=False)
        actor_state = loaded.get("actor_state_dict", {})
        target_actor_state = self.alg.actor.state_dict()
        needs_migration = is_stack_mlp_history_migration_needed(actor_state, target_actor_state)
        if not needs_migration and "optimizer_state_dict" in loaded:
            return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)

        if needs_migration:
            obs_dim = int(actor_state["mlp.0.weight"].shape[1])
            action_dim = int(self.env.num_actions)
            migrated, report = migrate_stack_mlp_checkpoint(
                loaded,
                StackMlpHistoryMigrationConfig(
                    history_len=self.task033_history_len,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                ),
            )
        else:
            migrated = loaded
            report = dict((loaded.get("infos") or {}).get("task033_history_migration") or {})
            report.update(
                {
                    "history_len": self.task033_history_len,
                    "target_actor_input_dim": int(target_actor_state["mlp.0.weight"].shape[1]),
                    "optimizer_policy": "fresh optimizer required after migration",
                    "already_history_shaped": True,
                }
            )
            report.setdefault("migration", "task033_stack_mlp_history")
            report.setdefault("source_actor_input_dim", int(actor_state["mlp.0.weight"].shape[1]))

        self.alg.actor.load_state_dict(migrated["actor_state_dict"], strict=strict)
        self.alg.critic.load_state_dict(migrated["critic_state_dict"], strict=strict)
        self.current_learning_iteration = int(migrated.get("iter", 0))
        infos = dict(migrated.get("infos") or {})
        infos["task033_history_migration"] = report
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
        print(f"[Task033] loaded StackMLP warmstart without optimizer: {report}")
        return infos


class _Task036AdaptationWarmstartMixin:
    """Load base MLP checkpoints with a fresh optimizer for adaptation conditioning."""

    task036_obs_dim = 104
    task036_adaptation_history_len = 4
    task036_adaptation_latent_dim = 32

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        torch = _require_torch()
        loaded = torch.load(path, map_location=map_location, weights_only=False)
        actor_state = loaded.get("actor_state_dict", {})
        target_actor_state = self.alg.actor.state_dict()
        needs_migration = is_adaptation_conditioning_migration_needed(
            actor_state,
            target_actor_state,
        )
        if not needs_migration and "optimizer_state_dict" in loaded:
            return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)

        if needs_migration:
            migrated, report = migrate_adaptation_conditioned_checkpoint(
                loaded,
                AdaptationConditioningMigrationConfig(
                    obs_dim=self.task036_obs_dim,
                    action_dim=int(self.env.num_actions),
                    history_len=self.task036_adaptation_history_len,
                    latent_dim=self.task036_adaptation_latent_dim,
                ),
            )
        else:
            migrated = loaded
            report = dict((loaded.get("infos") or {}).get("task036_adaptation_conditioning_migration") or {})
            report.update(
                {
                    "target_actor_input_dim": int(target_actor_state["mlp.0.weight"].shape[1]),
                    "optimizer_policy": "fresh optimizer required after migration",
                    "already_adaptation_shaped": True,
                }
            )
            report.setdefault("migration", "task036_adaptation_conditioning")

        actor_state_for_load = dict(target_actor_state)
        actor_state_for_load.update(migrated["actor_state_dict"])
        self.alg.actor.load_state_dict(actor_state_for_load, strict=strict)
        self.alg.critic.load_state_dict(migrated["critic_state_dict"], strict=strict)
        self.current_learning_iteration = int(migrated.get("iter", 0))
        infos = dict(migrated.get("infos") or {})
        infos["task036_adaptation_conditioning_migration"] = report
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
        print(f"[Task036] loaded adaptation-conditioned warmstart without optimizer: {report}")
        return infos


class _Task037AdaptK160WarmstartMixin(_Task036AdaptationWarmstartMixin):
    """Warm-start a K160 adaptation actor from a shorter adaptation checkpoint."""

    task036_adaptation_history_len = 160

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        torch = _require_torch()
        loaded = torch.load(path, map_location=map_location, weights_only=False)
        actor_state = loaded.get("actor_state_dict", {})
        target_actor_state = self.alg.actor.state_dict()
        first_key = "mlp.0.weight"
        source_first = actor_state.get(first_key)

        if source_first is not None and int(source_first.shape[1]) == self.task036_obs_dim:
            return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)

        migrated_actor, report = _migrate_adaptation_history_len_actor_state_dict(
            actor_state,
            target_actor_state,
            obs_dim=self.task036_obs_dim,
            action_dim=int(self.env.num_actions),
            target_history_len=self.task036_adaptation_history_len,
        )
        actor_state_for_load = dict(target_actor_state)
        actor_state_for_load.update(migrated_actor)
        self.alg.actor.load_state_dict(actor_state_for_load, strict=strict)
        self.alg.critic.load_state_dict(loaded["critic_state_dict"], strict=strict)
        self.current_learning_iteration = int(loaded.get("iter", 0))
        infos = dict(loaded.get("infos") or {})
        infos["task037_adapt_k160_warmstart_migration"] = report
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
        print(f"[Task037] loaded AdaptK160 warmstart without optimizer: {report}")
        return infos


class Task033BufferOnlyK4Runner(_base_runner()):
    """Maintain K=4 history while keeping the actor on the original obs."""

    def __init__(self, env: Any, train_cfg: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(
            Task033HistoryVecEnvWrapper(env, history_len=4),
            train_cfg,
            *args,
            **kwargs,
        )


class Task037BufferOnlyK4AutoResetRunner(_base_runner()):
    """MJLab auto-reset multi-trial smoke runner over the existing K4 buffer."""

    def __init__(self, env: Any, train_cfg: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        env = Task037MultiTrialVecEnvWrapper(
            env,
            num_trials=3,
            reset_strategy="auto",
        )
        env = Task033HistoryVecEnvWrapper(env, history_len=4)
        super().__init__(env, train_cfg, *args, **kwargs)


class Task037BufferOnlyK4DeterministicInnerResetRunner(_base_runner()):
    """Task037 runner that preserves condition across inner MJLab resets."""

    def __init__(self, env: Any, train_cfg: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        install_task037_inner_reset_controller(env.unwrapped, num_trials=3)
        env.reset()
        env = Task037MultiTrialVecEnvWrapper(
            env,
            num_trials=3,
            reset_strategy="auto",
        )
        env = Task033HistoryVecEnvWrapper(env, history_len=4)
        super().__init__(env, train_cfg, *args, **kwargs)


class Task037AdaptK4DeterministicInnerResetRunner(_Task036AdaptationWarmstartMixin, _base_runner()):
    """Task037 deterministic multi-trial runner for Task036 AdaptK4 checkpoints."""

    def __init__(self, env: Any, train_cfg: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        install_task037_inner_reset_controller(env.unwrapped, num_trials=3)
        env.reset()
        env = Task037MultiTrialVecEnvWrapper(
            env,
            num_trials=3,
            reset_strategy="auto",
        )
        env = Task033HistoryVecEnvWrapper(env, history_len=4)
        train_cfg["obs_groups"] = {"actor": ["actor_history"], "critic": ["critic"]}
        train_cfg["actor"]["class_name"] = (
            "h200_locomotion_lab.training.rsl_history_wrapper:Task036AdaptationConditionedMlpModel"
        )
        train_cfg["actor"]["history_len"] = 4
        train_cfg["actor"]["action_dim"] = int(env.num_actions)
        train_cfg["actor"]["adaptation_latent_dim"] = self.task036_adaptation_latent_dim
        train_cfg["actor"]["adaptation_hidden_dim"] = 128
        super().__init__(env, train_cfg, *args, **kwargs)


class Task037TxlMemoryK160DeterministicRunner(_base_runner()):
    """Task037 runner that consumes 3.2s actor-visible history as segment memory."""

    task037_history_len = 160
    task037_segment_len = 16
    task037_token_dim = 128

    def __init__(self, env: Any, train_cfg: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        install_task037_inner_reset_controller(env.unwrapped, num_trials=3)
        env.reset()
        env = Task037MultiTrialVecEnvWrapper(
            env,
            num_trials=3,
            reset_strategy="auto",
        )
        env = Task033HistoryVecEnvWrapper(env, history_len=self.task037_history_len)
        train_cfg["obs_groups"] = {"actor": ["actor_history"], "critic": ["critic"]}
        train_cfg["actor"]["class_name"] = (
            "h200_locomotion_lab.training.rsl_history_wrapper:Task037TxlStyleMemoryModel"
        )
        train_cfg["actor"]["history_len"] = self.task037_history_len
        train_cfg["actor"]["segment_len"] = self.task037_segment_len
        train_cfg["actor"]["token_dim"] = self.task037_token_dim
        super().__init__(env, train_cfg, *args, **kwargs)


class Task038TrueTxlMemoryK160Runner(_base_runner()):
    """Task038 runner smoke path that consumes a stateful true-TXL cache actor."""

    task038_history_len = 160
    task038_segment_len = 16
    task038_token_dim = 128
    task038_memory_len = 64

    def __init__(self, env: Any, train_cfg: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        install_task037_inner_reset_controller(env.unwrapped, num_trials=3)
        env.reset()
        env = Task037MultiTrialVecEnvWrapper(
            env,
            num_trials=3,
            reset_strategy="auto",
        )
        env = Task033HistoryVecEnvWrapper(env, history_len=self.task038_history_len)
        env = Task038TrueTxlResetHookVecEnvWrapper(env)
        train_cfg.setdefault("actor", {})
        train_cfg["obs_groups"] = {"actor": ["actor_history"], "critic": ["critic"]}
        train_cfg["actor"]["class_name"] = (
            "h200_locomotion_lab.training.rsl_history_wrapper:Task038TrueTxlMemoryModel"
        )
        train_cfg["actor"]["history_len"] = self.task038_history_len
        train_cfg["actor"]["segment_len"] = self.task038_segment_len
        train_cfg["actor"]["token_dim"] = self.task038_token_dim
        train_cfg["actor"]["memory_len"] = self.task038_memory_len
        super().__init__(env, train_cfg, *args, **kwargs)
        env.attach_task038_txl_actor(_task038_find_actor(self))


class Task044TrueTxlMemoryK160ClearHistoryRunner(_base_runner()):
    """Task044 runner: clear visible history on inner reset, preserve TXL cache."""

    task044_history_len = 160
    task044_segment_len = 16
    task044_token_dim = 128
    task044_memory_len = 64

    def __init__(self, env: Any, train_cfg: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        install_task037_inner_reset_controller(env.unwrapped, num_trials=3)
        env.reset()
        env = Task037MultiTrialVecEnvWrapper(
            env,
            num_trials=3,
            reset_strategy="auto",
        )
        recovery_reward_cfg = _task046_post_reset_recovery_reward_cfg(train_cfg)
        if recovery_reward_cfg is not None:
            env = Task046PostResetRecoveryRewardVecEnvWrapper(env, **recovery_reward_cfg)
        retry_context_cfg = _task046_retry_context_cfg(train_cfg)
        if retry_context_cfg is not None:
            env = Task046RetryContextVecEnvWrapper(env, **retry_context_cfg)
        env = Task033HistoryVecEnvWrapper(
            env,
            history_len=self.task044_history_len,
            clear_history_on_inner_reset=True,
        )
        env = Task044FaultLabelVecEnvWrapper(env)
        env = Task038TrueTxlResetHookVecEnvWrapper(env)
        train_cfg.setdefault("actor", {})
        train_cfg["obs_groups"] = {"actor": ["actor_history"], "critic": ["critic"]}
        train_cfg["actor"]["class_name"] = (
            "h200_locomotion_lab.training.rsl_history_wrapper:Task038TrueTxlMemoryModel"
        )
        train_cfg["actor"]["history_len"] = self.task044_history_len
        train_cfg["actor"]["segment_len"] = self.task044_segment_len
        train_cfg["actor"]["token_dim"] = self.task044_token_dim
        train_cfg["actor"]["memory_len"] = self.task044_memory_len
        super().__init__(env, train_cfg, *args, **kwargs)
        env.attach_task038_txl_actor(_task038_find_actor(self))


class Task044TrueTxlMemoryK160ContinuousRunner(_base_runner()):
    """Task044 eval-only runner: no Task037 inner reset or multi-trial wrapper."""

    task044_history_len = 160
    task044_segment_len = 16
    task044_token_dim = 128
    task044_memory_len = 64

    def __init__(self, env: Any, train_cfg: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        env = Task033HistoryVecEnvWrapper(
            env,
            history_len=self.task044_history_len,
            clear_history_on_inner_reset=False,
        )
        env = Task044FaultLabelVecEnvWrapper(env)
        env = Task038TrueTxlResetHookVecEnvWrapper(env)
        train_cfg.setdefault("actor", {})
        train_cfg["obs_groups"] = {"actor": ["actor_history"], "critic": ["critic"]}
        train_cfg["actor"]["class_name"] = (
            "h200_locomotion_lab.training.rsl_history_wrapper:Task038TrueTxlMemoryModel"
        )
        train_cfg["actor"]["history_len"] = self.task044_history_len
        train_cfg["actor"]["segment_len"] = self.task044_segment_len
        train_cfg["actor"]["token_dim"] = self.task044_token_dim
        train_cfg["actor"]["memory_len"] = self.task044_memory_len
        super().__init__(env, train_cfg, *args, **kwargs)
        env.attach_task038_txl_actor(_task038_find_actor(self))


class Task040SequenceAwareTrueTxlPPO(_base_ppo()):
    """PPO update path that preserves rollout time order for true-TXL actors."""

    def __init__(
        self,
        *args: Any,
        task044_fault_aux_loss_weight: float = 0.0,
        task044_fault_aux_obs_key: str = "task044_fault_label",
        task044_fault_aux_num_classes: int = 0,
        task044_fault_aux_max_trial_step: int = -1,
        task044_fault_aux_min_trial_index: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.task044_fault_aux_loss_weight = float(task044_fault_aux_loss_weight)
        self.task044_fault_aux_obs_key = str(task044_fault_aux_obs_key)
        self.task044_fault_aux_num_classes = int(task044_fault_aux_num_classes)
        self.task044_fault_aux_max_trial_step = int(task044_fault_aux_max_trial_step)
        self.task044_fault_aux_min_trial_index = int(task044_fault_aux_min_trial_index)
        self.task044_fault_aux_head = None
        if self.task044_fault_aux_loss_weight > 0.0 and self.task044_fault_aux_num_classes > 0:
            torch = _require_torch()
            nn = torch.nn
            memory_latent_dim = int(getattr(self.actor, "memory_latent_dim", 0) or 0)
            if memory_latent_dim <= 0:
                raise ValueError("Task044 fault auxiliary loss requires actor.memory_latent_dim > 0")
            self.task044_fault_aux_head = nn.Linear(
                memory_latent_dim,
                self.task044_fault_aux_num_classes,
            ).to(self.device)
            nn.init.orthogonal_(self.task044_fault_aux_head.weight)
            nn.init.zeros_(self.task044_fault_aux_head.bias)
            self.optimizer.add_param_group({"params": self.task044_fault_aux_head.parameters()})
        self.task044_fault_aux_updates = 0
        self.task044_fault_aux_last_loss: float | None = None
        self.task040_sequence_update_batches = 0
        self.task040_sequence_update_samples = 0
        self.task040_sequence_update_steps = 0
        self.task040_last_loss_dict: dict[str, float] | None = None

    def update(self) -> dict[str, float]:
        if self.rnd:
            raise NotImplementedError("Task040 sequence-aware PPO does not support RND yet")
        if self.symmetry:
            raise NotImplementedError("Task040 sequence-aware PPO does not support symmetry yet")
        if not hasattr(self.actor, "task040_forward_sequence"):
            return super().update()

        torch = _require_torch()
        nn = torch.nn
        functional = nn.functional
        st = self.storage
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_fault_aux_loss = 0.0
        update_count = 0

        for _epoch in range(self.num_learning_epochs):
            for env_start, env_stop in _task040_sequence_env_slices(
                st.num_envs,
                self.num_mini_batches,
            ):
                obs_seq = st.observations[:, env_start:env_stop]
                reset_mask = _task040_reset_mask_from_dones(st.dones[:, env_start:env_stop])
                actions = _task040_flatten_time_env(st.actions[:, env_start:env_stop])
                old_actions_log_prob = _task040_flatten_time_env(
                    st.actions_log_prob[:, env_start:env_stop]
                )
                old_distribution_params = tuple(
                    _task040_flatten_time_env(param[:, env_start:env_stop])
                    for param in st.distribution_params
                )
                values_old = _task040_flatten_time_env(st.values[:, env_start:env_stop])
                returns = _task040_flatten_time_env(st.returns[:, env_start:env_stop])
                advantages = _task040_flatten_time_env(st.advantages[:, env_start:env_stop])
                if self.normalize_advantage_per_mini_batch:
                    with torch.no_grad():
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                self.actor.task040_forward_sequence(
                    obs_seq,
                    reset_mask=reset_mask,
                    stochastic_output=True,
                )
                fault_aux_loss = None
                if self.task044_fault_aux_loss_weight > 0.0:
                    fault_aux_loss = self._task044_fault_aux_loss(
                        functional,
                        st.observations,
                        env_start=env_start,
                        env_stop=env_stop,
                    )
                actions_log_prob = self.actor.get_output_log_prob(actions)
                obs_flat = obs_seq.flatten(0, 1)
                values = self.critic(obs_flat)
                distribution_params = tuple(self.actor.output_distribution_params)
                entropy = self.actor.output_entropy

                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = self.actor.get_kl_divergence(
                            old_distribution_params,
                            distribution_params,
                        )
                        kl_mean = torch.mean(kl)
                        if self.is_multi_gpu:
                            torch.distributed.all_reduce(
                                kl_mean,
                                op=torch.distributed.ReduceOp.SUM,
                            )
                            kl_mean /= self.gpu_world_size
                        if self.gpu_global_rank == 0:
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        if self.is_multi_gpu:
                            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                            torch.distributed.broadcast(lr_tensor, src=0)
                            self.learning_rate = lr_tensor.item()
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                ratio = torch.exp(actions_log_prob - torch.squeeze(old_actions_log_prob))
                surrogate = -torch.squeeze(advantages) * ratio
                surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param,
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                if self.use_clipped_value_loss:
                    value_clipped = values_old + (values - values_old).clamp(
                        -self.clip_param,
                        self.clip_param,
                    )
                    value_losses = (values - returns).pow(2)
                    value_losses_clipped = (value_clipped - returns).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns - values).pow(2).mean()

                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )
                if fault_aux_loss is not None:
                    loss = loss + self.task044_fault_aux_loss_weight * fault_aux_loss
                self.optimizer.zero_grad()
                loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                sample_count = int(actions.shape[0])
                step_count = int(obs_seq.batch_size[0])
                self.task040_sequence_update_batches += 1
                self.task040_sequence_update_samples += sample_count
                self.task040_sequence_update_steps += step_count
                mean_value_loss += float(value_loss.item())
                mean_surrogate_loss += float(surrogate_loss.item())
                mean_entropy += float(entropy.mean().item())
                if fault_aux_loss is not None:
                    mean_fault_aux_loss += float(fault_aux_loss.item())
                    self.task044_fault_aux_updates += 1
                    self.task044_fault_aux_last_loss = float(fault_aux_loss.item())
                update_count += 1

        if update_count <= 0:
            raise RuntimeError("Task040 sequence-aware PPO produced no update batches")
        loss_dict = {
            "value": mean_value_loss / update_count,
            "surrogate": mean_surrogate_loss / update_count,
            "entropy": mean_entropy / update_count,
        }
        if self.task044_fault_aux_loss_weight > 0.0:
            loss_dict["task044_fault_aux"] = mean_fault_aux_loss / update_count
        self.task040_last_loss_dict = loss_dict
        st.clear()
        return loss_dict

    def _task044_fault_aux_loss(
        self,
        functional: Any,
        observations: Any,
        *,
        env_start: int,
        env_stop: int,
    ) -> Any | None:
        if self.task044_fault_aux_head is None:
            return None
        if self.task044_fault_aux_obs_key not in observations.keys():
            return None
        latent_fn = getattr(self.actor, "task044_memory_latents_from_last_sequence", None)
        if not callable(latent_fn):
            return None
        memory_latents = latent_fn()
        if memory_latents is None:
            return None
        logits = self.task044_fault_aux_head(memory_latents)
        labels = observations[self.task044_fault_aux_obs_key][:, env_start:env_stop]
        labels = labels.reshape(-1).to(device=logits.device, dtype=_require_torch().long)
        valid = (labels >= 0) & (labels < int(logits.shape[-1]))
        valid = self._task044_apply_fault_aux_trial_mask(
            valid,
            observations,
            env_start=env_start,
            env_stop=env_stop,
            device=logits.device,
        )
        if not bool(valid.any().item()):
            return None
        return functional.cross_entropy(logits[valid], labels[valid])

    def _task044_apply_fault_aux_trial_mask(
        self,
        valid: Any,
        observations: Any,
        *,
        env_start: int,
        env_stop: int,
        device: Any,
    ) -> Any:
        torch = _require_torch()
        if self.task044_fault_aux_max_trial_step >= 0:
            key = Task044FaultLabelVecEnvWrapper.trial_step_group
            if key not in observations.keys():
                return torch.zeros_like(valid, dtype=torch.bool, device=device)
            trial_steps = observations[key][:, env_start:env_stop]
            trial_steps = trial_steps.reshape(-1).to(device=device, dtype=torch.long)
            valid = valid & (trial_steps <= self.task044_fault_aux_max_trial_step)
        if self.task044_fault_aux_min_trial_index > 0:
            key = Task044FaultLabelVecEnvWrapper.trial_index_group
            if key not in observations.keys():
                return torch.zeros_like(valid, dtype=torch.bool, device=device)
            trial_indices = observations[key][:, env_start:env_stop]
            trial_indices = trial_indices.reshape(-1).to(device=device, dtype=torch.long)
            valid = valid & (trial_indices >= self.task044_fault_aux_min_trial_index)
        return valid

    def task040_sequence_update_debug_snapshot(self) -> dict[str, Any]:
        return {
            "algorithm_class": type(self).__name__,
            "sequence_update_batches": int(self.task040_sequence_update_batches),
            "sequence_update_samples": int(self.task040_sequence_update_samples),
            "sequence_update_steps": int(self.task040_sequence_update_steps),
            "last_loss_dict": dict(self.task040_last_loss_dict or {}),
            "task044_fault_aux_loss_weight": float(self.task044_fault_aux_loss_weight),
            "task044_fault_aux_obs_key": self.task044_fault_aux_obs_key,
            "task044_fault_aux_num_classes": int(self.task044_fault_aux_num_classes),
            "task044_fault_aux_max_trial_step": int(self.task044_fault_aux_max_trial_step),
            "task044_fault_aux_min_trial_index": int(self.task044_fault_aux_min_trial_index),
            "task044_fault_aux_updates": int(self.task044_fault_aux_updates),
            "task044_fault_aux_last_loss": self.task044_fault_aux_last_loss,
        }


class Task037AdaptK160DeterministicInnerResetRunner(_Task037AdaptK160WarmstartMixin, _base_runner()):
    """Task037 K160 adaptation-conditioned runner with a warm-startable actor head."""

    task037_history_len = 160

    def __init__(self, env: Any, train_cfg: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        install_task037_inner_reset_controller(env.unwrapped, num_trials=3)
        env.reset()
        env = Task037MultiTrialVecEnvWrapper(
            env,
            num_trials=3,
            reset_strategy="auto",
        )
        env = Task033HistoryVecEnvWrapper(env, history_len=self.task037_history_len)
        train_cfg["obs_groups"] = {"actor": ["actor_history"], "critic": ["critic"]}
        train_cfg["actor"]["class_name"] = (
            "h200_locomotion_lab.training.rsl_history_wrapper:Task036AdaptationConditionedMlpModel"
        )
        train_cfg["actor"]["history_len"] = self.task037_history_len
        train_cfg["actor"]["action_dim"] = int(env.num_actions)
        train_cfg["actor"]["adaptation_latent_dim"] = self.task036_adaptation_latent_dim
        train_cfg["actor"]["adaptation_hidden_dim"] = 128
        super().__init__(env, train_cfg, *args, **kwargs)


class Task033StackMlpK4Runner(_Task033StackMlpWarmstartMixin, _base_runner()):
    """Flatten K=4 shared history frames into the actor MLP input."""

    task033_history_len = 4

    def __init__(self, env: Any, train_cfg: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        env = Task033HistoryVecEnvWrapper(env, history_len=4)
        train_cfg["obs_groups"] = {"actor": ["actor_history"], "critic": ["critic"]}
        super().__init__(env, train_cfg, *args, **kwargs)


class Task033StackMlpK8Runner(_Task033StackMlpWarmstartMixin, _base_runner()):
    """Flatten K=8 shared history frames into the actor MLP input."""

    task033_history_len = 8

    def __init__(self, env: Any, train_cfg: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        env = Task033HistoryVecEnvWrapper(env, history_len=8)
        train_cfg["obs_groups"] = {"actor": ["actor_history"], "critic": ["critic"]}
        super().__init__(env, train_cfg, *args, **kwargs)


class _Task033FrozenBasePathMixin:
    """Freeze the migrated base actor path and train only new history columns."""

    def __init__(self, env: Any, train_cfg: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(env, train_cfg, *args, **kwargs)
        self._freeze_task033_base_actor_path()

    def _freeze_task033_base_actor_path(self) -> None:
        torch = _require_torch()
        actor = self.alg.actor
        history_len = int(self.task033_history_len)
        input_dim = int(actor.mlp[0].weight.shape[1])
        frame_dim = input_dim // history_len
        obs_dim = frame_dim - int(self.env.num_actions)
        newest_obs_start = (history_len - 1) * frame_dim
        newest_obs_stop = newest_obs_start + obs_dim

        for name, param in actor.named_parameters():
            if name != "mlp.0.weight":
                param.requires_grad_(False)

        mask = torch.ones_like(actor.mlp[0].weight)
        mask[:, newest_obs_start:newest_obs_stop] = 0.0
        actor.mlp[0].weight.register_hook(lambda grad: grad * mask)
        actor.update_normalization = lambda obs: None
        print(
            "[Task033] frozen base actor path; trainable first-layer columns exclude "
            f"newest obs slice [{newest_obs_start}, {newest_obs_stop})"
        )


class Task033StackMlpK4FrozenBaseRunner(_Task033FrozenBasePathMixin, Task033StackMlpK4Runner):
    """StackMLP K4 runner that preserves the migrated base policy path."""


class Task033GruK4Runner(_base_runner()):
    """Use RSL-RL GRU models while consuming the shared K=4 history group."""

    def __init__(self, env: Any, train_cfg: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        env = Task033HistoryVecEnvWrapper(env, history_len=4)
        train_cfg["obs_groups"] = {"actor": ["actor_history"], "critic": ["critic"]}
        train_cfg["actor"]["class_name"] = "RNNModel"
        train_cfg["actor"]["rnn_type"] = "gru"
        train_cfg["actor"]["rnn_hidden_dim"] = 256
        train_cfg["actor"]["rnn_num_layers"] = 1
        train_cfg["critic"]["class_name"] = "RNNModel"
        train_cfg["critic"]["rnn_type"] = "gru"
        train_cfg["critic"]["rnn_hidden_dim"] = 256
        train_cfg["critic"]["rnn_num_layers"] = 1
        super().__init__(env, train_cfg, *args, **kwargs)


class Task033TokenK4Runner(_base_runner()):
    """Use a minimal tokenized history actor over the shared K=4 history group."""

    def __init__(self, env: Any, train_cfg: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        env = Task033HistoryVecEnvWrapper(env, history_len=4)
        train_cfg["obs_groups"] = {"actor": ["actor_history"], "critic": ["critic"]}
        train_cfg["actor"]["class_name"] = (
            "h200_locomotion_lab.training.rsl_history_wrapper:Task033HistoryTokenMlpModel"
        )
        train_cfg["actor"]["history_len"] = 4
        train_cfg["actor"]["token_dim"] = 128
        super().__init__(env, train_cfg, *args, **kwargs)


class Task036AdaptK4Runner(_Task036AdaptationWarmstartMixin, _base_runner()):
    """Use shared K4 history only to infer an adaptation latent for an MLP actor."""

    def __init__(self, env: Any, train_cfg: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        env = Task033HistoryVecEnvWrapper(env, history_len=4)
        train_cfg["obs_groups"] = {"actor": ["actor_history"], "critic": ["critic"]}
        train_cfg["actor"]["class_name"] = (
            "h200_locomotion_lab.training.rsl_history_wrapper:Task036AdaptationConditionedMlpModel"
        )
        train_cfg["actor"]["history_len"] = 4
        train_cfg["actor"]["action_dim"] = int(env.num_actions)
        train_cfg["actor"]["adaptation_latent_dim"] = self.task036_adaptation_latent_dim
        train_cfg["actor"]["adaptation_hidden_dim"] = 128
        super().__init__(env, train_cfg, *args, **kwargs)


def _task040_sequence_env_slices(num_envs: int, num_mini_batches: int) -> list[tuple[int, int]]:
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    if num_mini_batches <= 0:
        raise ValueError(f"num_mini_batches must be positive, got {num_mini_batches}")
    if num_envs % num_mini_batches != 0:
        raise ValueError(
            f"Task040 sequence PPO requires num_envs divisible by num_mini_batches, "
            f"got num_envs={num_envs}, num_mini_batches={num_mini_batches}"
        )
    mini_batch_size = num_envs // num_mini_batches
    return [
        (index * mini_batch_size, (index + 1) * mini_batch_size)
        for index in range(num_mini_batches)
    ]


def _task040_flatten_time_env(tensor: Any) -> Any:
    if len(tensor.shape) < 2:
        raise ValueError(f"expected tensor with at least two dims, got {tuple(tensor.shape)}")
    return tensor.reshape(int(tensor.shape[0]) * int(tensor.shape[1]), *tensor.shape[2:])


def _task040_reset_mask_from_dones(dones: Any) -> Any:
    torch = _require_torch()
    if len(dones.shape) == 3 and int(dones.shape[-1]) == 1:
        dones = dones.squeeze(-1)
    if len(dones.shape) != 2:
        raise ValueError(f"dones must have shape [time, env] or [time, env, 1], got {tuple(dones.shape)}")
    reset_mask = torch.zeros_like(dones, dtype=torch.bool)
    if int(dones.shape[0]) > 1:
        reset_mask[1:] = dones[:-1].to(dtype=torch.bool)
    return reset_mask


def _task040_sequence_reset_mask(
    torch: Any,
    reset_mask: Any | None,
    *,
    steps: int,
    batch: int,
    device: Any,
) -> Any:
    if reset_mask is None:
        return torch.zeros((steps, batch), device=device, dtype=torch.bool)
    if hasattr(reset_mask, "to"):
        reset_mask = reset_mask.to(device=device, dtype=torch.bool)
    else:
        reset_mask = torch.as_tensor(reset_mask, device=device, dtype=torch.bool)
    if tuple(reset_mask.shape) != (steps, batch):
        raise ValueError(
            f"Task040 reset_mask must have shape ({steps}, {batch}), got {tuple(reset_mask.shape)}"
        )
    return reset_mask


def _task040_append_sequence_memory(
    torch: Any,
    memory: Any,
    lengths: Any,
    segment_tokens: Any,
    *,
    memory_len: int,
) -> tuple[Any, Any]:
    updated = torch.zeros_like(memory)
    updated_lengths = lengths.clone()
    for env_id in range(int(segment_tokens.shape[0])):
        length = int(lengths[env_id].detach().cpu().item())
        combined = torch.cat((memory[env_id, :length], segment_tokens[env_id]), dim=0)
        if int(combined.shape[0]) > memory_len:
            combined = combined[-memory_len:]
        new_len = int(combined.shape[0])
        updated[env_id, :new_len] = combined
        updated_lengths[env_id] = new_len
    return updated, updated_lengths


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - H200-only import path.
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch


def _task038_clone_inference_tensor(tensor: Any) -> Any:
    if tensor is None:
        return None
    is_inference = getattr(tensor, "is_inference", None)
    if not callable(is_inference):
        return tensor
    try:
        needs_clone = bool(is_inference())
    except RuntimeError:
        return tensor
    if not needs_clone:
        return tensor
    return tensor.clone()


def _task042_tensor_mean_l2_norm(torch: Any, tensor: Any) -> float:
    if int(tensor.numel()) == 0:
        return 0.0
    flattened = tensor.detach().float().reshape(int(tensor.shape[0]), -1)
    return float(torch.linalg.norm(flattened, dim=-1).mean().detach().cpu().item())


def _dtype_name(torch: Any, dtype: Any) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise ValueError(f"Unsupported history buffer dtype: {dtype}")


def _combined_reset_mask(
    done: Any | None,
    zero_action_mask: Any | None,
    device: Any,
    num_envs: int,
) -> Any | None:
    if done is None and zero_action_mask is None:
        return None
    if done is None:
        return _bool_mask(zero_action_mask, device, num_envs)
    if zero_action_mask is None:
        return _bool_mask(done, device, num_envs)
    return _bool_mask(done, device, num_envs) | _bool_mask(zero_action_mask, device, num_envs)


def _any_extras_mask(
    extras: dict[str, Any],
    keys: tuple[str, ...],
    device: Any,
    num_envs: int,
) -> Any | None:
    mask = None
    for key in keys:
        if key not in extras:
            continue
        key_mask = _bool_mask(extras[key], device, num_envs)
        mask = key_mask if mask is None else mask | key_mask
    return mask


def _mask_to_env_ids(mask: Any) -> Any:
    return mask.nonzero(as_tuple=False).flatten()


def _all_env_ids(num_envs: int, device: Any) -> Any:
    torch = _require_torch()
    return torch.arange(num_envs, device=device, dtype=torch.long)


def _task038_find_actor(runner: Any) -> Any:
    for path in (
        ("alg", "actor"),
        ("alg", "actor_critic", "actor"),
        ("alg", "actor_critic"),
        ("actor_critic", "actor"),
        ("actor_critic",),
        ("policy",),
    ):
        obj = runner
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "task038_txl_outer_reset"):
            return obj
    raise AttributeError("Task038 true-TXL runner could not find an actor with TXL reset hooks")


def _bool_mask(mask: Any, device: Any, num_envs: int) -> Any:
    torch = _require_torch()
    if hasattr(mask, "to"):
        tensor = mask.to(device=device, dtype=torch.bool)
    else:
        tensor = torch.as_tensor(mask, device=device, dtype=torch.bool)
    if tuple(tensor.shape) != (num_envs,):
        raise ValueError(f"mask must have shape ({num_envs},), got {tuple(tensor.shape)}")
    return tensor


def _migrate_adaptation_history_len_actor_state_dict(
    source_state: dict[str, Any],
    target_state: dict[str, Any],
    *,
    obs_dim: int,
    action_dim: int,
    target_history_len: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy matching AdaptK actor weights while resizing history normalizers."""

    frame_dim = obs_dim + action_dim
    source_history_len = _infer_history_len(source_state, frame_dim)
    migrated: dict[str, Any] = {}
    copied_keys: list[str] = []
    expanded_keys: list[str] = []
    skipped_keys: list[str] = []

    for key, target_value in target_state.items():
        if key not in source_state:
            skipped_keys.append(key)
            continue
        source_value = source_state[key]
        if tuple(source_value.shape) == tuple(target_value.shape):
            migrated[key] = source_value.clone()
            copied_keys.append(key)
        elif key in {"obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"}:
            migrated[key] = _resize_history_normalizer(
                source_value,
                target_value,
                source_history_len=source_history_len,
                target_history_len=target_history_len,
                frame_dim=frame_dim,
            )
            expanded_keys.append(key)
        else:
            skipped_keys.append(key)

    report = {
        "migration": "task037_adapt_k160_from_short_adapt",
        "source_history_len": source_history_len,
        "target_history_len": target_history_len,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "frame_dim": frame_dim,
        "copied_key_count": len(copied_keys),
        "expanded_keys": expanded_keys,
        "skipped_key_count": len(skipped_keys),
        "skipped_keys": skipped_keys,
        "optimizer_policy": "fresh optimizer required after history-length migration",
    }
    return migrated, report


def migrate_adaptk160_to_task041_true_txl_checkpoint(
    source_checkpoint: Mapping[str, Any],
    *,
    target_actor_state: Mapping[str, Any],
    target_critic_state: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a shape-complete true-TXL warmstart from an AdaptK160 checkpoint.

    The bridge preserves the proven AdaptK160 base policy path when target actor
    shapes match, while leaving TXL projection, attention, and memory state fresh.
    The optimizer is intentionally dropped so continuation starts with a fresh
    optimizer under the Task041 sequence-aware PPO update.
    """

    source_actor = dict(source_checkpoint.get("actor_state_dict") or {})
    source_critic = dict(source_checkpoint.get("critic_state_dict") or {})
    actor_state, actor_report = _copy_shape_matching_state(
        source_actor,
        target_actor_state,
        report_prefix="actor",
    )
    critic_state, critic_report = _copy_shape_matching_state(
        source_critic,
        target_critic_state,
        report_prefix="critic",
    )
    infos = dict(source_checkpoint.get("infos") or {})
    report = {
        "migration": "task041_adaptk160_to_true_txl_warmstart",
        "source_iter": int(source_checkpoint.get("iter") or 0),
        "target_iter": 0,
        "optimizer_policy": "fresh optimizer required after cross-architecture warmstart",
        **actor_report,
        **critic_report,
    }
    infos["task041_adaptk160_true_txl_warmstart"] = report
    migrated = {
        "actor_state_dict": actor_state,
        "critic_state_dict": critic_state,
        "iter": 0,
        "infos": infos,
    }
    return migrated, report


def _copy_shape_matching_state(
    source_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
    *,
    report_prefix: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    copied_state = {
        key: value.clone() if hasattr(value, "clone") else value
        for key, value in target_state.items()
    }
    copied_keys: list[str] = []
    partial_keys: list[str] = []
    fresh_keys: list[str] = []
    skipped_shape_keys: list[str] = []
    unexpected_source_keys: list[str] = []
    for key, source_value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            unexpected_source_keys.append(key)
            continue
        if tuple(source_value.shape) == tuple(target_value.shape):
            copied_state[key] = source_value.clone()
            copied_keys.append(key)
        elif _can_copy_feature_prefix(key, source_value, target_value):
            copied_state[key] = _copy_feature_prefix(
                key,
                source_value,
                target_value,
                source_state=source_state,
            )
            partial_keys.append(key)
        else:
            skipped_shape_keys.append(key)
    for key in target_state:
        if key not in copied_keys and key not in partial_keys:
            fresh_keys.append(key)
    return copied_state, {
        f"{report_prefix}_copied_key_count": len(copied_keys),
        f"{report_prefix}_copied_keys": copied_keys,
        f"{report_prefix}_partial_key_count": len(partial_keys),
        f"{report_prefix}_partial_keys": partial_keys,
        f"{report_prefix}_fresh_key_count": len(fresh_keys),
        f"{report_prefix}_fresh_keys": fresh_keys,
        f"{report_prefix}_skipped_shape_keys": skipped_shape_keys,
        f"{report_prefix}_unexpected_source_keys": unexpected_source_keys,
    }


def _can_copy_feature_prefix(key: str, source_value: Any, target_value: Any) -> bool:
    if key == "mlp.0.weight":
        return (
            len(source_value.shape) == 2
            and len(target_value.shape) == 2
            and int(source_value.shape[0]) == int(target_value.shape[0])
            and int(source_value.shape[1]) < int(target_value.shape[1])
        )
    if key in {"obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"}:
        return (
            len(source_value.shape) == len(target_value.shape)
            and tuple(source_value.shape[:-1]) == tuple(target_value.shape[:-1])
            and int(source_value.shape[-1]) < int(target_value.shape[-1])
        )
    return False


def _copy_feature_prefix(
    key: str,
    source_value: Any,
    target_value: Any,
    *,
    source_state: Mapping[str, Any],
) -> Any:
    if key == "mlp.0.weight":
        copied = target_value.new_zeros(target_value.shape)
    elif key in {"obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"}:
        expanded = _copy_base_obs_normalizer_to_history(
            key,
            source_value,
            target_value,
            source_state=source_state,
        )
        if expanded is not None:
            return expanded
        copied = target_value.clone()
    else:
        copied = target_value.clone()
    source_dim = int(source_value.shape[-1])
    copied[..., :source_dim] = source_value.to(device=target_value.device, dtype=target_value.dtype)
    return copied


def _copy_base_obs_normalizer_to_history(
    key: str,
    source_value: Any,
    target_value: Any,
    *,
    source_state: Mapping[str, Any],
) -> Any | None:
    source_obs_dim = int(source_value.shape[-1])
    action_dim = _infer_action_dim_from_source_state(source_state)
    if action_dim is None:
        return None
    frame_dim = source_obs_dim + action_dim
    target_dim = int(target_value.shape[-1])
    if target_dim <= frame_dim or target_dim % frame_dim != 0:
        return None
    fill = 0.0 if key == "obs_normalizer._mean" else 1.0
    copied = target_value.new_full(target_value.shape, fill)
    history_len = target_dim // frame_dim
    source_cast = source_value.to(device=target_value.device, dtype=target_value.dtype)
    for frame_index in range(history_len):
        start = frame_index * frame_dim
        copied[..., start : start + source_obs_dim] = source_cast
    return copied


def _infer_action_dim_from_source_state(source_state: Mapping[str, Any]) -> int | None:
    std = source_state.get("distribution.std_param")
    if std is None or not hasattr(std, "shape") or len(std.shape) != 1:
        return None
    return int(std.shape[0])


def _infer_history_len(source_state: dict[str, Any], frame_dim: int) -> int:
    for key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
        value = source_state.get(key)
        if value is not None and int(value.shape[-1]) % frame_dim == 0:
            return int(value.shape[-1]) // frame_dim
    return 4


def _resize_history_normalizer(
    source_value: Any,
    target_value: Any,
    *,
    source_history_len: int,
    target_history_len: int,
    frame_dim: int,
) -> Any:
    if int(source_value.shape[-1]) != source_history_len * frame_dim:
        raise ValueError(
            "source history normalizer dim does not match "
            f"source_history_len={source_history_len}, frame_dim={frame_dim}"
        )
    if int(target_value.shape[-1]) != target_history_len * frame_dim:
        raise ValueError(
            "target history normalizer dim does not match "
            f"target_history_len={target_history_len}, frame_dim={frame_dim}"
        )
    source_frames = source_value.reshape(*source_value.shape[:-1], source_history_len, frame_dim)
    newest_frame = source_frames[..., -1:, :]
    target_frames = newest_frame.expand(*source_value.shape[:-1], target_history_len, frame_dim)
    return target_frames.reshape_as(target_value).to(device=target_value.device, dtype=target_value.dtype)
