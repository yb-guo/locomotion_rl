"""RSL-RL wrappers for Task033 shared-history policy consumers."""

from __future__ import annotations

from typing import Any

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
    TASK037_ZERO_ACTION_RESET_KEY,
    Task037MultiTrialVecEnvWrapper,
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


class Task033HistoryVecEnvWrapper:
    """Add an actor-visible history observation group without changing the env."""

    def __init__(
        self,
        env: Any,
        *,
        history_len: int,
        actor_group: str = "actor",
        history_group: str = "actor_history",
    ) -> None:
        self.env = env
        self.history_len = history_len
        self.actor_group = actor_group
        self.history_group = history_group
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
        self._append_from_obs(
            obs,
            actions,
            done=dones,
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
                    history_len=4,
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


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - H200-only import path.
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch


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


def _bool_mask(mask: Any, device: Any, num_envs: int) -> Any:
    torch = _require_torch()
    if hasattr(mask, "to"):
        tensor = mask.to(device=device, dtype=torch.bool)
    else:
        tensor = torch.as_tensor(mask, device=device, dtype=torch.bool)
    if tuple(tensor.shape) != (num_envs,):
        raise ValueError(f"mask must have shape ({num_envs},), got {tuple(tensor.shape)}")
    return tensor
