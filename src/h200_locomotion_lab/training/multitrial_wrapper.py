"""Task037 multi-trial vectorized environment contract wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TASK037_TRIAL_DONE_KEY = "task037_trial_done"
TASK037_EPISODE_DONE_KEY = "task037_episode_done"
TASK037_INNER_RESET_KEY = "task037_inner_reset"
TASK037_OUTER_RESET_KEY = "task037_outer_reset"
TASK037_ZERO_ACTION_RESET_KEY = "task037_zero_action_reset"
TASK037_TRIAL_INDEX_KEY = "task037_trial_index"
TASK037_FINAL_TRIAL_KEY = "task037_final_trial"
TASK037_FALL_KEY = "task037_fall"
TASK037_TIMEOUT_KEY = "task037_trial_timeout"
TASK037_RESET_REASON_KEY = "task037_reset_reason"
TASK037_CONDITION_ID_KEY = "task037_condition_id"
TRIAL_DONE_KEY = "trial_done"
EPISODE_DONE_KEY = "episode_done"
INNER_RESET_KEY = "inner_reset"
OUTER_RESET_KEY = "outer_reset"
TRIAL_INDEX_KEY = "trial_index"
FINAL_TRIAL_KEY = "final_trial"
RESET_REASON_KEY = "reset_reason"


@dataclass(frozen=True, slots=True)
class Task037MultiTrialConfig:
    """Fixed multi-trial reset contract for in-context adaptation."""

    num_trials: int = 3
    reset_strategy: str = "explicit"
    trial_timeout_steps: int | None = None

    def __post_init__(self) -> None:
        if self.num_trials <= 0:
            raise ValueError("num_trials must be positive")
        if self.reset_strategy not in {"explicit", "auto"}:
            raise ValueError("reset_strategy must be 'explicit' or 'auto'")
        if self.trial_timeout_steps is not None and self.trial_timeout_steps <= 0:
            raise ValueError("trial_timeout_steps must be positive when set")


class Task037MultiTrialVecEnvWrapper:
    """Convert raw per-trial resets into outer episode resets.

    The wrapped env is expected to emit normal vector-env step tuples. A raw
    trial end is computed as `fall OR trial_timeout OR raw_done`. Inner trials
    reset the wrapped simulator but keep runner-facing `done=False`; the final
    trial maps to runner-facing `done=True`.
    """

    def __init__(
        self,
        env: Any,
        *,
        num_trials: int = 3,
        reset_strategy: str = "explicit",
        trial_timeout_steps: int | None = None,
    ) -> None:
        self.env = env
        self.config = Task037MultiTrialConfig(
            num_trials=num_trials,
            reset_strategy=reset_strategy,
            trial_timeout_steps=trial_timeout_steps,
        )
        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.num_actions = env.num_actions
        torch = _require_torch()
        self.trial_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.trial_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

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
        self.trial_index.zero_()
        self.trial_step.zero_()
        return obs, self._with_contract_extras(extras or {})

    def get_observations(self) -> Any:
        return self.env.get_observations()

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        obs, rewards, raw_done, extras = _coerce_step_result(self.env.step(actions))
        extras = dict(extras or {})
        self.trial_step += 1
        raw_done = _as_bool_tensor(raw_done, self.device, self.num_envs)
        fall = _extras_bool(extras, TASK037_FALL_KEY, self.device, self.num_envs)
        timeout = _extras_bool(extras, TASK037_TIMEOUT_KEY, self.device, self.num_envs)
        if "time_outs" in extras:
            timeout = timeout | _as_bool_tensor(extras["time_outs"], self.device, self.num_envs)
        if self.config.trial_timeout_steps is not None:
            timeout = timeout | (self.trial_step >= self.config.trial_timeout_steps)
        fall = fall | (raw_done & ~timeout)
        trial_done = raw_done | fall | timeout
        final_trial = self.trial_index >= self.config.num_trials - 1
        episode_done = trial_done & final_trial
        inner_reset = trial_done & ~episode_done
        outer_reset = episode_done

        if bool(inner_reset.any().item()):
            inner_ids = _mask_to_env_ids(inner_reset)
            self.trial_index[inner_ids] += 1
            self.trial_step[inner_ids] = 0
            if self.config.reset_strategy == "explicit":
                reset_obs = self._reset_env_ids(inner_ids, episode=False)
                obs = _merge_obs_for_env_ids(obs, reset_obs, inner_ids)

        if bool(outer_reset.any().item()):
            outer_ids = _mask_to_env_ids(outer_reset)
            self.trial_index[outer_ids] = 0
            self.trial_step[outer_ids] = 0
            if self.config.reset_strategy == "explicit":
                reset_obs = self._reset_env_ids(outer_ids, episode=True)
                obs = _merge_obs_for_env_ids(obs, reset_obs, outer_ids)

        extras[TASK037_TRIAL_DONE_KEY] = trial_done
        extras[TASK037_EPISODE_DONE_KEY] = episode_done
        extras[TASK037_INNER_RESET_KEY] = inner_reset
        extras[TASK037_OUTER_RESET_KEY] = outer_reset
        extras[TASK037_ZERO_ACTION_RESET_KEY] = trial_done
        extras[TASK037_TRIAL_INDEX_KEY] = self.trial_index.clone()
        extras[TASK037_FINAL_TRIAL_KEY] = final_trial
        reset_reason = _reset_reason(raw_done, fall, timeout)
        extras[TASK037_RESET_REASON_KEY] = reset_reason
        extras[TRIAL_DONE_KEY] = trial_done
        extras[EPISODE_DONE_KEY] = episode_done
        extras[INNER_RESET_KEY] = inner_reset
        extras[OUTER_RESET_KEY] = outer_reset
        extras[TRIAL_INDEX_KEY] = self.trial_index.clone()
        extras[FINAL_TRIAL_KEY] = final_trial
        extras[RESET_REASON_KEY] = reset_reason
        extras["task037_reset_strategy"] = self.config.reset_strategy
        extras["task037_trial_step"] = self.trial_step.clone()
        condition_id = self._condition_id()
        if condition_id is not None:
            extras[TASK037_CONDITION_ID_KEY] = condition_id
        return obs, rewards, episode_done, extras

    def close(self) -> None:
        return self.env.close()

    def _reset_env_ids(self, env_ids: Any, *, episode: bool) -> Any:
        if episode and hasattr(self.env, "reset_episode"):
            return self.env.reset_episode(env_ids)
        if not episode and hasattr(self.env, "reset_trial"):
            return self.env.reset_trial(env_ids)
        if hasattr(self.env, "reset_envs"):
            return self.env.reset_envs(env_ids)
        if hasattr(self.env, "reset_idx"):
            result = self.env.reset_idx(env_ids)
            return result if result is not None else self.env.get_observations()
        raise AttributeError(
            "wrapped env must provide reset_trial/reset_episode, reset_envs, or reset_idx"
        )

    def _condition_id(self) -> Any | None:
        value = getattr(self.env, "condition_id", None)
        if value is None:
            return None
        return value.clone() if hasattr(value, "clone") else value

    def _with_contract_extras(self, extras: dict[str, Any]) -> dict[str, Any]:
        torch = _require_torch()
        false_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        extras = dict(extras)
        extras.setdefault(TASK037_TRIAL_DONE_KEY, false_mask.clone())
        extras.setdefault(TASK037_EPISODE_DONE_KEY, false_mask.clone())
        extras.setdefault(TASK037_INNER_RESET_KEY, false_mask.clone())
        extras.setdefault(TASK037_OUTER_RESET_KEY, false_mask.clone())
        extras.setdefault(TASK037_ZERO_ACTION_RESET_KEY, false_mask.clone())
        extras.setdefault(TASK037_TRIAL_INDEX_KEY, self.trial_index.clone())
        extras.setdefault(TASK037_FINAL_TRIAL_KEY, false_mask.clone())
        extras.setdefault(
            TASK037_RESET_REASON_KEY,
            torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
        )
        extras.setdefault(TRIAL_DONE_KEY, false_mask.clone())
        extras.setdefault(EPISODE_DONE_KEY, false_mask.clone())
        extras.setdefault(INNER_RESET_KEY, false_mask.clone())
        extras.setdefault(OUTER_RESET_KEY, false_mask.clone())
        extras.setdefault(TRIAL_INDEX_KEY, self.trial_index.clone())
        extras.setdefault(FINAL_TRIAL_KEY, false_mask.clone())
        extras.setdefault(
            RESET_REASON_KEY,
            torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
        )
        extras.setdefault("task037_reset_strategy", self.config.reset_strategy)
        extras.setdefault("task037_trial_step", self.trial_step.clone())
        condition_id = self._condition_id()
        if condition_id is not None:
            extras.setdefault(TASK037_CONDITION_ID_KEY, condition_id)
        return extras


def _coerce_step_result(result: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    if len(result) == 4:
        obs, rewards, done, extras = result
        return obs, rewards, done, extras
    if len(result) == 5:
        obs, rewards, terminated, truncated, extras = result
        return obs, rewards, terminated | truncated, extras
    if len(result) == 3:
        obs, rewards, done = result
        return obs, rewards, done, {}
    raise ValueError(f"Unsupported env.step result length: {len(result)}")


def _extras_bool(extras: dict[str, Any], key: str, device: Any, num_envs: int) -> Any:
    torch = _require_torch()
    if key not in extras:
        return torch.zeros(num_envs, device=device, dtype=torch.bool)
    return _as_bool_tensor(extras[key], device, num_envs)


def _as_bool_tensor(value: Any, device: Any, num_envs: int) -> Any:
    torch = _require_torch()
    if hasattr(value, "to"):
        tensor = value.to(device=device, dtype=torch.bool)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=torch.bool)
    if tuple(tensor.shape) != (num_envs,):
        raise ValueError(f"mask must have shape ({num_envs},), got {tuple(tensor.shape)}")
    return tensor


def _mask_to_env_ids(mask: Any) -> Any:
    return mask.nonzero(as_tuple=False).flatten()


def _reset_reason(raw_done: Any, fall: Any, timeout: Any) -> Any:
    torch = _require_torch()
    reason = torch.zeros_like(raw_done, dtype=torch.long)
    reason[raw_done] = 3
    reason[fall] = 1
    reason[timeout] = 2
    return reason


def _merge_obs_for_env_ids(obs: Any, reset_obs: Any, env_ids: Any) -> Any:
    if isinstance(obs, dict):
        merged = dict(obs)
        for key, value in obs.items():
            merged[key] = _merge_obs_for_env_ids(value, reset_obs[key], env_ids)
        return merged

    merged = obs.clone() if hasattr(obs, "clone") else obs.copy()
    if reset_obs.shape[0] == obs.shape[0]:
        selected = reset_obs.index_select(0, env_ids)
    elif reset_obs.shape[0] == env_ids.numel():
        selected = reset_obs
    else:
        raise ValueError(
            "reset observation first dimension must match num_envs or selected env count"
        )
    merged[env_ids] = selected
    return merged


def _require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - local no-torch path.
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch
