import importlib.util
import sys

import pytest

from h200_locomotion_lab.training.ppo_loop import (
    PPOConfig,
    RolloutBatch,
    build_actor_critic,
    compute_gae,
    describe_training_plan,
    parameter_l1_sum,
    ppo_update,
    collect_rollout,
    tanh_gaussian_log_prob_from_action,
)


def test_ppo_loop_imports_without_requiring_torch() -> None:
    assert "Run a small PPO baseline." in describe_training_plan()
    if importlib.util.find_spec("torch") is None:
        assert "torch" not in sys.modules


def test_ppo_config_validates_batch_shape() -> None:
    with pytest.raises(ValueError, match="minibatch_size must not exceed"):
        PPOConfig(n_envs=2, rollout_steps=2, minibatch_size=5)


def test_compute_gae_known_values() -> None:
    torch = pytest.importorskip("torch")
    config = PPOConfig(n_envs=1, rollout_steps=3, minibatch_size=3, gamma=1.0, gae_lambda=1.0)
    batch = RolloutBatch(
        observations=torch.zeros((3, 1, 90)),
        actions=torch.zeros((3, 1, 27)),
        rewards=torch.tensor([[1.0], [1.0], [1.0]]),
        dones=torch.tensor([[False], [False], [True]]),
        values=torch.zeros((3, 1)),
        log_probs=torch.zeros((3, 1)),
        next_observation=torch.zeros((1, 90)),
        next_value=torch.zeros((1,)),
        collect_time_s=1.0,
        env_steps=3,
        reward_mean=1.0,
        done_count=1,
        timeout_count=0,
        fallen_count=1,
        reset_count=1,
        height_bad_count=0,
        termination_height_bad_count=0,
        tilt_bad_count=1,
        height_reset_count=0,
        tilt_reset_count=1,
        full_env_reset_wave=False,
        full_env_reset_wave_count=0,
        episode_length_mean=2.0,
        episode_length_min=1.0,
        episode_length_max=3.0,
        completed_episode_length_mean=3.0,
        completed_episode_count=1,
        root_height_mean=0.8,
        root_height_min=0.4,
        upright_mean=1.0,
    )

    advantages, returns = compute_gae(batch, config)

    assert advantages.squeeze(1).tolist() == pytest.approx([3.0, 2.0, 1.0])
    assert returns.squeeze(1).tolist() == pytest.approx([3.0, 2.0, 1.0])


def test_tanh_gaussian_log_prob_is_finite_near_bounds() -> None:
    torch = pytest.importorskip("torch")
    config = PPOConfig(n_envs=1, rollout_steps=1, minibatch_size=1)
    mean = torch.zeros((2, config.action_dim))
    log_std = torch.zeros_like(mean)
    action = torch.full_like(mean, 0.999999)

    log_prob = tanh_gaussian_log_prob_from_action(action, mean, log_std, config)

    assert torch.isfinite(log_prob).all()
    assert log_prob.shape == (2,)


def test_actor_critic_shapes() -> None:
    torch = pytest.importorskip("torch")
    config = PPOConfig(n_envs=4, rollout_steps=2, minibatch_size=4)
    model = build_actor_critic(config, device="cpu")
    observation = torch.zeros((4, config.obs_dim))

    action, log_prob, value, entropy = model.act(observation)

    assert action.shape == (4, config.action_dim)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)
    assert entropy.shape == (4,)
    assert torch.isfinite(action).all()
    assert action.min() >= -1.0
    assert action.max() <= 1.0


def test_ppo_update_changes_actor_and_value_params() -> None:
    torch = pytest.importorskip("torch")
    config = PPOConfig(n_envs=4, rollout_steps=3, minibatch_size=6, epochs=2)
    model = build_actor_critic(config, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    observations = torch.randn((config.rollout_steps, config.n_envs, config.obs_dim))
    with torch.no_grad():
        flat_obs = observations.reshape(-1, config.obs_dim)
        actions, log_probs, values, _entropy = model.act(flat_obs)
    batch = RolloutBatch(
        observations=observations,
        actions=actions.reshape(config.rollout_steps, config.n_envs, config.action_dim),
        rewards=torch.randn((config.rollout_steps, config.n_envs)),
        dones=torch.zeros((config.rollout_steps, config.n_envs), dtype=torch.bool),
        values=values.reshape(config.rollout_steps, config.n_envs),
        log_probs=log_probs.reshape(config.rollout_steps, config.n_envs),
        next_observation=torch.zeros((config.n_envs, config.obs_dim)),
        next_value=torch.zeros((config.n_envs,)),
        collect_time_s=1.0,
        env_steps=config.rollout_steps * config.n_envs,
        reward_mean=0.0,
        done_count=0,
        timeout_count=0,
        fallen_count=0,
        reset_count=0,
        height_bad_count=0,
        termination_height_bad_count=0,
        tilt_bad_count=0,
        height_reset_count=0,
        tilt_reset_count=0,
        full_env_reset_wave=False,
        full_env_reset_wave_count=0,
        episode_length_mean=1.0,
        episode_length_min=1.0,
        episode_length_max=1.0,
        completed_episode_length_mean=0.0,
        completed_episode_count=0,
        root_height_mean=0.8,
        root_height_min=0.8,
        upright_mean=1.0,
    )
    advantages, returns = compute_gae(batch, config)
    actor_before = parameter_l1_sum(model.actor)
    value_before = parameter_l1_sum(model.value)

    diagnostics = ppo_update(model, optimizer, batch, advantages, returns, config)

    assert parameter_l1_sum(model.actor) != pytest.approx(actor_before)
    assert parameter_l1_sum(model.value) != pytest.approx(value_before)
    assert diagnostics.update_samples_per_sec > 0.0
    assert diagnostics.entropy > 0.0


def test_collect_rollout_reports_episode_length_and_reset_rates_inputs() -> None:
    torch = pytest.importorskip("torch")
    config = PPOConfig(n_envs=2, rollout_steps=3, minibatch_size=3)
    env = _MetricEnv(torch, config)
    model = _MetricModel(torch, config)

    batch = collect_rollout(
        env,
        model,
        torch.zeros((config.n_envs, config.obs_dim)),
        config,
    )

    assert batch.reset_count == 3
    assert batch.height_bad_count == 2
    assert batch.termination_height_bad_count == 1
    assert batch.height_reset_count == 1
    assert batch.tilt_bad_count == 1
    assert batch.tilt_reset_count == 1
    assert batch.timeout_count == 1
    assert batch.full_env_reset_wave
    assert batch.full_env_reset_wave_count == 1
    assert batch.episode_length_mean == pytest.approx(2.0)
    assert batch.episode_length_min == 1.0
    assert batch.episode_length_max == 3.0
    assert batch.completed_episode_count == 2
    assert batch.completed_episode_length_mean == pytest.approx(2.5)


class _MetricModel:
    def __init__(self, torch: object, config: PPOConfig) -> None:
        self.torch = torch
        self.config = config

    def act(self, observation: object) -> tuple[object, object, object, object]:
        action = self.torch.zeros((self.config.n_envs, self.config.action_dim))
        log_prob = self.torch.zeros((self.config.n_envs,))
        value = self.torch.zeros((self.config.n_envs,))
        entropy = self.torch.zeros((self.config.n_envs,))
        return action, log_prob, value, entropy

    def forward(self, observation: object) -> tuple[object, object]:
        return (
            self.torch.zeros((self.config.n_envs, self.config.action_dim)),
            self.torch.zeros((self.config.n_envs,)),
        )


class _MetricEnv:
    def __init__(self, torch: object, config: PPOConfig) -> None:
        self.torch = torch
        self.config = config
        self.step_index = 0

    def step(self, action: object) -> object:
        self.step_index += 1
        done = self.torch.tensor(
            [
                [False, False],
                [True, False],
                [True, True],
            ][self.step_index - 1]
        )
        terminated = self.torch.tensor(
            [
                [False, False],
                [True, False],
                [False, True],
            ][self.step_index - 1]
        )
        truncated = self.torch.tensor(
            [
                [False, False],
                [False, False],
                [True, False],
            ][self.step_index - 1]
        )
        height_bad = self.torch.tensor(
            [
                [False, False],
                [True, False],
                [False, True],
            ][self.step_index - 1]
        )
        termination_height_bad = self.torch.tensor(
            [
                [False, False],
                [True, False],
                [False, False],
            ][self.step_index - 1]
        )
        tilt_bad = self.torch.tensor(
            [
                [False, False],
                [False, False],
                [False, True],
            ][self.step_index - 1]
        )
        completed = [
            self.torch.tensor([], dtype=self.torch.long),
            self.torch.tensor([2]),
            self.torch.tensor([3]),
        ][self.step_index - 1]
        return _Transition(
            observation=self.torch.zeros((self.config.n_envs, self.config.obs_dim)),
            reward=self.torch.ones((self.config.n_envs,)),
            terminated=terminated,
            truncated=truncated,
            done=done,
            info={
                "reset_count": int(done.sum().item()),
                "full_env_reset_wave": bool(done.all().item()),
                "episode_lengths": self.torch.full((self.config.n_envs,), self.step_index),
                "completed_episode_lengths": completed,
                "components": {
                    "height_bad": height_bad,
                    "termination_height_bad": termination_height_bad,
                    "tilt_bad": tilt_bad,
                    "root_height": self.torch.full((self.config.n_envs,), 0.8),
                    "upright": self.torch.ones((self.config.n_envs,)),
                },
            },
        )


class _Transition:
    def __init__(
        self,
        *,
        observation: object,
        reward: object,
        terminated: object,
        truncated: object,
        done: object,
        info: dict[str, object],
    ) -> None:
        self.observation = observation
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.done = done
        self.info = info
