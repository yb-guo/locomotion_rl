from types import SimpleNamespace

import pytest

from h200_locomotion_lab.training.multitrial_wrapper import (
    TASK037_CONDITION_ID_KEY,
    TASK037_EPISODE_DONE_KEY,
    TASK037_FALL_KEY,
    TASK037_INNER_RESET_KEY,
    TASK037_OUTER_RESET_KEY,
    TASK037_TIMEOUT_KEY,
    TASK037_TRIAL_DONE_KEY,
    TASK037_TRIAL_INDEX_KEY,
    Task037MultiTrialVecEnvWrapper,
)
from h200_locomotion_lab.training.rsl_history_wrapper import Task033HistoryVecEnvWrapper


class ScriptedTask037FakeEnv:
    def __init__(self, torch, events: list[dict[str, list[bool]]]) -> None:
        self.torch = torch
        self.events = events
        self.num_envs = 2
        self.device = torch.device("cpu")
        self.max_episode_length = 128
        self.num_actions = 1
        self.cfg = SimpleNamespace()
        self.unwrapped = self
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.condition_id = torch.tensor([10, 20], dtype=torch.long)
        self.last_action = torch.zeros((self.num_envs, self.num_actions))
        self.pose_marker = torch.zeros(self.num_envs)
        self.step_index = 0
        self.trial_reset_count = torch.zeros(self.num_envs, dtype=torch.long)
        self.episode_reset_count = torch.zeros(self.num_envs, dtype=torch.long)

    def seed(self, seed: int = -1) -> int:
        return seed

    def reset(self):
        self.step_index = 0
        self.episode_length_buf.zero_()
        self.last_action.zero_()
        self.pose_marker.zero_()
        return self.get_observations(), {"reset": "outer"}

    def reset_trial(self, env_ids):
        env_ids = env_ids.to(dtype=self.torch.long)
        self.trial_reset_count[env_ids] += 1
        self.episode_length_buf[env_ids] = 0
        self.last_action[env_ids] = 0.0
        self.pose_marker[env_ids] = 100.0 + self.trial_reset_count[env_ids].to(dtype=self.torch.float32)
        return self.get_observations()

    def reset_episode(self, env_ids):
        env_ids = env_ids.to(dtype=self.torch.long)
        self.episode_reset_count[env_ids] += 1
        self.episode_length_buf[env_ids] = 0
        self.condition_id[env_ids] += 1000
        self.last_action[env_ids] = 0.0
        self.pose_marker[env_ids] = 0.0
        return self.get_observations()

    def get_observations(self):
        actor = self.torch.stack((self.pose_marker, self.last_action[:, 0]), dim=-1)
        critic = self.torch.stack((self.pose_marker + 1.0, self.last_action[:, 0]), dim=-1)
        return {"actor": actor.clone(), "critic": critic.clone()}

    def step(self, actions):
        self.last_action = actions.clone()
        self.pose_marker += 1.0
        self.episode_length_buf += 1
        event = self.events[self.step_index] if self.step_index < len(self.events) else {}
        self.step_index += 1
        fall = self.torch.tensor(event.get("fall", [False, False]), dtype=self.torch.bool)
        timeout = self.torch.tensor(event.get("timeout", [False, False]), dtype=self.torch.bool)
        raw_done = fall | timeout
        reward = self.torch.ones(self.num_envs)
        return self.get_observations(), reward, raw_done, {
            TASK037_FALL_KEY: fall,
            TASK037_TIMEOUT_KEY: timeout,
        }

    def close(self) -> None:
        pass


def test_task037_done_mapping_preserves_conditions_until_final_trial() -> None:
    torch = pytest.importorskip("torch")
    base = ScriptedTask037FakeEnv(
        torch,
        events=[
            {"fall": [True, False]},
            {"timeout": [True, False], "fall": [False, True]},
            {"fall": [True, False]},
        ],
    )
    env = Task037MultiTrialVecEnvWrapper(base, num_trials=3)

    _obs, reset_extras = env.reset()
    assert reset_extras[TASK037_TRIAL_INDEX_KEY].tolist() == [0, 0]

    _obs, _reward, done, extras = env.step(torch.tensor([[5.0], [7.0]]))
    assert done.tolist() == [False, False]
    assert extras[TASK037_TRIAL_DONE_KEY].tolist() == [True, False]
    assert extras[TASK037_EPISODE_DONE_KEY].tolist() == [False, False]
    assert extras[TASK037_INNER_RESET_KEY].tolist() == [True, False]
    assert extras[TASK037_OUTER_RESET_KEY].tolist() == [False, False]
    assert extras[TASK037_CONDITION_ID_KEY].tolist() == [10, 20]
    assert env.trial_index.tolist() == [1, 0]

    _obs, _reward, done, extras = env.step(torch.tensor([[2.0], [3.0]]))
    assert done.tolist() == [False, False]
    assert extras[TASK037_TRIAL_DONE_KEY].tolist() == [True, True]
    assert extras[TASK037_EPISODE_DONE_KEY].tolist() == [False, False]
    assert env.trial_index.tolist() == [2, 1]
    assert extras[TASK037_CONDITION_ID_KEY].tolist() == [10, 20]

    _obs, _reward, done, extras = env.step(torch.tensor([[1.0], [1.0]]))
    assert done.tolist() == [True, False]
    assert extras[TASK037_TRIAL_DONE_KEY].tolist() == [True, False]
    assert extras[TASK037_EPISODE_DONE_KEY].tolist() == [True, False]
    assert extras[TASK037_INNER_RESET_KEY].tolist() == [False, False]
    assert extras[TASK037_OUTER_RESET_KEY].tolist() == [True, False]
    assert env.trial_index.tolist() == [0, 1]
    assert extras[TASK037_CONDITION_ID_KEY].tolist() == [1010, 20]


def test_task037_history_preserves_inner_trials_and_clears_outer_done() -> None:
    torch = pytest.importorskip("torch")
    base = ScriptedTask037FakeEnv(
        torch,
        events=[
            {"fall": [True, False]},
            {"timeout": [True, False]},
            {"fall": [True, False]},
        ],
    )
    env = Task033HistoryVecEnvWrapper(
        Task037MultiTrialVecEnvWrapper(base, num_trials=3),
        history_len=4,
    )

    obs, _extras = env.reset()
    assert not any(key.startswith("task037") for key in obs)
    assert env._buffer is not None
    assert env._buffer.valid_counts.tolist() == [1, 1]

    obs, _reward, done, extras = env.step(torch.tensor([[5.0], [7.0]]))
    assert done.tolist() == [False, False]
    assert extras[TASK037_INNER_RESET_KEY].tolist() == [True, False]
    assert env._buffer.valid_counts.tolist() == [2, 2]
    newest = env._buffer.newest()
    assert newest[0, -1].item() == pytest.approx(0.0)
    assert newest[1, -1].item() == pytest.approx(7.0)
    assert not any(key.startswith("task037") for key in obs)

    _obs, _reward, done, _extras = env.step(torch.tensor([[4.0], [8.0]]))
    assert done.tolist() == [False, False]
    assert env._buffer.valid_counts.tolist() == [3, 3]
    newest = env._buffer.newest()
    assert newest[0, -1].item() == pytest.approx(0.0)

    _obs, _reward, done, extras = env.step(torch.tensor([[2.0], [9.0]]))
    assert done.tolist() == [True, False]
    assert extras[TASK037_OUTER_RESET_KEY].tolist() == [True, False]
    assert env._buffer.valid_counts.tolist() == [1, 4]
    newest = env._buffer.newest()
    assert newest[0, -1].item() == pytest.approx(0.0)
    assert newest[1, -1].item() == pytest.approx(9.0)
