from types import SimpleNamespace

import pytest

from h200_locomotion_lab.training.multitrial_wrapper import (
    EPISODE_DONE_KEY,
    FINAL_TRIAL_KEY,
    RESET_REASON_KEY,
    TASK037_CONDITION_ID_KEY,
    TASK037_EPISODE_DONE_KEY,
    TASK037_FALL_KEY,
    TASK037_INNER_RESET_KEY,
    TASK037_OUTER_RESET_KEY,
    TASK037_TIMEOUT_KEY,
    TASK037_TRIAL_DONE_KEY,
    TASK037_TRIAL_INDEX_KEY,
    TRIAL_DONE_KEY,
    TRIAL_INDEX_KEY,
    Task037MultiTrialVecEnvWrapper,
)
from h200_locomotion_lab.training.rsl_history_wrapper import (
    Task033HistoryVecEnvWrapper,
    _migrate_adaptation_history_len_actor_state_dict,
)


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
    assert extras[TRIAL_DONE_KEY].tolist() == [True, False]
    assert extras[EPISODE_DONE_KEY].tolist() == [False, False]
    assert extras[TRIAL_INDEX_KEY].tolist() == [1, 0]
    assert extras[FINAL_TRIAL_KEY].tolist() == [False, False]
    assert extras[RESET_REASON_KEY].tolist() == [1, 0]
    assert extras[TASK037_INNER_RESET_KEY].tolist() == [True, False]
    assert extras[TASK037_OUTER_RESET_KEY].tolist() == [False, False]
    assert extras[TASK037_CONDITION_ID_KEY].tolist() == [10, 20]
    assert env.trial_index.tolist() == [1, 0]

    _obs, _reward, done, extras = env.step(torch.tensor([[2.0], [3.0]]))
    assert done.tolist() == [False, False]
    assert extras[TASK037_TRIAL_DONE_KEY].tolist() == [True, True]
    assert extras[TASK037_EPISODE_DONE_KEY].tolist() == [False, False]
    assert extras[RESET_REASON_KEY].tolist() == [2, 1]
    assert env.trial_index.tolist() == [2, 1]
    assert extras[TASK037_CONDITION_ID_KEY].tolist() == [10, 20]

    _obs, _reward, done, extras = env.step(torch.tensor([[1.0], [1.0]]))
    assert done.tolist() == [True, False]
    assert extras[TASK037_TRIAL_DONE_KEY].tolist() == [True, False]
    assert extras[TASK037_EPISODE_DONE_KEY].tolist() == [True, False]
    assert extras[RESET_REASON_KEY].tolist() == [1, 0]
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


def test_task044_history_can_clear_visible_history_on_inner_reset() -> None:
    torch = pytest.importorskip("torch")
    base = ScriptedTask037FakeEnv(
        torch,
        events=[
            {"fall": [True, False]},
            {"timeout": [True, False]},
        ],
    )
    env = Task033HistoryVecEnvWrapper(
        Task037MultiTrialVecEnvWrapper(base, num_trials=3),
        history_len=4,
        clear_history_on_inner_reset=True,
    )

    env.reset()
    assert env._buffer is not None
    assert env._buffer.valid_counts.tolist() == [1, 1]

    _obs, _reward, done, extras = env.step(torch.tensor([[5.0], [7.0]]))

    assert done.tolist() == [False, False]
    assert extras[TASK037_INNER_RESET_KEY].tolist() == [True, False]
    assert env._buffer.valid_counts.tolist() == [1, 2]
    newest = env._buffer.newest()
    assert newest[0, -1].item() == pytest.approx(0.0)
    assert newest[1, -1].item() == pytest.approx(7.0)

    _obs, _reward, done, extras = env.step(torch.tensor([[4.0], [8.0]]))

    assert done.tolist() == [False, False]
    assert extras[TASK037_INNER_RESET_KEY].tolist() == [True, False]
    assert env._buffer.valid_counts.tolist() == [1, 3]


def test_task037_local_trial_timeout_takes_precedence_over_raw_done_fall() -> None:
    torch = pytest.importorskip("torch")
    base = ScriptedTask037FakeEnv(
        torch,
        events=[
            {"fall": [True, True]},
        ],
    )
    env = Task037MultiTrialVecEnvWrapper(
        base,
        num_trials=3,
        trial_timeout_steps=1,
    )

    env.reset()
    _obs, _reward, done, extras = env.step(torch.tensor([[1.0], [1.0]]))

    assert done.tolist() == [False, False]
    assert extras[TASK037_TRIAL_DONE_KEY].tolist() == [True, True]
    assert extras[RESET_REASON_KEY].tolist() == [2, 2]


def test_task037_eval_accumulator_reports_velocity_components() -> None:
    torch = pytest.importorskip("torch")
    from h200_locomotion_lab.tools.task037_multitrial_eval_checkpoint import _TrialAccumulator

    stats = _TrialAccumulator(torch, num_envs=2, device="cpu")
    mask = torch.tensor([True, True])
    command = torch.tensor([[1.6, 0.0, 0.0], [1.6, 0.0, 0.0]])
    lin_vel = torch.tensor([[1.2, 0.1], [1.4, -0.3]])

    stats.add_sample(
        mask,
        reward=torch.tensor([1.0, 3.0]),
        command=command,
        lin_vel=lin_vel,
        lin_error=torch.linalg.norm(command[:, :2] - lin_vel, dim=-1),
        yaw_error=torch.tensor([0.0, 0.2]),
        gravity_xy=torch.tensor([0.1, 0.3]),
        root_z=torch.tensor([0.7, 0.8]),
    )

    data = stats.to_json(trial_idx=0, num_envs=2)

    assert data["lin_vel_command"]["mean_x"] == pytest.approx(1.6)
    assert data["lin_vel_command"]["mean_y"] == pytest.approx(0.0)
    assert data["lin_vel_actual"]["mean_x"] == pytest.approx(1.3)
    assert data["lin_vel_actual"]["mean_y"] == pytest.approx(-0.1)
    assert data["lin_vel_error_components"]["mean_abs_x"] == pytest.approx(0.3)
    assert data["lin_vel_error_components"]["mean_abs_y"] == pytest.approx(0.2)


def test_task037_eval_accumulator_caps_completion_ratio_at_one() -> None:
    torch = pytest.importorskip("torch")
    from h200_locomotion_lab.tools.task037_multitrial_eval_checkpoint import _TrialAccumulator

    stats = _TrialAccumulator(torch, num_envs=2, device="cpu")
    stats.add_reset_events(torch.tensor([True, True, True]), torch.tensor([2, 2, 2]))

    data = stats.to_json(trial_idx=0, num_envs=2)

    assert data["completion_count"] == 3
    assert data["completion_ratio"] == pytest.approx(1.0)


def test_task037_txl_k160_memory_preserves_inner_and_clears_outer() -> None:
    torch = pytest.importorskip("torch")
    base = ScriptedTask037FakeEnv(
        torch,
        events=[
            {"timeout": [True, False]},
            {"timeout": [True, False]},
            {"timeout": [True, False]},
        ],
    )
    env = Task033HistoryVecEnvWrapper(
        Task037MultiTrialVecEnvWrapper(base, num_trials=3),
        history_len=160,
    )

    env.reset()
    assert env._buffer is not None
    assert env._buffer.valid_counts.tolist() == [1, 1]

    _obs, _reward, done, extras = env.step(torch.tensor([[1.0], [2.0]]))
    assert done.tolist() == [False, False]
    assert extras[TASK037_INNER_RESET_KEY].tolist() == [True, False]
    assert env._buffer.valid_counts.tolist() == [2, 2]
    assert env._buffer.newest()[0, -1].item() == pytest.approx(0.0)

    _obs, _reward, done, extras = env.step(torch.tensor([[3.0], [4.0]]))
    assert done.tolist() == [False, False]
    assert extras[TASK037_INNER_RESET_KEY].tolist() == [True, False]
    assert env._buffer.valid_counts.tolist() == [3, 3]

    _obs, _reward, done, extras = env.step(torch.tensor([[5.0], [6.0]]))
    assert done.tolist() == [True, False]
    assert extras[TASK037_OUTER_RESET_KEY].tolist() == [True, False]
    assert env._buffer.valid_counts.tolist() == [1, 4]
    assert env._buffer.flatten_latest().shape == (2, 160 * 3)


def test_task037_adapt_k160_warmstart_migration_resizes_history_normalizer() -> None:
    torch = pytest.importorskip("torch")
    frame_dim = 135
    source_history_len = 4
    target_history_len = 160
    source_mean = torch.arange(source_history_len * frame_dim, dtype=torch.float32).reshape(1, -1)
    source_state = {
        "obs_normalizer._mean": source_mean,
        "obs_normalizer._var": torch.ones_like(source_mean),
        "obs_normalizer._std": torch.ones_like(source_mean),
        "mlp.0.weight": torch.ones(512, 136),
        "mlp.0.bias": torch.ones(512),
        "adaptation_encoder.0.weight": torch.ones(128, source_history_len * frame_dim),
        "adaptation_encoder.0.bias": torch.full((128,), 2.0),
    }
    target_state = {
        "obs_normalizer._mean": torch.zeros(1, target_history_len * frame_dim),
        "obs_normalizer._var": torch.ones(1, target_history_len * frame_dim),
        "obs_normalizer._std": torch.ones(1, target_history_len * frame_dim),
        "mlp.0.weight": torch.zeros(512, 136),
        "mlp.0.bias": torch.zeros(512),
        "adaptation_encoder.0.weight": torch.zeros(128, target_history_len * frame_dim),
        "adaptation_encoder.0.bias": torch.zeros(128),
    }

    migrated, report = _migrate_adaptation_history_len_actor_state_dict(
        source_state,
        target_state,
        obs_dim=104,
        action_dim=31,
        target_history_len=target_history_len,
    )

    assert report["source_history_len"] == source_history_len
    assert report["target_history_len"] == target_history_len
    assert migrated["mlp.0.weight"].shape == (512, 136)
    assert migrated["obs_normalizer._mean"].shape == (1, target_history_len * frame_dim)
    latest_source_frame = source_mean.reshape(1, source_history_len, frame_dim)[:, -1, :]
    target_frames = migrated["obs_normalizer._mean"].reshape(1, target_history_len, frame_dim)
    assert torch.equal(target_frames[:, 0, :], latest_source_frame)
    assert torch.equal(target_frames[:, -1, :], latest_source_frame)
    assert "adaptation_encoder.0.weight" not in migrated
    assert torch.equal(migrated["adaptation_encoder.0.bias"], source_state["adaptation_encoder.0.bias"])
