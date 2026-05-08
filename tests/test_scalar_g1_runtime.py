import math

import pytest

from h200_locomotion_lab.envs.robot_backend import LogReplayG1RobotBackend
from h200_locomotion_lab.runtime.scalar_g1_runtime import (
    FakeActionProvider,
    ScalarG1Runtime,
    SequenceActionProvider,
    ZeroActionProvider,
)
from h200_locomotion_lab.sonic.g1_observation import SONIC_ACTION_DIM


def test_scalar_runtime_drives_log_replay_backend_with_finite_zero_commands() -> None:
    backend = LogReplayG1RobotBackend.from_mujoco_qpos_rows(_qpos_rows())
    runtime = ScalarG1Runtime(backend, ZeroActionProvider())

    steps = runtime.run(2)

    assert len(steps) == 2
    assert len(backend.commands) == 2
    assert steps[0].state.root_z == 0.79
    assert steps[1].next_state.root_qpos[0] == 0.2
    for command in backend.commands:
        assert command.raw_action_isaaclab == (0.0,) * SONIC_ACTION_DIM
        assert all(math.isfinite(value) for value in command.motor_position_targets_mujoco)


def test_sequence_provider_feeds_runtime_actions_in_order() -> None:
    backend = LogReplayG1RobotBackend.from_mujoco_qpos_rows(_qpos_rows())
    provider = SequenceActionProvider(
        (
            (0.1,) * SONIC_ACTION_DIM,
            (0.2,) * SONIC_ACTION_DIM,
        )
    )
    runtime = ScalarG1Runtime(backend, provider)

    steps = runtime.run(2)

    assert steps[0].raw_action_isaaclab == (0.1,) * SONIC_ACTION_DIM
    assert steps[1].raw_action_isaaclab == (0.2,) * SONIC_ACTION_DIM
    assert backend.read_state().last_action_isaaclab == (0.2,) * SONIC_ACTION_DIM


def test_fake_provider_is_deterministic_and_finite() -> None:
    backend = LogReplayG1RobotBackend.from_mujoco_qpos_rows(_qpos_rows())
    provider = FakeActionProvider(amplitude=0.03)

    first = provider.action_for_state(backend.reset(), 3)
    second = provider.action_for_state(backend.read_state(), 3)

    assert first == second
    assert len(first) == SONIC_ACTION_DIM
    assert max(abs(value) for value in first) <= 0.03
    assert all(math.isfinite(value) for value in first)


def test_runtime_rejects_non_finite_provider_action() -> None:
    backend = LogReplayG1RobotBackend.from_mujoco_qpos_rows(_qpos_rows())
    runtime = ScalarG1Runtime(backend, _NonFiniteProvider())

    with pytest.raises(ValueError, match="contains a non-finite value"):
        runtime.run(1)


class _NonFiniteProvider:
    def action_for_state(self, state: object, step_index: int) -> tuple[float, ...]:
        return (float("nan"),) + (0.0,) * (SONIC_ACTION_DIM - 1)


def _qpos_rows() -> tuple[tuple[float, ...], ...]:
    return (
        (0.0, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0) + (0.0,) * SONIC_ACTION_DIM,
        (0.1, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0) + (0.01,) * SONIC_ACTION_DIM,
        (0.2, 0.0, 0.77, 1.0, 0.0, 0.0, 0.0) + (0.02,) * SONIC_ACTION_DIM,
    )
