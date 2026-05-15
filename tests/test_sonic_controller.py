from __future__ import annotations

from h200_locomotion_lab.envs.robot_backend import G1RobotState
from h200_locomotion_lab.sonic.controller import SonicPlannerEncoderActionProvider
from h200_locomotion_lab.sonic.g1_observation import SONIC_ACTION_DIM, SONIC_TOKEN_DIM
from h200_locomotion_lab.sonic.g1_planner_encoder import (
    SONIC_PLANNER_DEFAULT_HEIGHT,
    SonicPlannerMotion50Hz,
)
from h200_locomotion_lab.sonic.g1_policy_bridge import SONIC_G1_DEFAULT_ANGLES


class FakePlanner:
    def __init__(
        self,
        root_positions: tuple[tuple[float, float, float], ...] | None = None,
    ) -> None:
        self.contexts = []
        self.root_positions = root_positions or ((0.0, 0.0, 0.8),) * 80

    def plan(self, context_qpos):
        self.contexts.append(context_qpos)
        zeros = (0.0,) * SONIC_ACTION_DIM
        return SonicPlannerMotion50Hz(
            root_positions=self.root_positions,
            root_quats=((1.0, 0.0, 0.0, 0.0),) * 80,
            joint_positions_policy_order=(zeros,) * 80,
            joint_velocities_policy_order=(zeros,) * 80,
        )


class FakeEncoder:
    def __init__(self) -> None:
        self.observations = []

    def run(self, observation):
        self.observations.append(observation)
        return tuple(0.01 * index for index in range(SONIC_TOKEN_DIM))


class FakeDecoder:
    def __init__(self) -> None:
        self.observations = []

    def run(self, observation):
        self.observations.append(observation)
        return tuple(0.02 * index for index in range(SONIC_ACTION_DIM))


def test_sonic_planner_encoder_provider_replans_and_returns_action() -> None:
    planner = FakePlanner()
    encoder = FakeEncoder()
    decoder = FakeDecoder()
    provider = SonicPlannerEncoderActionProvider(
        planner=planner,
        encoder=encoder,
        decoder=decoder,
        replan_interval=2,
    )
    state = G1RobotState(
        root_qpos=(0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0),
        motor_positions_mujoco=SONIC_G1_DEFAULT_ANGLES,
    )

    action0 = provider.action_for_state(state, 0)
    action1 = provider.action_for_state(state, 1)
    action2 = provider.action_for_state(state, 2)

    expected = tuple(0.02 * index for index in range(SONIC_ACTION_DIM))
    assert action0 == expected
    assert action1 == expected
    assert action2 == expected
    assert provider.planner_calls == 2
    assert len(planner.contexts) == 2
    assert all(len(context) == 4 for context in planner.contexts)
    assert len(encoder.observations) == 3
    assert len(decoder.observations) == 3


def test_sonic_planner_encoder_provider_can_replan_from_previous_motion_context() -> None:
    planner = FakePlanner(
        root_positions=tuple((0.01 * index, 0.0, 0.8) for index in range(80))
    )
    provider = SonicPlannerEncoderActionProvider(
        planner=planner,
        encoder=FakeEncoder(),
        decoder=FakeDecoder(),
        replan_interval=2,
        planner_context_source="motion",
    )
    state = G1RobotState(
        root_qpos=(9.0, 8.0, 0.5, 1.0, 0.0, 0.0, 0.0),
        motor_positions_mujoco=SONIC_G1_DEFAULT_ANGLES,
    )

    provider.action_for_state(state, 0)
    provider.action_for_state(state, 1)
    provider.action_for_state(state, 2)

    assert provider.planner_calls == 2
    assert len(planner.contexts) == 2
    first_context = planner.contexts[0]
    second_context = planner.contexts[1]
    assert first_context[0][:7] == (
        0.0,
        0.0,
        SONIC_PLANNER_DEFAULT_HEIGHT,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    assert first_context[0][7:] == SONIC_G1_DEFAULT_ANGLES
    assert 0.03 < second_context[0][0] < 0.05
    assert second_context[0][2] == 0.8
