"""Action providers that turn backend state into SONIC raw actions."""

from __future__ import annotations

from typing import Literal, Protocol, Sequence

from h200_locomotion_lab.envs.robot_backend import (
    G1RobotState,
    robot_state_to_planner_qpos,
)
from h200_locomotion_lab.sonic.g1_observation import (
    SONIC_ACTION_DIM,
    SONIC_TOKEN_DIM,
    SonicG1HistoryBuffer,
    SonicG1HistoryFrame,
    build_sonic_g1_decoder_observation,
    mujoco_motor_state_to_sonic_body_state,
)
from h200_locomotion_lab.sonic.g1_planner_encoder import (
    SONIC_ENCODER_OBS_DIM,
    SonicPlannerMotion50Hz,
    build_g1_encoder_observation_from_planner_motion,
    build_initial_planner_context,
    build_planner_context_from_motion,
    build_planner_context_from_mujoco_qpos_history,
)

PlannerContextSource = Literal["live", "motion"]


class SonicPlanner(Protocol):
    def plan(self, context_qpos: Sequence[Sequence[float]] | None) -> SonicPlannerMotion50Hz:
        """Return a 50 Hz planner motion for the supplied 4x36 context."""


class SonicEncoder(Protocol):
    def run(self, observation: Sequence[float]) -> Sequence[float]:
        """Return one 64D token row."""


class SonicDecoder(Protocol):
    def run(self, observation: Sequence[float]) -> Sequence[float]:
        """Return one raw 29D SONIC policy action."""


class SonicPlannerEncoderActionProvider:
    """Online planner -> encoder -> decoder provider for `ScalarG1Runtime`."""

    def __init__(
        self,
        *,
        planner: SonicPlanner,
        encoder: SonicEncoder,
        decoder: SonicDecoder,
        replan_interval: int = 10,
        planner_context_source: PlannerContextSource = "live",
        motion_context_lookahead_steps: int = 2,
    ) -> None:
        if replan_interval < 0:
            raise ValueError("replan_interval must be non-negative")
        if planner_context_source not in ("live", "motion"):
            raise ValueError("planner_context_source must be 'live' or 'motion'")
        if motion_context_lookahead_steps < 0:
            raise ValueError("motion_context_lookahead_steps must be non-negative")
        self.planner = planner
        self.encoder = encoder
        self.decoder = decoder
        self.replan_interval = int(replan_interval)
        self.planner_context_source = planner_context_source
        self.motion_context_lookahead_steps = int(motion_context_lookahead_steps)
        self.history = SonicG1HistoryBuffer()
        self.qpos_history: list[tuple[float, ...]] = []
        self.motion: SonicPlannerMotion50Hz | None = None
        self.motion_start_step = 0
        self.planner_calls = 0

    def reset(self) -> None:
        self.history = SonicG1HistoryBuffer()
        self.qpos_history.clear()
        self.motion = None
        self.motion_start_step = 0
        self.planner_calls = 0

    def action_for_state(self, state: G1RobotState, step_index: int) -> tuple[float, ...]:
        self._record_state(state)
        if self._needs_replan(step_index):
            self._replan(step_index)
        assert self.motion is not None
        motion_frame = step_index - self.motion_start_step
        encoder_observation = build_g1_encoder_observation_from_planner_motion(
            self.motion,
            current_frame=motion_frame,
            robot_base_quat=state.base_quat,
        )
        if len(encoder_observation) != SONIC_ENCODER_OBS_DIM:
            raise RuntimeError("invalid SONIC encoder observation dimension")
        token_state = _coerce_vector(
            self.encoder.run(encoder_observation),
            SONIC_TOKEN_DIM,
            "token_state",
        )
        decoder_observation = build_sonic_g1_decoder_observation(
            token_state,
            self.history.latest_oldest_first(),
        )
        return _coerce_vector(
            self.decoder.run(decoder_observation),
            SONIC_ACTION_DIM,
            "raw_action_isaaclab",
        )

    def _record_state(self, state: G1RobotState) -> None:
        self.qpos_history.append(robot_state_to_planner_qpos(state))
        body_q, body_dq = mujoco_motor_state_to_sonic_body_state(
            state.motor_positions_mujoco,
            state.motor_velocities_mujoco,
        )
        self.history.append(
            SonicG1HistoryFrame(
                base_ang_vel=state.base_angular_velocity,
                body_q=body_q,
                body_dq=body_dq,
                last_action=state.last_action_isaaclab,
                base_quat=state.base_quat,
            )
        )

    def _needs_replan(self, step_index: int) -> bool:
        if self.motion is None:
            return True
        return self.replan_interval > 0 and step_index > 0 and step_index % self.replan_interval == 0

    def _replan(self, step_index: int) -> None:
        context = self._build_replan_context(step_index)
        self.motion = self.planner.plan(context)
        self.motion_start_step = step_index
        self.planner_calls += 1

    def _build_replan_context(self, step_index: int) -> tuple[tuple[float, ...], ...]:
        if self.planner_context_source == "live" or self.motion is None:
            if self.planner_context_source == "motion" and self.motion is None:
                return build_initial_planner_context(self.qpos_history[-1][7:])
            return build_planner_context_from_mujoco_qpos_history(self.qpos_history)
        return build_planner_context_from_motion(
            self.motion,
            gen_frame=step_index - self.motion_start_step,
            motion_look_ahead_steps=self.motion_context_lookahead_steps,
        )


def _coerce_vector(values: Sequence[float], expected_dim: int, name: str) -> tuple[float, ...]:
    if len(values) != expected_dim:
        raise ValueError(f"{name} expected dim={expected_dim}, got {len(values)}")
    return tuple(float(value) for value in values)
