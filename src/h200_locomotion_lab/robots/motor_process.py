"""Hidden, time-varying actuator conditions for whole-body adaptation tests."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

MotorEventKind = Literal["weak", "dead", "latency"]


@dataclass(frozen=True, slots=True)
class MotorEvent:
    """One actuator event in control-step coordinates."""

    slot: str
    kind: MotorEventKind
    onset_step: int
    duration_steps: int
    value: float
    persistent: bool = False

    @property
    def end_step(self) -> int:
        return self.onset_step + self.duration_steps

    def active_at(self, step: int) -> bool:
        return step >= self.onset_step and (self.persistent or step < self.end_step)


@dataclass(frozen=True, slots=True)
class MotorProcessConfig:
    """Distribution and timing defaults from the whole-body plan."""

    control_hz: float = 50.0
    no_event_probability: float = 0.30
    max_events: int = 2
    onset_range_seconds: tuple[float, float] = (1.0, 7.0)
    duration_range_seconds: tuple[float, float] = (1.0, 3.0)
    persistent_probability: float = 0.50
    weak_probability: float = 0.60
    dead_probability: float = 0.20
    weak_range: tuple[float, float] = (0.3, 0.7)
    dead_range: tuple[float, float] = (0.0, 0.1)
    extra_latency_steps: tuple[int, int] = (1, 3)

    def __post_init__(self) -> None:
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if not 0.0 <= self.no_event_probability <= 1.0:
            raise ValueError("no_event_probability must be in [0, 1]")
        if self.max_events < 1:
            raise ValueError("max_events must be positive")
        if self.onset_range_seconds[0] < 0 or self.onset_range_seconds[0] > self.onset_range_seconds[1]:
            raise ValueError("onset range must be non-negative and ordered")
        if self.duration_range_seconds[0] <= 0 or self.duration_range_seconds[0] > self.duration_range_seconds[1]:
            raise ValueError("duration range must be positive and ordered")


@dataclass(frozen=True, slots=True)
class MotorState:
    """Physical actuator state exposed to critic/diagnostics, not actor."""

    strength: tuple[float, ...]
    extra_latency_steps: tuple[int, ...]
    ema_alpha: tuple[float, ...]
    events: tuple[MotorEvent, ...]

    def critic_payload(self) -> dict[str, object]:
        return {
            "strength": self.strength,
            "extra_latency_steps": self.extra_latency_steps,
            "ema_alpha": self.ema_alpha,
        }


class MotorProcess:
    """Deterministic event scheduler with context and trial reset boundaries."""

    def __init__(
        self,
        active_slots: tuple[str, ...],
        *,
        config: MotorProcessConfig | None = None,
        baseline_strength: tuple[float, ...] | None = None,
        baseline_latency_steps: tuple[int, ...] | None = None,
        baseline_ema_alpha: tuple[float, ...] | None = None,
    ) -> None:
        self.active_slots = tuple(active_slots)
        self.config = config or MotorProcessConfig()
        count = len(self.active_slots)
        self._baseline_strength = baseline_strength or (1.0,) * count
        self._baseline_latency = baseline_latency_steps or (0,) * count
        self._baseline_ema = baseline_ema_alpha or (1.0,) * count
        if not (
            len(self._baseline_strength) == len(self._baseline_latency) == len(self._baseline_ema) == count
        ):
            raise ValueError("all baseline actuator vectors must match active slot count")
        if any(value <= 0 for value in self._baseline_strength):
            raise ValueError("baseline strength must be positive")
        if any(not 0.0 < value <= 1.0 for value in self._baseline_ema):
            raise ValueError("baseline EMA alpha must be in (0, 1]")
        self._events: tuple[MotorEvent, ...] = ()
        self._action_history: list[tuple[float, ...]] = []
        self._last_processed_action = (0.0,) * count

    def reset_context(self, seed: int, *, trial_seconds: float = 10.0) -> MotorState:
        """Resample baseline faults/events at a new latent physical context."""

        rng = random.Random(seed)
        self._events = self._sample_events(rng, trial_seconds=trial_seconds)
        self._action_history: list[tuple[float, ...]] = []
        self._last_processed_action = (0.0,) * len(self.active_slots)
        return self.state_at(0)

    def reset_trial(self) -> MotorState:
        """Reset physical pose while preserving persistent context events."""

        self._action_history = []
        self._last_processed_action = (0.0,) * len(self.active_slots)
        return self.state_at(0)

    def process_action(self, action: tuple[float, ...], step: int) -> tuple[float, ...]:
        """Apply hidden strength, delay, and EMA dynamics to one joint action."""

        if len(action) != len(self.active_slots):
            raise ValueError("action length must match active slots")
        if step < 0:
            raise ValueError("step must be non-negative")
        state = self.state_at(step)
        self._action_history.append(tuple(float(value) for value in action))
        output: list[float] = []
        for index, _slot in enumerate(self.active_slots):
            delay = state.extra_latency_steps[index]
            source_index = max(0, len(self._action_history) - 1 - delay)
            delayed = self._action_history[source_index][index]
            target = delayed * state.strength[index]
            alpha = state.ema_alpha[index]
            previous = self._last_processed_action[index]
            output.append(alpha * target + (1.0 - alpha) * previous)
        self._last_processed_action = tuple(output)
        return self._last_processed_action

    def state_at(self, step: int) -> MotorState:
        if step < 0:
            raise ValueError("step must be non-negative")
        strength = list(self._baseline_strength)
        latency = list(self._baseline_latency)
        ema = list(self._baseline_ema)
        slot_to_index = {slot: index for index, slot in enumerate(self.active_slots)}
        for event in self._events:
            index = slot_to_index[event.slot]
            if not event.active_at(step):
                continue
            if event.kind == "weak" or event.kind == "dead":
                strength[index] = min(strength[index], event.value)
            elif event.kind == "latency":
                latency[index] = max(latency[index], int(event.value))
        return MotorState(tuple(strength), tuple(latency), tuple(ema), self._events)

    @property
    def events(self) -> tuple[MotorEvent, ...]:
        return self._events

    def trace(self, steps: int) -> tuple[MotorState, ...]:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        return tuple(self.state_at(step) for step in range(steps))

    def _sample_events(self, rng: random.Random, *, trial_seconds: float) -> tuple[MotorEvent, ...]:
        if rng.random() < self.config.no_event_probability or not self.active_slots:
            return ()
        event_count = rng.randint(1, self.config.max_events)
        onset_max = max(self.config.onset_range_seconds[0], min(
            self.config.onset_range_seconds[1], trial_seconds - self.config.duration_range_seconds[0]
        ))
        events: list[MotorEvent] = []
        selected_slots = self._sample_slots(rng, min(event_count, len(self.active_slots)))
        for slot in selected_slots:
            onset_seconds = rng.uniform(self.config.onset_range_seconds[0], onset_max)
            duration_seconds = rng.uniform(*self.config.duration_range_seconds)
            onset = round(onset_seconds * self.config.control_hz)
            duration = max(1, round(duration_seconds * self.config.control_hz))
            kind_draw = rng.random()
            if kind_draw < self.config.weak_probability:
                kind: MotorEventKind = "weak"
                value = rng.uniform(*self.config.weak_range)
            elif kind_draw < self.config.weak_probability + self.config.dead_probability:
                kind = "dead"
                value = rng.uniform(*self.config.dead_range)
            else:
                kind = "latency"
                value = float(rng.randint(*self.config.extra_latency_steps))
            events.append(
                MotorEvent(
                    slot=slot,
                    kind=kind,
                    onset_step=onset,
                    duration_steps=duration,
                    value=value,
                    persistent=rng.random() < self.config.persistent_probability,
                )
            )
        return tuple(sorted(events, key=lambda event: (event.onset_step, event.slot)))

    def _sample_slots(self, rng: random.Random, count: int) -> list[str]:
        """Prefer leg/waist events with the planned 70/30 split when possible."""

        lower_body = [slot for slot in self.active_slots if slot.startswith(("limb", "waist_"))]
        upper_body = [slot for slot in self.active_slots if slot.startswith(("left_arm_", "right_arm_"))]
        selected: list[str] = []
        for _ in range(count):
            available_lower = [slot for slot in lower_body if slot not in selected]
            available_upper = [slot for slot in upper_body if slot not in selected]
            if not available_lower and not available_upper:
                break
            if available_lower and available_upper:
                pool = available_lower if rng.random() < 0.7 else available_upper
            else:
                pool = available_lower or available_upper
            selected.append(rng.choice(pool))
        return selected
