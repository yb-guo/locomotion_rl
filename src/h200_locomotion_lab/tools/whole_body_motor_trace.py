"""Write a deterministic hidden-motor event trace for Task055 review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from h200_locomotion_lab.robots.motor_process import MotorProcess, MotorProcessConfig


def run_trace(*, seed: int, steps: int) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    active = (
        "limb0_hip_pitch",
        "limb0_knee_pitch",
        "limb1_knee_pitch",
        "waist_yaw",
        "left_arm_elbow_pitch",
        "right_arm_elbow_pitch",
    )
    process = MotorProcess(
        active,
        config=MotorProcessConfig(no_event_probability=0.0, max_events=2),
    )
    process.reset_context(seed, trial_seconds=10.0)
    trace = process.trace(steps)
    return {
        "seed": seed,
        "active_slots": active,
        "events": [
            {
                "slot": event.slot,
                "kind": event.kind,
                "onset_step": event.onset_step,
                "duration_steps": event.duration_steps,
                "persistent": event.persistent,
                "value": event.value,
            }
            for event in process.events
        ],
        "strength_trace": [list(state.strength) for state in trace],
        "latency_trace": [list(state.extra_latency_steps) for state in trace],
        "actor_payload_keys": sorted(process.state_at(0).critic_payload()),
        "context_reset_events": [event.slot for event in process.events],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=55001)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_trace(seed=args.seed, steps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
