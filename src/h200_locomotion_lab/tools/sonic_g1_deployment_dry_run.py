"""Dry-run the SONIC G1 body deployment bridge from recorded qpos/action rows."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

from h200_locomotion_lab.envs.robot_backend import (
    G1MotorCommand,
    LogReplayG1RobotBackend,
    robot_state_to_sonic_history_frame,
)
from h200_locomotion_lab.sonic.g1_observation import (
    SONIC_ACTION_DIM,
    SONIC_DECODER_OBS_DIM,
    SONIC_TOKEN_DIM,
    SonicG1HistoryBuffer,
    build_sonic_g1_decoder_observation,
)
from h200_locomotion_lab.sonic.g1_planner_encoder import SONIC_PLANNER_QPOS_DIM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qpos-csv", required=True, help="36D root+motor qpos rows.")
    parser.add_argument("--raw-action-csv", help="Optional 29D raw SONIC action rows.")
    parser.add_argument("--token-csv", help="Optional 64D token rows.")
    parser.add_argument("--frames", type=int, help="Number of replay frames.")
    parser.add_argument("--policy-rate-hz", type=float, default=50.0)
    parser.add_argument("--output-summary-csv", required=True)
    args = parser.parse_args()

    qpos_rows = read_numeric_csv(Path(args.qpos_csv), SONIC_PLANNER_QPOS_DIM)
    action_rows = (
        read_numeric_csv(Path(args.raw_action_csv), SONIC_ACTION_DIM)
        if args.raw_action_csv
        else None
    )
    token_rows = read_numeric_csv(Path(args.token_csv), SONIC_TOKEN_DIM) if args.token_csv else None
    frames = args.frames if args.frames is not None else len(qpos_rows)
    rows = run_dry_run(
        qpos_rows,
        raw_action_rows=action_rows,
        token_rows=token_rows,
        frames=frames,
        policy_rate_hz=args.policy_rate_hz,
    )
    output_summary = Path(args.output_summary_csv)
    write_summary_csv(output_summary, rows)

    obs_finite = all(row["obs_finite"] == "True" for row in rows)
    action_finite = all(row["raw_action_finite"] == "True" for row in rows)
    target_finite = all(row["target_finite"] == "True" for row in rows)
    print("SONIC_G1_DEPLOYMENT_DRY_RUN_MODE log_replay")
    print("QPOS_SOURCE", args.qpos_csv)
    print("RAW_ACTION_SOURCE", args.raw_action_csv or "zero")
    print("TOKEN_SOURCE", args.token_csv or "zero")
    print("FRAMES", frames)
    print("OBS_DIM", SONIC_DECODER_OBS_DIM)
    print("OBS_FINITE", obs_finite)
    print("RAW_ACTION_FINITE", action_finite)
    print("TARGET_FINITE", target_finite)
    print("ROOT_Z_MIN", min(float(row["root_z"]) for row in rows))
    print("ROOT_Z_MAX", max(float(row["root_z"]) for row in rows))
    print("RAW_ACTION_MAX_ABS", max(float(row["raw_action_max_abs"]) for row in rows))
    print("TARGET_MAX_ABS", max(float(row["target_max_abs"]) for row in rows))
    print("OUTPUT_SUMMARY_CSV", output_summary)
    if not obs_finite or not action_finite or not target_finite:
        raise SystemExit("SONIC_G1_DEPLOYMENT_DRY_RUN_FAILED")
    print("SONIC_G1_DEPLOYMENT_DRY_RUN_OK")


def run_dry_run(
    qpos_rows: Sequence[Sequence[float]],
    *,
    raw_action_rows: Sequence[Sequence[float]] | None = None,
    token_rows: Sequence[Sequence[float]] | None = None,
    frames: int | None = None,
    policy_rate_hz: float = 50.0,
) -> list[dict[str, str]]:
    if frames is None:
        frames = len(qpos_rows)
    if frames <= 0:
        raise ValueError("frames must be positive")
    if frames > len(qpos_rows):
        raise ValueError(f"frames={frames} exceeds qpos rows={len(qpos_rows)}")
    backend = LogReplayG1RobotBackend.from_mujoco_qpos_rows(
        qpos_rows[:frames],
        policy_rate_hz=policy_rate_hz,
    )
    history = SonicG1HistoryBuffer()
    backend.reset()
    summary_rows: list[dict[str, str]] = []
    for frame_index in range(frames):
        state = backend.read_state()
        history.append(robot_state_to_sonic_history_frame(state))
        token = row_for_frame(token_rows, frame_index, SONIC_TOKEN_DIM, "token")
        observation = build_sonic_g1_decoder_observation(token, history.latest_oldest_first())
        raw_action = row_for_frame(
            raw_action_rows,
            frame_index,
            SONIC_ACTION_DIM,
            "raw_action",
        )
        command = G1MotorCommand.from_raw_sonic_action(raw_action)
        backend.write_command(command)
        summary_rows.append(frame_summary(frame_index, state, observation, command))
        if frame_index < frames - 1:
            backend.advance()
    return summary_rows


def frame_summary(
    frame_index: int,
    state: object,
    observation: Sequence[float],
    command: G1MotorCommand,
) -> dict[str, str]:
    raw_action = command.raw_action_isaaclab
    target = command.motor_position_targets_mujoco
    row = {
        "frame": str(frame_index),
        "root_x": f"{state.root_qpos[0]:.9g}",  # type: ignore[attr-defined]
        "root_y": f"{state.root_qpos[1]:.9g}",  # type: ignore[attr-defined]
        "root_z": f"{state.root_qpos[2]:.9g}",  # type: ignore[attr-defined]
        "obs_dim": str(len(observation)),
        "obs_finite": str(is_finite(observation)),
        "raw_action_finite": str(is_finite(raw_action)),
        "target_finite": str(is_finite(target)),
        "raw_action_max_abs": f"{max_abs(raw_action):.9g}",
        "target_min": f"{min(target):.9g}",
        "target_max": f"{max(target):.9g}",
        "target_max_abs": f"{max_abs(target):.9g}",
    }
    for index, value in enumerate(raw_action):
        row[f"raw_action_{index:02d}"] = f"{value:.9g}"
    for index, value in enumerate(target):
        row[f"target_mujoco_{index:02d}"] = f"{value:.9g}"
    return row


def row_for_frame(
    rows: Sequence[Sequence[float]] | None,
    frame_index: int,
    dim: int,
    name: str,
) -> tuple[float, ...]:
    if rows is None:
        return (0.0,) * dim
    if not rows:
        raise ValueError(f"{name} rows must not be empty")
    row_index = 0 if len(rows) == 1 else frame_index
    if row_index >= len(rows):
        raise ValueError(f"{name} rows={len(rows)} are shorter than frame {frame_index}")
    return coerce_vector(rows[row_index], dim, name)


def read_numeric_csv(path: Path, dim: int) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        for row_index, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue
            fields = tuple(value.strip() for value in row if value.strip())
            try:
                values = tuple(float(value) for value in fields)
            except ValueError:
                if row_index == 1:
                    continue
                raise ValueError(f"{path}:{row_index} contains a non-numeric value")
            rows.append(coerce_vector(values, dim, f"{path}:{row_index}"))
    if not rows:
        raise ValueError(f"{path} did not contain any numeric rows")
    return tuple(rows)


def write_summary_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("summary rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0].keys())
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def coerce_vector(values: Sequence[float], dim: int, name: str) -> tuple[float, ...]:
    if len(values) != dim:
        raise ValueError(f"{name} expected dim={dim}, got {len(values)}")
    result = tuple(float(value) for value in values)
    if not is_finite(result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def is_finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def max_abs(values: Sequence[float]) -> float:
    return max(abs(float(value)) for value in values)


if __name__ == "__main__":
    main()

