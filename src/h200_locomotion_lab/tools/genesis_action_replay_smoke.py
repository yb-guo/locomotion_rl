"""Replay explicit 29D normalized actions through the Genesis G1 env."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Sequence

from h200_locomotion_lab.envs.genesis_adapter import GenesisG1Contract, GenesisG1Env
from h200_locomotion_lab.tools.sonic_reference_replay_smoke import (
    apply_sonic_g1_motor_config,
    _flatten_numeric,
    _read_floating_base_position,
    _is_finite,
    _read_min_link_height,
)


ActionRows = list[tuple[float, ...]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, help="Path to SONIC G1 29-motor MJCF.")
    parser.add_argument("--actions-csv", help="CSV with one 29D normalized action per row.")
    parser.add_argument(
        "--fixture",
        choices=("zero", "sine", "pulse"),
        default="sine",
        help="Built-in deterministic action sequence used when --actions-csv is omitted.",
    )
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--amplitude", type=float, default=0.12)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--logging-level", default="warning")
    parser.add_argument("--convexify", action="store_true")
    parser.add_argument("--decimate", action="store_true")
    parser.add_argument("--base-pos", nargs=3, type=float, default=(0.0, 0.0, 0.8))
    parser.add_argument("--base-quat", nargs=4, type=float, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument(
        "--default-joint-pos-csv",
        help="CSV whose selected 29D row is used as the nominal action-zero pose.",
    )
    parser.add_argument("--default-joint-pos-row", type=int, default=0)
    parser.add_argument("--min-base-height", type=float, default=0.2)
    parser.add_argument("--max-base-height", type=float, default=1.5)
    parser.add_argument("--no-sonic-motor-config", action="store_true")
    args = parser.parse_args()

    contract = GenesisG1Contract()
    actions = load_action_sequence(
        actions_csv=Path(args.actions_csv) if args.actions_csv else None,
        fixture=args.fixture,
        frames=args.frames,
        action_dim=contract.action_dim,
        amplitude=args.amplitude,
    )

    action_min, action_max, action_max_abs = action_range(actions)
    clipped_count = count_out_of_range_actions(actions)
    print("GENESIS_ACTION_REPLAY_MODE normalized_actions")
    print("ASSET", Path(args.asset))
    print("ACTIONS_SOURCE", args.actions_csv or f"fixture:{args.fixture}")
    print("REPLAY_FRAMES", len(actions))
    print("ACTION_DIM", contract.action_dim)
    print("ACTION_MIN_MAX", action_min, action_max)
    print("ACTION_MAX_ABS", action_max_abs)
    print("ACTION_OUT_OF_RANGE_VALUES", clipped_count)
    print("ACTION_SCALE_RAD", contract.action_scale_rad)
    print("BASE_POS", tuple(args.base_pos))
    print("BASE_QUAT", tuple(args.base_quat))
    default_motor_positions = (
        read_default_joint_positions(
            Path(args.default_joint_pos_csv),
            args.default_joint_pos_row,
            contract.action_dim,
        )
        if args.default_joint_pos_csv
        else None
    )
    print("DEFAULT_JOINT_POS_SOURCE", args.default_joint_pos_csv or "asset_qpos0")
    if default_motor_positions is not None:
        print("DEFAULT_JOINT_POS_ROW", args.default_joint_pos_row)
        print("DEFAULT_JOINT_POS_MIN_MAX", min(default_motor_positions), max(default_motor_positions))

    env = GenesisG1Env.from_genesis_asset(
        args.asset,
        backend=args.backend,
        base_pos=tuple(args.base_pos),
        base_quat=tuple(args.base_quat),
        default_motor_positions=default_motor_positions,
        convexify=args.convexify,
        decimate=args.decimate,
        logging_level=args.logging_level,
    )
    backend = env.backend
    if backend is None:
        raise RuntimeError("Genesis backend was not created")
    robot = backend.robot  # type: ignore[attr-defined]
    motor_idx = backend.motor_dof_indices  # type: ignore[attr-defined]
    print("MOTOR_DOF_COUNT", len(motor_idx))
    print("MOTOR_DOF_INDICES", motor_idx)
    print("BASE_HEIGHT_SOURCE", "floating_base_dof" if min(motor_idx) >= 3 else "spawn_pose")

    if not args.no_sonic_motor_config:
        apply_sonic_g1_motor_config(robot, motor_idx)
        print("MOTOR_CONFIG", "sonic_g1_kp_kv_force_range")
    else:
        print("MOTOR_CONFIG", "genesis_default")

    observation = env.reset(seed=0)
    print("RESET_OBS_LEN", len(observation))

    base_heights: list[float] = []
    min_link_heights: list[float] = []
    max_abs_qvel: list[float] = []
    finite_ok = _is_finite(observation)
    start = time.time()
    for frame, action in enumerate(actions):
        result = env.step(action)
        observation = result.observation
        qvel = _flatten_numeric(robot.get_dofs_velocity(dofs_idx_local=motor_idx))
        base_pos = _read_floating_base_position(robot, motor_idx)
        min_link_z = _read_min_link_height(robot)

        finite_ok = (
            finite_ok
            and _is_finite(observation)
            and _is_finite(qvel)
            and _is_finite(base_pos)
            and (min_link_z is None or math.isfinite(min_link_z))
        )
        base_heights.append(base_pos[2])
        if min_link_z is not None:
            min_link_heights.append(min_link_z)
        max_abs_qvel.append(max(abs(value) for value in qvel))

        if frame in {0, 1, 2, 9, 49, len(actions) - 1}:
            clipped = clip_action(action)
            print(
                "FRAME",
                frame,
                "base_z",
                base_pos[2],
                "min_link_z",
                min_link_z,
                "action_min",
                min(clipped),
                "action_max",
                max(clipped),
                "max_abs_qvel",
                max_abs_qvel[-1],
                "obs_len",
                len(observation),
            )

    elapsed = time.time() - start
    base_min = min(base_heights)
    base_max = max(base_heights)
    base_final = base_heights[-1]
    height_ok = (
        args.min_base_height <= base_min <= args.max_base_height
        and args.min_base_height <= base_final <= args.max_base_height
    )
    print("FINITE_OK", finite_ok)
    print("BASE_HEIGHT_MIN", base_min)
    print("BASE_HEIGHT_MAX", base_max)
    print("BASE_HEIGHT_FINAL", base_final)
    if min_link_heights:
        print("MIN_LINK_HEIGHT_MIN", min(min_link_heights))
        print("MIN_LINK_HEIGHT_FINAL", min_link_heights[-1])
    print("MAX_ABS_QVEL", max(max_abs_qvel))
    print("ELAPSED_S", elapsed)
    print("POLICY_STEPS", len(actions))
    print("SIM_STEPS", len(actions) * contract.decimation)
    print("HEIGHT_OK_RANGE", args.min_base_height, args.max_base_height, height_ok)
    if not finite_ok:
        raise SystemExit("non-finite state during Genesis action replay")
    if not height_ok:
        raise SystemExit("base height left smoke range during Genesis action replay")
    print("GENESIS_ACTION_REPLAY_SMOKE_OK")


def load_action_sequence(
    *,
    actions_csv: Path | None,
    fixture: str,
    frames: int,
    action_dim: int,
    amplitude: float,
) -> ActionRows:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if actions_csv is not None:
        rows = read_action_csv(actions_csv, action_dim)
        if not rows:
            raise ValueError(f"{actions_csv} contains no action rows")
        return rows[: min(frames, len(rows))]
    return build_action_fixture(fixture, frames, action_dim, amplitude)


def read_action_csv(path: Path, action_dim: int) -> ActionRows:
    rows: ActionRows = []
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        for row_index, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue
            try:
                values = tuple(float(value) for value in row)
            except ValueError:
                if row_index == 1:
                    continue
                raise ValueError(f"{path}:{row_index} contains a non-numeric action value")
            if len(values) != action_dim:
                raise ValueError(
                    f"{path}:{row_index} expected {action_dim} action columns, got {len(values)}"
                )
            rows.append(values)
    return rows


def read_default_joint_positions(path: Path, row_index: int, action_dim: int) -> tuple[float, ...]:
    if row_index < 0:
        raise ValueError("default joint position row must be non-negative")
    rows = read_action_csv(path, action_dim)
    if row_index >= len(rows):
        raise ValueError(f"{path} has {len(rows)} data rows, cannot read row {row_index}")
    return rows[row_index]


def build_action_fixture(
    fixture: str,
    frames: int,
    action_dim: int,
    amplitude: float,
) -> ActionRows:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative")

    if fixture == "zero":
        return [(0.0,) * action_dim for _ in range(frames)]

    if fixture == "sine":
        rows: ActionRows = []
        denom = max(frames - 1, 1)
        for frame in range(frames):
            phase = 2.0 * math.pi * frame / denom
            rows.append(
                tuple(amplitude * math.sin(phase + joint_index * 0.37) for joint_index in range(action_dim))
            )
        return rows

    if fixture == "pulse":
        rows = []
        active = {0, 3, 6, 9, 12, 15, 18, 22, 25}
        for frame in range(frames):
            sign = 1.0 if (frame // 10) % 2 == 0 else -1.0
            rows.append(
                tuple(sign * amplitude if joint_index in active else 0.0 for joint_index in range(action_dim))
            )
        return rows

    raise ValueError(f"Unknown action fixture: {fixture}")


def clip_action(action: Sequence[float]) -> tuple[float, ...]:
    return tuple(max(-1.0, min(1.0, float(value))) for value in action)


def count_out_of_range_actions(actions: Sequence[Sequence[float]]) -> int:
    return sum(1 for row in actions for value in row if float(value) < -1.0 or float(value) > 1.0)


def action_range(actions: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    values = [float(value) for row in actions for value in row]
    if not values:
        raise ValueError("actions must not be empty")
    return min(values), max(values), max(abs(value) for value in values)


if __name__ == "__main__":
    main()
