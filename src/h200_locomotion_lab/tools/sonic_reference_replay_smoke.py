"""Replay SONIC reference joint targets through the Genesis G1 backend."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any, Sequence

from h200_locomotion_lab.envs.genesis_adapter import GenesisG1Env


ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10 * 2.0 * math.pi
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ * NATURAL_FREQ

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

SONIC_G1_KPS: tuple[float, ...] = (
    STIFFNESS_7520_22,
    STIFFNESS_7520_22,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
    2.0 * STIFFNESS_5020,
    2.0 * STIFFNESS_5020,
    STIFFNESS_7520_22,
    STIFFNESS_7520_22,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
    2.0 * STIFFNESS_5020,
    2.0 * STIFFNESS_5020,
    STIFFNESS_7520_14,
    2.0 * STIFFNESS_5020,
    2.0 * STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_4010,
    STIFFNESS_4010,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_4010,
    STIFFNESS_4010,
)

SONIC_G1_KDS: tuple[float, ...] = (
    DAMPING_7520_22,
    DAMPING_7520_22,
    DAMPING_7520_14,
    DAMPING_7520_22,
    2.0 * DAMPING_5020,
    2.0 * DAMPING_5020,
    DAMPING_7520_22,
    DAMPING_7520_22,
    DAMPING_7520_14,
    DAMPING_7520_22,
    2.0 * DAMPING_5020,
    2.0 * DAMPING_5020,
    DAMPING_7520_14,
    2.0 * DAMPING_5020,
    2.0 * DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_4010,
    DAMPING_4010,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_4010,
    DAMPING_4010,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, help="Path to SONIC G1 29-motor MJCF.")
    parser.add_argument("--reference-dir", required=True, help="SONIC reference clip directory.")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--logging-level", default="warning")
    parser.add_argument("--convexify", action="store_true")
    parser.add_argument("--decimate", action="store_true")
    parser.add_argument("--min-base-height", type=float, default=0.2)
    parser.add_argument("--max-base-height", type=float, default=1.5)
    args = parser.parse_args()

    asset = Path(args.asset)
    reference_dir = Path(args.reference_dir)
    joint_rows = _read_csv_rows(reference_dir / "joint_pos.csv", 29)
    body_rows = _read_csv_rows(reference_dir / "body_pos.csv", 42)
    frames = min(args.frames, len(joint_rows))
    if frames <= 0:
        raise ValueError("frames must be positive")

    root0 = tuple(body_rows[0][:3])
    print("SONIC_REFERENCE_REPLAY_MODE joint_pos_as_position_targets")
    print("ASSET", asset)
    print("REF_DIR", reference_dir)
    print("JOINT_POS_ROWS", len(joint_rows))
    print("REPLAY_FRAMES", frames)
    print("ROOT0", root0)
    print("TARGET0_MIN_MAX", min(joint_rows[0]), max(joint_rows[0]))
    print("TARGET_LAST_MIN_MAX", min(joint_rows[frames - 1]), max(joint_rows[frames - 1]))
    print("KPS_MIN_MAX", min(SONIC_G1_KPS), max(SONIC_G1_KPS))
    print("KDS_MIN_MAX", min(SONIC_G1_KDS), max(SONIC_G1_KDS))

    env = GenesisG1Env.from_genesis_asset(
        str(asset),
        backend=args.backend,
        convexify=args.convexify,
        decimate=args.decimate,
        logging_level=args.logging_level,
    )
    backend = env.backend
    robot = backend.robot  # type: ignore[attr-defined]
    motor_idx = backend.motor_dof_indices  # type: ignore[attr-defined]
    print("MOTOR_DOF_COUNT", len(motor_idx))
    print("MOTOR_DOF_INDICES", motor_idx)

    robot.set_dofs_kp(SONIC_G1_KPS, dofs_idx_local=motor_idx)
    robot.set_dofs_kv(SONIC_G1_KDS, dofs_idx_local=motor_idx)
    robot.set_pos(root0)
    robot.set_dofs_position(tuple(joint_rows[0]), dofs_idx_local=motor_idx, zero_velocity=True)
    robot.set_dofs_velocity(None)

    base_heights: list[float] = []
    mean_abs_errors: list[float] = []
    max_abs_errors: list[float] = []
    max_abs_qvel: list[float] = []
    finite_ok = True
    start = time.time()
    for frame, target in enumerate(joint_rows[:frames]):
        robot.control_dofs_position(tuple(target), dofs_idx_local=motor_idx)
        for _ in range(env.contract.decimation):
            backend.scene.step()  # type: ignore[attr-defined]
        q = _flatten_numeric(robot.get_dofs_position(dofs_idx_local=motor_idx))
        v = _flatten_numeric(robot.get_dofs_velocity(dofs_idx_local=motor_idx))
        pos = _flatten_numeric(robot.get_pos())
        finite_ok = finite_ok and _is_finite(q) and _is_finite(v) and _is_finite(pos)
        err = [abs(actual - expected) for actual, expected in zip(q, target)]
        mean_abs_errors.append(sum(err) / len(err))
        max_abs_errors.append(max(err))
        max_abs_qvel.append(max(abs(value) for value in v))
        base_heights.append(pos[2])
        if frame in {0, 1, 2, 9, 49, frames - 1}:
            print(
                "FRAME",
                frame,
                "base_z",
                pos[2],
                "mean_abs_err",
                mean_abs_errors[-1],
                "max_abs_err",
                max_abs_errors[-1],
                "max_abs_qvel",
                max_abs_qvel[-1],
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
    print("MEAN_ABS_TRACKING_ERROR_AVG", sum(mean_abs_errors) / len(mean_abs_errors))
    print("MAX_ABS_TRACKING_ERROR", max(max_abs_errors))
    print("MAX_ABS_QVEL", max(max_abs_qvel))
    print("ELAPSED_S", elapsed)
    print("SIM_STEPS", frames * env.contract.decimation)
    print("HEIGHT_OK_RANGE", args.min_base_height, args.max_base_height, height_ok)
    if not finite_ok:
        raise SystemExit("non-finite state during SONIC reference replay")
    if not height_ok:
        raise SystemExit("base height left smoke range during SONIC reference replay")
    print("SONIC_REFERENCE_REPLAY_GENESIS_SMOKE_OK")


def _read_csv_rows(path: Path, cols: int) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != cols:
            raise ValueError(f"{path} expected {cols} columns, got {len(header)}")
        for row in reader:
            rows.append([float(value) for value in row])
    return rows


def _flatten_numeric(values: Any) -> tuple[float, ...]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "reshape"):
        values = values.reshape(-1)
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        return (float(values),)
    return tuple(float(value) for value in values)


def _is_finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


if __name__ == "__main__":
    main()
