"""Replay SONIC reference joint targets through the Genesis G1 backend."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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

EFFORT_LIMIT_5020 = 25.0
EFFORT_LIMIT_7520_14 = 88.0
EFFORT_LIMIT_7520_22 = 139.0
EFFORT_LIMIT_4010 = 5.0

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

SONIC_G1_FORCE_LIMITS: tuple[float, ...] = (
    EFFORT_LIMIT_7520_22,
    EFFORT_LIMIT_7520_22,
    EFFORT_LIMIT_7520_14,
    EFFORT_LIMIT_7520_22,
    EFFORT_LIMIT_5020,
    EFFORT_LIMIT_5020,
    EFFORT_LIMIT_7520_22,
    EFFORT_LIMIT_7520_22,
    EFFORT_LIMIT_7520_14,
    EFFORT_LIMIT_7520_22,
    EFFORT_LIMIT_5020,
    EFFORT_LIMIT_5020,
    EFFORT_LIMIT_7520_14,
    EFFORT_LIMIT_5020,
    EFFORT_LIMIT_5020,
    EFFORT_LIMIT_5020,
    EFFORT_LIMIT_5020,
    EFFORT_LIMIT_5020,
    EFFORT_LIMIT_5020,
    EFFORT_LIMIT_5020,
    EFFORT_LIMIT_4010,
    EFFORT_LIMIT_4010,
    EFFORT_LIMIT_5020,
    EFFORT_LIMIT_5020,
    EFFORT_LIMIT_5020,
    EFFORT_LIMIT_5020,
    EFFORT_LIMIT_5020,
    EFFORT_LIMIT_4010,
    EFFORT_LIMIT_4010,
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
    body_quat_rows = _read_csv_rows(reference_dir / "body_quat.csv", 56)
    frames = min(args.frames, len(joint_rows))
    if frames <= 0:
        raise ValueError("frames must be positive")

    root0 = tuple(body_rows[0][:3])
    root_quat0 = tuple(body_quat_rows[0][:4])
    print("SONIC_REFERENCE_REPLAY_MODE joint_pos_as_position_targets")
    print("ASSET", asset)
    print("REF_DIR", reference_dir)
    print("JOINT_POS_ROWS", len(joint_rows))
    print("REPLAY_FRAMES", frames)
    print("ROOT0", root0)
    print("ROOT_QUAT0", root_quat0)
    print("TARGET0_MIN_MAX", min(joint_rows[0]), max(joint_rows[0]))
    print("TARGET_LAST_MIN_MAX", min(joint_rows[frames - 1]), max(joint_rows[frames - 1]))
    print("KPS_MIN_MAX", min(SONIC_G1_KPS), max(SONIC_G1_KPS))
    print("KDS_MIN_MAX", min(SONIC_G1_KDS), max(SONIC_G1_KDS))
    print("FORCE_LIMITS_MIN_MAX", min(SONIC_G1_FORCE_LIMITS), max(SONIC_G1_FORCE_LIMITS))

    env = GenesisG1Env.from_genesis_asset(
        str(asset),
        backend=args.backend,
        base_pos=(0.0, 0.0, 0.0),
        base_quat=(1.0, 0.0, 0.0, 0.0),
        root_qpos=root0 + root_quat0,
        convexify=args.convexify,
        decimate=args.decimate,
        logging_level=args.logging_level,
    )
    backend = env.backend
    robot = backend.robot  # type: ignore[attr-defined]
    motor_idx = backend.motor_dof_indices  # type: ignore[attr-defined]
    print("MOTOR_DOF_COUNT", len(motor_idx))
    print("MOTOR_DOF_INDICES", motor_idx)

    apply_sonic_g1_motor_config(robot, motor_idx)
    _set_floating_root_qpos(robot, root0 + root_quat0)
    robot.set_dofs_position(tuple(joint_rows[0]), dofs_idx_local=motor_idx, zero_velocity=True)
    robot.set_dofs_velocity(None)

    base_heights: list[float] = []
    base_positions: list[tuple[float, float, float]] = []
    min_link_heights: list[float] = []
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
        pos = _read_floating_base_position(robot, motor_idx)
        min_link_z = _read_min_link_height(robot)
        contact_count, max_contact_force = _read_contact_metrics(robot)
        if min_link_z is not None:
            min_link_heights.append(min_link_z)
        base_positions.append(pos)
        finite_ok = (
            finite_ok
            and _is_finite(q)
            and _is_finite(v)
            and _is_finite(pos)
            and (max_contact_force is None or math.isfinite(max_contact_force))
        )
        err = [abs(actual - expected) for actual, expected in zip(q, target)]
        mean_abs_errors.append(sum(err) / len(err))
        max_abs_errors.append(max(err))
        max_abs_qvel.append(max(abs(value) for value in v))
        base_heights.append(pos[2])
        if frame in {0, 1, 2, 9, 49, frames - 1}:
            print(
                "FRAME",
                frame,
                "base_x",
                pos[0],
                "base_y",
                pos[1],
                "base_z",
                pos[2],
                "min_link_z",
                min_link_z,
                "contact_count",
                contact_count,
                "max_contact_force",
                max_contact_force,
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
    displacement_xy = _horizontal_distance(base_positions[0], base_positions[-1])
    path_xy = _xy_path_length(base_positions)
    duration_s = len(base_positions) / 50.0
    height_ok = (
        args.min_base_height <= base_min <= args.max_base_height
        and args.min_base_height <= base_final <= args.max_base_height
    )
    print("FINITE_OK", finite_ok)
    print("ROOT_X_START", base_positions[0][0])
    print("ROOT_Y_START", base_positions[0][1])
    print("ROOT_X_FINAL", base_positions[-1][0])
    print("ROOT_Y_FINAL", base_positions[-1][1])
    print("HORIZONTAL_DISPLACEMENT", displacement_xy)
    print("PATH_LENGTH_XY", path_xy)
    print("AVERAGE_SPEED_XY", displacement_xy / duration_s if duration_s > 0 else 0.0)
    print("BASE_HEIGHT_MIN", base_min)
    print("BASE_HEIGHT_MAX", base_max)
    print("BASE_HEIGHT_FINAL", base_final)
    if min_link_heights:
        print("MIN_LINK_HEIGHT_MIN", min(min_link_heights))
        print("MIN_LINK_HEIGHT_FINAL", min_link_heights[-1])
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


def apply_sonic_g1_motor_config(robot: Any, motor_idx: Sequence[int]) -> None:
    """Apply SONIC motor gains using the Genesis official control-demo API shape."""

    robot.set_dofs_kp(SONIC_G1_KPS, dofs_idx_local=motor_idx)
    robot.set_dofs_kv(SONIC_G1_KDS, dofs_idx_local=motor_idx)
    robot.set_dofs_force_range(
        tuple(-limit for limit in SONIC_G1_FORCE_LIMITS),
        SONIC_G1_FORCE_LIMITS,
        dofs_idx_local=motor_idx,
    )


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


def _read_min_link_height(robot: Any) -> float | None:
    if not hasattr(robot, "get_links_pos"):
        return None
    link_pos = _flatten_numeric(robot.get_links_pos())
    if len(link_pos) < 3:
        return None
    return min(link_pos[index] for index in range(2, len(link_pos), 3))


def _read_contact_metrics(robot: Any) -> tuple[int | None, float | None]:
    return (_read_contact_count(robot), _read_max_link_contact_force(robot))


def _read_contact_count(robot: Any) -> int | None:
    if not hasattr(robot, "get_contacts"):
        return None
    try:
        contacts = robot.get_contacts(exclude_self_contact=True)
    except TypeError:
        contacts = robot.get_contacts()
    if not contacts:
        return 0
    if "valid_mask" in contacts:
        return sum(1 for value in _flatten_numeric(contacts["valid_mask"]) if bool(value))
    for key in ("geom_a", "link_a", "position"):
        if key not in contacts:
            continue
        values = _flatten_numeric(contacts[key])
        return len(values) // 3 if key == "position" else len(values)
    return 0


def _read_max_link_contact_force(robot: Any) -> float | None:
    if not hasattr(robot, "get_links_net_contact_force"):
        return None
    forces = _flatten_numeric(robot.get_links_net_contact_force())
    if len(forces) < 3:
        return None
    return max(
        math.sqrt(
            forces[index] * forces[index]
            + forces[index + 1] * forces[index + 1]
            + forces[index + 2] * forces[index + 2]
        )
        for index in range(0, len(forces) - 2, 3)
    )


def _read_floating_base_position(
    robot: Any,
    motor_dof_indices: Sequence[int],
) -> tuple[float, float, float]:
    """Read dynamic root translation from free-joint DOFs when present.

    The first controlled motor DOF starts after the root free-joint DOFs. Prefer
    qpos because a free joint has 7 q coordinates, while DOF position exposes a
    6-DoF representation.
    """

    if motor_dof_indices:
        root_dof_count = min(int(index) for index in motor_dof_indices)
        if root_dof_count >= 3:
            all_positions = _read_all_qpos(robot)
            if len(all_positions) >= 3:
                return (
                    float(all_positions[0]),
                    float(all_positions[1]),
                    float(all_positions[2]),
                )
    return _read_spawn_position(robot)


def _read_spawn_position(robot: Any) -> tuple[float, float, float]:
    pos = _flatten_numeric(robot.get_pos())
    if len(pos) < 3:
        raise ValueError("robot.get_pos() returned fewer than 3 values")
    return (pos[0], pos[1], pos[2])


def _set_floating_root_qpos(
    robot: Any,
    root_qpos: Sequence[float],
) -> None:
    values = tuple(float(value) for value in root_qpos)
    if len(values) != 7:
        raise ValueError(f"Expected root_qpos length 7, got {len(values)}")
    if hasattr(robot, "set_qpos"):
        robot.set_qpos(values, qs_idx_local=tuple(range(7)), zero_velocity=True)
        return
    robot.set_pos(values[:3], zero_velocity=True)
    robot.set_quat(values[3:], zero_velocity=True)


def _read_all_qpos(robot: Any) -> tuple[float, ...]:
    if hasattr(robot, "get_qpos"):
        return _flatten_numeric(robot.get_qpos())
    return _read_all_dof_positions(robot)


def _read_all_dof_positions(robot: Any) -> tuple[float, ...]:
    try:
        return _flatten_numeric(robot.get_dofs_position())
    except TypeError:
        n_dofs = int(robot.n_dofs)
        return _flatten_numeric(robot.get_dofs_position(dofs_idx_local=tuple(range(n_dofs))))


def _horizontal_distance(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    return math.sqrt(dx * dx + dy * dy)


def _xy_path_length(positions: Sequence[tuple[float, float, float]]) -> float:
    return sum(_horizontal_distance(prev, curr) for prev, curr in itertools.pairwise(positions))


if __name__ == "__main__":
    main()
