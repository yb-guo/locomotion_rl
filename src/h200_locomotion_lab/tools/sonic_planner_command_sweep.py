"""Sweep SONIC planner command inputs without stepping a simulator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from h200_locomotion_lab.sonic.g1_planner_encoder import build_initial_planner_context
from h200_locomotion_lab.sonic.planner_runner import (
    SonicPlannerCommand,
    SubprocessSonicPlanner,
)


DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "posx": (1.0, 0.0, 0.0),
    "negx": (-1.0, 0.0, 0.0),
    "posy": (0.0, 1.0, 0.0),
    "negy": (0.0, -1.0, 0.0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", required=True)
    parser.add_argument("--planner-runner", required=True)
    parser.add_argument("--planner-work-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", type=int, default=2)
    parser.add_argument("--target-vels", default="-1.0,-0.5,0.5,1.0")
    parser.add_argument("--directions", default="posx,negx")
    parser.add_argument("--height", type=float, default=-1.0)
    parser.add_argument("--random-seed", type=int, default=1234)
    args = parser.parse_args()

    rows = run_sweep(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "rows": rows}, indent=2, sort_keys=True))


def run_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_vels = parse_csv_floats(args.target_vels)
    direction_names = parse_csv_strings(args.directions)
    for direction_name in direction_names:
        direction = direction_by_name(direction_name)
        for target_vel in target_vels:
            planner = SubprocessSonicPlanner(
                planner=Path(args.planner),
                planner_runner=Path(args.planner_runner),
                work_dir=Path(args.planner_work_dir) / f"{direction_name}_{format_float_tag(target_vel)}",
                command=SonicPlannerCommand(
                    mode=int(args.mode),
                    target_vel=float(target_vel),
                    height=float(args.height),
                    random_seed=int(args.random_seed),
                    movement_direction=direction,
                    facing_direction=direction,
                ),
            )
            motion = planner.plan(build_initial_planner_context())
            rows.append(
                {
                    "direction": direction_name,
                    "movement_direction": list(direction),
                    "target_vel": float(target_vel),
                    **summarize_motion(motion.root_positions),
                }
            )
    return rows


def summarize_motion(root_positions: Sequence[Sequence[float]]) -> dict[str, Any]:
    if not root_positions:
        raise ValueError("root_positions must not be empty")
    duration_s = max((len(root_positions) - 1) / 50.0, 1.0e-9)
    root_delta_xyz = [
        float(root_positions[-1][axis]) - float(root_positions[0][axis])
        for axis in range(3)
    ]
    root_z = [float(row[2]) for row in root_positions]
    return {
        "timesteps_50hz": len(root_positions),
        "duration_s": duration_s,
        "root_start_xyz": [float(value) for value in root_positions[0][:3]],
        "root_final_xyz": [float(value) for value in root_positions[-1][:3]],
        "root_delta_xyz": root_delta_xyz,
        "root_delta_xy_per_s": [
            root_delta_xyz[0] / duration_s,
            root_delta_xyz[1] / duration_s,
        ],
        "root_z_min": min(root_z),
        "root_z_mean": mean(root_z),
        "root_z_final": root_z[-1],
    }


def parse_csv_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one float")
    if not all(math.isfinite(item) for item in values):
        raise ValueError("floats must be finite")
    return values


def parse_csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one value")
    return values


def direction_by_name(name: str) -> tuple[float, float, float]:
    try:
        return DIRECTIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown direction {name!r}; expected one of {sorted(DIRECTIONS)}") from exc


def format_float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "neg").replace(".", "p")


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(float(value) for value in values) / len(values)


if __name__ == "__main__":
    main()
