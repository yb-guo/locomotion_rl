"""Subprocess wrapper for the SONIC C++ ONNX Runtime planner runner."""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from h200_locomotion_lab.sonic.g1_planner_encoder import (
    SonicPlannerMotion50Hz,
    resample_planner_mujoco_qpos_to_50hz,
)


@dataclass(frozen=True, slots=True)
class SonicPlannerCommand:
    mode: int = 2
    target_vel: float = -1.0
    height: float = -1.0
    random_seed: int = 1234
    movement_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    facing_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)


class SubprocessSonicPlanner:
    """Run `sonic_planner_ort_runner` and return resampled 50 Hz motion."""

    def __init__(
        self,
        *,
        planner: Path | str,
        planner_runner: Path | str,
        work_dir: Path | str,
        command: SonicPlannerCommand | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        self.planner = Path(planner)
        self.planner_runner = Path(planner_runner)
        self.work_dir = Path(work_dir)
        self.command = command or SonicPlannerCommand()
        self.timeout_s = float(timeout_s)
        self._call_index = 0

    def plan(
        self,
        context_qpos: Sequence[Sequence[float]] | None,
    ) -> SonicPlannerMotion50Hz:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        tag = f"call_{self._call_index:04d}"
        self._call_index += 1
        qpos_csv = self.work_dir / f"planner_{tag}_qpos.csv"
        stdout_log = self.work_dir / f"planner_{tag}.stdout.log"
        stderr_log = self.work_dir / f"planner_{tag}.stderr.log"
        cmd = [
            str(self.planner_runner),
            "--planner",
            str(self.planner),
            "--output-qpos-csv",
            str(qpos_csv),
            "--mode",
            str(self.command.mode),
            "--target-vel",
            str(self.command.target_vel),
            "--height",
            str(self.command.height),
            "--random-seed",
            str(self.command.random_seed),
            "--movement-direction",
            *(str(value) for value in self.command.movement_direction),
            "--facing-direction",
            *(str(value) for value in self.command.facing_direction),
        ]
        if context_qpos is not None:
            context_csv = self.work_dir / f"planner_{tag}_context.csv"
            write_numeric_rows(context_csv, context_qpos)
            cmd.extend(["--context-qpos-csv", str(context_csv)])

        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )
        stdout_log.write_text(completed.stdout)
        stderr_log.write_text(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                f"planner runner failed with code {completed.returncode}; "
                f"stdout={stdout_log} stderr={stderr_log}"
            )
        qpos_rows = read_numeric_rows(qpos_csv)
        return resample_planner_mujoco_qpos_to_50hz(
            qpos_rows,
            num_pred_frames=parse_num_pred_frames(completed.stdout),
        )


def write_numeric_rows(path: Path, rows: Sequence[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for row in rows:
            writer.writerow([f"{float(value):.9g}" for value in row])


def read_numeric_rows(path: Path) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            if row:
                rows.append(tuple(float(value) for value in row))
    if not rows:
        raise ValueError(f"{path} contained no numeric rows")
    return tuple(rows)


def parse_num_pred_frames(stdout: str) -> int | None:
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and parts[0] == "NUM_PRED_FRAMES":
            return int(parts[1])
    return None
