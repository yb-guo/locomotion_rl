"""Replay official SONIC motion targets through mjlab for plant-response traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from h200_locomotion_lab.envs.mjlab_backend import MjlabG1RobotBackend
from h200_locomotion_lab.envs.robot_backend import G1MotorCommand
from h200_locomotion_lab.runtime.scalar_g1_runtime import ScalarG1Runtime
from h200_locomotion_lab.sonic.g1_observation import SONIC_ACTION_DIM
from h200_locomotion_lab.sonic.g1_planner_encoder import SonicPlannerMotion50Hz
from h200_locomotion_lab.sonic.g1_policy_bridge import (
    SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX,
    get_default_sonic_g1_action_bridge,
)
from h200_locomotion_lab.tools.mjlab_sonic_alignment_trace import (
    TraceProbeState,
    build_action_bridge,
    build_mjlab_env,
    close_env,
    l2_norm,
    percentile,
    quat_to_rpy,
    summarize_alignment_trace,
    top_joint_abs_max,
    top_joint_abs_mean,
    top_joint_error_rms,
    trace_row,
    RecordingDecoder,
    RecordingEncoder,
)


@dataclass(frozen=True)
class OfficialMotionCsv:
    rows: tuple[tuple[float, ...], ...]
    segment_lengths: tuple[int, ...]


@dataclass(frozen=True)
class ReplayStep:
    step_index: int
    raw_action_isaaclab: tuple[float, ...]
    command: G1MotorCommand
    next_state: Any


class FixedMotionReference:
    def __init__(self, motion: SonicPlannerMotion50Hz) -> None:
        self.motion = motion
        self.motion_start_step = 0
        self.planner_calls = 1


class FixedMotionPlanner:
    def __init__(self, motion: SonicPlannerMotion50Hz) -> None:
        self.motion = motion
        self.contexts: list[Sequence[Sequence[float]] | None] = []

    def plan(self, context_qpos: Sequence[Sequence[float]] | None) -> SonicPlannerMotion50Hz:
        self.contexts.append(context_qpos)
        return self.motion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="Unitree-G1-Flat")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trace-name", default="official_motion_replay")
    parser.add_argument("--motion-csv", required=True)
    parser.add_argument("--planner-motion-csv")
    parser.add_argument("--official-log-dir")
    parser.add_argument(
        "--replay-mode",
        choices=("target-command", "sonic-decoder"),
        default="target-command",
    )
    parser.add_argument("--encoder")
    parser.add_argument("--decoder")
    parser.add_argument("--sonic-action-scale-mult", type=float, default=1.0)
    parser.add_argument("--fixed-base-reset", action="store_true")
    parser.add_argument("--disable-startup-randomization", action="store_true")
    parser.add_argument("--sonic-default-reset", action="store_true")
    parser.add_argument("--sonic-hip-pitch-actuator", action="store_true")
    parser.add_argument("--disable-terminations", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command_motion_csv = read_official_motion_qpos_csv(Path(args.motion_csv))
    command_rows = command_motion_csv.rows[: args.steps]
    if len(command_rows) < args.steps:
        raise ValueError(
            f"--motion-csv has only {len(command_rows)} usable rows, "
            f"cannot replay {args.steps} steps"
        )
    planner_motion_csv = (
        read_official_motion_qpos_csv(Path(args.planner_motion_csv))
        if args.planner_motion_csv
        else None
    )
    reference_motion = motion_from_official_qpos_rows(command_rows)
    if args.replay_mode == "sonic-decoder" and (not args.encoder or not args.decoder):
        raise ValueError("--encoder and --decoder are required for --replay-mode sonic-decoder")

    raw_env = build_mjlab_env(args)
    action_bridge = build_action_bridge(args.sonic_action_scale_mult)
    bridge = action_bridge or get_default_sonic_g1_action_bridge()
    backend = MjlabG1RobotBackend(raw_env)
    probe = TraceProbeState()

    rows: list[dict[str, Any]]
    done_steps: list[int]
    try:
        if args.replay_mode == "target-command":
            rows, done_steps = run_target_command_replay(
                backend,
                command_rows,
                reference_motion,
                bridge=bridge,
                action_bridge=action_bridge,
                probe=probe,
            )
        else:
            rows, done_steps = run_sonic_decoder_replay(
                backend,
                command_rows,
                reference_motion,
                encoder_path=Path(args.encoder),
                decoder_path=Path(args.decoder),
                action_bridge=action_bridge,
                probe=probe,
            )
    finally:
        close_env(backend.raw_env)

    summary = summarize_alignment_trace(
        rows,
        done_steps=done_steps,
        joint_names=backend.sonic_joint_order,
    )
    summary["options"] = {
        "fixed_base_reset": bool(args.fixed_base_reset),
        "disable_startup_randomization": bool(args.disable_startup_randomization),
        "sonic_default_reset": bool(args.sonic_default_reset),
        "sonic_hip_pitch_actuator": bool(args.sonic_hip_pitch_actuator),
        "sonic_action_scale_mult": float(args.sonic_action_scale_mult),
        "seed": args.seed,
        "motion_csv": str(Path(args.motion_csv)),
        "planner_motion_csv": str(Path(args.planner_motion_csv))
        if args.planner_motion_csv
        else None,
        "official_log_dir": str(Path(args.official_log_dir))
        if args.official_log_dir
        else None,
        "replay_mode": args.replay_mode,
        "encoder": str(Path(args.encoder)) if args.encoder else None,
        "decoder": str(Path(args.decoder)) if args.decoder else None,
    }
    summary["motion_csv"] = summarize_motion_csv(
        command_motion_csv,
        replay_rows=len(command_rows),
    )
    if planner_motion_csv is not None:
        summary["planner_motion_csv"] = summarize_motion_csv(
            planner_motion_csv,
            replay_rows=None,
        )
    official_response = (
        summarize_official_deploy_response(
            Path(args.official_log_dir),
            official_response_targets(
                Path(args.official_log_dir),
                command_rows,
                replay_mode=args.replay_mode,
                bridge=bridge,
                steps=args.steps,
            ),
            backend.sonic_joint_order,
            steps=args.steps,
        )
        if args.official_log_dir
        else None
    )
    if official_response is not None:
        summary["official_deploy_response"] = official_response
        summary["official_vs_mjlab_response"] = compare_official_and_mjlab(
            official_response,
            summary,
        )

    payload = {"summary": summary, "rows": rows}
    trace_path = output_dir / f"{args.trace_name}.json"
    trace_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"trace": str(trace_path.resolve()), **summary}, indent=2, sort_keys=True))


def run_target_command_replay(
    backend: MjlabG1RobotBackend,
    command_rows: Sequence[Sequence[float]],
    reference_motion: SonicPlannerMotion50Hz,
    *,
    bridge: Any,
    action_bridge: Any,
    probe: TraceProbeState,
) -> tuple[list[dict[str, Any]], list[int]]:
    provider = FixedMotionReference(reference_motion)
    rows: list[dict[str, Any]] = []
    done_steps: list[int] = []
    backend.reset()
    for step_index, qpos in enumerate(command_rows):
        target = tuple(qpos[7:])
        raw_action = bridge.command_targets_to_policy_action(target)
        command = G1MotorCommand(
            raw_action_isaaclab=raw_action,
            motor_position_targets_mujoco=target,
        )
        backend.write_command(command)
        next_state = backend.advance()
        if backend.last_step_result and backend.last_step_result.done:
            done_steps.append(step_index)
        probe.raw_action_norm = l2_norm(raw_action)
        step = ReplayStep(
            step_index=step_index,
            raw_action_isaaclab=raw_action,
            command=command,
            next_state=next_state,
        )
        row = trace_row(step, backend, provider, probe, action_bridge)
        append_command_motion_fields(row, qpos)
        rows.append(row)
    return rows, done_steps


def run_sonic_decoder_replay(
    backend: MjlabG1RobotBackend,
    command_rows: Sequence[Sequence[float]],
    reference_motion: SonicPlannerMotion50Hz,
    *,
    encoder_path: Path,
    decoder_path: Path,
    action_bridge: Any,
    probe: TraceProbeState,
) -> tuple[list[dict[str, Any]], list[int]]:
    from h200_locomotion_lab.sonic.controller import SonicPlannerEncoderActionProvider
    from h200_locomotion_lab.sonic.onnx_models import SonicOnnxDecoder, SonicOnnxEncoder

    provider = SonicPlannerEncoderActionProvider(
        planner=FixedMotionPlanner(reference_motion),
        encoder=RecordingEncoder(SonicOnnxEncoder(encoder_path), probe),
        decoder=RecordingDecoder(SonicOnnxDecoder(decoder_path), probe),
        replan_interval=0,
        planner_context_source="motion",
    )
    runtime = ScalarG1Runtime(
        backend,
        provider,
        action_bridge=action_bridge,
    )
    rows: list[dict[str, Any]] = []
    done_steps: list[int] = []
    runtime.reset()
    for step_index, qpos in enumerate(command_rows):
        step = runtime.step()
        if backend.last_step_result and backend.last_step_result.done:
            done_steps.append(step.step_index)
        row = trace_row(step, backend, provider, probe, action_bridge)
        append_command_motion_fields(row, qpos)
        rows.append(row)
    return rows, done_steps


def append_command_motion_fields(row: dict[str, Any], qpos: Sequence[float]) -> None:
    row["command_motion_root_xyz"] = list(qpos[:3])
    row["command_motion_root_quat"] = list(qpos[3:7])
    row["command_motion_pitch"] = quat_to_rpy(qpos[3:7])[1]
    row["command_motion_joint_positions"] = list(qpos[7:])


def read_official_motion_qpos_csv(path: Path) -> OfficialMotionCsv:
    """Read official no-header qpos CSVs with trailing commas and blank segments."""

    rows: list[tuple[float, ...]] = []
    segment_lengths: list[int] = []
    current_segment_len = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, raw_row in enumerate(csv.reader(handle), start=1):
            cells = [cell.strip() for cell in raw_row if cell.strip()]
            if not cells:
                if current_segment_len:
                    segment_lengths.append(current_segment_len)
                    current_segment_len = 0
                continue
            values = tuple(float(cell) for cell in cells)
            if len(values) != 7 + SONIC_ACTION_DIM:
                raise ValueError(
                    f"{path}:{line_number} expected 36 qpos values, got {len(values)}"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}:{line_number} contains non-finite values")
            rows.append(values)
            current_segment_len += 1
    if current_segment_len:
        segment_lengths.append(current_segment_len)
    if not rows:
        raise ValueError(f"{path} contains no qpos rows")
    return OfficialMotionCsv(rows=tuple(rows), segment_lengths=tuple(segment_lengths))


def motion_from_official_qpos_rows(
    rows: Sequence[Sequence[float]],
    *,
    rate_hz: float = 50.0,
) -> SonicPlannerMotion50Hz:
    qpos_rows = tuple(
        _coerce_qpos(row, f"rows[{index}]")
        for index, row in enumerate(rows)
    )
    if len(qpos_rows) < 2:
        raise ValueError("at least two qpos rows are required")
    joint_positions_policy_order = tuple(
        tuple(row[7 + mujoco_index] for mujoco_index in SONIC_G1_POLICY_INDEX_TO_MUJOCO_INDEX)
        for row in qpos_rows
    )
    joint_velocities_policy_order = [
        tuple(
            (
                joint_positions_policy_order[index + 1][joint]
                - joint_positions_policy_order[index][joint]
            )
            * rate_hz
            for joint in range(SONIC_ACTION_DIM)
        )
        for index in range(len(qpos_rows) - 1)
    ]
    joint_velocities_policy_order.append(joint_velocities_policy_order[-1])
    return SonicPlannerMotion50Hz(
        root_positions=tuple(tuple(row[:3]) for row in qpos_rows),  # type: ignore[arg-type]
        root_quats=tuple(tuple(row[3:7]) for row in qpos_rows),  # type: ignore[arg-type]
        joint_positions_policy_order=joint_positions_policy_order,
        joint_velocities_policy_order=tuple(joint_velocities_policy_order),
    )


def summarize_motion_csv(
    motion: OfficialMotionCsv,
    *,
    replay_rows: int | None,
) -> dict[str, Any]:
    rows = motion.rows if replay_rows is None else motion.rows[:replay_rows]
    root_z = [row[2] for row in rows]
    root_pitch = [quat_to_rpy(row[3:7])[1] for row in rows]
    root_delta = [rows[-1][axis] - rows[0][axis] for axis in range(3)]
    duration_s = max(len(rows), 1) * 0.02
    return {
        "rows": len(motion.rows),
        "replay_rows": replay_rows,
        "segment_count": len(motion.segment_lengths),
        "segment_lengths_first10": list(motion.segment_lengths[:10]),
        "root_z_start": root_z[0],
        "root_z_final": root_z[-1],
        "root_z_min": min(root_z),
        "root_z_mean": sum(root_z) / len(root_z),
        "root_delta_xyz": root_delta,
        "root_delta_xy_per_s": [
            root_delta[0] / duration_s,
            root_delta[1] / duration_s,
        ],
        "abs_pitch_p95": percentile([abs(value) for value in root_pitch], 95.0),
        "abs_pitch_max": max(abs(value) for value in root_pitch),
    }


def summarize_official_deploy_response(
    log_dir: Path,
    target_rows: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    steps: int,
) -> dict[str, Any]:
    q_rows = read_official_log_matrix(log_dir / "q.csv", "q", steps)
    torque_rows = read_official_log_matrix(log_dir / "motor_torque.csv", "tau", steps)
    pitch = read_official_base_pitch(log_dir / "base_quat.csv", steps)
    targets = [tuple(row) for row in target_rows[:steps]]
    joint_error_rows = [
        [q - target for q, target in zip(q_row, target_row, strict=True)]
        for q_row, target_row in zip(q_rows, targets, strict=True)
    ]
    return {
        "steps": len(q_rows),
        "abs_pitch_p95": percentile([abs(value) for value in pitch], 95.0),
        "abs_pitch_max": max(abs(value) for value in pitch),
        "joint_error_rms_mean": sum(rms(row) for row in joint_error_rows)
        / len(joint_error_rows),
        "joint_error_rms_max": max(rms(row) for row in joint_error_rows),
        "top_joint_error_rms": top_joint_error_rms(
            [{"joint_error": row} for row in joint_error_rows],
            joint_names,
        ),
        "top_joint_torque_abs_mean": top_joint_abs_mean(torque_rows, joint_names),
        "top_joint_torque_abs_max": top_joint_abs_max(torque_rows, joint_names),
    }


def official_response_targets(
    log_dir: Path,
    command_rows: Sequence[Sequence[float]],
    *,
    replay_mode: str,
    bridge: Any,
    steps: int,
) -> tuple[tuple[float, ...], ...]:
    if replay_mode == "target-command":
        return tuple(tuple(row[7:]) for row in command_rows[:steps])
    if replay_mode == "sonic-decoder":
        raw_actions = read_official_log_matrix(log_dir / "action.csv", "act", steps)
        return tuple(
            bridge.policy_action_to_command_targets(raw_action)
            for raw_action in raw_actions
        )
    raise ValueError(f"unknown replay_mode: {replay_mode}")


def compare_official_and_mjlab(
    official: dict[str, Any],
    mjlab: dict[str, Any],
) -> dict[str, Any]:
    return {
        "abs_pitch_p95_delta_mjlab_minus_official": (
            float(mjlab["abs_pitch_p95"]) - float(official["abs_pitch_p95"])
        ),
        "joint_error_rms_mean_delta_mjlab_minus_official": (
            float(mjlab["joint_error_rms_mean"])
            - float(official["joint_error_rms_mean"])
        ),
        "official_top_joint_error_rms": official["top_joint_error_rms"][:5],
        "mjlab_top_joint_error_rms": mjlab["top_joint_error_rms"][:5],
        "official_top_torque_abs_max": official["top_joint_torque_abs_max"][:5],
        "mjlab_top_actuator_force_abs_max": mjlab.get(
            "top_joint_actuator_force_abs_max",
            [],
        )[:5],
        "mjlab_top_force_saturation_fraction": mjlab.get(
            "top_joint_force_saturation_fraction",
            [],
        )[:5],
        "mjlab_foot_contact_force_norm_mean": mjlab.get(
            "foot_contact_force_norm_mean"
        ),
        "mjlab_foot_contact_force_norm_max": mjlab.get(
            "foot_contact_force_norm_max"
        ),
    }


def read_official_log_matrix(path: Path, prefix: str, limit: int) -> list[list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[list[float]] = []
        for row in reader:
            rows.append([float(row[f"{prefix}_{index}"]) for index in range(SONIC_ACTION_DIM)])
            if len(rows) >= limit:
                break
    if len(rows) < limit:
        raise ValueError(f"{path} has only {len(rows)} rows, expected {limit}")
    return rows


def read_official_base_pitch(path: Path, limit: int) -> list[float]:
    pitches: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            quat = (
                float(row["base_qw"]),
                float(row["base_qx"]),
                float(row["base_qy"]),
                float(row["base_qz"]),
            )
            pitches.append(quat_to_rpy(quat)[1])
            if len(pitches) >= limit:
                break
    if len(pitches) < limit:
        raise ValueError(f"{path} has only {len(pitches)} rows, expected {limit}")
    return pitches


def rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values) / len(values))


def _coerce_qpos(row: Sequence[float], name: str) -> tuple[float, ...]:
    if len(row) != 7 + SONIC_ACTION_DIM:
        raise ValueError(f"{name} expected dim=36, got {len(row)}")
    values = tuple(float(value) for value in row)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} contains non-finite values")
    return values


if __name__ == "__main__":
    main()
