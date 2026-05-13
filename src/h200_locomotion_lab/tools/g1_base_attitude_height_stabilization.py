"""Deterministic G1 base attitude/height stabilization probe for task023."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import random
import shlex
import time
from typing import Any, Sequence

from h200_locomotion_lab.robots import (
    G1_27DOF_NOHAND_ACTUATOR_ORDER,
    load_g1_27dof_nohand_profile,
)


TASK_NAME = "task023-base-attitude-height-stabilization"
DEFAULT_OUTPUT_ROOT = Path("outputs/task023/base_attitude_height_stabilization")
STABILIZER_MODES = ("none", "attitude", "height", "attitude_height")
RUNNERS = ("local_toy", "genesis")
POSE_PROFILES = ("current", "unitree_gym")
DEFAULT_TARGET_HEIGHT = 0.78
DEFAULT_MIN_UPRIGHT = 0.30
DEFAULT_TERMINATION_HEIGHT_MIN = 0.20
DEFAULT_TERMINATION_HEIGHT_MAX = 1.20
DEFAULT_MAX_GAIN = 10.0
DEFAULT_MAX_JOINT_DELTA = 0.08
TOP_JOINT_COUNT = 6
ANKLE_ROLL_JOINTS = ("left_ankle_roll_joint", "right_ankle_roll_joint")
ANKLE_PITCH_JOINTS = ("left_ankle_pitch_joint", "right_ankle_pitch_joint")
ANKLE_ROLL_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")
ANKLE_PITCH_LINKS = ("left_ankle_pitch_link", "right_ankle_pitch_link")
ATTITUDE_JOINTS = (
    "left_hip_roll_joint",
    "left_ankle_roll_joint",
    "right_hip_roll_joint",
    "right_ankle_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
)
HEIGHT_JOINTS = (
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
)
CURRENT_POSE_LEG_VALUES = {
    "left_hip_pitch_joint": -0.06,
    "right_hip_pitch_joint": -0.06,
    "left_knee_joint": 0.12,
    "right_knee_joint": 0.12,
    "left_ankle_pitch_joint": -0.07,
    "right_ankle_pitch_joint": -0.07,
}


@dataclass(frozen=True)
class StabilizerGains:
    attitude_kp: float
    attitude_kd: float
    height_kp: float
    height_kd: float
    max_joint_delta: float


@dataclass(frozen=True)
class ToyState:
    step: int
    root_height: float
    root_height_velocity: float
    roll: float
    pitch: float
    roll_velocity: float
    pitch_velocity: float

    @property
    def tilt(self) -> float:
        return math.sqrt(self.roll * self.roll + self.pitch * self.pitch)

    @property
    def upright(self) -> float:
        return clamp(1.0 - self.tilt, 0.0, 1.0)


@dataclass(frozen=True)
class ControllerOutput:
    roll_delta: float
    pitch_delta: float
    height_delta: float
    clipped: bool

    @property
    def max_abs_delta(self) -> float:
        return max(abs(self.roll_delta), abs(self.pitch_delta), abs(self.height_delta))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result: dict[str, Any] = {"status": "error", "blocker": ""}
    exit_code = 0
    try:
        if args.print_genesis_command:
            result = {
                "status": "completed",
                "blocker": "",
                "runner": "genesis_command",
                "command": build_h200_genesis_command(args),
            }
        else:
            result = run_probe(args)
    except Exception as exc:
        result["blocker"] = f"{exc.__class__.__name__}:{exc}"
        exit_code = 1
    print(json.dumps(result, sort_keys=True), flush=True)
    if exit_code:
        raise SystemExit(exit_code)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=RUNNERS, default="local_toy")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--n-envs", type=positive_int, default=1)
    parser.add_argument("--mode", choices=STABILIZER_MODES, default="none")
    parser.add_argument("--pose-profile", choices=POSE_PROFILES, default="current")
    parser.add_argument("--steps", type=positive_int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--asset-path", type=Path, default=None)
    parser.add_argument("--asset-variant-label", default="source")
    parser.add_argument("--asset-source-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--attitude-kp", type=nonnegative_float, default=1.6)
    parser.add_argument("--attitude-kd", type=nonnegative_float, default=0.45)
    parser.add_argument("--height-kp", type=nonnegative_float, default=1.25)
    parser.add_argument("--height-kd", type=nonnegative_float, default=0.30)
    parser.add_argument("--max-gain", type=positive_float, default=DEFAULT_MAX_GAIN)
    parser.add_argument("--max-joint-delta", type=positive_float, default=DEFAULT_MAX_JOINT_DELTA)
    parser.add_argument("--target-height", type=positive_float, default=DEFAULT_TARGET_HEIGHT)
    parser.add_argument("--min-upright", type=positive_float, default=DEFAULT_MIN_UPRIGHT)
    parser.add_argument(
        "--termination-height-min",
        type=positive_float,
        default=DEFAULT_TERMINATION_HEIGHT_MIN,
    )
    parser.add_argument(
        "--termination-height-max",
        type=positive_float,
        default=DEFAULT_TERMINATION_HEIGHT_MAX,
    )
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--logical-cuda-device", default="cuda:0")
    parser.add_argument("--print-genesis-command", action="store_true")
    return parser.parse_args(argv)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.runner == "genesis":
        return run_genesis_probe(args)
    if args.runner != "local_toy":
        raise ValueError(f"unknown runner: {args.runner}")

    run_dir = resolve_run_dir(args.output_root, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = run_local_toy_probe(args=args, run_dir=run_dir)
    write_json(run_dir / "summary.json", summary)
    if args.summary_json is not None:
        write_json(resolve_output_file(args.summary_json), summary)
    return summary


def run_local_toy_probe(*, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    asset_path = effective_asset_path(args.asset_path)
    requested_gains = StabilizerGains(
        attitude_kp=float(args.attitude_kp),
        attitude_kd=float(args.attitude_kd),
        height_kp=float(args.height_kp),
        height_kd=float(args.height_kd),
        max_joint_delta=float(args.max_joint_delta),
    )
    gains = clip_gains(requested_gains, max_gain=float(args.max_gain))
    config = build_run_config(
        args=args,
        run_dir=run_dir,
        effective_asset_path=asset_path,
        requested_gains=requested_gains,
        effective_gains=gains,
    )
    write_json(run_dir / "config.json", config)
    metrics = run_toy_rollout(args=args, gains=gains)
    metrics_path = run_dir / "metrics.jsonl"
    for row in metrics:
        append_jsonl(metrics_path, row)

    baseline_metrics = (
        metrics
        if args.mode == "none"
        else run_toy_rollout(args=args, gains=gains, override_mode="none")
    )
    summary = summarize_rollout(
        args=args,
        run_dir=run_dir,
        effective_asset_path=asset_path,
        requested_gains=requested_gains,
        effective_gains=gains,
        rows=metrics,
        baseline_rows=baseline_metrics,
    )
    return summary


def run_toy_rollout(
    *,
    args: argparse.Namespace,
    gains: StabilizerGains,
    override_mode: str | None = None,
) -> list[dict[str, Any]]:
    mode = override_mode or args.mode
    state = initial_toy_state(seed=int(args.seed))
    rng = random.Random(int(args.seed) + 1009)
    joint_errors = [0.0 for _ in G1_27DOF_NOHAND_ACTUATOR_ORDER]
    rows: list[dict[str, Any]] = []
    asset_instability = asset_instability_factor(str(args.asset_variant_label))
    reset_seen = False
    for step in range(int(args.steps)):
        control = compute_controller_output(
            mode=mode,
            gains=gains,
            state=state,
            target_height=float(args.target_height),
        )
        joint_errors = update_joint_errors(joint_errors, control)
        ankle_roll_force, ankle_pitch_force = contact_forces(
            state=state,
            control=control,
            asset_instability=asset_instability,
            rng=rng,
        )
        tilt_bad = state.upright < float(args.min_upright)
        height_bad = (
            state.root_height < float(args.termination_height_min)
            or state.root_height > float(args.termination_height_max)
        )
        reset = tilt_bad or height_bad
        rows.append(
            {
                "step": step,
                "mode": mode,
                "root_height": state.root_height,
                "upright": state.upright,
                "tilt": state.tilt,
                "roll": state.roll,
                "pitch": state.pitch,
                "controller": {
                    "roll_delta": control.roll_delta,
                    "pitch_delta": control.pitch_delta,
                    "height_delta": control.height_delta,
                    "max_abs_delta": control.max_abs_delta,
                    "clipped": control.clipped,
                },
                "tilt_bad": tilt_bad,
                "height_bad": height_bad,
                "reset": reset,
                "reset_reason": reset_reason(tilt_bad=tilt_bad, height_bad=height_bad),
                "joint_errors": dict(zip(G1_27DOF_NOHAND_ACTUATOR_ORDER, joint_errors)),
                "ankle_roll_contact_force": ankle_roll_force,
                "ankle_pitch_contact_force": ankle_pitch_force,
            }
        )
        if reset and not reset_seen:
            reset_seen = True
        state = advance_toy_state(
            state=state,
            control=control if not reset_seen else ControllerOutput(0.0, 0.0, 0.0, False),
            asset_instability=asset_instability,
        )
    return rows


def compute_controller_output(
    *,
    mode: str,
    gains: StabilizerGains,
    state: ToyState,
    target_height: float,
) -> ControllerOutput:
    raw_roll = 0.0
    raw_pitch = 0.0
    raw_height = 0.0
    if mode in {"attitude", "attitude_height"}:
        raw_roll = -(gains.attitude_kp * state.roll + gains.attitude_kd * state.roll_velocity)
        raw_pitch = -(gains.attitude_kp * state.pitch + gains.attitude_kd * state.pitch_velocity)
    if mode in {"height", "attitude_height"}:
        raw_height = gains.height_kp * (target_height - state.root_height) - (
            gains.height_kd * state.root_height_velocity
        )
    if mode == "none":
        pass
    elif mode not in STABILIZER_MODES:
        raise ValueError(f"unknown stabilizer mode: {mode}")

    limit = gains.max_joint_delta
    roll_delta = clamp(raw_roll, -limit, limit)
    pitch_delta = clamp(raw_pitch, -limit, limit)
    height_delta = clamp(raw_height, -limit, limit)
    clipped = (
        roll_delta != raw_roll
        or pitch_delta != raw_pitch
        or height_delta != raw_height
    )
    return ControllerOutput(
        roll_delta=roll_delta,
        pitch_delta=pitch_delta,
        height_delta=height_delta,
        clipped=clipped,
    )


def advance_toy_state(
    *,
    state: ToyState,
    control: ControllerOutput,
    asset_instability: float,
) -> ToyState:
    tilt_bias = 0.0060 * asset_instability + 0.000035 * state.step
    pitch_bias = -0.0038 * asset_instability - 0.000020 * state.step
    height_sag = 0.0026 * asset_instability + 0.000006 * state.step
    roll_velocity = (state.roll_velocity * 0.72) + tilt_bias + (0.18 * control.roll_delta)
    pitch_velocity = (state.pitch_velocity * 0.74) + pitch_bias + (0.16 * control.pitch_delta)
    height_velocity = (
        state.root_height_velocity * 0.70
        - height_sag
        + (0.065 * control.height_delta)
        - (0.0016 * state.tilt)
    )
    return ToyState(
        step=state.step + 1,
        root_height=state.root_height + height_velocity,
        root_height_velocity=height_velocity,
        roll=state.roll + roll_velocity,
        pitch=state.pitch + pitch_velocity,
        roll_velocity=roll_velocity,
        pitch_velocity=pitch_velocity,
    )


def update_joint_errors(
    joint_errors: Sequence[float],
    control: ControllerOutput,
) -> list[float]:
    values = [float(value) * 0.90 for value in joint_errors]
    for joint_name in ANKLE_ROLL_JOINTS:
        values[joint_index(joint_name)] += control.roll_delta
    for joint_name in ANKLE_PITCH_JOINTS:
        values[joint_index(joint_name)] += control.pitch_delta + (0.5 * control.height_delta)
    for joint_name in ("left_knee_joint", "right_knee_joint"):
        values[joint_index(joint_name)] += control.height_delta
    for joint_name in ("left_hip_roll_joint", "right_hip_roll_joint"):
        values[joint_index(joint_name)] += 0.5 * control.roll_delta
    for joint_name in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
        values[joint_index(joint_name)] += 0.5 * control.pitch_delta + 0.25 * control.height_delta
    return values


def contact_forces(
    *,
    state: ToyState,
    control: ControllerOutput,
    asset_instability: float,
    rng: random.Random,
) -> tuple[float, float]:
    deterministic_jitter = rng.random() * 0.0001
    ankle_roll = (
        1.4 * asset_instability
        + 9.0 * state.tilt
        + 95.0 * abs(control.roll_delta)
        + deterministic_jitter
    )
    ankle_pitch = (
        1.2 * asset_instability
        + 6.0 * abs(state.pitch)
        + 70.0 * abs(control.pitch_delta)
        + 55.0 * abs(control.height_delta)
        + deterministic_jitter
    )
    return ankle_roll, ankle_pitch


def summarize_rollout(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    effective_asset_path: str,
    requested_gains: StabilizerGains,
    effective_gains: StabilizerGains,
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    pose = pose_profile_values(
        args.pose_profile,
        load_g1_27dof_nohand_profile().control.default_angles_rad,
    )
    first_tilt = first_step(rows, "tilt_bad")
    first_reset = first_step(rows, "reset")
    baseline_first_tilt = first_step(baseline_rows, "tilt_bad")
    baseline_first_reset = first_step(baseline_rows, "reset")
    return {
        "status": "completed",
        "blocker": "",
        "task": TASK_NAME,
        "runner": args.runner,
        "run_dir": str(run_dir),
        "seed": int(args.seed),
        "pose_profile": args.pose_profile,
        "pose_leg_values_rad": leg_value_summary(pose),
        "steps_requested": int(args.steps),
        "steps_completed": len(rows),
        "effective_asset_path": effective_asset_path,
        "asset_metadata": {
            "variant_label": args.asset_variant_label,
            "source_path": None
            if args.asset_source_path is None
            else str(args.asset_source_path),
            "effective_path": effective_asset_path,
        },
        "stabilizer": {
            "mode": args.mode,
            "gains": {
                "requested": gains_to_dict(requested_gains),
                "effective": gains_to_dict(effective_gains),
                "max_gain": float(args.max_gain),
                "gain_clipped": gains_to_dict(requested_gains) != gains_to_dict(effective_gains),
            },
            "clipping": controller_clipping_summary(rows),
        },
        "root_height_timeline_summary": numeric_timeline_summary(rows, "root_height"),
        "upright_timeline_summary": numeric_timeline_summary(rows, "upright"),
        "first_tilt_step": first_tilt,
        "first_reset_step": first_reset,
        "first_reset_reason": first_reset_reason(rows),
        "baseline_first_tilt_step": baseline_first_tilt,
        "baseline_first_reset_step": baseline_first_reset,
        "top_joint_errors": top_joint_errors(rows),
        "contact_trace_summary": {
            "ankle_roll": contact_summary(rows, "ankle_roll_contact_force"),
            "ankle_pitch": contact_summary(rows, "ankle_pitch_contact_force"),
        },
        "improvement_classification": classify_improvement(
            mode=args.mode,
            first_reset=first_reset,
            baseline_first_reset=baseline_first_reset,
            steps=len(rows),
            max_contact_force=max_contact_force(rows),
        ),
    }


def build_run_config(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    effective_asset_path: str,
    requested_gains: StabilizerGains,
    effective_gains: StabilizerGains,
) -> dict[str, Any]:
    pose = pose_profile_values(
        args.pose_profile,
        load_g1_27dof_nohand_profile().control.default_angles_rad,
    )
    return {
        "task": TASK_NAME,
        "runner": args.runner,
        "run_dir": str(run_dir),
        "mode": args.mode,
        "pose_profile": args.pose_profile,
        "pose_leg_values_rad": leg_value_summary(pose),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "effective_asset_path": effective_asset_path,
        "asset_metadata": {
            "variant_label": args.asset_variant_label,
            "source_path": None
            if args.asset_source_path is None
            else str(args.asset_source_path),
            "effective_path": effective_asset_path,
        },
        "target_height": float(args.target_height),
        "min_upright": float(args.min_upright),
        "termination_height_min": float(args.termination_height_min),
        "termination_height_max": float(args.termination_height_max),
        "stabilizer_gains": {
            "requested": gains_to_dict(requested_gains),
            "effective": gains_to_dict(effective_gains),
            "max_gain": float(args.max_gain),
        },
    }


def run_genesis_probe(args: argparse.Namespace) -> dict[str, Any]:
    VectorizedGenesisBackend, VectorizedGenesisConfig, torch = load_genesis_runtime()
    if hasattr(torch, "manual_seed"):
        torch.manual_seed(int(args.seed))
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    base_profile = load_g1_27dof_nohand_profile()
    profile = profile_with_asset_path(base_profile, args.asset_path)
    pose = pose_profile_values(args.pose_profile, profile.control.default_angles_rad)
    asset_path = str(profile.asset.path)
    run_dir = resolve_run_dir(args.output_root, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    requested_gains = StabilizerGains(
        attitude_kp=float(args.attitude_kp),
        attitude_kd=float(args.attitude_kd),
        height_kp=float(args.height_kp),
        height_kd=float(args.height_kd),
        max_joint_delta=float(args.max_joint_delta),
    )
    gains = clip_gains(requested_gains, max_gain=float(args.max_gain))
    config = build_run_config(
        args=args,
        run_dir=run_dir,
        effective_asset_path=asset_path,
        requested_gains=requested_gains,
        effective_gains=gains,
    )
    config["hardware_metadata"] = hardware_metadata(args)
    config["genesis"] = {
        "backend": args.backend,
        "n_envs": int(args.n_envs),
        "profile": profile_metadata(profile),
    }
    write_json(run_dir / "config.json", config)

    backend = VectorizedGenesisBackend(
        VectorizedGenesisConfig(
            n_envs=int(args.n_envs),
            backend=args.backend,
            logical_cuda_device=args.logical_cuda_device,
            default_positions_rad=pose,
        ),
        profile=profile,
    )
    rows = run_genesis_rollout(
        args=args,
        backend=backend,
        gains=gains,
        mode=args.mode,
    )
    metrics_path = run_dir / "metrics.jsonl"
    for row in rows:
        append_jsonl(metrics_path, row)

    baseline_rows = rows
    if args.mode != "none":
        baseline_backend = VectorizedGenesisBackend(
            VectorizedGenesisConfig(
                n_envs=int(args.n_envs),
                backend=args.backend,
                logical_cuda_device=args.logical_cuda_device,
                default_positions_rad=pose,
            ),
            profile=profile,
        )
        baseline_rows = run_genesis_rollout(
            args=args,
            backend=baseline_backend,
            gains=gains,
            mode="none",
        )
        baseline_metrics_path = run_dir / "baseline_metrics.jsonl"
        for row in baseline_rows:
            append_jsonl(baseline_metrics_path, row)

    summary = summarize_rollout(
        args=args,
        run_dir=run_dir,
        effective_asset_path=asset_path,
        requested_gains=requested_gains,
        effective_gains=gains,
        rows=rows,
        baseline_rows=baseline_rows,
    )
    summary["hardware_metadata"] = hardware_metadata(args)
    summary["genesis"] = {
        "backend": args.backend,
        "n_envs": int(args.n_envs),
        "profile": profile_metadata(profile),
        "tensor_device_report": optional_call(backend, "tensor_device_report"),
        "tensor_device_ok": optional_call(backend, "tensor_device_ok"),
        "contact_solver_config_report": optional_call(backend, "contact_solver_config_report"),
    }
    write_json(run_dir / "summary.json", summary)
    if args.summary_json is not None:
        write_json(resolve_output_file(args.summary_json), summary)
    return summary


def load_genesis_runtime() -> tuple[Any, Any, Any]:
    from h200_locomotion_lab.envs.vectorized_genesis_backend import (  # noqa: PLC0415
        VectorizedGenesisBackend,
        VectorizedGenesisConfig,
    )

    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - H200 environment path.
        raise RuntimeError(f"torch import failed for Genesis runner: {exc}") from exc
    return VectorizedGenesisBackend, VectorizedGenesisConfig, torch


def profile_with_asset_path(profile: Any, asset_path: Path | None) -> Any:
    if asset_path is None:
        return profile
    asset_path_value = path_cli(asset_path)
    try:
        return replace(profile, asset=replace(profile.asset, path=asset_path_value))
    except TypeError:
        profile.asset.path = asset_path_value
        return profile


def run_genesis_rollout(
    *,
    args: argparse.Namespace,
    backend: Any,
    gains: StabilizerGains,
    mode: str,
) -> list[dict[str, Any]]:
    backend.reset()
    contact_reader = build_contact_reader(backend.robot)
    rows: list[dict[str, Any]] = []
    for step in range(int(args.steps)):
        pre_state = backend.state()
        action, controller_rows = genesis_action_rows(
            args=args,
            backend=backend,
            state=pre_state,
            gains=gains,
            mode=mode,
        )
        clipped_action = rows_from_tensorlike(backend.step_physics(action))
        post_state = backend.state()
        row = genesis_metric_row(
            args=args,
            backend=backend,
            state=post_state,
            controller_rows=controller_rows,
            clipped_action=clipped_action,
            contact_reader=contact_reader,
            mode=mode,
            step=step,
        )
        rows.append(row)
    return rows


def genesis_action_rows(
    *,
    args: argparse.Namespace,
    backend: Any,
    state: Any,
    gains: StabilizerGains,
    mode: str,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    root_rows = rows_from_tensorlike(state.root_pos)
    quat_rows = rows_from_tensorlike(state.root_quat)
    root_vel_rows = rows_from_tensorlike(getattr(state, "root_vel", []))
    root_ang_vel_rows = rows_from_tensorlike(getattr(state, "root_ang_vel", []))
    action_scales = tuple(float(value) for value in backend.profile.control.action_scales_rad)
    rows: list[list[float]] = []
    controller_rows: list[dict[str, Any]] = []
    for env_index in range(backend.n_envs):
        roll, pitch = roll_pitch_from_quat(quat_rows[env_index])
        root_vel = root_vel_rows[env_index] if root_vel_rows else [0.0, 0.0, 0.0]
        root_ang_vel = root_ang_vel_rows[env_index] if root_ang_vel_rows else [0.0, 0.0, 0.0]
        control = compute_controller_output(
            mode=mode,
            gains=gains,
            state=ToyState(
                step=0,
                root_height=float(root_rows[env_index][2]),
                root_height_velocity=float(root_vel[2]) if len(root_vel) > 2 else 0.0,
                roll=roll,
                pitch=pitch,
                roll_velocity=float(root_ang_vel[0]) if root_ang_vel else 0.0,
                pitch_velocity=float(root_ang_vel[1]) if len(root_ang_vel) > 1 else 0.0,
            ),
            target_height=float(args.target_height),
        )
        joint_deltas = genesis_joint_deltas(control)
        action_row = []
        for joint_name, scale in zip(backend.profile.actuator_order, action_scales):
            delta = joint_deltas.get(joint_name, 0.0)
            normalized = 0.0 if scale == 0.0 else delta / scale
            action_row.append(clamp(normalized, -1.0, 1.0))
        clipped_by_normalization = any(abs(value) >= 1.0 for value in action_row)
        rows.append(action_row)
        controller_rows.append(
            {
                "roll_delta": control.roll_delta,
                "pitch_delta": control.pitch_delta,
                "height_delta": control.height_delta,
                "max_abs_delta": control.max_abs_delta,
                "clipped": control.clipped or clipped_by_normalization,
                "max_abs_normalized_action": max((abs(value) for value in action_row), default=0.0),
            }
        )
    return rows, controller_rows


def genesis_joint_deltas(control: ControllerOutput) -> dict[str, float]:
    values = {joint_name: 0.0 for joint_name in G1_27DOF_NOHAND_ACTUATOR_ORDER}
    for joint_name in ANKLE_ROLL_JOINTS:
        values[joint_name] += control.roll_delta
    for joint_name in ANKLE_PITCH_JOINTS:
        values[joint_name] += control.pitch_delta + (0.5 * control.height_delta)
    for joint_name in ("left_knee_joint", "right_knee_joint"):
        values[joint_name] += control.height_delta
    for joint_name in ("left_hip_roll_joint", "right_hip_roll_joint"):
        values[joint_name] += 0.5 * control.roll_delta
    for joint_name in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
        values[joint_name] += 0.5 * control.pitch_delta + 0.25 * control.height_delta
    return values


def genesis_metric_row(
    *,
    args: argparse.Namespace,
    backend: Any,
    state: Any,
    controller_rows: Sequence[dict[str, Any]],
    clipped_action: Sequence[Sequence[float]],
    contact_reader: dict[str, Any],
    mode: str,
    step: int,
) -> dict[str, Any]:
    root_rows = rows_from_tensorlike(state.root_pos)
    quat_rows = rows_from_tensorlike(state.root_quat)
    dof_rows = rows_from_tensorlike(state.dof_pos)
    default_positions = tuple(float(value) for value in backend.default_positions_values)
    heights = [float(row[2]) for row in root_rows]
    attitudes = [attitude_from_quat(row) for row in quat_rows]
    tilts = [row["tilt"] for row in attitudes]
    uprights = [row["upright"] for row in attitudes]
    joint_errors = aggregate_joint_errors(dof_rows, default_positions, backend.profile.actuator_order)
    contacts = read_contact_metrics(backend.robot, contact_reader)
    tilt_bad = any(value < float(args.min_upright) for value in uprights)
    height_bad = any(
        value < float(args.termination_height_min)
        or value > float(args.termination_height_max)
        for value in heights
    )
    action_max = max(
        (abs(value) for row in clipped_action for value in row),
        default=0.0,
    )
    controller = aggregate_controller_rows(controller_rows)
    controller["max_abs_normalized_action"] = max(
        float(controller["max_abs_normalized_action"]),
        action_max,
    )
    return {
        "step": step,
        "mode": mode,
        "root_height": min(heights),
        "upright": min(uprights),
        "tilt": max(tilts),
        "roll": max((abs(row["roll"]) for row in attitudes), default=0.0),
        "pitch": max((abs(row["pitch"]) for row in attitudes), default=0.0),
        "controller": controller,
        "tilt_bad": tilt_bad,
        "height_bad": height_bad,
        "reset": tilt_bad or height_bad,
        "reset_reason": reset_reason(tilt_bad=tilt_bad, height_bad=height_bad),
        "joint_errors": joint_errors,
        "ankle_roll_contact_force": contacts["ankle_roll"]["max_force"],
        "ankle_pitch_contact_force": contacts["ankle_pitch"]["max_force"],
        "contact_metrics": contacts,
    }


def build_h200_genesis_command(args: argparse.Namespace) -> str:
    project = "h200-locomotion-lab-task023-base-attitude-height-stabilization"
    command = [
        "python",
        "-m",
        "h200_locomotion_lab.tools.g1_base_attitude_height_stabilization",
        "--runner",
        "genesis",
        "--backend",
        args.backend,
        "--n-envs",
        str(args.n_envs),
        "--mode",
        args.mode,
        "--pose-profile",
        args.pose_profile,
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--asset-variant-label",
        args.asset_variant_label,
        "--output-root",
        path_cli(args.output_root),
        "--run-name",
        args.run_name or time.strftime("%Y%m%d-%H%M%S"),
        "--physical-gpu",
        str(args.physical_gpu),
        "--logical-cuda-device",
        args.logical_cuda_device,
        "--attitude-kp",
        str(args.attitude_kp),
        "--attitude-kd",
        str(args.attitude_kd),
        "--height-kp",
        str(args.height_kp),
        "--height-kd",
        str(args.height_kd),
        "--max-gain",
        str(args.max_gain),
        "--max-joint-delta",
        str(args.max_joint_delta),
        "--target-height",
        str(args.target_height),
        "--min-upright",
        str(args.min_upright),
        "--termination-height-min",
        str(args.termination_height_min),
        "--termination-height-max",
        str(args.termination_height_max),
    ]
    if args.asset_path is not None:
        command.extend(["--asset-path", path_cli(args.asset_path)])
    if args.asset_source_path is not None:
        command.extend(["--asset-source-path", path_cli(args.asset_source_path)])
    command_text = " ".join(shlex.quote(part) for part in command)
    env_prefix = (
        f"CUDA_VISIBLE_DEVICES={shlex.quote(str(args.physical_gpu))} "
        "PYTHONPATH=src "
    )
    inner = "cd /root/agent_workspace/project/" + project + " && " + env_prefix + command_text
    return "/root/agent_workspace/safe_agent/run_guarded.sh bash -lc " + shlex.quote(inner)


def aggregate_controller_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "roll_delta": max((abs(float(row["roll_delta"])) for row in rows), default=0.0),
        "pitch_delta": max((abs(float(row["pitch_delta"])) for row in rows), default=0.0),
        "height_delta": max((abs(float(row["height_delta"])) for row in rows), default=0.0),
        "max_abs_delta": max((float(row["max_abs_delta"]) for row in rows), default=0.0),
        "max_abs_normalized_action": max(
            (float(row.get("max_abs_normalized_action", 0.0)) for row in rows),
            default=0.0,
        ),
        "clipped": any(bool(row["clipped"]) for row in rows),
    }


def aggregate_joint_errors(
    dof_rows: Sequence[Sequence[float]],
    default_positions: Sequence[float],
    joint_names: Sequence[str],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for joint_index, joint_name in enumerate(joint_names):
        values[joint_name] = max(
            (
                abs(float(row[joint_index]) - float(default_positions[joint_index]))
                for row in dof_rows
            ),
            default=0.0,
        )
    return values


def attitude_from_quat(quat: Sequence[float]) -> dict[str, float]:
    normalized = normalize_quat_wxyz(quat)
    roll, pitch = roll_pitch_from_quat(normalized)
    tilt = math.sqrt(roll * roll + pitch * pitch)
    projected_gravity_z = -1.0 + 2.0 * (
        normalized[1] * normalized[1] + normalized[2] * normalized[2]
    )
    return {
        "roll": roll,
        "pitch": pitch,
        "tilt": tilt,
        "upright": clamp(-projected_gravity_z, 0.0, 1.0),
    }


def normalize_quat_wxyz(quat: Sequence[float]) -> tuple[float, float, float, float]:
    if len(quat) < 4:
        return (1.0, 0.0, 0.0, 0.0)
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    norm = math.sqrt((w * w) + (x * x) + (y * y) + (z * z))
    if norm < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / norm, x / norm, y / norm, z / norm)


def roll_pitch_from_quat(quat: Sequence[float]) -> tuple[float, float]:
    if len(quat) < 4:
        return 0.0, 0.0
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    return roll, pitch


def build_contact_reader(robot: Any) -> dict[str, Any]:
    return {
        "ankle_roll": build_contact_group(robot, ANKLE_ROLL_LINKS),
        "ankle_pitch": build_contact_group(robot, ANKLE_PITCH_LINKS),
    }


def build_contact_group(robot: Any, link_names: Sequence[str]) -> dict[str, Any]:
    links = []
    missing = []
    for link_name in link_names:
        link_index = resolve_link_index(robot, link_name)
        if link_index is None:
            missing.append({"link": link_name, "reason": "link_index_unavailable"})
        links.append({"name": link_name, "index": link_index})
    return {"links": links, "missing": missing}


def read_contact_metrics(robot: Any, reader: dict[str, Any]) -> dict[str, Any]:
    return {
        group_name: read_contact_group(robot, group)
        for group_name, group in reader.items()
    }


def read_contact_group(robot: Any, group: dict[str, Any]) -> dict[str, Any]:
    link_forces = []
    missing = list(group["missing"])
    for link in group["links"]:
        link_name = str(link["name"])
        link_index = link["index"]
        if link_index is None:
            link_forces.append({"link": link_name, "index": None, "force": None})
            continue
        force = read_link_contact_force(robot, int(link_index))
        if force is None:
            missing.append({"link": link_name, "reason": "contact_force_unavailable"})
        link_forces.append({"link": link_name, "index": int(link_index), "force": force})
    forces = [
        float(row["force"])
        for row in link_forces
        if row["force"] is not None
    ]
    return {
        "available": bool(forces),
        "missing": missing,
        "links": link_forces,
        "max_force": max(forces) if forces else None,
        "mean_force": (sum(forces) / len(forces)) if forces else None,
        "active_links": sum(1 for value in forces if value > 0.0),
    }


def resolve_link_index(robot: Any, link_name: str) -> int | None:
    if not hasattr(robot, "get_link"):
        return None
    try:
        link = robot.get_link(link_name)
    except Exception:
        return None
    for attr in ("idx_local", "idx", "link_idx", "id"):
        if not hasattr(link, attr):
            continue
        value = getattr(link, attr)
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            value = value[0]
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def read_link_contact_force(robot: Any, link_index: int) -> float | None:
    for method_name in ("get_links_net_contact_force", "get_links_net_contact_forces"):
        if not hasattr(robot, method_name):
            continue
        method = getattr(robot, method_name)
        try:
            selected = method(links_idx_local=(link_index,))
            force = max_vector_norm(selected)
            if force is not None:
                return force
        except TypeError:
            pass
        except Exception:
            pass
        try:
            full = method()
            force = indexed_contact_force(full, link_index)
            if force is not None:
                return force
        except Exception:
            continue
    return None


def indexed_contact_force(value: Any, link_index: int) -> float | None:
    vectors = contact_vectors_for_link(value, link_index)
    if not vectors:
        return None
    norms = [
        math.sqrt(
            (float(row[0]) * float(row[0]))
            + (float(row[1]) * float(row[1]))
            + (float(row[2]) * float(row[2]))
        )
        for row in vectors
    ]
    return max(norms) if norms else None


def contact_vectors_for_link(value: Any, link_index: int) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None:
        return []
    if not isinstance(value, list):
        value = list(value)
    if not value:
        return []
    if all(isinstance(item, (int, float, bool)) for item in value):
        start = link_index * 3
        if len(value) < start + 3:
            return []
        return [[float(item) for item in value[start : start + 3]]]
    if all(
        isinstance(row, list)
        and all(isinstance(item, (int, float, bool)) for item in row)
        for row in value
    ):
        if all(len(row) == 3 for row in value):
            if len(value) <= link_index:
                return []
            return [[float(item) for item in value[link_index]]]
        start = link_index * 3
        return [
            [float(item) for item in row[start : start + 3]]
            for row in value
            if len(row) >= start + 3
        ]
    vectors = []
    for env_row in value:
        if hasattr(env_row, "tolist"):
            env_row = env_row.tolist()
        if not isinstance(env_row, list) or len(env_row) <= link_index:
            continue
        vector = env_row[link_index]
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if isinstance(vector, list) and len(vector) >= 3:
            vectors.append([float(item) for item in vector[:3]])
    return vectors


def max_vector_norm(value: Any) -> float | None:
    flat = flatten_numeric(value)
    if len(flat) < 3:
        return None
    norms = []
    for index in range(0, len(flat) - 2, 3):
        fx, fy, fz = flat[index : index + 3]
        norms.append(math.sqrt(fx * fx + fy * fy + fz * fz))
    return max(norms) if norms else None


def flatten_numeric(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float, bool)):
        return [float(value)]
    flattened: list[float] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            flattened.extend(flatten_numeric(item))
        else:
            flattened.append(float(item))
    return flattened


def rows_from_tensorlike(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float, bool)):
        return [[float(value)]]
    if value is None:
        return []
    if not isinstance(value, list):
        value = list(value)
    if not value:
        return []
    if all(isinstance(item, (int, float, bool)) for item in value):
        return [[float(item) for item in value]]
    rows: list[list[float]] = []
    for row in value:
        if hasattr(row, "tolist"):
            row = row.tolist()
        rows.append([float(item) for item in row])
    return rows


def hardware_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "guarded_command_cuda_visible_devices": str(args.physical_gpu),
    }


def profile_metadata(profile: Any) -> dict[str, Any]:
    asset = getattr(profile, "asset", None)
    return {
        "name": getattr(profile, "name", None),
        "family": getattr(profile, "family", None),
        "route": getattr(profile, "route", None),
        "asset_path": getattr(asset, "path", None),
        "asset_format": getattr(asset, "format", None),
        "asset_genesis_morph": getattr(asset, "genesis_morph", None),
        "asset_usage": getattr(asset, "usage", None),
        "source_path": None
        if getattr(profile, "source_path", None) is None
        else str(profile.source_path),
    }


def optional_call(target: Any, method_name: str) -> Any:
    if not hasattr(target, method_name):
        return None
    try:
        return getattr(target, method_name)()
    except Exception as exc:
        return {"unavailable": True, "reason": f"{exc.__class__.__name__}:{exc}"}


def initial_toy_state(*, seed: int) -> ToyState:
    rng = random.Random(seed)
    return ToyState(
        step=0,
        root_height=DEFAULT_TARGET_HEIGHT,
        root_height_velocity=0.0,
        roll=0.020 + (0.006 * rng.random()),
        pitch=-0.015 - (0.006 * rng.random()),
        roll_velocity=0.0,
        pitch_velocity=0.0,
    )


def clip_gains(gains: StabilizerGains, *, max_gain: float) -> StabilizerGains:
    return StabilizerGains(
        attitude_kp=min(gains.attitude_kp, max_gain),
        attitude_kd=min(gains.attitude_kd, max_gain),
        height_kp=min(gains.height_kp, max_gain),
        height_kd=min(gains.height_kd, max_gain),
        max_joint_delta=gains.max_joint_delta,
    )


def controller_clipping_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    clipped_steps = sum(1 for row in rows if bool(row["controller"]["clipped"]))
    max_abs_delta = max(
        (float(row["controller"]["max_abs_delta"]) for row in rows),
        default=0.0,
    )
    return {
        "clipped_steps": clipped_steps,
        "clipping_ratio": clipped_steps / len(rows) if rows else 0.0,
        "max_abs_delta": max_abs_delta,
    }


def numeric_timeline_summary(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    if not values:
        return {"initial": None, "final": None, "min": None, "max": None, "mean": None, "samples": []}
    return {
        "initial": values[0],
        "final": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "samples": sample_timeline(rows, key),
    }


def sample_timeline(rows: Sequence[dict[str, Any]], key: str, count: int = 6) -> list[dict[str, Any]]:
    if not rows:
        return []
    if len(rows) <= count:
        selected = list(rows)
    else:
        indexes = sorted({round(index * (len(rows) - 1) / (count - 1)) for index in range(count)})
        selected = [rows[index] for index in indexes]
    return [{"step": int(row["step"]), "value": float(row[key])} for row in selected]


def top_joint_errors(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    joint_names = G1_27DOF_NOHAND_ACTUATOR_ORDER
    entries = []
    for joint_name in joint_names:
        values = [float(row["joint_errors"][joint_name]) for row in rows]
        rms = math.sqrt(sum(value * value for value in values) / len(values))
        entries.append(
            {
                "joint": joint_name,
                "rms": rms,
                "max_abs": max(abs(value) for value in values),
                "final_abs": abs(values[-1]),
            }
        )
    return sorted(entries, key=lambda row: row["rms"], reverse=True)[:TOP_JOINT_COUNT]


def contact_summary(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None
    ]
    if not values:
        group_name = "ankle_roll" if "roll" in key else "ankle_pitch"
        missing = []
        if rows and "contact_metrics" in rows[0]:
            missing = rows[0]["contact_metrics"].get(group_name, {}).get("missing", [])
        return {
            "available": False,
            "missing": missing
            or [{"path": key, "reason": "no_contact_force_values"}],
            "max_force": None,
            "mean_force": None,
            "active_steps": 0,
            "samples": sample_optional_timeline(rows, key),
        }
    return {
        "available": True,
        "missing": [],
        "max_force": max(values),
        "mean_force": sum(values) / len(values),
        "active_steps": sum(1 for value in values if value > 0.0),
        "samples": sample_optional_timeline(rows, key),
    }


def first_step(rows: Sequence[dict[str, Any]], key: str) -> int | None:
    for row in rows:
        if bool(row[key]):
            return int(row["step"])
    return None


def first_reset_reason(rows: Sequence[dict[str, Any]]) -> str | None:
    for row in rows:
        if bool(row["reset"]):
            return str(row["reset_reason"])
    return None


def reset_reason(*, tilt_bad: bool, height_bad: bool) -> str | None:
    if tilt_bad and height_bad:
        return "tilt_and_height"
    if tilt_bad:
        return "tilt"
    if height_bad:
        return "height"
    return None


def classify_improvement(
    *,
    mode: str,
    first_reset: int | None,
    baseline_first_reset: int | None,
    steps: int,
    max_contact_force: float,
) -> str:
    if mode == "none":
        return "baseline"
    if baseline_first_reset is None:
        return "baseline_stable"
    if first_reset is None:
        return "physical_stability" if max_contact_force < 30.0 else "stability_with_high_contact"
    if first_reset > baseline_first_reset:
        return "delayed_reset"
    if first_reset == baseline_first_reset:
        return "no_improvement"
    if first_reset >= steps - 1:
        return "delayed_reset"
    return "regressed"


def max_contact_force(rows: Sequence[dict[str, Any]]) -> float:
    return max(
        (
            max(
                float(row["ankle_roll_contact_force"] or 0.0),
                float(row["ankle_pitch_contact_force"] or 0.0),
            )
            for row in rows
        ),
        default=0.0,
    )


def sample_optional_timeline(
    rows: Sequence[dict[str, Any]],
    key: str,
    count: int = 6,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    if len(rows) <= count:
        selected = list(rows)
    else:
        indexes = sorted({round(index * (len(rows) - 1) / (count - 1)) for index in range(count)})
        selected = [rows[index] for index in indexes]
    return [
        {
            "step": int(row["step"]),
            "value": None if row.get(key) is None else float(row[key]),
        }
        for row in selected
    ]


def asset_instability_factor(label: str) -> float:
    normalized = label.strip().lower()
    if normalized == "ankle_roll_larger_spheres":
        return 0.88
    if normalized == "ankle_roll_box_support":
        return 0.80
    return 1.0


def effective_asset_path(asset_path: Path | None) -> str:
    if asset_path is not None:
        return path_cli(asset_path)
    profile = load_g1_27dof_nohand_profile()
    return str(profile.asset.path)


def path_cli(path: Path) -> str:
    return path.as_posix()


def resolve_run_dir(output_root: Path, run_name: str) -> Path:
    root = output_root if output_root.is_absolute() else Path.cwd() / output_root
    name = run_name.strip() or time.strftime("%Y%m%d-%H%M%S")
    return (root / name).resolve()


def resolve_output_file(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def gains_to_dict(gains: StabilizerGains) -> dict[str, float]:
    return {
        "attitude_kp": float(gains.attitude_kp),
        "attitude_kd": float(gains.attitude_kd),
        "height_kp": float(gains.height_kp),
        "height_kd": float(gains.height_kd),
        "max_joint_delta": float(gains.max_joint_delta),
    }


def pose_profile_values(name: str, current_pose: Sequence[float]) -> tuple[float, ...]:
    if name == "current":
        values = task018_tall_crouch_pose_values(current_pose)
    elif name == "unitree_gym":
        values = unitree_gym_pose_values()
    else:
        raise ValueError(f"unknown pose profile: {name}")
    if len(values) != 27:
        raise ValueError("pose profile must have 27 values")
    return values


def task018_tall_crouch_pose_values(current_pose: Sequence[float]) -> tuple[float, ...]:
    values = [float(value) for value in current_pose]
    for joint_name, value in CURRENT_POSE_LEG_VALUES.items():
        values[G1_27DOF_NOHAND_ACTUATOR_ORDER.index(joint_name)] = value
    return tuple(values)


def unitree_gym_pose_values() -> tuple[float, ...]:
    values = dict.fromkeys(G1_27DOF_NOHAND_ACTUATOR_ORDER, 0.0)
    values.update(
        {
            "left_hip_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "right_hip_pitch_joint": -0.1,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "left_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.2,
            "left_elbow_joint": 0.6,
            "right_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_elbow_joint": 0.6,
        }
    )
    return tuple(float(values[joint_name]) for joint_name in G1_27DOF_NOHAND_ACTUATOR_ORDER)


def leg_value_summary(pose: Sequence[float]) -> dict[str, float]:
    return {
        "hip_pitch": float(
            pose[G1_27DOF_NOHAND_ACTUATOR_ORDER.index("left_hip_pitch_joint")]
        ),
        "knee": float(pose[G1_27DOF_NOHAND_ACTUATOR_ORDER.index("left_knee_joint")]),
        "ankle_pitch": float(
            pose[G1_27DOF_NOHAND_ACTUATOR_ORDER.index("left_ankle_pitch_joint")]
        ),
    }


def joint_index(joint_name: str) -> int:
    return G1_27DOF_NOHAND_ACTUATOR_ORDER.index(joint_name)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, sort_keys=True) + "\n")


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


if __name__ == "__main__":
    main()
