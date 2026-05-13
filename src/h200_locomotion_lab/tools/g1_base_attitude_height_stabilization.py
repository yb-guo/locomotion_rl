"""Deterministic G1 base attitude/height stabilization probe for task023."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
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
DEFAULT_TARGET_HEIGHT = 0.78
DEFAULT_MIN_UPRIGHT = 0.30
DEFAULT_TERMINATION_HEIGHT_MIN = 0.20
DEFAULT_TERMINATION_HEIGHT_MAX = 1.20
DEFAULT_MAX_GAIN = 10.0
DEFAULT_MAX_JOINT_DELTA = 0.08
TOP_JOINT_COUNT = 6
ANKLE_ROLL_JOINTS = ("left_ankle_roll_joint", "right_ankle_roll_joint")
ANKLE_PITCH_JOINTS = ("left_ankle_pitch_joint", "right_ankle_pitch_joint")
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
    parser.add_argument("--mode", choices=STABILIZER_MODES, default="none")
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
    return {
        "task": TASK_NAME,
        "runner": args.runner,
        "run_dir": str(run_dir),
        "mode": args.mode,
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
    # The actual Genesis path is intentionally delayed to subtask 002.
    from h200_locomotion_lab.envs import vectorized_genesis_backend as _genesis_backend  # noqa: F401

    raise NotImplementedError(
        "Genesis runner entry is reserved for task023 subtask 002; "
        f"use guarded H200 command: {build_h200_genesis_command(args)}"
    )


def build_h200_genesis_command(args: argparse.Namespace) -> str:
    project = "h200-locomotion-lab-task023-base-attitude-height-stabilization"
    command = [
        "python",
        "-m",
        "h200_locomotion_lab.tools.g1_base_attitude_height_stabilization",
        "--runner",
        "genesis",
        "--mode",
        args.mode,
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--asset-variant-label",
        args.asset_variant_label,
        "--output-root",
        str(args.output_root),
        "--run-name",
        args.run_name or time.strftime("%Y%m%d-%H%M%S"),
        "--physical-gpu",
        str(args.physical_gpu),
        "--logical-cuda-device",
        args.logical_cuda_device,
    ]
    if args.asset_path is not None:
        command.extend(["--asset-path", str(args.asset_path)])
    if args.asset_source_path is not None:
        command.extend(["--asset-source-path", str(args.asset_source_path)])
    inner = "cd /root/agent_workspace/project/" + project + " && " + " ".join(
        shlex.quote(part) for part in command
    )
    return "/root/agent_workspace/safe_agent/run_guarded.sh bash -lc " + shlex.quote(inner)


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
    values = [float(row[key]) for row in rows]
    if not values:
        return {"max_force": 0.0, "mean_force": 0.0, "active_steps": 0, "samples": []}
    return {
        "max_force": max(values),
        "mean_force": sum(values) / len(values),
        "active_steps": sum(1 for value in values if value > 0.0),
        "samples": sample_timeline(rows, key),
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
                float(row["ankle_roll_contact_force"]),
                float(row["ankle_pitch_contact_force"]),
            )
            for row in rows
        ),
        default=0.0,
    )


def asset_instability_factor(label: str) -> float:
    normalized = label.strip().lower()
    if normalized == "ankle_roll_larger_spheres":
        return 0.88
    if normalized == "ankle_roll_box_support":
        return 0.80
    return 1.0


def effective_asset_path(asset_path: Path | None) -> str:
    if asset_path is not None:
        return str(asset_path)
    profile = load_g1_27dof_nohand_profile()
    return str(profile.asset.path)


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
