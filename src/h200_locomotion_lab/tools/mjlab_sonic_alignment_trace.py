"""Trace SONIC/mjlab alignment metrics for diagnosis runs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from h200_locomotion_lab.envs.mjlab_backend import MjlabG1RobotBackend
from h200_locomotion_lab.runtime.scalar_g1_runtime import ScalarG1Runtime
from h200_locomotion_lab.sonic.controller import SonicPlannerEncoderActionProvider
from h200_locomotion_lab.sonic.g1_observation import (
    SONIC_G1_DECODER_OBSERVATION_FIELDS,
    SONIC_ACTION_DIM,
    SONIC_DECODER_OBS_DIM,
    SONIC_TOKEN_DIM,
)
from h200_locomotion_lab.sonic.g1_planner_encoder import (
    SONIC_ENCODER_OBS_DIM,
    SONIC_G1_ENCODER_FIELDS,
    SONIC_PLANNER_DEFAULT_HEIGHT,
)
from h200_locomotion_lab.sonic.planner_runner import (
    SonicPlannerCommand,
    SubprocessSonicPlanner,
)


STARTUP_RANDOMIZATION_EVENTS = ("foot_friction", "encoder_bias", "base_com")


@dataclass
class TraceProbeState:
    encoder_field_norms: dict[str, float] = field(default_factory=dict)
    decoder_field_norms: dict[str, float] = field(default_factory=dict)
    token_norm: float | None = None
    raw_action_norm: float | None = None
    planner_context_root_z: tuple[float, ...] | None = None


class RecordingPlanner:
    def __init__(self, inner: Any, probe: TraceProbeState) -> None:
        self.inner = inner
        self.probe = probe

    def plan(self, context_qpos):
        if context_qpos is not None:
            self.probe.planner_context_root_z = tuple(float(row[2]) for row in context_qpos)
        return self.inner.plan(context_qpos)


class RecordingEncoder:
    def __init__(self, inner: Any, probe: TraceProbeState) -> None:
        self.inner = inner
        self.probe = probe

    def run(self, observation: Sequence[float]) -> Sequence[float]:
        obs = _coerce_vector(observation, SONIC_ENCODER_OBS_DIM, "encoder_observation")
        self.probe.encoder_field_norms = field_norms(obs, SONIC_G1_ENCODER_FIELDS)
        token = _coerce_vector(self.inner.run(obs), SONIC_TOKEN_DIM, "token_state")
        self.probe.token_norm = l2_norm(token)
        return token


class RecordingDecoder:
    def __init__(self, inner: Any, probe: TraceProbeState) -> None:
        self.inner = inner
        self.probe = probe

    def run(self, observation: Sequence[float]) -> Sequence[float]:
        obs = _coerce_vector(observation, SONIC_DECODER_OBS_DIM, "decoder_observation")
        self.probe.decoder_field_norms = field_norms(obs, SONIC_G1_DECODER_OBSERVATION_FIELDS)
        action = _coerce_vector(self.inner.run(obs), SONIC_ACTION_DIM, "raw_action")
        self.probe.raw_action_norm = l2_norm(action)
        return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="Unitree-G1-Flat")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trace-name", default="alignment_trace")
    parser.add_argument("--planner", required=True)
    parser.add_argument("--planner-runner", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--planner-work-dir", required=True)
    parser.add_argument("--replan-interval", type=int, default=10)
    parser.add_argument("--planner-context-source", choices=("live", "motion"), default="live")
    parser.add_argument("--mode", type=int, default=2)
    parser.add_argument("--target-vel", type=float, default=-1.0)
    parser.add_argument("--movement-direction", nargs=3, type=float, default=(1.0, 0.0, 0.0))
    parser.add_argument("--facing-direction", nargs=3, type=float, default=(1.0, 0.0, 0.0))
    parser.add_argument("--height", type=float, default=-1.0)
    parser.add_argument("--fixed-base-reset", action="store_true")
    parser.add_argument("--disable-startup-randomization", action="store_true")
    parser.add_argument("--sonic-default-reset", action="store_true")
    parser.add_argument("--disable-terminations", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_env = build_mjlab_env(args)
    backend = MjlabG1RobotBackend(raw_env)
    probe = TraceProbeState()
    provider = build_online_provider(args, probe)
    runtime = ScalarG1Runtime(backend, provider)

    rows: list[dict[str, Any]] = []
    done_steps: list[int] = []
    try:
        runtime.reset()
        for _ in range(args.steps):
            step = runtime.step()
            if backend.last_step_result and backend.last_step_result.done:
                done_steps.append(step.step_index)
            rows.append(trace_row(step, backend, provider, probe))
    finally:
        close_env(backend.raw_env)

    summary = summarize_alignment_trace(
        rows,
        done_steps=done_steps,
        joint_names=backend.sonic_joint_order,
        args=args,
    )
    payload = {"summary": summary, "rows": rows}
    trace_path = output_dir / f"{args.trace_name}.json"
    trace_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"trace": str(trace_path.resolve()), **summary}, indent=2, sort_keys=True))


def build_mjlab_env(args: argparse.Namespace):
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    cfg = load_env_cfg(args.task_id, play=True)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    cfg.scene.num_envs = 1
    if args.disable_terminations:
        cfg.terminations = {}
    if args.fixed_base_reset:
        set_fixed_base_reset(cfg)
    if args.disable_startup_randomization:
        for event_name in STARTUP_RANDOMIZATION_EVENTS:
            cfg.events.pop(event_name, None)
    if args.sonic_default_reset:
        set_sonic_default_reset(cfg)
    return ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode=None)


def build_online_provider(args: argparse.Namespace, probe: TraceProbeState):
    from h200_locomotion_lab.sonic.onnx_models import SonicOnnxDecoder, SonicOnnxEncoder

    planner = RecordingPlanner(
        SubprocessSonicPlanner(
            planner=Path(args.planner),
            planner_runner=Path(args.planner_runner),
            work_dir=Path(args.planner_work_dir),
            command=SonicPlannerCommand(
                mode=args.mode,
                target_vel=args.target_vel,
                movement_direction=tuple(args.movement_direction),
                facing_direction=tuple(args.facing_direction),
                height=args.height,
            ),
        ),
        probe,
    )
    return SonicPlannerEncoderActionProvider(
        planner=planner,
        encoder=RecordingEncoder(SonicOnnxEncoder(Path(args.encoder)), probe),
        decoder=RecordingDecoder(SonicOnnxDecoder(Path(args.decoder)), probe),
        replan_interval=args.replan_interval,
        planner_context_source=args.planner_context_source,
    )


def set_fixed_base_reset(cfg: Any) -> None:
    reset_base = cfg.events.get("reset_base")
    if reset_base is None:
        return
    reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    reset_base.params["velocity_range"] = {}


def set_sonic_default_reset(cfg: Any) -> None:
    from src.assets.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME

    cfg.scene.entities["robot"].init_state = KNEES_BENT_KEYFRAME
    reset_base = cfg.events.get("reset_base")
    if reset_base is not None:
        reset_base.params["pose_range"]["z"] = (0.0, 0.0)


def trace_row(
    step: Any,
    backend: MjlabG1RobotBackend,
    provider: SonicPlannerEncoderActionProvider,
    probe: TraceProbeState,
) -> dict[str, Any]:
    actual = step.next_state.motor_positions_mujoco
    target = step.command.motor_position_targets_mujoco
    joint_error = tuple(actual_value - target_value for actual_value, target_value in zip(actual, target))
    mjlab_action = backend.motor_targets_to_mjlab_action(target)
    roll, pitch, yaw = quat_to_rpy(step.next_state.base_quat)
    motion_frame = step.step_index - provider.motion_start_step
    planner_root_z = None
    if provider.motion is not None and 0 <= motion_frame < provider.motion.timesteps:
        planner_root_z = provider.motion.root_positions[motion_frame][2]
    done = backend.last_step_result.done if backend.last_step_result else None
    return {
        "step": step.step_index,
        "done": done,
        "root_xyz": list(step.next_state.root_qpos[:3]),
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "planner_calls": provider.planner_calls,
        "motion_frame": motion_frame,
        "planner_root_z": planner_root_z,
        "planner_context_root_z": list(probe.planner_context_root_z)
        if probe.planner_context_root_z is not None
        else None,
        "joint_error_rms": rms(joint_error),
        "joint_error_absmax": max_abs(joint_error),
        "joint_error": list(joint_error),
        "raw_action_min": min(step.raw_action_isaaclab),
        "raw_action_max": max(step.raw_action_isaaclab),
        "raw_action_absmax": max_abs(step.raw_action_isaaclab),
        "raw_action_norm": probe.raw_action_norm,
        "mjlab_action_min": min(mjlab_action),
        "mjlab_action_max": max(mjlab_action),
        "mjlab_action_absmax": max_abs(mjlab_action),
        "target_min": min(target),
        "target_max": max(target),
        "actual_min": min(actual),
        "actual_max": max(actual),
        "token_norm": probe.token_norm,
        "encoder_field_norms": dict(probe.encoder_field_norms),
        "decoder_field_norms": dict(probe.decoder_field_norms),
    }


def summarize_alignment_trace(
    rows: Sequence[dict[str, Any]],
    *,
    done_steps: Sequence[int],
    joint_names: Sequence[str],
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    root_z = [float(row["root_xyz"][2]) for row in rows]
    pitch_abs = [abs(float(row["pitch"])) for row in rows]
    roll_abs = [abs(float(row["roll"])) for row in rows]
    joint_error_rms = [float(row["joint_error_rms"]) for row in rows]
    raw_action_absmax = [float(row["raw_action_absmax"]) for row in rows]
    mjlab_action_absmax = [float(row["mjlab_action_absmax"]) for row in rows]
    planner_root_z = [
        float(row["planner_root_z"])
        for row in rows
        if row.get("planner_root_z") is not None
    ]
    summary = {
        "steps": len(rows),
        "done_steps": list(done_steps[:20]),
        "root_z_start": root_z[0],
        "root_z_final": root_z[-1],
        "root_z_min": min(root_z),
        "root_z_mean": mean(root_z),
        "root_delta_xyz": [
            float(rows[-1]["root_xyz"][axis] - rows[0]["root_xyz"][axis])
            for axis in range(3)
        ],
        "abs_pitch_p95": percentile(pitch_abs, 95.0),
        "abs_pitch_max": max(pitch_abs),
        "abs_roll_p95": percentile(roll_abs, 95.0),
        "abs_roll_max": max(roll_abs),
        "joint_error_rms_mean": mean(joint_error_rms),
        "joint_error_rms_max": max(joint_error_rms),
        "raw_action_absmax_max": max(raw_action_absmax),
        "mjlab_action_absmax_max": max(mjlab_action_absmax),
        "top_joint_error_rms": top_joint_error_rms(rows, joint_names),
        "encoder_zero_fields_last": zero_fields(rows[-1]["encoder_field_norms"]),
    }
    if planner_root_z:
        summary["planner_root_z_min"] = min(planner_root_z)
        summary["planner_root_z_mean"] = mean(planner_root_z)
        summary["root_minus_planner_z_mean"] = mean(
            [
                float(row["root_xyz"][2]) - float(row["planner_root_z"])
                for row in rows
                if row.get("planner_root_z") is not None
            ]
        )
    if args is not None:
        summary["options"] = {
            "fixed_base_reset": bool(args.fixed_base_reset),
            "disable_startup_randomization": bool(args.disable_startup_randomization),
            "sonic_default_reset": bool(args.sonic_default_reset),
            "mode": int(args.mode),
            "target_vel": float(args.target_vel),
            "height": float(args.height),
            "replan_interval": int(args.replan_interval),
            "planner_context_source": args.planner_context_source,
            "seed": args.seed,
        }
    return summary


def field_norms(values: Sequence[float], fields: Sequence[Any]) -> dict[str, float]:
    return {
        field.name: l2_norm(values[field.offset : field.offset + field.dim])
        for field in fields
    }


def top_joint_error_rms(
    rows: Sequence[dict[str, Any]],
    joint_names: Sequence[str],
    *,
    count: int = 8,
) -> list[dict[str, float | str]]:
    accum = [0.0] * len(joint_names)
    for row in rows:
        for index, value in enumerate(row["joint_error"]):
            accum[index] += float(value) * float(value)
    scored = [
        {
            "joint": joint_name,
            "rms": math.sqrt(accum[index] / len(rows)),
        }
        for index, joint_name in enumerate(joint_names)
    ]
    return sorted(scored, key=lambda row: float(row["rms"]), reverse=True)[:count]


def zero_fields(field_values: dict[str, float], *, eps: float = 1.0e-9) -> list[str]:
    return sorted(name for name, value in field_values.items() if abs(float(value)) <= eps)


def quat_to_rpy(quat: Sequence[float]) -> tuple[float, float, float]:
    qw, qx, qy, qz = _coerce_vector(quat, 4, "quat")
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def rms(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return math.sqrt(sum(float(value) * float(value) for value in values) / len(values))


def l2_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def max_abs(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return max(abs(float(value)) for value in values)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(float(value) for value in values) / len(values)


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile_value must be in [0, 100]")
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile_value / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    alpha = position - lower
    return sorted_values[lower] * (1.0 - alpha) + sorted_values[upper] * alpha


def close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _coerce_vector(values: Sequence[float], expected_dim: int, name: str) -> tuple[float, ...]:
    if len(values) != expected_dim:
        raise ValueError(f"{name} expected dim={expected_dim}, got {len(values)}")
    vector = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} contains a non-finite value")
    return vector


if __name__ == "__main__":
    main()
