"""Trace SONIC/mjlab alignment metrics for diagnosis runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from h200_locomotion_lab.envs.mjlab_backend import MjlabG1RobotBackend
from h200_locomotion_lab.envs.robot_backend import G1MotorCommand
from h200_locomotion_lab.runtime import ScalarActionBridge
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
from h200_locomotion_lab.sonic.g1_policy_bridge import get_default_sonic_g1_action_bridge
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
    parser.add_argument("--sonic-action-scale-mult", type=float, default=1.0)
    parser.add_argument("--mode", type=int, default=2)
    parser.add_argument("--target-vel", type=float, default=-1.0)
    parser.add_argument("--movement-direction", nargs=3, type=float, default=(1.0, 0.0, 0.0))
    parser.add_argument("--facing-direction", nargs=3, type=float, default=(1.0, 0.0, 0.0))
    parser.add_argument("--height", type=float, default=-1.0)
    parser.add_argument("--fixed-base-reset", action="store_true")
    parser.add_argument("--disable-startup-randomization", action="store_true")
    parser.add_argument("--sonic-default-reset", action="store_true")
    parser.add_argument("--sonic-hip-pitch-actuator", action="store_true")
    parser.add_argument("--clamp-targets-to-soft-limits", action="store_true")
    parser.add_argument("--disable-terminations", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_env = build_mjlab_env(args)
    backend: MjlabG1RobotBackend = (
        SoftLimitClampedMjlabG1RobotBackend(raw_env)
        if args.clamp_targets_to_soft_limits
        else MjlabG1RobotBackend(raw_env)
    )
    probe = TraceProbeState()
    provider = build_online_provider(args, probe)
    runtime = ScalarG1Runtime(
        backend,
        provider,
        action_bridge=build_action_bridge(args.sonic_action_scale_mult),
    )

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
    if args.sonic_hip_pitch_actuator:
        set_sonic_hip_pitch_actuator(cfg)
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


def build_action_bridge(scale_mult: float) -> ScalarActionBridge | None:
    if not math.isfinite(scale_mult) or scale_mult <= 0.0:
        raise ValueError("--sonic-action-scale-mult must be finite and positive")
    if abs(scale_mult - 1.0) <= 1.0e-12:
        return None
    bridge = get_default_sonic_g1_action_bridge()
    return ScalarActionBridge(
        action_dim=bridge.action_dim,
        command_to_policy=bridge.command_to_policy,
        default_angles_command=bridge.default_angles_command,
        action_scale_command=tuple(
            float(scale) * float(scale_mult)
            for scale in bridge.action_scale_command
        ),
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


def set_sonic_hip_pitch_actuator(cfg: Any) -> None:
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from src.assets.robots.unitree_g1 import g1_constants as g1

    robot_cfg = cfg.scene.entities["robot"]
    articulation = robot_cfg.articulation
    actuators = []
    for actuator in articulation.actuators:
        names = tuple(
            name
            for name in actuator.target_names_expr
            if name != ".*_hip_pitch_joint"
        )
        if names:
            actuator.target_names_expr = names
            actuators.append(actuator)
    actuators.append(
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hip_pitch_joint",),
            stiffness=g1.STIFFNESS_7520_22,
            damping=g1.DAMPING_7520_22,
            effort_limit=g1.ACTUATOR_7520_22.effort_limit,
            armature=g1.ACTUATOR_7520_22.reflected_inertia,
        )
    )
    articulation.actuators = tuple(actuators)


class SoftLimitClampedMjlabG1RobotBackend(MjlabG1RobotBackend):
    """Trace-only backend that clamps motor position targets to mjlab soft limits."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_unclamped_command: G1MotorCommand | None = None
        self.last_target_clip_delta: tuple[float, ...] = (0.0,) * SONIC_ACTION_DIM

    def reset(self):
        state = super().reset()
        self.last_unclamped_command = None
        self.last_target_clip_delta = (0.0,) * SONIC_ACTION_DIM
        return state

    def write_command(self, command: G1MotorCommand) -> None:
        self.last_unclamped_command = command
        limits = read_sonic_robot_data_matrix(self, "soft_joint_pos_limits", 2)
        if limits is None:
            self.last_target_clip_delta = (0.0,) * SONIC_ACTION_DIM
            super().write_command(command)
            return
        clamped_targets = clamp_joint_targets(
            command.motor_position_targets_mujoco,
            limits,
        )
        self.last_target_clip_delta = tuple(
            clamped - raw
            for clamped, raw in zip(
                clamped_targets,
                command.motor_position_targets_mujoco,
                strict=True,
            )
        )
        super().write_command(
            G1MotorCommand(
                raw_action_isaaclab=command.raw_action_isaaclab,
                motor_position_targets_mujoco=clamped_targets,
            )
        )


def trace_row(
    step: Any,
    backend: MjlabG1RobotBackend,
    provider: SonicPlannerEncoderActionProvider,
    probe: TraceProbeState,
) -> dict[str, Any]:
    actual = step.next_state.motor_positions_mujoco
    raw_target = step.command.motor_position_targets_mujoco
    target = backend._last_command.motor_position_targets_mujoco
    target_clip_delta = getattr(
        backend,
        "last_target_clip_delta",
        (0.0,) * SONIC_ACTION_DIM,
    )
    joint_error = tuple(actual_value - target_value for actual_value, target_value in zip(actual, target))
    mjlab_action = backend.motor_targets_to_mjlab_action(target)
    roll, pitch, yaw = quat_to_rpy(step.next_state.base_quat)
    motion_frame = step.step_index - provider.motion_start_step
    planner_root_xyz = None
    planner_root_vel_xyz = None
    planner_root_z = None
    if provider.motion is not None and 0 <= motion_frame < provider.motion.timesteps:
        planner_root_xyz = provider.motion.root_positions[motion_frame]
        planner_root_vel_xyz = planner_root_velocity(provider.motion.root_positions, motion_frame)
        planner_root_z = planner_root_xyz[2]
    done = backend.last_step_result.done if backend.last_step_result else None
    root_lin_vel_b = read_robot_data_vector(backend, "root_link_lin_vel_b", 3)
    root_ang_vel_b = read_robot_data_vector(backend, "root_link_ang_vel_b", 3)
    mjlab_twist_command = read_mjlab_command(backend.raw_env, "twist", 3)
    actuator_force = read_sonic_robot_data_vector(backend, "actuator_force")
    joint_effort_target = read_sonic_robot_data_vector(backend, "joint_effort_target")
    effort_limits = mjlab_effort_limits_by_sonic_joint(backend)
    force_utilization = (
        tuple(
            abs(force) / limit if limit > 0.0 else 0.0
            for force, limit in zip(actuator_force, effort_limits, strict=True)
        )
        if actuator_force is not None and effort_limits is not None
        else None
    )
    soft_limit_margins = joint_limit_margins(
        actual,
        read_sonic_robot_data_matrix(backend, "soft_joint_pos_limits", 2),
    )
    raw_target_soft_limit_margins = joint_limit_margins(
        raw_target,
        read_sonic_robot_data_matrix(backend, "soft_joint_pos_limits", 2),
    )
    target_soft_limit_margins = joint_limit_margins(
        target,
        read_sonic_robot_data_matrix(backend, "soft_joint_pos_limits", 2),
    )
    foot_contact = read_foot_contact(backend.raw_env)
    return {
        "step": step.step_index,
        "done": done,
        "root_xyz": list(step.next_state.root_qpos[:3]),
        "root_lin_vel_b": list(root_lin_vel_b) if root_lin_vel_b is not None else None,
        "root_ang_vel_b": list(root_ang_vel_b) if root_ang_vel_b is not None else None,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "planner_calls": provider.planner_calls,
        "motion_frame": motion_frame,
        "planner_root_xyz": list(planner_root_xyz) if planner_root_xyz is not None else None,
        "planner_root_vel_xyz": (
            list(planner_root_vel_xyz) if planner_root_vel_xyz is not None else None
        ),
        "planner_root_z": planner_root_z,
        "mjlab_twist_command": (
            list(mjlab_twist_command) if mjlab_twist_command is not None else None
        ),
        "actuator_force": list(actuator_force) if actuator_force is not None else None,
        "joint_effort_target": (
            list(joint_effort_target) if joint_effort_target is not None else None
        ),
        "effort_limits": list(effort_limits) if effort_limits is not None else None,
        "actuator_force_utilization": (
            list(force_utilization) if force_utilization is not None else None
        ),
        "actual_soft_limit_margin": (
            list(soft_limit_margins) if soft_limit_margins is not None else None
        ),
        "target_soft_limit_margin": (
            list(target_soft_limit_margins) if target_soft_limit_margins is not None else None
        ),
        "raw_target_soft_limit_margin": (
            list(raw_target_soft_limit_margins)
            if raw_target_soft_limit_margins is not None
            else None
        ),
        "target_clip_delta": list(target_clip_delta),
        "target_clip_rms": rms(target_clip_delta),
        "target_clip_absmax": max_abs(target_clip_delta),
        "foot_contact_found": (
            list(foot_contact["found"]) if foot_contact is not None else None
        ),
        "foot_contact_force_norm": (
            list(foot_contact["force_norm"]) if foot_contact is not None else None
        ),
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
    duration_s = len(rows) * 0.02
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
    root_delta_xyz = [
        float(rows[-1]["root_xyz"][axis] - rows[0]["root_xyz"][axis])
        for axis in range(3)
    ]
    summary = {
        "steps": len(rows),
        "done_steps": list(done_steps[:20]),
        "root_z_start": root_z[0],
        "root_z_final": root_z[-1],
        "root_z_min": min(root_z),
        "root_z_mean": mean(root_z),
        "root_delta_xyz": root_delta_xyz,
        "root_delta_xy_per_s": [
            root_delta_xyz[0] / duration_s,
            root_delta_xyz[1] / duration_s,
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
    force_rows = rows_with_vector(rows, "actuator_force")
    force_util_rows = rows_with_vector(rows, "actuator_force_utilization")
    actual_margin_rows = rows_with_vector(rows, "actual_soft_limit_margin")
    target_margin_rows = rows_with_vector(rows, "target_soft_limit_margin")
    raw_target_margin_rows = rows_with_vector(rows, "raw_target_soft_limit_margin")
    foot_force_rows = rows_with_vector(rows, "foot_contact_force_norm")
    target_clip_rms = [float(row.get("target_clip_rms", 0.0)) for row in rows]
    target_clip_absmax = [float(row.get("target_clip_absmax", 0.0)) for row in rows]
    summary["target_clip_rms_mean"] = mean(target_clip_rms)
    summary["target_clip_absmax_max"] = max(target_clip_absmax)
    target_clip_rows = rows_with_vector(rows, "target_clip_delta")
    if target_clip_rows:
        summary["top_joint_target_clip_absmax"] = top_joint_abs_max(
            target_clip_rows,
            joint_names,
        )
    if force_rows:
        summary["top_joint_actuator_force_abs_mean"] = top_joint_abs_mean(
            force_rows,
            joint_names,
        )
        summary["top_joint_actuator_force_abs_max"] = top_joint_abs_max(
            force_rows,
            joint_names,
        )
    if force_util_rows:
        summary["top_joint_force_utilization_mean"] = top_joint_abs_mean(
            force_util_rows,
            joint_names,
        )
        summary["top_joint_force_utilization_max"] = top_joint_abs_max(
            force_util_rows,
            joint_names,
        )
        summary["top_joint_force_saturation_fraction"] = top_joint_fraction_above(
            force_util_rows,
            joint_names,
            threshold=0.95,
        )
    if actual_margin_rows:
        summary["top_joint_actual_soft_limit_margin_min"] = top_joint_min_value(
            actual_margin_rows,
            joint_names,
        )
        summary["top_joint_actual_soft_limit_violation_fraction"] = (
            top_joint_fraction_below(
                actual_margin_rows,
                joint_names,
                threshold=0.0,
            )
        )
    if target_margin_rows:
        summary["top_joint_target_soft_limit_margin_min"] = top_joint_min_value(
            target_margin_rows,
            joint_names,
        )
        summary["top_joint_target_soft_limit_violation_fraction"] = (
            top_joint_fraction_below(
                target_margin_rows,
                joint_names,
                threshold=0.0,
            )
        )
    if raw_target_margin_rows:
        summary["top_joint_raw_target_soft_limit_margin_min"] = top_joint_min_value(
            raw_target_margin_rows,
            joint_names,
        )
        summary["top_joint_raw_target_soft_limit_violation_fraction"] = (
            top_joint_fraction_below(
                raw_target_margin_rows,
                joint_names,
                threshold=0.0,
            )
        )
    if foot_force_rows:
        summary["foot_contact_force_norm_mean"] = column_means(foot_force_rows)
        summary["foot_contact_force_norm_max"] = column_maxes(foot_force_rows)
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
    planner_vel_x = vector_component_values(rows, "planner_root_vel_xyz", 0)
    planner_vel_y = vector_component_values(rows, "planner_root_vel_xyz", 1)
    root_lin_vel_x = vector_component_values(rows, "root_lin_vel_b", 0)
    root_lin_vel_y = vector_component_values(rows, "root_lin_vel_b", 1)
    mjlab_command_x = vector_component_values(rows, "mjlab_twist_command", 0)
    mjlab_command_y = vector_component_values(rows, "mjlab_twist_command", 1)
    if planner_vel_x:
        summary["planner_root_vel_x_mean"] = mean(planner_vel_x)
        summary["planner_root_vel_y_mean"] = mean(planner_vel_y)
    if root_lin_vel_x:
        summary["root_lin_vel_b_x_mean"] = mean(root_lin_vel_x)
        summary["root_lin_vel_b_y_mean"] = mean(root_lin_vel_y)
    if mjlab_command_x:
        summary["mjlab_twist_command_x_mean"] = mean(mjlab_command_x)
        summary["mjlab_twist_command_y_mean"] = mean(mjlab_command_y)
    if args is not None:
        summary["options"] = {
            "fixed_base_reset": bool(args.fixed_base_reset),
            "disable_startup_randomization": bool(args.disable_startup_randomization),
            "sonic_default_reset": bool(args.sonic_default_reset),
            "sonic_hip_pitch_actuator": bool(args.sonic_hip_pitch_actuator),
            "clamp_targets_to_soft_limits": bool(args.clamp_targets_to_soft_limits),
            "mode": int(args.mode),
            "target_vel": float(args.target_vel),
            "height": float(args.height),
            "replan_interval": int(args.replan_interval),
            "planner_context_source": args.planner_context_source,
            "sonic_action_scale_mult": float(args.sonic_action_scale_mult),
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


def rows_with_vector(rows: Sequence[dict[str, Any]], field: str) -> list[list[float]]:
    values: list[list[float]] = []
    for row in rows:
        vector = row.get(field)
        if vector is not None:
            values.append([float(value) for value in vector])
    return values


def top_joint_abs_mean(
    rows: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    count: int = 8,
) -> list[dict[str, float | str]]:
    scored = [
        {
            "joint": joint_name,
            "value": mean([abs(float(row[index])) for row in rows]),
        }
        for index, joint_name in enumerate(joint_names)
    ]
    return sorted(scored, key=lambda row: float(row["value"]), reverse=True)[:count]


def top_joint_abs_max(
    rows: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    count: int = 8,
) -> list[dict[str, float | str]]:
    scored = [
        {
            "joint": joint_name,
            "value": max(abs(float(row[index])) for row in rows),
        }
        for index, joint_name in enumerate(joint_names)
    ]
    return sorted(scored, key=lambda row: float(row["value"]), reverse=True)[:count]


def top_joint_fraction_above(
    rows: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    threshold: float,
    count: int = 8,
) -> list[dict[str, float | str]]:
    scored = [
        {
            "joint": joint_name,
            "value": sum(
                1.0 for row in rows if abs(float(row[index])) >= threshold
            )
            / len(rows),
        }
        for index, joint_name in enumerate(joint_names)
    ]
    return sorted(scored, key=lambda row: float(row["value"]), reverse=True)[:count]


def top_joint_fraction_below(
    rows: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    threshold: float,
    count: int = 8,
) -> list[dict[str, float | str]]:
    scored = [
        {
            "joint": joint_name,
            "value": sum(1.0 for row in rows if float(row[index]) < threshold)
            / len(rows),
        }
        for index, joint_name in enumerate(joint_names)
    ]
    return sorted(scored, key=lambda row: float(row["value"]), reverse=True)[:count]


def top_joint_min_value(
    rows: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    count: int = 8,
) -> list[dict[str, float | str]]:
    scored = [
        {
            "joint": joint_name,
            "value": min(float(row[index]) for row in rows),
        }
        for index, joint_name in enumerate(joint_names)
    ]
    return sorted(scored, key=lambda row: float(row["value"]))[:count]


def column_means(rows: Sequence[Sequence[float]]) -> list[float]:
    if not rows:
        raise ValueError("rows must not be empty")
    return [
        mean([float(row[index]) for row in rows])
        for index in range(len(rows[0]))
    ]


def column_maxes(rows: Sequence[Sequence[float]]) -> list[float]:
    if not rows:
        raise ValueError("rows must not be empty")
    return [
        max(float(row[index]) for row in rows)
        for index in range(len(rows[0]))
    ]


def zero_fields(field_values: dict[str, float], *, eps: float = 1.0e-9) -> list[str]:
    return sorted(name for name, value in field_values.items() if abs(float(value)) <= eps)


def vector_component_values(
    rows: Sequence[dict[str, Any]],
    field: str,
    component: int,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        vector = row.get(field)
        if vector is not None:
            values.append(float(vector[component]))
    return values


def planner_root_velocity(
    root_positions: Sequence[Sequence[float]],
    frame: int,
    *,
    rate_hz: float = 50.0,
) -> tuple[float, float, float]:
    if not root_positions:
        raise ValueError("root_positions must not be empty")
    index = max(0, min(int(frame), len(root_positions) - 1))
    if index < len(root_positions) - 1:
        current = root_positions[index]
        other = root_positions[index + 1]
    else:
        current = root_positions[index - 1] if index > 0 else root_positions[index]
        other = root_positions[index]
    return tuple(
        (float(other[axis]) - float(current[axis])) * float(rate_hz)
        for axis in range(3)
    )  # type: ignore[return-value]


def read_robot_data_vector(
    backend: MjlabG1RobotBackend,
    attr: str,
    expected_dim: int,
) -> tuple[float, ...] | None:
    values = getattr(backend.robot.data, attr, None)
    if values is None:
        return None
    flat = flatten_numeric(values)
    if len(flat) < expected_dim:
        return None
    return tuple(flat[:expected_dim])


def read_sonic_robot_data_vector(
    backend: MjlabG1RobotBackend,
    attr: str,
) -> tuple[float, ...] | None:
    values = getattr(backend.robot.data, attr, None)
    if values is None:
        return None
    flat = flatten_numeric(values)
    if len(flat) < len(backend.robot_joint_names):
        return None
    robot_order = tuple(flat[: len(backend.robot_joint_names)])
    return tuple(
        robot_order[index]
        for index in getattr(backend, "_sonic_to_robot_indices")
    )


def read_sonic_robot_data_matrix(
    backend: MjlabG1RobotBackend,
    attr: str,
    cols: int,
) -> tuple[tuple[float, ...], ...] | None:
    values = getattr(backend.robot.data, attr, None)
    if values is None:
        return None
    flat = flatten_numeric(values)
    rows = len(backend.robot_joint_names)
    if len(flat) < rows * cols:
        return None
    robot_order = tuple(
        tuple(flat[row * cols : row * cols + cols])
        for row in range(rows)
    )
    return tuple(
        robot_order[index]
        for index in getattr(backend, "_sonic_to_robot_indices")
    )


def joint_limit_margins(
    values: Sequence[float],
    limits: Sequence[Sequence[float]] | None,
) -> tuple[float, ...] | None:
    if limits is None:
        return None
    return tuple(
        min(float(value) - float(limit[0]), float(limit[1]) - float(value))
        for value, limit in zip(values, limits, strict=True)
    )


def clamp_joint_targets(
    values: Sequence[float],
    limits: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    return tuple(
        min(max(float(value), float(limit[0])), float(limit[1]))
        for value, limit in zip(values, limits, strict=True)
    )


def mjlab_effort_limits_by_sonic_joint(
    backend: MjlabG1RobotBackend,
) -> tuple[float, ...] | None:
    articulation = getattr(getattr(backend.robot, "cfg", None), "articulation", None)
    actuators = getattr(articulation, "actuators", None)
    if not actuators:
        return None
    limits: list[float] = []
    for joint_name in backend.sonic_joint_order:
        limit = None
        for actuator in actuators:
            patterns = getattr(actuator, "target_names_expr", ())
            if any(re.fullmatch(pattern, joint_name) for pattern in patterns):
                limit = getattr(actuator, "effort_limit", None)
        if limit is None:
            return None
        limits.append(float(limit))
    return tuple(limits)


def read_foot_contact(env: Any) -> dict[str, tuple[float, ...]] | None:
    sensors = getattr(getattr(env, "scene", None), "sensors", None)
    if sensors is None:
        return None
    sensor = sensors.get("feet_ground_contact")
    data = getattr(sensor, "data", None)
    if data is None:
        return None
    found_values = getattr(data, "found", None)
    force_values = getattr(data, "force", None)
    if found_values is None or force_values is None:
        return None
    found = flatten_numeric(found_values)
    force = flatten_numeric(force_values)
    if len(found) < 2 or len(force) < 6:
        return None
    force_norm = (
        math.sqrt(sum(value * value for value in force[0:3])),
        math.sqrt(sum(value * value for value in force[3:6])),
    )
    return {"found": tuple(found[:2]), "force_norm": force_norm}


def read_mjlab_command(env: Any, command_name: str, expected_dim: int) -> tuple[float, ...] | None:
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return None
    get_term = getattr(command_manager, "get_term", None)
    if not callable(get_term):
        return None
    try:
        command_term = get_term(command_name)
    except Exception:
        return None
    command = getattr(command_term, "command", None)
    if command is None:
        return None
    flat = flatten_numeric(command)
    if len(flat) < expected_dim:
        return None
    return tuple(flat[:expected_dim])


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


def flatten_numeric(values: Any) -> tuple[float, ...]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "reshape") and hasattr(values, "tolist"):
        try:
            values = values.reshape(-1).tolist()
        except TypeError:
            values = values.tolist()
    elif hasattr(values, "flatten") and hasattr(values, "tolist"):
        values = values.flatten().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        return (float(values),)
    flat: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flat.extend(flatten_numeric(value))
        else:
            flat.append(float(value))
    return tuple(flat)


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
