"""Standalone zero-action G1 standing causality probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from h200_locomotion_lab.envs.g1_velocity_tracking_env import G1VelocityTrackingConfig
from h200_locomotion_lab.envs.vectorized_genesis_backend import (
    VectorizedGenesisBackend,
    VectorizedGenesisConfig,
    as_rows,
    is_tensor_like,
)
from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.robots import (
    G1_27DOF_NOHAND_ACTUATOR_ORDER,
    load_g1_27dof_nohand_profile,
)

PROJECT_PREFIX = Path("/root/agent_workspace/project")
DEFAULT_OUTPUT_ROOT = Path("outputs/task019/zero_action_standing_causality")
CONTROL_MODES = (
    "genesis_position",
    "genesis_position_resend_physics",
    "custom_pd_torque",
)
POSE_PROFILES = ("current", "unitree_gym")
GAIN_PROFILES = (
    "current",
    "global_kv_2x",
    "global_kv_4x",
    "global_kp_0_5x_kv_2x",
    "ankle_kp_2x_kv_2x",
    "knee_ankle_kp_2x_kv_2x",
    "unitree_leg_gains",
    "force_limit_2x",
)
DEFAULT_ROOT_Z = 0.78
DEFAULT_MIN_UPRIGHT = 0.30
CURRENT_POSE_LEG_VALUES = {
    "left_hip_pitch_joint": -0.06,
    "right_hip_pitch_joint": -0.06,
    "left_knee_joint": 0.12,
    "right_knee_joint": 0.12,
    "left_ankle_pitch_joint": -0.07,
    "right_ankle_pitch_joint": -0.07,
}
ANKLE_JOINTS = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
KNEE_JOINTS = ("left_knee_joint", "right_knee_joint")
UNITREE_LEG_GAINS = {
    "hip_pitch": (100.0, 2.0),
    "hip_roll": (100.0, 2.0),
    "hip_yaw": (100.0, 2.0),
    "knee": (150.0, 4.0),
    "ankle_pitch": (40.0, 2.0),
    "ankle_roll": (40.0, 2.0),
}


@dataclass(frozen=True)
class DiagnosticGainProfile:
    name: str
    kp: tuple[float, ...]
    kv: tuple[float, ...]
    force_limits: tuple[float, ...]


def main() -> None:
    args = parse_args()
    result: dict[str, Any] = {
        "status": "error",
        "blocker": "",
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }
    exit_code = 0
    try:
        summary = run_probe(args)
        result.update(summary)
        result["status"] = summary["status"]
    except RECOVERABLE_RUNTIME_ERRORS as exc:  # pragma: no cover - H200 failure path.
        result["status"] = "error"
        result["blocker"] = f"{exc.__class__.__name__}:{exc}"
        exit_code = 1
    print(json.dumps(result, sort_keys=True), flush=True)
    if exit_code:
        raise SystemExit(exit_code)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=positive_int, default=1024)
    parser.add_argument("--chunks", type=positive_int, default=50)
    parser.add_argument("--chunk-steps", type=positive_int, default=32)
    parser.add_argument("--warmup-policy-steps", type=nonnegative_int, default=0)
    parser.add_argument("--pre-eval-reset", action="store_true")
    parser.add_argument(
        "--pre-eval-reset-scope",
        choices=("full", "all_env_ids"),
        default="full",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-mode", choices=CONTROL_MODES, default="genesis_position")
    parser.add_argument("--pose-profile", choices=POSE_PROFILES, default="current")
    parser.add_argument("--gain-profile", choices=GAIN_PROFILES, default="current")
    parser.add_argument("--root-z", type=positive_float, default=DEFAULT_ROOT_Z)
    parser.add_argument("--height-min", type=positive_float, default=0.45)
    parser.add_argument("--height-max", type=positive_float, default=1.20)
    parser.add_argument("--termination-height-min", type=positive_float, default=0.20)
    parser.add_argument("--termination-height-max", type=positive_float, default=1.20)
    parser.add_argument("--min-upright", type=positive_float, default=DEFAULT_MIN_UPRIGHT)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--logical-cuda-device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--asset-path", type=Path, default=None)
    return parser.parse_args(argv)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    verify_cuda_isolation(
        backend=args.backend,
        physical_gpu=str(args.physical_gpu),
        logical_cuda_device=args.logical_cuda_device,
    )
    torch = require_torch()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_profile = load_g1_27dof_nohand_profile()
    profile = profile_with_asset_path(base_profile, args.asset_path)
    effective_asset_path = str(profile.asset.path)
    pose = pose_profile_values(args.pose_profile, profile.control.default_angles_rad)
    gains = gain_profile_values(args.gain_profile, profile.control)
    run_dir = resolve_run_dir(args.output_root, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    config_payload = build_run_config(
        args=args,
        default_pose=pose,
        gains=gains,
        asset_path=effective_asset_path,
    )
    write_json(run_dir / "config.json", config_payload)

    backend = VectorizedGenesisBackend(
        VectorizedGenesisConfig(
            n_envs=args.n_envs,
            backend=args.backend,
            logical_cuda_device=args.logical_cuda_device,
            root_qpos=root_qpos(args.root_z),
            default_positions_rad=pose,
        ),
        profile=profile,
    )
    apply_gain_profile_to_backend(backend, gains)
    backend.reset()
    zero_action = torch.zeros((args.n_envs, profile.action_dim), device=args.logical_cuda_device)
    warmup_diagnostics = run_warmup(
        torch=torch,
        backend=backend,
        zero_action=zero_action,
        args=args,
    )
    if args.pre_eval_reset:
        reset_before_eval(torch=torch, backend=backend, args=args)
    rows: list[dict[str, Any]] = []
    metrics_path = run_dir / "metrics.jsonl"
    total_policy_steps = 0
    for chunk_index in range(args.chunks):
        row = run_chunk(
            torch=torch,
            backend=backend,
            zero_action=zero_action,
            args=args,
            action_joint_names=profile.actuator_order,
            chunk_index=chunk_index,
            total_policy_steps=total_policy_steps,
        )
        total_policy_steps += args.chunk_steps
        rows.append(row)
        append_jsonl(metrics_path, row)

    summary = summarize_run(
        rows=rows,
        args=args,
        run_dir=run_dir,
        warmup_diagnostics=warmup_diagnostics,
        asset_path=effective_asset_path,
    )
    write_json(run_dir / "summary.json", summary)
    return summary


def profile_with_asset_path(profile: Any, asset_path: Path | None) -> Any:
    if asset_path is None:
        return profile
    return replace(profile, asset=replace(profile.asset, path=asset_path.as_posix()))


def reset_before_eval(
    *,
    torch: Any,
    backend: VectorizedGenesisBackend,
    args: argparse.Namespace,
) -> None:
    if args.pre_eval_reset_scope == "full":
        backend.reset()
        return
    if args.pre_eval_reset_scope == "all_env_ids":
        env_ids = torch.arange(
            backend.n_envs,
            dtype=torch.long,
            device=args.logical_cuda_device,
        )
        backend.reset(env_ids)
        return
    raise ValueError(f"unknown pre-eval reset scope: {args.pre_eval_reset_scope}")


def run_warmup(
    *,
    torch: Any,
    backend: VectorizedGenesisBackend,
    zero_action: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.warmup_policy_steps == 0:
        return empty_warmup_diagnostics()

    height_bad_flags: list[Any] = []
    termination_height_bad_flags: list[Any] = []
    tilt_bad_flags: list[Any] = []
    root_height_values: list[Any] = []
    upright_values: list[Any] = []
    joint_error_values: list[Any] = []
    joint_velocity_values: list[Any] = []
    saturation_values: list[Any] = []

    synchronize_device(torch, args.logical_cuda_device)
    with torch.no_grad():
        for _ in range(args.warmup_policy_steps):
            control = apply_control_mode(
                torch=torch,
                backend=backend,
                action=zero_action,
                mode=args.control_mode,
            )
            state = backend.state()
            flags = standing_flags(torch=torch, state=state, config=env_config(args))
            height_bad_flags.append(flags["height_bad"])
            termination_height_bad_flags.append(flags["termination_height_bad"])
            tilt_bad_flags.append(flags["tilt_bad"])
            root_height_values.append(flags["root_height"])
            upright_values.append(flags["upright"])
            joint_error_values.append(state.dof_pos - backend.default_positions.unsqueeze(0))
            joint_velocity_values.append(state.dof_vel)
            saturation_values.append(control["saturated"])
    synchronize_device(torch, args.logical_cuda_device)

    joint_error_tensor = torch.stack(joint_error_values).float()
    joint_velocity_tensor = torch.stack(joint_velocity_values).float()
    saturation_tensor = torch.stack(saturation_values).bool()
    return {
        "policy_steps": args.warmup_policy_steps,
        "height_bad_count": sum_bool_tensors(torch, height_bad_flags),
        "termination_height_bad_count": sum_bool_tensors(
            torch,
            termination_height_bad_flags,
        ),
        "tilt_bad_count": sum_bool_tensors(torch, tilt_bad_flags),
        "root_height_mean": mean_tensors(torch, root_height_values),
        "root_height_min": min_tensors(torch, root_height_values),
        "upright_mean": mean_tensors(torch, upright_values),
        "upright_min": min_tensors(torch, upright_values),
        "joint_position_error_rms": rms_tensor(torch, joint_error_tensor),
        "joint_velocity_rms": rms_tensor(torch, joint_velocity_tensor),
        "force_saturation_ratio": float(saturation_tensor.float().mean().item()),
    }


def empty_warmup_diagnostics() -> dict[str, Any]:
    return {
        "policy_steps": 0,
        "height_bad_count": 0,
        "termination_height_bad_count": 0,
        "tilt_bad_count": 0,
        "root_height_mean": 0.0,
        "root_height_min": 0.0,
        "upright_mean": 0.0,
        "upright_min": 0.0,
        "joint_position_error_rms": 0.0,
        "joint_velocity_rms": 0.0,
        "force_saturation_ratio": 0.0,
    }


def run_chunk(
    *,
    torch: Any,
    backend: VectorizedGenesisBackend,
    zero_action: Any,
    args: argparse.Namespace,
    action_joint_names: Sequence[str],
    chunk_index: int,
    total_policy_steps: int,
) -> dict[str, Any]:
    height_bad_flags: list[Any] = []
    termination_height_bad_flags: list[Any] = []
    tilt_bad_flags: list[Any] = []
    root_height_values: list[Any] = []
    upright_values: list[Any] = []
    joint_error_values: list[Any] = []
    joint_velocity_values: list[Any] = []
    control_values: list[Any] = []
    saturation_values: list[Any] = []
    reset_count = 0
    contact_count_total = 0
    max_contact_force = 0.0

    synchronize_device(torch, args.logical_cuda_device)
    started = time.perf_counter()
    with torch.no_grad():
        for local_step in range(args.chunk_steps):
            control = apply_control_mode(
                torch=torch,
                backend=backend,
                action=zero_action,
                mode=args.control_mode,
            )
            state = backend.state()
            flags = standing_flags(torch=torch, state=state, config=env_config(args))
            height_bad_flags.append(flags["height_bad"])
            termination_height_bad_flags.append(flags["termination_height_bad"])
            tilt_bad_flags.append(flags["tilt_bad"])
            root_height_values.append(flags["root_height"])
            upright_values.append(flags["upright"])
            joint_error_values.append(state.dof_pos - backend.default_positions.unsqueeze(0))
            joint_velocity_values.append(state.dof_vel)
            control_values.append(control["values"])
            saturation_values.append(control["saturated"])
            contacts = read_contact_metrics(backend, torch=torch)
            contact_count_total += int(contacts["count"])
            max_contact_force = max(max_contact_force, float(contacts["max_force"]))
            done = flags["termination_height_bad"] | flags["tilt_bad"]
            done_env_ids = torch.nonzero(done, as_tuple=False).flatten()
            done_count = int(done_env_ids.numel())
            reset_count += done_count
            if done_count:
                backend.reset(done_env_ids)
            if done_count and chunk_index == 0 and local_step == 0:
                synchronize_device(torch, args.logical_cuda_device)
    synchronize_device(torch, args.logical_cuda_device)
    elapsed = time.perf_counter() - started

    joint_error_tensor = torch.stack(joint_error_values).float()
    joint_velocity_tensor = torch.stack(joint_velocity_values).float()
    control_tensor = torch.stack(control_values).float()
    saturation_tensor = torch.stack(saturation_values).bool()
    env_steps = args.chunk_steps * backend.n_envs
    throughput = env_steps / elapsed if elapsed > 0.0 else 0.0
    row = {
        "seed": args.seed,
        "control_mode": args.control_mode,
        "pose_profile": args.pose_profile,
        "gain_profile": args.gain_profile,
        "chunk_index": chunk_index,
        "chunk_steps": args.chunk_steps,
        "n_envs": backend.n_envs,
        "env_steps": env_steps,
        "total_policy_steps": total_policy_steps + args.chunk_steps,
        "reset_count": reset_count,
        "height_bad_count": sum_bool_tensors(torch, height_bad_flags),
        "termination_height_bad_count": sum_bool_tensors(
            torch,
            termination_height_bad_flags,
        ),
        "tilt_bad_count": sum_bool_tensors(torch, tilt_bad_flags),
        "root_height_mean": mean_tensors(torch, root_height_values),
        "root_height_min": min_tensors(torch, root_height_values),
        "upright_mean": mean_tensors(torch, upright_values),
        "upright_min": min_tensors(torch, upright_values),
        "joint_position_error_rms": rms_tensor(torch, joint_error_tensor),
        "joint_position_error_max": max_abs_tensor(joint_error_tensor),
        "joint_velocity_rms": rms_tensor(torch, joint_velocity_tensor),
        "joint_velocity_max": max_abs_tensor(joint_velocity_tensor),
        "control_kind": control["kind"],
        "control_rms": rms_tensor(torch, control_tensor),
        "control_max": max_abs_tensor(control_tensor),
        "force_saturation_ratio": float(saturation_tensor.float().mean().item()),
        "foot_or_body_contact_count": contact_count_total,
        "max_contact_force": max_contact_force,
        "throughput_env_steps_per_sec": throughput,
        "logical_cuda_device": args.logical_cuda_device,
        "tensor_device_ok": tensor_device_ok(
            (
                joint_error_tensor,
                joint_velocity_tensor,
                control_tensor,
            ),
            args.logical_cuda_device,
        ),
        "top_joint_position_error_rms": top_rms_entries(
            torch,
            joint_error_tensor,
            action_joint_names,
        ),
    }
    assert_metric_row_ok(row)
    return row


def apply_control_mode(
    *,
    torch: Any,
    backend: VectorizedGenesisBackend,
    action: Any,
    mode: str,
) -> dict[str, Any]:
    if mode == "genesis_position":
        clipped_action = backend.step_physics(action)
        targets = backend._action_targets(clipped_action)
        saturated = torch.zeros_like(targets, dtype=torch.bool)
        return {"kind": "position", "values": targets, "saturated": saturated}

    clipped_action = backend._clip_action(backend._coerce_action(action))
    targets = backend._action_targets(clipped_action)
    if mode == "genesis_position_resend_physics":
        for _ in range(backend.decimation):
            backend._control_dofs_position(targets)
            backend.scene.step()
        backend.previous_action = clipped_action
        backend.step_count += 1
        saturated = torch.zeros_like(targets, dtype=torch.bool)
        return {"kind": "position", "values": targets, "saturated": saturated}

    if mode == "custom_pd_torque":
        values = None
        saturated = None
        for _ in range(backend.decimation):
            state = backend.state()
            torque = compute_pd_torque(torch=torch, backend=backend, targets=targets, state=state)
            values = torque["clipped"]
            saturated = torque["saturated"]
            control_dofs_torque(backend, values)
            backend.scene.step()
        backend.previous_action = clipped_action
        backend.step_count += 1
        if values is None or saturated is None:
            raise RuntimeError("custom_pd_torque produced no control values")
        return {"kind": "torque", "values": values, "saturated": saturated}

    raise ValueError(f"unknown control mode: {mode}")


def compute_pd_torque(
    *,
    torch: Any,
    backend: VectorizedGenesisBackend,
    targets: Any,
    state: Any,
) -> dict[str, Any]:
    gains = diagnostic_gains_for_backend(backend)
    kp = torch.tensor(
        gains.kp,
        dtype=torch.float32,
        device=backend.config.logical_cuda_device,
    ) * float(backend.motor_kp_mult)
    kv = torch.tensor(
        gains.kv,
        dtype=torch.float32,
        device=backend.config.logical_cuda_device,
    ) * float(backend.motor_kv_mult)
    limits = torch.tensor(
        gains.force_limits,
        dtype=torch.float32,
        device=backend.config.logical_cuda_device,
    ) * float(backend.motor_force_limit_mult)
    raw = kp.unsqueeze(0) * (targets - state.dof_pos) - kv.unsqueeze(0) * state.dof_vel
    clipped = raw.clamp(-limits.unsqueeze(0), limits.unsqueeze(0))
    return {
        "raw": raw,
        "clipped": clipped,
        "saturated": raw.abs() >= limits.unsqueeze(0),
    }


def control_dofs_torque(backend: VectorizedGenesisBackend, torque: Any) -> None:
    robot = backend.robot
    for method_name in ("control_dofs_force", "control_dofs_torque"):
        if hasattr(robot, method_name):
            method = getattr(robot, method_name)
            try:
                method(torque, dofs_idx_local=backend.motor_dof_indices)
            except TypeError:
                method(torque, backend.motor_dof_indices)
            return
    raise RuntimeError("Genesis robot has no torque/force control method")


def standing_flags(*, torch: Any, state: Any, config: G1VelocityTrackingConfig) -> dict[str, Any]:
    root_height = state.root_pos[:, 2]
    projected_gravity = projected_gravity_torch(torch, state.root_quat)
    upright = (-projected_gravity[:, 2]).clamp(0.0, 1.0)
    height_bad = (root_height < config.height_min) | (root_height > config.height_max)
    termination_height_bad = (root_height < config.termination_height_min) | (
        root_height > config.termination_height_max
    )
    tilt_bad = upright < config.min_upright
    return {
        "root_height": root_height,
        "upright": upright,
        "height_bad": height_bad,
        "termination_height_bad": termination_height_bad,
        "tilt_bad": tilt_bad,
    }


def projected_gravity_torch(torch: Any, root_quat: Any) -> Any:
    quat = root_quat / root_quat.norm(dim=1, keepdim=True).clamp_min(1e-6)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    return torch.stack(
        (
            2.0 * (w * y - x * z),
            -2.0 * (w * x + y * z),
            -1.0 + 2.0 * (x.square() + y.square()),
        ),
        dim=1,
    )


def env_config(args: argparse.Namespace) -> G1VelocityTrackingConfig:
    return G1VelocityTrackingConfig(
        command_vx_min=0.0,
        command_vx_max=0.0,
        command_yaw_min=0.0,
        command_yaw_max=0.0,
        height_min=args.height_min,
        height_max=args.height_max,
        termination_height_min=args.termination_height_min,
        termination_height_max=args.termination_height_max,
        min_upright=args.min_upright,
    )


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


def gain_profile_values(name: str, control: Any) -> DiagnosticGainProfile:
    kp = [float(value) for value in control.kp]
    kv = [float(value) for value in control.kv]
    force_limits = [float(value) for value in control.force_limits]
    if len(kp) != 27 or len(kv) != 27 or len(force_limits) != 27:
        raise ValueError("gain profile inputs must have 27 values")

    if name == "current":
        pass
    elif name == "global_kv_2x":
        kv = scale_selected(kv, 2.0)
    elif name == "global_kv_4x":
        kv = scale_selected(kv, 4.0)
    elif name == "global_kp_0_5x_kv_2x":
        kp = scale_selected(kp, 0.5)
        kv = scale_selected(kv, 2.0)
    elif name == "ankle_kp_2x_kv_2x":
        kp = scale_selected(kp, 2.0, ANKLE_JOINTS)
        kv = scale_selected(kv, 2.0, ANKLE_JOINTS)
    elif name == "knee_ankle_kp_2x_kv_2x":
        selected = (*KNEE_JOINTS, *ANKLE_JOINTS)
        kp = scale_selected(kp, 2.0, selected)
        kv = scale_selected(kv, 2.0, selected)
    elif name == "unitree_leg_gains":
        for index, joint_name in enumerate(G1_27DOF_NOHAND_ACTUATOR_ORDER):
            for suffix, values in UNITREE_LEG_GAINS.items():
                if joint_name.endswith(f"_{suffix}_joint"):
                    kp[index], kv[index] = values
                    break
    elif name == "force_limit_2x":
        force_limits = scale_selected(force_limits, 2.0)
    else:
        raise ValueError(f"unknown gain profile: {name}")

    return DiagnosticGainProfile(
        name=name,
        kp=tuple(kp),
        kv=tuple(kv),
        force_limits=tuple(force_limits),
    )


def scale_selected(
    values: Sequence[float],
    multiplier: float,
    joint_names: Sequence[str] | None = None,
) -> list[float]:
    scaled = [float(value) for value in values]
    if joint_names is None:
        return [value * multiplier for value in scaled]
    for joint_name in joint_names:
        scaled[G1_27DOF_NOHAND_ACTUATOR_ORDER.index(joint_name)] *= multiplier
    return scaled


def apply_gain_profile_to_backend(
    backend: VectorizedGenesisBackend,
    gains: DiagnosticGainProfile,
) -> None:
    backend._diagnostic_gain_profile = gains
    set_robot_dofs_kp(
        backend,
        scale_selected(gains.kp, float(backend.motor_kp_mult)),
    )
    set_robot_dofs_kv(
        backend,
        scale_selected(gains.kv, float(backend.motor_kv_mult)),
    )
    set_robot_dofs_force_range(
        backend,
        scale_selected(gains.force_limits, float(backend.motor_force_limit_mult)),
    )


def diagnostic_gains_for_backend(backend: VectorizedGenesisBackend) -> DiagnosticGainProfile:
    gains = getattr(backend, "_diagnostic_gain_profile", None)
    if isinstance(gains, DiagnosticGainProfile):
        return gains
    return DiagnosticGainProfile(
        name="current",
        kp=tuple(float(value) for value in backend.profile.control.kp),
        kv=tuple(float(value) for value in backend.profile.control.kv),
        force_limits=tuple(float(value) for value in backend.profile.control.force_limits),
    )


def set_robot_dofs_kp(backend: VectorizedGenesisBackend, values: Sequence[float]) -> None:
    robot = backend.robot
    if not hasattr(robot, "set_dofs_kp"):
        return
    gains = tuple(float(value) for value in values)
    try:
        robot.set_dofs_kp(gains, dofs_idx_local=backend.motor_dof_indices)
    except TypeError:
        robot.set_dofs_kp(gains, backend.motor_dof_indices)


def set_robot_dofs_kv(backend: VectorizedGenesisBackend, values: Sequence[float]) -> None:
    robot = backend.robot
    if not hasattr(robot, "set_dofs_kv"):
        return
    gains = tuple(float(value) for value in values)
    try:
        robot.set_dofs_kv(gains, dofs_idx_local=backend.motor_dof_indices)
    except TypeError:
        robot.set_dofs_kv(gains, backend.motor_dof_indices)


def set_robot_dofs_force_range(
    backend: VectorizedGenesisBackend,
    values: Sequence[float],
) -> None:
    robot = backend.robot
    if not hasattr(robot, "set_dofs_force_range"):
        return
    lower = tuple(-float(value) for value in values)
    upper = tuple(float(value) for value in values)
    try:
        robot.set_dofs_force_range(lower, upper, dofs_idx_local=backend.motor_dof_indices)
    except TypeError:
        robot.set_dofs_force_range(lower, upper, backend.motor_dof_indices)


def gain_value_summary(gains: DiagnosticGainProfile) -> dict[str, dict[str, float]]:
    index = G1_27DOF_NOHAND_ACTUATOR_ORDER.index
    return {
        joint_name: {
            "kp": float(gains.kp[index(joint_name)]),
            "kv": float(gains.kv[index(joint_name)]),
            "force_limit": float(gains.force_limits[index(joint_name)]),
        }
        for joint_name in (
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
        )
    }


def build_run_config(
    *,
    args: argparse.Namespace,
    default_pose: Sequence[float],
    gains: DiagnosticGainProfile,
    asset_path: str = "",
) -> dict[str, Any]:
    return {
        "task": "task019-g1-zero-action-standing-causality-diagnosis",
        "seed": args.seed,
        "n_envs": args.n_envs,
        "chunks": args.chunks,
        "chunk_steps": args.chunk_steps,
        "warmup_policy_steps": args.warmup_policy_steps,
        "pre_eval_reset": args.pre_eval_reset,
        "pre_eval_reset_scope": args.pre_eval_reset_scope,
        "control_mode": args.control_mode,
        "pose_profile": args.pose_profile,
        "gain_profile": args.gain_profile,
        "pose_leg_values_rad": leg_value_summary(default_pose),
        "gain_values": gain_value_summary(gains),
        "root_z": args.root_z,
        "height_min": args.height_min,
        "height_max": args.height_max,
        "termination_height_min": args.termination_height_min,
        "termination_height_max": args.termination_height_max,
        "min_upright": args.min_upright,
        "backend": args.backend,
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "asset_path": asset_path,
    }


def summarize_run(
    *,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
    warmup_diagnostics: dict[str, Any] | None = None,
    asset_path: str = "",
) -> dict[str, Any]:
    warmup = warmup_diagnostics or empty_warmup_diagnostics()
    diagnostics = summarize_chunk_diagnostics(rows)
    completed = len(rows) == args.chunks
    tensors_ok = all(bool(row["tensor_device_ok"]) for row in rows)
    evaluation_passed = (
        completed
        and tensors_ok
        and diagnostics["max_reset_count"] == 0
        and diagnostics["max_tilt_bad_count"] == 0
    )
    diagnostic_passed = (
        evaluation_passed
        and args.warmup_policy_steps == 0
        and not args.pre_eval_reset
    )
    if diagnostic_passed:
        status = "passed"
    elif evaluation_passed:
        status = "diagnostic_passed"
    else:
        status = "failed"
    return {
        "status": status,
        "passed": diagnostic_passed,
        "evaluation_passed": evaluation_passed,
        "diagnostic_passed": diagnostic_passed,
        "run_dir": str(run_dir),
        "control_mode": args.control_mode,
        "pose_profile": args.pose_profile,
        "gain_profile": args.gain_profile,
        "warmup_policy_steps": args.warmup_policy_steps,
        "pre_eval_reset": args.pre_eval_reset,
        "pre_eval_reset_scope": args.pre_eval_reset_scope,
        "warmup_diagnostics": warmup,
        "chunks_completed": len(rows),
        "chunks_expected": args.chunks,
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "asset_path": asset_path,
        **diagnostics,
    }


def summarize_chunk_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "first_tilt_chunk": None,
            "first_tilt_step": None,
            "max_reset_count": 0,
            "final_reset_count": 0,
            "max_tilt_bad_count": 0,
            "final_tilt_bad_count": 0,
            "final_root_height_mean": 0.0,
            "final_root_height_min": 0.0,
            "final_upright_mean": 0.0,
            "final_joint_position_error_rms": 0.0,
            "final_joint_velocity_rms": 0.0,
            "min_throughput_env_steps_per_sec": 0.0,
        }
    final = rows[-1]
    first_tilt = next((row for row in rows if int(row["tilt_bad_count"]) > 0), None)
    return {
        "first_tilt_chunk": None if first_tilt is None else int(first_tilt["chunk_index"]),
        "first_tilt_step": (
            None
            if first_tilt is None
            else int(first_tilt["chunk_index"]) * int(first_tilt["chunk_steps"])
        ),
        "max_reset_count": max(int(row["reset_count"]) for row in rows),
        "final_reset_count": int(final["reset_count"]),
        "max_tilt_bad_count": max(int(row["tilt_bad_count"]) for row in rows),
        "final_tilt_bad_count": int(final["tilt_bad_count"]),
        "final_termination_height_bad_count": int(final["termination_height_bad_count"]),
        "final_root_height_mean": float(final["root_height_mean"]),
        "final_root_height_min": float(final["root_height_min"]),
        "final_upright_mean": float(final["upright_mean"]),
        "final_joint_position_error_rms": float(final["joint_position_error_rms"]),
        "final_joint_velocity_rms": float(final["joint_velocity_rms"]),
        "min_throughput_env_steps_per_sec": min(
            float(row["throughput_env_steps_per_sec"]) for row in rows
        ),
    }


def read_contact_metrics(backend: VectorizedGenesisBackend, *, torch: Any) -> dict[str, float]:
    robot = backend.robot
    for method_name in (
        "get_links_net_contact_force",
        "get_links_net_contact_forces",
        "get_contact_forces",
    ):
        if not hasattr(robot, method_name):
            continue
        value_available = True
        try:
            value = getattr(robot, method_name)()
        except RECOVERABLE_RUNTIME_ERRORS:
            value_available = False
        if not value_available:
            continue
        if value is None:
            continue
        if is_tensor_like(value):
            magnitudes = value.detach().float().norm(dim=-1)
            active = magnitudes > 0.0
            return {
                "count": float(active.sum().item()),
                "max_force": float(magnitudes.max().item()) if magnitudes.numel() else 0.0,
            }
        rows = as_rows(value)
        if not rows:
            continue
        magnitudes = [
            sum(component * component for component in row) ** 0.5 for row in rows
        ]
        return {
            "count": float(sum(1 for item in magnitudes if item > 0.0)),
            "max_force": max(magnitudes) if magnitudes else 0.0,
        }
    return {"count": 0.0, "max_force": 0.0}


def resolve_run_dir(output_root: Path, run_id: str) -> Path:
    root = output_root if output_root.is_absolute() else Path.cwd() / output_root
    run_name = run_id.strip() or time.strftime("%Y%m%d-%H%M%S")
    run_dir = (root / run_name).resolve()
    project_prefix = PROJECT_PREFIX.resolve()
    if project_prefix not in (run_dir, *run_dir.parents):
        raise RuntimeError(f"output path must stay under {project_prefix}: {run_dir}")
    return run_dir


def verify_cuda_isolation(
    *,
    backend: str,
    physical_gpu: str,
    logical_cuda_device: str,
) -> None:
    if backend != "cuda":
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if tokens != [physical_gpu] or physical_gpu != "1":
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES expected physical GPU 1, got {visible}")
    if logical_cuda_device != "cuda:0":
        raise RuntimeError("logical_cuda_device must be cuda:0")
    torch = require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("torch CUDA unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible CUDA device, got {torch.cuda.device_count()}")


def root_qpos(root_z: float) -> tuple[float, float, float, float, float, float, float]:
    return (0.0, 0.0, float(root_z), 1.0, 0.0, 0.0, 0.0)


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


def sum_bool_tensors(torch: Any, values: list[Any]) -> int:
    if not values:
        return 0
    return int(torch.stack(values).sum().item())


def mean_tensors(torch: Any, values: list[Any]) -> float:
    if not values:
        return 0.0
    return float(torch.stack(values).float().mean().item())


def min_tensors(torch: Any, values: list[Any]) -> float:
    if not values:
        return 0.0
    return float(torch.stack(values).float().min().item())


def rms_tensor(torch: Any, value: Any) -> float:
    return float(value.square().mean().sqrt().item()) if value.numel() else 0.0


def max_abs_tensor(value: Any) -> float:
    return float(value.abs().max().item()) if value.numel() else 0.0


def top_rms_entries(
    torch: Any,
    values: Any,
    joint_names: Sequence[str],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    rms = values.square().mean(dim=(0, 1)).sqrt()
    count = min(top_k, int(rms.numel()), len(joint_names))
    top_values, top_indices = torch.topk(rms, count)
    return [
        {"joint": joint_names[int(index.item())], "rms": float(value.item())}
        for value, index in zip(top_values, top_indices)
    ]


def assert_metric_row_ok(row: dict[str, Any]) -> None:
    finite_keys = (
        "root_height_mean",
        "root_height_min",
        "upright_mean",
        "upright_min",
        "joint_position_error_rms",
        "joint_position_error_max",
        "joint_velocity_rms",
        "joint_velocity_max",
        "control_rms",
        "control_max",
        "force_saturation_ratio",
        "throughput_env_steps_per_sec",
    )
    for key in finite_keys:
        value = float(row[key])
        if not math.isfinite(value):
            raise ValueError(f"{key} is not finite: {value}")
    if not row["tensor_device_ok"]:
        raise ValueError("tensor_device_ok is false")


def tensor_device_ok(values: Sequence[Any], logical_cuda_device: str) -> bool:
    return all(str(getattr(value, "device", "")) == logical_cuda_device for value in values)


def synchronize_device(torch: Any, logical_cuda_device: str) -> None:
    if logical_cuda_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - H200 environment path.
        raise RuntimeError(f"torch import failed: {exc}") from exc
    return torch


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, sort_keys=True) + "\n")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


if __name__ == "__main__":
    main()
