"""Standalone zero-action G1 standing causality probe."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

from h200_locomotion_lab.envs.g1_velocity_tracking_env import G1VelocityTrackingConfig
from h200_locomotion_lab.envs.vectorized_genesis_backend import (
    VectorizedGenesisBackend,
    VectorizedGenesisConfig,
    as_rows,
    is_tensor_like,
)
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
    except Exception as exc:  # pragma: no cover - H200 failure path.
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-mode", choices=CONTROL_MODES, default="genesis_position")
    parser.add_argument("--pose-profile", choices=POSE_PROFILES, default="current")
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

    profile = load_g1_27dof_nohand_profile()
    pose = pose_profile_values(args.pose_profile, profile.control.default_angles_rad)
    run_dir = resolve_run_dir(args.output_root, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    config_payload = build_run_config(args=args, default_pose=pose)
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
    backend.reset()
    zero_action = torch.zeros((args.n_envs, profile.action_dim), device=args.logical_cuda_device)
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

    summary = summarize_run(rows=rows, args=args, run_dir=run_dir)
    write_json(run_dir / "summary.json", summary)
    return summary


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
    kp = torch.tensor(
        backend.profile.control.kp,
        dtype=torch.float32,
        device=backend.config.logical_cuda_device,
    ) * float(backend.motor_kp_mult)
    kv = torch.tensor(
        backend.profile.control.kv,
        dtype=torch.float32,
        device=backend.config.logical_cuda_device,
    ) * float(backend.motor_kv_mult)
    limits = torch.tensor(
        backend.profile.control.force_limits,
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


def build_run_config(*, args: argparse.Namespace, default_pose: Sequence[float]) -> dict[str, Any]:
    return {
        "task": "task019-g1-zero-action-standing-causality-diagnosis",
        "seed": args.seed,
        "n_envs": args.n_envs,
        "chunks": args.chunks,
        "chunk_steps": args.chunk_steps,
        "control_mode": args.control_mode,
        "pose_profile": args.pose_profile,
        "pose_leg_values_rad": leg_value_summary(default_pose),
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
    }


def summarize_run(
    *,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    diagnostics = summarize_chunk_diagnostics(rows)
    completed = len(rows) == args.chunks
    tensors_ok = all(bool(row["tensor_device_ok"]) for row in rows)
    passed = (
        completed
        and tensors_ok
        and diagnostics["max_reset_count"] == 0
        and diagnostics["max_tilt_bad_count"] == 0
    )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "run_dir": str(run_dir),
        "control_mode": args.control_mode,
        "pose_profile": args.pose_profile,
        "chunks_completed": len(rows),
        "chunks_expected": args.chunks,
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
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
        try:
            value = getattr(robot, method_name)()
        except Exception:
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
        if value != value or value in (float("inf"), float("-inf")):
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


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


if __name__ == "__main__":
    main()
