"""Probe zero-action standing reset poses for the G1 vectorized Genesis env."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from h200_locomotion_lab.envs.g1_velocity_tracking_env import (
    G1VelocityTrackingConfig,
    G1VelocityTrackingVectorizedEnv,
)
from h200_locomotion_lab.envs.g1_reset_poses import (
    G1_STANDING_RESET_POSE_NAMES,
    build_g1_standing_reset_pose_candidates,
    leg_value_summary,
)
from h200_locomotion_lab.envs.vectorized_genesis_backend import (
    VectorizedGenesisBackend,
    VectorizedGenesisConfig,
)
from h200_locomotion_lab.robots import load_g1_27dof_nohand_profile
from h200_locomotion_lab.training.ppo_loop import require_torch, synchronize_device
from h200_locomotion_lab.tools.g1_ppo_smoke import (
    PROJECT_PREFIX,
    non_negative_float,
    positive_float,
    positive_int,
    resolve_run_dir,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/task014/standing_reset_pose_probe")
DEFAULT_ROOT_Z_VALUES = "0.90,1.00,1.10,1.20"
DEFAULT_POSE_CANDIDATES = ",".join(G1_STANDING_RESET_POSE_NAMES)
DEFAULT_TERMINATION_HEIGHT_MIN_VALUES = "0.20,0.25,0.30,0.35,0.40,0.45"


def main() -> None:
    args = parse_args()
    result: dict[str, Any] = {
        "status": "failed",
        "blocker": "",
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }
    try:
        result.update(run_probe(args))
        result["status"] = "ok"
    except Exception as exc:  # pragma: no cover - H200 failure path.
        result["blocker"] = f"{exc.__class__.__name__}:{exc}"
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "ok":
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=positive_int, default=1024)
    parser.add_argument("--steps", type=positive_int, default=64)
    parser.add_argument("--root-z-values", default=DEFAULT_ROOT_Z_VALUES)
    parser.add_argument("--pose-candidates", default=DEFAULT_POSE_CANDIDATES)
    parser.add_argument("--height-min", type=non_negative_float, default=0.45)
    parser.add_argument("--height-max", type=positive_float, default=1.20)
    parser.add_argument(
        "--termination-height-min-values",
        default=DEFAULT_TERMINATION_HEIGHT_MIN_VALUES,
    )
    parser.add_argument("--termination-height-max", type=positive_float, default=1.20)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--logical-cuda-device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    verify_cuda_isolation(
        backend=args.backend,
        physical_gpu=str(args.physical_gpu),
        logical_cuda_device=args.logical_cuda_device,
    )
    torch = require_torch()
    profile = load_g1_27dof_nohand_profile()
    root_z_values = parse_float_list(args.root_z_values)
    pose_names = parse_name_list(args.pose_candidates)
    termination_height_min_values = parse_float_list(args.termination_height_min_values)
    candidates = build_g1_standing_reset_pose_candidates(profile.control.default_angles_rad)
    missing = [name for name in pose_names if name not in candidates]
    if missing:
        raise ValueError(f"unknown pose candidate: {missing[0]}")

    run_dir = resolve_run_dir(args.output_root, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        run_dir / "config.json",
        {
            "n_envs": args.n_envs,
            "steps": args.steps,
            "root_z_values": root_z_values,
            "pose_candidates": pose_names,
            "height_min": args.height_min,
            "height_max": args.height_max,
            "termination_height_min_values": termination_height_min_values,
            "termination_height_max": args.termination_height_max,
            "backend": args.backend,
            "physical_gpu": str(args.physical_gpu),
            "logical_cuda_device": args.logical_cuda_device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        },
    )

    first_pose = candidates[pose_names[0]]
    backend = VectorizedGenesisBackend(
        VectorizedGenesisConfig(
            n_envs=args.n_envs,
            backend=args.backend,
            logical_cuda_device=args.logical_cuda_device,
            root_qpos=root_qpos(root_z_values[0]),
            default_positions_rad=first_pose,
        ),
        profile=profile,
    )
    zero_action = torch.zeros((args.n_envs, backend.action_dim), device=args.logical_cuda_device)
    rows: list[dict[str, Any]] = []
    rows_path = run_dir / "rows.jsonl"
    for pose_name in pose_names:
        pose = candidates[pose_name]
        for root_z in root_z_values:
            for termination_height_min in termination_height_min_values:
                env_config = G1VelocityTrackingConfig(
                    command_vx_min=0.0,
                    command_vx_max=0.0,
                    command_yaw_min=0.0,
                    command_yaw_max=0.0,
                    height_min=args.height_min,
                    height_max=args.height_max,
                    termination_height_min=termination_height_min,
                    termination_height_max=args.termination_height_max,
                )
                row = run_candidate(
                    torch=torch,
                    backend=backend,
                    env_config=env_config,
                    zero_action=zero_action,
                    pose_name=pose_name,
                    pose=pose,
                    root_z=root_z,
                    steps=args.steps,
                    n_envs=args.n_envs,
                    logical_cuda_device=args.logical_cuda_device,
                )
                rows.append(row)
                append_jsonl(rows_path, row)

    stable_rows = [
        row
        for row in rows
        if row["reset_count"] == 0
        and row["termination_height_bad_count"] == 0
        and row["tilt_bad_count"] == 0
    ]
    best = choose_best_row(rows)
    summary = {
        "run_dir": str(run_dir),
        "stable_found": bool(stable_rows),
        "stable_rows": stable_rows,
        "best": best,
        "row_count": len(rows),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def run_candidate(
    *,
    torch: Any,
    backend: VectorizedGenesisBackend,
    env_config: G1VelocityTrackingConfig,
    zero_action: Any,
    pose_name: str,
    pose: tuple[float, ...],
    root_z: float,
    steps: int,
    n_envs: int,
    logical_cuda_device: str,
) -> dict[str, Any]:
    backend.set_reset_pose(root_qpos=root_qpos(root_z), default_positions_rad=pose)
    env = G1VelocityTrackingVectorizedEnv(backend, env_config)
    observation = env.reset()
    if str(getattr(observation, "device", "")) != logical_cuda_device:
        raise RuntimeError(f"observation device mismatch: {getattr(observation, 'device', None)}")
    reset_count = 0
    done_values: list[Any] = []
    height_bad_values: list[Any] = []
    termination_height_bad_values: list[Any] = []
    tilt_bad_values: list[Any] = []
    root_height_values: list[Any] = []
    upright_values: list[Any] = []
    synchronize_device(getattr(observation, "device", None))
    started = time.perf_counter()
    for _ in range(steps):
        transition = env.step(zero_action)
        components = transition.info["components"]
        reset_count += int(transition.info["reset_count"])
        done_values.append(transition.done)
        height_bad_values.append(components["height_bad"])
        termination_height_bad_values.append(components["termination_height_bad"])
        tilt_bad_values.append(components["tilt_bad"])
        root_height_values.append(components["root_height"])
        upright_values.append(components["upright"])
    synchronize_device(getattr(observation, "device", None))
    elapsed = time.perf_counter() - started
    done_tensor = torch.stack(done_values)
    height_bad_tensor = torch.stack(height_bad_values)
    termination_height_bad_tensor = torch.stack(termination_height_bad_values)
    tilt_bad_tensor = torch.stack(tilt_bad_values)
    root_height_tensor = torch.stack(root_height_values).float()
    upright_tensor = torch.stack(upright_values).float()
    return {
        "pose": pose_name,
        "root_z": root_z,
        "height_min": env_config.height_min,
        "height_max": env_config.height_max,
        "termination_height_min": env_config.termination_height_min,
        "termination_height_max": env_config.termination_height_max,
        "steps": steps,
        "n_envs": n_envs,
        "env_steps": steps * n_envs,
        "time_s": elapsed,
        "env_policy_steps_per_sec": (steps * n_envs) / elapsed if elapsed > 0.0 else 0.0,
        "reset_count": reset_count,
        "done_count": int(done_tensor.sum().item()),
        "height_bad_count": int(height_bad_tensor.sum().item()),
        "termination_height_bad_count": int(termination_height_bad_tensor.sum().item()),
        "tilt_bad_count": int(tilt_bad_tensor.sum().item()),
        "root_height_mean": float(root_height_tensor.mean().item()),
        "root_height_min": float(root_height_tensor.min().item()),
        "root_height_final_mean": float(root_height_tensor[-1].mean().item()),
        "upright_mean": float(upright_tensor.mean().item()),
        "upright_min": float(upright_tensor.min().item()),
        "leg_values": leg_value_summary(pose),
    }


def choose_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no rows to choose from")
    return min(
        rows,
        key=lambda row: (
            row["reset_count"],
            row["tilt_bad_count"],
            row.get("termination_height_bad_count", row["height_bad_count"]),
            row["height_bad_count"],
            -row["root_height_min"],
            -row["env_policy_steps_per_sec"],
        ),
    )


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
    if tokens != [physical_gpu]:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES expected {physical_gpu}, got {visible}")
    if logical_cuda_device != "cuda:0":
        raise RuntimeError("logical_cuda_device must be cuda:0")
    torch = require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("torch CUDA unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible CUDA device, got {torch.cuda.device_count()}")


def root_qpos(root_z: float) -> tuple[float, float, float, float, float, float, float]:
    return (0.0, 0.0, root_z, 1.0, 0.0, 0.0, 0.0)


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one float value is required")
    return values


def parse_name_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one pose candidate is required")
    return values


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
