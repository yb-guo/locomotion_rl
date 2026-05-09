"""Component timing profile for Genesis official batched APIs.

This diagnostic separates action-write, state-read, raw scene-step, and
combined policy-loop costs. It reuses the independent official API probe
helpers and does not import Genesis or Torch at module import time.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from h200_locomotion_lab.tools.genesis_official_batched_api_probe import (
    ASSET_KINDS,
    ASSET_VARIANTS,
    UNAVAILABLE,
    BlockedProbe,
    build_scene,
    compute_throughput,
    exercise_selected_reset,
    format_key_value,
    gpu_snapshot,
    import_genesis_module,
    init_genesis,
    make_action_target,
    positive_int,
    read_state,
    resolve_dof_indices,
    state_device_metrics,
    tensor_device_name,
    tensor_devices_ok,
    verify_single_visible_cuda_device,
    write_action_targets,
)


PROFILE_KEYS: tuple[str, ...] = (
    "status",
    "asset_kind",
    "asset_variant",
    "cuda_visible_devices",
    "physical_gpu",
    "logical_cuda_device",
    "backend",
    "n_envs",
    "build_time_s",
    "warmup_time_s",
    "action_write_time_s",
    "action_writes_per_sec",
    "state_read_time_s",
    "state_reads_per_sec",
    "scene_step_time_s",
    "scene_steps_per_sec",
    "env_scene_steps_per_sec",
    "combined_loop_time_s",
    "combined_policy_steps_per_sec",
    "combined_env_policy_steps_per_sec",
    "combined_env_sim_steps_per_sec",
    "profile_policy_steps",
    "decimation",
    "action_device",
    "qpos_device",
    "dofs_pos_device",
    "dofs_vel_device",
    "root_pos_device",
    "root_quat_device",
    "root_vel_device",
    "tensor_device_ok",
    "selected_reset_supported",
    "selected_reset_changes_only_target_envs",
    "selected_reset_time_s",
    "gpu_snapshot_before",
    "gpu_snapshot_after",
    "blocker",
)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    metrics = initial_profile_metrics(args)

    try:
        result = run_component_profile(args, metrics)
        emit_ordered_metrics(result)
    except BlockedProbe as exc:
        metrics["status"] = "blocked"
        metrics["blocker"] = str(exc).replace("\n", "\\n")
        emit_ordered_metrics(metrics)
    except Exception as exc:  # pragma: no cover - target-only failure path.
        metrics["status"] = "failed"
        metrics["blocker"] = f"{exc.__class__.__name__}: {exc}".replace("\n", "\\n")
        emit_ordered_metrics(metrics)
        raise SystemExit(1) from None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Genesis batched API component costs.")
    parser.add_argument("--asset-kind", choices=ASSET_KINDS, required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--n-envs", type=positive_int, default=1024)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--variant", choices=ASSET_VARIANTS, default="default")
    parser.add_argument("--warmup-policy-steps", type=positive_int, default=20)
    parser.add_argument("--profile-policy-steps", type=positive_int, default=100)
    parser.add_argument("--decimation", type=positive_int, default=4)
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--logical-cuda-device", default="cuda:0")
    parser.add_argument("--sim-dt", type=float, default=0.005)
    return parser


def initial_profile_metrics(args: argparse.Namespace) -> dict[str, Any]:
    metrics = {key: UNAVAILABLE for key in PROFILE_KEYS}
    metrics.update(
        {
            "status": "blocked",
            "asset_kind": args.asset_kind,
            "asset_variant": args.variant,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
            "physical_gpu": str(args.physical_gpu),
            "logical_cuda_device": args.logical_cuda_device,
            "backend": args.backend,
            "n_envs": args.n_envs,
            "profile_policy_steps": args.profile_policy_steps,
            "decimation": args.decimation,
            "tensor_device_ok": False,
            "selected_reset_supported": False,
            "selected_reset_changes_only_target_envs": False,
            "blocker": "",
        }
    )
    return metrics


def run_component_profile(
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    torch_module = None
    if args.backend == "cuda":
        cuda_result = verify_single_visible_cuda_device(
            physical_gpu=str(args.physical_gpu),
            logical_cuda_device=args.logical_cuda_device,
        )
        if not cuda_result.ok:
            raise BlockedProbe(cuda_result.blocker)
        torch_module = cuda_result.torch_module

    asset_path = Path(args.asset)
    if not asset_path.is_file():
        raise BlockedProbe(f"asset_not_found:{asset_path}")

    metrics["gpu_snapshot_before"] = gpu_snapshot()
    gs = import_genesis_module()
    init_genesis(gs, args.backend)

    started = time.perf_counter()
    scene, robot = build_scene(
        gs,
        asset_kind=args.asset_kind,
        asset_path=str(asset_path),
        asset_variant=args.variant,
        n_envs=args.n_envs,
        sim_dt=args.sim_dt,
    )
    metrics["build_time_s"] = elapsed_since(started)

    dof_indices = resolve_dof_indices(robot)
    action = make_action_target(
        torch_module=torch_module,
        backend=args.backend,
        logical_cuda_device=args.logical_cuda_device,
        n_envs=args.n_envs,
        n_dofs=len(dof_indices),
    )
    metrics["action_device"] = tensor_device_name(action)

    selected_reset = exercise_selected_reset(
        robot,
        asset_kind=args.asset_kind,
        n_envs=args.n_envs,
        dof_indices=dof_indices,
        torch_module=torch_module,
        logical_cuda_device=args.logical_cuda_device,
    )
    metrics["selected_reset_supported"] = selected_reset["supported"]
    metrics["selected_reset_changes_only_target_envs"] = selected_reset["changes_only_target_envs"]
    metrics["selected_reset_time_s"] = selected_reset["time_s"]
    if args.n_envs >= 2 and not selected_reset["changes_only_target_envs"]:
        raise BlockedProbe(f"selected_reset_not_verified:{selected_reset['reason']}")

    warmup_started = time.perf_counter()
    run_combined_loop(
        scene,
        robot,
        action=action,
        dof_indices=dof_indices,
        policy_steps=args.warmup_policy_steps,
        decimation=args.decimation,
    )
    metrics["warmup_time_s"] = elapsed_since(warmup_started)

    action_started = time.perf_counter()
    for _ in range(args.profile_policy_steps):
        write_action_targets(robot, action, dof_indices)
    action_write_time_s = elapsed_since(action_started)
    metrics.update(action_write_rates(args.profile_policy_steps, action_write_time_s))

    state_started = time.perf_counter()
    state = None
    for _ in range(args.profile_policy_steps):
        state = read_state(robot, dof_indices)
    state_read_time_s = elapsed_since(state_started)
    metrics.update(state_read_rates(args.profile_policy_steps, state_read_time_s))
    if state is None:
        raise BlockedProbe("state_read_unavailable")

    scene_steps = args.profile_policy_steps * args.decimation
    scene_started = time.perf_counter()
    for _ in range(scene_steps):
        scene.step()
    scene_step_time_s = elapsed_since(scene_started)
    metrics.update(scene_step_rates(scene_steps, args.n_envs, scene_step_time_s))

    combined_started = time.perf_counter()
    state = run_combined_loop(
        scene,
        robot,
        action=action,
        dof_indices=dof_indices,
        policy_steps=args.profile_policy_steps,
        decimation=args.decimation,
    )
    combined_loop_time_s = elapsed_since(combined_started)
    metrics["combined_loop_time_s"] = combined_loop_time_s
    combined_rates = compute_throughput(
        policy_steps=args.profile_policy_steps,
        decimation=args.decimation,
        n_envs=args.n_envs,
        elapsed_s=combined_loop_time_s,
    )
    metrics["combined_policy_steps_per_sec"] = combined_rates["policy_steps_per_sec"]
    metrics["combined_env_policy_steps_per_sec"] = combined_rates["env_policy_steps_per_sec"]
    metrics["combined_env_sim_steps_per_sec"] = combined_rates["env_sim_steps_per_sec"]

    metrics.update(state_device_metrics(state))
    device_values = (
        metrics["action_device"],
        metrics["qpos_device"],
        metrics["dofs_pos_device"],
        metrics["dofs_vel_device"],
        metrics["root_pos_device"],
        metrics["root_quat_device"],
        metrics["root_vel_device"],
    )
    metrics["tensor_device_ok"] = tensor_devices_ok(
        device_values,
        backend=args.backend,
        logical_cuda_device=args.logical_cuda_device,
    )
    if not metrics["tensor_device_ok"]:
        raise BlockedProbe(f"tensor_device_mismatch:{device_values}")

    metrics["gpu_snapshot_after"] = gpu_snapshot()
    metrics["status"] = "ok"
    metrics["blocker"] = ""
    return metrics


def run_combined_loop(
    scene: Any,
    robot: Any,
    *,
    action: Any,
    dof_indices: tuple[int, ...],
    policy_steps: int,
    decimation: int,
) -> Any:
    state = read_state(robot, dof_indices)
    for _ in range(policy_steps):
        write_action_targets(robot, action, dof_indices)
        state = read_state(robot, dof_indices)
        for _ in range(decimation):
            scene.step()
    return state


def action_write_rates(iterations: int, elapsed_s: float) -> dict[str, float]:
    return {
        "action_write_time_s": elapsed_s,
        "action_writes_per_sec": per_second(iterations, elapsed_s),
    }


def state_read_rates(iterations: int, elapsed_s: float) -> dict[str, float]:
    return {
        "state_read_time_s": elapsed_s,
        "state_reads_per_sec": per_second(iterations, elapsed_s),
    }


def scene_step_rates(scene_steps: int, n_envs: int, elapsed_s: float) -> dict[str, float]:
    return {
        "scene_step_time_s": elapsed_s,
        "scene_steps_per_sec": per_second(scene_steps, elapsed_s),
        "env_scene_steps_per_sec": per_second(scene_steps * n_envs, elapsed_s),
    }


def per_second(count: int, elapsed_s: float) -> float:
    if elapsed_s <= 0.0:
        return 0.0
    return count / elapsed_s


def emit_ordered_metrics(metrics: Mapping[str, Any]) -> None:
    for key in PROFILE_KEYS:
        print(format_key_value(key, metrics.get(key, UNAVAILABLE)), flush=True)


def elapsed_since(started_at: float) -> float:
    return time.perf_counter() - started_at


if __name__ == "__main__":
    main()
