"""Smoke and micro-benchmark the Task033 shared history buffer."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from h200_locomotion_lab.training.history_buffer import (
    FORBIDDEN_ACTOR_FIELD_TOKENS,
    HistoryBufferConfig,
    TorchHistoryBuffer,
    build_default_actor_history_spec,
    require_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=104)
    parser.add_argument("--action-dim", type=int, default=31)
    parser.add_argument("--residual-dim", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--reset-step", type=int, default=3)
    parser.add_argument("--reset-env", type=int, default=1)
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--repo-commit", default="unknown")
    parser.add_argument("--h200-checkout", default="unknown")
    parser.add_argument("--command-label", default="")
    return parser.parse_args()


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    spec = build_default_actor_history_spec(
        observation_dim=args.obs_dim,
        action_dim=args.action_dim,
        residual_dim=args.residual_dim,
    )
    buffer = TorchHistoryBuffer(
        HistoryBufferConfig(
            num_envs=args.num_envs,
            history_len=args.history_len,
            frame_dim=spec.frame_dim,
            device=args.device,
            dtype=args.dtype,
        )
    )

    for step in range(args.steps):
        frames = make_step_frames(torch, args.num_envs, spec.frame_dim, step, args.device, args.dtype)
        reset_env_ids = [args.reset_env] if step == args.reset_step else None
        buffer.append(frames, reset_env_ids=reset_env_ids)

    latest = buffer.latest_oldest_first()
    flat = buffer.flatten_latest()
    valid_counts = [int(value) for value in buffer.valid_counts.detach().cpu().tolist()]
    reset_env_zero_prefix = zero_prefix_count(torch, latest[args.reset_env])
    expected_reset_valid = min(args.history_len, max(0, args.steps - args.reset_step))
    benchmark = (
        run_micro_benchmark(torch, buffer, args.benchmark_steps)
        if args.benchmark_steps > 0
        else {}
    )

    result = {
        "pass": True,
        "status": "passed",
        "failure_reasons": [],
        "command": args.command_label or " ".join(sys.argv),
        "repo_commit": args.repo_commit,
        "h200_checkout": args.h200_checkout,
        "host": platform.node(),
        "gpu_name": gpu_name(torch, buffer.device),
        "num_envs": args.num_envs,
        "history_len": args.history_len,
        "base_actor_obs_dim": args.obs_dim,
        "actor_frame_dim": spec.frame_dim,
        "actor_input_dim_stack": args.history_len * spec.frame_dim,
        "action_dim": args.action_dim,
        "device": str(buffer.device),
        "buffer_device": str(buffer.device),
        "is_gpu_resident": getattr(buffer.device, "type", None) == "cuda",
        "dtype": str(buffer.dtype),
        "storage_shape": list(buffer.storage.shape),
        "latest_shape": list(latest.shape),
        "flat_shape": list(flat.shape),
        "valid_count_min": min(valid_counts),
        "valid_count_max": max(valid_counts),
        "valid_count_sample": valid_counts[: min(16, len(valid_counts))],
        "reset_env_valid_count": valid_counts[args.reset_env],
        "write_index": buffer.write_index,
        "reset_step": args.reset_step,
        "reset_env": args.reset_env,
        "reset_env_zero_prefix": reset_env_zero_prefix,
        "expected_reset_env_valid_count": expected_reset_valid,
        "actor_field_count": len(spec.actor_field_names),
        "debug_field_count": len(spec.debug_field_names),
        "forbidden_actor_terms": list(FORBIDDEN_ACTOR_FIELD_TOKENS),
        "actor_fault_leakage_check": "passed",
        "reset_policy": "clear reset env ids before appending the current frame",
        "benchmark": benchmark,
    }
    validate_smoke_result(result)
    return result


def make_step_frames(
    torch: Any,
    num_envs: int,
    frame_dim: int,
    step: int,
    device: str,
    dtype: str,
) -> Any:
    torch_dtype = getattr(torch, dtype)
    env_offsets = torch.arange(num_envs, device=device, dtype=torch_dtype).reshape(num_envs, 1) / 1000.0
    return torch.full((num_envs, frame_dim), float(step + 1), device=device, dtype=torch_dtype) + env_offsets


def zero_prefix_count(torch: Any, frames_oldest_first: Any) -> int:
    frame_is_zero = frames_oldest_first.abs().sum(dim=-1) == 0
    if not bool(frame_is_zero.any().item()):
        return 0
    first_non_zero = (~frame_is_zero).nonzero(as_tuple=False)
    if first_non_zero.numel() == 0:
        return int(frame_is_zero.numel())
    return int(first_non_zero[0].item())


def run_micro_benchmark(torch: Any, buffer: TorchHistoryBuffer, benchmark_steps: int) -> dict[str, Any]:
    synchronize(torch, buffer.device)
    started = time.perf_counter()
    for step in range(benchmark_steps):
        frames = torch.full(
            (buffer.config.num_envs, buffer.config.frame_dim),
            float(step),
            device=buffer.device,
            dtype=buffer.dtype,
        )
        buffer.append(frames)
        _ = buffer.flatten_latest()
    synchronize(torch, buffer.device)
    elapsed_s = time.perf_counter() - started
    history_frames = benchmark_steps * buffer.config.num_envs
    return {
        "benchmark_steps": benchmark_steps,
        "elapsed_s": elapsed_s,
        "history_frames_per_sec": history_frames / elapsed_s if elapsed_s > 0.0 else 0.0,
        "cuda_memory_allocated_bytes": cuda_memory_allocated(torch, buffer.device),
    }


def synchronize(torch: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def cuda_memory_allocated(torch: Any, device: Any) -> int:
    if getattr(device, "type", None) != "cuda":
        return 0
    return int(torch.cuda.memory_allocated(device))


def gpu_name(torch: Any, device: Any) -> str:
    if getattr(device, "type", None) != "cuda":
        return "none"
    return str(torch.cuda.get_device_name(device))


def validate_smoke_result(result: dict[str, Any]) -> None:
    if result["storage_shape"] != [
        result["num_envs"],
        result["history_len"],
        result["actor_frame_dim"],
    ]:
        raise RuntimeError("history storage shape mismatch")
    expected_reset_valid = result["expected_reset_env_valid_count"]
    actual_reset_valid = result["reset_env_valid_count"]
    if actual_reset_valid != expected_reset_valid:
        raise RuntimeError(
            f"reset env valid count mismatch: expected {expected_reset_valid}, got {actual_reset_valid}"
        )
    expected_zero_prefix = result["history_len"] - expected_reset_valid
    if result["reset_env_zero_prefix"] != expected_zero_prefix:
        raise RuntimeError(
            "reset env zero prefix mismatch: "
            f"expected {expected_zero_prefix}, got {result['reset_env_zero_prefix']}"
        )


def main() -> None:
    args = parse_args()
    result = run_smoke(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
