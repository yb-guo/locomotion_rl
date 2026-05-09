"""Run task014 minimal PPO smoke on the vectorized G1 Genesis env."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from h200_locomotion_lab.envs.g1_velocity_tracking_env import (
    G1VelocityTrackingConfig,
    G1VelocityTrackingVectorizedEnv,
)
from h200_locomotion_lab.envs.vectorized_genesis_backend import (
    VectorizedGenesisBackend,
    VectorizedGenesisConfig,
)
from h200_locomotion_lab.training.ppo_loop import (
    PPOConfig,
    build_actor_critic,
    collect_rollout,
    compute_gae,
    parameter_l1_sum,
    ppo_update,
    require_torch,
    tensor_device_ok,
)


PROJECT_PREFIX = Path("/root/agent_workspace/project")
DEFAULT_OUTPUT_ROOT = Path("outputs/task014/minimal_ppo_smoke")


def main() -> None:
    args = parse_args()
    metrics: dict[str, Any] = {
        "status": "failed",
        "blocker": "",
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }
    try:
        summary = run_smoke(args)
        metrics.update(summary)
        metrics["status"] = "ok"
    except Exception as exc:  # pragma: no cover - H200 failure path.
        metrics["blocker"] = f"{exc.__class__.__name__}:{exc}"
    print(json.dumps(metrics, sort_keys=True), flush=True)
    if metrics["status"] != "ok":
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=positive_int, default=1024)
    parser.add_argument("--rollout-steps", type=positive_int, default=32)
    parser.add_argument("--ppo-updates", type=positive_int, default=5)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=positive_int, default=2)
    parser.add_argument("--minibatch-size", type=positive_int, default=8192)
    parser.add_argument("--lr", type=positive_float, default=3e-4)
    parser.add_argument("--gamma", type=positive_float, default=0.99)
    parser.add_argument("--gae-lambda", type=positive_float, default=0.95)
    parser.add_argument("--clip", type=positive_float, default=0.2)
    parser.add_argument("--value-coef", type=positive_float, default=0.5)
    parser.add_argument("--entropy-coef", type=non_negative_float, default=0.0)
    parser.add_argument("--max-grad-norm", type=positive_float, default=1.0)
    parser.add_argument("--log-std-init", type=float, default=-0.5)
    parser.add_argument("--height-min", type=positive_float, default=0.45)
    parser.add_argument("--height-max", type=positive_float, default=1.20)
    parser.add_argument("--root-z", type=positive_float, default=1.10)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--logical-cuda-device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--min-collect-env-steps-per-sec", type=positive_float, default=10000.0)
    return parser.parse_args()


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    verify_cuda_isolation(
        backend=args.backend,
        physical_gpu=str(args.physical_gpu),
        logical_cuda_device=args.logical_cuda_device,
    )
    torch = require_torch()
    config = PPOConfig(
        n_envs=args.n_envs,
        rollout_steps=args.rollout_steps,
        ppo_updates=args.ppo_updates,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip=args.clip,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        log_std_init=args.log_std_init,
    )
    seeds = parse_seeds(args.seeds)
    run_dir = resolve_run_dir(args.output_root, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        run_dir / "config.json",
        {
            "ppo": asdict(config),
            "env": {
                "height_min": args.height_min,
                "height_max": args.height_max,
                "root_z": args.root_z,
            },
            "seeds": seeds,
            "backend": args.backend,
            "physical_gpu": str(args.physical_gpu),
            "logical_cuda_device": args.logical_cuda_device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        },
    )

    backend = VectorizedGenesisBackend(
        VectorizedGenesisConfig(
            n_envs=config.n_envs,
            backend=args.backend,
            logical_cuda_device=args.logical_cuda_device,
            root_qpos=(0.0, 0.0, args.root_z, 1.0, 0.0, 0.0, 0.0),
        )
    )
    env = G1VelocityTrackingVectorizedEnv(
        backend,
        G1VelocityTrackingConfig(
            height_min=args.height_min,
            height_max=args.height_max,
        ),
    )
    seed_summaries = []
    checkpoints: dict[int, dict[str, Any]] = {}
    metrics_path = run_dir / "metrics.jsonl"
    for seed in seeds:
        seed_summary, checkpoint = run_seed(
            torch=torch,
            env=env,
            config=config,
            seed=seed,
            metrics_path=metrics_path,
            min_collect_env_steps_per_sec=args.min_collect_env_steps_per_sec,
            logical_cuda_device=args.logical_cuda_device,
        )
        seed_summaries.append(seed_summary)
        checkpoints[seed] = checkpoint
    summary = {
        "status": "ok",
        "run_dir": str(run_dir),
        "seeds": seed_summaries,
        "all_seeds_passed": all(seed["passed"] for seed in seed_summaries),
        "min_collect_env_policy_steps_per_sec": min(
            seed["min_collect_env_policy_steps_per_sec"] for seed in seed_summaries
        ),
        "mean_reward_mean": sum(seed["final_reward_mean"] for seed in seed_summaries)
        / len(seed_summaries),
    }
    if not summary["all_seeds_passed"]:
        raise RuntimeError("one or more seeds failed pass criteria")
    write_json(run_dir / "summary.json", summary)
    torch.save(
        {
            "config": asdict(config),
            "seed_checkpoints": checkpoints,
            "summary": summary,
        },
        run_dir / "final_checkpoint.pt",
    )
    return summary


def run_seed(
    *,
    torch: Any,
    env: G1VelocityTrackingVectorizedEnv,
    config: PPOConfig,
    seed: int,
    metrics_path: Path,
    min_collect_env_steps_per_sec: float,
    logical_cuda_device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_actor_critic(config, device=logical_cuda_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    observation = env.reset()
    initial_actor_l1 = parameter_l1_sum(model.actor)
    initial_value_l1 = parameter_l1_sum(model.value)
    min_collect_rate = float("inf")
    final_reward_mean = 0.0
    final_metrics: dict[str, Any] = {}
    for update in range(config.ppo_updates):
        batch = collect_rollout(env, model, observation, config)
        observation = batch.next_observation
        advantages, returns = compute_gae(batch, config)
        diagnostics = ppo_update(model, optimizer, batch, advantages, returns, config)
        collect_rate = (
            batch.env_steps / batch.collect_time_s if batch.collect_time_s > 0.0 else 0.0
        )
        min_collect_rate = min(min_collect_rate, collect_rate)
        device_ok = tensor_device_ok(
            {
                "observation": batch.observations,
                "action": batch.actions,
                "reward": batch.rewards,
                "done": batch.dones,
                "value": batch.values,
                "log_prob": batch.log_probs,
            },
            logical_cuda_device,
        )
        row = {
            "seed": seed,
            "update": update,
            "env_steps": batch.env_steps * (update + 1),
            "reward_mean": batch.reward_mean,
            "done_count": batch.done_count,
            "timeout_count": batch.timeout_count,
            "fallen_count": batch.fallen_count,
            "reset_count": batch.reset_count,
            "height_bad_count": batch.height_bad_count,
            "tilt_bad_count": batch.tilt_bad_count,
            "root_height_mean": batch.root_height_mean,
            "root_height_min": batch.root_height_min,
            "upright_mean": batch.upright_mean,
            "policy_loss": diagnostics.policy_loss,
            "value_loss": diagnostics.value_loss,
            "entropy": diagnostics.entropy,
            "approx_kl": diagnostics.approx_kl,
            "clip_fraction": diagnostics.clip_fraction,
            "grad_norm": diagnostics.grad_norm,
            "collect_time_s": batch.collect_time_s,
            "collect_env_policy_steps_per_sec": collect_rate,
            "update_time_s": diagnostics.update_time_s,
            "update_samples_per_sec": diagnostics.update_samples_per_sec,
            "tensor_device_ok": device_ok,
        }
        assert_metric_row_ok(row)
        append_jsonl(metrics_path, row)
        final_reward_mean = batch.reward_mean
        final_metrics = row
    final_actor_l1 = parameter_l1_sum(model.actor)
    final_value_l1 = parameter_l1_sum(model.value)
    actor_changed = abs(final_actor_l1 - initial_actor_l1) > 1e-9
    value_changed = abs(final_value_l1 - initial_value_l1) > 1e-9
    passed = (
        actor_changed
        and value_changed
        and min_collect_rate >= min_collect_env_steps_per_sec
        and bool(final_metrics.get("tensor_device_ok", False))
    )
    summary = {
        "seed": seed,
        "passed": passed,
        "actor_params_changed": actor_changed,
        "value_params_changed": value_changed,
        "min_collect_env_policy_steps_per_sec": min_collect_rate,
        "final_reward_mean": final_reward_mean,
        "final_metrics": final_metrics,
    }
    if not passed:
        raise RuntimeError(f"seed {seed} failed pass criteria: {summary}")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "summary": summary,
    }
    return summary, checkpoint


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


def resolve_run_dir(output_root: Path, run_id: str) -> Path:
    root = output_root if output_root.is_absolute() else Path.cwd() / output_root
    run_name = run_id.strip() or time.strftime("%Y%m%d-%H%M%S")
    run_dir = (root / run_name).resolve()
    project_prefix = PROJECT_PREFIX.resolve()
    if project_prefix not in (run_dir, *run_dir.parents):
        raise RuntimeError(f"output path must stay under {project_prefix}: {run_dir}")
    return run_dir


def assert_metric_row_ok(row: dict[str, Any]) -> None:
    finite_keys = (
        "reward_mean",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
        "root_height_mean",
        "root_height_min",
        "upright_mean",
        "collect_time_s",
        "collect_env_policy_steps_per_sec",
        "update_time_s",
        "update_samples_per_sec",
    )
    for key in finite_keys:
        value = float(row[key])
        if not math_is_finite(value):
            raise ValueError(f"{key} is not finite: {value}")
    if not row["tensor_device_ok"]:
        raise ValueError("tensor_device_ok is false")


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


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


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def math_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


if __name__ == "__main__":
    main()
