"""Run task014 minimal PPO smoke on the vectorized G1 Genesis env."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from h200_locomotion_lab.envs.g1_reset_poses import (
    G1_STANDING_RESET_POSE_NAMES,
    build_g1_standing_reset_pose_candidates,
    leg_value_summary,
)
from h200_locomotion_lab.envs.g1_velocity_tracking_env import (
    G1VelocityTrackingConfig,
    G1VelocityTrackingVectorizedEnv,
)
from h200_locomotion_lab.envs.vectorized_genesis_backend import (
    ACTION_JOINT_GROUPS,
    VectorizedGenesisBackend,
    VectorizedGenesisConfig,
)
from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.robots import load_g1_27dof_nohand_profile
from h200_locomotion_lab.tools import g1_ankle_roll_contact_patch as contact_patch
from h200_locomotion_lab.training.ppo_loop import (
    REWARD_COMPONENT_NAMES,
    PPOConfig,
    build_actor_critic,
    collect_rollout,
    compute_gae,
    parameter_l1_sum,
    ppo_update,
    require_torch,
    synchronize_device,
    tensor_device_ok,
)

PROJECT_PREFIX = Path("/root/agent_workspace/project")
DEFAULT_OUTPUT_ROOT = Path("outputs/task014/minimal_ppo_smoke")
DEFAULT_RESET_POSE = "tall_crouch"
DEFAULT_ROOT_Z = 1.20
COMMAND_MODES = ("vx_yaw", "standing")
ASSET_VARIANTS = ("task023_hybrid", "profile")
DEFAULT_ASSET_VARIANT = "task023_hybrid"
TASK023_HYBRID_PATCH_VARIANT = "ankle_roll_hybrid_edge_boxes_no_points"


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
    except RECOVERABLE_RUNTIME_ERRORS as exc:  # pragma: no cover - H200 failure path.
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
    parser.add_argument("--termination-height-min", type=positive_float, default=0.20)
    parser.add_argument("--termination-height-max", type=positive_float, default=1.20)
    parser.add_argument("--root-z", type=positive_float, default=DEFAULT_ROOT_Z)
    parser.add_argument("--action-scale-mult", type=positive_float, default=1.0)
    parser.add_argument("--action-joint-group", choices=ACTION_JOINT_GROUPS, default="all")
    parser.add_argument("--command-mode", choices=COMMAND_MODES, default="vx_yaw")
    parser.add_argument("--base-height-target", type=positive_float, default=0.85)
    parser.add_argument("--base-height-sigma", type=positive_float, default=0.10)
    parser.add_argument("--base-height-reward-scale", type=non_negative_float, default=0.0)
    parser.add_argument("--action-rate-penalty-scale", type=non_negative_float, default=0.01)
    parser.add_argument("--joint-velocity-penalty-scale", type=non_negative_float, default=0.0)
    parser.add_argument("--joint-deviation-penalty-scale", type=non_negative_float, default=0.05)
    parser.add_argument("--termination-penalty", type=float, default=0.0)
    parser.add_argument(
        "--default-pose",
        choices=G1_STANDING_RESET_POSE_NAMES,
        default=DEFAULT_RESET_POSE,
    )
    parser.add_argument(
        "--asset-variant",
        choices=ASSET_VARIANTS,
        default=DEFAULT_ASSET_VARIANT,
        help=(
            "Training asset selector. task023_hybrid generates the current best "
            "hybrid edge-box foot asset under the run directory; profile uses "
            "the source asset from the robot profile."
        ),
    )
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--logical-cuda-device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--min-collect-env-steps-per-sec", type=positive_float, default=10000.0)
    parser.add_argument("--warmup-steps", type=non_negative_int, default=1)
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
    base_profile = load_g1_27dof_nohand_profile()
    default_pose = build_g1_standing_reset_pose_candidates(
        base_profile.control.default_angles_rad
    )[args.default_pose]
    command_ranges = command_ranges_for_mode(args.command_mode)
    run_dir = resolve_run_dir(args.output_root, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    profile, asset_resolution = resolve_training_profile_for_asset_variant(
        base_profile,
        asset_variant=args.asset_variant,
        run_dir=run_dir,
    )
    write_json(run_dir / "asset_resolution.json", asset_resolution)
    write_json(
        run_dir / "config.json",
        {
            "ppo": asdict(config),
            "env": {
                "height_min": args.height_min,
                "height_max": args.height_max,
                "termination_height_min": args.termination_height_min,
                "termination_height_max": args.termination_height_max,
                "root_z": args.root_z,
                "default_pose": args.default_pose,
                "default_pose_leg_values_rad": leg_value_summary(default_pose),
                "asset_variant": args.asset_variant,
                "asset_path": profile.asset.path,
                "base_profile_asset_path": base_profile.asset.path,
                "asset_resolution": asset_resolution,
                "action_scale_mult": args.action_scale_mult,
                "action_joint_group": args.action_joint_group,
                "command_mode": args.command_mode,
                "command_ranges": command_ranges,
                "base_height_target": args.base_height_target,
                "base_height_sigma": args.base_height_sigma,
                "base_height_reward_scale": args.base_height_reward_scale,
                "action_rate_penalty_scale": args.action_rate_penalty_scale,
                "joint_velocity_penalty_scale": args.joint_velocity_penalty_scale,
                "joint_deviation_penalty_scale": args.joint_deviation_penalty_scale,
                "termination_penalty": args.termination_penalty,
                "warmup_steps": args.warmup_steps,
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
            default_positions_rad=default_pose,
            action_scale_mult=args.action_scale_mult,
            action_joint_group=args.action_joint_group,
        ),
        profile=profile,
    )
    env = G1VelocityTrackingVectorizedEnv(
        backend,
        G1VelocityTrackingConfig(
            height_min=args.height_min,
            height_max=args.height_max,
            termination_height_min=args.termination_height_min,
            termination_height_max=args.termination_height_max,
            command_vx_min=command_ranges["command_vx_min"],
            command_vx_max=command_ranges["command_vx_max"],
            command_yaw_min=command_ranges["command_yaw_min"],
            command_yaw_max=command_ranges["command_yaw_max"],
            base_height_target=args.base_height_target,
            base_height_sigma=args.base_height_sigma,
            base_height_reward_scale=args.base_height_reward_scale,
            action_rate_penalty_scale=args.action_rate_penalty_scale,
            joint_velocity_penalty_scale=args.joint_velocity_penalty_scale,
            joint_deviation_penalty_scale=args.joint_deviation_penalty_scale,
            termination_penalty=args.termination_penalty,
        ),
    )
    warmup_env(
        torch=torch,
        env=env,
        action_dim=profile.action_dim,
        steps=args.warmup_steps,
        logical_cuda_device=args.logical_cuda_device,
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
        "asset_variant": args.asset_variant,
        "asset_path": profile.asset.path,
        "seeds": seed_summaries,
        "all_seeds_passed": all(seed["passed"] for seed in seed_summaries),
        "min_collect_env_policy_steps_per_sec": min(
            seed["min_collect_env_policy_steps_per_sec"] for seed in seed_summaries
        ),
        "mean_reward_mean": sum(seed["final_reward_mean"] for seed in seed_summaries)
        / len(seed_summaries),
        "mean_final_episode_length_mean": mean_seed_final_metric(
            seed_summaries,
            "episode_length_mean",
        ),
        "mean_final_survival_rate": mean_seed_final_metric(
            seed_summaries,
            "survival_rate",
        ),
        "max_final_height_reset_rate": max_seed_final_metric(
            seed_summaries,
            "height_reset_rate",
        ),
        "max_final_tilt_reset_rate": max_seed_final_metric(
            seed_summaries,
            "tilt_reset_rate",
        ),
        "max_final_timeout_rate": max_seed_final_metric(
            seed_summaries,
            "timeout_rate",
        ),
        "any_final_full_env_reset_wave": any(
            bool(seed["final_metrics"].get("full_env_reset_wave", False))
            for seed in seed_summaries
        ),
    }
    summary.update(summarize_run_training_metrics(seed_summaries))
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


def resolve_training_profile_for_asset_variant(
    profile: Any,
    *,
    asset_variant: str,
    run_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    if asset_variant == "profile":
        return profile, {
            "asset_variant": asset_variant,
            "source_asset_path": profile.asset.path,
            "effective_asset_path": profile.asset.path,
            "generated": False,
        }
    if asset_variant != "task023_hybrid":
        raise ValueError(f"unknown asset_variant: {asset_variant}")

    patch_args = argparse.Namespace(
        source_asset=Path(profile.asset.path),
        output_root=run_dir / "asset_generation",
        run_id="task023_hybrid",
        target_bodies=",".join(contact_patch.DEFAULT_TARGET_BODIES),
        variants=TASK023_HYBRID_PATCH_VARIANT,
        larger_sphere_size=contact_patch.DEFAULT_LARGER_SPHERE_SIZE,
        friction=contact_patch.DEFAULT_FRICTION,
        condim=contact_patch.DEFAULT_CONDIM,
        priority=contact_patch.DEFAULT_PRIORITY,
        box_size=contact_patch.BOX_SUPPORT_SIZE,
        box_pos=contact_patch.BOX_SUPPORT_POS,
    )
    patch_summary = contact_patch.run_patch_generation(patch_args)
    if patch_summary.get("missing") or patch_summary.get("errors"):
        raise RuntimeError(
            "task023 hybrid asset generation failed: "
            f"missing={patch_summary.get('missing')} errors={patch_summary.get('errors')}"
        )
    variant_report = patch_summary["variants"][TASK023_HYBRID_PATCH_VARIANT]
    effective_asset = Path(variant_report["path"]).resolve()
    return profile_with_asset_path(profile, effective_asset), {
        "asset_variant": asset_variant,
        "patch_variant": TASK023_HYBRID_PATCH_VARIANT,
        "source_asset_path": profile.asset.path,
        "effective_asset_path": effective_asset.as_posix(),
        "generated": True,
        "patch_run_dir": patch_summary["run_dir"],
        "patch_summary_path": str(Path(patch_summary["run_dir"]) / "summary.json"),
        "missing": patch_summary.get("missing", []),
        "errors": patch_summary.get("errors", []),
    }


def profile_with_asset_path(profile: Any, asset_path: Path) -> Any:
    return replace(profile, asset=replace(profile.asset, path=asset_path.as_posix()))


def warmup_env(
    *,
    torch: Any,
    env: G1VelocityTrackingVectorizedEnv,
    action_dim: int,
    steps: int,
    logical_cuda_device: str,
) -> None:
    if steps <= 0:
        return
    observation = env.reset()
    synchronize_device(getattr(observation, "device", None))
    zero_action = torch.zeros((env.n_envs, action_dim), device=logical_cuda_device)
    for _ in range(steps):
        transition = env.step(zero_action)
        observation = transition.observation
    synchronize_device(getattr(observation, "device", None))
    env.reset()


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
    training_rows: list[dict[str, Any]] = []
    for update in range(config.ppo_updates):
        batch = collect_rollout(env, model, observation, config)
        observation = batch.next_observation
        advantages, returns = compute_gae(batch, config)
        profile_stats = rollout_profile_stats(
            batch=batch,
            advantages=advantages,
            returns=returns,
            model=model,
            env_config=env.config,
            action_saturation_threshold=0.95,
        )
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
            "termination_height_bad_count": batch.termination_height_bad_count,
            "tilt_bad_count": batch.tilt_bad_count,
            "height_reset_count": batch.height_reset_count,
            "tilt_reset_count": batch.tilt_reset_count,
            "reset_rate": rate(batch.reset_count, batch.env_steps),
            "height_reset_rate": rate(batch.height_reset_count, batch.env_steps),
            "tilt_reset_rate": rate(batch.tilt_reset_count, batch.env_steps),
            "timeout_rate": rate(batch.timeout_count, batch.env_steps),
            "survival_rate": 1.0 - rate(batch.reset_count, batch.env_steps),
            "full_env_reset_wave": batch.full_env_reset_wave,
            "full_env_reset_wave_count": batch.full_env_reset_wave_count,
            "episode_length_mean": batch.episode_length_mean,
            "episode_length_min": batch.episode_length_min,
            "episode_length_max": batch.episode_length_max,
            "completed_episode_length_mean": batch.completed_episode_length_mean,
            "completed_episode_count": batch.completed_episode_count,
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
        row.update(profile_stats)
        assert_metric_row_ok(row)
        append_jsonl(metrics_path, row)
        training_rows.append(row)
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
    summary.update(summarize_seed_training_metrics(training_rows))
    if not passed:
        raise RuntimeError(f"seed {seed} failed pass criteria: {summary}")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "summary": summary,
    }
    return summary, checkpoint


def summarize_seed_training_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full_env_reset_wave_updates = [
        int(row["update"]) for row in rows if bool(row.get("full_env_reset_wave", False))
    ]
    full_env_reset_wave_count = sum(
        int(row.get("full_env_reset_wave_count", 0)) for row in rows
    )
    return {
        "any_training_full_env_reset_wave": bool(full_env_reset_wave_updates),
        "training_full_env_reset_wave_updates": full_env_reset_wave_updates,
        "training_full_env_reset_wave_count": full_env_reset_wave_count,
        "max_training_reset_rate": max_row_metric(rows, "reset_rate"),
        "max_training_tilt_reset_rate": max_row_metric(rows, "tilt_reset_rate"),
        "max_training_height_reset_rate": max_row_metric(rows, "height_reset_rate"),
        "max_training_episode_length_mean": max_row_metric(rows, "episode_length_mean"),
        "max_training_reward_mean": max_row_metric(rows, "reward_mean"),
    }


def summarize_run_training_metrics(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "any_training_full_env_reset_wave": any(
            bool(seed.get("any_training_full_env_reset_wave", False))
            for seed in seed_summaries
        ),
        "training_full_env_reset_wave_seed_count": sum(
            1
            for seed in seed_summaries
            if bool(seed.get("any_training_full_env_reset_wave", False))
        ),
        "training_full_env_reset_wave_count": sum(
            int(seed.get("training_full_env_reset_wave_count", 0))
            for seed in seed_summaries
        ),
        "max_training_reset_rate": max_seed_metric(
            seed_summaries,
            "max_training_reset_rate",
        ),
        "max_training_tilt_reset_rate": max_seed_metric(
            seed_summaries,
            "max_training_tilt_reset_rate",
        ),
        "max_training_height_reset_rate": max_seed_metric(
            seed_summaries,
            "max_training_height_reset_rate",
        ),
        "max_training_episode_length_mean": max_seed_metric(
            seed_summaries,
            "max_training_episode_length_mean",
        ),
        "max_training_reward_mean": max_seed_metric(
            seed_summaries,
            "max_training_reward_mean",
        ),
    }


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
        "reset_rate",
        "height_reset_rate",
        "tilt_reset_rate",
        "timeout_rate",
        "survival_rate",
        "episode_length_mean",
        "episode_length_min",
        "episode_length_max",
        "completed_episode_length_mean",
        "collect_time_s",
        "collect_env_policy_steps_per_sec",
        "update_time_s",
        "update_samples_per_sec",
        "observation_mean",
        "observation_std",
        "observation_min",
        "observation_max",
        "action_mean",
        "action_std",
        "action_min",
        "action_max",
        "action_saturation_ratio",
        "reward_std",
        "reward_min",
        "reward_max",
        "value_prediction_mean",
        "value_prediction_std",
        "value_prediction_min",
        "value_prediction_max",
        "return_mean",
        "return_std",
        "return_min",
        "return_max",
        "advantage_mean",
        "advantage_std",
        "advantage_min",
        "advantage_max",
        "log_std_mean",
        "log_std_min",
        "log_std_max",
    )
    for key in finite_keys:
        value = float(row[key])
        if not math_is_finite(value):
            raise ValueError(f"{key} is not finite: {value}")
    for key, value in row.items():
        if (
            key.startswith(("reward_component_", "reward_contribution_"))
        ) and not math_is_finite(float(value)):
            raise ValueError(f"{key} is not finite: {value}")
    if not row["tensor_device_ok"]:
        raise ValueError("tensor_device_ok is false")


def rollout_profile_stats(
    *,
    batch: Any,
    advantages: Any,
    returns: Any,
    model: Any,
    env_config: Any,
    action_saturation_threshold: float,
) -> dict[str, float]:
    stats = {}
    stats.update(tensor_profile_stats(batch.observations, "observation"))
    stats.update(tensor_profile_stats(batch.actions, "action"))
    stats.update(tensor_profile_stats(batch.rewards, "reward"))
    stats.update(tensor_profile_stats(batch.values, "value_prediction"))
    stats.update(tensor_profile_stats(returns, "return"))
    stats.update(tensor_profile_stats(advantages, "advantage"))
    stats["action_saturation_ratio"] = action_saturation_ratio(
        batch.actions,
        threshold=action_saturation_threshold,
    )
    log_std = model.log_std.detach()
    stats["log_std_mean"] = float(log_std.float().mean().item())
    stats["log_std_min"] = float(log_std.float().min().item())
    stats["log_std_max"] = float(log_std.float().max().item())
    for name in REWARD_COMPONENT_NAMES:
        if name in batch.reward_component_means:
            stats[f"reward_component_{name}_mean"] = float(batch.reward_component_means[name])
    stats.update(reward_contribution_stats(batch.reward_component_means, env_config))
    return stats


def reward_contribution_stats(
    component_means: dict[str, float],
    env_config: Any,
) -> dict[str, float]:
    stats = {
        "reward_contribution_alive_mean": float(getattr(env_config, "alive_reward", 0.0)),
    }
    scale_map = {
        "tracking_lin_vel": ("lin_vel_reward_scale", 1.0),
        "tracking_yaw_rate": ("yaw_rate_reward_scale", 1.0),
        "upright": ("upright_reward_scale", 1.0),
        "tracking_base_height": ("base_height_reward_scale", 1.0),
        "action_rate_penalty": ("action_rate_penalty_scale", -1.0),
        "joint_velocity_penalty": ("joint_velocity_penalty_scale", -1.0),
        "joint_deviation_penalty": ("joint_deviation_penalty_scale", -1.0),
    }
    for name, (scale_attr, sign) in scale_map.items():
        if name not in component_means:
            continue
        scale = float(getattr(env_config, scale_attr, 0.0))
        stats[f"reward_contribution_{name}_mean"] = (
            sign * scale * float(component_means[name])
        )
    if "termination_penalty" in component_means:
        stats["reward_contribution_termination_penalty_mean"] = float(
            component_means["termination_penalty"]
        )
    return stats


def tensor_profile_stats(tensor: Any, prefix: str) -> dict[str, float]:
    values = tensor.detach().float().reshape(-1)
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()),
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
    }


def action_saturation_ratio(actions: Any, *, threshold: float) -> float:
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    saturated = actions.detach().abs() >= threshold
    return float(saturated.float().mean().item())


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def command_ranges_for_mode(mode: str) -> dict[str, float]:
    if mode == "vx_yaw":
        return {
            "command_vx_min": 0.0,
            "command_vx_max": 0.8,
            "command_yaw_min": -0.5,
            "command_yaw_max": 0.5,
        }
    if mode == "standing":
        return {
            "command_vx_min": 0.0,
            "command_vx_max": 0.0,
            "command_yaw_min": 0.0,
            "command_yaw_max": 0.0,
        }
    raise ValueError(f"unknown command mode: {mode}")


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


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
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


def rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count) / float(total)


def mean_seed_final_metric(seed_summaries: list[dict[str, Any]], key: str) -> float:
    if not seed_summaries:
        return 0.0
    return sum(float(seed["final_metrics"].get(key, 0.0)) for seed in seed_summaries) / len(
        seed_summaries
    )


def max_seed_final_metric(seed_summaries: list[dict[str, Any]], key: str) -> float:
    if not seed_summaries:
        return 0.0
    return max(float(seed["final_metrics"].get(key, 0.0)) for seed in seed_summaries)


def max_seed_metric(seed_summaries: list[dict[str, Any]], key: str) -> float:
    if not seed_summaries:
        return 0.0
    return max(float(seed.get(key, 0.0)) for seed in seed_summaries)


def max_row_metric(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return max(float(row.get(key, 0.0)) for row in rows)


def math_is_finite(value: float) -> bool:
    return math.isfinite(value)


if __name__ == "__main__":
    main()
