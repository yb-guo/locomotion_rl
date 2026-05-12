"""Run task015 staged curriculum PPO smoke on the vectorized G1 Genesis env."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
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
    ACTION_JOINT_GROUPS,
    VectorizedGenesisBackend,
    VectorizedGenesisConfig,
)
from h200_locomotion_lab.robots import load_g1_27dof_nohand_profile
from h200_locomotion_lab.training.ppo_loop import (
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
DEFAULT_OUTPUT_ROOT = Path("outputs/task015/g1_curriculum_ppo")
DEFAULT_RESET_POSE = "tall_crouch"
DEFAULT_ROOT_Z = 1.20
DEFAULT_LOG_STD_INIT = -2.5
DEFAULT_ACTION_SCALE_MULT = 0.10
DEFAULT_TERMINATION_HEIGHT_MIN = 0.20
DEFAULT_WARMUP_STEPS = 1
DEFAULT_UPDATES_PER_STAGE = 50


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    name: str
    command_vx_min: float
    command_vx_max: float
    command_yaw_min: float
    command_yaw_max: float


CURRICULUM_STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage(
        name="standing",
        command_vx_min=0.0,
        command_vx_max=0.0,
        command_yaw_min=0.0,
        command_yaw_max=0.0,
    ),
    CurriculumStage(
        name="small_vx",
        command_vx_min=0.0,
        command_vx_max=0.25,
        command_yaw_min=0.0,
        command_yaw_max=0.0,
    ),
    CurriculumStage(
        name="small_yaw",
        command_vx_min=0.0,
        command_vx_max=0.0,
        command_yaw_min=-0.25,
        command_yaw_max=0.25,
    ),
    CurriculumStage(
        name="small_vxyaw",
        command_vx_min=0.0,
        command_vx_max=0.25,
        command_yaw_min=-0.25,
        command_yaw_max=0.25,
    ),
)


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
        if summary["all_seeds_passed"]:
            metrics["status"] = "ok"
        else:
            metrics["blocker"] = "one or more seeds failed pass criteria"
    except Exception as exc:  # pragma: no cover - H200 failure path.
        metrics["blocker"] = f"{exc.__class__.__name__}:{exc}"
    print(json.dumps(metrics, sort_keys=True), flush=True)
    if metrics["status"] != "ok":
        raise SystemExit(1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=positive_int, default=1024)
    parser.add_argument("--rollout-steps", type=positive_int, default=32)
    parser.add_argument("--updates-per-stage", type=positive_int, default=DEFAULT_UPDATES_PER_STAGE)
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
    parser.add_argument("--log-std-init", type=float, default=DEFAULT_LOG_STD_INIT)
    parser.add_argument("--height-min", type=positive_float, default=0.45)
    parser.add_argument("--height-max", type=positive_float, default=1.20)
    parser.add_argument(
        "--termination-height-min",
        type=positive_float,
        default=DEFAULT_TERMINATION_HEIGHT_MIN,
    )
    parser.add_argument("--termination-height-max", type=positive_float, default=1.20)
    parser.add_argument("--root-z", type=positive_float, default=DEFAULT_ROOT_Z)
    parser.add_argument(
        "--action-scale-mult",
        type=positive_float,
        default=DEFAULT_ACTION_SCALE_MULT,
    )
    parser.add_argument("--action-joint-group", choices=ACTION_JOINT_GROUPS, default="all")
    parser.add_argument("--base-height-target", type=positive_float, default=0.85)
    parser.add_argument("--base-height-sigma", type=positive_float, default=0.10)
    parser.add_argument("--base-height-reward-scale", type=non_negative_float, default=0.0)
    parser.add_argument("--action-rate-penalty-scale", type=non_negative_float, default=0.01)
    parser.add_argument("--joint-deviation-penalty-scale", type=non_negative_float, default=0.05)
    parser.add_argument("--termination-penalty", type=float, default=0.0)
    parser.add_argument(
        "--default-pose",
        choices=G1_STANDING_RESET_POSE_NAMES,
        default=DEFAULT_RESET_POSE,
    )
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--logical-cuda-device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--stage-names",
        default="",
        help="Comma-separated curriculum stages to run. Defaults to the full curriculum.",
    )
    parser.add_argument("--min-collect-env-steps-per-sec", type=positive_float, default=10000.0)
    parser.add_argument("--warmup-steps", type=non_negative_int, default=DEFAULT_WARMUP_STEPS)
    return parser.parse_args(argv)


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
        ppo_updates=args.updates_per_stage,
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
    stages = selected_curriculum_stages(args.stage_names)
    stage_env_configs = {
        stage.name: build_stage_env_config(args=args, stage=stage) for stage in stages
    }
    profile = load_g1_27dof_nohand_profile()
    default_pose = build_g1_standing_reset_pose_candidates(
        profile.control.default_angles_rad
    )[args.default_pose]
    run_dir = resolve_run_dir(args.output_root, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        run_dir / "config.json",
        build_run_config(
            args=args,
            ppo_config=config,
            seeds=seeds,
            stages=stages,
            default_pose=default_pose,
        ),
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
    env = G1VelocityTrackingVectorizedEnv(backend, stage_env_configs[stages[0].name])
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
            stages=stages,
            stage_env_configs=stage_env_configs,
            metrics_path=metrics_path,
            min_collect_env_steps_per_sec=args.min_collect_env_steps_per_sec,
            logical_cuda_device=args.logical_cuda_device,
            action_joint_names=profile.actuator_order,
        )
        seed_summaries.append(seed_summary)
        checkpoints[seed] = checkpoint

    completed_stage_rewards = [
        stage["final_reward_mean"]
        for seed in seed_summaries
        for stage in seed["stages"]
        if stage["status"] == "passed"
    ]
    min_collect_rates = [
        seed["min_collect_env_policy_steps_per_sec"] for seed in seed_summaries
    ]
    summary = {
        "status": "passed" if all(seed["passed"] for seed in seed_summaries) else "failed",
        "run_dir": str(run_dir),
        "stages": [asdict(stage) for stage in stages],
        "seeds": seed_summaries,
        "all_seeds_passed": all(seed["passed"] for seed in seed_summaries),
        "min_collect_env_policy_steps_per_sec": min(min_collect_rates),
        "mean_reward_mean": (
            sum(completed_stage_rewards) / len(completed_stage_rewards)
            if completed_stage_rewards
            else 0.0
        ),
    }
    write_json(run_dir / "summary.json", summary)
    torch.save(
        {
            "config": build_run_config(
                args=args,
                ppo_config=config,
                seeds=seeds,
                stages=stages,
                default_pose=default_pose,
            ),
            "seed_checkpoints": checkpoints,
            "summary": summary,
        },
        run_dir / "final_checkpoint.pt",
    )
    return summary


def curriculum_stages() -> tuple[CurriculumStage, ...]:
    return CURRICULUM_STAGES


def selected_curriculum_stages(raw_stage_names: str) -> tuple[CurriculumStage, ...]:
    requested = [name.strip() for name in raw_stage_names.split(",") if name.strip()]
    if not requested:
        return curriculum_stages()
    stage_by_name = {stage.name: stage for stage in CURRICULUM_STAGES}
    unknown = [name for name in requested if name not in stage_by_name]
    if unknown:
        valid = ",".join(stage_by_name)
        raise argparse.ArgumentTypeError(
            f"unknown curriculum stage {unknown[0]!r}; valid stages: {valid}"
        )
    if len(set(requested)) != len(requested):
        raise argparse.ArgumentTypeError("stage names must be unique")
    return tuple(stage_by_name[name] for name in requested)


def build_run_config(
    *,
    args: argparse.Namespace,
    ppo_config: PPOConfig,
    seeds: list[int],
    stages: tuple[CurriculumStage, ...],
    default_pose: Any,
) -> dict[str, Any]:
    return {
        "ppo": asdict(ppo_config),
        "curriculum": {
            "updates_per_stage": args.updates_per_stage,
            "stage_names": [stage.name for stage in stages],
            "stages": [asdict(stage) for stage in stages],
        },
        "env": {
            "height_min": args.height_min,
            "height_max": args.height_max,
            "termination_height_min": args.termination_height_min,
            "termination_height_max": args.termination_height_max,
            "root_z": args.root_z,
            "default_pose": args.default_pose,
            "default_pose_leg_values_rad": leg_value_summary(default_pose),
            "action_scale_mult": args.action_scale_mult,
            "action_joint_group": args.action_joint_group,
            "base_height_target": args.base_height_target,
            "base_height_sigma": args.base_height_sigma,
            "base_height_reward_scale": args.base_height_reward_scale,
            "action_rate_penalty_scale": args.action_rate_penalty_scale,
            "joint_deviation_penalty_scale": args.joint_deviation_penalty_scale,
            "termination_penalty": args.termination_penalty,
            "warmup_steps": args.warmup_steps,
        },
        "seeds": seeds,
        "backend": args.backend,
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }


def build_stage_env_config(
    *,
    args: argparse.Namespace,
    stage: CurriculumStage,
) -> G1VelocityTrackingConfig:
    return G1VelocityTrackingConfig(
        height_min=args.height_min,
        height_max=args.height_max,
        termination_height_min=args.termination_height_min,
        termination_height_max=args.termination_height_max,
        command_vx_min=stage.command_vx_min,
        command_vx_max=stage.command_vx_max,
        command_yaw_min=stage.command_yaw_min,
        command_yaw_max=stage.command_yaw_max,
        base_height_target=args.base_height_target,
        base_height_sigma=args.base_height_sigma,
        base_height_reward_scale=args.base_height_reward_scale,
        action_rate_penalty_scale=args.action_rate_penalty_scale,
        joint_deviation_penalty_scale=args.joint_deviation_penalty_scale,
        termination_penalty=args.termination_penalty,
    )


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
    stages: tuple[CurriculumStage, ...],
    stage_env_configs: dict[str, G1VelocityTrackingConfig],
    metrics_path: Path,
    min_collect_env_steps_per_sec: float,
    logical_cuda_device: str,
    action_joint_names: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_actor_critic(config, device=logical_cuda_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    stage_summaries: list[dict[str, Any]] = []
    min_collect_rate = float("inf")
    total_env_steps = 0
    global_update = 0
    blocker = ""

    for stage_index, stage in enumerate(stages):
        env.config = stage_env_configs[stage.name]
        observation = env.reset()
        stage_summary, total_env_steps, global_update = run_stage(
            env=env,
            model=model,
            optimizer=optimizer,
            config=config,
            seed=seed,
            stage=stage,
            stage_index=stage_index,
            observation=observation,
            metrics_path=metrics_path,
            total_env_steps=total_env_steps,
            global_update=global_update,
            min_collect_env_steps_per_sec=min_collect_env_steps_per_sec,
            logical_cuda_device=logical_cuda_device,
            action_joint_names=action_joint_names,
        )
        stage_summaries.append(stage_summary)
        if stage_summary["updates_completed"] > 0:
            min_collect_rate = min(
                min_collect_rate,
                stage_summary["min_collect_env_policy_steps_per_sec"],
            )
        if not stage_summary["passed"]:
            blocker = stage_summary["blocker"]
            stage_summaries.extend(
                skipped_stage_summary(
                    seed=seed,
                    stages=stages,
                    start_index=stage_index + 1,
                    blocker=f"upstream stage {stage.name} failed: {blocker}",
                )
            )
            break

    passed = all(stage["passed"] for stage in stage_summaries)
    completed_stages = [stage for stage in stage_summaries if stage["status"] == "passed"]
    final_reward_mean = completed_stages[-1]["final_reward_mean"] if completed_stages else 0.0
    summary = {
        "seed": seed,
        "passed": passed,
        "blocker": "" if passed else blocker,
        "stages": stage_summaries,
        "completed_stage_count": len(completed_stages),
        "global_updates_completed": global_update,
        "env_steps": total_env_steps,
        "min_collect_env_policy_steps_per_sec": (
            min_collect_rate if min_collect_rate != float("inf") else 0.0
        ),
        "final_reward_mean": final_reward_mean,
    }
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "summary": summary,
    }
    return summary, checkpoint


def run_stage(
    *,
    env: G1VelocityTrackingVectorizedEnv,
    model: Any,
    optimizer: Any,
    config: PPOConfig,
    seed: int,
    stage: CurriculumStage,
    stage_index: int,
    observation: Any,
    metrics_path: Path,
    total_env_steps: int,
    global_update: int,
    min_collect_env_steps_per_sec: float,
    logical_cuda_device: str,
    action_joint_names: tuple[str, ...],
) -> tuple[dict[str, Any], int, int]:
    initial_actor_l1 = parameter_l1_sum(model.actor)
    initial_value_l1 = parameter_l1_sum(model.value)
    min_collect_rate = float("inf")
    final_reward_mean = 0.0
    final_metrics: dict[str, Any] = {}
    stage_rows: list[dict[str, Any]] = []
    updates_completed = 0
    stage_env_steps = 0
    blocker = ""

    for stage_update in range(config.ppo_updates):
        try:
            batch = collect_rollout(env, model, observation, config)
            observation = batch.next_observation
            advantages, returns = compute_gae(batch, config)
            diagnostics = ppo_update(model, optimizer, batch, advantages, returns, config)
            collect_rate = (
                batch.env_steps / batch.collect_time_s if batch.collect_time_s > 0.0 else 0.0
            )
            min_collect_rate = min(min_collect_rate, collect_rate)
            total_env_steps += batch.env_steps
            stage_env_steps += batch.env_steps
            row = build_metric_row(
                env=env,
                batch=batch,
                diagnostics=diagnostics,
                seed=seed,
                stage=stage,
                stage_index=stage_index,
                stage_update=stage_update,
                global_update=global_update,
                env_steps=total_env_steps,
                stage_env_steps=stage_env_steps,
                collect_rate=collect_rate,
                logical_cuda_device=logical_cuda_device,
                action_joint_names=action_joint_names,
            )
            assert_metric_row_ok(row)
            append_jsonl(metrics_path, row)
            stage_rows.append(row)
            final_reward_mean = batch.reward_mean
            final_metrics = row
            updates_completed += 1
            global_update += 1
        except Exception as exc:
            blocker = f"{exc.__class__.__name__}:{exc}"
            break

    final_actor_l1 = parameter_l1_sum(model.actor)
    final_value_l1 = parameter_l1_sum(model.value)
    actor_changed = abs(final_actor_l1 - initial_actor_l1) > 1e-9
    value_changed = abs(final_value_l1 - initial_value_l1) > 1e-9
    tensor_ok = bool(final_metrics.get("tensor_device_ok", False))
    throughput_ok = min_collect_rate >= min_collect_env_steps_per_sec
    passed = (
        blocker == ""
        and updates_completed == config.ppo_updates
        and actor_changed
        and value_changed
        and throughput_ok
        and tensor_ok
    )
    failure_reasons = []
    if blocker:
        failure_reasons.append(blocker)
    if updates_completed != config.ppo_updates:
        failure_reasons.append(
            f"completed {updates_completed}/{config.ppo_updates} updates"
        )
    if not actor_changed:
        failure_reasons.append("actor params did not change")
    if not value_changed:
        failure_reasons.append("value params did not change")
    if not throughput_ok:
        failure_reasons.append(
            "collect throughput below threshold: "
            f"{min_collect_rate:.3f} < {min_collect_env_steps_per_sec:.3f}"
        )
    if not tensor_ok:
        failure_reasons.append("tensor_device_ok is false")

    stage_summary = {
        "seed": seed,
        "stage": stage.name,
        "stage_index": stage_index,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "blocker": "; ".join(failure_reasons),
        "updates_completed": updates_completed,
        "env_steps": stage_env_steps,
        "global_update_start": global_update - updates_completed,
        "global_update_end": global_update - 1 if updates_completed else None,
        "actor_params_changed": actor_changed,
        "value_params_changed": value_changed,
        "min_collect_env_policy_steps_per_sec": (
            min_collect_rate if min_collect_rate != float("inf") else 0.0
        ),
        "final_reward_mean": final_reward_mean,
        "final_metrics": final_metrics,
    }
    stage_summary.update(summarize_stage_diagnostics(stage_rows))
    return (
        stage_summary,
        total_env_steps,
        global_update,
    )


def summarize_stage_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return empty_stage_diagnostics()

    final_row = rows[-1]
    reset_counts = [int(row["reset_count"]) for row in rows]
    tilt_bad_counts = [int(row["tilt_bad_count"]) for row in rows]
    approx_kls = [float(row["approx_kl"]) for row in rows]
    root_height_mins = [float(row["root_height_min"]) for row in rows]
    upright_means = [float(row["upright_mean"]) for row in rows]
    action_abs_maxes = [float(row["action_abs_max"]) for row in rows]
    first_tilt_update = next(
        (
            int(row["stage_update"])
            for row in rows
            if int(row["tilt_bad_count"]) > 0
        ),
        None,
    )
    return {
        "first_tilt_update": first_tilt_update,
        "max_reset_count": max(reset_counts),
        "mean_reset_count": sum(reset_counts) / len(reset_counts),
        "final_reset_count": int(final_row["reset_count"]),
        "max_tilt_bad_count": max(tilt_bad_counts),
        "final_tilt_bad_count": int(final_row["tilt_bad_count"]),
        "final_termination_height_bad_count": int(
            final_row["termination_height_bad_count"]
        ),
        "max_approx_kl": max(approx_kls),
        "final_approx_kl": float(final_row["approx_kl"]),
        "final_entropy": float(final_row["entropy"]),
        "final_reward_mean": float(final_row["reward_mean"]),
        "min_root_height_min": min(root_height_mins),
        "final_root_height_mean": float(final_row["root_height_mean"]),
        "final_root_height_min": float(final_row["root_height_min"]),
        "min_upright_mean": min(upright_means),
        "final_upright_mean": float(final_row["upright_mean"]),
        "final_action_abs_mean": float(final_row["action_abs_mean"]),
        "max_action_abs_max": max(action_abs_maxes),
        "final_action_abs_max": float(final_row["action_abs_max"]),
        "final_action_std": float(final_row["action_std"]),
        "final_top_action_rms_joints": final_row["top_action_rms_joints"],
    }


def empty_stage_diagnostics() -> dict[str, Any]:
    return {
        "first_tilt_update": None,
        "max_reset_count": 0,
        "mean_reset_count": 0.0,
        "final_reset_count": 0,
        "max_tilt_bad_count": 0,
        "final_tilt_bad_count": 0,
        "final_termination_height_bad_count": 0,
        "max_approx_kl": 0.0,
        "final_approx_kl": 0.0,
        "final_entropy": 0.0,
        "final_reward_mean": 0.0,
        "min_root_height_min": 0.0,
        "final_root_height_mean": 0.0,
        "final_root_height_min": 0.0,
        "min_upright_mean": 0.0,
        "final_upright_mean": 0.0,
        "final_action_abs_mean": 0.0,
        "max_action_abs_max": 0.0,
        "final_action_abs_max": 0.0,
        "final_action_std": 0.0,
        "final_top_action_rms_joints": [],
    }


def build_metric_row(
    *,
    env: G1VelocityTrackingVectorizedEnv,
    batch: Any,
    diagnostics: Any,
    seed: int,
    stage: CurriculumStage,
    stage_index: int,
    stage_update: int,
    global_update: int,
    env_steps: int,
    stage_env_steps: int,
    collect_rate: float,
    logical_cuda_device: str,
    action_joint_names: tuple[str, ...],
) -> dict[str, Any]:
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
    env_device_ok = True
    if hasattr(env, "tensor_device_ok"):
        env_device_ok = bool(env.tensor_device_ok())
    action_stats = normalized_action_stats(
        batch.actions,
        action_joint_names=action_joint_names,
    )
    return {
        "seed": seed,
        "stage": stage.name,
        "stage_index": stage_index,
        "stage_update": stage_update,
        "global_update": global_update,
        "env_steps": env_steps,
        "stage_env_steps": stage_env_steps,
        "reward_mean": batch.reward_mean,
        "done_count": batch.done_count,
        "timeout_count": batch.timeout_count,
        "fallen_count": batch.fallen_count,
        "reset_count": batch.reset_count,
        "height_bad_count": batch.height_bad_count,
        "termination_height_bad_count": batch.termination_height_bad_count,
        "tilt_bad_count": batch.tilt_bad_count,
        "root_height_mean": batch.root_height_mean,
        "root_height_min": batch.root_height_min,
        "upright_mean": batch.upright_mean,
        **action_stats,
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
        "logical_cuda_device": logical_cuda_device,
        "observation_device": str(getattr(batch.observations, "device", "")),
        "action_device": str(getattr(batch.actions, "device", "")),
        "reward_device": str(getattr(batch.rewards, "device", "")),
        "value_device": str(getattr(batch.values, "device", "")),
        "log_prob_device": str(getattr(batch.log_probs, "device", "")),
        "tensor_device_ok": device_ok,
        "env_tensor_device_ok": env_device_ok,
    }


def skipped_stage_summary(
    *,
    seed: int,
    stages: tuple[CurriculumStage, ...],
    start_index: int,
    blocker: str,
) -> list[dict[str, Any]]:
    skipped = []
    for index, stage in enumerate(stages[start_index:], start=start_index):
        stage_summary = {
            "seed": seed,
            "stage": stage.name,
            "stage_index": index,
            "status": "skipped",
            "passed": False,
            "blocker": blocker,
            "updates_completed": 0,
            "env_steps": 0,
            "global_update_start": None,
            "global_update_end": None,
            "actor_params_changed": False,
            "value_params_changed": False,
            "min_collect_env_policy_steps_per_sec": 0.0,
            "final_reward_mean": 0.0,
            "final_metrics": {},
        }
        stage_summary.update(empty_stage_diagnostics())
        skipped.append(stage_summary)
    return skipped


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
        "action_abs_mean",
        "action_abs_max",
        "action_std",
    )
    for key in finite_keys:
        value = float(row[key])
        if not math_is_finite(value):
            raise ValueError(f"{key} is not finite: {value}")
    if not row["tensor_device_ok"]:
        raise ValueError("tensor_device_ok is false")
    if not row["env_tensor_device_ok"]:
        raise ValueError("env_tensor_device_ok is false")


def normalized_action_stats(
    actions: Any,
    *,
    action_joint_names: tuple[str, ...],
    top_k: int = 5,
) -> dict[str, Any]:
    action_abs = actions.detach().abs()
    rms_by_joint = actions.detach().square().mean(dim=(0, 1)).sqrt()
    top_count = min(top_k, int(rms_by_joint.numel()), len(action_joint_names))
    top_values, top_indices = rms_by_joint.topk(top_count)
    top_entries = [
        {
            "joint": action_joint_names[int(index.item())],
            "rms": float(value.item()),
        }
        for value, index in zip(top_values, top_indices)
    ]
    return {
        "action_abs_mean": float(action_abs.mean().item()),
        "action_abs_max": float(action_abs.max().item()),
        "action_std": float(actions.detach().std(unbiased=False).item()),
        "top_action_rms_joints": top_entries,
    }


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


def math_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


if __name__ == "__main__":
    main()
