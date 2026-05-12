"""Run task020 standing PPO action-energy ablation candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from h200_locomotion_lab.tools import g1_ppo_smoke


DEFAULT_OUTPUT_ROOT = Path("outputs/task020/action_energy_ablation")
DEFAULT_ACTION_SCALE_MULTS = (0.10, 0.20, 0.25, 0.35)
DEFAULT_LOG_STD_INITS = (-2.0, -1.5, -1.0)

FIXED_BASE_HEIGHT_REWARD_SCALE = 0.20
FIXED_JOINT_VELOCITY_PENALTY_SCALE = 0.001
FIXED_TERMINATION_PENALTY = -1.0
FIXED_TERMINATION_HEIGHT_MIN = 0.20
FIXED_ROOT_Z = 1.20
FIXED_COMMAND_MODE = "standing"


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
        summary = run_ablation(args)
        metrics.update(summary)
        metrics["status"] = summary["status"]
        if summary["status"] != "passed":
            metrics["blocker"] = summary["blocker"]
    except Exception as exc:  # pragma: no cover - H200 failure path.
        metrics["blocker"] = f"{exc.__class__.__name__}:{exc}"
    print(json.dumps(metrics, sort_keys=True), flush=True)
    if metrics["status"] != "passed":
        raise SystemExit(1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=g1_ppo_smoke.positive_int, default=1024)
    parser.add_argument("--rollout-steps", type=g1_ppo_smoke.positive_int, default=32)
    parser.add_argument("--ppo-updates", type=g1_ppo_smoke.positive_int, default=5)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--epochs", type=g1_ppo_smoke.positive_int, default=2)
    parser.add_argument("--minibatch-size", type=g1_ppo_smoke.positive_int, default=8192)
    parser.add_argument("--lr", type=g1_ppo_smoke.positive_float, default=3e-4)
    parser.add_argument("--gamma", type=g1_ppo_smoke.positive_float, default=0.99)
    parser.add_argument("--gae-lambda", type=g1_ppo_smoke.positive_float, default=0.95)
    parser.add_argument("--clip", type=g1_ppo_smoke.positive_float, default=0.2)
    parser.add_argument("--value-coef", type=g1_ppo_smoke.positive_float, default=0.5)
    parser.add_argument("--entropy-coef", type=g1_ppo_smoke.non_negative_float, default=0.0)
    parser.add_argument("--max-grad-norm", type=g1_ppo_smoke.positive_float, default=1.0)
    parser.add_argument(
        "--action-scale-mults",
        default=",".join(str(value) for value in DEFAULT_ACTION_SCALE_MULTS),
    )
    parser.add_argument(
        "--log-std-inits",
        default=",".join(str(value) for value in DEFAULT_LOG_STD_INITS),
    )
    parser.add_argument("--height-min", type=g1_ppo_smoke.positive_float, default=0.45)
    parser.add_argument("--height-max", type=g1_ppo_smoke.positive_float, default=1.20)
    parser.add_argument(
        "--termination-height-max",
        type=g1_ppo_smoke.positive_float,
        default=1.20,
    )
    parser.add_argument(
        "--action-joint-group",
        choices=g1_ppo_smoke.ACTION_JOINT_GROUPS,
        default="all",
    )
    parser.add_argument("--base-height-target", type=g1_ppo_smoke.positive_float, default=0.85)
    parser.add_argument("--base-height-sigma", type=g1_ppo_smoke.positive_float, default=0.10)
    parser.add_argument(
        "--action-rate-penalty-scale",
        type=g1_ppo_smoke.non_negative_float,
        default=0.01,
    )
    parser.add_argument(
        "--joint-deviation-penalty-scale",
        type=g1_ppo_smoke.non_negative_float,
        default=0.05,
    )
    parser.add_argument(
        "--default-pose",
        choices=g1_ppo_smoke.G1_STANDING_RESET_POSE_NAMES,
        default=g1_ppo_smoke.DEFAULT_RESET_POSE,
    )
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--logical-cuda-device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--min-collect-env-steps-per-sec",
        type=g1_ppo_smoke.positive_float,
        default=10000.0,
    )
    parser.add_argument("--warmup-steps", type=g1_ppo_smoke.non_negative_int, default=1)
    return parser.parse_args(argv)


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    action_scale_mults = parse_float_list(args.action_scale_mults, name="action_scale_mults")
    log_std_inits = parse_float_list(args.log_std_inits, name="log_std_inits")
    require_bounded_values(
        values=action_scale_mults,
        allowed=DEFAULT_ACTION_SCALE_MULTS,
        name="action_scale_mults",
    )
    require_bounded_values(
        values=log_std_inits,
        allowed=DEFAULT_LOG_STD_INITS,
        name="log_std_inits",
    )
    run_dir = g1_ppo_smoke.resolve_run_dir(args.output_root, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        run_dir / "config.json",
        {
            "task": "task020-standing-ppo-stabilization/005-action-energy-ablation",
            "action_scale_mults": action_scale_mults,
            "log_std_inits": log_std_inits,
            "smoke_defaults": json_safe_args(args),
            "fixed_reward_reset_config": fixed_reward_reset_config(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        },
    )

    candidates = []
    for action_scale_mult in action_scale_mults:
        for log_std_init in log_std_inits:
            candidate_args = build_smoke_args(
                args=args,
                output_root=run_dir,
                action_scale_mult=action_scale_mult,
                log_std_init=log_std_init,
            )
            candidate = run_candidate(
                candidate_args=candidate_args,
                action_scale_mult=action_scale_mult,
                log_std_init=log_std_init,
            )
            candidates.append(candidate)
            append_jsonl(run_dir / "candidates.jsonl", candidate)

    selected = choose_candidate(candidates)
    summary = {
        "status": "passed" if selected is not None else "blocked",
        "blocker": "" if selected is not None else "no candidate passed standing PPO criteria",
        "run_dir": str(run_dir),
        "fixed_reward_reset_config": fixed_reward_reset_config(),
        "candidate_count": len(candidates),
        "selected_candidate": selected,
        "candidates": candidates,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def build_smoke_args(
    *,
    args: argparse.Namespace,
    output_root: Path,
    action_scale_mult: float,
    log_std_init: float,
) -> argparse.Namespace:
    return argparse.Namespace(
        n_envs=args.n_envs,
        rollout_steps=args.rollout_steps,
        ppo_updates=args.ppo_updates,
        seeds=args.seeds,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip=args.clip,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        log_std_init=log_std_init,
        height_min=args.height_min,
        height_max=args.height_max,
        termination_height_min=FIXED_TERMINATION_HEIGHT_MIN,
        termination_height_max=args.termination_height_max,
        root_z=FIXED_ROOT_Z,
        action_scale_mult=action_scale_mult,
        action_joint_group=args.action_joint_group,
        command_mode=FIXED_COMMAND_MODE,
        base_height_target=args.base_height_target,
        base_height_sigma=args.base_height_sigma,
        base_height_reward_scale=FIXED_BASE_HEIGHT_REWARD_SCALE,
        action_rate_penalty_scale=args.action_rate_penalty_scale,
        joint_velocity_penalty_scale=FIXED_JOINT_VELOCITY_PENALTY_SCALE,
        joint_deviation_penalty_scale=args.joint_deviation_penalty_scale,
        termination_penalty=FIXED_TERMINATION_PENALTY,
        default_pose=args.default_pose,
        backend=args.backend,
        physical_gpu=args.physical_gpu,
        logical_cuda_device=args.logical_cuda_device,
        output_root=output_root,
        run_id=candidate_name(action_scale_mult, log_std_init),
        min_collect_env_steps_per_sec=args.min_collect_env_steps_per_sec,
        warmup_steps=args.warmup_steps,
    )


def run_candidate(
    *,
    candidate_args: argparse.Namespace,
    action_scale_mult: float,
    log_std_init: float,
) -> dict[str, Any]:
    try:
        smoke_summary = g1_ppo_smoke.run_smoke(candidate_args)
        status = "completed"
        blocker = ""
    except Exception as exc:  # pragma: no cover - H200 failure path.
        smoke_summary = {}
        status = "failed"
        blocker = f"{exc.__class__.__name__}:{exc}"
    return summarize_candidate(
        action_scale_mult=action_scale_mult,
        log_std_init=log_std_init,
        candidate_args=candidate_args,
        status=status,
        blocker=blocker,
        smoke_summary=smoke_summary,
    )


def summarize_candidate(
    *,
    action_scale_mult: float,
    log_std_init: float,
    candidate_args: argparse.Namespace,
    status: str,
    blocker: str,
    smoke_summary: dict[str, Any],
) -> dict[str, Any]:
    seed_summaries = smoke_summary.get("seeds", [])
    final_metrics = [seed.get("final_metrics", {}) for seed in seed_summaries]
    actor_changed = all(bool(seed.get("actor_params_changed", False)) for seed in seed_summaries)
    value_changed = all(bool(seed.get("value_params_changed", False)) for seed in seed_summaries)
    all_seeds_passed = bool(smoke_summary.get("all_seeds_passed", False))
    return {
        "name": candidate_name(action_scale_mult, log_std_init),
        "status": status,
        "blocker": blocker,
        "run_dir": smoke_summary.get(
            "run_dir",
            str((Path(candidate_args.output_root) / candidate_args.run_id).resolve()),
        ),
        "action_scale_mult": action_scale_mult,
        "log_std_init": log_std_init,
        "all_seeds_passed": all_seeds_passed,
        "passed": status == "completed" and all_seeds_passed,
        "min_collect_env_policy_steps_per_sec": smoke_summary.get(
            "min_collect_env_policy_steps_per_sec",
            0.0,
        ),
        "mean_reward_mean": smoke_summary.get("mean_reward_mean", 0.0),
        "mean_final_episode_length_mean": smoke_summary.get(
            "mean_final_episode_length_mean",
            mean_metric(final_metrics, "episode_length_mean"),
        ),
        "mean_final_survival_rate": smoke_summary.get(
            "mean_final_survival_rate",
            mean_metric(final_metrics, "survival_rate"),
        ),
        "max_final_reset_rate": max_metric(final_metrics, "reset_rate"),
        "max_final_height_reset_rate": smoke_summary.get(
            "max_final_height_reset_rate",
            max_metric(final_metrics, "height_reset_rate"),
        ),
        "max_final_tilt_reset_rate": smoke_summary.get(
            "max_final_tilt_reset_rate",
            max_metric(final_metrics, "tilt_reset_rate"),
        ),
        "max_final_timeout_rate": smoke_summary.get(
            "max_final_timeout_rate",
            max_metric(final_metrics, "timeout_rate"),
        ),
        "any_final_full_env_reset_wave": bool(
            smoke_summary.get("any_final_full_env_reset_wave", False)
            or any(metric.get("full_env_reset_wave", False) for metric in final_metrics)
        ),
        "action_saturation_ratio": max_metric(final_metrics, "action_saturation_ratio"),
        "log_std_mean": mean_metric(final_metrics, "log_std_mean"),
        "log_std_min": min_metric(final_metrics, "log_std_min"),
        "log_std_max": max_metric(final_metrics, "log_std_max"),
        "approx_kl": max_metric(final_metrics, "approx_kl"),
        "clip_fraction": max_metric(final_metrics, "clip_fraction"),
        "actor_params_changed": actor_changed,
        "value_params_changed": value_changed,
        "seeds": [
            {
                "seed": seed.get("seed"),
                "passed": seed.get("passed", False),
                "actor_params_changed": seed.get("actor_params_changed", False),
                "value_params_changed": seed.get("value_params_changed", False),
                "final_reward_mean": seed.get("final_reward_mean", 0.0),
                "final_metrics": seed.get("final_metrics", {}),
            }
            for seed in seed_summaries
        ],
    }


def choose_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    viable = [
        candidate
        for candidate in candidates
        if candidate["passed"]
        and bool(candidate["actor_params_changed"])
        and bool(candidate["value_params_changed"])
        and float(candidate["mean_final_survival_rate"]) >= 1.0
        and float(candidate["max_final_reset_rate"]) == 0.0
        and not bool(candidate["any_final_full_env_reset_wave"])
    ]
    if not viable:
        return None
    return sorted(viable, key=candidate_sort_key)[0]


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    has_reset_or_wave = (
        float(candidate["max_final_reset_rate"]) > 0.0
        or float(candidate["max_final_height_reset_rate"]) > 0.0
        or float(candidate["max_final_tilt_reset_rate"]) > 0.0
        or bool(candidate["any_final_full_env_reset_wave"])
    )
    params_unchanged = not (
        bool(candidate["actor_params_changed"]) and bool(candidate["value_params_changed"])
    )
    return (
        has_reset_or_wave,
        -float(candidate["mean_final_survival_rate"]),
        -float(candidate["mean_final_episode_length_mean"]),
        -float(candidate["mean_reward_mean"]),
        float(candidate["action_saturation_ratio"]),
        params_unchanged,
        float(candidate["action_scale_mult"]),
        float(candidate["log_std_init"]),
    )


def parse_float_list(raw: str, *, name: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"{name} requires at least one value")
    if any(value <= 0.0 for value in values) and name == "action_scale_mults":
        raise argparse.ArgumentTypeError("action_scale_mults must be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} values must be unique")
    return values


def require_bounded_values(
    *,
    values: list[float],
    allowed: tuple[float, ...],
    name: str,
) -> None:
    invalid = [value for value in values if value not in allowed]
    if invalid:
        allowed_values = ",".join(str(value) for value in allowed)
        raise ValueError(f"{name} must be a subset of task005 values: {allowed_values}")


def candidate_name(action_scale_mult: float, log_std_init: float) -> str:
    scale = str(action_scale_mult).replace(".", "p").replace("-", "neg")
    log_std = str(log_std_init).replace(".", "p").replace("-", "neg")
    return f"scale_{scale}_logstd_{log_std}"


def fixed_reward_reset_config() -> dict[str, Any]:
    return {
        "base_height_reward_scale": FIXED_BASE_HEIGHT_REWARD_SCALE,
        "joint_velocity_penalty_scale": FIXED_JOINT_VELOCITY_PENALTY_SCALE,
        "termination_penalty": FIXED_TERMINATION_PENALTY,
        "termination_height_min": FIXED_TERMINATION_HEIGHT_MIN,
        "root_z": FIXED_ROOT_Z,
        "command_mode": FIXED_COMMAND_MODE,
    }


def json_safe_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, sort_keys=True) + "\n")


def mean_metric(metrics: list[dict[str, Any]], key: str) -> float:
    values = [float(metric[key]) for metric in metrics if key in metric]
    return sum(values) / len(values) if values else 0.0


def max_metric(metrics: list[dict[str, Any]], key: str) -> float:
    values = [float(metric[key]) for metric in metrics if key in metric]
    return max(values) if values else 0.0


def min_metric(metrics: list[dict[str, Any]], key: str) -> float:
    values = [float(metric[key]) for metric in metrics if key in metric]
    return min(values) if values else 0.0


if __name__ == "__main__":
    main()
