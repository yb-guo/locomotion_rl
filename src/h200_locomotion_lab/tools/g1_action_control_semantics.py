"""Run task017 standing-only action/control semantics ablations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.tools import g1_curriculum_ppo_smoke as curriculum

DEFAULT_OUTPUT_ROOT = Path("outputs/task017/action_control_semantics")
DEFAULT_STAGE_NAMES = "standing"
DEFAULT_UPDATES_PER_STAGE = 10


@dataclass(frozen=True, slots=True)
class ActionControlVariant:
    name: str
    overrides: dict[str, str]


ACTION_CONTROL_VARIANTS: tuple[ActionControlVariant, ...] = (
    ActionControlVariant(
        name="baseline",
        overrides={
            "action_scale_mult": "0.10",
            "action_joint_group": "all",
            "log_std_init": "-2.5",
        },
    ),
    ActionControlVariant(name="action_scale_0_05", overrides={"action_scale_mult": "0.05"}),
    ActionControlVariant(name="action_scale_0_03", overrides={"action_scale_mult": "0.03"}),
    ActionControlVariant(name="action_scale_0_01", overrides={"action_scale_mult": "0.01"}),
    ActionControlVariant(name="action_group_legs", overrides={"action_joint_group": "legs"}),
    ActionControlVariant(
        name="action_group_legs_waist",
        overrides={"action_joint_group": "legs_waist"},
    ),
    ActionControlVariant(name="log_std_neg3_5", overrides={"log_std_init": "-3.5"}),
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
        summary = run_ablation(args)
        metrics.update(summary)
        if summary["all_variants_completed"]:
            metrics["status"] = "ok"
        else:
            metrics["blocker"] = summary.get("blocker") or "one or more variants failed"
    except RECOVERABLE_RUNTIME_ERRORS as exc:  # pragma: no cover - H200 failure path.
        metrics["blocker"] = f"{exc.__class__.__name__}:{exc}"
    print(json.dumps(metrics, sort_keys=True), flush=True)
    if metrics["status"] != "ok":
        raise SystemExit(1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=curriculum.positive_int, default=1024)
    parser.add_argument("--rollout-steps", type=curriculum.positive_int, default=32)
    parser.add_argument(
        "--updates-per-stage",
        type=curriculum.positive_int,
        default=DEFAULT_UPDATES_PER_STAGE,
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--epochs", type=curriculum.positive_int, default=2)
    parser.add_argument("--minibatch-size", type=curriculum.positive_int, default=8192)
    parser.add_argument("--lr", type=curriculum.positive_float, default=3e-4)
    parser.add_argument("--gamma", type=curriculum.positive_float, default=0.99)
    parser.add_argument("--gae-lambda", type=curriculum.positive_float, default=0.95)
    parser.add_argument("--clip", type=curriculum.positive_float, default=0.2)
    parser.add_argument("--value-coef", type=curriculum.positive_float, default=0.5)
    parser.add_argument("--entropy-coef", type=curriculum.non_negative_float, default=0.0)
    parser.add_argument("--max-grad-norm", type=curriculum.positive_float, default=1.0)
    parser.add_argument("--log-std-init", type=float, default=curriculum.DEFAULT_LOG_STD_INIT)
    parser.add_argument("--height-min", type=curriculum.positive_float, default=0.45)
    parser.add_argument("--height-max", type=curriculum.positive_float, default=1.20)
    parser.add_argument(
        "--termination-height-min",
        type=curriculum.positive_float,
        default=curriculum.DEFAULT_TERMINATION_HEIGHT_MIN,
    )
    parser.add_argument("--termination-height-max", type=curriculum.positive_float, default=1.20)
    parser.add_argument("--root-z", type=curriculum.positive_float, default=curriculum.DEFAULT_ROOT_Z)
    parser.add_argument(
        "--action-scale-mult",
        type=curriculum.positive_float,
        default=curriculum.DEFAULT_ACTION_SCALE_MULT,
    )
    parser.add_argument(
        "--action-joint-group",
        choices=curriculum.ACTION_JOINT_GROUPS,
        default="all",
    )
    parser.add_argument("--base-height-target", type=curriculum.positive_float, default=0.85)
    parser.add_argument("--base-height-sigma", type=curriculum.positive_float, default=0.10)
    parser.add_argument(
        "--base-height-reward-scale",
        type=curriculum.non_negative_float,
        default=0.0,
    )
    parser.add_argument(
        "--action-rate-penalty-scale",
        type=curriculum.non_negative_float,
        default=0.01,
    )
    parser.add_argument(
        "--joint-deviation-penalty-scale",
        type=curriculum.non_negative_float,
        default=0.05,
    )
    parser.add_argument("--termination-penalty", type=float, default=0.0)
    parser.add_argument(
        "--default-pose",
        choices=curriculum.G1_STANDING_RESET_POSE_NAMES,
        default=curriculum.DEFAULT_RESET_POSE,
    )
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--logical-cuda-device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--stage-names", default=DEFAULT_STAGE_NAMES)
    parser.add_argument(
        "--min-collect-env-steps-per-sec",
        type=curriculum.positive_float,
        default=10000.0,
    )
    parser.add_argument(
        "--warmup-steps",
        type=curriculum.non_negative_int,
        default=curriculum.DEFAULT_WARMUP_STEPS,
    )
    return parser.parse_args(argv)


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = curriculum.resolve_run_dir(args.output_root, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    curriculum.write_json(run_dir / "config.json", build_ablation_config(args))

    variant_summaries = []
    for variant in ACTION_CONTROL_VARIANTS:
        variant_args = build_curriculum_args(
            args=args,
            variant=variant,
            output_root=run_dir,
        )
        try:
            smoke_summary, status, blocker = run_curriculum_variant(
                variant_args=variant_args,
                log_dir=run_dir / "logs" / variant.name,
            )
        except RECOVERABLE_RUNTIME_ERRORS as exc:  # pragma: no cover - H200 failure path.
            smoke_summary = {}
            status = "failed"
            blocker = f"{exc.__class__.__name__}:{exc}"
        variant_summaries.append(
            summarize_variant(
                variant=variant,
                status=status,
                blocker=blocker,
                smoke_summary=smoke_summary,
                variant_args=variant_args,
            )
        )
        if variant.name == "baseline" and not baseline_reproduced_tilt_reset(
            variant_summaries[-1]
        ):
            blocker = (
                variant_summaries[-1]["blocker"]
                or "baseline did not reproduce tilt reset waves"
            )
            summary = {
                "status": "blocked_no_baseline_repro",
                "blocker": blocker,
                "run_dir": str(run_dir),
                "variants": variant_summaries,
                "baseline_reproduced_tilt_reset": False,
                "all_variants_completed": False,
            }
            curriculum.write_json(run_dir / "summary.json", summary)
            return summary

    all_completed = all(
        variant["status"] == "completed" for variant in variant_summaries
    )
    summary = {
        "status": "passed" if all_completed else "failed",
        "blocker": "" if all_completed else "one or more variants failed",
        "run_dir": str(run_dir),
        "variants": variant_summaries,
        "baseline_reproduced_tilt_reset": True,
        "all_variants_completed": all_completed,
    }
    curriculum.write_json(run_dir / "summary.json", summary)
    return summary


def build_ablation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task": "task017-g1-action-control-semantics-diagnosis",
        "variants": [asdict(variant) for variant in ACTION_CONTROL_VARIANTS],
        "curriculum_defaults": json_safe_args(args),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }


def json_safe_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def build_curriculum_args(
    *,
    args: argparse.Namespace,
    variant: ActionControlVariant,
    output_root: Path,
) -> argparse.Namespace:
    values = {
        "n_envs": args.n_envs,
        "rollout_steps": args.rollout_steps,
        "updates_per_stage": args.updates_per_stage,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "minibatch_size": args.minibatch_size,
        "lr": args.lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip": args.clip,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "max_grad_norm": args.max_grad_norm,
        "log_std_init": args.log_std_init,
        "height_min": args.height_min,
        "height_max": args.height_max,
        "termination_height_min": args.termination_height_min,
        "termination_height_max": args.termination_height_max,
        "root_z": args.root_z,
        "action_scale_mult": args.action_scale_mult,
        "action_joint_group": args.action_joint_group,
        "base_height_target": args.base_height_target,
        "base_height_sigma": args.base_height_sigma,
        "base_height_reward_scale": args.base_height_reward_scale,
        "action_rate_penalty_scale": args.action_rate_penalty_scale,
        "joint_deviation_penalty_scale": args.joint_deviation_penalty_scale,
        "termination_penalty": args.termination_penalty,
        "default_pose": args.default_pose,
        "backend": args.backend,
        "physical_gpu": args.physical_gpu,
        "logical_cuda_device": args.logical_cuda_device,
        "output_root": output_root,
        "run_id": variant.name,
        "stage_names": args.stage_names,
        "min_collect_env_steps_per_sec": args.min_collect_env_steps_per_sec,
        "warmup_steps": args.warmup_steps,
    }
    for key, value in variant.overrides.items():
        values[key] = value
    return curriculum.parse_args(namespace_to_curriculum_argv(values))


def run_curriculum_variant(
    *,
    variant_args: argparse.Namespace,
    log_dir: Path,
) -> tuple[dict[str, Any], str, str]:
    log_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "h200_locomotion_lab.tools.g1_curriculum_ppo_smoke",
        *namespace_to_curriculum_argv(json_safe_args(variant_args)),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    (log_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (log_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")

    variant_run_dir = Path(variant_args.output_root) / variant_args.run_id
    summary_path = variant_run_dir / "summary.json"
    smoke_summary = read_json(summary_path) if summary_path.exists() else {}
    output_metrics = parse_last_json_object(completed.stdout)
    if completed.returncode == 0 and smoke_summary.get("all_seeds_passed", False):
        return smoke_summary, "completed", ""
    blocker = (
        output_metrics.get("blocker")
        or summarize_failed_seed_blocker(smoke_summary)
        or f"curriculum subprocess exited {completed.returncode}"
    )
    return smoke_summary, "failed", blocker


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_last_json_object(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def summarize_failed_seed_blocker(smoke_summary: dict[str, Any]) -> str:
    for seed_summary in smoke_summary.get("seeds", []):
        blocker = seed_summary.get("blocker", "")
        if blocker:
            return str(blocker)
    if smoke_summary:
        return "run_smoke pass criteria failed"
    return ""


def namespace_to_curriculum_argv(values: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in values.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return argv


def summarize_variant(
    *,
    variant: ActionControlVariant,
    status: str,
    blocker: str,
    smoke_summary: dict[str, Any],
    variant_args: argparse.Namespace,
) -> dict[str, Any]:
    seed_stage_summaries = [
        {
            "seed": seed_summary["seed"],
            "stage": stage["stage"],
            "first_tilt_update": stage["first_tilt_update"],
            "max_reset_count": stage["max_reset_count"],
            "mean_reset_count": stage["mean_reset_count"],
            "final_reset_count": stage["final_reset_count"],
            "max_tilt_bad_count": stage["max_tilt_bad_count"],
            "final_tilt_bad_count": stage["final_tilt_bad_count"],
            "final_termination_height_bad_count": (
                stage["final_termination_height_bad_count"]
            ),
            "max_approx_kl": stage["max_approx_kl"],
            "final_approx_kl": stage["final_approx_kl"],
            "final_entropy": stage["final_entropy"],
            "final_reward_mean": stage["final_reward_mean"],
            "min_root_height_min": stage["min_root_height_min"],
            "final_root_height_mean": stage["final_root_height_mean"],
            "final_root_height_min": stage["final_root_height_min"],
            "min_upright_mean": stage["min_upright_mean"],
            "final_upright_mean": stage["final_upright_mean"],
            "final_action_abs_mean": stage["final_action_abs_mean"],
            "max_action_abs_max": stage["max_action_abs_max"],
            "final_action_abs_max": stage["final_action_abs_max"],
            "final_action_std": stage["final_action_std"],
            "final_top_action_rms_joints": stage["final_top_action_rms_joints"],
            "min_collect_env_policy_steps_per_sec": (
                stage["min_collect_env_policy_steps_per_sec"]
            ),
        }
        for seed_summary in smoke_summary.get("seeds", [])
        for stage in seed_summary.get("stages", [])
    ]
    return {
        "name": variant.name,
        "status": status,
        "blocker": blocker,
        "overrides": variant.overrides,
        "stage_names": variant_args.stage_names,
        "run_dir": smoke_summary.get(
            "run_dir",
            str((Path(variant_args.output_root) / variant_args.run_id).resolve()),
        ),
        "all_seeds_passed": smoke_summary.get("all_seeds_passed", False),
        "min_collect_env_policy_steps_per_sec": smoke_summary.get(
            "min_collect_env_policy_steps_per_sec",
            0.0,
        ),
        "mean_reward_mean": smoke_summary.get("mean_reward_mean", 0.0),
        "seed_stage_summaries": seed_stage_summaries,
    }


def baseline_reproduced_tilt_reset(variant_summary: dict[str, Any]) -> bool:
    if variant_summary["status"] != "completed":
        return False
    return any(
        int(stage["max_tilt_bad_count"]) > 0 and int(stage["max_reset_count"]) > 0
        for stage in variant_summary["seed_stage_summaries"]
    )


if __name__ == "__main__":
    main()
