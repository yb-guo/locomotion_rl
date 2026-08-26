"""Create a Task041 true-TXL warmstart checkpoint from AdaptK160."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.tools.task038_true_txl_runner_smoke_probe import (
    DEFAULT_EXPECTED_ACTION_DIM,
    _load_env_cfg,
)
from h200_locomotion_lab.tools.task040_sequence_txl_ppo_update_smoke import (
    _install_ipython_display_stub,
    _install_wandb_stub,
    _install_wcwidth_stub,
)
from h200_locomotion_lab.tools.task041_sequence_txl_clean_train import (
    DEFAULT_TASK,
    mutate_agent_cfg_for_task041_train,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a shape-complete Task041 Task038TrueTxlMemoryModel checkpoint "
            "from a Task037 AdaptK160 clean prior. This is warmstart provenance "
            "only; eval quality must be proven separately."
        )
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4100301)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-latent-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=DEFAULT_EXPECTED_ACTION_DIM)
    parser.add_argument("--adaptation-hidden-dim", type=int, default=128)
    parser.set_defaults(base_obs_passthrough=True, adaptation_warmstart=True)
    parser.add_argument("--base-obs-passthrough", dest="base_obs_passthrough", action="store_true")
    parser.add_argument("--no-base-obs-passthrough", dest="base_obs_passthrough", action="store_false")
    parser.add_argument("--adaptation-warmstart", dest="adaptation_warmstart", action="store_true")
    parser.add_argument("--no-adaptation-warmstart", dest="adaptation_warmstart", action="store_false")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    preflight_args(args)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("WANDB_DISABLED", "true")
    _install_ipython_display_stub()
    _install_wandb_stub()
    _install_wcwidth_stub()

    import mjlab.tasks as _mjlab_tasks
    import src.tasks as _project_tasks

    del _mjlab_tasks, _project_tasks  # Imports register task packages by side effect.
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    from h200_locomotion_lab.training.rsl_history_wrapper import (
        migrate_adaptk160_to_task041_true_txl_checkpoint,
    )

    configure_torch_backends()
    torch.set_grad_enabled(False)

    env_cfg = _load_env_cfg(load_env_cfg, args.task)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.seed = args.seed
    if hasattr(getattr(env_cfg, "scene", None), "num_envs"):
        env_cfg.scene.num_envs = args.num_envs
    train_cfg = mutate_agent_cfg_for_task041_train(
        asdict(agent_cfg),
        rollout_steps=2,
        iterations=1,
        save_interval=1000000,
        seed=args.seed,
        num_mini_batches=1,
        num_learning_epochs=1,
        experiment_name="task041_adaptk160_true_txl_warmstart",
        run_name="task041_adaptk160_true_txl_warmstart_template",
        memory_latent_dim=args.memory_latent_dim,
        action_dim=args.action_dim,
        base_obs_passthrough=args.base_obs_passthrough,
        adaptation_warmstart=args.adaptation_warmstart,
        adaptation_hidden_dim=args.adaptation_hidden_dim,
    )

    outer_env = None
    try:
        base = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        outer_env = RslRlVecEnvWrapper(base, clip_actions=getattr(agent_cfg, "clip_actions", None))
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        runner = runner_cls(
            outer_env,
            train_cfg,
            log_dir=str(args.target_checkpoint.expanduser().resolve().parent / "template_logs"),
            device=args.device,
        )
        source = torch.load(args.source_checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
        migrated, report = migrate_adaptk160_to_task041_true_txl_checkpoint(
            source,
            target_actor_state=runner.alg.actor.state_dict(),
            target_critic_state=runner.alg.critic.state_dict(),
        )
        target = args.target_checkpoint.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(migrated, target)
        runner.load(
            str(target),
            load_cfg={
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            strict=True,
            map_location=args.device,
        )
        summary = {
            "schema": "task041_adaptk160_true_txl_warmstart_v1",
            "task": args.task,
            "command": list(sys.argv),
            "source_checkpoint": str(args.source_checkpoint.expanduser().resolve()),
            "target_checkpoint": str(target),
            "target_checkpoint_exists": target.exists(),
            "num_envs": args.num_envs,
            "seed": args.seed,
            "device": args.device,
            "memory_latent_dim": args.memory_latent_dim,
            "base_obs_passthrough": args.base_obs_passthrough,
            "adaptation_warmstart": args.adaptation_warmstart,
            "actor_model_class": type(runner.alg.actor).__name__,
            "algorithm_class": type(getattr(runner, "alg", None)).__name__,
            "migration_report": report,
            "warmstart_pipeline_pass": False,
            "quality_gate_pass": False,
            "pass": False,
            "quality_claim": False,
            "training_claim": False,
            "eval_claim": False,
            "reproduction_claim": False,
            "superiority_claim": False,
            "diagnostic_note": (
                "Warmstart checkpoint construction only. Clean-gait quality is "
                "proven only by task041_sequence_txl_clean_eval."
            ),
        }
        summary["warmstart_pipeline_pass"], summary["failure_reasons"] = evaluate_pipeline_pass(summary)
        return summary
    finally:
        if outer_env is not None and hasattr(outer_env, "close"):
            outer_env.close()


def preflight_args(args: argparse.Namespace) -> None:
    reasons: list[str] = []
    if args.task != DEFAULT_TASK:
        reasons.append("task_not_task041_sequence_txl_clean_train")
    if not args.source_checkpoint.expanduser().exists():
        reasons.append("source_checkpoint_missing")
    if int(args.num_envs) <= 0:
        reasons.append("num_envs_not_positive")
    if int(args.memory_latent_dim) <= 0:
        reasons.append("memory_latent_dim_not_positive")
    if int(args.action_dim) <= 0:
        reasons.append("action_dim_not_positive")
    if int(args.adaptation_hidden_dim) <= 0:
        reasons.append("adaptation_hidden_dim_not_positive")
    if reasons:
        raise ValueError(", ".join(reasons))


def evaluate_pipeline_pass(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary.get("actor_model_class") != "Task038TrueTxlMemoryModel":
        reasons.append("actor_model_class_mismatch")
    if not summary.get("target_checkpoint_exists"):
        reasons.append("target_checkpoint_missing")
    report = summary.get("migration_report") or {}
    if int(report.get("actor_copied_key_count") or 0) <= 0:
        reasons.append("no_actor_keys_copied")
    copied = set(report.get("actor_copied_keys") or [])
    partial = set(report.get("actor_partial_keys") or [])
    for key in (
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
        "mlp.0.weight",
        "mlp.0.bias",
    ):
        if key not in copied and key not in partial:
            reasons.append(f"missing_copied_actor_key:{key}")
    if (
        summary.get("quality_claim") is not False
        or summary.get("training_claim") is not False
        or summary.get("eval_claim") is not False
        or summary.get("reproduction_claim") is not False
        or summary.get("superiority_claim") is not False
    ):
        reasons.append("claim_boundary_violation")
    return not reasons, reasons


def write_json_summary(path: Path, summary: dict[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary["json_path"] = str(output)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        summary = run(args)
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        summary = {
            "schema": "task041_adaptk160_true_txl_warmstart_v1",
            "task": getattr(args, "task", DEFAULT_TASK),
            "command": list(sys.argv),
            "source_checkpoint": str(getattr(args, "source_checkpoint", "")),
            "target_checkpoint": str(getattr(args, "target_checkpoint", "")),
            "warmstart_pipeline_pass": False,
            "quality_gate_pass": False,
            "pass": False,
            "failure_reasons": ["warmstart_exception"],
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "quality_claim": False,
            "training_claim": False,
            "eval_claim": False,
            "reproduction_claim": False,
            "superiority_claim": False,
        }
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
