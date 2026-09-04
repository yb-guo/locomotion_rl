"""Train Task041 sequence-aware true-TXL on the clean G1-like variant."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.tools.task038_true_txl_ppo_update_smoke import (
    PreflightError,
    collect_post_learn_diagnostics,
    summarize_log_dir_files,
)
from h200_locomotion_lab.tools.task038_true_txl_runner_smoke_probe import (
    DEFAULT_EXPECTED_ACTION_DIM,
    DEFAULT_EXPECTED_ACTOR_MODEL_CLASS,
    DEFAULT_EXPECTED_RUNNER_CLS,
    TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID,
    _action_dim,
    _actor_model,
    _agent_cfg_as_dict,
    _load_env_cfg,
    _set_if_present,
    _total_action_dim,
)
from h200_locomotion_lab.tools.task040_sequence_txl_ppo_update_smoke import (
    DEFAULT_EXPECTED_ALGORITHM_CLASS,
    TASK040_ALGORITHM_CLASS,
    _algorithm_debug_snapshot,
    _install_ipython_display_stub,
    _install_wandb_stub,
    _install_wcwidth_stub,
)

DEFAULT_TASK = TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID
DEFAULT_LOG_DIR = Path("outputs/task041/sequence_txl_clean_train/logs")
DEFAULT_OUTPUT_JSON = Path("outputs/task041/sequence_txl_clean_train/train_summary.json")
ACTOR_TRAINABLE_SCOPES = (
    "all",
    "txl_residual_only",
    "txl_residual_and_mlp_memory_input",
    "memory_output_projection_only",
)
TXL_RESIDUAL_PARAMETER_PREFIXES = (
    "token_projection",
    "position_embedding",
    "attention_layers",
    "norm_layers",
    "memory_output_projection",
)
MEMORY_OUTPUT_PROJECTION_PARAMETER_PREFIXES = ("memory_output_projection",)


def parse_args(argv: list[str] | None = None, *, description: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description
        or (
            "Train Task041 sequence-aware true-TXL on the Task038 clean train "
            "variant. This records training pipeline evidence only; eval quality "
            "must be proven by task041_sequence_txl_clean_eval."
        )
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=4100101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-mini-batches", type=int, default=4)
    parser.add_argument("--num-learning-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--desired-kl", type=float)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--experiment-name", default="task041_sequence_txl_clean_train")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument(
        "--actor-trainable-scope",
        choices=ACTOR_TRAINABLE_SCOPES,
        default="all",
        help=(
            "Limit actor parameters updated by PPO. 'all' preserves the Task041 "
            "behavior; 'txl_residual_and_mlp_memory_input' keeps the base MLP "
            "prior frozen while training the memory-latent input columns."
        ),
    )
    parser.add_argument("--memory-latent-dim", type=int, default=32)
    parser.add_argument("--memory-latent-scale", type=float, default=1.0)
    parser.add_argument("--base-obs-passthrough-scale", type=float, default=1.0)
    parser.add_argument("--adaptation-warmstart-scale", type=float, default=1.0)
    parser.add_argument("--task044-fault-aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--task044-fault-aux-num-classes", type=int, default=0)
    parser.add_argument("--task044-fault-aux-max-trial-step", type=int, default=-1)
    parser.add_argument("--task044-fault-aux-min-trial-index", type=int, default=0)
    parser.add_argument(
        "--task046-post-reset-recovery-reward",
        action="store_true",
        help="Enable final-trial post-reset recovery reward shaping for Task046.",
    )
    parser.add_argument("--task046-final-trial-index", type=int, default=2)
    parser.add_argument("--task046-recovery-window-steps", type=int, default=50)
    parser.add_argument("--task046-tail-window-steps", type=int, default=50)
    parser.add_argument("--task046-early-velocity-weight", type=float, default=0.0)
    parser.add_argument("--task046-tail-velocity-weight", type=float, default=0.0)
    parser.add_argument("--task046-orientation-weight", type=float, default=0.0)
    parser.add_argument("--task046-root-height-weight", type=float, default=0.0)
    parser.add_argument("--task046-min-root-z", type=float, default=0.70)
    parser.add_argument(
        "--task046-retry-context",
        action="store_true",
        help="Append retry/reset context features to actor observations for Task046.",
    )
    parser.add_argument("--task046-retry-context-num-trials", type=int, default=3)
    parser.add_argument("--task046-retry-context-final-trial-index", type=int, default=2)
    parser.add_argument("--task046-retry-context-step-window-steps", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=DEFAULT_EXPECTED_ACTION_DIM)
    parser.add_argument("--adaptation-hidden-dim", type=int, default=128)
    parser.set_defaults(base_obs_passthrough=True, adaptation_warmstart=True)
    parser.add_argument("--base-obs-passthrough", dest="base_obs_passthrough", action="store_true")
    parser.add_argument("--no-base-obs-passthrough", dest="base_obs_passthrough", action="store_false")
    parser.add_argument("--adaptation-warmstart", dest="adaptation_warmstart", action="store_true")
    parser.add_argument("--no-adaptation-warmstart", dest="adaptation_warmstart", action="store_false")
    parser.add_argument("--expected-action-dim", type=int, default=DEFAULT_EXPECTED_ACTION_DIM)
    parser.add_argument("--expected-runner-cls", default=DEFAULT_EXPECTED_RUNNER_CLS)
    parser.add_argument(
        "--expected-actor-model-class",
        default=DEFAULT_EXPECTED_ACTOR_MODEL_CLASS,
    )
    parser.add_argument(
        "--expected-algorithm-class",
        default=DEFAULT_EXPECTED_ALGORITHM_CLASS,
    )
    return parser.parse_args(argv)


def preflight_args(args: argparse.Namespace) -> None:
    reasons: list[str] = []
    if args.task != DEFAULT_TASK:
        reasons.append("task_not_task041_sequence_txl_clean_train")
    if int(args.num_envs) <= 0:
        reasons.append("num_envs_not_positive")
    if int(args.rollout_steps) <= 0:
        reasons.append("rollout_steps_not_positive")
    if int(args.iterations) <= 0:
        reasons.append("iterations_not_positive")
    if int(args.save_interval) <= 0:
        reasons.append("save_interval_not_positive")
    if int(args.num_mini_batches) <= 0:
        reasons.append("num_mini_batches_not_positive")
    elif int(args.num_envs) % int(args.num_mini_batches) != 0:
        reasons.append("num_envs_not_divisible_by_num_mini_batches")
    if args.num_learning_epochs is not None and int(args.num_learning_epochs) <= 0:
        reasons.append("num_learning_epochs_not_positive")
    if args.learning_rate is not None and float(args.learning_rate) <= 0.0:
        reasons.append("learning_rate_not_positive")
    if args.desired_kl is not None and float(args.desired_kl) <= 0.0:
        reasons.append("desired_kl_not_positive")
    if int(args.memory_latent_dim) <= 0:
        reasons.append("memory_latent_dim_not_positive")
    if float(args.memory_latent_scale) <= 0.0:
        reasons.append("memory_latent_scale_not_positive")
    if not 0.0 <= float(args.base_obs_passthrough_scale) <= 1.0:
        reasons.append("base_obs_passthrough_scale_out_of_range")
    if not 0.0 <= float(args.adaptation_warmstart_scale) <= 1.0:
        reasons.append("adaptation_warmstart_scale_out_of_range")
    if float(args.task044_fault_aux_loss_weight) < 0.0:
        reasons.append("task044_fault_aux_loss_weight_negative")
    if int(args.task044_fault_aux_num_classes) < 0:
        reasons.append("task044_fault_aux_num_classes_negative")
    if int(args.task044_fault_aux_max_trial_step) < -1:
        reasons.append("task044_fault_aux_max_trial_step_less_than_negative_one")
    if int(args.task044_fault_aux_min_trial_index) < 0:
        reasons.append("task044_fault_aux_min_trial_index_negative")
    if float(args.task044_fault_aux_loss_weight) > 0.0 and int(args.task044_fault_aux_num_classes) <= 1:
        reasons.append("task044_fault_aux_num_classes_too_small")
    if int(args.task046_final_trial_index) < 0:
        reasons.append("task046_final_trial_index_negative")
    if int(args.task046_recovery_window_steps) <= 0:
        reasons.append("task046_recovery_window_steps_not_positive")
    if int(args.task046_tail_window_steps) < 0:
        reasons.append("task046_tail_window_steps_negative")
    for attr in (
        "task046_early_velocity_weight",
        "task046_tail_velocity_weight",
        "task046_orientation_weight",
        "task046_root_height_weight",
    ):
        if float(getattr(args, attr)) < 0.0:
            reasons.append(f"{attr}_negative")
    if float(args.task046_min_root_z) <= 0.0:
        reasons.append("task046_min_root_z_not_positive")
    if int(args.task046_retry_context_num_trials) <= 0:
        reasons.append("task046_retry_context_num_trials_not_positive")
    if int(args.task046_retry_context_final_trial_index) < 0:
        reasons.append("task046_retry_context_final_trial_index_negative")
    if int(args.task046_retry_context_step_window_steps) <= 0:
        reasons.append("task046_retry_context_step_window_steps_not_positive")
    if int(args.action_dim) <= 0:
        reasons.append("action_dim_not_positive")
    if int(args.adaptation_hidden_dim) <= 0:
        reasons.append("adaptation_hidden_dim_not_positive")
    if int(args.expected_action_dim) <= 0:
        reasons.append("expected_action_dim_not_positive")
    if args.resume_checkpoint and not Path(args.resume_checkpoint).expanduser().exists():
        reasons.append("resume_checkpoint_missing")
    if reasons:
        raise PreflightError(reasons)


def run_train(args: argparse.Namespace) -> dict[str, Any]:
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

    configure_torch_backends()
    torch.set_grad_enabled(True)

    env_cfg = _load_env_cfg(load_env_cfg, args.task)
    agent_cfg = load_rl_cfg(args.task)
    _set_if_present(env_cfg, "seed", args.seed)
    if hasattr(getattr(env_cfg, "scene", None), "num_envs"):
        env_cfg.scene.num_envs = args.num_envs

    train_cfg = mutate_agent_cfg_for_task041_train(
        _agent_cfg_as_dict(agent_cfg),
        rollout_steps=args.rollout_steps,
        iterations=args.iterations,
        save_interval=args.save_interval,
        seed=args.seed,
        num_mini_batches=args.num_mini_batches,
        num_learning_epochs=args.num_learning_epochs,
        learning_rate=args.learning_rate,
        desired_kl=args.desired_kl,
        experiment_name=args.experiment_name,
        run_name=args.run_name or _default_run_name(args),
        memory_latent_dim=args.memory_latent_dim,
        memory_latent_scale=args.memory_latent_scale,
        base_obs_passthrough_scale=args.base_obs_passthrough_scale,
        adaptation_warmstart_scale=args.adaptation_warmstart_scale,
        task044_fault_aux_loss_weight=args.task044_fault_aux_loss_weight,
        task044_fault_aux_num_classes=args.task044_fault_aux_num_classes,
        task044_fault_aux_max_trial_step=args.task044_fault_aux_max_trial_step,
        task044_fault_aux_min_trial_index=args.task044_fault_aux_min_trial_index,
        task046_post_reset_recovery_reward=args.task046_post_reset_recovery_reward,
        task046_final_trial_index=args.task046_final_trial_index,
        task046_recovery_window_steps=args.task046_recovery_window_steps,
        task046_tail_window_steps=args.task046_tail_window_steps,
        task046_early_velocity_weight=args.task046_early_velocity_weight,
        task046_tail_velocity_weight=args.task046_tail_velocity_weight,
        task046_orientation_weight=args.task046_orientation_weight,
        task046_root_height_weight=args.task046_root_height_weight,
        task046_min_root_z=args.task046_min_root_z,
        task046_retry_context=args.task046_retry_context,
        task046_retry_context_num_trials=args.task046_retry_context_num_trials,
        task046_retry_context_final_trial_index=args.task046_retry_context_final_trial_index,
        task046_retry_context_step_window_steps=args.task046_retry_context_step_window_steps,
        action_dim=args.action_dim,
        base_obs_passthrough=args.base_obs_passthrough,
        adaptation_warmstart=args.adaptation_warmstart,
        adaptation_hidden_dim=args.adaptation_hidden_dim,
    )

    log_dir = Path(args.log_dir).expanduser().resolve()
    start = time.time()
    outer_env = None
    learn_returned = False
    try:
        base = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        clip_actions = getattr(agent_cfg, "clip_actions", None)
        outer_env = RslRlVecEnvWrapper(base, clip_actions=clip_actions)
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        runner_cls_name = getattr(runner_cls, "__name__", type(runner_cls).__name__)
        runner = runner_cls(
            outer_env,
            train_cfg,
            log_dir=str(log_dir),
            device=args.device,
        )
        if args.resume_checkpoint:
            runner.load(
                str(Path(args.resume_checkpoint).expanduser().resolve()),
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
        actor_before_learn = _actor_model(runner)
        actor_trainable_scope_report = _set_actor_trainable_scope(
            actor_before_learn,
            args.actor_trainable_scope,
        )
        all_actor_parameters_before = _actor_parameter_snapshot(
            torch,
            actor_before_learn,
            trainable_only=False,
        )
        txl_parameters_before = _actor_parameter_snapshot(
            torch,
            actor_before_learn,
            trainable_only=True,
        )
        rollout_env = runner.env
        action_dim = _action_dim(rollout_env, base)
        total_action_dim = _total_action_dim(base) or action_dim
        actual_num_envs = int(getattr(rollout_env, "num_envs", args.num_envs))

        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        learn_returned = True

        post_learn = collect_post_learn_diagnostics(
            torch=torch,
            runner=runner,
            rollout_env=rollout_env,
            device=args.device,
        )
        actor_after_learn = _actor_model(runner)
        all_actor_parameters_after = _actor_parameter_snapshot(
            torch,
            actor_after_learn,
            trainable_only=False,
        )
        txl_parameter_stats = _txl_parameter_stats(
            torch,
            txl_parameters_before,
            _actor_parameter_snapshot(torch, actor_after_learn, trainable_only=True),
        )
        actor_scope_parameter_stats = _actor_scope_parameter_stats(
            torch,
            all_actor_parameters_before,
            all_actor_parameters_after,
            actor_trainable_scope_report,
        )
        algorithm_debug = _algorithm_debug_snapshot(runner)
        task046_recovery_debug = _env_wrapper_debug_snapshot(
            rollout_env,
            "task046_recovery_debug_snapshot",
        )
        task046_retry_context_debug = _env_wrapper_debug_snapshot(
            rollout_env,
            "task046_retry_context_debug_snapshot",
        )
        final_iteration = int(getattr(runner, "current_learning_iteration", args.iterations - 1))
        checkpoint = log_dir / f"model_{final_iteration}.pt"
        summary = {
            "schema": "task041_sequence_txl_clean_train_v1",
            "task": args.task,
            "command": list(sys.argv),
            "seed": args.seed,
            "device": args.device,
            "num_envs": args.num_envs,
            "actual_num_envs": actual_num_envs,
            "rollout_steps": args.rollout_steps,
            "iterations": args.iterations,
            "final_iteration": final_iteration,
            "save_interval": args.save_interval,
            "num_mini_batches": args.num_mini_batches,
            "num_learning_epochs": _num_learning_epochs(train_cfg),
            "learning_rate": _algorithm_float(train_cfg, "learning_rate"),
            "desired_kl": _algorithm_float(train_cfg, "desired_kl"),
            "memory_latent_dim": args.memory_latent_dim,
            "memory_latent_scale": args.memory_latent_scale,
            "base_obs_passthrough_scale": args.base_obs_passthrough_scale,
            "adaptation_warmstart_scale": args.adaptation_warmstart_scale,
            "task044_fault_aux_loss_weight": args.task044_fault_aux_loss_weight,
            "task044_fault_aux_num_classes": args.task044_fault_aux_num_classes,
            "task044_fault_aux_max_trial_step": args.task044_fault_aux_max_trial_step,
            "task044_fault_aux_min_trial_index": args.task044_fault_aux_min_trial_index,
            "task046_post_reset_recovery_reward": args.task046_post_reset_recovery_reward,
            "task046_post_reset_recovery_reward_cfg": train_cfg.get(
                "task046_post_reset_recovery_reward",
                {},
            ),
            "task046_post_reset_recovery_reward_debug": task046_recovery_debug,
            "task046_retry_context": args.task046_retry_context,
            "task046_retry_context_cfg": train_cfg.get("task046_retry_context", {}),
            "task046_retry_context_debug": task046_retry_context_debug,
            "base_obs_passthrough": args.base_obs_passthrough,
            "adaptation_warmstart": args.adaptation_warmstart,
            "actor_trainable_scope": args.actor_trainable_scope,
            "actor_trainable_scope_report": actor_trainable_scope_report,
            "expected_action_dim": args.expected_action_dim,
            "expected_runner_cls": args.expected_runner_cls,
            "expected_actor_model_class": args.expected_actor_model_class,
            "expected_algorithm_class": args.expected_algorithm_class,
            "runner_cls": runner_cls_name,
            "algorithm_class": type(getattr(runner, "alg", None)).__name__,
            "actor_model_class": post_learn["actor_model_class"],
            "action_dim": action_dim,
            "total_action_dim": total_action_dim,
            "learn_returned": learn_returned,
            "checkpoint": str(checkpoint),
            "checkpoint_exists": checkpoint.exists(),
            "policy_action_shape": post_learn["policy_action_shape"],
            "policy_action_finite": post_learn["policy_action_finite"],
            "policy_error": post_learn["policy_error"],
            "obs": post_learn["obs"],
            "obs_all_finite": post_learn["obs_all_finite"],
            "obs_error": post_learn["obs_error"],
            "txl_debug": post_learn["txl_debug"],
            "txl_debug_error": post_learn["txl_debug_error"],
            "actor_model_error": post_learn["actor_model_error"],
            "algorithm_debug": algorithm_debug,
            "txl_parameter_stats": txl_parameter_stats,
            "actor_scope_parameter_stats": actor_scope_parameter_stats,
            "log_dir": str(log_dir),
            "log_dir_exists": log_dir.exists(),
            "log_dir_files": summarize_log_dir_files(log_dir),
            "experiment_name": args.experiment_name,
            "run_name": train_cfg.get("run_name"),
            "resume_checkpoint": args.resume_checkpoint,
            "task041_sequence_txl_clean_train_only": True,
            "train_pipeline_pass": False,
            "quality_gate_pass": False,
            "pass": False,
            "quality_claim": False,
            "training_claim": False,
            "eval_claim": False,
            "reproduction_claim": False,
            "superiority_claim": False,
            "diagnostic_note": (
                "Task041 train pipeline evidence only; eval quality is proven "
                "only by task041_sequence_txl_clean_eval."
            ),
            "wall_time_s": time.time() - start,
        }
        summary["train_pipeline_pass"], summary["failure_reasons"] = evaluate_train_pipeline_pass(summary)
        return summary
    finally:
        if outer_env is not None and hasattr(outer_env, "close"):
            outer_env.close()


def mutate_agent_cfg_for_task041_train(
    train_cfg: dict[str, Any],
    *,
    rollout_steps: int,
    iterations: int,
    save_interval: int,
    seed: int,
    num_mini_batches: int,
    num_learning_epochs: int | None,
    learning_rate: float | None = None,
    desired_kl: float | None = None,
    experiment_name: str,
    run_name: str,
    memory_latent_dim: int = 32,
    memory_latent_scale: float = 1.0,
    base_obs_passthrough_scale: float = 1.0,
    adaptation_warmstart_scale: float = 1.0,
    task044_fault_aux_loss_weight: float = 0.0,
    task044_fault_aux_num_classes: int = 0,
    task044_fault_aux_max_trial_step: int = -1,
    task044_fault_aux_min_trial_index: int = 0,
    task046_post_reset_recovery_reward: bool = False,
    task046_final_trial_index: int = 2,
    task046_recovery_window_steps: int = 50,
    task046_tail_window_steps: int = 50,
    task046_early_velocity_weight: float = 0.0,
    task046_tail_velocity_weight: float = 0.0,
    task046_orientation_weight: float = 0.0,
    task046_root_height_weight: float = 0.0,
    task046_min_root_z: float = 0.70,
    task046_retry_context: bool = False,
    task046_retry_context_num_trials: int = 3,
    task046_retry_context_final_trial_index: int = 2,
    task046_retry_context_step_window_steps: int = 50,
    action_dim: int = DEFAULT_EXPECTED_ACTION_DIM,
    base_obs_passthrough: bool = True,
    adaptation_warmstart: bool = True,
    adaptation_hidden_dim: int = 128,
) -> dict[str, Any]:
    train_cfg["num_steps_per_env"] = int(rollout_steps)
    train_cfg["max_iterations"] = int(iterations)
    train_cfg["save_interval"] = int(save_interval)
    train_cfg["logger"] = "tensorboard"
    train_cfg["experiment_name"] = experiment_name
    train_cfg["run_name"] = run_name
    train_cfg["seed"] = int(seed)
    train_cfg["upload_model"] = False
    train_cfg["resume"] = False
    train_cfg["task046_post_reset_recovery_reward"] = {
        "enabled": bool(task046_post_reset_recovery_reward),
        "final_trial_index": int(task046_final_trial_index),
        "recovery_window_steps": int(task046_recovery_window_steps),
        "tail_window_steps": int(task046_tail_window_steps),
        "early_velocity_weight": float(task046_early_velocity_weight),
        "tail_velocity_weight": float(task046_tail_velocity_weight),
        "orientation_weight": float(task046_orientation_weight),
        "root_height_weight": float(task046_root_height_weight),
        "min_root_z": float(task046_min_root_z),
    }
    train_cfg["task046_retry_context"] = {
        "enabled": bool(task046_retry_context),
        "num_trials": int(task046_retry_context_num_trials),
        "final_trial_index": int(task046_retry_context_final_trial_index),
        "step_window_steps": int(task046_retry_context_step_window_steps),
    }
    actor = train_cfg.setdefault("actor", {})
    if isinstance(actor, dict):
        actor["memory_latent_dim"] = int(memory_latent_dim)
        actor["memory_latent_scale"] = float(memory_latent_scale)
        actor["base_obs_passthrough_scale"] = float(base_obs_passthrough_scale)
        actor["adaptation_warmstart_scale"] = float(adaptation_warmstart_scale)
        actor["action_dim"] = int(action_dim)
        actor["base_obs_passthrough"] = bool(base_obs_passthrough)
        actor["adaptation_warmstart"] = bool(adaptation_warmstart)
        actor["adaptation_hidden_dim"] = int(adaptation_hidden_dim)
    algorithm = train_cfg.get("algorithm")
    if isinstance(algorithm, dict):
        algorithm["class_name"] = TASK040_ALGORITHM_CLASS
        algorithm["num_mini_batches"] = int(num_mini_batches)
        if learning_rate is not None:
            algorithm["learning_rate"] = float(learning_rate)
        if desired_kl is not None:
            algorithm["desired_kl"] = float(desired_kl)
        algorithm["task044_fault_aux_loss_weight"] = float(task044_fault_aux_loss_weight)
        algorithm["task044_fault_aux_num_classes"] = int(task044_fault_aux_num_classes)
        algorithm["task044_fault_aux_max_trial_step"] = int(task044_fault_aux_max_trial_step)
        algorithm["task044_fault_aux_min_trial_index"] = int(task044_fault_aux_min_trial_index)
        if num_learning_epochs is not None:
            algorithm["num_learning_epochs"] = int(num_learning_epochs)
    return train_cfg


def evaluate_train_pipeline_pass(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary.get("task") != DEFAULT_TASK:
        reasons.append("task_not_task041_sequence_txl_clean_train")
    if summary.get("runner_cls") != summary.get("expected_runner_cls"):
        reasons.append("runner_cls_mismatch")
    if summary.get("algorithm_class") != summary.get("expected_algorithm_class"):
        reasons.append("algorithm_class_mismatch")
    if summary.get("actor_model_class") != summary.get("expected_actor_model_class"):
        reasons.append("actor_model_class_mismatch")
    expected_action_dim = int(summary.get("expected_action_dim") or -1)
    if int(summary.get("action_dim") or -1) != expected_action_dim:
        reasons.append("action_dim_mismatch")
    if int(summary.get("total_action_dim") or -1) != expected_action_dim:
        reasons.append("total_action_dim_mismatch")
    if not summary.get("learn_returned"):
        reasons.append("learn_not_returned")
    if not summary.get("checkpoint_exists"):
        reasons.append("checkpoint_missing")
    debug = summary.get("txl_debug") or {}
    if int(debug.get("stateless_fallback_forward_batches") or 0) != 0:
        reasons.append("txl_debug_stateless_fallback_seen")
    if int(debug.get("sequence_update_forward_batches") or 0) <= 0:
        reasons.append("txl_debug_no_sequence_update_forward")
    algorithm_debug = summary.get("algorithm_debug") or {}
    if int(algorithm_debug.get("sequence_update_batches") or 0) <= 0:
        reasons.append("algorithm_debug_no_sequence_update_batches")
    if not algorithm_debug.get("last_loss_dict"):
        reasons.append("algorithm_debug_missing_loss_dict")
    fault_aux_weight = float(summary.get("task044_fault_aux_loss_weight") or 0.0)
    if fault_aux_weight > 0.0:
        if int(algorithm_debug.get("task044_fault_aux_updates") or 0) <= 0:
            reasons.append("task044_fault_aux_no_updates")
        if algorithm_debug.get("task044_fault_aux_last_loss") is None:
            reasons.append("task044_fault_aux_missing_loss")
    scope = str(summary.get("actor_trainable_scope") or "all")
    scope_report = summary.get("actor_trainable_scope_report") or {}
    if scope not in ACTOR_TRAINABLE_SCOPES:
        reasons.append("actor_trainable_scope_unknown")
    elif scope != "all":
        if int(scope_report.get("trainable_parameter_count") or 0) <= 0:
            reasons.append("actor_trainable_scope_no_trainable_parameters")
        scope_stats = summary.get("actor_scope_parameter_stats") or {}
        frozen_delta_norm = float(scope_stats.get("frozen_parameter_delta_norm") or 0.0)
        if frozen_delta_norm > 1e-9:
            reasons.append("actor_trainable_scope_frozen_parameters_changed")
        frozen_normalizer_delta_norm = float(scope_stats.get("frozen_obs_normalizer_delta_norm") or 0.0)
        if frozen_normalizer_delta_norm > 1e-9:
            reasons.append("actor_trainable_scope_obs_normalizer_changed")
        partial_frozen_delta_norm = float(scope_stats.get("partial_frozen_delta_norm") or 0.0)
        if partial_frozen_delta_norm > 1e-9:
            reasons.append("actor_trainable_scope_partial_frozen_columns_changed")
    if summary.get("policy_action_shape") is not None and not summary.get("policy_action_finite"):
        reasons.append("policy_action_not_finite")
    if not summary.get("log_dir_exists"):
        reasons.append("log_dir_missing")
    if (
        not summary.get("task041_sequence_txl_clean_train_only")
        or summary.get("quality_claim") is not False
        or summary.get("training_claim") is not False
        or summary.get("eval_claim") is not False
        or summary.get("reproduction_claim") is not False
        or summary.get("superiority_claim") is not False
    ):
        reasons.append("claim_boundary_violation")
    return not reasons, reasons


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    log_dir = Path(args.log_dir).expanduser().resolve()
    summary = {
        "schema": "task041_sequence_txl_clean_train_v1",
        "task": getattr(args, "task", DEFAULT_TASK),
        "command": list(sys.argv),
        "seed": getattr(args, "seed", None),
        "device": getattr(args, "device", ""),
        "num_envs": getattr(args, "num_envs", None),
        "rollout_steps": getattr(args, "rollout_steps", None),
        "iterations": getattr(args, "iterations", None),
        "save_interval": getattr(args, "save_interval", None),
        "num_mini_batches": getattr(args, "num_mini_batches", None),
        "actor_trainable_scope": getattr(args, "actor_trainable_scope", "all"),
        "task044_fault_aux_loss_weight": getattr(args, "task044_fault_aux_loss_weight", 0.0),
        "base_obs_passthrough_scale": getattr(args, "base_obs_passthrough_scale", 1.0),
        "adaptation_warmstart_scale": getattr(args, "adaptation_warmstart_scale", 1.0),
        "task044_fault_aux_num_classes": getattr(args, "task044_fault_aux_num_classes", 0),
        "task044_fault_aux_max_trial_step": getattr(args, "task044_fault_aux_max_trial_step", -1),
        "task044_fault_aux_min_trial_index": getattr(args, "task044_fault_aux_min_trial_index", 0),
        "task046_post_reset_recovery_reward": getattr(
            args,
            "task046_post_reset_recovery_reward",
            False,
        ),
        "task046_retry_context": getattr(args, "task046_retry_context", False),
        "expected_action_dim": getattr(args, "expected_action_dim", None),
        "expected_runner_cls": getattr(args, "expected_runner_cls", ""),
        "expected_actor_model_class": getattr(args, "expected_actor_model_class", ""),
        "expected_algorithm_class": getattr(args, "expected_algorithm_class", ""),
        "learn_returned": False,
        "checkpoint_exists": False,
        "log_dir": str(log_dir),
        "log_dir_exists": log_dir.exists(),
        "task041_sequence_txl_clean_train_only": True,
        "train_pipeline_pass": False,
        "quality_gate_pass": False,
        "pass": False,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "failure_reasons": ["train_exception"],
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }
    if isinstance(exc, PreflightError):
        summary["preflight_rejected"] = True
        summary["failure_reasons"] = list(exc.reasons)
    return summary


def write_json_summary(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary["json_path"] = str(output)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_run_name(args: argparse.Namespace) -> str:
    return f"seq_txl_clean_env{args.num_envs}_iter{args.iterations}_seed{args.seed}"


def _env_wrapper_debug_snapshot(env: Any, method_name: str) -> dict[str, Any]:
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        method = getattr(current, method_name, None)
        if callable(method):
            data = method()
            return dict(data) if isinstance(data, dict) else {}
        current = getattr(current, "env", None)
    return {}


def _num_learning_epochs(train_cfg: dict[str, Any]) -> int | None:
    algorithm = train_cfg.get("algorithm")
    if not isinstance(algorithm, dict):
        return None
    value = algorithm.get("num_learning_epochs")
    return int(value) if value is not None else None


def _algorithm_float(train_cfg: dict[str, Any], key: str) -> float | None:
    algorithm = train_cfg.get("algorithm")
    if not isinstance(algorithm, dict):
        return None
    value = algorithm.get(key)
    return float(value) if value is not None else None


def _actor_parameter_snapshot(
    torch: Any,
    actor: Any | None,
    *,
    trainable_only: bool = True,
) -> dict[str, Any]:
    if actor is None or not hasattr(actor, "named_parameters"):
        return {}
    snapshot = {
        name: parameter.detach().clone()
        for name, parameter in actor.named_parameters()
        if not trainable_only or bool(getattr(parameter, "requires_grad", False))
    }
    if not trainable_only and hasattr(actor, "named_buffers"):
        snapshot.update(
            {
                name: buffer.detach().clone()
                for name, buffer in actor.named_buffers()
                if hasattr(buffer, "detach")
            }
        )
    return snapshot


def _set_actor_trainable_scope(actor: Any | None, scope: str) -> dict[str, Any]:
    if scope not in ACTOR_TRAINABLE_SCOPES:
        raise ValueError(f"unknown actor trainable scope: {scope}")
    if actor is None or not hasattr(actor, "named_parameters"):
        return {
            "scope": scope,
            "actor_present": False,
            "normalization_update_disabled": False,
            "trainable_parameter_count": 0,
            "frozen_parameter_count": 0,
            "trainable_tensor_count": 0,
            "frozen_tensor_count": 0,
            "trainable_parameter_names": [],
            "frozen_parameter_names": [],
        }

    allowed_prefixes = _actor_trainable_prefixes(scope)
    partial_specs = _partial_trainable_specs(actor, scope)
    partial_names = {str(spec["name"]) for spec in partial_specs}
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    trainable_count = 0
    frozen_count = 0
    for name, parameter in actor.named_parameters():
        trainable = (
            scope == "all"
            or name in partial_names
            or any(name.startswith(prefix) for prefix in allowed_prefixes)
        )
        if scope != "all" and hasattr(parameter, "requires_grad_"):
            parameter.requires_grad_(trainable)
        if name in partial_names:
            _register_partial_gradient_mask(parameter, partial_specs, name)
        if bool(getattr(parameter, "requires_grad", False)):
            trainable_names.append(name)
            trainable_count += int(parameter.numel())
        else:
            frozen_names.append(name)
            frozen_count += int(parameter.numel())

    normalization_update_disabled = False
    if scope != "all" and hasattr(actor, "update_normalization"):
        actor.update_normalization = lambda obs: None
        normalization_update_disabled = True

    return {
        "scope": scope,
        "actor_present": True,
        "normalization_update_disabled": normalization_update_disabled,
        "allowed_prefixes": list(allowed_prefixes),
        "partial_trainable_parameters": partial_specs,
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_count": frozen_count,
        "trainable_tensor_count": len(trainable_names),
        "frozen_tensor_count": len(frozen_names),
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
    }


def _actor_trainable_prefixes(scope: str) -> tuple[str, ...]:
    if scope == "all":
        return ()
    if scope in {"txl_residual_only", "txl_residual_and_mlp_memory_input"}:
        return TXL_RESIDUAL_PARAMETER_PREFIXES
    if scope == "memory_output_projection_only":
        return MEMORY_OUTPUT_PROJECTION_PARAMETER_PREFIXES
    raise ValueError(f"unknown actor trainable scope: {scope}")


def _partial_trainable_specs(actor: Any, scope: str) -> list[dict[str, Any]]:
    if scope != "txl_residual_and_mlp_memory_input":
        return []
    memory_latent_dim = int(getattr(actor, "memory_latent_dim", 0) or 0)
    if memory_latent_dim <= 0:
        return []
    for name, parameter in actor.named_parameters():
        if name != "mlp.0.weight" or not hasattr(parameter, "shape"):
            continue
        shape = tuple(int(dim) for dim in parameter.shape)
        if len(shape) != 2 or shape[1] <= memory_latent_dim:
            continue
        trainable_start = shape[1] - memory_latent_dim
        return [
            {
                "name": name,
                "trainable_columns": [trainable_start, shape[1]],
                "frozen_columns": [0, trainable_start],
                "reason": "memory_latent_input_columns_only",
            }
        ]
    return []


def _register_partial_gradient_mask(
    parameter: Any,
    partial_specs: list[dict[str, Any]],
    parameter_name: str,
) -> None:
    spec = next((item for item in partial_specs if item.get("name") == parameter_name), None)
    if spec is None or not hasattr(parameter, "register_hook"):
        return
    trainable_start, trainable_end = [int(value) for value in spec["trainable_columns"]]

    def _mask_gradient(grad: Any) -> Any:
        masked = grad.clone()
        masked[:, :trainable_start] = 0
        if trainable_end < masked.shape[1]:
            masked[:, trainable_end:] = 0
        return masked

    parameter.register_hook(_mask_gradient)


def _actor_scope_parameter_stats(
    torch: Any,
    before: dict[str, Any],
    after: dict[str, Any],
    scope_report: dict[str, Any],
) -> dict[str, Any]:
    trainable_names = set(scope_report.get("trainable_parameter_names") or [])
    frozen_names = set(scope_report.get("frozen_parameter_names") or [])
    trainable_delta_tensors = _matching_delta_tensors(before, after, trainable_names)
    frozen_delta_tensors = _matching_delta_tensors(before, after, frozen_names)
    normalizer_names = {name for name in before if name.startswith("obs_normalizer.")}
    normalizer_delta_tensors = _matching_delta_tensors(before, after, normalizer_names)
    partial_stats = _partial_trainable_delta_stats(torch, before, after, scope_report)
    return {
        "trainable_parameter_delta_norm": _tensor_collection_l2_norm(torch, trainable_delta_tensors),
        "trainable_parameter_delta_max_abs": _tensor_collection_max_abs(torch, trainable_delta_tensors),
        "frozen_parameter_delta_norm": _tensor_collection_l2_norm(torch, frozen_delta_tensors),
        "frozen_parameter_delta_max_abs": _tensor_collection_max_abs(torch, frozen_delta_tensors),
        "frozen_obs_normalizer_delta_norm": _tensor_collection_l2_norm(torch, normalizer_delta_tensors),
        "frozen_obs_normalizer_delta_max_abs": _tensor_collection_max_abs(torch, normalizer_delta_tensors),
        "trainable_tensor_count": len(trainable_delta_tensors),
        "frozen_tensor_count": len(frozen_delta_tensors),
        "frozen_obs_normalizer_tensor_count": len(normalizer_delta_tensors),
        "frozen_parameters_unchanged": _tensor_collection_l2_norm(torch, frozen_delta_tensors) <= 1e-9,
        "frozen_obs_normalizer_unchanged": _tensor_collection_l2_norm(torch, normalizer_delta_tensors) <= 1e-9,
        **partial_stats,
    }


def _partial_trainable_delta_stats(
    torch: Any,
    before: dict[str, Any],
    after: dict[str, Any],
    scope_report: dict[str, Any],
) -> dict[str, Any]:
    trainable_tensors: list[Any] = []
    frozen_tensors: list[Any] = []
    for spec in scope_report.get("partial_trainable_parameters") or []:
        name = str(spec.get("name") or "")
        if name not in before or name not in after:
            continue
        if tuple(before[name].shape) != tuple(after[name].shape):
            continue
        train_start, train_end = [int(value) for value in spec.get("trainable_columns", [0, 0])]
        frozen_start, frozen_end = [int(value) for value in spec.get("frozen_columns", [0, 0])]
        delta = after[name] - before[name]
        trainable_tensors.append(delta[:, train_start:train_end])
        frozen_tensors.append(delta[:, frozen_start:frozen_end])
        if train_end < delta.shape[1]:
            frozen_tensors.append(delta[:, train_end:])
    return {
        "partial_trainable_delta_norm": _tensor_collection_l2_norm(torch, trainable_tensors),
        "partial_trainable_delta_max_abs": _tensor_collection_max_abs(torch, trainable_tensors),
        "partial_frozen_delta_norm": _tensor_collection_l2_norm(torch, frozen_tensors),
        "partial_frozen_delta_max_abs": _tensor_collection_max_abs(torch, frozen_tensors),
        "partial_frozen_columns_unchanged": _tensor_collection_l2_norm(torch, frozen_tensors) <= 1e-9,
    }


def _matching_delta_tensors(
    before: dict[str, Any],
    after: dict[str, Any],
    names: set[str],
) -> list[Any]:
    return [
        after[name] - before[name]
        for name in sorted(names)
        if name in before and name in after and tuple(after[name].shape) == tuple(before[name].shape)
    ]


def _txl_parameter_stats(
    torch: Any,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    groups = {
        "memory_output_projection": ("memory_output_projection",),
        "attention_layers": ("attention_layers",),
        "norm_layers": ("norm_layers",),
        "token_projection": ("token_projection",),
        "position_embedding": ("position_embedding",),
    }
    stats: dict[str, Any] = {
        "trainable_param_count": int(sum(int(tensor.numel()) for tensor in after.values())),
        "tracked_param_count": int(
            sum(
                int(tensor.numel())
                for name, tensor in after.items()
                if _txl_parameter_group_name(name, groups) is not None
            )
        ),
        "groups": {},
    }
    for group_name, prefixes in groups.items():
        stats["groups"][group_name] = _parameter_group_delta_stats(
            torch,
            before,
            after,
            prefixes,
        )
    memory_group = stats["groups"]["memory_output_projection"]
    stats["memory_output_projection_weight_norm_before"] = memory_group["before_norm"]
    stats["memory_output_projection_weight_norm_after"] = memory_group["after_norm"]
    stats["memory_output_projection_delta_norm"] = memory_group["delta_norm"]
    stats["attention_layers_delta_norm"] = stats["groups"]["attention_layers"]["delta_norm"]
    stats["norm_layers_delta_norm"] = stats["groups"]["norm_layers"]["delta_norm"]
    stats["token_projection_delta_norm"] = stats["groups"]["token_projection"]["delta_norm"]
    stats["position_embedding_delta_norm"] = stats["groups"]["position_embedding"]["delta_norm"]
    return stats


def _txl_parameter_group_name(
    parameter_name: str,
    groups: dict[str, tuple[str, ...]],
) -> str | None:
    for group_name, prefixes in groups.items():
        if any(parameter_name.startswith(prefix) for prefix in prefixes):
            return group_name
    return None


def _parameter_group_delta_stats(
    torch: Any,
    before: dict[str, Any],
    after: dict[str, Any],
    prefixes: tuple[str, ...],
) -> dict[str, Any]:
    keys = sorted(
        name
        for name in after
        if any(name.startswith(prefix) for prefix in prefixes)
    )
    before_tensors = [before[name] for name in keys if name in before]
    after_tensors = [after[name] for name in keys]
    delta_tensors = [
        after[name] - before[name]
        for name in keys
        if name in before and tuple(after[name].shape) == tuple(before[name].shape)
    ]
    return {
        "parameter_count": int(sum(int(after[name].numel()) for name in keys)),
        "tensor_count": len(keys),
        "before_norm": _tensor_collection_l2_norm(torch, before_tensors),
        "after_norm": _tensor_collection_l2_norm(torch, after_tensors),
        "delta_norm": _tensor_collection_l2_norm(torch, delta_tensors),
    }


def _tensor_collection_l2_norm(torch: Any, tensors: list[Any]) -> float:
    if not tensors:
        return 0.0
    total = torch.zeros((), device=tensors[0].device, dtype=torch.float32)
    for tensor in tensors:
        total = total + tensor.detach().float().pow(2).sum()
    return float(torch.sqrt(total).detach().cpu().item())


def _tensor_collection_max_abs(torch: Any, tensors: list[Any]) -> float:
    if not tensors:
        return 0.0
    max_abs = torch.zeros((), device=tensors[0].device, dtype=torch.float32)
    for tensor in tensors:
        tensor_max = tensor.detach().float().abs().max()
        max_abs = torch.maximum(max_abs, tensor_max)
    return float(max_abs.detach().cpu().item())


def main() -> None:
    args = parse_args()
    try:
        preflight_args(args)
        summary = run_train(args)
    except PreflightError as exc:
        summary = build_failure_summary(args, exc)
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        summary = build_failure_summary(args, exc)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
