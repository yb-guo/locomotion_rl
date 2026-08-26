"""Run a tiny Task038 true-TXL PPO update smoke on the train variant only."""

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
from h200_locomotion_lab.tools.task038_true_txl_runner_smoke_probe import (
    DEFAULT_EXPECTED_ACTION_DIM,
    DEFAULT_EXPECTED_ACTOR_MODEL_CLASS,
    DEFAULT_EXPECTED_RUNNER_CLS,
    TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID,
    _action_dim,
    _actor_model,
    _agent_cfg_as_dict,
    _finite,
    _first,
    _get_policy,
    _load_env_cfg,
    _obs_all_finite,
    _obs_summary,
    _policy_action_shape_matches,
    _set_if_present,
    _shape,
    _total_action_dim,
    _txl_debug_snapshot,
)

DEFAULT_TASK = TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID
DEFAULT_LOG_DIR = Path("outputs/task038/true_txl_ppo_update_smoke")


class PreflightError(ValueError):
    def __init__(self, reasons: list[str]) -> None:
        super().__init__(", ".join(reasons))
        self.reasons = reasons


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one tiny Task038 true-TXL PPO update smoke on the train variant. "
            "This is a training-path crash smoke only, not eval or quality evidence."
        )
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3801301)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-action-dim", type=int, default=DEFAULT_EXPECTED_ACTION_DIM)
    parser.add_argument("--expected-runner-cls", default=DEFAULT_EXPECTED_RUNNER_CLS)
    parser.add_argument(
        "--expected-actor-model-class",
        default=DEFAULT_EXPECTED_ACTOR_MODEL_CLASS,
    )
    return parser.parse_args(argv)


def preflight_args(args: argparse.Namespace) -> None:
    reasons: list[str] = []
    if args.task != DEFAULT_TASK:
        reasons.append("task_not_train_true_txl_runner_smoke")
    if int(args.iterations) != 1:
        reasons.append("iterations_not_one")
    if reasons:
        raise PreflightError(reasons)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    preflight_args(args)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("WANDB_DISABLED", "true")

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

    train_cfg = mutate_agent_cfg_for_smoke(
        _agent_cfg_as_dict(agent_cfg),
        rollout_steps=args.rollout_steps,
        iterations=args.iterations,
        seed=args.seed,
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
        rollout_env = runner.env
        action_dim = _action_dim(rollout_env, base)
        total_action_dim = _total_action_dim(base) or action_dim
        actual_num_envs = int(getattr(rollout_env, "num_envs", args.num_envs))

        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        learn_returned = True

        post_learn = collect_post_learn_diagnostics(
            torch=torch,
            runner=runner,
            rollout_env=rollout_env,
            device=args.device,
        )

        summary = {
            "task": args.task,
            "command": list(sys.argv),
            "seed": args.seed,
            "device": args.device,
            "num_envs": args.num_envs,
            "actual_num_envs": actual_num_envs,
            "rollout_steps": args.rollout_steps,
            "iterations": args.iterations,
            "expected_action_dim": args.expected_action_dim,
            "expected_runner_cls": args.expected_runner_cls,
            "expected_actor_model_class": args.expected_actor_model_class,
            "runner_cls": runner_cls_name,
            "actor_model_class": post_learn["actor_model_class"],
            "action_dim": action_dim,
            "total_action_dim": total_action_dim,
            "learn_returned": learn_returned,
            "policy_action_shape": post_learn["policy_action_shape"],
            "policy_action_finite": post_learn["policy_action_finite"],
            "policy_error": post_learn["policy_error"],
            "obs": post_learn["obs"],
            "obs_all_finite": post_learn["obs_all_finite"],
            "obs_error": post_learn["obs_error"],
            "txl_debug": post_learn["txl_debug"],
            "txl_debug_error": post_learn["txl_debug_error"],
            "actor_model_error": post_learn["actor_model_error"],
            "log_dir": str(log_dir),
            "log_dir_exists": log_dir.exists(),
            "log_dir_files": summarize_log_dir_files(log_dir),
            "quality_claim": False,
            "training_claim": False,
            "eval_claim": False,
            "reproduction_claim": False,
            "superiority_claim": False,
            "ppo_update_smoke_only": True,
            "wall_time_s": time.time() - start,
        }
        summary["pass"], summary["failure_reasons"] = evaluate_probe_pass(summary)
        return summary
    finally:
        if outer_env is not None and hasattr(outer_env, "close"):
            outer_env.close()


def collect_post_learn_diagnostics(
    *,
    torch: Any,
    runner: Any,
    rollout_env: Any,
    device: str,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "actor_model_class": None,
        "actor_model_error": None,
        "txl_debug": {},
        "txl_debug_error": None,
        "policy_action_shape": None,
        "policy_action_finite": False,
        "policy_error": None,
        "obs": {},
        "obs_all_finite": None,
        "obs_error": None,
    }

    actor_model = None
    try:
        actor_model = _actor_model(runner)
        diagnostics["actor_model_class"] = type(actor_model).__name__ if actor_model else None
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        diagnostics["actor_model_error"] = repr(exc)

    if actor_model is not None:
        try:
            diagnostics["txl_debug"] = _txl_debug_snapshot(actor_model)
        except RECOVERABLE_RUNTIME_ERRORS as exc:
            diagnostics["txl_debug_error"] = repr(exc)

    policy = None
    try:
        policy, diagnostics["policy_error"] = _get_policy(runner, device)
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        diagnostics["policy_error"] = repr(exc)

    obs = None
    try:
        obs = _get_observations(rollout_env)
        if obs is not None:
            obs_summary = _obs_summary(torch, obs)
            diagnostics["obs"] = obs_summary
            diagnostics["obs_all_finite"] = _obs_all_finite(obs_summary)
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        diagnostics["obs_error"] = repr(exc)

    if policy is not None and obs is not None:
        try:
            with torch.no_grad():
                policy_action = policy(obs)
            diagnostics["policy_action_shape"] = _shape(policy_action)
            diagnostics["policy_action_finite"] = _finite(torch, policy_action)
        except RECOVERABLE_RUNTIME_ERRORS as exc:
            diagnostics["policy_error"] = repr(exc)

    return diagnostics


def mutate_agent_cfg_for_smoke(
    train_cfg: dict[str, Any],
    *,
    rollout_steps: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    train_cfg["num_steps_per_env"] = int(rollout_steps)
    train_cfg["max_iterations"] = int(iterations)
    train_cfg["save_interval"] = 1000000
    train_cfg["logger"] = "tensorboard"
    train_cfg["experiment_name"] = "task038_true_txl_ppo_update_smoke"
    train_cfg["run_name"] = f"env_smoke_seed{seed}"
    train_cfg["seed"] = int(seed)
    train_cfg["upload_model"] = False
    train_cfg["resume"] = False
    algorithm = train_cfg.get("algorithm")
    if isinstance(algorithm, dict):
        algorithm["num_learning_epochs"] = 1
        algorithm["num_mini_batches"] = 1
    return train_cfg


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    log_dir = Path(args.log_dir).expanduser().resolve()
    return {
        "task": args.task,
        "command": list(sys.argv),
        "seed": args.seed,
        "device": args.device,
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "iterations": args.iterations,
        "expected_action_dim": args.expected_action_dim,
        "expected_runner_cls": args.expected_runner_cls,
        "expected_actor_model_class": args.expected_actor_model_class,
        "learn_returned": False,
        "log_dir": str(log_dir),
        "log_dir_exists": log_dir.exists(),
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "ppo_update_smoke_only": True,
        "pass": False,
        "failure_reasons": ["probe_exception"],
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }


def build_preflight_failure_summary(
    args: argparse.Namespace, exc: PreflightError
) -> dict[str, Any]:
    summary = build_failure_summary(args, exc)
    summary["failure_reasons"] = list(exc.reasons)
    summary["preflight_rejected"] = True
    return summary


def evaluate_probe_pass(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary.get("task") != DEFAULT_TASK:
        reasons.append("task_not_train_true_txl_runner_smoke")
    if summary.get("runner_cls") != summary.get("expected_runner_cls"):
        reasons.append("runner_cls_mismatch")
    if summary.get("actor_model_class") != summary.get("expected_actor_model_class"):
        reasons.append("actor_model_class_mismatch")
    expected_action_dim = int(summary.get("expected_action_dim") or -1)
    if int(summary.get("action_dim") or -1) != expected_action_dim:
        reasons.append("action_dim_mismatch")
    if int(summary.get("total_action_dim") or -1) != expected_action_dim:
        reasons.append("total_action_dim_mismatch")
    if not summary.get("learn_returned"):
        reasons.append("learn_not_returned")
    elif int((summary.get("txl_debug") or {}).get("stateless_forward_batches") or 0) <= 0:
        reasons.append("txl_debug_no_stateless_minibatch_fallback")
    if int(summary.get("iterations") or 0) != 1:
        reasons.append("iterations_not_one")
    if int(summary.get("num_envs") or 0) <= 0:
        reasons.append("num_envs_not_positive")
    if int(summary.get("rollout_steps") or 0) <= 0:
        reasons.append("rollout_steps_not_positive")
    if summary.get("policy_action_shape") is not None:
        if not summary.get("policy_action_finite"):
            reasons.append("policy_action_not_finite")
        elif not _policy_action_shape_matches(summary, expected_action_dim):
            reasons.append("policy_action_shape_mismatch")
    if not summary.get("log_dir_exists"):
        reasons.append("log_dir_missing")
    if summary.get("wall_time_s") is None:
        reasons.append("wall_time_missing")
    if (
        summary.get("quality_claim")
        or summary.get("training_claim")
        or summary.get("eval_claim")
        or summary.get("reproduction_claim")
        or summary.get("superiority_claim")
        or not summary.get("ppo_update_smoke_only")
    ):
        reasons.append("claim_boundary_violation")
    return not reasons, reasons


def summarize_log_dir_files(log_dir: Path, limit: int = 40) -> list[str]:
    if not log_dir.exists():
        return []
    files: list[str] = []
    for path in sorted(log_dir.rglob("*")):
        if path.is_file():
            files.append(str(path.relative_to(log_dir)))
        if len(files) >= limit:
            break
    return files


def write_json_summary(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary["json_path"] = str(output)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _get_observations(env: Any) -> Any | None:
    getter = getattr(env, "get_observations", None)
    if callable(getter):
        return getter()
    try:
        return _first(env.reset())
    except RECOVERABLE_RUNTIME_ERRORS:
        return None


def main() -> None:
    args = parse_args()
    try:
        preflight_args(args)
        summary = run_probe(args)
    except PreflightError as exc:
        summary = build_preflight_failure_summary(args, exc)
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        summary = build_failure_summary(args, exc)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
