"""Run a Task040 sequence-aware true-TXL PPO update smoke."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import types
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
    _agent_cfg_as_dict,
    _load_env_cfg,
    _policy_action_shape_matches,
    _set_if_present,
    _total_action_dim,
)

DEFAULT_TASK = TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID
DEFAULT_LOG_DIR = Path("outputs/task040/sequence_txl_ppo_update_smoke")
DEFAULT_EXPECTED_ALGORITHM_CLASS = "Task040SequenceAwareTrueTxlPPO"
TASK040_ALGORITHM_CLASS = (
    "h200_locomotion_lab.training.rsl_history_wrapper:Task040SequenceAwareTrueTxlPPO"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one tiny Task040 true-TXL PPO update smoke using a sequence-aware "
            "actor update path. This is plumbing evidence only, not policy-quality evidence."
        )
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4004001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-mini-batches", type=int, default=1)
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
        reasons.append("task_not_train_true_txl_runner_smoke")
    if int(args.iterations) != 1:
        reasons.append("iterations_not_one")
    if int(args.num_mini_batches) <= 0:
        reasons.append("num_mini_batches_not_positive")
    if int(args.num_envs) % int(args.num_mini_batches) != 0:
        reasons.append("num_envs_not_divisible_by_num_mini_batches")
    if reasons:
        raise PreflightError(reasons)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
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

    train_cfg = mutate_agent_cfg_for_sequence_smoke(
        _agent_cfg_as_dict(agent_cfg),
        rollout_steps=args.rollout_steps,
        iterations=args.iterations,
        seed=args.seed,
        num_mini_batches=args.num_mini_batches,
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
        algorithm_debug = _algorithm_debug_snapshot(runner)
        summary = {
            "task": args.task,
            "command": list(sys.argv),
            "seed": args.seed,
            "device": args.device,
            "num_envs": args.num_envs,
            "actual_num_envs": actual_num_envs,
            "rollout_steps": args.rollout_steps,
            "iterations": args.iterations,
            "num_mini_batches": args.num_mini_batches,
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
            "log_dir": str(log_dir),
            "log_dir_exists": log_dir.exists(),
            "log_dir_files": summarize_log_dir_files(log_dir),
            "quality_claim": False,
            "training_claim": False,
            "eval_claim": False,
            "reproduction_claim": False,
            "superiority_claim": False,
            "sequence_aware_ppo_update_smoke_only": True,
            "wall_time_s": time.time() - start,
        }
        summary["pass"], summary["failure_reasons"] = evaluate_probe_pass(summary)
        return summary
    finally:
        if outer_env is not None and hasattr(outer_env, "close"):
            outer_env.close()


def mutate_agent_cfg_for_sequence_smoke(
    train_cfg: dict[str, Any],
    *,
    rollout_steps: int,
    iterations: int,
    seed: int,
    num_mini_batches: int,
) -> dict[str, Any]:
    train_cfg["num_steps_per_env"] = int(rollout_steps)
    train_cfg["max_iterations"] = int(iterations)
    train_cfg["save_interval"] = 1000000
    train_cfg["logger"] = "tensorboard"
    train_cfg["experiment_name"] = "task040_sequence_txl_ppo_update_smoke"
    train_cfg["run_name"] = f"env_smoke_seed{seed}"
    train_cfg["seed"] = int(seed)
    train_cfg["upload_model"] = False
    train_cfg["resume"] = False
    algorithm = train_cfg.get("algorithm")
    if isinstance(algorithm, dict):
        algorithm["class_name"] = TASK040_ALGORITHM_CLASS
        algorithm["num_learning_epochs"] = 1
        algorithm["num_mini_batches"] = int(num_mini_batches)
    return train_cfg


def evaluate_probe_pass(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary.get("task") != DEFAULT_TASK:
        reasons.append("task_not_train_true_txl_runner_smoke")
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
    debug = summary.get("txl_debug") or {}
    if int(debug.get("stateless_fallback_forward_batches") or 0) != 0:
        reasons.append("txl_debug_stateless_fallback_seen")
    if int(debug.get("stateless_fallback_forward_samples") or 0) != 0:
        reasons.append("txl_debug_stateless_fallback_samples_seen")
    if int(debug.get("sequence_update_forward_batches") or 0) <= 0:
        reasons.append("txl_debug_no_sequence_update_forward")
    algorithm_debug = summary.get("algorithm_debug") or {}
    if int(algorithm_debug.get("sequence_update_batches") or 0) <= 0:
        reasons.append("algorithm_debug_no_sequence_update_batches")
    if not algorithm_debug.get("last_loss_dict"):
        reasons.append("algorithm_debug_missing_loss_dict")
    if int(summary.get("iterations") or 0) != 1:
        reasons.append("iterations_not_one")
    if int(summary.get("num_envs") or 0) <= 0:
        reasons.append("num_envs_not_positive")
    if int(summary.get("rollout_steps") or 0) <= 0:
        reasons.append("rollout_steps_not_positive")
    if int(summary.get("num_mini_batches") or 0) <= 0:
        reasons.append("num_mini_batches_not_positive")
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
        or not summary.get("sequence_aware_ppo_update_smoke_only")
    ):
        reasons.append("claim_boundary_violation")
    return not reasons, reasons


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
        "num_mini_batches": args.num_mini_batches,
        "expected_action_dim": args.expected_action_dim,
        "expected_runner_cls": args.expected_runner_cls,
        "expected_actor_model_class": args.expected_actor_model_class,
        "expected_algorithm_class": args.expected_algorithm_class,
        "learn_returned": False,
        "log_dir": str(log_dir),
        "log_dir_exists": log_dir.exists(),
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "sequence_aware_ppo_update_smoke_only": True,
        "pass": False,
        "failure_reasons": ["probe_exception"],
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }


def build_preflight_failure_summary(
    args: argparse.Namespace,
    exc: PreflightError,
) -> dict[str, Any]:
    summary = build_failure_summary(args, exc)
    summary["failure_reasons"] = list(exc.reasons)
    summary["preflight_rejected"] = True
    return summary


def _install_ipython_display_stub() -> None:
    """Provide the tiny optional display API that mediapy imports on H200."""

    if "IPython.display" in sys.modules:
        return
    ipython_module = sys.modules.get("IPython") or types.ModuleType("IPython")
    display_module = types.ModuleType("IPython.display")

    class _DisplayObject:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def _repr_html_(self) -> str:
            return ""

    def _display(*args: Any, **kwargs: Any) -> None:
        return None

    display_module.display = _display
    display_module.HTML = _DisplayObject
    display_module.Image = _DisplayObject
    display_module.Video = _DisplayObject
    display_module.clear_output = _display
    ipython_module.display = display_module
    sys.modules.setdefault("IPython", ipython_module)
    sys.modules["IPython.display"] = display_module


def _install_wandb_stub() -> None:
    """Avoid importing optional W&B transitive deps during task registration."""

    if "wandb" in sys.modules:
        return
    wandb_module = types.ModuleType("wandb")

    class _Api:
        pass

    def _noop(*args: Any, **kwargs: Any) -> None:
        return None

    wandb_module.run = None
    wandb_module.Api = _Api
    wandb_module.init = _noop
    wandb_module.log = _noop
    wandb_module.finish = _noop
    wandb_module.save = _noop
    wandb_module.login = _noop
    sys.modules["wandb"] = wandb_module


def _install_wcwidth_stub() -> None:
    """Provide width helpers for prettytable in the slim H200 conda env."""

    if "wcwidth" in sys.modules:
        return
    wcwidth_module = types.ModuleType("wcwidth")

    def _wcwidth(char: str) -> int:
        return 0 if not char else 1

    def _wcswidth(text: str) -> int:
        return len(str(text))

    wcwidth_module.wcwidth = _wcwidth
    wcwidth_module.wcswidth = _wcswidth
    sys.modules["wcwidth"] = wcwidth_module


def write_json_summary(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary["json_path"] = str(output)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _algorithm_debug_snapshot(runner: Any) -> dict[str, Any]:
    alg = getattr(runner, "alg", None)
    snapshot = getattr(alg, "task040_sequence_update_debug_snapshot", None)
    if not callable(snapshot):
        return {}
    data = snapshot()
    return dict(data) if isinstance(data, dict) else {}


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
