"""Load a Task038 true-TXL checkpoint and run a tiny policy rollout smoke."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from h200_locomotion_lab.tools.task038_true_txl_runner_smoke_probe import (
    DEFAULT_EXPECTED_ACTION_DIM,
    DEFAULT_EXPECTED_ACTOR_MODEL_CLASS,
    DEFAULT_EXPECTED_RUNNER_CLS,
    HELDOUT_TRUE_TXL_RUNNER_SMOKE_TASK_ID,
    TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID,
    _action_dim,
    _actor_model,
    _actor_model_class,
    _agent_cfg_as_dict,
    _extras_from_reset,
    _extras_from_step,
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
    _variant_label,
)


DEFAULT_TASK = TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID
ALLOWED_TASKS = (
    TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID,
    HELDOUT_TRUE_TXL_RUNNER_SMOKE_TASK_ID,
)


class PreflightError(ValueError):
    def __init__(self, reasons: list[str]) -> None:
        super().__init__(", ".join(reasons))
        self.reasons = reasons


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a Task038 true-TXL checkpoint and run a short policy rollout "
            "smoke. This is eval-load/rollout plumbing only, not quality eval."
        )
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3801501)
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
    if args.task not in ALLOWED_TASKS:
        reasons.append("task_not_true_txl_runner_smoke")
    if int(args.num_envs) <= 0:
        reasons.append("num_envs_not_positive")
    if int(args.steps) <= 0:
        reasons.append("steps_not_positive")
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.exists():
        reasons.append("checkpoint_missing")
    if reasons:
        raise PreflightError(reasons)


def _optional_txl_debug_snapshot(actor_model: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return _txl_debug_snapshot(actor_model), None
    except Exception as exc:
        return None, repr(exc)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    preflight_args(args)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import torch
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    torch.set_grad_enabled(False)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    env_cfg = _load_env_cfg(load_env_cfg, args.task)
    agent_cfg = load_rl_cfg(args.task)
    _set_if_present(env_cfg, "seed", args.seed)
    if hasattr(getattr(env_cfg, "scene", None), "num_envs"):
        env_cfg.scene.num_envs = args.num_envs

    start = time.time()
    outer_env = None
    try:
        base = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        clip_actions = getattr(agent_cfg, "clip_actions", None)
        outer_env = RslRlVecEnvWrapper(base, clip_actions=clip_actions)
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        runner_cls_name = getattr(runner_cls, "__name__", type(runner_cls).__name__)
        runner = runner_cls(outer_env, _agent_cfg_as_dict(agent_cfg), device=args.device)
        rollout_env = runner.env

        action_dim = _action_dim(rollout_env, base)
        total_action_dim = _total_action_dim(base) or action_dim
        actual_num_envs = int(getattr(rollout_env, "num_envs", args.num_envs))
        load_infos = runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        load_infos_keys = sorted(load_infos) if isinstance(load_infos, dict) else []
        policy, policy_error = _get_policy(runner, args.device)
        actor_model = _actor_model(runner)
        txl_debug_before, txl_debug_before_error = _optional_txl_debug_snapshot(
            actor_model
        )

        reset_result = rollout_env.reset()
        obs = _first(reset_result)
        reset_extras = _extras_from_reset(reset_result)
        last_extra_keys = sorted(reset_extras)
        policy_action_shape = None
        policy_action_finite = False
        step_count = 0

        if policy is not None:
            for _ in range(args.steps):
                try:
                    action = policy(obs)
                    policy_action_shape = _shape(action)
                    policy_action_finite = _finite(torch, action)
                except Exception as exc:
                    policy_error = repr(exc)
                    break
                step_result = rollout_env.step(action)
                step_count += 1
                obs = _first(step_result)
                last_extra_keys = sorted(_extras_from_step(step_result))

        obs_summary = _obs_summary(torch, obs)
        txl_debug_after, txl_debug_after_error = _optional_txl_debug_snapshot(
            actor_model
        )
        summary = {
            "task": args.task,
            "variant_label": _variant_label(args.task),
            "checkpoint": str(checkpoint),
            "checkpoint_exists": checkpoint.exists(),
            "command": list(sys.argv),
            "seed": args.seed,
            "device": args.device,
            "num_envs": args.num_envs,
            "actual_num_envs": actual_num_envs,
            "steps": args.steps,
            "step_count": step_count,
            "expected_action_dim": args.expected_action_dim,
            "expected_runner_cls": args.expected_runner_cls,
            "expected_actor_model_class": args.expected_actor_model_class,
            "runner_cls": runner_cls_name,
            "actor_model_class": _actor_model_class(runner),
            "action_dim": action_dim,
            "total_action_dim": total_action_dim,
            "load_returned": True,
            "load_infos_keys": load_infos_keys,
            "policy_action_shape": policy_action_shape,
            "policy_action_finite": policy_action_finite,
            "policy_error": policy_error,
            "obs": obs_summary,
            "obs_all_finite": _obs_all_finite(obs_summary),
            "last_extra_keys": last_extra_keys,
            "txl_debug_before": txl_debug_before,
            "txl_debug_before_error": txl_debug_before_error,
            "txl_debug_after": txl_debug_after,
            "txl_debug_after_error": txl_debug_after_error,
            "quality_claim": False,
            "training_claim": False,
            "eval_claim": False,
            "reproduction_claim": False,
            "superiority_claim": False,
            "checkpoint_eval_load_smoke_only": True,
            "wall_time_s": time.time() - start,
        }
        summary["pass"], summary["failure_reasons"] = evaluate_probe_pass(summary)
        return summary
    finally:
        if outer_env is not None and hasattr(outer_env, "close"):
            outer_env.close()


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser()
    return {
        "task": args.task,
        "variant_label": _variant_label(args.task),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "command": list(sys.argv),
        "seed": args.seed,
        "device": args.device,
        "num_envs": args.num_envs,
        "actual_num_envs": None,
        "steps": args.steps,
        "step_count": 0,
        "expected_action_dim": args.expected_action_dim,
        "expected_runner_cls": args.expected_runner_cls,
        "expected_actor_model_class": args.expected_actor_model_class,
        "load_returned": False,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "checkpoint_eval_load_smoke_only": True,
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
    if summary.get("task") not in ALLOWED_TASKS:
        reasons.append("task_not_true_txl_runner_smoke")
    if not summary.get("checkpoint_exists"):
        reasons.append("checkpoint_missing")
    if summary.get("runner_cls") != summary.get("expected_runner_cls"):
        reasons.append("runner_cls_mismatch")
    if summary.get("actor_model_class") != summary.get("expected_actor_model_class"):
        reasons.append("actor_model_class_mismatch")
    expected_action_dim = int(summary.get("expected_action_dim") or -1)
    if int(summary.get("action_dim") or -1) != expected_action_dim:
        reasons.append("action_dim_mismatch")
    if int(summary.get("total_action_dim") or -1) != expected_action_dim:
        reasons.append("total_action_dim_mismatch")
    if not summary.get("load_returned"):
        reasons.append("load_not_returned")
    if summary.get("policy_error") is not None:
        reasons.append("policy_error")
    if not summary.get("policy_action_finite"):
        reasons.append("policy_action_not_finite")
    if not summary.get("policy_action_shape"):
        reasons.append("policy_action_shape_missing")
    elif not _policy_action_shape_matches(summary, expected_action_dim):
        reasons.append("policy_action_shape_mismatch")
    if int(summary.get("steps") or 0) <= 0 or int(summary.get("step_count") or 0) <= 0:
        reasons.append("no_policy_steps_executed")
    if not summary.get("obs"):
        reasons.append("obs_summary_empty")
    elif not summary.get("obs_all_finite"):
        reasons.append("obs_not_finite")
    if summary.get("wall_time_s") is None:
        reasons.append("wall_time_missing")
    if (
        summary.get("quality_claim")
        or summary.get("training_claim")
        or summary.get("eval_claim")
        or summary.get("reproduction_claim")
        or summary.get("superiority_claim")
        or not summary.get("checkpoint_eval_load_smoke_only")
    ):
        reasons.append("claim_boundary_violation")
    return not reasons, reasons


def write_json_summary(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary["json_path"] = str(output)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        preflight_args(args)
        summary = run_probe(args)
    except PreflightError as exc:
        summary = build_preflight_failure_summary(args, exc)
    except Exception as exc:
        summary = build_failure_summary(args, exc)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
