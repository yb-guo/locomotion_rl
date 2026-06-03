"""Probe Task038 true-TXL reset-hook integration without training."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from h200_locomotion_lab.tools.task038_true_txl_runner_smoke_probe import (
    DEFAULT_EXPECTED_ACTION_DIM,
    DEFAULT_EXPECTED_ACTOR_MODEL_CLASS,
    DEFAULT_EXPECTED_RUNNER_CLS,
    DEFAULT_TASK,
    REQUIRED_EXTRA_KEYS,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a no-training Task038 true-TXL reset-hook integration smoke."
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--variant-label", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--episode-length-s", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3801201)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-action-dim", type=int, default=DEFAULT_EXPECTED_ACTION_DIM)
    parser.add_argument("--expected-runner-cls", default=DEFAULT_EXPECTED_RUNNER_CLS)
    parser.add_argument(
        "--expected-actor-model-class",
        default=DEFAULT_EXPECTED_ACTOR_MODEL_CLASS,
    )
    return parser.parse_args(argv)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
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

    env_cfg = _load_env_cfg(load_env_cfg, args.task)
    agent_cfg = load_rl_cfg(args.task)
    _set_if_present(env_cfg, "seed", args.seed)
    _set_episode_length_s(env_cfg, args.episode_length_s)
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

        reset_result = rollout_env.reset()
        obs = _first(reset_result)
        reset_extras = _extras_from_reset(reset_result)
        required_seen = {key: key in reset_extras for key in REQUIRED_EXTRA_KEYS}
        step_required_seen = {key: False for key in REQUIRED_EXTRA_KEYS}

        action_dim = _action_dim(rollout_env, base)
        total_action_dim = _total_action_dim(base) or action_dim
        actual_num_envs = int(getattr(rollout_env, "num_envs", args.num_envs))
        policy, policy_error = _get_policy(runner, args.device)
        actor_model = _actor_model(runner)

        saw_inner_reset = False
        saw_outer_reset = False
        inner_preserved = False
        outer_cleared = False
        inner_reset_examples: list[dict[str, Any]] = []
        outer_reset_examples: list[dict[str, Any]] = []
        policy_action_shape = None
        policy_action_finite = False
        last_extra_keys = sorted(reset_extras)
        step_count = 0

        if policy is not None:
            for _step in range(args.steps):
                pre_action = policy(obs)
                policy_action_shape = _shape(pre_action)
                policy_action_finite = _finite(torch, pre_action)
                pre_debug = _txl_debug_snapshot(actor_model)
                step_result = rollout_env.step(pre_action)
                step_count += 1
                obs = _first(step_result)
                extras = _extras_from_step(step_result)
                last_extra_keys = sorted(extras)
                for key in REQUIRED_EXTRA_KEYS:
                    required_seen[key] = required_seen[key] or key in extras
                    step_required_seen[key] = step_required_seen[key] or key in extras

                post_debug = _txl_debug_snapshot(actor_model)
                inner_ids = _reset_ids_from_extras(
                    torch,
                    extras,
                    ("inner_reset", "task037_inner_reset"),
                    args.device,
                    actual_num_envs,
                )
                outer_ids = _reset_ids_from_extras(
                    torch,
                    extras,
                    ("outer_reset", "episode_done", "task037_outer_reset", "task037_episode_done"),
                    args.device,
                    actual_num_envs,
                )
                if inner_ids:
                    saw_inner_reset = True
                    for env_id in inner_ids:
                        before = _memory_lengths_for_env(pre_debug, env_id)
                        after = _memory_lengths_for_env(post_debug, env_id)
                        example = {"env_id": env_id, "before": before, "after": after}
                        inner_reset_examples.append(example)
                        if before and before == after and _has_positive_int(before):
                            inner_preserved = True
                if outer_ids:
                    saw_outer_reset = True
                    for env_id in outer_ids:
                        before = _memory_lengths_for_env(pre_debug, env_id)
                        after = _memory_lengths_for_env(post_debug, env_id)
                        example = {"env_id": env_id, "before": before, "after": after}
                        outer_reset_examples.append(example)
                        if before and after and all(int(value) == 0 for value in after):
                            outer_cleared = True
                if inner_preserved and outer_cleared:
                    break

        txl_debug = _txl_debug_snapshot(actor_model)
        obs_summary = _obs_summary(torch, obs)
        summary = {
            "task": args.task,
            "variant_label": args.variant_label or _variant_label(args.task),
            "command": list(sys.argv),
            "seed": args.seed,
            "device": args.device,
            "num_envs": args.num_envs,
            "actual_num_envs": actual_num_envs,
            "steps": args.steps,
            "episode_length_s": args.episode_length_s,
            "step_count": step_count,
            "expected_action_dim": args.expected_action_dim,
            "expected_runner_cls": args.expected_runner_cls,
            "expected_actor_model_class": args.expected_actor_model_class,
            "runner_cls": runner_cls_name,
            "actor_model_class": _actor_model_class(runner),
            "action_dim": action_dim,
            "total_action_dim": total_action_dim,
            "policy_action_shape": policy_action_shape,
            "policy_action_finite": policy_action_finite,
            "policy_error": policy_error,
            "obs": obs_summary,
            "obs_all_finite": _obs_all_finite(obs_summary),
            "required_extras_missing": [
                key for key, seen in required_seen.items() if not seen
            ],
            "step_required_extras_missing": [
                key for key, seen in step_required_seen.items() if not seen
            ],
            "last_extra_keys": last_extra_keys,
            "saw_inner_reset": saw_inner_reset,
            "saw_outer_reset": saw_outer_reset,
            "inner_reset_preserved_memory_before_next_policy": inner_preserved,
            "outer_reset_cleared_memory_before_next_policy": outer_cleared,
            "inner_reset_examples": inner_reset_examples[:8],
            "outer_reset_examples": outer_reset_examples[:8],
            "actor_inner_reset_events_total": _debug_event_total(txl_debug, "inner_reset_events"),
            "actor_outer_reset_events_total": _debug_event_total(txl_debug, "outer_reset_events"),
            "txl_debug": txl_debug,
            "quality_claim": False,
            "training_claim": False,
            "eval_claim": False,
            "reproduction_claim": False,
            "superiority_claim": False,
            "reset_hook_integration_smoke_only": True,
            "wall_time_s": time.time() - start,
        }
        summary["pass"], summary["failure_reasons"] = evaluate_probe_pass(summary)
        return summary
    finally:
        if outer_env is not None and hasattr(outer_env, "close"):
            outer_env.close()


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    return {
        "task": args.task,
        "variant_label": args.variant_label or _variant_label(args.task),
        "command": list(sys.argv),
        "seed": args.seed,
        "device": args.device,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "episode_length_s": args.episode_length_s,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "reset_hook_integration_smoke_only": True,
        "pass": False,
        "failure_reasons": ["probe_exception"],
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }


def evaluate_probe_pass(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary.get("runner_cls") != summary.get("expected_runner_cls"):
        reasons.append("runner_cls_mismatch")
    if summary.get("actor_model_class") != summary.get("expected_actor_model_class"):
        reasons.append("actor_model_class_mismatch")
    expected_action_dim = int(summary.get("expected_action_dim") or -1)
    if int(summary.get("action_dim") or -1) != expected_action_dim:
        reasons.append("action_dim_mismatch")
    if int(summary.get("total_action_dim") or -1) != expected_action_dim:
        reasons.append("total_action_dim_mismatch")
    if not summary.get("policy_action_finite"):
        reasons.append("policy_action_not_finite")
    if not summary.get("policy_action_shape"):
        reasons.append("policy_action_shape_missing")
    elif not _policy_action_shape_matches(summary, expected_action_dim):
        reasons.append("policy_action_shape_mismatch")
    if int(summary.get("step_count") or 0) <= 0:
        reasons.append("no_steps_executed")
    if summary.get("required_extras_missing"):
        reasons.append("required_extras_missing")
    if summary.get("step_required_extras_missing"):
        reasons.append("step_required_extras_missing")
    if not summary.get("obs"):
        reasons.append("obs_summary_empty")
    elif not summary.get("obs_all_finite"):
        reasons.append("obs_not_finite")
    if not summary.get("saw_inner_reset"):
        reasons.append("inner_reset_not_seen")
    if not summary.get("saw_outer_reset"):
        reasons.append("outer_reset_not_seen")
    if int(summary.get("actor_inner_reset_events_total") or 0) <= 0:
        reasons.append("actor_inner_reset_event_missing")
    if int(summary.get("actor_outer_reset_events_total") or 0) <= 0:
        reasons.append("actor_outer_reset_event_missing")
    if not summary.get("inner_reset_preserved_memory_before_next_policy"):
        reasons.append("inner_reset_memory_not_preserved")
    if not summary.get("outer_reset_cleared_memory_before_next_policy"):
        reasons.append("outer_reset_memory_not_cleared")
    if (
        summary.get("quality_claim")
        or summary.get("training_claim")
        or summary.get("eval_claim")
        or summary.get("reproduction_claim")
        or summary.get("superiority_claim")
        or not summary.get("reset_hook_integration_smoke_only")
    ):
        reasons.append("claim_boundary_violation")
    return not reasons, reasons


def write_json_summary(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary["json_path"] = str(output)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reset_ids_from_extras(
    torch: Any,
    extras: Mapping[str, Any],
    keys: tuple[str, ...],
    device: str,
    num_envs: int,
) -> list[int]:
    mask = None
    for key in keys:
        if key not in extras:
            continue
        key_mask = _bool_tensor(torch, extras[key], device)
        if tuple(key_mask.shape) != (num_envs,):
            continue
        mask = key_mask if mask is None else mask | key_mask
    if mask is None:
        return []
    return [int(value) for value in mask.nonzero(as_tuple=False).flatten().detach().cpu().tolist()]


def _bool_tensor(torch: Any, value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device=device, dtype=torch.bool)
    return torch.as_tensor(value, device=device, dtype=torch.bool)


def _memory_lengths_for_env(debug: Any, env_id: int) -> list[int]:
    if not isinstance(debug, Mapping):
        return []
    envs = debug.get("envs")
    if not isinstance(envs, list):
        return []
    for env in envs:
        if isinstance(env, Mapping) and int(env.get("env_id", -1)) == env_id:
            lengths = env.get("memory_lengths")
            if isinstance(lengths, list):
                return [int(value) for value in lengths]
    return []


def _debug_event_total(debug: Any, key: str) -> int:
    if not isinstance(debug, Mapping):
        return 0
    envs = debug.get("envs")
    if not isinstance(envs, list):
        return 0
    return sum(int(env.get(key) or 0) for env in envs if isinstance(env, Mapping))


def _has_positive_int(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    if isinstance(value, (list, tuple)):
        return any(_has_positive_int(item) for item in value)
    return False


def _set_episode_length_s(env_cfg: Any, episode_length_s: float) -> None:
    if episode_length_s <= 0:
        return
    for obj in (
        env_cfg,
        getattr(env_cfg, "env", None),
        getattr(env_cfg, "episode", None),
        getattr(env_cfg, "terminations", None),
    ):
        if obj is not None and hasattr(obj, "episode_length_s"):
            setattr(obj, "episode_length_s", episode_length_s)


def main() -> None:
    args = parse_args()
    try:
        summary = run_probe(args)
    except Exception as exc:
        summary = build_failure_summary(args, exc)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
