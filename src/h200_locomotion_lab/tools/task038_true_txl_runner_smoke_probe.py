"""Probe Task038 true-TXL runner construction and cache-consuming policy forwards."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from numbers import Number
from pathlib import Path
from typing import Any

from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS

TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke"
)
HELDOUT_TRUE_TXL_RUNNER_SMOKE_TASK_ID = (
    "Unitree-G1-Gripper-Flat-Task038-HeldoutTrueTxlRunnerSmoke"
)
DEFAULT_TASK = TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID
DEFAULT_EXPECTED_ACTION_DIM = 31
DEFAULT_EXPECTED_RUNNER_CLS = "Task038TrueTxlMemoryK160Runner"
DEFAULT_EXPECTED_ACTOR_MODEL_CLASS = "Task038TrueTxlMemoryModel"

REQUIRED_EXTRA_KEYS = (
    "trial_done",
    "episode_done",
    "trial_index",
    "final_trial",
    "reset_reason",
    "task037_trial_done",
    "task037_episode_done",
    "task037_trial_index",
    "task037_final_trial",
    "task037_reset_reason",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal Task038 true-TXL runner-consumer smoke."
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--variant-label", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3801101)
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

    import mjlab.tasks as _mjlab_tasks
    import src.tasks as _project_tasks

    del _mjlab_tasks, _project_tasks  # Imports register task packages by side effect.
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    torch.set_grad_enabled(False)

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

        reset_result = rollout_env.reset()
        obs = _first(reset_result)
        reset_extras = _extras_from_reset(reset_result)
        required_seen = {key: key in reset_extras for key in REQUIRED_EXTRA_KEYS}

        action_dim = _action_dim(rollout_env, base)
        total_action_dim = _total_action_dim(base) or action_dim
        actual_num_envs = int(getattr(rollout_env, "num_envs", args.num_envs))
        policy, policy_error = _get_policy(runner, args.device)
        actor_model = _actor_model(runner)
        policy_action = None
        policy_action_shape = None
        policy_action_finite = False
        txl_debug = {}
        if policy is not None:
            policy_action = policy(obs)
            policy_action = policy(obs)
            policy_action_shape = _shape(policy_action)
            policy_action_finite = _finite(torch, policy_action)
            txl_debug = _txl_debug_snapshot(actor_model)

        step_count = 0
        last_extra_keys = sorted(reset_extras)
        step_required_seen = {key: False for key in REQUIRED_EXTRA_KEYS}
        for _step in range(args.steps):
            actions = torch.zeros(
                (actual_num_envs, int(action_dim)),
                device=args.device,
            )
            step_result = rollout_env.step(actions)
            step_count += 1
            obs = _first(step_result)
            extras = _extras_from_step(step_result)
            last_extra_keys = sorted(extras)
            for key in REQUIRED_EXTRA_KEYS:
                required_seen[key] = required_seen[key] or key in extras
                step_required_seen[key] = step_required_seen[key] or key in extras

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
            "txl_debug": txl_debug,
            "quality_claim": False,
            "training_claim": False,
            "eval_claim": False,
            "runner_smoke_only": True,
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
        "actual_num_envs": None,
        "steps": args.steps,
        "step_count": 0,
        "expected_action_dim": args.expected_action_dim,
        "expected_runner_cls": args.expected_runner_cls,
        "expected_actor_model_class": args.expected_actor_model_class,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "runner_smoke_only": True,
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
    if int(summary.get("steps") or 0) <= 0 or int(summary.get("step_count") or 0) <= 0:
        reasons.append("no_steps_executed")
    if summary.get("required_extras_missing"):
        reasons.append("required_extras_missing")
    if summary.get("step_required_extras_missing"):
        reasons.append("step_required_extras_missing")
    if not summary.get("obs"):
        reasons.append("obs_summary_empty")
    elif not summary.get("obs_all_finite"):
        reasons.append("obs_not_finite")
    if not _txl_debug_envs_valid(summary.get("txl_debug")):
        reasons.append("txl_debug_invalid")
    if not _txl_debug_has_incremental_steps(summary.get("txl_debug")):
        reasons.append("txl_debug_no_incremental_steps")
    if not _txl_debug_has_previous_memory_exposure(summary.get("txl_debug")):
        reasons.append("txl_debug_no_previous_memory_exposure")
    if (
        summary.get("quality_claim")
        or summary.get("training_claim")
        or summary.get("eval_claim")
        or not summary.get("runner_smoke_only")
    ):
        reasons.append("claim_boundary_violation")
    return not reasons, reasons


def _policy_action_shape_matches(summary: dict[str, Any], expected_action_dim: int) -> bool:
    shape = summary.get("policy_action_shape")
    if not isinstance(shape, list) or len(shape) != 2:
        return False
    actual_num_envs = summary.get("actual_num_envs")
    if not isinstance(actual_num_envs, int):
        return False
    return int(shape[0]) == actual_num_envs and int(shape[-1]) == expected_action_dim


def _txl_debug_envs_valid(debug: Any) -> bool:
    if not isinstance(debug, Mapping):
        return False
    envs = debug.get("envs")
    if not isinstance(envs, list) or not envs:
        return False
    required = {
        "env_id",
        "memory_lengths",
        "inner_reset_events",
        "outer_reset_events",
        "incremental_steps",
    }
    for env in envs:
        if not isinstance(env, Mapping) or not required.issubset(env):
            return False
        if not isinstance(env["memory_lengths"], list) or not env["memory_lengths"]:
            return False
    return True


def _txl_debug_has_incremental_steps(debug: Any) -> bool:
    if not isinstance(debug, Mapping):
        return False
    envs = debug.get("envs")
    if not isinstance(envs, list):
        return False
    return any(int(env.get("incremental_steps") or 0) > 0 for env in envs if isinstance(env, Mapping))


def _txl_debug_has_previous_memory_exposure(debug: Any) -> bool:
    if not isinstance(debug, Mapping):
        return False
    envs = debug.get("envs")
    if isinstance(envs, list):
        for env in envs:
            if not isinstance(env, Mapping):
                continue
            lengths = env.get("last_attended_previous_memory_lengths")
            if _has_positive_int(lengths):
                return True
    return _has_positive_int(debug.get("last_attended_previous_memory_lengths"))


def _has_positive_int(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    if isinstance(value, (list, tuple)):
        return any(_has_positive_int(item) for item in value)
    return False


def write_json_summary(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary["json_path"] = str(output)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_env_cfg(load_env_cfg: Any, task: str) -> Any:
    try:
        return load_env_cfg(task, play=True)
    except TypeError:
        return load_env_cfg(task)


def _agent_cfg_as_dict(agent_cfg: Any) -> dict[str, Any]:
    if is_dataclass(agent_cfg):
        return asdict(agent_cfg)
    if isinstance(agent_cfg, Mapping):
        return dict(agent_cfg)
    return dict(vars(agent_cfg))


def _set_if_present(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _get_policy(runner: Any, device: str) -> tuple[Any | None, str | None]:
    get_policy = getattr(runner, "get_inference_policy", None)
    if not callable(get_policy):
        return None, "runner has no get_inference_policy"
    try:
        policy = get_policy(device=device)
        eval_fn = getattr(policy, "eval", None)
        if callable(eval_fn):
            eval_fn()
        return policy, None
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        return None, repr(exc)


def _actor_model(runner: Any) -> Any | None:
    for path in (
        ("alg", "actor_critic", "actor"),
        ("alg", "actor_critic"),
        ("actor_critic", "actor"),
        ("actor_critic",),
        ("policy",),
        ("alg", "actor"),
    ):
        obj = runner
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _actor_model_class(runner: Any) -> str | None:
    actor = _actor_model(runner)
    if actor is None:
        return None
    return type(actor).__name__


def _txl_debug_snapshot(actor_model: Any) -> dict[str, Any]:
    snapshot = getattr(actor_model, "txl_debug_snapshot", None)
    if not callable(snapshot):
        return {}
    data = snapshot()
    return dict(data) if isinstance(data, Mapping) else {}


def _first(value: Any) -> Any:
    if isinstance(value, tuple):
        return value[0]
    return value


def _extras_from_reset(reset_result: Any) -> dict[str, Any]:
    if isinstance(reset_result, tuple) and len(reset_result) >= 2 and isinstance(reset_result[1], dict):
        return reset_result[1]
    return {}


def _extras_from_step(step_result: Any) -> dict[str, Any]:
    if isinstance(step_result, tuple) and step_result and isinstance(step_result[-1], dict):
        return step_result[-1]
    return {}


def _action_dim(env: Any, base: Any) -> int | None:
    for obj in (env, base, getattr(base, "action_manager", None)):
        for name in ("num_actions", "action_dim", "total_action_dim"):
            value = getattr(obj, name, None)
            if isinstance(value, int):
                return value
    return None


def _total_action_dim(base: Any) -> int | None:
    manager = getattr(base, "action_manager", None)
    for name in ("total_action_dim", "action_dim", "num_actions"):
        value = getattr(manager, name, None)
        if isinstance(value, int):
            return value
    return None


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _finite(torch: Any, value: Any) -> bool:
    try:
        return bool(torch.isfinite(value.detach().float()).all().item())
    except RECOVERABLE_RUNTIME_ERRORS:
        return False


def _obs_summary(torch: Any, obs: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    mapping_items = _mapping_items(obs)
    if mapping_items is not None:
        for key, value in mapping_items:
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(_obs_summary(torch, value, name))
        return out

    tensor = _obs_leaf_tensor(torch, obs)
    if tensor is not None:
        out[prefix or "obs"] = {
            "shape": _shape(tensor) or [],
            "finite": _finite(torch, tensor),
        }
    return out


def _obs_all_finite(obs_summary: dict[str, dict[str, Any]]) -> bool:
    return bool(obs_summary) and all(item["finite"] for item in obs_summary.values())


def _mapping_items(value: Any) -> list[tuple[Any, Any]] | None:
    if isinstance(value, Mapping):
        return list(value.items())
    items = getattr(value, "items", None)
    keys = getattr(value, "keys", None)
    if callable(items) and callable(keys):
        try:
            return list(items())
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            pass
    getitem = getattr(value, "__getitem__", None)
    if callable(keys) and callable(getitem):
        try:
            return [(key, value[key]) for key in keys()]
        except (AttributeError, KeyError, IndexError, NotImplementedError, TypeError, ValueError):
            return None
    return None


def _obs_leaf_tensor(torch: Any, obs: Any) -> Any | None:
    if obs is None or isinstance(obs, (str, bytes, bytearray)):
        return None
    is_tensor = getattr(torch, "is_tensor", None)
    if callable(is_tensor):
        try:
            if is_tensor(obs):
                return obs
        except TypeError:
            pass
    if isinstance(obs, Number) or hasattr(obs, "__array__") or isinstance(obs, (list, tuple)):
        try:
            return torch.as_tensor(obs)
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _variant_label(task: str) -> str:
    lowered = task.lower()
    if "heldout" in lowered:
        return "heldout"
    if "train" in lowered:
        return "train"
    return "unknown"


def main() -> None:
    args = parse_args()
    try:
        summary = run_probe(args)
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        summary = build_failure_summary(args, exc)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
