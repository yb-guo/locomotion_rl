"""Probe Task038 MJLab variant env-load-only task ids."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from collections.abc import Mapping
from numbers import Number
from pathlib import Path
from typing import Any

from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS

DEFAULT_TASK = "Unitree-G1-Gripper-Flat-Task038-TrainAssetSmoke"
DEFAULT_EXPECTED_ACTION_DIM = 31
TRAIN_TASK_ID = "Unitree-G1-Gripper-Flat-Task038-TrainAssetSmoke"
HELDOUT_TASK_ID = "Unitree-G1-Gripper-Flat-Task038-HeldoutAssetSmoke"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal MJLab env-load smoke for Task038 XML variants."
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--variant-label", default=None)
    parser.add_argument("--expected-xml-path", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3800901)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-action-dim", type=int, default=DEFAULT_EXPECTED_ACTION_DIM)
    return parser.parse_args(argv)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import mjlab.tasks as _mjlab_tasks
    import src.tasks as _project_tasks

    del _mjlab_tasks, _project_tasks  # Imports register task packages by side effect.
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    torch.set_grad_enabled(False)

    registered_xml_path, xml_resolution_error = resolve_registered_xml_path(args.task)
    xml_path_matches_expected = xml_path_match(args.expected_xml_path, registered_xml_path)
    env_cfg = _load_env_cfg(load_env_cfg, args.task)
    agent_cfg = load_rl_cfg(args.task)
    _set_if_present(env_cfg, "seed", args.seed)
    if hasattr(getattr(env_cfg, "scene", None), "num_envs"):
        env_cfg.scene.num_envs = args.num_envs

    start = time.time()
    outer_env = None
    zero_step_ok = False
    last_obs: Any = None
    done_count = 0
    reset_count = 0
    action_dim = None
    total_action_dim = None
    try:
        base = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        outer_env = base
        clip_actions = getattr(agent_cfg, "clip_actions", None)
        env = RslRlVecEnvWrapper(base, clip_actions=clip_actions)
        outer_env = env

        reset_result = env.reset()
        last_obs = _first(reset_result)
        action_dim = _action_dim(env, base)
        total_action_dim = _total_action_dim(base) or action_dim
        if action_dim is None:
            raise RuntimeError("could not infer MJLab action dimension")

        for _step in range(args.steps):
            actions = torch.zeros((args.num_envs, int(action_dim)), device=args.device)
            step_result = env.step(actions)
            last_obs = _first(step_result)
            done = _done_from_step(step_result)
            extras = _extras_from_step(step_result)
            done_count += _bool_count(torch, done, args.device)
            reset_count += _reset_count(torch, extras, args.device)
        zero_step_ok = True

        obs_summary = _obs_summary(torch, last_obs)
        finite_obs = all(item["finite"] for item in obs_summary.values()) if obs_summary else False
        result = {
            "task": args.task,
            "variant_label": args.variant_label or _variant_label(args.task),
            "expected_xml_path": args.expected_xml_path,
            "registered_xml_path": registered_xml_path,
            "xml_path_matches_expected": xml_path_matches_expected,
            "xml_resolution_error": xml_resolution_error,
            "command": list(sys.argv),
            "seed": args.seed,
            "device": args.device,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "expected_action_dim": args.expected_action_dim,
            "action_dim": action_dim,
            "total_action_dim": total_action_dim,
            "action_terms": _action_terms(base),
            "obs": obs_summary,
            "obs_all_finite": finite_obs,
            "zero_step_ok": zero_step_ok,
            "done_count": done_count,
            "reset_count": reset_count,
            "wall_time_s": time.time() - start,
        }
        result["pass"], result["failure_reasons"] = evaluate_probe_pass(result)
        return result
    finally:
        if outer_env is not None and hasattr(outer_env, "close"):
            outer_env.close()


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    registered_xml_path, xml_resolution_error = _safe_resolve_registered_xml_path(args.task)
    xml_path_matches_expected = xml_path_match(args.expected_xml_path, registered_xml_path)
    return {
        "task": args.task,
        "variant_label": args.variant_label or _variant_label(args.task),
        "expected_xml_path": args.expected_xml_path,
        "registered_xml_path": registered_xml_path,
        "xml_path_matches_expected": xml_path_matches_expected,
        "xml_resolution_error": xml_resolution_error,
        "command": list(sys.argv),
        "seed": args.seed,
        "device": args.device,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "expected_action_dim": args.expected_action_dim,
        "zero_step_ok": False,
        "pass": False,
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }


def resolve_registered_xml_path(task: str) -> tuple[str | None, str | None]:
    try:
        constants = importlib.import_module(
            "src.assets.robots.unitree_g1_gripper.g1_gripper_constants"
        )
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        return None, repr(exc)

    if task == TRAIN_TASK_ID:
        value = getattr(constants, "TASK038_TRAIN_XML", None)
    elif task == HELDOUT_TASK_ID:
        value = getattr(constants, "TASK038_HELDOUT_XML", None)
    else:
        return None, f"unknown Task038 XML smoke task id: {task}"
    if value is None:
        return None, f"Task038 XML constant missing for task: {task}"
    return str(value), None


def xml_path_match(expected: str | None, registered: str | None) -> bool | None:
    if registered is None:
        return False
    if expected is None:
        return None
    return _normalize_path(expected) == _normalize_path(registered)


def evaluate_probe_pass(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    expected_action_dim = int(summary["expected_action_dim"])
    if not summary.get("registered_xml_path"):
        reasons.append("registered_xml_path_unavailable")
    if summary.get("expected_xml_path") is not None and not summary.get("xml_path_matches_expected"):
        reasons.append("expected_xml_path_mismatch")
    if not summary.get("zero_step_ok"):
        reasons.append("zero_step_failed")
    if int(summary.get("action_dim") or -1) != expected_action_dim:
        reasons.append("action_dim_mismatch")
    if int(summary.get("total_action_dim") or -1) != expected_action_dim:
        reasons.append("total_action_dim_mismatch")
    if not summary.get("obs"):
        reasons.append("obs_summary_empty")
    elif not summary.get("obs_all_finite"):
        reasons.append("obs_not_finite")
    return not reasons, reasons


def _safe_resolve_registered_xml_path(task: str) -> tuple[str | None, str | None]:
    try:
        return resolve_registered_xml_path(task)
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        return None, repr(exc)


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


def _set_if_present(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _first(value: Any) -> Any:
    if isinstance(value, tuple):
        return value[0]
    return value


def _done_from_step(step_result: Any) -> Any:
    if not isinstance(step_result, tuple):
        return None
    if len(step_result) >= 4:
        return step_result[2]
    return None


def _extras_from_step(step_result: Any) -> Any:
    if isinstance(step_result, tuple) and step_result:
        maybe = step_result[-1]
        if isinstance(maybe, dict):
            return maybe
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


def _action_terms(base: Any) -> list[dict[str, Any]]:
    manager = getattr(base, "action_manager", None)
    if manager is None:
        return []
    terms = getattr(manager, "_terms", None) or getattr(manager, "terms", None) or {}
    if isinstance(terms, dict):
        iterator = terms.items()
    else:
        iterator = ((getattr(term, "name", str(index)), term) for index, term in enumerate(terms or ()))
    out = []
    for name, term in iterator:
        dim = None
        for attr in ("action_dim", "num_actions", "total_action_dim"):
            value = getattr(term, attr, None)
            if isinstance(value, int):
                dim = value
                break
        out.append({"name": str(name), "dim": dim, "type": type(term).__name__})
    return out


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
            "shape": list(tensor.shape),
            "finite": bool(torch.isfinite(tensor.detach().float()).all().item()),
        }
    return out


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


def _bool_count(torch: Any, value: Any, device: str) -> int:
    if value is None:
        return 0
    tensor = value if hasattr(value, "to") else torch.as_tensor(value, device=device)
    return int(tensor.to(device=device, dtype=torch.bool).sum().item())


def _reset_count(torch: Any, extras: dict[str, Any], device: str) -> int:
    for key in ("reset", "resets", "reset_buf", "episode_done", "time_outs"):
        if key in extras:
            return _bool_count(torch, extras[key], device)
    return 0


def _normalize_path(path: str) -> str:
    return str(Path(path).expanduser())


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
