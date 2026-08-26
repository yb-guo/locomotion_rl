"""Wrap the Task038 true-TXL PPO smoke with Task039 memory-update diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.tools import task038_true_txl_ppo_update_smoke
from h200_locomotion_lab.tools.task038_true_txl_runner_smoke_probe import (
    DEFAULT_EXPECTED_ACTION_DIM,
    DEFAULT_EXPECTED_ACTOR_MODEL_CLASS,
    DEFAULT_EXPECTED_RUNNER_CLS,
    TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID,
)

DEFAULT_TASK = TRAIN_TRUE_TXL_RUNNER_SMOKE_TASK_ID
DEFAULT_LOG_DIR = Path("outputs/task039/txl_update_memory_diagnostics")
NO_OVERCLAIM_FIELDS = (
    "quality_claim",
    "training_claim",
    "eval_claim",
    "reproduction_claim",
    "superiority_claim",
)
INTERPRETATION_HIGH_FALLBACK = (
    "High stateless fallback ratio means this diagnostic does not support a "
    "long-memory training claim; sequence-aware TXL PPO update remains required."
)


class PreflightError(ValueError):
    def __init__(self, reasons: list[str]) -> None:
        super().__init__(", ".join(reasons))
        self.reasons = reasons


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Task039 TXL update-memory diagnostics by wrapping the Task038 "
            "train-only true-TXL PPO update smoke. This records instrumentation "
            "evidence only, not quality, training, eval, reproduction, or "
            "superiority evidence."
        )
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3900401)
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
    if int(args.num_envs) <= 0:
        reasons.append("num_envs_not_positive")
    if int(args.rollout_steps) <= 0:
        reasons.append("rollout_steps_not_positive")
    if int(args.expected_action_dim) <= 0:
        reasons.append("expected_action_dim_not_positive")
    if reasons:
        raise PreflightError(reasons)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    preflight_args(args)
    task038_summary = task038_true_txl_ppo_update_smoke.run_probe(_task038_args(args))
    return wrap_task038_result(args, task038_summary)


def wrap_task038_result(
    args: argparse.Namespace, task038_summary: Mapping[str, Any]
) -> dict[str, Any]:
    wrapped = dict(task038_summary)
    wrapped["task039_txl_update_memory_diagnostics_only"] = True
    wrapped["diagnostic_kind"] = "task039_txl_update_memory_diagnostics"
    wrapped["train_variant_only"] = True
    wrapped["heldout_variant_included"] = False
    wrapped["task038_ppo_update_smoke_pass"] = bool(task038_summary.get("pass", False))
    wrapped["task038_ppo_update_smoke_failure_reasons"] = list(
        _as_list(task038_summary.get("failure_reasons"))
    )
    wrapped["expected_runner_cls"] = args.expected_runner_cls
    wrapped["expected_actor_model_class"] = args.expected_actor_model_class
    wrapped["expected_action_dim"] = args.expected_action_dim
    for field in NO_OVERCLAIM_FIELDS:
        wrapped[field] = False

    diagnostic = build_update_memory_diagnostic_summary(wrapped)
    wrapped.update(diagnostic)
    wrapped["task039_update_memory_debug_summary"] = diagnostic
    passed, reasons = evaluate_diagnostic_pass(wrapped)
    wrapped["diagnostic_evidence_valid"] = passed
    wrapped["pass"] = passed
    wrapped["failure_reasons"] = reasons
    return wrapped


def build_update_memory_diagnostic_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    debug = _debug_mapping(summary)
    counter_summary = _forward_counter_summary(debug)
    memory_summary = _memory_length_observation_summary(summary)
    temporal = _temporal_segment_summary(counter_summary)
    fallback_batches = counter_summary["stateless_fallback_forward_batches"]
    fallback_samples = counter_summary["stateless_fallback_forward_samples"]
    fallback_present = _positive_int(fallback_batches) or _positive_int(fallback_samples)
    counters_present = bool(counter_summary["forward_counters_present"])
    long_memory_supported = (
        counters_present
        and not fallback_present
        and temporal["minibatches_preserve_temporal_segments"] is True
    )
    router_decision = (
        "sequence_aware_txl_ppo_update_required_next"
        if fallback_present
        else "diagnostic_inconclusive_without_minibatch_segment_metadata"
    )
    interpretation = (
        INTERPRETATION_HIGH_FALLBACK
        if fallback_present
        else (
            "No stateless fallback was observed, but this wrapper still needs "
            "explicit temporal minibatch metadata before making a long-memory "
            "training claim."
        )
    )

    return {
        **counter_summary,
        **memory_summary,
        **temporal,
        "stateless_fallback_present": fallback_present,
        "long_memory_training_claim_supported": long_memory_supported,
        "router_decision": router_decision,
        "interpretation": interpretation,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
    }


def evaluate_diagnostic_pass(summary: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary.get("task") != DEFAULT_TASK:
        reasons.append("task_not_train_true_txl_runner_smoke")
    if not summary.get("train_variant_only") or summary.get("heldout_variant_included"):
        reasons.append("variant_boundary_violation")
    if not summary.get("task039_txl_update_memory_diagnostics_only"):
        reasons.append("diagnostic_wrapper_flag_missing")
    if not summary.get("task038_ppo_update_smoke_pass"):
        reasons.append("task038_ppo_update_smoke_not_passed")
    if not summary.get("forward_counters_present"):
        reasons.append("forward_counters_missing")
    if _int_or_none(summary.get("total_actor_forward_batches")) in (None, 0):
        reasons.append("total_actor_forward_batches_missing")
    if _int_or_none(summary.get("total_actor_forward_samples")) in (None, 0):
        reasons.append("total_actor_forward_samples_missing")
    if (
        not _positive_int(summary.get("stateless_fallback_forward_batches"))
        and not _positive_int(summary.get("stateless_fallback_forward_samples"))
    ):
        reasons.append("stateless_fallback_not_observed")
    if (
        summary.get("long_memory_training_claim_supported")
        and summary.get("stateless_fallback_present")
    ):
        reasons.append("long_memory_claim_supported_despite_fallback")
    if summary.get("memory_lengths_observable") not in (True, False):
        reasons.append("memory_lengths_observable_missing")
    if summary.get("minibatches_preserve_temporal_segments") not in (True, False, "unknown"):
        reasons.append("temporal_segment_preservation_missing")
    if not summary.get("minibatches_preserve_temporal_segments_reason"):
        reasons.append("temporal_segment_preservation_reason_missing")
    if any(summary.get(field) is not False for field in NO_OVERCLAIM_FIELDS):
        reasons.append("claim_boundary_violation")
    return not reasons, reasons


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    log_dir = Path(args.log_dir).expanduser().resolve()
    summary = {
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
        "log_dir": str(log_dir),
        "log_dir_exists": log_dir.exists(),
        "task039_txl_update_memory_diagnostics_only": True,
        "diagnostic_kind": "task039_txl_update_memory_diagnostics",
        "train_variant_only": True,
        "heldout_variant_included": False,
        "diagnostic_evidence_valid": False,
        "long_memory_training_claim_supported": False,
        "memory_lengths_observable": False,
        "memory_lengths_observable_reason": "probe_failed_before_memory_debug",
        "minibatches_preserve_temporal_segments": "unknown",
        "minibatches_preserve_temporal_segments_reason": "probe_failed_before_forward_counters",
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "pass": False,
        "failure_reasons": ["probe_exception"],
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


def _task038_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        task=args.task,
        output_json=args.output_json,
        log_dir=args.log_dir,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        iterations=args.iterations,
        seed=args.seed,
        device=args.device,
        expected_action_dim=args.expected_action_dim,
        expected_runner_cls=args.expected_runner_cls,
        expected_actor_model_class=args.expected_actor_model_class,
    )


def _debug_mapping(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    debug = summary.get("txl_debug")
    return debug if isinstance(debug, Mapping) else {}


def _forward_counter_summary(debug: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "total_actor_forward_batches": ("total_actor_forward_batches",),
        "total_actor_forward_samples": ("total_actor_forward_samples",),
        "env_cache_stateful_forward_batches": ("env_cache_stateful_forward_batches",),
        "env_cache_stateful_forward_samples": ("env_cache_stateful_forward_samples",),
        "stateless_fallback_forward_batches": (
            "stateless_fallback_forward_batches",
            "stateless_forward_batches",
        ),
        "stateless_fallback_forward_samples": (
            "stateless_fallback_forward_samples",
            "stateless_forward_samples",
        ),
    }
    values: dict[str, int | None] = {}
    missing: list[str] = []
    for canonical, aliases in fields.items():
        value = _first_nonnegative_int(debug, aliases)
        values[canonical] = value
        if value is None:
            missing.append(canonical)

    total_batches = values["total_actor_forward_batches"]
    total_samples = values["total_actor_forward_samples"]
    fallback_batches = values["stateless_fallback_forward_batches"]
    fallback_samples = values["stateless_fallback_forward_samples"]
    return {
        **values,
        "forward_counters_present": not missing,
        "forward_counter_missing_fields": missing,
        "stateless_fallback_ratio_by_batches": _ratio(fallback_batches, total_batches),
        "stateless_fallback_ratio_by_samples": _ratio(fallback_samples, total_samples),
    }


def _memory_length_observation_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    before = _memory_debug_for(summary, "txl_debug_before")
    after = _memory_debug_for(summary, "txl_debug_after") or _memory_debug_for(summary, "txl_debug")
    before_summary = _memory_lengths_summary(before) if before else None
    after_summary = _memory_lengths_summary(after) if after else None
    observable = before_summary is not None and after_summary is not None
    reason = None
    if not observable:
        reason = (
            "wrapped_task038_ppo_update_smoke_exposes_post_learn_txl_debug_only"
            if after_summary is not None
            else "wrapped_task038_ppo_update_smoke_exposes_no_memory_length_snapshot"
        )
    return {
        "memory_lengths_observable": observable,
        "memory_lengths_observable_reason": reason,
        "memory_lengths_before_update": before_summary,
        "memory_lengths_after_update": after_summary,
        "memory_length_summary": {
            "before_update": before_summary,
            "after_update": after_summary,
            "observable": observable,
            "reason": reason,
        },
    }


def _temporal_segment_summary(counter_summary: Mapping[str, Any]) -> dict[str, Any]:
    if not counter_summary.get("forward_counters_present"):
        return {
            "minibatches_preserve_temporal_segments": "unknown",
            "minibatches_preserve_temporal_segments_reason": (
                "forward counters missing, cannot infer PPO minibatch temporal structure"
            ),
        }
    if _positive_int(counter_summary.get("stateless_fallback_forward_batches")):
        return {
            "minibatches_preserve_temporal_segments": False,
            "minibatches_preserve_temporal_segments_reason": (
                "stateless fallback batches indicate flattened PPO minibatches did not "
                "match the env-cache stateful forward layout"
            ),
        }
    return {
        "minibatches_preserve_temporal_segments": "unknown",
        "minibatches_preserve_temporal_segments_reason": (
            "no stateless fallback was observed, but this diagnostic does not inspect "
            "PPO minibatch segment metadata directly"
        ),
    }


def _memory_debug_for(summary: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = summary.get(key)
    return value if isinstance(value, Mapping) else None


def _memory_lengths_summary(debug: Mapping[str, Any]) -> dict[str, Any] | None:
    lengths = _memory_lengths_from_debug(debug)
    if not lengths:
        return None
    return {
        "value_count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "sum": sum(lengths),
        "mean": sum(lengths) / len(lengths),
    }


def _memory_lengths_from_debug(debug: Mapping[str, Any]) -> list[int]:
    lengths: list[int] = []
    envs = debug.get("envs")
    if isinstance(envs, Sequence) and not isinstance(envs, (str, bytes)):
        for env in envs:
            if isinstance(env, Mapping):
                lengths.extend(_numeric_sequence(env.get("memory_lengths")))
    lengths.extend(_numeric_sequence(debug.get("memory_lengths")))
    return lengths


def _numeric_sequence(value: Any) -> list[int]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values: list[int] = []
        for item in value:
            if not isinstance(item, bool) and isinstance(item, (int, float)):
                values.append(int(item))
        return values
    return []


def _first_nonnegative_int(debug: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _int_or_none(debug.get(key))
        if value is not None and value >= 0:
            return value
    return None


def _ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _positive_int(value: Any) -> bool:
    parsed = _int_or_none(value)
    return parsed is not None and parsed > 0


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    return [value]


def main() -> None:
    args = parse_args()
    try:
        preflight_args(args)
        summary = run_probe(args)
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        summary = build_failure_summary(args, exc)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
