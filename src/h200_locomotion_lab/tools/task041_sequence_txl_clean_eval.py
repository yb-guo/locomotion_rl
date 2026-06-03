"""Evaluate Task041 sequence-aware true-TXL clean checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from h200_locomotion_lab.tools import task039_true_txl_clean_eval
from h200_locomotion_lab.tools.task038_true_txl_runner_smoke_probe import (
    DEFAULT_EXPECTED_ACTION_DIM,
    DEFAULT_EXPECTED_ACTOR_MODEL_CLASS,
    DEFAULT_EXPECTED_RUNNER_CLS,
)
from h200_locomotion_lab.tools.task040_sequence_txl_ppo_update_smoke import (
    _install_ipython_display_stub,
    _install_wandb_stub,
    _install_wcwidth_stub,
)


TASK042_MEMORY_ABLATION_MODES = (
    "none",
    "zero_txl_residual",
    "stateless_txl_memory",
    "zero_memory_latent",
)


def parse_args(argv: list[str] | None = None, *, description: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description
        or (
            "Run Task041 sequence-aware true-TXL clean-gait eval using the Task039 "
            "quality gate. Top-level pass requires pipeline and quality gate pass."
        )
    )
    parser.add_argument("--task", default=task039_true_txl_clean_eval.TASK039_TRUE_TXL_CLEAN_TASK_ID)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--seed", type=int, default=4100201)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trial-length-s", type=float, default=2.0)
    parser.add_argument("--lin-vel-x", type=float, default=0.4)
    parser.add_argument("--lin-vel-y", type=float, default=0.0)
    parser.add_argument("--ang-vel-z", type=float, default=0.0)
    parser.add_argument(
        "--dynamic-case",
        choices=sorted(task039_true_txl_clean_eval.task037_multitrial_eval_checkpoint.DYNAMIC_CASES),
        default="none",
    )
    parser.add_argument(
        "--dynamic-dead-joint",
        choices=task039_true_txl_clean_eval.task037_multitrial_eval_checkpoint.DEFAULT_JOINTS,
    )
    parser.add_argument("--dynamic-onset-s", type=float, default=0.5)
    parser.add_argument("--dynamic-recovery-s", type=float, default=1.5)
    parser.add_argument(
        "--force-dead-joint",
        choices=task039_true_txl_clean_eval.task037_multitrial_eval_checkpoint.DEFAULT_JOINTS,
    )
    parser.add_argument("--dead-scale", type=float, default=0.0)
    parser.add_argument("--final-window-s", type=float, default=0.0)
    parser.add_argument("--min-final-completion-ratio", type=float, default=0.95)
    parser.add_argument("--max-final-fall-ratio", type=float, default=0.50)
    parser.add_argument("--max-final-lin-vel-error", type=float, default=1.20)
    parser.add_argument("--max-final-yaw-vel-error", type=float, default=1.00)
    parser.add_argument("--max-final-gravity-xy", type=float, default=0.90)
    parser.add_argument("--min-final-root-z", type=float, default=0.35)
    parser.add_argument("--memory-latent-dim", type=int, default=32)
    parser.add_argument("--memory-latent-scale", type=float, default=1.0)
    parser.add_argument("--base-obs-passthrough-scale", type=float, default=1.0)
    parser.add_argument("--adaptation-warmstart-scale", type=float, default=1.0)
    parser.add_argument("--action-dim", type=int, default=DEFAULT_EXPECTED_ACTION_DIM)
    parser.add_argument("--adaptation-hidden-dim", type=int, default=128)
    parser.set_defaults(base_obs_passthrough=True, adaptation_warmstart=True)
    parser.add_argument("--base-obs-passthrough", dest="base_obs_passthrough", action="store_true")
    parser.add_argument("--no-base-obs-passthrough", dest="base_obs_passthrough", action="store_false")
    parser.add_argument("--adaptation-warmstart", dest="adaptation_warmstart", action="store_true")
    parser.add_argument("--no-adaptation-warmstart", dest="adaptation_warmstart", action="store_false")
    parser.add_argument(
        "--memory-ablation-mode",
        choices=TASK042_MEMORY_ABLATION_MODES,
        default="none",
    )
    parser.add_argument("--expected-action-dim", type=int, default=DEFAULT_EXPECTED_ACTION_DIM)
    parser.add_argument("--expected-runner-cls", default=DEFAULT_EXPECTED_RUNNER_CLS)
    parser.add_argument(
        "--expected-actor-model-class",
        default=DEFAULT_EXPECTED_ACTOR_MODEL_CLASS,
    )
    parser.add_argument("--train-summary-json")
    return parser.parse_args(argv)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    _install_ipython_display_stub()
    _install_wandb_stub()
    _install_wcwidth_stub()
    task039_true_txl_clean_eval.preflight_args(args)
    result = task039_true_txl_clean_eval.run_eval(args)
    return wrap_task039_result(args, result)


def wrap_task039_result(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    wrapped = dict(result)
    train_summary = _load_train_summary(args.train_summary_json)
    wrapped["task041_sequence_txl_clean_eval"] = True
    wrapped["policy_label"] = "SequenceAwareTrueTXL"
    wrapped["memory_latent_dim"] = args.memory_latent_dim
    wrapped["memory_latent_scale"] = args.memory_latent_scale
    wrapped["base_obs_passthrough"] = args.base_obs_passthrough
    wrapped["adaptation_warmstart"] = args.adaptation_warmstart
    wrapped["base_obs_passthrough_scale"] = args.base_obs_passthrough_scale
    wrapped["adaptation_warmstart_scale"] = args.adaptation_warmstart_scale
    wrapped.update(_memory_ablation_summary(wrapped.get("txl_debug"), args.memory_ablation_mode))
    wrapped["train_summary_json"] = args.train_summary_json
    wrapped["train_summary"] = train_summary
    wrapped["sequence_aware_update_train_pipeline_pass"] = bool(
        (train_summary or {}).get("train_pipeline_pass")
    )
    wrapped["sequence_aware_algorithm_class"] = (train_summary or {}).get("algorithm_class")
    wrapped["sequence_aware_checkpoint_match"] = _checkpoint_matches_train_summary(
        args.checkpoint,
        train_summary,
    )
    wrapped["quality_claim"] = False
    wrapped["training_claim"] = False
    wrapped["eval_claim"] = False
    wrapped["reproduction_claim"] = False
    wrapped["superiority_claim"] = False
    pipeline_pass, task041_reasons = evaluate_task041_eval_pipeline_pass(wrapped)
    wrapped["task041_pipeline_pass"] = pipeline_pass
    wrapped["failure_reasons"] = list(dict.fromkeys(
        list(wrapped.get("failure_reasons") or []) + task041_reasons
    ))
    wrapped["pass"] = pipeline_pass and bool(wrapped.get("quality_gate_pass"))
    return wrapped


def evaluate_task041_eval_pipeline_pass(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not summary.get("pipeline_pass"):
        reasons.append("task039_pipeline_not_passed")
    if summary.get("sequence_aware_algorithm_class") not in {
        "Task040SequenceAwareTrueTxlPPO",
        None,
    }:
        reasons.append("sequence_aware_algorithm_class_mismatch")
    if summary.get("train_summary_json") and not summary.get("train_summary"):
        reasons.append("train_summary_missing")
    if summary.get("train_summary") and not summary.get("sequence_aware_update_train_pipeline_pass"):
        reasons.append("train_summary_pipeline_not_passed")
    if summary.get("train_summary") and not summary.get("sequence_aware_checkpoint_match"):
        reasons.append("train_summary_checkpoint_mismatch")
    if (
        not summary.get("task041_sequence_txl_clean_eval")
        or summary.get("quality_claim") is not False
        or summary.get("training_claim") is not False
        or summary.get("eval_claim") is not False
        or summary.get("reproduction_claim") is not False
        or summary.get("superiority_claim") is not False
    ):
        reasons.append("claim_boundary_violation")
    return not reasons, reasons


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    summary = task039_true_txl_clean_eval.build_failure_summary(args, exc)
    summary["task041_sequence_txl_clean_eval"] = True
    summary["policy_label"] = "SequenceAwareTrueTXL"
    summary.update(_memory_ablation_summary(summary.get("txl_debug"), getattr(args, "memory_ablation_mode", "none")))
    summary["task041_pipeline_pass"] = False
    summary["pass"] = False
    return summary


def write_json_summary(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary["json_path"] = str(output)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_train_summary(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    summary_path = Path(path).expanduser()
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _checkpoint_matches_train_summary(checkpoint: str, train_summary: dict[str, Any] | None) -> bool:
    if not train_summary:
        return False
    return Path(checkpoint).expanduser().resolve() == Path(
        str(train_summary.get("checkpoint", ""))
    ).expanduser().resolve()


def _memory_ablation_summary(debug: Any, requested_mode: str) -> dict[str, Any]:
    debug_map = debug if isinstance(debug, dict) else {}
    actual_mode = debug_map.get("task042_memory_ablation_mode", requested_mode)
    residual_enabled = debug_map.get("memory_residual_enabled")
    memory_latent_enabled = debug_map.get("memory_latent_enabled")
    stateful_enabled = debug_map.get("stateful_memory_enabled")
    return {
        "memory_ablation_mode": requested_mode,
        "memory_ablation_mode_reported": actual_mode,
        "memory_ablation_mode_match": actual_mode == requested_mode,
        "memory_residual_enabled": (
            bool(residual_enabled)
            if residual_enabled is not None
            else requested_mode not in {"zero_txl_residual", "zero_memory_latent"}
        ),
        "memory_latent_enabled": (
            bool(memory_latent_enabled)
            if memory_latent_enabled is not None
            else requested_mode != "zero_memory_latent"
        ),
        "stateful_memory_enabled": (
            bool(stateful_enabled)
            if stateful_enabled is not None
            else requested_mode != "stateless_txl_memory"
        ),
        "txl_residual_output_norm": debug_map.get("txl_residual_output_norm_last"),
        "txl_residual_raw_norm": debug_map.get("txl_residual_raw_norm_last"),
        "adaptation_output_norm": debug_map.get("adaptation_output_norm_last"),
        "policy_memory_latent_norm": debug_map.get("policy_memory_latent_norm_last"),
    }


def main() -> None:
    args = parse_args()
    try:
        summary = run_eval(args)
    except Exception as exc:
        summary = build_failure_summary(args, exc)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
