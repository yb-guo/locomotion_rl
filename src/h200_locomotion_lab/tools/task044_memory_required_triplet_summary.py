"""Build a Task044 memory-required triplet summary from eval JSON files."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from h200_locomotion_lab.training.task044_memory_required_contract import (
    Task044TripletThresholds,
    evaluate_task044_memory_required_triplet,
)


HIDDEN_FAULT_CONTRACT = {
    "fault_identity_in_actor_obs": False,
    "fault_severity_in_actor_obs": False,
    "fault_onset_in_actor_obs": False,
    "fault_recovery_in_actor_obs": False,
    "fault_schedule_in_actor_obs": False,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read normal, zero-residual, and stateless-memory eval JSON files "
            "and write a Task044 memory-required triplet summary."
        )
    )
    parser.add_argument("--normal-json", required=True)
    parser.add_argument("--zero-residual-json", required=True)
    parser.add_argument("--stateless-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--confirm-hidden-fault-labels",
        action="store_true",
        help=(
            "Annotate the triplet with the Task044 hidden-fault observation "
            "contract. Use only after checking the actor observation does not "
            "include direct fault identity, severity, onset, recovery, or "
            "schedule labels."
        ),
    )
    parser.add_argument("--min-ablation-lin-vel-error-delta", type=float, default=0.03)
    parser.add_argument("--min-ablation-fall-ratio-delta", type=float, default=0.10)
    parser.add_argument("--min-ablation-completion-ratio-drop", type=float, default=0.10)
    parser.add_argument(
        "--metric-scope",
        choices=(
            "final_trial",
            "final_trial_window",
            "final_trial_tail_window",
            "post_fault_window",
        ),
        default="final_trial",
        help=(
            "Metric object used for ablation degradation deltas. The normal "
            "quality gate still uses the eval JSON quality_gate_pass field."
        ),
    )
    return parser.parse_args(argv)


def build_triplet_summary(args: argparse.Namespace) -> dict[str, Any]:
    normal_path = Path(args.normal_json).expanduser().resolve()
    zero_path = Path(args.zero_residual_json).expanduser().resolve()
    stateless_path = Path(args.stateless_json).expanduser().resolve()
    normal = _read_json(normal_path)
    zero_residual = _read_json(zero_path)
    stateless = _read_json(stateless_path)

    if args.confirm_hidden_fault_labels:
        _ensure_hidden_fault_contract(normal)
        _ensure_hidden_fault_contract(zero_residual)
        _ensure_hidden_fault_contract(stateless)

    thresholds = Task044TripletThresholds(
        min_ablation_lin_vel_error_delta=args.min_ablation_lin_vel_error_delta,
        min_ablation_fall_ratio_delta=args.min_ablation_fall_ratio_delta,
        min_ablation_completion_ratio_drop=args.min_ablation_completion_ratio_drop,
        metric_scope=args.metric_scope,
    )
    contract = evaluate_task044_memory_required_triplet(
        normal,
        zero_residual,
        stateless,
        thresholds,
    )
    return {
        "task044_triplet_summary": True,
        "normal_json": str(normal_path),
        "zero_residual_json": str(zero_path),
        "stateless_json": str(stateless_path),
        "task044_memory_required_pass": contract["task044_memory_required_pass"],
        "memory_required_evidence_pass": contract["memory_required_evidence_pass"],
        "task044_contract": contract,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "memory_causality_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "failure_reasons": list(contract["failure_reasons"]),
    }


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    return {
        "task044_triplet_summary": True,
        "task044_memory_required_pass": False,
        "memory_required_evidence_pass": False,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "memory_causality_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "normal_json": getattr(args, "normal_json", None),
        "zero_residual_json": getattr(args, "zero_residual_json", None),
        "stateless_json": getattr(args, "stateless_json", None),
        "error": repr(exc),
        "traceback": traceback.format_exc(),
        "failure_reasons": ["task044_triplet_summary_error"],
    }


def write_json_summary(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary["json_path"] = str(output)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _ensure_hidden_fault_contract(summary: dict[str, Any]) -> None:
    summary.setdefault("task044_hidden_fault_contract", dict(HIDDEN_FAULT_CONTRACT))


def main() -> None:
    args = parse_args()
    try:
        summary = build_triplet_summary(args)
    except Exception as exc:
        summary = build_failure_summary(args, exc)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
