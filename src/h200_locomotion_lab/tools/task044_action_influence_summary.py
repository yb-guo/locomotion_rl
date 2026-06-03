"""Build a Task044 action-influence diagnostic summary from eval JSON files."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from h200_locomotion_lab.training.task044_action_influence_contract import (
    Task044ActionInfluenceThresholds,
    evaluate_task044_action_influence_triplet,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read normal, zero-residual, and stateless-memory eval JSON files "
            "and compare action statistics. Diagnostic-only."
        )
    )
    parser.add_argument("--normal-json", required=True)
    parser.add_argument("--zero-residual-json", required=True)
    parser.add_argument("--stateless-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--metric-scope",
        choices=("final_trial", "final_trial_window", "root"),
        default="final_trial_window",
    )
    parser.add_argument("--min-mean-abs-l1-delta", type=float, default=0.01)
    parser.add_argument("--min-mean-l2-delta", type=float, default=0.01)
    return parser.parse_args(argv)


def build_action_influence_summary(args: argparse.Namespace) -> dict[str, Any]:
    normal_path = Path(args.normal_json).expanduser().resolve()
    zero_path = Path(args.zero_residual_json).expanduser().resolve()
    stateless_path = Path(args.stateless_json).expanduser().resolve()
    thresholds = Task044ActionInfluenceThresholds(
        min_mean_abs_l1_delta=args.min_mean_abs_l1_delta,
        min_mean_l2_delta=args.min_mean_l2_delta,
        metric_scope=args.metric_scope,
    )
    contract = evaluate_task044_action_influence_triplet(
        _read_json(normal_path),
        _read_json(zero_path),
        _read_json(stateless_path),
        thresholds,
    )
    return {
        "task044_action_influence_summary": True,
        "normal_json": str(normal_path),
        "zero_residual_json": str(zero_path),
        "stateless_json": str(stateless_path),
        "action_influence_detected": contract["action_influence_detected"],
        "diagnostic_pass": contract["diagnostic_pass"],
        "task044_action_influence_contract": contract,
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
        "task044_action_influence_summary": True,
        "action_influence_detected": False,
        "diagnostic_pass": False,
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
        "failure_reasons": ["task044_action_influence_summary_error"],
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


def main() -> None:
    args = parse_args()
    try:
        summary = build_action_influence_summary(args)
    except Exception as exc:
        summary = build_failure_summary(args, exc)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
