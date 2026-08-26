"""Calibrate Task039 quality gate on an existing multi-trial eval JSON."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.training.task039_quality_feedback import evaluate_quality_feedback


class PreflightError(ValueError):
    def __init__(self, reasons: list[str]) -> None:
        super().__init__(", ".join(reasons))
        self.reasons = reasons


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Task039 quality gate on an existing eval JSON as diagnostic "
            "calibration evidence. This does not create a baseline or reproduction claim."
        )
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--policy-label", default="calibration")
    parser.add_argument("--checkpoint-label", default="")
    return parser.parse_args(argv)


def preflight_args(args: argparse.Namespace) -> None:
    reasons: list[str] = []
    if not Path(args.input_json).expanduser().exists():
        reasons.append("input_json_missing")
    if not str(args.policy_label).strip():
        reasons.append("policy_label_missing")
    if reasons:
        raise PreflightError(reasons)


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    preflight_args(args)
    source_path = Path(args.input_json).expanduser()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    summary = normalize_source_summary(args, source, source_path=source_path)
    feedback = evaluate_quality_feedback(summary)
    summary["quality_feedback"] = feedback.to_json()
    summary["quality_gate_pass"] = feedback.quality_gate_pass
    summary["quality_failure_reasons"] = list(feedback.failure_reasons)
    summary["failure_reasons"] = list(feedback.failure_reasons)
    summary["pass"] = bool(summary["pipeline_pass"] and feedback.quality_gate_pass)
    return summary


def normalize_source_summary(
    args: argparse.Namespace,
    source: dict[str, Any],
    *,
    source_path: Path,
) -> dict[str, Any]:
    summary = dict(source)
    pipeline_pass = bool(source.get("pipeline_pass", source.get("pass", False)))
    summary["source_json"] = str(source_path)
    summary["task039_quality_calibration_only"] = True
    summary["policy_label"] = args.policy_label
    summary["checkpoint_label"] = args.checkpoint_label
    summary["pipeline_pass"] = pipeline_pass
    summary["quality_claim"] = False
    summary["training_claim"] = False
    summary["eval_claim"] = False
    summary["reproduction_claim"] = False
    summary["superiority_claim"] = False
    summary["diagnostic_note"] = (
        "Task039 quality-gate calibration only. This is not a baseline, training "
        "success, eval success, reproduction, or superiority claim."
    )
    return summary


def build_failure_summary(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    summary = {
        "source_json": str(Path(getattr(args, "input_json", "")).expanduser()),
        "task039_quality_calibration_only": True,
        "policy_label": getattr(args, "policy_label", ""),
        "checkpoint_label": getattr(args, "checkpoint_label", ""),
        "pipeline_pass": False,
        "quality_gate_pass": False,
        "pass": False,
        "quality_claim": False,
        "training_claim": False,
        "eval_claim": False,
        "reproduction_claim": False,
        "superiority_claim": False,
        "failure_reasons": ["quality_calibration_exception"],
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


def main() -> None:
    args = parse_args()
    try:
        summary = run_calibration(args)
    except RECOVERABLE_RUNTIME_ERRORS as exc:
        summary = build_failure_summary(args, exc)
    summary["command"] = list(sys.argv)
    write_json_summary(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
