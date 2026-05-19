"""Task023 Jacobian static-load payload mass estimator.

Uses step-aligned traces:

    delta_tau = effort_control_payload - effort_control_0kg
    delta_tau ~= m * J_tool(q).T * [0, 0, +g]

The positive sign is intentional: the measured control effort is the actuator
reaction required to hold a downward payload.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DEFAULT_ROOT = Path("outputs/task023/franka_current_force_estimation")
DEFAULT_RUNS = (
    "0:full_mass_0kg,"
    "0.25:full_mass_0p25kg,"
    "0.5:full_mass_0p5kg,"
    "1.0:full_mass_1kg,"
    "2.0:full_mass_2kg"
)


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_estimator(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.strict and result["status"] != "ok":
        raise SystemExit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate endpoint payload mass from effort/Jacobian traces.")
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_ROOT / "traces")
    parser.add_argument("--runs", default=DEFAULT_RUNS)
    parser.add_argument("--baseline-mass-kg", type=float, default=0.0)
    parser.add_argument("--hold-steps", type=int, default=500)
    parser.add_argument("--window-steps", type=int, default=128)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "summaries" / "jacobian_static_payload_estimator.json",
    )
    parser.add_argument("--strict", action="store_true")
    return parser


def run_estimator(args: argparse.Namespace) -> dict[str, Any]:
    if args.hold_steps < 0:
        return failed("hold_steps_must_be_non_negative")
    if args.window_steps <= 0:
        return failed("window_steps_must_be_positive")

    run_map = parse_runs(args.runs)
    if args.baseline_mass_kg not in run_map:
        return failed(f"baseline_mass_missing:{args.baseline_mass_kg}")
    traces = {mass: load_trace(args.trace_root, run_id) for mass, run_id in run_map.items()}
    baseline = traces[args.baseline_mass_kg]

    per_mass: list[dict[str, Any]] = []
    all_true: list[np.ndarray] = []
    all_raw: list[np.ndarray] = []
    predictions: dict[float, np.ndarray] = {}

    for mass, run_id in sorted(run_map.items()):
        if mass == args.baseline_mass_kg:
            continue
        raw = estimate_mass_windows(
            payload_trace=traces[mass],
            baseline_trace=baseline,
            hold_steps=args.hold_steps,
            window_steps=args.window_steps,
            gravity=args.gravity,
        )
        predictions[mass] = raw
        true = np.full_like(raw, fill_value=mass, dtype=float)
        all_true.append(true)
        all_raw.append(raw)
        per_mass.append(
            {
                "mass_kg": mass,
                "run_id": run_id,
                "window_count": int(raw.shape[0]),
                **regression_metrics(true, raw),
                "prediction_mean": float(np.mean(raw)),
                "prediction_median": float(np.median(raw)),
                "prediction_std": float(np.std(raw)),
                "prediction_p05": float(np.percentile(raw, 5)),
                "prediction_p95": float(np.percentile(raw, 95)),
                "low_confidence": bool(mass >= 2.0),
                "low_confidence_reason": (
                    "2kg trace has nontrivial effort saturation" if mass >= 2.0 else ""
                ),
            }
        )

    if not all_true:
        return failed("no_nonzero_payload_runs")

    y_true = np.concatenate(all_true)
    y_raw = np.concatenate(all_raw)
    lomo = leave_one_mass_out_calibration(predictions)
    return {
        "status": "ok",
        "blocker": "",
        "trace_root": str(args.trace_root),
        "runs": {str(mass): run_id for mass, run_id in sorted(run_map.items())},
        "baseline_mass_kg": args.baseline_mass_kg,
        "hold_steps_excluded": args.hold_steps,
        "window_steps": args.window_steps,
        "gravity": args.gravity,
        "model": "Jacobian static-load least squares",
        "equation": "delta_tau ~= mass_kg * J_tool_trans(q).T * [0, 0, +g]",
        "sign_note": "control effort is the actuator reaction to downward payload gravity",
        "per_mass": per_mass,
        "overall_raw": regression_metrics(y_true, y_raw),
        "leave_one_mass_out_linear_calibration": lomo,
    }


def parse_runs(value: str) -> dict[float, str]:
    out: dict[float, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        mass_text, run_id = item.split(":", 1)
        out[float(mass_text)] = run_id
    return out


def load_trace(root: Path, run_id: str) -> Mapping[str, Any]:
    path = root / f"{run_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path)


def estimate_mass_windows(
    *,
    payload_trace: Mapping[str, Any],
    baseline_trace: Mapping[str, Any],
    hold_steps: int,
    window_steps: int,
    gravity: float,
) -> np.ndarray:
    delta_tau = payload_trace["effort_control"][hold_steps:] - baseline_trace["effort_control"][hold_steps:]
    jacobian = payload_trace["jacobian"][hold_steps:, :3, :]
    if delta_tau.shape[0] < window_steps:
        raise ValueError(f"trace too short for window_steps={window_steps}")
    gravity_reaction = np.array([0.0, 0.0, gravity], dtype=float)
    regressor = np.einsum("tij,i->tj", jacobian, gravity_reaction)
    estimates = []
    for start in range(0, delta_tau.shape[0] - window_steps + 1):
        a = regressor[start : start + window_steps]
        y = delta_tau[start : start + window_steps]
        denominator = float(np.sum(a * a))
        if denominator <= 1e-12:
            estimates.append(float("nan"))
            continue
        estimates.append(float(np.sum(a * y) / denominator))
    raw = np.asarray(estimates, dtype=float)
    return raw[np.isfinite(raw)]


def leave_one_mass_out_calibration(predictions: Mapping[float, np.ndarray]) -> list[dict[str, Any]]:
    masses = sorted(predictions)
    rows: list[dict[str, Any]] = []
    for held_out in masses:
        train_true = []
        train_raw = []
        for mass in masses:
            if mass == held_out:
                continue
            raw = predictions[mass]
            train_true.append(np.full_like(raw, fill_value=mass, dtype=float))
            train_raw.append(raw)
        x_train = np.concatenate(train_raw)
        y_train = np.concatenate(train_true)
        slope, intercept = fit_line(x_train, y_train)
        raw_test = predictions[held_out]
        true_test = np.full_like(raw_test, fill_value=held_out, dtype=float)
        calibrated = slope * raw_test + intercept
        rows.append(
            {
                "held_out_mass_kg": held_out,
                "calibration_slope": slope,
                "calibration_intercept": intercept,
                **regression_metrics(true_test, calibrated),
                "prediction_mean": float(np.mean(calibrated)),
                "prediction_std": float(np.std(calibrated)),
            }
        )
    return rows


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(math.sqrt(np.mean(error * error)))
    ss_res = float(np.sum(error * error))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {
        "mae_kg": mae,
        "rmse_kg": rmse,
        "r2": float(r2),
        "bias_kg": float(np.mean(error)),
    }


def failed(blocker: str) -> dict[str, Any]:
    return {"status": "failed", "blocker": blocker}


if __name__ == "__main__":
    main()
