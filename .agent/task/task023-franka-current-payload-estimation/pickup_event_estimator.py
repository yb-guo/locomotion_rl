"""Task023 pickup-event online payload estimator.

The estimator remains traditional and task-local. For transport episodes it
uses a causal sliding-window Jacobian least-squares estimate after subtracting
a pre-collected unloaded trajectory baseline. For return-home episodes it uses
the same-pose before/after effort difference at q_home.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ROOT = Path("outputs/task023/franka_current_force_estimation")
GRAVITY_REACTION = np.array([0.0, 0.0, 9.81], dtype=float)


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_estimator(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.strict and result["status"] != "ok":
        raise SystemExit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate pickup-event payload mass from effort traces.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--mode", choices=("auto", "pickup_transport", "return_home_diff"), default="auto")
    parser.add_argument("--window-steps", type=int, default=256)
    parser.add_argument("--min-window-steps", type=int, default=16)
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Use every Nth trace sample before estimating. For 500Hz traces, 10 means 50Hz.",
    )
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument(
        "--sample-method",
        choices=("point", "mean"),
        default="point",
        help="point takes every Nth sample; mean averages each causal N-sample block.",
    )
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--home-tail-s", type=float, default=1.0)
    parser.add_argument("--convergence-sustain-s", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "summaries" / "pickup_event_estimator.json")
    parser.add_argument("--estimate-trace", type=Path)
    parser.add_argument("--visual", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser


def run_estimator(args: argparse.Namespace) -> dict[str, Any]:
    if args.window_steps <= 0:
        return failed("window_steps_must_be_positive")
    if args.min_window_steps <= 0:
        return failed("min_window_steps_must_be_positive")
    if args.min_window_steps > args.window_steps:
        return failed("min_window_gt_window")
    if args.sample_stride <= 0:
        return failed("sample_stride_must_be_positive")
    if args.sample_offset < 0:
        return failed("sample_offset_must_be_non_negative")
    if args.sample_offset >= args.sample_stride:
        return failed("sample_offset_must_be_lt_sample_stride")
    if args.home_tail_s <= 0.0:
        return failed("home_tail_s_must_be_positive")

    trace = load_trace(args.trace, args.sample_stride, args.sample_offset, args.sample_method)
    scenario = scalar_string(trace["scenario"]) if "scenario" in trace else ""
    mode = scenario if args.mode == "auto" else args.mode
    if mode == "pickup_transport":
        if args.baseline is None:
            return failed("pickup_transport_requires_baseline")
        result, estimate = estimate_pickup_transport(
            args,
            trace,
            load_trace(args.baseline, args.sample_stride, args.sample_offset, args.sample_method),
        )
    elif mode == "return_home_diff":
        result, estimate = estimate_return_home_diff(args, trace)
    else:
        return failed(f"unknown_mode:{mode}")
    if result.get("status") == "failed":
        return result

    estimate_trace = args.estimate_trace or default_estimate_trace_path(args.trace, mode)
    visual = args.visual or default_visual_path(args.trace, mode)
    write_estimate_trace(estimate_trace, trace, estimate)
    write_mass_hat_svg(visual, trace, estimate, result)
    result.update(
        {
            "status": "ok",
            "blocker": "",
            "trace": str(args.trace),
            "baseline": str(args.baseline) if args.baseline is not None else "",
            "mode": mode,
            "scenario": scenario,
            "window_steps": args.window_steps,
            "min_window_steps": args.min_window_steps,
            "sample_stride": args.sample_stride,
            "sample_offset": args.sample_offset,
            "sample_method": args.sample_method,
            "effective_sample_hz": effective_sample_hz(trace),
            "gravity": args.gravity,
            "estimate_trace": str(estimate_trace),
            "visual": str(visual),
        }
    )
    return result


def load_trace(path: Path, sample_stride: int, sample_offset: int, sample_method: str) -> dict[str, Any]:
    source = np.load(path)
    if "t" not in source:
        raise KeyError(f"trace_missing_t:{path}")
    n_steps = int(source["t"].shape[0])
    if sample_offset >= n_steps:
        raise ValueError(f"sample_offset_out_of_trace:{sample_offset}>={n_steps}")
    if sample_method == "mean":
        return load_trace_block_mean(source, n_steps, sample_stride, sample_offset)
    indices = np.arange(sample_offset, n_steps, sample_stride, dtype=np.int64)
    out: dict[str, Any] = {}
    for key in source.files:
        value = source[key]
        if value.shape != () and value.shape[0] == n_steps:
            out[key] = value[indices]
        else:
            out[key] = value
    if "pickup_event_step" in source:
        pickup_original = scalar_int(source["pickup_event_step"])
        sampled_pickup = int(np.searchsorted(indices, pickup_original, side="left"))
        if sampled_pickup >= indices.shape[0]:
            sampled_pickup = int(indices.shape[0] - 1)
        out["pickup_event_step"] = np.array(sampled_pickup, dtype=np.int64)
        out["pickup_event_source_step"] = np.array(pickup_original, dtype=np.int64)
        out["sampled_source_step"] = indices
    return out


def load_trace_block_mean(
    source: Mapping[str, Any],
    n_steps: int,
    sample_stride: int,
    sample_offset: int,
) -> dict[str, Any]:
    block_starts = np.arange(sample_offset, n_steps, sample_stride, dtype=np.int64)
    block_ends = np.minimum(block_starts + sample_stride, n_steps)
    out: dict[str, Any] = {}
    for key in source.files:
        value = source[key]
        if value.shape == () or value.shape[0] != n_steps:
            out[key] = value
            continue
        if key == "phase":
            out[key] = np.asarray([value[end - 1] for end in block_ends])
        elif key in {"step", "phase_id", "payload_attached", "force_saturation"}:
            reducer = np.max if key in {"payload_attached", "force_saturation"} else last_value
            out[key] = np.asarray([reducer(value[start:end], axis=0) for start, end in zip(block_starts, block_ends)])
        else:
            out[key] = np.asarray([np.mean(value[start:end], axis=0) for start, end in zip(block_starts, block_ends)])
    if "pickup_event_step" in source:
        pickup_original = scalar_int(source["pickup_event_step"])
        sampled_pickup = int(np.searchsorted(block_ends - 1, pickup_original, side="left"))
        if sampled_pickup >= block_starts.shape[0]:
            sampled_pickup = int(block_starts.shape[0] - 1)
        out["pickup_event_step"] = np.array(sampled_pickup, dtype=np.int64)
        out["pickup_event_source_step"] = np.array(pickup_original, dtype=np.int64)
        out["sampled_source_step"] = block_ends - 1
    return out


def last_value(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.take(values, -1, axis=axis)


def effective_sample_hz(trace: Mapping[str, Any]) -> float:
    t = trace["t"]
    if t.shape[0] < 2:
        return float("nan")
    return float(1.0 / np.median(np.diff(t)))


def estimate_pickup_transport(
    args: argparse.Namespace,
    trace: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if trace["effort_control"].shape != baseline["effort_control"].shape:
        return failed("baseline_shape_mismatch"), empty_estimate(trace)
    if scalar_int(trace["pickup_event_step"]) != scalar_int(baseline["pickup_event_step"]):
        return failed("baseline_pickup_step_mismatch"), empty_estimate(trace)

    true_mass = trace["payload_mass_kg"]
    pickup_step = scalar_int(trace["pickup_event_step"])
    nominal_mass = float(trace["payload_mass_nominal_kg"])
    gravity = np.array([0.0, 0.0, args.gravity], dtype=float)
    delta_tau = trace["effort_control"] - baseline["effort_control"]
    regressor = np.einsum("tij,i->tj", trace["jacobian"][:, :3, :], gravity)
    mass_hat, residual_norm = causal_mass_windows(
        delta_tau=delta_tau,
        regressor=regressor,
        window_steps=args.window_steps,
        min_window_steps=args.min_window_steps,
    )
    metrics = transport_metrics(args, trace, mass_hat, residual_norm, pickup_step, nominal_mass)
    estimate = {
        "mass_hat_kg": mass_hat,
        "residual_norm": residual_norm,
        "true_mass_kg": true_mass,
    }
    return (
        {
            "estimator_variant": "trajectory_baseline",
            "equation": "delta_tau(t)=effort_pickup(t)-effort_unloaded_baseline(t); delta_tau ~= m*J_tool_trans(q).T*[0,0,+g]",
            "uses_future_pickup_episode_samples": False,
            **metrics,
        },
        estimate,
    )


def estimate_return_home_diff(
    args: argparse.Namespace,
    trace: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    t = trace["t"]
    dt = infer_dt(t)
    tail_steps = max(1, round(args.home_tail_s / dt))
    phase = trace["phase"].astype(str)
    before_idx = np.flatnonzero(phase == "home_hold_before")
    after_idx = np.flatnonzero(phase == "home_hold_after")
    if before_idx.size == 0 or after_idx.size == 0:
        return failed("home_hold_phase_missing"), empty_estimate(trace)
    before_tail = before_idx[-min(tail_steps, before_idx.size) :]
    after_tail = after_idx[-min(tail_steps, after_idx.size) :]

    tau_before = np.mean(trace["effort_control"][before_tail], axis=0)
    tau_after = np.mean(trace["effort_control"][after_tail], axis=0)
    delta_tau_home = tau_after - tau_before
    q_before = np.mean(trace["q"][before_tail], axis=0)
    q_after = np.mean(trace["q"][after_tail], axis=0)
    jac_home = np.mean(trace["jacobian"][after_tail, :3, :], axis=0)
    gravity = np.array([0.0, 0.0, args.gravity], dtype=float)
    regressor = np.einsum("ij,i->j", jac_home, gravity)
    denominator = float(np.sum(regressor * regressor))
    if denominator <= 1e-12:
        return failed("home_regressor_degenerate"), empty_estimate(trace)

    mass_hat_home = float(np.sum(regressor * delta_tau_home) / denominator)
    nominal_mass = float(trace["payload_mass_nominal_kg"])
    abs_error = abs(mass_hat_home - nominal_mass)
    mass_hat = np.full_like(t, fill_value=np.nan, dtype=float)
    residual_norm = np.full_like(t, fill_value=np.nan, dtype=float)
    mass_hat[before_idx] = 0.0
    for step in after_idx:
        start = max(after_idx[0], step - args.window_steps + 1)
        window = np.arange(start, step + 1)
        if window.size < args.min_window_steps:
            continue
        delta_window = trace["effort_control"][window] - tau_before
        reg_window = np.einsum("tij,i->tj", trace["jacobian"][window, :3, :], gravity)
        denominator_window = float(np.sum(reg_window * reg_window))
        if denominator_window <= 1e-12:
            continue
        value = float(np.sum(reg_window * delta_window) / denominator_window)
        mass_hat[step] = value
        residual_norm[step] = float(np.sqrt(np.mean((delta_window - value * reg_window) ** 2)))

    estimate = {
        "mass_hat_kg": mass_hat,
        "residual_norm": residual_norm,
        "true_mass_kg": trace["payload_mass_kg"],
    }
    metrics = {
        "estimator_variant": "return_home_same_pose_diff",
        "equation": "delta_tau_home=mean(tau_home_after)-mean(tau_home_before); delta_tau_home ~= m*J_tool_trans(q_home).T*[0,0,+g]",
        "uses_future_pickup_episode_samples": False,
        "payload_mass_true_kg": nominal_mass,
        "mass_hat_home_kg": mass_hat_home,
        "home_abs_error_kg": abs_error,
        "home_q_diff_norm_rad": float(np.linalg.norm(q_after - q_before)),
        "home_tail_steps": int(before_tail.size),
        "home_tail_s": float(before_tail.size * dt),
        "force_saturation_ratio": force_saturation_ratio(trace),
        "tracking_error_rms": tracking_error_rms(trace),
        "pass_first_0p5": bool(abs_error <= 0.05) if abs(nominal_mass - 0.5) <= 1e-9 else None,
    }
    return metrics, estimate


def causal_mass_windows(
    *,
    delta_tau: np.ndarray,
    regressor: np.ndarray,
    window_steps: int,
    min_window_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    mass_hat = np.full((delta_tau.shape[0],), fill_value=np.nan, dtype=float)
    residual_norm = np.full((delta_tau.shape[0],), fill_value=np.nan, dtype=float)
    for step in range(delta_tau.shape[0]):
        start = max(0, step - window_steps + 1)
        if step - start + 1 < min_window_steps:
            continue
        a = regressor[start : step + 1]
        y = delta_tau[start : step + 1]
        denominator = float(np.sum(a * a))
        if denominator <= 1e-12:
            continue
        value = float(np.sum(a * y) / denominator)
        mass_hat[step] = value
        residual_norm[step] = float(np.sqrt(np.mean((y - value * a) ** 2)))
    return mass_hat, residual_norm


def transport_metrics(
    args: argparse.Namespace,
    trace: Mapping[str, Any],
    mass_hat: np.ndarray,
    residual_norm: np.ndarray,
    pickup_step: int,
    nominal_mass: float,
) -> dict[str, Any]:
    t = trace["t"]
    dt = infer_dt(t)
    finite = np.isfinite(mass_hat)
    pre = (np.arange(mass_hat.shape[0]) < pickup_step) & finite
    post = (np.arange(mass_hat.shape[0]) >= pickup_step) & finite
    tolerance = max(0.05, 0.10 * nominal_mass)
    detection_threshold = max(0.05, 0.20 * nominal_mass)
    detection_step = first_detection_step(mass_hat, pickup_step, detection_threshold)
    convergence_step = first_sustained_convergence_step(
        mass_hat=mass_hat,
        pickup_step=pickup_step,
        target=nominal_mass,
        tolerance=tolerance,
        sustain_steps=max(1, round(args.convergence_sustain_s / dt)),
    )
    post_eval_start = convergence_step if convergence_step is not None else min(
        mass_hat.shape[0] - 1,
        pickup_step + round(1.0 / dt),
    )
    post_eval = (np.arange(mass_hat.shape[0]) >= post_eval_start) & finite
    post_error = mass_hat[post_eval] - nominal_mass
    pass_first_0p5 = None
    if abs(nominal_mass - 0.5) <= 1e-9:
        pass_first_0p5 = bool(
            safe_mean_abs(mass_hat[pre]) <= 0.05
            and detection_step is not None
            and (detection_step - pickup_step) * dt <= 0.5
            and convergence_step is not None
            and (convergence_step - pickup_step) * dt <= 1.0
            and float(np.mean(np.abs(post_error))) <= 0.10
        )
    return {
        "payload_mass_true_kg": nominal_mass,
        "pickup_event_step": pickup_step,
        "pickup_time_s": float(t[pickup_step]) if pickup_step < t.shape[0] else float(pickup_step * dt),
        "pre_pickup_abs_mean_kg": safe_mean_abs(mass_hat[pre]),
        "pre_pickup_abs_max_kg": safe_max_abs(mass_hat[pre]),
        "detection_threshold_kg": detection_threshold,
        "detection_step": detection_step,
        "detection_delay_ms": (
            float((detection_step - pickup_step) * dt * 1000.0) if detection_step is not None else None
        ),
        "convergence_tolerance_kg": tolerance,
        "convergence_step": convergence_step,
        "convergence_time_ms": (
            float((convergence_step - pickup_step) * dt * 1000.0)
            if convergence_step is not None
            else None
        ),
        "post_eval_start_step": int(post_eval_start),
        "post_convergence_mae_kg": float(np.mean(np.abs(post_error))) if post_error.size else None,
        "post_convergence_rmse_kg": float(math.sqrt(np.mean(post_error * post_error))) if post_error.size else None,
        "post_pickup_mean_kg": float(np.nanmean(mass_hat[post])) if np.any(post) else None,
        "residual_norm_mean": float(np.nanmean(residual_norm)) if np.any(np.isfinite(residual_norm)) else None,
        "force_saturation_ratio": force_saturation_ratio(trace),
        "tracking_error_rms": tracking_error_rms(trace),
        "pass_first_0p5": pass_first_0p5,
    }


def first_detection_step(mass_hat: np.ndarray, pickup_step: int, threshold: float) -> int | None:
    for step in range(pickup_step, mass_hat.shape[0]):
        value = mass_hat[step]
        if np.isfinite(value) and value >= threshold:
            return int(step)
    return None


def first_sustained_convergence_step(
    *,
    mass_hat: np.ndarray,
    pickup_step: int,
    target: float,
    tolerance: float,
    sustain_steps: int,
) -> int | None:
    finite = np.isfinite(mass_hat)
    ok = finite & (np.abs(mass_hat - target) <= tolerance)
    last_start = mass_hat.shape[0] - sustain_steps
    for step in range(pickup_step, max(pickup_step, last_start) + 1):
        if np.all(ok[step : step + sustain_steps]):
            return int(step)
    return None


def write_estimate_trace(path: Path, trace: Mapping[str, Any], estimate: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t=trace["t"],
        phase=trace["phase"],
        payload_attached=trace["payload_attached"],
        payload_mass_kg=trace["payload_mass_kg"],
        pickup_event_step=trace["pickup_event_step"],
        mass_hat_kg=estimate["mass_hat_kg"],
        residual_norm=estimate["residual_norm"],
        true_mass_kg=estimate["true_mass_kg"],
    )


def write_mass_hat_svg(
    path: Path,
    trace: Mapping[str, Any],
    estimate: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = trace["t"]
    mass_hat = estimate["mass_hat_kg"]
    true_mass = estimate["true_mass_kg"]
    phase = trace["phase"].astype(str)
    pickup_step = scalar_int(trace["pickup_event_step"])
    width = 1100
    height = 420
    margin_left = 70
    margin_right = 30
    margin_top = 30
    margin_bottom = 55
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    finite_hat = mass_hat[np.isfinite(mass_hat)]
    y_max_data = max(float(np.nanmax(true_mass)), float(np.nanmax(finite_hat)) if finite_hat.size else 0.0, 0.2)
    y_min_data = min(0.0, float(np.nanmin(finite_hat)) if finite_hat.size else 0.0)
    y_min = y_min_data - 0.08
    y_max = y_max_data + 0.12

    def x_of(value: float) -> float:
        return margin_left + (float(value) - float(t[0])) / max(1e-9, float(t[-1] - t[0])) * plot_w

    def y_of(value: float) -> float:
        return margin_top + (y_max - float(value)) / max(1e-9, y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="22" font-family="Arial" font-size="16" fill="#111">Pickup event mass estimate</text>',
    ]
    parts.extend(phase_backgrounds(t, phase, x_of, margin_top, plot_h))
    parts.append(
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222" stroke-width="1"/>'
    )
    for y_value in nice_ticks(y_min, y_max, 5):
        y = y_of(y_value)
        parts.append(f'<line x1="{margin_left}" x2="{margin_left + plot_w}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11" fill="#374151">{y_value:.2f}</text>'
        )
    pickup_x = x_of(t[pickup_step] if pickup_step < t.shape[0] else t[-1])
    parts.append(f'<line x1="{pickup_x:.2f}" x2="{pickup_x:.2f}" y1="{margin_top}" y2="{margin_top + plot_h}" stroke="#dc2626" stroke-width="2"/>')
    parts.append(
        f'<text x="{pickup_x + 5:.2f}" y="{margin_top + 16}" font-family="Arial" font-size="11" fill="#dc2626">pickup</text>'
    )
    parts.extend(polyline_segments(t, true_mass, x_of, y_of, "#16a34a", 2.0))
    parts.extend(polyline_segments(t, mass_hat, x_of, y_of, "#2563eb", 2.0))
    parts.append(f'<text x="{margin_left}" y="{height - 20}" font-family="Arial" font-size="12" fill="#111">blue: mass_hat_kg, green: true endpoint payload mass</text>')
    verdict = metrics.get("pass_first_0p5")
    verdict_text = "pass_first_0p5=" + str(verdict)
    parts.append(f'<text x="{width - margin_right}" y="{height - 20}" text-anchor="end" font-family="Arial" font-size="12" fill="#111">{html.escape(verdict_text)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def phase_backgrounds(
    t: np.ndarray,
    phase: np.ndarray,
    x_of: Any,
    margin_top: int,
    plot_h: int,
) -> list[str]:
    out = []
    colors = ("#f8fafc", "#f1f5f9")
    start = 0
    color_index = 0
    while start < phase.shape[0]:
        end = start + 1
        while end < phase.shape[0] and phase[end] == phase[start]:
            end += 1
        x0 = x_of(t[start])
        x1 = x_of(t[end - 1])
        out.append(
            f'<rect x="{x0:.2f}" y="{margin_top}" width="{max(1.0, x1 - x0):.2f}" height="{plot_h}" fill="{colors[color_index % 2]}" opacity="0.7"/>'
        )
        out.append(
            f'<text x="{x0 + 4:.2f}" y="{margin_top + plot_h - 6}" font-family="Arial" font-size="10" fill="#64748b">{html.escape(str(phase[start]))}</text>'
        )
        start = end
        color_index += 1
    return out


def polyline_segments(
    t: np.ndarray,
    values: np.ndarray,
    x_of: Any,
    y_of: Any,
    color: str,
    stroke_width: float,
) -> list[str]:
    max_points = 1400
    stride = max(1, math.ceil(values.shape[0] / max_points))
    out = []
    current: list[str] = []
    for index in range(0, values.shape[0], stride):
        value = values[index]
        if not np.isfinite(value):
            if len(current) >= 2:
                out.append(polyline(current, color, stroke_width))
            current = []
            continue
        current.append(f"{x_of(t[index]):.2f},{y_of(float(value)):.2f}")
    if len(current) >= 2:
        out.append(polyline(current, color, stroke_width))
    return out


def polyline(points: list[str], color: str, stroke_width: float) -> str:
    return f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linejoin="round" stroke-linecap="round"/>'


def nice_ticks(y_min: float, y_max: float, count: int) -> list[float]:
    if count <= 1:
        return [y_min]
    step = (y_max - y_min) / float(count - 1)
    return [y_min + step * index for index in range(count)]


def default_estimate_trace_path(trace_path: Path, mode: str) -> Path:
    return DEFAULT_ROOT / "estimates" / f"{trace_path.stem}_{mode}_estimate.npz"


def default_visual_path(trace_path: Path, mode: str) -> Path:
    return DEFAULT_ROOT / "visuals" / f"{trace_path.stem}_{mode}_mass_hat.svg"


def empty_estimate(trace: Mapping[str, Any]) -> dict[str, Any]:
    t = trace["t"]
    return {
        "mass_hat_kg": np.full_like(t, fill_value=np.nan, dtype=float),
        "residual_norm": np.full_like(t, fill_value=np.nan, dtype=float),
        "true_mass_kg": trace["payload_mass_kg"],
    }


def infer_dt(t: np.ndarray) -> float:
    if t.shape[0] < 2:
        return 0.002
    return float(np.median(np.diff(t)))


def safe_mean_abs(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(np.abs(values)))


def safe_max_abs(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.max(np.abs(values)))


def force_saturation_ratio(trace: Mapping[str, Any]) -> float:
    if "force_saturation" not in trace:
        return float("nan")
    return float(np.mean(trace["force_saturation"]))


def tracking_error_rms(trace: Mapping[str, Any]) -> float:
    return float(np.sqrt(np.mean(trace["tracking_error"] * trace["tracking_error"])))


def scalar_string(value: Any) -> str:
    if hasattr(value, "shape") and value.shape == ():
        return str(value.item())
    return str(value)


def scalar_int(value: Any) -> int:
    array = np.asarray(value)
    return int(array.item()) if array.shape == () else int(array.reshape(-1)[0])


def failed(blocker: str) -> dict[str, Any]:
    return {"status": "failed", "blocker": blocker}


if __name__ == "__main__":
    main()
