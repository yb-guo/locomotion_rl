"""Visualize task023 Genesis Franka payload collection from saved NPZ traces.

No plotting dependencies are required. This script writes plain SVG files so it
can run in the local Windows environment without matplotlib/Pillow.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("outputs/task023/franka_current_force_estimation")
FULL_RUNS = (
    (0.0, "full_mass_0kg"),
    (0.25, "full_mass_0p25kg"),
    (0.5, "full_mass_0p5kg"),
    (1.0, "full_mass_1kg"),
    (2.0, "full_mass_2kg"),
)
COLORS = {
    0.0: "#3d5a80",
    0.25: "#2a9d8f",
    0.5: "#e9c46a",
    1.0: "#f4a261",
    2.0: "#e76f51",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write SVG visualizations for task023 traces.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "visuals")
    parser.add_argument("--hold-steps", type=int, default=500)
    parser.add_argument("--window-steps", type=int, default=128)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = {mass: np.load(args.root / "traces" / f"{run_id}.npz") for mass, run_id in FULL_RUNS}
    estimator = load_json(args.root / "summaries" / "jacobian_static_payload_estimator.json")
    force_compare = load_json(args.root / "summaries" / "force_limit_2kg_compare.json")

    dashboard = args.output_dir / "genesis_collection_dashboard.svg"
    force_svg = args.output_dir / "force_limit_2kg_compare.svg"
    schema_svg = args.output_dir / "trace_schema_and_residual.svg"

    write_svg(dashboard, collection_dashboard(traces, estimator, args.hold_steps, args.window_steps))
    write_svg(force_svg, force_limit_compare_svg(force_compare))
    write_svg(schema_svg, trace_schema_svg())

    print(json.dumps({
        "dashboard": str(dashboard),
        "force_limit_compare": str(force_svg),
        "trace_schema": str(schema_svg),
    }, indent=2, sort_keys=True))


def collection_dashboard(
    traces: dict[float, np.lib.npyio.NpzFile],
    estimator: dict,
    hold_steps: int,
    window_steps: int,
) -> str:
    width, height = 1600, 1120
    parts = [svg_open(width, height)]
    parts.append(rect(0, 0, width, height, "#fbfbf8"))
    parts.append(text(40, 46, "Task023 Genesis Franka Payload Collection", 30, "#202124", weight="700"))
    parts.append(text(40, 78, "H200 Genesis traces: one process per mass, same q_target, link7 payload weld, 500 Hz", 16, "#4b5563"))

    parts.append(panel_timeline(40, 110, 720, 210))
    parts.append(panel_tool_path(800, 110, 740, 430, traces))
    parts.append(panel_effort_rms(40, 360, 720, 330, traces, hold_steps))
    parts.append(panel_mass_estimate(800, 580, 740, 420, traces, hold_steps, window_steps))
    parts.append(panel_estimator_summary(40, 730, 720, 270, estimator))
    parts.append(svg_close())
    return "\n".join(parts)


def panel_timeline(x: float, y: float, w: float, h: float) -> str:
    parts = [panel_bg(x, y, w, h, "Collection timeline")]
    x0, x1 = x + 40, x + w - 40
    yb = y + 95
    hold_end = x0 + (x1 - x0) * (1.0 / 13.0)
    parts.append(line(x0, yb, x1, yb, "#9aa0a6", 2))
    parts.append(rect(x0, yb - 18, hold_end - x0, 36, "#d9e2ec"))
    parts.append(rect(hold_end, yb - 18, x1 - hold_end, 36, "#c7d7f2"))
    parts.append(text(x0, yb - 34, "0.0s", 12, "#4b5563"))
    parts.append(text(hold_end - 16, yb - 34, "1.0s", 12, "#4b5563"))
    parts.append(text(x1 - 34, yb - 34, "13.0s", 12, "#4b5563"))
    parts.append(text(x0 + 12, yb + 6, "hold q0", 14, "#1f2937", weight="600"))
    parts.append(text(hold_end + 18, yb + 6, "deterministic slow sine sweep", 14, "#1f2937", weight="600"))
    parts.append(text(x + 40, y + 150, "Rows per mass: 6500    dt: 0.002s    sample rate: 500Hz", 14, "#374151"))
    parts.append(text(x + 40, y + 178, "Mass runs: 0, 0.25, 0.5, 1.0, 2.0 kg    Output: NPZ traces + JSON summaries", 14, "#374151"))
    return "\n".join(parts)


def panel_tool_path(
    x: float,
    y: float,
    w: float,
    h: float,
    traces: dict[float, np.lib.npyio.NpzFile],
) -> str:
    parts = [panel_bg(x, y, w, h, "Tool path from Genesis link7 position (X-Z projection)")]
    plot = PlotBox(x + 75, y + 62, w - 115, h - 115)
    xs = np.concatenate([trace["tool_pos"][:, 0] for trace in traces.values()])
    zs = np.concatenate([trace["tool_pos"][:, 2] for trace in traces.values()])
    mapper = Mapper(float(xs.min()), float(xs.max()), float(zs.min()), float(zs.max()), plot)
    parts.extend(axes(plot, "tool x (m)", "tool z (m)"))
    for mass, trace in traces.items():
        points = downsample_xy(trace["tool_pos"][:, 0], trace["tool_pos"][:, 2], 700)
        parts.append(polyline([mapper.map(px, pz) for px, pz in points], COLORS[mass], 2.2, opacity=0.9))
    parts.extend(legend(x + w - 185, y + 62, traces.keys()))
    return "\n".join(parts)


def panel_effort_rms(
    x: float,
    y: float,
    w: float,
    h: float,
    traces: dict[float, np.lib.npyio.NpzFile],
    hold_steps: int,
) -> str:
    parts = [panel_bg(x, y, w, h, "Control effort RMS during collection")]
    plot = PlotBox(x + 75, y + 60, w - 120, h - 108)
    values = {}
    for mass, trace in traces.items():
        effort = trace["effort_control"][hold_steps:]
        rms = np.sqrt(np.mean(effort * effort, axis=1))
        t = trace["t"][hold_steps:]
        values[mass] = (t, rms)
    ymax = max(float(np.max(v[1])) for v in values.values()) * 1.05
    mapper = Mapper(1.0, 13.0, 0.0, ymax, plot)
    parts.extend(axes(plot, "time (s)", "joint effort RMS (Nm)"))
    for mass, (t, rms) in values.items():
        points = downsample_xy(t, rms, 700)
        parts.append(polyline([mapper.map(px, py) for px, py in points], COLORS[mass], 2.0))
    parts.extend(legend(x + w - 170, y + 62, values.keys()))
    return "\n".join(parts)


def panel_mass_estimate(
    x: float,
    y: float,
    w: float,
    h: float,
    traces: dict[float, np.lib.npyio.NpzFile],
    hold_steps: int,
    window_steps: int,
) -> str:
    parts = [panel_bg(x, y, w, h, "Sliding-window Jacobian mass estimate")]
    plot = PlotBox(x + 75, y + 60, w - 120, h - 108)
    base = traces[0.0]
    curves = {}
    for mass in (0.25, 0.5, 1.0, 2.0):
        t, estimate = estimate_mass_curve(base, traces[mass], hold_steps, window_steps)
        curves[mass] = (t, estimate)
    mapper = Mapper(1.0, 13.0, 0.0, 2.25, plot)
    parts.extend(axes(plot, "time (s)", "mass_hat (kg)"))
    for mass in (0.25, 0.5, 1.0, 2.0):
        parts.append(line(plot.x0, mapper.y(mass), plot.x1, mapper.y(mass), COLORS[mass], 1.0, dash="6 8", opacity=0.45))
        points = downsample_xy(curves[mass][0], curves[mass][1], 700)
        parts.append(polyline([mapper.map(px, py) for px, py in points], COLORS[mass], 2.0))
    parts.extend(legend(x + w - 170, y + 62, (0.25, 0.5, 1.0, 2.0)))
    return "\n".join(parts)


def panel_estimator_summary(x: float, y: float, w: float, h: float, estimator: dict) -> str:
    parts = [panel_bg(x, y, w, h, "Estimator result")]
    overall = estimator["overall_raw"]
    parts.append(big_metric(x + 45, y + 75, "MAE", f"{overall['mae_kg']:.4f} kg", "#2a9d8f"))
    parts.append(big_metric(x + 245, y + 75, "RMSE", f"{overall['rmse_kg']:.4f} kg", "#3d5a80"))
    parts.append(big_metric(x + 470, y + 75, "R2", f"{overall['r2']:.4f}", "#e76f51"))
    parts.append(text(x + 45, y + 178, "Estimator: delta_tau ~= mass * J_tool_trans(q)^T * [0, 0, +9.81]", 14, "#374151"))
    parts.append(text(x + 45, y + 208, "Input: get_dofs_control_force only. get_dofs_force was recorded as diagnostic.", 14, "#374151"))
    parts.append(text(x + 45, y + 238, "Interpretation: endpoint payload mass under known gravity and known tool attachment.", 14, "#374151"))
    return "\n".join(parts)


def force_limit_compare_svg(compare: dict) -> str:
    width, height = 1100, 560
    parts = [svg_open(width, height), rect(0, 0, width, height, "#fbfbf8")]
    parts.append(text(40, 48, "2kg Force-Limit Diagnostic", 30, "#202124", weight="700"))
    rows = compare["rows"]
    labels = ["saturation_ratio_abs_ge_limit_minus_0p1", "tracking_error_rms", "mass_hat_mae"]
    titles = ["Saturation ratio", "Tracking error RMS", "Mass estimate MAE"]
    colors = ["#e76f51", "#3d5a80"]
    for idx, (key, title) in enumerate(zip(labels, titles)):
        px = 55 + idx * 345
        py = 110
        parts.append(panel_bg(px, py, 300, 360, title))
        vmax = max(float(row[key]) for row in rows) * 1.25
        if vmax <= 0:
            vmax = 1.0
        for j, row in enumerate(rows):
            value = float(row[key])
            bh = 230 * value / vmax
            bx = px + 75 + j * 80
            by = py + 285 - bh
            parts.append(rect(bx, by, 48, bh, colors[j]))
            parts.append(text(bx - 12, py + 315, "default" if j == 0 else "2x", 13, "#374151"))
            parts.append(text(bx - 22, by - 10, f"{value:.4f}", 12, "#374151"))
        parts.append(line(px + 55, py + 285, px + 255, py + 285, "#9aa0a6", 1.2))
    parts.append(text(50, 520, "2x force limit reduces saturation and tracking error. The mass estimate remains essentially unchanged.", 16, "#374151"))
    parts.append(svg_close())
    return "\n".join(parts)


def trace_schema_svg() -> str:
    width, height = 1300, 620
    parts = [svg_open(width, height), rect(0, 0, width, height, "#fbfbf8")]
    parts.append(text(40, 48, "Genesis Trace Row and Estimation Inputs", 30, "#202124", weight="700"))
    boxes = [
        (70, 110, 210, 70, "scene.step()", "Genesis physics"),
        (330, 110, 210, 70, "read state", "q, dq, tool_pos"),
        (590, 110, 210, 70, "read effort", "control + diagnostic"),
        (850, 110, 210, 70, "read Jacobian", "get_jacobian(link7)"),
        (460, 270, 360, 88, "trace row", "q, dq, q_target, tracking_error, effort_control, get_dofs_force, J, mass label"),
        (460, 430, 360, 88, "estimator input", "delta_tau from effort_control + J_tool_trans"),
    ]
    for bx, by, bw, bh, title, subtitle in boxes:
        parts.append(rounded_rect(bx, by, bw, bh, "#ffffff", "#ccd5df"))
        parts.append(text(bx + 18, by + 30, title, 17, "#111827", weight="700"))
        parts.append(text(bx + 18, by + 55, subtitle, 13, "#4b5563"))
    arrows = [
        (280, 145, 330, 145),
        (540, 145, 590, 145),
        (800, 145, 850, 145),
        (695, 180, 640, 270),
        (640, 358, 640, 430),
    ]
    for x0, y0, x1, y1 in arrows:
        parts.append(arrow(x0, y0, x1, y1, "#52616f"))
    parts.append(text(70, 560, "Only effort_control enters the estimator. get_dofs_force is stored for diagnostics because it can include simulator internal effects.", 16, "#374151"))
    parts.append(svg_close())
    return "\n".join(parts)


def estimate_mass_curve(
    base: np.lib.npyio.NpzFile,
    trace: np.lib.npyio.NpzFile,
    hold_steps: int,
    window_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    delta = trace["effort_control"][hold_steps:] - base["effort_control"][hold_steps:]
    jacobian = trace["jacobian"][hold_steps:, :3, :]
    a = np.einsum("tij,i->tj", jacobian, np.array([0.0, 0.0, 9.81]))
    estimates = []
    times = []
    t = trace["t"][hold_steps:]
    for start in range(delta.shape[0] - window_steps + 1):
        aw = a[start : start + window_steps]
        dw = delta[start : start + window_steps]
        estimates.append(float(np.sum(aw * dw) / (np.sum(aw * aw) + 1e-9)))
        times.append(float(t[start + window_steps // 2]))
    return np.asarray(times), np.asarray(estimates)


def downsample_xy(x: Sequence[float], y: Sequence[float], max_points: int) -> list[tuple[float, float]]:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    if len(x_arr) <= max_points:
        idx = np.arange(len(x_arr))
    else:
        idx = np.linspace(0, len(x_arr) - 1, max_points).astype(int)
    return [(float(x_arr[i]), float(y_arr[i])) for i in idx]


class PlotBox:
    def __init__(self, x: float, y: float, w: float, h: float) -> None:
        self.x0 = x
        self.y0 = y
        self.w = w
        self.h = h
        self.x1 = x + w
        self.y1 = y + h


class Mapper:
    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float, box: PlotBox) -> None:
        pad_x = (xmax - xmin) * 0.04 if xmax > xmin else 1.0
        pad_y = (ymax - ymin) * 0.08 if ymax > ymin else 1.0
        self.xmin = xmin - pad_x
        self.xmax = xmax + pad_x
        self.ymin = ymin - pad_y
        self.ymax = ymax + pad_y
        self.box = box

    def x(self, value: float) -> float:
        return self.box.x0 + (value - self.xmin) / (self.xmax - self.xmin) * self.box.w

    def y(self, value: float) -> float:
        return self.box.y1 - (value - self.ymin) / (self.ymax - self.ymin) * self.box.h

    def map(self, x: float, y: float) -> tuple[float, float]:
        return self.x(x), self.y(y)


def axes(box: PlotBox, xlabel: str, ylabel: str) -> list[str]:
    return [
        line(box.x0, box.y1, box.x1, box.y1, "#9aa0a6", 1.2),
        line(box.x0, box.y0, box.x0, box.y1, "#9aa0a6", 1.2),
        text(box.x0 + box.w / 2 - 35, box.y1 + 36, xlabel, 13, "#4b5563"),
        text(box.x0 - 58, box.y0 + box.h / 2, ylabel, 13, "#4b5563", rotate=-90),
    ]


def legend(x: float, y: float, masses: Iterable[float]) -> list[str]:
    parts = []
    for i, mass in enumerate(masses):
        yy = y + i * 24
        parts.append(rect(x, yy - 10, 16, 4, COLORS[float(mass)]))
        parts.append(text(x + 24, yy - 5, f"{mass:g} kg", 13, "#374151"))
    return parts


def big_metric(x: float, y: float, label: str, value: str, color: str) -> str:
    return "\n".join([
        rounded_rect(x, y, 160, 78, "#ffffff", "#d0d7de"),
        text(x + 18, y + 28, label, 15, "#4b5563", weight="700"),
        text(x + 18, y + 58, value, 20, color, weight="700"),
    ])


def panel_bg(x: float, y: float, w: float, h: float, title: str) -> str:
    return "\n".join([
        rounded_rect(x, y, w, h, "#ffffff", "#d7dee8"),
        text(x + 22, y + 32, title, 18, "#111827", weight="700"),
    ])


def svg_open(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def svg_close() -> str:
    return "</svg>"


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none") -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}"/>'


def rounded_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" fill="{fill}" stroke="{stroke}"/>'


def line(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
    width: float,
    dash: str | None = None,
    opacity: float = 1.0,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{color}" stroke-width="{width:.1f}" opacity="{opacity:.3f}"{dash_attr}/>'


def arrow(x0: float, y0: float, x1: float, y1: float, color: str) -> str:
    angle = math.atan2(y1 - y0, x1 - x0)
    head = 9
    a1 = angle + math.pi * 0.82
    a2 = angle - math.pi * 0.82
    hx1, hy1 = x1 + head * math.cos(a1), y1 + head * math.sin(a1)
    hx2, hy2 = x1 + head * math.cos(a2), y1 + head * math.sin(a2)
    return "\n".join([
        line(x0, y0, x1, y1, color, 2.0),
        f'<polygon points="{x1:.1f},{y1:.1f} {hx1:.1f},{hy1:.1f} {hx2:.1f},{hy2:.1f}" fill="{color}"/>',
    ])


def polyline(points: Sequence[tuple[float, float]], color: str, width: float, opacity: float = 1.0) -> str:
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="{width:.1f}" opacity="{opacity:.3f}" stroke-linejoin="round" stroke-linecap="round"/>'


def text(
    x: float,
    y: float,
    value: str,
    size: int,
    color: str,
    weight: str = "400",
    rotate: int | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    escaped = html_escape(value)
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}"{transform}>{escaped}</text>'


def html_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_svg(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
