"""Trace zero-action G1 support polygon and COM signals on Genesis."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from h200_locomotion_lab.envs.vectorized_genesis_backend import (
    VectorizedGenesisBackend,
    VectorizedGenesisConfig,
)
from h200_locomotion_lab.error_policy import RECOVERABLE_RUNTIME_ERRORS
from h200_locomotion_lab.robots import load_g1_27dof_nohand_profile
from h200_locomotion_lab.tools import g1_ppo_smoke
from h200_locomotion_lab.tools import g1_zero_action_standing_causality as zero_action

DEFAULT_OUTPUT_ROOT = Path("outputs/task023/zero_action_support_trace")
DEFAULT_START_STEP = 80
DEFAULT_END_STEP = 130
FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")


def main() -> None:
    args = parse_args()
    result: dict[str, Any] = {
        "status": "failed",
        "blocker": "",
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }
    try:
        summary = run_trace(args)
        result.update(summary)
        result["status"] = "ok"
    except RECOVERABLE_RUNTIME_ERRORS as exc:  # pragma: no cover - H200 failure path.
        result["blocker"] = f"{exc.__class__.__name__}:{exc}"
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "ok":
        raise SystemExit(1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=zero_action.positive_int, default=1)
    parser.add_argument("--start-step", type=zero_action.positive_int, default=DEFAULT_START_STEP)
    parser.add_argument("--end-step", type=zero_action.positive_int, default=DEFAULT_END_STEP)
    parser.add_argument("--asset-variant", choices=g1_ppo_smoke.ASSET_VARIANTS, default="task023_hybrid")
    parser.add_argument("--contact-threshold", type=float, default=1.0)
    parser.add_argument("--root-z", type=zero_action.positive_float, default=zero_action.DEFAULT_ROOT_Z)
    parser.add_argument("--termination-height-min", type=zero_action.positive_float, default=0.20)
    parser.add_argument("--termination-height-max", type=zero_action.positive_float, default=1.20)
    parser.add_argument("--height-min", type=zero_action.positive_float, default=0.45)
    parser.add_argument("--height-max", type=zero_action.positive_float, default=1.20)
    parser.add_argument("--min-upright", type=zero_action.positive_float, default=zero_action.DEFAULT_MIN_UPRIGHT)
    parser.add_argument("--pose-profile", choices=zero_action.POSE_PROFILES, default="current")
    parser.add_argument("--gain-profile", choices=zero_action.GAIN_PROFILES, default="current")
    parser.add_argument(
        "--control-mode",
        choices=zero_action.CONTROL_MODES,
        default="genesis_position",
    )
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--logical-cuda-device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def run_trace(args: argparse.Namespace) -> dict[str, Any]:
    if args.start_step > args.end_step:
        raise ValueError("start-step must be <= end-step")
    zero_action.verify_cuda_isolation(
        backend=args.backend,
        physical_gpu=str(args.physical_gpu),
        logical_cuda_device=args.logical_cuda_device,
    )
    torch = zero_action.require_torch()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_profile = load_g1_27dof_nohand_profile()
    run_dir = g1_ppo_smoke.resolve_run_dir(args.output_root, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    profile, asset_resolution = g1_ppo_smoke.resolve_training_profile_for_asset_variant(
        base_profile,
        asset_variant=args.asset_variant,
        run_dir=run_dir,
    )
    write_json(run_dir / "asset_resolution.json", asset_resolution)
    write_json(run_dir / "config.json", build_run_config(args, profile.asset.path, asset_resolution))

    mass_model = parse_mass_model(profile.asset.path)
    support_model = parse_foot_support_model(profile.asset.path, FOOT_LINKS)
    pose = zero_action.pose_profile_values(args.pose_profile, profile.control.default_angles_rad)
    gains = zero_action.gain_profile_values(args.gain_profile, profile.control)
    backend = VectorizedGenesisBackend(
        VectorizedGenesisConfig(
            n_envs=args.n_envs,
            backend=args.backend,
            logical_cuda_device=args.logical_cuda_device,
            root_qpos=zero_action.root_qpos(args.root_z),
            default_positions_rad=pose,
        ),
        profile=profile,
    )
    zero_action.apply_gain_profile_to_backend(backend, gains)
    backend.reset()

    mass_link_indices = resolve_link_indices(backend.robot, mass_model.keys())
    foot_link_indices = resolve_link_indices(backend.robot, FOOT_LINKS)
    trace_path = run_dir / "trace.jsonl"
    rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    zero_action_tensor = torch.zeros((args.n_envs, profile.action_dim), device=args.logical_cuda_device)
    start_time = time.perf_counter()
    with torch.no_grad():
        for step in range(1, args.end_step + 1):
            zero_action.apply_control_mode(
                torch=torch,
                backend=backend,
                action=zero_action_tensor,
                mode=args.control_mode,
            )
            state = backend.state()
            flags = zero_action.standing_flags(
                torch=torch,
                state=state,
                config=zero_action.env_config(args),
            )
            row = trace_row(
                robot=backend.robot,
                state=state,
                flags=flags,
                step=step,
                mass_model=mass_model,
                mass_link_indices=mass_link_indices,
                support_model=support_model,
                foot_link_indices=foot_link_indices,
                contact_threshold=float(args.contact_threshold),
            )
            all_rows.append(row)
            if args.start_step <= step <= args.end_step:
                rows.append(row)
                append_jsonl(trace_path, row)

    summary = summarize_trace(
        rows=rows,
        all_rows=all_rows,
        args=args,
        run_dir=run_dir,
        trace_path=trace_path,
        elapsed_s=time.perf_counter() - start_time,
        asset_path=profile.asset.path,
        asset_resolution=asset_resolution,
        mass_model=mass_model,
        support_model=support_model,
        mass_link_indices=mass_link_indices,
        foot_link_indices=foot_link_indices,
    )
    write_json(run_dir / "summary.json", summary)
    return summary


def build_run_config(
    args: argparse.Namespace,
    asset_path: str,
    asset_resolution: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": "task023-zero-action-support-trace",
        "n_envs": args.n_envs,
        "start_step": args.start_step,
        "end_step": args.end_step,
        "asset_variant": args.asset_variant,
        "asset_path": asset_path,
        "asset_resolution": asset_resolution,
        "root_z": args.root_z,
        "termination_height_min": args.termination_height_min,
        "termination_height_max": args.termination_height_max,
        "height_min": args.height_min,
        "height_max": args.height_max,
        "min_upright": args.min_upright,
        "pose_profile": args.pose_profile,
        "gain_profile": args.gain_profile,
        "control_mode": args.control_mode,
        "backend": args.backend,
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }


def trace_row(
    *,
    robot: Any,
    state: Any,
    flags: dict[str, Any],
    step: int,
    mass_model: dict[str, float],
    mass_link_indices: dict[str, int | None],
    support_model: dict[str, list[tuple[float, float, float]]],
    foot_link_indices: dict[str, int | None],
    contact_threshold: float,
) -> dict[str, Any]:
    root_quat = first_vector(state.root_quat)
    roll, pitch = roll_pitch_from_quat(root_quat)
    com = estimate_com(robot=robot, mass_model=mass_model, link_indices=mass_link_indices)
    foot_contacts = {
        name: read_link_contact(robot, link_idx, contact_threshold)
        for name, link_idx in foot_link_indices.items()
    }
    support_points = support_polygon_points(
        robot=robot,
        support_model=support_model,
        foot_link_indices=foot_link_indices,
        foot_contacts=foot_contacts,
        contact_threshold=contact_threshold,
    )
    hull = convex_hull([(point[0], point[1]) for point in support_points])
    margin = None
    inside = None
    if com is not None and len(hull) >= 3:
        margin = signed_point_polygon_margin((com[0], com[1]), hull)
        inside = margin >= 0.0
    return {
        "policy_step": step,
        "root_height": float(first_vector(state.root_pos)[2]),
        "root_roll": roll,
        "root_pitch": pitch,
        "upright": tensor_mean(flags["upright"]),
        "height_bad_count": tensor_bool_count(flags["height_bad"]),
        "termination_height_bad_count": tensor_bool_count(flags["termination_height_bad"]),
        "tilt_bad_count": tensor_bool_count(flags["tilt_bad"]),
        "com": None if com is None else {"x": com[0], "y": com[1], "z": com[2]},
        "com_source": "mass_weighted_link_positions" if com is not None else "unavailable",
        "support_polygon": {
            "active_foot_count": sum(1 for item in foot_contacts.values() if item["active"]),
            "point_count": len(support_points),
            "hull": [{"x": x, "y": y} for x, y in hull],
            "area": polygon_area(hull),
            "com_inside": inside,
            "com_signed_margin": margin,
        },
        "feet": foot_contacts,
    }


def summarize_trace(
    *,
    rows: Sequence[dict[str, Any]],
    all_rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
    trace_path: Path,
    elapsed_s: float,
    asset_path: str,
    asset_resolution: dict[str, Any],
    mass_model: dict[str, float],
    support_model: dict[str, list[tuple[float, float, float]]],
    mass_link_indices: dict[str, int | None],
    foot_link_indices: dict[str, int | None],
) -> dict[str, Any]:
    first_tilt = first_step_where(all_rows, "tilt_bad_count")
    first_height_bad = first_step_where(all_rows, "height_bad_count")
    first_termination_height = first_step_where(all_rows, "termination_height_bad_count")
    first_com_outside = next(
        (
            int(row["policy_step"])
            for row in all_rows
            if row["support_polygon"]["com_inside"] is False
        ),
        None,
    )
    min_margin_row = min(
        (row for row in all_rows if row["support_polygon"]["com_signed_margin"] is not None),
        key=lambda row: float(row["support_polygon"]["com_signed_margin"]),
        default=None,
    )
    rows_by_step = {int(row["policy_step"]): row for row in all_rows}
    key_steps = sorted(
        step
        for step in {
            args.start_step,
            100,
            110,
            116,
            120,
            122,
            123,
            args.end_step,
            first_com_outside,
            first_tilt,
        }
        if step is not None and 1 <= step <= args.end_step
    )
    return {
        "status": "completed",
        "run_dir": str(run_dir),
        "trace_path": str(trace_path),
        "asset_variant": args.asset_variant,
        "asset_path": asset_path,
        "asset_resolution": asset_resolution,
        "steps_simulated": args.end_step,
        "steps_recorded": len(rows),
        "elapsed_s": elapsed_s,
        "first_tilt_step": first_tilt,
        "first_height_bad_step": first_height_bad,
        "first_termination_height_bad_step": first_termination_height,
        "first_com_outside_support_step": first_com_outside,
        "min_com_signed_margin": None
        if min_margin_row is None
        else float(min_margin_row["support_polygon"]["com_signed_margin"]),
        "min_com_signed_margin_step": None if min_margin_row is None else int(min_margin_row["policy_step"]),
        "key_rows": {str(step): compact_row(rows_by_step[step]) for step in key_steps if step in rows_by_step},
        "mass_body_count": len(mass_model),
        "resolved_mass_body_count": sum(1 for value in mass_link_indices.values() if value is not None),
        "support_model_point_counts": {name: len(points) for name, points in support_model.items()},
        "unresolved_mass_links": [name for name, value in mass_link_indices.items() if value is None],
        "unresolved_foot_links": [name for name, value in foot_link_indices.items() if value is None],
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "root_height": row["root_height"],
        "root_roll": row["root_roll"],
        "root_pitch": row["root_pitch"],
        "upright": row["upright"],
        "tilt_bad_count": row["tilt_bad_count"],
        "height_bad_count": row["height_bad_count"],
        "termination_height_bad_count": row["termination_height_bad_count"],
        "com": row["com"],
        "support_polygon": row["support_polygon"],
        "feet": row["feet"],
    }


def parse_mass_model(asset_path: str | Path) -> dict[str, float]:
    root = ElementTree.parse(asset_path).getroot()
    masses: dict[str, float] = {}
    for body in root.iter("body"):
        name = body.get("name")
        inertial = body.find("inertial")
        if not name or inertial is None or inertial.get("mass") is None:
            continue
        mass = float(inertial.get("mass", "0"))
        if mass > 0.0:
            masses[name] = mass
    return masses


def parse_foot_support_model(
    asset_path: str | Path,
    foot_links: Sequence[str],
) -> dict[str, list[tuple[float, float, float]]]:
    root = ElementTree.parse(asset_path).getroot()
    return {
        link_name: support_points_for_body(root, link_name)
        for link_name in foot_links
    }


def support_points_for_body(
    root: ElementTree.Element,
    body_name: str,
) -> list[tuple[float, float, float]]:
    body = find_body(root, body_name)
    if body is None:
        return []
    points: list[tuple[float, float, float]] = []
    for geom in body.findall("geom"):
        if not contact_enabled(geom) or geom.get("type") == "mesh":
            continue
        pos = parse_xyz(geom.get("pos"), default=(0.0, 0.0, 0.0))
        size = parse_size(geom.get("size"))
        geom_type = geom.get("type", "sphere")
        if geom_type == "box" and len(size) >= 2:
            sx, sy = size[0], size[1]
            for dx in (-sx, sx):
                for dy in (-sy, sy):
                    points.append((pos[0] + dx, pos[1] + dy, pos[2]))
        else:
            points.append(pos)
    return points


def contact_enabled(geom: ElementTree.Element) -> bool:
    return geom.get("contype") != "0" and geom.get("conaffinity") != "0"


def find_body(root: ElementTree.Element, body_name: str) -> ElementTree.Element | None:
    for body in root.iter("body"):
        if body.get("name") == body_name:
            return body
    return None


def resolve_link_indices(robot: Any, names: Iterable[str]) -> dict[str, int | None]:
    return {name: resolve_link_index(robot, name) for name in names}


def resolve_link_index(robot: Any, name: str) -> int | None:
    if not hasattr(robot, "get_link"):
        return None
    try:
        link = robot.get_link(name)
    except RECOVERABLE_RUNTIME_ERRORS:
        return None
    for attr in ("idx_local", "idx", "id"):
        if hasattr(link, attr):
            return int(getattr(link, attr))
    return None


def estimate_com(
    *,
    robot: Any,
    mass_model: dict[str, float],
    link_indices: dict[str, int | None],
) -> tuple[float, float, float] | None:
    positions: list[tuple[float, float, float, float]] = []
    for name, mass in mass_model.items():
        link_idx = link_indices.get(name)
        if link_idx is None:
            continue
        vector = read_link_vector(robot, "get_links_pos", link_idx)
        if vector is None or len(vector) < 3:
            continue
        positions.append((mass, float(vector[0]), float(vector[1]), float(vector[2])))
    total_mass = sum(item[0] for item in positions)
    if total_mass <= 0.0:
        return None
    return (
        sum(mass * x for mass, x, _y, _z in positions) / total_mass,
        sum(mass * y for mass, _x, y, _z in positions) / total_mass,
        sum(mass * z for mass, _x, _y, z in positions) / total_mass,
    )


def support_polygon_points(
    *,
    robot: Any,
    support_model: dict[str, list[tuple[float, float, float]]],
    foot_link_indices: dict[str, int | None],
    foot_contacts: dict[str, dict[str, Any]],
    contact_threshold: float,
) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for link_name, local_points in support_model.items():
        if foot_contacts.get(link_name, {}).get("force", 0.0) < contact_threshold:
            continue
        link_idx = foot_link_indices.get(link_name)
        if link_idx is None:
            continue
        pos = read_link_vector(robot, "get_links_pos", link_idx)
        if pos is None or len(pos) < 3:
            continue
        quat = read_link_vector(robot, "get_links_quat", link_idx)
        for local in local_points:
            rotated = rotate_vector_by_quat(local, quat) if quat and len(quat) >= 4 else local
            points.append(
                (
                    float(pos[0]) + rotated[0],
                    float(pos[1]) + rotated[1],
                    float(pos[2]) + rotated[2],
                )
            )
    return points


def read_link_contact(
    robot: Any,
    link_idx: int | None,
    contact_threshold: float,
) -> dict[str, Any]:
    if link_idx is None:
        return {"force": 0.0, "active": False, "available": False}
    vector = None
    for method_name in ("get_links_net_contact_force", "get_links_net_contact_forces"):
        vector = read_link_vector(robot, method_name, link_idx)
        if vector is not None:
            break
    force = 0.0 if vector is None else math.sqrt(sum(float(value) ** 2 for value in vector[-3:]))
    return {
        "force": force,
        "active": force >= contact_threshold,
        "available": vector is not None,
    }


def read_link_vector(robot: Any, method_name: str, link_idx: int) -> list[float] | None:
    if not hasattr(robot, method_name):
        return None
    method = getattr(robot, method_name)
    try:
        values = method(links_idx_local=(link_idx,))
    except TypeError:
        try:
            values = method()
        except RECOVERABLE_RUNTIME_ERRORS:
            return None
    except RECOVERABLE_RUNTIME_ERRORS:
        return None
    vectors = vectors_from_any(values, link_idx=link_idx)
    return vectors[0] if vectors else None


def vectors_from_any(value: Any, *, link_idx: int) -> list[list[float]]:
    value = python_value(value)
    if is_numeric_sequence(value):
        numbers = [float(item) for item in value]
        if len(numbers) > 3 and len(numbers) % 3 == 0:
            start = link_idx * 3
            if len(numbers) >= start + 3:
                return [numbers[start : start + 3]]
        return [numbers]
    if not isinstance(value, list):
        value = list(value)
    if not value:
        return []
    if all(is_numeric_sequence(row) for row in value):
        return [[float(item) for item in row] for row in value]
    vectors: list[list[float]] = []
    for row in value:
        row = python_value(row)
        if is_numeric_sequence(row):
            vectors.append([float(item) for item in row])
            continue
        if not isinstance(row, list) or not row:
            continue
        if all(is_numeric_sequence(item) for item in row):
            if len(row) == 1:
                vectors.append([float(item) for item in row[0]])
            elif len(row) > link_idx:
                vectors.append([float(item) for item in row[link_idx]])
            continue
        vectors.extend(vectors_from_any(row, link_idx=link_idx))
    return vectors


def first_vector(value: Any) -> list[float]:
    value = python_value(value)
    if is_numeric_sequence(value):
        return [float(item) for item in value]
    if not isinstance(value, list):
        value = list(value)
    if not value:
        return []
    first = python_value(value[0])
    if is_numeric_sequence(first):
        return [float(item) for item in first]
    return first_vector(first)


def roll_pitch_from_quat(quat: Sequence[float]) -> tuple[float, float]:
    if len(quat) < 4:
        return 0.0, 0.0
    w, x, y, z = normalize_quat(quat[:4])
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    return roll, pitch


def rotate_vector_by_quat(
    vector: Sequence[float],
    quat: Sequence[float],
) -> tuple[float, float, float]:
    w, x, y, z = normalize_quat(quat[:4])
    vx, vy, vz = (float(vector[0]), float(vector[1]), float(vector[2]))
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def normalize_quat(quat: Sequence[float]) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in quat[:4])
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return list(unique)

    def cross(
        origin: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (
            b[0] - origin[0]
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def signed_point_polygon_margin(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
) -> float:
    if len(polygon) < 3:
        return float("-inf")
    inside = point_in_polygon(point, polygon)
    distances = [
        point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    ]
    distance = min(distances) if distances else 0.0
    return distance if inside else -distance


def point_in_polygon(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, vertex in enumerate(polygon):
        xi, yi = vertex
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (
            (yj - yi) or 1e-12
        ) + xi
        if intersects:
            inside = not inside
        j = i
    return inside or point_on_polygon_boundary(point, polygon)


def point_on_polygon_boundary(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
) -> bool:
    return any(
        point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        <= 1e-9
        for index in range(len(polygon))
    )


def point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
    closest = (sx + t * dx, sy + t * dy)
    return math.hypot(px - closest[0], py - closest[1])


def polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
        / 2.0
    )


def parse_xyz(value: str | None, *, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not value:
        return default
    parts = [float(part) for part in value.split()]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    return tuple(parts[:3])  # type: ignore[return-value]


def parse_size(value: str | None) -> tuple[float, ...]:
    if not value:
        return ()
    return tuple(float(part) for part in value.split())


def first_step_where(rows: Sequence[dict[str, Any]], count_key: str) -> int | None:
    return next(
        (int(row["policy_step"]) for row in rows if int(row.get(count_key, 0) or 0) > 0),
        None,
    )


def tensor_bool_count(value: Any) -> int:
    value = python_value(value)
    if isinstance(value, bool):
        return int(value)
    if is_numeric_sequence(value):
        return sum(1 for item in value if bool(item))
    if not isinstance(value, list):
        value = list(value)
    return sum(tensor_bool_count(item) for item in value)


def tensor_mean(value: Any) -> float:
    value = python_value(value)
    if isinstance(value, (int, float, bool)):
        return float(value)
    flat = flatten_numbers(value)
    return sum(flat) / len(flat) if flat else 0.0


def flatten_numbers(value: Any) -> list[float]:
    value = python_value(value)
    if isinstance(value, (int, float, bool)):
        return [float(value)]
    if is_numeric_sequence(value):
        return [float(item) for item in value]
    if not isinstance(value, list):
        value = list(value)
    result: list[float] = []
    for item in value:
        result.extend(flatten_numbers(item))
    return result


def is_numeric_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(
        isinstance(item, (int, float, bool)) for item in value
    )


def python_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
