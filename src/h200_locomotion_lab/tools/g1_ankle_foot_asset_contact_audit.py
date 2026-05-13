"""Audit G1 ankle/foot MJCF contact fields and optional Genesis link traces."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree

from h200_locomotion_lab.envs.vectorized_genesis_backend import VectorizedGenesisBackend
from h200_locomotion_lab.envs.vectorized_genesis_backend import VectorizedGenesisConfig
from h200_locomotion_lab.robots import load_g1_27dof_nohand_profile
from h200_locomotion_lab.tools import g1_zero_action_standing_causality as zero_action


DEFAULT_OUTPUT_ROOT = Path("outputs/task021/ankle_foot_asset_contact_audit")
DEFAULT_TARGET_BODIES = (
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
)
CONTACT_ATTRS = (
    "friction",
    "condim",
    "solref",
    "solimp",
    "priority",
    "contype",
    "conaffinity",
)
INERTIAL_ATTRS = ("mass", "pos", "diaginertia", "fullinertia")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result: dict[str, Any] = {
        "status": "error",
        "blocker": "",
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }
    exit_code = 0
    try:
        summary = run_audit(args)
        result.update(summary)
        result["status"] = summary["status"]
    except Exception as exc:  # pragma: no cover - setup failure path.
        result["status"] = "error"
        result["blocker"] = f"{exc.__class__.__name__}:{exc}"
        exit_code = 1
    print(json.dumps(result, sort_keys=True), flush=True)
    if exit_code:
        raise SystemExit(exit_code)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-path", type=Path, default=None)
    parser.add_argument("--target-bodies", default=",".join(DEFAULT_TARGET_BODIES))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-link-trace", action="store_true")
    parser.add_argument("--n-envs", type=zero_action.positive_int, default=512)
    parser.add_argument("--steps", type=zero_action.positive_int, default=100)
    parser.add_argument(
        "--control-mode",
        choices=zero_action.CONTROL_MODES,
        default="genesis_position",
    )
    parser.add_argument("--pose-profile", choices=zero_action.POSE_PROFILES, default="current")
    parser.add_argument("--gain-profile", choices=zero_action.GAIN_PROFILES, default="current")
    parser.add_argument("--root-z", type=zero_action.positive_float, default=zero_action.DEFAULT_ROOT_Z)
    parser.add_argument("--height-min", type=zero_action.positive_float, default=0.45)
    parser.add_argument("--height-max", type=zero_action.positive_float, default=1.20)
    parser.add_argument("--termination-height-min", type=zero_action.positive_float, default=0.20)
    parser.add_argument("--termination-height-max", type=zero_action.positive_float, default=1.20)
    parser.add_argument("--min-upright", type=zero_action.positive_float, default=zero_action.DEFAULT_MIN_UPRIGHT)
    parser.add_argument("--contact-threshold", type=float, default=1.0)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument("--logical-cuda-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_g1_27dof_nohand_profile()
    asset_path = args.asset_path if args.asset_path is not None else Path(profile.asset.path)
    effective_asset_path = asset_path.as_posix()
    targets = parse_targets(args.target_bodies)
    run_dir = resolve_run_dir(args.output_root, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)

    asset_audit = audit_asset_xml(asset_path=asset_path, target_bodies=targets)
    write_json(run_dir / "asset_audit.json", asset_audit)

    link_trace_summary: dict[str, Any] | None = None
    if args.run_link_trace:
        link_trace_summary = run_link_trace(
            args=args,
            profile=zero_action.profile_with_asset_path(profile, asset_path),
            targets=targets,
            output_path=run_dir / "link_trace.jsonl",
        )

    summary = {
        "status": "completed",
        "blocker": "",
        "run_dir": str(run_dir),
        "asset_audit_path": str(run_dir / "asset_audit.json"),
        "asset_path": effective_asset_path,
        "link_trace_path": None if link_trace_summary is None else str(run_dir / "link_trace.jsonl"),
        "asset_present": asset_audit["asset_present"],
        "target_bodies": targets,
        "missing_count": len(asset_audit["missing"]),
        "link_trace": link_trace_summary,
        "physical_gpu": str(args.physical_gpu),
        "logical_cuda_device": args.logical_cuda_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def audit_asset_xml(*, asset_path: str | Path, target_bodies: Sequence[str]) -> dict[str, Any]:
    path = Path(asset_path)
    missing: list[dict[str, str]] = []
    report: dict[str, Any] = {
        "asset_path": str(asset_path),
        "asset_present": path.is_file(),
        "target_bodies": list(target_bodies),
        "bodies": {},
        "symmetry": [],
        "missing": missing,
    }
    if not path.is_file():
        missing.append({"path": "asset_path", "reason": "asset_missing"})
        return report

    root = ElementTree.parse(path).getroot()
    for body_name in target_bodies:
        body = find_body(root, body_name)
        report["bodies"][body_name] = audit_body(body=body, body_name=body_name, missing=missing)
    report["symmetry"] = compare_left_right_symmetry(report["bodies"])
    return report


def audit_body(
    *,
    body: ElementTree.Element | None,
    body_name: str,
    missing: list[dict[str, str]],
) -> dict[str, Any]:
    if body is None:
        missing.append({"path": f"bodies.{body_name}", "reason": "body_absent"})
        return {
            "present": False,
            "inertial": None,
            "direct_geoms": [],
        }
    inertial_element = body.find("inertial")
    inertial = None if inertial_element is None else select_attrs(inertial_element, INERTIAL_ATTRS)
    if inertial_element is None:
        missing.append({"path": f"bodies.{body_name}.inertial", "reason": "inertial_absent"})
    else:
        for attr in INERTIAL_ATTRS:
            if attr not in inertial:
                missing.append(
                    {
                        "path": f"bodies.{body_name}.inertial.{attr}",
                        "reason": "xml_field_absent",
                    }
                )

    geoms = [audit_geom(geom=geom, body_name=body_name, index=index, missing=missing) for index, geom in enumerate(body.findall("geom"))]
    if not geoms:
        missing.append({"path": f"bodies.{body_name}.direct_geoms", "reason": "geom_absent"})
    return {
        "present": True,
        "inertial": inertial,
        "direct_geoms": geoms,
    }


def audit_geom(
    *,
    geom: ElementTree.Element,
    body_name: str,
    index: int,
    missing: list[dict[str, str]],
) -> dict[str, Any]:
    attrs = dict(sorted(geom.attrib.items()))
    contact_attrs = select_attrs(geom, CONTACT_ATTRS)
    for attr in CONTACT_ATTRS:
        if attr not in contact_attrs:
            missing.append(
                {
                    "path": f"bodies.{body_name}.direct_geoms[{index}].{attr}",
                    "reason": "xml_field_absent",
                }
            )
    return {
        "name": geom.get("name"),
        "attrs": attrs,
        "contact_attrs": contact_attrs,
        "missing_contact_attrs": [attr for attr in CONTACT_ATTRS if attr not in contact_attrs],
    }


def compare_left_right_symmetry(bodies: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_name, left_body in bodies.items():
        if not left_name.startswith("left_"):
            continue
        right_name = "right_" + left_name.removeprefix("left_")
        right_body = bodies.get(right_name)
        if right_body is None:
            continue
        rows.append(
            {
                "left": left_name,
                "right": right_name,
                "both_present": bool(left_body.get("present")) and bool(right_body.get("present")),
                "inertial_match": left_body.get("inertial") == right_body.get("inertial"),
                "direct_geom_count_match": len(left_body.get("direct_geoms", []))
                == len(right_body.get("direct_geoms", [])),
                "contact_attrs_match": geom_contact_attrs(left_body)
                == geom_contact_attrs(right_body),
            }
        )
    return rows


def run_link_trace(
    *,
    args: argparse.Namespace,
    profile: Any,
    targets: Sequence[str],
    output_path: Path,
) -> dict[str, Any]:
    zero_action.verify_cuda_isolation(
        backend=args.backend,
        physical_gpu=str(args.physical_gpu),
        logical_cuda_device=args.logical_cuda_device,
    )
    torch = zero_action.require_torch()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    pose = zero_action.pose_profile_values(args.pose_profile, profile.control.default_angles_rad)
    gains = zero_action.gain_profile_values(args.gain_profile, profile.control)
    profile = zero_action.profile_with_asset_path(profile, args.asset_path)
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
    link_indices = {name: resolve_link_index(backend.robot, name) for name in targets}
    zero_action_tensor = torch.zeros((args.n_envs, profile.action_dim), device=args.logical_cuda_device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for step in range(args.steps):
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
            row = link_trace_row(
                torch=torch,
                robot=backend.robot,
                link_indices=link_indices,
                flags=flags,
                step=step + 1,
                contact_threshold=float(args.contact_threshold),
            )
            rows.append(row)
            append_jsonl(output_path, row)
    return summarize_link_trace(rows=rows, link_indices=link_indices)


def link_trace_row(
    *,
    torch: Any,
    robot: Any,
    link_indices: dict[str, int | None],
    flags: dict[str, Any],
    step: int,
    contact_threshold: float,
) -> dict[str, Any]:
    return {
        "policy_step": step,
        "root_height_min": tensor_min(flags["root_height"]),
        "root_height_mean": tensor_mean(flags["root_height"]),
        "upright_min": tensor_min(flags["upright"]),
        "upright_mean": tensor_mean(flags["upright"]),
        "height_bad_count": tensor_bool_count(flags["height_bad"]),
        "termination_height_bad_count": tensor_bool_count(flags["termination_height_bad"]),
        "tilt_bad_count": tensor_bool_count(flags["tilt_bad"]),
        "links": {
            name: read_link_sample(
                robot=robot,
                link_idx=link_idx,
                contact_threshold=contact_threshold,
            )
            for name, link_idx in link_indices.items()
        },
    }


def summarize_link_trace(
    *,
    rows: Sequence[dict[str, Any]],
    link_indices: dict[str, int | None],
) -> dict[str, Any]:
    link_values: dict[str, dict[str, Any]] = {}
    for name in link_indices:
        samples = [
            row.get("links", {}).get(
                name,
                {
                    "z_mean": None,
                    "contact_force_max": None,
                    "contact_env_count": 0,
                },
            )
            for row in rows
        ]
        z_values = [sample["z_mean"] for sample in samples if sample.get("z_mean") is not None]
        force_values = [
            sample["contact_force_max"]
            for sample in samples
            if sample.get("contact_force_max") is not None
        ]
        link_values[name] = {
            "link_idx": link_indices[name],
            "z_min": min(z_values) if z_values else None,
            "contact_force_max": max(force_values) if force_values else None,
            "max_contact_env_count": max(
                int(sample.get("contact_env_count", 0) or 0) for sample in samples
            )
            if samples
            else 0,
        }
    return {
        "steps": len(rows),
        "first_tilt_step": next(
            (int(row["policy_step"]) for row in rows if int(row["tilt_bad_count"]) > 0),
            None,
        ),
        "max_tilt_bad_count": max((int(row["tilt_bad_count"]) for row in rows), default=0),
        "max_termination_height_bad_count": max(
            (int(row["termination_height_bad_count"]) for row in rows),
            default=0,
        ),
        "links": link_values,
        "unresolved_links": [name for name, index in link_indices.items() if index is None],
    }


def read_link_sample(
    *,
    robot: Any,
    link_idx: int | None,
    contact_threshold: float,
) -> dict[str, Any]:
    if link_idx is None:
        return {
            "z_min": None,
            "z_mean": None,
            "contact_force_max": None,
            "contact_force_mean": None,
            "contact_env_count": 0,
            "available": False,
        }
    z_values = read_link_z_values(robot, link_idx)
    force_values = read_link_contact_force_values(robot, link_idx)
    return {
        "z_min": min(z_values) if z_values else None,
        "z_mean": sum(z_values) / len(z_values) if z_values else None,
        "contact_force_max": max(force_values) if force_values else None,
        "contact_force_mean": sum(force_values) / len(force_values) if force_values else None,
        "contact_env_count": sum(1 for value in force_values if value >= contact_threshold),
        "available": bool(z_values or force_values),
    }


def read_link_z_values(robot: Any, link_idx: int) -> list[float]:
    if not hasattr(robot, "get_links_pos"):
        return []
    try:
        return z_values_from_any(robot.get_links_pos(links_idx_local=(link_idx,)), link_idx=link_idx)
    except Exception:
        pass
    try:
        return z_values_from_any(robot.get_links_pos(), link_idx=link_idx)
    except Exception:
        return []


def read_link_contact_force_values(robot: Any, link_idx: int) -> list[float]:
    for method_name in ("get_links_net_contact_force", "get_links_net_contact_forces"):
        if not hasattr(robot, method_name):
            continue
        method = getattr(robot, method_name)
        try:
            values = method(links_idx_local=(link_idx,))
        except TypeError:
            values = method()
        except Exception:
            continue
        return force_values_from_any(values, link_idx=link_idx)
    return []


def z_values_from_any(value: Any, *, link_idx: int) -> list[float]:
    return [float(vector[-1]) for vector in link_vectors_from_any(value, link_idx=link_idx) if vector]


def force_values_from_any(value: Any, *, link_idx: int) -> list[float]:
    vectors = link_vectors_from_any(value, link_idx=link_idx)
    if not vectors:
        return []
    return [
        math.sqrt(sum(float(component) * float(component) for component in vector[-3:]))
        for vector in vectors
        if len(vector) >= 3
    ]


def link_vectors_from_any(value: Any, *, link_idx: int) -> list[list[float]]:
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
        vectors.extend(link_vectors_from_any(row, link_idx=link_idx))
    return vectors


def vectors_from_any(value: Any) -> list[list[float]]:
    return list(iter_numeric_vectors(python_value(value)))


def python_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def is_numeric_sequence(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, (int, float, bool)) for item in value)


def iter_numeric_vectors(value: Any) -> Iterable[list[float]]:
    if isinstance(value, (int, float, bool)):
        yield [float(value)]
        return
    if not isinstance(value, list):
        value = list(value)
    if not value:
        return
    if all(isinstance(item, (int, float, bool)) for item in value):
        numbers = [float(item) for item in value]
        if len(numbers) > 3 and len(numbers) % 3 == 0:
            for index in range(0, len(numbers), 3):
                yield numbers[index : index + 3]
        else:
            yield numbers
        return
    for item in value:
        yield from iter_numeric_vectors(item)


def rows_from_any(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [[float(value)]]
    if not isinstance(value, list):
        value = list(value)
    if not value:
        return []
    if all(isinstance(item, (int, float, bool)) for item in value):
        return [[float(item) for item in value]]
    return [[float(item) for item in row] for row in value]


def resolve_link_index(robot: Any, link_name: str) -> int | None:
    if not hasattr(robot, "get_link"):
        return None
    try:
        link = robot.get_link(link_name)
    except Exception:
        return None
    for attr in ("idx_local", "idx", "link_idx", "id"):
        if not hasattr(link, attr):
            continue
        value = getattr(link, attr)
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            value = value[0]
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def parse_targets(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def find_body(root: ElementTree.Element, name: str) -> ElementTree.Element | None:
    return root.find(f".//body[@name='{name}']")


def select_attrs(element: ElementTree.Element, names: Iterable[str]) -> dict[str, str]:
    return {name: element.attrib[name] for name in names if name in element.attrib}


def geom_contact_attrs(body: dict[str, Any]) -> list[dict[str, str]]:
    return [geom.get("contact_attrs", {}) for geom in body.get("direct_geoms", [])]


def flatten_numeric(value: Any) -> list[float]:
    return [item for row in rows_from_any(value) for item in row]


def tensor_min(value: Any) -> float:
    if hasattr(value, "min"):
        return float(value.min().item())
    values = flatten_numeric(value)
    return min(values) if values else 0.0


def tensor_mean(value: Any) -> float:
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "mean"):
        return float(value.mean().item())
    values = flatten_numeric(value)
    return sum(values) / len(values) if values else 0.0


def tensor_bool_count(value: Any) -> int:
    if hasattr(value, "sum"):
        return int(value.sum().item())
    return sum(1 for item in flatten_numeric(value) if bool(item))


def resolve_run_dir(output_root: Path, run_id: str) -> Path:
    root = output_root if output_root.is_absolute() else Path.cwd() / output_root
    run_name = run_id.strip() or time.strftime("%Y%m%d-%H%M%S")
    run_dir = (root / run_name).resolve()
    project_prefix = zero_action.PROJECT_PREFIX.resolve()
    if project_prefix.exists() and project_prefix not in (run_dir, *run_dir.parents):
        raise RuntimeError(f"output path must stay under {project_prefix}: {run_dir}")
    return run_dir


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
