"""Generate project-local G1 ankle-roll MJCF contact patch variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import time
from typing import Any, Sequence
from xml.etree import ElementTree

from h200_locomotion_lab.robots import load_g1_27dof_nohand_profile


DEFAULT_OUTPUT_ROOT = Path("outputs/task022/ankle_roll_contact_patch")
DEFAULT_TARGET_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
VARIANTS = (
    "ankle_roll_friction_attrs",
    "ankle_roll_larger_spheres",
    "ankle_roll_box_support",
    "ankle_roll_mesh_collision",
    "ankle_roll_sole_collision",
    "ankle_roll_center_sole_keep_points",
    "ankle_roll_center_sole_no_points",
    "ankle_roll_mesh_bbox_sole_no_points",
    "ankle_roll_edge_boxes_no_points",
    "ankle_roll_hybrid_edge_boxes_no_points",
)
SUPPORT_SIZE = "0.005"
DEFAULT_LARGER_SPHERE_SIZE = "0.012"
DEFAULT_FRICTION = "1.0 0.02 0.001"
DEFAULT_CONDIM = "4"
DEFAULT_PRIORITY = "1"
BOX_SUPPORT_SIZE = "0.035 0.020 0.006"
BOX_SUPPORT_POS = "0 0 -0.006"
SOLE_HALF_HEIGHT = 0.004
MESH_BBOX_SOLE_HALF_HEIGHT = 0.006
MESH_BBOX_SOLE_CENTER_Z = -0.006
EDGE_BOX_SIZE = "0.010 0.030 0.004"
EDGE_BOX_Z = -0.031
EDGE_BOX_XS = (-0.05, 0.12)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_patch_generation(args)
    print(json.dumps(summary, sort_keys=True), flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-asset", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--target-bodies", default=",".join(DEFAULT_TARGET_BODIES))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--larger-sphere-size", default=DEFAULT_LARGER_SPHERE_SIZE)
    parser.add_argument("--friction", default=DEFAULT_FRICTION)
    parser.add_argument("--condim", default=DEFAULT_CONDIM)
    parser.add_argument("--priority", default=DEFAULT_PRIORITY)
    parser.add_argument("--box-size", default=BOX_SUPPORT_SIZE)
    parser.add_argument("--box-pos", default=BOX_SUPPORT_POS)
    return parser.parse_args(argv)


def run_patch_generation(args: argparse.Namespace) -> dict[str, Any]:
    source_asset = resolve_source_asset(args.source_asset)
    variants = parse_csv(args.variants)
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    targets = parse_csv(args.target_bodies)
    run_dir = resolve_run_dir(args.output_root, args.run_id)
    assets_dir = run_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=False)

    source_before = source_asset.read_bytes()
    variant_reports: dict[str, Any] = {}
    for variant in variants:
        output_path = assets_dir / f"{source_asset.stem}.{variant}{source_asset.suffix}"
        if output_path.resolve() == source_asset.resolve():
            raise RuntimeError(f"refusing to overwrite source asset: {source_asset}")
        variant_reports[variant] = write_variant(
            source_asset=source_asset,
            output_path=output_path,
            variant=variant,
            target_bodies=targets,
            args=args,
        )

    source_after = source_asset.read_bytes()
    meshdir_reports = [
        report["meshdir_handling"] for report in variant_reports.values()
    ]
    summary = {
        "status": "completed",
        "source_path": str(source_asset),
        "source_unchanged": source_before == source_after,
        "run_dir": str(run_dir),
        "assets_dir": str(assets_dir),
        "meshdir_handling": meshdir_reports[0]
        if meshdir_reports
        else inspect_compiler_meshdir(source_asset),
        "target_bodies": targets,
        "variants": variant_reports,
        "missing": [
            row
            for report in variant_reports.values()
            for row in report.get("missing", [])
        ],
        "errors": [
            row
            for report in variant_reports.values()
            for row in report.get("errors", [])
        ],
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def write_variant(
    *,
    source_asset: Path,
    output_path: Path,
    variant: str,
    target_bodies: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    tree = ElementTree.parse(source_asset)
    root = tree.getroot()
    missing: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    changed_bodies: dict[str, Any] = {}
    meshdir_handling = rewrite_relative_compiler_meshdir(
        root=root,
        source_asset=source_asset,
    )

    for body_name in target_bodies:
        body = find_body(root, body_name)
        if body is None:
            missing.append(missing_row(variant, body_name, "body", "body_absent"))
            changed_bodies[body_name] = {"present": False, "changed_geoms": []}
            continue
        support_geoms = select_support_geoms(body)
        if not support_geoms:
            missing.append(
                missing_row(variant, body_name, "support_geoms", "support_geom_absent")
            )
        if variant == "ankle_roll_friction_attrs":
            changed = apply_friction_attrs(
                support_geoms=support_geoms,
                body_name=body_name,
                friction=args.friction,
                condim=args.condim,
                priority=args.priority,
            )
        elif variant == "ankle_roll_larger_spheres":
            changed = apply_larger_spheres(
                support_geoms=support_geoms,
                body_name=body_name,
                larger_size=args.larger_sphere_size,
            )
        elif variant == "ankle_roll_box_support":
            changed = apply_box_support(
                body=body,
                body_name=body_name,
                box_size=args.box_size,
                box_pos=args.box_pos,
                friction=args.friction,
                condim=args.condim,
                priority=args.priority,
            )
        elif variant == "ankle_roll_mesh_collision":
            changed = apply_mesh_collision(
                body=body,
                body_name=body_name,
                friction=args.friction,
                condim=args.condim,
                priority=args.priority,
                missing=missing,
                variant=variant,
            )
        elif variant == "ankle_roll_sole_collision":
            changed = apply_sole_collision(
                body=body,
                body_name=body_name,
                support_geoms=support_geoms,
                friction=args.friction,
                condim=args.condim,
                priority=args.priority,
                missing=missing,
                variant=variant,
            )
        elif variant == "ankle_roll_center_sole_keep_points":
            changed = apply_explicit_box_support(
                body=body,
                body_name=body_name,
                name=f"{body_name}_task023_center_sole_keep_points",
                box_size=args.box_size,
                box_pos=args.box_pos,
                friction=args.friction,
                condim=args.condim,
                priority=args.priority,
            )
        elif variant == "ankle_roll_center_sole_no_points":
            changed = disable_support_geoms(support_geoms, body_name=body_name)
            changed.extend(
                apply_explicit_box_support(
                    body=body,
                    body_name=body_name,
                    name=f"{body_name}_task023_center_sole_no_points",
                    box_size=args.box_size,
                    box_pos=args.box_pos,
                    friction=args.friction,
                    condim=args.condim,
                    priority=args.priority,
                )
            )
        elif variant == "ankle_roll_mesh_bbox_sole_no_points":
            changed = apply_mesh_bbox_sole(
                root=root,
                source_asset=source_asset,
                body=body,
                body_name=body_name,
                support_geoms=support_geoms,
                friction=args.friction,
                condim=args.condim,
                priority=args.priority,
                missing=missing,
                variant=variant,
            )
        elif variant == "ankle_roll_edge_boxes_no_points":
            changed = disable_support_geoms(support_geoms, body_name=body_name)
            changed.extend(
                apply_edge_boxes(
                    body=body,
                    body_name=body_name,
                    prefix="edge_boxes_no_points",
                    friction=args.friction,
                    condim=args.condim,
                    priority=args.priority,
                )
            )
        elif variant == "ankle_roll_hybrid_edge_boxes_no_points":
            changed = disable_support_geoms(support_geoms, body_name=body_name)
            changed.extend(
                apply_explicit_box_support(
                    body=body,
                    body_name=body_name,
                    name=f"{body_name}_task023_hybrid_center_pad",
                    box_size=args.box_size,
                    box_pos=args.box_pos,
                    friction=args.friction,
                    condim=args.condim,
                    priority=args.priority,
                )
            )
            changed.extend(
                apply_edge_boxes(
                    body=body,
                    body_name=body_name,
                    prefix="hybrid_edge_boxes",
                    friction=args.friction,
                    condim=args.condim,
                    priority=args.priority,
                )
            )
        else:  # pragma: no cover - guarded by run_patch_generation.
            errors.append(missing_row(variant, body_name, "variant", "unknown_variant"))
            changed = []
        changed_bodies[body_name] = {
            "present": True,
            "support_geom_count": len(support_geoms),
            "changed_geoms": changed,
        }

    indent_xml(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return {
        "path": str(output_path),
        "target_bodies": target_bodies,
        "meshdir_handling": meshdir_handling,
        "changed_bodies": changed_bodies,
        "changed_geom_count": sum(
            len(body_report.get("changed_geoms", []))
            for body_report in changed_bodies.values()
        ),
        "missing": missing,
        "errors": errors,
    }


def rewrite_relative_compiler_meshdir(
    *,
    root: ElementTree.Element,
    source_asset: Path,
) -> dict[str, Any]:
    compiler = root.find("compiler")
    if compiler is None:
        return {
            "compiler_present": False,
            "source_meshdir": None,
            "resolved_source_meshdir": None,
            "output_meshdir": None,
            "rewritten": False,
        }

    source_meshdir = compiler.get("meshdir")
    if source_meshdir is None:
        return {
            "compiler_present": True,
            "source_meshdir": None,
            "resolved_source_meshdir": None,
            "output_meshdir": None,
            "rewritten": False,
        }

    source_meshdir_path = Path(source_meshdir)
    if source_meshdir_path.is_absolute():
        resolved_source_meshdir = source_meshdir_path.resolve()
        rewritten = False
    else:
        resolved_source_meshdir = (source_asset.parent / source_meshdir_path).resolve()
        compiler.set("meshdir", str(resolved_source_meshdir))
        rewritten = True

    return {
        "compiler_present": True,
        "source_meshdir": source_meshdir,
        "resolved_source_meshdir": str(resolved_source_meshdir),
        "output_meshdir": compiler.get("meshdir"),
        "rewritten": rewritten,
    }


def inspect_compiler_meshdir(source_asset: Path) -> dict[str, Any]:
    tree = ElementTree.parse(source_asset)
    return rewrite_relative_compiler_meshdir(
        root=tree.getroot(),
        source_asset=source_asset,
    )


def apply_friction_attrs(
    *,
    support_geoms: Sequence[ElementTree.Element],
    body_name: str,
    friction: str,
    condim: str,
    priority: str,
) -> list[dict[str, Any]]:
    changes = []
    for index, geom in enumerate(support_geoms):
        changes.append(
            set_attrs(
                geom,
                body_name=body_name,
                geom_index=index,
                attrs={"friction": friction, "condim": condim, "priority": priority},
            )
        )
    return changes


def apply_larger_spheres(
    *,
    support_geoms: Sequence[ElementTree.Element],
    body_name: str,
    larger_size: str,
) -> list[dict[str, Any]]:
    changes = []
    for index, geom in enumerate(support_geoms):
        changes.append(
            set_attrs(
                geom,
                body_name=body_name,
                geom_index=index,
                attrs={"size": larger_size},
            )
        )
    return changes


def apply_box_support(
    *,
    body: ElementTree.Element,
    body_name: str,
    box_size: str,
    box_pos: str,
    friction: str,
    condim: str,
    priority: str,
) -> list[dict[str, Any]]:
    name = f"{body_name}_task022_box_support"
    geom = ElementTree.Element(
        "geom",
        {
            "name": name,
            "type": "box",
            "size": box_size,
            "pos": box_pos,
            "friction": friction,
            "condim": condim,
            "priority": priority,
        },
    )
    body.append(geom)
    return [
        {
            "body": body_name,
            "geom": name,
            "geom_index": len(body.findall("geom")) - 1,
            "before": None,
            "after": dict(sorted(geom.attrib.items())),
            "changed_attrs": dict(sorted(geom.attrib.items())),
        }
    ]


def apply_explicit_box_support(
    *,
    body: ElementTree.Element,
    body_name: str,
    name: str,
    box_size: str,
    box_pos: str,
    friction: str,
    condim: str,
    priority: str,
) -> list[dict[str, Any]]:
    geom = ElementTree.Element(
        "geom",
        {
            "name": name,
            "type": "box",
            "size": box_size,
            "pos": box_pos,
            "friction": friction,
            "condim": condim,
            "priority": priority,
            "contype": "1",
            "conaffinity": "1",
        },
    )
    body.append(geom)
    return [
        {
            "body": body_name,
            "geom": name,
            "geom_index": len(body.findall("geom")) - 1,
            "before": None,
            "after": dict(sorted(geom.attrib.items())),
            "changed_attrs": dict(sorted(geom.attrib.items())),
        }
    ]


def disable_support_geoms(
    support_geoms: Sequence[ElementTree.Element],
    *,
    body_name: str,
) -> list[dict[str, Any]]:
    return [
        set_attrs(
            geom,
            body_name=body_name,
            geom_index=index,
            attrs={"contype": "0", "conaffinity": "0"},
        )
        for index, geom in enumerate(support_geoms)
    ]


def apply_edge_boxes(
    *,
    body: ElementTree.Element,
    body_name: str,
    prefix: str,
    friction: str,
    condim: str,
    priority: str,
) -> list[dict[str, Any]]:
    changes = []
    for label, x_pos in (("heel", EDGE_BOX_XS[0]), ("toe", EDGE_BOX_XS[1])):
        name = f"{body_name}_task023_{prefix}_{label}"
        changes.extend(
            apply_explicit_box_support(
                body=body,
                body_name=body_name,
                name=name,
                box_size=EDGE_BOX_SIZE,
                box_pos=format_vec3((x_pos, 0.0, EDGE_BOX_Z)),
                friction=friction,
                condim=condim,
                priority=priority,
            )
        )
    return changes


def apply_mesh_collision(
    *,
    body: ElementTree.Element,
    body_name: str,
    friction: str,
    condim: str,
    priority: str,
    missing: list[dict[str, str]],
    variant: str,
) -> list[dict[str, Any]]:
    visual = select_visual_mesh_geom(body)
    if visual is None or not visual.get("mesh"):
        missing.append(missing_row(variant, body_name, "visual_mesh", "visual_mesh_absent"))
        return []

    name = f"{body_name}_task023_mesh_collision"
    geom = ElementTree.Element(
        "geom",
        {
            "name": name,
            "type": "mesh",
            "mesh": str(visual.get("mesh")),
            "friction": friction,
            "condim": condim,
            "priority": priority,
            "contype": "1",
            "conaffinity": "1",
        },
    )
    body.append(geom)
    return [
        {
            "body": body_name,
            "geom": name,
            "geom_index": len(body.findall("geom")) - 1,
            "before": None,
            "after": dict(sorted(geom.attrib.items())),
            "changed_attrs": dict(sorted(geom.attrib.items())),
        }
    ]


def apply_sole_collision(
    *,
    body: ElementTree.Element,
    body_name: str,
    support_geoms: Sequence[ElementTree.Element],
    friction: str,
    condim: str,
    priority: str,
    missing: list[dict[str, str]],
    variant: str,
) -> list[dict[str, Any]]:
    footprint = support_footprint(support_geoms)
    if footprint is None:
        missing.append(missing_row(variant, body_name, "support_footprint", "support_footprint_absent"))
        return []

    changes = [
        set_attrs(
            geom,
            body_name=body_name,
            geom_index=index,
            attrs={"contype": "0", "conaffinity": "0"},
        )
        for index, geom in enumerate(support_geoms)
    ]
    name = f"{body_name}_task023_sole_collision"
    geom = ElementTree.Element(
        "geom",
        {
            "name": name,
            "type": "box",
            "size": footprint["size"],
            "pos": footprint["pos"],
            "friction": friction,
            "condim": condim,
            "priority": priority,
            "contype": "1",
            "conaffinity": "1",
        },
    )
    body.append(geom)
    changes.append(
        {
            "body": body_name,
            "geom": name,
            "geom_index": len(body.findall("geom")) - 1,
            "before": None,
            "after": dict(sorted(geom.attrib.items())),
            "changed_attrs": dict(sorted(geom.attrib.items())),
        }
    )
    return changes


def apply_mesh_bbox_sole(
    *,
    root: ElementTree.Element,
    source_asset: Path,
    body: ElementTree.Element,
    body_name: str,
    support_geoms: Sequence[ElementTree.Element],
    friction: str,
    condim: str,
    priority: str,
    missing: list[dict[str, str]],
    variant: str,
) -> list[dict[str, Any]]:
    bbox = visual_mesh_bbox(
        root=root,
        source_asset=source_asset,
        body=body,
        body_name=body_name,
        missing=missing,
        variant=variant,
    )
    if bbox is None:
        return []

    mins, maxs = bbox
    box_size = format_vec3(
        (
            (maxs[0] - mins[0]) / 2.0,
            (maxs[1] - mins[1]) / 2.0,
            MESH_BBOX_SOLE_HALF_HEIGHT,
        )
    )
    box_pos = format_vec3(
        (
            (mins[0] + maxs[0]) / 2.0,
            (mins[1] + maxs[1]) / 2.0,
            MESH_BBOX_SOLE_CENTER_Z,
        )
    )
    changes = disable_support_geoms(support_geoms, body_name=body_name)
    changes.extend(
        apply_explicit_box_support(
            body=body,
            body_name=body_name,
            name=f"{body_name}_task023_mesh_bbox_sole_no_points",
            box_size=box_size,
            box_pos=box_pos,
            friction=friction,
            condim=condim,
            priority=priority,
        )
    )
    return changes


def visual_mesh_bbox(
    *,
    root: ElementTree.Element,
    source_asset: Path,
    body: ElementTree.Element,
    body_name: str,
    missing: list[dict[str, str]],
    variant: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    visual = select_visual_mesh_geom(body)
    if visual is None or not visual.get("mesh"):
        missing.append(missing_row(variant, body_name, "visual_mesh", "visual_mesh_absent"))
        return None

    mesh = find_mesh_asset(root, str(visual.get("mesh")))
    if mesh is None or not mesh.get("file"):
        missing.append(missing_row(variant, body_name, "mesh_asset", "mesh_asset_absent"))
        return None

    mesh_file = resolve_mesh_file(root=root, source_asset=source_asset, mesh=mesh)
    if not mesh_file.is_file():
        missing.append(missing_row(variant, body_name, "mesh_file", "mesh_file_absent"))
        return None

    vertices = read_mesh_vertices(mesh_file)
    if not vertices:
        missing.append(missing_row(variant, body_name, "mesh_vertices", "mesh_vertices_absent"))
        return None

    scale = parse_mesh_scale(mesh.get("scale"))
    scaled = [
        (vertex[0] * scale[0], vertex[1] * scale[1], vertex[2] * scale[2])
        for vertex in vertices
    ]
    mins = tuple(min(vertex[index] for vertex in scaled) for index in range(3))
    maxs = tuple(max(vertex[index] for vertex in scaled) for index in range(3))
    return mins, maxs


def support_footprint(
    support_geoms: Sequence[ElementTree.Element],
) -> dict[str, str] | None:
    samples = []
    for geom in support_geoms:
        pos = parse_vec3(geom.get("pos"))
        radius = parse_first_float(geom.get("size"))
        if pos is None or radius is None:
            continue
        samples.append((pos, radius))
    if not samples:
        return None

    min_x = min(pos[0] - radius for pos, radius in samples)
    max_x = max(pos[0] + radius for pos, radius in samples)
    min_y = min(pos[1] - radius for pos, radius in samples)
    max_y = max(pos[1] + radius for pos, radius in samples)
    min_z = min(pos[2] - radius for pos, radius in samples)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    center_z = min_z + SOLE_HALF_HEIGHT
    size_x = (max_x - min_x) / 2.0
    size_y = (max_y - min_y) / 2.0
    return {
        "pos": format_vec3((center_x, center_y, center_z)),
        "size": format_vec3((size_x, size_y, SOLE_HALF_HEIGHT)),
    }


def parse_vec3(value: str | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    tokens = value.split()
    if len(tokens) < 3:
        return None
    try:
        return float(tokens[0]), float(tokens[1]), float(tokens[2])
    except ValueError:
        return None


def parse_mesh_scale(value: str | None) -> tuple[float, float, float]:
    if value is None:
        return (1.0, 1.0, 1.0)
    tokens = value.split()
    if len(tokens) == 1:
        try:
            scale = float(tokens[0])
        except ValueError:
            return (1.0, 1.0, 1.0)
        return (scale, scale, scale)
    if len(tokens) >= 3:
        try:
            return float(tokens[0]), float(tokens[1]), float(tokens[2])
        except ValueError:
            return (1.0, 1.0, 1.0)
    return (1.0, 1.0, 1.0)


def find_mesh_asset(root: ElementTree.Element, mesh_name: str) -> ElementTree.Element | None:
    return root.find(f".//mesh[@name='{mesh_name}']")


def resolve_mesh_file(
    *,
    root: ElementTree.Element,
    source_asset: Path,
    mesh: ElementTree.Element,
) -> Path:
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else None
    base = Path(meshdir) if meshdir else source_asset.parent
    if not base.is_absolute():
        base = source_asset.parent / base
    return (base / str(mesh.get("file"))).resolve()


def read_mesh_vertices(path: Path) -> list[tuple[float, float, float]]:
    suffix = path.suffix.lower()
    if suffix == ".stl":
        return read_stl_vertices(path)
    if suffix == ".obj":
        return read_obj_vertices(path)
    return []


def read_obj_vertices(path: Path) -> list[tuple[float, float, float]]:
    vertices = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        _, x, y, z, *_ = line.split()
        vertices.append((float(x), float(y), float(z)))
    return vertices


def read_stl_vertices(path: Path) -> list[tuple[float, float, float]]:
    data = path.read_bytes()
    prefix = data[:512].decode(errors="ignore").lower()
    if prefix.lstrip().startswith("solid") and "vertex" in prefix:
        vertices = []
        for line in data.decode(errors="ignore").splitlines():
            line = line.strip()
            if not line.startswith("vertex "):
                continue
            _, x, y, z = line.split()
            vertices.append((float(x), float(y), float(z)))
        return vertices

    if len(data) < 84:
        return []
    triangle_count = int.from_bytes(data[80:84], byteorder="little", signed=False)
    expected_length = 84 + triangle_count * 50
    if len(data) < expected_length:
        return []

    vertices = []
    offset = 84
    for _ in range(triangle_count):
        offset += 12
        for _ in range(3):
            vertices.append(struct.unpack_from("<fff", data, offset))
            offset += 12
        offset += 2
    return vertices


def parse_first_float(value: str | None) -> float | None:
    token = first_size_token(value)
    if token is None:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def format_vec3(values: Sequence[float]) -> str:
    return " ".join(format_float(value) for value in values)


def format_float(value: float) -> str:
    text = f"{value:.9g}"
    return "0" if text == "-0" else text


def set_attrs(
    geom: ElementTree.Element,
    *,
    body_name: str,
    geom_index: int,
    attrs: dict[str, str],
) -> dict[str, Any]:
    before = dict(sorted(geom.attrib.items()))
    for name, value in attrs.items():
        geom.set(name, value)
    after = dict(sorted(geom.attrib.items()))
    return {
        "body": body_name,
        "geom": geom.get("name"),
        "geom_index": geom_index,
        "before": before,
        "after": after,
        "changed_attrs": {
            name: {"before": before.get(name), "after": after.get(name)}
            for name in attrs
            if before.get(name) != after.get(name)
        },
    }


def select_support_geoms(body: ElementTree.Element) -> list[ElementTree.Element]:
    geoms = []
    for geom in body.findall("geom"):
        if is_visual_noncolliding_geom(geom):
            continue
        if is_point_support_geom(geom):
            geoms.append(geom)
    return geoms


def select_visual_mesh_geom(body: ElementTree.Element) -> ElementTree.Element | None:
    for geom in body.findall("geom"):
        if geom.get("type") == "mesh" and is_visual_noncolliding_geom(geom):
            return geom
    return None


def is_visual_noncolliding_geom(geom: ElementTree.Element) -> bool:
    return geom.get("contype") == "0" and geom.get("conaffinity") == "0"


def is_point_support_geom(geom: ElementTree.Element) -> bool:
    geom_type = geom.get("type", "sphere")
    return (
        geom_type in {"sphere", "ellipsoid"}
        and first_size_token(geom.get("size")) == SUPPORT_SIZE
    )


def first_size_token(size: str | None) -> str | None:
    if size is None:
        return None
    tokens = size.split()
    return tokens[0] if tokens else None


def resolve_source_asset(source_asset: Path | None) -> Path:
    if source_asset is None:
        profile = load_g1_27dof_nohand_profile()
        source_asset = Path(profile.asset.path)
    path = source_asset if source_asset.is_absolute() else (Path.cwd() / source_asset)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source asset does not exist: {path}")
    return path


def resolve_run_dir(output_root: Path, run_id: str) -> Path:
    root = output_root if output_root.is_absolute() else Path.cwd() / output_root
    run_name = run_id.strip() or time.strftime("%Y%m%d-%H%M%S")
    return (root / run_name).resolve()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def find_body(root: ElementTree.Element, name: str) -> ElementTree.Element | None:
    return root.find(f".//body[@name='{name}']")


def missing_row(variant: str, body_name: str, path: str, reason: str) -> dict[str, str]:
    return {
        "variant": variant,
        "body": body_name,
        "path": f"bodies.{body_name}.{path}",
        "reason": reason,
    }


def indent_xml(element: ElementTree.Element) -> None:
    try:
        ElementTree.indent(element, space="  ")
    except AttributeError:  # pragma: no cover - Python >=3.11 in pyproject.
        return


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
