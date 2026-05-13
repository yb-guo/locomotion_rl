"""Generate project-local G1 ankle-roll MJCF contact patch variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
)
SUPPORT_SIZE = "0.005"
DEFAULT_LARGER_SPHERE_SIZE = "0.012"
DEFAULT_FRICTION = "1.0 0.02 0.001"
DEFAULT_CONDIM = "4"
DEFAULT_PRIORITY = "1"
BOX_SUPPORT_SIZE = "0.035 0.020 0.006"
BOX_SUPPORT_POS = "0 0 -0.006"


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
