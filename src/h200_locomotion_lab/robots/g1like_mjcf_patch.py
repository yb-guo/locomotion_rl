"""Conservative MJCF patching for Task038 G1-like morphology variants."""

from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from h200_locomotion_lab.robots.g1like_morphology import (
    G1LIKE_SLOT_SCHEMA_ID,
    NONPHYSICAL_SCALING_NOTE,
    SCALE_FIELDS,
    joint_order_hash,
    slot_schema_hash,
)
from h200_locomotion_lab.robots.g1like_slots import G1LIKE_ACTION_DIM


POSITION_TAGS = ("body", "joint", "geom", "site")
ACTUATOR_RANGE_ATTRS = ("ctrlrange", "forcerange")
ACTUATOR_VECTOR_ATTRS = ("gear",)


class G1LikeMJCFPatchError(ValueError):
    """Raised when a G1-like MJCF patch request is invalid."""


def patch_g1like_mjcf_variant(
    *,
    source_mjcf: str | Path,
    output_mjcf: str | Path,
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    """Patch a source MJCF into one variant artifact and return a JSON summary."""

    source_path = Path(source_mjcf)
    output_path = Path(output_mjcf)
    _validate_variant(variant)

    tree = ET.parse(source_path)
    root = tree.getroot()
    topology_before = mjcf_topology(root)
    meshdir_info = _rewrite_relative_meshdir_for_output(root, source_path, output_path)
    patched_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    limitation_notes: list[str] = []

    link_scale = float(variant["link_scale"])
    mass_scale = float(variant["mass_scale"])
    com_scale = float(variant["com_scale"])
    inertia_scale = float(variant["inertia_scale"])
    motor_strength_scale = float(variant["motor_strength_scale"])

    _scale_pos_attrs(root, link_scale, patched_counts, skipped_counts)
    _scale_inertials(root, mass_scale, com_scale, inertia_scale, patched_counts, skipped_counts)
    _scale_motor_attrs(root, motor_strength_scale, patched_counts, skipped_counts)

    if any(scale != 1.0 for scale in (mass_scale, com_scale, inertia_scale)):
        limitation_notes.append(NONPHYSICAL_SCALING_NOTE)
    _append_missing_attr_limitations(skipped_counts, limitation_notes)

    topology_after = mjcf_topology(root)
    if topology_after != topology_before:
        raise G1LikeMJCFPatchError("patch mutated MJCF topology names/order")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    local_parse_ok = _local_parse_ok(output_path)
    summary = _summary(
        variant=variant,
        source_path=source_path,
        output_path=output_path,
        topology_before=topology_before,
        topology_after=topology_after,
        patched_counts=patched_counts,
        skipped_counts=skipped_counts,
        limitation_notes=limitation_notes,
        local_parse_ok=local_parse_ok,
        meshdir_info=meshdir_info,
    )
    if not local_parse_ok:
        summary["pass"] = False
        summary["limitation_notes"].append("Patched XML failed local ElementTree parse.")
    return summary


def patch_g1like_mjcf_string(
    source_xml: str,
    *,
    variant: Mapping[str, Any],
    source_path: str = "<string>",
    output_path: str = "<memory>",
) -> tuple[str, dict[str, Any]]:
    """Patch MJCF text in memory for unit tests and small contract checks."""

    _validate_variant(variant)
    root = ET.fromstring(source_xml)
    topology_before = mjcf_topology(root)
    meshdir_info = _meshdir_info(root, Path(source_path))
    patched_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    limitation_notes: list[str] = []

    _scale_pos_attrs(root, float(variant["link_scale"]), patched_counts, skipped_counts)
    _scale_inertials(
        root,
        float(variant["mass_scale"]),
        float(variant["com_scale"]),
        float(variant["inertia_scale"]),
        patched_counts,
        skipped_counts,
    )
    _scale_motor_attrs(
        root,
        float(variant["motor_strength_scale"]),
        patched_counts,
        skipped_counts,
    )
    if any(
        float(variant[field]) != 1.0
        for field in ("mass_scale", "com_scale", "inertia_scale")
    ):
        limitation_notes.append(NONPHYSICAL_SCALING_NOTE)
    _append_missing_attr_limitations(skipped_counts, limitation_notes)

    topology_after = mjcf_topology(root)
    if topology_after != topology_before:
        raise G1LikeMJCFPatchError("patch mutated MJCF topology names/order")

    patched_xml = ET.tostring(root, encoding="unicode")
    ET.fromstring(patched_xml)
    return patched_xml, _summary(
        variant=variant,
        source_path=Path(source_path),
        output_path=Path(output_path),
        topology_before=topology_before,
        topology_after=topology_after,
        patched_counts=patched_counts,
        skipped_counts=skipped_counts,
        limitation_notes=limitation_notes,
        local_parse_ok=True,
        meshdir_info=meshdir_info,
    )


def mjcf_topology(root: ET.Element) -> dict[str, list[str]]:
    """Return topology-sensitive names in document order."""

    return {
        "body_names": _names(root.iter("body")),
        "joint_names": _names(root.iter("joint")),
        "actuator_names": _actuator_names(root),
    }


def _scale_pos_attrs(
    root: ET.Element,
    scale: float,
    patched_counts: Counter[str],
    skipped_counts: Counter[str],
) -> None:
    for tag in POSITION_TAGS:
        for element in root.iter(tag):
            key = f"{tag}_pos"
            if "pos" not in element.attrib:
                skipped_counts[f"{key}_missing"] += 1
                continue
            element.set("pos", _scale_attr(element.attrib["pos"], scale, expected_len=3))
            patched_counts[key] += 1


def _scale_inertials(
    root: ET.Element,
    mass_scale: float,
    com_scale: float,
    inertia_scale: float,
    patched_counts: Counter[str],
    skipped_counts: Counter[str],
) -> None:
    for inertial in root.iter("inertial"):
        if "mass" in inertial.attrib:
            inertial.set("mass", _scale_attr(inertial.attrib["mass"], mass_scale, expected_len=1))
            patched_counts["inertial_mass"] += 1
        else:
            skipped_counts["inertial_mass_missing"] += 1

        if "pos" in inertial.attrib:
            inertial.set("pos", _scale_attr(inertial.attrib["pos"], com_scale, expected_len=3))
            patched_counts["inertial_pos"] += 1
        else:
            skipped_counts["inertial_pos_missing"] += 1

        if "diaginertia" in inertial.attrib:
            inertial.set(
                "diaginertia",
                _scale_attr(inertial.attrib["diaginertia"], inertia_scale, expected_len=3),
            )
            patched_counts["inertial_diaginertia"] += 1
        else:
            skipped_counts["inertial_diaginertia_missing"] += 1

        if "fullinertia" in inertial.attrib:
            inertial.set(
                "fullinertia",
                _scale_attr(inertial.attrib["fullinertia"], inertia_scale, expected_len=6),
            )
            patched_counts["inertial_fullinertia"] += 1
        else:
            skipped_counts["inertial_fullinertia_missing"] += 1


def _scale_motor_attrs(
    root: ET.Element,
    scale: float,
    patched_counts: Counter[str],
    skipped_counts: Counter[str],
) -> None:
    actuator_root = root.find("actuator")
    actuators = list(actuator_root) if actuator_root is not None else []
    if actuator_root is None:
        skipped_counts["actuator_section_missing"] += 1

    for actuator in actuators:
        for attr in ACTUATOR_RANGE_ATTRS:
            key = f"actuator_{attr}"
            if attr not in actuator.attrib:
                skipped_counts[f"{key}_missing"] += 1
                continue
            actuator.set(attr, _scale_attr(actuator.attrib[attr], scale))
            patched_counts[key] += 1
        for attr in ACTUATOR_VECTOR_ATTRS:
            key = f"actuator_{attr}"
            if attr not in actuator.attrib:
                skipped_counts[f"{key}_missing"] += 1
                continue
            actuator.set(attr, _scale_attr(actuator.attrib[attr], scale))
            patched_counts[key] += 1

    for joint in root.iter("joint"):
        if "actuatorfrcrange" not in joint.attrib:
            skipped_counts["joint_actuatorfrcrange_missing"] += 1
            continue
        joint.set("actuatorfrcrange", _scale_attr(joint.attrib["actuatorfrcrange"], scale))
        patched_counts["joint_actuatorfrcrange"] += 1


def _summary(
    *,
    variant: Mapping[str, Any],
    source_path: Path,
    output_path: Path,
    topology_before: Mapping[str, list[str]],
    topology_after: Mapping[str, list[str]],
    patched_counts: Counter[str],
    skipped_counts: Counter[str],
    limitation_notes: list[str],
    local_parse_ok: bool,
    meshdir_info: Mapping[str, Any],
) -> dict[str, Any]:
    scales = {field: float(variant[field]) for field in SCALE_FIELDS}
    pass_ok = local_parse_ok and topology_before == topology_after
    return {
        "variant_id": str(variant["variant_id"]),
        "split": str(variant["split"]),
        "heldout_condition": str(variant["heldout_condition"]),
        "scales": scales,
        "slot_schema_id": G1LIKE_SLOT_SCHEMA_ID,
        "slot_schema_hash": slot_schema_hash(),
        "joint_order_hash": joint_order_hash(),
        "action_dim": G1LIKE_ACTION_DIM,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "source_xml_dir": str(meshdir_info["source_xml_dir"]),
        "meshdir_before": meshdir_info["meshdir_before"],
        "meshdir_after": meshdir_info["meshdir_after"],
        "meshdir_rewritten": bool(meshdir_info["meshdir_rewritten"]),
        "topology_before": copy.deepcopy(dict(topology_before)),
        "topology_after": copy.deepcopy(dict(topology_after)),
        "patched_counts": dict(sorted(patched_counts.items())),
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "limitation_notes": list(dict.fromkeys(limitation_notes)),
        "local_parse_ok": bool(local_parse_ok),
        "pass": bool(pass_ok),
    }


def _validate_variant(variant: Mapping[str, Any]) -> None:
    required = ("variant_id", "split", "heldout_condition", *SCALE_FIELDS)
    missing = [field for field in required if field not in variant]
    if missing:
        raise G1LikeMJCFPatchError(f"variant is missing required fields: {', '.join(missing)}")
    if variant["split"] not in ("train", "heldout"):
        raise G1LikeMJCFPatchError("variant split must be train or heldout")
    if variant["split"] == "train" and variant["heldout_condition"] != "none":
        raise G1LikeMJCFPatchError("train variants must use heldout_condition=none")
    for field in SCALE_FIELDS:
        value = variant[field]
        if not isinstance(value, int | float):
            raise G1LikeMJCFPatchError(f"{field} must be numeric")
        if float(value) <= 0:
            raise G1LikeMJCFPatchError(f"{field} must be positive")


def _scale_attr(value: str, scale: float, *, expected_len: int | None = None) -> str:
    values = _parse_float_list(value)
    if expected_len is not None and len(values) != expected_len:
        raise G1LikeMJCFPatchError(
            f"expected {expected_len} numeric values, got {len(values)} from {value!r}"
        )
    return _format_float_list([item * scale for item in values])


def _parse_float_list(value: str) -> list[float]:
    parts = value.split()
    if not parts:
        raise G1LikeMJCFPatchError("numeric MJCF attribute must not be empty")
    try:
        return [float(part) for part in parts]
    except ValueError as error:
        raise G1LikeMJCFPatchError(f"invalid numeric MJCF attribute: {value!r}") from error


def _format_float_list(values: list[float]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def _names(elements: Any) -> list[str]:
    return [element.attrib.get("name", "") for element in elements]


def _actuator_names(root: ET.Element) -> list[str]:
    actuator_root = root.find("actuator")
    if actuator_root is None:
        return []
    return [element.attrib.get("name", "") for element in list(actuator_root)]


def _rewrite_relative_meshdir_for_output(
    root: ET.Element,
    source_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    info = _meshdir_info(root, source_path)
    meshdir = info["meshdir_before"]
    if meshdir is None:
        return info
    if Path(meshdir).is_absolute():
        return info
    if source_path.parent.resolve() == output_path.parent.resolve():
        return info

    compiler = root.find("compiler")
    if compiler is None:
        return info
    rewritten = str((source_path.parent / meshdir).resolve())
    compiler.set("meshdir", rewritten)
    info["meshdir_after"] = rewritten
    info["meshdir_rewritten"] = True
    return info


def _meshdir_info(root: ET.Element, source_path: Path) -> dict[str, Any]:
    compiler = root.find("compiler")
    meshdir = compiler.attrib.get("meshdir") if compiler is not None else None
    return {
        "source_xml_dir": str(source_path.parent.resolve()),
        "meshdir_before": meshdir,
        "meshdir_after": meshdir,
        "meshdir_rewritten": False,
    }


def _append_missing_attr_limitations(
    skipped_counts: Counter[str],
    limitation_notes: list[str],
) -> None:
    if not skipped_counts:
        return
    limitation_notes.append(
        "Some optional MJCF attributes were absent and therefore skipped; see skipped_counts."
    )
    unsupported = [
        key for key, count in skipped_counts.items() if key.endswith("_missing") and count > 0
    ]
    if unsupported:
        limitation_notes.append(
            "Missing fields are not treated as successfully patched: "
            + ", ".join(sorted(unsupported))
        )


def _local_parse_ok(path: Path) -> bool:
    try:
        ET.parse(path)
    except ET.ParseError:
        return False
    return True


def write_patch_summary(path: str | Path, summary: Mapping[str, Any]) -> None:
    """Write a small deterministic JSON summary."""

    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
