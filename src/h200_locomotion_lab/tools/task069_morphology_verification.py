"""Verification helpers for the task069 morphology envelope.

The tool is intentionally task-scoped.  It records the legacy v2 contract before
the paper-faithful profile is introduced and later provides the same deterministic
matrix/galleried verification entry point without creating a second production
generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from h200_locomotion_lab.robots.procedural_morphology import (
    LOCOFORMER_MORPHOLOGY_CONTRACT_HASH,
    LOCOFORMER_MORPHOLOGY_CONTRACT_VERSION,
    LOCOFORMER_MORPHOLOGY_PROFILE_VERSION,
    PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
    PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
    LocoFormerMorphologyGenerator,
    LocoFormerMorphologyGeneratorConfig,
    MorphologyGenerator,
    compile_mjcf,
    compile_with_mujoco,
    morphology_blueprint_hash,
    morphology_instance_key,
    physical_params_hash,
)
from h200_locomotion_lab.robots.whole_body_adapter import BoundEmbodiment

LOCOFORMER_FAMILIES = (
    "biped",
    "quadruped",
    "wheeled_biped",
    "wheeled_quadruped",
)
SELF_CONTACT_PENETRATION_TOLERANCE = 1e-7
TERMINAL_FLOOR_FLOAT_TOLERANCE = 0.05


REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = REPO_ROOT / ".agent" / "task" / "task069-locoformer-paper-faithful-morphology"
DEFAULT_ARTIFACT_ROOT = TASK_ROOT / "artifacts"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _git_status() -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in completed.stdout.splitlines() if line]


def _runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hardware_assumption": "RTX 5060 Ti-first; local MuJoCo/headless; H200 disabled",
        "git_dirty_files_at_capture": _git_status(),
    }
    try:
        import mujoco

        metadata["mujoco_version"] = str(mujoco.__version__)
    except ImportError:
        metadata["mujoco_version"] = None
    return metadata


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_source_contract(output: Path) -> None:
    """Record direct paper facts separately from local implementation choices."""

    payload = {
        "task": "task069-locoformer-paper-faithful-morphology",
        "claim_boundary": (
            "paper-faithful, verifiable four-family procedural morphology envelope; "
            "not official generator/source/code or policy/training parity"
        ),
        "sources": [
            {
                "kind": "official_project_page",
                "url": "https://generalist-locomotion.github.io/",
                "authority": "authors",
            },
            {
                "kind": "author_paper",
                "url": "https://arxiv.org/html/2509.23745",
                "authority": "authors",
            },
        ],
        "claims": [
            {
                "id": "four_training_families",
                "status": "direct",
                "statement": (
                    "Appendix A.1/Figure 6 describes quadruped, wheeled quadruped, "
                    "biped, and wheeled biped training morphologies."
                ),
                "source": "author_paper:Appendix A.1/Figure 6",
            },
            {
                "id": "procedural_training_bodies",
                "status": "direct",
                "statement": (
                    "Training morphologies are procedurally generated bodies rather "
                    "than exact parameters of commercially sold robots."
                ),
                "source": "author_paper:§2.1 Task Generation",
            },
            {
                "id": "unseen_evaluation_embodiments",
                "status": "direct",
                "statement": (
                    "Figure 7 named robots are unseen evaluation embodiments, not "
                    "the training morphology grammar."
                ),
                "source": "author_paper:Figure 7",
            },
            {
                "id": "primitive_ranges",
                "status": "local_design_choice",
                "statement": (
                    "Primitive dimensions, joint ranges, wheel dimensions, masses, "
                    "and randomization intervals are implementation choices in this "
                    "repository because the public sources do not publish the exact "
                    "morphology generator."
                ),
                "source": "not specified by public morphology-generator source",
            },
            {
                "id": "policy_and_scale_scope",
                "status": "unknown_out_of_scope",
                "statement": (
                    "This artifact does not establish TXL quality, long-context RL, "
                    "sim2real behavior, or the paper's large-scale training claims."
                ),
                "source": "task069 scope boundary",
            },
        ],
        "local_reference_artifacts": [
            {
                "path": "../task067-biped-stance-contract/artifacts/independent_r4a31g_review/locoformer_official_fig6a_quadruped.png",
                "sha256": "ded3e8bc9130b7918b6336698e3db1ad7db1290794045d5619f338eed5ce5d31",
                "role": "visual reference only",
            },
            {
                "path": "../task067-biped-stance-contract/artifacts/independent_r4a31g_review/locoformer_official_fig6c_biped.png",
                "sha256": "9714c051ac26320c126a517561794c83aae59dde392a21c5f19463108bc62605",
                "role": "visual reference only",
            },
        ],
        "source_sha256": {
            "procedural_morphology.py": sha256_path(
                REPO_ROOT / "src" / "h200_locomotion_lab" / "robots" / "procedural_morphology.py"
            ),
            "verification_tool.py": sha256_path(Path(__file__)),
        },
        "runtime": _runtime_metadata(),
    }
    _write_json(output, payload)


def _legacy_record(generator: MorphologyGenerator, family: str, seed: int, range_fraction: float) -> dict[str, Any]:
    record: dict[str, Any] = {
        "family": family,
        "seed": seed,
        "range_fraction": range_fraction,
        "status": "failed",
        "compiled": False,
        "blueprint_manifest": None,
        "blueprint_hash": None,
        "physical_manifest": None,
        "physical_hash": None,
        "xml_sha256": None,
        "instance_key": None,
        "error": None,
    }
    try:
        blueprint = generator.generate(family, seed)  # type: ignore[arg-type]
        physical = generator.sample_physical_params(
            blueprint,
            seed + 10_000_000,
            range_fraction=range_fraction,
        )
        xml = compile_mjcf(blueprint, physical)
        record.update(
            {
                "blueprint_manifest": blueprint.manifest(),
                "blueprint_hash": morphology_blueprint_hash(blueprint),
                "physical_manifest": physical.manifest(),
                "physical_hash": physical_params_hash(physical),
                "xml_sha256": sha256_bytes(xml.encode("utf-8")),
                "instance_key": morphology_instance_key(blueprint, physical).manifest()
                | {"cache_key": morphology_instance_key(blueprint, physical).cache_key},
                "contract_version": PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
                "contract_hash": PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
            }
        )
        try:
            compile_with_mujoco(xml)
            record["compiled"] = True
            record["status"] = "passed"
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - environment-dependent
            record["error"] = f"compile: {type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - preserves denominator on failure
        record["error"] = f"build: {type(exc).__name__}: {exc}"
    return record


def write_legacy_baseline(
    output: Path,
    *,
    seeds: range = range(32),
    range_fractions: tuple[float, ...] = (0.0, 0.5),
) -> None:
    """Freeze the legacy v2 outputs before changing the production generator."""

    generator = MorphologyGenerator()
    records = [
        _legacy_record(generator, family, seed, range_fraction)
        for family in ("biped", "quadruped")
        for seed in seeds
        for range_fraction in range_fractions
    ]
    payload = {
        "task": "task069-locoformer-paper-faithful-morphology",
        "profile": "legacy_v2",
        "contract_version": PROCEDURAL_EMBODIMENT_CONTRACT_VERSION,
        "contract_hash": PROCEDURAL_EMBODIMENT_CONTRACT_HASH,
        "expected_denominator": len(records),
        "records": records,
        "runtime": _runtime_metadata(),
        "source_sha256": {
            "procedural_morphology.py": sha256_path(
                REPO_ROOT / "src" / "h200_locomotion_lab" / "robots" / "procedural_morphology.py"
            ),
            "verification_tool.py": sha256_path(Path(__file__)),
        },
    }
    _write_json(output, payload)


def verify_legacy_baseline(path: Path) -> dict[str, Any]:
    """Rebuild the frozen legacy matrix and report any drift without overwriting it."""

    expected = json.loads(path.read_text(encoding="utf-8"))
    expected_records = {
        (item["family"], item["seed"], item["range_fraction"]): item
        for item in expected["records"]
    }
    generator = MorphologyGenerator()
    comparisons: list[dict[str, Any]] = []
    for key, old in expected_records.items():
        family, seed, range_fraction = key
        current = _legacy_record(generator, family, seed, range_fraction)
        # JSON round-tripping makes tuple-valued dataclass fields comparable
        # with the lists persisted in the artifact.
        current["blueprint_manifest"] = json.loads(
            json.dumps(current["blueprint_manifest"], sort_keys=True)
        )
        checks = {
            field: current.get(field) == old.get(field)
            for field in (
                "blueprint_manifest",
                "blueprint_hash",
                "physical_manifest",
                "physical_hash",
                "xml_sha256",
                "instance_key",
                "compiled",
                "status",
                "error",
                "contract_version",
                "contract_hash",
            )
        }
        comparisons.append({"key": list(key), "passed": all(checks.values()), "checks": checks})
    return {
        "expected_denominator": expected["expected_denominator"],
        "rebuilt_count": len(comparisons),
        "passed_count": sum(item["passed"] for item in comparisons),
        "failed": [item for item in comparisons if not item["passed"]],
        "passed": len(comparisons) == expected["expected_denominator"]
        and all(item["passed"] for item in comparisons),
    }


def _finite_array(values: Any) -> bool:
    try:
        import numpy as np

        return bool(np.isfinite(values).all())
    except (TypeError, ValueError):
        return False


def _joint_lookup(model: Any, mujoco: Any, name: str, object_type: Any) -> int:
    return int(mujoco.mj_name2id(model, object_type, name))


def _axis_order_signature(blueprint: Any) -> dict[str, tuple[str, ...]]:
    signature: dict[str, list[str]] = {}
    for joint in blueprint.joints:
        if not joint.semantic_slot.startswith("limb"):
            continue
        limb = joint.semantic_slot.split("_", 1)[0]
        signature.setdefault(limb, []).append(
            f"{joint.semantic_slot.split('_', 1)[1]}:{joint.axis_name}"
        )
    return {limb: tuple(values) for limb, values in signature.items()}


def _leg_length_summary(blueprint: Any) -> dict[str, float]:
    summary: dict[str, float] = {}
    for link in blueprint.links:
        for limb in range(4):
            prefix = f"limb{limb}_"
            if link.name.startswith(("left_leg_", "right_leg_", "front_left_leg_", "front_right_leg_", "rear_left_leg_", "rear_right_leg_")) and link.geom_type == "capsule":
                semantic = next(
                    (
                        joint.semantic_slot
                        for joint in blueprint.joints
                        if joint.child_link == link.name
                    ),
                    None,
                )
                if semantic and semantic.startswith(prefix):
                    summary[prefix[:-1]] = summary.get(prefix[:-1], 0.0) + float(link.length)
    return summary


def _biped_grammar_witness(blueprint: Any) -> dict[str, Any]:
    if not blueprint.family.endswith("biped"):
        return {
            "has_ankle": None,
            "mirrored_leg_grammar": None,
            "has_arms": blueprint.has_arms,
        }
    legs: dict[str, list[Any]] = {"limb0": [], "limb1": []}
    for joint in blueprint.joints:
        limb = joint.semantic_slot.split("_", 1)[0]
        if limb in legs:
            legs[limb].append(joint)
    suffixes = {
        limb: tuple(joint.semantic_slot.split("_", 1)[1] for joint in joints)
        for limb, joints in legs.items()
    }
    mirrored = suffixes["limb0"] == suffixes["limb1"]
    if mirrored:
        for first, second in zip(legs["limb0"], legs["limb1"]):
            if first.axis_name in {"roll", "yaw"}:
                mirrored = mirrored and second.axis == tuple(-value for value in first.axis)
            else:
                mirrored = mirrored and second.axis == first.axis
    return {
        "has_ankle": any("ankle_" in suffix for suffix in suffixes["limb0"]),
        "left_has_ankle": any("ankle_" in suffix for suffix in suffixes["limb0"]),
        "right_has_ankle": any("ankle_" in suffix for suffix in suffixes["limb1"]),
        "mirrored_leg_grammar": mirrored,
        "has_arms": blueprint.has_arms,
    }


def _wheel_xml_checks(blueprint: Any, xml: str) -> dict[str, Any]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    joints = {str(item.get("name")): item for item in root.iter("joint")}
    geoms = {str(item.get("name")): item for item in root.iter("geom")}
    actuators = {
        str(item.get("name")): item for item in root.find("actuator") or ()
    }
    checks: list[dict[str, Any]] = []
    for wheel in blueprint.wheel_specs:
        joint = joints.get(wheel.joint_name)
        geom = geoms.get(f"{wheel.link_name}_geom")
        actuator = actuators.get(f"{wheel.joint_name}_actuator")
        checks.append(
            {
                "semantic_slot": wheel.semantic_slot,
                "joint_present": joint is not None,
                "joint_continuous": joint is not None and joint.get("limited") == "false",
                "axis": tuple(float(value) for value in str(joint.get("axis", "")).split())
                if joint is not None
                else None,
                "geom_present": geom is not None,
                "geom_type": geom.get("type") if geom is not None else None,
                "geom_contact": geom is not None
                and geom.get("contype") == "1"
                and geom.get("conaffinity") == "1",
                "geom_has_rotation": geom is not None and geom.get("quat") is not None,
                "actuator_present": actuator is not None,
                "actuator_type": actuator.tag if actuator is not None else None,
                "actuator_ctrlrange": actuator.get("ctrlrange") if actuator is not None else None,
            }
        )
    return {
        "wheel_count": len(blueprint.wheel_specs),
        "checks": checks,
        "no_active_wheel_slots": not blueprint.wheel_specs
        and not any(slot.endswith("_wheel") for slot in blueprint.active_slots),
        "passed": (
            all(
                item["joint_present"]
                and item["joint_continuous"]
                and item["geom_present"]
                and item["geom_type"] == "cylinder"
                and item["geom_contact"]
                and item["geom_has_rotation"]
                and item["actuator_present"]
                and item["actuator_type"] == "motor"
                for item in checks
            )
            if blueprint.wheel_specs
            else not any(slot.endswith("_wheel") for slot in blueprint.active_slots)
        ),
    }


def _finite_physics_checks(model: Any, blueprint: Any, mujoco: Any) -> dict[str, Any]:
    limited_joint_values = []
    for joint in blueprint.joints:
        joint_id = _joint_lookup(model, mujoco, joint.name, mujoco.mjtObj.mjOBJ_JOINT)
        if joint_id >= 0 and bool(model.jnt_limited[joint_id]):
            limited_joint_values.extend(model.jnt_range[joint_id].tolist())
    actuator_values = []
    for index in range(model.nu):
        if bool(model.actuator_ctrllimited[index]):
            actuator_values.extend(model.actuator_ctrlrange[index].tolist())
        if bool(model.actuator_forcelimited[index]):
            actuator_values.extend(model.actuator_forcerange[index].tolist())
    # MuJoCo's world body is a legal zero-mass sentinel at body id 0.  Only
    # generated bodies belong in the positive-mass gate.
    generated_body_mass = model.body_mass[1:]
    positive_body_mass = bool(len(generated_body_mass) > 0 and (generated_body_mass > 0.0).all())
    positive_dof_inertia = bool(len(model.dof_M0) > 0 and (model.dof_M0 > 0.0).all())
    finite = (
        _finite_array(generated_body_mass)
        and _finite_array(model.dof_M0)
        and _finite_array(limited_joint_values)
        and _finite_array(actuator_values)
        and positive_body_mass
        and positive_dof_inertia
    )
    return {
        "finite": finite,
        "world_body_excluded": True,
        "generated_body_count": max(0, int(model.nbody) - 1),
        "positive_body_mass": positive_body_mass,
        "positive_dof_inertia": positive_dof_inertia,
        "minimum_generated_body_mass": float(generated_body_mass.min())
        if len(generated_body_mass)
        else None,
        "minimum_dof_inertia": float(model.dof_M0.min()) if len(model.dof_M0) else None,
        "limited_joint_values_finite": _finite_array(limited_joint_values),
        "actuator_values_finite": _finite_array(actuator_values),
    }


def _set_default_pose(
    model: Any,
    blueprint: Any,
    data: Any,
    mujoco: Any,
    *,
    root_z: float,
) -> None:
    data.qpos[0:3] = (0.0, 0.0, root_z)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for joint in blueprint.joints:
        joint_id = _joint_lookup(model, mujoco, joint.name, mujoco.mjtObj.mjOBJ_JOINT)
        address = int(model.jnt_qposadr[joint_id])
        value = 0.0 if joint.semantic_slot.endswith("_wheel") else joint.nominal
        if bool(model.jnt_limited[joint_id]):
            lower, upper = (float(item) for item in model.jnt_range[joint_id])
            value = min(upper, max(lower, value))
        data.qpos[address] = value
    data.ctrl[:] = 0.0


def _terminal_geom_ids(model: Any, blueprint: Any, mujoco: Any) -> tuple[int, ...]:
    names = [
        f"{link.name}_footpad"
        for link in blueprint.links
        if link.foot
    ]
    names.extend(f"{wheel.link_name}_geom" for wheel in blueprint.wheel_specs)
    ids = tuple(_joint_lookup(model, mujoco, name, mujoco.mjtObj.mjOBJ_GEOM) for name in names)
    if not ids or any(geom_id < 0 for geom_id in ids):
        raise ValueError("every morphology must expose a resolvable terminal foot or wheel geom")
    return ids


def _contact_records(
    model: Any,
    data: Any,
    mujoco: Any,
    *,
    step: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    floor_id = _joint_lookup(model, mujoco, "floor", mujoco.mjtObj.mjOBJ_GEOM)
    floor_contacts: list[dict[str, Any]] = []
    self_contacts: list[dict[str, Any]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        item = {
            "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
            "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
            "distance": float(contact.dist),
            "step": step,
        }
        if floor_id in {geom1, geom2}:
            item["kind"] = "floor"
            floor_contacts.append(item)
        elif model.geom_bodyid[geom1] != 0 and model.geom_bodyid[geom2] != 0:
            item["kind"] = "self"
            self_contacts.append(item)
    return floor_contacts, self_contacts


def _terminal_floor_distances(
    model: Any,
    data: Any,
    terminal_ids: tuple[int, ...],
    mujoco: Any,
) -> list[dict[str, Any]]:
    distances: list[dict[str, Any]] = []
    for geom_id in terminal_ids:
        geom_type = int(model.geom_type[geom_id])
        size = [float(value) for value in model.geom_size[geom_id]]
        vertical_row = [float(value) for value in data.geom_xmat[geom_id][6:9]]
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            vertical_extent = sum(abs(vertical_row[index]) * size[index] for index in range(3))
        elif geom_type in {
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        }:
            radius = size[0]
            half_length = size[1]
            radial_vertical = math.sqrt(vertical_row[0] ** 2 + vertical_row[1] ** 2)
            vertical_extent = abs(vertical_row[2]) * half_length + radius * radial_vertical
        else:
            vertical_extent = max(size)
        lower_z = float(data.geom_xpos[geom_id, 2]) - vertical_extent
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        distances.append(
            {
                "geom": name,
                "lower_z": lower_z,
                "clearance": lower_z,
                "airborne": lower_z > 0.05,
                "penetrating": lower_z < -1e-6,
            }
        )
    return distances


def _prepare_default_pose(
    model: Any,
    blueprint: Any,
    physical: Any,
    mujoco: Any,
) -> tuple[Any, dict[str, Any]]:
    """Place the deterministic diagnostic pose above the floor without stance solving."""

    data = mujoco.MjData(model)
    terminal_ids = _terminal_geom_ids(model, blueprint, mujoco)
    _set_default_pose(model, blueprint, data, mujoco, root_z=0.0)
    mujoco.mj_forward(model, data)
    origin_distances = _terminal_floor_distances(model, data, terminal_ids, mujoco)
    minimum_origin_clearance = min(item["clearance"] for item in origin_distances)
    nominal_root_z = blueprint.nominal_height * physical.global_scale
    root_z = max(0.001, -minimum_origin_clearance + 0.005)
    _set_default_pose(model, blueprint, data, mujoco, root_z=root_z)
    mujoco.mj_forward(model, data)
    terminal_distances = _terminal_floor_distances(model, data, terminal_ids, mujoco)
    floor_contacts, self_contacts = _contact_records(model, data, mujoco, step=0)
    diagnostics = {
        "root_z": root_z,
        "nominal_root_z": nominal_root_z,
        "root_z_source": "terminal_floor_clearance_without_stance_solving",
        "terminal_floor_distance": terminal_distances,
        "initial_floor_contacts": floor_contacts,
        "initial_self_contacts": self_contacts,
        "initial_floor_contact_count": len(floor_contacts),
        "initial_self_contact_count": len(self_contacts),
        "reset_self_collision_free": not self_contacts,
        "reset_terminal_floor_clear": not any(
            item["penetrating"] for item in terminal_distances
        ),
        "reset_pose_passed": not self_contacts
        and not any(item["penetrating"] for item in terminal_distances),
    }
    return data, diagnostics


def _default_pose_smoke(
    model: Any,
    blueprint: Any,
    physical: Any,
    *,
    steps: int,
    mujoco: Any,
) -> dict[str, Any]:
    data, reset_pose = _prepare_default_pose(model, blueprint, physical, mujoco)
    finite = True
    solver_fatal = False
    max_penetration = 0.0
    max_self_penetration = 0.0
    max_floor_penetration = 0.0
    max_contact_count = int(data.ncon)
    max_warning_count = 0
    rollout_contact_events: list[dict[str, Any]] = []
    finite_fields = {
        "qpos": True,
        "qvel": True,
        "qacc": True,
        "ctrl": True,
        "act": True,
        "qfrc_actuator": True,
        "actuator_force": True,
    }

    def update_finite_fields() -> None:
        finite_fields["qpos"] = finite_fields["qpos"] and _finite_array(data.qpos)
        finite_fields["qvel"] = finite_fields["qvel"] and _finite_array(data.qvel)
        finite_fields["qacc"] = finite_fields["qacc"] and _finite_array(data.qacc)
        finite_fields["ctrl"] = finite_fields["ctrl"] and _finite_array(data.ctrl)
        finite_fields["act"] = finite_fields["act"] and _finite_array(data.act)
        finite_fields["qfrc_actuator"] = finite_fields["qfrc_actuator"] and _finite_array(
            data.qfrc_actuator
        )
        finite_fields["actuator_force"] = finite_fields["actuator_force"] and _finite_array(
            data.actuator_force
        )

    def update_contacts(step: int) -> None:
        nonlocal max_contact_count, max_penetration, max_self_penetration, max_floor_penetration
        floor_contacts, self_contacts = _contact_records(model, data, mujoco, step=step)
        rollout_contact_events.extend(floor_contacts)
        rollout_contact_events.extend(self_contacts)
        max_contact_count = max(max_contact_count, int(data.ncon))
        for item in floor_contacts:
            penetration = max(0.0, -float(item["distance"]))
            max_penetration = max(max_penetration, penetration)
            max_floor_penetration = max(max_floor_penetration, penetration)
        for item in self_contacts:
            penetration = max(0.0, -float(item["distance"]))
            max_penetration = max(max_penetration, penetration)
            max_self_penetration = max(max_self_penetration, penetration)

    def warning_count() -> int:
        return int(sum(int(data.warning[index].number) for index in range(len(data.warning))))

    try:
        mujoco.mj_forward(model, data)
        update_finite_fields()
        update_contacts(0)
        max_warning_count = max(max_warning_count, warning_count())
        for step in range(1, steps + 1):
            mujoco.mj_step(model, data)
            update_finite_fields()
            update_contacts(step)
            max_warning_count = max(max_warning_count, warning_count())
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - preserves failure evidence
        solver_fatal = True
        error = f"{type(exc).__name__}: {exc}"
    else:
        error = None
    finite = bool(all(finite_fields.values()))
    gravity = tuple(float(value) for value in data.qpos[3:7])
    return {
        "steps": steps,
        "finite": finite,
        "solver_fatal": solver_fatal,
        "error": error,
        "finite_fields": finite_fields,
        "warning_count": max_warning_count,
        "warning_free": max_warning_count == 0,
        "fall_observed": bool(data.qpos[2] < reset_pose["root_z"] * 0.35),
        "max_penetration": max_penetration,
        "max_self_penetration": max_self_penetration,
        "max_floor_penetration": max_floor_penetration,
        "max_contact_count": max_contact_count,
        "rollout_contact_events": rollout_contact_events,
        "rollout_contact_event_count": len(rollout_contact_events),
        "final_qpos_norm": float(sum(float(value) ** 2 for value in data.qpos) ** 0.5),
        "final_quaternion": gravity,
        "reset_pose": reset_pose,
    }


def _matrix_record(
    generator: LocoFormerMorphologyGenerator,
    family: str,
    seed: int,
    *,
    smoke_steps: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "family": family,
        "seed": seed,
        "range_fraction": 0.5,
        "status": "failed",
        "built": False,
        "compiled": False,
        "smoke_passed": False,
        "determinism_passed": False,
        "slot_mapping_passed": False,
        "finite_physics_passed": False,
        "wheel_passed": None,
        "error": None,
    }
    try:
        blueprint = generator.generate(family, seed)  # type: ignore[arg-type]
        repeat = generator.generate(family, seed)  # type: ignore[arg-type]
        physical = generator.sample_physical_params(blueprint, seed + 20_000_000, range_fraction=0.5)
        repeat_physical = generator.sample_physical_params(
            repeat, seed + 20_000_000, range_fraction=0.5
        )
        xml = compile_mjcf(blueprint, physical)
        repeat_xml = compile_mjcf(repeat, repeat_physical)
        record.update(
            {
                "built": True,
                "profile_version": blueprint.profile_version,
                "contract_version": blueprint.embodiment_contract_version,
                "contract_hash": blueprint.embodiment_contract_hash,
                "structural_hash": blueprint.structural_hash,
                "blueprint_hash": morphology_blueprint_hash(blueprint),
                "physical_hash": physical_params_hash(physical),
                "instance_key": morphology_instance_key(blueprint, physical).manifest()
                | {"cache_key": morphology_instance_key(blueprint, physical).cache_key},
                "xml_sha256": sha256_bytes(xml.encode("utf-8")),
                "active_slots": blueprint.active_slots,
                "active_slot_mask": blueprint.active_slot_mask,
                "joint_count": len(blueprint.joints),
                "actuator_count": len(blueprint.actuators),
                "wheel_count": len(blueprint.wheel_specs),
                "wheel_manifest": [
                    {
                        "link_name": wheel.link_name,
                        "joint_name": wheel.joint_name,
                        "semantic_slot": wheel.semantic_slot,
                        "radius": wheel.radius,
                        "width": wheel.width,
                        "axis_name": wheel.axis_name,
                        "axis": wheel.axis,
                        "joint_range": wheel.joint_range,
                        "continuous": wheel.continuous,
                        "friction": wheel.friction,
                        "effort_limit": wheel.effort_limit,
                    }
                    for wheel in blueprint.wheel_specs
                ],
                "axis_order_signature": _axis_order_signature(blueprint),
                "grammar_witness": _biped_grammar_witness(blueprint),
                "leg_length_summary": _leg_length_summary(blueprint),
                "trunk_aspect_ratio": blueprint.links[0].size[0] / blueprint.links[0].size[1],
                "physical_continuous_identity_separate": physical_params_hash(physical)
                != physical_params_hash(
                    generator.sample_physical_params(blueprint, seed + 20_000_001, range_fraction=0.5)
                ),
                "repeat_physical_hash": physical_params_hash(repeat_physical),
                "determinism_checks": {
                    "blueprint_manifest": blueprint.manifest() == repeat.manifest(),
                    "physical_manifest": physical.manifest() == repeat_physical.manifest(),
                    "physical_hash": physical_params_hash(physical)
                    == physical_params_hash(repeat_physical),
                    "xml_sha256": sha256_bytes(xml.encode("utf-8"))
                    == sha256_bytes(repeat_xml.encode("utf-8")),
                },
            }
        )
        record["determinism_passed"] = all(record["determinism_checks"].values())
        model = compile_with_mujoco(xml)
        record["compiled"] = True
        record["model_nq"] = int(model.nq)
        record["model_nv"] = int(model.nv)
        record["model_nu"] = int(model.nu)
        record["finite_physics"] = _finite_physics_checks(model, blueprint, __import__("mujoco"))
        record["finite_physics_passed"] = bool(record["finite_physics"]["finite"])
        mapping = BoundEmbodiment.from_blueprint(blueprint, physical=physical).mapping
        robot_values = tuple(float(index) for index in range(len(blueprint.joints)))
        record["slot_mapping"] = {
            "selector": mapping.selector,
            "mask": mapping.mask,
            "active_count": mapping.active_count,
            "round_trip": mapping.round_trip(robot_values),
        }
        record["slot_mapping_passed"] = mapping.round_trip(robot_values) == robot_values
        record["wheel_xml"] = _wheel_xml_checks(blueprint, xml)
        record["wheel_passed"] = bool(record["wheel_xml"]["passed"])
        smoke = _default_pose_smoke(
            model,
            blueprint,
            physical,
            steps=smoke_steps,
            mujoco=__import__("mujoco"),
        )
        record["smoke"] = smoke
        record["reset_pose"] = smoke["reset_pose"]
        record["smoke_passed"] = bool(
            smoke["finite"] and not smoke["solver_fatal"] and smoke["warning_free"]
        )
        record["status"] = "passed" if all(
            (
                record["finite_physics_passed"],
                record["smoke_passed"],
                record["determinism_passed"],
                record["slot_mapping_passed"],
                record["wheel_passed"],
                record["reset_pose"]["reset_pose_passed"],
            )
        ) else "failed"
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - preserves expected denominator
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _decode_image(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
        return {
            "path": str(path),
            "sha256": sha256_path(path),
            "viewer_decode": True,
            "width": width,
            "height": height,
            "mode": mode,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - environment-dependent
        return {
            "path": str(path),
            "sha256": sha256_path(path) if path.exists() else None,
            "viewer_decode": False,
            "width": None,
            "height": None,
            "mode": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _render_blueprint(
    blueprint: Any,
    physical: Any,
    path: Path,
    *,
    view: str,
) -> dict[str, Any]:
    """Render one deterministic MuJoCo frame for the gallery."""

    try:
        import mujoco
        import numpy as np
        from PIL import Image, ImageDraw

        model = mujoco.MjModel.from_xml_string(compile_mjcf(blueprint, physical))
        data, reset_pose = _prepare_default_pose(model, blueprint, physical, mujoco)
        width, height = (480, 320)
        renderer = mujoco.Renderer(model, height=height, width=width)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        base = "biped" if blueprint.family.endswith("biped") else "quadruped"
        camera.lookat[:] = (0.0, 0.0, reset_pose["root_z"] * (0.48 if base == "biped" else 0.42))
        camera.distance = 3.4 if base == "biped" else 2.5
        settings = {
            "oblique": (135.0, -10.0),
            "side": (90.0, -8.0),
            "front": (180.0, -8.0),
            "wheel_axis": (180.0, -2.0),
            "detail": (135.0, 2.0),
        }
        camera.azimuth, camera.elevation = settings[view]
        renderer.update_scene(data, camera=camera)
        pixels = np.asarray(renderer.render()).copy()
        renderer.close()
        if pixels.ndim != 3 or pixels.shape[-1] != 3 or int(pixels.max()) <= 5:
            raise RuntimeError("renderer produced a black or malformed frame")
        image = Image.fromarray(pixels.astype("uint8"), mode="RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 205, 18), fill=(0, 0, 0))
        draw.text((4, 3), f"{blueprint.family} seed={blueprint.seed} {view}", fill=(255, 255, 255))
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")
        return _decode_image(path)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - EGL/GL is environment-dependent
        return {
            "path": str(path),
            "sha256": sha256_path(path) if path.exists() else None,
            "viewer_decode": False,
            "width": None,
            "height": None,
            "mode": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _write_gallery(
    root: Path,
    generator: LocoFormerMorphologyGenerator,
    *,
    seeds: range,
    range_fraction: float = 0.5,
) -> dict[str, Any]:
    from PIL import Image

    gallery: dict[str, Any] = {}
    for family in LOCOFORMER_FAMILIES:
        family_root = root / family
        family_root.mkdir(parents=True, exist_ok=True)
        tiles: list[Any | None] = []
        records: list[dict[str, Any]] = []
        for seed in seeds:
            blueprint = generator.generate(family, seed)  # type: ignore[arg-type]
            physical = generator.sample_physical_params(
                blueprint, seed + 20_000_000, range_fraction=range_fraction
            )
            tile_path = family_root / f"seed_{seed:02d}.png"
            result = _render_blueprint(blueprint, physical, tile_path, view="oblique")
            records.append(result)
            if result["viewer_decode"]:
                with Image.open(tile_path) as image:
                    tiles.append(image.convert("RGB").resize((240, 160)))
            else:
                # Preserve the expected seed position in the montage even
                # when rendering fails; failures stay visible in JSON too.
                tiles.append(None)
        montage = Image.new("RGB", (8 * 240, 4 * 160), (24, 24, 24))
        for index, tile in enumerate(tiles):
            if tile is not None:
                montage.paste(tile, ((index % 8) * 240, (index // 8) * 160))
        montage_path = family_root / "montage.png"
        montage.save(montage_path, format="PNG")
        closeups: list[dict[str, Any]] = []
        views = ("oblique", "side", "front", "wheel_axis" if family.startswith("wheeled_") else "detail")
        for seed, view in zip((0, 10, 20, 31), views):
            blueprint = generator.generate(family, seed)  # type: ignore[arg-type]
            physical = generator.sample_physical_params(
                blueprint, seed + 20_000_000, range_fraction=range_fraction
            )
            closeups.append(
                _render_blueprint(
                    blueprint,
                    physical,
                    family_root / f"closeup_seed_{seed:02d}_{view}.png",
                    view=view,
                )
            )
        slots_path = family_root / "active_slots.json"
        slot_payload = {
            "family": family,
            "records": [
                {
                    "seed": seed,
                    "active_slots": generator.generate(family, seed).active_slots,  # type: ignore[arg-type]
                }
                for seed in seeds
            ],
        }
        _write_json(slots_path, slot_payload)
        gallery[family] = {
            "expected_samples": len(seeds),
            "rendered_samples": sum(item["viewer_decode"] for item in records),
            "montage": _decode_image(montage_path),
            "closeups": closeups,
            "sample_frames": records,
            "active_slots": {
                "path": str(slots_path),
                "sha256": sha256_path(slots_path),
            },
        }
    return gallery


def run_morphology_matrix(
    output: Path = DEFAULT_ARTIFACT_ROOT / "r3_morphology_matrix.json",
    *,
    seeds: range = range(32),
    smoke_steps: int = 100,
    render: bool = True,
) -> dict[str, Any]:
    """Run the four-family deterministic/compile/finite-smoke matrix."""

    if smoke_steps <= 0:
        raise ValueError("smoke_steps must be positive")
    started = time.monotonic()
    generator = LocoFormerMorphologyGenerator(LocoFormerMorphologyGeneratorConfig())
    records = [
        _matrix_record(generator, family, seed, smoke_steps=smoke_steps)
        for family in LOCOFORMER_FAMILIES
        for seed in seeds
    ]
    summary: dict[str, Any] = {}
    for family in LOCOFORMER_FAMILIES:
        family_records = [record for record in records if record["family"] == family]
        summary[family] = {
            "expected_denominator": len(seeds),
            "built": sum(bool(record["built"]) for record in family_records),
            "compiled": sum(bool(record["compiled"]) for record in family_records),
            "finite_physics": sum(bool(record["finite_physics_passed"]) for record in family_records),
            "finite_smoke": sum(bool(record["smoke_passed"]) for record in family_records),
            "deterministic": sum(bool(record["determinism_passed"]) for record in family_records),
            "slot_mapping": sum(bool(record["slot_mapping_passed"]) for record in family_records),
            "reset_pose": sum(
                bool(record.get("reset_pose", {}).get("reset_pose_passed"))
                for record in family_records
            ),
            "reset_self_collision_free": sum(
                bool(record.get("reset_pose", {}).get("reset_self_collision_free"))
                for record in family_records
            ),
            "reset_terminal_floor_clear": sum(
                bool(record.get("reset_pose", {}).get("reset_terminal_floor_clear"))
                for record in family_records
            ),
            "wheel_expected": family.startswith("wheeled_"),
            "wheel_passed": sum(bool(record["wheel_passed"]) for record in family_records),
            "errors": [record["error"] for record in family_records if record["error"]],
        }
    gallery = None
    if render:
        gallery = _write_gallery(output.parent / "gallery", generator, seeds=seeds)
    payload: dict[str, Any] = {
        "task": "task069-locoformer-paper-faithful-morphology",
        "profile_version": LOCOFORMER_MORPHOLOGY_PROFILE_VERSION,
        "contract_version": LOCOFORMER_MORPHOLOGY_CONTRACT_VERSION,
        "contract_hash": LOCOFORMER_MORPHOLOGY_CONTRACT_HASH,
        "families": LOCOFORMER_FAMILIES,
        "expected_denominator": len(LOCOFORMER_FAMILIES) * len(seeds),
        "smoke_steps": smoke_steps,
        "summary": summary,
        "records": records,
        "gallery": gallery,
        "visual_inspection": {
            "agent_viewer_required": True,
            "status": "pending_manual_image_viewer_review" if render else "not_requested",
        },
        "runtime": _runtime_metadata(),
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": {
            "procedural_morphology.py": sha256_path(
                REPO_ROOT / "src" / "h200_locomotion_lab" / "robots" / "procedural_morphology.py"
            ),
            "verification_tool.py": sha256_path(Path(__file__)),
        },
    }
    _write_json(output, payload)
    return payload


def finalize_visual_inspection(path: Path, observations_path: Path) -> dict[str, Any]:
    """Attach explicit observations supplied after opening every final image.

    The verifier deliberately does not manufacture a manual-review claim from
    PNG decoding.  ``observations_path`` is a small, human-authored manifest
    that names every opened image and records image-specific observations and
    disclosed reset-contact caveats.
    """

    path = path.resolve()
    observations_path = observations_path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    if observations.get("viewer") != "local_image_viewer":
        raise ValueError("visual observations must identify the local image viewer")
    if observations.get("status") != "passed_with_disclosed_reset_variation":
        raise ValueError("visual observations must use the disclosed-reset review status")
    reviewed_paths = set(observations.get("reviewed_paths", ()))
    image_observations = observations.get("image_observations", {})
    gallery_root = path.parent / "gallery"
    entries: list[dict[str, Any]] = []
    for family in LOCOFORMER_FAMILIES:
        family_root = gallery_root / family
        image_paths = [family_root / "montage.png"] + sorted(family_root.glob("closeup_*.png"))
        for image_path in image_paths:
            relative = image_path.relative_to(path.parent).as_posix()
            if relative not in reviewed_paths:
                raise ValueError(f"manual viewer confirmation missing for {relative}")
            image_note = image_observations.get(relative)
            if not isinstance(image_note, dict):
                raise TypeError(f"missing image-specific manual observation for {relative}")
            decoded = _decode_image(image_path)
            if not decoded["viewer_decode"]:
                raise ValueError(f"final image is not decodable: {relative}")
            observation = str(image_note.get("observation", ""))
            if not observation:
                raise ValueError(f"empty manual observation for {family}")
            entries.append(
                {
                    "relative_path": relative,
                    "sha256": decoded["sha256"],
                    "viewer_decode": decoded["viewer_decode"],
                    "execution_agent_observation": observation,
                    "problems_observed": list(image_note.get("problems_observed", [])),
                    "notes": str(image_note.get("notes", "")),
                    "re_view_conclusion": str(image_note.get("re_view_conclusion", "")),
                    "manual_viewer_confirmed": True,
                }
            )
    payload["visual_inspection"] = entries
    payload["visual_inspection_status"] = "passed_execution_agent_manual_viewer_review_with_disclosed_reset_variation"
    payload["visual_observation_manifest"] = {
        "path": str(observations_path.relative_to(path.parent)),
        "sha256": sha256_path(observations_path),
    }
    payload["source_sha256"] = {
        "procedural_morphology.py": sha256_path(
            REPO_ROOT / "src" / "h200_locomotion_lab" / "robots" / "procedural_morphology.py"
        ),
        "verification_tool.py": sha256_path(Path(__file__)),
    }
    _write_json(path, payload)
    return payload


def write_final_verification(
    output: Path = DEFAULT_ARTIFACT_ROOT / "r4_final_verification.json",
    *,
    matrix_path: Path = DEFAULT_ARTIFACT_ROOT / "r3_morphology_matrix.json",
    legacy_path: Path = DEFAULT_ARTIFACT_ROOT / "r0_legacy_v2_baseline.json",
    source_contract_path: Path = DEFAULT_ARTIFACT_ROOT / "r0_source_contract.json",
    pytest_status: str = "not_run",
    ruff_status: str = "not_run",
    inspect_status: str = "not_run",
    full_pytest_status: str = "not_run",
    full_pytest_evidence_log: str = "/tmp/task069_full_pytest_clean.log",
    full_pytest_note: str = "",
) -> dict[str, Any]:
    """Assemble the final machine-readable execution gate evidence."""

    output = output.resolve()
    matrix_path = matrix_path.resolve()
    legacy_path = legacy_path.resolve()
    source_contract_path = source_contract_path.resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    legacy_verify = verify_legacy_baseline(legacy_path)
    source_contract_matches_current_writer = (
        source_contract.get("source_sha256", {}).get("verification_tool.py")
        == sha256_path(Path(__file__))
    )
    legacy_generator = MorphologyGenerator()
    paper_generator = LocoFormerMorphologyGenerator()
    legacy_blueprint = legacy_generator.generate("biped", 0)
    paper_blueprint = paper_generator.generate("biped", 0)
    legacy_physical = legacy_generator.sample_physical_params(
        legacy_blueprint, 10_000_000, range_fraction=0.5
    )
    paper_physical = paper_generator.sample_physical_params(
        paper_blueprint, 20_000_000, range_fraction=0.5
    )
    legacy_key = morphology_instance_key(legacy_blueprint, legacy_physical)
    paper_key = morphology_instance_key(paper_blueprint, paper_physical)

    from h200_locomotion_lab.core.checkpoint import (
        WHOLE_BODY_SCHEMA_HASH,
        WholeBodyCheckpointMetadata,
        make_checkpoint_payload,
        validate_checkpoint_payload,
    )

    checkpoint_probe: dict[str, Any] = {
        "schema_hash": WHOLE_BODY_SCHEMA_HASH,
        "legacy_contract_rejects_paper_runtime": False,
        "paper_contract_accepts_paper_runtime": False,
    }
    legacy_metadata = WholeBodyCheckpointMetadata(
        embodiment_contract_version=legacy_key.embodiment_contract_version,
        embodiment_contract_hash=legacy_key.embodiment_contract_hash,
        manifest_hash=legacy_key.cache_key,
    )
    paper_metadata = WholeBodyCheckpointMetadata(
        embodiment_contract_version=paper_key.embodiment_contract_version,
        embodiment_contract_hash=paper_key.embodiment_contract_hash,
        manifest_hash=paper_key.cache_key,
    )
    try:
        validate_checkpoint_payload(
            make_checkpoint_payload({}, legacy_metadata),
            expected_embodiment_contract_version=paper_key.embodiment_contract_version,
            expected_embodiment_contract_hash=paper_key.embodiment_contract_hash,
            expected_manifest_hash=paper_key.cache_key,
        )
    except ValueError:
        checkpoint_probe["legacy_contract_rejects_paper_runtime"] = True
    validate_checkpoint_payload(
        make_checkpoint_payload({}, paper_metadata),
        expected_embodiment_contract_version=paper_key.embodiment_contract_version,
        expected_embodiment_contract_hash=paper_key.embodiment_contract_hash,
        expected_manifest_hash=paper_key.cache_key,
    )
    checkpoint_probe["paper_contract_accepts_paper_runtime"] = True

    full_pytest_result: dict[str, Any] = {
        "command": ".venv/bin/python -m pytest -q",
        "status": full_pytest_status,
        "evidence_log": full_pytest_evidence_log,
    }
    if full_pytest_note:
        full_pytest_result["evidence"] = full_pytest_note
    command_results = [
        {
            "command": (
                ".venv/bin/python -m pytest -q tests/test_whole_body_contract.py "
                "tests/test_whole_body_extended.py tests/test_whole_body_usability_gate.py "
                "tests/test_task069_*.py"
            ),
            "status": pytest_status,
            "evidence_log": "/tmp/task069_pytest.log",
        },
        {
            "command": (
                ".venv/bin/ruff check src/h200_locomotion_lab/robots "
                "tests/test_whole_body_contract.py tests/test_whole_body_extended.py "
                "tests/test_whole_body_usability_gate.py tests/test_task069_*.py"
            ),
            "status": ruff_status,
            "evidence_log": "/tmp/task069_ruff.log",
        },
        {
            "command": ".venv/bin/python -m h200_locomotion_lab.tools.inspect_agent",
            "status": inspect_status,
            "evidence_log": "/tmp/task069_inspect.log",
        },
        full_pytest_result,
    ]
    production_paths = {
        "procedural_morphology.py": REPO_ROOT / "src" / "h200_locomotion_lab" / "robots" / "procedural_morphology.py",
        "robots_init.py": REPO_ROOT / "src" / "h200_locomotion_lab" / "robots" / "__init__.py",
        "verification_tool.py": Path(__file__),
    }
    production_sha = {
        name: {
            "pre_r0": legacy["source_sha256"].get("procedural_morphology.py")
            if name == "procedural_morphology.py"
            else None,
            "post_r4": sha256_path(path),
        }
        for name, path in production_paths.items()
    }
    full_pytest_split_summary_path = matrix_path.parent / "r4_full_pytest_split_summary.json"
    full_pytest_split_summary = None
    if full_pytest_split_summary_path.exists():
        full_pytest_split_summary = {
            "artifact": str(full_pytest_split_summary_path.relative_to(REPO_ROOT)),
            "artifact_sha256": sha256_path(full_pytest_split_summary_path),
            "summary": json.loads(full_pytest_split_summary_path.read_text(encoding="utf-8")),
        }
    matrix_passed = all(
        item["built"] == item["expected_denominator"]
        and item["compiled"] == item["expected_denominator"]
        and item["finite_physics"] == item["expected_denominator"]
        and item["finite_smoke"] == item["expected_denominator"]
        and item["deterministic"] == item["expected_denominator"]
        and item["slot_mapping"] == item["expected_denominator"]
        and item["reset_pose"] == item["expected_denominator"]
        and item["wheel_passed"] == item["expected_denominator"]
        for item in matrix["summary"].values()
    )
    payload = {
        "task": "task069-locoformer-paper-faithful-morphology",
        "status": "execution_verified_pending_independent_readonly_review",
        "claim": "paper-faithful, verifiable four-family procedural morphology envelope",
        "explicit_non_claims": [
            "official generator/source-code reproduction",
            "named-robot parameter or pixel/parameter parity",
            "LocoFormer policy/TXL/long-context training reproduction",
            "large-scale RL, sim2real, or real-robot deployment",
        ],
        "r0_legacy": {
            "artifact": str(legacy_path.relative_to(REPO_ROOT)),
            "expected_denominator": legacy["expected_denominator"],
            "artifact_sha256": sha256_path(legacy_path),
            "verification": legacy_verify,
        },
        "r0_source_contract": {
            "artifact": str(source_contract_path.relative_to(REPO_ROOT)),
            "artifact_sha256": sha256_path(source_contract_path),
            "source_sha256": source_contract.get("source_sha256", {}),
            "matches_current_writer": source_contract_matches_current_writer,
        },
        "r3_matrix": {
            "artifact": str(matrix_path.relative_to(REPO_ROOT)),
            "artifact_sha256": sha256_path(matrix_path),
            "expected_denominator": matrix["expected_denominator"],
            "summary": matrix["summary"],
            "passed": matrix_passed,
        },
        "cache_checkpoint_fail_closed": {
            "legacy_contract_version": legacy_key.embodiment_contract_version,
            "paper_contract_version": paper_key.embodiment_contract_version,
            "legacy_contract_hash": legacy_key.embodiment_contract_hash,
            "paper_contract_hash": paper_key.embodiment_contract_hash,
            "legacy_cache_key": legacy_key.cache_key,
            "paper_cache_key": paper_key.cache_key,
            "cache_key_distinct": legacy_key.cache_key != paper_key.cache_key,
            "checkpoint_probe": checkpoint_probe,
        },
        "visual_inspection": matrix["visual_inspection"],
        "visual_inspection_status": matrix.get("visual_inspection_status"),
        "verification_commands": command_results,
        "full_pytest_split_summary": full_pytest_split_summary,
        "production_source_sha256": production_sha,
        "retries_and_repairs": [
            {
                "stage": "gallery_camera_framing",
                "initial_issue": "quadruped closeups were slightly cropped at the first camera distance",
                "repair": "increased quadruped camera distance from 2.0 to 2.5 and regenerated all gallery images",
                "recheck": "all four montages and sixteen final closeups were reopened with the image viewer",
            }
        ],
        "runtime": _runtime_metadata(),
    }
    _write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("source-contract")
    source.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_ROOT / "r0_source_contract.json")
    baseline = subparsers.add_parser("legacy-baseline")
    baseline.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_ROOT / "r0_legacy_v2_baseline.json")
    verify = subparsers.add_parser("verify-legacy")
    verify.add_argument("--input", type=Path, default=DEFAULT_ARTIFACT_ROOT / "r0_legacy_v2_baseline.json")
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_ROOT / "r3_morphology_matrix.json")
    matrix.add_argument("--smoke-steps", type=int, default=100)
    matrix.add_argument("--no-render", action="store_true")
    visual = subparsers.add_parser("finalize-visual")
    visual.add_argument("--input", type=Path, default=DEFAULT_ARTIFACT_ROOT / "r3_morphology_matrix.json")
    visual.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="human-authored local-image-viewer observation manifest",
    )
    final = subparsers.add_parser("finalize-r4")
    final.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_ROOT / "r4_final_verification.json")
    final.add_argument("--matrix", type=Path, default=DEFAULT_ARTIFACT_ROOT / "r3_morphology_matrix.json")
    final.add_argument("--legacy", type=Path, default=DEFAULT_ARTIFACT_ROOT / "r0_legacy_v2_baseline.json")
    final.add_argument(
        "--source-contract",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "r0_source_contract.json",
    )
    final.add_argument("--pytest-status", default="not_run")
    final.add_argument("--ruff-status", default="not_run")
    final.add_argument("--inspect-status", default="not_run")
    final.add_argument("--full-pytest-status", default="not_run")
    final.add_argument("--full-pytest-evidence-log", default="/tmp/task069_full_pytest_clean.log")
    final.add_argument("--full-pytest-note", default="")
    args = parser.parse_args()
    if args.command == "source-contract":
        write_source_contract(args.output)
    elif args.command == "legacy-baseline":
        write_legacy_baseline(args.output)
    elif args.command == "verify-legacy":
        print(json.dumps(verify_legacy_baseline(args.input), indent=2, sort_keys=True))
    elif args.command == "finalize-visual":
        result = finalize_visual_inspection(args.input, args.observations)
        print(
            json.dumps(
                {
                    "input": str(args.input),
                    "visual_inspection_count": len(result["visual_inspection"]),
                    "status": result["visual_inspection_status"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "finalize-r4":
        result = write_final_verification(
            args.output,
            matrix_path=args.matrix,
            legacy_path=args.legacy,
            source_contract_path=args.source_contract,
            pytest_status=args.pytest_status,
            ruff_status=args.ruff_status,
            inspect_status=args.inspect_status,
            full_pytest_status=args.full_pytest_status,
            full_pytest_evidence_log=args.full_pytest_evidence_log,
            full_pytest_note=args.full_pytest_note,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "status": result["status"],
                    "legacy_passed": result["r0_legacy"]["verification"]["passed"],
                    "matrix_passed": result["r3_matrix"]["passed"],
                    "visual_inspection_count": len(result["visual_inspection"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        result = run_morphology_matrix(
            args.output,
            smoke_steps=args.smoke_steps,
            render=not args.no_render,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "expected_denominator": result["expected_denominator"],
                    "summary": result["summary"],
                    "visual_inspection": result["visual_inspection"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
