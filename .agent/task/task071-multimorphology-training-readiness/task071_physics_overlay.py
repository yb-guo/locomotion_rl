"""Bind official Unitree simulation physics to frozen Task070 witnesses.

The Task070 descriptor, manifest, and anonymous primitive XML remain immutable.
This module emits a separate, versioned physics overlay and runs the Task071 R1
gate directly on the bound XML; it never regenerates morphology from a vendor
source tree.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
TASK070_ATTEMPT010 = (
    ROOT
    / ".agent/task/task070-archetype-constrained-standable-morphology/artifacts/"
    "preview_task070_v2_descriptor_driven_attempt010"
)
FROZEN = Path(
    os.environ.get(
        "TASK070_FROZEN_ATTEMPT010",
        str(
            TASK070_ATTEMPT010
            if TASK070_ATTEMPT010.is_dir()
            else Path(
                "/home/admin1/workspace/proj/locomotion_rl/.agent/task/"
                "task070-archetype-constrained-standable-morphology/artifacts/"
                "preview_task070_v2_descriptor_driven_attempt010"
            )
        ),
    )
)
SOURCE = ROOT / ".external/task071_full_sim_assets/unitree_mujoco"
OUT = ROOT / ".agent/task/task071-multimorphology-training-readiness/artifacts"
OVERLAY_PATH = OUT / "official_sim_physics_overlay_v1.json"
BOUND_R1_PATH = OUT / "r1_g1_go2_bound_official_sim_physics_overlay_v1.json"

OVERLAY_VERSION = "official_sim_physics_overlay_v1"
EXPECTED_MUJOCO_VERSION = "3.12.0"
EXPECTED_REPO_COMMIT = "4134cb5dc7ff1ba7f484deda48b5274b58694519"
EXPECTED_REPO_ORIGIN = "https://github.com/unitreerobotics/unitree_mujoco.git"
COMMAND = (
    "UV_CACHE_DIR=/home/admin1/workspace/store/cache/uv "
    "uv run --isolated --locked --offline --python 3.11 --extra mujoco "
    "python .agent/task/task071-multimorphology-training-readiness/"
    "task071_physics_overlay.py"
)

CASE_SPECS: dict[str, dict[str, Any]] = {
    "unitree_g1": {
        "family": "biped",
        "motor_count": 29,
        "link_count": 30,
        "frozen_dir": "unitree_g1_seed000",
        "frozen_stem": "unitree_g1_29dof",
        "source_xml": "unitree_robots/g1/g1_29dof.xml",
        "source_xml_sha256": (
            "423e28bd718b19f7a65cda539b6f794ddbb268b4b9bdbd85f4bd982b30729617"
        ),
        "descriptor_sha256": (
            "6464ad8af464956ca8c722a95fddd94b7183c0cdd153134b0cbda12f6199662e"
        ),
        "manifest_sha256": (
            "fcb581ac1feb5454bebf7251098548f10648f9f478160adfee0fa764b3405967"
        ),
        "frozen_xml_sha256": (
            "35f6e56eb17b018fa1288db6f74eb8c42fc6616c599008c5050a6af8805120f1"
        ),
        "structural_descriptor_sha256": (
            "6663fc23b3179a3411bf366c7d7f8233ff0526331c91e752634dd5a08a78f246"
        ),
        "blueprint_hash": (
            "33ef2c9c633b4ac2efdccaab615f257a62928de6a1d1e7036de7e29aa36fdbce"
        ),
        "physical_hash": (
            "5b5871f783df0cc4332afe541336eb2ef49905d5cc34f3b42421adcb32aaa987"
        ),
        "dimensions": {"nq": 36, "nv": 35, "nu": 29},
    },
    "unitree_go2": {
        "family": "quadruped",
        "motor_count": 12,
        "link_count": 13,
        "frozen_dir": "unitree_go2_seed000",
        "frozen_stem": "unitree_go2_12dof",
        "source_xml": "unitree_robots/go2/go2.xml",
        "source_xml_sha256": (
            "2014a3d76e30f17ab9447d8a67bd015291f74fa4d71ae30d005f1a32bd693d4b"
        ),
        "descriptor_sha256": (
            "795fd0549643cf96ca83385d0c67ba7fb68485b074c16f610a4c197179e82bac"
        ),
        "manifest_sha256": (
            "a7afd7b32706c27d276c1b71dc527d05ac3c3fede16edde32b6152633169f398"
        ),
        "frozen_xml_sha256": (
            "296ad8fb2ae42f1bb1e437c5e722914794676c7c6f0da51b9a60c674d85ebfa9"
        ),
        "structural_descriptor_sha256": (
            "54478c5cce9dda3fea9fd72d2ab8c56c6fa97a543829e5a2748e387614ed8273"
        ),
        "blueprint_hash": (
            "253e7d1aef63b8ff154a7f91cc627feea8fdc2e05bb478760fd2ac354ab27e5b"
        ),
        "physical_hash": (
            "1d12a22ca9e350ef653f3df8e01831a28825f889dafd57c174ec092756bca642"
        ),
        "dimensions": {"nq": 19, "nv": 18, "nu": 12},
    },
}
STANCE_PROFILE_VERSION = "task071_instance_bound_inverse_static_position_hold_v1"
STANCE_PROFILES: dict[str, dict[str, Any]] = {
    "unitree_g1": {
        "penetration": 0.000618,
        "pose_source": "contact_height_equalization_and_inverse_static_search",
        "overrides": {
            "limb0_hip_pitch": -0.15,
            "limb0_knee_pitch": 0.35,
            "limb0_ankle_pitch": -0.20,
            "limb1_hip_pitch": -0.197755,
            "limb1_knee_pitch": 0.44551,
            "limb1_ankle_pitch": -0.247755,
        },
    },
    "unitree_go2": {
        "penetration": 0.0006902719354629518,
        "pose_source": "official_home_keyframe_neighborhood_and_inverse_static_validation",
        "overrides": {
            f"limb{i}_{part}": value
            for i in range(4)
            for part, value in (("hip_pitch", 0.95), ("knee_pitch", -1.70))
        },
    },
}


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_artifact_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def json_artifact_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_artifact_bytes(payload)).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, _json_artifact_bytes(payload))


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_stance_profile(
    blueprint: Any,
    profile: dict[str, Any],
) -> dict[str, Any]:
    if set(profile) != {"penetration", "pose_source", "overrides"}:
        raise ValueError("stance profile fields are invalid")
    penetration = float(profile["penetration"])
    if not math.isfinite(penetration) or not 0.0 < penetration <= 0.006:
        raise ValueError("stance profile penetration out of range")
    slots = {joint.semantic_slot: joint for joint in blueprint.joints}
    overrides = profile.get("overrides", {})
    if set(overrides) - set(slots) or not overrides:
        raise ValueError("stance profile contains unknown or missing slots")
    normalized: dict[str, float] = {}
    for slot, value in overrides.items():
        target = float(value)
        if not math.isfinite(target) or not (
            slots[slot].joint_range[0] <= target <= slots[slot].joint_range[1]
        ):
            raise ValueError(f"stance profile target out of range: {slot}")
        normalized[slot] = target
    return {
        "joint_nominal_overrides": normalized,
        "pose_source": str(profile["pose_source"]),
        "target_floor_penetration_m": penetration,
    }


def _inverse_static_position_targets(
    model: Any,
    data: Any,
    addresses: list[dict[str, Any]],
    mujoco: Any,
) -> tuple[dict[int, float], dict[str, Any]]:
    root_joint_free = bool(
        model.njnt
        and int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    if not root_joint_free:
        raise ValueError("inverse-static stance requires a free root")
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    root_residual = [float(value) for value in data.qfrc_inverse[:6]]
    if not all(math.isfinite(value) for value in root_residual):
        raise ValueError("inverse-static free-root residual is non-finite")
    targets: dict[int, float] = {}
    evidence: list[dict[str, Any]] = []
    for item in addresses:
        aid, did = int(item["actuator_id"]), int(item["dof_address"])
        gain = float(model.actuator_gainprm[aid, 0])
        torque = float(data.qfrc_inverse[did])
        if not math.isfinite(gain) or gain <= 0.0 or not math.isfinite(torque):
            raise ValueError("invalid inverse-static actuator gain/torque")
        qpos = float(data.qpos[int(item["qpos_address"])])
        target = qpos + torque / gain
        low, high = (float(x) for x in model.actuator_ctrlrange[aid])
        flow, fhigh = (float(x) for x in model.actuator_forcerange[aid])
        if (
            not math.isfinite(target)
            or not low <= target <= high
            or not flow < torque < fhigh
        ):
            raise ValueError(
                f"inverse-static target outside actuator limits: {item['joint'].name}"
            )
        effort_fraction = abs(torque) / max(abs(flow), abs(fhigh), 1e-12)
        targets[aid] = target
        evidence.append(
            {
                "actuator_name": str(model.actuator(aid).name),
                "joint_name": item["joint"].name,
                "semantic_slot": item["joint"].semantic_slot,
                "qpos_rad": qpos,
                "kp": gain,
                "required_torque_nm": torque,
                "position_target_rad": target,
                "ctrlrange_rad": [low, high],
                "forcerange_nm": [flow, fhigh],
                "effort_fraction": effort_fraction,
            }
        )
    if len(targets) != model.nu:
        raise ValueError("inverse-static target count does not match actuator count")
    no_external_wrench = bool(
        not data.xfrc_applied.any() and not data.qfrc_applied.any()
    )
    if not no_external_wrench:
        raise ValueError("inverse-static stance cannot use an external wrench")
    return targets, {
        "solver": "mujoco_mj_inverse_with_zero_qvel_qacc",
        "actuators": evidence,
        "control_target_count": len(targets),
        "max_effort_fraction": max(row["effort_fraction"] for row in evidence),
        "free_root_inverse_residual": root_residual,
        "root_joint_free": root_joint_free,
        "gravity_m_s2": [float(value) for value in model.opt.gravity],
        "external_wrench_applied": False,
        "equality_support_constraint_count": int(model.neq),
        "position_targets_clamped": False,
        "qvel_zero": True,
        "qacc_zero": True,
    }


@dataclass(frozen=True)
class FrozenCase:
    reference_id: str
    spec: dict[str, Any]
    descriptor_path: Path
    manifest_path: Path
    xml_path: Path
    descriptor: dict[str, Any]
    manifest: dict[str, Any]
    blueprint: Any
    physical: Any


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt(values: Any) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _run_git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(SOURCE), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_mujoco() -> Any:
    import mujoco

    if mujoco.__version__ != EXPECTED_MUJOCO_VERSION:
        raise ValueError(
            f"MuJoCo {EXPECTED_MUJOCO_VERSION} required, got {mujoco.__version__}"
        )
    return mujoco


def validate_official_repo() -> dict[str, Any]:
    head = _run_git("rev-parse", "HEAD")
    origin = _run_git("remote", "get-url", "origin")
    status = _run_git("status", "--porcelain=v1", "--untracked-files=all")
    if head != EXPECTED_REPO_COMMIT:
        raise ValueError(f"official Unitree commit mismatch: {head}")
    if origin != EXPECTED_REPO_ORIGIN:
        raise ValueError(f"official Unitree origin mismatch: {origin}")
    if status:
        raise ValueError("official Unitree worktree is dirty")
    return {"commit": head, "origin": origin, "clean": True}


def _frozen_paths(spec: dict[str, Any]) -> tuple[Path, Path, Path]:
    directory = FROZEN / spec["frozen_dir"]
    stem = spec["frozen_stem"]
    return (
        directory / f"{stem}_structural_descriptor.json",
        directory / f"{stem}_anonymous_preview_manifest.json",
        directory / f"{stem}_anonymous_preview.xml",
    )


def _deserialize_blueprint(payload: dict[str, Any]) -> Any:
    from h200_locomotion_lab.robots.procedural_morphology import (
        ActuatorBlueprint,
        JointBlueprint,
        LinkBlueprint,
        MorphologyBlueprint,
        WheelBlueprint,
    )

    links = tuple(
        LinkBlueprint(
            **(
                row
                | {
                    "size": tuple(row["size"]),
                    "pos": tuple(row["pos"]),
                    "com": tuple(row["com"]),
                    "foot_size": (
                        None
                        if row["foot_size"] is None
                        else tuple(row["foot_size"])
                    ),
                }
            )
        )
        for row in payload["links"]
    )
    joints = tuple(
        JointBlueprint(
            **(
                row
                | {
                    "axis": tuple(row["axis"]),
                    "joint_range": tuple(row["joint_range"]),
                }
            )
        )
        for row in payload["joints"]
    )
    actuators = tuple(ActuatorBlueprint(**row) for row in payload["actuators"])
    wheels = tuple(
        WheelBlueprint(
            **(
                row
                | {
                    "axis": tuple(row["axis"]),
                    "joint_range": tuple(row["joint_range"]),
                    "geom_quat": tuple(row["geom_quat"]),
                }
            )
        )
        for row in payload["wheel_specs"]
    )
    return MorphologyBlueprint(
        family=payload["family"],
        seed=int(payload["seed"]),
        links=links,
        joints=joints,
        actuators=actuators,
        nominal_height=float(payload["nominal_height"]),
        has_arms=bool(payload["has_arms"]),
        structural_hash=str(payload["structural_hash"]),
        end_sites=tuple(payload["end_sites"]),
        profile_version=str(payload["profile_version"]),
        contract_version=str(payload["contract_version"]),
        contract_hash=str(payload["contract_hash"]),
        wheel_specs=wheels,
        profile_metadata=payload["profile_metadata"],
    )


def _deserialize_physical(payload: dict[str, Any]) -> Any:
    from h200_locomotion_lab.robots.procedural_morphology import PhysicalParams

    values = dict(payload)
    values["com_offsets"] = {
        name: tuple(offset) for name, offset in payload["com_offsets"].items()
    }
    return PhysicalParams(**values)


def load_frozen_case(reference_id: str) -> FrozenCase:
    from h200_locomotion_lab.robots.procedural_morphology import (
        morphology_blueprint_hash,
        physical_params_hash,
    )

    spec = CASE_SPECS[reference_id]
    descriptor_path, manifest_path, xml_path = _frozen_paths(spec)
    observed = {
        "descriptor": sha256_path(descriptor_path),
        "manifest": sha256_path(manifest_path),
        "xml": sha256_path(xml_path),
    }
    expected = {
        "descriptor": spec["descriptor_sha256"],
        "manifest": spec["manifest_sha256"],
        "xml": spec["frozen_xml_sha256"],
    }
    if observed != expected:
        raise ValueError(f"frozen raw SHA mismatch for {reference_id}: {observed}")

    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    motor_count = int(spec["motor_count"])
    if not (
        descriptor["reference_id"] == manifest["source_reference_id"] == reference_id
        and descriptor["source_motor_count"]
        == descriptor["anonymous_motor_count"]
        == manifest["joint_count"]
        == manifest["actuator_count"]
        == motor_count
        and len(descriptor["source_body_tree"]) == int(spec["link_count"])
        and manifest["link_count"] == int(spec["link_count"])
        and manifest["family"] == spec["family"]
        and manifest["descriptor_sha256"] == spec["descriptor_sha256"]
        and descriptor["descriptor_sha256"]
        == manifest["structural_descriptor_sha256"]
        == spec["structural_descriptor_sha256"]
        and manifest["xml_sha256"] == spec["frozen_xml_sha256"]
        and manifest["blueprint_hash"] == spec["blueprint_hash"]
        and manifest["physical_hash"] == spec["physical_hash"]
    ):
        raise ValueError(f"frozen manifest/descriptor lineage mismatch: {reference_id}")

    blueprint = _deserialize_blueprint(manifest["blueprint_manifest"])
    physical = _deserialize_physical(manifest["physical_manifest"])
    if morphology_blueprint_hash(blueprint) != spec["blueprint_hash"]:
        raise ValueError(f"frozen blueprint hash mismatch: {reference_id}")
    if physical_params_hash(physical) != spec["physical_hash"]:
        raise ValueError(f"frozen physical hash mismatch: {reference_id}")
    metadata = blueprint.profile_metadata
    if not (
        blueprint.family == spec["family"]
        and metadata["source_reference_id"] == reference_id
        and metadata["structural_descriptor_sha256"]
        == spec["structural_descriptor_sha256"]
        and len(blueprint.links) == int(spec["link_count"])
        and len(blueprint.joints) == len(blueprint.actuators) == motor_count
    ):
        raise ValueError(f"frozen blueprint identity mismatch: {reference_id}")
    return FrozenCase(
        reference_id=reference_id,
        spec=spec,
        descriptor_path=descriptor_path,
        manifest_path=manifest_path,
        xml_path=xml_path,
        descriptor=descriptor,
        manifest=manifest,
        blueprint=blueprint,
        physical=physical,
    )


def _close(actual: Any, expected: Any, *, tolerance: float = 5e-6) -> bool:
    if len(actual) != len(expected):
        return False
    return all(
        math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        for left, right in zip(actual, expected, strict=True)
    )


def _xml_body_parent_map(root: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}

    def visit(parent: ET.Element, parent_name: str) -> None:
        for body in parent.findall("body"):
            name = body.get("name")
            if name is None:
                raise ValueError("anonymous XML body has no name")
            result[name] = parent_name
            visit(body, name)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("anonymous XML has no worldbody")
    visit(worldbody, "world")
    return result


def _ordered_mapping(
    case: FrozenCase,
    source_model: Any,
    frozen_root: ET.Element,
    mujoco: Any,
) -> list[dict[str, Any]]:
    tree = case.descriptor["source_body_tree"]
    motors = case.descriptor["source_to_anonymous_motor_bijection"]
    blueprint = case.blueprint
    motor_count = int(case.spec["motor_count"])
    if len({row["source_body_name"] for row in tree}) != len(tree):
        raise ValueError("source body mapping is not unique")
    if len({row["anonymous_link"] for row in tree}) != len(tree):
        raise ValueError("anonymous body mapping is not unique")
    if [row["anonymous_link"] for row in tree] != [link.name for link in blueprint.links]:
        raise ValueError("descriptor body order differs from frozen blueprint")

    frozen_body_elements = [
        body
        for body in frozen_root.iter("body")
        if body.get("name") != "root"
    ]
    if [body.get("name") for body in frozen_body_elements] != [
        link.name for link in blueprint.links
    ]:
        raise ValueError("frozen XML body order differs from frozen blueprint")
    frozen_bodies = {body.get("name"): body for body in frozen_body_elements}
    parent_map = _xml_body_parent_map(frozen_root)

    source_body_names = {row["source_body_name"] for row in tree}
    mapped_joint_ids: list[int] = []
    mapped_actuator_ids: set[int] = set()
    records: list[dict[str, Any]] = []
    if [row["selected_motor_joint"] for row in tree if row["selected_motor_joint"]] != [
        row["source_joint_name"] for row in motors
    ]:
        raise ValueError("descriptor tree and motor order differ")

    for index, motor in enumerate(motors):
        body_row = tree[index + 1]
        joint = blueprint.joints[index]
        actuator = blueprint.actuators[index]
        if not (
            motor["source_joint_name"] == body_row["selected_motor_joint"]
            and motor["source_child_body"] == body_row["source_body_name"]
            and motor["anonymous_child_link"] == body_row["anonymous_link"]
            and joint.name == actuator.joint_name
            and joint.semantic_slot
            == actuator.semantic_slot
            == motor["anonymous_semantic_slot"]
            and joint.parent_link == motor["anonymous_parent_link"]
            and joint.child_link == motor["anonymous_child_link"]
        ):
            raise ValueError(f"ordered motor mapping mismatch at index {index}")

        source_body_id = int(
            mujoco.mj_name2id(
                source_model,
                mujoco.mjtObj.mjOBJ_BODY,
                motor["source_child_body"],
            )
        )
        if source_body_id < 0:
            raise ValueError(f"missing official body: {motor['source_child_body']}")
        source_parent_id = int(source_model.body_parentid[source_body_id])
        source_parent = (
            "root"
            if source_parent_id == 0
            else str(
                mujoco.mj_id2name(
                    source_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    source_parent_id,
                )
            )
        )
        if source_parent != motor["source_parent_body"]:
            raise ValueError(
                f"official parent mismatch: {motor['source_child_body']}"
            )
        if parent_map[motor["anonymous_child_link"]] != motor["anonymous_parent_link"]:
            raise ValueError(
                f"anonymous parent mismatch: {motor['anonymous_child_link']}"
            )

        joint_id = int(
            mujoco.mj_name2id(
                source_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                motor["source_joint_name"],
            )
        )
        if joint_id < 0 or int(source_model.jnt_bodyid[joint_id]) != source_body_id:
            raise ValueError(f"official joint/body mismatch: {motor['source_joint_name']}")
        if not (
            _close(source_model.jnt_axis[joint_id], motor["normalized_local_axis"])
            and _close(source_model.jnt_range[joint_id], motor["joint_range"])
            and _close(joint.axis, motor["normalized_local_axis"])
            and _close(joint.joint_range, motor["joint_range"])
        ):
            raise ValueError(f"joint axis/range mismatch: {motor['source_joint_name']}")
        frozen_joint = frozen_bodies[motor["anonymous_child_link"]].find("joint")
        if frozen_joint is None or not (
            frozen_joint.get("name") == joint.name
            and _close(
                tuple(float(value) for value in frozen_joint.get("axis", "").split()),
                joint.axis,
            )
            and _close(
                tuple(float(value) for value in frozen_joint.get("range", "").split()),
                joint.joint_range,
            )
        ):
            raise ValueError(f"frozen joint mismatch: {joint.name}")

        actuator_ids = [
            actuator_id
            for actuator_id in range(source_model.nu)
            if int(source_model.actuator_trnid[actuator_id, 0]) == joint_id
        ]
        if len(actuator_ids) != 1 or actuator_ids[0] in mapped_actuator_ids:
            raise ValueError(f"official actuator bijection failed: {motor['source_joint_name']}")
        actuator_id = actuator_ids[0]
        mapped_actuator_ids.add(actuator_id)
        mapped_joint_ids.append(joint_id)
        records.append(
            {
                "index": index,
                "module": motor["module"],
                "semantic_slot": motor["anonymous_semantic_slot"],
                "official_parent_body": source_parent,
                "official_child_body": motor["source_child_body"],
                "official_joint": motor["source_joint_name"],
                "official_joint_id": joint_id,
                "official_actuator": str(
                    mujoco.mj_id2name(
                        source_model,
                        mujoco.mjtObj.mjOBJ_ACTUATOR,
                        actuator_id,
                    )
                ),
                "official_actuator_id": actuator_id,
                "anonymous_parent_body": motor["anonymous_parent_link"],
                "anonymous_child_body": motor["anonymous_child_link"],
                "anonymous_joint": joint.name,
                "anonymous_actuator": actuator.name,
            }
        )

    nonfree_joint_ids = [
        joint_id
        for joint_id in range(source_model.njnt)
        if int(source_model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if mapped_joint_ids != nonfree_joint_ids or len(mapped_actuator_ids) != motor_count:
        raise ValueError("official joint/actuator order is not fully mapped")
    if len(records) != motor_count:
        raise ValueError("motor mapping count mismatch")

    for body_row in tree:
        body_id = int(
            mujoco.mj_name2id(
                source_model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_row["source_body_name"],
            )
        )
        if body_id < 0:
            raise ValueError(f"missing official body: {body_row['source_body_name']}")
        children = {
            str(
                mujoco.mj_id2name(
                    source_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    candidate,
                )
            )
            for candidate in range(1, source_model.nbody)
            if int(source_model.body_parentid[candidate]) == body_id
            and str(
                mujoco.mj_id2name(
                    source_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    candidate,
                )
            )
            in source_body_names
        }
        if children != set(body_row["child_bodies"]):
            raise ValueError(f"official child set mismatch: {body_row['source_body_name']}")
    return records


def _scale_by_anonymous(case: FrozenCase) -> dict[str, float]:
    geometry = case.manifest["blueprint_manifest"]["profile_metadata"][
        "geometry_randomization"
    ]
    realized = geometry["realized_source_tree_positions"]
    scale_by_joint: dict[str, float] = {}
    for row in realized:
        joint_name = row["source_joint_name"]
        scale = row.get("module_scale", row.get("uniform_visual_scale"))
        if joint_name in scale_by_joint or scale is None or float(scale) <= 0.0:
            raise ValueError(f"invalid morphology scale: {joint_name}")
        scale_by_joint[joint_name] = float(scale)
    if len(scale_by_joint) != int(case.spec["motor_count"]):
        raise ValueError("incomplete morphology scale mapping")
    root_scale = (
        geometry["module_scales"]["waist"]
        if case.reference_id == "unitree_g1"
        else geometry["uniform_visual_scale"]
    )
    result: dict[str, float] = {}
    for row in case.descriptor["source_body_tree"]:
        selected = row["selected_motor_joint"]
        result[row["anonymous_link"]] = (
            float(root_scale) if selected is None else scale_by_joint[selected]
        )
    if len(result) != int(case.spec["link_count"]):
        raise ValueError("anonymous morphology scale mapping is incomplete")
    return result


def scaled_inertial(
    mass: float,
    position: Any,
    quaternion: Any,
    inertia: Any,
    scale: float,
) -> dict[str, Any]:
    if scale <= 0.0 or mass <= 0.0:
        raise ValueError("mass and morphology scale must be positive")
    return {
        "mass": float(mass),
        "pos": tuple(float(value) * scale for value in position),
        "quat": tuple(float(value) for value in quaternion),
        "diaginertia": tuple(float(value) * scale * scale for value in inertia),
    }


def _structural_signature(
    root: ET.Element,
    terminal_geoms: set[str],
) -> tuple[Any, ...]:
    def visit(element: ET.Element) -> tuple[Any, ...]:
        attrs = dict(element.attrib)
        if element.tag == "inertial":
            for name in ("mass", "pos", "quat", "diaginertia", "fullinertia"):
                attrs.pop(name, None)
        elif element.tag == "joint" and attrs.get("type") != "free":
            for name in ("damping", "armature", "frictionloss"):
                attrs.pop(name, None)
        elif element.tag == "position":
            attrs.pop("forcerange", None)
        elif element.tag == "geom" and attrs.get("name") in terminal_geoms:
            attrs.pop("friction", None)
        return (
            element.tag,
            tuple(sorted(attrs.items())),
            tuple(visit(child) for child in element),
        )

    return visit(root)


def _finite_compile_evidence(model: Any, mujoco: Any) -> dict[str, Any]:
    import numpy as np

    arrays = {
        "body_mass": model.body_mass,
        "body_ipos": model.body_ipos,
        "body_iquat": model.body_iquat,
        "body_inertia": model.body_inertia,
        "dof_M0": model.dof_M0,
        "dof_damping": model.dof_damping,
        "dof_armature": model.dof_armature,
        "dof_frictionloss": model.dof_frictionloss,
        "actuator_ctrlrange": model.actuator_ctrlrange,
        "actuator_forcerange": model.actuator_forcerange,
        "geom_friction": model.geom_friction,
    }
    finite = {name: bool(np.isfinite(values).all()) for name, values in arrays.items()}
    positive_generated_mass = bool((model.body_mass[2:] > 0.0).all())
    positive_dof_inertia = bool((model.dof_M0 > 0.0).all())
    passed = all(finite.values()) and positive_generated_mass and positive_dof_inertia
    if not passed:
        raise ValueError("bound XML compiled physics is not finite/positive")
    return {
        "mujoco_version": mujoco.__version__,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "total_mass_kg": float(model.body_mass.sum()),
        "finite_arrays": finite,
        "positive_generated_body_mass": positive_generated_mass,
        "positive_dof_inertia": positive_dof_inertia,
        "passed": True,
    }


def _terminal_friction(
    case: FrozenCase,
    mapping: list[dict[str, Any]],
    source_model: Any,
    mujoco: Any,
) -> dict[str, tuple[float, float, float]]:
    mapping_by_anon = {row["anonymous_child_body"]: row for row in mapping}
    foot_links = [link for link in case.blueprint.links if link.foot]
    expected_count = 2 if case.reference_id == "unitree_g1" else 4
    if len(foot_links) != expected_count:
        raise ValueError("unexpected frozen terminal foot count")
    result: dict[str, tuple[float, float, float]] = {}
    for link in foot_links:
        source_body = mapping_by_anon[link.name]["official_child_body"]
        body_id = int(
            mujoco.mj_name2id(
                source_model,
                mujoco.mjtObj.mjOBJ_BODY,
                source_body,
            )
        )
        if case.reference_id == "unitree_go2":
            geom_name = source_body.split("_", maxsplit=1)[0]
            geom_id = int(
                mujoco.mj_name2id(
                    source_model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    geom_name,
                )
            )
            if geom_id < 0 or int(source_model.geom_bodyid[geom_id]) != body_id:
                raise ValueError(f"official terminal geom mismatch: {source_body}")
            friction = tuple(float(value) for value in source_model.geom_friction[geom_id])
        else:
            candidates = {
                tuple(float(value) for value in source_model.geom_friction[geom_id])
                for geom_id in range(source_model.ngeom)
                if int(source_model.geom_bodyid[geom_id]) == body_id
                and (
                    int(source_model.geom_contype[geom_id]) != 0
                    or int(source_model.geom_conaffinity[geom_id]) != 0
                )
            }
            if len(candidates) != 1:
                raise ValueError(f"official terminal friction is ambiguous: {source_body}")
            friction = candidates.pop()
        result[f"{link.name}_footpad"] = friction
    return result


def _bind_case(
    case: FrozenCase,
    mujoco: Any,
    *,
    write_artifact: bool,
) -> tuple[dict[str, Any], str]:
    source_path = SOURCE / case.spec["source_xml"]
    if sha256_path(source_path) != case.spec["source_xml_sha256"]:
        raise ValueError(f"official source XML SHA mismatch: {case.reference_id}")
    source_model = mujoco.MjModel.from_xml_path(str(source_path))
    frozen_root = ET.parse(case.xml_path).getroot()
    mapping = _ordered_mapping(case, source_model, frozen_root, mujoco)
    scale_by_anonymous = _scale_by_anonymous(case)
    terminal_friction = _terminal_friction(
        case,
        mapping,
        source_model,
        mujoco,
    )
    body_elements = {
        body.get("name"): body
        for body in frozen_root.iter("body")
        if body.get("name") is not None
    }
    body_records: list[dict[str, Any]] = []
    for body_row in case.descriptor["source_body_tree"]:
        anonymous_body = body_row["anonymous_link"]
        official_body = body_row["source_body_name"]
        body_id = int(
            mujoco.mj_name2id(
                source_model,
                mujoco.mjtObj.mjOBJ_BODY,
                official_body,
            )
        )
        scale = scale_by_anonymous[anonymous_body]
        inertial = scaled_inertial(
            float(source_model.body_mass[body_id]),
            source_model.body_ipos[body_id],
            source_model.body_iquat[body_id],
            source_model.body_inertia[body_id],
            scale,
        )
        target = body_elements[anonymous_body].find("inertial")
        if target is None:
            raise ValueError(f"frozen body has no inertial: {anonymous_body}")
        target.attrib.clear()
        target.attrib.update(
            {
                "mass": f"{inertial['mass']:.12g}",
                "pos": _fmt(inertial["pos"]),
                "quat": _fmt(inertial["quat"]),
                "diaginertia": _fmt(inertial["diaginertia"]),
            }
        )
        body_records.append(
            {
                "official_body": official_body,
                "anonymous_body": anonymous_body,
                "morphology_scale": scale,
                "mass_kg": inertial["mass"],
                "com_m": list(inertial["pos"]),
                "inertial_quaternion_wxyz": list(inertial["quat"]),
                "diagonal_inertia_kg_m2": list(inertial["diaginertia"]),
            }
        )

    frozen_actuators = {
        actuator.get("joint"): actuator
        for actuator in frozen_root.findall(".//actuator/*")
    }
    motor_records: list[dict[str, Any]] = []
    for row in mapping:
        body = body_elements[row["anonymous_child_body"]]
        joint = body.find("joint")
        if joint is None or joint.get("name") != row["anonymous_joint"]:
            raise ValueError(f"missing frozen joint: {row['anonymous_joint']}")
        joint_id = int(row["official_joint_id"])
        dof_id = int(source_model.jnt_dofadr[joint_id])
        actuator_id = int(row["official_actuator_id"])
        joint.set("damping", f"{float(source_model.dof_damping[dof_id]):.12g}")
        joint.set("armature", f"{float(source_model.dof_armature[dof_id]):.12g}")
        joint.set(
            "frictionloss",
            f"{float(source_model.dof_frictionloss[dof_id]):.12g}",
        )
        actuator = frozen_actuators.get(row["anonymous_joint"])
        if actuator is None or actuator.tag != "position":
            raise ValueError(
                f"frozen position-actuator semantics missing: {row['anonymous_joint']}"
            )
        force_range = tuple(
            float(value) for value in source_model.actuator_ctrlrange[actuator_id]
        )
        actuator.set("forcerange", _fmt(force_range))
        motor_records.append(
            row
            | {
                "axis": [float(value) for value in source_model.jnt_axis[joint_id]],
                "range_rad": [
                    float(value) for value in source_model.jnt_range[joint_id]
                ],
                "joint_damping": float(source_model.dof_damping[dof_id]),
                "joint_armature": float(source_model.dof_armature[dof_id]),
                "joint_frictionloss": float(source_model.dof_frictionloss[dof_id]),
                "force_range": list(force_range),
                "position_kp": float(actuator.get("kp", "nan")),
                "position_kd": float(actuator.get("kv", "nan")),
                "gain_provenance": "frozen_Task070_companion_motor_config",
                "force_and_joint_dynamics_provenance": (
                    "official_unitree_mujoco_compiled_effective_values"
                ),
            }
        )

    friction_records: list[dict[str, Any]] = []
    for geom_name, friction in terminal_friction.items():
        candidates = [
            geom
            for geom in frozen_root.iter("geom")
            if geom.get("name") == geom_name
        ]
        if len(candidates) != 1:
            raise ValueError(f"frozen terminal geom mapping failed: {geom_name}")
        candidates[0].set("friction", _fmt(friction))
        friction_records.append(
            {
                "anonymous_terminal_geom": geom_name,
                "friction": list(friction),
                "provenance": (
                    "official Unitree terminal contact geom effective friction"
                ),
            }
        )

    for element in frozen_root.iter():
        if element.tag in {"mesh", "texture", "material"} or "mesh" in element.attrib:
            raise ValueError("bound anonymous XML contains vendor asset identity")
    frozen_for_signature = ET.parse(case.xml_path).getroot()
    terminal_names = set(terminal_friction)
    if _structural_signature(frozen_for_signature, terminal_names) != _structural_signature(
        frozen_root,
        terminal_names,
    ):
        raise ValueError("physics overlay changed frozen structural XML")

    xml = ET.tostring(frozen_root, encoding="unicode") + "\n"
    if any(token in xml.lower() for token in ("unitree", "logo", "g1_", "go2")):
        raise ValueError("bound anonymous XML leaked source model identity")
    bound_model = mujoco.MjModel.from_xml_string(xml)
    compile_evidence = _finite_compile_evidence(bound_model, mujoco)
    if {
        key: compile_evidence[key] for key in ("nq", "nv", "nu")
    } != case.spec["dimensions"]:
        raise ValueError(f"bound compile dimensions mismatch: {case.reference_id}")

    output = (
        OUT
        / "models"
        / (
            f"{case.reference_id}_{case.spec['family']}_"
            f"{OVERLAY_VERSION}.xml"
        )
    )
    output_sha256 = _text_sha256(xml)
    if write_artifact:
        _atomic_write(output, xml.encode("utf-8"))
    frozen_after = {
        "descriptor": sha256_path(case.descriptor_path),
        "manifest": sha256_path(case.manifest_path),
        "xml": sha256_path(case.xml_path),
    }
    if frozen_after != {
        "descriptor": case.spec["descriptor_sha256"],
        "manifest": case.spec["manifest_sha256"],
        "xml": case.spec["frozen_xml_sha256"],
    }:
        raise ValueError("frozen input changed during overlay generation")

    record = {
        "reference_id": case.reference_id,
        "family": case.spec["family"],
        "mapping_counts": {
            "bodies": len(body_records),
            "joints": len(motor_records),
            "actuators": len(motor_records),
            "terminal_contacts": len(friction_records),
        },
        "official_source": {
            "path": str(source_path.relative_to(ROOT)),
            "sha256": sha256_path(source_path),
        },
        "frozen_input": {
            "descriptor_path": str(case.descriptor_path),
            "descriptor_sha256": sha256_path(case.descriptor_path),
            "manifest_path": str(case.manifest_path),
            "manifest_sha256": sha256_path(case.manifest_path),
            "xml_path": str(case.xml_path),
            "xml_sha256": sha256_path(case.xml_path),
            "structural_descriptor_sha256": case.spec[
                "structural_descriptor_sha256"
            ],
            "blueprint_hash": case.spec["blueprint_hash"],
            "physical_hash": case.spec["physical_hash"],
            "unchanged_after_generation": True,
        },
        "binding_formula": {
            "mass": "official nominal mass",
            "local_com": "official local COM * frozen morphology scale",
            "diagonal_inertia": (
                "official diagonal inertia * frozen morphology scale^2"
            ),
            "position_kp_kd": "frozen Task070 companion motor config",
            "joint_dynamics_and_force_limit": (
                "official Unitree MuJoCo compiled effective values"
            ),
            "runtime_fault_process": "not applied in nominal overlay",
        },
        "body_mapping": body_records,
        "motor_mapping": motor_records,
        "terminal_contact_mapping": friction_records,
        "structure_preservation": {
            "frozen_structural_signature_equal": True,
            "canonical_root_preserved": True,
            "position_actuator_semantics_preserved": True,
            "primitive_geometry_only": True,
            "mesh_texture_logo_copied": False,
        },
        "compile_evidence": compile_evidence,
        "output_xml": str(output.relative_to(ROOT)),
        "output_xml_sha256": output_sha256,
        "claim_boundary": {
            "real_system_identified": False,
            "nominal_sim_prior_only": True,
            "walking_claimed": False,
            "train_ready_claimed": False,
        },
    }
    return record, xml


def _generate_overlay_bundle(
    *,
    write_artifact: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    mujoco = _require_mujoco()
    repo = validate_official_repo()
    records = []
    bound_xml: dict[str, str] = {}
    for reference_id in CASE_SPECS:
        case = load_frozen_case(reference_id)
        record, xml = _bind_case(case, mujoco, write_artifact=write_artifact)
        records.append(record)
        bound_xml[reference_id] = xml
    payload = {
        "artifact": OVERLAY_VERSION,
        "overlay_version": OVERLAY_VERSION,
        "official_repo": repo,
        "mujoco_version": mujoco.__version__,
        "denominator": 2,
        "records": records,
        "summary": {
            "bound": sum(record["compile_evidence"]["passed"] for record in records),
            "lineage_preserved": sum(
                record["frozen_input"]["unchanged_after_generation"]
                for record in records
            ),
            "structure_preserved": sum(
                record["structure_preservation"][
                    "frozen_structural_signature_equal"
                ]
                for record in records
            ),
        },
        "claim_boundary": {
            "real_system_identified": False,
            "nominal_sim_prior_only": True,
            "task071_training_readiness_passed": False,
            "ppo_or_long_training_started": False,
        },
    }
    if write_artifact:
        write_json_artifact(OVERLAY_PATH, payload)
    return payload, bound_xml


def generate_overlay(*, write_artifact: bool = True) -> dict[str, Any]:
    payload, _ = _generate_overlay_bundle(write_artifact=write_artifact)
    return payload


def _validate_overlay_for_r1(
    overlay: dict[str, Any],
    *,
    bound_xml: dict[str, str],
    require_persisted: bool,
) -> dict[str, dict[str, Any]]:
    if require_persisted:
        persisted = json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))
        if overlay != persisted:
            raise ValueError("R1 overlay payload does not match regenerated artifact")
    if not (
        overlay.get("artifact") == overlay.get("overlay_version") == OVERLAY_VERSION
        and overlay.get("official_repo") == validate_official_repo()
        and overlay.get("mujoco_version") == EXPECTED_MUJOCO_VERSION
        and overlay.get("denominator") == 2
        and len(overlay.get("records", ())) == 2
    ):
        raise ValueError("R1 overlay metadata is invalid")
    records = {record.get("reference_id"): record for record in overlay["records"]}
    if set(records) != set(CASE_SPECS):
        raise ValueError("R1 overlay case denominator is invalid")
    for reference_id, spec in CASE_SPECS.items():
        record = records[reference_id]
        expected_output = (
            OUT
            / "models"
            / f"{reference_id}_{spec['family']}_{OVERLAY_VERSION}.xml"
        )
        expected_counts = {
            "bodies": spec["link_count"],
            "joints": spec["motor_count"],
            "actuators": spec["motor_count"],
            "terminal_contacts": 2 if reference_id == "unitree_g1" else 4,
        }
        frozen = record.get("frozen_input", {})
        structure = record.get("structure_preservation", {})
        compile_evidence = record.get("compile_evidence", {})
        if not (
            record.get("family") == spec["family"]
            and record.get("mapping_counts") == expected_counts
            and record.get("output_xml") == str(expected_output.relative_to(ROOT))
            and (
                not require_persisted
                or (
                    expected_output.is_file()
                    and record.get("output_xml_sha256") == sha256_path(expected_output)
                )
            )
            and record.get("output_xml_sha256") == _text_sha256(bound_xml[reference_id])
            and frozen.get("descriptor_sha256") == spec["descriptor_sha256"]
            and frozen.get("manifest_sha256") == spec["manifest_sha256"]
            and frozen.get("xml_sha256") == spec["frozen_xml_sha256"]
            and frozen.get("structural_descriptor_sha256")
            == spec["structural_descriptor_sha256"]
            and frozen.get("blueprint_hash") == spec["blueprint_hash"]
            and frozen.get("physical_hash") == spec["physical_hash"]
            and frozen.get("unchanged_after_generation") is True
            and structure
            == {
                "frozen_structural_signature_equal": True,
                "canonical_root_preserved": True,
                "position_actuator_semantics_preserved": True,
                "primitive_geometry_only": True,
                "mesh_texture_logo_copied": False,
            }
            and compile_evidence.get("passed") is True
            and {
                key: compile_evidence.get(key) for key in ("nq", "nv", "nu")
            }
            == spec["dimensions"]
        ):
            raise ValueError(f"R1 overlay binding is invalid: {reference_id}")
    return records


def run_bound_r1(
    overlay: dict[str, Any] | None = None,
    *,
    write_artifact: bool = True,
) -> dict[str, Any]:
    mujoco = _require_mujoco()
    regenerated, bound_xml = _generate_overlay_bundle(write_artifact=write_artifact)
    if overlay is not None and overlay != regenerated:
        raise ValueError("caller-supplied R1 overlay differs from regenerated binding")
    overlay = regenerated
    from h200_locomotion_lab.robots import whole_body_stance as stance_contract
    from h200_locomotion_lab.robots.procedural_morphology import morphology_instance_key
    from h200_locomotion_lab.tools import task070_morphology_verification as verification

    overlay_by_case = _validate_overlay_for_r1(
        overlay,
        bound_xml=bound_xml,
        require_persisted=write_artifact,
    )
    overlay_artifact_sha256 = json_artifact_sha256(overlay)
    if write_artifact and sha256_path(OVERLAY_PATH) != overlay_artifact_sha256:
        raise ValueError("persisted overlay artifact SHA mismatch")
    records: list[dict[str, Any]] = []
    for reference_id in CASE_SPECS:
        case = load_frozen_case(reference_id)
        overlay_record = overlay_by_case[reference_id]
        xml_path = ROOT / overlay_record["output_xml"]
        xml = bound_xml[reference_id]
        if _text_sha256(xml) != overlay_record["output_xml_sha256"]:
            raise ValueError(f"bound XML SHA mismatch: {reference_id}")
        model = mujoco.MjModel.from_xml_string(xml)
        if not math.isclose(
            float(model.opt.timestep),
            0.002,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"bound R1 timestep mismatch: {reference_id}")
        profile = _validate_stance_profile(
            case.blueprint,
            STANCE_PROFILES[reference_id],
        )
        overrides = profile["joint_nominal_overrides"]
        penetration = profile["target_floor_penetration_m"]
        profile_manifest = {
            "version": STANCE_PROFILE_VERSION,
            "reference_id": reference_id,
            "pose_source": profile["pose_source"],
            "target_floor_penetration_m": penetration,
            "joint_nominal_overrides": overrides,
            "base_frozen_blueprint_hash": case.spec["blueprint_hash"],
            "frozen_morphology_changed": False,
            "joint_order_axis_range_changed": False,
            "physics_overlay_changed": False,
        }
        profile_manifest["sha256"] = _payload_sha256(profile_manifest)
        _, reset, _, _ = verification._prepare_stance_pose(
            model,
            case.blueprint,
            case.physical,
            mujoco,
            target_floor_penetration_m=penetration,
            joint_nominal_overrides=overrides,
        )
        profile_data, _, _, profile_addresses = verification._prepare_stance_pose(
            model,
            case.blueprint,
            case.physical,
            mujoco,
            target_floor_penetration_m=penetration,
            joint_nominal_overrides=overrides,
        )
        explicit_targets, feedforward = _inverse_static_position_targets(
            model,
            profile_data,
            profile_addresses,
            mujoco,
        )
        feedforward["stance_profile_sha256"] = profile_manifest["sha256"]
        feedforward["bound_xml_sha256"] = overlay_record["output_xml_sha256"]
        feedforward["hidden_support_added"] = False
        stance_solution = stance_contract.StanceSolution(
            instance_key=morphology_instance_key(case.blueprint, case.physical),
            base_height=float(reset["root_z"]),
            joint_qpos={
                joint.semantic_slot: float(overrides.get(joint.semantic_slot, joint.nominal))
                for joint in case.blueprint.joints
            },
            actuator_ctrl={
                str(row["semantic_slot"]): float(row["position_target_rad"])
                for row in feedforward["actuators"]
            },
            model_xml_sha256=overlay_record["output_xml_sha256"],
        )
        stance_solution.validate_for(
            case.blueprint,
            case.physical,
            expected_model_xml_sha256=overlay_record["output_xml_sha256"],
        )
        response = verification._arena_actuator_response(
            mujoco.MjModel.from_xml_string(xml),
            case.blueprint,
            case.physical,
            response_steps=32,
            mujoco=mujoco,
        )
        _, stance = verification._stance_rollout(
            mujoco.MjModel.from_xml_string(xml),
            case.blueprint,
            case.physical,
            steps=1000,
            wheel_velocity_hold=True,
            disturbance=False,
            mujoco=mujoco,
            target_floor_penetration_m=penetration,
            joint_nominal_overrides=overrides,
            explicit_targets=explicit_targets,
        )
        dimensions = {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
        }
        accounting_exact = dimensions == case.spec["dimensions"]
        record = {
            "reference_id": reference_id,
            "family": case.spec["family"],
            "frozen_task070_input_match": True,
            "structural_descriptor_sha256": case.spec[
                "structural_descriptor_sha256"
            ],
            "frozen_blueprint_hash": case.spec["blueprint_hash"],
            "frozen_physical_hash": case.spec["physical_hash"],
            "overlay_artifact_sha256": overlay_artifact_sha256,
            "bound_xml_path": str(xml_path.relative_to(ROOT)),
            "bound_xml_sha256": _text_sha256(xml),
            "compiled": True,
            "model_dimensions": dimensions,
            "accounting_exact": accounting_exact,
            "reset_pose": reset,
            "reset_pose_passed": bool(reset["reset_pose_passed"]),
            "actuator_response": response,
            "all_actuators_responsive": bool(response["all_actuators_responsive"]),
            "stance_hold": stance,
            "stance_profile": profile_manifest,
            "inverse_static_feedforward": feedforward,
            "stance_solution": {
                "manifest": stance_solution.manifest(),
                "sha256": stance_solution.solution_hash,
            },
            "stance_hold_passed": bool(stance.get("passed", False)),
            "operational_actuator_smoke_passed": bool(
                accounting_exact
                and reset["reset_pose_passed"]
                and response["all_actuators_responsive"]
            ),
        }
        records.append(record)

    stance_passed = sum(record["stance_hold_passed"] for record in records)
    payload = {
        "artifact": "task071_bound_official_sim_physics_r1_v1",
        "task": "task071-multimorphology-training-readiness",
        "runtime": {
            "command": COMMAND,
            "git_head": _git_head(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mujoco_version": mujoco.__version__,
            "source": {
                "physics_overlay_path": str(Path(__file__).relative_to(ROOT)),
                "physics_overlay_sha256": sha256_path(Path(__file__)),
                "stance_helper_path": str(
                    Path(verification.__file__).relative_to(ROOT)
                ),
                "stance_helper_sha256": sha256_path(Path(verification.__file__)),
                "stance_contract_path": str(
                    Path(stance_contract.__file__).relative_to(ROOT)
                ),
                "stance_contract_sha256": sha256_path(Path(stance_contract.__file__)),
            },
        },
        "overlay_version": OVERLAY_VERSION,
        "overlay_artifact_path": str(OVERLAY_PATH.relative_to(ROOT)),
        "overlay_artifact_sha256": overlay_artifact_sha256,
        "mujoco_version": mujoco.__version__,
        "denominator": 2,
        "response_steps_per_actuator": 32,
        "stance_steps": 1000,
        "timestep_seconds": 0.002,
        "stance_duration_seconds": 2.0,
        "records": records,
        "summary": {
            "compiled": sum(record["compiled"] for record in records),
            "accounting_exact": sum(record["accounting_exact"] for record in records),
            "frozen_lineage_match": sum(
                record["frozen_task070_input_match"] for record in records
            ),
            "reset_pose_passed": sum(
                record["reset_pose_passed"] for record in records
            ),
            "all_actuators_responsive": sum(
                record["all_actuators_responsive"] for record in records
            ),
            "stance_hold_passed": stance_passed,
        },
        "task071_r1_admission_passed": all(
            record["compiled"]
            and record["accounting_exact"]
            and record["frozen_task070_input_match"]
            and record["reset_pose_passed"]
            and record["all_actuators_responsive"]
            and record["stance_hold_passed"]
            for record in records
        ),
        "failure_reasons": (
            [] if stance_passed == 2 else [f"bound_stance_passed_{stance_passed}_of_2"]
        ),
        "claim_boundary": {
            "walking_claimed": False,
            "train_ready_claimed": False,
            "real_system_identified": False,
            "ppo_or_long_training_started": False,
        },
    }
    if write_artifact:
        write_json_artifact(BOUND_R1_PATH, payload)
    return payload


def bound_r1_matrix(bound_r1: dict[str, Any]) -> dict[str, Any]:
    records = [
        {
            "reference_id": record["reference_id"],
            "family": record["family"],
            "compiled": record["compiled"],
            "accounting_exact": record["accounting_exact"],
            "frozen_task070_input_match": record["frozen_task070_input_match"],
            "structural_descriptor_sha256": record[
                "structural_descriptor_sha256"
            ],
            "reset_pose_passed": record["reset_pose_passed"],
            "all_actuators_responsive": record["all_actuators_responsive"],
            "stance_hold_passed": record["stance_hold_passed"],
            "stance_steps": record["stance_hold"].get("steps"),
            "stance_duration_seconds": record["stance_hold"].get(
                "duration_seconds"
            ),
            "support_gate_passed": record["stance_hold"].get(
                "support_gate_passed"
            ),
            "finite": record["stance_hold"].get("finite"),
            "bound_xml_sha256": record["bound_xml_sha256"],
        }
        for record in bound_r1["records"]
    ]
    return {
        "artifact": "task071_r1_reset_stance_matrix_bound_official_sim_physics_v1",
        "source": {
            "path": str(BOUND_R1_PATH.relative_to(ROOT)),
            "sha256": sha256_path(BOUND_R1_PATH),
            "classification": "fresh bound Task071 R1 evidence",
        },
        "denominator": 2,
        "records": records,
        "frozen_task070_input_match_all": all(
            record["frozen_task070_input_match"] for record in records
        ),
        "fresh_stance_passed": sum(
            record["stance_hold_passed"] for record in records
        ),
        "fresh_stance_denominator": 2,
        "task071_r1_admission_passed": bound_r1["task071_r1_admission_passed"],
        "failure_reasons": bound_r1["failure_reasons"],
        "ppo_or_long_training_started": False,
    }


def main() -> int:
    overlay = generate_overlay()
    bound_r1 = run_bound_r1(overlay)
    print(
        "Task071 official physics overlay: "
        f"bound={overlay['summary']['bound']}/2, "
        f"lineage={bound_r1['summary']['frozen_lineage_match']}/2, "
        f"response={bound_r1['summary']['all_actuators_responsive']}/2, "
        f"stance={bound_r1['summary']['stance_hold_passed']}/2, "
        f"r1_admission={bound_r1['task071_r1_admission_passed']}, ppo=False"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
