"""Task071 G1/Go2 fail-closed physical-attribute probe.

This probe intentionally avoids importing MuJoCo. It audits the frozen Task070
v2 compiler inputs and outputs, then records why a dynamic Task071 rerun is not
admitted in the current locked environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
TASK070 = ROOT / ".agent/task/task070-archetype-constrained-standable-morphology"
TASK071 = ROOT / ".agent/task/task071-multimorphology-training-readiness"
OUT = TASK071 / "artifacts"
ARENA = TASK070 / "artifacts/arena_task070_v2_attempt010/flat_arena_smoke.json"
PROBE_COMMAND = (
    "UV_CACHE_DIR=/home/admin1/workspace/store/cache/uv "
    "uv run --isolated --locked --python 3.11 python "
    ".agent/task/task071-multimorphology-training-readiness/"
    "task071_physical_probe.py"
)
EXPECTED_CASES = {
    "unitree_g1": {"family": "biped", "motor_count": 29},
    "unitree_go2": {"family": "quadruped", "motor_count": 12},
}
TASK070_EVIDENCE = {
    "canonical_root_audit": {
        "path": TASK070
        / "artifacts/preview_task070_v2_descriptor_driven_attempt010/"
        "canonical_root_frame_audit.json",
        "sha256": "f96da04079f8155221b4067cac6af31968182209f809a08ee1face32d28b8547",
    },
    "visual_observation": {
        "path": TASK070
        / "artifacts/preview_task070_v2_descriptor_driven_attempt010/"
        "all_configuration_agent_visual_observation.json",
        "sha256": "0251b4b8e3cbcdf7984676a659669a8860c6d13664f846c8ff06c307451c141a",
    },
    "arena_smoke": {
        "path": ARENA,
        "sha256": "8a8d281d9ede3d32713ceb92c96976f8eacab864d35d6feba1c31fb2db52436d",
    },
}
COVERAGE_KEYS = (
    "source_motor_count",
    "config_record_count",
    "usable_quantitative_prior_count",
    "rejected_placeholder_count",
    "declared_effort_count",
    "declared_velocity_count",
    "source_pd_gain_count",
    "source_armature_count",
    "source_rotor_inertia_count",
    "source_gear_ratio_count",
    "source_control_mode_count",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _write(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _runtime() -> dict[str, object]:
    return {
        "command": PROBE_COMMAND,
        "git_head": _git_head(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hardware_scope": (
            "RTX 5060 Ti-first project; this R0 probe is CPU/static and did not use GPU"
        ),
        "robot_asset_or_dataset_downloads_performed": False,
        "locked_dependency_download_attempted": True,
        "locked_dependency_download_completed": False,
        "training_performed": False,
    }


def _offline_mujoco_probe() -> dict[str, object]:
    command = (
        "uv",
        "run",
        "--isolated",
        "--locked",
        "--python",
        "3.11",
        "--offline",
        "--extra",
        "mujoco",
        "python",
        "-c",
        "import mujoco; print(mujoco.__version__)",
    )
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = "/home/admin1/workspace/store/cache/uv"
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    return {
        "command": (
            "UV_CACHE_DIR=/home/admin1/workspace/store/cache/uv "
            "uv run --isolated --locked --python 3.11 --offline --extra mujoco "
            "python -c 'import mujoco; print(mujoco.__version__)'"
        ),
        "returncode": result.returncode,
        "output": output,
        "output_sha256": _sha256_bytes(output.encode()),
        "dependency_available": result.returncode == 0,
    }


def _schema_identity() -> tuple[str, int, int]:
    from h200_locomotion_lab.robots.whole_body_slots import (
        WHOLE_BODY_ACTION_DIM,
        WHOLE_BODY_ACTOR_OBS_DIM,
        WHOLE_BODY_SCHEMA_HASH,
    )

    return (
        WHOLE_BODY_SCHEMA_HASH,
        WHOLE_BODY_ACTION_DIM,
        WHOLE_BODY_ACTOR_OBS_DIM,
    )


def _mapping_hash(values: Mapping[str, object]) -> str:
    return _sha256_bytes(
        json.dumps(dict(values), sort_keys=True, separators=(",", ":")).encode()
    )


def _relative_residual(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-12)


def _compact_coverage(coverage: Mapping[str, object]) -> dict[str, object]:
    return {key: coverage.get(key) for key in COVERAGE_KEYS}


def _changed(
    nominal: Mapping[str, object],
    full_a: Mapping[str, object],
    full_b: Mapping[str, object],
    key: str,
) -> dict[str, bool]:
    return {
        "changed_from_nominal": (
            nominal[key] != full_a[key] or nominal[key] != full_b[key]
        ),
        "full_samples_distinct": full_a[key] != full_b[key],
    }


def _sample_summary(
    blueprint: Any,
    physical: Any,
    xml: str,
    seed: int,
) -> dict[str, object]:
    from h200_locomotion_lab.robots.procedural_morphology import physical_params_hash

    root = ET.fromstring(xml)
    geometry_payload = [
        {
            key: node.attrib[key]
            for key in ("name", "type", "size", "pos", "fromto", "quat")
            if key in node.attrib
        }
        for node in root.findall(".//geom")
    ]
    inertials = root.findall(".//inertial")
    masses = [float(node.attrib["mass"]) for node in inertials]
    inertia_triples = [
        tuple(float(value) for value in node.attrib["diaginertia"].split())
        for node in inertials
    ]
    actuator_nodes = root.findall(".//actuator/*")
    force_ranges = [
        tuple(float(value) for value in node.attrib["forcerange"].split())
        for node in actuator_nodes
        if "forcerange" in node.attrib
    ]
    frictions = [
        float(node.attrib["friction"].split()[0])
        for node in root.findall(".//geom")
        if "friction" in node.attrib
    ]
    bodies = {node.attrib.get("name"): node for node in root.findall(".//body")}
    com_residuals = []
    mass_relative_residuals = []
    for link in blueprint.links:
        inertial = bodies[link.name].find("inertial")
        if inertial is None:
            raise ValueError(f"missing inertial for {link.name}")
        actual = tuple(float(value) for value in inertial.attrib["pos"].split())
        actual_mass = float(inertial.attrib["mass"])
        expected_mass = link.mass * physical.mass_scales[link.name]
        mass_relative_residuals.append(
            _relative_residual(actual_mass, expected_mass)
        )
        expected = physical.com_offsets[link.name]
        base_scaled = tuple(
            value * physical.global_scale * physical.link_scales[link.name]
            for value in link.com
        )
        com_residuals.append(
            max(
                max(
                    abs(actual_value - expected_value)
                    for actual_value, expected_value in zip(actual, expected)
                ),
                max(
                    abs(offset - scaled)
                    for offset, scaled in zip(expected, base_scaled)
                ),
            )
        )
    coverage = _compact_coverage(
        blueprint.profile_metadata["motor_configuration"]["source_config_coverage"]
    )
    inertia_valid = all(
        all(math.isfinite(value) and value > 0.0 for value in triple)
        and max(triple) <= sum(triple) - max(triple) + 1e-12
        for triple in inertia_triples
    )
    finite_positive_masses = all(
        math.isfinite(value) and value > 0.0 for value in masses
    )
    finite_positive_friction = bool(frictions) and all(
        math.isfinite(value) and value > 0.0 for value in frictions
    )
    force_ranges_valid = len(force_ranges) == len(blueprint.actuators) and all(
        math.isfinite(low) and math.isfinite(high) and low < 0.0 < high
        for low, high in force_ranges
    )
    friction_relative_residual = max(
        _relative_residual(value, physical.friction) for value in frictions
    )
    actuator_nodes_by_name = {node.attrib["name"]: node for node in actuator_nodes}
    resolved_records = blueprint.profile_metadata["motor_configuration"][
        "resolved_anonymous_actuators"
    ]
    resolved_by_slot = {
        record["anonymous_semantic_slot"]: record["final_compiled"]
        for record in resolved_records
    }
    correlated = physical.metadata.get("task071_correlated_actuation", {})
    slot_composition = correlated.get("slot_composition", {})
    correlated_coverage_exact = set(slot_composition) == {
        joint.semantic_slot for joint in blueprint.joints
    }
    correlated_eligible = correlated.get("eligible") is True
    topology_coverage_exact = correlated.get("topology_coverage_exact") is True
    exact_mapping_not_claimed = (
        correlated.get("exact_physical_transmission_mapping_claimed") is False
    )
    factors_match_physical = correlated_coverage_exact and all(
        slot_composition[slot]["motor_strength"] == physical.motor_strength[slot]
        and slot_composition[slot]["kp_scale"] == physical.kp_scales[slot]
        and slot_composition[slot]["kd_scale"] == physical.kd_scales[slot]
        for slot in slot_composition
    )
    blueprint_config_residuals = []
    compiled_actuator_residuals = []
    for actuator in blueprint.actuators:
        resolved = resolved_by_slot[actuator.semantic_slot]
        blueprint_config_residuals.extend(
            (
                _relative_residual(actuator.kp, float(resolved["kp"])),
                _relative_residual(actuator.kd, float(resolved["kd"])),
                _relative_residual(
                    actuator.effort_limit,
                    float(resolved["effort_limit"]),
                ),
            )
        )
        node = actuator_nodes_by_name[actuator.name]
        low, high = (float(value) for value in node.attrib["forcerange"].split())
        expected_effort = (
            float(resolved["effort_limit"])
            * physical.motor_strength[actuator.semantic_slot]
        )
        expected_kp = float(resolved["kp"]) * physical.kp_scales[
            actuator.semantic_slot
        ]
        expected_kd = float(resolved["kd"]) * physical.kd_scales[
            actuator.semantic_slot
        ]
        compiled_actuator_residuals.extend(
            (
                _relative_residual(-low, expected_effort),
                _relative_residual(high, expected_effort),
                _relative_residual(float(node.attrib["kp"]), expected_kp),
                _relative_residual(float(node.attrib["kv"]), expected_kd),
            )
        )
    mass_compiler_coherent = max(mass_relative_residuals) <= 5e-6
    friction_compiler_coherent = friction_relative_residual <= 5e-6
    resolved_motor_config_matches_blueprint = max(blueprint_config_residuals) <= 1e-12
    compiled_actuator_config_coherent = max(compiled_actuator_residuals) <= 5e-6
    return {
        "seed": seed,
        "motor_count": len(blueprint.actuators),
        "xml_actuator_count": len(actuator_nodes),
        "link_count": len(blueprint.links),
        "inertial_count": len(inertials),
        "total_mass_kg": sum(masses),
        "min_link_mass_kg": min(masses),
        "finite_positive_masses": finite_positive_masses,
        "mass_compiler_max_relative_residual": max(mass_relative_residuals),
        "mass_compiler_coherent": mass_compiler_coherent,
        "principal_inertia_valid": inertia_valid,
        "finite_positive_friction": finite_positive_friction,
        "friction": physical.friction,
        "friction_compiler_max_relative_residual": friction_relative_residual,
        "friction_compiler_coherent": friction_compiler_coherent,
        "effort_abs_limit_min": min(
            max(abs(low), abs(high)) for low, high in force_ranges
        ),
        "effort_abs_limit_max": max(
            max(abs(low), abs(high)) for low, high in force_ranges
        ),
        "force_ranges_valid": force_ranges_valid,
        "resolved_motor_config_matches_blueprint": (
            resolved_motor_config_matches_blueprint
        ),
        "compiled_actuator_max_relative_residual": max(
            compiled_actuator_residuals
        ),
        "compiled_actuator_config_coherent": compiled_actuator_config_coherent,
        "physical_hash": physical_params_hash(physical),
        "xml_sha256": _sha256_bytes(xml.encode()),
        "compiled_geometry_sha256": _sha256_bytes(
            json.dumps(
                geometry_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ),
        "source_config_coverage": coverage,
        "com_max_scale_residual_m": max(com_residuals),
        "com_scale_coherent": max(com_residuals) <= 5e-10,
        "global_scale": physical.global_scale,
        "link_scales_hash": _mapping_hash(physical.link_scales),
        "mass_scales_hash": _mapping_hash(physical.mass_scales),
        "motor_strength_hash": _mapping_hash(physical.motor_strength),
        "kp_scales_hash": _mapping_hash(physical.kp_scales),
        "kd_scales_hash": _mapping_hash(physical.kd_scales),
        "delay_ms": physical.delay_ms,
        "correlated_metadata_coverage_exact": correlated_coverage_exact,
        "correlated_eligible": correlated_eligible,
        "topology_coverage_exact": topology_coverage_exact,
        "exact_mapping_not_claimed": exact_mapping_not_claimed,
        "correlated_factors_match_physical": factors_match_physical,
        "correlated_non_independent": correlated.get("independent_per_slot_noise") is False,
        "delay_runtime_owner": correlated.get("delay_runtime_owner"),
    }


def _physical_case(case: str) -> dict[str, object]:
    from h200_locomotion_lab.envs.whole_body_mujoco import _motor_process_baselines
    from h200_locomotion_lab.robots.archetype_morphology import (
        MotorDofPreservingArchetypePreviewGenerator,
    )
    from h200_locomotion_lab.robots.procedural_morphology import (
        compile_mjcf,
        physical_params_hash,
    )

    expected = EXPECTED_CASES[case]
    generator = MotorDofPreservingArchetypePreviewGenerator(reference_id=case)
    blueprint = generator.generate(str(expected["family"]), 0)
    sampled = []
    for seed, range_fraction in (
        (70000000, 0.0),
        (70000001, 1.0),
        (70000002, 1.0),
    ):
        physical = generator.sample_physical_params(
            blueprint,
            seed,
            range_fraction=range_fraction,
        )
        xml = compile_mjcf(blueprint, physical)
        sampled.append(
            {
                "physical": physical,
                "xml": xml,
                "summary": _sample_summary(blueprint, physical, xml, seed),
            }
        )
    nominal, full_a, full_b = sampled
    replay = generator.sample_physical_params(
        blueprint,
        70000001,
        range_fraction=1.0,
    )
    replay_xml = compile_mjcf(blueprint, replay)
    deterministic = (
        physical_params_hash(replay) == full_a["summary"]["physical_hash"]
        and _sha256_bytes(replay_xml.encode())
        == full_a["summary"]["xml_sha256"]
    )
    randomization_keys = (
        "global_scale",
        "link_scales_hash",
        "compiled_geometry_sha256",
        "mass_scales_hash",
        "friction",
        "motor_strength_hash",
        "kp_scales_hash",
        "kd_scales_hash",
        "delay_ms",
    )
    randomization = {
        key: _changed(
            nominal["summary"],
            full_a["summary"],
            full_b["summary"],
            key,
        )
        for key in randomization_keys
    }
    expected_count = int(expected["motor_count"])
    runtime_baselines = [
        _motor_process_baselines(blueprint, item["physical"], 50.0)
        for item in sampled
    ]
    runtime_strength_identity = all(
        all(value == 1.0 for value in baseline[0]) for baseline in runtime_baselines
    )
    runtime_delay_steps_exact = all(
        baseline[1]
        == (round(item["physical"].delay_ms * 50.0 / 1000.0),) * expected_count
        for baseline, item in zip(runtime_baselines, sampled)
    )
    coverage = nominal["summary"]["source_config_coverage"]
    coverage_exact = all(
        coverage[key] == expected_count
        for key in (
            "source_motor_count",
            "config_record_count",
            "usable_quantitative_prior_count",
            "declared_effort_count",
            "source_pd_gain_count",
            "source_armature_count",
            "source_control_mode_count",
        )
    ) and coverage["rejected_placeholder_count"] == 0
    static = deterministic and coverage_exact and all(
        summary["motor_count"] == expected_count
        and summary["xml_actuator_count"] == expected_count
        and summary["link_count"] == summary["inertial_count"]
        and summary["finite_positive_masses"]
        and summary["mass_compiler_coherent"]
        and summary["principal_inertia_valid"]
        and summary["finite_positive_friction"]
        and summary["friction_compiler_coherent"]
        and summary["force_ranges_valid"]
        and summary["resolved_motor_config_matches_blueprint"]
        and summary["compiled_actuator_config_coherent"]
        for summary in (item["summary"] for item in sampled)
    )
    physical_dimensions = (
        "global_scale",
        "link_scales_hash",
        "compiled_geometry_sha256",
        "mass_scales_hash",
        "friction",
    )
    control_dimensions = (
        "motor_strength_hash",
        "kp_scales_hash",
        "kd_scales_hash",
        "delay_ms",
    )
    geometry_mass_friction_exercised = all(
        all(randomization[key].values()) for key in physical_dimensions
    )
    control_randomization_required = bool(
        blueprint.profile_metadata["motor_configuration"][
            "control_gain_randomization_required_for_training_distribution"
        ]
    )
    control_randomization_exercised = all(
        all(randomization[key].values()) for key in control_dimensions
    )
    com_coherent = all(item["summary"]["com_scale_coherent"] for item in sampled)
    training_admission = (
        static
        and geometry_mass_friction_exercised
        and com_coherent
        and all(
            summary["correlated_metadata_coverage_exact"]
            and summary["correlated_eligible"]
            and summary["topology_coverage_exact"]
            and summary["exact_mapping_not_claimed"]
            and summary["correlated_factors_match_physical"]
            and summary["correlated_non_independent"]
            and summary["delay_runtime_owner"] == "WholeBodyMuJoCoShard→MotorProcess"
            for summary in (item["summary"] for item in sampled)
        )
        and runtime_strength_identity
        and runtime_delay_steps_exact
        and len({baseline[1][0] for baseline in runtime_baselines[1:]}) > 1
        and (not control_randomization_required or control_randomization_exercised)
    )
    failures = []
    if not static:
        failures.append("static_integrity_failed")
    if not geometry_mass_friction_exercised:
        failures.append("geometry_mass_friction_randomization_not_exercised")
    if not com_coherent:
        failures.append("com_does_not_scale_with_randomized_link_geometry")
    if control_randomization_required and not control_randomization_exercised:
        missing = [
            key
            for key in control_dimensions
            if not all(randomization[key].values())
        ]
        failures.append("required_control_randomization_missing:" + ",".join(missing))
    if not all(
        summary["correlated_metadata_coverage_exact"]
        and summary["correlated_eligible"]
        and summary["topology_coverage_exact"]
        and summary["exact_mapping_not_claimed"]
        and summary["correlated_factors_match_physical"]
        and summary["correlated_non_independent"]
        and summary["delay_runtime_owner"] == "WholeBodyMuJoCoShard→MotorProcess"
        for summary in (item["summary"] for item in sampled)
    ):
        failures.append("correlated_actuation_metadata_invalid")
    if not runtime_strength_identity or not runtime_delay_steps_exact:
        failures.append("runtime_baseline_contract_invalid")
    if len({baseline[1][0] for baseline in runtime_baselines[1:]}) <= 1:
        failures.append("runtime_delay_steps_not_exercised")
    metadata = blueprint.profile_metadata
    return {
        "case": case,
        "family": expected["family"],
        "profile_version": blueprint.profile_version,
        "contract_version": blueprint.contract_version,
        "contract_hash": blueprint.contract_hash,
        "structural_descriptor_sha256": metadata["structural_descriptor_sha256"],
        "source_sha256": metadata["source_sha256"],
        "expected_motor_count": expected_count,
        "source_motor_config_coverage": coverage,
        "source_motor_config_coverage_exact": coverage_exact,
        "samples": [item["summary"] for item in sampled],
        "deterministic_replay_passed": deterministic,
        "randomization": randomization,
        "geometry_mass_friction_randomization_exercised": (
            geometry_mass_friction_exercised
        ),
        "control_gain_randomization_required": control_randomization_required,
        "control_randomization_exercised": control_randomization_exercised,
        "com_scale_coherent_all_samples": com_coherent,
        "com_max_scale_residual_m": max(
            item["summary"]["com_max_scale_residual_m"] for item in sampled
        ),
        "static_integrity_passed": static,
        "r0_physical_stack_admission_passed": training_admission,
        "runtime_nominal_strength_identity": runtime_strength_identity,
        "runtime_delay_steps_exact": runtime_delay_steps_exact,
        "delay_steps_at_50hz": [baseline[1][0] for baseline in runtime_baselines],
        "failure_reasons": failures,
    }


def _task070_registry_evidence() -> dict[str, object]:
    records = {}
    for name, expected in TASK070_EVIDENCE.items():
        path = expected["path"]
        actual = _sha256_path(path)
        records[name] = {
            "path": _relative(path),
            "expected_sha256": expected["sha256"],
            "actual_sha256": actual,
            "matched": actual == expected["sha256"],
        }
    return records


def _registry() -> dict[str, object]:
    schema_hash, action_dim, observation_dim = _schema_identity()
    tiers = {
        "A": [
            {"id": "unitree_g1", "family": "biped"},
            {"id": "engineai_pm01", "family": "biped"},
            {"id": "spot_base", "family": "quadruped"},
            {"id": "unitree_go2", "family": "quadruped"},
            {"id": "deeprobotics_lite3", "family": "quadruped"},
        ],
        "B": [
            {"id": "unitree_g1_wheeled", "family": "wheeled_biped"},
            {"id": "engineai_pm01_wheeled", "family": "wheeled_biped"},
            {"id": "spot_base_wheeled", "family": "wheeled_quadruped"},
            {"id": "unitree_go2_wheeled", "family": "wheeled_quadruped"},
            {
                "id": "deeprobotics_lite3_wheeled",
                "family": "wheeled_quadruped",
            },
        ],
        "C": [
            {"id": case, "family": "biped", "status": "candidate_fail_closed"}
            for case in (
                "agibot_x1_serial",
                "agibot_x2_ultra",
                "engineai_t800",
                "engineai_t800pro",
                "limx_hu_d04",
                "booster_t1_23",
                "booster_t1_29",
                "robotera_star1",
            )
        ],
    }
    expected_denominator = {"A": 5, "B": 5, "C": 8}
    actual_denominator = {name: len(cases) for name, cases in tiers.items()}
    evidence = _task070_registry_evidence()
    overlay = TASK070 / "016-user-visual-acceptance.md"
    freeze_passed = (
        actual_denominator == expected_denominator
        and action_dim == 45
        and observation_dim == 193
        and all(record["matched"] for record in evidence.values())
    )
    return {
        "artifact": "task071_r0_training_case_registry",
        "runtime": _runtime(),
        "schema_version": "whole_body_v1_45",
        "schema_hash": schema_hash,
        "action_dim": action_dim,
        "observation_dim": observation_dim,
        "tiers": tiers,
        "expected_denominator": expected_denominator,
        "actual_denominator": actual_denominator,
        "total_denominator": sum(actual_denominator.values()),
        "task070_evidence": evidence,
        "user_visual_acceptance_overlay": {
            "path": _relative(overlay),
            "sha256": _sha256_path(overlay),
            "user_visual_acceptance": True,
            "counts_toward_task070_v2_pass": False,
        },
        "task070_passed": False,
        "input_freeze_passed": freeze_passed,
        "clean_clone_portability": False,
        "portability_blocker": (
            "Task070 artifacts and external sources are ignored local inputs; "
            "this frame uses read-only symlinks"
        ),
    }


def _reset_stance_matrix() -> dict[str, object]:
    arena = json.loads(ARENA.read_text(encoding="utf-8"))
    offline_probe = _offline_mujoco_probe()
    expected = {
        ("unitree_g1", "biped"),
        ("unitree_go2", "quadruped"),
    }
    records = []
    for record in arena["records"]:
        identity = (record.get("reference_id"), record.get("family"))
        if identity not in expected:
            continue
        response = record.get("actuator_response") or {}
        stance = record.get("stance_hold") or {}
        records.append(
            {
                "reference_id": record.get("reference_id"),
                "family": record.get("family"),
                "compiled": record.get("compiled"),
                "accounting_exact": record.get("accounting_exact"),
                "reset_pose_passed": record.get("reset_pose_passed"),
                "actuator_count": response.get("actuator_count"),
                "all_actuators_responsive": response.get(
                    "all_actuators_responsive"
                ),
                "stance_hold_passed": record.get("stance_hold_passed"),
                "stance_steps": stance.get("steps"),
                "stance_duration_seconds": stance.get("duration_seconds"),
                "support_gate_passed": stance.get("support_gate_passed"),
                "finite": stance.get("finite"),
            }
        )
    identities = {(record["reference_id"], record["family"]) for record in records}
    if identities != expected or len(records) != 2:
        raise ValueError("Task070 arena evidence does not contain exact G1/Go2 denominator")
    stance_passed = sum(bool(record["stance_hold_passed"]) for record in records)
    dependency_available = bool(offline_probe["dependency_available"])
    dynamic_status = (
        "not_run_dependency_available"
        if dependency_available
        else "blocked_environment"
    )
    dynamic_failure_reason = (
        "the locked MuJoCo dependency is available, but no fresh dynamic rerun was executed"
        if dependency_available
        else (
            "the exact locked/offline dependency probe failed because the "
            "mujoco==3.12.0 CPython 3.11 wheel is absent from the shared cache"
        )
    )
    return {
        "artifact": "task071_r1_reset_stance_matrix",
        "runtime": _runtime(),
        "denominator": 2,
        "source": {
            "path": _relative(ARENA),
            "sha256": _sha256_path(ARENA),
            "classification": "Task070 prior evidence only",
        },
        "previous_compile_accounting_reset_actuator_stance": records,
        "previous_stance_passed": stance_passed,
        "previous_stance_denominator": 2,
        "fresh_dynamic_rerun": {
            "status": dynamic_status,
            "decisive_offline_dependency_probe": offline_probe,
            "failure_reason": dynamic_failure_reason,
            "reused_task070_result_as_task071_rerun": False,
            "historical_online_attempts": {
                "count": 4,
                "latest_exact_command": (
                    "UV_CACHE_DIR=/home/admin1/workspace/store/cache/uv "
                    "uv run --isolated --locked --python 3.11 --extra mujoco "
                    "python -c 'import mujoco; print(mujoco.__version__)'"
                ),
                "latest_outcome": (
                    "failed after 4 retries in 127.7 seconds: operation timed out"
                ),
                "earlier_outcome": "three stalled/interrupted attempts",
                "transcript_retained": False,
                "used_as_reproducible_gate_evidence": False,
            },
        },
        "task071_r1_admission_passed": False,
        "failure_reasons": [
            f"fresh_dynamic_rerun_{dynamic_status}",
            f"prior_generic_stance_passed_{stance_passed}_of_2",
        ],
        "ppo_or_long_training_started": False,
    }


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    cases = [_physical_case(case) for case in EXPECTED_CASES]
    probe = {
        "artifact": "task071_r0_physical_attribute_probe",
        "runtime": _runtime(),
        "denominator": len(EXPECTED_CASES),
        "cases": cases,
        "static_integrity_passed_count": sum(
            case["static_integrity_passed"] for case in cases
        ),
        "static_integrity_passed": all(
            case["static_integrity_passed"] for case in cases
        ),
        "r0_physical_stack_admission_passed_count": sum(
            case["r0_physical_stack_admission_passed"] for case in cases
        ),
        "r0_physical_stack_admission_passed": all(
            case["r0_physical_stack_admission_passed"] for case in cases
        ),
        "next_gate_allowed": False,
        "ppo_or_long_training_started": False,
    }
    _write("r0_physical_attribute_probe.json", probe)
    _write("r0_training_case_registry.json", _registry())
    _write("r1_reset_stance_matrix.json", _reset_stance_matrix())
    print(
        "Task071 R0: "
        f"static={probe['static_integrity_passed_count']}/2, "
        f"physical_stack={probe['r0_physical_stack_admission_passed_count']}/2, "
        "fresh_dynamic=blocked_environment"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
