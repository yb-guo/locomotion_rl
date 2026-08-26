from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from h200_locomotion_lab.robots.procedural_morphology import (
    LOCOFORMER_MORPHOLOGY_CONTRACT_HASH,
    LOCOFORMER_MORPHOLOGY_CONTRACT_VERSION,
    LOCOFORMER_MORPHOLOGY_PROFILE_VERSION,
    LocoFormerMorphologyGenerator,
    LocoFormerMorphologyGeneratorConfig,
    compile_mjcf,
    compile_with_mujoco,
    morphology_blueprint_hash,
    morphology_instance_key,
)
from h200_locomotion_lab.robots.whole_body_adapter import BoundEmbodiment
from h200_locomotion_lab.tools.task069_morphology_verification import (
    _finite_physics_checks,
    _prepare_default_pose,
    run_morphology_matrix,
    verify_legacy_baseline,
)


def test_task069_profile_is_explicit_and_deterministic() -> None:
    generator = LocoFormerMorphologyGenerator()
    for family in ("biped", "quadruped", "wheeled_biped", "wheeled_quadruped"):
        first = generator.generate(family, 7)
        second = generator.generate(family, 7)
        assert first.profile_version == LOCOFORMER_MORPHOLOGY_PROFILE_VERSION
        assert first.contract_version == LOCOFORMER_MORPHOLOGY_CONTRACT_VERSION
        assert first.contract_hash == LOCOFORMER_MORPHOLOGY_CONTRACT_HASH
        assert first.manifest() == second.manifest()
        assert compile_with_mujoco(compile_mjcf(first)).nu == len(first.actuators)


def test_task069_wheel_topology_is_dynamical_and_slot_mapped() -> None:
    generator = LocoFormerMorphologyGenerator()
    for family, expected_wheels in (("wheeled_biped", 2), ("wheeled_quadruped", 4)):
        blueprint = generator.generate(family, 3)
        assert len(blueprint.wheel_specs) == expected_wheels
        assert {wheel.semantic_slot for wheel in blueprint.wheel_specs} == {
            f"limb{index}_wheel" for index in range(expected_wheels)
        }
        mapping = BoundEmbodiment.from_blueprint(blueprint).mapping
        assert mapping.active_count == len(blueprint.joints)
        assert mapping.round_trip(tuple(range(len(blueprint.joints)))) == tuple(
            float(index) for index in range(len(blueprint.joints))
        )

        root = ET.fromstring(compile_mjcf(blueprint))
        joints = {str(item.get("name")): item for item in root.iter("joint")}
        geoms = {str(item.get("name")): item for item in root.iter("geom")}
        actuators = {str(item.get("name")): item for item in root.find("actuator") or ()}
        for wheel in blueprint.wheel_specs:
            assert joints[wheel.joint_name].get("limited") == "false"
            assert joints[wheel.joint_name].get("axis") == "0 1 0"
            assert geoms[f"{wheel.link_name}_geom"].get("type") == "cylinder"
            assert geoms[f"{wheel.link_name}_geom"].get("contype") == "1"
            assert geoms[f"{wheel.link_name}_geom"].get("conaffinity") == "1"
            assert actuators[f"{wheel.joint_name}_actuator"].tag == "motor"


def test_task069_non_wheeled_families_have_no_wheel_slots() -> None:
    generator = LocoFormerMorphologyGenerator()
    for family in ("biped", "quadruped"):
        blueprint = generator.generate(family, 4)
        assert blueprint.wheel_specs == ()
        assert not any(slot.endswith("_wheel") for slot in blueprint.active_slots)


def test_task069_biped_grammar_exposes_ankle_mirror_and_arm_witnesses() -> None:
    generator = LocoFormerMorphologyGenerator()
    no_ankle = next(
        generator.generate("biped", seed)
        for seed in range(64)
        if not any(joint.semantic_slot == "limb0_ankle_pitch" for joint in generator.generate("biped", seed).joints)
    )
    with_ankle = next(
        generator.generate("biped", seed)
        for seed in range(64)
        if any(joint.semantic_slot == "limb0_ankle_pitch" for joint in generator.generate("biped", seed).joints)
    )
    with_arms = next(generator.generate("biped", seed) for seed in range(64) if generator.generate("biped", seed).has_arms)
    assert not any(joint.semantic_slot.startswith("limb0_ankle_") for joint in no_ankle.joints)
    assert any(joint.semantic_slot.startswith("limb0_ankle_") for joint in with_ankle.joints)
    assert with_arms.has_arms

    mirrored = LocoFormerMorphologyGenerator(
        LocoFormerMorphologyGeneratorConfig(
            require_biped_ankle=True,
            mirror_biped_legs=True,
            arm_probability=0.0,
            waist_max_joints=0,
        )
    ).generate("biped", 0)
    left = [joint for joint in mirrored.joints if joint.semantic_slot.startswith("limb0_")]
    right = [joint for joint in mirrored.joints if joint.semantic_slot.startswith("limb1_")]
    assert [joint.semantic_slot.split("_", 1)[1] for joint in left] == [
        joint.semantic_slot.split("_", 1)[1] for joint in right
    ]


def test_task069_biped_arm_attachment_has_trunk_clearance() -> None:
    generator = LocoFormerMorphologyGenerator()
    for seed in range(32):
        blueprint = generator.generate("biped", seed)
        if not blueprint.has_arms:
            continue
        trunk_half_width = blueprint.links[0].size[1]
        first_arm_links = [
            link
            for link in blueprint.links
            if link.parent == "trunk" and "_arm_" in link.name
        ]
        assert first_arm_links
        assert all(
            abs(link.pos[1]) > trunk_half_width + link.size[0]
            for link in first_arm_links
        )


def test_task069_reset_and_finite_gates_exclude_world_body() -> None:
    generator = LocoFormerMorphologyGenerator()
    for family in ("biped", "quadruped", "wheeled_biped", "wheeled_quadruped"):
        blueprint = generator.generate(family, 0)
        physical = generator.sample_physical_params(blueprint, 20_000_000, range_fraction=0.5)
        import mujoco

        model = mujoco.MjModel.from_xml_string(compile_mjcf(blueprint, physical))
        _, reset_pose = _prepare_default_pose(model, blueprint, physical, mujoco)
        assert reset_pose["reset_pose_passed"]
        assert reset_pose["initial_self_contact_count"] == 0
        assert _finite_physics_checks(model, blueprint, mujoco)["finite"]


def test_task069_structural_identity_is_separate_from_continuous_geometry() -> None:
    config = LocoFormerMorphologyGeneratorConfig(
        arm_probability=0.0,
        waist_max_joints=0,
        require_biped_ankle=True,
        mirror_biped_legs=True,
        biped_extra_hip_yaw_probability=0.0,
    )
    generator = LocoFormerMorphologyGenerator(config)
    first = generator.generate("biped", 0)
    second = generator.generate("biped", 1)
    assert first.structural_hash == second.structural_hash
    assert morphology_blueprint_hash(first) != morphology_blueprint_hash(second)
    assert morphology_instance_key(first) != morphology_instance_key(second)
    assert morphology_instance_key(first).embodiment_contract_version == LOCOFORMER_MORPHOLOGY_CONTRACT_VERSION


def test_task069_legacy_baseline_is_present_and_stable() -> None:
    baseline = Path(".agent/task/task069-locoformer-paper-faithful-morphology/artifacts/r0_legacy_v2_baseline.json")
    source = Path(".agent/task/task069-locoformer-paper-faithful-morphology/artifacts/r0_source_contract.json")
    assert baseline.exists()
    assert source.exists()
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert payload["expected_denominator"] == 128
    assert len(payload["records"]) == 128
    assert all(record["status"] == "passed" for record in payload["records"])
    assert verify_legacy_baseline(baseline)["passed"]


def test_task069_small_matrix_preserves_expected_denominator(tmp_path: Path) -> None:
    payload = run_morphology_matrix(
        tmp_path / "matrix.json",
        seeds=range(2),
        smoke_steps=3,
        render=False,
    )
    assert payload["expected_denominator"] == 8
    assert all(item["built"] == 2 for item in payload["summary"].values())
    assert all(item["compiled"] == 2 for item in payload["summary"].values())
    assert all(item["finite_physics"] == 2 for item in payload["summary"].values())
    assert all(item["reset_pose"] == 2 for item in payload["summary"].values())
