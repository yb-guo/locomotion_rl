from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from h200_locomotion_lab.robots.archetype_morphology import (
    DISTANCE_BANDS,
    REGION_EXPECTED_PER_FAMILY,
    TASK070_ADDITIONAL_HUMANOID_REFERENCE_IDS,
    TASK070_DISTANCE_CONTRACT_HASH,
    TASK070_PRIOR_SET_ID,
    TASK070_R0_DESIGN_CONTRACT_SHA256,
    TASK070_REFERENCE_REGISTRY_SHA256,
    TASK070_SOURCE_LICENSE_MATRIX_SHA256,
    TASK070_STANCE_CONTRACT_HASH,
    ArchetypeConstrainedMorphologyGenerator,
    MotorDofPreservingArchetypePreviewGenerator,
    Task070ArchetypeConfig,
    load_additional_humanoid_motor_dof_preserving_descriptor,
    load_g1_motor_dof_preserving_descriptor,
    load_pm01_motor_dof_preserving_descriptor,
    load_quadruped_motor_dof_preserving_descriptor,
)
from h200_locomotion_lab.robots.procedural_morphology import (
    ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_HASH,
    ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_VERSION,
    ARCHETYPE_CONSTRAINED_MORPHOLOGY_PROFILE_VERSION,
    MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_HASH,
    MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_VERSION,
    MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_PROFILE_VERSION,
    LocoFormerMorphologyGenerator,
    MorphologyGenerator,
    compile_mjcf,
    compile_with_mujoco,
    morphology_blueprint_hash,
    morphology_instance_key,
    physical_params_hash,
    read_canonical_root_state,
)
from h200_locomotion_lab.robots.whole_body_slots import WHOLE_BODY_SLOT_NAMES
from h200_locomotion_lab.tools.task070_morphology_verification import (
    TASK070_V2_CURRENT_ARENA_CASES,
    VISUAL_REVIEW_STATUS,
    _final_matrix_gate,
    run_archetype_matrix,
    run_v2_arena_smoke,
    verify_r0_compatibility_baseline,
    write_final_verification,
)

TASK_ROOT = Path(".agent/task/task070-archetype-constrained-standable-morphology")
ARTIFACT_ROOT = TASK_ROOT / "artifacts"
FAMILIES = ("biped", "quadruped", "wheeled_biped", "wheeled_quadruped")


def _load_artifact(name: str) -> dict[str, object]:
    return json.loads((ARTIFACT_ROOT / name).read_text(encoding="utf-8"))


def test_task070_profile_identity_is_explicit_deterministic_and_compilable() -> None:
    generator = ArchetypeConstrainedMorphologyGenerator()
    for family in FAMILIES:
        first = generator.generate(family, 7)
        second = generator.generate(family, 7)
        metadata = first.profile_metadata
        assert first.profile_version == ARCHETYPE_CONSTRAINED_MORPHOLOGY_PROFILE_VERSION
        assert first.contract_version == ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_VERSION
        assert first.contract_hash == ARCHETYPE_CONSTRAINED_MORPHOLOGY_CONTRACT_HASH
        assert first.manifest() == second.manifest()
        assert metadata["reference_registry_sha256"] == TASK070_REFERENCE_REGISTRY_SHA256
        assert metadata["source_license_matrix_sha256"] == TASK070_SOURCE_LICENSE_MATRIX_SHA256
        assert metadata["r0_design_contract_sha256"] == TASK070_R0_DESIGN_CONTRACT_SHA256
        assert metadata["prior_set_id"] == TASK070_PRIOR_SET_ID
        assert metadata["distance_contract_hash"] == TASK070_DISTANCE_CONTRACT_HASH
        assert metadata["stance_contract_hash"] == TASK070_STANCE_CONTRACT_HASH
        assert metadata["primitive_geometry_only"] is True
        assert compile_with_mujoco(compile_mjcf(first)).nu == len(first.actuators)


def test_task070_v2_g1_preview_preserves_29_motor_dofs_and_compiles() -> None:
    generator = MotorDofPreservingArchetypePreviewGenerator()
    blueprint = generator.generate("biped", 0)
    xml = compile_mjcf(blueprint, generator.sample_physical_params(blueprint, 70_000_000))
    metadata = blueprint.profile_metadata
    accounting = metadata["motor_accounting"]
    assert blueprint.profile_version == MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_PROFILE_VERSION
    assert blueprint.contract_version == MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_VERSION
    assert blueprint.contract_hash == MOTOR_DOF_PRESERVING_ARCHETYPE_MORPHOLOGY_CONTRACT_HASH
    assert accounting["source_actuated_motor_count"] == 29
    assert accounting["anonymous_motor_count"] == 29
    assert accounting["total_actuator_count"] == 29
    assert accounting["bijection_passed"] is True
    assert len(metadata["source_to_anonymous_motor_bijection"]) == 29
    assert len(blueprint.joints) == len(blueprint.actuators) == compile_with_mujoco(xml).nu == 29
    assert blueprint.has_arms is True
    assert "mesh" not in xml
    assert "left_arm_wrist_yaw" in blueprint.active_slots
    assert "right_arm_wrist_yaw" in blueprint.active_slots
    assert "waist_pitch" in blueprint.active_slots
    descriptor = metadata["source_tree_descriptor"]
    assert descriptor["source_motor_count"] == 29
    assert descriptor["anonymous_motor_count"] == 29
    assert metadata["joint_marker_sites"] == "source_motor_origins"
    audit_pose = metadata["visual_audit_nominal_joint_pose"]
    assert audit_pose["left_arm_elbow_pitch"] == 0.75
    assert audit_pose["right_arm_elbow_pitch"] == 0.75


def test_task070_t1_visual_audit_pose_breaks_arm_chain_collinearity() -> None:
    for reference_id, expected_count in (("booster_t1_23", 23), ("booster_t1_29", 29)):
        blueprint = MotorDofPreservingArchetypePreviewGenerator(
            reference_id=reference_id
        ).generate("biped", 0)
        assert len(blueprint.joints) == len(blueprint.actuators) == expected_count
        pose = blueprint.profile_metadata["visual_audit_nominal_joint_pose"]
        assert pose["left_arm_shoulder_roll"] == pytest.approx(-0.42)
        assert pose["right_arm_shoulder_roll"] == pytest.approx(0.42)
        assert pose["left_arm_elbow_pitch"] == pytest.approx(0.62)
        assert pose["right_arm_elbow_pitch"] == pytest.approx(0.62)
        assert pose["left_arm_elbow_yaw"] == pytest.approx(0.58)
        assert pose["right_arm_elbow_yaw"] == pytest.approx(-0.58)

    # The T1-only audit pose must not alter other candidate nominal conventions.
    x1 = MotorDofPreservingArchetypePreviewGenerator(
        reference_id="agibot_x1_serial"
    ).generate("biped", 0)
    x1_pose = x1.profile_metadata["visual_audit_nominal_joint_pose"]
    assert x1_pose["left_arm_shoulder_roll"] == -0.02
    assert x1_pose["right_arm_shoulder_roll"] == -0.02


def test_task070_v2_g1_descriptor_preserves_source_body_tree_edges() -> None:
    descriptor = load_g1_motor_dof_preserving_descriptor()
    motors = descriptor.motors
    source_names = [motor.source_joint_name for motor in motors]
    slots = [motor.anonymous_semantic_slot for motor in motors]
    body_nodes = {node.source_body_name: node for node in descriptor.body_tree}
    source_edges = {
        (node.source_parent_body, node.source_body_name)
        for node in descriptor.body_tree
        if node.source_parent_body != "root"
    }
    assert len(motors) == len(set(source_names)) == len(set(slots)) == 29
    assert descriptor.module_dof_counts == {
        "left_leg": 6,
        "right_leg": 6,
        "waist": 3,
        "left_arm": 7,
        "right_arm": 7,
    }
    assert {
        "left_arm_wrist_roll",
        "left_arm_wrist_pitch",
        "left_arm_wrist_yaw",
        "right_arm_wrist_roll",
        "right_arm_wrist_pitch",
        "right_arm_wrist_yaw",
        "waist_yaw",
        "waist_roll",
        "waist_pitch",
    } <= set(slots)
    for motor in motors:
        assert motor.module != "other"
        assert motor.source_joint_type == "hinge"
        assert motor.source_parent_body in body_nodes
        assert motor.source_child_body in body_nodes
        assert (motor.source_parent_body, motor.source_child_body) in source_edges
        assert motor.source_body_local_pos == body_nodes[motor.source_child_body].source_body_local_pos
        assert (
            motor.source_body_local_quat
            == body_nodes[motor.source_child_body].source_body_local_quat
        )
        assert len(motor.source_body_local_pos) == 3
        assert len(motor.source_body_local_quat) == 4
        assert len(motor.source_joint_local_pos) == 3
        assert len(motor.joint_range) == 2
        assert motor.joint_range[0] < motor.joint_range[1]
        assert math.isclose(
            sum(value * value for value in motor.normalized_local_axis),
            1.0,
            rel_tol=1e-12,
        )
        assert math.isclose(
            sum(value * value for value in motor.source_body_local_quat),
            1.0,
            rel_tol=1e-12,
        )
        assert motor.anonymous_child_link.startswith("anon_")
        assert motor.anonymous_parent_link.startswith("anon_")

    by_slot = {motor.anonymous_semantic_slot: motor for motor in motors}
    assert by_slot["left_arm_wrist_yaw"].source_parent_body == "left_wrist_pitch_link"
    assert by_slot["right_arm_wrist_yaw"].source_parent_body == "right_wrist_pitch_link"
    assert by_slot["limb0_knee_pitch"].source_parent_body == "left_hip_yaw_link"
    assert by_slot["limb1_knee_pitch"].source_parent_body == "right_hip_yaw_link"
    assert by_slot["left_arm_shoulder_pitch"].source_body_local_quat != (
        1.0,
        0.0,
        0.0,
        0.0,
    )


def test_task070_v2_g1_preview_edges_are_descriptor_driven() -> None:
    descriptor = load_g1_motor_dof_preserving_descriptor()
    generator = MotorDofPreservingArchetypePreviewGenerator()
    blueprint = generator.generate("biped", 0)
    anonymous_edges = {
        (joint.parent_link, joint.child_link, joint.semantic_slot)
        for joint in blueprint.joints
    }
    assert len(blueprint.joints) == len(descriptor.motors) == 29
    for motor in descriptor.motors:
        assert (
            motor.anonymous_parent_link,
            motor.anonymous_child_link,
            motor.anonymous_semantic_slot,
        ) in anonymous_edges
    realized = blueprint.profile_metadata["geometry_randomization"][
        "realized_source_tree_positions"
    ]
    assert len(realized) == 29
    assert all("source_body_local_pos" in item for item in realized)
    xml = compile_mjcf(blueprint, generator.sample_physical_params(blueprint, 70_000_000))
    root = ET.fromstring(xml)
    geoms = {str(item.get("name")): item for item in root.iter("geom")}
    bodies = {str(item.get("name")): item for item in root.iter("body")}
    wrist_geom = geoms["anon_left_arm_wrist_roll_link_geom"]
    wrist_fromto = tuple(float(value) for value in wrist_geom.get("fromto", "").split())
    assert len(wrist_fromto) == 6
    assert abs(wrist_fromto[3] - wrist_fromto[0]) > 0.04
    assert wrist_geom.get("rgba") is not None
    assert geoms["anon_pelvis_core_geom"].get("rgba") is not None
    assert "anon_pelvis_core_limb0_hip_pitch_attachment_visual" in geoms
    assert "anon_pelvis_core_limb1_hip_pitch_attachment_visual" in geoms
    assert "anon_waist_pitch_link_left_arm_shoulder_pitch_attachment_visual" in geoms
    assert "anon_waist_pitch_link_right_arm_shoulder_pitch_attachment_visual" in geoms
    ankle_geom = geoms["anon_limb0_ankle_pitch_link_geom"]
    ankle_fromto = tuple(float(value) for value in ankle_geom.get("fromto", "").split())
    ankle_roll_pos = tuple(
        float(value)
        for value in bodies["anon_limb0_ankle_roll_link"].get("pos", "").split()
    )
    assert ankle_fromto[3:] == ankle_roll_pos
    shoulder_quat = tuple(
        float(value)
        for value in bodies["anon_left_arm_shoulder_pitch_link"].get("quat", "").split()
    )
    assert all(
        math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9)
        for actual, expected in zip(
            shoulder_quat,
            descriptor.motors[15].source_body_local_quat,
            strict=True,
        )
    )
    torso_geom = geoms["anon_waist_pitch_link_geom"]
    torso_pos_z = float(torso_geom.get("pos", "0 0 0").split()[2])
    torso_half_height = float(torso_geom.get("size", "").split()[2])
    assert torso_pos_z > torso_half_height
    pelvis_geom = geoms["anon_pelvis_core_geom"]
    pelvis_pos_z = float(pelvis_geom.get("pos", "0 0 0").split()[2])
    pelvis_half_height = float(pelvis_geom.get("size", "").split()[2])
    assert pelvis_pos_z + pelvis_half_height <= 1e-9
    footpad = geoms["anon_limb0_ankle_roll_link_footpad"]
    assert float(footpad.get("pos", "").split()[0]) > 0.0


def test_task070_v2_other_source_descriptors_preserve_motor_order_and_semantics() -> None:
    pm01 = load_pm01_motor_dof_preserving_descriptor()
    assert len(pm01.motors) == 23
    pm01_by_source = {motor.source_joint_name: motor for motor in pm01.motors}
    assert pm01_by_source["J16_ELBOW_PITCH_L"].anonymous_semantic_slot == (
        "left_arm_elbow_pitch"
    )
    assert pm01_by_source["J17_ELBOW_YAW_L"].anonymous_semantic_slot == (
        "left_arm_wrist_yaw"
    )
    assert pm01_by_source["J21_ELBOW_PITCH_R"].anonymous_semantic_slot == (
        "right_arm_elbow_pitch"
    )
    assert pm01_by_source["J22_ELBOW_YAW_R"].anonymous_semantic_slot == (
        "right_arm_wrist_yaw"
    )
    assert len({motor.anonymous_semantic_slot for motor in pm01.motors}) == 23

    expected_modules = {
        "front_left_leg": 3,
        "front_right_leg": 3,
        "rear_left_leg": 3,
        "rear_right_leg": 3,
    }
    for reference_id in ("spot_base", "unitree_go2", "deeprobotics_lite3"):
        descriptor = load_quadruped_motor_dof_preserving_descriptor(reference_id)
        slots = tuple(motor.anonymous_semantic_slot for motor in descriptor.motors)
        assert len(descriptor.motors) == len(set(slots)) == 12
        assert descriptor.module_dof_counts == expected_modules
        assert slots == tuple(
            f"limb{limb}_{joint}"
            for limb in range(4)
            for joint in ("hip_roll", "hip_pitch", "knee_pitch")
        )
        terminal_offsets = descriptor.source_to_anonymous_frame_transform[
            "terminal_local_offsets"
        ]
        assert len(terminal_offsets) == 4
        assert all(
            motor.source_joint_type in {"hinge", "revolute"}
            for motor in descriptor.motors
        )
        assert all(
            math.isclose(
                sum(value * value for value in motor.normalized_local_axis),
                1.0,
                rel_tol=1e-12,
            )
            for motor in descriptor.motors
        )


def test_task070_v2_g1_motor_config_preserves_real_source_classes_as_hints() -> None:
    descriptor = load_g1_motor_dof_preserving_descriptor()
    by_slot = {motor.anonymous_semantic_slot: motor for motor in descriptor.motors}
    expected = {
        "limb0_hip_pitch": ("7520_14", 88.0, 32.0),
        "limb0_hip_roll": ("7520_22", 139.0, 20.0),
        "limb0_ankle_pitch": ("parallel_5020_x2", 50.0, 37.0),
        "left_arm_shoulder_pitch": ("5020", 25.0, 37.0),
        "left_arm_wrist_pitch": ("4010", 5.0, 22.0),
    }
    for slot, (source_class, effort, velocity) in expected.items():
        config = by_slot[slot].source_motor_config
        assert config is not None
        assert config.source_motor_class == source_class
        assert config.declared_effort_limit == effort
        assert config.declared_velocity_limit == velocity
        assert config.stiffness is not None
        assert config.damping is not None
        assert config.armature is not None
        assert config.rotor_inertias is not None
        assert config.gear_ratios is not None
        assert config.control_mode == "builtin_position_pd"
        assert config.usable_as_quantitative_prior is True
        assert len(config.source_sha256) == 64

    generator = MotorDofPreservingArchetypePreviewGenerator()
    blueprint = generator.generate("biped", 0)
    actuators = {actuator.semantic_slot: actuator for actuator in blueprint.actuators}
    assert actuators["limb0_hip_roll"].effort_limit > actuators["limb0_hip_pitch"].effort_limit
    assert actuators["limb0_hip_pitch"].effort_limit > actuators["limb0_ankle_pitch"].effort_limit
    assert actuators["left_arm_shoulder_pitch"].effort_limit > actuators[
        "left_arm_wrist_pitch"
    ].effort_limit
    config_manifest = blueprint.profile_metadata["motor_configuration"]
    assert config_manifest["exact_named_robot_parameter_parity_claimed"] is False
    assert config_manifest["source_config_coverage"]["usable_quantitative_prior_count"] == 29
    assert config_manifest["source_config_coverage"]["source_rotor_inertia_count"] == 29
    assert config_manifest["source_config_coverage"]["source_gear_ratio_count"] == 29
    resolved = {
        item["anonymous_semantic_slot"]: item
        for item in config_manifest["resolved_anonymous_actuators"]
    }
    assert resolved["limb0_hip_roll"]["source_hint_used"] is True
    assert resolved["limb0_hip_roll"]["velocity_limit_runtime_enforced"] is False
    assert not math.isclose(
        resolved["limb0_hip_roll"]["final_compiled"]["effort_limit"],
        by_slot["limb0_hip_roll"].source_motor_config.declared_effort_limit,
    )
    xml = compile_mjcf(blueprint, generator.sample_physical_params(blueprint, 70_000_000))
    xml_actuators = {
        str(item.get("name")): item for item in ET.fromstring(xml).find("actuator") or ()
    }
    knee_actuator = actuators["limb0_knee_pitch"]
    force_range = tuple(
        float(value)
        for value in xml_actuators[knee_actuator.name].get("forcerange", "").split()
    )
    assert math.isclose(force_range[1], knee_actuator.effort_limit, rel_tol=1e-6)


def test_task070_v2_other_motor_configs_keep_classes_and_reject_spot_placeholders() -> None:
    pm01 = load_pm01_motor_dof_preserving_descriptor()
    pm01_by_slot = {motor.anonymous_semantic_slot: motor for motor in pm01.motors}
    assert pm01_by_slot["limb0_hip_pitch"].source_motor_config.declared_effort_limit == 164.0
    assert pm01_by_slot["limb0_hip_yaw"].source_motor_config.declared_effort_limit == 52.0
    assert pm01_by_slot["limb0_hip_pitch"].source_motor_config.declared_velocity_limit == 26.3
    assert pm01_by_slot["limb0_hip_yaw"].source_motor_config.declared_velocity_limit == 35.2

    go2 = load_quadruped_motor_dof_preserving_descriptor("unitree_go2")
    go2_by_slot = {motor.anonymous_semantic_slot: motor for motor in go2.motors}
    assert go2_by_slot["limb0_hip_pitch"].source_motor_config.stiffness == 20.0
    assert go2_by_slot["limb0_hip_pitch"].source_motor_config.declared_effort_limit == 23.5
    assert go2_by_slot["limb0_knee_pitch"].source_motor_config.stiffness == 40.0
    assert go2_by_slot["limb0_knee_pitch"].source_motor_config.declared_effort_limit == 45.0

    lite3 = load_quadruped_motor_dof_preserving_descriptor("deeprobotics_lite3")
    lite3_by_slot = {motor.anonymous_semantic_slot: motor for motor in lite3.motors}
    assert lite3_by_slot["limb0_hip_pitch"].source_motor_config.declared_effort_limit == 24.0
    assert lite3_by_slot["limb0_knee_pitch"].source_motor_config.declared_effort_limit == 36.0
    assert lite3_by_slot["limb0_hip_pitch"].source_motor_config.declared_velocity_limit == 26.2
    assert lite3_by_slot["limb0_knee_pitch"].source_motor_config.declared_velocity_limit == 17.3

    spot = load_quadruped_motor_dof_preserving_descriptor("spot_base")
    assert all(motor.source_motor_config.declared_effort_limit == 1000.0 for motor in spot.motors)
    assert all(motor.source_motor_config.usable_as_quantitative_prior is False for motor in spot.motors)
    for reference_id in ("unitree_go2", "deeprobotics_lite3"):
        generator = MotorDofPreservingArchetypePreviewGenerator(reference_id=reference_id)
        blueprint = generator.generate("quadruped", 0)
        actuators = {actuator.semantic_slot: actuator for actuator in blueprint.actuators}
        assert actuators["limb0_knee_pitch"].kp > actuators["limb0_hip_pitch"].kp
        assert actuators["limb0_knee_pitch"].effort_limit > actuators[
            "limb0_hip_pitch"
        ].effort_limit
    spot_blueprint = MotorDofPreservingArchetypePreviewGenerator(
        reference_id="spot_base"
    ).generate("quadruped", 0)
    spot_config = spot_blueprint.profile_metadata["motor_configuration"]
    assert spot_config["source_config_coverage"]["rejected_placeholder_count"] == 12
    assert all(
        record["source_hint_used"] is False
        for record in spot_config["resolved_anonymous_actuators"]
    )


def test_task070_v2_wheel_motor_config_is_explicitly_local_not_vendor_derived() -> None:
    generator = MotorDofPreservingArchetypePreviewGenerator(reference_id="unitree_go2")
    blueprint = generator.generate("wheeled_quadruped", 0)
    config = blueprint.profile_metadata["motor_configuration"]
    assert config["local_wheel_motor_config_count"] == 4
    assert config["wheel_motor_config_provenance"] == "local_engineering_module"
    wheel_records = [
        record
        for record in config["resolved_anonymous_actuators"]
        if record["anonymous_semantic_slot"].endswith("_wheel")
    ]
    assert len(wheel_records) == 4
    assert all(record["source_hint_used"] is False for record in wheel_records)
    assert all(
        record["final_compiled"]["effort_limit"] == 45.0
        for record in wheel_records
    )


def test_task070_v2_actuation_stack_exactly_covers_current_source_centers() -> None:
    cases = (
        ("unitree_g1", "biped", 29),
        ("engineai_pm01", "biped", 23),
        ("spot_base", "quadruped", 12),
        ("unitree_go2", "quadruped", 12),
        ("deeprobotics_lite3", "quadruped", 12),
    )
    for reference_id, family, expected_count in cases:
        blueprint = MotorDofPreservingArchetypePreviewGenerator(
            reference_id=reference_id
        ).generate(family, 0)
        stack = blueprint.profile_metadata["actuation_stack"]
        transmission = stack["transmission_model"]
        covered = [
            slot
            for group in transmission["groups"]
            for slot in group["anonymous_semantic_slots"]
        ]
        expected_slots = [joint.semantic_slot for joint in blueprint.joints]
        assert covered == list(dict.fromkeys(covered))
        assert set(covered) == set(expected_slots)
        assert transmission["generalized_joint_coverage_exact"] is True
        assert transmission["generalized_joint_slot_count"] == expected_count
        assert transmission["modeled_physical_actuator_count"] == expected_count
        assert transmission["exact_physical_mapping_claimed"] is False
        coherent = stack["coherent_motor_config"]
        family_slots = {
            slot
            for motor_family in coherent["families"]
            for slot in motor_family["anonymous_semantic_slots"]
        }
        assert family_slots == set(expected_slots)
        assert coherent["independent_scalar_randomization_allowed"] is False
        runtime = stack["runtime_fault_process"]
        assert runtime["status"] == "declared_not_applied_in_task070_preview"
        assert runtime["current_process_coordinate"] == "generalized_joint_action_slot"
        assert runtime["current_supported_events"] == ["weak", "dead", "latency"]
        assert runtime["physical_parallel_motor_fault_projection"] == "not_implemented"


def test_task070_v2_g1_parallel_groups_preserve_29_physical_motor_accounting() -> None:
    blueprint = MotorDofPreservingArchetypePreviewGenerator().generate("biped", 0)
    groups = blueprint.profile_metadata["actuation_stack"]["transmission_model"][
        "groups"
    ]
    parallel = [
        group
        for group in groups
        if group["kind"] == "parallel_two_axis_two_motor_nominal_aggregate"
    ]
    assert len(parallel) == 3
    assert {tuple(group["anonymous_semantic_slots"]) for group in parallel} == {
        ("limb0_ankle_pitch", "limb0_ankle_roll"),
        ("limb1_ankle_pitch", "limb1_ankle_roll"),
        ("waist_roll", "waist_pitch"),
    }
    assert all(group["modeled_physical_actuator_count"] == 2 for group in parallel)
    assert all(group["exact_kinematic_mapping_available"] is False for group in parallel)
    assert all(group["mapping_usable_as_quantitative_prior"] is False for group in parallel)
    assert sum(group["modeled_physical_actuator_count"] for group in groups) == 29


def test_task070_v2_unspecified_transmissions_fail_closed() -> None:
    for reference_id, family in (
        ("spot_base", "quadruped"),
        ("deeprobotics_lite3", "quadruped"),
    ):
        blueprint = MotorDofPreservingArchetypePreviewGenerator(
            reference_id=reference_id
        ).generate(family, 0)
        groups = blueprint.profile_metadata["actuation_stack"]["transmission_model"][
            "groups"
        ]
        assert all(group["kind"] == "source_unspecified_joint_space_proxy" for group in groups)
        assert all(group["mapping_usable_as_quantitative_prior"] is False for group in groups)
        assert all(group["fail_closed_reason"] for group in groups)


def test_task070_v2_pm01_related_native_config_is_recorded_but_not_silently_merged() -> None:
    blueprint = MotorDofPreservingArchetypePreviewGenerator(
        reference_id="engineai_pm01"
    ).generate("biped", 0)
    stack = blueprint.profile_metadata["actuation_stack"]
    groups = stack["transmission_model"]["groups"]
    parallel = [
        group
        for group in groups
        if group["kind"] == "parallel_two_axis_two_motor_related_variant_mapping"
    ]
    assert len(parallel) == 2
    assert {tuple(group["anonymous_semantic_slots"]) for group in parallel} == {
        ("limb0_ankle_pitch", "limb0_ankle_roll"),
        ("limb1_ankle_pitch", "limb1_ankle_roll"),
    }
    assert sum(group["modeled_physical_actuator_count"] for group in groups) == 23
    related = stack["coherent_motor_config"]["related_official_config_evidence"]
    assert len(related) == 1
    assert related[0]["published_enabled_motor_count"] == 24
    assert related[0]["selected_descriptor_shared_named_motor_count"] == 23
    assert related[0]["extra_related_variant_motor"] == "J23_HEAD_YAW"
    assert related[0]["applied_to_resolved_anonymous_actuators"] is False
    assert all(len(value) == 64 for value in related[0]["sha256"])


def test_task070_v2_wheels_extend_the_actuation_stack_as_local_direct_motors() -> None:
    blueprint = MotorDofPreservingArchetypePreviewGenerator(
        reference_id="unitree_go2"
    ).generate("wheeled_quadruped", 0)
    stack = blueprint.profile_metadata["actuation_stack"]
    transmission = stack["transmission_model"]
    wheel_groups = [
        group for group in transmission["groups"] if group["kind"] == "continuous_wheel_direct"
    ]
    assert len(wheel_groups) == 4
    assert transmission["generalized_joint_slot_count"] == 16
    assert transmission["modeled_physical_actuator_count"] == 16
    assert transmission["local_direct_wheel_mapping_count"] == 4
    assert stack["structural_descriptor"]["added_local_wheel_slots"] == [
        "limb0_wheel",
        "limb1_wheel",
        "limb2_wheel",
        "limb3_wheel",
    ]
    wheel_families = [
        family
        for family in stack["coherent_motor_config"]["families"]
        if family["source_motor_class"] == "local_continuous_wheel"
    ]
    assert len(wheel_families) == 1
    assert blueprint.profile_metadata["motor_configuration"]["policy_action"] == (
        "mixed_joint_position_and_continuous_wheel_torque"
    )


def test_task070_v2_quadruped_preview_has_auditable_three_segment_limbs() -> None:
    for reference_id in ("spot_base", "unitree_go2", "deeprobotics_lite3"):
        generator = MotorDofPreservingArchetypePreviewGenerator(reference_id=reference_id)
        blueprint = generator.generate("quadruped", 0)
        physical = generator.sample_physical_params(blueprint, 70_000_000)
        xml = compile_mjcf(blueprint, physical)
        model = compile_with_mujoco(xml)
        metadata = blueprint.profile_metadata
        accounting = metadata["motor_accounting"]
        assert model.nu == len(blueprint.joints) == len(blueprint.actuators) == 12
        assert accounting["source_actuated_motor_count"] == 12
        assert accounting["anonymous_non_wheel_motor_count"] == 12
        assert accounting["added_wheel_motor_count"] == 0
        assert accounting["bijection_passed"] is True
        assert "mesh" not in xml
        root = ET.fromstring(xml)
        geoms = {str(item.get("name")): item for item in root.iter("geom")}
        bodies = {str(item.get("name")): item for item in root.iter("body")}
        for limb in range(4):
            hip_roll_link = f"anon_limb{limb}_hip_roll_link"
            hip_pitch_link = f"anon_limb{limb}_hip_pitch_link"
            knee_link = f"anon_limb{limb}_knee_pitch_link"
            assert bodies[hip_roll_link].get("pos") is not None
            assert bodies[hip_pitch_link].get("pos") is not None
            assert bodies[knee_link].get("pos") is not None
            hip_connector = tuple(
                float(value)
                for value in geoms[f"{hip_roll_link}_geom"].get("fromto", "").split()
            )
            hip_pitch_pos = tuple(
                float(value) for value in bodies[hip_pitch_link].get("pos", "").split()
            )
            upper_leg = tuple(
                float(value)
                for value in geoms[f"{hip_pitch_link}_geom"].get("fromto", "").split()
            )
            knee_pos = tuple(
                float(value) for value in bodies[knee_link].get("pos", "").split()
            )
            assert hip_connector[3:] == hip_pitch_pos
            assert upper_leg[3:] == knee_pos
            assert geoms[f"{knee_link}_footpad"].get("pos") is not None
        assert len(metadata["auxiliary_capsule_visuals"]["anon_trunk_core"]) == 4
        audit_pose = metadata["visual_audit_nominal_joint_pose"]
        assert all(abs(audit_pose[f"limb{limb}_hip_pitch"]) > 0.4 for limb in range(4))
        assert all(abs(audit_pose[f"limb{limb}_knee_pitch"]) > 0.8 for limb in range(4))


def test_task070_v2_terminal_wheels_append_without_replacing_source_motors() -> None:
    cases = (
        ("unitree_g1", "wheeled_biped", 29, 2),
        ("engineai_pm01", "wheeled_biped", 23, 2),
        ("spot_base", "wheeled_quadruped", 12, 4),
        ("unitree_go2", "wheeled_quadruped", 12, 4),
        ("deeprobotics_lite3", "wheeled_quadruped", 12, 4),
    )
    for reference_id, family, source_count, wheel_count in cases:
        generator = MotorDofPreservingArchetypePreviewGenerator(reference_id=reference_id)
        source_family = "biped" if family == "wheeled_biped" else "quadruped"
        source_blueprint = generator.generate(source_family, 0)
        blueprint = generator.generate(family, 0)
        physical = generator.sample_physical_params(blueprint, 70_000_000)
        xml = compile_mjcf(blueprint, physical)
        model = compile_with_mujoco(xml)
        accounting = blueprint.profile_metadata["motor_accounting"]
        assert blueprint.joints[:source_count] == source_blueprint.joints
        assert blueprint.actuators[:source_count] == source_blueprint.actuators
        assert len(blueprint.wheel_specs) == wheel_count
        assert len(blueprint.joints) == model.nu == source_count + wheel_count
        assert accounting["source_non_wheel_motor_count"] == source_count
        assert accounting["anonymous_non_wheel_motor_count"] == source_count
        assert accounting["added_wheel_motor_count"] == wheel_count
        assert accounting["total_actuator_count"] == source_count + wheel_count
        assert accounting["bijection_passed"] is True
        assert "footpad" not in xml
        root = ET.fromstring(xml)
        joint_elements = {str(item.get("name")): item for item in root.iter("joint")}
        for wheel, record in zip(
            blueprint.wheel_specs,
            blueprint.profile_metadata["terminal_wheel_composition"],
            strict=True,
        ):
            assert wheel.axis == (0.0, 1.0, 0.0)
            assert record["local_axis"] == (0.0, 1.0, 0.0)
            assert "local lateral basis" in record["axis_derivation"]
            assert joint_elements[wheel.joint_name].get("limited") == "false"


def test_task070_r0_artifacts_clear_multivendor_sources_and_heldout_guard() -> None:
    registry = _load_artifact("r0_reference_registry.json")
    matrix = _load_artifact("r0_source_license_matrix.json")
    compatibility = _load_artifact("r0_compatibility_baseline.json")
    reference_ids = {entry["reference_id"] for entry in registry["entries"]}  # type: ignore[index]
    assert {
        "unitree_g1",
        "engineai_pm01",
        "spot_base",
        "unitree_go2",
        "deeprobotics_lite3",
    } <= reference_ids
    assert matrix["minimum_prior_pool_passed"]["passed"] is True  # type: ignore[index]
    rows = {row["reference_id"]: row for row in matrix["rows"]}  # type: ignore[index]
    assert rows["spot_base"]["quantitative_prior_allowed"] is True
    assert "SDK Software" in " ".join(rows["spot_base"]["restrictions"])
    assert all(row["official_source"] for row in rows.values())
    assert compatibility["task067_artifact_guard"]["accepted_as_task070_stance_evidence"] is False  # type: ignore[index]


def test_task070_region_denominators_distance_bands_and_clone_guard_are_frozen() -> None:
    generator = ArchetypeConstrainedMorphologyGenerator()
    for family in FAMILIES:
        counts: Counter[str] = Counter()
        for seed in range(32):
            blueprint = generator.generate(family, seed)
            metadata = blueprint.profile_metadata
            region = str(metadata["sampling_region"])
            counts[region] += 1
            lower, upper = DISTANCE_BANDS[region]
            distance = float(metadata["nearest_prior_distance"])
            assert lower <= distance <= upper
            assert metadata["clone_guard"]["passed"] is True
            assert region == generator.expected_sampling_region(seed)
        assert dict(counts) == REGION_EXPECTED_PER_FAMILY


def test_task070_physical_identity_is_separate_from_topology_identity() -> None:
    generator = ArchetypeConstrainedMorphologyGenerator()
    blueprint = generator.generate("wheeled_quadruped", 4)
    first = generator.sample_physical_params(blueprint, 30_000_004, range_fraction=0.5)
    second = generator.sample_physical_params(blueprint, 30_000_005, range_fraction=0.5)
    assert blueprint.structural_hash == generator.generate("wheeled_quadruped", 4).structural_hash
    assert morphology_blueprint_hash(blueprint) == morphology_blueprint_hash(
        generator.generate("wheeled_quadruped", 4)
    )
    assert physical_params_hash(first) != physical_params_hash(second)
    assert morphology_instance_key(blueprint, first).cache_key != morphology_instance_key(
        blueprint,
        second,
    ).cache_key
    other = generator.generate("wheeled_quadruped", 5)
    same_physical_seed = generator.sample_physical_params(other, 30_000_004, range_fraction=0.5)
    assert first.metadata["task070_actuator_scaling"]["contract"] == (
        "task070_morphology_aware_sampled_mass_scale_lever_arm_scaling_v2"
    )
    assert (
        first.metadata["task070_actuator_scaling"]["total_sampled_mass_kg"]
        != second.metadata["task070_actuator_scaling"]["total_sampled_mass_kg"]
    )
    assert (
        first.metadata["task070_actuator_scaling"]["global_scale"]
        != second.metadata["task070_actuator_scaling"]["global_scale"]
    )
    shared_slots = set(first.motor_strength) & set(same_physical_seed.motor_strength)
    assert any(
        first.motor_strength[slot] != same_physical_seed.motor_strength[slot]
        or first.kp_scales[slot] != same_physical_seed.kp_scales[slot]
        or first.kd_scales[slot] != same_physical_seed.kd_scales[slot]
        for slot in shared_slots
    )


def test_task070_sampler_does_not_fail_open_after_retry_exhaustion() -> None:
    generator = ArchetypeConstrainedMorphologyGenerator()
    for family in FAMILIES:
        for seed in (4239, 7991, 8255, 9999):
            blueprint = generator.generate(family, seed)
            metadata = blueprint.profile_metadata
            region = str(metadata["sampling_region"])
            lower, upper = DISTANCE_BANDS[region]
            distance = float(metadata["nearest_prior_distance"])
            assert lower <= distance <= upper
            assert metadata["clone_guard"]["passed"] is True
            assert any(item["region_band_passed"] for item in metadata["retry_trace"])


def test_task070_biped_feet_and_wheeled_biped_attachments_are_bounded() -> None:
    generator = ArchetypeConstrainedMorphologyGenerator()
    for seed in range(32):
        biped = generator.generate("biped", seed)
        leg_length = float(biped.profile_metadata["normalized_feature_vector"]["leg_length_norm"])
        for link in biped.links:
            if link.foot:
                assert link.foot_size is not None
                assert 2.0 * link.foot_size[0] / leg_length <= 0.78 + 1e-12

        wheeled = generator.generate("wheeled_biped", seed)
        trunk = next(link for link in wheeled.links if link.name == "trunk")
        hips = [link for link in wheeled.links if link.parent == "trunk" and link.name.endswith("_0_link")]
        assert len(hips) == 2
        assert hips[0].pos[0] == hips[1].pos[0]
        assert hips[0].pos[1] == -hips[1].pos[1]
        assert abs(hips[0].pos[0]) <= trunk.size[0]


def test_task070_wheel_topology_uses_continuous_contact_cylinders_and_slots() -> None:
    generator = ArchetypeConstrainedMorphologyGenerator()
    for family, expected_wheels in (("wheeled_biped", 2), ("wheeled_quadruped", 4)):
        blueprint = generator.generate(family, 3)
        assert {wheel.semantic_slot for wheel in blueprint.wheel_specs} == {
            f"limb{index}_wheel" for index in range(expected_wheels)
        }
        root = ET.fromstring(compile_mjcf(blueprint))
        joints = {str(item.get("name")): item for item in root.iter("joint")}
        geoms = {str(item.get("name")): item for item in root.iter("geom")}
        actuators = {str(item.get("name")): item for item in root.find("actuator") or ()}
        for wheel in blueprint.wheel_specs:
            assert joints[wheel.joint_name].get("limited") == "false"
            assert float(joints[wheel.joint_name].get("damping", "inf")) <= 1.0
            assert float(joints[wheel.joint_name].get("frictionloss", "inf")) <= 0.05
            assert geoms[f"{wheel.link_name}_geom"].get("type") == "cylinder"
            assert geoms[f"{wheel.link_name}_geom"].get("contype") == "1"
            assert geoms[f"{wheel.link_name}_geom"].get("conaffinity") == "1"
            assert actuators[f"{wheel.joint_name}_actuator"].tag == "motor"


def test_task070_contract_identity_changes_with_registry_or_distance_hash() -> None:
    baseline = ArchetypeConstrainedMorphologyGenerator().generate("biped", 0)
    changed_registry = ArchetypeConstrainedMorphologyGenerator(
        Task070ArchetypeConfig(reference_registry_sha256="0" * 64)
    ).generate("biped", 0)
    changed_distance = ArchetypeConstrainedMorphologyGenerator(
        Task070ArchetypeConfig(distance_contract_hash="1" * 64)
    ).generate("biped", 0)
    assert morphology_blueprint_hash(baseline) != morphology_blueprint_hash(changed_registry)
    assert morphology_blueprint_hash(baseline) != morphology_blueprint_hash(changed_distance)
    assert baseline.profile_metadata["contract_identity_hash"] != changed_registry.profile_metadata[
        "contract_identity_hash"
    ]


def test_task070_small_matrix_preserves_expected_denominator(tmp_path: Path) -> None:
    payload = run_archetype_matrix(
        tmp_path / "matrix.json",
        seeds=range(1),
        steps=50,
        render=False,
    )
    assert payload["expected_denominator"] == 4
    assert all(item["built"] == 1 for item in payload["summary"].values())
    assert all(item["compiled"] == 1 for item in payload["summary"].values())
    assert all(item["stance_hold"] == 1 for item in payload["summary"].values())
    assert all(item["support_gate"] == 1 for item in payload["summary"].values())
    assert all(item["contact_residual"] == 1 for item in payload["summary"].values())
    assert all(item["contact_wrench_residual"] == 1 for item in payload["summary"].values())
    assert all(item["self_contact_clear"] == 1 for item in payload["summary"].values())
    assert all(item["wheel_wheel_contact_clear"] == 1 for item in payload["summary"].values())
    assert all(item["region_band"] == 1 for item in payload["summary"].values())
    assert all(item["clone_guard"] == 1 for item in payload["summary"].values())
    for record in payload["records"]:
        stance = record["stance_hold"]
        assert stance["finite_fields"]["contact_wrench"] is True
        assert (
            stance["final_contact_wrench_force_residual_fraction"]
            <= payload["stance_thresholds"]["max_contact_wrench_force_residual_fraction"]
        )
        assert (
            stance["final_contact_wrench_torque_residual_fraction"]
            <= payload["stance_thresholds"]["max_contact_wrench_torque_residual_fraction"]
        )


def test_task070_final_matrix_gate_requires_exact_family_seed_denominator() -> None:
    matrix = _load_artifact("r4_archetype_morphology_matrix.json")
    gate = _final_matrix_gate(matrix)
    assert gate["passed"] is True
    assert gate["expected_denominator_exact"] is True
    assert gate["record_count_exact"] is True
    assert gate["record_key_set_exact"] is True
    assert set(gate["families"]) == set(FAMILIES)


def test_task070_r5_matrix_gate_fails_closed_on_empty_records(tmp_path: Path) -> None:
    matrix = _load_artifact("r4_archetype_morphology_matrix.json")
    matrix["summary"] = {}
    matrix["records"] = []
    bad_matrix_path = tmp_path / "bad_empty_matrix.json"
    bad_matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gate = _final_matrix_gate(matrix)
    assert gate["passed"] is False
    assert gate["summary_families_exact"] is False
    assert gate["record_count_exact"] is False
    assert gate["record_key_set_exact"] is False

    payload = write_final_verification(
        tmp_path / "r5_bad_empty_matrix.json",
        matrix_path=bad_matrix_path,
        compatibility_path=ARTIFACT_ROOT / "r0_compatibility_baseline.json",
        pytest_status="passed",
        ruff_status="passed",
        inspect_status="passed",
        full_pytest_status="passed",
        visual_status=VISUAL_REVIEW_STATUS,
        pytest_exit_code=0,
        ruff_exit_code=0,
        inspect_exit_code=0,
        full_pytest_exit_code=0,
    )
    assert payload["visual_review"]["passed"] is True
    assert payload["matrix_passed"] is False
    assert payload["matrix_gate"]["passed"] is False


def test_task070_r0_compatibility_baseline_still_matches_legacy_and_task069() -> None:
    result = verify_r0_compatibility_baseline(ARTIFACT_ROOT / "r0_compatibility_baseline.json")
    assert result["passed"]
    assert result["passed_count"] == result["expected_denominator"]


def test_task070_additional_humanoid_candidates_preserve_complete_source_motor_trees() -> None:
    import mujoco
    import numpy as np

    expected_counts = {
        "agibot_x1_serial": 29,
        "agibot_x2_ultra": 31,
        "engineai_t800": 25,
        "engineai_t800pro": 43,
        "limx_hu_d04": 31,
        "booster_t1_23": 23,
        "booster_t1_29": 29,
        "robotera_star1": 55,
    }
    assert set(TASK070_ADDITIONAL_HUMANOID_REFERENCE_IDS) == set(expected_counts)
    standard_slots = set(WHOLE_BODY_SLOT_NAMES)
    for reference_id, expected_count in expected_counts.items():
        descriptor = load_additional_humanoid_motor_dof_preserving_descriptor(
            reference_id
        )
        assert len(descriptor.motors) == expected_count
        assert len({motor.source_joint_name for motor in descriptor.motors}) == expected_count
        assert (
            len({motor.anonymous_semantic_slot for motor in descriptor.motors})
            == expected_count
        )
        assert all(motor.module != "other" for motor in descriptor.motors)
        body_names = {node.source_body_name for node in descriptor.body_tree}
        assert all(motor.source_parent_body in body_names for motor in descriptor.motors)
        assert all(motor.source_child_body in body_names for motor in descriptor.motors)

        generator = MotorDofPreservingArchetypePreviewGenerator(
            reference_id=reference_id
        )
        blueprint = generator.generate("biped", 0)
        physical = generator.sample_physical_params(blueprint, 70_000_000)
        xml = compile_mjcf(blueprint, physical)
        metadata = blueprint.profile_metadata
        assert compile_with_mujoco(xml).nu == expected_count
        assert len(blueprint.joints) == len(blueprint.actuators) == expected_count
        assert metadata["candidate_prior_status"] == "candidate_fail_closed"
        assert metadata["counts_toward_task070_v2_pass"] is False
        assert metadata["policy_adapter_compatible"] is False
        assert set(metadata["task070_candidate_extra_semantic_slots"]) == (
            set(blueprint.active_slots) - standard_slots
        )
        assert metadata["motor_accounting"]["bijection_passed"] is True
        assert metadata["motor_configuration"]["source_config_coverage"][
            "usable_quantitative_prior_count"
        ] == 0
        assert metadata["primitive_geometry_only"] is True
        assert metadata["mesh_texture_logo_copied"] is False
        assert "<mesh" not in xml

        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.qpos[2] = blueprint.nominal_height * physical.global_scale
        for joint in blueprint.joints:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint.name
            )
            data.qpos[model.jnt_qposadr[joint_id]] = joint.nominal
        mujoco.mj_forward(model, data)
        torso_boxes = [
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
            and "footpad" not in (mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            ) or "")
        ]
        assert len(torso_boxes) >= 2
        torso_pelvis_gap = mujoco.mj_geomDistance(
            model, data, torso_boxes[0], torso_boxes[1], 10.0, np.zeros(6)
        )
        assert torso_pelvis_gap > 0.008
        assert torso_pelvis_gap < 0.060

    x1 = MotorDofPreservingArchetypePreviewGenerator(
        reference_id="agibot_x1_serial"
    ).generate("biped", 0)
    assert x1.profile_metadata["motor_accounting"][
        "configured_physical_motor_count"
    ] == 31
    assert x1.profile_metadata["motor_accounting"][
        "source_model_config_motor_count_gap"
    ] == 2


def test_task070_candidate_wheel_composition_keeps_complete_source_motor_set() -> None:
    generator = MotorDofPreservingArchetypePreviewGenerator(
        reference_id="robotera_star1"
    )
    blueprint = generator.generate("wheeled_biped", 0)
    physical = generator.sample_physical_params(blueprint, 70_000_000)
    accounting = blueprint.profile_metadata["motor_accounting"]
    assert accounting["source_non_wheel_motor_count"] == 55
    assert accounting["anonymous_non_wheel_motor_count"] == 55
    assert accounting["added_wheel_motor_count"] == 2
    assert accounting["total_actuator_count"] == 57
    assert accounting["bijection_passed"] is True
    assert compile_with_mujoco(compile_mjcf(blueprint, physical)).nu == 57


def test_task070_candidate_extra_slots_fail_closed_without_every_gate() -> None:
    blueprint = MotorDofPreservingArchetypePreviewGenerator(
        reference_id="robotera_star1"
    ).generate("biped", 0)
    metadata = dict(blueprint.profile_metadata)
    invalid_updates = (
        {"candidate_prior_status": "promoted"},
        {"policy_adapter_compatible": True},
        {"counts_toward_task070_v2_pass": True},
        {"task070_candidate_extra_semantic_slots": []},
    )
    for update in invalid_updates:
        with pytest.raises(ValueError, match="unknown whole-body slots"):
            replace(blueprint, profile_metadata={**metadata, **update})


def test_task070_v2_canonical_root_site_and_runtime_reader_cover_18_cases() -> None:
    import mujoco
    import numpy as np

    from h200_locomotion_lab.robots.whole_body_stance import _leg_joint_indices

    cases = tuple(TASK070_V2_CURRENT_ARENA_CASES) + tuple(
        (reference_id, "biped")
        for reference_id in TASK070_ADDITIONAL_HUMANOID_REFERENCE_IDS
    )
    assert len(cases) == 18
    for reference_id, family in cases:
        blueprint = MotorDofPreservingArchetypePreviewGenerator(
            reference_id=reference_id
        ).generate(family, 0)
        frame = blueprint.profile_metadata["canonical_root_frame"]
        assert frame["contract_version"] == "canonical_root_frame_v1"
        assert frame["site_name"] == "canonical_root"
        assert frame["native_free_root_qpos_is_canonical"] is False
        assert frame["coordinate_convention"] == {
            "handedness": "right_handed",
            "x": "forward",
            "y": "left",
            "z": "up",
            "quaternion_order": "wxyz",
            "pose": "site_to_world",
            "twist": "expressed_in_canonical_frame/angular_then_linear",
            "projected_gravity": "world_minus_z_expressed_in_canonical",
            "transform_units": (
                "translation_is_blueprint_length_scaled_by_physical.global_scale_at_compile"
            ),
        }
        leg_joint_indices = _leg_joint_indices(blueprint)
        assert leg_joint_indices == tuple(
            index
            for index, joint in enumerate(blueprint.joints)
            if joint.semantic_slot.startswith("limb") and joint.axis_name != "wheel"
        )
        assert leg_joint_indices
        anchor_transform = frame["anchor_body_from_canonical"]
        inverse_transform = frame["canonical_from_anchor_body"]
        origin = np.asarray(frame["origin"], dtype=float)
        quaternion = np.asarray(anchor_transform["quaternion_wxyz"], dtype=float)
        inverse_quaternion = np.asarray(
            inverse_transform["quaternion_wxyz"],
            dtype=float,
        )
        np.testing.assert_allclose(anchor_transform["translation"], origin, atol=1e-12)
        np.testing.assert_allclose(
            inverse_quaternion,
            quaternion * np.asarray((1.0, -1.0, -1.0, -1.0)),
            atol=1e-12,
        )
        assert np.linalg.norm(quaternion) == pytest.approx(1.0, abs=1e-12)
        rotated_origin = np.zeros(3, dtype=float)
        mujoco.mju_rotVecQuat(rotated_origin, origin, inverse_quaternion)
        np.testing.assert_allclose(
            inverse_transform["translation"],
            -rotated_origin,
            atol=1e-12,
        )
        model = compile_with_mujoco(compile_mjcf(blueprint))
        assert sum(
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) == "canonical_root")
            for site_id in range(model.nsite)
        ) == 1
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "canonical_root")
        first = []
        for limb in range(4 if family.endswith("quadruped") else 2):
            joint = next(
                joint for joint in blueprint.joints
                if joint.semantic_slot.startswith(f"limb{limb}_")
                and joint.parent_link == frame["site_body_link"]
            )
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
            first.append(data.xanchor[joint_id])
        expected = np.mean(np.asarray(first), axis=0)
        np.testing.assert_allclose(data.site_xpos[site_id], expected, atol=1e-8)
        axes = np.asarray(data.site_xmat[site_id]).reshape(3, 3)
        np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-8)
        assert np.linalg.det(axes) == pytest.approx(1.0, abs=1e-8)
        np.testing.assert_allclose(axes, np.eye(3), atol=1e-8)

    blueprint = MotorDofPreservingArchetypePreviewGenerator().generate("biped", 0)
    model = compile_with_mujoco(compile_mjcf(blueprint))
    data = mujoco.MjData(model)
    root_quaternion = np.asarray((0.91, 0.19, -0.11, 0.31), dtype=float)
    data.qpos[3:7] = root_quaternion / np.linalg.norm(root_quaternion)
    data.qvel[:6] = (0.2, -0.3, 0.4, -0.5, 0.6, -0.7)
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "canonical_root")
    state = read_canonical_root_state(model, data, site_id)
    np.testing.assert_allclose(state.world_position, data.site_xpos[site_id], atol=1e-10)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    np.testing.assert_allclose(state.world_quaternion_wxyz, quat, atol=1e-10)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, velocity, 1)
    np.testing.assert_allclose(state.local_angular_velocity, velocity[:3], atol=1e-10)
    np.testing.assert_allclose(state.local_linear_velocity, velocity[3:], atol=1e-10)
    matrix = np.asarray(data.site_xmat[site_id]).reshape(3, 3)
    np.testing.assert_allclose(state.projected_gravity, -matrix[2], atol=1e-10)
    assert read_canonical_root_state(model, data) == state


@pytest.mark.parametrize("reference_id", ("booster_t1_23", "robotera_star1"))
def test_task070_canonical_root_tracks_waist_below_native_free_root(
    reference_id: str,
) -> None:
    import mujoco
    import numpy as np

    generator = MotorDofPreservingArchetypePreviewGenerator(
        reference_id=reference_id
    )
    blueprint = generator.generate("biped", 0)
    frame = blueprint.profile_metadata["canonical_root_frame"]
    model = compile_with_mujoco(compile_mjcf(blueprint))
    data = mujoco.MjData(model)
    waist_joint = next(
        joint
        for joint in blueprint.joints
        if joint.child_link == frame["site_body_link"]
    )
    waist_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        waist_joint.name,
    )
    data.qpos[model.jnt_qposadr[waist_id]] = min(
        0.31,
        waist_joint.joint_range[1] - 0.05,
    )
    data.qvel[:6] = (0.2, -0.3, 0.4, -0.5, 0.6, -0.7)
    data.qvel[model.jnt_dofadr[waist_id]] = 0.8
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "canonical_root",
    )
    anchors = []
    for limb in range(2):
        hip_joint = next(
            joint
            for joint in blueprint.joints
            if joint.semantic_slot.startswith(f"limb{limb}_")
            and joint.parent_link == frame["site_body_link"]
        )
        hip_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            hip_joint.name,
        )
        anchors.append(data.xanchor[hip_id])
    np.testing.assert_allclose(
        data.site_xpos[site_id],
        np.mean(np.asarray(anchors), axis=0),
        atol=1e-8,
    )
    parent_body_id = int(model.site_bodyid[site_id])
    np.testing.assert_allclose(
        data.site_xmat[site_id],
        data.xmat[parent_body_id],
        atol=1e-8,
    )
    state = read_canonical_root_state(model, data, site_id)
    velocity = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_SITE,
        site_id,
        velocity,
        1,
    )
    np.testing.assert_allclose(state.local_angular_velocity, velocity[:3], atol=1e-10)
    np.testing.assert_allclose(state.local_linear_velocity, velocity[3:], atol=1e-10)
    matrix = np.asarray(data.site_xmat[site_id]).reshape(3, 3)
    np.testing.assert_allclose(state.projected_gravity, -matrix[2], atol=1e-10)


def test_task070_g1_whole_body_shard_consumes_canonical_root(monkeypatch: pytest.MonkeyPatch) -> None:
    import mujoco
    import numpy as np

    import h200_locomotion_lab.envs.whole_body_mujoco as shard_module
    from h200_locomotion_lab.robots.whole_body_stance import StanceSolution

    generator = MotorDofPreservingArchetypePreviewGenerator()
    blueprint = generator.generate("biped", 0)
    physical = generator.sample_physical_params(blueprint, 70_000_000)

    def nominal_stance(model, data, candidate, candidate_physical):
        del model, data
        joint_qpos = {joint.semantic_slot: joint.nominal for joint in candidate.joints}
        return StanceSolution(
            instance_key=morphology_instance_key(candidate, candidate_physical),
            base_height=candidate.nominal_height * candidate_physical.global_scale,
            joint_qpos=joint_qpos,
            actuator_ctrl=joint_qpos,
        )

    monkeypatch.setattr(shard_module, "solve_static_stance", nominal_stance)
    shard = shard_module.WholeBodyMuJoCoShard(
        blueprint,
        physical=physical,
        num_envs=1,
    )
    assert shard._canonical_root_site_id is not None
    data = shard.data[0]
    root_quaternion = np.asarray((0.91, 0.19, -0.11, 0.31), dtype=float)
    data.qpos[3:7] = root_quaternion / np.linalg.norm(root_quaternion)
    data.qvel[:6] = (0.2, -0.3, 0.4, -0.5, 0.6, -0.7)
    mujoco.mj_forward(shard.model, data)

    state = read_canonical_root_state(
        shard.model,
        data,
        shard._canonical_root_site_id,
    )
    observation = shard._observation(data, 0, False)
    assert observation[:3] == pytest.approx(state.local_linear_velocity)
    assert observation[3:6] == pytest.approx(state.local_angular_velocity)
    assert observation[6:9] == pytest.approx(state.projected_gravity)

    shard._commands[0] = (
        state.local_linear_velocity[0],
        state.local_linear_velocity[1],
        state.local_angular_velocity[2],
    )
    reward, normalized_error, non_foot = shard._reward(data, 0)
    upright = max(0.0, min(1.0, -state.projected_gravity[2]))
    assert reward == pytest.approx(1.25 + 0.25 * upright - 0.10 * non_foot)
    assert normalized_error == pytest.approx(0.0)

    target_height = shard._fall_height_threshold() - 1e-4
    data.qpos[2] += target_height - state.world_position[2]
    mujoco.mj_forward(shard.model, data)
    assert float(data.qpos[2]) > shard._fall_height_threshold()
    assert shard._is_fallen(data) is True


def test_task070_flat_arena_smoke_checks_motion_without_claiming_walking(
    tmp_path: Path,
) -> None:
    payload = run_v2_arena_smoke(
        tmp_path / "arena" / "smoke.json",
        cases=(
            ("unitree_g1", "biped"),
            ("unitree_go2", "quadruped"),
            ("unitree_g1", "wheeled_biped"),
            ("agibot_x1_serial", "biped"),
        ),
        stance_steps=1000,
        response_steps=12,
        render=False,
    )
    assert payload["walking_claimed"] is False
    assert payload["dynamic_locomotion_policy_present"] is False
    assert payload["counts_toward_task070_v2_pass"] is False
    assert payload["summary"]["case_count"] == 4
    assert payload["summary"]["compiled"] == 4
    assert payload["summary"]["accounting_exact"] == 4
    assert payload["summary"]["reset_pose_passed"] == 4
    assert payload["summary"]["all_actuators_responsive"] == 4
    for record in payload["records"]:
        assert record["walking_claimed"] is False
        assert record["error"] is None
        response = record["actuator_response"]
        assert response["contract"] == (
            "task070_flat_arena_paired_baseline_actuator_pulse_v2"
        )
        assert response["actuator_count"] == record["model_nu"]
        assert response["responsive_actuator_count"] == record["model_nu"]
        assert response["all_actuators_responsive"] is True
        for actuator_record in response["records"]:
            metric = actuator_record["response_metric"]
            assert metric.startswith("max_command_induced_")
            assert actuator_record[metric] >= actuator_record["response_threshold"]
        assert record["stance_hold"]["solver_fatal"] is False


def test_task071_v2_correlated_physical_sampling_is_com_and_slot_auditable() -> None:
    from h200_locomotion_lab.envs.whole_body_mujoco import _motor_process_baselines

    for reference_id, family in (("unitree_g1", "biped"), ("unitree_go2", "quadruped")):
        generator = MotorDofPreservingArchetypePreviewGenerator(reference_id=reference_id)
        blueprint = generator.generate(family, 0)
        nominal = generator.sample_physical_params(blueprint, 701, range_fraction=0.0)
        full = generator.sample_physical_params(blueprint, 701, range_fraction=1.0)
        other = generator.sample_physical_params(blueprint, 70000002, range_fraction=1.0)
        replay = generator.sample_physical_params(blueprint, 701, range_fraction=1.0)
        assert full == replay
        assert full.motor_strength != other.motor_strength
        assert full.kp_scales != other.kp_scales
        assert full.kd_scales != other.kd_scales
        assert full.delay_ms != other.delay_ms
        strength, latency, _ = _motor_process_baselines(blueprint, full, 50.0)
        assert strength == (1.0,) * len(blueprint.joints)
        assert latency == (round(full.delay_ms * 50.0 / 1000.0),) * len(blueprint.joints)
        _, other_latency, _ = _motor_process_baselines(blueprint, other, 50.0)
        assert latency != other_latency
        assert nominal.motor_strength == {slot: 1.0 for slot in nominal.motor_strength}
        assert nominal.kp_scales == nominal.kd_scales == nominal.motor_strength
        assert nominal.delay_ms == 0.0
        assert full.metadata["task071_correlated_actuation"]["eligible"] is True
        audit = full.metadata["task071_correlated_actuation"]
        assert set(audit["slot_composition"]) == {joint.semantic_slot for joint in blueprint.joints}
        assert audit["independent_per_slot_noise"] is False
        assert audit["delay_runtime_owner"] == "WholeBodyMuJoCoShard→MotorProcess"
        families = audit["family_latents"]
        groups = audit["group_efficiency_latents"]
        for factors in audit["slot_composition"].values():
            family = families[factors["family_id"]]
            efficiency = groups[factors["group_id"]]
            assert factors["motor_strength"] == pytest.approx(
                family["effort"] * efficiency
            )
            assert factors["kp_scale"] == pytest.approx(
                family["bandwidth"] * (0.5 + 0.5 * family["effort"])
            )
            assert factors["kd_scale"] == pytest.approx(
                family["bandwidth"] * (0.75 + 0.25 * family["effort"])
            )
        xml = ET.fromstring(compile_mjcf(blueprint, full))
        bodies = {node.attrib["name"]: node for node in xml.findall(".//body")}
        for link in blueprint.links:
            actual = tuple(float(value) for value in bodies[link.name].find("inertial").attrib["pos"].split())
            assert actual == pytest.approx(full.com_offsets[link.name])
            assert full.com_offsets[link.name] == pytest.approx(
                tuple(value * full.global_scale * full.link_scales[link.name] for value in link.com)
            )
        actuators = {node.attrib["name"]: node for node in xml.findall(".//actuator/*")}
        resolved = {
            record["anonymous_semantic_slot"]: record["final_compiled"]
            for record in blueprint.profile_metadata["motor_configuration"]["resolved_anonymous_actuators"]
        }
        for actuator in blueprint.actuators:
            node = actuators[actuator.name]
            source = resolved[actuator.semantic_slot]
            assert float(node.attrib["kp"]) == pytest.approx(source["kp"] * full.kp_scales[actuator.semantic_slot], rel=1e-5)
            assert float(node.attrib["kv"]) == pytest.approx(source["kd"] * full.kd_scales[actuator.semantic_slot], rel=1e-5)
            assert float(node.attrib["forcerange"].split()[1]) == pytest.approx(source["effort_limit"] * full.motor_strength[actuator.semantic_slot], rel=1e-5)


@pytest.mark.parametrize("reference_id, family", [("spot_base", "quadruped"), ("agibot_x1_serial", "biped")])
def test_task071_v2_incomplete_or_candidate_evidence_remains_fail_closed_identity(
    reference_id: str, family: str
) -> None:
    generator = MotorDofPreservingArchetypePreviewGenerator(reference_id=reference_id)
    blueprint = generator.generate(family, 0)
    physical = generator.sample_physical_params(blueprint, 702, range_fraction=1.0)
    audit = physical.metadata["task071_correlated_actuation"]
    assert audit["eligible"] is False
    assert physical.motor_strength == {slot: 1.0 for slot in physical.motor_strength}
    assert physical.kp_scales == physical.kd_scales == physical.motor_strength
    assert physical.delay_ms == 0.0


def test_task071_legacy_profiles_keep_physical_strength_runtime_baselines() -> None:
    from h200_locomotion_lab.envs.whole_body_mujoco import _motor_process_baselines

    for generator in (
        MorphologyGenerator(),
        LocoFormerMorphologyGenerator(),
        ArchetypeConstrainedMorphologyGenerator(),
    ):
        blueprint = generator.generate("biped", 3)
        physical = generator.sample_physical_params(blueprint, 4)
        strength, latency, ema = _motor_process_baselines(blueprint, physical, 50.0)
        expected = tuple(physical.motor_strength[joint.semantic_slot] for joint in blueprint.joints)
        assert strength == expected
        assert latency == (round(physical.delay_ms * 50.0 / 1000.0),) * len(blueprint.joints)
        assert ema == (physical.ema_alpha,) * len(blueprint.joints)
