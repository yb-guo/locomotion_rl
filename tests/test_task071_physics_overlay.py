import copy
import importlib.util
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).parents[1]
    / ".agent/task/task071-multimorphology-training-readiness/"
    "task071_physics_overlay.py"
)
SPEC = importlib.util.spec_from_file_location("task071_physics_overlay", MODULE)
OVERLAY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = OVERLAY
SPEC.loader.exec_module(OVERLAY)


def test_scaled_inertial_uses_body_specific_scale():
    result = OVERLAY.scaled_inertial(
        2.0,
        (1.0, -2.0, 0.5),
        (1.0, 0.0, 0.0, 0.0),
        (3.0, 4.0, 5.0),
        1.5,
    )
    assert result == {
        "mass": 2.0,
        "pos": (1.5, -3.0, 0.75),
        "quat": (1.0, 0.0, 0.0, 0.0),
        "diaginertia": (6.75, 9.0, 11.25),
    }
    with pytest.raises(ValueError, match="positive"):
        OVERLAY.scaled_inertial(2.0, (0, 0, 0), (1, 0, 0, 0), (1, 1, 1), 0)


def test_structural_signature_allows_only_declared_physics_fields():
    frozen = ET.fromstring(
        """
        <mujoco model="procedural_x">
          <worldbody>
            <body name="root"><joint name="root_free" type="free"/>
              <body name="anon_link" pos="1 2 3">
                <joint name="anon_joint" type="hinge" axis="0 1 0" range="-1 1"
                  damping="1" armature="2" frictionloss="3"/>
                <geom name="anon_link_footpad" type="box" friction=".9 .1 .1"/>
                <inertial pos="0 0 0" quat="1 0 0 0" mass="1"
                  diaginertia="1 1 1"/>
                <site name="canonical_root" pos="0 0 0"/>
              </body>
            </body>
          </worldbody>
          <actuator><position name="anon_actuator" joint="anon_joint" kp="1" kv="2"
            forcerange="-3 3"/></actuator>
        </mujoco>
        """
    )
    bound = copy.deepcopy(frozen)
    bound.find(".//inertial").set("mass", "7")
    bound.find(".//joint[@name='anon_joint']").set("damping", "8")
    bound.find(".//position").set("forcerange", "-9 9")
    bound.find(".//geom").set("friction", ".4 .02 .01")
    terminals = {"anon_link_footpad"}
    assert OVERLAY._structural_signature(frozen, terminals) == OVERLAY._structural_signature(
        bound,
        terminals,
    )

    bound.find(".//joint[@name='anon_joint']").set("axis", "1 0 0")
    assert OVERLAY._structural_signature(frozen, terminals) != OVERLAY._structural_signature(
        bound,
        terminals,
    )


LOCAL_INPUTS = OVERLAY.FROZEN.is_dir() and OVERLAY.SOURCE.is_dir()


@pytest.mark.skipif(not LOCAL_INPUTS, reason="Task071 ignored frozen/official inputs unavailable")
def test_official_overlay_preserves_frozen_lineage_and_maps_effective_physics():
    result = OVERLAY.generate_overlay()
    assert result["official_repo"] == {
        "commit": OVERLAY.EXPECTED_REPO_COMMIT,
        "origin": OVERLAY.EXPECTED_REPO_ORIGIN,
        "clean": True,
    }
    assert result["summary"] == {
        "bound": 2,
        "lineage_preserved": 2,
        "structure_preserved": 2,
    }
    records = {record["reference_id"]: record for record in result["records"]}
    assert records["unitree_g1"]["mapping_counts"] == {
        "bodies": 30,
        "joints": 29,
        "actuators": 29,
        "terminal_contacts": 2,
    }
    assert records["unitree_go2"]["mapping_counts"] == {
        "bodies": 13,
        "joints": 12,
        "actuators": 12,
        "terminal_contacts": 4,
    }

    g1 = records["unitree_g1"]
    g1_motors = {row["official_joint"]: row for row in g1["motor_mapping"]}
    assert g1_motors["left_wrist_roll_joint"]["joint_frictionloss"] == 0.2
    assert g1_motors["left_wrist_pitch_joint"]["joint_frictionloss"] == 0.1
    pelvis = g1["body_mapping"][0]
    assert pelvis["official_body"] == "pelvis"
    assert math.isclose(
        pelvis["com_m"][2],
        -0.07605 * 1.0058724582641345,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    go2 = records["unitree_go2"]
    go2_motors = {row["official_joint"]: row for row in go2["motor_mapping"]}
    assert go2_motors["FL_calf_joint"]["force_range"] == [-45.43, 45.43]
    assert go2_motors["FR_calf_joint"]["force_range"] == [-45.43, 45.43]
    assert go2_motors["FL_hip_joint"]["force_range"] == [-23.7, 23.7]
    assert {
        tuple(record["friction"]) for record in go2["terminal_contact_mapping"]
    } == {(0.4, 0.02, 0.01)}
    assert all(record["compile_evidence"]["passed"] for record in records.values())
    assert all(
        record["frozen_input"]["unchanged_after_generation"]
        and record["structure_preservation"]["frozen_structural_signature_equal"]
        and record["structure_preservation"]["primitive_geometry_only"]
        and not record["structure_preservation"]["mesh_texture_logo_copied"]
        for record in records.values()
    )


@pytest.mark.skipif(not LOCAL_INPUTS, reason="Task071 ignored frozen/official inputs unavailable")
def test_bound_r1_uses_frozen_manifests_and_exact_timing():
    overlay = OVERLAY.generate_overlay()
    result = OVERLAY.run_bound_r1(overlay)
    assert result["summary"]["compiled"] == 2
    assert result["summary"]["accounting_exact"] == 2
    assert result["summary"]["frozen_lineage_match"] == 2
    assert result["summary"]["reset_pose_passed"] == 2
    assert result["summary"]["all_actuators_responsive"] == 2
    assert result["response_steps_per_actuator"] == 32
    assert result["stance_steps"] == 1000
    assert result["timestep_seconds"] == 0.002
    assert result["stance_duration_seconds"] == 2.0
    for record in result["records"]:
        assert record["frozen_task070_input_match"] is True
        assert record["actuator_response"]["response_steps_per_actuator"] == 32
        assert record["stance_hold"]["steps"] == 1000
        assert record["stance_hold"]["timestep_seconds"] == 0.002
        assert record["stance_hold"]["duration_seconds"] == 2.0
    assert result["task071_r1_admission_passed"] is (
        result["summary"]["stance_hold_passed"] == 2
    )


@pytest.mark.skipif(not LOCAL_INPUTS, reason="Task071 ignored frozen/official inputs unavailable")
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("output_xml", "artifacts/models/tampered.xml"),
        ("output_xml_sha256", "0" * 64),
    ),
)
def test_bound_r1_rejects_tampered_overlay_binding(field, value):
    overlay = OVERLAY.generate_overlay()
    overlay["records"][0][field] = value
    with pytest.raises(ValueError, match="caller-supplied R1 overlay"):
        OVERLAY.run_bound_r1(overlay)
