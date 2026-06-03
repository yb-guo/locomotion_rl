import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from h200_locomotion_lab.robots.g1like_mjcf_patch import (
    G1LikeMJCFPatchError,
    mjcf_topology,
    patch_g1like_mjcf_string,
    patch_g1like_mjcf_variant,
)
from h200_locomotion_lab.tools import task038_g1like_variant_asset_smoke as asset_smoke
from h200_locomotion_lab.robots.g1like_morphology import (
    G1LIKE_SLOT_SCHEMA_ID,
    generate_g1like_morphology_manifest,
    joint_order_hash,
    slot_schema_hash,
)


MINIMAL_MJCF = """\
<mujoco model="tiny_g1like">
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <inertial pos="0 0 0.1" mass="10" diaginertia="1 2 3"/>
      <joint name="left_hip_pitch_joint" pos="0.1 0.2 0.3" actuatorfrcrange="-100 100"/>
      <geom name="pelvis_geom" pos="0 0 0" size="0.1"/>
      <site name="imu" pos="0 0 0.2"/>
      <body name="left_thigh" pos="0 0 -0.3">
        <inertial pos="0 0 -0.1" mass="2" fullinertia="1 2 3 0.1 0.2 0.3"/>
        <joint name="left_knee_joint" pos="0 0 -0.4"/>
        <geom name="left_thigh_geom" pos="0 0 -0.2" size="0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="left_hip_pitch_joint" joint="left_hip_pitch_joint" gear="100" ctrlrange="-1 1" forcerange="-50 50"/>
    <motor name="left_knee_joint" joint="left_knee_joint" gear="80"/>
  </actuator>
</mujoco>
"""


MISSING_ATTR_MJCF = """\
<mujoco model="missing_attrs">
  <worldbody>
    <body name="pelvis">
      <inertial mass="1"/>
      <joint name="left_hip_pitch_joint"/>
      <geom name="pelvis_geom"/>
      <site name="imu"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="left_hip_pitch_joint" joint="left_hip_pitch_joint"/>
  </actuator>
</mujoco>
"""


RELATIVE_MESHDIR_MJCF = MINIMAL_MJCF.replace(
    '<mujoco model="tiny_g1like">',
    '<mujoco model="tiny_g1like">\n  <compiler meshdir="assets"/>',
)


ABSOLUTE_MESHDIR_MJCF = MINIMAL_MJCF.replace(
    '<mujoco model="tiny_g1like">',
    '<mujoco model="tiny_g1like">\n  <compiler meshdir="C:/robot/assets"/>',
)


def test_patcher_preserves_topology_names_and_order() -> None:
    variant = _variant()
    before = mjcf_topology(ET.fromstring(MINIMAL_MJCF))

    patched_xml, summary = patch_g1like_mjcf_string(MINIMAL_MJCF, variant=variant)
    after = mjcf_topology(ET.fromstring(patched_xml))

    assert after == before
    assert summary["topology_before"] == before
    assert summary["topology_after"] == before
    assert summary["pass"] is True


def test_scales_expected_xml_attrs_and_records_counts() -> None:
    variant = _variant(
        link_scale=2.0,
        mass_scale=3.0,
        com_scale=4.0,
        inertia_scale=5.0,
        motor_strength_scale=6.0,
    )

    patched_xml, summary = patch_g1like_mjcf_string(MINIMAL_MJCF, variant=variant)
    root = ET.fromstring(patched_xml)
    pelvis = root.find(".//body[@name='pelvis']")
    inertial = root.find(".//body[@name='pelvis']/inertial")
    hip_joint = root.find(".//joint[@name='left_hip_pitch_joint']")
    hip_motor = root.find(".//actuator/motor[@name='left_hip_pitch_joint']")

    assert pelvis is not None
    assert inertial is not None
    assert hip_joint is not None
    assert hip_motor is not None
    assert pelvis.attrib["pos"] == "0 0 2"
    assert inertial.attrib["mass"] == "30"
    assert inertial.attrib["pos"] == "0 0 0.4"
    assert inertial.attrib["diaginertia"] == "5 10 15"
    assert hip_joint.attrib["actuatorfrcrange"] == "-600 600"
    assert hip_motor.attrib["ctrlrange"] == "-6 6"
    assert hip_motor.attrib["forcerange"] == "-300 300"
    assert hip_motor.attrib["gear"] == "600"
    assert summary["patched_counts"]["body_pos"] == 2
    assert summary["patched_counts"]["joint_pos"] == 2
    assert summary["patched_counts"]["geom_pos"] == 2
    assert summary["patched_counts"]["site_pos"] == 1
    assert summary["patched_counts"]["inertial_mass"] == 2
    assert summary["patched_counts"]["inertial_diaginertia"] == 1
    assert summary["patched_counts"]["inertial_fullinertia"] == 1
    assert summary["patched_counts"]["actuator_ctrlrange"] == 1
    assert summary["patched_counts"]["actuator_forcerange"] == 1
    assert summary["patched_counts"]["actuator_gear"] == 2
    assert summary["patched_counts"]["joint_actuatorfrcrange"] == 1
    assert any("nonphysical" in note.lower() for note in summary["limitation_notes"])


def test_missing_attrs_are_recorded_as_skipped_and_limited_not_success_claims() -> None:
    patched_xml, summary = patch_g1like_mjcf_string(
        MISSING_ATTR_MJCF,
        variant=_variant(),
    )

    ET.fromstring(patched_xml)
    assert summary["local_parse_ok"] is True
    assert summary["pass"] is True
    assert summary["skipped_counts"]["body_pos_missing"] == 1
    assert summary["skipped_counts"]["joint_pos_missing"] == 1
    assert summary["skipped_counts"]["geom_pos_missing"] == 1
    assert summary["skipped_counts"]["site_pos_missing"] == 1
    assert summary["skipped_counts"]["actuator_ctrlrange_missing"] == 1
    assert summary["skipped_counts"]["actuator_forcerange_missing"] == 1
    assert summary["skipped_counts"]["actuator_gear_missing"] == 1
    assert any("not treated as successfully patched" in note for note in summary["limitation_notes"])


def test_file_patcher_summary_contains_required_contract_fields(tmp_path: Path) -> None:
    source = tmp_path / "source.xml"
    output = tmp_path / "patched.xml"
    source.write_text(MINIMAL_MJCF, encoding="utf-8")

    summary = patch_g1like_mjcf_variant(
        source_mjcf=source,
        output_mjcf=output,
        variant=_variant(),
    )

    assert output.exists()
    assert summary["variant_id"] == "fixture-train"
    assert summary["split"] == "train"
    assert summary["heldout_condition"] == "none"
    assert summary["slot_schema_id"] == G1LIKE_SLOT_SCHEMA_ID
    assert summary["slot_schema_hash"] == slot_schema_hash()
    assert summary["joint_order_hash"] == joint_order_hash()
    assert summary["action_dim"] == 29
    assert summary["source_path"] == str(source)
    assert summary["output_path"] == str(output)
    assert summary["source_xml_dir"] == str(source.parent)
    assert summary["meshdir_before"] is None
    assert summary["meshdir_after"] is None
    assert summary["meshdir_rewritten"] is False
    assert summary["local_parse_ok"] is True


def test_relative_meshdir_is_rewritten_when_output_dir_differs(tmp_path: Path) -> None:
    source_dir = tmp_path / "robot"
    output_dir = tmp_path / "outputs"
    source_dir.mkdir()
    source = source_dir / "source.xml"
    output = output_dir / "patched.xml"
    source.write_text(RELATIVE_MESHDIR_MJCF, encoding="utf-8")

    summary = patch_g1like_mjcf_variant(
        source_mjcf=source,
        output_mjcf=output,
        variant=_variant(),
    )

    compiler = ET.parse(output).getroot().find("compiler")
    expected_meshdir = str((source_dir / "assets").resolve())
    assert compiler is not None
    assert compiler.attrib["meshdir"] == expected_meshdir
    assert summary["meshdir_before"] == "assets"
    assert summary["meshdir_after"] == expected_meshdir
    assert summary["meshdir_rewritten"] is True
    assert summary["source_xml_dir"] == str(source_dir)


def test_absolute_meshdir_is_not_rewritten_when_output_dir_differs(tmp_path: Path) -> None:
    source_dir = tmp_path / "robot"
    output_dir = tmp_path / "outputs"
    source_dir.mkdir()
    source = source_dir / "source.xml"
    output = output_dir / "patched.xml"
    source.write_text(ABSOLUTE_MESHDIR_MJCF, encoding="utf-8")

    summary = patch_g1like_mjcf_variant(
        source_mjcf=source,
        output_mjcf=output,
        variant=_variant(),
    )

    compiler = ET.parse(output).getroot().find("compiler")
    assert compiler is not None
    assert compiler.attrib["meshdir"] == "C:/robot/assets"
    assert summary["meshdir_before"] == "C:/robot/assets"
    assert summary["meshdir_after"] == "C:/robot/assets"
    assert summary["meshdir_rewritten"] is False


def test_cli_writes_summary_with_train_and_heldout_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.xml"
    output_dir = tmp_path / "variants"
    summary_json = tmp_path / "summary.json"
    source.write_text(MINIMAL_MJCF, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "h200_locomotion_lab.tools.task038_g1like_variant_asset_smoke",
            "--source-mjcf",
            str(source),
            "--output-dir",
            str(output_dir),
            "--summary-json",
            str(summary_json),
            "--seed",
            "38",
            "--heldout-condition",
            "combined",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert json.loads(result.stdout)["pass"] is True
    assert summary["local_parse_ok"] is True
    assert summary["pass"] is True
    assert summary["h200_load_smoke"] == "pending"
    assert summary["simulator_started"] is False
    assert summary["asset_downloaded"] is False
    assert summary["mujoco_compile_requested"] is False
    assert summary["mujoco_compile"] == {"status": "not_requested", "variants": []}
    assert [item["split"] for item in summary["variant_summaries"]] == ["train", "heldout"]
    assert len(list(output_dir.glob("*.xml"))) == 2


def test_cli_compile_path_records_model_metadata_without_requiring_mujoco(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.xml"
    output_dir = tmp_path / "variants"
    summary_json = tmp_path / "summary.json"
    source.write_text(MINIMAL_MJCF, encoding="utf-8")

    def fake_compile(path: Path) -> dict:
        return {
            "compile_ok": True,
            "blocked": False,
            "nq": 7,
            "nv": 6,
            "nu": 2,
            "njnt": 3,
        }

    monkeypatch.setattr(asset_smoke, "_compile_mujoco_xml", fake_compile)
    args = asset_smoke.parse_args(
        [
            "--source-mjcf",
            str(source),
            "--output-dir",
            str(output_dir),
            "--summary-json",
            str(summary_json),
            "--compile-mujoco",
        ]
    )

    summary = asset_smoke.run_smoke(args)

    assert summary["pass"] is True
    assert summary["simulator_started"] is False
    assert summary["mujoco_compile_requested"] is True
    assert summary["mujoco_compile"]["status"] == "ok"
    assert len(summary["mujoco_compile"]["variants"]) == 2
    assert all(item["compile_ok"] is True for item in summary["mujoco_compile"]["variants"])
    assert all(item["nq"] == 7 for item in summary["mujoco_compile"]["variants"])


def test_cli_help_works() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "h200_locomotion_lab.tools.task038_g1like_variant_asset_smoke",
            "--help",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--source-mjcf" in result.stdout
    assert "--heldout-condition" in result.stdout
    assert "--compile-mujoco" in result.stdout


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda variant: variant.pop("variant_id"), "missing required fields"),
        (lambda variant: variant.update({"split": "eval"}), "split must be train or heldout"),
        (lambda variant: variant.update({"link_scale": 0.0}), "link_scale must be positive"),
        (lambda variant: variant.update({"mass_scale": "heavy"}), "mass_scale must be numeric"),
    ],
)
def test_invalid_variant_fails_clearly(mutator, message: str) -> None:
    variant = _variant()
    mutator(variant)

    with pytest.raises(G1LikeMJCFPatchError, match=message):
        patch_g1like_mjcf_string(MINIMAL_MJCF, variant=variant)


def test_invalid_numeric_xml_attr_fails_clearly() -> None:
    broken = MINIMAL_MJCF.replace('pos="0 0 1"', 'pos="0 bad 1"', 1)

    with pytest.raises(G1LikeMJCFPatchError, match="invalid numeric MJCF attribute"):
        patch_g1like_mjcf_string(broken, variant=_variant())


def _variant(**overrides) -> dict:
    variant = generate_g1like_morphology_manifest(
        seed=38,
        train_count=1,
        heldout_conditions=("combined",),
    )["variants"][0]
    variant = dict(variant)
    variant.update(
        {
            "variant_id": "fixture-train",
            "split": "train",
            "heldout_condition": "none",
            "link_scale": 1.1,
            "mass_scale": 1.2,
            "com_scale": 1.3,
            "inertia_scale": 1.4,
            "motor_strength_scale": 1.5,
        }
    )
    variant.update(overrides)
    return variant
