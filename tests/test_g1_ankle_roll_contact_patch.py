import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from xml.etree import ElementTree

import pytest

from h200_locomotion_lab.tools import g1_ankle_roll_contact_patch as patcher

TARGETS = ("left_ankle_roll_link", "right_ankle_roll_link")


def test_parse_args_defaults_to_profile_source_asset(monkeypatch) -> None:
    root = fresh_test_dir("default_source")
    source = write_fixture_xml(root / "fixture.xml")
    monkeypatch.setattr(
        patcher,
        "load_g1_27dof_nohand_profile",
        lambda: SimpleNamespace(asset=SimpleNamespace(path=str(source))),
    )
    args = patcher.parse_args(
        [
            "--output-root",
            str(root / "out"),
            "--run-id",
            "default_source",
            "--variants",
            "ankle_roll_friction_attrs",
        ]
    )

    summary = patcher.run_patch_generation(args)

    assert summary["source_path"] == str(source.resolve())
    assert summary["source_unchanged"] is True
    assert Path(summary["variants"]["ankle_roll_friction_attrs"]["path"]).is_file()


def test_patch_generation_writes_all_variants_and_preserves_source_and_visual() -> None:
    root = fresh_test_dir("all_variants")
    source = write_fixture_xml(root / "g1.xml")
    source_before = source.read_text(encoding="utf-8")
    args = patcher.parse_args(
        [
            "--source-asset",
            str(source),
            "--output-root",
            str(root / "outputs"),
            "--run-id",
            "patch",
            "--larger-sphere-size",
            "0.012",
        ]
    )

    summary = patcher.run_patch_generation(args)

    run_dir = root / "outputs" / "patch"
    written_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert source.read_text(encoding="utf-8") == source_before
    assert written_summary["source_unchanged"] is True
    assert written_summary["meshdir_handling"]["compiler_present"] is False
    assert written_summary["meshdir_handling"]["rewritten"] is False
    assert set(written_summary["variants"]) == set(patcher.VARIANTS)

    friction_xml = parse_variant(summary, "ankle_roll_friction_attrs")
    larger_xml = parse_variant(summary, "ankle_roll_larger_spheres")
    box_xml = parse_variant(summary, "ankle_roll_box_support")
    mesh_collision_xml = parse_variant(summary, "ankle_roll_mesh_collision")
    sole_collision_xml = parse_variant(summary, "ankle_roll_sole_collision")
    center_keep_xml = parse_variant(summary, "ankle_roll_center_sole_keep_points")
    center_no_points_xml = parse_variant(summary, "ankle_roll_center_sole_no_points")
    mesh_bbox_xml = parse_variant(summary, "ankle_roll_mesh_bbox_sole_no_points")
    edge_boxes_xml = parse_variant(summary, "ankle_roll_edge_boxes_no_points")
    hybrid_edge_boxes_xml = parse_variant(summary, "ankle_roll_hybrid_edge_boxes_no_points")

    for root in (
        friction_xml,
        larger_xml,
        box_xml,
        mesh_collision_xml,
        sole_collision_xml,
        center_keep_xml,
        center_no_points_xml,
        mesh_bbox_xml,
        edge_boxes_xml,
        hybrid_edge_boxes_xml,
    ):
        assert root.find("compiler") is None
        for body_name in TARGETS:
            visual = geom_by_name(root, f"{body_name}_visual")
            assert visual.attrib == {
                "name": f"{body_name}_visual",
                "type": "mesh",
                "mesh": f"{body_name}_mesh",
                "contype": "0",
                "conaffinity": "0",
            }

    for body_name in TARGETS:
        support_geoms = support_geoms_for_body(friction_xml, body_name)
        assert len(support_geoms) == 4
        assert all(geom.get("friction") == "1.0 0.02 0.001" for geom in support_geoms)
        assert all(geom.get("condim") == "4" for geom in support_geoms)
        assert all(geom.get("priority") == "1" for geom in support_geoms)
        assert all(geom.get("size") == "0.005" for geom in support_geoms)

        larger_support_geoms = support_geoms_for_body(larger_xml, body_name, size="0.012")
        assert len(larger_support_geoms) == 4
        assert [geom.get("pos") for geom in larger_support_geoms] == [
            "0.03 0.02 -0.01",
            "0.03 -0.02 -0.01",
            "-0.03 0.02 -0.01",
            "-0.03 -0.02 -0.01",
        ]

        body = find_body(box_xml, body_name)
        assert body is not None
        assert body.find("inertial").attrib == {"mass": "0.608", "pos": "0 0 0"}
        box = geom_by_name(box_xml, f"{body_name}_task022_box_support")
        assert box.get("type") == "box"
        assert box.get("size") == patcher.BOX_SUPPORT_SIZE
        assert box.get("pos") == patcher.BOX_SUPPORT_POS

        mesh_collision = geom_by_name(
            mesh_collision_xml,
            f"{body_name}_task023_mesh_collision",
        )
        assert mesh_collision.attrib == {
            "name": f"{body_name}_task023_mesh_collision",
            "type": "mesh",
            "mesh": f"{body_name}_mesh",
            "friction": "1.0 0.02 0.001",
            "condim": "4",
            "priority": "1",
            "contype": "1",
            "conaffinity": "1",
        }
        sole_collision = geom_by_name(
            sole_collision_xml,
            f"{body_name}_task023_sole_collision",
        )
        assert sole_collision.attrib == {
            "name": f"{body_name}_task023_sole_collision",
            "type": "box",
            "size": "0.035 0.025 0.004",
            "pos": "0 0 -0.011",
            "friction": "1.0 0.02 0.001",
            "condim": "4",
            "priority": "1",
            "contype": "1",
            "conaffinity": "1",
        }
        support_prefix = body_name.removesuffix("_ankle_roll_link")
        for index in range(4):
            support = geom_by_name(sole_collision_xml, f"{support_prefix}_support_{index}")
            assert support.get("contype") == "0"
            assert support.get("conaffinity") == "0"

        center_keep = geom_by_name(
            center_keep_xml,
            f"{body_name}_task023_center_sole_keep_points",
        )
        assert center_keep.attrib == {
            "name": f"{body_name}_task023_center_sole_keep_points",
            "type": "box",
            "size": patcher.BOX_SUPPORT_SIZE,
            "pos": patcher.BOX_SUPPORT_POS,
            "friction": "1.0 0.02 0.001",
            "condim": "4",
            "priority": "1",
            "contype": "1",
            "conaffinity": "1",
        }
        for index in range(4):
            support = geom_by_name(center_keep_xml, f"{support_prefix}_support_{index}")
            assert support.get("contype") is None
            assert support.get("conaffinity") is None

        center_no_points = geom_by_name(
            center_no_points_xml,
            f"{body_name}_task023_center_sole_no_points",
        )
        assert center_no_points.get("size") == patcher.BOX_SUPPORT_SIZE
        assert center_no_points.get("pos") == patcher.BOX_SUPPORT_POS
        for index in range(4):
            support = geom_by_name(center_no_points_xml, f"{support_prefix}_support_{index}")
            assert support.get("contype") == "0"
            assert support.get("conaffinity") == "0"

        mesh_bbox = geom_by_name(
            mesh_bbox_xml,
            f"{body_name}_task023_mesh_bbox_sole_no_points",
        )
        assert mesh_bbox.get("type") == "box"
        assert mesh_bbox.get("size") == "0.1 0.04 0.006"
        assert mesh_bbox.get("pos") == "0.04 0 -0.006"
        for index in range(4):
            support = geom_by_name(mesh_bbox_xml, f"{support_prefix}_support_{index}")
            assert support.get("contype") == "0"
            assert support.get("conaffinity") == "0"

        edge_heel = geom_by_name(
            edge_boxes_xml,
            f"{body_name}_task023_edge_boxes_no_points_heel",
        )
        edge_toe = geom_by_name(
            edge_boxes_xml,
            f"{body_name}_task023_edge_boxes_no_points_toe",
        )
        assert_edge_box(edge_heel, "heel")
        assert_edge_box(edge_toe, "toe")
        for index in range(4):
            support = geom_by_name(edge_boxes_xml, f"{support_prefix}_support_{index}")
            assert support.get("contype") == "0"
            assert support.get("conaffinity") == "0"

        hybrid_center = geom_by_name(
            hybrid_edge_boxes_xml,
            f"{body_name}_task023_hybrid_center_pad",
        )
        assert hybrid_center.get("type") == "box"
        assert hybrid_center.get("size") == patcher.BOX_SUPPORT_SIZE
        assert hybrid_center.get("pos") == patcher.BOX_SUPPORT_POS
        hybrid_heel = geom_by_name(
            hybrid_edge_boxes_xml,
            f"{body_name}_task023_hybrid_edge_boxes_heel",
        )
        hybrid_toe = geom_by_name(
            hybrid_edge_boxes_xml,
            f"{body_name}_task023_hybrid_edge_boxes_toe",
        )
        assert_edge_box(hybrid_heel, "heel")
        assert_edge_box(hybrid_toe, "toe")
        for index in range(4):
            support = geom_by_name(hybrid_edge_boxes_xml, f"{support_prefix}_support_{index}")
            assert support.get("contype") == "0"
            assert support.get("conaffinity") == "0"

    for variant, report in written_summary["variants"].items():
        assert Path(report["path"]).is_file(), variant
        assert report["changed_geom_count"] > 0
        assert report["missing"] == []
        assert report["errors"] == []


def test_missing_target_body_and_support_geoms_are_reported() -> None:
    root = fresh_test_dir("missing")
    source = root / "missing.xml"
    source.write_text(
        """<mujoco>
  <worldbody>
    <body name="left_ankle_roll_link">
      <geom name="left_ankle_roll_link_visual" type="mesh" contype="0" conaffinity="0" />
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    args = patcher.parse_args(
        [
            "--source-asset",
            str(source),
            "--output-root",
            str(root / "outputs"),
            "--run-id",
            "missing",
            "--variants",
            "ankle_roll_friction_attrs",
        ]
    )

    summary = patcher.run_patch_generation(args)

    assert {
        "variant": "ankle_roll_friction_attrs",
        "body": "left_ankle_roll_link",
        "path": "bodies.left_ankle_roll_link.support_geoms",
        "reason": "support_geom_absent",
    } in summary["missing"]
    assert {
        "variant": "ankle_roll_friction_attrs",
        "body": "right_ankle_roll_link",
        "path": "bodies.right_ankle_roll_link.body",
        "reason": "body_absent",
    } in summary["missing"]
    assert summary["variants"]["ankle_roll_friction_attrs"]["changed_geom_count"] == 0


def test_relative_compiler_meshdir_is_rewritten_without_mutating_source() -> None:
    root = fresh_test_dir("relative_meshdir")
    source = write_fixture_xml(root / "robot" / "g1.xml", meshdir="../meshes")
    source_before = source.read_text(encoding="utf-8")
    expected_meshdir = str((source.parent / "../meshes").resolve())
    args = patcher.parse_args(
        [
            "--source-asset",
            str(source),
            "--output-root",
            str(root / "outputs"),
            "--run-id",
            "patch",
            "--variants",
            "ankle_roll_friction_attrs",
        ]
    )

    summary = patcher.run_patch_generation(args)

    assert source.read_text(encoding="utf-8") == source_before
    assert summary["source_unchanged"] is True
    assert summary["meshdir_handling"] == {
        "compiler_present": True,
        "source_meshdir": "../meshes",
        "resolved_source_meshdir": expected_meshdir,
        "output_meshdir": expected_meshdir,
        "rewritten": True,
    }
    variant = summary["variants"]["ankle_roll_friction_attrs"]
    assert variant["meshdir_handling"] == summary["meshdir_handling"]
    output_root = ElementTree.parse(variant["path"]).getroot()
    output_compiler = output_root.find("compiler")
    assert output_compiler is not None
    assert output_compiler.get("meshdir") == expected_meshdir

    source_root = ElementTree.parse(source).getroot()
    source_compiler = source_root.find("compiler")
    assert source_compiler is not None
    assert source_compiler.get("meshdir") == "../meshes"


def test_unknown_variant_is_rejected() -> None:
    root = fresh_test_dir("unknown")
    source = write_fixture_xml(root / "g1.xml")
    args = patcher.parse_args(
        [
            "--source-asset",
            str(source),
            "--output-root",
            str(root / "outputs"),
            "--run-id",
            "bad",
            "--variants",
            "unknown",
        ]
    )

    with pytest.raises(ValueError, match="unknown variants"):
        patcher.run_patch_generation(args)


def write_fixture_xml(path: Path, *, meshdir: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh_root = (path.parent / meshdir).resolve() if meshdir is not None else path.parent
    mesh_root.mkdir(parents=True, exist_ok=True)
    for name in ("left.obj", "right.obj"):
        (mesh_root / name).write_text(
            """v -0.06 -0.04 -0.03
v -0.06 0.04 -0.03
v 0.14 -0.04 -0.03
v 0.14 0.04 0.02
""",
            encoding="utf-8",
        )
    compiler = f'  <compiler meshdir="{meshdir}" />\n' if meshdir is not None else ""
    path.write_text(
        f"""<mujoco>
{compiler}  <asset>
    <mesh name="left_ankle_roll_link_mesh" file="left.obj" />
    <mesh name="right_ankle_roll_link_mesh" file="right.obj" />
  </asset>
  <worldbody>
    <body name="left_ankle_roll_link">
      <inertial mass="0.608" pos="0 0 0" />
      <geom name="left_ankle_roll_link_visual" type="mesh"
            mesh="left_ankle_roll_link_mesh" contype="0" conaffinity="0" />
      <geom name="left_support_0" type="sphere" size="0.005" pos="0.03 0.02 -0.01" />
      <geom name="left_support_1" type="sphere" size="0.005" pos="0.03 -0.02 -0.01" />
      <geom name="left_support_2" type="sphere" size="0.005" pos="-0.03 0.02 -0.01" />
      <geom name="left_support_3" type="sphere" size="0.005" pos="-0.03 -0.02 -0.01" />
    </body>
    <body name="right_ankle_roll_link">
      <inertial mass="0.608" pos="0 0 0" />
      <geom name="right_ankle_roll_link_visual" type="mesh"
            mesh="right_ankle_roll_link_mesh" contype="0" conaffinity="0" />
      <geom name="right_support_0" type="sphere" size="0.005" pos="0.03 0.02 -0.01" />
      <geom name="right_support_1" type="sphere" size="0.005" pos="0.03 -0.02 -0.01" />
      <geom name="right_support_2" type="sphere" size="0.005" pos="-0.03 0.02 -0.01" />
      <geom name="right_support_3" type="sphere" size="0.005" pos="-0.03 -0.02 -0.01" />
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return path


def fresh_test_dir(name: str) -> Path:
    root = (Path.cwd() / "outputs" / ".test_tmp_task022" / f"{name}_{uuid4().hex}").resolve()
    root.mkdir(parents=True)
    return root


def parse_variant(summary: dict[str, object], variant: str) -> ElementTree.Element:
    path = Path(summary["variants"][variant]["path"])
    return ElementTree.parse(path).getroot()


def find_body(root: ElementTree.Element, body_name: str) -> ElementTree.Element | None:
    return root.find(f".//body[@name='{body_name}']")


def geom_by_name(root: ElementTree.Element, geom_name: str) -> ElementTree.Element:
    geom = root.find(f".//geom[@name='{geom_name}']")
    assert geom is not None
    return geom


def assert_edge_box(geom: ElementTree.Element, label: str) -> None:
    expected_x = {"heel": "-0.05", "toe": "0.12"}[label]
    assert geom.get("type") == "box"
    assert geom.get("size") == patcher.EDGE_BOX_SIZE
    assert geom.get("pos") == f"{expected_x} 0 {patcher.EDGE_BOX_Z}"
    assert geom.get("friction") == "1.0 0.02 0.001"
    assert geom.get("condim") == "4"
    assert geom.get("priority") == "1"
    assert geom.get("contype") == "1"
    assert geom.get("conaffinity") == "1"


def support_geoms_for_body(
    root: ElementTree.Element,
    body_name: str,
    *,
    size: str = "0.005",
) -> list[ElementTree.Element]:
    body = find_body(root, body_name)
    assert body is not None
    return [
        geom
        for geom in body.findall("geom")
        if geom.get("type", "sphere") == "sphere" and geom.get("size") == size
    ]
