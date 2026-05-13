import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from h200_locomotion_lab.tools import g1_ankle_foot_asset_contact_audit as probe


def test_parse_targets_and_default_args() -> None:
    args = probe.parse_args([])

    assert args.run_link_trace is False
    assert args.n_envs == 512
    assert args.steps == 100
    assert args.pose_profile == "current"
    assert probe.parse_targets("a,b, c ,,") == ["a", "b", "c"]
    assert probe.parse_targets(args.target_bodies) == list(probe.DEFAULT_TARGET_BODIES)


def test_xml_audit_reports_inertial_geom_contact_attrs_and_absent_fields() -> None:
    xml_path = write_fixture_xml()

    report = probe.audit_asset_xml(
        asset_path=xml_path,
        target_bodies=[
            "left_ankle_pitch_link",
            "left_ankle_roll_link",
            "right_ankle_pitch_link",
            "right_ankle_roll_link",
        ],
    )

    left_pitch = report["bodies"]["left_ankle_pitch_link"]
    left_roll = report["bodies"]["left_ankle_roll_link"]
    assert report["asset_present"] is True
    assert left_pitch["inertial"] == {
        "mass": "1.2",
        "pos": "0 0 0",
        "diaginertia": "1 2 3",
    }
    assert left_pitch["direct_geoms"][0]["contact_attrs"] == {
        "condim": "3",
        "friction": "0.8 0.1 0.1",
        "contype": "1",
        "conaffinity": "1",
    }
    assert left_roll["direct_geoms"] == []
    assert {
        "path": "bodies.left_ankle_roll_link.direct_geoms",
        "reason": "geom_absent",
    } in report["missing"]
    assert {
        "path": "bodies.left_ankle_pitch_link.direct_geoms[0].solref",
        "reason": "xml_field_absent",
    } in report["missing"]
    assert report["symmetry"][0]["left"] == "left_ankle_pitch_link"
    assert report["symmetry"][0]["right"] == "right_ankle_pitch_link"
    assert report["symmetry"][0]["inertial_match"] is True
    assert report["symmetry"][1]["both_present"] is True


def test_run_audit_writes_asset_and_summary_without_trace(monkeypatch) -> None:
    xml_path = write_fixture_xml()
    run_root = fresh_test_dir("run_audit")
    monkeypatch.setattr(
        probe,
        "load_g1_27dof_nohand_profile",
        lambda: SimpleNamespace(asset=SimpleNamespace(path=str(xml_path))),
    )
    args = probe.parse_args(
        [
            "--output-root",
            str(run_root),
            "--run-id",
            "audit",
        ]
    )

    summary = probe.run_audit(args)

    run_dir = run_root / "audit"
    assert summary["status"] == "completed"
    assert summary["link_trace"] is None
    assert (run_dir / "asset_audit.json").is_file()
    assert (run_dir / "summary.json").is_file()
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["missing_count"] >= 1


def test_link_sample_and_trace_summary_use_fake_robot_arrays() -> None:
    robot = FakeRobot()
    link_idx = probe.resolve_link_index(robot, "left_ankle_pitch_link")
    sample = probe.read_link_sample(robot=robot, link_idx=link_idx, contact_threshold=4.0)

    assert link_idx == 2
    assert sample["z_min"] == 0.1
    assert sample["z_mean"] == 0.15000000000000002
    assert sample["contact_force_max"] == 5.0
    assert sample["contact_force_mean"] == 2.5
    assert sample["contact_env_count"] == 1

    rows = [
        {
            "policy_step": 1,
            "tilt_bad_count": 0,
            "termination_height_bad_count": 0,
            "links": {"left_ankle_pitch_link": sample},
        },
        {
            "policy_step": 2,
            "tilt_bad_count": 3,
            "termination_height_bad_count": 1,
            "links": {
                "left_ankle_pitch_link": {
                    **sample,
                    "z_mean": 0.05,
                    "contact_force_max": 7.0,
                    "contact_env_count": 2,
                }
            },
        },
    ]

    summary = probe.summarize_link_trace(
        rows=rows,
        link_indices={"left_ankle_pitch_link": link_idx, "missing": None},
    )

    assert summary["steps"] == 2
    assert summary["first_tilt_step"] == 2
    assert summary["max_tilt_bad_count"] == 3
    assert summary["links"]["left_ankle_pitch_link"]["z_min"] == 0.05
    assert summary["links"]["left_ankle_pitch_link"]["contact_force_max"] == 7.0
    assert summary["links"]["left_ankle_pitch_link"]["max_contact_env_count"] == 2
    assert summary["unresolved_links"] == ["missing"]


def test_link_sample_handles_genesis_like_nested_env_link_vectors() -> None:
    robot = NestedFakeRobot()

    sample = probe.read_link_sample(robot=robot, link_idx=2, contact_threshold=4.0)

    assert sample["z_min"] == 0.1
    assert sample["z_mean"] == 0.15000000000000002
    assert sample["contact_force_max"] == 5.0
    assert sample["contact_force_mean"] == 2.5
    assert sample["contact_env_count"] == 1


def test_link_sample_selects_requested_link_from_all_link_tensor() -> None:
    robot = AllLinksFakeRobot()

    sample = probe.read_link_sample(robot=robot, link_idx=1, contact_threshold=4.0)

    assert sample["z_min"] == 0.2
    assert sample["z_mean"] == 0.25
    assert sample["contact_force_max"] == 6.0
    assert sample["contact_force_mean"] == 3.0
    assert sample["contact_env_count"] == 1


def write_fixture_xml() -> Path:
    root = fresh_test_dir("xml")
    path = root / "g1.xml"
    path.write_text(
        """<mujoco>
  <worldbody>
    <body name="left_ankle_pitch_link">
      <inertial mass="1.2" pos="0 0 0" diaginertia="1 2 3" />
      <geom name="left_ankle_pitch_geom" type="box" friction="0.8 0.1 0.1" condim="3" contype="1" conaffinity="1" />
    </body>
    <body name="right_ankle_pitch_link">
      <inertial mass="1.2" pos="0 0 0" diaginertia="1 2 3" />
      <geom name="right_ankle_pitch_geom" type="box" friction="0.8 0.1 0.1" condim="3" contype="1" conaffinity="1" />
    </body>
    <body name="left_ankle_roll_link">
      <inertial mass="0.3" pos="0 0 0" />
    </body>
    <body name="right_ankle_roll_link">
      <inertial mass="0.3" pos="0 0 0" />
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return path


def fresh_test_dir(name: str) -> Path:
    root = (Path.cwd() / ".test_tmp_task021" / f"{name}_{uuid4().hex}").resolve()
    root.mkdir(parents=True)
    return root


class FakeLink:
    idx_local = 2


class FakeRobot:
    def get_link(self, name: str) -> FakeLink:
        if name != "left_ankle_pitch_link":
            raise KeyError(name)
        return FakeLink()

    def get_links_pos(self, links_idx_local: tuple[int, ...]) -> list[list[float]]:
        assert links_idx_local == (2,)
        return [[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]]

    def get_links_net_contact_force(self, links_idx_local: tuple[int, ...]) -> list[list[float]]:
        assert links_idx_local == (2,)
        return [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]


class NestedFakeRobot:
    def get_links_pos(self, links_idx_local: tuple[int, ...]) -> list[list[list[float]]]:
        assert links_idx_local == (2,)
        return [[[0.0, 0.0, 0.1]], [[0.0, 0.0, 0.2]]]

    def get_links_net_contact_force(
        self,
        links_idx_local: tuple[int, ...],
    ) -> list[list[list[float]]]:
        assert links_idx_local == (2,)
        return [[[3.0, 4.0, 0.0]], [[0.0, 0.0, 0.0]]]


class AllLinksFakeRobot:
    def get_links_pos(self, links_idx_local: tuple[int, ...]) -> list[list[list[float]]]:
        assert links_idx_local == (1,)
        return [
            [[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]],
            [[0.0, 0.0, 0.3], [0.0, 0.0, 0.3]],
        ]

    def get_links_net_contact_force(
        self,
        links_idx_local: tuple[int, ...],
    ) -> list[list[list[float]]]:
        assert links_idx_local == (1,)
        return [
            [[100.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            [[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
