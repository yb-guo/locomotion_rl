import json
from pathlib import Path

from h200_locomotion_lab.tools.g1_genesis_alignment_bundle import (
    build_alignment_bundle,
    main,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "g1_genesis_alignment_bundle"
MISSING_ASSET = FIXTURE_DIR / "does_not_exist.xml"
MINIMAL_XML = FIXTURE_DIR / "minimal_no_contact.xml"


def test_alignment_bundle_emits_stable_top_level_json(capsys) -> None:
    main(["--asset-path", str(MISSING_ASSET)])
    data = json.loads(capsys.readouterr().out)

    assert list(data) == [
        "alignment_status",
        "contact_friction_solver",
        "control_timing_comparison",
        "genesis_27dof_training_profile",
        "mapped_control_comparison",
        "missing",
        "root_pose_defaults",
        "sonic_29dof_profile",
        "vectorized_genesis_backend_contact_solver_config",
    ]
    assert data["sonic_29dof_profile"]["dof_count"] == 29
    assert data["genesis_27dof_training_profile"]["dof_count"] == 27


def test_mapped_29dof_sonic_control_matches_27dof_genesis_profile() -> None:
    report = build_alignment_bundle(asset_path=MISSING_ASSET)
    comparison = report["mapped_control_comparison"]

    assert comparison["removed_joints"] == ["waist_roll_joint", "waist_pitch_joint"]
    assert comparison["joint_order_match"] is True
    assert comparison["all_match"] is True
    for field_name in (
        "default_angles_rad",
        "action_scales_rad",
        "kp",
        "kv",
        "force_limits",
    ):
        field = comparison["fields"][field_name]
        assert field["match"] is True
        assert field["mapped_sonic"] == field["genesis"]
        assert field["mismatches"] == []
        assert len(field["mapped_sonic"]) == 27


def test_missing_xml_and_contact_fields_are_reported() -> None:
    report = build_alignment_bundle(asset_path=MISSING_ASSET)

    assert report["contact_friction_solver"]["asset_present"] is False
    assert report["contact_friction_solver"]["asset_path"] == str(MISSING_ASSET)
    assert {
        "path": "contact_friction_solver.asset_path",
        "reason": "asset_missing",
    } in report["missing"]
    assert {
        "path": "contact_friction_solver.contact",
        "reason": "asset_missing",
    } in report["missing"]
    assert {
        "path": "contact_friction_solver.option.timestep",
        "reason": "asset_missing",
    } in report["missing"]
    assert {
        "path": "genesis_27dof_training_profile.contact_friction_solver_config",
        "reason": "not_represented_in_profile",
    } in report["missing"]
    assert {
        "path": "vectorized_genesis_backend.rigid_contact_solver",
        "reason": "unset_defaults",
    } in report["missing"]
    assert all(
        item["path"] != "vectorized_genesis_backend.contact_friction_solver_config"
        for item in report["missing"]
    )


def test_present_xml_reports_absent_contact_values_without_inference() -> None:
    report = build_alignment_bundle(asset_path=MINIMAL_XML)

    assert report["contact_friction_solver"]["asset_present"] is True
    assert report["contact_friction_solver"]["compiler"] is None
    assert report["contact_friction_solver"]["option"] is None
    assert report["contact_friction_solver"]["contact"] is None
    assert {
        "path": "contact_friction_solver.geoms_with_contact_fields",
        "reason": "xml_field_absent",
    } in report["missing"]


def test_backend_runtime_defaults_are_represented() -> None:
    report = build_alignment_bundle(asset_path=MISSING_ASSET)
    backend_defaults = report["root_pose_defaults"]["backend_vectorized_genesis_config"]

    assert backend_defaults["root_qpos"] == [0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0]
    assert backend_defaults["default_positions_rad"] is None
    assert backend_defaults["uses_profile_default_positions_when_none"] is True
    assert backend_defaults["action_scale_mult"] == 1.0
    assert backend_defaults["action_joint_group"] == "all"
    assert backend_defaults["motor_kp_mult"] == 1.0
    assert backend_defaults["motor_kv_mult"] == 1.0
    assert backend_defaults["motor_force_limit_mult"] == 1.0
    assert backend_defaults["rigid_contact_solver_configured"] is False
    assert report["root_pose_defaults"]["sonic_planner"]["context_root_qpos"] == [
        0.0,
        0.0,
        0.78874,
        1.0,
        0.0,
        0.0,
        0.0,
    ]


def test_alignment_json_records_unset_backend_contact_solver_defaults() -> None:
    report = build_alignment_bundle(asset_path=MISSING_ASSET)
    config = report["vectorized_genesis_backend_contact_solver_config"]

    assert config["boundary"] == "gs.Scene(rigid_options=gs.options.RigidOptions(...))"
    assert config["configured"] is False
    assert config["requested_rigid_options"] == {}
    assert "iterations" in config["unset_fields"]
    assert "enable_mujoco_compatibility" in config["unset_fields"]
    assert config["missing"] == [
        {
            "path": "vectorized_genesis_backend.rigid_contact_solver",
            "reason": "unset_defaults",
        }
    ]


def test_control_timing_comparison_reports_genesis_and_missing_sonic_mjcf_timing() -> None:
    report = build_alignment_bundle(asset_path=MISSING_ASSET)
    timing = report["control_timing_comparison"]

    assert timing["genesis_training_contract"] == {
        "sim_dt_s": 0.005,
        "decimation": 4,
        "policy_rate_hz": 50,
        "derived_policy_rate_hz": 50.0,
        "self_consistent": True,
    }
    assert timing["vectorized_genesis_backend"]["sim_dt_source"] == (
        "profile.training_contract.sim_dt_s"
    )
    assert timing["sonic_29dof_profile"] == {
        "sim_dt_s": None,
        "decimation": None,
        "policy_rate_hz": None,
        "source": "not_represented_in_profile",
    }
    assert timing["comparison"] == {
        "genesis_vs_mjcf_timestep_match": None,
        "sonic_profile_timing_present": False,
        "mjcf_decimation_present": False,
    }
    assert {
        "path": "sonic_29dof_profile.training_contract.policy_rate_hz",
        "reason": "not_represented_in_profile",
    } in report["missing"]
    assert {
        "path": "mjcf.control_decimation",
        "reason": "not_represented_in_mjcf",
    } in report["missing"]


def test_default_remote_asset_path_is_preserved_in_local_report() -> None:
    report = build_alignment_bundle()

    assert report["contact_friction_solver"]["asset_path"].startswith("/root/")
