import inspect
import json
from pathlib import Path
from uuid import uuid4

import pytest

from h200_locomotion_lab.tools import g1_base_attitude_height_stabilization as probe


def test_parse_args_exposes_task023_modes_and_metadata() -> None:
    args = probe.parse_args(
        [
            "--mode",
            "attitude_height",
            "--steps",
            "32",
            "--seed",
            "7",
            "--asset-path",
            "outputs/task023/assets/g1.xml",
            "--asset-variant-label",
            "ankle_roll_larger_spheres",
            "--asset-source-path",
            "/source/g1.xml",
            "--output-root",
            "outputs/task023/probe",
            "--run-name",
            "local",
        ]
    )

    assert set(probe.STABILIZER_MODES) == {"none", "attitude", "height", "attitude_height"}
    assert args.runner == "local_toy"
    assert args.mode == "attitude_height"
    assert args.steps == 32
    assert args.seed == 7
    assert args.asset_path == Path("outputs/task023/assets/g1.xml")
    assert args.asset_variant_label == "ankle_roll_larger_spheres"
    assert args.asset_source_path == Path("/source/g1.xml")
    assert args.output_root == Path("outputs/task023/probe")
    assert args.run_name == "local"


def test_module_import_path_does_not_pull_genesis_backend() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")
    top_level = source.split("def run_genesis_probe", maxsplit=1)[0]

    assert "vectorized_genesis_backend" not in top_level
    assert "GenesisG1SceneBackend" not in source
    assert "genesis" not in inspect.getsource(probe.run_toy_rollout).lower()


def test_controller_mode_and_gain_clipping_are_bounded() -> None:
    args = probe.parse_args(
        [
            "--mode",
            "attitude_height",
            "--attitude-kp",
            "99",
            "--height-kp",
            "21",
            "--max-gain",
            "5",
            "--max-joint-delta",
            "0.03",
        ]
    )
    requested = probe.StabilizerGains(
        attitude_kp=args.attitude_kp,
        attitude_kd=args.attitude_kd,
        height_kp=args.height_kp,
        height_kd=args.height_kd,
        max_joint_delta=args.max_joint_delta,
    )
    gains = probe.clip_gains(requested, max_gain=args.max_gain)
    state = probe.ToyState(
        step=10,
        root_height=0.40,
        root_height_velocity=-0.2,
        roll=0.5,
        pitch=-0.4,
        roll_velocity=0.1,
        pitch_velocity=-0.1,
    )

    output = probe.compute_controller_output(
        mode=args.mode,
        gains=gains,
        state=state,
        target_height=args.target_height,
    )

    assert gains.attitude_kp == pytest.approx(5.0)
    assert gains.height_kp == pytest.approx(5.0)
    assert output.clipped is True
    assert output.max_abs_delta <= 0.03 + 1e-12
    assert output.roll_delta == pytest.approx(-0.03)
    assert output.pitch_delta == pytest.approx(0.03)
    assert output.height_delta == pytest.approx(0.03)


def test_local_none_rollout_reports_first_tilt_reset_and_schema() -> None:
    root = fresh_test_dir("none")
    args = probe.parse_args(
        [
            "--mode",
            "none",
            "--steps",
            "180",
            "--seed",
            "0",
            "--asset-path",
            "source.xml",
            "--asset-variant-label",
            "source",
            "--output-root",
            str(root),
            "--run-name",
            "none",
        ]
    )

    summary = probe.run_probe(args)

    run_dir = root / "none"
    assert summary["status"] == "completed"
    assert summary["effective_asset_path"] == "source.xml"
    assert summary["asset_metadata"]["variant_label"] == "source"
    assert summary["stabilizer"]["mode"] == "none"
    assert summary["improvement_classification"] == "baseline"
    assert summary["first_tilt_step"] is not None
    assert summary["first_reset_step"] == summary["first_tilt_step"]
    assert summary["root_height_timeline_summary"]["min"] < 0.78
    assert summary["upright_timeline_summary"]["final"] < summary["upright_timeline_summary"]["initial"]
    assert summary["top_joint_errors"]
    assert {"ankle_roll", "ankle_pitch"} <= set(summary["contact_trace_summary"])
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "summary.json").is_file()
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 180
    assert {"ankle_roll_contact_force", "ankle_pitch_contact_force", "joint_errors"} <= set(
        rows[0]
    )


def test_local_attitude_height_improves_over_local_baseline() -> None:
    root = fresh_test_dir("attitude_height")
    args = probe.parse_args(
        [
            "--mode",
            "attitude_height",
            "--steps",
            "180",
            "--seed",
            "0",
            "--asset-path",
            "larger.xml",
            "--asset-variant-label",
            "ankle_roll_larger_spheres",
            "--asset-source-path",
            "source.xml",
            "--max-joint-delta",
            "0.05",
            "--output-root",
            str(root),
            "--run-name",
            "attitude-height",
        ]
    )

    summary = probe.run_probe(args)

    assert summary["stabilizer"]["mode"] == "attitude_height"
    assert summary["asset_metadata"]["source_path"] == "source.xml"
    assert summary["baseline_first_reset_step"] is not None
    assert summary["first_reset_step"] is None
    assert summary["improvement_classification"] == "physical_stability"
    assert summary["stabilizer"]["clipping"]["clipped_steps"] > 0
    top_joints = {row["joint"] for row in summary["top_joint_errors"]}
    assert top_joints & {
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_knee_joint",
        "right_knee_joint",
    }
    assert summary["contact_trace_summary"]["ankle_roll"]["max_force"] > 0.0
    assert summary["contact_trace_summary"]["ankle_pitch"]["active_steps"] == 180


def test_summary_json_option_writes_requested_file() -> None:
    root = fresh_test_dir("summary_json")
    summary_json = root / "summary-copy.json"
    args = probe.parse_args(
        [
            "--mode",
            "height",
            "--steps",
            "24",
            "--asset-path",
            "source.xml",
            "--output-root",
            str(root),
            "--run-name",
            "height",
            "--summary-json",
            str(summary_json),
        ]
    )

    summary = probe.run_probe(args)

    copied = json.loads(summary_json.read_text(encoding="utf-8"))
    assert copied["run_dir"] == summary["run_dir"]
    assert copied["stabilizer"]["mode"] == "height"


def test_genesis_command_uses_guarded_wrapper_without_running_h200() -> None:
    args = probe.parse_args(
        [
            "--runner",
            "genesis",
            "--mode",
            "attitude",
            "--steps",
            "96",
            "--asset-path",
            "/root/project/assets/g1.xml",
            "--run-name",
            "candidate",
        ]
    )

    command = probe.build_h200_genesis_command(args)

    assert command.startswith("/root/agent_workspace/safe_agent/run_guarded.sh bash -lc ")
    assert "h200-locomotion-lab-task023-base-attitude-height-stabilization" in command
    assert "h200_locomotion_lab.tools.g1_base_attitude_height_stabilization" in command
    assert "--runner genesis" in command


def fresh_test_dir(name: str) -> Path:
    root = (Path.cwd() / "outputs" / "task023" / ".test_tmp" / f"{name}_{uuid4().hex}").resolve()
    root.mkdir(parents=True)
    return root
