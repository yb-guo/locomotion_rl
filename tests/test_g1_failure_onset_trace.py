import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from h200_locomotion_lab.tools import g1_failure_onset_trace as probe


def test_default_trace_scenarios_and_one_step_defaults() -> None:
    args = probe.parse_args(["--backend", "cpu", "--logical-cuda-device", "cpu"])
    scenarios = probe.parse_scenarios(args.scenarios)

    assert [scenario.name for scenario in scenarios] == list(probe.DEFAULT_SCENARIOS)
    assert args.chunks == 96
    assert args.chunk_steps == 1
    assert probe.SCENARIOS["combo_custom_pd_unitree_leg"].args == (
        "--control-mode",
        "custom_pd_torque",
        "--gain-profile",
        "unitree_leg_gains",
    )


def test_build_zero_action_command_adds_combo_args() -> None:
    args = probe.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--n-envs",
            "4",
            "--chunks",
            "128",
            "--chunk-steps",
            "1",
            "--asset-path",
            "outputs/task022/assets/patched.xml",
        ]
    )

    command = probe.build_zero_action_command(
        args=args,
        scenario=probe.SCENARIOS["combo_custom_pd_unitree_leg"],
        run_id="trace-combo",
    )

    assert command[:3] == [
        probe.sys.executable,
        "-m",
        "h200_locomotion_lab.tools.g1_zero_action_standing_causality",
    ]
    assert command[command.index("--chunks") + 1] == "128"
    assert command[command.index("--chunk-steps") + 1] == "1"
    assert command[command.index("--run-id") + 1] == "trace-combo"
    assert command[command.index("--asset-path") + 1] == "outputs/task022/assets/patched.xml"
    assert command[-4:] == [
        "--control-mode",
        "custom_pd_torque",
        "--gain-profile",
        "unitree_leg_gains",
    ]


def test_timeline_focus_selects_rows_around_first_tilt() -> None:
    rows = [
        metric_row(step=step, tilt_bad_count=0, root_height_min=0.8 - step * 0.01)
        for step in range(1, 8)
    ]
    rows[4]["tilt_bad_count"] = 512
    rows[4]["top_joint_position_error_rms"] = [{"joint": "left_knee_joint", "rms": 0.2}]

    focus = probe.timeline_focus(rows=rows, summary={"first_tilt_step": 4})

    assert focus["first_tilt_row"]["total_policy_steps"] == 5
    assert focus["first_tilt_row"]["tilt_bad_count"] == 512
    assert focus["first_tilt_row"]["top_joint_position_error_rms"] == [
        {"joint": "left_knee_joint", "rms": 0.2}
    ]
    assert [row["total_policy_steps"] for row in focus["rows_around_first_tilt"]] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert focus["max_contact_force"] == 10.0
    assert focus["max_contact_count"] == 3
    assert focus["max_force_saturation_ratio"] == 0.25


def test_first_tilt_row_falls_back_to_tilt_count_when_summary_step_is_missing() -> None:
    rows = [
        metric_row(step=1, tilt_bad_count=0, root_height_min=0.8),
        metric_row(step=2, tilt_bad_count=512, root_height_min=0.6),
    ]

    first = probe.first_tilt_row(rows=rows, summary={})

    assert first is rows[1]


def test_run_scenario_reads_metrics_jsonl_and_writes_summary(monkeypatch) -> None:
    run_dir = fresh_test_dir("run_scenario")
    output_root = run_dir / "zero_action_outputs"
    args = probe.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--output-root",
            str(output_root),
            "--run-id",
            "trace",
            "--scenario-timeout-s",
            "1",
        ]
    )
    captured_metrics_paths: list[Path] = []
    rows = [
        metric_row(step=1, tilt_bad_count=0, root_height_min=0.78),
        metric_row(step=2, tilt_bad_count=512, root_height_min=0.62),
    ]

    def fake_run(command, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "failed",
                    "blocker": "",
                    "first_tilt_step": 2,
                    "max_reset_count": 512,
                    "final_root_height_min": 0.62,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    monkeypatch.setattr(
        probe,
        "read_jsonl",
        lambda path: captured_metrics_paths.append(path) or rows,
    )

    result = probe.run_scenario(
        args=args,
        run_dir=run_dir,
        scenario=probe.SCENARIOS["baseline_current"],
    )

    assert result["status"] == "failed"
    assert result["key_metrics"]["first_tilt_step"] == 2
    assert result["timeline_focus"]["first_tilt_row"]["total_policy_steps"] == 2
    assert captured_metrics_paths == [
        output_root / f"{run_dir.name}-baseline_current" / "metrics.jsonl"
    ]
    assert (run_dir / "baseline_current.json").is_file()


def test_run_trace_captures_scenario_exception_and_continues(monkeypatch) -> None:
    run_dir = fresh_test_dir("run_trace_error")
    calls: list[str] = []
    args = probe.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--output-root",
            str(run_dir),
            "--run-id",
            "trace",
            "--scenarios",
            "baseline_current,control_custom_pd_torque",
        ]
    )

    def fake_run_scenario(*, args, run_dir, scenario):
        calls.append(scenario.name)
        if scenario.name == "baseline_current":
            raise TimeoutError("timed out")
        return {
            "name": scenario.name,
            "layer": scenario.layer,
            "status": "failed",
            "blocker": "",
            "key_metrics": {"first_tilt_step": 64},
            "timeline_focus": {},
        }

    monkeypatch.setattr(probe, "run_scenario", fake_run_scenario)

    summary = probe.run_trace(args)

    assert calls == ["baseline_current", "control_custom_pd_torque"]
    assert summary["status"] == "completed"
    assert [row["status"] for row in summary["scenarios"]] == ["error", "failed"]
    assert summary["scenarios"][0]["blocker"] == "TimeoutError:timed out"
    assert (run_dir / "trace" / "baseline_current.json").is_file()
    assert (run_dir / "trace" / "summary.json").is_file()


def metric_row(*, step: int, tilt_bad_count: int, root_height_min: float) -> dict[str, object]:
    return {
        "chunk_index": step - 1,
        "total_policy_steps": step,
        "reset_count": 512 if tilt_bad_count else 0,
        "height_bad_count": 0,
        "termination_height_bad_count": 0,
        "tilt_bad_count": tilt_bad_count,
        "root_height_mean": root_height_min + 0.02,
        "root_height_min": root_height_min,
        "upright_mean": 0.9,
        "upright_min": 0.8,
        "joint_position_error_rms": 0.1,
        "joint_position_error_max": 0.2,
        "joint_velocity_rms": 0.3,
        "joint_velocity_max": 0.4,
        "control_kind": "position",
        "control_rms": 0.5,
        "control_max": 0.6,
        "force_saturation_ratio": 0.25 if step == 5 else 0.0,
        "foot_or_body_contact_count": 3 if step == 5 else 1,
        "max_contact_force": 10.0 if step == 5 else 2.0,
    }


def fresh_test_dir(name: str) -> Path:
    root = (Path.cwd() / ".test_tmp_task021" / f"{name}_{uuid4().hex}").resolve()
    root.mkdir(parents=True)
    return root
