import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from h200_locomotion_lab.tools import g1_standing_semantics_matrix as probe


def test_default_matrix_scenarios_are_layered() -> None:
    args = probe.parse_args(["--backend", "cpu", "--logical-cuda-device", "cpu"])
    scenarios = probe.parse_scenarios(args.scenarios)

    assert [scenario.name for scenario in scenarios] == list(probe.DEFAULT_SCENARIOS)
    assert [scenario.layer for scenario in scenarios[:3]] == [
        "baseline",
        "control",
        "control",
    ]
    assert probe.SCENARIOS["control_custom_pd_torque"].args == (
        "--control-mode",
        "custom_pd_torque",
    )
    assert probe.SCENARIOS["root_z_1_00"].args == ("--root-z", "1.00")


def test_build_zero_action_command_adds_common_and_scenario_args() -> None:
    args = probe.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--n-envs",
            "4",
            "--chunks",
            "2",
            "--chunk-steps",
            "3",
        ]
    )

    command = probe.build_zero_action_command(
        args=args,
        scenario=probe.SCENARIOS["gain_unitree_leg"],
        run_id="matrix-gain",
    )

    assert command[:3] == [
        probe.sys.executable,
        "-m",
        "h200_locomotion_lab.tools.g1_zero_action_standing_causality",
    ]
    assert command[command.index("--n-envs") + 1] == "4"
    assert command[command.index("--run-id") + 1] == "matrix-gain"
    assert command[-2:] == ["--gain-profile", "unitree_leg_gains"]


def test_parse_probe_stdout_uses_last_json_line() -> None:
    parsed = probe.parse_probe_stdout(
        "noise\n"
        "{\"status\":\"old\"}\n"
        "more noise\n"
        "{\"status\":\"passed\",\"first_tilt_step\":null}\n"
    )

    assert parsed == {"status": "passed", "first_tilt_step": None}


def test_run_scenario_captures_subprocess_summary(
    monkeypatch,
) -> None:
    run_dir = fresh_test_dir("run_scenario")
    args = probe.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--n-envs",
            "4",
            "--chunks",
            "1",
            "--chunk-steps",
            "1",
            "--output-root",
            str(run_dir / "outputs"),
            "--scenario-timeout-s",
            "1",
        ]
    )

    def fake_run(command, **kwargs):
        assert command[2] == "h200_locomotion_lab.tools.g1_zero_action_standing_causality"
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "failed",
                    "blocker": "",
                    "first_tilt_step": 64,
                    "max_reset_count": 4,
                    "final_root_height_min": 0.5,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(probe.subprocess, "run", fake_run)

    result = probe.run_scenario(
        args=args,
        run_dir=run_dir,
        scenario=probe.SCENARIOS["baseline_current"],
    )

    assert result["status"] == "failed"
    assert result["key_metrics"]["first_tilt_step"] == 64
    assert (run_dir / "baseline_current.json").is_file()


def test_rank_results_prefers_later_tilt_then_fewer_resets() -> None:
    results = [
        {
            "name": "early",
            "layer": "control",
            "status": "failed",
            "key_metrics": {
                "evaluation_passed": False,
                "first_tilt_step": 64,
                "max_reset_count": 512,
                "final_root_height_min": 0.6,
            },
        },
        {
            "name": "late",
            "layer": "pose",
            "status": "failed",
            "key_metrics": {
                "evaluation_passed": False,
                "first_tilt_step": 128,
                "max_reset_count": 64,
                "final_root_height_min": 0.7,
            },
        },
    ]

    assert probe.rank_results(results)[0]["name"] == "late"


def test_run_matrix_captures_scenario_exception_and_continues(monkeypatch) -> None:
    run_dir = fresh_test_dir("run_matrix_error")
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
            "matrix",
            "--scenarios",
            "baseline_current,control_resend_physics",
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
            "key_metrics": {"first_tilt_step": 64, "max_reset_count": 1},
        }

    monkeypatch.setattr(probe, "run_scenario", fake_run_scenario)

    summary = probe.run_matrix(args)

    assert calls == ["baseline_current", "control_resend_physics"]
    assert summary["status"] == "completed"
    assert [row["status"] for row in summary["scenarios"]] == ["error", "failed"]
    assert summary["scenarios"][0]["blocker"] == "TimeoutError:timed out"
    assert summary["scenarios"][0]["key_metrics"] == {}
    assert (run_dir / "matrix" / "baseline_current.json").is_file()
    assert (run_dir / "matrix" / "summary.json").is_file()


def test_rank_results_places_error_empty_metrics_after_measured_failure() -> None:
    results = [
        {
            "name": "error",
            "layer": "control",
            "status": "error",
            "key_metrics": {},
        },
        {
            "name": "measured",
            "layer": "baseline",
            "status": "failed",
            "key_metrics": {
                "evaluation_passed": False,
                "first_tilt_step": 64,
                "max_reset_count": 512,
                "final_root_height_min": 0.6,
            },
        },
    ]

    ranked = probe.rank_results(results)

    assert [row["name"] for row in ranked] == ["measured", "error"]


def fresh_test_dir(name: str) -> Path:
    root = (Path.cwd() / ".test_tmp_task021" / f"{name}_{uuid4().hex}").resolve()
    root.mkdir(parents=True)
    return root
