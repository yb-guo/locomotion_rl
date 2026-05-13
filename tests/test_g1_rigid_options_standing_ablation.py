import json
from pathlib import Path
from uuid import uuid4

import pytest

from h200_locomotion_lab.robots import G1_27DOF_NOHAND_ACTUATOR_ORDER
from h200_locomotion_lab.tools import g1_rigid_options_standing_ablation as probe


def test_default_scenarios_and_rigid_option_payloads() -> None:
    args = probe.parse_args([])
    scenarios = probe.parse_scenarios(args.scenarios)

    assert probe.scenario_names(scenarios) == list(probe.DEFAULT_SCENARIOS)
    assert probe.scenario_payload(probe.SCENARIOS["default_unset"]) == {
        "name": "default_unset",
        "rigid_options_request": {},
    }
    assert probe.scenario_payload(probe.SCENARIOS["newton_solver_only"]) == {
        "name": "newton_solver_only",
        "rigid_options_request": {"constraint_solver": "Newton"},
    }
    assert probe.scenario_payload(probe.SCENARIOS["newton_mujoco_contact"])[
        "rigid_options_request"
    ] == {
        "constraint_solver": "Newton",
        "enable_mujoco_compatibility": True,
        "enable_multi_contact": True,
    }
    assert probe.scenario_payload(probe.SCENARIOS["newton_solver_bundle"])[
        "rigid_options_request"
    ] == {
        "constraint_solver": "Newton",
        "constraint_timeconst": 0.02,
        "enable_mujoco_compatibility": True,
        "enable_multi_contact": True,
        "iterations": 10,
        "ls_iterations": 10,
        "tolerance": 1e-6,
    }


def test_run_scenario_passes_rigid_config_into_vectorized_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = fresh_test_dir("run_scenario")
    captured: dict[str, object] = {}
    monkeypatch.setattr(probe, "load_g1_27dof_nohand_profile", lambda: FakeProfile())
    monkeypatch.setattr(probe.zero_action, "apply_gain_profile_to_backend", lambda *args: None)
    monkeypatch.setattr(probe.zero_action, "run_warmup", lambda **kwargs: {})
    monkeypatch.setattr(
        probe.zero_action,
        "run_chunk",
        lambda **kwargs: metric_row(chunk_index=kwargs["chunk_index"]),
    )
    monkeypatch.setattr(
        probe.zero_action,
        "summarize_run",
        lambda **kwargs: {
            "status": "passed",
            "passed": True,
            "evaluation_passed": True,
            "diagnostic_passed": True,
            "chunks_completed": 1,
            "chunks_expected": 1,
            "max_reset_count": 0,
            "final_root_height_mean": 0.78,
        },
    )

    class FakeVectorizedBackend:
        def __init__(self, config: object, *, profile: object) -> None:
            captured["config"] = config
            captured["profile"] = profile
            self.config = config
            self.profile = profile

        def contact_solver_config_report(self) -> dict[str, object]:
            return self.config.rigid_contact_solver.report()

        def reset(self) -> None:
            captured["reset"] = True

    monkeypatch.setattr(probe, "VectorizedGenesisBackend", FakeVectorizedBackend)
    args = probe.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--n-envs",
            "2",
            "--chunks",
            "1",
            "--chunk-steps",
            "1",
        ]
    )
    scenario = probe.SCENARIOS["newton_solver_bundle"]

    result = probe.run_scenario(
        args=args,
        torch=FakeTorch(),
        run_dir=run_dir,
        scenario=scenario,
    )

    config = captured["config"]
    assert config.rigid_contact_solver is scenario.rigid_contact_solver
    assert config.rigid_contact_solver.to_genesis_kwargs() == {
        "constraint_solver": "Newton",
        "enable_mujoco_compatibility": True,
        "enable_multi_contact": True,
        "iterations": 10,
        "tolerance": 1e-6,
        "ls_iterations": 10,
        "constraint_timeconst": 0.02,
    }
    assert result["name"] == "newton_solver_bundle"
    assert result["status"] == "passed"
    assert (run_dir / "newton_solver_bundle" / "config.json").is_file()
    assert (run_dir / "newton_solver_bundle" / "metrics.jsonl").is_file()
    config_payload = json.loads(
        (run_dir / "newton_solver_bundle" / "config.json").read_text(encoding="utf-8")
    )
    assert config_payload["scenario"]["rigid_options_request"] == (
        config.rigid_contact_solver.to_genesis_kwargs()
    )


def test_run_ablation_captures_scenario_error_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = fresh_test_dir("run_ablation")
    calls: list[str] = []
    monkeypatch.setattr(probe.zero_action, "verify_cuda_isolation", lambda **kwargs: None)
    monkeypatch.setattr(probe.zero_action, "require_torch", lambda: FakeTorch())
    monkeypatch.setattr(probe, "resolve_run_dir", lambda output_root, run_id: run_root / "run")

    def fake_run_scenario(**kwargs: object) -> dict[str, object]:
        scenario = kwargs["scenario"]
        calls.append(scenario.name)
        if scenario.name == "default_unset":
            raise RuntimeError("boom")
        return {
            "name": scenario.name,
            "rigid_options_request": scenario.rigid_contact_solver.to_genesis_kwargs(),
            "rigid_options_report": scenario.rigid_contact_solver.report(),
            "status": "passed",
            "blocker": "",
            "key_metrics": {"status": "passed"},
            "run_dir": str(kwargs["run_dir"] / scenario.name),
        }

    monkeypatch.setattr(probe, "run_scenario", fake_run_scenario)
    args = probe.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--scenarios",
            "default_unset,newton_solver_only",
        ]
    )

    summary = probe.run_ablation(args)

    assert calls == ["default_unset", "newton_solver_only"]
    assert summary["status"] == "completed"
    assert [item["status"] for item in summary["scenarios"]] == ["error", "passed"]
    assert summary["scenarios"][0]["blocker"] == "RuntimeError:boom"
    assert (run_root / "run" / "default_unset" / "config.json").is_file()
    assert (run_root / "run" / "default_unset" / "metrics.jsonl").is_file()
    assert (run_root / "run" / "default_unset" / "summary.json").is_file()
    assert (run_root / "run" / "summary.json").is_file()


def test_contact_summary_promotes_contact_metrics_to_key_metrics() -> None:
    summary = {
        "status": "failed",
        "passed": False,
        **probe.contact_summary(
            [
                {
                    "foot_or_body_contact_count": 3,
                    "max_contact_force": 10.0,
                },
                {
                    "foot_or_body_contact_count": 5,
                    "max_contact_force": 7.0,
                },
            ]
        ),
    }

    metrics = probe.key_metrics(summary)

    assert metrics["final_foot_or_body_contact_count"] == 5
    assert metrics["max_foot_or_body_contact_count"] == 5
    assert metrics["final_max_contact_force"] == 7.0
    assert metrics["max_contact_force"] == 10.0


def test_resolve_run_dir_enforces_project_prefix_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_prefix = fresh_test_dir("project_prefix")

    class FakeProjectPrefix:
        @staticmethod
        def resolve() -> Path:
            return project_prefix

    monkeypatch.setattr(probe.zero_action, "PROJECT_PREFIX", FakeProjectPrefix())

    inside_root = project_prefix / "h200-locomotion-lab-task021"
    inside = probe.resolve_run_dir(inside_root, "run")
    assert inside == inside_root / "run"
    with pytest.raises(RuntimeError, match="output path must stay under"):
        probe.resolve_run_dir(Path("/tmp/not-project"), "run")


def metric_row(*, chunk_index: int) -> dict[str, object]:
    return {
        "chunk_index": chunk_index,
        "chunk_steps": 1,
        "reset_count": 0,
        "tilt_bad_count": 0,
        "termination_height_bad_count": 0,
        "root_height_mean": 0.78,
        "root_height_min": 0.78,
        "upright_mean": 1.0,
        "joint_position_error_rms": 0.0,
        "joint_velocity_rms": 0.0,
        "foot_or_body_contact_count": 0,
        "max_contact_force": 0.0,
        "throughput_env_steps_per_sec": 100.0,
        "tensor_device_ok": True,
    }


def fresh_test_dir(name: str) -> Path:
    root = (Path.cwd() / ".test_tmp_task021" / f"{name}_{uuid4().hex}").resolve()
    root.mkdir(parents=True)
    return root


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def manual_seed_all(seed: int) -> None:
            return None

    def manual_seed(self, seed: int) -> None:
        return None

    def zeros(self, shape: tuple[int, int], *, device: str) -> list[list[float]]:
        rows, columns = shape
        return [[0.0 for _ in range(columns)] for _ in range(rows)]


class FakeControl:
    default_angles_rad = (0.0,) * 27
    kp = (10.0,) * 27
    kv = (1.0,) * 27
    force_limits = (100.0,) * 27


class FakeProfile:
    action_dim = 27
    actuator_order = G1_27DOF_NOHAND_ACTUATOR_ORDER
    control = FakeControl()
