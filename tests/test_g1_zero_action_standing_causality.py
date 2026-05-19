import inspect
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from h200_locomotion_lab.robots import G1_27DOF_NOHAND_ACTUATOR_ORDER
from h200_locomotion_lab.tools import g1_zero_action_standing_causality as probe


def test_parse_args_defaults_to_task019_contract() -> None:
    args = probe.parse_args([])

    assert args.n_envs == 1024
    assert args.chunks == 50
    assert args.chunk_steps == 32
    assert args.warmup_policy_steps == 0
    assert args.pre_eval_reset is False
    assert args.pre_eval_reset_scope == "full"
    assert args.seed == 0
    assert args.control_mode == "genesis_position"
    assert args.pose_profile == "current"
    assert args.gain_profile == "current"
    assert args.physical_gpu == "1"
    assert args.logical_cuda_device == "cuda:0"
    assert args.output_root == probe.DEFAULT_OUTPUT_ROOT
    assert args.asset_path is None


def test_profile_with_asset_path_replaces_nested_asset_without_mutating_original() -> None:
    profile = DataclassProfile(asset=DataclassAsset(path="/source/g1_27dof_nohand.xml"))

    overridden = probe.profile_with_asset_path(profile, Path("/patched/variant.xml"))

    assert overridden is not profile
    assert overridden.asset is not profile.asset
    assert profile.asset.path == "/source/g1_27dof_nohand.xml"
    assert overridden.asset.path == "/patched/variant.xml"


def test_parse_args_warmup_policy_steps_must_be_nonnegative() -> None:
    assert probe.parse_args(["--warmup-policy-steps", "0"]).warmup_policy_steps == 0
    assert probe.parse_args(["--warmup-policy-steps", "3"]).warmup_policy_steps == 3
    with pytest.raises(SystemExit):
        probe.parse_args(["--warmup-policy-steps", "-1"])


def test_parse_args_pre_eval_reset_flag_defaults_false_and_sets_true() -> None:
    assert probe.parse_args([]).pre_eval_reset is False
    assert probe.parse_args(["--pre-eval-reset"]).pre_eval_reset is True


def test_parse_args_pre_eval_reset_scope_choices() -> None:
    assert probe.parse_args([]).pre_eval_reset_scope == "full"
    assert (
        probe.parse_args(["--pre-eval-reset-scope", "full"]).pre_eval_reset_scope
        == "full"
    )
    assert (
        probe.parse_args(["--pre-eval-reset-scope", "all_env_ids"]).pre_eval_reset_scope
        == "all_env_ids"
    )
    with pytest.raises(SystemExit):
        probe.parse_args(["--pre-eval-reset-scope", "bad"])


def test_parse_args_accepts_control_modes_and_pose_profiles() -> None:
    for control_mode in probe.CONTROL_MODES:
        args = probe.parse_args(["--control-mode", control_mode])
        assert args.control_mode == control_mode
    for pose_profile in probe.POSE_PROFILES:
        args = probe.parse_args(["--pose-profile", pose_profile])
        assert args.pose_profile == pose_profile
    for gain_profile in probe.GAIN_PROFILES:
        args = probe.parse_args(["--gain-profile", gain_profile])
        assert args.gain_profile == gain_profile


def test_pose_profile_values_current_and_unitree_gym() -> None:
    current = tuple(0.01 * index for index in range(27))
    index = G1_27DOF_NOHAND_ACTUATOR_ORDER.index

    current_profile = probe.pose_profile_values("current", current)
    assert current_profile[index("left_hip_pitch_joint")] == pytest.approx(-0.06)
    assert current_profile[index("right_hip_pitch_joint")] == pytest.approx(-0.06)
    assert current_profile[index("left_knee_joint")] == pytest.approx(0.12)
    assert current_profile[index("right_knee_joint")] == pytest.approx(0.12)
    assert current_profile[index("left_ankle_pitch_joint")] == pytest.approx(-0.07)
    assert current_profile[index("right_ankle_pitch_joint")] == pytest.approx(-0.07)
    assert current_profile[index("waist_yaw_joint")] == pytest.approx(
        current[index("waist_yaw_joint")]
    )
    assert current_profile[index("left_shoulder_pitch_joint")] == pytest.approx(
        current[index("left_shoulder_pitch_joint")]
    )

    unitree = probe.pose_profile_values("unitree_gym", current)
    assert len(unitree) == 27
    assert unitree[index("left_hip_pitch_joint")] == pytest.approx(-0.1)
    assert unitree[index("right_knee_joint")] == pytest.approx(0.3)
    assert unitree[index("left_ankle_pitch_joint")] == pytest.approx(-0.2)
    assert unitree[index("left_shoulder_roll_joint")] == pytest.approx(0.2)
    assert unitree[index("right_shoulder_roll_joint")] == pytest.approx(-0.2)
    assert unitree[index("waist_yaw_joint")] == pytest.approx(0.0)


def test_probe_source_has_no_forbidden_update_path_strings() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")
    forbidden = (
        "actor" + "-critic",
        "collect" + "_rollout",
        "compute" + "_gae",
        "ppo" + "_update",
        "g1_curriculum" + "_ppo_smoke",
    )
    for item in forbidden:
        assert item not in source
        assert not hasattr(probe, item)
    assert "build_actor" not in source
    assert "run_chunk" in inspect.getsource(probe.run_probe)


def test_gain_profile_values_anchor_on_27dof_nohand_order() -> None:
    index = G1_27DOF_NOHAND_ACTUATOR_ORDER.index

    current = probe.gain_profile_values("current", FakeControl())
    assert current.kp == FakeControl.kp
    assert current.kv == FakeControl.kv
    assert current.force_limits == FakeControl.force_limits

    kv_2x = probe.gain_profile_values("global_kv_2x", FakeControl())
    assert kv_2x.kp == FakeControl.kp
    assert kv_2x.kv[index("left_hip_pitch_joint")] == pytest.approx(2.0)
    assert kv_2x.kv[index("right_wrist_yaw_joint")] == pytest.approx(2.0)

    kv_4x = probe.gain_profile_values("global_kv_4x", FakeControl())
    assert kv_4x.kv[index("left_hip_pitch_joint")] == pytest.approx(4.0)

    soft_stiffness = probe.gain_profile_values("global_kp_0_5x_kv_2x", FakeControl())
    assert soft_stiffness.kp[index("left_knee_joint")] == pytest.approx(5.0)
    assert soft_stiffness.kv[index("left_knee_joint")] == pytest.approx(2.0)

    ankle = probe.gain_profile_values("ankle_kp_2x_kv_2x", FakeControl())
    assert ankle.kp[index("left_ankle_pitch_joint")] == pytest.approx(20.0)
    assert ankle.kv[index("right_ankle_roll_joint")] == pytest.approx(2.0)
    assert ankle.kp[index("left_knee_joint")] == pytest.approx(10.0)

    knee_ankle = probe.gain_profile_values("knee_ankle_kp_2x_kv_2x", FakeControl())
    assert knee_ankle.kp[index("left_knee_joint")] == pytest.approx(20.0)
    assert knee_ankle.kv[index("right_knee_joint")] == pytest.approx(2.0)
    assert knee_ankle.kp[index("left_hip_pitch_joint")] == pytest.approx(10.0)

    unitree = probe.gain_profile_values("unitree_leg_gains", FakeControl())
    assert unitree.kp[index("left_hip_pitch_joint")] == pytest.approx(100.0)
    assert unitree.kv[index("right_hip_yaw_joint")] == pytest.approx(2.0)
    assert unitree.kp[index("left_knee_joint")] == pytest.approx(150.0)
    assert unitree.kv[index("right_knee_joint")] == pytest.approx(4.0)
    assert unitree.kp[index("left_ankle_pitch_joint")] == pytest.approx(40.0)
    assert unitree.kv[index("right_ankle_roll_joint")] == pytest.approx(2.0)
    assert unitree.kp[index("waist_yaw_joint")] == pytest.approx(10.0)

    force_limit = probe.gain_profile_values("force_limit_2x", FakeControl())
    assert force_limit.kp == FakeControl.kp
    assert force_limit.kv == FakeControl.kv
    assert force_limit.force_limits[index("left_hip_pitch_joint")] == pytest.approx(200.0)
    assert force_limit.force_limits[index("right_wrist_yaw_joint")] == pytest.approx(200.0)


def test_apply_gain_profile_to_fake_backend_and_robot() -> None:
    backend = GainOnlyBackend()
    gains = probe.gain_profile_values("force_limit_2x", FakeControl())

    probe.apply_gain_profile_to_backend(backend, gains)

    assert probe.diagnostic_gains_for_backend(backend) is gains
    assert backend.robot.kp_calls == [(gains.kp, backend.motor_dof_indices)]
    assert backend.robot.kv_calls == [(gains.kv, backend.motor_dof_indices)]
    assert backend.robot.force_range_calls == [
        (
            tuple(-value for value in gains.force_limits),
            gains.force_limits,
            backend.motor_dof_indices,
        )
    ]


def test_custom_pd_torque_uses_diagnostic_gains() -> None:
    torch = NumpyTorch()
    backend = GainOnlyBackend()
    backend.n_envs = 1
    gains = probe.gain_profile_values("unitree_leg_gains", FakeControl())
    probe.apply_gain_profile_to_backend(backend, gains)
    index = G1_27DOF_NOHAND_ACTUATOR_ORDER.index
    joint_index = index("left_knee_joint")
    dof_pos = NumpyTensor(np.zeros((1, 27), dtype=np.float32))
    dof_vel = NumpyTensor(np.zeros((1, 27), dtype=np.float32))
    state = SimpleNamespace(dof_pos=dof_pos, dof_vel=dof_vel)
    targets = NumpyTensor(np.zeros((1, 27), dtype=np.float32))
    targets.array[:, joint_index] += 0.1

    torque = probe.compute_pd_torque(
        torch=torch,
        backend=backend,
        targets=targets,
        state=state,
    )

    assert torque["raw"][0, joint_index].item() == pytest.approx(15.0)
    assert torque["clipped"][0, joint_index].item() == pytest.approx(15.0)
    assert torque["saturated"][0, joint_index].item() is False


def test_summarize_run_pass_and_fail() -> None:
    args = probe.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--chunks",
            "2",
        ]
    )
    pass_rows = [
        metric_row(chunk_index=0, reset_count=0, tilt_bad_count=0),
        metric_row(chunk_index=1, reset_count=0, tilt_bad_count=0),
    ]
    fail_rows = [
        metric_row(chunk_index=0, reset_count=0, tilt_bad_count=0),
        metric_row(chunk_index=1, reset_count=3, tilt_bad_count=2),
    ]

    passed = probe.summarize_run(rows=pass_rows, args=args, run_dir=Path("run"))
    failed = probe.summarize_run(rows=fail_rows, args=args, run_dir=Path("run"))

    assert passed["status"] == "passed"
    assert passed["passed"] is True
    assert passed["evaluation_passed"] is True
    assert passed["diagnostic_passed"] is True
    assert passed["pre_eval_reset"] is False
    assert passed["pre_eval_reset_scope"] == "full"
    assert passed["first_tilt_chunk"] is None
    assert failed["status"] == "failed"
    assert failed["passed"] is False
    assert failed["evaluation_passed"] is False
    assert failed["diagnostic_passed"] is False
    assert failed["first_tilt_chunk"] == 1
    assert failed["first_tilt_step"] == 32


def test_summarize_run_distinguishes_warmup_diagnostic_pass() -> None:
    args = probe.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--chunks",
            "1",
            "--warmup-policy-steps",
            "2",
        ]
    )
    rows = [metric_row(chunk_index=0, reset_count=0, tilt_bad_count=0)]

    summary = probe.summarize_run(rows=rows, args=args, run_dir=Path("run"))

    assert summary["status"] == "diagnostic_passed"
    assert summary["evaluation_passed"] is True
    assert summary["diagnostic_passed"] is False
    assert summary["passed"] is False
    assert summary["warmup_policy_steps"] == 2
    assert summary["pre_eval_reset"] is False


def test_summarize_run_distinguishes_pre_eval_reset_diagnostic_pass() -> None:
    args = probe.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--chunks",
            "1",
            "--pre-eval-reset",
        ]
    )
    rows = [metric_row(chunk_index=0, reset_count=0, tilt_bad_count=0)]

    summary = probe.summarize_run(rows=rows, args=args, run_dir=Path("run"))

    assert summary["status"] == "diagnostic_passed"
    assert summary["evaluation_passed"] is True
    assert summary["diagnostic_passed"] is False
    assert summary["passed"] is False
    assert summary["pre_eval_reset"] is True
    assert summary["pre_eval_reset_scope"] == "full"


def test_main_diagnostic_failure_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = probe.parse_args([])
    monkeypatch.setattr(probe, "parse_args", lambda: args)
    monkeypatch.setattr(
        probe,
        "run_probe",
        lambda args: {
            "status": "failed",
            "passed": False,
            "first_tilt_chunk": 0,
            "max_reset_count": 2,
        },
    )

    probe.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["blocker"] == ""
    assert payload["passed"] is False
    assert payload["max_reset_count"] == 2


def test_main_execution_error_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_probe(args: object) -> dict[str, object]:
        raise RuntimeError("boom")

    args = probe.parse_args([])
    monkeypatch.setattr(probe, "parse_args", lambda: args)
    monkeypatch.setattr(probe, "run_probe", fail_probe)

    with pytest.raises(SystemExit) as exc_info:
        probe.main()

    payload = json.loads(capsys.readouterr().out)
    assert exc_info.value.code == 1
    assert payload["status"] == "error"
    assert payload["blocker"] == "RuntimeError:boom"


def test_fake_control_mode_call_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        probe,
        "compute_pd_torque",
        lambda **kwargs: {
            "clipped": kwargs["targets"],
            "saturated": [[False] * 27 for _ in range(kwargs["backend"].n_envs)],
        },
    )
    for control_mode in probe.CONTROL_MODES:
        backend = ListFakeBackend(n_envs=2, decimation=4)
        action = [[0.0] * 27 for _ in range(2)]

        result = probe.apply_control_mode(
            torch=MiniTorch(),
            backend=backend,
            action=action,
            mode=control_mode,
        )

        if control_mode == "genesis_position":
            assert backend.position_calls == 1
            assert backend.robot.force_calls == 0
        elif control_mode == "genesis_position_resend_physics":
            assert backend.position_calls == backend.decimation
            assert backend.robot.force_calls == 0
        else:
            assert backend.position_calls == 0
            assert backend.robot.force_calls == backend.decimation
            assert result["kind"] == "torque"
        assert backend.scene.step_count == backend.decimation
        assert backend.step_count == 1


def test_run_probe_writes_artifacts_with_fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    tmp_path = fresh_test_dir("artifacts")
    monkeypatch.setattr(probe, "PROJECT_PREFIX", tmp_path)
    monkeypatch.setattr(probe, "load_g1_27dof_nohand_profile", lambda: FakeProfile())

    class RuntimeFakeBackend(FakeBackend):
        def __init__(self, config: object, *, profile: object) -> None:
            super().__init__(
                torch=torch,
                n_envs=config.n_envs,
                decimation=2,
                default_positions=config.default_positions_rad,
            )
            self.config = config
            self.profile = profile

    monkeypatch.setattr(probe, "VectorizedGenesisBackend", RuntimeFakeBackend)
    args = probe.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--n-envs",
            "2",
            "--chunks",
            "2",
            "--chunk-steps",
            "2",
            "--control-mode",
            "custom_pd_torque",
            "--pose-profile",
            "unitree_gym",
            "--gain-profile",
            "global_kv_2x",
            "--output-root",
            str(tmp_path / "outputs"),
            "--run-id",
            "fake",
        ]
    )

    summary = probe.run_probe(args)

    run_dir = tmp_path / "outputs" / "fake"
    assert summary["status"] == "passed"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "summary.json").is_file()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["control_mode"] == "custom_pd_torque"
    assert config["pose_profile"] == "unitree_gym"
    assert config["gain_profile"] == "global_kv_2x"
    assert config["pre_eval_reset"] is False
    assert config["pre_eval_reset_scope"] == "full"
    assert config["asset_path"] == "default.xml"
    assert config["gain_values"]["left_knee_joint"]["kv"] == pytest.approx(2.0)
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["chunk_index"] for row in rows] == [0, 1]
    assert rows[0]["gain_profile"] == "global_kv_2x"
    assert summary["gain_profile"] == "global_kv_2x"
    assert summary["asset_path"] == "default.xml"
    assert summary["pre_eval_reset"] is False
    assert summary["pre_eval_reset_scope"] == "full"
    assert {"joint_position_error_rms", "joint_velocity_rms", "control_rms"} <= rows[0].keys()
    cleanup_test_dir(tmp_path)


def test_run_probe_warmup_records_diagnostics_without_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    tmp_path = fresh_test_dir("warmup")
    monkeypatch.setattr(probe, "PROJECT_PREFIX", tmp_path)
    monkeypatch.setattr(probe, "load_g1_27dof_nohand_profile", lambda: FakeProfile())
    backend_holder = {}

    class RuntimeWarmupBackend(WarmupTiltBackend):
        def __init__(self, config: object, *, profile: object) -> None:
            super().__init__(
                torch=torch,
                n_envs=config.n_envs,
                decimation=1,
                default_positions=config.default_positions_rad,
            )
            self.config = config
            self.profile = profile
            backend_holder["backend"] = self

    monkeypatch.setattr(probe, "VectorizedGenesisBackend", RuntimeWarmupBackend)
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
            "2",
            "--warmup-policy-steps",
            "2",
            "--output-root",
            str(tmp_path / "outputs"),
            "--run-id",
            "fake-warmup",
        ]
    )

    summary = probe.run_probe(args)

    backend = backend_holder["backend"]
    run_dir = tmp_path / "outputs" / "fake-warmup"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    warmup = summary["warmup_diagnostics"]
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert backend.reset_calls == [None]
    assert config["warmup_policy_steps"] == 2
    assert warmup["policy_steps"] == 2
    assert warmup["tilt_bad_count"] == 4
    assert warmup["termination_height_bad_count"] == 4
    assert warmup["root_height_min"] == pytest.approx(0.1)
    assert warmup["upright_min"] == pytest.approx(0.0)
    assert warmup["joint_position_error_rms"] >= 0.0
    assert warmup["joint_velocity_rms"] >= 0.0
    assert warmup["force_saturation_ratio"] == pytest.approx(0.0)
    assert rows[0]["tilt_bad_count"] == 0
    assert summary["status"] == "diagnostic_passed"
    assert summary["evaluation_passed"] is True
    assert summary["diagnostic_passed"] is False
    assert summary["passed"] is False
    cleanup_test_dir(tmp_path)


def test_run_probe_pre_eval_reset_records_one_extra_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    tmp_path = fresh_test_dir("pre_eval_reset")
    monkeypatch.setattr(probe, "PROJECT_PREFIX", tmp_path)
    monkeypatch.setattr(probe, "load_g1_27dof_nohand_profile", lambda: FakeProfile())
    backend_holder = {}

    class RuntimeWarmupBackend(WarmupTiltBackend):
        def __init__(self, config: object, *, profile: object) -> None:
            super().__init__(
                torch=torch,
                n_envs=config.n_envs,
                decimation=1,
                default_positions=config.default_positions_rad,
            )
            self.config = config
            self.profile = profile
            backend_holder["backend"] = self

    monkeypatch.setattr(probe, "VectorizedGenesisBackend", RuntimeWarmupBackend)
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
            "2",
            "--warmup-policy-steps",
            "2",
            "--pre-eval-reset",
            "--output-root",
            str(tmp_path / "outputs"),
            "--run-id",
            "fake-pre-eval-reset",
        ]
    )

    summary = probe.run_probe(args)

    backend = backend_holder["backend"]
    run_dir = tmp_path / "outputs" / "fake-pre-eval-reset"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert backend.reset_calls == [None, None]
    assert config["pre_eval_reset"] is True
    assert config["pre_eval_reset_scope"] == "full"
    assert summary["pre_eval_reset"] is True
    assert summary["pre_eval_reset_scope"] == "full"
    assert summary["status"] == "diagnostic_passed"
    assert summary["evaluation_passed"] is True
    assert summary["diagnostic_passed"] is False
    assert summary["passed"] is False
    cleanup_test_dir(tmp_path)


def test_run_probe_pre_eval_reset_all_env_ids_uses_selected_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    tmp_path = fresh_test_dir("pre_eval_reset_all_env_ids")
    monkeypatch.setattr(probe, "PROJECT_PREFIX", tmp_path)
    monkeypatch.setattr(probe, "load_g1_27dof_nohand_profile", lambda: FakeProfile())
    backend_holder = {}

    class RuntimeWarmupBackend(WarmupTiltBackend):
        def __init__(self, config: object, *, profile: object) -> None:
            super().__init__(
                torch=torch,
                n_envs=config.n_envs,
                decimation=1,
                default_positions=config.default_positions_rad,
            )
            self.config = config
            self.profile = profile
            backend_holder["backend"] = self

    monkeypatch.setattr(probe, "VectorizedGenesisBackend", RuntimeWarmupBackend)
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
            "2",
            "--warmup-policy-steps",
            "2",
            "--pre-eval-reset",
            "--pre-eval-reset-scope",
            "all_env_ids",
            "--output-root",
            str(tmp_path / "outputs"),
            "--run-id",
            "fake-pre-eval-reset-all-env-ids",
        ]
    )

    summary = probe.run_probe(args)

    backend = backend_holder["backend"]
    run_dir = tmp_path / "outputs" / "fake-pre-eval-reset-all-env-ids"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    env_ids = backend.reset_calls[1]
    assert backend.reset_calls[0] is None
    assert env_ids is not None
    assert env_ids.tolist() == [0, 1]
    assert str(env_ids.device) == "cpu"
    assert config["pre_eval_reset"] is True
    assert config["pre_eval_reset_scope"] == "all_env_ids"
    assert summary["pre_eval_reset"] is True
    assert summary["pre_eval_reset_scope"] == "all_env_ids"
    assert summary["evaluation_passed"] is True
    assert summary["diagnostic_passed"] is False
    assert summary["passed"] is False
    cleanup_test_dir(tmp_path)


def test_resolve_run_dir_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    project = Path.cwd() / ".test_tmp_task019" / "project"
    monkeypatch.setattr(probe, "PROJECT_PREFIX", project)

    run_dir = probe.resolve_run_dir(project / "outputs", "run-a")

    assert run_dir == (project / "outputs" / "run-a").resolve()
    with pytest.raises(RuntimeError, match="output path must stay under"):
        probe.resolve_run_dir(Path.cwd() / ".test_tmp_task019_elsewhere", "run-a")


def metric_row(
    *,
    chunk_index: int,
    reset_count: int,
    tilt_bad_count: int,
) -> dict[str, object]:
    return {
        "chunk_index": chunk_index,
        "chunk_steps": 32,
        "reset_count": reset_count,
        "tilt_bad_count": tilt_bad_count,
        "termination_height_bad_count": tilt_bad_count,
        "root_height_mean": 0.78,
        "root_height_min": 0.78,
        "upright_mean": 1.0,
        "joint_position_error_rms": 0.0,
        "joint_velocity_rms": 0.0,
        "throughput_env_steps_per_sec": 100.0,
        "tensor_device_ok": True,
    }


def fresh_test_dir(name: str) -> Path:
    root = (Path.cwd() / ".test_tmp_task019" / name).resolve()
    cleanup_test_dir(root)
    root.mkdir(parents=True)
    return root


def cleanup_test_dir(path: Path) -> None:
    workspace = Path.cwd().resolve()
    resolved = path.resolve()
    if workspace not in (resolved, *resolved.parents):
        raise RuntimeError(f"refusing to clean path outside workspace: {resolved}")
    if ".test_tmp_task019" not in resolved.parts:
        raise RuntimeError(f"refusing to clean non-test path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


class FakeControl:
    default_angles_rad = (0.0,) * 27
    kp = (10.0,) * 27
    kv = (1.0,) * 27
    force_limits = (100.0,) * 27


class FakeProfile:
    action_dim = 27
    actuator_order = G1_27DOF_NOHAND_ACTUATOR_ORDER
    asset = SimpleNamespace(path="default.xml")
    control = FakeControl()


@dataclass(frozen=True)
class DataclassAsset:
    path: str


@dataclass(frozen=True)
class DataclassProfile:
    asset: DataclassAsset


class FakeScene:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1


class FakeRobot:
    def __init__(self) -> None:
        self.force_calls = 0
        self.kp_calls: list[tuple[tuple[float, ...], tuple[int, ...]]] = []
        self.kv_calls: list[tuple[tuple[float, ...], tuple[int, ...]]] = []
        self.force_range_calls: list[
            tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...]]
        ] = []

    def control_dofs_force(self, values: object, dofs_idx_local: tuple[int, ...]) -> None:
        self.force_calls += 1

    def set_dofs_kp(self, values: tuple[float, ...], dofs_idx_local: tuple[int, ...]) -> None:
        self.kp_calls.append((values, dofs_idx_local))

    def set_dofs_kv(self, values: tuple[float, ...], dofs_idx_local: tuple[int, ...]) -> None:
        self.kv_calls.append((values, dofs_idx_local))

    def set_dofs_force_range(
        self,
        lower: tuple[float, ...],
        upper: tuple[float, ...],
        dofs_idx_local: tuple[int, ...],
    ) -> None:
        self.force_range_calls.append((lower, upper, dofs_idx_local))


class GainOnlyBackend:
    def __init__(self) -> None:
        self.motor_dof_indices = tuple(range(27))
        self.motor_kp_mult = 1.0
        self.motor_kv_mult = 1.0
        self.motor_force_limit_mult = 1.0
        self.config = SimpleNamespace(logical_cuda_device="cpu")
        self.profile = FakeProfile()
        self.robot = FakeRobot()


class MiniTorch:
    bool = bool

    def zeros_like(self, values: list[list[float]], dtype: object | None = None) -> list[list[bool]]:
        return [[False for _ in row] for row in values]


class NumpyTorch:
    float32 = np.float32

    def tensor(self, values: object, *, dtype: object, device: str) -> "NumpyTensor":
        return NumpyTensor(np.array(values, dtype=dtype))


class NumpyTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def unsqueeze(self, axis: int) -> "NumpyTensor":
        return NumpyTensor(np.expand_dims(self.array, axis))

    def clamp(self, min_value: object, max_value: object) -> "NumpyTensor":
        lower = min_value.array if isinstance(min_value, NumpyTensor) else min_value
        upper = max_value.array if isinstance(max_value, NumpyTensor) else max_value
        return NumpyTensor(np.clip(self.array, lower, upper))

    def abs(self) -> "NumpyTensor":
        return NumpyTensor(np.abs(self.array))

    def item(self) -> object:
        return self.array.item()

    def __getitem__(self, key: object) -> "NumpyTensor":
        return NumpyTensor(np.asarray(self.array[key]))

    def __add__(self, other: object) -> "NumpyTensor":
        return self._binary(other, np.add)

    def __sub__(self, other: object) -> "NumpyTensor":
        return self._binary(other, np.subtract)

    def __rsub__(self, other: object) -> "NumpyTensor":
        other_array = other.array if isinstance(other, NumpyTensor) else other
        return NumpyTensor(np.subtract(other_array, self.array))

    def __mul__(self, other: object) -> "NumpyTensor":
        return self._binary(other, np.multiply)

    def __rmul__(self, other: object) -> "NumpyTensor":
        return self.__mul__(other)

    def __neg__(self) -> "NumpyTensor":
        return NumpyTensor(-self.array)

    def __ge__(self, other: object) -> "NumpyTensor":
        return self._binary(other, np.greater_equal)

    def _binary(self, other: object, operation: object) -> "NumpyTensor":
        other_array = other.array if isinstance(other, NumpyTensor) else other
        return NumpyTensor(operation(self.array, other_array))


class ListFakeBackend:
    def __init__(self, *, n_envs: int, decimation: int) -> None:
        self.n_envs = n_envs
        self.decimation = decimation
        self.action_dim = 27
        self.motor_dof_indices = tuple(range(27))
        self.default_positions_values = (0.0,) * 27
        self.scene = FakeScene()
        self.robot = FakeRobot()
        self.position_calls = 0
        self.step_count = 0
        self.previous_action = [[0.0] * 27 for _ in range(n_envs)]

    def _coerce_action(self, action: list[list[float]]) -> list[list[float]]:
        return action

    def _clip_action(self, action: list[list[float]]) -> list[list[float]]:
        return [[max(-1.0, min(1.0, value)) for value in row] for row in action]

    def _action_targets(self, action: list[list[float]]) -> list[list[float]]:
        return [[0.0 for _ in row] for row in action]

    def _control_dofs_position(self, targets: object) -> None:
        self.position_calls += 1

    def step_physics(self, action: list[list[float]]) -> list[list[float]]:
        clipped = self._clip_action(self._coerce_action(action))
        targets = self._action_targets(clipped)
        self._control_dofs_position(targets)
        for _ in range(self.decimation):
            self.scene.step()
        self.previous_action = clipped
        self.step_count += 1
        return clipped

    def state(self) -> SimpleNamespace:
        return SimpleNamespace()


class FakeBackend:
    def __init__(
        self,
        *,
        torch: object,
        n_envs: int,
        decimation: int,
        default_positions: tuple[float, ...] | None = None,
    ) -> None:
        self.torch = torch
        self.n_envs = n_envs
        self.decimation = decimation
        self.action_dim = 27
        self.motor_dof_indices = tuple(range(27))
        self.motor_kp_mult = 1.0
        self.motor_kv_mult = 1.0
        self.motor_force_limit_mult = 1.0
        self.config = SimpleNamespace(logical_cuda_device="cpu")
        self.profile = FakeProfile()
        values = default_positions or (0.0,) * 27
        self.default_positions_values = tuple(values)
        self.default_positions = torch.tensor(values, dtype=torch.float32, device="cpu")
        self.scene = FakeScene()
        self.robot = FakeRobot()
        self.position_calls = 0
        self.force_calls = 0
        self.reset_calls: list[object | None] = []
        self.step_count = 0
        self.previous_action = torch.zeros((n_envs, 27), device="cpu")
        self.dof_pos = self.default_positions.unsqueeze(0).repeat(n_envs, 1)
        self.dof_vel = torch.zeros((n_envs, 27), device="cpu")

    def reset(self, env_ids: object | None = None) -> object:
        self.reset_calls.append(env_ids)
        return self.torch.zeros((self.n_envs, 90), device="cpu")

    def _coerce_action(self, action: object) -> object:
        return action

    def _clip_action(self, action: object) -> object:
        return action.clamp(-1.0, 1.0)

    def _action_targets(self, action: object) -> object:
        return self.default_positions.unsqueeze(0) + action * 0.0

    def _control_dofs_position(self, targets: object) -> None:
        self.position_calls += 1
        self.dof_pos = targets

    def step_physics(self, action: object) -> object:
        clipped = self._clip_action(self._coerce_action(action))
        targets = self._action_targets(clipped)
        self._control_dofs_position(targets)
        for _ in range(self.decimation):
            self.scene.step()
        self.previous_action = clipped
        self.step_count += 1
        return clipped

    def state(self) -> SimpleNamespace:
        root_pos = self.torch.zeros((self.n_envs, 3), device="cpu")
        root_pos[:, 2] = 0.78
        root_quat = self.torch.zeros((self.n_envs, 4), device="cpu")
        root_quat[:, 0] = 1.0
        return SimpleNamespace(
            root_pos=root_pos,
            root_quat=root_quat,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
        )


class WarmupTiltBackend(FakeBackend):
    def state(self) -> SimpleNamespace:
        state = super().state()
        if self.step_count <= 2:
            state.root_pos[:, 2] = 0.1
            state.root_quat[:, 0] = 0.0
            state.root_quat[:, 1] = 1.0
        return state
