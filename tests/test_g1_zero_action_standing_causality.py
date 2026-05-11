import inspect
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from h200_locomotion_lab.robots import G1_27DOF_NOHAND_ACTUATOR_ORDER
from h200_locomotion_lab.tools import g1_zero_action_standing_causality as probe


def test_parse_args_defaults_to_task019_contract() -> None:
    args = probe.parse_args([])

    assert args.n_envs == 1024
    assert args.chunks == 50
    assert args.chunk_steps == 32
    assert args.seed == 0
    assert args.control_mode == "genesis_position"
    assert args.pose_profile == "current"
    assert args.physical_gpu == "1"
    assert args.logical_cuda_device == "cuda:0"
    assert args.output_root == probe.DEFAULT_OUTPUT_ROOT


def test_parse_args_accepts_control_modes_and_pose_profiles() -> None:
    for control_mode in probe.CONTROL_MODES:
        args = probe.parse_args(["--control-mode", control_mode])
        assert args.control_mode == control_mode
    for pose_profile in probe.POSE_PROFILES:
        args = probe.parse_args(["--pose-profile", pose_profile])
        assert args.pose_profile == pose_profile


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
    assert passed["first_tilt_chunk"] is None
    assert failed["status"] == "failed"
    assert failed["passed"] is False
    assert failed["first_tilt_chunk"] == 1
    assert failed["first_tilt_step"] == 32


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
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["chunk_index"] for row in rows] == [0, 1]
    assert {"joint_position_error_rms", "joint_velocity_rms", "control_rms"} <= rows[0].keys()
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
    control = FakeControl()


class FakeScene:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1


class FakeRobot:
    def __init__(self) -> None:
        self.force_calls = 0

    def control_dofs_force(self, values: object, dofs_idx_local: tuple[int, ...]) -> None:
        self.force_calls += 1


class MiniTorch:
    bool = bool

    def zeros_like(self, values: list[list[float]], dtype: object | None = None) -> list[list[bool]]:
        return [[False for _ in row] for row in values]


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
        self.step_count = 0
        self.previous_action = torch.zeros((n_envs, 27), device="cpu")
        self.dof_pos = self.default_positions.unsqueeze(0).repeat(n_envs, 1)
        self.dof_vel = torch.zeros((n_envs, 27), device="cpu")

    def reset(self, env_ids: object | None = None) -> object:
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
