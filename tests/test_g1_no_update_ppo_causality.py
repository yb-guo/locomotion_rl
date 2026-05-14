import inspect
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from h200_locomotion_lab.tools import g1_no_update_ppo_causality as probe
from h200_locomotion_lab.training import ppo_loop


def test_parse_args_defaults_to_task018_contract() -> None:
    args = probe.parse_args([])

    assert args.n_envs == 1024
    assert args.chunks == 50
    assert args.chunk_steps == 32
    assert args.seed == 0
    assert args.mode == "zero_action"
    assert args.log_std_init == -2.5
    assert args.action_scale_mult == 0.10
    assert args.root_z == 1.20
    assert args.default_pose == "tall_crouch"
    assert args.termination_height_min == 0.20
    assert args.output_root == probe.DEFAULT_OUTPUT_ROOT


def test_parse_args_accepts_all_action_modes() -> None:
    for mode in probe.ACTION_MODES:
        args = probe.parse_args(["--mode", mode])

        assert args.mode == mode


def test_parse_args_rejects_unknown_action_mode() -> None:
    with pytest.raises(SystemExit):
        probe.parse_args(["--mode", "ppo_update"])


def test_probe_module_does_not_import_update_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_update(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("PPO update helpers must not be called")

    monkeypatch.setattr(ppo_loop, "compute_gae", fail_update)
    monkeypatch.setattr(ppo_loop, "ppo_update", fail_update)

    source = Path(probe.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "g1_curriculum_ppo_smoke",
        "compute_gae",
        "ppo_update",
        "collect_rollout",
    ):
        assert forbidden not in source
        assert not hasattr(probe, forbidden)

    assert not hasattr(probe, "compute_gae")
    assert not hasattr(probe, "ppo_update")
    assert "compute_gae" not in inspect.getsource(probe.run_probe)
    assert "ppo_update" not in inspect.getsource(probe.run_probe)
    assert "compute_gae" not in inspect.getsource(probe.run_chunk)
    assert "ppo_update" not in inspect.getsource(probe.run_chunk)


def test_run_probe_writes_artifacts_without_update_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    tmp_path = fresh_test_dir("artifacts")
    install_fake_runtime(monkeypatch, torch)
    monkeypatch.setattr(probe, "PROJECT_PREFIX", tmp_path)

    def fail_update(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("PPO update helpers must not be called")

    monkeypatch.setattr(ppo_loop, "compute_gae", fail_update)
    monkeypatch.setattr(ppo_loop, "ppo_update", fail_update)
    args = small_args(
        tmp_path,
        "--run-id",
        "mean-action",
        "--mode",
        "untrained_mean_action",
    )

    summary = probe.run_probe(args)

    run_dir = tmp_path / "outputs" / "mean-action"
    assert summary["mode_passed"] is True
    assert summary["all_modes_passed"] is True
    assert summary["chunks_completed"] == 2
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "summary.json").is_file()

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["no_update"] is True
    assert config["mode"] == "untrained_mean_action"
    assert config["env"]["stage"] == "standing"
    assert config["env"]["action_scale_mult"] == 0.10
    assert config["env"]["action_joint_group"] == "all"
    assert config["physical_gpu"] == "1"
    assert config["logical_cuda_device"] == "cpu"

    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["chunk_index"] for row in rows] == [0, 1]
    assert all(row["mode"] == "untrained_mean_action" for row in rows)
    assert {"reset_count", "tilt_bad_count", "termination_height_bad_count"} <= rows[0].keys()
    assert {"root_height_mean", "root_height_min", "upright_mean"} <= rows[0].keys()
    assert {"action_abs_mean", "action_abs_max", "action_std"} <= rows[0].keys()
    assert {"throughput_env_steps_per_sec", "logical_cuda_device"} <= rows[0].keys()
    assert rows[0]["top_action_rms_joints"]
    cleanup_test_dir(tmp_path)


def test_zero_action_stats_are_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    tmp_path = fresh_test_dir("zero-action")
    install_fake_runtime(monkeypatch, torch)
    monkeypatch.setattr(probe, "PROJECT_PREFIX", tmp_path)
    args = small_args(tmp_path, "--run-id", "zero-action", "--mode", "zero_action")

    summary = probe.run_probe(args)

    assert summary["final_action_abs_mean"] == pytest.approx(0.0)
    assert summary["final_action_abs_max"] == pytest.approx(0.0)
    assert summary["final_action_std"] == pytest.approx(0.0)
    assert summary["final_top_action_rms_joints"][0]["rms"] == pytest.approx(0.0)
    cleanup_test_dir(tmp_path)


def test_summarize_chunk_diagnostics_exposes_required_fields() -> None:
    rows = [
        metric_row(chunk_index=0, reset_count=0, tilt_bad_count=0, throughput=100.0),
        metric_row(chunk_index=1, reset_count=3, tilt_bad_count=2, throughput=80.0),
    ]

    summary = probe.summarize_chunk_diagnostics(rows)

    assert summary["first_tilt_chunk"] == 1
    assert summary["max_reset_count"] == 3
    assert summary["mean_reset_count"] == pytest.approx(1.5)
    assert summary["final_reset_count"] == 3
    assert summary["max_tilt_bad_count"] == 2
    assert summary["final_tilt_bad_count"] == 2
    assert summary["final_termination_height_bad_count"] == 1
    assert summary["final_root_height_mean"] == pytest.approx(1.01)
    assert summary["final_upright_mean"] == pytest.approx(0.91)
    assert summary["final_action_abs_mean"] == pytest.approx(0.2)
    assert summary["final_action_abs_max"] == pytest.approx(0.4)
    assert summary["final_action_std"] == pytest.approx(0.1)
    assert summary["min_collect_env_policy_steps_per_sec"] == pytest.approx(80.0)


def test_resolve_run_dir_guard_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    project = Path.cwd() / "project-root-for-test"
    monkeypatch.setattr(probe, "PROJECT_PREFIX", project)

    run_dir = probe.resolve_run_dir(project / "outputs", "run-a")

    assert run_dir == (project / "outputs" / "run-a").resolve()
    with pytest.raises(RuntimeError, match="output path must stay under"):
        probe.resolve_run_dir(Path.cwd() / "elsewhere", "run-a")


def test_select_action_sampled_uses_model_act() -> None:
    torch = pytest.importorskip("torch")
    model = FakeModel(torch)
    observation = torch.zeros((2, 90))

    action = probe.select_action(
        torch=torch,
        observation=observation,
        model=model,
        mode="untrained_sampled_action",
        n_envs=2,
        action_dim=3,
        logical_cuda_device="cpu",
    )

    assert model.act_calls == 1
    assert action.shape == (2, 3)
    assert action.abs().mean().item() > 0.0


def small_args(tmp_path: Path, *extra: str) -> probe.argparse.Namespace:
    return probe.parse_args(
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
            "--output-root",
            str(tmp_path / "outputs"),
            "--warmup-steps",
            "0",
            *extra,
        ]
    )


def fresh_test_dir(name: str) -> Path:
    root = (Path.cwd() / ".test_tmp_task018" / name).resolve()
    cleanup_test_dir(root)
    root.mkdir(parents=True)
    return root


def cleanup_test_dir(path: Path) -> None:
    workspace = Path.cwd().resolve()
    resolved = path.resolve()
    if workspace not in (resolved, *resolved.parents):
        raise RuntimeError(f"refusing to clean path outside workspace: {resolved}")
    if ".test_tmp_task018" not in resolved.parts:
        raise RuntimeError(f"refusing to clean non-test path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def install_fake_runtime(monkeypatch: pytest.MonkeyPatch, torch: object) -> None:
    monkeypatch.setattr(probe, "load_g1_27dof_nohand_profile", lambda: FakeProfile())
    monkeypatch.setattr(probe, "VectorizedGenesisBackend", FakeBackend)
    monkeypatch.setattr(probe, "G1VelocityTrackingVectorizedEnv", fake_env_factory(torch))


def fake_env_factory(torch: object) -> type["FakeEnv"]:
    class RuntimeFakeEnv(FakeEnv):
        def __init__(self, backend: FakeBackend, config: object) -> None:
            super().__init__(
                torch=torch,
                n_envs=backend.config.n_envs,
                observation_dim=90,
            )
            self.backend = backend
            self.config = config

    return RuntimeFakeEnv


class FakeControl:
    default_angles_rad = (0.0,) * 27


class FakeAsset:
    path = "/source/fake_g1.xml"


class FakeProfile:
    asset = FakeAsset()
    action_dim = 27
    actuator_order = tuple(f"joint_{index}" for index in range(27))
    control = FakeControl()


class FakeBackend:
    def __init__(self, config: object, *, profile: FakeProfile) -> None:
        self.config = config
        self.profile = profile


class FakeEnv:
    def __init__(
        self,
        *,
        torch: object,
        n_envs: int,
        observation_dim: int,
    ) -> None:
        self.torch = torch
        self.n_envs = n_envs
        self.observation_dim = observation_dim
        self.step_count = 0

    def reset(self) -> object:
        self.step_count = 0
        return self.torch.zeros((self.n_envs, self.observation_dim))

    def step(self, action: object) -> SimpleNamespace:
        self.step_count += 1
        device = action.device
        observation = (
            self.torch.randn((self.n_envs, self.observation_dim), device=device) * 0.01
        )
        reward = self.torch.ones((self.n_envs,), device=device)
        false_flags = self.torch.zeros((self.n_envs,), dtype=self.torch.bool, device=device)
        tilt = self.torch.full(
            (self.n_envs,),
            self.step_count == 3,
            dtype=self.torch.bool,
            device=device,
        )
        components = {
            "height_bad": false_flags,
            "termination_height_bad": tilt,
            "tilt_bad": tilt,
            "root_height": self.torch.full((self.n_envs,), 1.0, device=device),
            "upright": self.torch.ones((self.n_envs,), device=device),
        }
        return SimpleNamespace(
            observation=observation,
            reward=reward,
            terminated=false_flags,
            truncated=false_flags,
            done=false_flags,
            info={"reset_count": int(tilt.sum().item()), "components": components},
        )

    def tensor_device_ok(self) -> bool:
        return True


class FakeModel:
    def __init__(self, torch: object) -> None:
        self.torch = torch
        self.act_calls = 0

    def act(self, observation: object) -> tuple[object, object, object, object]:
        self.act_calls += 1
        action = self.torch.full((observation.shape[0], 3), 0.25)
        zeros = self.torch.zeros((observation.shape[0],))
        return action, zeros, zeros, zeros


def metric_row(
    *,
    chunk_index: int,
    reset_count: int,
    tilt_bad_count: int,
    throughput: float,
) -> dict[str, object]:
    return {
        "chunk_index": chunk_index,
        "reset_count": reset_count,
        "tilt_bad_count": tilt_bad_count,
        "termination_height_bad_count": 1,
        "root_height_mean": 1.0 + chunk_index * 0.01,
        "root_height_min": 0.9 + chunk_index * 0.01,
        "upright_mean": 0.9 + chunk_index * 0.01,
        "action_abs_mean": 0.2,
        "action_abs_max": 0.4,
        "action_std": 0.1,
        "top_action_rms_joints": [{"joint": "joint_0", "rms": 0.2}],
        "collect_env_policy_steps_per_sec": throughput,
    }
