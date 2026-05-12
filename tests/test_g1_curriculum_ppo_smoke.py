import json
import shutil
from types import SimpleNamespace

import pytest

from h200_locomotion_lab.tools import g1_curriculum_ppo_smoke as curriculum
from h200_locomotion_lab.training.ppo_loop import PPOConfig


def test_parse_args_defaults_to_task015_stable_config() -> None:
    args = curriculum.parse_args([])

    assert args.n_envs == 1024
    assert args.rollout_steps == 32
    assert args.updates_per_stage == 50
    assert args.log_std_init == -2.5
    assert args.action_scale_mult == 0.10
    assert args.root_z == 1.20
    assert args.default_pose == "tall_crouch"
    assert args.termination_height_min == 0.20
    assert args.warmup_steps == 1
    assert args.output_root == curriculum.Path("outputs/task015/g1_curriculum_ppo")


def test_curriculum_stages_are_explicit_and_ordered() -> None:
    stages = curriculum.curriculum_stages()

    assert [stage.name for stage in stages] == [
        "standing",
        "small_vx",
        "small_yaw",
        "small_vxyaw",
    ]
    assert stages[0].command_vx_max == 0.0
    assert stages[1].command_vx_max > 0.0
    assert stages[1].command_yaw_max == 0.0
    assert stages[2].command_vx_max == 0.0
    assert stages[2].command_yaw_min < 0.0
    assert stages[3].command_vx_max > 0.0
    assert stages[3].command_yaw_max > 0.0


def test_resolve_run_dir_stays_under_project_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    project = curriculum.Path.cwd() / "project-root-for-test"
    monkeypatch.setattr(curriculum, "PROJECT_PREFIX", project)

    run_dir = curriculum.resolve_run_dir(project / "outputs", "run-a")

    assert run_dir == (project / "outputs" / "run-a").resolve()


def test_resolve_run_dir_rejects_non_project_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = curriculum.Path.cwd() / "project-root-for-test"
    monkeypatch.setattr(curriculum, "PROJECT_PREFIX", project)

    with pytest.raises(RuntimeError, match="output path must stay under"):
        curriculum.resolve_run_dir(curriculum.Path.cwd() / "elsewhere", "run-a")


def test_run_smoke_writes_curriculum_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    tmp_path = fresh_test_dir("artifacts")
    install_fake_runtime(monkeypatch, torch)
    monkeypatch.setattr(curriculum, "PROJECT_PREFIX", tmp_path)
    args = small_args(
        tmp_path,
        "--run-id",
        "artifact-run",
    )

    summary = curriculum.run_smoke(args)

    run_dir = tmp_path / "outputs" / "artifact-run"
    assert summary["all_seeds_passed"] is True
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "final_checkpoint.pt").is_file()

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["curriculum"]["updates_per_stage"] == 1
    assert [stage["name"] for stage in config["curriculum"]["stages"]] == [
        "standing",
        "small_vx",
        "small_yaw",
        "small_vxyaw",
    ]

    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["stage"] for row in rows] == [
        "standing",
        "small_vx",
        "small_yaw",
        "small_vxyaw",
    ]
    assert [row["stage_update"] for row in rows] == [0, 0, 0, 0]
    assert [row["global_update"] for row in rows] == [0, 1, 2, 3]
    assert all(row["tensor_device_ok"] for row in rows)
    assert all(row["env_tensor_device_ok"] for row in rows)
    assert {"reward_mean", "reset_count", "root_height_mean", "tilt_bad_count"} <= rows[0].keys()
    assert {"approx_kl", "policy_loss", "value_loss"} <= rows[0].keys()
    cleanup_test_dir(tmp_path)


def test_standing_failure_skips_velocity_stages() -> None:
    torch = pytest.importorskip("torch")
    tmp_path = fresh_test_dir("stop-rule")
    env = FakeEnv(torch=torch, n_envs=2, action_dim=27, observation_dim=90)
    config = PPOConfig(
        n_envs=2,
        rollout_steps=2,
        ppo_updates=1,
        epochs=1,
        minibatch_size=2,
        log_std_init=-2.5,
    )
    stages = curriculum.curriculum_stages()
    args = small_args(tmp_path)
    stage_env_configs = {
        stage.name: curriculum.build_stage_env_config(args=args, stage=stage)
        for stage in stages
    }

    summary, _checkpoint = curriculum.run_seed(
        torch=torch,
        env=env,
        config=config,
        seed=0,
        stages=stages,
        stage_env_configs=stage_env_configs,
        metrics_path=tmp_path / "metrics.jsonl",
        min_collect_env_steps_per_sec=1e12,
        logical_cuda_device="cpu",
    )

    assert summary["passed"] is False
    assert [stage["status"] for stage in summary["stages"]] == [
        "failed",
        "skipped",
        "skipped",
        "skipped",
    ]
    assert "standing failed" in summary["stages"][1]["blocker"]
    cleanup_test_dir(tmp_path)


def test_parse_seeds_rejects_duplicates() -> None:
    with pytest.raises(argparse_error(), match="seeds must be unique"):
        curriculum.parse_seeds("0,1,0")


def small_args(tmp_path: curriculum.Path, *extra: str) -> curriculum.argparse.Namespace:
    return curriculum.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--n-envs",
            "2",
            "--rollout-steps",
            "2",
            "--updates-per-stage",
            "1",
            "--seeds",
            "0",
            "--epochs",
            "1",
            "--minibatch-size",
            "2",
            "--output-root",
            str(tmp_path / "outputs"),
            "--warmup-steps",
            "0",
            "--min-collect-env-steps-per-sec",
            "0.0001",
            *extra,
        ]
    )


def fresh_test_dir(name: str) -> curriculum.Path:
    root = (curriculum.Path.cwd() / ".test_tmp_task015" / name).resolve()
    cleanup_test_dir(root)
    root.mkdir(parents=True)
    return root


def cleanup_test_dir(path: curriculum.Path) -> None:
    workspace = curriculum.Path.cwd().resolve()
    resolved = path.resolve()
    if workspace not in (resolved, *resolved.parents):
        raise RuntimeError(f"refusing to clean path outside workspace: {resolved}")
    if ".test_tmp_task015" not in resolved.parts:
        raise RuntimeError(f"refusing to clean non-test path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def install_fake_runtime(monkeypatch: pytest.MonkeyPatch, torch: object) -> None:
    monkeypatch.setattr(curriculum, "load_g1_27dof_nohand_profile", lambda: FakeProfile())
    monkeypatch.setattr(curriculum, "VectorizedGenesisBackend", FakeBackend)
    monkeypatch.setattr(curriculum, "G1VelocityTrackingVectorizedEnv", fake_env_factory(torch))


def fake_env_factory(torch: object) -> type["FakeEnv"]:
    class RuntimeFakeEnv(FakeEnv):
        def __init__(self, backend: FakeBackend, config: object) -> None:
            super().__init__(
                torch=torch,
                n_envs=backend.config.n_envs,
                action_dim=backend.profile.action_dim,
                observation_dim=90,
            )
            self.backend = backend
            self.config = config

    return RuntimeFakeEnv


class FakeControl:
    default_angles_rad = (0.0,) * 27


class FakeProfile:
    action_dim = 27
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
        action_dim: int,
        observation_dim: int,
    ) -> None:
        self.torch = torch
        self.n_envs = n_envs
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.config = SimpleNamespace(
            command_vx_max=0.0,
            command_yaw_min=0.0,
            command_yaw_max=0.0,
        )
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
        reward_scale = (
            1.0
            + float(self.config.command_vx_max)
            + abs(float(self.config.command_yaw_min))
            + abs(float(self.config.command_yaw_max))
        )
        reward = self.torch.full((self.n_envs,), reward_scale, device=device)
        false_flags = self.torch.zeros((self.n_envs,), dtype=self.torch.bool, device=device)
        components = {
            "height_bad": false_flags,
            "termination_height_bad": false_flags,
            "tilt_bad": false_flags,
            "root_height": self.torch.full((self.n_envs,), 1.0, device=device),
            "upright": self.torch.ones((self.n_envs,), device=device),
        }
        return SimpleNamespace(
            observation=observation,
            reward=reward,
            terminated=false_flags,
            truncated=false_flags,
            done=false_flags,
            info={"reset_count": 0, "components": components},
        )

    def tensor_device_ok(self) -> bool:
        return True


def argparse_error() -> type[Exception]:
    return Exception
