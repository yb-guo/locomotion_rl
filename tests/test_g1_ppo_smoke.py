from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from h200_locomotion_lab.tools import g1_ppo_smoke


def test_parse_seeds_rejects_duplicates() -> None:
    with pytest.raises(argparse_error(), match="seeds must be unique"):
        g1_ppo_smoke.parse_seeds("0,1,0")


def test_parse_args_accepts_log_std_init(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "g1_ppo_smoke",
            "--log-std-init",
            "-2.0",
            "--height-min",
            "0.40",
            "--termination-height-min",
            "0.20",
            "--root-z",
            "0.90",
            "--default-pose",
            "profile",
            "--action-scale-mult",
            "0.25",
            "--action-joint-group",
            "legs",
            "--command-mode",
            "standing",
            "--base-height-reward-scale",
            "0.5",
            "--joint-velocity-penalty-scale",
            "0.02",
            "--termination-penalty",
            "-5.0",
            "--warmup-steps",
            "2",
        ],
    )

    args = g1_ppo_smoke.parse_args()

    assert args.log_std_init == -2.0
    assert args.height_min == 0.40
    assert args.termination_height_min == 0.20
    assert args.root_z == 0.90
    assert args.default_pose == "profile"
    assert args.action_scale_mult == 0.25
    assert args.action_joint_group == "legs"
    assert args.command_mode == "standing"
    assert args.base_height_reward_scale == 0.5
    assert args.joint_velocity_penalty_scale == 0.02
    assert args.termination_penalty == -5.0
    assert args.warmup_steps == 2
    assert args.asset_variant == "task023_hybrid"


def test_parse_args_defaults_to_stable_tall_crouch_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["g1_ppo_smoke"])

    args = g1_ppo_smoke.parse_args()

    assert args.root_z == 1.20
    assert args.default_pose == "tall_crouch"
    assert args.asset_variant == "task023_hybrid"


def test_parse_args_can_use_profile_asset_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["g1_ppo_smoke", "--asset-variant", "profile"])

    args = g1_ppo_smoke.parse_args()

    assert args.asset_variant == "profile"


def test_resolve_training_profile_profile_variant_keeps_source_asset() -> None:
    profile = FakeProfile(asset=FakeAsset(path="/source/g1_27dof_nohand.xml"))

    resolved, report = g1_ppo_smoke.resolve_training_profile_for_asset_variant(
        profile,
        asset_variant="profile",
        run_dir=g1_ppo_smoke.Path.cwd() / "unused",
    )

    assert resolved is profile
    assert report == {
        "asset_variant": "profile",
        "source_asset_path": "/source/g1_27dof_nohand.xml",
        "effective_asset_path": "/source/g1_27dof_nohand.xml",
        "generated": False,
    }


def test_resolve_training_profile_task023_hybrid_generates_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = g1_ppo_smoke.Path.cwd() / "outputs" / ".test_tmp_ppo_hybrid"
    profile = FakeProfile(asset=FakeAsset(path="/source/g1_27dof_nohand.xml"))
    generated_asset = run_dir / "asset_generation" / "task023_hybrid" / "assets" / "hybrid.xml"

    def fake_run_patch_generation(args):
        assert args.source_asset == g1_ppo_smoke.Path("/source/g1_27dof_nohand.xml")
        assert args.output_root == run_dir / "asset_generation"
        assert args.run_id == "task023_hybrid"
        assert args.variants == g1_ppo_smoke.TASK023_HYBRID_PATCH_VARIANT
        return {
            "run_dir": str(run_dir / "asset_generation" / "task023_hybrid"),
            "missing": [],
            "errors": [],
            "variants": {
                g1_ppo_smoke.TASK023_HYBRID_PATCH_VARIANT: {
                    "path": str(generated_asset),
                },
            },
        }

    monkeypatch.setattr(
        g1_ppo_smoke.contact_patch,
        "run_patch_generation",
        fake_run_patch_generation,
    )

    resolved, report = g1_ppo_smoke.resolve_training_profile_for_asset_variant(
        profile,
        asset_variant="task023_hybrid",
        run_dir=run_dir,
    )

    assert profile.asset.path == "/source/g1_27dof_nohand.xml"
    assert resolved.asset.path == generated_asset.resolve().as_posix()
    assert report["asset_variant"] == "task023_hybrid"
    assert report["patch_variant"] == g1_ppo_smoke.TASK023_HYBRID_PATCH_VARIANT
    assert report["source_asset_path"] == "/source/g1_27dof_nohand.xml"
    assert report["effective_asset_path"] == generated_asset.resolve().as_posix()
    assert report["generated"] is True


def test_resolve_run_dir_stays_under_project_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    project = g1_ppo_smoke.Path.cwd() / "project-root-for-test"
    monkeypatch.setattr(g1_ppo_smoke, "PROJECT_PREFIX", project)

    run_dir = g1_ppo_smoke.resolve_run_dir(project / "outputs", "run-a")

    assert run_dir == (project / "outputs" / "run-a").resolve()


def test_resolve_run_dir_rejects_non_project_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = g1_ppo_smoke.Path.cwd() / "project-root-for-test"
    monkeypatch.setattr(g1_ppo_smoke, "PROJECT_PREFIX", project)

    with pytest.raises(RuntimeError, match="output path must stay under"):
        g1_ppo_smoke.resolve_run_dir(g1_ppo_smoke.Path.cwd() / "elsewhere", "run-a")


def test_command_ranges_for_mode() -> None:
    assert g1_ppo_smoke.command_ranges_for_mode("standing") == {
        "command_vx_min": 0.0,
        "command_vx_max": 0.0,
        "command_yaw_min": 0.0,
        "command_yaw_max": 0.0,
    }
    assert g1_ppo_smoke.command_ranges_for_mode("vx_yaw")["command_vx_max"] == 0.8


def test_metric_row_rejects_non_finite_value() -> None:
    row = _valid_metric_row()
    row["reward_mean"] = float("nan")

    with pytest.raises(ValueError, match="reward_mean"):
        g1_ppo_smoke.assert_metric_row_ok(row)


def test_metric_row_rejects_non_finite_reward_component() -> None:
    row = _valid_metric_row()
    row["reward_component_tracking_lin_vel_mean"] = float("inf")

    with pytest.raises(ValueError, match="reward_component_tracking_lin_vel_mean"):
        g1_ppo_smoke.assert_metric_row_ok(row)


def test_profile_stat_helpers_report_distribution_and_saturation() -> None:
    torch = pytest.importorskip("torch")
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    stats = g1_ppo_smoke.tensor_profile_stats(values, "observation")
    saturation = g1_ppo_smoke.action_saturation_ratio(
        torch.tensor([[-0.96, -0.94], [0.95, 0.0]]),
        threshold=0.95,
    )

    assert stats == {
        "observation_mean": pytest.approx(2.5),
        "observation_std": pytest.approx(1.118034, rel=1e-5),
        "observation_min": pytest.approx(1.0),
        "observation_max": pytest.approx(4.0),
    }
    assert saturation == pytest.approx(0.5)


def test_reward_contribution_stats_apply_env_scales_and_signs() -> None:
    stats = g1_ppo_smoke.reward_contribution_stats(
        {
            "tracking_lin_vel": 0.5,
            "tracking_yaw_rate": 0.25,
            "upright": 0.8,
            "tracking_base_height": 0.9,
            "action_rate_penalty": 0.4,
            "joint_velocity_penalty": 0.3,
            "joint_deviation_penalty": 0.2,
            "termination_penalty": -0.5,
        },
        SimpleNamespace(
            alive_reward=0.05,
            lin_vel_reward_scale=1.0,
            yaw_rate_reward_scale=0.5,
            upright_reward_scale=0.5,
            base_height_reward_scale=0.0,
            action_rate_penalty_scale=0.01,
            joint_velocity_penalty_scale=0.02,
            joint_deviation_penalty_scale=0.05,
        ),
    )

    assert stats == {
        "reward_contribution_alive_mean": pytest.approx(0.05),
        "reward_contribution_tracking_lin_vel_mean": pytest.approx(0.5),
        "reward_contribution_tracking_yaw_rate_mean": pytest.approx(0.125),
        "reward_contribution_upright_mean": pytest.approx(0.4),
        "reward_contribution_tracking_base_height_mean": pytest.approx(0.0),
        "reward_contribution_action_rate_penalty_mean": pytest.approx(-0.004),
        "reward_contribution_joint_velocity_penalty_mean": pytest.approx(-0.006),
        "reward_contribution_joint_deviation_penalty_mean": pytest.approx(-0.01),
        "reward_contribution_termination_penalty_mean": pytest.approx(-0.5),
    }


def test_rate_handles_empty_totals() -> None:
    assert g1_ppo_smoke.rate(3, 6) == pytest.approx(0.5)
    assert g1_ppo_smoke.rate(3, 0) == 0.0


def test_seed_final_metric_helpers() -> None:
    seed_summaries = [
        {"final_metrics": {"height_reset_rate": 0.25}},
        {"final_metrics": {"height_reset_rate": 0.75}},
    ]

    assert g1_ppo_smoke.mean_seed_final_metric(
        seed_summaries,
        "height_reset_rate",
    ) == pytest.approx(0.5)
    assert g1_ppo_smoke.max_seed_final_metric(
        seed_summaries,
        "height_reset_rate",
    ) == pytest.approx(0.75)


def test_summarize_seed_training_metrics_reports_full_training_window() -> None:
    rows = [
        _training_metric_row(
            update=0,
            full_env_reset_wave=False,
            reset_rate=0.10,
            tilt_reset_rate=0.0,
            height_reset_rate=0.10,
            episode_length_mean=16.0,
            reward_mean=0.25,
        ),
        _training_metric_row(
            update=1,
            full_env_reset_wave=True,
            full_env_reset_wave_count=3,
            reset_rate=1.0,
            tilt_reset_rate=1.0,
            height_reset_rate=0.0,
            episode_length_mean=2.0,
            reward_mean=-1.0,
        ),
        _training_metric_row(
            update=2,
            full_env_reset_wave=True,
            reset_rate=0.75,
            tilt_reset_rate=0.25,
            height_reset_rate=0.50,
            episode_length_mean=24.0,
            reward_mean=0.75,
        ),
    ]

    summary = g1_ppo_smoke.summarize_seed_training_metrics(rows)

    assert summary == {
        "any_training_full_env_reset_wave": True,
        "training_full_env_reset_wave_updates": [1, 2],
        "training_full_env_reset_wave_count": 4,
        "max_training_reset_rate": pytest.approx(1.0),
        "max_training_tilt_reset_rate": pytest.approx(1.0),
        "max_training_height_reset_rate": pytest.approx(0.50),
        "max_training_episode_length_mean": pytest.approx(24.0),
        "max_training_reward_mean": pytest.approx(0.75),
    }


def test_summarize_seed_training_metrics_handles_empty_rows() -> None:
    summary = g1_ppo_smoke.summarize_seed_training_metrics([])

    assert summary == {
        "any_training_full_env_reset_wave": False,
        "training_full_env_reset_wave_updates": [],
        "training_full_env_reset_wave_count": 0,
        "max_training_reset_rate": pytest.approx(0.0),
        "max_training_tilt_reset_rate": pytest.approx(0.0),
        "max_training_height_reset_rate": pytest.approx(0.0),
        "max_training_episode_length_mean": pytest.approx(0.0),
        "max_training_reward_mean": pytest.approx(0.0),
    }


def test_summarize_run_training_metrics_aggregates_across_seeds() -> None:
    seed_summaries = [
        {
            "seed": 0,
            "any_training_full_env_reset_wave": True,
            "training_full_env_reset_wave_count": 2,
            "max_training_reset_rate": 1.0,
            "max_training_tilt_reset_rate": 1.0,
            "max_training_height_reset_rate": 0.25,
            "max_training_episode_length_mean": 24.0,
            "max_training_reward_mean": 0.75,
        },
        {
            "seed": 1,
            "any_training_full_env_reset_wave": False,
            "training_full_env_reset_wave_count": 0,
            "max_training_reset_rate": 0.5,
            "max_training_tilt_reset_rate": 0.0,
            "max_training_height_reset_rate": 0.5,
            "max_training_episode_length_mean": 32.0,
            "max_training_reward_mean": 1.25,
        },
    ]

    summary = g1_ppo_smoke.summarize_run_training_metrics(seed_summaries)

    assert summary == {
        "any_training_full_env_reset_wave": True,
        "training_full_env_reset_wave_seed_count": 1,
        "training_full_env_reset_wave_count": 2,
        "max_training_reset_rate": pytest.approx(1.0),
        "max_training_tilt_reset_rate": pytest.approx(1.0),
        "max_training_height_reset_rate": pytest.approx(0.5),
        "max_training_episode_length_mean": pytest.approx(32.0),
        "max_training_reward_mean": pytest.approx(1.25),
    }


def argparse_error() -> type[Exception]:
    return Exception


@dataclass(frozen=True)
class FakeAsset:
    path: str


@dataclass(frozen=True)
class FakeProfile:
    asset: FakeAsset


def _valid_metric_row() -> dict[str, float | bool]:
    return {
        "reward_mean": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 1.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "grad_norm": 0.0,
        "root_height_mean": 1.0,
        "root_height_min": 1.0,
        "upright_mean": 1.0,
        "reset_rate": 0.0,
        "height_reset_rate": 0.0,
        "tilt_reset_rate": 0.0,
        "timeout_rate": 0.0,
        "survival_rate": 1.0,
        "episode_length_mean": 1.0,
        "episode_length_min": 1.0,
        "episode_length_max": 1.0,
        "completed_episode_length_mean": 0.0,
        "collect_time_s": 1.0,
        "collect_env_policy_steps_per_sec": 1.0,
        "update_time_s": 1.0,
        "update_samples_per_sec": 1.0,
        "observation_mean": 0.0,
        "observation_std": 1.0,
        "observation_min": -1.0,
        "observation_max": 1.0,
        "action_mean": 0.0,
        "action_std": 1.0,
        "action_min": -1.0,
        "action_max": 1.0,
        "action_saturation_ratio": 0.0,
        "reward_std": 1.0,
        "reward_min": -1.0,
        "reward_max": 1.0,
        "value_prediction_mean": 0.0,
        "value_prediction_std": 1.0,
        "value_prediction_min": -1.0,
        "value_prediction_max": 1.0,
        "return_mean": 0.0,
        "return_std": 1.0,
        "return_min": -1.0,
        "return_max": 1.0,
        "advantage_mean": 0.0,
        "advantage_std": 1.0,
        "advantage_min": -1.0,
        "advantage_max": 1.0,
        "log_std_mean": -0.5,
        "log_std_min": -0.5,
        "log_std_max": -0.5,
        "tensor_device_ok": True,
    }


def _training_metric_row(
    *,
    update: int,
    full_env_reset_wave: bool,
    reset_rate: float,
    tilt_reset_rate: float,
    height_reset_rate: float,
    episode_length_mean: float,
    reward_mean: float,
    full_env_reset_wave_count: int = 1,
) -> dict[str, float | int | bool]:
    return {
        "update": update,
        "full_env_reset_wave": full_env_reset_wave,
        "full_env_reset_wave_count": full_env_reset_wave_count if full_env_reset_wave else 0,
        "reset_rate": reset_rate,
        "tilt_reset_rate": tilt_reset_rate,
        "height_reset_rate": height_reset_rate,
        "episode_length_mean": episode_length_mean,
        "reward_mean": reward_mean,
    }
