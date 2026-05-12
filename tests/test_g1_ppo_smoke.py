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
    assert args.termination_penalty == -5.0
    assert args.warmup_steps == 2


def test_parse_args_defaults_to_stable_tall_crouch_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["g1_ppo_smoke"])

    args = g1_ppo_smoke.parse_args()

    assert args.root_z == 1.20
    assert args.default_pose == "tall_crouch"


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


def argparse_error() -> type[Exception]:
    return Exception


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
