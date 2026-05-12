import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from h200_locomotion_lab.tools import g1_action_energy_ablation as ablation


def test_parse_args_defaults_to_task020_route() -> None:
    args = ablation.parse_args([])

    assert args.seeds == "0"
    assert args.action_scale_mults == "0.1,0.2,0.25,0.35"
    assert args.log_std_inits == "-2.0,-1.5,-1.0"
    assert args.output_root == Path("outputs/task020/action_energy_ablation")


def test_build_smoke_args_keeps_task004_standing_config() -> None:
    args = ablation.parse_args(["--seeds", "0,1"])
    smoke_args = ablation.build_smoke_args(
        args=args,
        output_root=Path("/tmp/out"),
        action_scale_mult=0.25,
        log_std_init=-1.5,
    )

    assert smoke_args.action_scale_mult == pytest.approx(0.25)
    assert smoke_args.log_std_init == pytest.approx(-1.5)
    assert smoke_args.seeds == "0,1"
    assert smoke_args.command_mode == "standing"
    assert smoke_args.root_z == pytest.approx(1.20)
    assert smoke_args.termination_height_min == pytest.approx(0.20)
    assert smoke_args.base_height_reward_scale == pytest.approx(0.20)
    assert smoke_args.joint_velocity_penalty_scale == pytest.approx(0.001)
    assert smoke_args.termination_penalty == pytest.approx(-1.0)
    assert smoke_args.run_id == "scale_0p25_logstd_neg1p5"


def test_candidate_smoke_command_uses_subprocess_module_with_fixed_standing_config() -> None:
    args = ablation.parse_args(["--seeds", "0,1"])
    smoke_args = ablation.build_smoke_args(
        args=args,
        output_root=Path("/tmp/out"),
        action_scale_mult=0.25,
        log_std_init=-1.5,
    )

    command = ablation.candidate_smoke_command(smoke_args)

    assert command[:3] == [
        sys.executable,
        "-m",
        "h200_locomotion_lab.tools.g1_ppo_smoke",
    ]
    assert cli_value(command, "--command-mode") == "standing"
    assert cli_value(command, "--root-z") == "1.2"
    assert cli_value(command, "--termination-height-min") == "0.2"
    assert cli_value(command, "--base-height-reward-scale") == "0.2"
    assert cli_value(command, "--joint-velocity-penalty-scale") == "0.001"
    assert cli_value(command, "--termination-penalty") == "-1.0"
    assert cli_value(command, "--action-scale-mult") == "0.25"
    assert cli_value(command, "--log-std-init") == "-1.5"
    assert cli_value(command, "--run-id") == "scale_0p25_logstd_neg1p5"


def test_run_candidate_subprocess_reads_summary_without_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_root = fresh_test_root("subprocess-summary")
    smoke_args = ablation.build_smoke_args(
        args=ablation.parse_args([]),
        output_root=test_root,
        action_scale_mult=0.25,
        log_std_init=-1.5,
    )
    calls = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        summary_path = Path(smoke_args.output_root) / smoke_args.run_id / "summary.json"
        summary_path.parent.mkdir(parents=True)
        summary_path.write_text(
            json.dumps(smoke_summary(
                args=smoke_args,
                reward=1.0,
                episode_length=20.0,
                action_saturation=0.0,
                survival_rate=1.0,
            )),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(ablation.subprocess, "run", fake_run)

    summary = ablation.run_candidate_subprocess(smoke_args)

    assert summary["all_seeds_passed"] is True
    assert calls
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["text"] is True
    assert calls[0][1]["check"] is False
    assert "shell" not in calls[0][1]


def test_run_candidate_subprocess_raises_on_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_root = fresh_test_root("subprocess-failure")
    smoke_args = ablation.build_smoke_args(
        args=ablation.parse_args([]),
        output_root=test_root,
        action_scale_mult=0.25,
        log_std_init=-1.5,
    )

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            7,
            stdout="partial stdout",
            stderr="GenesisException:Genesis already initialized.",
        )

    monkeypatch.setattr(ablation.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="returncode=7.*Genesis already initialized"):
        ablation.run_candidate_subprocess(smoke_args)


def test_run_ablation_records_candidates_and_selects_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_root = fresh_test_root("selects-best")
    monkeypatch.setattr(ablation.g1_ppo_smoke, "PROJECT_PREFIX", test_root)
    calls = []

    def fake_run_smoke(args: object) -> dict[str, object]:
        calls.append(args)
        if args.action_scale_mult == 0.10:
            return smoke_summary(
                args=args,
                reward=1.0,
                episode_length=20.0,
                action_saturation=0.0,
                survival_rate=1.0,
            )
        if args.action_scale_mult == 0.20:
            return smoke_summary(
                args=args,
                reward=2.0,
                episode_length=40.0,
                action_saturation=0.01,
                survival_rate=1.0,
            )
        return smoke_summary(
            args=args,
            reward=3.0,
            episode_length=60.0,
            action_saturation=0.50,
            survival_rate=1.0,
        )

    args = ablation.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--output-root",
            str(test_root / "outputs"),
            "--run-id",
            "matrix",
            "--action-scale-mults",
            "0.10,0.20,0.25",
            "--log-std-inits=-2.0",
        ]
    )

    summary = ablation.run_ablation(args, runner=fake_run_smoke)

    assert [call.command_mode for call in calls] == ["standing", "standing", "standing"]
    assert [call.action_scale_mult for call in calls] == [0.10, 0.20, 0.25]
    assert summary["status"] == "passed"
    assert summary["selected_candidate"]["name"] == "scale_0p25_logstd_neg2p0"
    run_dir = test_root / "outputs" / "matrix"
    rows = [
        json.loads(line)
        for line in (run_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["name"] for row in rows] == [
        "scale_0p1_logstd_neg2p0",
        "scale_0p2_logstd_neg2p0",
        "scale_0p25_logstd_neg2p0",
    ]
    assert (run_dir / "summary.json").is_file()


def test_run_ablation_blocks_parent_when_any_candidate_runtime_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_root = fresh_test_root("records-failure")
    monkeypatch.setattr(ablation.g1_ppo_smoke, "PROJECT_PREFIX", test_root)

    def fake_run_smoke(args: object) -> dict[str, object]:
        if args.log_std_init == -1.0:
            raise RuntimeError("GenesisException:Genesis already initialized")
        return smoke_summary(
            args=args,
            reward=1.0,
            episode_length=30.0,
            action_saturation=0.0,
            survival_rate=1.0,
        )

    args = ablation.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--output-root",
            str(test_root / "outputs"),
            "--run-id",
            "matrix",
            "--action-scale-mults",
            "0.10",
            "--log-std-inits=-2.0,-1.0",
        ]
    )

    summary = ablation.run_ablation(args, runner=fake_run_smoke)

    failed = [candidate for candidate in summary["candidates"] if candidate["status"] == "failed"]
    assert summary["status"] == "blocked"
    assert summary["selected_candidate"]["name"] == "scale_0p1_logstd_neg2p0"
    assert "candidate runtime/subprocess failures" in summary["blocker"]
    assert "RuntimeError:GenesisException:Genesis already initialized" in summary["blocker"]
    assert len(failed) == 1
    assert failed[0]["name"] == "scale_0p1_logstd_neg1p0"
    assert "RuntimeError:GenesisException:Genesis already initialized" in failed[0]["blocker"]


def test_choose_candidate_returns_none_when_no_candidate_passes() -> None:
    assert ablation.choose_candidate([{"passed": False}]) is None


def test_choose_candidate_rejects_reset_or_survival_regression() -> None:
    candidates = [
        candidate_summary(
            name="reset",
            survival_rate=1.0,
            reset_rate=0.1,
            reward=3.0,
            episode_length=60.0,
        ),
        candidate_summary(
            name="survival",
            survival_rate=0.9,
            reset_rate=0.0,
            reward=4.0,
            episode_length=70.0,
        ),
    ]

    assert ablation.choose_candidate(candidates) is None


def test_run_ablation_rejects_out_of_scope_matrix_values() -> None:
    args = ablation.parse_args(
        [
            "--action-scale-mults",
            "0.10,1.0",
            "--log-std-inits=-2.0",
        ]
    )

    with pytest.raises(ValueError, match="action_scale_mults must be a subset"):
        ablation.run_ablation(args)


def test_run_ablation_rejects_out_of_scope_log_std_values() -> None:
    args = ablation.parse_args(
        [
            "--action-scale-mults",
            "0.10",
            "--log-std-inits=-2.0,0.0",
        ]
    )

    with pytest.raises(ValueError, match="log_std_inits must be a subset"):
        ablation.run_ablation(args)


def cli_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def smoke_summary(
    *,
    args: object,
    reward: float,
    episode_length: float,
    action_saturation: float,
    survival_rate: float,
) -> dict[str, object]:
    final_metrics = {
        "reset_rate": 0.0,
        "height_reset_rate": 0.0,
        "tilt_reset_rate": 0.0,
        "timeout_rate": 0.0,
        "episode_length_mean": episode_length,
        "action_saturation_ratio": action_saturation,
        "log_std_mean": args.log_std_init,
        "log_std_min": args.log_std_init,
        "log_std_max": args.log_std_init,
        "approx_kl": 0.01,
        "clip_fraction": 0.02,
        "full_env_reset_wave": False,
        "survival_rate": survival_rate,
    }
    return {
        "status": "ok",
        "run_dir": str((Path(args.output_root) / args.run_id).resolve()),
        "all_seeds_passed": True,
        "min_collect_env_policy_steps_per_sec": 20000.0,
        "mean_reward_mean": reward,
        "mean_final_episode_length_mean": episode_length,
        "mean_final_survival_rate": survival_rate,
        "max_final_height_reset_rate": 0.0,
        "max_final_tilt_reset_rate": 0.0,
        "max_final_timeout_rate": 0.0,
        "any_final_full_env_reset_wave": False,
        "seeds": [
            {
                "seed": 0,
                "passed": True,
                "actor_params_changed": True,
                "value_params_changed": True,
                "final_reward_mean": reward,
                "final_metrics": final_metrics,
            }
        ],
    }


def candidate_summary(
    *,
    name: str,
    survival_rate: float,
    reset_rate: float,
    reward: float,
    episode_length: float,
) -> dict[str, object]:
    return {
        "name": name,
        "passed": True,
        "actor_params_changed": True,
        "value_params_changed": True,
        "mean_final_survival_rate": survival_rate,
        "max_final_reset_rate": reset_rate,
        "max_final_height_reset_rate": 0.0,
        "max_final_tilt_reset_rate": 0.0,
        "any_final_full_env_reset_wave": False,
        "action_saturation_ratio": 0.0,
        "mean_reward_mean": reward,
        "mean_final_episode_length_mean": episode_length,
        "action_scale_mult": 0.1,
        "log_std_init": -2.0,
    }


def fresh_test_root(name: str) -> Path:
    root = (Path.cwd() / "outputs" / "test_task020" / f"{name}_{uuid.uuid4().hex}").resolve()
    root.mkdir(parents=True)
    return root
