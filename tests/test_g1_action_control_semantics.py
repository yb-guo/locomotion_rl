import json
import time

import pytest

from h200_locomotion_lab.tools import g1_action_control_semantics as ablation


def test_parse_args_defaults_to_task017_standing_only_output_root() -> None:
    args = ablation.parse_args([])

    assert args.seeds == "0"
    assert args.updates_per_stage == 10
    assert args.stage_names == "standing"
    assert args.output_root == ablation.Path("outputs/task017/action_control_semantics")


def test_action_control_variants_are_one_variable_at_a_time() -> None:
    variants = {variant.name: variant.overrides for variant in ablation.ACTION_CONTROL_VARIANTS}

    assert list(variants) == [
        "baseline",
        "action_scale_0_05",
        "action_scale_0_03",
        "action_scale_0_01",
        "action_group_legs",
        "action_group_legs_waist",
        "log_std_neg3_5",
    ]
    assert variants["baseline"] == {
        "action_scale_mult": "0.10",
        "action_joint_group": "all",
        "log_std_init": "-2.5",
    }
    assert variants["action_scale_0_05"] == {"action_scale_mult": "0.05"}
    assert variants["action_scale_0_03"] == {"action_scale_mult": "0.03"}
    assert variants["action_scale_0_01"] == {"action_scale_mult": "0.01"}
    assert variants["action_group_legs"] == {"action_joint_group": "legs"}
    assert variants["action_group_legs_waist"] == {"action_joint_group": "legs_waist"}
    assert variants["log_std_neg3_5"] == {"log_std_init": "-3.5"}


def test_build_curriculum_args_applies_variant_override_and_standing_stage() -> None:
    args = ablation.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--action-scale-mult",
            "0.10",
            "--action-joint-group",
            "all",
            "--log-std-init",
            "-2.5",
        ]
    )

    scale_args = ablation.build_curriculum_args(
        args=args,
        variant=ablation.ACTION_CONTROL_VARIANTS[1],
        output_root=ablation.Path("outputs"),
    )
    group_args = ablation.build_curriculum_args(
        args=args,
        variant=ablation.ACTION_CONTROL_VARIANTS[4],
        output_root=ablation.Path("outputs"),
    )

    assert scale_args.stage_names == "standing"
    assert scale_args.action_scale_mult == pytest.approx(0.05)
    assert scale_args.action_joint_group == "all"
    assert scale_args.log_std_init == pytest.approx(-2.5)
    assert group_args.action_scale_mult == pytest.approx(0.10)
    assert group_args.action_joint_group == "legs"
    assert group_args.log_std_init == pytest.approx(-2.5)


def test_run_ablation_writes_summary_and_runs_all_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = fresh_test_dir("ablation-artifacts")
    monkeypatch.setattr(ablation.curriculum, "PROJECT_PREFIX", tmp_path)
    calls = []

    def fake_run_curriculum_variant(
        *,
        variant_args: object,
        log_dir: object,
    ) -> tuple[dict[str, object], str, str]:
        calls.append(variant_args)
        return smoke_summary(
            run_dir=tmp_path / "outputs" / "run-a" / variant_args.run_id,
            max_reset_count=4,
            max_tilt_bad_count=4,
        ), "completed", ""

    monkeypatch.setattr(ablation, "run_curriculum_variant", fake_run_curriculum_variant)
    args = ablation.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--output-root",
            str(tmp_path / "outputs"),
            "--run-id",
            "run-a",
        ]
    )

    summary = ablation.run_ablation(args)

    run_dir = tmp_path / "outputs" / "run-a"
    assert summary["all_variants_completed"] is True
    assert [call.run_id for call in calls] == [
        "baseline",
        "action_scale_0_05",
        "action_scale_0_03",
        "action_scale_0_01",
        "action_group_legs",
        "action_group_legs_waist",
        "log_std_neg3_5",
    ]
    assert all(call.stage_names == "standing" for call in calls)
    written_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    stage = written_summary["variants"][0]["seed_stage_summaries"][0]
    assert stage["first_tilt_update"] == 2
    assert stage["final_action_abs_mean"] == pytest.approx(0.12)
    assert stage["max_action_abs_max"] == pytest.approx(0.91)
    assert stage["final_top_action_rms_joints"] == [{"joint": "left_knee_joint", "rms": 0.42}]
    assert written_summary["baseline_reproduced_tilt_reset"] is True


def test_run_ablation_stops_when_baseline_does_not_reproduce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = fresh_test_dir("ablation-no-baseline-repro")
    monkeypatch.setattr(ablation.curriculum, "PROJECT_PREFIX", tmp_path)
    calls = []

    def fake_run_curriculum_variant(
        *,
        variant_args: object,
        log_dir: object,
    ) -> tuple[dict[str, object], str, str]:
        calls.append(variant_args)
        return smoke_summary(
            run_dir=tmp_path / "outputs" / "run-b" / variant_args.run_id,
            max_reset_count=0,
            max_tilt_bad_count=0,
        ), "completed", ""

    monkeypatch.setattr(ablation, "run_curriculum_variant", fake_run_curriculum_variant)
    args = ablation.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--output-root",
            str(tmp_path / "outputs"),
            "--run-id",
            "run-b",
        ]
    )

    summary = ablation.run_ablation(args)

    assert summary["status"] == "blocked_no_baseline_repro"
    assert summary["baseline_reproduced_tilt_reset"] is False
    assert summary["all_variants_completed"] is False
    assert summary["blocker"] == "baseline did not reproduce tilt reset waves"
    assert [call.run_id for call in calls] == ["baseline"]


def test_run_ablation_preserves_subprocess_failure_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = fresh_test_dir("ablation-failed-summary")
    monkeypatch.setattr(ablation.curriculum, "PROJECT_PREFIX", tmp_path)
    calls = []

    def fake_run_curriculum_variant(
        *,
        variant_args: object,
        log_dir: object,
    ) -> tuple[dict[str, object], str, str]:
        calls.append(variant_args)
        return smoke_summary(
            run_dir=tmp_path / "outputs" / "run-c" / variant_args.run_id,
            max_reset_count=2,
            max_tilt_bad_count=2,
            all_seeds_passed=False,
        ), "failed", "run_smoke pass criteria failed"

    monkeypatch.setattr(ablation, "run_curriculum_variant", fake_run_curriculum_variant)
    args = ablation.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--output-root",
            str(tmp_path / "outputs"),
            "--run-id",
            "run-c",
        ]
    )

    summary = ablation.run_ablation(args)

    assert summary["status"] == "blocked_no_baseline_repro"
    assert summary["blocker"] == "run_smoke pass criteria failed"
    assert summary["variants"][0]["status"] == "failed"
    assert [call.run_id for call in calls] == ["baseline"]


def test_run_curriculum_variant_reads_subprocess_success_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = fresh_test_dir("subprocess-success")
    variant_args = ablation.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--output-root",
            str(tmp_path / "outputs"),
            "--run-id",
            "baseline",
        ]
    )
    run_dir = tmp_path / "outputs" / "baseline"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(smoke_summary(run_dir=run_dir, max_reset_count=1, max_tilt_bad_count=1)),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ablation.subprocess,
        "run",
        lambda *args, **kwargs: completed_process(0, '{"status":"ok"}\n', ""),
    )

    summary, status, blocker = ablation.run_curriculum_variant(
        variant_args=variant_args,
        log_dir=tmp_path / "logs",
    )

    assert status == "completed"
    assert blocker == ""
    assert summary["all_seeds_passed"] is True


def test_run_curriculum_variant_reports_subprocess_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = fresh_test_dir("subprocess-failure")
    variant_args = ablation.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--output-root",
            str(tmp_path / "outputs"),
            "--run-id",
            "baseline",
        ]
    )
    monkeypatch.setattr(
        ablation.subprocess,
        "run",
        lambda *args, **kwargs: completed_process(
            1,
            '{"blocker":"synthetic failure"}\n',
            "stderr text",
        ),
    )

    summary, status, blocker = ablation.run_curriculum_variant(
        variant_args=variant_args,
        log_dir=tmp_path / "logs",
    )

    assert summary == {}
    assert status == "failed"
    assert blocker == "synthetic failure"
    assert (tmp_path / "logs" / "stdout.txt").read_text(encoding="utf-8")
    assert (tmp_path / "logs" / "stderr.txt").read_text(encoding="utf-8") == "stderr text"


def smoke_summary(
    *,
    run_dir: ablation.Path,
    max_reset_count: int,
    max_tilt_bad_count: int,
    all_seeds_passed: bool = True,
) -> dict[str, object]:
    return {
        "run_dir": str(run_dir),
        "all_seeds_passed": all_seeds_passed,
        "min_collect_env_policy_steps_per_sec": 12345.0,
        "mean_reward_mean": 1.5,
        "seeds": [
            {
                "seed": 0,
                "stages": [
                    {
                        "stage": "standing",
                        "first_tilt_update": 2 if max_tilt_bad_count else None,
                        "max_reset_count": max_reset_count,
                        "mean_reset_count": float(max_reset_count),
                        "final_reset_count": max_reset_count,
                        "max_tilt_bad_count": max_tilt_bad_count,
                        "final_tilt_bad_count": max_tilt_bad_count,
                        "final_termination_height_bad_count": 0,
                        "max_approx_kl": 0.02,
                        "final_approx_kl": 0.01,
                        "final_entropy": 10.0,
                        "final_reward_mean": 1.5,
                        "min_root_height_min": 0.55,
                        "final_root_height_mean": 0.80,
                        "final_root_height_min": 0.60,
                        "min_upright_mean": 0.65,
                        "final_upright_mean": 0.70,
                        "final_action_abs_mean": 0.12,
                        "max_action_abs_max": 0.91,
                        "final_action_abs_max": 0.80,
                        "final_action_std": 0.21,
                        "final_top_action_rms_joints": [
                            {"joint": "left_knee_joint", "rms": 0.42}
                        ],
                        "min_collect_env_policy_steps_per_sec": 12345.0,
                    }
                ],
            }
        ],
    }


def completed_process(returncode: int, stdout: str, stderr: str) -> object:
    return ablation.subprocess.CompletedProcess(
        args=["python", "-m", "h200_locomotion_lab.tools.g1_curriculum_ppo_smoke"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def fresh_test_dir(name: str) -> ablation.Path:
    root = (
        ablation.Path.cwd()
        / ".test_tmp_task017"
        / f"{name}-{time.time_ns()}"
    ).resolve()
    root.mkdir(parents=True)
    return root
