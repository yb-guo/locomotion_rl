import json
import time

import pytest

from h200_locomotion_lab.tools import g1_tilt_reset_ablation as ablation


def test_parse_args_defaults_to_task016_output_root_and_seed0() -> None:
    args = ablation.parse_args([])

    assert args.seeds == "0"
    assert args.updates_per_stage == 50
    assert args.output_root == ablation.Path("outputs/task016/tilt_reset_ablation")


def test_ablation_variants_are_one_variable_at_a_time() -> None:
    variants = {variant.name: variant.overrides for variant in ablation.ABLATION_VARIANTS}

    assert list(variants) == [
        "baseline",
        "lr_1e4",
        "termination_penalty_neg5",
        "action_rate_penalty_high",
    ]
    assert variants["baseline"] == {}
    assert variants["lr_1e4"] == {"lr": "0.0001"}
    assert variants["termination_penalty_neg5"] == {"termination_penalty": "-5.0"}
    assert variants["action_rate_penalty_high"] == {"action_rate_penalty_scale": "0.05"}


def test_build_curriculum_args_applies_variant_override_only() -> None:
    args = ablation.parse_args(
        [
            "--backend",
            "cpu",
            "--logical-cuda-device",
            "cpu",
            "--seeds",
            "0",
            "--lr",
            "0.0003",
            "--termination-penalty",
            "0.0",
        ]
    )

    lr_args = ablation.build_curriculum_args(
        args=args,
        variant=ablation.ABLATION_VARIANTS[1],
        output_root=ablation.Path("outputs"),
    )
    penalty_args = ablation.build_curriculum_args(
        args=args,
        variant=ablation.ABLATION_VARIANTS[2],
        output_root=ablation.Path("outputs"),
    )

    assert lr_args.lr == pytest.approx(0.0001)
    assert lr_args.termination_penalty == 0.0
    assert penalty_args.lr == pytest.approx(0.0003)
    assert penalty_args.termination_penalty == -5.0


def test_run_ablation_writes_aggregate_and_variant_references(
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
        return {
            "run_dir": str(tmp_path / "outputs" / "run-a" / variant_args.run_id),
            "all_seeds_passed": True,
            "min_collect_env_policy_steps_per_sec": 12345.0,
            "mean_reward_mean": 1.5,
            "seeds": [
                {
                    "seed": 0,
                    "stages": [
                        {
                            "stage": "standing",
                            "first_tilt_update": 2,
                            "max_reset_count": 4,
                            "mean_reset_count": 2.0,
                            "final_reset_count": 3,
                            "max_tilt_bad_count": 4,
                            "final_tilt_bad_count": 3,
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
                            "min_collect_env_policy_steps_per_sec": 12345.0,
                        }
                    ],
                }
            ],
        }, "completed", ""

    monkeypatch.setattr(
        ablation,
        "run_curriculum_variant",
        fake_run_curriculum_variant,
    )
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
        "lr_1e4",
        "termination_penalty_neg5",
        "action_rate_penalty_high",
    ]
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "summary.json").is_file()
    written_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert written_summary["variants"][0]["seed_stage_summaries"][0][
        "first_tilt_update"
    ] == 2
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
        return {
            "run_dir": str(tmp_path / "outputs" / "run-b" / variant_args.run_id),
            "all_seeds_passed": True,
            "min_collect_env_policy_steps_per_sec": 12345.0,
            "mean_reward_mean": 1.5,
            "seeds": [
                {
                    "seed": 0,
                    "stages": [
                        {
                            "stage": "standing",
                            "first_tilt_update": None,
                            "max_reset_count": 0,
                            "mean_reset_count": 0.0,
                            "final_reset_count": 0,
                            "max_tilt_bad_count": 0,
                            "final_tilt_bad_count": 0,
                            "final_termination_height_bad_count": 0,
                            "max_approx_kl": 0.02,
                            "final_approx_kl": 0.01,
                            "final_entropy": 10.0,
                            "final_reward_mean": 1.5,
                            "min_root_height_min": 1.1,
                            "final_root_height_mean": 1.2,
                            "final_root_height_min": 1.1,
                            "min_upright_mean": 1.0,
                            "final_upright_mean": 1.0,
                            "min_collect_env_policy_steps_per_sec": 12345.0,
                        }
                    ],
                }
            ],
        }, "completed", ""

    monkeypatch.setattr(
        ablation,
        "run_curriculum_variant",
        fake_run_curriculum_variant,
    )
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
    assert [call.run_id for call in calls] == ["baseline"]


def test_run_ablation_marks_failed_run_smoke_summary_as_failed(
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
        return {
            "run_dir": str(tmp_path / "outputs" / "run-c" / variant_args.run_id),
            "status": "failed",
            "all_seeds_passed": False,
            "min_collect_env_policy_steps_per_sec": 0.0,
            "mean_reward_mean": 0.0,
            "seeds": [
                {
                    "seed": 0,
                    "stages": [
                        {
                            "stage": "standing",
                            "first_tilt_update": 0,
                            "max_reset_count": 2,
                            "mean_reset_count": 2.0,
                            "final_reset_count": 2,
                            "max_tilt_bad_count": 2,
                            "final_tilt_bad_count": 2,
                            "final_termination_height_bad_count": 0,
                            "max_approx_kl": 0.0,
                            "final_approx_kl": 0.0,
                            "final_entropy": 0.0,
                            "final_reward_mean": 0.0,
                            "min_root_height_min": 0.0,
                            "final_root_height_mean": 0.0,
                            "final_root_height_min": 0.0,
                            "min_upright_mean": 0.0,
                            "final_upright_mean": 0.0,
                            "min_collect_env_policy_steps_per_sec": 0.0,
                        }
                    ],
                }
            ],
        }, "failed", "run_smoke pass criteria failed"

    monkeypatch.setattr(
        ablation,
        "run_curriculum_variant",
        fake_run_curriculum_variant,
    )
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


def fresh_test_dir(name: str) -> ablation.Path:
    root = (
        ablation.Path.cwd()
        / ".test_tmp_task016"
        / f"{name}-{time.time_ns()}"
    ).resolve()
    root.mkdir(parents=True)
    return root
