import importlib.util
import inspect
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_task044_continuous_fault_eval_parse_args_defaults() -> None:
    module = _load_src_tool("task044_continuous_fault_eval.py")

    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])

    assert args.task == module.TASK044_PERSISTENT_HIDDEN_POSE_TIGHT_TASK_ID
    assert args.dynamic_dead_joint == "left_knee_joint"
    assert args.dynamic_onset_s == 2.0
    assert args.dynamic_recovery_s == 999.0
    assert args.startup_excluded_s == 0.5
    assert args.post_fault_window_s == 2.0
    assert args.reset_time_bin_s == 0.5
    assert args.base_obs_passthrough_scale == 1.0
    assert args.adaptation_warmstart_scale == 1.0
    assert args.memory_ablation_mode == "none"
    assert args.expected_runner_cls == module.TASK044_CONTINUOUS_EXPECTED_RUNNER_CLS


def test_task044_continuous_fault_eval_accepts_zero_memory_latent_ablation() -> None:
    module = _load_src_tool("task044_continuous_fault_eval.py")

    args = module.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--output-json",
            "out.json",
            "--memory-ablation-mode",
            "zero_memory_latent",
        ]
    )

    assert args.memory_ablation_mode == "zero_memory_latent"


def test_task044_continuous_runner_does_not_install_multitrial_inner_reset() -> None:
    import h200_locomotion_lab.training.rsl_history_wrapper as wrapper

    source = inspect.getsource(wrapper.Task044TrueTxlMemoryK160ContinuousRunner)

    assert "install_task037_inner_reset_controller" not in source
    assert "Task037MultiTrialVecEnvWrapper" not in source
    assert "Task033HistoryVecEnvWrapper" in source
    assert "Task038TrueTxlResetHookVecEnvWrapper" in source


def test_task044_continuous_window_json_uses_coverage_as_completion_ratio() -> None:
    module = _load_src_tool("task044_continuous_fault_eval.py")

    window = module._continuous_window_json(
        _FakeStats(sample_count=80, fall_count=1),
        trial_idx=0,
        num_envs=4,
        expected_samples=100,
    )

    assert window["sample_count"] == 80
    assert window["expected_sample_count"] == 100
    assert window["completion_count"] == 80
    assert window["completion_ratio"] == 0.8
    assert window["coverage_ratio"] == 0.8
    assert window["fall_ratio"] == 0.25


def test_task044_continuous_expected_window_samples() -> None:
    module = _load_src_tool("task044_continuous_fault_eval.py")

    assert module._expected_window_samples(
        num_envs=2,
        steps=10,
        dt=0.1,
        start_s=0.2,
        end_s=0.5,
    ) == 6


def test_task044_reset_time_diagnostic_bins_and_segments() -> None:
    module = _load_src_tool("task044_continuous_fault_eval.py")

    diagnostic = module._make_physical_reset_time_diagnostic(
        num_envs=4,
        steps=8,
        dt=0.5,
        bin_s=1.0,
        dynamic_onset_s=1.0,
        post_start_s=1.5,
        post_end_s=3.0,
    )
    module._add_physical_reset_time_events(
        diagnostic,
        step=0,
        reset_count=1,
        fall_count=1,
        timeout_count=0,
    )
    module._add_physical_reset_time_events(
        diagnostic,
        step=2,
        reset_count=2,
        fall_count=1,
        timeout_count=1,
    )
    module._add_physical_reset_time_events(
        diagnostic,
        step=7,
        reset_count=1,
        fall_count=0,
        timeout_count=1,
    )

    final = module._finalize_physical_reset_time_diagnostic(
        diagnostic,
        num_envs=4,
        dt=0.5,
        first_reset_steps=[0, 2, 7, -1],
        first_fall_steps=[0, 2, -1, -1],
    )

    assert final["schema"] == "task044_physical_reset_time_diagnostic_v1"
    assert final["diagnostic_only"]
    assert final["event_scope"] == "physical_env_done"
    assert final["totals"] == {"reset_count": 4, "fall_count": 2, "timeout_count": 2}
    assert final["totals"]["reset_count"] == (
        final["totals"]["fall_count"] + final["totals"]["timeout_count"]
    )
    assert final["segments"]["pre_fault"]["reset_count"] == 1
    assert final["segments"]["fault_onset_to_post_window_start"]["reset_count"] == 2
    assert final["segments"]["post_fault_after_window"]["reset_count"] == 1
    assert final["bins"][0]["reset_count"] == 1
    assert final["bins"][1]["reset_count"] == 2
    assert final["bins"][3]["reset_count"] == 1
    assert final["first_reset"]["env_count"] == 3
    assert final["first_reset"]["max_time_s"] == 3.5
    assert final["first_fall"]["env_ratio"] == 0.5
    assert final["segments"]["fault_onset_to_post_window_start"]["reset_events_per_env"] == 0.5


def test_task044_continuous_failure_reasons_report_quality_and_continuity() -> None:
    module = _load_src_tool("task044_continuous_fault_eval.py")
    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "out.json"])
    thresholds = module._thresholds(args)
    window = _FakeStats(sample_count=100).to_json(trial_idx=0, num_envs=4)
    window["expected_sample_count"] = 100
    window["coverage_ratio"] = 1.0
    window["completion_ratio"] = 1.0
    window["lin_vel_error"]["mean"] = thresholds["max_post_fault_lin_vel_error"] + 0.1

    reasons = module._failure_reasons(
        args=args,
        action_dim=31,
        total_action_dim=31,
        actor_model_class="Task038TrueTxlMemoryModel",
        physical_continuity_pass=False,
        post_fault_window=window,
        post_fault_window_pass=False,
        thresholds=thresholds,
    )

    assert "physical_continuity_not_preserved" in reasons
    assert "post_fault_window_quality_not_passed" in reasons
    assert "post_fault_lin_vel_error_too_high" in reasons


def test_task044_continuous_memory_debug_active_requires_forward_samples_and_memory() -> None:
    module = _load_src_tool("task044_continuous_fault_eval.py")

    assert module._memory_debug_active(
        {
            "total_actor_forward_batches": 1,
            "total_actor_forward_samples": 4,
            "last_attended_previous_memory_lengths": [[64, 64]],
        }
    )
    assert not module._memory_debug_active(
        {
            "total_actor_forward_batches": 1,
            "total_actor_forward_samples": 4,
            "last_attended_previous_memory_lengths": [],
        }
    )


class _FakeStats:
    def __init__(self, *, sample_count: int, fall_count: int = 0) -> None:
        self.sample_count = sample_count
        self.fall_count = fall_count

    def to_json(self, *, trial_idx: int, num_envs: int) -> dict:
        return {
            "trial_index": trial_idx,
            "sample_count": self.sample_count,
            "completion_count": 0,
            "completion_ratio": 0.0,
            "fall_count": self.fall_count,
            "fall_ratio": 0.0,
            "zero_fall_ratio": 1.0,
            "timeout_count": 0,
            "reset_reason_counts": {"1": self.fall_count} if self.fall_count else {},
            "reward_mean": 0.0,
            "lin_vel_command": {"mean_x": 1.6, "mean_y": 0.0},
            "lin_vel_actual": {"mean_x": 1.4, "mean_y": 0.0},
            "lin_vel_error_components": {"mean_abs_x": 0.2, "mean_abs_y": 0.0},
            "lin_vel_error": {"mean": 0.2},
            "yaw_vel_error": {"mean": 0.0},
            "gravity_xy": {"mean": 0.0, "max": 0.1},
            "root_z": {"mean": 0.8, "min": 0.75},
        }


def _load_src_tool(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module
