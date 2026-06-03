import importlib.util
from pathlib import Path


TASK037_DIR = (
    Path(__file__).resolve().parents[1]
    / ".agent"
    / "task"
    / "task037-locoformer-style-multitrial-long-context-training"
)


def test_task037_registration_patch_is_idempotent() -> None:
    module = _load_script("task037_register_multitrial_stages.py")
    init_path = MemoryPath(
        "from mjlab.rl import MjlabOnPolicyRunner\n"
        "from mjlab.tasks.registry import register_mjlab_task\n",
    )

    module.patch_init(init_path)
    once = init_path.read_text(encoding="utf-8")
    module.patch_init(init_path)
    twice = init_path.read_text(encoding="utf-8")

    task_id = "Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-AutoReset-Fast2p0"
    deterministic_task_id = (
        "Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-DeterministicInnerReset-Fast2p0"
    )
    adapt_task_id = "Unitree-G1-Gripper-Flat-Task037-AdaptK4-DeterministicInnerReset-Fast2p0"
    adapt_k160_clean_task_id = (
        "Unitree-G1-Gripper-Flat-Task037-AdaptK160-CleanUnified-Fast2p0"
    )
    adapt_k160_weak_task_id = (
        "Unitree-G1-Gripper-Flat-Task037-AdaptK160-WeakPersistent-Fast2p0"
    )
    adapt_k160_mixed_task_id = (
        "Unitree-G1-Gripper-Flat-Task037-AdaptK160-MixedPersistent-Fast2p0"
    )
    adapt_k160_deadgrid_task_id = (
        "Unitree-G1-Gripper-Flat-Task037-AdaptK160-FocusedDeadGrid-Fast2p0"
    )
    adapt_k160_dynamic_task_id = (
        "Unitree-G1-Gripper-Flat-Task037-AdaptK160-DynamicMotorFailure-Fast1p6"
    )
    adapt_k160_knee2p0_task_id = (
        "Unitree-G1-Gripper-Flat-Task037-AdaptK160-KneeHipRollVx2p0"
    )
    adapt_k160_rightknee_task_id = (
        "Unitree-G1-Gripper-Flat-Task037-AdaptK160-RightKneeMixedVx2p0"
    )
    txl_task_id = (
        "Unitree-G1-Gripper-Flat-Task037-TxlMemoryK160-DeterministicInnerReset-Fast2p0"
    )
    txl_deadgrid_task_id = "Unitree-G1-Gripper-Flat-Task037-TxlMemoryK160-FocusedDeadGrid-Fast2p0"
    txl_dynamic_task_id = "Unitree-G1-Gripper-Flat-Task037-TxlMemoryK160-DynamicMotorFailure-Fast1p6"
    assert once == twice
    assert once.count("Task037BufferOnlyK4AutoResetRunner") == 2
    assert once.count("Task037BufferOnlyK4DeterministicInnerResetRunner") == 2
    assert once.count("Task037AdaptK4DeterministicInnerResetRunner") == 2
    assert once.count("Task037AdaptK160DeterministicInnerResetRunner") == 8
    assert once.count("Task037TxlMemoryK160DeterministicRunner") == 4
    assert once.count(task_id) == 1
    assert once.count(deterministic_task_id) == 1
    assert once.count(adapt_task_id) == 1
    assert once.count(adapt_k160_clean_task_id) == 1
    assert once.count(adapt_k160_weak_task_id) == 1
    assert once.count(adapt_k160_mixed_task_id) == 1
    assert once.count(adapt_k160_deadgrid_task_id) == 1
    assert once.count(adapt_k160_dynamic_task_id) == 1
    assert once.count(adapt_k160_knee2p0_task_id) == 1
    assert once.count(adapt_k160_rightknee_task_id) == 1
    assert once.count("unitree_g1_gripper_flat_task035_clean_unified_env_cfg") == 2
    assert once.count("unitree_g1_gripper_flat_task035_weak_persistent_env_cfg") == 2
    assert once.count("unitree_g1_gripper_flat_task035_mixed_persistent_env_cfg") == 2
    assert once.count(txl_task_id) == 1
    assert once.count(txl_deadgrid_task_id) == 1
    assert once.count(txl_dynamic_task_id) == 1


def test_task037_extras_probe_parse_args_defaults_to_task037_id() -> None:
    module = _load_src_tool("task037_mjlab_multitrial_extras_probe.py")

    args = module.parse_args(["--output-json", "out.json"])

    assert args.task == "Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-AutoReset-Fast2p0"
    assert args.output_json == "out.json"
    assert args.episode_length_s > 0


def test_task037_inner_reset_probe_parse_args_defaults_to_deterministic_task_id() -> None:
    module = _load_src_tool("task037_mjlab_inner_reset_probe.py")

    args = module.parse_args(["--output-json", "inner.json"])

    assert args.task == (
        "Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-DeterministicInnerReset-Fast2p0"
    )
    assert args.output_json == "inner.json"
    assert args.max_command_delta > 0


def test_task037_multitrial_eval_parse_args_defaults_to_adapt_task_id() -> None:
    module = _load_src_tool("task037_multitrial_eval_checkpoint.py")

    args = module.parse_args(["--checkpoint", "model.pt", "--output-json", "eval.json"])

    assert args.task == "Unitree-G1-Gripper-Flat-Task037-AdaptK4-DeterministicInnerReset-Fast2p0"
    assert args.checkpoint == "model.pt"
    assert args.output_json == "eval.json"
    assert args.trial_length_s == 2.0
    assert args.final_window_s == 0.0
    assert args.min_final_completion_ratio > 0
    assert args.memory_latent_dim is None
    assert args.base_obs_passthrough_scale is None
    assert args.adaptation_warmstart_scale is None
    assert args.action_dim is None
    assert args.adaptation_hidden_dim is None
    assert args.base_obs_passthrough is None
    assert args.adaptation_warmstart is None


def test_task037_multitrial_eval_parse_args_supports_matrix_modes() -> None:
    module = _load_src_tool("task037_multitrial_eval_checkpoint.py")

    args = module.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--output-json",
            "eval.json",
            "--dynamic-case",
            "switch",
            "--force-dead-joint",
            "right_knee_joint",
            "--dynamic-dead-joint",
            "left_hip_yaw_joint",
            "--dynamic-onset-s",
            "0.5",
            "--dynamic-recovery-s",
            "1.5",
            "--final-window-s",
            "0.5",
            "--memory-latent-dim",
            "32",
            "--base-obs-passthrough-scale",
            "0.5",
            "--adaptation-warmstart-scale",
            "0.5",
            "--action-dim",
            "31",
            "--adaptation-hidden-dim",
            "128",
            "--base-obs-passthrough",
            "--adaptation-warmstart",
        ]
    )

    assert args.dynamic_case == "switch"
    assert args.force_dead_joint == "right_knee_joint"
    assert args.dynamic_dead_joint == "left_hip_yaw_joint"
    assert args.dynamic_onset_s == 0.5
    assert args.dynamic_recovery_s == 1.5
    assert args.final_window_s == 0.5
    assert args.dead_scale == 0.0
    assert args.memory_latent_dim == 32
    assert args.base_obs_passthrough_scale == 0.5
    assert args.adaptation_warmstart_scale == 0.5
    assert args.action_dim == 31
    assert args.adaptation_hidden_dim == 128
    assert args.base_obs_passthrough is True
    assert args.adaptation_warmstart is True


def test_task037_multitrial_eval_final_window_steps_helper() -> None:
    module = _load_src_tool("task037_multitrial_eval_checkpoint.py")

    assert module._final_window_steps(0.0, 0.02) is None
    assert module._final_window_steps(0.5, 0.02) == 25
    assert module._final_window_steps(0.501, 0.02) == 26


def test_task037_multitrial_eval_action_accumulator_records_masked_stats() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    module = _load_src_tool("task037_multitrial_eval_checkpoint.py")
    accumulator = module._ActionAccumulator(torch, "cpu")

    accumulator.add_sample(
        torch.tensor([True, False, True]),
        torch.tensor(
            [
                [1.0, -2.0],
                [100.0, 100.0],
                [3.0, 4.0],
            ]
        ),
    )

    stats = accumulator.to_json(top_k=2)

    assert stats["sample_count"] == 2
    assert stats["action_dim"] == 2
    assert stats["mean_abs_by_dim"] == [2.0, 3.0]
    assert stats["max_abs_by_dim"] == [3.0, 4.0]
    assert stats["top_abs_dims"][0]["dim"] == 1


def test_task037_multitrial_eval_metadata_helpers_are_non_breaking() -> None:
    module = _load_src_tool("task037_multitrial_eval_checkpoint.py")
    env = type("Env", (), {"num_actions": 31})()
    base = type(
        "Base",
        (),
        {
            "action_manager": type("ActionManager", (), {"total_action_dim": 31})(),
        },
    )()

    assert module._action_dim(env, base) == 31
    assert module._total_action_dim(base) == 31
    runner = type("Runner", (), {"alg": type("Alg", (), {"actor": object()})()})()
    assert module._find_actor(runner) is not None


def test_task037_multitrial_eval_optional_txl_cfg_skips_none_values() -> None:
    from types import SimpleNamespace

    module = _load_src_tool("task037_multitrial_eval_checkpoint.py")
    train_cfg = {
        "actor": {
            "memory_latent_dim": 32,
            "base_obs_passthrough_scale": 1.0,
        }
    }

    module._apply_optional_txl_actor_cfg(
        SimpleNamespace(memory_latent_dim=None, base_obs_passthrough_scale=0.5),
        train_cfg,
    )

    assert train_cfg["actor"]["memory_latent_dim"] == 32
    assert train_cfg["actor"]["base_obs_passthrough_scale"] == 0.5


def test_task037_multitrial_eval_txl_debug_snapshot_is_non_breaking() -> None:
    module = _load_src_tool("task037_multitrial_eval_checkpoint.py")

    class ActorWithDebug:
        def txl_debug_snapshot(self) -> dict:
            return {
                "envs": [
                    {
                        "env_id": 0,
                        "incremental_steps": 3,
                        "last_attended_previous_memory_lengths": [0, 1],
                    }
                ]
            }

    assert module._txl_debug_snapshot(ActorWithDebug())["envs"][0]["incremental_steps"] == 3
    assert module._txl_debug_snapshot(None) == {}
    assert module._txl_debug_snapshot(object()) == {}

    class ActorWithInvalidDebug:
        def txl_debug_snapshot(self) -> list:
            return []

    assert module._txl_debug_snapshot(ActorWithInvalidDebug()) == {}


def test_task037_clean_gait_prior_launch_script_uses_clean_adapt_k160_task() -> None:
    script = (TASK037_DIR / "task037_launch_clean_gait_prior.sh").read_text(encoding="utf-8")

    assert "Unitree-G1-Gripper-Flat-Task037-AdaptK160-CleanUnified-Fast2p0" in script
    assert "task037_adaptk4_model5408_k160_warmstart" in script
    assert "DRY_RUN" in script


def test_task037_adaptk160_failure_curriculum_launch_script_has_stage_tasks() -> None:
    script = (TASK037_DIR / "task037_launch_adaptk160_failure_curriculum.sh").read_text(
        encoding="utf-8"
    )

    assert "STAGE=weak|mixed|deadgrid|dynamic|knee2p0|rightknee2p0" in script
    assert "Unitree-G1-Gripper-Flat-Task037-AdaptK160-WeakPersistent-Fast2p0" in script
    assert "Unitree-G1-Gripper-Flat-Task037-AdaptK160-MixedPersistent-Fast2p0" in script
    assert "Unitree-G1-Gripper-Flat-Task037-AdaptK160-FocusedDeadGrid-Fast2p0" in script
    assert "Unitree-G1-Gripper-Flat-Task037-AdaptK160-DynamicMotorFailure-Fast1p6" in script
    assert "Unitree-G1-Gripper-Flat-Task037-AdaptK160-KneeHipRollVx2p0" in script
    assert "Unitree-G1-Gripper-Flat-Task037-AdaptK160-RightKneeMixedVx2p0" in script
    assert "model_5467.pt" in script


def _load_script(name: str):
    path = TASK037_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_src_tool(name: str):
    path = Path(__file__).resolve().parents[1] / "src/h200_locomotion_lab/tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MemoryPath:
    def __init__(self, text: str) -> None:
        self.text = text

    def read_text(self, encoding: str) -> str:
        assert encoding == "utf-8"
        return self.text

    def write_text(self, text: str, encoding: str) -> None:
        assert encoding == "utf-8"
        self.text = text
