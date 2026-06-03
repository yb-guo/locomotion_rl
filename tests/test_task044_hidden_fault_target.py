import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_task044_hidden_fault_train_defaults_to_task044_id() -> None:
    module = _load_src_tool("task044_hidden_fault_train.py")

    args = module.parse_args(["--iterations", "1", "--num-envs", "8", "--num-mini-batches", "4"])

    assert args.task == module.TASK044_HIDDEN_FAULT_TASK_ID
    assert args.output_json == module.DEFAULT_OUTPUT_JSON
    assert args.log_dir == module.DEFAULT_LOG_DIR
    assert args.experiment_name == module.DEFAULT_EXPERIMENT_NAME
    assert args.run_name.startswith("seq_txl_hidden_fault_env")
    assert args.expected_runner_cls == module.TASK044_EXPECTED_RUNNER_CLS


def test_task044_hidden_fault_train_accepts_task046_retry_context_consumer_flags() -> None:
    module = _load_src_tool("task044_hidden_fault_train.py")

    args = module.parse_args(
        [
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
            "--task046-retry-context",
            "--task046-retry-context-num-trials",
            "3",
            "--task046-retry-context-final-trial-index",
            "2",
            "--task046-retry-context-step-window-steps",
            "50",
            "--task046-post-reset-recovery-reward",
            "--task046-early-velocity-weight",
            "1.2",
            "--task046-tail-velocity-weight",
            "0.15",
            "--task046-orientation-weight",
            "0.25",
        ]
    )

    assert args.task == module.TASK044_HIDDEN_FAULT_TASK_ID
    assert args.expected_runner_cls == module.TASK044_EXPECTED_RUNNER_CLS
    assert args.task046_retry_context is True
    assert args.task046_retry_context_num_trials == 3
    assert args.task046_retry_context_final_trial_index == 2
    assert args.task046_retry_context_step_window_steps == 50
    assert args.task046_post_reset_recovery_reward is True
    module.preflight_args(args)


def test_task044_hidden_fault_train_preflight_uses_task044_expected_task() -> None:
    module = _load_src_tool("task044_hidden_fault_train.py")
    args = module.parse_args(["--iterations", "1", "--num-envs", "8", "--num-mini-batches", "4"])

    module.preflight_args(args)

    wrong = module.parse_args(
        [
            "--task",
            "WrongTask",
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    with pytest.raises(module.base_train.PreflightError) as exc_info:
        module.preflight_args(wrong)
    assert exc_info.value.reasons == ["task_not_task044_hidden_fault_allowed"]

    aligned = module.parse_args(
        [
            "--task",
            module.TASK044_EVAL_ALIGNED_LEFT_KNEE_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(aligned)

    persistent = module.parse_args(
        [
            "--task",
            module.TASK044_PERSISTENT_HIDDEN_VELBOOST_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(persistent)

    immediate = module.parse_args(
        [
            "--task",
            module.TASK044_PERSISTENT_IMMEDIATE_VELBOOST_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(immediate)

    left_knee = module.parse_args(
        [
            "--task",
            module.TASK044_PERSISTENT_IMMEDIATE_LEFT_KNEE_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(left_knee)

    speed_push = module.parse_args(
        [
            "--task",
            module.TASK044_PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(speed_push)

    speed_stability = module.parse_args(
        [
            "--task",
            module.TASK044_PERSISTENT_HIDDEN_SPEED_STABILITY_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(speed_stability)

    height_guard = module.parse_args(
        [
            "--task",
            module.TASK044_PERSISTENT_HIDDEN_HEIGHT_GUARD_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(height_guard)

    height_guard_strong = module.parse_args(
        [
            "--task",
            module.TASK044_PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(height_guard_strong)

    low_root_terminate = module.parse_args(
        [
            "--task",
            module.TASK044_PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(low_root_terminate)

    pose_terminate = module.parse_args(
        [
            "--task",
            module.TASK044_PERSISTENT_HIDDEN_POSE_TERMINATE_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(pose_terminate)

    pose_tight = module.parse_args(
        [
            "--task",
            module.TASK044_PERSISTENT_HIDDEN_POSE_TIGHT_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(pose_tight)

    speed_pose_balance = module.parse_args(
        [
            "--task",
            module.TASK044_PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(speed_pose_balance)

    forward_floor = module.parse_args(
        [
            "--task",
            module.TASK044_PERSISTENT_HIDDEN_FORWARD_FLOOR_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(forward_floor)

    forward_target = module.parse_args(
        [
            "--task",
            module.TASK044_PERSISTENT_HIDDEN_FORWARD_TARGET_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(forward_target)

    speed_curriculum = module.parse_args(
        [
            "--task",
            module.TASK044_PERSISTENT_HIDDEN_SPEED_CURRICULUM_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(speed_curriculum)

    immediate_left_knee_pose_forward = module.parse_args(
        [
            "--task",
            module.TASK044_PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(immediate_left_knee_pose_forward)

    immediate_left_knee_survival = module.parse_args(
        [
            "--task",
            module.TASK045_PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(immediate_left_knee_survival)

    immediate_left_knee_long_survival = module.parse_args(
        [
            "--task",
            module.TASK045_PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(immediate_left_knee_long_survival)

    pose_tight_gate_long_tail = module.parse_args(
        [
            "--task",
            module.TASK045_POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(pose_tight_gate_long_tail)

    startup_boost = module.parse_args(
        [
            "--task",
            module.TASK044_PERSISTENT_HIDDEN_STARTUP_BOOST_TASK_ID,
            "--iterations",
            "1",
            "--num-envs",
            "8",
            "--num-mini-batches",
            "4",
        ]
    )
    module.preflight_args(startup_boost)

    wrapped = module.wrap_train_summary(
        {
            "task": module.TASK044_PERSISTENT_HIDDEN_VELBOOST_TASK_ID,
            "train_pipeline_pass": True,
        }
    )
    assert wrapped["task044_train_task_id"] == module.TASK044_PERSISTENT_HIDDEN_VELBOOST_TASK_ID


def test_task044_hidden_fault_eval_defaults_and_wraps_hidden_contract() -> None:
    module = _load_src_tool("task044_hidden_fault_eval.py")

    args = module.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--output-json",
            "out.json",
        ]
    )
    wrapped = module.wrap_eval_summary(
        {
            "task": module.TASK044_PERSISTENT_HIDDEN_VELBOOST_TASK_ID,
            "pipeline_pass": True,
            "quality_gate_pass": True,
            "pass": True,
            "memory_ablation_mode": "none",
        }
    )

    assert args.task == module.TASK044_HIDDEN_FAULT_TASK_ID
    assert args.lin_vel_x == 1.6
    assert args.dynamic_dead_joint == "left_knee_joint"
    assert args.dynamic_onset_s == 0.0
    assert args.dynamic_recovery_s == 2.0
    assert args.final_window_s == 0.5
    assert wrapped["task044_eval_task_id"] == module.TASK044_PERSISTENT_HIDDEN_VELBOOST_TASK_ID
    assert module.TASK044_EXPECTED_RUNNER_CLS == "Task044TrueTxlMemoryK160ClearHistoryRunner"
    assert wrapped["task044_hidden_fault_eval"] is True
    assert wrapped["task044_eval_pipeline_pass"] is True
    assert wrapped["task044_hidden_fault_contract"]["fault_identity_in_actor_obs"] is False
    assert wrapped["memory_causality_claim"] is False
    assert wrapped["reproduction_claim"] is False
    assert wrapped["superiority_claim"] is False


def test_task044_register_hidden_fault_stage_inserts_task_and_runner() -> None:
    module = _load_task044_script("task044_register_hidden_fault_stage.py")
    source = "from mjlab.rl import MjlabOnPolicyRunner\n"

    patched = module._ensure_runner_import(source)

    assert "Task044TrueTxlMemoryK160ClearHistoryRunner" in patched
    assert module.TASK_ID == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-Fast1p6"
    assert (
        module.EVAL_ALIGNED_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-EvalLeftKnee1p6"
    )
    assert (
        module.EVAL_ALIGNED_VELBOOST_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-EvalLeftKneeVelBoost1p6"
    )
    assert (
        module.PERSISTENT_HIDDEN_VELBOOST_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenVelBoost1p6"
    )
    assert (
        module.PERSISTENT_IMMEDIATE_VELBOOST_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateVelBoost1p6"
    )
    assert (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneeDeadVelBoost1p6"
    )
    assert (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneeDeadSpeedPush1p6"
    )
    assert (
        module.PERSISTENT_HIDDEN_SPEED_STABILITY_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenSpeedStability1p6"
    )
    assert (
        module.PERSISTENT_HIDDEN_HEIGHT_GUARD_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenHeightGuard1p6"
    )
    assert (
        module.PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenHeightGuardStrong1p6"
    )
    assert (
        module.PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenLowRootTerminate1p6"
    )
    assert (
        module.PERSISTENT_HIDDEN_POSE_TERMINATE_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenPoseTerminate1p6"
    )
    assert (
        module.PERSISTENT_HIDDEN_POSE_TIGHT_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenPoseTight1p6"
    )
    assert (
        module.PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenSpeedPoseBalance1p6"
    )
    assert (
        module.PERSISTENT_HIDDEN_FORWARD_FLOOR_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenForwardFloor1p6"
    )
    assert (
        module.PERSISTENT_HIDDEN_FORWARD_TARGET_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenForwardTarget1p6"
    )
    assert (
        module.PERSISTENT_HIDDEN_SPEED_CURRICULUM_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenSpeedCurriculum1p4To1p6"
    )
    assert (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneePoseForward1p6"
    )
    assert (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task045-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneePoseForwardSurvival1p6"
    )
    assert (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task045-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneeLongSurvival1p6"
    )
    assert (
        module.POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task045-TrueTxlHiddenFaultTrain-PoseTightGateLeftKneeLongTail1p6"
    )
    assert (
        module.PERSISTENT_HIDDEN_STARTUP_BOOST_TASK_ID
        == "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenStartupBoost1p6"
    )
    assert "unitree_g1_gripper_flat_dynamic_failure_train_fast1p6_env_cfg" in module.FIXED_SPEED_HELPER_BLOCK
    assert 'params["clean_probability"] = 0.05' in module.FIXED_SPEED_HELPER_BLOCK
    assert 'params["dynamic_single_probability"] = 0.45' in module.FIXED_SPEED_HELPER_BLOCK
    assert "cfg.episode_length_s = 2.0" in module.EVAL_ALIGNED_HELPER_BLOCK
    assert '"left_knee_joint", "dead", 0.0' in module.EVAL_ALIGNED_HELPER_BLOCK
    assert 'lin_vel_reward.weight = 3.0' in module.EVAL_ALIGNED_VELBOOST_HELPER_BLOCK
    assert 'lin_vel_reward.params["std"] = 1.0' in module.EVAL_ALIGNED_VELBOOST_HELPER_BLOCK
    assert 'params["preserve_schedule_across_inner_resets"] = True' in (
        module.PERSISTENT_HIDDEN_VELBOOST_HELPER_BLOCK
    )
    assert 'params["left_knee_probability"] = 0.65' in module.PERSISTENT_HIDDEN_VELBOOST_HELPER_BLOCK
    assert 'params["dynamic_dead_probability"] = 0.85' in module.PERSISTENT_HIDDEN_VELBOOST_HELPER_BLOCK
    assert 'params["dynamic_single_onset_range_s"] = (0.0, 0.0)' in (
        module.PERSISTENT_IMMEDIATE_VELBOOST_HELPER_BLOCK
    )
    assert 'params["dynamic_single_duration_range_s"] = (2.0, 2.0)' in (
        module.PERSISTENT_IMMEDIATE_VELBOOST_HELPER_BLOCK
    )
    assert 'params["left_knee_probability"] = 1.0' in (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_BLOCK
    )
    assert 'params["dynamic_dead_probability"] = 1.0' in (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_BLOCK
    )
    assert 'params["dead_scale_range"] = (0.0, 0.0)' in (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_BLOCK
    )
    assert "lin_vel_reward.weight = 6.0" in module.PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_HELPER_BLOCK
    assert 'lin_vel_reward.params["std"] = 0.5' in (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_HELPER_BLOCK
    )
    assert "lin_vel_reward.weight = 4.0" in module.PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_BLOCK
    assert 'lin_vel_reward.params["std"] = 0.8' in module.PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_BLOCK
    assert 'cfg.rewards["body_orientation_l2"].weight = -2.0' in (
        module.PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_BLOCK
    )
    assert 'cfg.rewards["is_terminated"].weight = -300.0' in (
        module.PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_BLOCK
    )
    assert "base_height_below_l2" in module.BASE_HEIGHT_REWARD_BLOCK
    assert "torch.clamp(float(min_height) - root_z, min=0.0)" in module.BASE_HEIGHT_REWARD_BLOCK
    assert 'cfg.rewards["base_height_below_l2"] = RewardTermCfg(' in (
        module.PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_BLOCK
    )
    assert 'weight=-8.0' in module.PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_BLOCK
    assert '"min_height": 0.70' in module.PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_BLOCK
    assert "lin_vel_reward.weight = 5.0" in module.PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_HELPER_BLOCK
    assert 'lin_vel_reward.params["std"] = 0.7' in module.PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_HELPER_BLOCK
    assert 'cfg.rewards["base_height_below_l2"].weight = -24.0' in (
        module.PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_HELPER_BLOCK
    )
    assert 'cfg.rewards["base_height_below_l2"].params["min_height"] = 0.72' in (
        module.PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_HELPER_BLOCK
    )
    assert "root_height_below" in module.ROOT_HEIGHT_TERMINATION_BLOCK
    assert "return root_z < float(min_height)" in module.ROOT_HEIGHT_TERMINATION_BLOCK
    assert 'cfg.terminations["root_too_low"] = TerminationTermCfg(' in (
        module.PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_HELPER_BLOCK
    )
    assert "lin_vel_reward.weight = 4.5" in module.PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_HELPER_BLOCK
    assert '"min_height": 0.58' in module.PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_HELPER_BLOCK
    assert "projected_gravity_xy_above" in module.PROJECTED_GRAVITY_TERMINATION_BLOCK
    assert "return gravity_xy > float(max_xy)" in module.PROJECTED_GRAVITY_TERMINATION_BLOCK
    assert 'cfg.terminations["gravity_xy_too_high"] = TerminationTermCfg(' in (
        module.PERSISTENT_HIDDEN_POSE_TERMINATE_HELPER_BLOCK
    )
    assert '"max_xy": 0.78' in module.PERSISTENT_HIDDEN_POSE_TERMINATE_HELPER_BLOCK
    assert "lin_vel_reward.weight = 5.0" in module.PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_BLOCK
    assert 'lin_vel_reward.params["std"] = 0.65' in module.PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_BLOCK
    assert 'cfg.terminations["gravity_xy_too_high"].params["max_xy"] = 0.74' in (
        module.PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_BLOCK
    )
    assert "lin_vel_reward.weight = 6.5" in module.PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_BLOCK
    assert 'lin_vel_reward.params["std"] = 0.55' in (
        module.PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_BLOCK
    )
    assert 'cfg.rewards["body_orientation_l2"].weight = -4.0' in (
        module.PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_BLOCK
    )
    assert 'cfg.rewards["is_terminated"].weight = -400.0' in (
        module.PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_BLOCK
    )
    assert 'cfg.terminations["gravity_xy_too_high"].params["max_xy"] = 0.72' in (
        module.PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_BLOCK
    )
    assert "forward_velocity_below_l1" in module.FORWARD_VELOCITY_BELOW_L1_BLOCK
    assert "asset.data.root_link_lin_vel_b[:, 0]" in module.FORWARD_VELOCITY_BELOW_L1_BLOCK
    assert "torch.clamp(float(target_x) - forward_vel, min=0.0)" in (
        module.FORWARD_VELOCITY_BELOW_L1_BLOCK
    )
    assert "forward_velocity_below_command_l1" in module.FORWARD_VELOCITY_BELOW_COMMAND_L1_BLOCK
    assert 'env.command_manager.get_command(command_name)' in (
        module.FORWARD_VELOCITY_BELOW_COMMAND_L1_BLOCK
    )
    assert "torch.clamp(command[:, 0] - forward_vel, min=0.0)" in (
        module.FORWARD_VELOCITY_BELOW_COMMAND_L1_BLOCK
    )
    assert "forward_velocity_below_command_early_l1" in (
        module.FORWARD_VELOCITY_BELOW_COMMAND_EARLY_L1_BLOCK
    )
    assert "env.episode_length_buf < int(max_step)" in (
        module.FORWARD_VELOCITY_BELOW_COMMAND_EARLY_L1_BLOCK
    )
    assert "lin_vel_reward.weight = 5.5" in module.PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_BLOCK
    assert 'lin_vel_reward.params["std"] = 0.60' in (
        module.PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_BLOCK
    )
    assert 'cfg.rewards["forward_velocity_below_l1"] = RewardTermCfg(' in (
        module.PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_BLOCK
    )
    assert '"target_x": 1.45' in module.PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_BLOCK
    assert 'weight=-3.0' in module.PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_BLOCK
    assert "lin_vel_reward.weight = 5.5" in module.PERSISTENT_HIDDEN_FORWARD_TARGET_HELPER_BLOCK
    assert 'lin_vel_reward.params["std"] = 0.60' in (
        module.PERSISTENT_HIDDEN_FORWARD_TARGET_HELPER_BLOCK
    )
    assert '"target_x": 1.55' in module.PERSISTENT_HIDDEN_FORWARD_TARGET_HELPER_BLOCK
    assert 'weight=-5.0' in module.PERSISTENT_HIDDEN_FORWARD_TARGET_HELPER_BLOCK
    assert "twist_cmd.ranges.lin_vel_x = (1.4, 1.6)" in (
        module.PERSISTENT_HIDDEN_SPEED_CURRICULUM_HELPER_BLOCK
    )
    assert 'cfg.rewards["forward_velocity_below_command_l1"] = RewardTermCfg(' in (
        module.PERSISTENT_HIDDEN_SPEED_CURRICULUM_HELPER_BLOCK
    )
    assert 'cfg = _task044_persistent_immediate_left_knee_dead_velboost_fixed1p6_env_cfg(play=play)' in (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_HELPER_BLOCK
    )
    assert 'cfg.terminations["root_too_low"] = TerminationTermCfg(' in (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_HELPER_BLOCK
    )
    assert 'cfg.terminations["gravity_xy_too_high"] = TerminationTermCfg(' in (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_HELPER_BLOCK
    )
    assert 'cfg = _task044_persistent_immediate_left_knee_pose_forward_fixed1p6_env_cfg(play=play)' in (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_BLOCK
    )
    assert 'cfg.rewards["body_orientation_l2"].weight = -5.0' in (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_BLOCK
    )
    assert 'cfg.rewards["is_terminated"].weight = -700.0' in (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_BLOCK
    )
    assert 'cfg.terminations["gravity_xy_too_high"].params["max_xy"] = 0.70' in (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_BLOCK
    )
    assert "lin_vel_reward.weight = 5.5" not in (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_BLOCK
    )
    assert 'cfg = _task045_persistent_immediate_left_knee_pose_forward_survival_fixed1p6_env_cfg(play=play)' in (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_HELPER_BLOCK
    )
    assert "cfg.episode_length_s = 8.0" in (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_HELPER_BLOCK
    )
    assert 'params["dynamic_single_duration_range_s"] = (8.0, 8.0)' in (
        module.PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_HELPER_BLOCK
    )
    assert 'cfg = _task044_persistent_hidden_pose_tight_fixed1p6_env_cfg(play=play)' in (
        module.POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_BLOCK
    )
    assert "cfg.episode_length_s = 8.0" in (
        module.POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_BLOCK
    )
    assert 'cfg.rewards["body_orientation_l2"].weight = -5.0' in (
        module.POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_BLOCK
    )
    assert 'cfg.rewards["is_terminated"].weight = -700.0' in (
        module.POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_BLOCK
    )
    assert 'cfg.terminations["gravity_xy_too_high"].params["max_xy"] = 0.70' in (
        module.POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_BLOCK
    )
    assert 'params["dynamic_single_onset_range_s"] = (2.0, 2.0)' in (
        module.POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_BLOCK
    )
    assert 'params["dynamic_single_duration_range_s"] = (8.0, 8.0)' in (
        module.POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_BLOCK
    )
    assert 'params["left_knee_probability"] = 1.0' in (
        module.POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_BLOCK
    )
    assert 'params["dead_scale_range"] = (0.0, 0.0)' in (
        module.POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_BLOCK
    )
    assert 'cfg.rewards["forward_velocity_below_command_early_l1"] = RewardTermCfg(' in (
        module.PERSISTENT_HIDDEN_STARTUP_BOOST_HELPER_BLOCK
    )
    assert '"max_step": 25' in module.PERSISTENT_HIDDEN_STARTUP_BOOST_HELPER_BLOCK
    assert 'weight=-6.0' in module.PERSISTENT_HIDDEN_STARTUP_BOOST_HELPER_BLOCK
    assert "runner_cls=Task044TrueTxlMemoryK160ClearHistoryRunner" in module.REGISTER_BLOCK


def test_task044_register_patch_adds_import_helper_and_single_registration() -> None:
    module = _load_task044_script("task044_register_hidden_fault_stage.py")
    init_path = MemoryPath(
        "\n".join(
            [
                "from tasks.velocity.config.g1_gripper.env_cfgs import (",
                "  unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg,",
                ")",
                "from mjlab.rl import MjlabOnPolicyRunner",
                "",
                "register_mjlab_task(",
                f'  task_id="{module.TASK_ID}",',
                "  env_cfg=old_cfg(),",
                "  play_env_cfg=old_cfg(play=True),",
                "  rl_cfg=unitree_g1_gripper_ppo_runner_cfg(),",
                "  runner_cls=Task038TrueTxlMemoryK160Runner,",
                ")",
            ]
        )
        + "\n"
    )

    module.patch_init(init_path)
    patched = init_path.read_text(encoding="utf-8")

    assert f"  {module.TRAIN_ENV_CFG_NAME}," in patched
    assert patched.count(module.TASK_ID) == 1
    assert patched.count(module.EVAL_ALIGNED_TASK_ID) == 1
    assert patched.count(module.EVAL_ALIGNED_VELBOOST_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_HIDDEN_VELBOOST_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_IMMEDIATE_VELBOOST_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_IMMEDIATE_LEFT_KNEE_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_HIDDEN_SPEED_STABILITY_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_HIDDEN_HEIGHT_GUARD_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_HIDDEN_POSE_TERMINATE_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_HIDDEN_POSE_TIGHT_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_HIDDEN_FORWARD_FLOOR_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_HIDDEN_FORWARD_TARGET_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_HIDDEN_SPEED_CURRICULUM_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_TASK_ID) == 1
    assert patched.count(module.POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_TASK_ID) == 1
    assert patched.count(module.PERSISTENT_HIDDEN_STARTUP_BOOST_TASK_ID) == 1
    assert "from mjlab.managers.reward_manager import RewardTermCfg" in patched
    assert "from mjlab.managers.scene_entity_config import SceneEntityCfg" in patched
    assert "from mjlab.managers.termination_manager import TerminationTermCfg" in patched
    assert "import src.tasks.velocity.mdp as mdp" in patched
    assert f"def {module.FIXED_SPEED_HELPER_NAME}" in patched
    assert f"def {module.EVAL_ALIGNED_HELPER_NAME}" in patched
    assert f"def {module.EVAL_ALIGNED_VELBOOST_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_HIDDEN_VELBOOST_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_IMMEDIATE_VELBOOST_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_IMMEDIATE_LEFT_KNEE_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_IMMEDIATE_LEFT_KNEE_SPEED_PUSH_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_HIDDEN_SPEED_STABILITY_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_HIDDEN_HEIGHT_GUARD_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_HIDDEN_HEIGHT_GUARD_STRONG_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_HIDDEN_LOW_ROOT_TERMINATE_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_HIDDEN_POSE_TERMINATE_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_HIDDEN_POSE_TIGHT_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_HIDDEN_SPEED_POSE_BALANCE_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_HIDDEN_FORWARD_FLOOR_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_HIDDEN_FORWARD_TARGET_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_HIDDEN_SPEED_CURRICULUM_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_IMMEDIATE_LEFT_KNEE_POSE_FORWARD_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_IMMEDIATE_LEFT_KNEE_SURVIVAL_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_IMMEDIATE_LEFT_KNEE_LONG_SURVIVAL_HELPER_NAME}" in patched
    assert f"def {module.POSE_TIGHT_GATE_LEFT_KNEE_LONG_TAIL_HELPER_NAME}" in patched
    assert f"def {module.PERSISTENT_HIDDEN_STARTUP_BOOST_HELPER_NAME}" in patched
    assert "twist_cmd.ranges.lin_vel_x = (1.6, 1.6)" in patched
    assert "env_cfg=_task044_hidden_fault_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_eval_left_knee_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_eval_left_knee_velboost_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_persistent_hidden_velboost_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_persistent_immediate_velboost_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_persistent_immediate_left_knee_dead_velboost_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_persistent_immediate_left_knee_dead_speedpush_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_persistent_hidden_speed_stability_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_persistent_hidden_height_guard_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_persistent_hidden_height_guard_strong_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_persistent_hidden_low_root_terminate_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_persistent_hidden_pose_terminate_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_persistent_hidden_pose_tight_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_persistent_hidden_speed_pose_balance_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_persistent_hidden_forward_floor_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_persistent_hidden_forward_target_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_persistent_hidden_speed_curriculum_1p4_to_1p6_env_cfg()" in patched
    assert "env_cfg=_task044_persistent_immediate_left_knee_pose_forward_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task045_persistent_immediate_left_knee_pose_forward_survival_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task045_persistent_immediate_left_knee_long_survival_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task045_pose_tight_gate_left_knee_long_tail_fixed1p6_env_cfg()" in patched
    assert "env_cfg=_task044_persistent_hidden_startup_boost_fixed1p6_env_cfg()" in patched
    assert 'params["preserve_schedule_across_inner_resets"] = True' in patched
    assert 'params["dynamic_single_onset_range_s"] = (0.0, 0.0)' in patched
    assert 'params["dynamic_single_onset_range_s"] = (2.0, 2.0)' in patched
    assert 'params["dynamic_single_duration_range_s"] = (8.0, 8.0)' in patched
    assert 'params["dead_scale_range"] = (0.0, 0.0)' in patched
    assert "lin_vel_reward.weight = 6.0" in patched
    assert "lin_vel_reward.weight = 4.0" in patched
    assert 'cfg.rewards["is_terminated"].weight = -300.0' in patched
    assert 'cfg.rewards["base_height_below_l2"] = RewardTermCfg(' in patched
    assert 'cfg.rewards["base_height_below_l2"].weight = -24.0' in patched
    assert 'cfg.terminations["root_too_low"] = TerminationTermCfg(' in patched
    assert 'cfg.terminations["gravity_xy_too_high"] = TerminationTermCfg(' in patched
    assert 'cfg.terminations["gravity_xy_too_high"].params["max_xy"] = 0.74' in patched
    assert 'cfg.terminations["gravity_xy_too_high"].params["max_xy"] = 0.72' in patched
    assert 'cfg.rewards["forward_velocity_below_l1"] = RewardTermCfg(' in patched
    assert 'cfg.rewards["forward_velocity_below_command_l1"] = RewardTermCfg(' in patched
    assert 'cfg.rewards["forward_velocity_below_command_early_l1"] = RewardTermCfg(' in patched
    assert "env_cfg=old_cfg()" not in patched


def test_task044_register_patch_adds_base_height_reward_function() -> None:
    module = _load_task044_script("task044_register_hidden_fault_stage.py")
    rewards_path = MemoryPath(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "from typing import TYPE_CHECKING",
                "",
                "import torch",
                "",
                "from mjlab.entity import Entity",
                "from mjlab.managers.scene_entity_config import SceneEntityCfg",
                "",
                "_DEFAULT_ASSET_CFG = SceneEntityCfg(\"robot\")",
            ]
        )
        + "\n"
    )

    module.patch_rewards(rewards_path)
    patched = rewards_path.read_text(encoding="utf-8")

    assert patched.count("def base_height_below_l2(") == 1
    assert "root_z = asset.data.root_link_pos_w[:, 2]" in patched
    assert "torch.clamp(float(min_height) - root_z, min=0.0)" in patched
    assert patched.count("def forward_velocity_below_l1(") == 1
    assert "forward_vel = asset.data.root_link_lin_vel_b[:, 0]" in patched
    assert "torch.clamp(float(target_x) - forward_vel, min=0.0)" in patched
    assert patched.count("def forward_velocity_below_command_l1(") == 1
    assert "torch.clamp(command[:, 0] - forward_vel, min=0.0)" in patched
    assert patched.count("def forward_velocity_below_command_early_l1(") == 1
    assert "return below * early" in patched

    module.patch_rewards(rewards_path)
    assert rewards_path.read_text(encoding="utf-8") == patched


def test_task044_register_patch_adds_root_height_termination_function() -> None:
    module = _load_task044_script("task044_register_hidden_fault_stage.py")
    terminations_path = MemoryPath(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "from typing import TYPE_CHECKING",
                "",
                "import torch",
                "",
                "from mjlab.sensor import ContactSensor",
                "",
                "if TYPE_CHECKING:",
                "  from mjlab.envs import ManagerBasedRlEnv",
                "",
                "def illegal_contact(env, sensor_name: str, force_threshold: float = 10.0):",
                "  return torch.zeros(1, dtype=torch.bool)",
            ]
        )
        + "\n"
    )

    module.patch_terminations(terminations_path)
    patched = terminations_path.read_text(encoding="utf-8")

    assert patched.count("def root_height_below(") == 1
    assert "from mjlab.managers.scene_entity_config import SceneEntityCfg" in patched
    assert '_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")' in patched
    assert "root_z = asset.data.root_link_pos_w[:, 2]" in patched
    assert "return root_z < float(min_height)" in patched
    assert "def projected_gravity_xy_above(" in patched
    assert "gravity_xy = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=-1)" in patched
    assert "return gravity_xy > float(max_xy)" in patched

    module.patch_terminations(terminations_path)
    assert terminations_path.read_text(encoding="utf-8") == patched


def test_task044_dynamic_scheduler_patch_preserves_hidden_schedule_across_inner_resets() -> None:
    module = _load_task044_script("task044_patch_dynamic_training_scheduler.py")
    env_cfg_path = MemoryPath(
        "\n".join(
            [
                "def _apply_task030_dynamic_training_failure(",
                "  env,",
                "  dynamic_dead_probability: float = 0.70,",
                "  transient_window_s: float = 0.3,",
                ") -> None:",
                "  reset_mask = (env.episode_length_buf <= 1) | (",
                "    env._task030_dynamic_training_case_id < 0",
                "  )",
                "  _task030_resample_dynamic_training_schedules(",
                "      env,",
                "      reset_env_ids,",
                "      targets,",
                "      clean_probability,",
                "      persistent_probability,",
                "      dynamic_single_probability,",
                "      left_knee_probability,",
                "      weak_scale_range,",
                "      dead_scale_range,",
                "      dynamic_dead_probability,",
                "    )",
                "  _task030_resample_dynamic_training_schedules(",
                "      env,",
                "      inactive_env_ids,",
                "      targets,",
                "      clean_probability,",
                "      persistent_probability,",
                "      dynamic_single_probability,",
                "      left_knee_probability,",
                "      weak_scale_range,",
                "      dead_scale_range,",
                "      dynamic_dead_probability,",
                "    )",
                "  return None",
                "",
                "def _task030_resample_dynamic_training_schedules(",
                "  env,",
                "  env_ids,",
                "  targets,",
                "  clean_probability: float,",
                "  persistent_probability: float,",
                "  dynamic_single_probability: float,",
                "  left_knee_probability: float,",
                "  weak_scale_range: tuple[float, float],",
                "  dead_scale_range: tuple[float, float],",
                "  dynamic_dead_probability: float,",
                ") -> None:",
                "  count = len(env_ids)",
                "  if count:",
                "    onset = 1.0 + 3.0 * torch.rand(count, device=env.device)",
                "    duration = 0.6 + 1.4 * torch.rand(count, device=env.device)",
                "",
                "def _add_task030_dynamic_training_failure_stage():",
                "  return {",
                '      "dynamic_dead_probability": 0.70,',
                '      "transient_window_s": 0.3,',
                "  }",
            ]
        )
        + "\n"
    )

    module.patch_env_cfgs(env_cfg_path)
    patched = env_cfg_path.read_text(encoding="utf-8")

    assert "preserve_schedule_across_inner_resets: bool = False" in patched
    assert "if preserve_schedule_across_inner_resets:" in patched
    assert "reset_mask = env._task030_dynamic_training_case_id < 0" in patched
    assert '"preserve_schedule_across_inner_resets": False' in patched
    assert "dynamic_single_onset_range_s: tuple[float, float] = (1.0, 4.0)" in patched
    assert "dynamic_single_duration_range_s: tuple[float, float] = (0.6, 2.0)" in patched
    assert '"dynamic_single_onset_range_s": (1.0, 4.0)' in patched
    assert '"dynamic_single_duration_range_s": (0.6, 2.0)' in patched

    module.patch_env_cfgs(env_cfg_path)
    assert env_cfg_path.read_text(encoding="utf-8") == patched


def test_task044_fault_label_wrapper_adds_non_actor_label_group() -> None:
    torch = pytest.importorskip("torch")
    module = _load_training_module("rsl_history_wrapper.py")
    env = FakeFaultEnv(torch)

    wrapper = module.Task044FaultLabelVecEnvWrapper(env)
    obs = wrapper.get_observations()

    assert "task044_fault_label" in obs
    assert tuple(obs["actor"].shape) == (3, 4)
    assert obs["task044_fault_label"].reshape(-1).tolist() == [0.0, 1.0, 6.0]
    assert obs["task044_trial_step"].reshape(-1).tolist() == [0.0, 3.0, 7.0]
    assert obs["task044_trial_index"].reshape(-1).tolist() == [0.0, 1.0, 2.0]


def test_task046_post_reset_recovery_reward_shapes_only_final_trial_windows() -> None:
    torch = pytest.importorskip("torch")
    module = _load_training_module("rsl_history_wrapper.py")
    env = FakeRecoveryEnv(torch)
    wrapper = module.Task046PostResetRecoveryRewardVecEnvWrapper(
        env,
        final_trial_index=2,
        recovery_window_steps=3,
        tail_window_steps=2,
        early_velocity_weight=1.0,
        tail_velocity_weight=0.5,
        orientation_weight=0.5,
        root_height_weight=2.0,
        min_root_z=0.70,
    )

    _obs, rewards, _dones, extras = wrapper.step(torch.zeros((3, 2)))

    assert rewards.tolist() == pytest.approx([-1.4, 0.0, -0.2])
    assert extras["task046_post_reset_recovery_reward"]["active_count"] == 2
    assert extras["task046_post_reset_recovery_reward"]["recovery_count"] == 1
    assert extras["task046_post_reset_recovery_reward"]["tail_count"] == 1
    debug = wrapper.task046_recovery_debug_snapshot()
    assert debug["sample_count"] == 2
    assert debug["recovery_sample_count"] == 1
    assert debug["tail_sample_count"] == 1


def test_task046_retry_context_adds_actor_visible_reset_features_without_fault_label() -> None:
    torch = FakeTorch()
    module = _load_training_module("rsl_history_wrapper.py")
    module._require_torch = lambda: torch
    from h200_locomotion_lab.training.history_buffer import HistoryFrameSpec

    HistoryFrameSpec(module.TASK046_RETRY_CONTEXT_FEATURE_NAMES).validate_no_actor_fault_leakage()
    env = FakeRetryContextEnv(torch)
    wrapper = module.Task046RetryContextVecEnvWrapper(
        env,
        num_trials=3,
        final_trial_index=2,
        step_window_steps=50,
    )

    obs, _extras = wrapper.reset()

    assert tuple(obs["actor"].shape) == (3, 8)
    _assert_nested_approx(
        obs["task046_retry_context"].tolist(),
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    )

    obs, _rewards, _dones, _extras = wrapper.step(torch.zeros((3, 2)))

    assert tuple(obs["actor"].shape) == (3, 8)
    _assert_nested_approx(
        obs["task046_retry_context"].tolist(),
        [
            [0.5, 0.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.2, 0.0, 0.0, 0.0],
        ],
    )

    obs, _rewards, _dones, _extras = wrapper.step(torch.zeros((3, 2)))

    _assert_nested_approx(
        obs["task046_retry_context"].tolist(),
        [
            [0.5, 0.0, 0.02, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.02, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.22, 0.0, 0.0, 0.0],
        ],
    )
    assert wrapper.task046_retry_context_debug_snapshot()["feature_dim"] == 6


def test_task046_retry_context_does_not_accumulate_when_env_reuses_obs_dict() -> None:
    torch = FakeTorch()
    module = _load_training_module("rsl_history_wrapper.py")
    module._require_torch = lambda: torch
    env = FakeRetryContextEnv(torch, reuse_observations=True)
    wrapper = module.Task046RetryContextVecEnvWrapper(env)

    first, _extras = wrapper.reset()
    second = wrapper.get_observations()
    third = wrapper.get_observations()

    assert tuple(first["actor"].shape) == (3, 8)
    assert tuple(second["actor"].shape) == (3, 8)
    assert tuple(third["actor"].shape) == (3, 8)
    assert tuple(env._cached_obs["actor"].shape) == (3, 2)
    assert wrapper.task046_retry_context_debug_snapshot()["base_actor_dim"] == 2


def test_task046_retry_context_preserves_observation_container_with_to_method() -> None:
    torch = FakeTorch()
    module = _load_training_module("rsl_history_wrapper.py")
    module._require_torch = lambda: torch
    env = FakeRetryContextEnv(torch, observation_container_cls=FakeObservationContainer)
    wrapper = module.Task046RetryContextVecEnvWrapper(env)

    obs, _extras = wrapper.reset()

    assert isinstance(obs, FakeObservationContainer)
    assert obs.to("cpu") is obs
    assert tuple(obs["actor"].shape) == (3, 8)


def _load_src_tool(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "tools" / name
    return _load_path(path)


def _assert_nested_approx(actual, expected) -> None:
    actual_flat = [value for row in actual for value in row]
    expected_flat = [value for row in expected for value in row]
    assert actual_flat == pytest.approx(expected_flat)


def _load_training_module(name: str):
    path = ROOT / "src" / "h200_locomotion_lab" / "training" / name
    return _load_path(path)


def _load_task044_script(name: str):
    path = ROOT / ".agent" / "task" / "task044-memory-required-fault-identification-target" / name
    return _load_path(path)


def _load_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
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


class FakeFaultEnv:
    def __init__(self, torch_module) -> None:
        self.torch = torch_module
        self.num_envs = 3
        self.device = "cpu"
        self.max_episode_length = 100
        self.num_actions = 2
        self.episode_length_buf = torch_module.zeros(3, dtype=torch_module.long)
        self._task030_dynamic_failure_target_index = torch_module.tensor([-1, 0, 5])
        self.trial_step = torch_module.tensor([0, 3, 7])
        self.trial_index = torch_module.tensor([0, 1, 2])

    @property
    def cfg(self):
        return object()

    @property
    def unwrapped(self):
        return self

    def seed(self, seed: int = -1) -> int:
        return seed

    def reset(self):
        return self.get_observations(), {}

    def get_observations(self):
        return {
            "actor": self.torch.zeros((3, 4)),
            "critic": self.torch.zeros((3, 5)),
        }

    def step(self, actions):
        rewards = self.torch.zeros(3)
        dones = self.torch.zeros(3, dtype=self.torch.bool)
        return self.get_observations(), rewards, dones, {}

    def close(self) -> None:
        return None


class FakeTorch:
    long = "long"
    bool = "bool"
    float32 = "float32"

    def __init__(self) -> None:
        import numpy as np

        self.np = np

    def zeros(self, shape, device=None, dtype=None):
        return FakeTensor(self.np.zeros(shape, dtype=self._dtype(dtype)))

    def tensor(self, values, device=None, dtype=None):
        return FakeTensor(self.np.asarray(values, dtype=self._dtype(dtype)))

    def clamp(self, tensor, min=None, max=None):
        array = tensor.array if isinstance(tensor, FakeTensor) else self.np.asarray(tensor)
        return FakeTensor(self.np.clip(array, min, max))

    def stack(self, tensors, dim=0):
        return FakeTensor(self.np.stack([self._array(tensor) for tensor in tensors], axis=dim))

    def cat(self, tensors, dim=0):
        return FakeTensor(self.np.concatenate([self._array(tensor) for tensor in tensors], axis=dim))

    def _array(self, value):
        return value.array if isinstance(value, FakeTensor) else self.np.asarray(value)

    def _dtype(self, dtype):
        if dtype == self.long:
            return self.np.int64
        if dtype == self.bool:
            return self.np.bool_
        if dtype == self.float32:
            return self.np.float32
        return None


class FakeTensor:
    def __init__(self, array) -> None:
        self.array = array

    @property
    def shape(self):
        return self.array.shape

    def to(self, device=None, dtype=None):
        if dtype is None:
            return FakeTensor(self.array.copy())
        return FakeTensor(self.array.astype(FakeTorch()._dtype(dtype)))

    def reshape(self, *shape):
        return FakeTensor(self.array.reshape(*shape))

    def zero_(self):
        self.array[...] = 0
        return self

    def any(self):
        import numpy as np

        return FakeTensor(np.asarray(self.array.any()))

    def item(self):
        return self.array.item()

    def tolist(self):
        return self.array.tolist()

    def __getitem__(self, key):
        return FakeTensor(self.array[self._key(key)])

    def __setitem__(self, key, value) -> None:
        self.array[self._key(key)] = value.array if isinstance(value, FakeTensor) else value

    def __truediv__(self, other):
        return FakeTensor(self.array / other)

    def __add__(self, other):
        other_array = other.array if isinstance(other, FakeTensor) else other
        return FakeTensor(self.array + other_array)

    def __eq__(self, other):
        other_array = other.array if isinstance(other, FakeTensor) else other
        return FakeTensor(self.array == other_array)

    def __ge__(self, other):
        other_array = other.array if isinstance(other, FakeTensor) else other
        return FakeTensor(self.array >= other_array)

    def __or__(self, other):
        other_array = other.array if isinstance(other, FakeTensor) else other
        return FakeTensor(self.array | other_array)

    def _key(self, key):
        if isinstance(key, FakeTensor):
            return key.array
        if isinstance(key, tuple):
            return tuple(item.array if isinstance(item, FakeTensor) else item for item in key)
        return key


class FakeObservationContainer(dict):
    def to(self, device):
        return self


class FakeRecoveryEnv:
    def __init__(self, torch_module) -> None:
        self.torch = torch_module
        self.num_envs = 3
        self.device = "cpu"
        self.max_episode_length = 100
        self.num_actions = 2
        self.episode_length_buf = torch_module.zeros(3, dtype=torch_module.long)
        data = type("RobotData", (), {})()
        data.root_link_lin_vel_b = torch_module.tensor(
            [
                [0.4, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [1.2, 0.0, 0.0],
            ]
        )
        data.projected_gravity_b = torch_module.tensor(
            [
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.1, 0.0, 0.0],
            ]
        )
        data.root_link_pos_w = torch_module.tensor(
            [
                [0.0, 0.0, 0.65],
                [0.0, 0.0, 0.65],
                [0.0, 0.0, 0.80],
            ]
        )
        self.scene = {"robot": type("Robot", (), {"data": data})()}
        self.command_manager = type(
            "CommandManager",
            (),
            {
                "get_command": lambda _self, _name: torch_module.tensor(
                    [
                        [1.6, 0.0, 0.0],
                        [1.6, 0.0, 0.0],
                        [1.6, 0.0, 0.0],
                    ]
                )
            },
        )()

    @property
    def cfg(self):
        return object()

    @property
    def unwrapped(self):
        return self

    def seed(self, seed: int = -1) -> int:
        return seed

    def reset(self):
        return self.get_observations(), {}

    def get_observations(self):
        return {
            "actor": self.torch.zeros((3, 4)),
            "critic": self.torch.zeros((3, 5)),
        }

    def step(self, actions):
        rewards = self.torch.zeros(3)
        dones = self.torch.zeros(3, dtype=self.torch.bool)
        extras = {
            "task037_trial_index": self.torch.tensor([2, 1, 2]),
            "task037_trial_step": self.torch.tensor([1, 1, 4]),
        }
        return self.get_observations(), rewards, dones, extras

    def close(self) -> None:
        return None


class FakeRetryContextEnv:
    def __init__(
        self,
        torch_module,
        *,
        reuse_observations: bool = False,
        observation_container_cls: type[dict] = dict,
    ) -> None:
        self.torch = torch_module
        self.num_envs = 3
        self.device = "cpu"
        self.max_episode_length = 100
        self.num_actions = 2
        self.episode_length_buf = torch_module.zeros(3, dtype=torch_module.long)
        self.trial_index = torch_module.zeros(3, dtype=torch_module.long)
        self.trial_step = torch_module.zeros(3, dtype=torch_module.long)
        self._step_index = 0
        self.reuse_observations = reuse_observations
        self.observation_container_cls = observation_container_cls
        self._cached_obs = self.observation_container_cls({
            "actor": self.torch.zeros((3, 2)),
            "critic": self.torch.zeros((3, 5)),
        })

    @property
    def cfg(self):
        return object()

    @property
    def unwrapped(self):
        return self

    def seed(self, seed: int = -1) -> int:
        return seed

    def reset(self):
        self._step_index = 0
        self.trial_index.zero_()
        self.trial_step.zero_()
        return self.get_observations(), {}

    def get_observations(self):
        if self.reuse_observations:
            return self._cached_obs
        return self.observation_container_cls({
            "actor": self.torch.zeros((3, 2)),
            "critic": self.torch.zeros((3, 5)),
        })

    def step(self, actions):
        rewards = self.torch.zeros(3)
        dones = self.torch.zeros(3, dtype=self.torch.bool)
        if self._step_index == 0:
            self.trial_index = self.torch.tensor([1, 2, 2])
            self.trial_step = self.torch.tensor([0, 0, 10])
            extras = {
                "task037_inner_reset": self.torch.tensor([True, True, False]),
                "task037_outer_reset": self.torch.tensor([False, False, False]),
                "task037_reset_reason": self.torch.tensor([1, 2, 0]),
                "task037_trial_index": self.trial_index,
                "task037_trial_step": self.trial_step,
            }
        else:
            self.trial_step = self.trial_step + 1
            extras = {
                "task037_inner_reset": self.torch.tensor([False, False, False]),
                "task037_outer_reset": self.torch.tensor([False, False, False]),
                "task037_reset_reason": self.torch.tensor([0, 0, 0]),
                "task037_trial_index": self.trial_index,
                "task037_trial_step": self.trial_step,
            }
        self._step_index += 1
        return self.get_observations(), rewards, dones, extras

    def close(self) -> None:
        return None
