# 012: Eval-Aligned Left-Knee Stage

## Route

Subtask 011 proved that memory latent can classify the hidden fault and affect
actions, but the behavior triplet still failed. The next closed unit aligns the
train target with the current Task044 triplet gate instead of continuing the
mixed hidden-fault distribution:

- add a Task044-specific train task id for the exact eval case;
- use fixed `vx=1.6`;
- use a 2.0 s episode/trial length, matching eval;
- use deterministic left-knee dead motor from `0.0` to `2.0` s;
- keep the actor-visible observation contract unchanged;
- continue using the existing Task044 clear-history runner and PPO infra.

This is a diagnostic bridge, not a claim that the final LocoFormer target is a
single left-knee policy.

## Acceptance

- Local tests lock the new task id and registry helper.
- H200 registry contains both the broad Task044 hidden-fault task and the
  eval-aligned left-knee task.
- H200 smoke passes for the new task.
- H200 long train records train-pipeline evidence with
  `task044_fault_aux_min_trial_index=1`.
- H200 triplet eval decides pass/fail; no pass is claimed from train or action
  influence alone.

## Log

- 2026-05-31 Added
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-EvalLeftKnee1p6`
  to the Task044 registry patch script. The helper fixes `episode_length_s=2.0`,
  `lin_vel_x=1.6`, and a deterministic left-knee dead template from `0.0` to
  `2.0` s.
- 2026-05-31 Updated Task044 train/eval wrappers and H200 scripts so `--task`
  can target the new eval-aligned task id. Task044 eval defaults now match the
  triplet diagnostic case: left-knee, onset `0.0`, recovery `2.0`, final window
  `0.5`, and `vx=1.6`.
- 2026-05-31 Local validation passed:
  `python -m pytest -q -p no:cacheprovider tests\test_task044_hidden_fault_target.py`
  with 5 passed and 1 skipped.
- 2026-05-31 H200 registry patch applied. Registry now contains both:
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-Fast1p6` and
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-EvalLeftKnee1p6`.
- 2026-05-31 H200 smoke passed for the eval-aligned task:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4402701.json`.
  Result: `train_pipeline_pass=true`, task id matched the eval-aligned task,
  and `task044_fault_aux_updates=20`.
- 2026-05-31 Started H200 long train from the previous 100-iteration bridge aux
  checkpoint. Background PID: `597946`. Expected output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_evalaligned_leftknee_aux002_early4_trial1_scale1_env1024_iter50_seed4402801.json`.
- 2026-05-31 H200 eval-aligned left-knee continuation completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_evalaligned_leftknee_aux002_early4_trial1_scale1_env1024_iter50_seed4402801.json`.
  Result: `train_pipeline_pass=true`, task id matched the eval-aligned task,
  and `task044_fault_aux_updates=750`.
- 2026-05-31 H200 triplet eval for that checkpoint failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/evalaligned_leftknee_aux002_early4_trial1_scale1_iter50_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4402901.json`.
  The action influence summary passed, but the behavior-level contract failed
  with `normal_quality_gate_not_passed`, `zero_residual_ablation_not_degraded`,
  and `stateless_memory_ablation_not_degraded`. Normal final trial was stable
  (`fall_ratio=0.0`, `gravity_xy.max=0.0912749320268631`,
  `root_z.min=0.7717834115028381`) but too slow:
  `lin_vel_error.mean=0.9456071853637695`.
- 2026-05-31 Started a longer eval-aligned continuation from that checkpoint.
  Background PID: `599403`. Expected output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_evalaligned_leftknee_aux002_early4_trial1_scale1_env1024_iter100_cont_seed4403001.json`.

## Review

Status: open.

This subtask directly addresses the observed train/eval mismatch. The first
eval-aligned run improved stability but not speed tracking or behavior-level
memory causality. It remains open while the longer continuation runs. Even if
action influence increases, Task044 is not passed unless the triplet JSON passes.
