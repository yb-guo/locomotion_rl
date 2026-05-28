# 003 Training Run

## Route

Run one bounded H200 curriculum training pass.

Constraints:

- env8192 target unless H200 occupancy says otherwise;
- frozen-base StackMLP K4;
- frequent checkpoints for sweep;
- no pass/fail claim from reward curves alone.

Curriculum order:

1. clean + unified-speed rehearsal;
2. weak persistent motor failures;
3. mixed weak/dead single-joint failures;
4. forced dead-grid rehearsal;
5. dynamic-switch rehearsal.

## Log

- 2026-05-28 Planned.
- 2026-05-28 Added launch artifact:
  `task035_launch_eval_gated_curriculum.sh`.
- 2026-05-28 Ran bounded mixed persistent curriculum continuation from
  `model_5350.pt`:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task035_eval_gated_curriculum_train/2026-05-28_15-31-23_035_mixed_from_model5350_env8192_iter20_gpu1_seed3503601`.
  Settings: env8192, GPU 1, seed `3503601`, max iter `20`,
  save interval `2`, learning rate `3e-6`, entropy coef `3e-4`.
  Training reached `model_5369.pt`; log:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task035/eval_gated_curriculum_train/035_mixed_from_model5350_env8192_iter20_gpu1_seed3503601.stdout.log`.

## Review

Status: completed. Training reward is not used as acceptance.
