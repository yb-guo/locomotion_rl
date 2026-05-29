# 005 GRU Token Longtrain Eval

## Route

Complete the Task033 missing policy-quality work:

- long-train GRU K4;
- long-train Token K4;
- checkpoint sweep non-final checkpoints;
- run the same blocker and full gates used for AdaptK4.

Do not claim policy-quality from existing smoke or overhead runs.

## Log

- 2026-05-28 H200 scratch 60-iteration GRU train completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task036/policy_train/036_gru_k4_scratch_env8192_iter60_gpu0_seed3603610.stdout.log`.
  Final checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task036_policy_quality_train/2026-05-28_23-06-49_036_gru_k4_scratch_env8192_iter60_gpu0_seed3603610/model_59.pt`.
- 2026-05-28 H200 scratch 60-iteration token train completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task036/policy_train/036_token_k4_scratch_env8192_iter60_gpu1_seed3603620.stdout.log`.
  Final checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task036_policy_quality_train/2026-05-28_23-06-49_036_token_k4_scratch_env8192_iter60_gpu1_seed3603620/model_59.pt`.
- 2026-05-28 Full matrix eval for GRU `model_59` completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task036/full_validation_gru_gru_model59_iter60/task036_full_validation_summary.json`.
  Result: `pass=false`; all speeds failed dynamic switch and all deadgrid
  cases failed.
- 2026-05-28 Full matrix eval for token `model_59` completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task036/full_validation_token_token_model59_iter60/task036_full_validation_summary.json`.
  Result: `pass=false`; all speeds failed dynamic switch. Deadgrid passed
  `6/12` at `0.4`, `0/12` at `1.2`, and `0/12` at `2.0`.
- 2026-05-28 Started longer GRU/token continuation runs from `model_59` toward
  approximately 300 total iterations:
  `036_gru_k4_resume59_env8192_iter240_gpu0_seed3603611` and
  `036_token_k4_resume59_env8192_iter240_gpu1_seed3603621`.
- 2026-05-28 GRU continuation completed, final checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task036_policy_quality_train/2026-05-28_23-28-23_036_gru_k4_resume59_env8192_iter240_gpu0_seed3603611/model_298.pt`.
  Full matrix:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task036/full_validation_gru_gru_model298_iter300/task036_full_validation_summary.json`.
  Result: `pass=false`; `0.4` dynamic passed but deadgrid was `6/12`,
  while `1.2` and `2.0` dynamic and deadgrid failed.
- 2026-05-28 Token continuation completed, final checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task036_policy_quality_train/2026-05-28_23-28-23_036_token_k4_resume59_env8192_iter240_gpu1_seed3603621/model_298.pt`.
  Full matrix:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task036/full_validation_token_token_model298_iter300/task036_full_validation_summary.json`.
  Result: `pass=false`; `0.4` dynamic passed but deadgrid was `4/12`,
  while `1.2` and `2.0` dynamic and deadgrid failed.

## Review

Status: complete for the initial Task036 GRU/token bakeoff. Neither GRU K4 nor
token K4 is promoted; longer training converged to stable but poor velocity
tracking behavior.
