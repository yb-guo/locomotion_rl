# 004 Multi-Trial Eval JSON

## Route

Add eval that reports per-trial behavior before any TXL quality claims.

Acceptance criteria:

- Eval JSON has `trial_0`, `trial_1`, `final_trial`, and aggregate sections.
- Per-trial metrics include fall ratio, velocity error, yaw error, root z,
  gravity xy, reward, and reset reason counts.
- Final-trial pass/fail is explicit and is the default promotion gate.
- Aggregate metrics are auxiliary and cannot hide final-trial failure.
- Existing AdaptK4 partial checkpoint is evaluated first.

## Log

- 2026-05-29 Planned.
- 2026-05-29 Added `Task037AdaptK4DeterministicInnerResetRunner` so
  Task036 AdaptK4 checkpoints load with the Task037 deterministic multi-trial
  wrapper.
- 2026-05-29 Added
  `h200_locomotion_lab.tools.task037_multitrial_eval_checkpoint`.
  The eval snapshots trial index before `env.step()` so terminal events are
  attributed to the trial that actually ended; step-after reset extras are not
  used for terminal bucketing.
- 2026-05-29 Fixed timeout classification in
  `Task037MultiTrialVecEnvWrapper` by treating RSL/MJLab `time_outs` as trial
  timeout instead of fall/raw failure. Added a fake-env regression for local
  `trial_timeout_steps` precedence when raw done and timeout occur on the same
  step.
- 2026-05-29 Local validation:
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_task037_mjlab_smoke_scripts.py tests/test_task037_multitrial_contract.py`
    -> `4 passed, 2 skipped in 0.08s`.
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m h200_locomotion_lab.tools.inspect_agent`
    -> passed.
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_agent_inventory.py`
    -> `2 passed in 0.02s`.
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m h200_locomotion_lab.tools.task037_multitrial_eval_checkpoint --help`
    -> passed.
- 2026-05-29 H200 validation after syncing to adapter checkout:
  - `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter`,
    `tests/test_task037_mjlab_smoke_scripts.py tests/test_task037_multitrial_contract.py`
    -> `7 passed in 2.54s`.
  - Existing checkpoint evaluated:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task036_policy_quality_train/2026-05-28_23-10-53_036_adapt_k4_warmstart_env8192_iter60_gpu0_seed3603630/model_5408.pt`.
  - Eval JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/multitrial_eval/task037_adaptk4_model5408_multitrial_eval_env64.json`.
  - Local evidence copy:
    `task037_adaptk4_model5408_multitrial_eval_env64.json`.
  - Conditions: H20D `cuda:0`, 64 envs, 360 steps, 2.0s trial length,
    fixed command 2.0 m/s forward.
  - Result: `final_trial_pass=true`, `pass=true`,
    `promotion_gate=final_trial`, `quality_claim=false`.
  - Final trial metrics: `completion_ratio=1.0`, `fall_ratio=0.0`,
    `reset_reason_counts={"2": 64}`, `lin_vel_error.mean=0.635804`,
    `yaw_vel_error.mean=0.121148`, `gravity_xy.max=0.140831`,
    `root_z.min=0.748780`.

## Review

Status: passed for the 004 eval JSON contract.

The eval now reports `trial_0`, `trial_1`, `final_trial`, and `aggregate`.
Promotion is explicitly gated by `final_trial_pass`; aggregate metrics are
marked auxiliary and cannot hide a final-trial failure. This is not a TXL
memory or long-training quality claim.
