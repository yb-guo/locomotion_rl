# 008 AdaptK160 Failure Curriculum Probe

## Route

Subtask 007 established a clean gait prior:

`AdaptK160 model_5467.pt` passes clean multi-trial eval at `0.4`, `1.2`, and
`2.0 m/s`.

This subtask tests whether that clean prior can enter motor-failure adaptation
without regressing immediately.

Fixed boundaries:

- Start from 007 `model_5467.pt`.
- Keep `Task037AdaptK160DeterministicInnerResetRunner`.
- Keep the same actor-visible K160 history and no explicit fault labels.
- First run failure baseline eval, then a bounded curriculum attempt.
- Do not promote unless both clean gait and failure gates have JSON evidence.

Planned route:

1. Register AdaptK160 failure task ids:
   - `WeakPersistent`;
   - `MixedPersistent`;
   - `FocusedDeadGrid`;
   - `DynamicMotorFailure`.
2. Baseline-evaluate `model_5467.pt` on:
   - dynamic switch at `0.4`, `1.2`, and `2.0 m/s`;
   - `2.0 m/s` 12-joint forced deadgrid.
3. Run one bounded weak or mixed persistent continuation from `model_5467.pt`.
4. Re-evaluate clean gait first, then failure matrix.
5. Decide whether AdaptK160 needs staged weak -> mixed -> deadgrid, or whether
   clean gait prior is not enough.

Acceptance:

- Failure task ids register on H200.
- Baseline failure eval writes JSON; missing event errors are not accepted as
  policy evidence.
- Training log/checkpoint path is recorded for the first failure continuation.
- Final decision separates:
  - clean gait regression;
  - dynamic switch result;
  - forced deadgrid result.

Evidence root:

`/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/adaptk160_failure_curriculum/`

## Log

- 2026-05-29 Opened after 007 clean gait prior passed and user asked to try the
  failure curriculum next.
- 2026-05-29 First probe attempt using the clean-only task id correctly failed
  as a setup issue: the clean task does not contain `dynamic_motor_failure` or
  `motor_failure` events, so dynamic/deadgrid evals produced missing-event JSON
  rather than policy metrics.
- 2026-05-29 Added AdaptK160 failure task registrations and a stage-selectable
  H200 launch script.
- 2026-05-29 Local validation after adding failure task ids:
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_task037_multitrial_contract.py tests/test_task037_mjlab_smoke_scripts.py tests/test_agent_inventory.py`
    -> `9 passed, 5 skipped`.
- 2026-05-29 H200 validation after syncing:
  - registry patch completed;
  - `tests/test_task037_multitrial_contract.py tests/test_task037_mjlab_smoke_scripts.py tests/test_task033_history_buffer.py`
    -> `19 passed`;
  - `task037_launch_adaptk160_failure_curriculum.sh --help` and
    `DRY_RUN=1 STAGE=weak` completed.
- 2026-05-29 Baseline failure eval for clean-prior `model_5467.pt` completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/adaptk160_failure_probe_model5467_registered/task037_adaptk160_model5467_failure_probe_summary.json`.
  Result: `pass=false`.
  - Dynamic switch: `3/3` speeds passed (`0.4`, `1.2`, `2.0 m/s`).
  - `2.0 m/s` forced deadgrid: `10/12` passed.
  - Failed cases:
    - `left_knee_joint`: fall ratio `0.0`, linear velocity error `1.291076`
      over the `1.20` gate.
    - `right_knee_joint`: fall ratio `0.046875`, gravity xy max `0.945125`,
      root z min `0.172143`.
- 2026-05-29 Ran one bounded mixed persistent continuation from `model_5467.pt`:
  - log:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/adaptk160_failure_curriculum/037_adapt_k160_mixed_from_clean5467_env8192_iter30_gpu0_seed3700810.stdout.log`
  - checkpoint:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task037_adaptk160_failure_curriculum/2026-05-29_17-05-46_037_adapt_k160_mixed_from_clean5467_env8192_iter30_gpu0_seed3700810/model_5496.pt`
  - throughput around `96k` steps/s; occasional training `fell_over` appeared,
    but the final eval preserved clean gait.
- 2026-05-29 Full eval for mixed final `model_5496.pt` completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/adaptk160_failure_curriculum/eval_mixed_model5496/task037_adaptk160_mixed_model5496_eval_summary.json`.
  Result: `pass=false`.
  - Clean: `3/3` speeds passed.
  - Dynamic switch: `3/3` speeds passed.
  - `2.0 m/s` forced deadgrid: `10/12` passed.
  - Failed cases remain:
    - `left_knee_joint`: fall ratio `0.0`, linear velocity error `1.329694`.
    - `right_knee_joint`: fall ratio `0.0625`, gravity xy max `0.937403`,
      root z min `0.184711`.
- 2026-05-29 Mixed-run knee checkpoint sweep completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/adaptk160_failure_curriculum/knee_sweep_mixed_seed3700810/task037_adaptk160_mixed_knee_sweep_summary.json`.
  Result: no checkpoint from `5470`, `5475`, `5480`, `5485`, `5490`, `5495`,
  or `5496` passed both knee blockers. The best observed checkpoint for both
  individual knee cases was `model_5475`, but it still failed:
  - left knee linear velocity error `1.300210`;
  - right knee root z min `0.216811`.

## Review

Status: complete for this probe. AdaptK160 clean-prior `model_5467.pt` already
passes dynamic switch but not the 2.0 m/s forced knee deadgrid. A generic mixed
persistent continuation preserves clean/dynamic gates but does not fix the
left/right knee blockers. No failure-adaptation checkpoint is promoted.
