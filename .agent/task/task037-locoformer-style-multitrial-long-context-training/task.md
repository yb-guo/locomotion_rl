# Task 037: LocoFormer-Style Multi-Trial Long-Context Training

## Route

Task036 showed that swapping policy consumers alone is not enough:

- AdaptK4 is the best partial candidate, but does not pass the full deadgrid.
- GRU K4 and token K4 are rejected for the current route.
- Longer memory should not be trained until the reset/memory/eval contract
  matches LocoFormer-style in-context adaptation.

This task adds a scaled-down LocoFormer-style training contract:

```text
outer episode
  sample one latent condition:
    command, motor failure, randomization condition

  trial 0
  trial 1
  final trial

  memory is preserved across inner trial resets
  memory is cleared only at outer episode reset
```

Fixed acceptance criteria:

- `num_trials = 3` for the first implementation.
- `trial_done = fall OR trial_timeout`.
- `episode_done = trial_done AND final_trial`.
- runner-facing `done = episode_done`.
- Inner trial reset preserves actor history, recurrent memory, and TXL memory.
- Outer episode reset clears actor history, recurrent memory, and TXL memory.
- Inner trial reset keeps command, failure, and randomization condition
  unchanged.
- Inner trial reset uses deterministic standing pose.
- Inner trial reset fixes phase to the start phase.
- Inner trial reset clears env/action-manager last action.
- Fall keeps the existing fall penalty.
- Trial timeout adds no extra penalty.
- Actor does not see trial index, final-trial flag, failure id, motor scale, or
  failure mask.
- Critic also does not see trial index or final-trial flag by default.
- Trial labels and condition debug state are allowed only in `extras` and JSON
  evidence.
- Task037 target long-context horizon is `3.2s`, i.e. `160` policy steps at
  `50Hz`; full `160`-step context is for TXL-style memory, not flattened MLP.

## Planned Slices

1. `001-fake-multitrial-contract.md`
   - Build a local fake vec-env harness and deterministic contract tests.

2. `002-mjlab-multitrial-smoke.md`
   - Connect the wrapper to MJLab and run construction/PPO smoke only.

3. `003-deterministic-inner-reset-mjlab.md`
   - Implement and verify real MJLab deterministic inner reset.

4. `004-multitrial-eval-json.md`
   - Add per-trial eval JSON with final-trial pass semantics.

5. `005-txl-style-memory-consumer.md`
   - Add TXL-style long-context consumer smoke with memory reset tests.

6. `006-long-context-training-decision.md`
   - Train/evaluate and compare against Task036 AdaptK4 partial.

7. `007-warmstart-clean-gait-prior.md`
   - Stop failure-matrix training and first establish a clean K160 gait prior
     via AdaptK4 warmstart or clean-only curriculum.

8. `008-adaptk160-failure-curriculum-probe.md`
   - Start from the clean AdaptK160 checkpoint and test weak/mixed/deadgrid or
     dynamic failure curriculum without confusing clean-task setup errors with
     policy failures.

9. `009-knee-focused-repair.md`
   - Focus the remaining `left_knee_joint` and `right_knee_joint` forced-dead
     blockers from the AdaptK160 clean prior.

10. `010-arbitrary-onset-multitrial-eval.md`
    - Evaluate per-joint dynamic dead onset with recovery under the Task037
      multi-trial final-trial gate, preserving AdaptK160 history across inner
      attempts.

## Minimal Closed Loop

1. Pass fake-env contract tests for `trial_done`, `episode_done`, condition
   preservation, and history reset behavior.
2. Run MJLab env64/env8192 smoke without claiming locomotion quality.
3. Prove deterministic inner reset preserves command/failure.
4. Run multi-trial eval JSON on an existing checkpoint.
5. Only then add TXL-style policy consumer and training smoke.
6. Long-train only after reset/memory/eval semantics are mechanically correct.

## Log

- 2026-05-29 Opened after Task036 closed with no promoted checkpoint and user
  approved the multi-trial contract decisions.
- 2026-05-29 Completed 001 fake-env contract tests. H200 torch validation:
  `/mnt/workspace/users/guoyubo/agent_workspace/task037_contract_tmp_20260529`,
  `tests/test_task037_multitrial_contract.py -q` -> `2 passed in 2.48s`.
- 2026-05-29 Completed 002 MJLab auto-reset smoke. H200 summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/mjlab_multitrial_smoke/task037_mjlab_multitrial_smoke_summary.json`
  with `pass=true`, env64 one-iter smoke, env8192 one-iter smoke, and extras
  probe `done_matches_episode_done=true`.
- 2026-05-29 Completed 003 deterministic inner reset probe. H200 summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/deterministic_inner_reset/task037_mjlab_inner_reset_probe_env64_poststep_restore.json`
  with `pass=true`, `inner_command_max_delta=0.0`,
  `inner_failure_max_delta=0.0`, fixed phase `[0.0, 1.0]`, and root z `0.8`.
- 2026-05-29 Completed 004 multi-trial eval JSON. Existing Task036 AdaptK4
  checkpoint `model_5408.pt` was evaluated on H200 with 64 envs, 360 steps,
  2.0s trials, fixed 2.0 m/s command. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/multitrial_eval/task037_adaptk4_model5408_multitrial_eval_env64.json`.
  Result: `final_trial_pass=true`, `pass=true`, final-trial fall ratio `0.0`,
  final-trial velocity error `0.635804`, `promotion_gate=final_trial`,
  `quality_claim=false`.
- 2026-05-29 Completed 005 TXL-style memory consumer smoke. Added K160
  segment-token memory consumer for `3.2s` context and H200 task id
  `Unitree-G1-Gripper-Flat-Task037-TxlMemoryK160-DeterministicInnerReset-Fast2p0`.
  H200 evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/txl_memory_smoke/037_txl_k160_env64_iter1_gpu0_seed3700501.stdout.log`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/txl_memory_smoke/037_txl_k160_env8192_iter1_gpu0_seed3700511.stdout.log`.
  Result: env64 one-iteration PPO smoke `1423` steps/s, env8192 overhead
  smoke `55595` steps/s, `quality_claim=false`.
- 2026-05-29 Completed 006 long-context training decision. TXL K160 scratch
  training to 60 iterations produced checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task037_long_context_train/2026-05-29_14-54-42_037_txl_k160_scratch_env8192_iter60_gpu0_seed3700601/model_59.pt`.
  Full validation:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/full_validation_txl_k160_model59_iter60/task037_full_validation_summary.json`.
  Result: `pass=false`; speeds `0.4`, `1.2`, and `2.0` all failed dynamic
  switch with final-trial fall ratio `1.0`, and all deadgrid results were
  `0/12`.
- 2026-05-29 Reopened with 007 after a clean multi-trial diagnosis showed TXL
  K160 scratch fails before any failure adaptation: clean `0.4`, `1.2`, and
  `2.0 m/s` all had final-trial fall ratio `1.0`, while AdaptK4 `model_5408`
  passed clean Task037 eval at `2.0 m/s`. 007 focuses on AdaptK4 warmstart and
  clean-only K160 gait prior before any more failure matrix work.
- 2026-05-29 Completed 007 warmstart clean gait prior. AdaptK160 clean-only
  training from AdaptK4 `model_5408` produced:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task037_clean_gait_prior_train/2026-05-29_16-44-07_037_adapt_k160_clean_from_adaptk4_env8192_iter60_gpu0_seed3700705/model_5467.pt`.
  Clean multi-trial eval summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/clean_gait_prior/task037_adaptk160_model5467_clean_eval_summary.json`.
  Result: `pass=true`; `0.4`, `1.2`, and `2.0 m/s` all passed with
  final-trial fall ratio `0.0`.
- 2026-05-29 Opened 008 after user asked to try the failure curriculum from
  `model_5467.pt`. The first failure probe with the clean-only task id exposed
  a setup boundary: clean task ids do not contain `dynamic_motor_failure` or
  `motor_failure` events. Added dedicated AdaptK160 failure task ids before
  rerunning policy evidence.
- 2026-05-29 Completed 008 AdaptK160 failure curriculum probe. Baseline
  `model_5467.pt` already passes dynamic switch at `0.4`, `1.2`, and `2.0 m/s`
  but only passes `10/12` forced deadgrid cases at `2.0 m/s`; failed cases are
  `left_knee_joint` and `right_knee_joint`. One mixed persistent continuation
  produced `model_5496.pt`, which preserved clean and dynamic gates but still
  passed only `10/12` forced deadgrid cases. Knee checkpoint sweep did not find
  a mixed checkpoint that passes both knee blockers. Local summary:
  `task037_adaptk160_failure_probe_summary.json`.
- 2026-05-29 Opened 009 knee-focused repair after user asked to try the
  remaining knee blocker route. Added AdaptK160 task ids for existing Task030
  `knee_hiproll_vx2p0` and Task034 right-knee mixed stages.
- 2026-05-29 Completed 009 knee-focused repair as a rejected route. The
  `knee2p0` continuation from `model_5467.pt` and a conservative low-LR
  continuation from `model_5496.pt` did not find a checkpoint that passes both
  `left_knee_joint` and `right_knee_joint` forced-dead eval. Right knee can pass
  intermittently; left knee remains upright but misses the velocity gate
  (`~1.276 > 1.20`). Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/adaptk160_knee_repair/knee2p0_lr1e6_sweep_seed3700830/task037_adaptk160_knee2p0_lr1e6_sweep_summary.json`.
- 2026-05-29 Opened 010 after user asked to test whether eval should give the
  policy multiple trials without clearing memory. Added `--dynamic-dead-joint`
  to `task037_multitrial_eval_checkpoint` so the same single-joint dead-onset
  and recovery schedule can repeat across inner trials while `final_trial_pass`
  remains the promotion gate.
- 2026-05-29 Completed 010 arbitrary-onset multi-trial eval diagnostic for
  AdaptK160 `model_5467.pt`. H200 full grid result: `35/36` final-trial pass
  over speeds `0.4`, `1.2`, `2.0` and 12 leg joints. `0.4` and `1.2` were
  `12/12`; `2.0` was `11/12`, failing only `left_hip_pitch_joint`. A focused
  `left_hip_pitch_joint @ 2.0 m/s` five-seed repeat was `0/5`, so this is a
  stable remaining blocker. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/arbitrary_onset_multitrial_model5467/full_grid_seed3700902/task037_adaptk160_model5467_arbitrary_onset_multitrial_summary.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/arbitrary_onset_multitrial_model5467/left_hip_pitch_vx2p0_s5/task037_adaptk160_model5467_left_hip_pitch_vx2p0_s5_summary.json`.

## Review

Status: reopened for 009. 001 fake-env contract, 002 MJLab auto-reset smoke, 003
deterministic inner-reset probe, 004 eval JSON contract, and 005 TXL-style K160
construction/overhead smoke passed. 006 rejects the current TXL K160 scratch
policy-quality route. 007 confirms the warmstart clean gait prior route:
AdaptK160 `model_5467.pt` passes clean multi-trial eval at `0.4`, `1.2`, and
`2.0 m/s`. 008 shows `model_5467.pt` already passes dynamic switch but not the
forced knee deadgrid; generic mixed persistent continuation is not enough. 009
shows the existing knee2p0 repair stage is also not enough. No
deadgrid-complete checkpoint is promoted yet. 010 shows multi-trial eval
substantially improves arbitrary single-joint dynamic onset, but full random
dynamic robustness is still not solved because `left_hip_pitch_joint @ 2.0 m/s`
fails multi-seed.
