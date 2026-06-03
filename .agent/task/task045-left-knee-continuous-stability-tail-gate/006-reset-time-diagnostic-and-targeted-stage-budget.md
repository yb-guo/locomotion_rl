# 006: Reset-Time Diagnostic And Targeted Stage Budget

## Route

Task045 should not become an unbounded left-knee reward tuning loop. The next
closed unit is to instrument the existing continuous eval so every failed JSON
shows where physical resets happen:

- before the dynamic fault;
- between fault onset and the scored post-fault window;
- inside the scored post-fault window;
- after the scored post-fault window, during the long-tail rollout.

The pass/fail contract stays unchanged. This diagnostic only explains a failed
gate; it does not relax `physical_reset_events == 0`, the post-fault quality
thresholds, action shape checks, or memory/runner checks.

## Acceptance

- `task044_continuous_fault_eval` writes `physical_reset_time_diagnostic`
  into JSON.
- The diagnostic includes bin-level and phase-level reset/fall/timeout counts.
- The diagnostic includes first-reset and first-fall env counts plus time range.
- Local tests cover bin and phase accounting.
- H200 reruns the current best checkpoint with the diagnostic enabled.
- Based on that JSON, choose at most two targeted continuation stages:
  1. onset-shock repair if resets cluster near fault onset;
  2. long-tail endurance repair if resets cluster after the scored window.
- If those targeted stages do not approach the gate, stop Task045 as a
  documented local-repair ceiling and return to the LocoFormer policy route.

## Log

- 2026-06-02 Opened after the long-survival stage reduced the best post-fault
  fall ratio to `0.078125` but still failed continuous eval with physical reset
  events. The unknown is no longer "does it fall"; the unknown is when it falls.
- 2026-06-02 Added `physical_reset_time_diagnostic` instrumentation to
  continuous eval. It records 0.5 s bins by default and phase summaries using
  the existing dynamic onset and post-fault window boundaries.
- 2026-06-02 H200 diagnostic on the current best checkpoint under the unchanged
  old gate:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/reset_time_diagnostic/pose_tight_gate_long_survival_env2048_lr5e6_model39_reset_time_seed4520501.json`.
  Result: `pass=false`, `physical_reset_events=63`, `post_fault_window.fall_ratio=0.08203125`.
  Reset phases:
  - `pre_fault=3`;
  - `fault_onset_to_post_window_start=4`;
  - `post_fault_window=21`;
  - `post_fault_after_window=35`.
  This rules out an onset-only repair. The largest residual is long-tail
  stability after the scored window, while the scored window itself is still
  above the `0.05` fall-ratio gate.
- 2026-06-02 Selected targeted stage 1:
  `Unitree-G1-Gripper-Flat-Task045-TrueTxlHiddenFaultTrain-PoseTightGateLeftKneeLongTail1p6`.
  It inherits the old `PersistentHiddenPoseTight1p6` gate, sets an 8.0 s
  episode, forces left-knee dead onset at 2.0 s, keeps the fault for the full
  long-tail horizon, and strengthens survival/posture pressure. This is the
  first of the allowed one or two targeted continuation stages.
- 2026-06-02 Stage 1 smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/hidden_fault_train/train_pose_tight_gate_long_tail_smoke_env64_iter1_seed4520901.json`.
- 2026-06-02 Stage 1 formal continuations completed but did not improve the old
  gate. Same-seed comparison on `4521101`:
  - original best:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/reset_time_diagnostic/pose_tight_gate_baseline_long_survival_model39_reset_time_seed4521101.json`,
    `physical_reset_events=56`, `post_fault_window.fall_ratio=0.078125`;
  - stage 1 `4096/lr2e-6/model39`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/reset_time_diagnostic/pose_tight_gate_long_tail_env4096_lr2e6_model39_reset_time_seed4521101.json`,
    `physical_reset_events=68`, `post_fault_window.fall_ratio=0.09765625`;
  - stage 1 `2048/lr5e-6/model39`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/reset_time_diagnostic/pose_tight_gate_long_tail_env2048_lr5e6_model39_reset_time_seed4521101.json`,
    `physical_reset_events=76`, `post_fault_window.fall_ratio=0.1171875`.
  Stage 1 is therefore a negative result, likely because full-actor PPO on the
  aligned long-tail task perturbs the existing gait prior.
- 2026-06-02 Selected targeted stage 2, the final allowed local-repair stage:
  rerun the same old-gate-aligned long-tail curriculum from the original best
  checkpoint, but with `actor_trainable_scope=txl_residual_and_mlp_memory_input`
  and smaller learning rates. This tests whether the memory/adaptation branch
  can repair the failure without degrading the base gait.
- 2026-06-02 Launched targeted stage 2 on H200:
  - `4096/lr1e-6/seed4521201`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/hidden_fault_train/train_pose_tight_gate_long_tail_scope_env4096_iter40_lr1e6_seed4521201.json`;
  - `2048/lr2e-6/seed4521202`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/hidden_fault_train/train_pose_tight_gate_long_tail_scope_env2048_iter40_lr2e6_seed4521202.json`.
- 2026-06-02 Targeted stage 2 also failed to improve the unchanged old gate.
  Representative eval JSONs:
  - original best, seed `4521302`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/reset_time_diagnostic/pose_tight_gate_baseline_long_survival_model39_reset_time_seed4521302.json`,
    `physical_reset_events=53`, `post_fault_window.fall_ratio=0.09765625`;
  - scope `2048/lr2e-6/model20`, seed `4521302`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/reset_time_diagnostic/pose_tight_gate_long_tail_scope_env2048_lr2e6_model20_reset_time_seed4521302.json`,
    `physical_reset_events=60`, `post_fault_window.fall_ratio=0.11328125`;
  - scope `4096/lr1e-6/model20`, seed `4521302`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/reset_time_diagnostic/pose_tight_gate_long_tail_scope_env4096_lr1e6_model20_reset_time_seed4521302.json`,
    `physical_reset_events=68`, `post_fault_window.fall_ratio=0.1328125`;
  - scope `4096/lr1e-6/model39`, seed `4521301`:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/reset_time_diagnostic/pose_tight_gate_long_tail_scope_env4096_lr1e6_model39_reset_time_seed4521301.json`,
    `physical_reset_events=92`, `post_fault_window.fall_ratio=0.13671875`.
  The second targeted stage is therefore also a negative result.

## Review

Status: complete as diagnostic, negative as local repair.

The reset-time diagnostic is implemented and verified. It showed that the old
gate failure is mostly post-fault-window and long-tail stability, not an
onset-only shock. Both allowed targeted continuation stages were run and neither
improved the unchanged old gate. Per the route, Task045 should stop local
reward/curriculum repair here and hand off to the LocoFormer policy route.
