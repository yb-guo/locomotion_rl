# Task 033: Shared History Memory Policy Bakeoff

## Route

Build one shared history infrastructure and use it to test three policy
consumers without duplicating env, failure scheduler, eval, or rendering code.

Motivation:

- Task031 showed `model_5349.pt` passes unified-speed canonical dynamic switch
  but fails forced persistent dead-grid.
- Task032 showed weak-focused MLP curriculum does not materially fix the
  dead-grid blocker.
- The next hypothesis is that the policy needs temporal evidence of action to
  joint-response mismatch, but the history collection path must be independent
  of the policy architecture.

Goal:

Implement and compare:

1. StackMLP consumer;
2. GRU consumer;
3. LocoFormer-style memory consumer.

All three must consume the same shared history stream and use the same eval
matrix.

Fixed boundaries:

- Keep MJLab envs, rewards, failure schedulers, and action space unchanged.
- Keep action `31D`.
- Do not expose explicit fault labels, motor scales, failure masks, or active
  joint ids to the actor.
- Preserve existing clean/random persistent/dead-grid/dynamic-switch eval
  semantics.
- Keep debug JSON allowed to record hidden failure state.
- Do not implement morphology randomization or full LocoFormer morphology
  generalization in this task.

Architecture target:

```text
MJLab env obs/action/done stream
        |
        v
SharedHistoryBuffer
        |
        v
PolicyInputBuilder
        |
        +-- StackMLP input
        +-- GRU input
        +-- LocoFormer-style token input
```

Planned slices:

1. `001-history-buffer-contract.md`
   - Define actor-visible history fields, debug-only fields, reset semantics,
     device placement, and no-leak constraints.

2. `002-overhead-harness.md`
   - Measure baseline MLP, buffer-only, StackMLP, GRU, and LocoFormer-style
     overhead with the same env count and horizon.

3. `003-stack-mlp-consumer.md`
   - First working consumer: flatten `K` frames from the shared buffer and feed
     an MLP.

4. `004-gru-consumer.md`
   - Add recurrent consumer using the same buffer stream and explicit hidden
     reset rules.

5. `005-locoformer-style-consumer.md`
   - Add a minimal tokenized history consumer, not full morphology
     generalization.

6. `006-bakeoff-eval-decision.md`
   - Run shared eval matrix and decide which policy route deserves further
     investment.

## Minimal Closed Loop

1. Implement GPU-resident shared history buffer smoke.
2. Prove buffer-only overhead before training any new policy.
3. Train or smoke each consumer behind the same interface.
4. Evaluate each consumer on the same blocker subset:
   - speeds `0.4`, `1.6`, `2.0 m/s`;
   - forced persistent dead-grid;
   - canonical dynamic switch.
5. Only run full `0.4..2.0` matrix for a consumer that improves the blocker
   subset without regressing dynamic switch.

Acceptance:

- Shared buffer is batched, GPU-resident, and reset-aware.
- History buffer does not expose explicit actor fault labels.
- Overhead JSON reports steps/s, policy forward time if available, GPU memory,
  actor input dim, and history length.
- All three consumers have at least one smoke result.
- At least the most promising consumer has blocker-subset eval JSON.
- Final decision is explicit: StackMLP enough, GRU needed, LocoFormer-style
  promising, or memory-policy route not yet sufficient.

Pass:

- This task passes when it produces comparable overhead and blocker-eval
  evidence for the three consumers. It does not require solving all dead-grid
  cases.

Fail:

- Each policy maintains its own incompatible history mechanism.
- Eval cases or thresholds change between consumers.
- History buffer is CPU/Python-loop bound at H200 env scale.
- Actor receives explicit failure ids, failure masks, or motor scales.
- Training reward is used as the only comparison.

Evidence root:

`/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task033/`

## Log

- 2026-05-28 Opened after user decision to implement StackMLP, GRU, and
  LocoFormer-style memory in one task with a shared history buffer and overhead
  measurements.
- 2026-05-28 Created branch
  `codex/task033-shared-history-memory-policy-bakeoff` from merged
  `origin/master` and started router/subagent execution.
- 2026-05-28 Completed 001 buffer contract smoke locally and on H200 CUDA.
  H200 evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task033/history_buffer_smoke/task033_history_buffer_cuda_smoke.json`.
- 2026-05-28 Registered Task033 MJLab/RSL-RL stages on H200:
  `BufferOnlyK4`, `StackMlpK4`, `StackMlpK8`, `GruK4`, and `TokenK4`.
  Added local summary artifact:
  `.agent/task/task033-shared-history-memory-policy-bakeoff/task033_h200_history_policy_smoke_summary.json`.
- 2026-05-28 H200 env64 train smokes passed for baseline MLP,
  buffer-only K4, StackMLP K4, GRU K4, and token K4. These validate runner
  construction and one short PPO loop, not policy quality.
- 2026-05-28 H200 env8192 overhead smokes passed initial gates for
  buffer-only K4, StackMLP K4, and StackMLP K8 against a baseline MLP
  denominator. GRU/token construction/cost smokes also completed at env8192:
  `52312` steps/s for GRU K4 and `56745` steps/s for token K4.
- 2026-05-28 Added StackMLP checkpoint migration from `model_5349.pt`.
  Migration maps the old `104D` actor first layer to the newest-frame obs slice
  inside the `540D` K4 history input, starts other history/action columns at
  zero, keeps the critic `119D`, and uses a fresh optimizer. H200 migration JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task033/migration_smoke/model_5349_stackmlp_k4_migration.json`.
- 2026-05-28 H200 StackMLP K4 migration load/train smoke completed one env64
  PPO iteration from the migrated checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_history_migration_warmstart/2026-05-28_12-20-22_033_stackmlp_k4_migration_load_smoke_env64_iter1_gpu0_seed3303341`.
- 2026-05-28 Naive StackMLP K4 PPO from the migrated checkpoint regressed
  policy quality. `model_5378.pt` from
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_stackmlp_k4_warmstart/2026-05-28_12-23-30_033_stackmlp_k4_from5349_env8192_iter30_gpu1_seed3303351`
  failed dynamic switch with `zero_fall_ratio=0.0`.
- 2026-05-28 Added and trained frozen-base StackMLP K4, which preserves the
  migrated base actor path while training only new history/action columns.
  Checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_stackmlp_k4_frozenbase_focused/2026-05-28_12-40-56_033_frozenbase_focused_from5349_env8192_iter30_gpu1_seed3303362_lr1e5/model_5378.pt`.
  It passed `2.0 m/s` dynamic switch:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task033/stackmlp_k4_eval/frozenbase_model5378_dynamicmotorfailure_vx2p0_seed3105349/task033_dynamic_eval_switch_vx2p0.json`
  (`pass=true`, `zero_fall_ratio=1.0`, `recovery_success_ratio=1.0`).
  It improved `2.0 m/s` forced persistent dead-grid to `11/12`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task033/stackmlp_k4_eval/frozenbase_model5378_deadgrid_vx2p0_seed3303500/task033_failure_grid_eval_aggregate.json`
  (`pass=false`; only `right_knee_joint` failed with
  `zero_fall_ratio=0.2109375`).

## Review

Status: decision-ready partial close. Shared history buffer, StackMLP, GRU,
and token consumers are wired through one history path with smoke/overhead
evidence. Frozen-base StackMLP K4 is the only consumer with policy-quality
eval evidence: it preserves dynamic-switch performance and improves the
  forced dead-grid blocker, but does not fully solve it. Do not claim full
  robust walking pass; the remaining blocker is `right_knee_joint` forced dead.
- 2026-05-28 Task034 follow-up checkpoint sweep superseded the Task033
  `model_5378.pt` selection: earlier checkpoint `model_5350.pt` from the same
  frozen-base run passes both `2.0 m/s` dynamic switch and full `2.0 m/s`
  forced dead-grid. Keep Task033 as the history-infra bakeoff; use Task034 for
  the corrected right-knee decision evidence.
