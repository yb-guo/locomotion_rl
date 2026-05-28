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

## Review

Status: planned. No implementation evidence yet.
