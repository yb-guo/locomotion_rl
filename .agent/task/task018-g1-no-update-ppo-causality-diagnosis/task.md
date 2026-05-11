# Task 018: G1 No-Update PPO Causality Diagnosis

## Goal

Separate simulator/control/reset instability from PPO-update-induced
instability for the G1 standing reset wave.

Task017 ruled out action amplitude, broad action group, and exploration noise
as sufficient root causes. Task018 asks:

```text
Does the same standing env fall without PPO updates, or only after learning
updates start changing the policy/value function?
```

This task is not a walking-quality claim.

## Scope

- Branch: `codex/task018-g1-no-update-ppo-causality-diagnosis`.
- Worktree:
  `../_worktrees/h200-locomotion-lab-task018-g1-no-update-ppo-causality-diagnosis`.
- Base: stacked on task017 commit `cfbd332`.
- Add a standalone no-update probe tool.
- Reuse the same G1 27DoF no-hand standing env config as task017.
- Compare no-update rollouts against task017 PPO standing u50 evidence.
- Only run PPO-update isolation if no-update probes show the env/control path
  can remain stable.

## Non-Goals

- No walking-quality claim.
- No reward redesign.
- No full PPO sweep.
- No LocoFormer integration.
- No SONIC integration.
- No ONNX export.
- No rendering/GIF/video.
- No domain randomization.
- No dataset/checkpoint/asset/upstream download.
- No writes/deletes under `/mnt/workspace` or `/mnt/workspace1`.
- No change to `GenesisG1SceneBackend`.

## H200 Protocol

Remote commands must use:

```bash
/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/<project> && <command>'
```

All remote code, outputs, and intermediate files must stay under:

```text
/root/agent_workspace/project
```

Default GPU:

```text
CUDA_VISIBLE_DEVICES=1
physical_gpu=1
logical_cuda_device=cuda:0
```

Remote project:

```text
/root/agent_workspace/project/h200-locomotion-lab-task018-g1-no-update-ppo-causality-diagnosis
```

Run dir:

```text
outputs/task018/no_update_ppo_causality/<run_id>/
```

## Diagnose Loop

### Feedback Loop

Use standing-only no-update rollouts over the same horizon as task017 u50:

- seed `0`;
- `50 chunks`;
- `32 policy steps per chunk`;
- total `1600` policy steps;
- one stage: `standing`;
- same reset pose/config as task017.

Record per chunk:

- `chunk_index`;
- `reset_count`;
- `tilt_bad_count`;
- `termination_height_bad_count`;
- `root_height_mean/min`;
- `upright_mean`;
- action abs mean/max/std;
- top action RMS joints;
- throughput.

### Ranked Hypotheses

1. **Control/reset semantics are unstable without learning**
   - Prediction: `zero_action` or deterministic no-update rollout reproduces
     reset waves.
2. **Sampling distribution is destabilizing before learning**
   - Prediction: `untrained_sampled_action` falls while zero/deterministic
     modes remain stable.
3. **PPO update destabilizes an otherwise stable rollout distribution**
   - Prediction: zero/deterministic/sampled no-update probes stay stable, but
     task017 PPO u50 falls.
4. **Actor update specifically destabilizes**
   - Prediction: actor-frozen or critic-only PPO update avoids the collapse
     after no-update probes prove the env path is stable.

## Stop Rules

- Run no-update probes before PPO-update isolation.
- If `zero_action` no-update falls, stop before actor/critic isolation and
  record reset/default-pose/PD/control as the next area.
- If no-update sampled policy falls but zero/deterministic do not, stop before
  actor/critic isolation and record sampling/action-interface as the next area.
- Run PPO-update isolation only if no-update probes are stable enough to make
  learning the likely cause.
- Change one variable per probe.
- Do not run final three-seed experiments in this task.
- Do not mark passed without local tests, H200 focused tests, H200 probe
  evidence, and read-only review.

## Acceptance

- Router creates task/subtask docs before coding.
- Coding subagent implements standalone no-update probe.
- Read-only reviewer subagent reviews boundary, correctness, and evidence.
- Local focused tests pass.
- Local full pytest passes.
- H200 focused tests pass.
- H200 seed-0 no-update matrix compares:
  - `zero_action`;
  - `untrained_mean_action`;
  - `untrained_sampled_action`.
- PPO-update isolation is either run with evidence or explicitly skipped by a
  stop rule.
- Decision states whether the next area is control/reset semantics,
  sampling/action-interface, or PPO update/reward/value learning.

# Route

1. `001-diagnosis-contract.md`
2. `002-no-update-rollout-probe.md`
3. `003-h200-no-update-baseline.md`
4. `004-h200-ppo-update-isolation.md`
5. `005-review-and-decision.md`

# Log

- 2026-05-11 Created task018 branch/worktree from task017 commit `cfbd332`.
- 2026-05-11 Router created diagnosis contract before coding.
- 2026-05-11 Coding subagent implemented
  `src/h200_locomotion_lab/tools/g1_no_update_ppo_causality.py` and focused
  tests.
- 2026-05-11 Read-only reviewer found a blocking standalone-boundary issue;
  coding subagent fixed it by removing the `g1_curriculum_ppo_smoke` import.
- 2026-05-11 Local verification passed: focused task018 tests, related PPO
  smoke tests, and full pytest (`220 passed, 11 skipped`).
- 2026-05-11 H200 focused tests passed: `43 passed in 5.06s`.
- 2026-05-11 H200 seed-0 no-update matrix completed under:
  `/root/agent_workspace/project/h200-locomotion-lab-task018-g1-no-update-ppo-causality-diagnosis/outputs/task018/no_update_ppo_causality/h200-gpu1-seed0-no-update-v1`.
- 2026-05-11 `zero_action`, `untrained_mean_action`, and
  `untrained_sampled_action` all reproduced `first_tilt_chunk=2`,
  `mean_reset_count=348.16`, `final_reset_count=1024`, and
  `final_tilt_bad_count=1024`.
- 2026-05-11 PPO-update isolation was skipped by stop rule because
  `zero_action` no-update already falls.
- 2026-05-11 Final read-only reviewer found no blocking findings and allowed
  marking task018 passed.

# Review

Status: passed.

- Verification evidence is complete: local focused tests, local full pytest,
  H200 focused tests, and H200 no-update matrix.
- Review is complete: the final read-only reviewer found no blocking findings.
- Decision: next work should target reset/default-pose/PD/control semantics
  before PPO reward/value/advantage changes.
