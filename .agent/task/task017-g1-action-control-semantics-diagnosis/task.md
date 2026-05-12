# Task 017: G1 Action Control Semantics Diagnosis

## Goal

Diagnose why G1 PPO standing resets begin at update 2 even after task016 ruled
out LR, termination penalty, and action-rate penalty as sufficient fixes.

The target question is:

```text
Is the policy/action interface pushing the robot out of balance, or is the
reward/optimizer loop destabilizing a physically valid control surface?
```

This task is not a walking-quality claim.

## Scope

- Branch: `codex/task017-g1-action-control-semantics-diagnosis`.
- Worktree:
  `../_worktrees/h200-locomotion-lab-task017-g1-action-control-semantics-diagnosis`.
- Base: stacked on task016 commit `e58e90d`.
- Reuse task016 diagnostic runner patterns.
- Add a standing-only fast diagnostic path.
- Add action statistics to PPO smoke metrics:
  - normalized action abs mean/max/std;
  - per-joint action RMS top entries if cheap enough.
- Run cheap H200 seed-0 matrix before any long curriculum run.

## Non-Goals

- No walking-quality claim.
- No full reward redesign.
- No PPO hyperparameter sweep beyond explicitly listed probes.
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
/root/agent_workspace/project/h200-locomotion-lab-task017-g1-action-control-semantics-diagnosis
```

Run dir:

```text
outputs/task017/action_control_semantics/<run_id>/
```

## Diagnose Loop

### Feedback Loop

Use standing-only PPO smoke as the fast repro:

- seed `0`;
- `updates_per_stage=10`;
- `n_envs=1024`;
- `rollout_steps=32`;
- one stage: `standing`.

Record per variant:

- first update with `tilt_bad_count > 0`;
- max/mean/final `reset_count`;
- max/final `tilt_bad_count`;
- final root height and upright;
- normalized action abs mean/max/std;
- top action RMS joints;
- KL/entropy/throughput.

### Ranked Hypotheses

1. **Action amplitude is too high**
   - Prediction: lower `action_scale_mult` delays or reduces reset waves.
2. **Action joint group is too broad**
   - Prediction: `legs` or `legs_waist` reduces reset waves versus `all`.
3. **Exploration noise is too high**
   - Prediction: lower `log_std_init` delays or reduces reset waves.
4. **Upright/default-joint shaping is still too weak**
   - Prediction: stronger upright or joint-deviation shaping helps only after
     action amplitude/control probes are bounded.

## Stop Rules

- If standing baseline does not reproduce the reset wave, stop and record the
  mismatch before tuning.
- Change one variable per variant.
- Do not run final three-seed experiments in this task.
- Do not move to longer curriculum until standing-only evidence points to a
  cause class.
- Do not mark passed without local tests, H200 focused tests, H200 ablation
  evidence, and read-only review.

## Acceptance

- Router creates task/subtask docs before coding.
- Coding subagent implements instrumentation/standing-only ablation runner.
- Read-only reviewer subagent reviews boundary, correctness, and evidence.
- Local focused tests pass.
- Local full pytest passes.
- H200 focused tests pass.
- H200 seed-0 standing baseline reproduces recurring tilt reset waves, or
  mismatch is recorded and the task stops.
- H200 seed-0 matrix compares at least:
  - baseline `action_scale_mult=0.10`, `action_joint_group=all`;
  - `action_scale_mult=0.05`;
  - `action_scale_mult=0.03`;
  - `action_scale_mult=0.01`;
  - `action_joint_group=legs`;
  - `action_joint_group=legs_waist`;
  - lower exploration noise.
- Decision states which hypothesis is most supported, or why evidence is
  inconclusive.

# Route

1. `001-diagnosis-contract.md`
2. `002-standing-only-runner-and-action-stats.md`
3. `003-h200-standing-baseline.md`
4. `004-h200-action-control-matrix.md`
5. `005-review-and-decision.md`

# Log

- 2026-05-11 Created task017 branch/worktree from task016 commit `e58e90d`.
- 2026-05-11 Router created diagnosis contract before coding.
- 2026-05-11 Coding subagent implemented standing-only stage selection, action
  stats, and task017 ablation runner.
- 2026-05-11 Router corrected task017 runner default to
  `updates_per_stage=10`.
- 2026-05-11 Local focused tests passed with 17 passed, 4 skipped.
- 2026-05-11 Local full pytest passed with 214 passed, 8 skipped.
- 2026-05-11 Read-only reviewer found no blocking findings before H200.
- 2026-05-11 H200 focused tests passed with 40 passed in 2.59s.
- 2026-05-11 H200 u10 standing-only matrix completed all variants:
  `/root/agent_workspace/project/h200-locomotion-lab-task017-g1-action-control-semantics-diagnosis/outputs/task017/action_control_semantics/h200-gpu1-seed0-standing-u10-v1`.
- 2026-05-11 H200 u50 targeted checks completed:
  `/root/agent_workspace/project/h200-locomotion-lab-task017-g1-action-control-semantics-diagnosis/outputs/task017/action_control_semantics/h200-gpu1-seed0-standing-u50-targeted-v1`.
- 2026-05-11 Final read-only reviewer found no blocking findings.

# Review

Status: passed.

- Baseline reproduced the update-2 reset wave.
- u10 matrix: all variants reproduced the update-2 reset wave, then recovered
  by final row.
- u50 targeted: baseline, `action_scale_mult=0.01`, and
  `log_std_init=-3.5` all ended with full reset/tilt collapse.
- Decision: action amplitude, broad action joint group, and exploration noise
  are not sufficient root causes. Next diagnostic should separate no-update
  simulator/control behavior from PPO-update reward/learning behavior.
