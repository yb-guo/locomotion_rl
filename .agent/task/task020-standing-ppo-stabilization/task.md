# Task 020: Standing PPO Stabilization

## Goal

Stabilize PPO active balance for the current Genesis G1 27DoF no-hand training
environment in standing mode.

This task is not a walking-quality claim. It does not try to prove passive
zero-action standing. Task019 already showed the current asset/contact dynamics
are not a clean passive-standing benchmark. Task020 asks whether PPO can learn a
stable active standing controller anyway.

## Scope

- Branch: `codex/task020-standing-ppo-stabilization`.
- Worktree:
  `../_worktrees/h200-locomotion-lab-task020-standing-ppo-stabilization`.
- Base: `master` after task014 minimal PPO smoke merge.
- Current environment: `VectorizedGenesisBackend` + G1 27DoF no-hand asset.
- Command mode: `standing` only until the standing gate passes.
- Physical GPU default: `CUDA_VISIBLE_DEVICES=1`,
  `physical_gpu=1`, `logical_cuda_device=cuda:0`.

## Non-Goals

- No walking-quality claim.
- No `vx_yaw` training before the readiness gate.
- No asset replacement.
- No Menagerie or `scene_mjx.xml` importer work.
- No MuJoCo reference work.
- No zero-action passive-standing proof.
- No LocoFormer.
- No SONIC.
- No ONNX.
- No rendering/GIF/video.
- No dataset/checkpoint/upstream repo download.
- No writes/deletes under `/mnt/workspace` or `/mnt/workspace1`.

## H200 Protocol

Remote commands must use:

```bash
/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/<project> && <command>'
```

All remote code, outputs, and intermediate files must stay under:

```text
/root/agent_workspace/project
```

Remote project:

```text
/root/agent_workspace/project/h200-locomotion-lab-task020-standing-ppo-stabilization
```

Default output root:

```text
outputs/task020/standing_ppo_stabilization/
```

## Success Metrics Contract

Standing PPO gate passes only if all hold:

- H200 physical GPU 1 evidence.
- 3 seeds.
- `command_mode=standing`.
- no NaN/Inf in obs/action/reward/value/logprob/loss/KL/entropy.
- actor and value params change.
- tensors stay on `cuda:0`.
- collect throughput stays above the task threshold.
- no full-env reset wave every rollout.
- `episode_len_mean >= 200` or `episode_len_mean >= 2x` baseline, whichever is
  easier to satisfy.
- `height_reset_rate` is near zero after reset-semantics hardening.
- `tilt_reset_rate` does not increase across training.
- deterministic standing eval confirms survival improvement or stable survival.
- read-only review finds no blocking boundary, correctness, or evidence issue.

## Diagnose Loop

Ranked hypotheses:

1. **Reset semantics still poison PPO.**
   - Prediction: hard height resets dominate even in standing; splitting
     diagnostic height from hard termination reduces reset noise.
2. **Reward scale gives weak or hackable balance signal.**
   - Prediction: reward improves without survival improvement, or one component
     dominates all others.
3. **Action energy is too high for early standing.**
   - Prediction: smaller action scale or lower initial std reduces tilt resets
     without freezing policy updates.
4. **Observation/action distributions destabilize value learning.**
   - Prediction: obs/value target/advantage/action stats are out of scale before
     reward changes.
5. **Environment/contact dynamics remain the blocker.**
   - Prediction: action is not saturated, PPO metrics are finite, but standing
     deterministic eval still shows repeated tilt/contact-collapse resets.

## Stop Rules

- If baseline has NaN/device/throughput failure, fix PPO loop plumbing before
  reward or action tuning.
- If hard height reset dominates, fix reset semantics before training longer.
- If reward components are not interpretable, stop before action ablation.
- If all action energy candidates still produce full-env reset waves, classify
  as environment/contact blocker and do not open yaw/vx.
- If deterministic standing eval does not improve, do not open yaw/vx.
- Do not add asset/importer/MuJoCo work to this task.
- Do not mark passed without H200 evidence and read-only review.

## Route

1. `000-success-metrics-contract.md`
2. `001-contract-standing-ppo-only.md`
3. `002-standing-baseline-repro.md`
4. `003-reset-metrics-hardening.md`
5. `003b-observation-action-distribution-profile.md`
6. `004-minimal-standing-reward-pack.md`
7. `005-action-energy-ablation.md`
8. `006-three-seed-standing-ppo-gate.md`
9. `006a-training-reset-wave-metrics.md`
10. `006c-rollout-tilt-sweep-causality.md`
11. `006b-deterministic-standing-eval.md`
12. `007-standing-to-yaw-readiness.md`
13. `008-review-and-decision.md`

## Acceptance

- Router creates task/subtask docs before coding.
- Coding subagents implement scoped code changes.
- Read-only reviewer reviews boundary, correctness, and evidence.
- Local focused tests pass.
- H200 focused tests pass.
- H200 standing PPO gate evidence is recorded.
- Deterministic standing eval evidence is recorded.
- Decision states one of:
  - standing PPO stable enough for yaw readiness gate;
  - PPO loop/reward/action still blocked;
  - environment/contact dynamics block current asset and a new asset/importer
    task is needed.

## Log

- 2026-05-12 Created after task019 diagnosed current Genesis G1 zero-action
  instability as foot/contact/ankle dynamics rather than PPO.
- 2026-05-12 Updated task020 branch with master task015-019 history before
  execution; resolved `.agent/index.md` so task015-020 are all listed.
- 2026-05-12 Local focused tests before H200 baseline:
  `PYTHONPATH=src python -m pytest tests/test_g1_ppo_smoke.py
  tests/test_ppo_loop.py -q -p no:cacheprovider` -> 9 passed, 4 skipped.
- 2026-05-12 H200 focused tests through guarded command:
  `PYTHONPATH=src python -m pytest tests/test_g1_ppo_smoke.py
  tests/test_ppo_loop.py -q -p no:cacheprovider` -> 13 passed.
- 2026-05-12 H200 standing baseline `h200-gpu1-standing-baseline-v1`:
  `CUDA_VISIBLE_DEVICES=1`, physical GPU 1, logical `cuda:0`,
  `command_mode=standing`, `action_scale_mult=0.25`, `root_z=1.20`,
  `termination_height_min=0.20`. Result: status ok, 3 seeds passed, no final
  reset/height/tilt failures, min collect throughput 35053.92 env-policy
  steps/s. Baseline shows PPO plumbing is healthy for short standing smoke, but
  task020 remains in progress because episode-length metrics, reset-rate
  hardening, deterministic eval, and review are still missing.
- 2026-05-12 Subtask003 reset metrics hardening committed in `bd73bc3` and
  verified locally and on H200. H200 reset-metrics run
  `h200-gpu1-standing-reset-metrics-v1`: status ok, 3 seeds passed,
  `CUDA_VISIBLE_DEVICES=1`, physical GPU 1, logical `cuda:0`, min collect
  throughput 35292.82 env-policy steps/s, mean final episode_length_mean
  51.9209, mean final survival_rate 1.0, max final height/tilt/timeout reset
  rates all 0.0, no final full-env reset wave.
- 2026-05-12 Subtask003b distribution profile committed through `55ab92d` and
  verified locally and on H200. H200 run `h200-gpu1-standing-profile-v2`:
  status ok, 3 seeds passed, physical GPU 1, logical `cuda:0`, min collect
  throughput 35548.72 env-policy steps/s, no final reset wave, action
  saturation ratio about 0.0026, observation_std about 0.314, log_std_mean
  about -0.499. Reward contributions show base_height contributes 0.0 under
  current config, while lin/yaw/upright dominate. No normalization or std clamp
  is justified before reward-pack work.
- 2026-05-12 Subtask004 minimal reward pack committed in `a68575f` and verified
  locally/on H200. H200 run `h200-gpu1-standing-reward-pack-v1`: status ok,
  3 seeds passed, physical GPU 1, logical `cuda:0`, `base_height_reward_scale=0.20`,
  `joint_velocity_penalty_scale=0.001`, `termination_penalty=-1.0`, min collect
  throughput 16598.49 env-policy steps/s, mean final reward 1.82327, mean final
  survival_rate 1.0, height/tilt/timeout reset rates 0.0, action saturation
  about 0.0026, no final full-env reset wave. Reward improved without survival
  regression.
- 2026-05-12 Subtask005 action-energy ablation tooling committed through
  `fa67354` after read-only review. H200 focused verification passed
  55 tests. H200 matrix `h200-gpu1-action-energy-v2` ran the bounded
  standing-only matrix on physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`, logical
  `cuda:0`): 12/12 candidates completed, no subprocess/runtime failures, parent
  summary status `passed`. Selected candidate is `scale_0p1_logstd_neg2p0`
  (`action_scale_mult=0.10`, `log_std_init=-2.0`) with reward 1.943375,
  episode_length_mean 51.659180, survival_rate 1.0, reset_rate 0.0,
  tilt_reset_rate 0.0, action_saturation_ratio 0.0, and min collect throughput
  18182.11 env-policy steps/s. This completes the one-seed action-energy
  selection step; task020 remains in progress because the 3-seed standing PPO
  gate and deterministic eval are still pending.
- 2026-05-12 Subtask006 H200 3-seed standing PPO gate
  `h200-gpu1-standing-gate-v1` ran with the selected subtask005 config
  (`action_scale_mult=0.10`, `log_std_init=-2.0`) and subtask004 reward/reset
  pack. Result: smoke status ok, all 3 seeds passed PPO plumbing checks,
  physical GPU 1, logical `cuda:0`, no final height/tilt/timeout resets, no
  final full-env reset wave, actor/value params changed, and min collect
  throughput 44411.12 env-policy steps/s. Gate did not meet the task episode
  threshold: final mean episode_length_mean was 67.295247 versus baseline
  51.9209 and the easier `2x` baseline threshold of about 103.84. Reward peaked
  early near 2.216 per seed but ended around 1.48-1.50 with low final root
  height, so task020 remains blocked at the standing gate and yaw/vx remains
  closed.
- 2026-05-12 Subtask006a added training-wide reset metrics to avoid reading
  only the final rollout. Local related tests passed 48/48 with 10 skipped;
  H200 related tests passed 58/58. H200 gate rerun
  `h200-gpu1-standing-gate-v2` kept the same blocked conclusion:
  final mean episode_length_mean 67.291992, max training episode_length_mean
  71.383545, min collect throughput 44088.53 env-policy steps/s, and all
  plumbing checks passed. The new metrics show no single-step full-env reset
  wave (`training_full_env_reset_wave_count=0`), but every seed has
  rollout-window tilt reset sweeps at updates 2, 5, 8, 11, 14, and 17
  (`reset_count=1024`, `tilt_reset_count=1024`, max training reset/tilt reset
  rate 0.03125). The task remains blocked in standing; yaw/vx remains closed.
- 2026-05-12 Subtask006c no-update causality probes reproduced the same reset
  sweep cadence without PPO updates. `zero_action`,
  `untrained_mean_action`, and `untrained_sampled_action` all had
  first_tilt_chunk 2, max_reset_count 1024, mean_reset_count 307.2,
  final_reset_count 0, and reset chunks `2,5,8,11,14,17`. The zero-action run
  had action mean/max 0.0/0.0, so the standing gate blocker does not require
  PPO updates or action noise. Deterministic standing eval and yaw readiness
  were skipped by stop rule.

## Review

Status: final read-only review passed with no blocking findings. Task020 remains
blocked at the standing PPO gate and is not passed.
