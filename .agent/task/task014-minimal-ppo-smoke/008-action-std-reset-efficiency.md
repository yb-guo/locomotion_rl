# 008: Action Std Reset Efficiency

## Goal

Diagnose whether high initial action noise is causing excessive fall/reset cost
in the task014 PPO smoke loop, then choose a safer initial exploration setting
if H200 evidence supports it.

## Route

1. Reproduce current v3 symptom:
   - all seeds pass;
   - final update still reports `fallen_count=2048`;
   - collect dominates update time.
2. Record ranked hypotheses:
   - `log_std_init=-0.5` samples actions too aggressively for the untrained G1;
   - lower initial action std reduces fall/reset frequency and collect time;
   - if collect is dominated by state reads/physics, fall count will change
     little or throughput will not improve;
   - if update dominates, action std will not materially affect wall time.
3. Add a CLI knob for `--log-std-init`.
4. Run H200 sweeps on physical GPU 1 with no other route changes.
5. Keep the best setting only if it preserves task014 pass criteria and improves
   reset/throughput evidence.

## Stop Rules

- Do not touch `GenesisG1SceneBackend`.
- Do not tune reward, curriculum, LocoFormer, SONIC, ONNX, planner, render, GIF,
  downloads, or `/mnt/workspace*`.
- If a lower std hides NaN/Inf or stops actor/value parameter updates, reject it.
- If fall/reset count does not improve, leave default unchanged and record the
  rejected hypothesis.

## Verification

- Local focused tests pass.
- H200 focused tests pass.
- H200 smoke or sweep evidence records:
  - `log_std_init`;
  - `fallen_count`;
  - collect throughput;
  - collect/update time;
  - pass/fail.

## Log

- 2026-05-09 Reproduced v3 symptom:
  - all seeds passed;
  - final rollouts still reported `fallen_count=2048`;
  - collect time dominated update time.
- Added CLI controls:
  - `--log-std-init`;
  - `--height-min`;
  - `--height-max`;
  - `--root-z`.
- Log-std sweep, seed 0:
  - `-0.5`: final `fallen_count=2048`, min collect
    `17093.651360569183`;
  - `-1.0`: final `fallen_count=2048`, min collect
    `15992.039674103658`;
  - `-1.5`: final `fallen_count=2048`, min collect
    `20059.93358830347`;
  - `-2.0`: final `fallen_count=2048`, min collect
    `23697.80576523341`.
  Lower action noise did not reduce reset count, so action noise was rejected as
  the root cause.
- Reset-cause profile:
  - `reset_count=2048`;
  - `height_bad_count=2048`;
  - `tilt_bad_count=0`;
  - `root_height_min=0.4272436201572418`;
  - `root_height_mean=0.6545456051826477`;
  - `upright_mean=0.9985527396202087`.
  The robot was upright but crossed the root-height threshold.
- Height threshold probes:
  - `height_min=0.40`: still `height_bad_count=2048`,
    `root_height_min=0.37821462750434875`;
  - `height_min=0.35`: still `height_bad_count=2048`,
    `root_height_min=0.32381224632263184`.
  Lowering the threshold only delayed reset and was rejected.
- Root height probes, seed 0:
  - `root_z=0.90`: min collect `43457.31948907588`,
    final collect `53887.8588314765`, `reset_count=2048`;
  - `root_z=1.00`: min collect `43121.12575640351`,
    final collect `54948.11222650299`, `reset_count=2048`;
  - `root_z=1.10`: min collect `44282.19463063956`,
    final collect `59589.964558626896`, `reset_count=2048`.
- Implemented default PPO smoke CLI root height:
  - `--root-z` default changed from `0.78` to `1.10`;
  - `config.json` now records env settings.
- H200 focused verification:
  - `11 passed in 2.77s`.
- H200 default three-seed PPO smoke:
  - command used `CUDA_VISIBLE_DEVICES=1`;
  - physical GPU `1`, logical device `cuda:0`;
  - run dir:
    `/root/agent_workspace/project/h200-locomotion-lab-task014-minimal-ppo-smoke/outputs/task014/minimal_ppo_smoke/h200-gpu1-rootz110-three-seed-v1`;
  - `metrics.jsonl`: 15 rows;
  - `summary.json`: `all_seeds_passed=true`;
  - `final_checkpoint.pt`: exists;
  - synchronized min collect throughput:
    `28997.905703819983 env_policy_steps_per_sec`;
  - mean final reward mean: `1.4842438697814941`;
  - observed total smoke command wall time: about `48.1s`.
- Per-seed final update metrics:
  - seed 0: collect `46325.40749661221/s`, `reset_count=1024`,
    `height_bad_count=1024`, `tilt_bad_count=0`,
    `root_height_mean=0.8669933676719666`, `upright_mean=0.9990267753601074`;
  - seed 1: collect `45960.52441677656/s`, `reset_count=1024`,
    `height_bad_count=1024`, `tilt_bad_count=0`,
    `root_height_mean=0.8671162128448486`, `upright_mean=0.9990234375`;
  - seed 2: collect `44952.96445143137/s`, `reset_count=1024`,
    `height_bad_count=1024`, `tilt_bad_count=0`,
    `root_height_mean=0.8668215870857239`, `upright_mean=0.9990200996398926`.

## Review

Status: passed.

- Correct hypothesis: reset root height was too low for this smoke loop,
  causing earlier height-threshold resets despite upright posture.
- Rejected hypotheses:
  - lower initial action noise did not reduce reset count;
  - lower height threshold did not remove resets and would weaken termination
    semantics.
- `root_z=1.10` preserves task014 acceptance and improves synchronized min
  collect throughput from `19381.815355781637` to `28997.905703819983`.
- Final rollout reset/fall count improved from `2048` to `1024`.
- PPO update remains small relative to collect; next optimization should target
  env/reset mechanics or a standing-stability initialization, not PPO math.
