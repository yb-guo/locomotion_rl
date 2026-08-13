# Task 053: Sampled Fault Curriculum

## Route

Replace manual fixed stages with one sampled hidden-fault curriculum task. Each
env samples a left-knee failure schedule at reset, and the sampling distribution
hardens with `common_step_counter`:

- target joint: hidden `left_knee_joint`;
- sampled motor scale: early range `0.30-0.60`, final range `0.0-0.30`;
- sampled dead-motor probability: early `0.0`, final `0.35`;
- sampled onset: early `0.90-1.40s`, final `0.50-0.90s`;
- sampled command: `vx=1.3` or `1.6`, with hard-speed probability increasing
  from `0.10` to `0.70`;
- curriculum progress reaches the final hard distribution after `4320` env
  steps, matching `rollout_steps=24` and `iterations=180`;
- actor observation keeps the hidden-fault boundary: no explicit fault identity,
  scale, onset, or recovery label is visible to the actor;
- privileged fault labels may be used only for auxiliary memory-latent training
  diagnostics.

Source checkpoint:

- `outputs/task052/curriculum_stage2/task052_stage2_left_knee_partial_continue_env512_step24_iter180_mb1_seed5200102/model_179.pt`
  is the preferred warm start because it already passes Stage2 partial-fault
  recovery.

## Planned Slices

1. `001-sampled-curriculum-train-smoke.md`
   - Register and smoke-train
     `Unitree-G1-Gripper-Flat-Task053-TrueTxl-SampledCurriculum-Train`.
   - This is a train-pipeline and checkpoint-producing gate only.

2. `002-hard-task050-eval.md`
   - Evaluate the sampled-curriculum checkpoint with hard Task050 continuous and
     retry gates: `vx=1.6`, scale `0.0`, onset `0.5s`.

## Acceptance Criteria

- Task053 train task registration exists and uses
  `Task044TrueTxlMemoryK160ClearHistoryRunner`.
- Stage-free sampled curriculum records train JSON with `env_cfg_mode=train`
  and strict sequence replay parity.
- Actor-visible observation does not include explicit fault labels.
- Hard Task050 continuous eval JSON exists and passes physical continuity plus
  post-fault quality gates.
- Hard Task050 retry eval JSON exists and passes the final-trial promotion gate.
- No all-joint damaged-joint claim is made from the left-knee-only target.
- No external checkpoint, dataset, asset, or upstream repository is downloaded.

## Log

- 2026-08-13 Opened at user request to avoid manual Stage3/4/5 switching and
  instead use one progressively sampled curriculum during training.
- 2026-08-13 Added Task053 sampled curriculum registration, train wrapper, and
  focused tests. Smoke-trained 20 iterations from Task052 Stage2 checkpoint:
  `.agent/task/task053-sampled-fault-curriculum/task053_sampled_curriculum_env512_step24_iter20_mb1_seed5300101.json`.
- 2026-08-13 Adjusted default curriculum hardening horizon from `12000` to
  `4320` env steps so a 180-iteration run reaches the hard distribution.
- 2026-08-13 Full 180-iteration sampled-curriculum train completed with
  `train_pipeline_pass=true`, but hard Task050 continuous and retry eval both
  failed. Continuous had `physical_reset_events=768` and post-fault
  `fall_ratio=1.0`; retry had `final_trial_pass=false` and final-trial
  `fall_count=256`.

## Review

Status: closed with negative hard-recovery evidence.

Task053 established the one-shot sampled curriculum pipeline and produced a
candidate checkpoint. The checkpoint is not accepted as hard hidden left-knee
dead-motor recovery because both required Task050 gates failed. No all-joint
damaged-joint claim is made.
