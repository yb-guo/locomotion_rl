# Task 052: Fixed-Stage Fault Curriculum

## Route

Run a fixed-stage hidden-fault curriculum before returning to the hard Task050
dead-motor gate. The first requested slice is Stage2:

- fixed command `vx=1.3 m/s`;
- hidden `left_knee_joint` motor scale `0.3`;
- dynamic onset at `1.0 s`, no recovery during the rollout;
- actor observation keeps the hidden-fault boundary: no explicit fault identity,
  scale, onset, or recovery label is visible to the actor;
- privileged fault labels may be used only for auxiliary memory-latent training
  diagnostics.

Source checkpoint:

- first preference:
  `outputs/task049/bridge_smoke_per_step_mlp_replay/task049_bridge_smoke_per_step_mlp_replay_env512_step24_iter10_mb1_seed4900107/model_9.pt`;
- do not treat the failed Task051 hard-mode checkpoint as a recovery prior
  unless a later subtask explicitly asks for that ablation.

## Planned Slices

1. `001-stage2-left-knee-partial-train.md`
   - Register and train
     `Unitree-G1-Gripper-Flat-Task052-TrueTxl-CurriculumStage2-Train`.
   - This is a train-pipeline and checkpoint-producing gate only.

2. `002-stage2-eval.md`
   - Run stage-matched Task050-style continuous and retry eval with
     `--dynamic-dead-scale 0.3`.
   - Optionally run the hard Task050 `scale=0.0` gate as a regression check,
     but do not require it to pass before earlier curriculum stages exist.

## Acceptance Criteria

- Task052 train task registration exists and uses
  `Task044TrueTxlMemoryK160ClearHistoryRunner`.
- Stage2 training JSON exists, records `env_cfg_mode=train`, and passes strict
  sequence replay parity.
- Actor-visible observation does not include explicit fault labels.
- Stage-matched continuous eval JSON exists and records physical continuity and
  post-fault quality gates.
- Stage-matched retry eval JSON exists and records the final-trial promotion
  gate.
- No hard dead-motor or all-joint damaged-joint claim is made from Stage2 alone.
- No external checkpoint, dataset, asset, or upstream repository is downloaded.

## Log

- 2026-08-13 Opened at user request to move from Task051 hard-mode direct
  training to fixed-stage curriculum. Interpreting "fixed stage 2" as the
  previously proposed Stage2: `vx=1.3 m/s`, left-knee motor scale `0.3`, onset
  `1.0 s`.
- 2026-08-13 Added Stage2 train registration
  `Unitree-G1-Gripper-Flat-Task052-TrueTxl-CurriculumStage2-Train` and wrapper
  `h200_locomotion_lab.tools.task052_fault_curriculum_train`. Also extended the
  Task050-style eval CLIs with `--dynamic-dead-scale` so stage-matched partial
  failures can be evaluated without changing the hard dead-motor default.
- 2026-08-13 Trained Stage2 from the Task049 clean checkpoint for 20 iterations,
  then continued Stage2 for 180 more iterations. Latest checkpoint:
  `outputs/task052/curriculum_stage2/task052_stage2_left_knee_partial_continue_env512_step24_iter180_mb1_seed5200102/model_179.pt`.
  Training JSON:
  `task052_stage2_left_knee_partial_continue_env512_step24_iter180_mb1_seed5200102.json`.
  It reports `train_pipeline_pass=true`, `task052_train_pipeline_pass=true`,
  runner `Task044TrueTxlMemoryK160ClearHistoryRunner`, algorithm
  `Task040SequenceAwareTrueTxlPPO`, actor `Task038TrueTxlMemoryModel`, strict
  logprob replay parity pass with max logprob and ratio error `0.0`, and
  `task044_fault_aux_last_loss=0.04792369529604912`.
- 2026-08-13 Stage-matched eval with `vx=1.3`, scale `0.3`, onset `1.0s`
  passed both gates. Continuous JSON:
  `outputs/task052/eval_stage2/task052_model179_stage2_left_knee_scale0p3_continuous_seed5200201.json`
  reports `pass=true`, `physical_continuity_pass=true`,
  `physical_reset_events=0`, `physical_fall_events=0`, post-fault
  `fall_ratio=0.0`, mean linear velocity error `0.097776859998703`, max
  `gravity_xy=0.05395696312189102`, and min `root_z=0.7685365676879883`.
  Retry JSON:
  `outputs/task052/eval_stage2/task052_model179_stage2_left_knee_scale0p3_retry_seed5200301.json`
  reports `pass=true`, `final_trial_pass=true`, aggregate `fall_count=0`, and
  final-trial `fall_count=0`.
- 2026-08-13 Optional hard Task050 migration check with `vx=1.6`, scale `0.0`,
  onset `0.5s` failed both gates. Continuous JSON:
  `outputs/task052/eval_hard_task050/task052_model179_hard_left_knee_scale0p0_vx1p6_continuous_seed5200401.json`
  reports `pass=false`, `physical_continuity_pass=false`,
  `physical_reset_events=1034`, `physical_fall_events=1034`, post-fault
  `fall_ratio=1.0`, mean linear velocity error `0.8940730094909668`, max
  `gravity_xy=0.9500682950019836`, min `root_z=0.1703503578901291`, and first
  fall across all envs with mean time `1.82625s`. Retry JSON:
  `outputs/task052/eval_hard_task050/task052_model179_hard_left_knee_scale0p0_vx1p6_retry_seed5200501.json`
  reports `pass=false`, `final_trial_pass=false`, aggregate
  `fall_count=1001`, aggregate `fall_ratio=0.9775390625`, and final-trial
  `fall_count=256`.

## Review

Status: active. Stage2 partial-fault recovery is verified. Hard Task050
dead-motor recovery is not verified; Stage2 alone does not support a hard
dead-motor or all-joint damaged-joint robustness claim.
