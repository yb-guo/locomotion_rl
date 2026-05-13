# Subtask 007: Failure Onset Trace And Combo Confirmation

## Route

- Keep this inside task021 and keep it diagnosis-only.
- Do not run PPO, rendering, downloads, or `GenesisG1SceneBackend`.
- Use the existing zero-action standing probe as the feedback loop.
- Run one-process-per-scenario on H200 with `CUDA_VISIBLE_DEVICES=1`.
- Compare:
  - current baseline;
  - best single control/gain improvements from subtask006;
  - combined `custom_pd_torque + unitree_leg_gains`;
  - one confirm rerun if a scenario delays or removes failure.

## Feedback Loop

Run short-horizon traces at 1-policy-step metric granularity around failure
onset. Aggregate:

- first tilt/reset step;
- root height and upright trajectory before failure;
- top joint position-error and velocity contributors;
- force saturation;
- contact count and max contact force;
- whether combining the best control/gain changes delays the step-64 failure.

## Ranked Hypotheses

1. If the current failure is a combined controller/gain issue, then
   `custom_pd_torque + unitree_leg_gains` will delay or remove the first
   tilt/reset wave more than either single change.
2. If a specific joint group seeds the fall, then 1-step trace metrics will
   show the same top joint errors or velocities before the first tilt event
   across baseline and improved scenarios.
3. If contact/inertial geometry seeds the fall, then contact force/count or
   root/upright drift will diverge before joint error/velocity changes explain
   the failure.

## Stop Rules

- Stop expanding the matrix if the combined controller/gain scenario still
  fails at the same first step as baseline.
- If a scenario improves first failure step, rerun it once before treating it
  as evidence.
- If top-joint/contact evidence is not present in summaries, add only the
  smallest reporting wrapper needed; do not refactor old tools.
- Do not mark passed without local tests, H200 evidence, and read-only review.

## Log

- 2026-05-13 Created after subtask006 showed isolated control-call frequency,
  simple gains, force limit, Unitree pose, and root-z changes do not remove the
  zero-action standing failure wave.
- 2026-05-13 Implemented
  `h200_locomotion_lab.tools.g1_failure_onset_trace`, a one-process-per-scenario
  wrapper around the existing zero-action standing probe. It defaults to
  `chunk_steps=1`, `chunks=96`, and scenarios `baseline_current`,
  `control_custom_pd_torque`, `gain_unitree_leg`, and
  `combo_custom_pd_unitree_leg`.
- 2026-05-13 Local focused tests:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_failure_onset_trace.py tests/test_g1_standing_semantics_matrix.py tests/test_g1_rigid_options_standing_ablation.py tests/test_vectorized_genesis_backend.py -q -p no:cacheprovider`
  -> 35 passed.
- 2026-05-13 Local expanded tests:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_failure_onset_trace.py tests/test_g1_standing_semantics_matrix.py tests/test_g1_rigid_options_standing_ablation.py tests/test_vectorized_genesis_backend.py tests/test_g1_genesis_alignment_bundle.py tests/test_g1_27dof_nohand_profile.py tests/test_robot_profile_loader.py -q -p no:cacheprovider`
  -> 70 passed.
- 2026-05-13 H200 focused and expanded tests used the same expanded command
  through `run_guarded.sh` with `CUDA_VISIBLE_DEVICES=1` -> 70 passed.
- 2026-05-13 H200 trace run:
  `outputs/task021/failure_onset_trace/h200-subtask007-onset-20260513-01/summary.json`.
  All scenarios used `physical_gpu=1`, `logical_cuda_device=cuda:0`,
  `n_envs=512`, `chunks=96`, `chunk_steps=1`.
  - `baseline_current`: failed, `first_tilt_step=88`, `max_reset=512`.
    First tilt row: `root_height_min=0.335023`, `upright_mean=0.267985`,
    `foot_or_body_contact_count=412`, `max_contact_force=32.416695`.
    Top joint position errors were both ankle pitch joints
    (`left=0.328898`, `right=0.328842` RMS).
  - `control_custom_pd_torque`: failed, `first_tilt_step=89`,
    `max_reset=512`. First tilt row top joint errors remained both ankle pitch
    joints (`left=0.328687`, `right=0.328629` RMS).
  - `gain_unitree_leg`: failed, `first_tilt_step=92`, `max_reset=512`.
    First tilt row ankle pitch error dropped to `0.032187` RMS; top errors
    moved to shoulder joints, but root/upright still fell through the same
    termination path.
  - `combo_custom_pd_unitree_leg`: failed, `first_tilt_step=93`,
    `max_reset=512`. First tilt row: `root_height_min=0.300558`,
    `upright_mean=0.256029`, `foot_or_body_contact_count=444`,
    `max_contact_force=107.707977`; max contact force over the run remained
    about `172.954`.
- 2026-05-13 H200 combo confirm rerun:
  `outputs/task021/failure_onset_trace/h200-subtask007-combo-confirm-20260513-01/summary.json`.
  The combo result repeated `first_tilt_step=93`, `max_reset=512`,
  `root_height_min=0.300557`, `upright_mean=0.256028`,
  `foot_or_body_contact_count=458`, and `max_contact_force=107.704498` at
  first tilt. `force_saturation_ratio=0.0`.
- 2026-05-13 The old subtask006 `first_tilt_step=64` was a coarse lower bound
  from `chunk_steps=32`. Subtask007 shows the precise one-step onset is
  baseline step 88, and the best combined controller/gain change only delays
  failure to step 93.

## Review

Status: passed. Read-only review found no blocking findings.

Reviewer residual risks:

- Step naming follows the existing probe convention: `first_tilt_step` is the
  chunk start index, while the focus row's `total_policy_steps` is one greater
  for `chunk_steps=1`.
- Baseline was not rerun in the confirm set, so the 5-policy-step delay depends
  on the single baseline trace plus repeated combo trace.
- Copied local evidence includes scenario/focus summaries. Full row-by-row
  `metrics.jsonl` remains in the remote run directories.

Decision:

- Combining `custom_pd_torque` with `unitree_leg_gains` helps but does not
  solve standing. It reproducibly delays onset by 5 policy steps.
- The failure is not force saturation: `force_saturation_ratio=0.0` in the
  confirm run.
- Baseline and custom-PD onset are dominated by ankle pitch error. Unitree gains
  reduce that error substantially, but root height/upright still collapse and
  contact spikes remain. The next diagnosis should inspect/repair asset
  inertial/contact geometry around the ankle/foot chain or add a controller that
  actively regulates base attitude/height, rather than only changing reset
  thresholds.
