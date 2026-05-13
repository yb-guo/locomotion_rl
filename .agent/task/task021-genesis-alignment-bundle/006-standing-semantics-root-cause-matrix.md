# Subtask 006: Standing Semantics Root-Cause Matrix

## Route

- Keep this inside task021 and keep it diagnosis-only.
- Add a bounded matrix runner that launches the existing zero-action standing
  probe as one process per scenario.
- Do not run PPO, rendering, downloads, or `GenesisG1SceneBackend`.
- Diagnose in layers:
  - control semantics;
  - gain and force limit;
  - reset pose and root height;
  - asset/contact evidence from metrics and Genesis warnings.

## Feedback Loop

Each scenario emits the normal zero-action standing `summary.json` and
`metrics.jsonl`. The matrix runner aggregates:

- scenario/layer/config;
- first tilt step;
- max/final reset and tilt counts;
- root height and upright metrics;
- joint error/velocity;
- contact count/force;
- status/blocker.

## Ranked Hypotheses

1. If control semantics are the cause, `genesis_position_resend_physics` or
   `custom_pd_torque` will delay or remove the step-64 all-env failure wave.
2. If PD/gain/force limits are the cause, one of the gain profiles will reduce
   reset/tilt counts without changing pose or asset.
3. If reset pose/root height is the cause, `unitree_gym` pose or higher
   `root_z` will improve first failure step or final stability.
4. If none of those layers help, remaining evidence should point toward
   inertial/contact geometry or asset semantics rather than PPO/reward tuning.

## Stop Rules

- Run one variable family at a time; do not create a full Cartesian product.
- If a scenario construction fails, record it and continue to the next
  scenario.
- If a scenario clearly improves stability, rerun that scenario once before
  making a recommendation.
- Do not mark passed without local tests, H200 evidence, and read-only review.

## Log

- 2026-05-12 Created after subtask005 showed tested `RigidOptions` scenarios
  do not remove the zero-action standing failure wave.
- 2026-05-12 Implemented
  `h200_locomotion_lab.tools.g1_standing_semantics_matrix`, a bounded matrix
  runner launching `g1_zero_action_standing_causality` as one subprocess per
  scenario.
- 2026-05-12 Local focused tests:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_standing_semantics_matrix.py tests/test_g1_rigid_options_standing_ablation.py tests/test_vectorized_genesis_backend.py -q -p no:cacheprovider`
  -> 27 passed.
- 2026-05-12 Local expanded tests:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_standing_semantics_matrix.py tests/test_g1_rigid_options_standing_ablation.py tests/test_vectorized_genesis_backend.py tests/test_g1_genesis_alignment_bundle.py tests/test_g1_27dof_nohand_profile.py tests/test_robot_profile_loader.py -q -p no:cacheprovider`
  -> 64 passed after read-only review fixes.
- 2026-05-12 H200 focused tests used the same expanded command -> 64 passed
  after read-only review fixes.
- 2026-05-12 H200 runs all used `CUDA_VISIBLE_DEVICES=1`,
  `physical_gpu=1`, `logical_cuda_device=cuda:0`, `n_envs=512`, `chunks=8`,
  `chunk_steps=32`.
- 2026-05-12 Control layer:
  - `baseline_current`: failed, `first_tilt_step=64`, `max_reset=512`,
    `final_reset=0`, `final_root_height_min=0.636994`,
    `final_upright_mean=0.934044`.
  - `control_resend_physics`: failed, `first_tilt_step=64`,
    `max_reset=512`, `final_reset=0`, `final_root_height_min=0.636994`,
    `final_upright_mean=0.934044`.
  - `control_custom_pd_torque`: failed, `first_tilt_step=64`,
    `max_reset=512`, `final_reset=0`, `final_root_height_min=0.681814`,
    `final_upright_mean=0.955294`.
- 2026-05-12 Gain layer:
  - `gain_unitree_leg`: failed, `first_tilt_step=64`, `max_reset=512`,
    `final_reset=0`, `final_root_height_min=0.742497`,
    `final_upright_mean=0.978485`.
  - `gain_global_kv_2x`: failed, `first_tilt_step=64`, `max_reset=512`,
    `final_reset=0`, `final_root_height_min=0.674267`,
    `final_upright_mean=0.950061`.
  - `gain_kp_half_kv_2x`: failed, `first_tilt_step=64`, `max_reset=512`,
    `final_reset=0`, `final_root_height_min=0.508916`,
    `final_upright_mean=0.890309`.
  - `gain_force_limit_2x`: failed, `first_tilt_step=64`, `max_reset=512`,
    `final_reset=0`, `final_root_height_min=0.636749`,
    `final_upright_mean=0.934155`.
- 2026-05-12 Pose/root layer:
  - `pose_unitree_gym`: failed, `first_tilt_step=32`, `max_reset=512`,
    `final_reset=512`, `final_tilt=512`, `final_root_height_min=0.292207`,
    `final_upright_mean=0.786237`.
  - `root_z_0_90`: failed, `first_tilt_step=64`, `max_reset=512`,
    `final_reset=0`, `final_tilt=0`, `final_root_height_min=0.429583`,
    `final_upright_mean=0.822169`.
  - `root_z_1_00`: failed, `first_tilt_step=64`, `max_reset=512`,
    `final_reset=0`, `final_tilt=0`, `final_root_height_min=0.521273`,
    `final_upright_mean=0.872298`.
  - `root_z_1_10`: failed, `first_tilt_step=64`, `max_reset=512`,
    `final_reset=0`, `final_tilt=0`, `final_root_height_min=0.582886`,
    `final_upright_mean=0.905982`.
- 2026-05-12 Remote summaries:
  - `outputs/task021/standing_semantics_matrix/h200-subtask006-control-20260512-01/summary.json`
  - `outputs/task021/standing_semantics_matrix/h200-subtask006-gain-20260512-01/summary.json`
  - `outputs/task021/standing_semantics_matrix/h200-subtask006-pose-root-20260512-01/summary.json`
- 2026-05-12 Read-only review found one blocking issue: scenario exceptions
  could abort the whole matrix instead of being recorded and continuing. Fixed
  per-scenario exception capture, wrote an error JSON for failed scenarios, and
  ranked empty/error metrics below measured scenarios.
- 2026-05-12 Copied H200 summaries into the local worktree under
  `outputs/task021/standing_semantics_matrix/`.

## Review

Status: passed. Read-only re-review found no blocking findings.

Decision:

- `resend_physics` is not the cause; it matches baseline.
- `custom_pd_torque` improves final height/upright but not first failure step.
- `unitree_leg_gains` is the best single improvement but does not delay/remove
  the failure wave.
- `unitree_gym` pose is worse.
- Raising `root_z` alone does not help.
- Evidence points away from isolated control-call frequency, simple gain/force
  limit changes, and root-z/Unitree-pose fixes. Remaining likely causes:
  combined controller/gain design or asset/inertial/contact geometry.
- Reviewer residual risk: ranking is heuristic and is useful for sorting
  evidence, not causal proof. The conclusion remains bounded to the tested
  isolated fixes.
