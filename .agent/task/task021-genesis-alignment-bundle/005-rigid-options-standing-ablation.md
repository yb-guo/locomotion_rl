# Subtask 005: Rigid Options Standing Ablation

## Route

- Add an independent task021 probe for zero-action standing diagnostics across
  rigid contact/solver scenarios.
- Reuse `g1_zero_action_standing_causality` helper functions for pose/gain
  setup, warmup/chunk execution, metric rows, summaries, and JSON writing.
- Keep the probe local to `VectorizedGenesisBackend` and
  `VectorizedGenesisConfig.rigid_contact_solver`.
- Do not touch `GenesisG1SceneBackend`, download assets, render, or run PPO.

## Log

- 2026-05-12 Implemented
  `h200_locomotion_lab.tools.g1_rigid_options_standing_ablation`, runnable via
  `python -m h200_locomotion_lab.tools.g1_rigid_options_standing_ablation`.
- 2026-05-12 Default scenarios:
  `default_unset`, `newton_solver_only`, `newton_mujoco_contact`, and
  `newton_solver_bundle`.
- 2026-05-12 The tool writes a run-level `summary.json` plus per-scenario
  `config.json`, `metrics.jsonl`, and `summary.json` for successful scenarios.
  Scenario exceptions are captured as `status=error` with a blocker string, and
  later scenarios continue.
- 2026-05-12 Added fake-only local tests for scenario payloads, passing
  `GenesisRigidContactSolverConfig` into `VectorizedGenesisConfig`, and
  continuing after a scenario error.
- 2026-05-12 Focused local command:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_rigid_options_standing_ablation.py tests/test_vectorized_genesis_backend.py -q -p no:cacheprovider`
  -> 19 passed.
- 2026-05-12 Router fixed cross-task helper compatibility with the H200 copy of
  `g1_zero_action_standing_causality` (`trace_path`, `contact_trace`,
  `trace_env_index`, `leg_pose_values`) and promoted contact count/force into
  scenario key metrics.
- 2026-05-12 First H200 multi-scenario run exposed a real Genesis lifecycle
  boundary: repeated backend construction in one process can raise
  `Genesis already initialized`. `VectorizedGenesisBackend._init_genesis()` now
  treats that exact message as reuse of an already initialized Genesis runtime;
  other init errors still raise.
- 2026-05-12 Expanded local related tests:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_rigid_options_standing_ablation.py tests/test_vectorized_genesis_backend.py tests/test_g1_genesis_alignment_bundle.py tests/test_g1_27dof_nohand_profile.py tests/test_robot_profile_loader.py -q -p no:cacheprovider`
  -> 57 passed.
- 2026-05-12 H200 focused tests passed 57/57 with guarded execution.
- 2026-05-12 H200 single-scenario ablations completed with
  `CUDA_VISIBLE_DEVICES=1`, `physical_gpu=1`, `logical_cuda_device=cuda:0`,
  `n_envs=512`, `chunks=8`, `chunk_steps=32`.

  | scenario | status | first_tilt_step | max_reset | final_reset | final_tilt | final_root_height_min | final_upright_mean | max_contact_force |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `default_unset` | failed | 64 | 512 | 0 | 0 | 0.636994 | 0.934044 | 173.049103 |
  | `newton_solver_only` | failed | 64 | 512 | 0 | 0 | 0.636768 | 0.934142 | 173.058075 |
  | `newton_mujoco_contact` | failed | 64 | 512 | 0 | 0 | 0.640997 | 0.936020 | 173.035492 |
  | `newton_solver_bundle` | failed | 64 | 512 | 512 | 512 | 0.354421 | 0.785159 | 172.949158 |

  Local copies of H200 summaries are under
  `outputs/task021/rigid_options_standing_ablation/h200-subtask005-single-*`.

## Decision

- `constraint_solver="Newton"` alone does not materially change the failure
  signature relative to default: same first tilt step, same all-env reset wave,
  and nearly identical contact force.
- Adding `enable_mujoco_compatibility=True` and `enable_multi_contact=True`
  gives a very small height/upright improvement but does not remove the failure
  wave.
- The stronger solver bundle is worse in this probe: all 512 envs are still in
  tilt/reset on the final chunk and root height collapses farther.
- The standing failure should not be treated as solved by the exposed
  `RigidOptions` surface. The next diagnosis should move to pose/gain/control
  semantics or asset inertial/contact geometry.

## Review

Status: passed. Read-only review found no blocking findings.

Notes:

- No Genesis import is required by the new tests.
- H200 execution completed without PPO, rendering, downloads, or
  `GenesisG1SceneBackend` changes.
- Reviewer residual risk: one-scenario-per-process H200 evidence is sufficient
  for this ablation, but long multi-scenario same-process SSH runs remain less
  proven; the result excludes this tested rigid-options surface, not pose,
  gains, inertial parameters, or contact geometry.
