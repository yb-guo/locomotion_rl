# 005: Reset/Contact State Follow-Up

## Goal

Only if control, pose, and targeted gain/force do not explain the failure,
inspect reset-state and contact/asset dynamics.

## Route

1. Confirm subtasks 003 and 004 permit this route.
2. Test root-z, hard velocity zeroing, settle-before-eval, and target
   reapplication one variable at a time.
3. Record foot contact, body contact, force saturation, joint error, and root
   trajectory evidence.
4. Decide whether the next task must inspect asset/contact/inertia details.

## Log

- 2026-05-11 Six-case gate permits later reset/contact diagnostics if targeted
  gain/force follow-up does not produce a stable candidate.
- Current evidence still shows tilt/fall resets with hard-height termination
  counts at zero.
- 2026-05-11 Subtask004 permits this route. Several gain profiles recover to
  stable final chunks after an early full-env tilt/reset wave, while
  `force_limit_2x` matches baseline and force saturation is zero. The next
  boundary is reset/contact settling, not force limit and not PPO.
- 2026-05-11 Added standalone `--warmup-policy-steps` support to run
  termination-disabled zero-action settle steps before evaluated chunks.
  Warmup records tilt, termination-height, root/upright, joint RMS, and force
  saturation diagnostics without calling backend reset. Warmup-clean evaluation
  reports `evaluation_passed=true` but `diagnostic_passed=false`.
- 2026-05-11 Added `--pre-eval-reset` for the post-warmup hard-reset probe.
  When enabled, the probe records warmup diagnostics first, then calls one
  full `backend.reset()` before evaluated chunks. Clean evaluation after this
  reset remains diagnostic-only and does not set strict `passed=true`.
- 2026-05-11 Added `--pre-eval-reset-scope` with `full` and `all_env_ids`.
  The `all_env_ids` mode uses a logical-device torch tensor of every env id
  before evaluation so the probe can directly compare full reset against the
  selected-env reset path used by evaluated chunk recovery.
- Verification: `$env:PYTHONPATH='src'; python -m pytest
  tests\test_g1_zero_action_standing_causality.py -q -p no:cacheprovider`
  passed with 17 passed, 4 skipped.
- 2026-05-11 H200 focused tests passed on physical GPU 1 / logical
  `cuda:0`: `44 passed in 3.97s`.
- 2026-05-11 H200 warmup-only sweep ran with `pose_profile=current`,
  `root_z=1.20`, `control_mode=genesis_position`,
  `gain_profile=unitree_leg_gains`, `n_envs=1024`, `chunks=50`,
  `chunk_steps=32`, `seed=0`.

| Warmup policy steps | Warmup tilt | Warmup term-height | Warmup root min | First eval tilt chunk | Max eval reset | Final eval reset | Final eval tilt | Final root mean | Evaluation pass |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 0 | 0 | 0.784 | 2 | 1024 | 0 | 0 | 0.787 | no |
| 32 | 0 | 0 | 0.784 | 2 | 1024 | 0 | 0 | 0.763 | no |
| 64 | 0 | 0 | 0.775 | 1 | 1024 | 1024 | 1024 | 0.771 | no |
| 96 | 0 | 0 | 0.339 | 0 | 1024 | 0 | 0 | 0.790 | no |
| 128 | 32768 | 30720 | 0.085 | 0 | 1024 | 0 | 0 | 0.790 | no |
| 256 | 163840 | 161792 | 0.085 | 0 | 1024 | 0 | 0 | 0.790 | no |

- 2026-05-11 Warmup without reset does not make the evaluated window clean.
  Long warmups show real fall during warmup, not just a false reset gate. The
  final evaluated chunks recover after reset, so the next boundary is whether
  a post-warmup hard reset before evaluation stabilizes the first evaluated
  episode.
- 2026-05-11 H200 post-warmup full-reset sweep ran with `--pre-eval-reset`.
  All evaluated windows still failed with `max_reset_count=1024`.

| Gain profile | Warmup steps | Warmup tilt | Warmup term-height | First eval tilt chunk | Max eval reset | Final eval reset | Final eval tilt | Final root mean | Evaluation pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `unitree_leg_gains` | 0 | 0 | 0 | 3 | 1024 | 0 | 0 | 0.790 | no |
| `unitree_leg_gains` | 16 | 0 | 0 | 3 | 1024 | 0 | 0 | 0.790 | no |
| `unitree_leg_gains` | 64 | 0 | 0 | 3 | 1024 | 0 | 0 | 0.790 | no |
| `unitree_leg_gains` | 128 | 32768 | 30720 | 3 | 1024 | 0 | 0 | 0.790 | no |
| `global_kv_4x` | 128 | 33792 | 30720 | 2 | 1024 | 0 | 0 | 0.786 | no |
| `global_kp_0_5x_kv_2x` | 128 | 38912 | 35840 | 2 | 1024 | 0 | 0 | 0.788 | no |

- 2026-05-11 Full pre-eval reset does not explain the recovery. The remaining
  reset-path difference is `backend.reset(None)` for full reset versus
  `backend.reset(done_env_ids)` for selected-env reset inside evaluated
  chunks. Next probe should reset all envs through the selected-env path before
  evaluation.
- 2026-05-11 H200 selected-env pre-eval reset sweep ran with
  `--pre-eval-reset-scope all_env_ids`. It matched full pre-eval reset: no
  evaluated window passed.

| Gain profile | Warmup steps | First eval tilt chunk | Max eval reset | Final eval reset | Final root mean | Evaluation pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `unitree_leg_gains` | 0 | 3 | 1024 | 0 | 0.790 | no |
| `unitree_leg_gains` | 16 | 3 | 1024 | 0 | 0.790 | no |
| `unitree_leg_gains` | 64 | 3 | 1024 | 0 | 0.790 | no |
| `unitree_leg_gains` | 128 | 3 | 1024 | 0 | 0.790 | no |
| `global_kv_4x` | 128 | 2 | 1024 | 0 | 0.786 | no |
| `global_kp_0_5x_kv_2x` | 128 | 2 | 1024 | 0 | 0.788 | no |

- 2026-05-11 Pulled representative H200 `metrics.jsonl` rows locally for
  inspection. For `unitree_leg_gains`, bad chunks repeat every three chunks:
  chunk `3, 6, 9, ... 48` all have `reset_count=1024` and
  `tilt_bad_count=1024`. This means the earlier `final_reset_count=0` rows were
  misleading because the final chunk happened to be between periodic falls.
- 2026-05-11 H200 current-pose root-z sweep with `unitree_leg_gains` ran at
  `root_z=0.76, 0.78, 0.80, 0.84, 0.90, 1.00`. No case passed; all had
  `max_reset_count=1024`.
- 2026-05-11 H200 `unitree_gym` pose root-z sweep with `unitree_leg_gains` ran
  at `root_z=0.70, 0.74, 0.78, 0.80, 0.84, 0.90`. No case passed; all had
  `max_reset_count=1024`.

## Review

Status: reset/contact follow-up complete; no strict pass.

- Candidate follow-up should keep the best stable gain profiles fixed while
  changing one reset/contact variable at a time.
- The first probes should compare immediate evaluation against
  settle-before-eval or termination-disabled warmup, because the symptom is an
  early reset wave followed by stable final chunks for some gain profiles.
- Evidence now shows periodic real falls, not a one-time reset artifact:
  warmup-only, full pre-eval reset, selected-env pre-eval reset, current-pose
  root-z sweep, and `unitree_gym` pose root-z sweep all fail.
- Decision: reset semantics and root-z alone are not the causal fix. The next
  task should add an explicit standing-pose micro-sweep or inspect
  asset/contact/inertia details, because the current two pose profiles do not
  provide a stable zero-action equilibrium.
