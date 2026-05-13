# Subtask 008: Ankle Foot Asset Contact Audit

## Route

- Keep this inside task021 and keep it diagnosis-only.
- Do not run PPO, rendering, downloads, or `GenesisG1SceneBackend`.
- Use prepared G1 assets only.
- Add the smallest standalone audit/probe needed to connect subtask007 onset
  evidence to ankle/foot asset/contact behavior.
- Prefer two feedback loops:
  - a pure XML asset audit for ankle/foot inertial and contact fields;
  - a short H200 link-level onset trace for target link z/contact force.

## Feedback Loop

The subtask passes only if it emits JSON evidence covering:

- ankle/foot link inertial mass and inertia values from the MJCF;
- direct geoms on ankle/foot bodies, including any explicit contact/friction/
  solver attributes;
- Genesis import warnings observed during the link-level trace;
- per-step target link z/contact force around first tilt/reset for baseline and
  `custom_pd_torque + unitree_leg_gains`;
- a decision that either identifies an asset/contact gap, or rules out
  link-level ankle/foot contact as the immediate trigger.

## Ranked Hypotheses

1. If ankle/foot asset geometry seeds the collapse, then XML audit will show
   missing or suspicious contact fields/geoms on ankle/foot bodies and H200
   link trace will show ankle/foot contact/height events before reset.
2. If the Genesis import warning is causal, then the same ankle roll links that
   warn about mass/geometry mismatch will show abnormal z/contact force near
   onset.
3. If ankle/foot contact is not the immediate trigger, link-level trace will
   show root/upright collapse without target ankle/foot force spikes, pushing
   the next task toward active base attitude/height stabilization.

## Stop Rules

- Stop if XML cannot be parsed without Genesis; fix the pure XML audit first.
- Do not infer contact/friction defaults that are absent from MJCF; report them
  as absent.
- If H200 link API cannot resolve target links or force tensors, record that as
  evidence instead of guessing.
- Do not mark passed without local tests, H200 evidence, and read-only review.

## Log

- 2026-05-13 Created after subtask007 showed baseline/custom-PD onset is ankle
  pitch dominated, Unitree gains reduce ankle pitch error, but root/upright
  still collapses with contact spikes and no force saturation.
- 2026-05-13 Implemented
  `h200_locomotion_lab.tools.g1_ankle_foot_asset_contact_audit`, with a pure
  MJCF XML audit path and optional `--run-link-trace` H200 runtime path.
- 2026-05-13 Router fixed two review-before-review issues before accepting H200
  trace evidence:
  - link trace now uses the same `pose_profile=current` semantics as the
    zero-action standing probe;
  - link force/z parsing now handles Genesis-like `(n_envs, n_links, 3)` values
    and selects the requested `link_idx`, after an invalid run showed identical
    force values for all target links.
- 2026-05-13 Local focused tests:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_ankle_foot_asset_contact_audit.py -q -p no:cacheprovider`
  -> 6 passed.
- 2026-05-13 Local expanded tests:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_ankle_foot_asset_contact_audit.py tests/test_g1_failure_onset_trace.py tests/test_g1_standing_semantics_matrix.py tests/test_g1_rigid_options_standing_ablation.py tests/test_vectorized_genesis_backend.py tests/test_g1_genesis_alignment_bundle.py tests/test_g1_27dof_nohand_profile.py tests/test_robot_profile_loader.py -q -p no:cacheprovider`
  -> 76 passed.
- 2026-05-13 H200 expanded tests used the same expanded command through
  `run_guarded.sh` with `CUDA_VISIBLE_DEVICES=1` -> 76 passed.
- 2026-05-13 H200 XML audit:
  `outputs/task021/ankle_foot_asset_contact_audit/h200-subtask008-asset-audit-20260513-01/summary.json`.
  The prepared asset is present. `missing_count=94` because ankle/foot geoms
  lack explicit `friction`, `condim`, `solref`, `solimp`, and `priority`.
  Key ankle/foot asset facts:
  - `left/right_ankle_pitch_link`: `mass=0.074`, two direct mesh geoms; visual
    mesh has `contype=0 conaffinity=0`, collision mesh has no explicit contact
    fields.
  - `left/right_ankle_roll_link`: `mass=0.608`, one visual mesh with
    `contype=0 conaffinity=0`, plus four direct `size=0.005` geoms with no
    explicit contact fields.
  - Left/right symmetry matches for target ankle pitch and ankle roll bodies.
- 2026-05-13 H200 link-level baseline trace:
  `outputs/task021/ankle_foot_asset_contact_audit/h200-subtask008-link-baseline-v3-20260513-01/summary.json`.
  `first_tilt_step=89`; target links resolved to local indices 14/18/15/19.
  `ankle_pitch` links had `contact_force_max=0`, while `ankle_roll` links had
  `contact_force_max=173.049095` left and `172.886348` right with
  `max_contact_env_count=512`. Around onset, step 87 already has all 512 envs
  in ankle-roll contact (`root_height_min=0.416222`,
  `upright_mean=0.382638`), and tilt appears at step 89.
- 2026-05-13 H200 combo link-level trace:
  `outputs/task021/ankle_foot_asset_contact_audit/h200-subtask008-link-combo-20260513-01/summary.json`.
  `first_tilt_step=94`; `ankle_pitch` links again had `contact_force_max=0`.
  `ankle_roll` links had `contact_force_max=405.916624` left and
  `405.811102` right with `max_contact_env_count=512`. Around onset, step 92
  has all 512 envs in ankle-roll contact (`root_height_min=0.390003`,
  `upright_mean=0.359521`), and tilt appears at step 94.

## Review

Status: passed. Read-only review found no blocking findings.

Reviewer residual risks:

- JSON summaries do not yet record exact pose/control/gain CLI settings for
  each link trace; the evidence relies on task log commands and run names.
- Genesis mass/geometry warnings are console evidence rather than structured
  JSON artifacts.
- The XML audit focuses on direct target-body inertial/geoms and explicit attrs;
  inherited MJCF defaults and broader child-body contact semantics need a
  follow-up before making an asset patch.

Decision:

- The immediate contact path is the ankle-roll link, not ankle-pitch contact:
  ankle-pitch link force is zero in both baseline and combo traces.
- The asset has a plausible contact/inertial mismatch: each ankle-roll link is
  `0.608kg`, Genesis warns that this mass is dubious compared with geometry,
  and the only active ankle-roll collision support is four tiny `size=0.005`
  point-like geoms with no explicit friction/contact solver attributes.
- The next implementation task should test a controlled asset-contact patch
  under project-local generated XML only: enlarge/replace ankle-roll collision
  support or add explicit foot contact/friction fields, then rerun the same
  zero-action/link trace before touching PPO.
