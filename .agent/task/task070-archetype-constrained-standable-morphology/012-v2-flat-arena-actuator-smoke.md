# 012 — V2 Flat-Arena Actuator Smoke

## Route

1. Compile each witness into its self-contained MuJoCo MJCF with the existing
   20 x 20 flat floor, gravity, free base, contact terminals, and exact actuator
   accounting.
2. From a contact-aware reset, independently pulse every position actuator and
   continuous wheel motor; require finite state, no solver warning, and
   measurable response in the commanded joint coordinate.
3. Separately run the documented controller-assisted stance for 1000 x 2 ms and
   record contact, support, pose, velocity, effort, and solver gates. Do not fold
   actuator response into stance success.
4. Report `walking_claimed=false`: this smoke verifies wiring and local response,
   not a gait policy, dynamic locomotion, or sim2real behavior.

## Log

- 2026-08-25：preflight confirmed that current v2 XML already contains floor,
  gravity, and a free base, but attempt003/005 only claimed compile/visual
  evidence. A no-terminal-contact path in the stance diagnostic exposed a
  `max(0.1, *empty)` TypeError; the empty-contact diagnostic was corrected
  before recording arena evidence.
- Attempt006 arena evidence:
  `artifacts/arena_task070_v2_attempt006/flat_arena_smoke.json` (SHA-256
  `c460eef106c024fdf026e171ebfb403b56439dc0ba94365cc355c95ce8206f3a`).
  The run covers the previous five centers in non-wheel/wheel compositions
  (10 cases) plus all eight additional humanoid candidates (18 total).
- Exact result: `18/18` compiled, `18/18` actuator accounting exact,
  `18/18` contact-aware reset passed, and `18/18` every actuator produced a
  measurable target-joint response relative to an identical-reset nominal-hold
  baseline, with finite state and no solver warning. X1 initially
  exposed unequal support-terminal height; its display nominal was corrected
  and the full run repeated.
- `0/18` passed the 1000-step stance gate under the generic nominal
  joint-hold controller. No candidate-specific balance controller or
  locomotion policy is present, so the evidence explicitly records
  `walking_claimed=false`.
- The execution agent opened representative snapshots and the complete
  `flat_arena_gallery.png` (SHA-256
  `5d5141194e7cb3687bf02a68fcb7bf6fbc6e9e2642943bc44a3c4775f0718945`).

## Review

- Flat-arena compile/reset/actuator-response smoke: **passed for 18/18 cases**.
- Stance/locomotion claim: **not passed** (`0/18` generic stance holds;
  no gait policy). “Motors respond” must not be reported as “the robot walks.”
- Artifact visual check is true, while `user_visual_acceptance=false` and
  `counts_toward_task070_v2_pass=false`. This microtask does not mark
  Task070 v2 passed.
