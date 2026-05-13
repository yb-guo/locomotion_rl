# Subtask 003: Review And Decision

## Route

- Run read-only review after local and H200 evidence.
- Fix blocking findings before recording a decision.
- Decide whether the alignment bundle is sufficient for standing PPO follow-up
  or whether another environment semantics task is needed.

## Log

- 2026-05-12 Initial read-only review found one blocking issue: timing was not
  explicitly compared or marked missing for SONIC/MJCF sources. Fixed in
  subtask001 and re-reviewed with no blocking findings.
- 2026-05-12 Final read-only review after H200 asset evidence found no
  blocking findings. Reviewer confirmed:
  - H200 copied report is valid JSON;
  - `xml_asset_present=true`;
  - mapped 29DoF to 27DoF control arrays match for `default_angles_rad`,
    `action_scales_rad`, `kp`, `kv`, and `force_limits`;
  - removed joints are `waist_roll_joint` and `waist_pitch_joint`;
  - tool import does not pull in `genesis` or `torch`;
  - timing and missing contact/timing semantics are represented explicitly.

## Decision

Alignment bundle is complete for the current diagnosis step, but it is not a
complete environment-semantics alignment.

Findings:

- Control profile alignment passes: 29DoF SONIC mapped to 27DoF no-hand matches
  default angles, action scales, PD gains, and force limits.
- Genesis training timing is explicit and self-consistent:
  `sim_dt_s=0.005`, `decimation=4`, `policy_rate_hz=50`.
- SONIC 29DoF robot profile does not encode timing, so timing parity cannot be
  proven from that profile alone.
- Prepared H200 MJCF is available and parseable, but it lacks explicit
  `<option>`, `<contact>`, and geom-level `friction/condim/solref/solimp/
  priority` fields. The report only finds compiler metadata and default joint
  armature/damping/frictionloss groups.
- Current 27DoF Genesis training profile and `VectorizedGenesisBackend` do not
  expose a structured contact/friction/solver config.

Next recommended task:

- Add a contact/solver semantics task that either:
  - builds a structured contact/friction/solver profile from an asset with
    explicit fields; or
  - adds backend-level reporting/override support for Genesis contact/solver
    defaults before more PPO tuning.

## Review

Status: passed. No blocking findings remain.
