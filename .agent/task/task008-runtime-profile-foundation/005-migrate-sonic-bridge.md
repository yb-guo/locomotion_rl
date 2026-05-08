# Route

Task: task008-runtime-profile-foundation

Goal: migrate existing SONIC G1 bridge users to profile-backed action/control
data without breaking task006/task007.

Pass condition:

- Existing `g1_policy_bridge.py` either delegates to the loaded profile path or
  is explicitly retained only as compatibility/test reference.
- Genesis SONIC motor config uses the loaded control profile.
- Task007 dry-run uses the loaded control profile.
- Existing tests are updated to compare against profile-backed behavior.

Fail condition:

- Two runtime authorities for default angles/action scales remain active.
- Migration changes task006 action semantics without evidence.

# Log

- 2026-05-08: Pending scalar/tensor bridge validation.

# Review

Status: pending.

