# 003: H200 Baseline Negative Gate

## Route

Run Task044's triplet gate against the existing `model_5349` True-TXL bridge.
The expected result is failure because Task043 already showed normal,
zero-residual, and stateless-memory behavior are tied.

If this baseline passes Task044, the target is invalid and must be tightened
before any new training.

## Acceptance

- H200 JSON summary path is recorded.
- The summary uses the same checkpoint, seed, speed, and fault schedule across
  normal, zero-residual, and stateless inputs.
- Expected result is a negative gate with reason such as
  `zero_residual_ablation_not_degraded` or
  `stateless_memory_ablation_not_degraded`.
- No training success or memory-causality success is claimed from this baseline.

## Log

- 2026-05-31 Planned after 002.
- 2026-05-31 Synced the Task044 contract and triplet-summary CLI to H200 source
  under
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/src`.
- 2026-05-31 Ran the `model_5349` bridge triplet summary on existing Task043
  eval JSONs. Output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/model5349_bridge_task044_baseline_negative_seed4301701.json`.
  Result: `task044_memory_required_pass=false`.
  Failure reasons: `zero_residual_ablation_not_degraded`,
  `stateless_memory_ablation_not_degraded`.

## Review

Status: passed.

This is the desired negative baseline. Normal mode passes the underlying
quality gate, but ablations remain tied: zero-residual final linear velocity
error is only `0.00015547871589660645` worse than normal, and stateless is
slightly better by `0.0002522468566894531`. Task044 therefore correctly rejects
the current bridge as memory-required evidence.
