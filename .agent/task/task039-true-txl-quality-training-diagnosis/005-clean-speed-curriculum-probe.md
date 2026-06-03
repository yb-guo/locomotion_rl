# 005 Clean Speed Curriculum Probe

## Route

If clean 0.4 m/s gait shows improvement, test whether the same route can extend
toward 1.2 and 2.0 m/s through a small clean-speed curriculum.

This is still train-variant quality diagnosis, not held-out morphology
adaptation.

## Minimal Closed Loop

Close this slice with:

- curriculum config or launch command;
- H200 training/eval JSON for at least two speeds;
- comparison against the fixed-speed checkpoint when available;
- failure classification for speed expansion.

## Evidence Gate

Evidence must record:

- speed stages;
- stage durations or iteration counts;
- checkpoint paths;
- eval JSON paths;
- root z, fall ratio, gravity xy, velocity error by speed;
- whether each speed is pipeline pass and quality gate pass.

## Subagent Ownership

Worker owns speed curriculum config/scripts/docs for this slice only. Worker
must not alter morphology randomization or true-TXL update internals here.

Reviewer checks that speed improvement is not inferred from aggregate reward
alone and that 2.0 m/s failure does not erase a valid 0.4/1.2 diagnosis.

## Failure Exit

If 0.4 m/s clean gait does not improve first, do not run this slice. Route back
to MLP/TXL clean-gait or memory-update diagnosis.

## Log

- 2026-05-30 Opened as a gated follow-up after clean-gait improvement.
- 2026-05-30 Gated not run. Task039 clean train-variant quality did not improve
  for the current MLP or true-TXL routes:
  - `002` MLP diagnostic has `pipeline_pass=true`,
    `quality_gate_pass=false`, `pass=false`;
  - `003` true-TXL diagnostic has `pipeline_pass=false`,
    `quality_gate_pass=false`, `pass=false`, with `memory_debug_missing`;
  - `004` update-memory diagnostic has
    `long_memory_training_claim_supported=false` and router decision
    `sequence_aware_txl_ppo_update_required_next`.
  Per this slice's Failure Exit, speed curriculum should not be launched before
  fixing the clean-gait/memory-update blocker.

## Review

Status: gated not run; no H200 speed-curriculum training/eval evidence and no
speed expansion claim.
