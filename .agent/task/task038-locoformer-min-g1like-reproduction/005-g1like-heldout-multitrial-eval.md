# 005 G1-like Heldout Multi-Trial Eval

## Route

Evaluate whether the G1-like LocoFormer-min setup produces the intended
few-shot adaptation signal.

Baselines:

- existing MLP/specialist policy where applicable;
- GRU history policy;
- AdaptK160 MLP latent;
- true TXL long-memory policy.

Eval groups:

- seen G1-like morphology variants;
- held-out link-length variants;
- held-out mass/COM/inertia variants;
- motor/dynamics OOD;
- combined morphology + motor dynamic cases.

Metrics:

- final-trial pass/fail;
- trial0 to final-trial improvement;
- fall ratio, velocity tracking, posture/root z;
- action smoothness and obvious gait cheating in render review;
- throughput and GPU cost.

## Minimal Closed Loop

Close this slice with a small comparable JSON summary before any broad claim:

- at least one seen variant and one held-out variant id;
- at least one baseline row and one TXL row attempted on the same variant ids;
- trial0 and final-trial metrics;
- final-trial pass/fail and trial0-to-final improvement;
- failure cases listed by morphology id and dynamic condition;
- explicit `not_attempted` entries for missing baselines instead of silently
  dropping them.

Representative videos are a later review aid; the minimal machine gate is JSON.

Before any Task038 superiority claim, the minimal JSON fixture must be expanded
to the full claim matrix from `task.md`: at least `2` seen variants, `4`
held-out variants, speeds `0.4/1.2/2.0 m/s`, at least `3` seeds per
variant/speed condition, and at least `4` inner trials per outer episode.

## Evidence Gate

Reviewable evidence must include the JSON summary path, command used to produce
it, git/code ref if available, H200 hardware note, and any skipped baseline
reason. TXL superiority may be claimed only if it improves held-out final-trial
performance against the attempted baselines under the same variant ids and eval
matrix. Task037 evidence is baseline/control evidence only; it cannot pass any
held-out morphology claim gate.

Acceptance:

- A JSON summary compares all attempted baselines on the same variants.
- The TXL claim requires better held-out final-trial performance than baselines,
  not just successful construction.
- The TXL claim requires at least `+10` percentage points over the best
  attempted non-TXL baseline on held-out final-trial pass rate and no more than
  `-5` percentage points seen/clean final-trial regression.
- Failure cases are listed by morphology id and dynamic condition.
- Representative videos exist for at least one pass and one failure.

## Subagent Ownership

- Worker owns eval contract docs, small JSON schema/fixtures, and, when
  authorized, eval runner wiring for this slice.
- Worker must consume `001` thresholds, `002` slot ids, `003` morphology ids,
  and `004` TXL debug metadata.
- Worker must not start large training runs or produce large video/checkpoint
  artifacts without explicit router approval.
- Reviewer checks same-variant comparability, skipped-baseline accounting, and
  that the claim is not made from construction-only smokes.

## Failure Exit

If baselines and TXL cannot be evaluated on the same variant ids and metrics,
stop and report non-comparability instead of emitting a partial claim.

## Log

- 2026-05-29 Opened as the final claim gate for G1-like LocoFormer-min.
- 2026-05-29 Implemented local simulator-free eval JSON contract in
  `src/h200_locomotion_lab/training/task038_eval_contract.py`.
- 2026-05-29 Added focused tests in `tests/test_task038_eval_contract.py` for
  tiny comparable fixtures, explicit `not_attempted` baselines, same-matrix
  enforcement, construction-only TXL claim blocking, failure grouping,
  full-gate delegation, and Task037 `control_reference` exclusion.
- 2026-05-29 Verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_eval_contract.py`
  -> `8 passed in 0.08s`.
- 2026-05-29 Reviewer fix: row-level split mismatch, heldout condition
  mismatch, invalid TXL debug metadata, and missing distinct final trial now
  hard-fail `schema_passed`, `local_passed`, and superiority claim eligibility.
- 2026-05-29 Verification after reviewer fix:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_eval_contract.py`
  -> `12 passed in 0.09s`.
- 2026-05-29 Review subagent rechecked the fixed local contract and reported
  no blocking findings. The prior false-pass cases now hard-fail local schema,
  local pass, and superiority-claim eligibility.

## Review

Status: local contract closed with reviewer confirmation; H200 eval/video gates
remain pending.

Review notes:

- Local tiny fixture passes schema/comparability aggregation only; default full
  Task038 superiority claim gate still rejects it through `evaluate_claim`.
- Superiority claim is blocked for construction-only TXL rows.
- Task037 rows are accepted only as `control_reference` and do not count toward
  heldout morphology pass evidence.
- Attempted rows must include trial0 and a distinct final trial index greater
  than zero; row-level validation failures block local pass and claim attempts.
- Representative pass/failure videos, H200 hardware run evidence, and full
  matrix eval remain unverified.
