# 001 Claim Contract and Feedback Loop

## Route

Define the minimal LocoFormer-style reproduction claim before implementing more
components.

Claim target:

```text
On held-out G1-like morphology/dynamics variants, a true TXL long-memory policy
shows better multi-trial final-trial adaptation than MLP, GRU, and AdaptK
baselines under the same unified slot contract and reward/eval matrix.
```

Non-claims:

- not arbitrary omni-bodied control;
- not quadruped or wheeled transfer;
- not real-robot deployment;
- not full LocoFormer scale;
- not a pass if only a single G1 checkpoint improves.

Required feedback loops:

- local fake-env tests for claim aggregation and per-trial improvement;
- CLI `--help` tests for eval/asset tools;
- H200 smoke for one generated variant;
- H200 JSON summary for every claim-level eval.

## Minimal Closed Loop

Close this slice with a tiny fake evidence fixture that runs without MuJoCo or
training:

- two morphology ids, one `seen` and one `heldout`;
- at least two baselines and one TXL row;
- trial0 and final-trial metrics;
- one passing case and one explicit failure case;
- a check that aggregate mean cannot hide failed final-trial adaptation.

## Evidence Gate

Reviewable evidence is either local test output or a small JSON fixture path
showing:

- baseline names: MLP/specialist where applicable, GRU history, AdaptK160, TXL;
- split names: seen, link-length heldout, mass/COM/inertia heldout,
  motor/dynamics OOD, combined morphology+dynamics;
- minimum claim matrix: at least `2` seen variant ids, `4` held-out variant ids,
  speeds `0.4/1.2/2.0 m/s`, at least `3` seeds per variant/speed condition,
  and at least `4` inner trials per outer episode;
- thresholds and stop conditions for final-trial pass/fail: TXL must beat the
  best attempted non-TXL baseline by at least `10` percentage points on held-out
  final-trial pass rate, while regressing seen/clean final-trial pass rate by no
  more than `5` percentage points;
- failure classification by morphology id and condition.

Acceptance:

- The task doc names exact baselines and eval splits.
- The task doc names numeric minimums for variant count, speeds, seeds, trials,
  and TXL-vs-baseline gates.
- The first fake-env test can fail if final-trial performance is hidden by
  aggregate metrics.
- No result is marked passed without H200 JSON evidence.

## Subagent Ownership

- Worker owns this file,
  `src/h200_locomotion_lab/training/task038_claim_contract.py`, and
  `tests/test_task038_claim_contract.py` for the local fake-evidence loop.
- Worker must not edit training configs, simulator envs, H200 runners, or large
  generated outputs from this slice.
- Reviewer checks that the claim is falsifiable and that every later subtask can
  produce the evidence shape named here.

## Failure Exit

If the claim cannot be reduced to local fake evidence plus later H200 JSON
evidence, stop and return the ambiguity to the router instead of starting
implementation.

## Log

- 2026-05-29 Opened with G1-like-only scope.
- 2026-05-29 Task038/001 implementation subagent added a pure-Python fake
  evidence claim evaluator in
  `src/h200_locomotion_lab/training/task038_claim_contract.py` plus focused
  unit coverage in `tests/test_task038_claim_contract.py`.
- 2026-05-29 Local verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_claim_contract.py`
  -> `5 passed in 0.03s`.
- 2026-05-29 Fixed reviewer blocking findings: coverage is now checked on the
  same required policy/variant/speed/seed/trial matrix; held-out condition
  coverage is explicit; tiny relaxed evidence uses two baselines plus TXL; and
  `ClaimResult.passed` is documented as a local fake-evidence shape gate rather
  than an H200 reproduction verdict.
- 2026-05-29 Local verification after reviewer fixes:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_claim_contract.py`
  -> `8 passed in 0.03s`.
- 2026-05-29 Review subagent rechecked the fixes and reported no blocking
  findings. Remaining note: later H200 JSON should preserve per-morphology
  failure classification, not only policy-level aggregate rates.

## Review

Status: local closed with reviewer confirmation for the fake-evidence loop.
This is not a Task038 reproduction pass; later H200 JSON evidence remains
required before any claim-level pass.
