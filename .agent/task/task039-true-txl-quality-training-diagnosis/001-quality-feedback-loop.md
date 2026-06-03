# 001 Quality Feedback Loop

## Route

Build the smallest deterministic feedback loop that can say whether a clean
G1-like checkpoint is learning to walk better than a random or one-iteration
smoke checkpoint.

This slice does not train. It defines and tests the metric contract that later
MLP and true-TXL runs must use.

## Minimal Closed Loop

Close this slice with:

- a JSON schema or helper that accepts one eval summary and returns diagnostic
  pass/fail;
- a fixture that fails a poor checkpoint because final-trial/root-z/fall or
  tracking metrics are bad;
- a fixture that passes a clearly improved clean checkpoint;
- explicit separation between `pipeline_pass` and `quality_gate_pass`;
- local tests and `--help` coverage if a CLI is added.

## Evidence Gate

Evidence must include local focused tests showing:

- missing metrics fail;
- non-finite metrics fail;
- low root z or high gravity xy fails;
- high fall ratio fails;
- poor velocity tracking fails;
- a complete improved fixture passes;
- claim flags remain diagnostic-only.

No H200 run is required for this first slice.

## Subagent Ownership

Worker owns only:

- this document;
- a small Task039 quality contract helper or CLI if needed;
- focused tests for the helper/CLI.

Reviewer checks that the feedback loop cannot pass a pipeline-only JSON and
does not encode a reproduction or superiority claim.

## Failure Exit

If the current eval summaries do not contain enough fields to distinguish
standing, falling, or bad tracking from walking, stop and route back to add
instrumentation before running more training.

## Log

- 2026-05-30 Opened as the first Task039 slice.
- 2026-05-30 Added pure-local quality feedback helper
  `src/h200_locomotion_lab/training/task039_quality_feedback.py` and focused
  tests in `tests/test_task039_quality_feedback.py`. The helper keeps pipeline
  health separate from `quality_gate_pass`, rejects missing/non-finite required
  metrics, gates final-trial completion/fall/gravity/root-z/tracking quality,
  accepts trial0 or aggregate context, and preserves diagnostic-only no-claim
  fields.
- 2026-05-30 Tightened no-overclaim contract: `quality_claim`,
  `training_claim`, `eval_claim`, `reproduction_claim`, and
  `superiority_claim` must be explicitly present and false; missing or true
  claim flags fail with clear reasons.
- 2026-05-30 Router verification passed:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider --basetemp .test_tmp_task039_001_router2 tests\test_task039_quality_feedback.py tests\test_agent_inventory.py`
  returned `13 passed in 0.07s`. `python -m h200_locomotion_lab.tools.inspect_agent`
  also completed successfully.
- 2026-05-30 Independent review subagent reported no blockers after the
  explicit-claim-flag tightening. Residual risk: aggregate can satisfy trend
  context but is not thresholded or compared; later baseline comparison should
  require explicit trial0 or baseline evidence.
- 2026-05-30 Added positive calibration evidence without making it a Task039
  baseline or reproduction claim:
  - local helper/CLI:
    `src/h200_locomotion_lab/tools/task039_quality_calibration.py`;
  - local tests:
    `tests/test_task039_quality_calibration.py`;
  - router verification:
    `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider --basetemp
    .test_tmp_task039_quality_calib tests\test_task039_quality_calibration.py
    tests\test_task039_quality_feedback.py tests\test_agent_inventory.py`
    returned `19 passed in 0.12s`;
  - H200 positive calibration JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task039/quality_calibration/task039_adaptk160_model5467_clean_vx0p4_positive_quality_calibration.json`.
- 2026-05-30 Positive calibration result from existing Task037 AdaptK160
  `model_5467` clean 0.4 eval:
  - `task039_quality_calibration_only=true`;
  - `policy_label=AdaptK160-positive-calibration`;
  - `pipeline_pass=true`;
  - `quality_gate_pass=true`;
  - `pass=true`;
  - `final_trial.fall_ratio=0.0`;
  - `final_trial.gravity_xy.max=0.06800129264593124`;
  - `final_trial.root_z.min=0.7846410870552063`;
  - `final_trial.lin_vel_error.mean=0.11638887971639633`;
  - all no-overclaim flags remained false.
  This proves the Task039 gate can pass a clearly improved clean checkpoint
  while `002`/`003` prove it fails poor checkpoints. It is not an MLP or
  true-TXL baseline.

## Review

Status: closed with reviewer confirmation and positive H200 calibration
evidence. No Task039 training, reproduction, or superiority claim is made by
this slice.
