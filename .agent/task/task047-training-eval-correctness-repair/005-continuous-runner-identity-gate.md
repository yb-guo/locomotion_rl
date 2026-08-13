# 005: Continuous Eval Runner Identity Gate

## Route

Make Task044 continuous-eval pipeline identity evidence come from the runtime
object rather than a constant.

Current behavior:

- `run_eval()` instantiates a runner but writes
  `runner_cls=TASK044_CONTINUOUS_EXPECTED_RUNNER_CLS`;
- `pipeline_pass` checks action dimensions, actor class, and physical
  continuity, but not actual runner class;
- `_failure_reasons()` checks whether the requested expected string equals the
  module constant, not whether the instantiated runner matches it.

Record `type(runner).__name__` (and, if wrappers/proxies require it, a clear
resolved consumer class) as actual runtime evidence. Gate actual versus
expected identity directly.

## Acceptance

- JSON contains distinct `runner_cls` and `expected_runner_cls` values sourced
  from actual runtime and CLI contract respectively.
- Exact mismatch forces `pipeline_pass=false`, `quality_gate_pass=false`, and
  `pass=false`.
- Failure reasons include `runner_cls_mismatch` for an actual mismatch.
- A fake/stub wrong-runner test cannot produce a passing summary even when
  action dimensions, actor class, continuity, and quality metrics pass.
- The correct continuous runner still passes the pipeline identity portion.
- Failure summaries do not hard-code successful runtime identity.
- Local RTX 4090 reruns at least one continuous eval after the fix and
  preserves the previous JSON as historical evidence.

## Log

- 2026-08-07 Opened from
  `tools/task044_continuous_fault_eval.py`: returned `runner_cls` is a constant
  and actual runner identity is absent from `pipeline_pass`.
- 2026-08-07 Existing Task044/Task045 continuous runs already fail on physical
  falls, so this is a future false-pass risk rather than evidence that the
  recorded failed standing gate actually passed.
- 2026-08-07 Updated `task044_continuous_fault_eval.py` to record
  `type(runner).__name__` and require exact actual-vs-expected runner identity
  inside `pipeline_pass`.
- 2026-08-07 Added a wrong-runner regression proving that action dimensions,
  actor class, physical continuity, and quality metrics cannot pass the
  pipeline when runtime runner identity mismatches. Local targeted pytest
  evidence is recorded in subtask 006 and passed.
- 2026-08-07 Runtime target changed to local RTX 4090. Corrected continuous
  eval is blocked locally by missing MJLab/task modules, missing configured G1
  asset, and no local checkpoint path. No checkpoint or asset was downloaded.

## Review

Status: local code/regression fixed; local RTX 4090 continuous eval rerun
blocked by missing runtime/assets/checkpoint.

This subtask tightens provenance only. It must not weaken physical continuity
or post-fault quality thresholds.
