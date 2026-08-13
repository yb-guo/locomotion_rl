# 001: True-TXL Rollout-Boundary State Consistency

## Route

Fix the mismatch between stateful sampling and update-time sequence replay.

Current boundary:

- `Task038TrueTxlMemoryModel.get_latent()` samples with persistent per-env
  `_memory_tensors` and `_memory_lengths`;
- `task040_get_sequence_latent()` creates zero memory for every update replay;
- `Task040SequenceAwareTrueTxlPPO.update()` reads rollout observations and done
  masks, but storage does not provide the memory state that existed before the
  first rollout observation.

Choose one exact reconstruction mechanism:

- store detached rollout-start memory tensors/lengths per layer and environment
  and feed the matching env slice into sequence replay; or
- use a verified burn-in prefix that reconstructs exactly the same state as
  sampling.

The implementation must preserve outer-reset clearing, inner-reset retention,
environment slicing, and inference-cache isolation. A zero-memory shortcut is
not accepted when sampling began with non-empty memory.

## Acceptance

- Add a regression with non-empty rollout-start memory and at least one env
  reset pattern.
- With parameters frozen and before `optimizer.step()`, recomputed action log
  probabilities match stored sampling values with max absolute error
  `<= 1e-5` and PPO ratios match one with max absolute error `<= 1e-5`.
- The test fails when rollout-start memory is forcibly zeroed, proving it
  exercises the defect rather than a zero-memory special case.
- Multi-minibatch environment slices receive their own correct memory rows.
- Replay does not mutate the live inference cache.
- Rollout storage/reset lifecycle cannot reuse a previous rollout's snapshot.
- Debug output records parity error, ratio error, snapshot/burn-in mode, and
  whether rollout-start memory was non-empty.
- The corrected Task040 local RTX 4090 smoke has zero stateless fallback and
  passes the new parity gate before any new True-TXL quality checkpoint is
  promoted.

## Log

- 2026-08-07 Opened from static code evidence in
  `training/rsl_history_wrapper.py`: stateful inference cache in `get_latent()`,
  zero initialization in `task040_get_sequence_latent()`, and update replay
  starting from rollout observations alone in
  `Task040SequenceAwareTrueTxlPPO.update()`.
- 2026-08-07 Existing tests were found to gate sequence counters and stateless
  fallback, but no test compares sampling and replay log probabilities before
  an optimizer step.
- 2026-08-07 Implemented rollout-start True-TXL memory snapshot capture at the
  first action of each rollout and replayed the matching per-env memory slice
  during `Task040SequenceAwareTrueTxlPPO.update()`.
- 2026-08-07 Added no-optimizer-step log-probability/ratio parity diagnostics
  to the Task040 algorithm debug output and tightened
  `task040_sequence_txl_ppo_update_smoke.py` to require two iterations,
  non-empty rollout-start memory, and parity pass.
- 2026-08-07 Local regression evidence:
  `UV_PROJECT_ENVIRONMENT=.venv uv run pytest -q -p no:cacheprovider
  tests/test_task040_sequence_txl_ppo_update_smoke.py
  tests/test_task037_multitrial_contract.py tests/test_ppo_loop.py
  tests/test_task044_hidden_fault_target.py
  tests/test_task044_continuous_fault_eval.py` -> 57 passed.
- 2026-08-07 User moved the runtime target to local RTX 4090. The repo-local
  CUDA environment now reports `torch 2.5.1+cu121` on `NVIDIA GeForce RTX
  4090`, but the real Task040 MJLab smoke is not runnable yet because
  `mjlab`, `src.tasks`, and the configured G1 asset are absent locally. No
  asset or upstream download was performed.
- 2026-08-10 Reproduced the local MJLab Task040 smoke after the runtime was
  made runnable. A one-iteration run had empty rollout-start memory by
  construction; a two-iteration run captured non-empty rollout-start memory but
  still failed parity with `max_logprob_abs_error=0.08075332641601562` and
  `max_ratio_abs_error=0.08410346508026123`.
- 2026-08-10 Fixed the remaining parity drift by snapshotting the actor
  observation normalizer once per rollout step before action sampling and
  replaying sequence updates with those per-step normalizer states. This
  complements the existing rollout-start True-TXL memory snapshot; both are
  required for exact no-optimizer-step logprob/ratio parity.
- 2026-08-10 Fresh regression and local CUDA evidence:
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m pytest -q -p
  no:cacheprovider tests/test_task040_sequence_txl_ppo_update_smoke.py` -> 18
  passed; final Task047 targeted matrix including this file -> 78 passed;
  `task047_local_4090_task040_smoke_after_normalizer_snapshot_fix_v2.json` ->
  `pass=true`, `normalizer_replay_mode=per_step_snapshot`,
  `normalizer_snapshot_count=2`, `rollout_start_memory_non_empty=true`,
  `max_logprob_abs_error=0.0`, `max_ratio_abs_error=0.0`, and zero stateless
  fallback.

## Review

Status: passed for the Task040 correctness-smoke boundary on the local RTX 4090
MJLab/MuJoCo Warp route.

The fresh CUDA smoke proves that sequence replay uses the non-empty
rollout-start True-TXL memory and the per-step sampling-time actor normalizer
state. This is still a PPO plumbing/correctness result, not a policy-quality
claim.
