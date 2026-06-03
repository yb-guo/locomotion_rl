# 004 True TXL Memory Policy Contract

## Route

Replace the current segment-pooling smoke with a true Transformer-XL memory
consumer.

Required semantics:

- Encode each `obs/action` frame as a token.
- Maintain per-layer hidden-state cache.
- Current segment attends to cached previous segment memory.
- Inner trial reset preserves TXL memory.
- Outer episode reset clears TXL memory.
- Episode masks prevent memory from leaking across independent outer episodes.
- Inference can update cache incrementally.

## Minimal Closed Loop

Close this slice first with a local fake-env/cache test, before any training:

- two parallel env ids;
- two inner trials per outer episode;
- memory persists across inner reset for the same outer episode;
- memory clears on outer reset;
- episode masks prevent one env's cache from leaking into another env;
- incremental inference updates cache shape and debug counters.

H200 runner smoke is a second evidence gate, not a locomotion-quality claim.

## Evidence Gate

Reviewable evidence must include local command output or a small JSON/debug log
showing cache length, reset events, env ids, mask events, and memory
clear/preserve decisions. H200 smoke evidence must name env count, runner path,
policy class, and whether the smoke was construction-only or stepping.

Acceptance:

- Local fake-env tests cover memory preservation/clearing and per-env isolation.
- RSL/MJLab runner smoke works on the existing Task037 G1 env.
- H200 env64 and env8192 one-iteration smokes pass.
- Eval JSON records TXL memory debug metadata.
- No locomotion quality claim is made from construction smoke alone.

## Subagent Ownership

- Worker owns TXL policy/cache contract docs and, when authorized, the smallest
  policy/cache implementation and fake-env tests needed for this slice.
- Worker does not own morphology generation, baseline training, or final
  held-out scoring.
- Reviewer checks reset semantics, per-env isolation, and that no construction
  smoke is presented as adaptation evidence.

## Failure Exit

If cache semantics cannot distinguish inner reset from outer reset in the
runner API, stop and route the API gap before training.

## Log

- 2026-05-29 Opened as the first true long-memory implementation slice for
  LocoFormer-min.
- 2026-05-29 Implemented local fake-env TXL memory/cache contract in
  `src/h200_locomotion_lab/training/task038_txl_memory.py`.
  - Pure Python/dataclass implementation; no torch, runner, simulator, H200, or
    training dependency.
  - Encodes each obs/action frame as one token.
  - Maintains per-env, per-layer hidden-state cache with `memory_len` cap.
  - `append_segment` reports `attended_previous_memory_lengths` before append.
  - Inner reset records an event and preserves memory.
  - Outer reset clears selected env memory only before the next append.
  - Env ids and episode indices are stored on cached hidden tokens and checked
    before append to catch env leakage or stale outer-episode memory.
  - Incremental `step` updates cache lengths and debug counters.
- 2026-05-29 Added local contract tests in
  `tests/test_task038_txl_memory_contract.py` covering parallel env isolation,
  first/second segment previous memory lengths, inner reset preservation, outer
  selected-env clearing, mask isolation, per-layer/env memory cap, incremental
  counters, and obs/action token encoding.
- 2026-05-29 Verification command:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_txl_memory_contract.py`
  - Result: `8 passed in 0.04s`.
- 2026-05-29 Added reviewer boundary coverage and pinned edge semantics.
  - Empty segments are rejected before reset/append side effects and do not
    increment `segments_appended`.
  - `outer_reset=True` plus `inner_reset=True` is ordered as outer clear first,
    then inner preserve event over cleared memory; the following append attends
    previous memory length `(0, ...)`.
  - Added failure coverage for `FrameToken` dimension mismatch, invalid
    `env_id`, corrupted private cache `env_id`, and stale cached
    `episode_index`.
- 2026-05-29 Verification command:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_txl_memory_contract.py`
  - Result: `14 passed in 0.05s`.
- 2026-05-29 Review subagent rechecked the boundary coverage and reported no
  blocking findings.

## Review

Status: local fake-cache closed with reviewer confirmation. H200 runner smoke
remains pending and is required before runner compatibility can be claimed.

Local evidence:

- Fake-env/cache contract tests pass locally.
- Debug output shape includes env ids, episode indices, reset events,
  preserve/clear decisions, cache lengths, and incremental counters.
- Boundary tests pin empty segment rejection, combined reset ordering,
  invalid-token/env failures, and leak-guard detection for corrupted cache
  metadata.

Pending gates:

- H200 runner smoke pending.
- No training, simulation, or locomotion-quality claim made for this slice.
