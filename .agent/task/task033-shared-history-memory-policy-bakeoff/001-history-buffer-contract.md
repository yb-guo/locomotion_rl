# 001 History Buffer Contract

## Route

Define the shared history stream before implementing any policy-specific
consumer.

Actor-visible frame fields:

- current actor observation;
- previous action already visible in the current actor contract;
- optional action-response residuals derived only from actor-visible state.

Debug-only fields:

- active failure joint id;
- motor scale;
- failure type;
- segment/case id;
- scheduler state.

Device and shape constraints:

- Buffer must live on GPU for H200 training/eval.
- No Python per-env loops in step-time append/reset.
- Ring buffer layout should be batched:
  `[num_envs, history_len, frame_dim]` or equivalent.
- Reset must clear or reinitialize only the reset `env_ids`.

Policy boundaries:

- StackMLP may flatten `K` frames.
- GRU may use current frame plus hidden state, but reset semantics still come
  from the shared buffer/done stream.
- LocoFormer-style consumer may tokenize history, but it must not own a
  separate env history path.

## Log

- 2026-05-28 Planned as the first Task033 implementation contract.

## Review

Status: planned. Pass requires a smoke test proving batched append/reset and
no actor fault-label leakage.
