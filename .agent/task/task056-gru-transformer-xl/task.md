# Task 056: GRU then Transformer-XL

## Route

Validate recurrent state propagation with GRU before enabling the canonical
six-layer, 256-hidden, eight-head TXL.  Use sequence-aware minibatches and
padding masks; clear memory only on `context_done`.

## Log

- 2026-08-19: Added GRU/TXL cores, explicit reset helpers, 128-step bounded
  memory, sequence padding/loss masks, sequence PPO minibatch utilities, and
  an explicit reset-memory-every-trial ablation switch.

## Review

Focused tests verify trial_done preserves state, context_done clears state,
TXL bounded memory, and padded sequence means.  The paired multi-seed
adaptation comparison against shared MLP and reset-memory baselines remains
pending.
