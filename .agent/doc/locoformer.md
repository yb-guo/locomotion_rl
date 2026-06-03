# LocoFormer Plan

LocoFormer is treated as a research reproduction target, not a first-day full
scale reproduction.

## Minimal Reproduction

The first useful target is:

- One robot family: G1-like humanoid variants first.
- Small morphology variation.
- Short but real context window.
- PPO or PPO-like baseline.
- Transformer policy core behind a stable observation/action schema.
- Online adaptation buffer.

## Current Direction

Task038 narrows the first LocoFormer-style reproduction to a G1-like family:

- fixed high-level humanoid topology;
- fixed unified joint-slot semantics and action dimension;
- randomized G1-like link lengths, mass, COM, inertia, and motor dynamics;
- held-out G1-like morphology variants for evaluation;
- multi-trial final-trial evaluation with memory retained across inner resets;
- comparison between non-transformer baselines and a true TXL memory policy.

Do not claim full LocoFormer reproduction from Task038. The first claim target is
only: a LocoFormer-style minimal reproduction where TXL long memory improves
held-out G1-like morphology/dynamics adaptation over MLP/GRU/AdaptK baselines.

## Agent Submodules

- `morphology_encoder`
- `proprio_tokenizer`
- `motion_context_encoder`
- `transformer_policy`
- `actor_critic_heads`
- `adaptation_buffer`

## Acceptance

Do not call it a LocoFormer reproduction until:

- A non-transformer baseline exists.
- The transformer policy runs in the same environment.
- Metrics show comparable or better adaptation under held-out dynamics or
  morphology randomization.
- Failure cases and hardware cost are recorded.

