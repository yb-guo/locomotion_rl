# LocoFormer Plan

LocoFormer is treated as a research reproduction target, not a first-day full
scale reproduction.

## Minimal Reproduction

The first useful target is:

- One robot family, likely Unitree G1 or a simple humanoid.
- Small morphology variation.
- Short but real context window.
- PPO or PPO-like baseline.
- Transformer policy core behind a stable observation/action schema.
- Online adaptation buffer.

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

