# Goal

Build the first Genesis-based G1 locomotion RL baseline on H200.

# Scope

- Genesis environment adapter.
- G1 robot asset/config inventory.
- Reset/step loop.
- PPO baseline before transformer policy.
- No SONIC or LocoFormer architectural claims until the baseline learns.

# Subtasks

- `001-g1-asset-and-control-contract.md`
- `002-genesis-env-reset-step.md`
- `003-ppo-baseline-loop.md`
- `004-baseline-metrics-and-logs.md`

# Result

in progress

- `001-g1-asset-and-control-contract`: local contract pass.
- `002-genesis-env-reset-step`: local reset/step boundary pass with contract-only
  backend; real H200 Genesis package is not installed yet.

# Lessons

- A simple baseline is required before long-context transformer experiments.
