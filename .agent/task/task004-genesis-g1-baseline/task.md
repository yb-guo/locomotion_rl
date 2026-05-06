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
- `002-genesis-env-reset-step`: pass. `GenesisG1Env` now has a real
  single-env Genesis backend for the 29-motor G1 asset, and H200 reset/step
  smoke passed with 96D observations, 29D actions, and `motor_dof_count=29`.
- `003-ppo-baseline-loop`: pending.

# Lessons

- A simple baseline is required before long-context transformer experiments.
