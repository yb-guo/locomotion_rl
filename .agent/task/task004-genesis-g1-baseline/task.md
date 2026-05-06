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
  backend; H200 `genesis-world==0.4.6` install and raw CUDA Genesis smoke pass.
  SONIC `model_data/g1/g1_29dof_with_hand.xml` builds and steps in Genesis on
  H200, but Genesis reports it as `49` DoF, so a true 29DoF runtime asset is
  still needed before PPO training.

# Lessons

- A simple baseline is required before long-context transformer experiments.
