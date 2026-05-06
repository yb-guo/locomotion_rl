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
  SONIC `gear_sonic/data/robots/g1/g1_29dof.xml` now builds and steps in Genesis
  on H200 after filling its 36 referenced mesh files from
  `gear_sonic_deploy/g1/meshes`. The XML has 29 motor actuators; Genesis reports
  `35` rigid DOFs because it includes the 6-DoF floating base.

# Lessons

- A simple baseline is required before long-context transformer experiments.
