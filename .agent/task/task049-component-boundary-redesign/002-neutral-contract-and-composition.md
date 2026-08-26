# 002: Neutral Contract and Composition

## Route

Implement the smallest useful shared vocabulary without importing Torch,
MuJoCo, Genesis, a robot, or a named learning algorithm. Compose components
from explicit YAML paths and reject real compatibility mismatches.

## Log

- 2026-08-19 Added `TensorSpace`, component specs, task/policy/algorithm
  protocols, `TaskStep`, `TransitionBatch`, and `UpdateReport` under `core/`.
- 2026-08-19 Added strict task/policy/algorithm/experiment YAMLs for the verified
  Unitree G1 flat-walking PPO baseline. Observation/action dimensions occur
  only in the task config.
- 2026-08-19 Added a minimal interaction loop that forwards opaque task and
  policy metrics without interpreting them.
- 2026-08-19 Moved the concrete G1 27DoF velocity-tracking MDP into `tasks/`,
  the tanh-Gaussian actor-critic implementation into `policies/`, and GAE/PPO
  update kernels into `algorithms/`. Legacy imports delegate to these owners.

## Review

Status: passed. The default composition loads without Torch/simulator imports,
and an incompatible JiT-style sample-only policy is rejected by PPO's declared
capability requirements.
