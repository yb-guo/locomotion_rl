# 003: SONIC Sequence Replay

## Route

Replay finite SONIC raw action rows through `SequenceActionProvider` and the
new mjlab backend before attempting online planner/encoder inference.

## Log

- 2026-05-15 Searched the current H200 workspace for existing task006 SONIC
  action/model artifacts under `/mnt/workspace/users/guoyubo/agent_workspace`.
  No usable `model_encoder.onnx`, `model_decoder.onnx`, `planner_sonic.onnx`,
  or official SONIC action CSV was present. The earlier `/root` task006 paths
  also appear unavailable in the current machine state.

## Review

Blocked until SONIC artifacts or action rows are restored. Do not download
them without explicit user approval.
