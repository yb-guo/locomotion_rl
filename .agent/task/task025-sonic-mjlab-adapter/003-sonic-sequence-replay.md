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
- 2026-05-15 Ran a synthetic sequence wiring smoke on H200 using the local
  `tests/fixtures/actions_limit_frames.csv` fixture. This is not an official
  SONIC replay; it only verifies the finite action CSV path through
  `SequenceActionProvider -> ScalarG1Runtime -> MjlabG1RobotBackend`.

  Command shape:

  ```bash
  python -m h200_locomotion_lab.tools.mjlab_sonic_rollout \
    --task-id Unitree-G1-Flat \
    --provider sequence \
    --actions-csv /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/actions_limit_frames.csv \
    --output-dir outputs/task025/synthetic_sequence_smoke \
    --steps 120 \
    --device cuda:0 \
    --disable-terminations
  ```

  Result:

  - `steps`: 120
  - `done_steps`: `[]`
  - `root_start_xyz`: `[0.14585301280021667, 0.01471762452274561, 0.7966601848602295]`
  - `root_end_xyz`: `[-0.5933108925819397, -0.5169636011123657, 0.1342436969280243]`
  - `root_delta_xyz`: `[-0.7391639053821564, -0.5316812256351113, -0.6624164879322052]`
  - remote video:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/outputs/task025/synthetic_sequence_smoke/synthetic_sequence-step-0.mp4`
  - local copy:
    `outputs/task025/synthetic_sequence_smoke/synthetic_sequence-step-0.mp4`

## Review

The sequence provider path is wired and can render video evidence in mjlab.
The robot falls because the fixture is a repeated synthetic raw action row, not
a stabilized SONIC action trace.

Official SONIC replay remains blocked until SONIC artifacts or action rows are
restored. Do not download them without explicit user approval.
