# 002: mjlab Backend Smoke

## Route

Implement `MjlabG1RobotBackend` against the existing `G1RobotBackend` protocol.
Verify action-target inversion with unit tests and then run a short H200 zero
action smoke through mjlab.

## Log

- 2026-05-15 Implemented:
  - `src/h200_locomotion_lab/envs/mjlab_backend.py`
  - `src/h200_locomotion_lab/tools/mjlab_sonic_rollout.py`
- Local verification:
  `PYTHONPATH=src python -m pytest -p no:cacheprovider tests/test_mjlab_backend.py tests/test_sonic_controller.py`
  passed with 4 tests.
- H200 smoke command ran in:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab`
- Provider:
  `zero`
- Output:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/outputs/task025/zero_smoke/zero_smoke-step-0.mp4`
- Local copy:
  `outputs/task025/zero_smoke/zero_smoke-step-0.mp4`
- Summary:
  - steps: 120
  - done steps: `[]`
  - root start: `[0.0807, -0.2581, 0.7967]`
  - root end: `[0.0614, -1.1458, 0.1335]`
  - root delta: `[-0.0193, -0.8878, -0.6632]`

## Review

Partial pass.

The backend can construct, step mjlab, and record video. The zero raw SONIC
action maps to SONIC default angles, not mjlab's current trained-policy stance,
so the robot collapses over 120 steps. This is expected evidence that the
adapter path executes, but it is not a locomotion pass.
