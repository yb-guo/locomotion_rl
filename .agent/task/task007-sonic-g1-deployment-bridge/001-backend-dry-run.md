# Route

Task: task007-sonic-g1-deployment-bridge

Goal: introduce the G1 29DoF backend boundary and prove it can run without
Genesis or hardware by replaying recorded/logged state rows.

Pass condition:

- A `RobotBackend` contract exposes reset, state read, target write, and
  control-frame advance.
- Genesis can be wrapped behind the same contract.
- Log replay can emit 36D root+motor qpos frames and accept 29D motor targets.
- Dry-run summary CSV records finite 994D decoder observations, raw actions,
  and MuJoCo-order motor targets.
- Local tests pass without importing Genesis.

Fail condition:

- Any path silently changes the existing task006 Genesis rollout semantics.
- Any state/action dimension mismatch is accepted.
- Dry-run requires real hardware or simulator installation.

# Log

- 2026-05-08: Opened route after task006 official-context 200-frame pass.
- 2026-05-08: Implemented the first backend dry-run layer:

```text
src/h200_locomotion_lab/envs/robot_backend.py
src/h200_locomotion_lab/tools/sonic_g1_deployment_dry_run.py
tests/test_robot_backend.py
tests/test_sonic_g1_deployment_dry_run.py
```

The backend-neutral contract now carries:

- 7D MuJoCo root qpos;
- 29D MuJoCo-order motor q/dq;
- 3D base angular velocity;
- previous raw SONIC action in policy/IsaacLab order;
- 29D MuJoCo-order motor position targets after the official action bridge.

`LogReplayG1RobotBackend` can replay 36D qpos rows and estimate motor
velocities from adjacent rows. It records commands but does not feed them back
into the state, which keeps it suitable for pre-hardware dry-run validation.

The dry-run tool builds the 994D decoder observation history, maps raw 29D
actions through the official SONIC G1 bridge, and writes a per-frame summary
CSV with finite gates and motor target columns.

Verification:

```text
PYTHONPATH=src python -m pytest -p no:cacheprovider \
  tests/test_robot_backend.py \
  tests/test_sonic_g1_deployment_dry_run.py \
  tests/test_genesis_sonic_planner_encoder_rollout_probe.py

10 passed
```

Full local verification:

```text
PYTHONPATH=src python -m pytest -p no:cacheprovider
76 passed
```

Formatting check:

```text
python -m ruff check src tests
No module named ruff
```

# Review

Status: pass for local backend dry-run scaffold.

This does not command hardware and does not prove real-time behavior. It proves
the I/O boundary can be exercised without Genesis or a robot, and it gives the
next H200 route a concrete dry-run tool.
