# Route

Task: task006-sonic-genesis-action-policy

Goal: Prove that the validated Genesis 29-motor G1 backend can be driven by
29D normalized action sequences through `GenesisG1Env.step(action)`.

Pass condition:

- H200 run completes without non-finite state.
- Action shape is exactly 29.
- Actions are clipped or validated in the same contract used by
  `GenesisG1Env.step`.
- Base height stays inside the configured smoke range.
- Max qvel and action range are recorded.
- Produce a short GIF/contact sheet if the smoke passes.

Fail condition:

- Any non-finite state.
- Action dimension mismatch.
- Robot falls below the smoke height range.
- Genesis scene build or step fails.
- H200 SSH/session failure before the command starts; record as infra failure,
  not simulator failure.

Implementation plan:

1. Add a small action replay tool that accepts either:
   - a CSV containing one 29D action row per policy step; or
   - a deterministic built-in action fixture for smoke testing.
2. Run local tests for parser/contract behavior without importing Genesis.
3. Sync to H200.
4. Run a short H200 smoke through `GenesisG1Env.step(action)`.
5. If the numeric smoke passes, render a short dynamic GIF/contact sheet.

# Log

- 2026-05-07: Opened route. This subtask is the next step requested by the
  user. It intentionally precedes SONIC policy integration.
- 2026-05-07: Added action replay tools:
  - `python -m h200_locomotion_lab.tools.genesis_action_replay_smoke`
  - `python -m h200_locomotion_lab.tools.genesis_action_replay_gif`
- The numeric smoke path calls `GenesisG1Env.step(action)` for each 29D
  normalized action row. The GIF path reuses the same action fixture and target
  mapping for visual inspection after the numeric smoke passes.
- Added local parser/fixture tests in
  `tests/test_genesis_action_replay_smoke.py` and fixed test fixtures under
  `tests/fixtures`.
- Added a `default_motor_positions` override to `GenesisSceneConfig` and
  `GenesisG1Env.from_genesis_asset(...)`. This is required because SONIC-style
  actions are deltas around a nominal joint pose; using the MJCF asset qpos0 as
  the action-zero pose caused the visual replay to crouch/collapse toward the
  plane even when the numeric smoke was finite.
- Local verification:

```text
Command: PYTHONPATH=src python -m pytest -p no:cacheprovider
Result: 18 passed
```

- H200 tool-level verification:

```text
Command: PYTHONPATH=src python3 -m pytest -p no:cacheprovider tests/test_genesis_adapter.py tests/test_genesis_action_replay_smoke.py
Result: 16 passed
```

- H200 numeric smoke with asset qpos0 and identity root was finite, but not
  accepted as pass evidence because the contact sheet showed the robot
  crouching/collapsing toward the plane. The useful diagnostic was:

```text
Log: /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_action_replay_sine_50f.log
GENESIS_ACTION_REPLAY_SMOKE_OK
MIN_LINK_HEIGHT_MIN 0.029549263417720795
```

- H200 numeric smoke passed using the SONIC reference root pose and the first
  `joint_pos.csv` row as the nominal action-zero pose:

```text
Log: /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_action_replay_sine_refroot_ref0_50f.log
GENESIS_ACTION_REPLAY_MODE normalized_actions
ACTIONS_SOURCE fixture:sine
REPLAY_FRAMES 50
ACTION_DIM 29
ACTION_MIN_MAX -0.11999990660629659 0.11999999778370125
ACTION_MAX_ABS 0.11999999778370125
ACTION_OUT_OF_RANGE_VALUES 0
ACTION_SCALE_RAD 0.25
BASE_POS (0.002389, 0.011728, 0.791166)
BASE_QUAT (0.711231, -0.00883, -0.004562, -0.702888)
DEFAULT_JOINT_POS_SOURCE .../walking_quip_360_R_002__A428/joint_pos.csv
DEFAULT_JOINT_POS_ROW 0
DEFAULT_JOINT_POS_MIN_MAX -0.72018 1.176074
MOTOR_DOF_COUNT 29
RESET_OBS_LEN 96
FRAME 0 base_z 0.7911660075187683 min_link_z 0.7911660075187683 action_min -0.11942877909229951 action_max 0.11957707376887206 max_abs_qvel 3.9576380252838135 obs_len 96
FRAME 49 base_z 0.7911660075187683 min_link_z 0.7911660075187683 action_min -0.11942877909229953 action_max 0.11957707376887206 max_abs_qvel 2.256716251373291 obs_len 96
FINITE_OK True
BASE_HEIGHT_MIN 0.7911660075187683
BASE_HEIGHT_MAX 0.7911660075187683
BASE_HEIGHT_FINAL 0.7911660075187683
MIN_LINK_HEIGHT_MIN 0.7911660075187683
MIN_LINK_HEIGHT_FINAL 0.7911660075187683
MAX_ABS_QVEL 6.885622501373291
POLICY_STEPS 50
SIM_STEPS 200
HEIGHT_OK_RANGE 0.2 1.5 True
GENESIS_ACTION_REPLAY_SMOKE_OK
```

- H200 GIF/contact-sheet verification passed for the same action sequence:

```text
Log: /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_action_replay_sine_refroot_ref0_50f_gif.log
Remote GIF: /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/videos/genesis_g1_action_replay_sine_refroot_ref0_50f.gif
Remote contact sheet: /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/videos/genesis_g1_action_replay_sine_refroot_ref0_50f_contact.png
Local GIF: .agent/task/task006-sonic-genesis-action-policy/artifacts/genesis_g1_action_replay_sine_refroot_ref0_50f.gif
Local contact sheet: .agent/task/task006-sonic-genesis-action-policy/artifacts/genesis_g1_action_replay_sine_refroot_ref0_50f_contact.png
GENESIS_ACTION_REPLAY_GIF_MODE normalized_actions
FRAMES 50
ACTION_MAX_ABS 0.11999999778370125
BASE_HEIGHT_MIN 0.7911660075187683
BASE_HEIGHT_MAX 0.7911660075187683
BASE_HEIGHT_FINAL 0.7911660075187683
RENDERED_FRAMES 50
GIF_BYTES 114221
GENESIS_ACTION_REPLAY_GIF_OK
```

Remote `imageio` read-back:

```text
READ_FRAMES 50
DIFF_MIN 0.04193204365079365
DIFF_MAX 0.12690724206349208
DIFF_AVG 0.08542785268869449
CONTACT shape (320, 2100, 3)
```

# Review

Status: pass.

H200 numeric action replay passed through `GenesisG1Env.step(action)` with a
50-step 29D normalized sine fixture. Visual GIF/contact-sheet review also passed
for the same root pose, nominal joint pose, and action sequence. L2 SONIC policy
rollout is now unblocked.
