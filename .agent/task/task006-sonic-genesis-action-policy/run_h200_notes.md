# Task 006 H200 Run Notes

## 2026-05-07

Target repo path:

```text
/root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/repo_replay_smoke
```

Important H200 execution detail: run from `/tmp` with explicit
`PYTHONPATH=/root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/repo_replay_smoke/src`.
An earlier sync left a stale top-level `h200_locomotion_lab` package under the
repo root, which can shadow `src/` when running from the repo directory.

Verification:

```text
Local full pytest: 44 passed
H200 targeted pytest from /tmp: 18 passed
Local ruff: not run; ruff is not installed locally
```

Decoder-only Genesis closed-loop:

```text
10 frames, token replay, official captured history init: passed
base_z final: 0.6546086668968201
action max abs: 5.1907243728637695

20 frames, same settings: failed height gate
base_z final: 0.2644214928150177
action max abs: 8.832411766052246
```

Conclusion: SONIC decoder is now connected to Genesis state/action feedback for
a short smoke, but stable closed-loop rollout is not passed.

Visual evidence:

```text
Remote GIF:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/videos/genesis_g1_sonic_closed_loop_20f.gif

Local GIF:
.agent/task/task006-sonic-genesis-action-policy/artifacts/genesis_g1_sonic_closed_loop_20f.gif

Frames: 20
base_z: 0.7887014746665955 -> 0.2644214630126953
action max abs: 8.832415580749512
```

## Diagnose 2026-05-07

Feedback loop:

```text
H200 20-frame decoder-only closed-loop smoke:
token replay + official_obs history init + root_qpos from captured SONIC clip
```

Initial symptom:

```text
Without SONIC motor config:
base_z: 0.7887014746665955 -> 0.2644214928150177
ACTION_MAX_ABS: 8.832411766052246
HEIGHT_OK_RANGE 0.3 1.2 False
```

Hypotheses tested:

- Online observation drift: probe compared Genesis-generated 994D obs against
  captured official obs by field.
- Base angular velocity order/frame mismatch: replacing only
  `his_base_angular_velocity_10frame_step1` with official values did not
  recover height.
- Joint velocity mismatch: replacing only
  `his_body_joint_velocities_10frame_step1` reduced action drift but did not
  recover height.
- Last-action feedback mismatch: replacing only `his_last_actions_10frame_step1`
  reduced action drift but did not recover height.
- Teacher forcing all official obs still fell when motor config was missing,
  so the immediate failure was not the decoder input layout.

Root cause found:

```text
The closed-loop smoke/probe/GIF tools used Genesis default motor control gains.
The passing action-replay path applied SONIC's official G1 kp/kv/force_range.
Raw SONIC policy targets without the matching motor config sagged/fell even
under teacher-forced official actions.
```

Fix:

```text
genesis_sonic_policy_rollout_smoke.py
genesis_sonic_policy_rollout_gif.py
genesis_sonic_policy_rollout_probe.py

Default now applies apply_sonic_g1_motor_config(...), with
--no-sonic-motor-config retained as an explicit negative-control/debug option.
```

H200 verification after fix:

```text
MOTOR_CONFIG sonic_g1_kp_kv_force_range
FRAME 0 base_z 0.788633406162262 action_max_abs 0.824930727481842
FRAME 5 base_z 0.751387357711792 action_max_abs 1.555012583732605
FRAME 10 base_z 0.7731913924217224 action_max_abs 2.3254857063293457
FRAME 15 base_z 0.785487949848175 action_max_abs 2.797891616821289
FRAME 19 base_z 0.7882418632507324 action_max_abs 3.2313764095306396
OBS_FINITE True
ACTION_FINITE True
BASE_HEIGHT_MIN 0.7512305378913879
BASE_HEIGHT_FINAL 0.7882418632507324
MAX_ABS_QVEL 4.189242839813232
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_SONIC_POLICY_ROLLOUT_SMOKE_OK
```

Review: the original 20-frame closed-loop failure no longer reproduces when the
tools use the official SONIC motor gains/force limits. Longer-horizon walking
quality is still a separate validation item; this only closes the 20-frame
smoke failure.
