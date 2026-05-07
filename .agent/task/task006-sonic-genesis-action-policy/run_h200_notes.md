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

## Longer Horizon 2026-05-07

Command class:

```text
genesis_sonic_policy_rollout_smoke.py
token replay + official_obs history init + SONIC motor config
frames: 100
root_qpos: 0.002389 0.011728 0.791166 0.711231 -0.00883 -0.004562 -0.702888
```

H200 100-frame verification:

```text
MOTOR_CONFIG sonic_g1_kp_kv_force_range
FRAME 0 base_z 0.788633406162262 action_max_abs 0.824930727481842
FRAME 20 base_z 0.7884613275527954 action_max_abs 3.3407137393951416
FRAME 40 base_z 0.7871977090835571 action_max_abs 3.4064137935638428
FRAME 60 base_z 0.7874587178230286 action_max_abs 2.485649585723877
FRAME 80 base_z 0.7869091629981995 action_max_abs 2.2736780643463135
FRAME 99 base_z 0.7874253988265991 action_max_abs 2.059873580932617
OBS_FINITE True
ACTION_FINITE True
BASE_HEIGHT_MIN 0.7512305378913879
BASE_HEIGHT_FINAL 0.7874253988265991
MAX_ABS_QVEL 4.189242839813232
ACTION_MAX_ABS 3.6153788566589355
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_SONIC_POLICY_ROLLOUT_SMOKE_OK
```

100-frame visual evidence:

```text
Remote GIF:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/videos/genesis_g1_sonic_closed_loop_100f_motor_config.gif

Local GIF:
.agent/task/task006-sonic-genesis-action-policy/artifacts/genesis_g1_sonic_closed_loop_100f_motor_config.gif

Rendered frames: 100
Resolution: 420 x 320
FPS: 12
GIF bytes: 205932
```

Review: the 100-frame decoder-only Genesis rollout remains upright under the
same token replay and official captured history initialization used for the
20-frame smoke. This is still replay-token validation, not a full online SONIC
planner/encoder rollout.

## Locomotion Probe 2026-05-07

Added a dedicated probe:

```text
python -m h200_locomotion_lab.tools.genesis_sonic_policy_locomotion_probe
```

The probe keeps the same decoder-only rollout path, but reports locomotion
metrics instead of only height:

- root x/y/z/yaw
- horizontal displacement and xy path length
- average xy speed at the 50 Hz policy rate
- left/right ankle-roll link height and net contact force
- single-support, double-support, no-support frames
- contact switches

Local verification:

```text
tests/test_genesis_sonic_policy_locomotion_probe.py: 5 passed
related local pytest group: 28 passed
```

H200 targeted pytest:

```text
tests/test_genesis_sonic_policy_locomotion_probe.py: 5 passed
```

H200 100-frame locomotion probe, same token replay and initial root qpos:

```text
ROOT_X_START 0.002389000030234456
ROOT_Y_START 0.011727999895811081
ROOT_Z_START 0.7911660075187683
ROOT_X_FINAL 0.0030075262766331434
ROOT_Y_FINAL -0.013038256205618382
ROOT_Z_FINAL 0.7874254584312439
ROOT_Z_MIN 0.7512305378913879
ROOT_Z_MAX 0.7911660075187683
HORIZONTAL_DISPLACEMENT 0.024773978606575816
PATH_LENGTH_XY 0.08982350716347963
AVERAGE_SPEED_XY 0.012386989303287908
YAW_DELTA -0.08904474184344568
LEFT_CONTACT_FRAMES 96
RIGHT_CONTACT_FRAMES 96
LEFT_CONTACT_SWITCHES 1
RIGHT_CONTACT_SWITCHES 1
TOTAL_CONTACT_SWITCHES 2
SINGLE_SUPPORT_FRAMES 0
DOUBLE_SUPPORT_FRAMES 96
NO_SUPPORT_FRAMES 4
TRANSLATION_OBSERVED False
FOOT_ALTERNATION_OBSERVED True
LOCOMOTION_OBSERVED False
GENESIS_SONIC_POLICY_LOCOMOTION_PROBE_OK
```

H200 300-frame attempt:

```text
The SSH connection was closed by the remote host after about 10 minutes before
the buffered output returned. This was treated as infrastructure timeout, not a
probe pass/fail result.
```

H200 200-frame locomotion probe with unbuffered output:

```text
ROOT_X_START 0.002389000030234456
ROOT_Y_START 0.011727999895811081
ROOT_Z_START 0.7911660075187683
ROOT_X_FINAL 0.0051521193236112595
ROOT_Y_FINAL -0.009658973664045334
ROOT_Z_FINAL 0.7868996262550354
ROOT_Z_MIN 0.7512305378913879
ROOT_Z_MAX 0.7911660075187683
HORIZONTAL_DISPLACEMENT 0.021564727363902112
PATH_LENGTH_XY 0.16066903187788673
AVERAGE_SPEED_XY 0.005391181840975528
YAW_DELTA -0.09394057327026872
LEFT_CONTACT_FRAMES 196
RIGHT_CONTACT_FRAMES 196
LEFT_CONTACT_SWITCHES 1
RIGHT_CONTACT_SWITCHES 1
TOTAL_CONTACT_SWITCHES 2
SINGLE_SUPPORT_FRAMES 0
DOUBLE_SUPPORT_FRAMES 196
NO_SUPPORT_FRAMES 4
TRANSLATION_OBSERVED False
FOOT_ALTERNATION_OBSERVED True
LOCOMOTION_OBSERVED False
GENESIS_SONIC_POLICY_LOCOMOTION_PROBE_OK
```

Review: the user's visual read was correct. Current token replay validates a
stable standing/settling decoder-only closed loop, not walking. The next route
should find a captured SONIC walking command/token/action segment or connect
the full online SONIC command -> encoder/planner -> decoder path, then require
non-trivial root translation and single-support foot alternation as pass
criteria.
