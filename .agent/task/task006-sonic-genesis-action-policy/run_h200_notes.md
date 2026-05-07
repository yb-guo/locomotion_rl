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

## Walking Capture And Triage 2026-05-07

Added a repeatable official capture harness:

```text
.agent/task/task006-sonic-genesis-action-policy/capture_official_walking.sh
```

The harness creates a walking-only reference directory that symlinks the
official `walking_quip_360_R_002__A428` clip, then runs the official deploy
binary with the same H200 MuJoCo sim2sim stack and sends:

```text
]  start control
T  play walking motion 0
O  stop
```

Official walking capture result:

```text
Deploy status: 0
Policy input CSV:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_policy_input_walking_capture.csv
Rows: 922
Dims: [994]
Finite: True

Target motion CSV:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_target_motion_walking_capture.csv
Rows: 922
Dims: [36]
Target xy path length: 10.128800215398531
Target net xy displacement: 0.0
```

The target motion has a long xy path but returns close to the start, consistent
with the `walking_quip_360` clip.

Genesis decoder-only closed-loop with official walking obs/token replay:

```text
Frames: 200
ROOT_X_START 0.002389000030234456
ROOT_Y_START 0.011727999895811081
ROOT_X_FINAL 0.01647598296403885
ROOT_Y_FINAL -0.03096807934343815
HORIZONTAL_DISPLACEMENT 0.04495996297353397
PATH_LENGTH_XY 0.18772288302671417
AVERAGE_SPEED_XY 0.011239990743383492
ROOT_Z_MIN 0.7476961612701416
ROOT_Z_FINAL 0.7856734991073608
LEFT_CONTACT_FRAMES 196
RIGHT_CONTACT_FRAMES 196
SINGLE_SUPPORT_FRAMES 0
DOUBLE_SUPPORT_FRAMES 196
TRANSLATION_OBSERVED False
LOCOMOTION_OBSERVED False
GENESIS_SONIC_POLICY_LOCOMOTION_PROBE_OK
```

Review: the walking token/obs capture did not turn the current Genesis
decoder-only closed loop into locomotion. It remains upright, but has negligible
net translation and no single-support gait.

Teacher-forced action replay from the same official walking obs:

```text
Decoder export:
official_policy_input_walking_capture.csv -> official_walking_decoder_actions_300f.csv
Rows decoded: 300
Action dim: 29
Action finite: True
Action min/max: -12.89730453491211 / 10.547945022583008

Genesis action replay, first 50 frames:
ACTION_MODE sonic_policy_raw
ACTION_MAX_ABS 5.41317034
FRAME 0 base_z 0.7885566353797913
FRAME 49 base_z 0.2155306339263916
BASE_HEIGHT_MIN 0.14942580461502075
BASE_HEIGHT_FINAL 0.2155306339263916
HEIGHT_OK_RANGE 0.3 1.2 False
```

Review: raw walking policy actions decoded from official MuJoCo observations
are not stable when directly teacher-forced into Genesis. The failure is not
just a missing walking token; the Genesis policy-action replay path still has a
dynamics/contact/feedback mismatch for walking-scale actions.

Added root-motion metrics to:

```text
python -m h200_locomotion_lab.tools.sonic_reference_replay_smoke
```

Verification:

```text
Local related pytest: 18 passed
H200 test_genesis_action_replay_smoke.py: 13 passed
```

Official walking reference joint-position replay in Genesis:

```text
Reference: walking_quip_360_R_002__A428
Frames: 50
ROOT_X_START 0.00381189095787704
ROOT_Y_START 0.010537360794842243
ROOT_X_FINAL -0.04089926555752754
ROOT_Y_FINAL 0.1428152471780777
HORIZONTAL_DISPLACEMENT 0.13962996362873248
PATH_LENGTH_XY 0.19400965954206936
AVERAGE_SPEED_XY 0.13962996362873248
BASE_HEIGHT_MIN 0.7714895009994507
BASE_HEIGHT_FINAL 0.7714895009994507
HEIGHT_OK_RANGE 0.3 1.2 True
SONIC_REFERENCE_REPLAY_GENESIS_SMOKE_OK
```

Review: Genesis can produce non-trivial root motion when driven by the official
walking reference joint positions. The current blocker is therefore narrower:
the SONIC policy-action path in Genesis does not yet reproduce the official
walking behavior, even when given official walking obs/token rows. Next work
should compare the generated motor targets, contact forces, and online history
against official MuJoCo for the same walking window, or move to a fuller
planner/encoder integration that keeps the target motion and policy feedback in
phase.

## Diagnose Root Pose 2026-05-07

Added focused instrumentation to `genesis_action_replay_smoke.py`:

- optional physical reset pose via `--initial-joint-pos-csv`;
- optional 29D reference q CSV via `--reference-joint-pos-csv`;
- per-frame policy-target tracking error and target-vs-reference error.

Verification:

```text
Local targeted pytest: 34 passed
H200 targeted pytest: 34 passed
```

Extracted pure 29D official walking logs:

```text
Script:
.agent/task/task006-sonic-genesis-action-policy/extract_official_log_29d.py

Actions:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_walking_raw_actions_log_300f.csv
rows: 300
finite: True
row0_absmax: 0.0
min/max: -12.897335052 / 10.547904968

Measured q:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_walking_q_log_300f.csv
rows: 300
finite: True
min/max: -1.419189334 / 1.354117393
```

Action-row alignment was tested by replaying the official `action.csv` rows
directly, including row 0 zero action. With the old root quaternion
`0.711231 -0.00883 -0.004562 -0.702888`, the 50-frame replay still fell:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_official_walking_raw_log_actions_50f_tracking.log

BASE_HEIGHT_MIN 0.07961206138134003
BASE_HEIGHT_FINAL 0.10619580000638962
HEIGHT_OK_RANGE 0.3 1.2 False
MEAN_ABS_TARGET_TRACKING_ERROR_AVG 0.24277533566380488
MEAN_ABS_REFERENCE_TRACKING_ERROR_AVG 0.12572087241865232
MEAN_ABS_TARGET_REFERENCE_ERROR_AVG 0.22703592721239374
```

Using official measured q0 as the physical reset pose did not fix the old-root
run:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_official_walking_raw_log_actions_50f_official_q0_init_tracking.log

BASE_HEIGHT_MIN 0.16823244094848633
BASE_HEIGHT_FINAL 0.19532467424869537
HEIGHT_OK_RANGE 0.3 1.2 False
MEAN_ABS_TARGET_TRACKING_ERROR_AVG 0.2502752844080102
MEAN_ABS_REFERENCE_TRACKING_ERROR_AVG 0.1218093288250086
MEAN_ABS_TARGET_REFERENCE_ERROR_AVG 0.22703592721239374
```

The walking capture's actual `base_quat.csv` row 0 is near identity:

```text
base_q = 0.999910712 -0.006119614 0.011878765 0.000052081
```

Replacing only the root quaternion with that walking capture quaternion, while
keeping official `action.csv` row alignment and measured q0 initialization,
changed the 50-frame replay from falling to passing the height smoke:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_official_walking_raw_log_actions_50f_walking_base_quat_tracking.log

ROOT_QPOS (0.002389, 0.011728, 0.791166, 0.999910712, -0.006119614, 0.011878765, 5.2081e-05)
FRAME 49 base_z 0.3062208294868469
BASE_HEIGHT_MIN 0.3062208294868469
BASE_HEIGHT_FINAL 0.3062208294868469
HEIGHT_OK_RANGE 0.3 1.2 True
MEAN_ABS_TARGET_TRACKING_ERROR_AVG 0.2502038281348184
MEAN_ABS_REFERENCE_TRACKING_ERROR_AVG 0.11873773822838386
MEAN_ABS_TARGET_REFERENCE_ERROR_AVG 0.22703592721239374
GENESIS_ACTION_REPLAY_SMOKE_OK
```

Diagnosis conclusion: the earlier official walking action replay failure was
primarily caused by using a root quaternion from the wrong capture. It was not
caused by the decoder, missing action row 0, or the physical motor q0 alone.
The replay is still marginal at 50 frames, so this is a smoke pass, not a
walking-quality pass.

L2 decoder-only locomotion probe with the corrected walking root quaternion and
official q0 was started for 100 frames:

```text
Script:
.agent/task/task006-sonic-genesis-action-policy/run_h200_walking_locomotion_probe.sh

Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_walking_obs_100f_walking_base_quat_q0_locomotion.log

FRAME 20 disp_xy 0.1339092214345273 path_xy 0.13691639215811013 root_z 0.760623037815094
FRAME 40 disp_xy 0.17986304279316542 path_xy 0.19964767986303977 root_z 0.7679933309555054
FRAME 60 disp_xy 0.21374395871356025 path_xy 0.2687822544722506 root_z 0.7868149876594543 left_contact True right_contact False
FRAME 80 disp_xy 0.24342315138543422 path_xy 0.30610471773944953 root_z 0.7728626728057861
```

The SSH session was closed by the remote host before the 100-frame summary
printed. A shorter 80-frame retry was interrupted after frame 0, and then the
H200 SSH endpoint started refusing connections. Treat the L2 result as
promising partial evidence only; it is not a pass until a complete run prints
the final `GENESIS_SONIC_POLICY_LOCOMOTION_PROBE_OK` summary.

After commit `0ecb2cf`, an 80-frame `nohup` retry was started so that SSH
disconnects would not kill the job:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_walking_obs_80f_walking_base_quat_q0_locomotion_nohup.log
```

That retry printed only frame 0 and then the Python process stayed at 99% CPU
for roughly 30 minutes:

```text
pid: 221946
command: python3 -u -m h200_locomotion_lab.tools.genesis_sonic_policy_locomotion_probe ...
```

The stuck process was killed. Do not rerun long L2 probes unchanged. The next
attempt should add an internal per-frame timeout/progress heartbeat or reduce
the tool to an action-export + shorter Genesis stepping loop before attempting
another 80/100-frame run.
