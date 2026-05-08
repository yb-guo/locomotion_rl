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

## L2 Short Walking Pass 2026-05-08

Added safer L2 probe controls:

- line-buffered output and per-frame stage heartbeats;
- optional `--progress-file`;
- optional `--skip-foot-metrics`;
- `--max-wall-time-s` guard checked between frames.

Updated helper script:

```text
.agent/task/task006-sonic-genesis-action-policy/run_h200_walking_locomotion_probe.sh
```

Verification:

```text
Local related pytest: 36 passed
H200 test_genesis_sonic_policy_locomotion_probe.py: 7 passed
```

5-frame heartbeat isolation, skipping foot metrics:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_walking_obs_5f_heartbeat_skip_foot.log

Result:
GENESIS_SONIC_POLICY_LOCOMOTION_PROBE_OK

Observation:
Each frame advanced. The expensive stage is `backend.step`, around 4.8-5.0 s
per policy frame on this H200/Genesis setup. No hang reproduced.
```

10-frame heartbeat run with foot/contact metrics enabled:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_walking_obs_10f_heartbeat_metrics.log

HORIZONTAL_DISPLACEMENT 0.0841102183964299
PATH_LENGTH_XY 0.08489193449579213
AVERAGE_SPEED_XY 0.4205510919821495
ROOT_Z_MIN 0.7292487621307373
ROOT_Z_FINAL 0.73612380027771
SINGLE_SUPPORT_FRAMES 3
DOUBLE_SUPPORT_FRAMES 5
NO_SUPPORT_FRAMES 2
TOTAL_CONTACT_SWITCHES 3
TRANSLATION_OBSERVED True
FOOT_ALTERNATION_OBSERVED True
LOCOMOTION_OBSERVED True
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_SONIC_POLICY_LOCOMOTION_PROBE_OK
```

20-frame walking probe:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_walking_obs_20f_metrics.log

HORIZONTAL_DISPLACEMENT 0.1291484013080984
PATH_LENGTH_XY 0.1319194914726968
AVERAGE_SPEED_XY 0.322871003270246
ROOT_Z_MIN 0.7292487025260925
ROOT_Z_FINAL 0.7582225203514099
SINGLE_SUPPORT_FRAMES 7
DOUBLE_SUPPORT_FRAMES 11
NO_SUPPORT_FRAMES 2
TOTAL_CONTACT_SWITCHES 4
TRANSLATION_OBSERVED True
FOOT_ALTERNATION_OBSERVED True
LOCOMOTION_OBSERVED True
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_SONIC_POLICY_LOCOMOTION_PROBE_OK
```

40-frame walking probe:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_walking_obs_40f_metrics.log

HORIZONTAL_DISPLACEMENT 0.1781279369041812
PATH_LENGTH_XY 0.1955194149409191
AVERAGE_SPEED_XY 0.22265992113022648
ROOT_Z_MIN 0.7292487025260925
ROOT_Z_FINAL 0.7679193019866943
SINGLE_SUPPORT_FRAMES 15
DOUBLE_SUPPORT_FRAMES 23
NO_SUPPORT_FRAMES 2
TOTAL_CONTACT_SWITCHES 6
TRANSLATION_OBSERVED True
FOOT_ALTERNATION_OBSERVED True
LOCOMOTION_OBSERVED True
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_SONIC_POLICY_LOCOMOTION_PROBE_OK
```

Visual evidence:

```text
Remote GIF:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/videos/genesis_g1_sonic_walking_obs_40f_correct_root.gif

Local GIF:
.agent/task/task006-sonic-genesis-action-policy/artifacts/genesis_g1_sonic_walking_obs_40f_correct_root.gif

Frames: 40
Resolution: 420 x 320
GIF bytes: 101840
BASE_HEIGHT_MIN 0.7292487025260925
BASE_HEIGHT_FINAL 0.767897367477417
GENESIS_SONIC_POLICY_ROLLOUT_GIF_OK
```

Review: L2 decoder-only SONIC walking replay now has complete H200 evidence
through 40 frames with corrected walking root pose, official q0 reset,
SONIC motor config, and official walking obs/token replay. This is a short
walking pass, not yet a long-horizon policy rollout. The next scale-up should
target 80/100 frames with the same heartbeat/progress controls and should keep
the corrected walking root quaternion.

80-frame scale-up completed with the same corrected configuration:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_walking_obs_80f_metrics.log

HORIZONTAL_DISPLACEMENT 0.24235538459989664
PATH_LENGTH_XY 0.3029986364942937
AVERAGE_SPEED_XY 0.15147211537493538
ROOT_Z_MIN 0.7292487621307373
ROOT_Z_FINAL 0.7751022577285767
SINGLE_SUPPORT_FRAMES 28
DOUBLE_SUPPORT_FRAMES 50
NO_SUPPORT_FRAMES 2
TOTAL_CONTACT_SWITCHES 13
TRANSLATION_OBSERVED True
FOOT_ALTERNATION_OBSERVED True
LOCOMOTION_OBSERVED True
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_SONIC_POLICY_LOCOMOTION_PROBE_OK
```

Review: the previous 80-frame hang did not reproduce with the guarded foreground
runner. L2 decoder-only walking now has complete H200 pass evidence through 80
frames. The remaining next step is 100/200-frame scale-up and then replacing
replayed official obs/token rows with the full SONIC encoder/planner path.

## L2 200-Frame Walking Scale-Up 2026-05-08

H200 command class:

```text
FRAMES=200 LOG_EVERY=50 PROGRESS_FILE=/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/progress_l2_200f_metrics.txt MAX_WALL_TIME_S=2400 /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/run_h200_walking_locomotion_probe.sh
```

Numeric log:

```text
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_walking_obs_200f_metrics.log
```

Result:

```text
FRAME 0 root_x 0.0023871322628110647 root_y 0.011728079989552498 root_z 0.7877528071403503 disp_xy 1.8694489130984784e-06 path_xy 1.8694489130984784e-06
FRAME 50 root_x 0.1955728381872177 root_y 0.057338517159223557 root_z 0.7777901887893677 disp_xy 0.198495124868869 path_xy 0.24467578403298504
FRAME 100 root_x 0.25905516743659973 root_y 0.11122798919677734 root_z 0.7843438982963562 disp_xy 0.2752776222579749 path_xy 0.3470539645304823
FRAME 150 root_x 0.3024607300758362 root_y 0.05134880542755127 root_z 0.8023837804794312 disp_xy 0.3026761493960159 path_xy 0.4232554565852338
FRAME 199 root_x 0.3019540011882782 root_y 0.08134350925683975 root_z 0.7878252863883972 disp_xy 0.3075475720314082 path_xy 0.5344631470862223
OBS_FINITE True
ACTION_FINITE True
ROOT_Z_MIN 0.7292487025260925
ROOT_Z_MAX 0.8026557564735413
HORIZONTAL_DISPLACEMENT 0.3075475720314082
PATH_LENGTH_XY 0.5344631470862223
AVERAGE_SPEED_XY 0.07688689300785205
YAW_DELTA 0.03219214842257979
ACTION_MAX_ABS 5.8859028816223145
LEFT_CONTACT_FRAMES 183
RIGHT_CONTACT_FRAMES 174
LEFT_CONTACT_SWITCHES 7
RIGHT_CONTACT_SWITCHES 11
TOTAL_CONTACT_SWITCHES 18
SINGLE_SUPPORT_FRAMES 39
DOUBLE_SUPPORT_FRAMES 159
NO_SUPPORT_FRAMES 2
TRANSLATION_OBSERVED True
FOOT_ALTERNATION_OBSERVED True
LOCOMOTION_OBSERVED True
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_SONIC_POLICY_LOCOMOTION_PROBE_OK
```

200-frame GIF:

```text
Remote:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/videos/genesis_g1_sonic_walking_obs_200f_correct_root.gif

Local:
.agent/task/task006-sonic-genesis-action-policy/artifacts/genesis_g1_sonic_walking_obs_200f_correct_root.gif

Rendered frames: 200
Resolution: 420 x 320
GIF bytes: 460316
BASE_HEIGHT_MIN 0.7292487025260925
BASE_HEIGHT_FINAL 0.7877992391586304
ACTION_MAX_ABS 5.885979652404785
GENESIS_SONIC_POLICY_ROLLOUT_GIF_OK
```

Review: L2 decoder-only SONIC walking replay now has complete 200-frame H200
evidence with corrected walking root pose, official q0 reset, SONIC motor
config, official walking obs replay, and replayed SONIC token. This validates
the Genesis action/decoder loop for a replayed walking clip. It is still not a
full SONIC rollout because the online encoder/planner path is not yet replacing
the replayed obs/token source.

## SONIC Planner/Encoder Bridge Start 2026-05-08

ONNX metadata inspection on H200:

```text
model_encoder.onnx:
INPUT obs_dict (1, 1762)
OUTPUT encoded_tokens (1, 64)

model_decoder.onnx:
INPUT obs_dict (1, 994)
OUTPUT action (1, 29)

planner_sonic.onnx:
INPUT context_mujoco_qpos (1, 4, 36)
INPUT target_vel (1,)
INPUT mode (1,)
INPUT movement_direction (1, 3)
INPUT facing_direction (1, 3)
INPUT random_seed (1,)
INPUT has_specific_target (1, 1)
INPUT specific_target_positions (1, 4, 3)
INPUT specific_target_headings (1, 4)
INPUT allowed_pred_num_tokens (1, 11)
INPUT height (1,)
OUTPUT mujoco_qpos (1, 64, 36)
OUTPUT num_pred_frames (1,)
```

Official C++ route confirmed:

- planner output is 30 Hz `mujoco_qpos`;
- official deploy resamples it to 50 Hz before encoder observations;
- encoder G1 mode 0 fills only `encoder_mode_4`,
  `motion_joint_positions_10frame_step5`,
  `motion_joint_velocities_10frame_step5`, and
  `motion_anchor_orientation_10frame_step5` inside the full 1762D encoder
  input; other enabled encoder fields are left zero by mode filtering.

Python `onnx.reference.ReferenceEvaluator` is not usable for the planner model:
the direct planner->encoder->decoder forward command stayed at about 103% CPU
for 10 minutes with an empty log and had to be killed. The fix was to use the
H200 system ONNX Runtime C++ install at `/opt/onnxruntime` for the planner and
keep Python ONNX ReferenceEvaluator for the smaller encoder/decoder models.

C++ planner runner evidence:

```text
Source:
.agent/task/task006-sonic-genesis-action-policy/sonic_planner_ort_runner.cpp

Remote binary:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/sonic_planner_ort_runner

Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/sonic_planner_ort_runner_walk.log

OUTPUT_QPOS_CSV /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/planner_ort_walk_qpos.csv
PLANNER_MODE 2
TARGET_VEL -1
PLANNER_QPOS_ROWS 64
PLANNER_QPOS_COLS 36
PLANNER_NUM_PRED_FRAMES 44
PLANNER_QPOS_FINITE 1
PLANNER_ROOT_Z_MIN_MAX 0.750778 0.787849
SONIC_PLANNER_ORT_RUNNER_OK
```

Planner->encoder->decoder forward without replayed obs/token:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/sonic_planner_encoder_decoder_forward_walk_20tokens.log

PLANNER_SOURCE csv
PLANNER_QPOS_CSV /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/planner_ort_walk_qpos.csv
REPLAY_OBS_USED False
REPLAY_TOKEN_USED False
PLANNER_NUM_PRED_FRAMES 44
MOTION_50HZ_TIMESTEPS 73
ENCODER_OBS_DIM 1762
ENCODER_OBS_ROWS 20
ENCODER_OBS_FINITE True
TOKEN_ROWS_GENERATED 20
TOKEN_DIM 64
TOKEN_FINITE True
DECODER_OBS_DIM 994
DECODER_OBS_FINITE True
ACTION_DIM 29
ACTION_FINITE True
ACTION_MAX_ABS 1.9110764265060425
OUTPUT_TOKEN_CSV /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/planner_encoder_token_walk_20f.csv
SONIC_PLANNER_ENCODER_DECODER_FORWARD_OK
```

Genesis smoke using planner+encoder generated token sequence, not official obs:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_planner_encoder_token_20f.log

TOKEN_SOURCE /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/planner_encoder_token_walk_20f.csv
OBS_SOURCE not_set
TOKEN_MODE replay
TOKEN_ROWS 20
HISTORY_INIT genesis
MOTOR_CONFIG sonic_g1_kp_kv_force_range
FRAMES 20
OBS_FINITE True
ACTION_FINITE True
ROOT_Z_MIN 0.7579326629638672
ROOT_Z_FINAL 0.759788453578949
HORIZONTAL_DISPLACEMENT 0.10920919963428287
PATH_LENGTH_XY 0.11964258357025515
AVERAGE_SPEED_XY 0.27302299908570715
ACTION_MAX_ABS 2.9726450443267822
SINGLE_SUPPORT_FRAMES 10
DOUBLE_SUPPORT_FRAMES 8
NO_SUPPORT_FRAMES 2
TOTAL_CONTACT_SWITCHES 4
TRANSLATION_OBSERVED True
FOOT_ALTERNATION_OBSERVED True
LOCOMOTION_OBSERVED True
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_SONIC_POLICY_LOCOMOTION_PROBE_OK
```

Review: the replayed official obs/token source has now been removed for a
short 20-frame Genesis smoke. The remaining gap is true online replanning:
planner qpos is still generated once up front, then encoded into a 20-token
sequence. The next step is to wrap planner+encoder as a runtime token provider
inside the Genesis rollout loop and refresh planner context from the simulated
motion/state instead of using a precomputed planner qpos CSV.

## Online Planner/Encoder Rollout 2026-05-08

Implemented the runtime path:

```text
Genesis state/history -> planner motion context -> C++ ONNX Runtime planner
-> 50 Hz planner motion -> Python encoder -> token -> decoder -> Genesis action
```

The C++ planner runner now accepts:

```text
--context-qpos-csv <4 x 36 MuJoCo qpos rows>
```

This enables replan calls from the live rollout loop instead of only using an
initial standing/walking q0 context.

Verification:

```text
Local full pytest: 66 passed
H200 targeted pytest from /tmp with absolute PYTHONPATH: 8 passed
H200 C++ runner compile: passed
```

H200 20-frame online rollout, one planner call:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_planner_encoder_runtime_20f.log

GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_MODE online_planner_encoder
REPLAY_OBS_USED False
REPLAY_TOKEN_USED False
PLANNER_CALLS 1
ENCODER_OBS_FINITE True
TOKEN_FINITE True
DECODER_OBS_FINITE True
ACTION_FINITE True
FINITE_OK True
ROOT_Z_MIN 0.7505621910095215
ROOT_Z_MAX 0.7911660075187683
HORIZONTAL_DISPLACEMENT 0.09537745650207112
PATH_LENGTH_XY 0.11395758913489372
SINGLE_SUPPORT_FRAMES 13
TOTAL_CONTACT_SWITCHES 4
LOCOMOTION_OBSERVED True
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_OK
```

H200 30-frame online rollout with replan every 10 frames:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_planner_encoder_runtime_replan10_30f.log

REPLAN 10 planner_calls 2 num_pred_frames 44 motion_timesteps 73
REPLAN 20 planner_calls 3 num_pred_frames 44 motion_timesteps 73
PLANNER_CALLS 3
ENCODER_OBS_FINITE True
TOKEN_FINITE True
DECODER_OBS_FINITE True
ACTION_FINITE True
FINITE_OK True
ROOT_Z_MIN 0.7509679198265076
ROOT_Z_MAX 0.7911660075187683
HORIZONTAL_DISPLACEMENT 0.22500730455222778
PATH_LENGTH_XY 0.24270576243687916
SINGLE_SUPPORT_FRAMES 22
TOTAL_CONTACT_SWITCHES 5
LOCOMOTION_OBSERVED True
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_OK
```

Review: the SONIC planner/encoder path is now connected inside the Genesis
rollout loop, and replayed official obs/token rows are no longer used. The
30-frame replan smoke proves the loop can refresh planner context and continue
walking under Genesis. Remaining work is scale-up and correctness hardening:
the replan context is sampled from the planner motion currently being tracked,
not yet reconciled against the simulated robot root/motor state in the same way
the official deployment stack may do.

## Genesis Feedback Planner Context 2026-05-08

Updated the online rollout probe so planner context can come from the live
Genesis state:

```text
Genesis root qpos + Genesis 29 motor qpos
-> 50 Hz qpos history
-> resampled 4 x 36 MuJoCo qpos context at 30 Hz
-> C++ ONNX Runtime planner
```

New probe controls:

```text
--initial-context-source genesis
--replan-context-source genesis
```

Verification:

```text
Local full pytest: 69 passed
H200 targeted pytest from /tmp with absolute PYTHONPATH: 11 passed
```

H200 30-frame Genesis-feedback closed-loop, replan every 10 frames:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_planner_encoder_runtime_genesisctx_replan10_30f.log

INITIAL_CONTEXT_SOURCE genesis
REPLAN_CONTEXT_SOURCE genesis
PLANNER_CALLS 3
GENESIS_QPOS_HISTORY_FRAMES 31
FINITE_OK True
ROOT_Z_MIN 0.7393287420272827
ROOT_Z_MAX 0.7911660075187683
HORIZONTAL_DISPLACEMENT 0.19177099548522342
PATH_LENGTH_XY 0.20562466234576518
SINGLE_SUPPORT_FRAMES 20
TOTAL_CONTACT_SWITCHES 5
LOCOMOTION_OBSERVED True
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_OK
```

H200 80-frame Genesis-feedback closed-loop, replan every 10 frames:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_planner_encoder_runtime_genesisctx_replan10_80f.log

INITIAL_CONTEXT_SOURCE genesis
REPLAN_CONTEXT_SOURCE genesis
PLANNER_CALLS 8
GENESIS_QPOS_HISTORY_FRAMES 81
FINITE_OK True
ROOT_Z_MIN 0.7064436078071594
ROOT_Z_MAX 0.7911660075187683
HORIZONTAL_DISPLACEMENT 0.8308150532633347
PATH_LENGTH_XY 0.8759600709577173
SINGLE_SUPPORT_FRAMES 51
TOTAL_CONTACT_SWITCHES 10
LOCOMOTION_OBSERVED True
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_OK
```

H200 120-frame Genesis-feedback closed-loop, replan every 10 frames:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_planner_encoder_runtime_genesisctx_replan10_120f.log

INITIAL_CONTEXT_SOURCE genesis
REPLAN_CONTEXT_SOURCE genesis
PLANNER_CALLS 12
GENESIS_QPOS_HISTORY_FRAMES 121
FINITE_OK True
ROOT_Z_MIN 0.7064440250396729
ROOT_Z_MAX 0.7911660075187683
HORIZONTAL_DISPLACEMENT 1.2903014294589257
PATH_LENGTH_XY 1.3514448254283475
SINGLE_SUPPORT_FRAMES 76
TOTAL_CONTACT_SWITCHES 12
LOCOMOTION_OBSERVED True
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_OK
```

Review: this is now a Genesis-feedback planner context loop, not just planner
motion self-sampling. The 120-frame result is stable under the current height
and locomotion gates. Remaining work is visual evidence for the online path,
200-frame scale-up, and closer parity checks against the official deployment
context convention.

## Online 200-Frame Scale-Up And Visual 2026-05-08

Added optional rendering to the same online planner/encoder rollout probe:

```text
--output-gif <path>
--output-mp4 <path>
```

Rendering happens inside the same loop as the numeric probe, after each Genesis
step, so the visual artifact corresponds to the Genesis-feedback planner context
run rather than a separate replay path.

H200 200-frame Genesis-feedback closed-loop, no rendering:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_planner_encoder_runtime_genesisctx_replan10_200f.log

INITIAL_CONTEXT_SOURCE genesis
REPLAN_CONTEXT_SOURCE genesis
PLANNER_CALLS 20
GENESIS_QPOS_HISTORY_FRAMES 201
FINITE_OK True
ROOT_Z_MIN 0.6891541481018066
ROOT_Z_MAX 0.7911660075187683
HORIZONTAL_DISPLACEMENT 2.249557914799456
PATH_LENGTH_XY 2.346827985821677
AVERAGE_SPEED_XY 0.562389478699864
ACTION_MAX_ABS 4.365278720855713
SINGLE_SUPPORT_FRAMES 128
TOTAL_CONTACT_SWITCHES 18
LOCOMOTION_OBSERVED True
HEIGHT_OK_RANGE 0.3 1.2 True
GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_OK
```

H200 200-frame Genesis-feedback closed-loop with GIF/MP4 rendering:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_planner_encoder_runtime_genesisctx_replan10_200f_render.log

Remote GIF:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/videos/genesis_g1_sonic_planner_encoder_genesisctx_replan10_200f.gif

Remote MP4:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/videos/genesis_g1_sonic_planner_encoder_genesisctx_replan10_200f.mp4

Local GIF:
.agent/task/task006-sonic-genesis-action-policy/artifacts/genesis_g1_sonic_planner_encoder_genesisctx_replan10_200f.gif

Local MP4:
.agent/task/task006-sonic-genesis-action-policy/artifacts/genesis_g1_sonic_planner_encoder_genesisctx_replan10_200f.mp4

RENDERED_FRAMES 200
GIF_BYTES 418700
MP4_BYTES 56405
GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_OK
```

The rendered run reproduced the same stability envelope:

```text
PLANNER_CALLS 20
GENESIS_QPOS_HISTORY_FRAMES 201
FINITE_OK True
ROOT_Z_MIN 0.6891546845436096
HORIZONTAL_DISPLACEMENT 2.249562539745548
PATH_LENGTH_XY 2.34683346652748
SINGLE_SUPPORT_FRAMES 128
TOTAL_CONTACT_SWITCHES 18
LOCOMOTION_OBSERVED True
HEIGHT_OK_RANGE 0.3 1.2 True
```

## Official Planner Context Parity Check 2026-05-08

Checked the H200 official SONIC deploy source:

```text
/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner.hpp
```

Relevant official behavior:

- `PlannerConfig.motion_look_ahead_steps = 2`.
- `InitializeContext(...)` uses live robot joint positions, but normalizes root
  to `x=0`, `y=0`, `z=default_height`, and identity quaternion.
- `UpdatePlanning(...)` sets `gen_frame_ = gen_frame + motion_look_ahead_steps`.
- `UpdateContextFromMotion(...)` samples 4 context rows at 30 Hz starting at
  `gen_frame_/50.0`, from a `MotionSequence`.
- The sampled context is `[root xyz, root quat wxyz, 29 joints]`.
- Joints are written into MuJoCo order with
  `context_qpos[7 + mujoco_to_isaaclab[i]] = motion_sequence->JointPositions(...)[i]`.

Official call-site behavior in:

```text
/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp
```

- Initial planner setup reads `base_quat` and 29 motor q from `LowState`, but
  `InitializeContext` only uses joint q for context; root context is normalized.
- Replanning calls `planner_->UpdatePlanning(current_frame_, planner_motion_, ...)`.
  That means the official replan context comes from `planner_motion_`, not
  directly from live low-level robot qpos.

Parity conclusion:

```text
initial_context_source=initial_joint_csv + replan_context_source=motion
```

is the closest current path to the official planner context convention.

```text
initial_context_source=genesis + replan_context_source=genesis
```

is intentionally a stronger Genesis-feedback closed-loop experiment, but it is
not exactly the official planner context convention because it feeds live
Genesis root/motor qpos history directly to planner context during replanning.

Review: the new 200-frame Genesis-feedback result is a valid closed-loop
stability result for this repo's Genesis integration, but it should not be
described as bit-for-bit or convention-exact official SONIC planner context.
For official parity, keep the motion-context route as the reference path and use
the Genesis-context route as a feedback robustness experiment.

## Official-Context Online 200-Frame Run 2026-05-08

Ran the closest current path to official SONIC planner context convention:

```text
--initial-context-source initial_joint_csv
--replan-context-source motion
--replan-interval 10
--frames 200
```

This path keeps planner replanning context sourced from the planner motion
buffer, matching the official deployment direction more closely than the
Genesis-feedback context experiment.

H200 200-frame official-context run with GIF/MP4 rendering:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/genesis_g1_sonic_planner_encoder_runtime_officialctx_replan10_200f_render.log

INITIAL_CONTEXT_SOURCE initial_joint_csv
REPLAN_CONTEXT_SOURCE motion
PLANNER_CALLS 20
GENESIS_QPOS_HISTORY_FRAMES 201
FINITE_OK True
TOKEN_MAX_ABS 0.625
ACTION_MAX_ABS 5.80940055847168
ROOT_Z_MIN 0.7183108925819397
ROOT_Z_MAX 0.7911660075187683
HORIZONTAL_DISPLACEMENT 3.6676499479337363
PATH_LENGTH_XY 3.7301093757850285
AVERAGE_SPEED_XY 0.9169124869834341
SINGLE_SUPPORT_FRAMES 151
DOUBLE_SUPPORT_FRAMES 35
NO_SUPPORT_FRAMES 14
TOTAL_CONTACT_SWITCHES 31
LOCOMOTION_OBSERVED True
HEIGHT_OK_RANGE 0.3 1.2 True
RENDERED_FRAMES 200
GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_OK
```

Visual artifacts:

```text
Remote GIF:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/videos/genesis_g1_sonic_planner_encoder_officialctx_replan10_200f.gif

Remote MP4:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/videos/genesis_g1_sonic_planner_encoder_officialctx_replan10_200f.mp4

Local GIF:
.agent/task/task006-sonic-genesis-action-policy/artifacts/genesis_g1_sonic_planner_encoder_officialctx_replan10_200f.gif

Local MP4:
.agent/task/task006-sonic-genesis-action-policy/artifacts/genesis_g1_sonic_planner_encoder_officialctx_replan10_200f.mp4

GIF_BYTES 456551
MP4_BYTES 67771
```

Review: the official-context path now passes the same 200-frame rendered H200
scale-up gate. This should be treated as the primary transfer-oriented route.
The Genesis-feedback context route remains useful as a robustness/diagnostic
ablation, but official-context is the closer deployment convention.
