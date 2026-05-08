# Route

Task: task009-sonic-action-rollout-matrix

Goal: run the official-context online SONIC walking rollout after task008.

Scope:

- `initial_context_source=initial_joint_csv`
- `replan_context_source=motion`
- profile-backed bridge from task008 code.
- Generate numeric summary and GIF/MP4 where practical.

Environment:

- interactive remote sessions must enter `/root/agent_workspace/safe_agent/agent_shell.sh`
- non-interactive remote commands must run through `/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/<project> && <command>'`
- code, generated files, and intermediate outputs must stay under `/root/agent_workspace/project`
- datasets must be accessed only through `/root/agent_workspace/datasets`
- dataset links must be created with `/root/agent_workspace/safe_agent/link_dataset.sh`
- do not write to or delete from `/mnt/workspace` or `/mnt/workspace1`
- do not manually clear `LD_PRELOAD`

Verify:

- `GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_OK`
- finite encoder/decoder/action flags
- stable root height
- locomotion/contact metrics
- artifact paths recorded

No Hack:

- Do not substitute replayed decoder obs/tokens for online planner/encoder.

Hardware: H200/Linux target via `myserver`.

# Log

- 2026-05-08: H200 official-context walking rollout completed through online
  SONIC planner -> encoder -> decoder -> profile-backed action bridge ->
  Genesis G1 execution.

  Guarded command wrapper:

```text
C:\Windows\System32\OpenSSH\ssh.exe myserver
"/root/agent_workspace/safe_agent/run_guarded.sh bash -lc '<inner command>'"
```

  Inner command:

```text
cd /root/agent_workspace/project/h200-locomotion-lab-task009-sonic-action-rollout-matrix &&
PYTHONPATH=/root/agent_workspace/project/h200-locomotion-lab-task009-sonic-action-rollout-matrix/src
python -m h200_locomotion_lab.tools.genesis_sonic_planner_encoder_rollout_probe
  --asset /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml
  --planner /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
  --planner-runner /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/sonic_planner_ort_runner
  --encoder /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx
  --decoder /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx
  --work-dir /root/agent_workspace/project/h200-locomotion-lab-task009-artifacts/walking_officialctx_40f_work
  --frames 40 --replan-interval 10
  --initial-context-source initial_joint_csv --replan-context-source motion
  --root-qpos 0.002389 0.011728 0.791166 0.999910712 -0.006119614 0.011878765 0.000052081
  --initial-joint-pos-csv /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_walking_q_log_300f.csv
  --initial-joint-pos-row 0 --log-every 10 --max-wall-time-s 360
  --progress-file /root/agent_workspace/project/h200-locomotion-lab-task009-artifacts/walking_officialctx_40f_progress.txt
  --output-gif /root/agent_workspace/project/h200-locomotion-lab-task009-artifacts/walking_officialctx_40f.gif
  --output-mp4 /root/agent_workspace/project/h200-locomotion-lab-task009-artifacts/walking_officialctx_40f.mp4
```

  Result:

```text
GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_OK
GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_MODE online_planner_encoder
REPLAY_OBS_USED False
REPLAY_TOKEN_USED False
MOTOR_CONFIG sonic_g1_kp_kv_force_range
PLANNER_CALLS 4
ENCODER_OBS_FINITE True
TOKEN_FINITE True
DECODER_OBS_FINITE True
ACTION_FINITE True
ROOT_Z_MIN 0.7183110117912292
ROOT_Z_MAX 0.7911660075187683
HORIZONTAL_DISPLACEMENT 0.4167171290308901
PATH_LENGTH_XY 0.43588413641877954
TOTAL_CONTACT_SWITCHES 6
SINGLE_SUPPORT_FRAMES 26
LOCOMOTION_OBSERVED True
GIF_OUTPUT /root/agent_workspace/project/h200-locomotion-lab-task009-artifacts/walking_officialctx_40f.gif
MP4_OUTPUT /root/agent_workspace/project/h200-locomotion-lab-task009-artifacts/walking_officialctx_40f.mp4
```

  Local copies:

```text
.agent/task/task009-sonic-action-rollout-matrix/artifacts/walking_officialctx_40f.gif
.agent/task/task009-sonic-action-rollout-matrix/artifacts/walking_officialctx_40f.mp4
```

# Review

Status: passed.

- 2026-05-08: Initial read-only reviewer blocked on documentation because the
  log showed only the inner H200 command and not the guarded wrapper. Router
  updated the log to include the `run_guarded.sh` wrapper and no-replay fields.
- 2026-05-08: Read-only re-review found no blocking issues and no suggestions.
