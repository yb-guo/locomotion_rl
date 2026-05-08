# Route

Task: task009-sonic-action-rollout-matrix

Goal: run at least one non-walking dance-like command sweep through the same
online SONIC action execution path.

Scope:

- Use existing planner command knobs such as movement/facing direction,
  target velocity, and planner mode.
- Treat this as a stress test unless official SONIC documentation/assets prove
  a real dance skill mode.
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

- Pass: finite flags, stable height, command path reaches robot execution, and
  artifact paths recorded.
- Fail: exact command, log path, and failure metric recorded.

No Hack:

- Do not relabel unstable random commands as a successful dance skill.

Hardware: H200/Linux target via `myserver`.

# Log

- 2026-05-08: H200 dance-like command stress completed through the same online
  SONIC planner -> encoder -> decoder -> profile-backed action bridge ->
  Genesis G1 execution path. This is not claimed as an official SONIC dance
  skill; it is a non-walking sidestep/facing stress command using existing
  planner knobs.

  Guarded command wrapper:

```text
C:\Windows\System32\OpenSSH\ssh.exe myserver
"/root/agent_workspace/safe_agent/run_guarded.sh bash -lc '<inner command>'"
```

  Inner command reuses the walking route's online planner/encoder/decoder
  command with these differences:

```text
--work-dir /root/agent_workspace/project/h200-locomotion-lab-task009-artifacts/dance_like_sidestep_facing_back_40f_work
--frames 40 --replan-interval 10
--movement-direction 0.0 1.0 0.0
--facing-direction -1.0 0.0 0.0
--target-vel 0.5
--output-gif /root/agent_workspace/project/h200-locomotion-lab-task009-artifacts/dance_like_sidestep_facing_back_40f.gif
--output-mp4 /root/agent_workspace/project/h200-locomotion-lab-task009-artifacts/dance_like_sidestep_facing_back_40f.mp4
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
ROOT_Z_MIN 0.7366165518760681
ROOT_Z_MAX 0.7911660075187683
HORIZONTAL_DISPLACEMENT 0.26222350185643367
PATH_LENGTH_XY 0.29529537557818253
YAW_DELTA -2.7321533960550517
TOTAL_CONTACT_SWITCHES 9
SINGLE_SUPPORT_FRAMES 31
LOCOMOTION_OBSERVED True
GIF_OUTPUT /root/agent_workspace/project/h200-locomotion-lab-task009-artifacts/dance_like_sidestep_facing_back_40f.gif
MP4_OUTPUT /root/agent_workspace/project/h200-locomotion-lab-task009-artifacts/dance_like_sidestep_facing_back_40f.mp4
```

  Local copies:

```text
.agent/task/task009-sonic-action-rollout-matrix/artifacts/dance_like_sidestep_facing_back_40f.gif
.agent/task/task009-sonic-action-rollout-matrix/artifacts/dance_like_sidestep_facing_back_40f.mp4
```

# Review

Status: passed.

- 2026-05-08: Initial read-only reviewer blocked on documentation because the
  log inherited the walking route's missing guarded wrapper evidence. Router
  updated the log to include the `run_guarded.sh` wrapper and no-replay fields.
- 2026-05-08: Read-only re-review found no blocking issues and no suggestions.
