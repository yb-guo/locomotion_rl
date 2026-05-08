# Route

Task: task009-sonic-action-rollout-matrix

Goal: confirm runnable H200 code/assets and exact command template.

Scope:

- Use task009 branch/worktree derived from task008.
- Use existing official SONIC assets on H200 only.
- Do not download or mutate datasets/assets.

Environment:

- interactive remote sessions must enter `/root/agent_workspace/safe_agent/agent_shell.sh`
- non-interactive remote commands must run through `/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/<project> && <command>'`
- code, generated files, and intermediate outputs must stay under `/root/agent_workspace/project`
- datasets must be accessed only through `/root/agent_workspace/datasets`
- dataset links must be created with `/root/agent_workspace/safe_agent/link_dataset.sh`
- do not write to or delete from `/mnt/workspace` or `/mnt/workspace1`
- do not manually clear `LD_PRELOAD`

Verify:

- H200 guarded shell alive.
- task009 code is present on H200.
- planner/encoder/decoder/asset/runner paths exist.

No Hack:

- No unguarded remote fallback.
- No hard-coded local Windows paths in remote scripts.

Hardware: H200/Linux target via `myserver`.

# Log

- 2026-05-08: Minimum route/assets closed loop executed from local worktree
  `D:\guoyubo.9\Documents\New project 2\_worktrees\h200-locomotion-lab-task009-sonic-action-rollout-matrix`
  on branch `codex/task009-sonic-action-rollout-matrix` at commit `c2acad0`.
  No long rollout was run.

  Guarded H200 shell check:

```text
Command:
/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/HeadPose || exit 2; echo LD_CHECK; if printenv LD_PRELOAD >/dev/null; then echo LD_PRELOAD_PRESENT; else echo LD_PRELOAD_ABSENT; fi; ...'

Result:
LD_CHECK
LD_PRELOAD_PRESENT
```

  The expected guarded project checkout
  `/root/agent_workspace/project/h200-locomotion-lab` is not present, and no
  task009 checkout is present yet:

```text
MISSING /root/agent_workspace/project/h200-locomotion-lab
MISSING /root/agent_workspace/project/h200-locomotion-lab-task009-sonic-action-rollout-matrix
drwxr-xr-x ... /root/agent_workspace/project/h200-locomotion-lab-task008-runtime-profile-foundation-49ceef3
drwxr-xr-x ... /root/agent_workspace/project/h200-locomotion-lab-task008-runtime-profile-foundation-ab6ae00
drwxr-xr-x ... /root/agent_workspace/project/h200-locomotion-lab-task008-artifacts
```

  Deployment note for task009 code: the current task009 code should be archived
  from this local worktree and copied into
  `/root/agent_workspace/project/h200-locomotion-lab-task009-sonic-action-rollout-matrix`
  before any rollout command. Existing H200 task008 checkouts/artifacts can be
  used as reference material, but task009 should not run from a missing or stale
  checkout. No deploy was performed for this subtask.

- 2026-05-08: Router deployed task009 code after this check. Exact commands:
  local `git archive --format=tar -o .agent/task/task009-sonic-action-rollout-matrix/artifacts/task009-route.tar HEAD`;
  `scp -O .../task009-route.tar myserver:/root/agent_workspace/project/h200-locomotion-lab-task009-artifacts/task009-route.tar`;
  guarded remote
  `cd /root/agent_workspace/project/h200-locomotion-lab-task009-sonic-action-rollout-matrix && tar -xf ../h200-locomotion-lab-task009-artifacts/task009-route.tar`.
  Extraction verified `src/h200_locomotion_lab/tools/genesis_sonic_planner_encoder_rollout_probe.py`.

  Required asset/policy/runner paths checked with `ls -l` through the guarded
  shell:

```text
-rw-r--r-- 1 root root 26111 Apr 30 17:44 /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml
-rw-rw-rw- 1 root root 50100513 May  5 04:32 /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx
-rw-rw-rw- 1 root root 40900688 May  5 04:35 /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx
-rw-rw-rw- 1 root root 773952989 May  5 05:28 /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
-rwxr-xr-x 1 root root 5464248 May  5 06:04 /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref
-rwxr-xr-x 1 root root 63872 May  7 22:18 /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/sonic_planner_ort_runner
-rw-r--r-- 1 root root 23510 May  8 00:42 /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/repo_replay_smoke/src/h200_locomotion_lab/tools/genesis_sonic_planner_encoder_rollout_probe.py
```

  Initial joint/root setup for the official-context walking route:

```text
-rw-r--r-- 1 root root 107190 May  7 05:15 /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_walking_q_log_300f.csv
-rw-rw-rw- 1 root root 8242922 May  7 03:39 /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_policy_input_walking_capture.csv
-rw-rw-rw- 1 root root 308725 May  7 03:39 /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_target_motion_walking_capture.csv
-rw-rw-rw- 1 root root 125123 May  5 04:15 /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428/joint_pos.csv
-rw-rw-rw- 1 root root 180909 May  5 04:15 /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428/body_pos.csv
-rw-rw-rw- 1 root root 242492 May  5 04:15 /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428/body_quat.csv
```

  The task006 official-context route uses:

```text
--root-qpos 0.002389 0.011728 0.791166 0.999910712 -0.006119614 0.011878765 0.000052081
--initial-joint-pos-csv /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_walking_q_log_300f.csv
--initial-joint-pos-row 0
--initial-context-source initial_joint_csv
--replan-context-source motion
--replan-interval 10
```

  Narrow reference-directory listing found no files named `base_quat.csv` or
  `root_pos.csv`; the available official reference files are `body_pos.csv`,
  `body_quat.csv`, `joint_pos.csv`, `joint_vel.csv`, `body_ang_vel.csv`, and
  `body_lin_vel.csv`. This is not a blocker for the task006 route because the
  corrected walking root quaternion is recorded as the explicit `--root-qpos`
  above, and the initial joint state comes from `official_walking_q_log_300f.csv`.

# Review

Status: passed.

- 2026-05-08: Read-only reviewer found no blocking issues. Suggestions:
  update Review after acceptance and record the exact deploy/archive command
  when deployment is performed.
