# 002 MJLab Multi-Trial Smoke

## Route

Connect the multi-trial wrapper to MJLab without making real deterministic
inner reset a blocker yet.

Acceptance criteria:

- Task ids register for at least one existing consumer.
- env64 construction smoke runs.
- env8192 one PPO iteration runs.
- `extras` include `trial_done`, `episode_done`, `trial_index`,
  `final_trial`, and reset reason.
- runner-facing `done` only follows `episode_done`.
- No policy-quality claim is made from this subtask.

## Log

- 2026-05-29 Planned.
- 2026-05-29 Implemented Task037 MJLab auto-reset smoke runner:
  `Task037BufferOnlyK4AutoResetRunner`.
- 2026-05-29 Added H200 patch/register script:
  `task037_register_multitrial_stages.py`.
- 2026-05-29 Added H200 extras probe:
  `python -m h200_locomotion_lab.tools.task037_mjlab_multitrial_extras_probe`.
- 2026-05-29 Added H200 PPO smoke launcher:
  `task037_launch_mjlab_multitrial_smoke.sh`.
- 2026-05-29 Local validation:
  `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -p
  no:cacheprovider tests/test_agent_inventory.py tests/test_task037_multitrial_contract.py
  tests/test_task037_mjlab_smoke_scripts.py` -> `4 passed, 2 skipped`.
- 2026-05-29 H200 validation:
  `/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python -m pytest
  -p no:cacheprovider tests/test_task037_multitrial_contract.py
  tests/test_task037_mjlab_smoke_scripts.py -q` -> `4 passed in 2.61s`.
- 2026-05-29 H200 registry check loaded env cfg, rl cfg, and
  `Task037BufferOnlyK4AutoResetRunner` for
  `Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-AutoReset-Fast2p0`.
- 2026-05-29 H200 extras probe summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/mjlab_multitrial_smoke/probe/task037_mjlab_multitrial_extras_probe_env64.json`.
  Result: `pass=true`, required extras missing `[]`, `done_matches_episode_done=true`,
  `saw_inner_trial_done=true`, `saw_outer_episode_done=true`, `max_trial_index=2`.
- 2026-05-29 H200 env64 PPO smoke:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/mjlab_multitrial_smoke/037_bufferonly_multitrial_env64_iter1_gpu0_seed3700202.stdout.log`.
  Result: one iteration passed, `steps_per_second=1389`.
- 2026-05-29 H200 env8192 PPO smoke:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/mjlab_multitrial_smoke/037_bufferonly_multitrial_env8192_iter1_gpu0_seed3700203.stdout.log`.
  Result: one iteration passed, `steps_per_second=56021`.
- 2026-05-29 H200 summary JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/mjlab_multitrial_smoke/task037_mjlab_multitrial_smoke_summary.json`.

## Review

Status: passed for MJLab auto-reset construction/PPO smoke only.

Evidence:

- Task id registration and runner class loading passed on H200.
- Required extras are present and runner-facing `done` equals
  `episode_done` in the H200 probe.
- env64 and env8192 one-iteration PPO smoke completed.

Limit:

- This subtask uses MJLab's existing auto-reset path. It proves wrapper
  integration, not deterministic condition-preserving inner reset.
- No policy-quality claim is made.
