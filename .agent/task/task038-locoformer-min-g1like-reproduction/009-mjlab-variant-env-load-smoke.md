# 009 MJLab Variant Env Load Smoke

## Route

Close the gap left by `008`:

```text
patched train/held-out XML artifacts
  -> external Unitree MJLab G1 gripper task registration
  -> MJLab env/action manager construction
  -> reset plus zero-action finite step smoke
```

This slice is env-load-only. It does not train, evaluate a checkpoint, render
video, or claim a LocoFormer reproduction result.

## Minimal Closed Loop

The implementation provides:

- an idempotent external MJLab patch script that accepts `--root` and touches
  only:
  - `src/assets/robots/unitree_g1_gripper/g1_gripper_constants.py`;
  - `src/tasks/velocity/config/g1_gripper/env_cfgs.py`;
  - `src/tasks/velocity/config/g1_gripper/__init__.py`;
- a robot cfg helper that loads a specified XML path while preserving
  `G1_GRIPPER_ARTICULATION`;
- train and held-out env cfg helpers using the patched XML paths produced by
  `008`;
- two env-load-only task ids:
  - `Unitree-G1-Gripper-Flat-Task038-TrainAssetSmoke`;
  - `Unitree-G1-Gripper-Flat-Task038-HeldoutAssetSmoke`;
- a local probe CLI that loads a task id on H200, resets, steps zero actions,
  inspects action and observation metadata, and writes small JSON.

## Evidence Gate

Local reviewable evidence:

- `.agent/task/task038-locoformer-min-g1like-reproduction/task038_register_mjlab_variant_assets.py`;
- `src/h200_locomotion_lab/tools/task038_mjlab_variant_env_load_probe.py`;
- `tests/test_task038_mjlab_variant_env_load.py`;
- command:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_mjlab_variant_env_load.py tests\test_task038_g1like_mjcf_patch.py`;
- command:
  `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task038_mjlab_variant_env_load_probe --help`.

H200 router gate, not run by this subagent:

1. sync the patch script and source package to the H200 MJLab environment;
2. run the patch script with the external MJLab `--root`;
3. run the probe once for the train task and once for the held-out task;
4. accept only JSON where `pass=true`, `zero_step_ok=true`,
   `action_dim=31`, `total_action_dim=31`, and observation finite flags are
   true.

The JSON is not training, eval, or reproduction evidence. It only proves the
registered MJLab env/action contract can load the patched XML and survive a
zero-action smoke.

## Subagent Ownership

Worker owns only:

- this `009` task note;
- `task038_register_mjlab_variant_assets.py`;
- `src/h200_locomotion_lab/tools/task038_mjlab_variant_env_load_probe.py`;
- `tests/test_task038_mjlab_variant_env_load.py`;
- narrow `task.md` status updates that register `009`;
- narrow `003` or `008` note updates only if needed to point at this follow-up.

Worker must adapt to the dirty worktree and must not revert or rewrite existing
Task037/Task038 work owned by other subagents.

## Failure Exit

Stop and report blocked if:

- MJLab changes its env cfg, action manager, or registry API enough that the
  probe cannot infer action dimension or zero-step safely;
- either registered task loads the wrong XML path;
- action dimension differs from the expected `31`;
- zero actions crash, produce non-finite observations, or reset accounting is
  unavailable in a way that hides crashes;
- anyone asks to mark runner, eval, video, reproduction, or TXL superiority as
  complete from this env-load-only smoke.

## Log

- 2026-05-29 Added an idempotent external MJLab patch script for Task038 train
  and held-out XML variant task ids. The script appends new helpers/registers
  rather than changing the default G1 gripper behavior.
- 2026-05-29 Added `task038_mjlab_variant_env_load_probe`, which imports MJLab
  only inside `run_probe`, writes structured failure JSON, and requires
  action dimension `31` plus finite zero-step observations for a positive JSON
  result.
- 2026-05-29 Added local tests for patch idempotence, task id/XML constants,
  probe CLI defaults, failure JSON writing, and documentation claim limits.
- 2026-05-29 Verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_mjlab_variant_env_load.py tests\test_task038_g1like_mjcf_patch.py`
  first hit the known Windows pytest temp permission issue under
  `AppData\Local\Temp\pytest-of-guoyubo.9`. Re-run with workspace temp:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider --basetemp .test_tmp_task038_009 tests\test_task038_mjlab_variant_env_load.py tests\test_task038_g1like_mjcf_patch.py`
  -> `19 passed in 0.38s`.
- 2026-05-29 Verification:
  `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task038_mjlab_variant_env_load_probe --help`
  -> help printed successfully with `--expected-action-dim`,
  `--expected-xml-path`, `--num-envs`, `--steps`, and `--device`.
- 2026-05-29 Addressed reviewer blockers: external `get_task038_variant_spec`
  now populates `spec.assets` for both absolute and relative `meshdir`; the
  probe resolves the registered Task038 XML constant, records
  `registered_xml_path` and `xml_path_matches_expected`, rejects expected-path
  mismatches, and requires a non-empty finite observation summary before
  `pass=true`.
- 2026-05-29 Verification after reviewer fixes:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider --basetemp .test_tmp_task038_009 tests\test_task038_mjlab_variant_env_load.py tests\test_task038_g1like_mjcf_patch.py`
  -> `22 passed in 0.41s`.
- 2026-05-29 Verification after reviewer fixes:
  `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task038_mjlab_variant_env_load_probe --help`
  -> help printed successfully.
- 2026-05-29 Addressed H200 MJLab probe failure where TensorDict-like
  observations reached `_obs_summary` as tensor leaves. The probe now recurses
  through mapping-style observations before converting numeric leaves, and
  failure JSON tries to preserve the registered Task038 XML path even after a
  later probe exception.
- 2026-05-29 Verification for the TensorDict-like fix:
  `python -m pytest -q -p no:cacheprovider --basetemp .test_tmp_task038_009 tests\test_task038_mjlab_variant_env_load.py tests\test_task038_g1like_mjcf_patch.py`
  first failed during collection because this local environment does not have
  the `src` package path installed as `h200_locomotion_lab`. Re-run with the
  repo's existing `PYTHONPATH=src` convention then hit the known Windows ACL
  issue removing `.test_tmp_task038_009`; elevated re-run:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider --basetemp .test_tmp_task038_009 tests\test_task038_mjlab_variant_env_load.py tests\test_task038_g1like_mjcf_patch.py`
  -> `24 passed in 0.44s`.
- 2026-05-29 H200 external MJLab patch application completed against
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab`;
  the patch script reported updates/checks for:
  `src/assets/robots/unitree_g1_gripper/g1_gripper_constants.py`,
  `src/tasks/velocity/config/g1_gripper/env_cfgs.py`, and
  `src/tasks/velocity/config/g1_gripper/__init__.py`.
- 2026-05-29 H200 train env-load probe passed on `cuda:0`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/mjlab_variant_env_load/train_env_load.json`.
  Evidence: `pass=true`, `zero_step_ok=true`, `registered_xml_path` matched
  the expected train XML, `action_dim=31`, `total_action_dim=31`,
  `obs.actor.shape=[1,104]`, `obs.critic.shape=[1,119]`, and both observation
  groups were finite after 10 zero-action steps.
- 2026-05-29 H200 held-out env-load probe passed on `cuda:0`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/mjlab_variant_env_load/heldout_env_load.json`.
  Evidence: `pass=true`, `zero_step_ok=true`, `registered_xml_path` matched
  the expected held-out XML, `action_dim=31`, `total_action_dim=31`,
  `obs.actor.shape=[1,104]`, `obs.critic.shape=[1,119]`, and both observation
  groups were finite after 10 zero-action steps.

## Review

Status: closed for the `009` env-load-only slice.

Final reviewer verdict:

- no blocking findings;
- TensorDict-like observation handling is sound and the pass gate did not
  become weaker;
- H200 train and held-out JSON evidence is sufficient to close only this
  env-load slice.

No runner, eval, video, reproduction, or TXL superiority claim is made by this
slice.
