# 008 G1-like MuJoCo Compile Load Smoke

## Route

Close the minimum readiness gap left by `007`:

```text
source G1 gripper MJCF
  -> copied/patched XML in output dir
  -> source-relative mesh assets remain resolvable
  -> optional MuJoCo compile/load smoke only when explicitly requested
```

This slice does not download robot assets, simulator assets, checkpoints,
datasets, or upstream repositories. It does not step simulation or run a policy.

## Minimal Closed Loop

The implementation must provide:

- source-relative `<compiler meshdir="...">` handling when the patched XML is
  written outside the source XML directory;
- summary fields for `meshdir_before`, `meshdir_after`, `meshdir_rewritten`,
  and `source_xml_dir`;
- CLI `--compile-mujoco`, default false, with no MuJoCo import unless requested;
- structured per-variant compile/load results with `compile_ok`, `nq`, `nv`,
  `nu`, `njnt`, or clear error text;
- local tests that do not require MuJoCo to be installed.

## Evidence Gate

Reviewable evidence:

- `src/h200_locomotion_lab/robots/g1like_mjcf_patch.py`;
- `src/h200_locomotion_lab/tools/task038_g1like_variant_asset_smoke.py`;
- `tests/test_task038_g1like_mjcf_patch.py`;
- command:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_g1like_mjcf_patch.py`;
- command:
  `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task038_g1like_variant_asset_smoke --help`.

Acceptance:

- local tests pass;
- `--help` includes `--compile-mujoco`;
- default CLI summaries record `mujoco_compile_requested=false` and do not
  import MuJoCo;
- compile-load evidence is recorded only when `--compile-mujoco` is actually
  requested and succeeds;
- `simulator_started=false` remains true because compile/load is not stepping
  simulation;
- no H200 compile/load pass is claimed by local readiness work.

## Subagent Ownership

Worker owns only:

- `src/h200_locomotion_lab/robots/g1like_mjcf_patch.py`;
- `src/h200_locomotion_lab/tools/task038_g1like_variant_asset_smoke.py`;
- `tests/test_task038_g1like_mjcf_patch.py`;
- this `008` task note;
- narrow `task.md` status updates that register `008`.

Worker must adapt to the dirty worktree and must not revert or rewrite existing
Task037/Task038 work owned by other subagents.

## Failure Exit

Stop and report blocked if:

- relative mesh assets cannot be made resolvable from copied patched XML without
  changing topology;
- MuJoCo is requested but not installed, in which case the CLI must return
  structured `blocked` compile results instead of crashing;
- MuJoCo compile/load fails for a patched XML, in which case the summary must
  record `compile_ok=false` and the error;
- anyone asks to mark H200 load, runner, eval, or reproduction passed without
  actual H200 evidence.

## Log

- 2026-05-29 Implemented source-relative meshdir preservation for copied
  patched MJCF outputs and added structured meshdir audit fields.
- 2026-05-29 Added optional CLI `--compile-mujoco`. Default remains local patch
  and parse only. Requested compile/load calls `mujoco.MjModel.from_xml_path`
  per variant, records model dimensions on success, and returns structured
  blocked/failure data when MuJoCo is unavailable or compile fails.
- 2026-05-29 Added local tests for relative meshdir rewrite, absolute meshdir
  preservation, default non-compile summary fields, injected compile/load
  metadata, and `--help` coverage.
- 2026-05-29 Verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_g1like_mjcf_patch.py`
  -> `14 passed in 0.37s` when run outside the sandbox because sandboxed
  Windows temp directory access denied pytest `tmp_path` setup.
- 2026-05-29 Verification:
  `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task038_g1like_variant_asset_smoke --help`
  -> help printed successfully with `--compile-mujoco`.
- 2026-05-29 Expanded local Task038 verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_g1like_mjcf_patch.py tests\test_task038_eval_contract.py tests\test_task038_txl_memory_contract.py tests\test_task038_g1like_morphology_manifest.py tests\test_task038_g1like_slot_contract.py tests\test_task038_claim_contract.py tests\test_agent_inventory.py`
  -> `76 passed in 0.48s` when run outside the sandbox because sandboxed
  Windows temp directory access denied pytest `tmp_path` setup.
- 2026-05-29 Review subagent reported no blocking findings. Non-blocking
  notes: the default path does not import MuJoCo by code review, and blocked
  compile paths are structured but not separately unit-tested.
- 2026-05-29 H200 explicit MuJoCo compile/load smoke ran with
  `--compile-mujoco` against the existing Unitree MJLab gripper source MJCF:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/src/assets/robots/unitree_g1_gripper/xmls/g1_gripper.xml`.
  Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/g1like_compile_smoke/task038_g1like_compile_smoke_summary.json`.
  Result: `pass=true`, `local_parse_ok=true`, `mujoco_compile.status=ok`;
  train variant `g1like-train-none-e6ba46370d` and held-out combined variant
  `g1like-heldout-combined-6ac730c265` both compiled with `nq=38`, `nv=37`,
  `njnt=32`, and `nu=0`. The `nu=0` result is expected for bare XML compile
  because MJLab gripper actions are created by the env action manager, not by
  XML actuators.
- 2026-05-29 Follow-up env/action-manager load gate moved to
  `009-mjlab-variant-env-load-smoke.md`. That slice registers MJLab
  env-load-only task ids and checks the expected 31-action contract; this `008`
  note remains only a bare MuJoCo compile/load record.

## Review

Status: H200 bare-MuJoCo compile/load smoke closed for one train and one
held-out patched XML artifact. MJLab env/action-manager load remains pending.

No MJLab runner, eval, video, reproduction, or TXL superiority claim is made by
this slice.
