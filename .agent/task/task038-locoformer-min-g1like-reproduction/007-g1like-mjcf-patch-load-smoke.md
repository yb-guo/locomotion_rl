# 007 G1-like MJCF Patch Load Smoke

## Route

Close the next minimum Task038 asset loop without starting H200:

```text
003 manifest variant
  -> conservative patched MJCF artifact
  -> local ElementTree parse
  -> small JSON contract summary for later H200 load smoke
```

This slice is local-only. It does not download robot assets, simulator assets,
checkpoints, datasets, or upstream repositories. It does not start MuJoCo,
Genesis, Isaac Sim, or any H200 job.

## Minimal Closed Loop

The implementation must provide:

- a pure Python `ElementTree` MJCF patcher;
- one CLI that consumes an existing source MJCF path and writes one train and
  one held-out patched XML artifact;
- a small JSON summary with variant id, split, held-out condition, scale
  factors, slot schema id/hash, joint order hash, `action_dim=29`, source path,
  output path, topology before/after, patched/skipped counts, limitation notes,
  and local parse/pass fields;
- local unit tests using temporary minimal MJCF fixtures only.

The patcher must preserve topology: no added/deleted bodies, joints, or
actuators, and no joint/actuator name or order changes.

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
- CLI `--help` works;
- generated summaries do not claim H200 or simulator load;
- skipped or unsupported MJCF fields are recorded as skipped/limitations, not
  as successful patch claims;
- nonphysical mass/COM/inertia scalar scaling is explicitly recorded.

Full H200 load smoke remains pending after this slice.

## Subagent Ownership

Worker owns only:

- `src/h200_locomotion_lab/robots/g1like_mjcf_patch.py`;
- `src/h200_locomotion_lab/tools/task038_g1like_variant_asset_smoke.py`;
- `tests/test_task038_g1like_mjcf_patch.py`;
- this 007 task note;
- narrow Task038 doc status updates that point to 007.

Worker must adapt to the dirty worktree and must not revert or rewrite existing
Task037/Task038 work owned by other subagents.

## Failure Exit

Stop and report blocked if:

- a source MJCF cannot be patched without changing body/joint/actuator topology;
- the local patched XML cannot be parsed by `ElementTree`;
- a required scale field or variant identity field is missing;
- H200 load evidence is requested but no simulator/H200 run has actually been
  performed.

## Log

- 2026-05-29 Implemented a conservative local MJCF patcher in
  `src/h200_locomotion_lab/robots/g1like_mjcf_patch.py`. It scales
  body/joint/geom/site `pos`, inertial `mass`/`pos`/`diaginertia`/`fullinertia`,
  actuator `ctrlrange`/`forcerange`/`gear`, and joint `actuatorfrcrange` when
  present. It records patched counts, skipped missing attributes, topology
  before/after, local parse status, and limitation notes. It raises clear
  errors for invalid variants or malformed numeric XML attributes.
- 2026-05-29 Added
  `src/h200_locomotion_lab/tools/task038_g1like_variant_asset_smoke.py`. The CLI
  generates one train and one held-out manifest variant, patches the caller's
  source MJCF into XML artifacts, and writes a small JSON summary with
  `h200_load_smoke="pending"`, `simulator_started=false`, and
  `asset_downloaded=false`.
- 2026-05-29 Added
  `tests/test_task038_g1like_mjcf_patch.py` with local tmp MJCF fixtures. Tests
  cover topology preservation, expected attribute scaling and counts, skipped
  missing fields and limitations, CLI summary output, CLI help, and clear
  invalid variant/XML failures.
- 2026-05-29 Verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_g1like_mjcf_patch.py`
  -> `11 passed in 0.42s` when run outside the sandbox because the sandboxed
  Windows temp directory denied pytest `tmp_path` creation.
- 2026-05-29 Verification:
  `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task038_g1like_variant_asset_smoke --help`
  -> help printed successfully with `--source-mjcf`, `--output-dir`,
  `--summary-json`, `--seed`, `--heldout-condition`, and `--heldout-band`.
- 2026-05-29 Expanded local Task038 verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_g1like_mjcf_patch.py tests\test_task038_eval_contract.py tests\test_task038_txl_memory_contract.py tests\test_task038_g1like_morphology_manifest.py tests\test_task038_g1like_slot_contract.py tests\test_task038_claim_contract.py tests\test_agent_inventory.py`
  -> `73 passed in 0.56s` when run outside the sandbox because the sandboxed
  Windows temp directories denied pytest `tmp_path` creation.
- 2026-05-29 Review subagent reported no blocking findings. Non-blocking note:
  JSON `pass=true` means local parse/topology contract pass only, not simulator
  readiness or a fully patched physical model.

## Review

Status: local MJCF patch artifact loop closed with reviewer confirmation.

Local evidence proves conservative XML patching and local parse/topology
preservation only. Full H200 load smoke remains pending and is not claimed by
this slice.
