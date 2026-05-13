# Subtask 002: H200 Zero Action Patch Ablation

## Route

- Run only after subtask001 passes local tests and read-only review.
- Use guarded H200 commands only.
- Compare source asset against generated project-local patched XML variants.
- No PPO.

## H200 Command Shape

Use:

```bash
/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/h200-locomotion-lab-task022-ankle-roll-contact-patch-ablation && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src <command>'
```

Record:

```text
physical_gpu=1
logical_cuda_device=cuda:0
```

## Evidence Required

- static `summary.json` from patch generator;
- zero-action standing/onset trace per asset variant;
- link-level ankle-roll contact trace per asset variant;
- comparison table with first tilt/reset, root/upright, ankle-roll force, and
  contact env count.

## Stop Rules

- If a patch variant fails Genesis import, record the importer error and
  continue to the next variant.
- If all variants fail import, stop and review patch semantics.
- If all variants reproduce first tilt near baseline, stop and mark contact
  patch insufficient.

## Log

- 2026-05-13 Created with task022; pending subtask001.
- 2026-05-13 Local preparation added `--asset-path` plumbing for zero-action,
  failure-onset, and contact-audit link traces so generated XML variants can be
  passed into `VectorizedGenesisBackend` without mutating the default profile.
  No H200 run performed.
- 2026-05-13 Read-only review found one blocking compatibility regression:
  `build_run_config` had made `asset_path` required and broke the existing
  rigid-options ablation caller. Coding subagent fixed the default argument and
  added regression coverage. Router verification passed:
  `PYTHONPATH=src python -m pytest
  tests/test_g1_rigid_options_standing_ablation.py
  tests/test_g1_zero_action_standing_causality.py
  tests/test_g1_failure_onset_trace.py
  tests/test_g1_ankle_foot_asset_contact_audit.py -q -p no:cacheprovider`
  -> 37 passed, 4 skipped. Re-review found no blocking findings.

## Review

Status: local asset-override preparation reviewed with no blocking findings.
H200 runtime evidence remains pending.
