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
- 2026-05-13 H200 deploy `36807e4` verified focused tests through
  `run_guarded.sh` with `CUDA_VISIBLE_DEVICES=1` -> 46 passed. Generated real
  patched XML variants under
  `outputs/task022/ankle_roll_contact_patch/h200-gpu1-patches-v2/`; static
  summary status `completed`, `source_unchanged=true`, `missing=[]`,
  `errors=[]`, and `compiler.meshdir` was rewritten from `meshes` to
  `/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/meshes`.
- 2026-05-13 H200 zero-action onset traces completed with physical GPU 1
  (`CUDA_VISIBLE_DEVICES=1`, logical `cuda:0`), `n_envs=512`, `chunks=128`,
  `chunk_steps=1`, `scenario=baseline_current`:

| asset | first_tilt_step | confirm | result |
| --- | ---: | ---: | --- |
| source `g1_27dof_nohand.xml` | 88 | n/a | baseline reproduced |
| `ankle_roll_friction_attrs` | 88 | n/a | no delay |
| `ankle_roll_larger_spheres` | 106 | 106 | delayed +18 steps, still failed |
| `ankle_roll_box_support` | 113 | 113 | delayed +25 steps, still failed |

  Onset summaries:
  `outputs/task022/failure_onset_trace/h200-gpu1-source-baseline-v1/summary.json`,
  `h200-gpu1-friction-baseline-v2/summary.json`,
  `h200-gpu1-larger-spheres-baseline-v1/summary.json`,
  `h200-gpu1-larger-spheres-baseline-confirm-v1/summary.json`,
  `h200-gpu1-box-support-baseline-v1/summary.json`, and
  `h200-gpu1-box-support-baseline-confirm-v1/summary.json`.
- 2026-05-13 H200 link-level traces completed for source and all three patch
  variants. Ankle-pitch links stayed at `contact_force_max=0.0`; ankle-roll
  links remained the contact path:

| asset | link first_tilt_step | left ankle-roll force max | right ankle-roll force max | note |
| --- | ---: | ---: | ---: | --- |
| source | 89 | 294.43 | 294.97 | baseline contact path |
| `ankle_roll_friction_attrs` | 89 | 294.55 | 294.76 | unchanged |
| `ankle_roll_larger_spheres` | 107 | 240.95 | 240.68 | delayed and lower roll force |
| `ankle_roll_box_support` | 114 | 662.77 | 638.69 | delayed but much higher roll force |

  Link summaries:
  `outputs/task022/ankle_foot_asset_contact_audit/h200-gpu1-source-link-v1/summary.json`,
  `h200-gpu1-friction-link-v1/summary.json`,
  `h200-gpu1-larger-spheres-link-v1/summary.json`, and
  `h200-gpu1-box-support-link-v1/summary.json`.

## Review

Status: passed with no blocking findings. H200 runtime evidence supports a
partial-help decision, not a stable-standing pass.
