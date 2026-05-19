# Subtask 013: PPO Hybrid Asset Switch

## Route

- Continue task023; do not start PPO training in this subtask.
- Switch the PPO smoke entrypoint to use the best current Genesis asset
  candidate by default:
  `ankle_roll_hybrid_edge_boxes_no_points`.
- Preserve an explicit source-asset fallback for controlled comparisons.
- Do not mutate the source MJCF or global robot profile YAML.
- Generate the task-local hybrid MJCF under each PPO run directory so the
  summary records the exact asset used.

## Log

- 2026-05-13 Opened after user asked to switch PPO to the best hybrid asset.
- 2026-05-13 Updated `g1_ppo_smoke` asset selection:
  - default `--asset-variant task023_hybrid`;
  - explicit `--asset-variant profile` preserves the old source-profile asset;
  - `task023_hybrid` generates `ankle_roll_hybrid_edge_boxes_no_points` under
    the PPO run directory before building the vectorized Genesis backend;
  - the global robot profile YAML and source MJCF are not mutated.
- 2026-05-13 Added `asset_resolution.json` plus `asset_variant` and
  `asset_path` fields in PPO run config/summary so later PPO evidence names the
  exact generated XML.
- 2026-05-13 Local verification:

```text
PYTHONPATH=src python -m pytest -p no:cacheprovider \
  tests/test_g1_ppo_smoke.py \
  tests/test_g1_ankle_roll_contact_patch.py
Result: 22 passed, 1 skipped

PYTHONPATH=src python -m pytest -p no:cacheprovider \
  tests/test_g1_action_energy_ablation.py \
  tests/test_g1_tilt_reset_ablation.py \
  tests/test_g1_curriculum_ppo_smoke.py
Result: 27 passed, 4 skipped
```

- 2026-05-13 H200 focused verification after syncing the patch:

```text
PYTHONPATH=src python3 -m pytest -p no:cacheprovider \
  tests/test_g1_ppo_smoke.py \
  tests/test_g1_ankle_roll_contact_patch.py
Result: 23 passed
```

- 2026-05-13 H200 real source-asset hybrid generation check:

```text
Command:
PYTHONPATH=src python3 -m h200_locomotion_lab.tools.g1_ankle_roll_contact_patch \
  --output-root outputs/task023/ppo_hybrid_asset_switch \
  --run-id verify_hybrid_asset_static_001 \
  --variants ankle_roll_hybrid_edge_boxes_no_points

Result:
status=completed
source_unchanged=true
missing=[]
errors=[]
changed_geom_count=14
asset:
/root/agent_workspace/project/h200-locomotion-lab-task023-base-attitude-height-stabilization/outputs/task023/ppo_hybrid_asset_switch/verify_hybrid_asset_static_001/assets/g1_27dof_nohand.ankle_roll_hybrid_edge_boxes_no_points.xml
```

## Review

Status: completed.

PPO now defaults to the best current task023 hybrid asset without changing the
global profile. No PPO training was run in this subtask.
