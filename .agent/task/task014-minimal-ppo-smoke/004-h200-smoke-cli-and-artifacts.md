# 004: H200 Smoke CLI And Artifacts

## Goal

Add the H200-facing PPO smoke CLI and artifact writer.

## Route

1. Add CLI:
   - module `h200_locomotion_lab.tools.g1_ppo_smoke`;
   - args for n envs, rollout steps, updates, seeds, GPU, output dir;
   - defaults match task014.
2. Enforce CUDA isolation:
   - `CUDA_VISIBLE_DEVICES=1`;
   - torch CUDA available;
   - visible device count `1`;
   - logical device `cuda:0`.
3. Write full run dir:
   - `config.json`;
   - `metrics.jsonl`;
   - `summary.json`;
   - `final_checkpoint.pt`.
4. Metrics:
   - one JSONL row per seed/update;
   - no per-step logging.
5. Output path guard:
   - remote output must start with
     `/root/agent_workspace/project/`;
   - never write `/mnt/workspace` or `/mnt/workspace1`.

## Stop Rules

- If output path is outside `/root/agent_workspace/project`, fail fast.
- If CLI imports Genesis/Torch at module import time on local no-torch path,
  fix boundary.
- If smoke does any render/GIF/ONNX/planner/SONIC path, stop.

## Verification

- Local CLI argument tests.
- Local output path guard tests.
- H200 dry run or smoke creates all expected files.

## Log

Pending implementation.

## Review

Status: pending.
