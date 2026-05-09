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

- 2026-05-09 Added CLI:
  `python -m h200_locomotion_lab.tools.g1_ppo_smoke`.
- CLI defaults match task014:
  - `1024` envs;
  - `32` rollout steps;
  - `5` PPO updates;
  - seeds `0,1,2`;
  - `epochs=2`;
  - `minibatch_size=8192`.
- CUDA guard enforces:
  - `CUDA_VISIBLE_DEVICES=1`;
  - one visible CUDA device;
  - `logical_cuda_device=cuda:0`.
- Output guard enforces run dirs under
  `/root/agent_workspace/project`.
- H200 run created:
  - `config.json` (`567` bytes);
  - `metrics.jsonl` (`7.0K`, 15 rows);
  - `summary.json` (`2.9K`);
  - `final_checkpoint.pt` (`2.2M`).

## Review

Status: passed.

- CLI produces only task014 smoke artifacts and does not enter render, GIF,
  SONIC, ONNX, planner, or download paths.
- Path guard rejects outputs outside `/root/agent_workspace/project`.
