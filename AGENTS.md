# Agent Instructions

This repo is for H200-first humanoid locomotion RL research.

## Entry Point

Start from `.agent/index.md`.

Use the HeadPose-style workflow:

- Project decisions go in `.agent/doc`.
- Work goes through `.agent/task/taskXXX-name/task.md`.
- Each subtask keeps `Route / Log / Review`.
- Do not mark a task passed without verification evidence.

## Priorities

- Prefer MuJoCo and Genesis for runnable experiments on H100/H200.
- Keep Isaac Lab integrations isolated and optional.
- Keep agent modules small enough to replace with upstream GEAR-SONIC or
  LocoFormer implementations later.
- Do not download datasets, checkpoints, or simulator assets unless explicitly
  requested.

## Useful Commands

```bash
python -m h200_locomotion_lab.tools.inspect_agent
python -m pytest
```

## Main Research Threads

- GEAR-SONIC: verify official MuJoCo sim2sim, then inspect deployment I/O.
- LocoFormer: implement the core policy decomposition first, then scale robots
  and domain randomization after the small loop works.
- H200 execution: use CUDA training and headless simulation; use an RTX machine
  only for Isaac Sim rendering, USD import/export, or sensor-heavy validation.

## Task Discipline

- Do not download checkpoints, robot assets, datasets, or upstream repos unless
  the user explicitly asks.
- Record hardware assumptions and failure modes in the relevant task log.
- If Isaac Lab fails on H200 due to Isaac Sim, RTX, Vulkan, or Kit startup, stop
  that route and move the experiment to MuJoCo or Genesis.
