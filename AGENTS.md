# Agent Instructions

This repo is for RTX 5060 Ti-first humanoid locomotion RL research. The
historical H200 path is paused and must not be selected by default.

## Entry Point

Start from `.agent/index.md`.

Use the HeadPose-style workflow:

- Project decisions go in `.agent/doc`.
- Work goes through `.agent/task/taskXXX-name/task.md`.
- Each subtask keeps `Route / Log / Review`.
- Do not mark a task passed without verification evidence.

## Priorities

- Prefer MuJoCo and MJLab for runnable experiments on the local RTX 5060 Ti.
- Keep Genesis optional and VRAM-safe for the RTX workstation.
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
- RTX execution: use CUDA training and headless MuJoCo/MJLab simulation. Do not
  start H200 jobs unless an explicit profile is re-enabled.

## Task Discipline

- Do not download checkpoints, robot assets, datasets, or upstream repos unless
  the user explicitly asks.
- Record hardware assumptions and failure modes in the relevant task log.
- If Isaac Lab fails due to Isaac Sim, RTX, Vulkan, or Kit startup, stop that
  route and keep the experiment on MuJoCo/MJLab.
