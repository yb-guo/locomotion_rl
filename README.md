# H200 Locomotion Lab

H200-first research scaffold for humanoid locomotion RL.

This is an agent-driven research project. The main entrypoint is:

```text
.agent/index.md
```

Use `.agent/doc` for long-lived decisions and `.agent/task` for executable work.
The repo is set up around the practical constraint that CUDA training GPUs and
RTX simulator workstations have different strengths. The current local training
route is RTX 4090 + MuJoCo-only:

1. Run GEAR-SONIC sim2sim through MuJoCo.
2. Run new RL experiments in MuJoCo.
3. Keep Isaac Lab as an optional headless smoke-test target, not as the core
   dependency for this repo.
4. Reproduce LocoFormer ideas in a small, inspectable setting before scaling.

## Layout

```text
h200-locomotion-lab/
  .agent/
    index.md           Agent entrypoint.
    doc/               Long-lived project memory.
    task/              Executable task breakdowns.
  configs/
    agents/             Agent architecture configs.
    envs/               Simulator and robot configs.
    experiments/        Small runnable experiment definitions.
  docs/
    agent_submodules.md Agent module map.
    h200_strategy.md    Hardware and simulator strategy.
  scripts/
    run_sonic_mujoco_smoke.sh
  src/h200_locomotion_lab/
    agents/             Policy and agent integration skeletons.
    envs/               Simulator adapters.
    training/           RL training loop placeholders.
    tools/              Local inspection utilities.
  tests/
```

## Local Inspection

These commands do not install heavy simulator dependencies:

```bash
python -m h200_locomotion_lab.tools.inspect_agent
python -m pytest
```

For editable development:

```bash
python -m pip install -e ".[dev]"
```

## Heavy Dependencies

Install the local MuJoCo training environment into this repo with uv:

```bash
UV_PROJECT_ENVIRONMENT=.venv uv sync --extra dev --extra training
```

Equivalent pip commands, if uv is unavailable:

```bash
python -m pip install -e ".[mujoco]"
python -m pip install -e ".[training]"
```

Genesis is not part of the current local training route. Historical Genesis
adapters and task notes remain in the repository, but they are not an
environment gate for MuJoCo training.

Isaac Lab is intentionally not a hard dependency here. If you test it on an H200,
run only the official GEAR-SONIC headless smoke test first and stop if the stack
fails on Isaac Sim or RTX/Vulkan initialization.

## Current Task Order

1. `task001-agent-setup`: establish the agent project structure.
2. `task002-sonic-mujoco-smoke`: run official GEAR-SONIC MuJoCo sim2sim.
3. `task003-h200-simulator-smoke`: decide MuJoCo / Genesis / Isaac Lab viability.
4. `task004-genesis-g1-baseline`: build a Genesis G1 PPO baseline.
5. `task005-locoformer-min-reproduction`: add minimal long-context transformer policy.
