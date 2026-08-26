# Locomotion Lab

RTX 5060 Ti-first research scaffold for humanoid locomotion RL. The Python
package keeps its historical `h200_locomotion_lab` name for compatibility;
H200 execution is an explicitly paused optional profile, not the default
development target.

This is an agent-driven research project. The main entrypoint is:

```text
.agent/index.md
```

Use `.agent/doc` for long-lived decisions and `.agent/task` for executable work.
The active hardware policy is a single local RTX 5060 Ti (16 GB) with
headless CUDA/MuJoCo/MJLab development. The default path is therefore:

1. Run GEAR-SONIC sim2sim through MuJoCo.
2. Run new RL experiments in Genesis or MuJoCo.
3. Keep Isaac Lab as an optional headless smoke-test target, not as the core
   dependency for this repo.
4. Reproduce LocoFormer ideas in a small, inspectable setting before scaling.

H200 jobs are disabled by default and are not part of the current development
loop. Historical H200 task logs remain in `.agent/task` as provenance only.

## Layout

```text
h200-locomotion-lab/
  .agent/
    index.md           Agent entrypoint.
    doc/               Long-lived project memory.
    task/              Executable task breakdowns.
  configs/
    tasks/              Observation/action/reward task contracts.
    policies/           Action-generator architecture configs.
    algorithms/         Policy-update rule configs.
    experiments/        Explicit component composition and runtime budget.
    agents/             Legacy agent configs during migration.
    envs/               Simulator and robot configs.
  docs/
    agent_submodules.md Agent module map.
    h200_strategy.md    Hardware and simulator strategy.
  scripts/
    run_sonic_mujoco_smoke.sh
  src/h200_locomotion_lab/
    core/               Framework-neutral RL contracts.
    tasks/              MDP task definitions.
    policies/           Task-independent action generators.
    algorithms/         Task-independent learning rules.
    experiments/        Composition and interaction entrypoints.
    envs/               Simulator/backend adapters.
    agents/             Legacy policy integrations during migration.
    training/           Legacy training paths during migration.
    tools/              Local inspection utilities.
  tests/
```

## Local Inspection

These commands do not install heavy simulator dependencies:

```bash
python -m h200_locomotion_lab.tools.inspect_agent
python -m h200_locomotion_lab.tools.inspect_components
python -m pytest
```

For editable development:

```bash
python -m pip install -e ".[dev]"
```

## Heavy Dependencies

Install these only on the target Linux machine:

```bash
# Genesis path
python -m pip install -e ".[genesis]"

# MuJoCo path
python -m pip install -e ".[mujoco]"

# Optional training utilities
python -m pip install -e ".[training]"
```

Isaac Lab is intentionally not a hard dependency here. If a future optional
H200 profile is re-enabled, run only the official GEAR-SONIC headless smoke
test first and stop if the stack fails on Isaac Sim or RTX/Vulkan
initialization.

## RTX 50-Series Unitree MJLab Setup

The verified workstation path uses Ubuntu 22.04, Python 3.11, an RTX 5060 Ti,
and the official Unitree MJLab checkout at revision
`1425b15f73bd4095f0df53709d7c389c3eb9e790`. Fetch the upstream checkout only
when intended, then run the reproducible installer:

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git \
  .external/unitree_rl_mjlab
git -C .external/unitree_rl_mjlab checkout \
  1425b15f73bd4095f0df53709d7c389c3eb9e790

scripts/setup_rtx_mjlab.sh
scripts/run_rtx_mjlab_smoke.sh
```

The constraint file pins MJLab 1.2.0 to MuJoCo 3.5.0 and Warp 1.12.0. Newer
MuJoCo/Warp releases remove APIs still used by that MJLab version. The smoke
defaults to 32 environments and one PPO iteration, which is intentionally much
smaller than the old 8192-environment H200 runs.

The official `Unitree-G1-Flat` smoke is a runtime check, not proof that the
project's 31-action Task044/046 algorithm has migrated. Audit that separate
source boundary with:

```bash
scripts/check_task044_migration.sh

# Also require a resumable policy checkpoint:
TASK044_CHECKPOINT=/absolute/path/model.pt \
REQUIRE_TASK044_CHECKPOINT=1 \
  scripts/check_task044_migration.sh
```

The tracked Task028 generator can rebuild the fixed-topology gripper base in a
fresh checkout. This gives the verified 31-action `[body29, gripper2]` contract,
but does not recreate the later motor-failure curriculum:

```bash
.venv/bin/python \
  .agent/task/task028-randomized-wholebody-morphology-env/artifacts/task028_create_g1_gripper_task.py \
  --root .external/unitree_rl_mjlab

RTX_MJLAB_SMOKE_TASK=Unitree-G1-Gripper-Flat \
RTX_MJLAB_SMOKE_RUN_NAME=g1_gripper_env32_iter1 \
  scripts/run_rtx_mjlab_smoke.sh
```

Run Unitree scripts from the upstream checkout root and prepend this project's
absolute `src` directory to `PYTHONPATH`; both projects use a top-level package
named `src`, so launching from the local repo root alone selects the wrong one.

## Current Task Order

1. `task001-agent-setup`: establish the agent project structure.
2. `task002-sonic-mujoco-smoke`: run official GEAR-SONIC MuJoCo sim2sim.
3. `task003-h200-simulator-smoke`: decide MuJoCo / Genesis / Isaac Lab viability.
4. `task004-genesis-g1-baseline`: build a Genesis G1 PPO baseline.
5. `task005-locoformer-min-reproduction`: add minimal long-context transformer policy.
