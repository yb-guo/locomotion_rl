# Route

Task: task002-sonic-mujoco-smoke

Goal: Create a reproducible MuJoCo sim2sim environment for SONIC.

Scope:

- environment notes
- official upstream env setup
- local smoke script updates if needed

Verify:

- Python imports MuJoCo and upstream SONIC modules.
- No Isaac Sim dependency is required for this subtask.
- `bash install_scripts/install_mujoco_sim.sh`
- `source .venv_sim/bin/activate && python -c "import mujoco; print(mujoco.__version__)"`

Environment:

- Linux target machine, ideally H200 server.

No Hack:

- no global site-package mutation
- no hard-coded personal checkpoint path in repo code
- no silent fallback when MuJoCo import fails

Hardware:

- should run without RTX
- record CUDA and driver versions when used

# Log

- Pending H200/Linux execution.
- Use `agent_execute.md` section L1.

# Review

Result: pending
Syntax:
Hack:
Scope:
Efficiency:
Hardware:
Verify:
Findings:
