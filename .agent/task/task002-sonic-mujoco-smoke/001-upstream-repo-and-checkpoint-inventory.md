# Route

Task: task002-sonic-mujoco-smoke

Goal: Inventory upstream SONIC repo, checkpoint source, and version constraints.

Scope:

- `.agent/doc/sonic.md`
- `configs/agents/sonic_adapter.yaml`
- `scripts/run_sonic_mujoco_smoke.sh`

Verify:

- List upstream repo URL, checkpoint repo, expected TensorRT version, and official commands.
- `Get-Content .agent/task/task002-sonic-mujoco-smoke/agent_execute.md`

Environment:

- local docs only unless user authorizes network/download.

No Hack:

- no vendored upstream copy
- no guessed checkpoint path
- no undocumented runtime version

Hardware:

- record whether each command needs H200, CPU, or RTX

# Log

- Upstream repo: `https://github.com/NVlabs/GR00T-WholeBodyControl`
- Official docs: `https://nvlabs.github.io/GR00T-WholeBodyControl/`
- SONIC page: `https://nvlabs.github.io/GEAR-SONIC/`
- Checkpoint/model repo: `nvidia/GEAR-SONIC`
- Deployment download: `python download_from_hf.py`
- Training/sample download: `hf download nvidia/GEAR-SONIC --include "sample_data/*" --include "sonic_release/*" --local-dir .`
- Required x86_64 TensorRT: `10.13`
- First target: MuJoCo sim2sim, not Isaac Lab.

# Review

Result: passed
Syntax: markdown only
Hack: no guessed checkpoint path
Scope: task docs and SONIC docs only
Efficiency: no heavy downloads during local planning
Hardware: H200/RTX/TensorRT constraints recorded
Verify: inventory written in `agent_execute.md`
Findings: none
