# Goal

Run the official GEAR-SONIC MuJoCo sim2sim path before any local algorithm work.

# Scope

- Clone or use upstream `NVlabs/GR00T-WholeBodyControl` on the H200/Linux target.
- Download `nvidia/GEAR-SONIC` checkpoint on the H200/Linux target.
- Record the exact environment, command, checkpoint, TensorRT version, and output.
- Do not rewrite upstream SONIC before the official loop runs.
- Use `agent_execute.md` as the execution runbook.

# Subtasks

- `001-upstream-repo-and-checkpoint-inventory.md`
- `002-mujoco-sim2sim-environment.md`
- `003-run-official-sim2sim.md`
- `004-record-policy-io-contract.md`
- `agent_execute.md`

# Result

ready for H200 execution

# Lessons

- MuJoCo sim2sim is the first validation path for H200.
- Do not attempt full training before official sim2sim and training smoke pass.
