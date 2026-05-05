# Goal

Run the official GEAR-SONIC MuJoCo sim2sim path before any local algorithm work.

# Scope

- Clone or use upstream `NVlabs/GR00T-WholeBodyControl` only when explicitly requested.
- Download `nvidia/GEAR-SONIC` checkpoint only when explicitly requested.
- Record the exact environment, command, checkpoint, TensorRT version, and output.
- Do not rewrite upstream SONIC before the official loop runs.

# Subtasks

- `001-upstream-repo-and-checkpoint-inventory.md`
- `002-mujoco-sim2sim-environment.md`
- `003-run-official-sim2sim.md`
- `004-record-policy-io-contract.md`

# Result

pending

# Lessons

- MuJoCo sim2sim is the first validation path for H200.

