# Route

Task: task002-sonic-mujoco-smoke

Goal: Record SONIC observation/action/runtime contracts for local adapter work.

Scope:

- `docs/agent_submodules.md`
- `configs/agents/sonic_adapter.yaml`
- optional `.agent/doc/sonic.md` update

Verify:

- Observation fields, shapes, units, action semantics, and runtime backend are listed.
- Read `gear_sonic_deploy/policy/release/observation_config.yaml`.
- Inspect upstream deployment config and C++ flow after sim2sim works.

Environment:

- local docs plus upstream code inspection

No Hack:

- no guessed tensor shapes
- no undocumented unit conversion
- no broad copy of upstream code

Hardware:

- not hardware-bound

# Log

- Pending. Do after official sim2sim succeeds.

# Review

Result: pending
Syntax:
Hack:
Scope:
Efficiency:
Hardware:
Verify:
Findings:
