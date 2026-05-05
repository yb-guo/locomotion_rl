# Route

Task: task002-sonic-mujoco-smoke

Goal: Run the official SONIC MuJoCo sim2sim loop end to end.

Scope:

- upstream SONIC command logs
- generated smoke-test notes
- no local policy rewrite

Verify:

- Simulator process starts.
- Deployment/control process connects.
- Checkpoint or engine loads.
- Robot moves in MuJoCo.
- Failure mode is captured if it does not run.
- Terminal 1: `source .venv_sim/bin/activate && python gear_sonic/scripts/run_sim_loop.py`
- Terminal 2: `cd gear_sonic_deploy && bash deploy.sh sim`

Environment:

- Linux target machine

No Hack:

- no fake success from partial startup
- no editing upstream code without recording the patch
- no accepting wrong TensorRT version silently

Hardware:

- H200 is acceptable
- record GPU utilization only if available

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
