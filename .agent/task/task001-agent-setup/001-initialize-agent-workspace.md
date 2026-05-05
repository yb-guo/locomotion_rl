# Route

Task: task001-agent-setup

Goal: Create the `.agent` workspace and project-level docs.

Scope:

- `.agent/index.md`
- `.agent/doc/project.md`
- `.agent/doc/h200_strategy.md`
- `.agent/doc/sonic.md`
- `.agent/doc/locoformer.md`

Verify:

- `Get-ChildItem -Recurse -File .agent`

Environment:

- local only

No Hack:

- no generated checkpoints
- no downloaded assets
- no hidden dependency on HeadPose paths

Hardware:

- document H200 constraints explicitly

# Log

- Created the `.agent` entrypoint and long-lived docs.

# Review

Result: passed
Syntax: markdown only
Hack: no downloaded assets or external paths
Scope: limited to `.agent` workspace and docs
Efficiency: lightweight docs only
Hardware: H200 constraints documented
Verify: `.agent` files created
Findings: none

