# Route

Task: task001-agent-setup

Goal: Point the repo README and AGENTS instructions at the `.agent` workflow.

Scope:

- `README.md`
- `AGENTS.md`

Verify:

- `Get-Content README.md`
- `Get-Content AGENTS.md`

Environment:

- local only

No Hack:

- no unrelated rewrite
- no fake completed external validation

Hardware:

- keep H200/RTX split visible

# Log

- Updated `README.md` and `AGENTS.md` to make `.agent/index.md` the project entrypoint.

# Review

Result: passed
Syntax: markdown only
Hack: no fake simulator validation
Scope: limited to project entrypoint docs
Efficiency: lightweight docs only
Hardware: H200/RTX split kept visible
Verify: README and AGENTS updated
Findings: none
