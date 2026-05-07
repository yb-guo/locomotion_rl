# Task 006 H200 Run Notes

## 2026-05-07

Target repo path:

```text
/root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/repo_replay_smoke
```

Important H200 execution detail: run from `/tmp` with explicit
`PYTHONPATH=/root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/repo_replay_smoke/src`.
An earlier sync left a stale top-level `h200_locomotion_lab` package under the
repo root, which can shadow `src/` when running from the repo directory.

Verification:

```text
Local full pytest: 44 passed
H200 targeted pytest from /tmp: 18 passed
Local ruff: not run; ruff is not installed locally
```

Decoder-only Genesis closed-loop:

```text
10 frames, token replay, official captured history init: passed
base_z final: 0.6546086668968201
action max abs: 5.1907243728637695

20 frames, same settings: failed height gate
base_z final: 0.2644214928150177
action max abs: 8.832411766052246
```

Conclusion: SONIC decoder is now connected to Genesis state/action feedback for
a short smoke, but stable closed-loop rollout is not passed.
