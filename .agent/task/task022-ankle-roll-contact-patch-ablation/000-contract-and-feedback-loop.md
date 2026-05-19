# Subtask 000: Contract And Feedback Loop

## Route

- Create the task contract before coding.
- Keep task022 diagnosis-first: patch generator, static XML evidence, then H200
  zero-action/link traces.
- Do not run PPO or modify the prepared source asset.
- Generate all patched XML files under the project output directory.

## Feedback Loop

The first runnable loop must be local and deterministic:

```text
patch generator reads a fixture/source XML -> writes named patched XML variants
-> emits JSON with exact changed bodies/geoms/attrs -> focused tests assert the
expected XML structure
```

H200 runtime loop comes only after static XML tests pass:

```text
run source asset and patch variants with zero-action/link trace -> compare
first tilt, root height/upright, ankle-roll contact force, and link contact env
counts
```

## Ranked Hypotheses

1. Larger ankle-roll support geoms delay or remove the tilt wave.
2. Explicit friction/contact attrs improve contact stability.
3. Contact-only patches are insufficient if mass/inertial mismatch dominates.

## Stop Rules

- Stop if patch generator cannot preserve the original XML except the intended
  target changes.
- Stop if generated XML is outside `/root/agent_workspace/project` on H200.
- Stop if no patch improves first tilt by at least 10 policy steps.

## Log

- 2026-05-13 Created with task022.

## Review

Status: pending.
