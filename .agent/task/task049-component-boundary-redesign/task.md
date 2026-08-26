# Task 049: Component Boundary Redesign

## Route

Redesign new experiments around independently replaceable task, policy, and
algorithm components. Derive only the boundaries required by ownership and
change rate, keep simulator/runtime details at the composition edge, and avoid
a big-bang rewrite of historical experiments.

Slices:

1. `001-first-principles-boundary.md`
   - Record the MDP-derived ownership and allowed dependency graph.
   - Classify MIP/flow/diffusion/JiT as policy parameterizations.
2. `002-neutral-contract-and-composition.md`
   - Add framework-neutral specs, protocols, transition data, strict component
     configs, and a minimal interaction composition.
3. `003-dependency-guard-and-handoff.md`
   - Enforce dependency direction and verify config/interaction behavior.
   - Document the incremental migration path for legacy modules.

Acceptance gate:

- task, policy, algorithm, and experiment configs load independently;
- task owns observation/action spaces and no policy config duplicates them;
- incompatible algorithm/policy capabilities fail at composition time;
- core/task/policy/algorithm dependency direction is mechanically tested;
- a fake end-to-end interaction/update loop passes without Torch or simulator;
- full repository tests and critical Ruff checks pass.

## Log

- 2026-08-19 Opened after the user requested a first-principles redesign with
  task and algorithm separated.
- 2026-08-19 Selected four independent change axes: backend physics, task MDP,
  policy/action generator, and algorithm/update rule. Experiment is the sole
  composition root rather than a fifth research abstraction.
- 2026-08-19 Added neutral contracts, separately owned configs, capability-based
  composition validation, an algorithm-agnostic interaction loop, and a
  dependency-direction guard.
- 2026-08-19 Migrated the concrete Genesis G1 velocity-tracking task, legacy
  tanh-Gaussian actor-critic, and GAE/clipped-PPO update kernel into `tasks/`,
  `policies/`, and `algorithms/`; old imports now delegate to those owners.
- 2026-08-19 Final verification: full repository pytest `724 passed` (35
  upstream TorchScript deprecation warnings), focused full-rule Ruff passed,
  repository critical Ruff passed, both inspection commands passed, and
  `git diff --check` passed.

## Review

Status: passed. The separation is enforced by executable contracts and import
guards, and existing call sites retain behavior through identity-tested
compatibility imports. This does not claim the external official MJLab runner
has been rewritten around the new composition root; that migration remains an
experiment-level follow-up.
