# Task 051: Procedural Whole-Body Generator

## Route

Implement deterministic primitive-link biped/quadruped grammar, physical
parameter sampling, MJCF compilation, structural manifests, and validation.

## Log

- 2026-08-19: Added the first dependency-light generator with box/capsule
  links, variable hip/ankle/wrist/waist topology, optional biped arms, and
  physics sampling.

## Review

Verified: `generator_smoke_1000x100.json` contains 2,000 deterministic
biped/quadruped records (1,000 per family), all MJCF models compile, all 100
passive steps remain finite and bounded, all generated link masses/inertias are
positive, default poses are within limits, and all 2,000 structural hashes are
unique.  The same artifact records arm/no-arm variation and actuator counts.
`morphology_split_manifest.json` records the 64/16/16 train/validation/heldout
topologies, structural hashes, and parameter seeds.  Long PPO training remains
an explicit RTX 5060 Ti follow-up; the MuJoCo reference shard is the current
backend.
