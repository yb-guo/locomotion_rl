# 003 Procedural G1-like Morphology Generator

## Route

Generate a small train/held-out distribution of G1-like morphology variants.

First generator scope:

- fixed high-level topology and joint-slot semantics;
- random thigh/calf/foot/link lengths within a declared train range;
- mass, COM, and inertia follow deterministic scaling rules;
- motor properties reuse Task029/Task030-style randomization where possible;
- train split and held-out split are explicit and reproducible by seed.

Suggested first ranges:

- train link scale: `0.95-1.05`;
- near holdout: `0.90-0.95` and `1.05-1.10`;
- OOD holdout: `0.80-0.90` and `1.10-1.20`.

## Minimal Closed Loop

Close this slice in two stages:

1. Local manifest loop: generate or specify a tiny manifest with one train
   variant and one held-out variant, deterministic by seed, without downloading
   assets.
2. H200 smoke loop: load at least one train and one held-out variant and write a
   small JSON record with morphology id, split, scale factors, slot schema id,
   joint order hash, and action dim.

## Evidence Gate

Reviewable evidence must include:

- manifest path or JSON fixture;
- deterministic seed and variant ids;
- train/held-out split labels;
- declared scale ranges actually used;
- explicit limitation notes for any nonphysical mass/COM/inertia scaling;
- H200 load-smoke command output or JSON before claiming simulator load.

Acceptance:

- Generate at least one local manifest without downloading assets.
- H200 loads at least one train variant and one held-out variant.
- Joint/action slot order is unchanged through the selector contract.
- Smoke step writes JSON with morphology id, split, scale factors, and action
  dim.
- If any mass/COM/inertia rule is nonphysical, it is explicitly recorded as a
  limitation.

## Subagent Ownership

- Worker owns only morphology manifest/config docs and small generator/smoke
  artifacts if implementation is later authorized.
- Worker must consume the `002` slot contract and must not invent alternate
  joint-order rules.
- Worker must not download assets, checkpoints, datasets, simulator assets, or
  upstream repos.
- Reviewer checks reproducibility, split separation, slot compatibility, and
  H200 evidence before any load claim is accepted.

## Failure Exit

If variants require downloaded assets, change topology, or break the fixed
action dimension, stop and report the blocked condition.

## Log

- 2026-05-29 Opened after deciding to start from G1-like variants only.
- 2026-05-29 Implemented the local deterministic manifest generator in
  `src/h200_locomotion_lab/robots/g1like_morphology.py`. The generator consumes
  the `002` `g1like_slots` action schema, records `action_dim=29`,
  `slot_schema_id`, `slot_schema_hash`, and `joint_order_hash`, and emits a
  small serializable manifest with train and held-out variants. No XML, assets,
  upstream repos, downloads, simulator startup, or H200 load path were touched.
- 2026-05-29 Added local tests in
  `tests/test_task038_g1like_morphology_manifest.py` covering deterministic
  same-seed output, seed-dependent ids/scales, train and held-out splits,
  held-out ranges outside the train range, 002-derived action/schema/order
  hashes, invalid split/range/action-dim rejection, and nonphysical
  mass/COM/inertia limitation notes.
- 2026-05-29 Verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_g1like_morphology_manifest.py`
  -> `15 passed in 0.09s`.
- 2026-05-29 Addressed reviewer blocking finding: manifest validation now
  rejects illegal `heldout_band` values and validates held-out out-of-train
  scales against the manifest's selected band only (`near` or `ood`) instead of
  accepting either declared band. Added regression tests for near manifests with
  OOD scales, OOD manifests with near scales, and illegal `heldout_band`.
- 2026-05-29 Verification after reviewer fix:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_g1like_morphology_manifest.py`
  -> `18 passed in 0.07s`.
- 2026-05-29 Addressed reviewer boundary blocker: held-out band validation now
  accepts values inside the manifest-selected declared ranges even when a value
  lies on the shared near/OOD boundary. It still rejects true band mixing such
  as near manifest scale `0.85` and OOD manifest scale `1.075`. Added a
  regression test proving `seed=5494`, `heldout_band="near"` generates and
  validates successfully.
- 2026-05-29 Verification after shared-boundary fix:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_g1like_morphology_manifest.py`
  -> `19 passed in 0.08s`.
- 2026-05-29 Review subagent rechecked the shared-boundary fix and reported no
  blocking findings.
- 2026-05-29 Follow-up local artifact loop moved to
  `007-g1like-mjcf-patch-load-smoke.md`: generated manifest variants can now be
  converted into conservative patched MJCF XML artifacts with local
  ElementTree parse/contract JSON evidence. This is not H200 load evidence.

## Review

Status: local manifest closed with reviewer confirmation. H200 load smoke
remains pending and is required before this subtask can claim simulator-load
compatibility.

Evidence:

- Local generator file:
  `src/h200_locomotion_lab/robots/g1like_morphology.py`.
- Local test file:
  `tests/test_task038_g1like_morphology_manifest.py`.
- Command results:
  - Initial local manifest verification: `15 passed in 0.09s`.
  - Reviewer band-validation fix: `18 passed in 0.07s`.
  - Shared-boundary band fix: `19 passed in 0.08s`.

Notes:

- Local manifest loop is implemented and verified with deterministic seed
  coverage and 002 slot compatibility checks.
- `007` provides the next local bridge from manifest variant to patched MJCF
  artifact and JSON contract for future H200 load smoke.
- H200 load smoke remains pending. This subtask does not claim a simulator load
  pass or H200 compatibility pass.
