# Whole-Body Procedural Locomotion Architecture

## Decision

The canonical research contract is `whole_body_v1_45`: a LocoFormer-style
lower-body superset (four limbs × seven slots) plus three waist slots and two
seven-slot arms.  Wheel slots remain reserved and masked in the current
biped/quadruped training distribution.  Head, neck, fingers, grippers,
vision, and manipulation are out of scope for this version.

Reference: [LocoFormer](https://arxiv.org/html/2509.23745v1).  Its published
28-slot lower-body superset is treated as the starting point, not as a vendor
asset or an unverified implementation dependency.

The actor observation is 193 dimensions:

```text
base linear velocity 3 + base angular velocity 3 + projected gravity 3
+ command 3 + joint position 45 + joint velocity 45 + previous action 45
+ active mask 45 + trial-start flag 1
```

The actor never receives morphology identifiers, physical parameters, or
motor-failure labels.  A critic may receive privileged simulator state.

## Ownership

`MorphologyBlueprint` owns discrete topology and semantic slot mapping.
`PhysicalParams` owns continuous geometry/dynamics samples.
`MorphologyInstanceKey` identifies one exact blueprint + physical realization;
it is the minimum safe cache key for any derived kinematic/dynamic state.
`StanceSolution` owns an absolute base height and absolute compiled-model joint
coordinates, and is valid only for its source `MorphologyInstanceKey`. It is
runtime-derived state and must not be stored on `MorphologyBlueprint`.
`MotorProcess` owns hidden, time-varying actuator conditions.
The velocity task owns command tracking, reward, and termination.  PPO/TXL
consume the rollout contract and do not import a simulator or robot module.

The reference runtime is `WholeBodyMuJoCoShard`: each fixed-topology shard
owns one compiled MJCF model and one `MjData` per environment, while
`WholeBodyRolloutMux` aggregates shards. The active development target is the
RTX 5060 Ti. Its safe local profile starts at 4 topology shards × 64
environments (256 total); the architecture still preserves the 8 × 256 and
16 × 256 scale targets for later measured expansion. The CPU reference shard
is replaceable by MJLab without changing the task contract.

The first trainable baseline is a masked MLP/PPO specialist.  The shared MLP
uses per-environment active masks, then GRU validates reset semantics before
the canonical TXL (6 layers, hidden 256, 8 heads, 128-step memory).  MIP/JiT/
flow matching remains an algorithm adapter: no fabricated PPO likelihood is
allowed when a flow policy cannot evaluate `log_prob`.

Topology is fixed for a compiled simulator shard/context.  Motor conditions may
change during a trial; `trial_done` resets physical state while preserving
recurrent context, and `context_done` resamples the physical context and clears
memory. Multiple `MjData` objects sharing the same compiled blueprint + physical
instance may share one stance solve. A physical resample must select/recompile
the corresponding instance and obtain a stance with the new exact key; a
topology-only stance cache is invalid.

## Checkpoint compatibility

Checkpoint compatibility has three independent identities:

- tensor `schema_hash`: the 45D slot and 193D observation layout only;
- `embodiment_contract_version/hash`: contact geometry, reset stance, and action
  midpoint semantics;
- `manifest_hash`: the concrete embodiment realization or full training
  distribution, including physical configuration and stance-solver contract.

Changing stance or foot geometry does not by itself change the tensor schema.
It must change the embodiment contract and concrete manifest instead. Loaders
compare all expected identities and reject legacy checkpoints missing the
embodiment fields. A solved stance already includes physical nominal offsets in
its absolute coordinates; reset and action code must not add them again.

## Evaluation split

Training uses only primitive box/capsule generated bipeds and quadrupeds.
Berkeley Humanoid, ANYmal C, G1, and Go2 are never used to fit the generator,
reward, or checkpoint selection.  Berkeley and ANYmal C are named OOD tests;
G1 and Go2 are regression tests.

Pinned external revisions are recorded in the task spec: Unitree MJLab
`1425b15f73bd4095f0df53709d7c389c3eb9e790` and MuJoCo Menagerie
`da76818e269b82289eba39808e2fb91d679d6994`.
