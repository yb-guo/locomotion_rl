# LocoFormer Plan

LocoFormer is treated as a research reproduction target, not a first-day full
scale reproduction.

## Minimal Reproduction

The first useful target is:

- One robot family: G1-like humanoid variants first.
- Small morphology variation.
- Short but real context window.
- PPO or PPO-like baseline.
- Transformer policy core behind a stable observation/action schema.
- Online adaptation buffer.

## Current Direction

Task038 narrows the first LocoFormer-style reproduction to a G1-like family:

- fixed high-level humanoid topology;
- fixed unified joint-slot semantics and action dimension;
- randomized G1-like link lengths, mass, COM, inertia, and motor dynamics;
- held-out G1-like morphology variants for evaluation;
- multi-trial final-trial evaluation with memory retained across inner resets;
- comparison between non-transformer baselines and a true TXL memory policy.

Do not claim full LocoFormer reproduction from Task038. The first claim target is
only: a LocoFormer-style minimal reproduction where TXL long memory improves
held-out G1-like morphology/dynamics adaptation over MLP/GRU/AdaptK baselines.

## Task069 Morphology Boundary

Task069 adds an explicitly opt-in `locoformer_paper_faithful_morphology_v1`
profile.  Its narrow claim is that the repository implements a verifiable
procedural morphology envelope consistent with the public paper description:
biped, quadruped, wheeled biped, and wheeled quadruped families.  The public
sources establish those family categories in Appendix A.1/Figure 6 and the
procedural-training-body point in §2.1 Task Generation; they do not publish the
exact morphology generator.  Primitive dimensions, joint ranges, wheel ranges, and
randomization intervals are therefore repository implementation choices.

The profile has real terminal wheel joints, rolling contact geoms, motor
actuators, and per-limb wheel slots.  Its profile and embodiment contract are
separate from the frozen
`procedural_whole_body_v2_footpad_actual_stance_feedforward` legacy profile;
legacy XML, manifests, cache keys, and checkpoint identity must remain stable.
The Task069 gallery and matrix artifacts are the evidence for this morphology
claim, not evidence of policy or training parity.  Reset floor/self-contact and
rollout penetration diagnostics are reported as morphology/smoke evidence only;
they do not establish a stance solution or training readiness.

Within the frozen Task069 claim, still out of scope: official generator/source-code
reproduction, named-robot parameter parity, TXL/long-context quality, large-scale RL,
sim2real, and real-robot deployment.  G1, H1, GR1, TRON1, Berkeley Humanoid, A1,
Spot, ANYmal C, TRON1-W, and Go2-W remain unseen evaluation embodiments rather
than hard-coded training templates.

## Task070 Engineering Morphology Direction

Task070 moves beyond Task069's minimum family envelope toward an engineering-useful,
standable morphology distribution.  It uses generic archetypes distilled from mature
robot kinematic design principles, normalized geometry ratios, constrained topology
variation, and a newly verified contact-aware stance.  Task069's profile and contract
remain frozen; Task070 must use a separate profile, contract, cache, and checkpoint
identity.

Named robots may be quantitative training priors or held-out evaluation embodiments,
but not both within the same claim.  Task070 uses a source- and license-audited
multi-vendor prior pool: G1 and at least one cleared EngineAI humanoid for the biped
centers, plus Spot, at least one cleared Unitree quadruped, and at least one cleared
Deep Robotics quadruped for the quadruped centers.  Exact models and files beyond G1
and Spot are selected and frozen during R0; a public repository alone is not clearance.
G1 and Spot remain explicit quantitative training priors and therefore seen/reference.
Every additional model used to set topology, ratios, ranges, distance normalization, or
gates becomes seen/reference as well and must not be reported as held-out/OOD.

Cleared descriptions may contribute only registry-declared generic joint trees, joint
semantics, attachment relationships, normalized ratios, and permitted scaling hints.
Task070 outputs use repository-generated primitive geometry and must not carry vendor
meshes, textures, logos, exact model identity, or parameter-parity claims.  Spot source
must retain its custom simulator/hardware restrictions in the registry rather than being
treated as a generic permissive asset.  Third-party or reverse-engineered descriptions
without verified provenance and file-level permission remain design-only or excluded.

Sampling is multi-center and stratified: prior neighborhoods, interpolation between
centers, and a bounded outward band inside frozen engineering limits.  Each generated
instance records source contributions, nearest prior, normalized distance, region, and
a clone guard; arbitrary unbounded extrapolation is not part of the Task070 claim.  A
separate held-out registry contains only embodiments unused for topology, ratio, range,
distance, or gate calibration.  Any leave-one-vendor-out result must rebuild those
choices without reading the excluded vendor.  These Task070 role changes do not rewrite
the frozen Task069 evidence or contract.

For Task070, "standable" means that the generated instance has a bound contact-aware
stance which survives the declared biped base-attitude hold, wheeled-biped active wheel
balance, quadruped position feedforward, or wheeled-quadruped zero-velocity hold gate.
It does not mean passive zero-torque standing, learned locomotion, policy parity,
sim2real readiness, or real-robot deployment.

Execution note, 2026-08-24: Task070 now has the opt-in
`archetype_constrained_morphology_v1` profile and
`procedural_archetype_constrained_morphology_v1` contract.  Its round-2 remediation R0
registry/license artifacts clear `unitree_g1`, `engineai_pm01`, `spot_base`,
`unitree_go2`, and `deeprobotics_lite3` as seen quantitative priors only and freeze the
actual executed 12D prior centers, feature limits, geometry envelope, fail-closed
sampler retry policy, sampled mass/scale-aware actuator scaling, and stance gates.
The remediation R4 matrix verifies 128/128 generated instances across biped,
quadruped, wheeled biped, and wheeled quadruped with 1000 steps at 2 ms, including
support polygon, static contact residual, 6D contact-wrench force/torque residual,
nonterminal-support exclusion, self-contact and wheel-wheel exclusion, region-band and
clone guard gates; R4 visual review binds a 174-image gallery manifest; R5 now
fail-closes on exact four-family, 128-record, seed `0..31`, per-region matrix
completeness and focused/full regression passed.  The task state remains
`execution_verified_pending_independent_readonly_review`, not `passed`.

### Task070 V2 Motor-Config Boundary

The LocoFormer paper does not publish a real-robot motor configuration table or an
official morphology-generator implementation. Its disclosed control interface emits
target joint positions in a unified superset joint space. Its procedural training
robots intentionally do not incorporate exact market-robot parameters; instead, the
paper reports broad randomization of control gains, joint limits, mass, center of mass,
and other standard dynamics parameters. Therefore Task070 must not present copied
vendor motor values as a LocoFormer reproduction or as exact named-robot parity.

Task070 v2 now keeps two explicit layers. The parsed source descriptor records local,
auditable motor hints and their provenance: the G1 companion configuration contributes
5020/7520/4010 class effort, speed, reflected inertia, and derived PD values; the Go2
companion configuration contributes hip/thigh/calf effort, PD, and armature; PM01 and
Lite3 contribute URDF effort/velocity limits. Spot's uniform `1000/1000` URDF limits
are retained as raw evidence but rejected as physical motor priors. Terminal wheel
motors remain a declared local engineering module because no authorized mature wheel
motor source is present.

The emitted anonymous MJCF consumes trusted source values only as scaling hints. It
records raw/proxy and final values separately, applies lever-aware gain/armature and
effort scaling to the anonymous primitive linkage, and keeps exact named-robot parameter
parity false. Source velocity limits are metadata for a future controller-side speed
gate; MuJoCo position actuators do not directly enforce a joint-speed limit attribute.
The eventual training distribution must still randomize gains and dynamics around these
centers rather than train on one fixed config.

## Agent Submodules

- `morphology_encoder`
- `proprio_tokenizer`
- `motion_context_encoder`
- `transformer_policy`
- `actor_critic_heads`
- `adaptation_buffer`

## Acceptance

Do not call it a LocoFormer reproduction until:

- A non-transformer baseline exists.
- The transformer policy runs in the same environment.
- Metrics show comparable or better adaptation under held-out dynamics or
  morphology randomization.
- Failure cases and hardware cost are recorded.
