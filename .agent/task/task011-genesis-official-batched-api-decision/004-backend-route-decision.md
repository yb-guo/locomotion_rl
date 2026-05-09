# Route

Task: task011-genesis-official-batched-api-decision

Goal: write the decision report that maps Franka, Go2, and G1 evidence to the
next backend route.

Scope:

- Summarize environment/version evidence.
- Summarize Franka official batched baseline.
- Summarize Go2 official locomotion baseline.
- Summarize G1 target asset/backend probe.
- Apply stop rules and classify blockers.
- Select the next task route.

Environment:

- local documentation update
- remote evidence must come from guarded H200 runs

Verify:

- report contains all required throughput metric definitions and observed
  values for attempted loops.
- report distinguishes build performance from steady-state performance.
- report distinguishes Genesis/H200 environment blockers from G1 asset/backend
  blockers.
- top-level task remains pending unless all required subtasks and review pass.

No Hack:

- Do not claim Genesis is unsuitable if only G1 failed after official assets
  passed.
- Do not claim G1 is suitable without tensor device and selected reset evidence.
- Do not hide skipped subtasks; cite the stop rule that skipped them.

Hardware: local docs plus H200 evidence from prior subtasks.

# Log

- 2026-05-09 Evidence source:
  H200 guarded runs saved under remote project
  `/root/agent_workspace/project/h200-locomotion-lab-task011-genesis-official-batched-api-decision/outputs/task011/`.
  Remote package evidence: `genesis-world==0.4.6`, `torch==2.5.1`.
  GPU inventory: two NVIDIA H200 GPUs, each 143771 MiB. Benchmarks used
  `CUDA_VISIBLE_DEVICES=1`, `physical_gpu=1`, and logical
  `logical_cuda_device=cuda:0`.
- Probe implementation:
  `h200_locomotion_lab.tools.genesis_official_batched_api_probe` is independent
  of `GenesisG1SceneBackend`. It builds Genesis scenes through official
  `gs.Scene`, `gs.morphs.MJCF` for Franka/G1, and `gs.morphs.URDF` for Go2.
  It excludes render, GIF/video, SONIC, ONNX, planner, reward, PPO, and scalar
  wrapper loops.
- Metric definitions:
  `build_time_s` measures only Genesis scene/entity construction through
  `scene.build(n_envs=...)`; `warmup_time_s` and `measure_time_s` exclude
  build time. `policy_steps_per_sec = measure_policy_steps / measure_time_s`.
  `sim_steps_per_sec = measure_policy_steps * decimation / measure_time_s`.
  `env_policy_steps_per_sec = measure_policy_steps * n_envs / measure_time_s`.
  `env_sim_steps_per_sec = measure_policy_steps * decimation * n_envs / measure_time_s`.
  Runs used `warmup_policy_steps=20`, `measure_policy_steps=100`, and
  `decimation=4`.
- Franka official MJCF baseline passed through `n_envs=1024`.

| n_envs | build_time_s | env_policy_steps_per_sec | env_sim_steps_per_sec | selected reset |
| ---: | ---: | ---: | ---: | --- |
| 1 | 10.328132 | 230.953171 | 923.812686 | not applicable |
| 16 | 52.939369 | 3536.146575 | 14144.586300 | verified target-only |
| 256 | 20.996516 | 61171.498607 | 244685.994430 | verified target-only |
| 1024 | 21.236085 | 225980.836743 | 903923.346973 | verified target-only |

- Go2 official URDF/floating-base baseline passed through `n_envs=1024`.
  These values are from the fixed probe version that records
  `root_vel_device=cuda:0` and verifies selected root qpos plus joint reset for
  batched runs.

| n_envs | build_time_s | env_policy_steps_per_sec | env_sim_steps_per_sec | selected reset |
| ---: | ---: | ---: | ---: | --- |
| 1 | 15.657439 | 161.341359 | 645.365435 | not applicable |
| 16 | 15.653096 | 3441.715713 | 13766.862852 | verified root+joint target-only |
| 256 | 15.509148 | 44440.862851 | 177763.451404 | verified root+joint target-only |
| 1024 | 17.131138 | 165521.561041 | 662086.244165 | verified root+joint target-only |

- G1 default target asset passed through `n_envs=1024`; optional `4096` also
  passed after `1024` showed healthy memory. These values are from the fixed
  probe version that records `root_vel_device=cuda:0` and verifies selected
  root qpos plus joint reset for batched runs.

| n_envs | build_time_s | env_policy_steps_per_sec | env_sim_steps_per_sec | selected reset |
| ---: | ---: | ---: | ---: | --- |
| 1 | 63.829883 | 72.098978 | 288.395914 | not applicable |
| 16 | 65.852252 | 570.518403 | 2282.073611 | verified root+joint target-only |
| 256 | 66.439508 | 6074.550575 | 24298.202302 | verified root+joint target-only |
| 1024 | 60.015459 | 22266.149334 | 89064.597337 | verified root+joint target-only |
| 4096 optional | 28.308660 | 78397.335490 | 313589.341962 | verified root+joint target-only |

- All Franka, Go2, and G1 attempted final runs emitted `status=ok`,
  `tensor_device_ok=true`, `action_device=cuda:0`, `qpos_device=cuda:0`,
  `dofs_pos_device=cuda:0`, `dofs_vel_device=cuda:0`,
  `root_pos_device=cuda:0`, and `root_quat_device=cuda:0`. Go2 and G1 fixed
  runs also emitted `root_vel_device=cuda:0`.
- Stop-rule application:
  Franka did not fail, so Go2 was allowed. Go2 did not fail, so G1 was
  allowed. G1 did not fail, so no blocker route was selected.
- Blocker classification:
  no H200/Genesis CUDA environment blocker; no official batched API blocker;
  no Go2 locomotion/floating-base/URDF blocker; no G1 target asset/backend
  blocker for official tensor I/O, batched build, stepping, or selected reset.
  The remaining G1 concerns are performance and asset-quality warnings.
- Non-blocking warnings to carry forward:
  Genesis warns that `torch<2.8.0` is unsupported while the environment has
  `torch==2.5.1`; G1 emits `trimesh` center-of-mass warning and dubious ankle
  link mass warnings. During the reviewer fix, the `set_pos`/`set_quat`
  selected root-reset path exposed a G1-specific Genesis message that selected
  pose writes to fixed-link geometries require `batch_fixed_verts=True`; the
  final evidence uses selected `set_qpos` root reset instead. These should be
  tracked in the next backend task but did not invalidate this route decision.
- Follow-up G1 performance diagnostics after the route decision:
  at `n_envs=1024`, `performance_mode=True` is not accepted by Genesis 0.4.6
  `MJCF`; `convexify=True` reached 22490.678338 env policy steps/s and
  `decimate=True` reached 22418.654409 env policy steps/s, versus the fixed
  default baseline of 22266.149334. These single-variable probes do not explain
  or materially improve the G1 slowdown, so the next backend task should
  profile solver/contact/entity counts and tensor API call costs rather than
  assuming mesh convexification or decimation is enough.
- Component profiling confirmed the bottleneck is raw physics stepping, not
  tensor API overhead. At `n_envs=1024`, G1 29DoF took 4.380875s for 400 raw
  `scene.step()` calls, while Go2 took 0.522974s. G1 action writes and state
  reads were each only about 0.02s for 100 policy iterations.
- Existing simpler G1 assets were inventoried without downloads. The best
  directly runnable candidate is `g1_27dof_nohand.xml`, which reached
  45827.527990 env policy steps/s at `n_envs=1024`, about 2.1x the 29DoF target
  default. `g1_27dof_fakehand.xml` also ran at 44164.266383 env policy steps/s.
  Several nominally simpler 12DoF/23DoF assets were blocked by existing mesh or
  importer issues, so they need an asset-cleanup task before Genesis training
  use.
- Decision:
  proceed to `VectorizedGenesisBackend`.

Rationale:

- Official Genesis batched scene construction and tensor I/O work on H200 with
  physical GPU 1 isolated as logical `cuda:0`.
- Selected reset was verified empirically for target-only behavior on all
  batched Franka, Go2, and G1 runs. Franka uses selected joint reset; Go2 and
  G1 use selected root qpos plus selected joint reset.
- G1 default asset reaches `n_envs=1024` and optional `4096` without OOM or
  tensor-device fallback, so the next route should implement a backend around
  the official batched API rather than fixing the environment, switching PPO
  backend, or first creating an asset simplification task.
- Build times for G1 are high and should be separated from steady-state
  training metrics in the next task.
- The next route should use 29DoF for policy-contract fidelity checks and add a
  parallel training-asset profile around `g1_27dof_nohand.xml` for faster PPO
  iteration.
- Follow-up implementation:
  the parallel training-asset profile has been added as
  `configs/robots/unitree_g1_27dof_nohand_genesis.yaml` with a separate
  `h200_locomotion_lab.robots.g1_27dof_nohand` loader. This keeps the existing
  29DoF SONIC profile strict while giving `VectorizedGenesisBackend` a
  validated 27D/90D no-hand training asset profile backed by the H200
  throughput and component-profile evidence above.
- Local profile verification:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_27dof_nohand_profile.py tests/test_robot_profile_loader.py -q -p no:cacheprovider`
  -> 27 passed.

# Review

Status: passed.

Read-only review completed after the root-reset/root-velocity fixes and found
no blocking or important findings. The decision report selects exactly one
route, `VectorizedGenesisBackend`, and does not classify H200/Genesis, Go2
URDF/floating-base, or G1 official tensor API behavior as blocked.
