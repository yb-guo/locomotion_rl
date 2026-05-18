# Task025: SONIC mjlab Adapter

## Goal

Run GEAR-SONIC as an external controller inside the current `unitree_rl_mjlab`
MuJoCo/ManagerBasedRlEnv stack and render rollout evidence.

This is adapter work, not training. The first version must not modify the
RSL-RL runner or the mjlab task implementation.

## Route

1. Add a backend adapter that exposes mjlab G1 as the existing
   `G1RobotBackend` contract.
2. Reuse the existing SONIC scalar runtime boundary:
   `ActionProvider -> ScalarG1Runtime -> G1MotorCommand -> G1RobotBackend`.
3. Keep planner/encoder/decoder runtime isolated behind small wrappers.
4. Start with zero/sequence action smokes before online planner rollout.
5. Record numeric evidence and video/contact-sheet output when H200 execution
   succeeds.

## Subtasks

- `001-contract-audit.md`
- `002-mjlab-backend-smoke.md`
- `003-sonic-sequence-replay.md`
- `004-online-planner-encoder-rollout.md`
- `005-review.md`
- `006-sonic-dependency-setup.md`
- `007-alignment-diagnosis.md`
- `008-command-and-actuator-alignment.md`
- `009-ankle-pitch-residual.md`
- `010-target-clamp-probe.md`
- `011-remaining-alignment-diagnosis.md`
- `012-action-history-and-range-trace.md`
- `013-official-sonic-contract-audit.md`
- `014-effective-action-history-probe.md`
- `015-official-sim2sim-target-contract.md`
- `016-official-limit-overlay-probe.md`
- `017-official-install-bootstrap.md`
- `018-official-sim2sim-runtime.md`
- `019-official-dds-lowstate-probe.md`
- `020-official-deploy-lowstate-instrumentation.md`
- `021-official-vs-mjlab-action-trace-comparison.md`
- `022-matched-input-official-planner-comparison.md`
- `023-official-motion-replay-plant-response.md`

## Acceptance

- mjlab backend reads finite 29DoF G1 state in SONIC MuJoCo command order.
- backend maps SONIC motor targets to mjlab joint-position actions by joint
  name, not by assumed index.
- zero-action smoke can step mjlab without non-finite state.
- SONIC sequence or online rollout records finite actions, root z, displacement,
  and rendered video or contact sheet.
- Failures document whether the blocker is joint order, action scale/default
  offset, motor gains, asset mismatch, or SONIC runtime.

## Log

- 2026-05-15 Opened task after task024 confirmed `unitree_rl_mjlab`
  `Unitree-G1-Flat` can render current RSL-RL checkpoint videos.
- 2026-05-15 Remote mjlab inspection found `joint_pos.target_names` matches the
  SONIC 29DoF command MuJoCo order:
  left leg, right leg, waist, left arm, right arm.
- 2026-05-15 H200 synthetic sequence smoke rendered a 120-step video with no
  termination. This verified CSV sequence plumbing only; it was not an official
  SONIC replay and the robot fell under repeated fixture actions.
- 2026-05-15 Added a `sonic` optional Python dependency extra for adapter-side
  ONNX helper code: `numpy` and `onnx`. H200 `unitree-rl-mjlab` already has
  these Python modules installed; Python `onnxruntime` is not required for the
  current adapter path.
- 2026-05-15 Downloaded a Linux CPython 3.11 wheelhouse locally and uploaded it
  to H200 for offline SONIC Python dependency setup:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/sonic_wheelhouse_linux_cp311`.
- 2026-05-15 Downloaded and uploaded ONNX Runtime C++ Linux x64 1.19.2, then
  built `bin/sonic_planner_ort_runner` on H200. The binary links against the
  uploaded runtime and starts correctly.
- 2026-05-15 Checked local agent/download paths before downloading. Official
  SONIC ONNX artifacts were not already present locally.
- 2026-05-15 User explicitly requested local download then upload. Downloaded
  official `nvidia/GEAR-SONIC` artifacts into
  `.external_downloads/gear_sonic_artifacts`, uploaded them to
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/gear_sonic_artifacts`,
  and verified remote SHA256.
- 2026-05-15 H200 true online rollout with official planner/encoder/decoder
  ran for 40 and 160 steps with no terminations and rendered videos. The
  160-step run moved forward about 1.27 m while root height dropped about 9 cm.
- 2026-05-15 Ran longer 400-step and 800-step online rollouts with normal mjlab
  terminations enabled. Neither triggered `fell_over`; the 800-step run moved
  about 7.96 m and ended with root z about 0.738. Contact sheet inspection
  shows the robot remains upright enough to step, but with a low, crouched,
  backward-leaning posture.
- 2026-05-15 Added alignment diagnosis after the 800-step visual review. The
  strongest immediate mismatch is mjlab HOME reset/action offset versus SONIC
  crouched default angles; hip-pitch action scale/controller constants also
  differ between mjlab and the SONIC profile.
- 2026-05-15 Added a deterministic alignment trace CLI and H200 ablations. The
  reset-only, startup-randomization-only, naive explicit-height, and unseeded
  velocity-tweak hypotheses did not hold up.
- 2026-05-15 Added `planner_context_source={live,motion}` to the online SONIC
  provider. On H200 with `seed=123`, 400-step fixed-base A/B improved from
  live-context `root_z_final=0.7049`, `abs_pitch_p95=0.5607` to motion-context
  `root_z_final=0.7598`, `abs_pitch_p95=0.2627`, with no terminations.
- 2026-05-15 Added planner/root velocity instrumentation and planner-only
  command sweep. `target_vel` is not a signed metric velocity for SONIC mode 2:
  direction comes from `movement_direction`, while `target_vel=0.5` gives a
  slower gait bucket than `-1.0/-0.5/1.0`.
- 2026-05-15 On H200 with motion context, `target_vel=0.5` reduced the 400-step
  trace from `abs_pitch_p95=0.2627`, `joint_error_rms_mean=0.2606` to
  `abs_pitch_p95=0.1827`, `joint_error_rms_mean=0.1703`.
- 2026-05-15 Confirmed mjlab G1 hip-pitch actuator profile is 7520_14
  (`kp=40.18`, `effort=88`) while SONIC profile uses 7520_22 (`kp=99.10`,
  `effort=139`). A trace-only hip-pitch profile override improved the 400-step
  `target_vel=0.5` run to `abs_pitch_p95=0.1229`,
  `joint_error_rms_mean=0.1528`.
- 2026-05-15 Added ankle-focused trace instrumentation. In the current best
  H200 400-step baseline, ankle pitch residual is not explained by actuator
  force saturation or actual joint-limit clipping. The actual ankle joints stay
  inside soft limits, while the SONIC motor targets sometimes command outside
  mjlab soft limits. During target violations, left/right ankle-pitch RMS error
  rises to about `1.15` / `0.95` rad versus `0.28` / `0.32` rad when targets
  remain inside limits.
- 2026-05-17 Added a trace-only `--clamp-targets-to-soft-limits` probe that
  clamps sent mjlab targets while preserving raw SONIC target diagnostics.
  Local tests pass. The first remote sync attempt was blocked by the approval
  reviewer until explicit user approval was provided.
- 2026-05-17 After explicit user approval, synced the updated trace tool to
  H200 and ran the clamp probe. The clamped rollout completed 400 steps with no
  done, zero sent-target soft-limit violations, similar root height/pitch, and
  modestly lower ankle-pitch RMS error. This shows the target can be made valid
  without breaking the rollout, but posture is not materially fixed by clamping
  alone.
- 2026-05-17 Ran offline diagnosis on the unclamped/clamped H200 trace pair.
  Remaining issues are upstream raw target range, production action-history
  semantics if clamping becomes real, unvalidated actuator-force utilization,
  and posture/style mismatch that is not solved by target validity alone.
- 2026-05-17 Added raw/effective action and target-range tracing. H200 detailed
  clamp trace shows left ankle pitch raw target range `[-0.8083, 1.5653]`
  against mjlab soft range `[-0.8029, 0.4538]`; target clamp changes effective
  left ankle pitch action by up to `2.5344`, too large to treat as a small
  safety correction.
- 2026-05-17 Audited official `NVlabs/GR00T-WholeBodyControl`
  `gear_sonic_deploy` G1 sources. Official G1 ankle-pitch hard range is
  `[-0.87267, 0.5236]`, still far below the observed left ankle-pitch raw
  target max `1.5653`. The G1 deploy reference constructs targets as
  `action * action_scale + default_angle`, records `last_action` as raw policy
  output, and did not show deploy-side target clipping with effective-history
  writeback.
- 2026-05-17 Added trace-only `--history-action-source raw|effective`. The
  effective mode inverts clamped sent targets back through the same
  `ScalarActionBridge` and feeds that value into the next decoder history while
  leaving the formal mjlab backend unchanged.
- 2026-05-17 Ran H200 raw/effective action-history A/B under the clamped best
  baseline. Effective history reduced max target clipping from `1.1099` to
  `0.6781` and left ankle-pitch raw target max from `1.5636` to `1.1319`, but
  did not materially improve rollout quality: both runs had no done, similar
  pitch, similar height, similar forward velocity, and nearly unchanged joint
  error.
- 2026-05-17 Started a `diagnose` loop for the remaining ankle target contract
  mismatch. Added `015` to build an official sim2sim target/action feedback
  loop and `016` to test whether replacing mjlab ankle-pitch soft limits with
  official G1 hard limits is enough to explain the clipping.
- 2026-05-17 `015` direct official sim2sim feedback loop is blocked because no
  runnable `GR00T-WholeBodyControl` checkout, `run_sim_loop.py`, or
  `deploy.sh` was found in the searched H200 workspaces. Per task rules, no new
  upstream repo was downloaded.
- 2026-05-17 Added and ran `016` trace-only official hard ankle-pitch limit
  overlay. It reduced max target clipping only from `1.1099` to `1.0712` and
  slightly worsened tracking, falsifying the hypothesis that mjlab's narrower
  soft ankle-pitch range is the primary cause.
- 2026-05-17 After explicit install permission, staged official
  `NVlabs/GR00T-WholeBodyControl` source on H200 at commit `0a87181` under
  `/mnt/workspace/users/guoyubo/agent_workspace/official/GR00T-WholeBodyControl`.
  The official deploy layout now symlinks to the existing task025 SONIC ONNX
  artifacts, and a pragmatic `.venv_sim` can import the official MuJoCo sim
  stack and show `gear_sonic/scripts/run_sim_loop.py --help`. The official C++
  deploy build remains blocked by missing TensorRT headers/libraries.
- 2026-05-17 Opened `018` to run the official SONIC sim2sim baseline. Preflight
  reconfirmed that H200 has no existing TensorRT install; official CMake
  requires a `TensorRT_ROOT` containing `NvInfer.h`, `NvInferVersion.h`,
  `libnvinfer`, `libnvinfer_plugin`, and `libnvonnxparser`.
- 2026-05-17 Built official `gear_sonic_deploy` on H200 using user-space
  TensorRT 10.16.1.11 CUDA 12.9 deb extracts plus Jammy C++ dev dependencies.
  The official `g1_deploy_onnx_ref` target now builds and generates TensorRT
  engines for the decoder, encoder, and planner from the task025 ONNX
  artifacts.
- 2026-05-17 Fixed the official sim venv's DDS mismatch by replacing
  `cyclonedds 11.0.1` with official-required `cyclonedds 0.10.2` built against
  the bundled Unitree `libddsc.so`. A Python LowState probe then confirmed the
  official sim loop continuously publishes `rt/lowstate` over `lo`.
- 2026-05-17 Official sim/deploy paired run now reaches `Init Done`, proving initial
  C++/Python DDS contact, but exits before control because C++ deploy reports
  `Lost LowState data connection from robot` and the deploy CSV logs stay
  empty. The remaining blocker is continuous C++ subscriber updates from the
  Python sim publisher, not TensorRT/model loading or adapter code.
- 2026-05-18 Added a minimal C++ `rt/lowstate` subscriber probe built against
  the same bundled Unitree SDK and CycloneDDS libraries as official
  `g1_deploy_onnx_ref`. With the official Python MuJoCo sim publisher live on
  `lo`, the probe received continuous `LowState_` callbacks even with the same
  subscriber queue depth `1` used by official deploy (`count=98` to `988` over
  the 5 second probe window). This falsifies the hypothesis that Python sim to
  C++ Unitree SDK DDS delivery is generally broken.
- 2026-05-18 Instrumented official `g1_deploy_onnx_ref` around
  `LowStateHandler` and `CheckSafety`, then restored the temporary patch. With
  a longer-lived official Python sim publisher, the full deploy process kept
  receiving fresh LowState after `Init Done`; `CheckSafety` saw about
  `0.1-5 ms` LowState age, not the `500 ms` absent threshold. The earlier
  `Lost LowState` was therefore a harness timing artifact from killing the sim
  publisher too early during deploy startup.
- 2026-05-18 Ran restored official deploy with stdin start key `]`. It
  transitioned into `CONTROL`, reported policy loop timing with fresh LowState,
  and wrote non-empty official CSV logs (`1237` rows each in `action.csv`,
  `q.csv`, `dq.csv`, and `token_state.csv`).
- 2026-05-18 Compared restored official deploy CSV logs against mjlab adapter
  raw/effective-history traces. Official deploy's `action.csv` records raw
  policy actions; the motor target is reconstructed as
  `default_angles[i] + action_buffer[isaaclab_to_mujoco[i]] * g1_action_scale[i]`.
  Official SONIC itself commands targets far outside the official G1 XML joint
  ranges: left/right ankle-pitch target violation absmax was `1.5687` /
  `1.4553 rad`, while measured ankle `q` stayed mostly inside limits. The
  mjlab left ankle-pitch raw target violation (`1.1099 rad` against mjlab soft
  limits) is therefore not an adapter-only limit bug. Effective-history feedback
  reduced later target excursions (`1.1099 -> 0.6781`) but did not materially
  improve posture, height, velocity, or mean joint tracking.
- 2026-05-18 Ran matched-input official planner-forward comparisons. Official
  `g1_deploy_onnx_ref` plus official Python MuJoCo sim produced non-empty
  planner-mode logs for both `WALK` and `SLOW_WALK target_vel=0.5` inputs. The
  matched `SLOW_WALK target_vel=0.5` run logged `planner_control_fraction=1.0`,
  `1240` rows, `base_pitch_abs_p95=0.1446`, left/right ankle target violation
  absmax `1.7254` / `1.3656 rad`, and ankle target-minus-q RMS `0.4215` /
  `0.3863`. The mjlab best motion-context `target_vel=0.5` 400-step baseline
  remained stable with `abs_pitch_p95=0.1229`, `root_z_final=0.7443`,
  `root_lin_vel_b_x_mean=0.7051`, and ankle joint-error RMS `0.4369` /
  `0.4004`. The result points away from controller I/O or planner-context
  target range as the remaining primary issue; the next diagnostic should
  compare plant/contact/actuator response under official planner/target motion.
- 2026-05-18 Added and ran official motion replay diagnostics in mjlab. Official
  `target_motion.csv` is a per-control-step current-motion qpos stream, while
  `planner_motion.csv` is segmented replan dumps. Directly commanding
  `target_motion.csv` joint positions in mjlab is a stress test, not the
  official controller contract, and it repeatedly terminated with
  `abs_pitch_p95=0.8887`, `root_z_final=0.2375`, and contact-force max about
  `2380 N`. The more faithful fixed-motion SONIC decoder replay also
  repeatedly terminated: official deploy first 400 rows had
  `abs_pitch_p95=0.1548` and motor-target residual RMS `0.7858`, while mjlab
  under the same motion-conditioned decoder route had `abs_pitch_p95=1.0294`,
  `root_z_final=0.6301`, `root_z_min=0.2480`, residual RMS `0.7160`, and large
  contact impulses. This points to whole-body plant/contact/actuator response,
  not a remaining target-range or action-history bug.

## Review

Status: adapter implementation passed; stable locomotion remains open.

Implemented the modular backend/controller boundary and verified local unit
tests. H200 zero-action and synthetic-sequence mjlab smokes executed and
rendered, but both collapsed because neither uses a stabilized SONIC action
trace. The C++ planner runner dependency is now buildable on H200. Official
SONIC ONNX artifacts were restored and the true online planner/encoder/decoder
path now runs and renders in `unitree_rl_mjlab`.

The remaining risk is policy/context quality, not adapter availability. The
deterministic traces now identify three concrete alignment issues:

- replanning from live qpos history instead of previous planner motion;
- using `target_vel=-1.0` as if it were a signed metric velocity;
- mjlab's hip-pitch actuator profile being weaker than the SONIC profile.

The current best diagnostic baseline is motion context, `target_vel=0.5`, and
the SONIC hip-pitch actuator profile. That still leaves ankle pitch as the
dominant residual tracking error. The first ankle probe points to target/limit
contract mismatch rather than force saturation: SONIC sometimes asks for ankle
targets outside mjlab soft limits, while the actual joints remain inside those
limits. The next route should compare official SONIC ankle limits with mjlab G1
limits, then run a trace-only target clamp probe before filling optional encoder
fields.

The trace-only clamp probe now runs on H200. It confirms target/limit mismatch
is real and correctable at the sent-target layer, but not sufficient to solve
the whole gait alignment issue.

Further diagnosis after clamping found no single larger downstream bug. The
next route should validate the upstream contract: official SONIC joint ranges,
effective action history semantics, and whether mjlab exposes a trustworthy
clipped actuator force signal.

The official upstream contract is now clearer: raw target excursions outside
joint limits are normal in official deploy, including in planner-forward mode.
Matched planner input did not make official actions smaller than mjlab's. Keep
the adapter on raw-history, unclamped fidelity by default. The remaining work
should shift from action clipping to same-trajectory plant response: replay or
sample official `planner_motion.csv` / `target_motion.csv` through mjlab and
compare measured q, base pitch, and contact/actuator response.

The same-trajectory replay now shows that plant response is the main remaining
axis. Under the official fixed motion input, mjlab's aggregate joint residual
is comparable to official, but base pitch, root height, contact impulses, and
termination behavior are much worse. The next route should narrow lower-body
plant/contact causes: hip/knee/ankle actuator constants, foot contact geometry
and friction, and reset/base-state alignment. Treat mjlab `actuator_force`
utilization cautiously until the signal is validated, because some reported
utilization values exceed one by large factors.

The detailed action/range trace narrows the next decision: either mirror an
official SONIC deploy-side clip including effective action history, or treat the
mjlab ankle soft limits as an asset mismatch and test a trace-only limit widening
patch.

Official source audit did not find an explicit SONIC deploy-side clamp or
effective-history writeback. Instead, official deploy appears to use raw policy
action in `last_action`, while the official G1 ankle-pitch hard limit is still
well below the observed raw target peak. The effective-history trace confirms
history semantics matter by reducing later target extremes, but it does not
solve posture or tracking. Production clamping should stay blocked until the
upstream joint-limit/asset contract is resolved.

The official hard-limit overlay further rules out a small mjlab soft-limit
width mismatch. Raw targets remain far outside official hard limits in the
current mjlab closed loop. A true official sim2sim target trace is now the next
missing feedback loop, but it requires an upstream checkout/environment or
explicit permission to fetch one.

The official source checkout/environment is now partially staged after explicit
install permission. Python MuJoCo sim entry is importable; C++ deploy is not
buildable yet because TensorRT is absent on H200.

`018` has moved the official runtime further: C++ deploy now builds and loads
the task025 SONIC artifacts through TensorRT, and the official Python sim loop
publishes continuous LowState under the official `cyclonedds==0.10.2` stack.
The apparent LowState freshness blocker was a test harness lifetime issue:
keeping the sim publisher alive long enough lets the restored official deploy
stay in `WAIT_FOR_CONTROL`, and piping the keyboard start key `]` moves it into
`CONTROL` with non-empty official CSV logs.

The official-vs-mjlab comparison reframes the ankle issue. "Raw target outside
joint limit" is part of the observed official deploy behavior, not a standalone
adapter defect. Official deploy sends out-of-range servo targets and relies on
the plant/simulator constraints for realized joint motion. Therefore production
mjlab should keep raw-action history and should not default to trace-only target
clamping unless we intentionally define a new controller contract. The next
comparison should match motion input, either by running official planner mode
with a forward command or by replaying the same official reference motion in the
mjlab adapter.

Verification:

- `PYTHONPATH=src python -m pytest -p no:cacheprovider`
  passed: `319 passed, 17 skipped`.
- `PYTHONPATH=src python -m pytest tests/test_sonic_controller.py
  tests/test_mjlab_sonic_alignment_trace.py -q`
  passed: `7 passed` with only a local pytest cache permission warning.
- H200 `seed=123`, 400-step fixed-base trace:
  - live context: no done, `root_z_final=0.7049`, `abs_pitch_p95=0.5607`.
  - motion context: no done, `root_z_final=0.7598`, `abs_pitch_p95=0.2627`.
- H200 `seed=123`, 400-step motion-context command/profile traces:
  - `target_vel=0.5`: no done, `abs_pitch_p95=0.1827`,
    `joint_error_rms_mean=0.1703`.
  - `target_vel=0.5` plus SONIC hip-pitch actuator profile: no done,
    `abs_pitch_p95=0.1229`, `joint_error_rms_mean=0.1528`.
- H200 `seed=123`, 400-step ankle-focused trace on the current best baseline:
  no done, `abs_pitch_p95=0.1225`, `joint_error_rms_mean=0.1527`. Actual
  soft-limit violation fraction was `0.0` for all joints; target soft-limit
  violation fraction was `0.0925` for left ankle pitch and `0.0700` for right
  ankle pitch.
- Local target-clamp probe tests:
  `PYTHONPATH=src python -m pytest -p no:cacheprovider` passed:
  `323 passed, 17 skipped`.
- H200 `seed=123`, 400-step target-clamp trace on the current best baseline:
  no done, sent target soft-limit violation fraction `0.0`, raw target
  violation fraction still visible (`0.0925` left ankle pitch, `0.0775` right
  ankle pitch), `abs_pitch_p95=0.1250`, `root_z_final=0.7465`,
  `joint_error_rms_mean=0.1486`.
- H200 `seed=123`, 400-step detailed target-clamp trace:
  no done, `abs_pitch_p95=0.1241`, `root_z_final=0.7441`,
  `joint_error_rms_mean=0.1500`, left ankle pitch raw target exceeds soft high
  by `1.1116 rad`, and effective action delta reaches `2.5344`.
- H200 `seed=123`, 400-step clamped history-source traces:
  - raw history: no done, `abs_pitch_p95=0.1241`,
    `root_z_final=0.7464`, `joint_error_rms_mean=0.1488`,
    `target_clip_absmax_max=1.1099`.
  - effective history: no done, `abs_pitch_p95=0.1216`,
    `root_z_final=0.7435`, `joint_error_rms_mean=0.1492`,
    `target_clip_absmax_max=0.6781`.
- H200 `seed=123`, 400-step raw-history official ankle hard-limit overlay:
  no done, `abs_pitch_p95=0.1247`, `root_z_final=0.7438`,
  `joint_error_rms_mean=0.1517`, `target_clip_absmax_max=1.0712`.
- H200 official install bootstrap:
  `gear_sonic/scripts/run_sim_loop.py --help` passed inside
  `/mnt/workspace/users/guoyubo/agent_workspace/official/
  GR00T-WholeBodyControl/.venv_sim`; C++ deploy `cmake` configure failed at
  `FindTensorRT.cmake` because `NvInferVersion.h` / `libnvinfer.so*` were not
  present.
- H200 official runtime build:
  `cmake --build build --target g1_deploy_onnx_ref --parallel 16` passed in
  `/mnt/workspace/users/guoyubo/agent_workspace/official/
  GR00T-WholeBodyControl/gear_sonic_deploy`.
- H200 official deploy-alone smoke:
  `g1_deploy_onnx_ref lo ... --planner-precision 32 --policy-precision 32`
  generated/loaded TensorRT engines for decoder, encoder, and planner, then
  waited for `LowState` as expected without the sim.
- H200 official Python sim LowState probe:
  after installing `cyclonedds==0.10.2`, a Python subscriber observed continuous
  `rt/lowstate` messages (`count=644` over a 5 second probe window).
- H200 official sim/deploy smoke:
  with the fixed DDS Python stack, deploy reached `Init Done` but exited before
  control with `Lost LowState data connection from robot`; deploy CSV files
  remained empty.
- H200 official C++ LowState subscriber probe:
  `/mnt/workspace/users/guoyubo/agent_workspace/official/lowstate_cpp_probe`
  built against official bundled `unitree_sdk2` and, while the official Python
  sim loop published `rt/lowstate`, received continuous callbacks with
  `InitChannel(..., 1)`: `count=98`, `197`, `296`, `395`, `494`, `593`, `691`,
  `790`, `889`, `988`.
- H200 official deploy LowState instrumentation:
  callbacks continued through `Init Done` and `CheckSafety` saw fresh
  LowState age around `0.1-5 ms`; the temporary debug patch was restored and
  removed from the official source.
- H200 restored official start-control smoke:
  piping `]` into stdin produced
  `[Control] DEBUG: operator_state.start=true, transitioning to CONTROL state`
  and wrote `1237` rows each to official `action.csv`, `q.csv`, `dq.csv`, and
  `token_state.csv` under
  `/mnt/workspace/users/guoyubo/agent_workspace/official/GR00T-WholeBodyControl/gear_sonic_deploy/outputs/task025/official_start_control_smoke/deploy_logs`.
- H200 official-vs-mjlab comparison:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/outputs/task025/official_vs_mjlab_compare/key_stats.json`
  generated from official deploy CSV and mjlab trace JSON. Official
  left/right ankle-pitch targets exceeded official hard limits by `1.5687` /
  `1.4553 rad`; mjlab raw-history left/right ankle-pitch targets exceeded
  mjlab soft limits by `1.1099` / `0.3611 rad`.
- `PYTHONPATH=src python -m pytest tests/test_mjlab_sonic_alignment_trace.py
  tests/test_scalar_action_bridge.py tests/test_sonic_controller.py -q`
  passed: `26 passed` with only the existing local pytest cache permission
  warning.
- `python -m ruff check ...` was not run because `ruff` is not installed in the
  local Python environment.
