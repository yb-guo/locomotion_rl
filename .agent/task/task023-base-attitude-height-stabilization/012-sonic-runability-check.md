# Subtask 012: SONIC Runability Check

## Route

- Continue inside task023; no new top-level task.
- User explicitly requested a SONIC runability check.
- No downloads, no PPO, no training, no upstream edits, no
  `GenesisG1SceneBackend` changes, and no `/mnt/workspace*` writes/deletes.
- Use only already-present H200 SONIC artifacts from task002.
- Treat official SONIC as a reference stack smoke, not as evidence that the
  task023 27DoF Genesis hybrid asset is PPO-ready.

## Feedback Loop

```text
preflight existing SONIC artifacts -> headless MuJoCo sim loop startup ->
optional deploy sim connection if sim loop starts
```

## Hypotheses

1. **Official SONIC stack is runnable on H200 with existing artifacts.**
   - Prediction: MuJoCo sim loop starts headless, deploy process starts, and no
     missing model/environment error appears.
2. **Official SONIC can be healthy while the task023 Genesis asset remains
   not PPO-ready.**
   - Prediction: SONIC runability does not contradict task023 because SONIC
     uses its own 29DoF asset, controller, model stack, and MuJoCo semantics.
3. **If official SONIC fails before control, the blocker is environment/model
   setup, not the task023 Genesis asset.**
   - Prediction: failure is missing runtime, display/headless issue, TensorRT,
     or process connection.

## Stop Rules

- Stop if a command would download or build new artifacts.
- Use timeouts so no long-running sim/deploy process is left alive.
- Do not run training or PPO.

## Log

- 2026-05-13 Created after user asked whether the asset is healthy and whether
  SONIC can run normally.
- 2026-05-13 Preflight on H200 found the existing task002 SONIC upstream tree,
  `run_sim_loop.py`, `deploy.sh`, `g1_29dof.xml`, `.venv_sim`, and the expected
  ONNX/config/reference artifacts. `.venv_sim` has MuJoCo `3.8.0`; `onnx` and
  `onnxruntime` are not installed there, but deploy uses the native ONNX/TensorRT
  path.
- 2026-05-13 Headless sim-only smoke used `--no-enable-onscreen
  --no-enable-offscreen --interface sim` and ran until the 30s timeout
  (`SIM_EXIT_CODE=124`) without a MuJoCo/XML crash. The only observed messages
  were non-fatal git metadata and loopback multicast warnings.
- 2026-05-13 Dual `sim_loop + deploy.sh sim` first stopped at the deploy
  confirmation prompt. A non-interactive confirmation then reached the native
  deploy program. It found model files, loaded 13 reference motion folders, and
  passed the ONNX hash check before failing while loading the policy model:
  `createInferRuntime: ... CUDA initialization failure with error: 35`, followed
  by segfault and `DEPLOY_EXIT_CODE=139`.
- 2026-05-13 Repeated the dual smoke with `CUDA_VISIBLE_DEVICES=1`; result was
  unchanged (`DEPLOY_EXIT_CODE=139`, same TensorRT/CUDA error 35). A read-only
  runtime diagnostic showed driver `570.195.03`, reported CUDA `12.8`, CUDA
  toolkit `12.8.61`, H200 GPU 1 visible, and system TensorRT candidates plus
  both `libcudart.so.12` and `libcudart.so.13` in the loader cache.
- 2026-05-13 Cleanup check found no residual `run_sim_loop.py`,
  `g1_deploy_onnx_ref`, or `deploy.sh` processes. `deploy.sh` did run its normal
  CMake/build step in the existing upstream run directory; no source files,
  checkpoints, datasets, upstream repos, or `/mnt/workspace*` paths were
  downloaded or modified by this task.
- 2026-05-13 Follow-up after user clarified that the goal was to check the
  SONIC asset and reproduce the old successful route. Task002
  `run_h200_notes.md` shows the same `CUDA initialization failure with error:
  35` was previously fixed by prepending extracted TensorRT
  `10.13.3.9-1+cuda12.9` runtime libraries from:

```text
/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/trt-10.13.3-cuda12.9-root/usr/lib/x86_64-linux-gnu
```

  That extracted runtime directory is no longer present on the H200 host, and
  the CUDA12.9 TensorRT `.deb` packages were not found in task002, task006,
  `/root/agent_workspace`, `/var/cache/apt/archives`, or local workspace
  caches. Current system packages are `libnvinfer10`,
  `libnvonnxparsers10`, and `tensorrt` `10.13.3.9-1+cuda13.0`, which is the
  known-bad path for this driver.
- 2026-05-13 Ran the asset-side portion of the old task002 route without
  TensorRT: default SONIC MuJoCo sim loop under `xvfb-run` against the official
  upstream tree. It ran until the 60s timeout (`SIM_XVFB_EXIT_CODE=124`) with
  no XML/mesh/MuJoCo crash. Log:

```text
/root/agent_workspace/project/h200-locomotion-lab-task023-base-attitude-height-stabilization/outputs/task023/sonic_asset_smoke/run_sim_loop_xvfb_60s.log
```

- 2026-05-13 Direct MuJoCo asset load/step check on
  `gear_sonic/data/robots/g1/g1_29dof.xml` passed:

```text
MUJOCO_VERSION 3.8.0
NQ 36
NV 35
NU 29
NBODY 31
NGEOM 76
FINITE True
QPOS_Z 0.7804043605468406
```

  No `run_sim_loop.py` or `g1_deploy_onnx_ref` process remained. Existing
  orphaned `Xvfb` processes are from May 5-7 and predate this smoke.

## Evidence

- Official SONIC asset/model presence: healthy enough for startup.
- Official SONIC MuJoCo XML/sim startup: healthy. Both direct MuJoCo
  load/step and default `xvfb-run` sim-loop startup passed at asset level.
- Official SONIC deploy runtime: not healthy on this H200 environment because
  TensorRT fails before policy inference/control.
- Task023 Genesis training asset: not certified by this check. SONIC uses its
  own 29DoF MuJoCo asset, controller, ONNX/TensorRT deploy path, and reference
  motion stack.

## Interpretation

- The SONIC failure is an environment/runtime failure, not evidence that the
  robot asset or reference motions are corrupt. The asset-side smoke passed.
- The immediate blocker for a full old-style policy/control rerun is missing
  TensorRT `10.13.3.9-1+cuda12.9` runtime extraction. GPU 1 is visible, but the
  current system TensorRT package is `+cuda13.0`, matching the task002
  documented failure mode.
- This does not overturn task023's Genesis conclusion: the best current Genesis
  asset/contact setup is improved but still not PPO-ready because active
  standing remains controller/contact-semantics dominated.

## Review

Status: diagnostic/not passed.

SONIC asset-side health: passed for the official MuJoCo 29DoF asset.

Full SONIC policy/control rerun: blocked by missing CUDA12.9 TensorRT runtime
that task002 used to avoid the known CUDA13 error 35 path. The blocker is not a
demonstrated bad asset.
