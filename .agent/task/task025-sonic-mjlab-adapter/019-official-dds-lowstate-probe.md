# 019: Official DDS LowState Probe

## Route

Build the smallest feedback loop for the official sim2sim blocker found in
`018`:

1. Compile a throwaway C++ subscriber against the same bundled Unitree SDK used
   by `g1_deploy_onnx_ref`.
2. Run the official Python MuJoCo sim loop as the `rt/lowstate` publisher.
3. Count C++ `LowState_` callback arrivals over a short fixed window.
4. Compare with the Python subscriber result from `018`.

This probe does not run SONIC policy, write repo source code, or modify the
official checkout. It only tests the DDS boundary.

## Log

- 2026-05-18 Opened after the user asked to test the next step. The target
  symptom is official deploy reaching `Init Done`, then failing
  `Lost LowState data connection from robot` despite the Python sim publisher
  being live.
- 2026-05-18 Built a throwaway H200 C++ subscriber at
  `/mnt/workspace/users/guoyubo/agent_workspace/official/lowstate_cpp_probe`
  against the official checkout's bundled `unitree_sdk2` and CycloneDDS
  libraries:
  `/mnt/workspace/users/guoyubo/agent_workspace/official/GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty/unitree_sdk2`.
- 2026-05-18 First probe with subscriber depth `10` received continuous
  callbacks while the official Python sim loop published `rt/lowstate` on
  `lo`: `count=0`, `0`, `69`, `168`, `266`, `365`, `464`, `563`, `662`,
  `761`.
- 2026-05-18 Re-ran with subscriber depth `1`, matching official
  `g1_deploy_onnx_ref`:
  `lowstate_subscriber_->InitChannel(..., 1)`. The probe still received
  continuous callbacks: `count=98`, `197`, `296`, `395`, `494`, `593`, `691`,
  `790`, `889`, `988`; `tick` advanced from `1394` to `5844`. Output was saved
  under
  `/mnt/workspace/users/guoyubo/agent_workspace/official/GR00T-WholeBodyControl/gear_sonic_deploy/outputs/task025/official_lowstate_cpp_probe`.
- 2026-05-18 Local regression group passed:
  `PYTHONPATH=src python -m pytest tests/test_mjlab_sonic_alignment_trace.py tests/test_scalar_action_bridge.py tests/test_sonic_controller.py -q`
  reported `26 passed` with the existing local pytest cache permission warning.

## Review

Status: passed.

The minimal C++ DDS boundary is not the blocker. A standalone C++ subscriber
using the same Unitree SDK, CycloneDDS libraries, topic, network interface, and
queue depth as official deploy can receive the official Python sim publisher's
`rt/lowstate` stream continuously.

This narrows the official sim2sim failure to the full deploy path after or
around `INIT`: the callback/timestamp buffer, thread scheduling, command writer
interaction, or state-machine freshness check in `g1_deploy_onnx_ref`.

Next route: instrument official deploy, preferably as a temporary remote patch,
to print `LowStateHandler` callback count and `CheckSafety` low-state age across
the `INIT -> WAIT_FOR_CONTROL` transition. If callbacks continue but age is
stale, focus on `DataBuffer` ownership/timestamp access; if callbacks stop only
inside full deploy, add a minimal command-writer/threaded reproduction.
