# 020: Official Deploy LowState Instrumentation

## Route

Instrument the official `g1_deploy_onnx_ref` process itself after `019`
proved that standalone C++ DDS subscription works:

1. Add temporary tagged logs around `LowStateHandler` and `CheckSafety`.
2. Run the official Python MuJoCo sim publisher plus the instrumented official
   deploy binary.
3. Decide whether `LowState` callbacks stop, timestamps stop refreshing, or
   safety reads stale data despite callbacks.
4. Remove the temporary remote patch after collecting evidence.

The remote debug prefix is `[DEBUG-task025-lowstate]`.

## Log

- 2026-05-18 Opened after `019` showed that the DDS boundary is healthy in a
  minimal C++ subscriber. Ranked hypotheses:
  1. callbacks stop only in the full deploy process;
  2. callbacks continue but `low_state_buffer_` timestamps are stale when read
     by `CheckSafety`;
  3. callback/timestamp are healthy and the state machine/freshness logic
     misclassifies them;
  4. command writer/control/planner thread interaction breaks subscription
     after the `INIT` phase.
- 2026-05-18 Temporarily patched the official H200 checkout with tagged
  `[DEBUG-task025-lowstate]` logs around `LowStateHandler`, the `Init Done`
  transition, and `CheckSafety`. Rebuilt `g1_deploy_onnx_ref`.
- 2026-05-18 Instrumented paired run with the official Python sim alive for a
  longer window did not reproduce `Lost LowState`. `LowStateHandler` callbacks
  continued through and after `Init Done`: callback counts advanced from `1`
  to over `5216`, `tick` advanced from `19100` to `45175`, and
  `CheckSafety` saw low-state ages around `0.1` to `5 ms`, far below the
  official `500 ms` absent threshold.
- 2026-05-18 Restored the official source from backup, verified no
  `[DEBUG-task025-lowstate]` strings remained, touched the restored source, and
  rebuilt so the binary no longer contained the debug patch.
- 2026-05-18 Re-ran the restored official deploy with a longer-lived sim
  publisher. It reached `Init Done` and did not print `Lost LowState`; the
  process was stopped only by the outer `timeout` (`deploy_status=124`). This
  shows the earlier `018` failure was not a persistent deploy-side DDS bug.
- 2026-05-18 Ran a start-control smoke by piping the official keyboard start
  key `]` into deploy stdin. The restored official binary reached:

  ```text
  Init Done
  [Control] DEBUG: operator_state.start=true, transitioning to CONTROL state
  Reference motion name: dance_in_da_party_001__A464
  Loop timing - LowState age: 5.222ms ... Policy: 57us ...
  ```

  The run wrote 1237 rows each to `action.csv`, `q.csv`, `dq.csv`, and
  `token_state.csv` under
  `/mnt/workspace/users/guoyubo/agent_workspace/official/GR00T-WholeBodyControl/gear_sonic_deploy/outputs/task025/official_start_control_smoke/deploy_logs`.
- 2026-05-18 Local regression group passed:
  `PYTHONPATH=src python -m pytest tests/test_mjlab_sonic_alignment_trace.py tests/test_scalar_action_bridge.py tests/test_sonic_controller.py -q`
  reported `26 passed` with the existing local pytest cache permission warning.

## Review

Status: passed.

The stale-LowState failure from `018` was a harness artifact, not a stable
official deploy defect. The earlier paired run kept the Python sim publisher
alive for too short a window while deploy still had TensorRT/model startup
work to finish. With the sim publisher kept alive long enough, official deploy
maintains fresh LowState across `INIT -> WAIT_FOR_CONTROL`.

Official SONIC now reaches the real `CONTROL` loop on H200 when stdin provides
the keyboard start key `]`. That produces non-empty official CSV logs, so the
next diagnostic can finally compare official deploy `action/q/dq` traces
against the mjlab adapter's raw/effective action and ankle range traces.

Temporary instrumentation was removed from the official checkout after the
probe; no `[DEBUG-task025-lowstate]` strings remain in the restored source.
