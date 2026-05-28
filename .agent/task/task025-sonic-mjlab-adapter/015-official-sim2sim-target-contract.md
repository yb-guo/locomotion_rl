# 015: Official Sim2Sim Target Contract

## Route

Use the diagnose loop to build a feedback signal that distinguishes:

- official SONIC itself can emit ankle targets above official G1 limits; or
- the current mjlab adapter closed loop is driving SONIC out of distribution.

Feedback loop target:

1. Use an existing H200 official `GR00T-WholeBodyControl` checkout if present.
2. Do not download a new upstream repo or new model/assets.
3. Inspect or instrument the smallest official sim2sim/deploy seam that can
   record:
   - policy raw action;
   - target computed as `action * action_scale + default_angle`;
   - official G1 ankle-pitch ranges;
   - `last_action` history source.
4. Compare official target ranges against current mjlab traces.

Ranked hypotheses before probing:

1. If official SONIC also emits ankle-pitch targets above `0.5236 rad`, then
   the policy/deploy contract relies on downstream saturation or unchecked
   target commands, and the adapter must make that behavior explicit.
2. If official SONIC target ranges stay near official limits while mjlab target
   ranges exceed them, then mjlab feedback, reset, actuator dynamics, or planner
   context mismatch is driving decoder history out of distribution.
3. If only the mjlab soft limit differs from official hard limit, then an
   official-limit overlay should nearly eliminate ankle clipping. Prior evidence
   predicts this will be false because raw target max is about `1.5653`.

## Log

- 2026-05-17 Opened after `014` showed effective action history reduces
  target extremes but does not fix posture or tracking.
- 2026-05-17 H200 inventory for an existing official sim2sim checkout did not
  find a runnable `GR00T-WholeBodyControl` tree in the searched user/workspace
  locations.

  Searches:

  ```text
  find ... -type d -name GR00T-WholeBodyControl
  find ... -name run_sim_loop.py
  find ... -name deploy.sh
  ```

  Results:

  ```text
  no GR00T-WholeBodyControl checkout found
  no gear_sonic/scripts/run_sim_loop.py found
  no gear_sonic_deploy/deploy.sh found
  current adapter ONNX artifacts are present
  old source snapshot found:
    /mnt/workspace/users/guoyubo/agent_workspace/
    h200-locomotion-lab-task023-franka-payload/
    .agent/task/task006-sonic-genesis-action-policy/artifacts/
    official_sources/g1_deploy_onnx_ref.cpp
  ```

  Per task rule, I did not download a new upstream repo.
- 2026-05-17 User gave explicit install permission. See `017` for the official
  source/environment bootstrap. The official Python MuJoCo sim entry point is
  now importable on H200, but the official C++ deploy loop is still blocked by
  missing TensorRT.

## Review

Result: unblocked for Python-side official sim-loop probing, still blocked for
the official C++ deploy loop.

The static source audit from `013` still stands, but it is not a runtime
feedback loop. Continue with a minimal official sim-loop probe that records the
same target/action contract fields. If the exact C++ deploy loop is required,
installing TensorRT must happen first.
