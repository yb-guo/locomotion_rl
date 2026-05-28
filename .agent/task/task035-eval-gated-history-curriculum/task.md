# Task 035: Eval-Gated History Curriculum

## Route

Task033 and Task034 produced one useful but dangerous lesson:

- frozen-base StackMLP K4 can produce a robust checkpoint;
- later PPO continuation can regress the same target;
- the best checkpoint was found by checkpoint sweep (`model_5350.pt`), not by
  taking the final checkpoint (`model_5378.pt`).

This task combines two things that must stay in the same closed loop:

1. validate `model_5350.pt` as the current baseline checkpoint;
2. build an eval-gated curriculum protocol that can reproduce or beat that
   baseline without accepting the final checkpoint blindly.

Candidate baseline checkpoint:

`/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_stackmlp_k4_frozenbase_focused/2026-05-28_12-40-56_033_frozenbase_focused_from5349_env8192_iter30_gpu1_seed3303362_lr1e5/model_5350.pt`

Fixed boundaries:

- Start with frozen-base StackMLP K4 and Task033 shared history input.
- Keep actor input K4 history, `540D`.
- Keep action `31D`, reward, observation contract, and failure scheduler
  semantics unchanged unless a subtask records a local stage registration.
- Do not expose explicit fault labels, motor scales, active joint ids, or
  failure masks to the actor.
- Do not switch to GRU/token policy in this task.
- Do not change link geometry, mass, COM, inertia, or motor topology.

Core curriculum:

```text
model_5350 baseline gate
        |
        v
clean + unified-speed rehearsal
        |
        v
weak motor persistent failures
        |
        v
mixed weak/dead single-joint failures
        |
        v
forced dead-grid rehearsal
        |
        v
dynamic switch rehearsal
        |
        v
checkpoint sweep + eval-gated selection
```

Planned slices:

1. `001-contract-and-baseline-gate.md`
   - Define checkpoint, thresholds, seeds, speed bins, and baseline validation.

2. `002-stage-registration.md`
   - Register H200 MJLab stages for eval-gated frozen-base StackMLP K4
     curriculum.

3. `003-training-run.md`
   - Run one bounded env8192 curriculum pass with frequent checkpoints.

4. `004-checkpoint-sweep.md`
   - Evaluate non-final checkpoints and select by eval score, not iteration.

5. `005-full-validation-and-video.md`
   - Validate best checkpoint against `model_5350.pt` and render videos.

6. `006-decision.md`
   - Promote curriculum checkpoint, keep `model_5350`, or reject the route.

## Minimal Closed Loop

1. Reproduce `model_5350.pt` on fresh seeds:
   - `2.0 m/s` dynamic switch;
   - `2.0 m/s` full 12-joint forced dead-grid.
2. Register and smoke a curriculum stage on H200.
3. Run one bounded curriculum continuation.
4. Sweep at least three checkpoints from the run, including non-final
   checkpoints.
5. Compare the selected checkpoint against:
   - Task034 accepted `model_5350.pt`;
   - Task033 final `model_5378.pt`.
6. Render representative videos before promotion.

Acceptance:

- All eval JSONs are written under:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task035/`.
- The checkpoint sweep includes non-final checkpoints.
- Eval JSON covers:
  - clean or no-failure sanity if available;
  - dynamic switch;
  - forced 12-joint dead-grid;
  - speeds `0.4`, `1.2`, `2.0 m/s` for the final candidate.
- Videos exist for at least low/mid/high representative cases.
- The final decision states whether curriculum training actually helped.

Pass:

- The eval-gated curriculum produces a checkpoint that matches or beats
  Task034 `model_5350.pt` on the validation gates.

Partial pass:

- `model_5350.pt` is validated as the current main checkpoint, but the new
  curriculum does not beat it.

Fail:

- The Task034 `model_5350.pt` pass does not reproduce across seeds.
- Curriculum checkpoints regress below `model_5350.pt`.
- Selection relies on training reward instead of eval JSON.
- The actor receives explicit fault state.

Evidence root:

`/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task035/`

## Log

- 2026-05-28 Opened as a merged task after user clarified that checkpoint
  validation and curriculum learning should be one closed loop.

## Review

Status: active planning. No curriculum pass claim yet.
