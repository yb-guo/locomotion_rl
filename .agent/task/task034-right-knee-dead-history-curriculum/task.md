# Task 034: Right-Knee Dead History Curriculum

## Route

Task033 found that shared history is useful but not yet complete:

- frozen-base StackMLP K4 preserves `2.0 m/s` dynamic-switch behavior;
- forced persistent dead-grid at `2.0 m/s` improves to `11/12`;
- the only remaining failed case is `right_knee_joint` forced dead.

This task is a focused curriculum and regression task, not a broad policy
bakeoff.

Starting checkpoint:

`/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_stackmlp_k4_frozenbase_focused/2026-05-28_12-40-56_033_frozenbase_focused_from5349_env8192_iter30_gpu1_seed3303362_lr1e5/model_5378.pt`

Fixed boundaries:

- Keep G1-like topology, rewards, action contract, and observation contract.
- Keep Task033 shared history input: K4 frames, actor input `540D`.
- Keep frozen-base StackMLP K4 as the only policy architecture in this task.
- Do not expose explicit fault labels, motor scales, active joint ids, or
  failure masks to the actor.
- Do not change link geometry, mass, COM, inertia, or action dimensions.
- Do not start GRU/token long training in this task.

Hypotheses:

1. Right-knee forced-dead is underrepresented or too hard too early in the
   focused curriculum; right-knee oversampling with staged severity should
   improve `zero_fall_ratio`.
2. Frozen-base history columns have enough capacity, but need more focused
   steps and lower-risk rehearsal to avoid regressing dynamic switch.
3. If right-knee still fails after focused frozen-base training, the remaining
   blocker likely needs a recurrent/token policy or explicit adaptation state.

Planned slices:

1. `001-repro-and-contract.md`
   - Record the exact Task033 failure JSON and acceptance thresholds.
   - Define the fast feedback loop for `right_knee_joint` forced dead.

2. `002-right-knee-focused-stage.md`
   - Register Task034 H200 env ids for right-knee-focused frozen-base training.
   - Keep actor no-leak and same Task033 runner path.

3. `003-frozenbase-train.md`
   - Continue from Task033 `model_5378.pt`.
   - Train staged right-knee-focused frozen-base StackMLP.

4. `004-right-knee-eval.md`
   - Evaluate `right_knee_joint` forced dead first.
   - If it improves, run the full 12-joint forced dead-grid at `2.0 m/s`.

5. `005-dynamic-switch-regression.md`
   - Re-run `2.0 m/s` canonical dynamic-switch regression.
   - Do not accept a dead-grid improvement that breaks dynamic switch.

6. `006-decision.md`
   - Decide whether frozen-base StackMLP is enough or Task035 should move to
     GRU/token long training.

## Minimal Closed Loop

1. Reproduce baseline failure:
   `right_knee_joint` forced-dead `zero_fall_ratio=0.2109375`.
2. Patch and smoke the Task034 right-knee-focused H200 stage.
3. Train one focused frozen-base continuation from Task033 `model_5378.pt`.
4. Evaluate:
   - right-knee forced dead;
   - full `2.0 m/s` 12-joint forced dead-grid if right-knee improves;
   - `2.0 m/s` dynamic switch regression.
5. Stop after one focused pass unless metrics clearly improve.

Acceptance:

- H200 stage smoke completes.
- Right-knee eval JSON exists with the same thresholds as Task033:
  `zero_fall_ratio >= 0.50`, `lin_vel_error_mean <= 1.0`,
  `yaw_vel_error_mean <= 1.0`, `gravity_xy_mean <= 0.75`.
- Full `2.0 m/s` forced dead-grid is `12/12` only if right-knee passes.
- Dynamic switch remains `pass=true`.
- Decision is explicit and evidence-backed.

Pass:

- `right_knee_joint` forced-dead passes at `2.0 m/s`.
- Full `2.0 m/s` forced dead-grid passes `12/12`.
- Dynamic switch remains passed.

Partial pass:

- Right-knee materially improves but remains below threshold, and dynamic
  switch remains passed.

Fail:

- Right-knee does not materially improve.
- Dynamic switch regresses.
- The actor receives explicit fault state.
- The task expands into GRU/token or from-scratch curriculum without a new
  task.

Evidence root:

`/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task034/`

## Log

- 2026-05-28 Opened from Task033 after frozen-base StackMLP K4 improved
  `2.0 m/s` forced dead-grid to `11/12` but failed `right_knee_joint`.
- 2026-05-28 Registered Task034 right-knee weak/mixed/hard frozen-base stages
  on H200 and ran env64 smoke:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task034_rightknee_frozenbase_smoke/2026-05-28_14-12-22_034_mixed_rightknee_env64_iter1_gpu1_seed3403400`.
- 2026-05-28 Ran focused mixed continuation from Task033 `model_5378.pt`:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task034_rightknee_frozenbase_train/2026-05-28_14-12-50_034_mixed_rightknee_from_task033_model5378_env8192_iter30_gpu1_seed3403401_lr5e6`.
  It regressed the target case; final `model_5407.pt` had
  `right_knee_joint` `zero_fall_ratio=0.015625`.
- 2026-05-28 Ran weaker 10-iteration continuation:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task034_rightknee_frozenbase_train/2026-05-28_14-21-49_034_weak_rightknee_from_task033_model5378_env8192_iter10_gpu1_seed3403411_lr1e6`.
  It also failed to improve the target case; best observed right-knee
  `zero_fall_ratio=0.1171875`.
- 2026-05-28 Diagnosed the issue by evaluating Task033 frozen-base checkpoints.
  Earlier checkpoint `model_5350.pt` already passes the right-knee target and
  the full `2.0 m/s` 12-joint forced dead-grid. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task034/frozenbase_model5350_deadgrid_vx2p0_seed3403500/task033_failure_grid_eval_aggregate.json`
  (`pass=true`, `pass_count=12`). Dynamic-switch regression also passes:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task034/frozenbase_model5350_dynamicmotorfailure_vx2p0_seed3105349/task033_dynamic_eval_switch_vx2p0.json`
  (`pass=true`, `zero_fall_ratio=1.0`, `recovery_success_ratio=1.0`).

## Review

Status: pass for the narrow Task034 target. The fix is checkpoint selection:
Task033 frozen-base `model_5350.pt`, not additional right-knee-focused
continuation. Continuing the focused stage from `model_5378.pt` regresses
right-knee robustness and should not be used.
