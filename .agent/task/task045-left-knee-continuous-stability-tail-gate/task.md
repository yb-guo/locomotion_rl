# Task 045: Left-Knee Continuous Stability Tail Gate

## Route

Task041 already exists as `task041-sequence-aware-txl-clean-gait`; this task is
opened as the next numbered continuation toward the user's current objective:
run the current memory-dependent locomotion line until the left-knee hidden
fault eval passes, then use that verified base to continue toward LocoFormer
reproduction.

Task044 produced a useful but incomplete state:

- scale-0.5 staged bypass annealing can make clean gait memory-dependent;
- `zero_memory_latent` breaks the clean gait checkpoint;
- hidden left-knee dead continuous eval no longer fails because of speed,
  posture, root height, or missing H200 execution;
- the current best checkpoint repeatedly fails because post-fault
  `fall_ratio=0.105-0.125`, above the `<=0.05` gate.

Task045 is therefore a narrow convergence task:

- keep the G1-like MJLab topology and action/observation contract unchanged;
- keep the Task044 continuous post-fault eval gate unchanged;
- reduce the residual left-knee dead-motor fall/reset tail until normal
  continuous eval passes;
- only after normal quality passes, run memory ablations/triplet evidence;
- do not claim LocoFormer reproduction until the verified base policy is ready
  for the next LocoFormer-specific task.

## Planned Slices

1. `001-current-best-and-gate-audit.md`
   - Freeze the current best checkpoint and repeated-seed baseline.
   - Record the exact pass gate and why the current failure is the residual
     stability tail.

2. `002-stability-tail-objective-stage.md`
   - Add or select the smallest stability-tail training objective that targets
     post-fault falls without relaxing eval.
   - Keep actor-visible observations and action shape unchanged.

3. `003-h200-convergence-train-eval-loop.md`
   - Run H200 training/eval loops from the current best checkpoint.
   - Track normal continuous eval first; do not spend triplet runs before normal
     quality passes.

4. `004-memory-ablation-after-normal-pass.md`
   - Once normal continuous eval passes, run zero-memory/stateless ablations on
     the same checkpoint, seed, speed, and fault schedule.
   - Record whether the result is memory-required or only a robust MLP/bridge
     policy.

5. `005-locoformer-reproduction-handoff.md`
   - Summarize what is reusable for the next LocoFormer reproduction task:
     task contract, eval gate, checkpoint, runner limitations, and remaining
     architecture gaps.

6. `006-reset-time-diagnostic-and-targeted-stage-budget.md`
   - Add reset-time diagnostic fields to continuous eval JSON without changing
     the pass/fail gate.
   - Use the diagnostic to choose at most one or two targeted continuation
     stages; if those do not approach the gate, stop this local repair route
     and hand off to the next LocoFormer policy task.

## Acceptance Criteria

Task045 is accepted only when current evidence proves:

- local docs and inventory include the task and every subtask has
  Route / Log / Review;
- H200 normal continuous left-knee dead eval records a passing JSON with:
  - `quality_gate_pass=true`;
  - `pipeline_pass=true`;
  - post-fault `fall_ratio <= 0.05`;
  - post-fault `lin_vel_error.mean <= 0.45`;
  - post-fault `gravity_xy.max <= 0.90`;
  - post-fault `root_z.min >= 0.35`;
- repeated seed eval does not show the pass is a one-seed accident;
- after normal pass, ablation/triplet JSONs are recorded before any
  memory-required claim;
- no claim is made that this reproduces LocoFormer; it is only the convergence
  gate needed before the LocoFormer-specific reproduction route.

## Evidence Gate

Local:

```powershell
$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.inspect_agent
```

H200 normal eval shape:

```bash
cd /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter
env PYTHONPATH=src:/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_DISABLED=true \
  /mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python \
  -m h200_locomotion_lab.tools.task044_continuous_fault_eval \
  --checkpoint /path/to/model.pt \
  --output-json /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/continuous_fault_eval/name.json \
  --num-envs 256 \
  --steps 360 \
  --seed 4520001 \
  --device cuda:0 \
  --lin-vel-x 1.6 \
  --dynamic-dead-joint left_knee_joint \
  --dynamic-onset-s 2.0 \
  --dynamic-recovery-s 999.0 \
  --startup-excluded-s 0.5 \
  --post-fault-window-s 2.0 \
  --memory-latent-dim 32 \
  --base-obs-passthrough \
  --adaptation-warmstart \
  --base-obs-passthrough-scale 0.5 \
  --adaptation-warmstart-scale 0.5 \
  --memory-ablation-mode none
```

## Log

- 2026-06-02 Opened because Task041 already exists and the live convergence
  blocker is downstream of Task044: scale-0.5 memory-dependent clean gait is
  available, but hidden left-knee continuous normal eval still fails on the
  residual post-fault fall/reset tail.
- 2026-06-02 Started subtask 002 with a narrow survival stage:
  `Unitree-G1-Gripper-Flat-Task045-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneePoseForwardSurvival1p6`.
  It preserves the existing checkpoint shape, visible observation contract, and
  eval gate; it only strengthens posture/survival pressure for the fixed
  left-knee dead curriculum.
- 2026-06-02 Added
  `Unitree-G1-Gripper-Flat-Task045-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneeLongSurvival1p6`
  because continuous eval also requires zero physical resets over the full
  rollout, not just low fall ratio in the first 2.0 s post-fault window.
- 2026-06-02 Opened reset-time diagnostic subtask after long-survival improved
  the best post-fault fall ratio to `0.078125` but still left physical reset
  events in continuous eval. The next decision must be based on where resets
  occur in the rollout, not another blind training extension.
- 2026-06-02 Clarified gate alignment: the current best checkpoint can pass when
  evaluated with its `Task045 LongSurvival` training task config, but the
  unchanged old `Task044 PersistentHiddenPoseTight1p6` gate still fails. Task045
  acceptance remains tied to the unchanged old continuous gate, not the
  training-task config.
- 2026-06-02 Selected targeted stage 1 from reset-time evidence:
  `Task045 PoseTightGateLeftKneeLongTail1p6`, a 2.0 s onset / 8.0 s long-tail
  left-knee-dead curriculum aligned to the old gate.
- 2026-06-02 Ran both allowed targeted stages. Stage 1 used full-actor PPO on
  the old-gate-aligned long-tail task; stage 2 limited training to
  `txl_residual_and_mlp_memory_input`. Both failed to improve the unchanged old
  continuous gate. The route should now stop local repair and return to the
  LocoFormer policy line.

## Review

Status: open, not passed.

Task045 deliberately keeps the scope smaller than "reproduce LocoFormer". The
current policy line must first produce a verified passing continuous left-knee
fault eval; otherwise downstream LocoFormer comparisons will be measuring an
unstable base task rather than architecture differences.

After reset-time diagnosis and two bounded targeted repair stages, the current
evidence says this local reward/curriculum route is not enough. The task should
not spend more runs on left-knee-specific tuning unless a new policy mechanism
or eval-contract change is explicitly opened in a later task.
