# Task 047: Training and Evaluation Correctness Repair

## Route

A 2026-08-07 read-only repository audit found five correctness defects at the
training/evaluation boundary. They affect how much confidence can be placed in
some Task040-Task046 evidence even though the existing smoke commands and
documented earlier jobs completed.

1. Sequence-aware True-TXL PPO samples actions with a persistent inference
   cache, but update-time sequence replay starts every rollout from zero
   memory. This can make the recomputed log probability differ from the stored
   sampling log probability before any optimizer step.
2. Task037 multi-trial eval reads robot state after `env.step()`, after the
   multi-trial/MJLab auto-reset path may already have replaced a failed state
   with a reset pose. Fall/reset counts remain meaningful, but terminal-frame
   velocity, gravity, and root-height metrics can be optimistic.
3. The legacy Genesis PPO rollout records `terminated` and `truncated`, but
   `compute_gae()` treats both as value-terminal. Correct timeout bootstrapping
   also requires the pre-reset terminal observation, not the next episode's
   reset observation.
4. Task046's post-reset recovery reward is selected by final-trial index and
   trial step only. It does not require the preceding inner reset to be a fall,
   so ordinary timeout retries are shaped under a retry-after-fall claim.
5. Task044 continuous eval writes a hard-coded runner class and does not include
   actual runner identity in `pipeline_pass`. A wrong runner can therefore be
   reported as the expected runner.

The task repairs these defects without changing quality thresholds or relaxing
the old continuous no-reset gate. Existing scientific conclusions must be
treated according to the impact boundary below until the affected evidence is
rerun.

## Impact Boundary

- Task040's sequence-update smoke proves that the branch executes, but not yet
  that PPO replay uses the sampling memory state.
- True-TXL checkpoints trained through the affected update path remain useful
  experimental artifacts, but their PPO correctness is provisional until
  subtask 001 passes and a fresh smoke/train check exists.
- Task037/Task046 fall and reset counts are not invalidated by subtask 002; the
  velocity/root/gravity values on reset transitions are provisional.
- Task046 `18/18` retry-gate results do not become continuous-standing passes.
  The old Task044/Task045 no-reset gate remains failed.
- The current Task044/Task045 JSONs already fail on falls, so the runner-gate
  defect does not turn those failures into passes. It is still a false-pass
  risk for future runs.
- The legacy Genesis GAE defect is isolated from upstream Unitree MJLab/RSL-RL
  PPO. It does not invalidate Task027's upstream MJLab baseline.

## Planned Slices

1. `001-txl-rollout-boundary-state-consistency.md`
   - Preserve or exactly reconstruct rollout-start True-TXL memory for PPO
     replay.
   - Add a no-optimizer-step old/new log-probability parity regression.

2. `002-multitrial-terminal-metric-capture.md`
   - Capture failure/timeout state metrics before physical auto-reset.
   - Prove reset-pose values cannot overwrite terminal-frame metrics.

3. `003-timeout-aware-genesis-gae.md`
   - Preserve per-step termination/truncation and terminal observations.
   - Bootstrap timeouts without crossing into the reset episode.

4. `004-fall-conditioned-recovery-shaping.md`
   - Make Task046 recovery shaping conditional on the preceding inner reset
     being a fall.
   - Keep timeout/no-prior-reset cases unshaped and rerun the affected smoke.

5. `005-continuous-runner-identity-gate.md`
   - Record the actual instantiated runner class.
   - Include exact runner identity in pipeline pass/failure reasons.

6. `006-regression-and-evidence-reconciliation.md`
   - Run the combined local/static and local RTX 4090 verification matrix.
   - Rerun affected eval evidence and update claims without overwriting old
     diagnostic JSONs.

7. `007-task041-train-env-mode-root-cause.md`
   - Explain the local Task041 collapse from the actual registry mode and
     TensorBoard evidence.
   - Make the training entry point explicitly load `play=False` and reject
     play-like episode lengths.

## Dependency Order

- Subtask 001 is the first blocker and must close before new True-TXL quality
  training is promoted.
- Subtasks 002, 003, 004, and 005 are code-independent and may proceed after
  their contracts are frozen.
- Subtask 004's local RTX 4090 consumer comparison depends on the corrected
  reward implementation.
- Subtask 006 closes only after subtasks 001-005 have test evidence.

## Acceptance Criteria

Task047 is accepted only when current evidence proves all of the following:

- a deterministic no-optimizer-step test shows recomputed sequence log
  probabilities match sampling log probabilities and PPO ratios start at one
  under non-empty rollout-start memory;
- terminal-frame multi-trial metrics come from the pre-reset state, with a fake
  env regression whose reset pose is deliberately different;
- legacy Genesis GAE bootstraps truncation from the pre-reset terminal
  observation, treats true termination as terminal, and does not propagate GAE
  across an auto-reset episode boundary;
- Task046 recovery shaping is active only when the previous inner reset was a
  fall, and state is cleared on outer reset;
- Task044 JSON records actual and expected runner identity, and runner mismatch
  forces `pipeline_pass=false`;
- targeted pytest, `inspect_agent`, Python compile, and shell syntax checks pass
  in an environment with the required dependencies;
- fresh local RTX 4090 smoke/eval JSONs exist for the affected True-TXL,
  Task037/Task046, and Task044 routes before their corrected claims are
  promoted;
- if local robot-runtime dependencies, checkpoints, or assets are unavailable,
  the missing artifact is recorded as a blocker and no corrected robot-quality
  claim is promoted from substitute smoke evidence;
- local runtime dependency evidence follows the current MuJoCo-only route;
  Genesis is not a required runtime gate for local training unless the route is
  explicitly reopened;
- old evidence is retained as historical diagnostic evidence and is not
  silently overwritten;
- no standing, memory-causality, LocoFormer reproduction, deployment, or
  superiority claim is made merely from correctness-smoke completion.

## Evidence Gate

Local RTX 4090 target shape:

```bash
UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m pytest -q -p no:cacheprovider \
  tests/test_task040_sequence_txl_ppo_update_smoke.py \
  tests/test_task037_multitrial_contract.py \
  tests/test_ppo_loop.py \
  tests/test_task044_hidden_fault_target.py \
  tests/test_task044_continuous_fault_eval.py
UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m h200_locomotion_lab.tools.inspect_agent
UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m compileall -q src tests
find .agent/task -type f -name '*.sh' -exec bash -n {} +
UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m h200_locomotion_lab.tools.task033_history_buffer_smoke \
  --num-envs 64 --history-len 4 --steps 8 --benchmark-steps 16 \
  --device cuda \
  --output-json .agent/task/task047-training-eval-correctness-repair/task047_local_4090_history_buffer_smoke.json
UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m h200_locomotion_lab.tools.task047_local_cuda_ppo_core_smoke \
  --device cuda:0 \
  --output-json .agent/task/task047-training-eval-correctness-repair/task047_local_4090_ppo_core_smoke.json
UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python - <<'PY'
import mujoco
xml = '<mujoco model="tiny"><worldbody><body pos="0 0 1"><joint type="free"/><geom type="box" size="0.05 0.05 0.05" mass="1"/></body></worldbody></mujoco>'
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
for _ in range(5):
    mujoco.mj_step(model, data)
print(mujoco.__version__, model.nq, model.nv, data.time)
PY
```

The exact local RTX 4090 commands and output paths are recorded in the owning
subtask before execution. No checkpoint, dataset, asset, or upstream repository
may be downloaded as part of this task without explicit user approval.

## Log

- 2026-08-07 Opened from a read-only project/task/code audit requested by the
  user.
- 2026-08-07 Split the five defects into independent closed units plus one
  final evidence-reconciliation unit. No production code was changed while
  opening this task.
- 2026-08-07 Opening local environment note: the initial shell lacked the
  simulation/training Python dependencies needed for pytest/runtime
  reproduction.
  Task creation and inventory checks could run before dependency setup; runtime
  evidence still requires the appropriate local simulator/RTX 4090 environment.
- 2026-08-07 Created branch `task047-training-eval-correctness-repair`.
- 2026-08-07 Created repo-local uv environment with
  `UV_PROJECT_ENVIRONMENT=.venv uv sync --extra dev --extra training`.
  `.venv/` is gitignored and `uv.lock` is generated in the repo. Genesis,
  MuJoCo, and MJLab extras were not installed; no assets/checkpoints/datasets
  were downloaded.
- 2026-08-07 Implemented local fixes for subtasks 001-005:
  rollout-start True-TXL memory replay/parity diagnostics; Task037 terminal
  metric capture and eval override; timeout-aware legacy Genesis GAE;
  fall-conditioned Task046 recovery shaping; and actual-runner identity gating
  for Task044 continuous eval.
- 2026-08-07 Local verification passed:
  `UV_PROJECT_ENVIRONMENT=.venv uv run pytest -q -p no:cacheprovider
  tests/test_task040_sequence_txl_ppo_update_smoke.py
  tests/test_task037_multitrial_contract.py tests/test_ppo_loop.py
  tests/test_task044_hidden_fault_target.py
  tests/test_task044_continuous_fault_eval.py` -> 57 passed;
  `UV_PROJECT_ENVIRONMENT=.venv uv run pytest -q -p no:cacheprovider` -> 714
  passed;
  `UV_PROJECT_ENVIRONMENT=.venv uv run python -m
  h200_locomotion_lab.tools.inspect_agent` -> passed;
  `UV_PROJECT_ENVIRONMENT=.venv uv run python -m compileall -q src tests` ->
  passed; `find .agent/task -type f -name '*.sh' -exec bash -n {} +` ->
  passed; `git diff --check` -> passed.
- 2026-08-07 Full pytest also surfaced and fixed a small pre-existing
  cross-platform MJCF `meshdir` absolute-path bug in
  `robots/g1like_mjcf_patch.py`; this was outside the five Task047 defects but
  required for full local regression to pass.
- 2026-08-07 User changed the runtime target from H200 to local training on an
  RTX 4090. Local `nvidia-smi` reported `NVIDIA GeForce RTX 4090`, driver
  `535.309.01`, CUDA runtime `12.2`, and high existing memory use
  (`45838MiB / 49140MiB`; process `539479` used `41262MiB`; Isaac process
  `794854` used `1706MiB`). No existing user process was stopped.
- 2026-08-07 Repaired the repo-local uv CUDA environment after the initial
  `uv sync` installed a torch build that could not initialize against the
  local driver. Evidence command
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python` reported
  `torch 2.5.1+cu121`, CUDA build `12.1`, `torch.cuda.is_available() == True`,
  and device `NVIDIA GeForce RTX 4090`.
- 2026-08-07 Added a `tool.uv.sources` entry for the PyTorch `cu121` index in
  `pyproject.toml` and regenerated `uv.lock`. The lock now selects
  `torch==2.5.1+cu121` from `https://download.pytorch.org/whl/cu121` for
  Linux x86_64. A full `uv sync --locked --extra dev --extra training`
  attempted to reconcile `.venv`, but stalled while downloading the 744MiB
  torch wheel and was stopped; `.venv` remains functionally CUDA-valid but
  `uv sync --check --locked --extra dev --extra training` still reports it
  outdated because torch was installed from a local `/tmp` wheel and old cu13
  packages remain installed. CUDA evidence commands continue to use
  `uv run --no-sync` until a complete sync finishes.
- 2026-08-07 Local robot-runtime blockers: `genesis`, `mujoco`, `mjlab`, and
  `src.tasks` are not installed in `.venv`; the configured G1 asset
  `/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_27dof_nohand.xml`
  is absent. No simulator asset, checkpoint, dataset, or upstream repository
  was downloaded.
- 2026-08-07 Local RTX 4090 CUDA smoke evidence:
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m
  h200_locomotion_lab.tools.task033_history_buffer_smoke --num-envs 64
  --history-len 4 --steps 8 --benchmark-steps 16 --device cuda
  --output-json
  .agent/task/task047-training-eval-correctness-repair/task047_local_4090_history_buffer_smoke.json`
  -> passed, `buffer_device=cuda:0`, GPU `NVIDIA GeForce RTX 4090`.
- 2026-08-07 Added and ran
  `h200_locomotion_lab.tools.task047_local_cuda_ppo_core_smoke` for a
  simulator-free local CUDA PPO-core training smoke:
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m
  h200_locomotion_lab.tools.task047_local_cuda_ppo_core_smoke --device cuda:0
  --output-json
  .agent/task/task047-training-eval-correctness-repair/task047_local_4090_ppo_core_smoke.json`
  -> passed, `torch=2.5.1+cu121`, `torch_cuda_build=12.1`, 2 PPO updates,
  rollout/action/terminal-value tensors on `cuda:0`, actor and value parameters
  changed.
- 2026-08-07 Post-RTX4090-target verification:
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m pytest -q -p
  no:cacheprovider tests/test_task040_sequence_txl_ppo_update_smoke.py
  tests/test_task037_multitrial_contract.py tests/test_ppo_loop.py
  tests/test_task044_hidden_fault_target.py
  tests/test_task044_continuous_fault_eval.py` -> 57 passed;
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m pytest -q -p
  no:cacheprovider` -> 714 passed;
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync ruff check
  src/h200_locomotion_lab/tools/task047_local_cuda_ppo_core_smoke.py` ->
  passed after import sorting;
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m compileall -q src
  tests`, `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m
  h200_locomotion_lab.tools.inspect_agent`, task shell syntax, and
  `git diff --check` -> passed.
- 2026-08-07 Lockfile validation after adding the cu121 source:
  `UV_PROJECT_ENVIRONMENT=.venv uv lock --check` -> passed;
  `UV_PROJECT_ENVIRONMENT=.venv uv sync --check --locked --extra dev --extra
  training` -> failed with an outdated-environment report only. The report
  would replace the local-file torch install with registry
  `torch==2.5.1+cu121` and uninstall stale cu13 packages; it did not indicate
  a code/test failure.
- 2026-08-09 Continued the MJLab local CUDA route without updating the driver
  or using Docker. Upgraded the editable local package pin in
  `external/unitree_rl_mjlab/setup.py` to `mujoco>=3.11.0` and
  `mujoco-warp>=3.11.0`, kept `warp-lang 1.16.0`, and preserved the existing
  `torch 2.5.1+cu121` + driver 12.2 setup.
- 2026-08-09 Patched the local MJLab site-packages compatibility layer for the
  new warp API (`get_cuda_driver_version()` and the removed `ls_parallel`
  setter) and fixed `reset_joints_by_offset()` to broadcast shared
  `soft_joint_pos_limits` instead of indexing env dim 0. Also stopped the
  task-local `wcwidth` stubs from shadowing the real package when it is
  installed.
- 2026-08-09 CUDA smoke progression: the Task040 sequence-aware true-TXL PPO
  update smoke now reaches and completes one training iteration on the RTX
  4090 with `learn_returned=true` and writes a fresh local model/checkpoint
  under `outputs/task040/sequence_txl_ppo_update_smoke/`. The smoke still
  fails the Task047 rollout-start logprob parity checks, so the correctness
  claim remains blocked, but the MJLab/MuJoCo Warp runtime path itself is now
  executing.
- 2026-08-07 User reiterated that the runtime target is local RTX 4090, not
  H200. Rechecked local state: torch/CUDA still imports successfully
  (`torch 2.5.1+cu121`, CUDA build `12.1`, CUDA available on
  `NVIDIA GeForce RTX 4090`). The installed module check still reports
  `mujoco`, `genesis`, `mjlab`, and `src.tasks` missing.
- 2026-08-07 Attempted to install local simulator Python dependencies without
  downloading robot assets or upstream repositories. `uv pip install --python
  .venv/bin/python 'mujoco>=3.2'` stalled while downloading the 18MiB MuJoCo
  wheel and was stopped after several minutes of near-zero CPU. `uv pip
  install --python .venv/bin/python --dry-run 'genesis-world>=0.3'` also
  stalled and was stopped. Environment was rechecked afterward:
  `mujoco/genesis/mjlab/src.tasks` remain missing; torch remains usable.
- 2026-08-07 Current GPU pressure is a hard local-training blocker:
  `nvidia-smi` reported only `269MiB` free on the RTX 4090, with process
  `539479` using `41262MiB` and Isaac processes `794854` and `1010343` using
  `1706MiB` and `2910MiB`. No existing user process was stopped.
- 2026-08-07 A deliberately tiny local CUDA PPO-core smoke rerun
  (`--n-envs 8 --rollout-steps 2 --ppo-updates 1`) failed at actor-critic CUDA
  allocation with `RuntimeError: CUDA error: out of memory`. This is recorded
  as machine-state evidence only; earlier successful CUDA-core evidence remains
  valid for the code path under available memory.
- 2026-08-07 User narrowed the local runtime route to MuJoCo-only training:
  Genesis must not be used for this run. Installed MuJoCo into the repo-local
  `.venv` via pip fallback after `uv pip install` stalled:
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m pip install
  --timeout 180 --progress-bar off 'mujoco>=3.2'` -> installed
  `mujoco==3.11.0`.
- 2026-08-07 A Genesis install had already completed before the MuJoCo-only
  instruction was processed. To honor the new route, uninstalled the Genesis
  top-level/runtime packages with
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m pip uninstall -y
  genesis-world gs-madrona quadrants`. Verification afterward reported
  `torch OK`, `mujoco OK`, `genesis MISSING`, `mjlab MISSING`, and
  `src.tasks MISSING`; `pip check` reported no broken requirements.
- 2026-08-07 MuJoCo-only runtime smoke passed:
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python` compiled a tiny MJCF
  string with `mujoco.MjModel.from_xml_string`, stepped five frames, and
  reported `mujoco 3.11.0`, `nq=7`, `nv=6`, `time=0.01`.
- 2026-08-07 Local search under `/home/xyzl/yubo` and `/home/xyzl` found no
  existing `unitree_rl_mjlab` or `mjlab` runtime directory. No upstream repo,
  robot asset, checkpoint, or dataset was downloaded.
- 2026-08-07 Post-MuJoCo-only verification:
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m pytest -q -p
  no:cacheprovider` -> 714 passed;
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m compileall -q src
  tests`, `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m
  h200_locomotion_lab.tools.inspect_agent`, and `git diff --check` -> passed.
  `nvidia-smi` still showed only `325MiB` free on the RTX 4090, with existing
  processes `539479`, `794854`, and `1010343` occupying the bulk of memory.
- 2026-08-07 Consolidated the user-selected MuJoCo-only route in project
  metadata: `pyproject.toml` now makes `mujoco>=3.2` part of the `training`
  extra and removes the advertised `genesis` extra; README and `.agent/doc`
  now state that local training is RTX 4090 + MuJoCo-only. Historical Genesis
  adapter code and task notes remain untouched but are not a runtime gate.
- 2026-08-07 Regenerated `uv.lock` after the MuJoCo-only dependency change:
  `UV_PROJECT_ENVIRONMENT=.venv uv lock` -> resolved 61 packages and removed
  `genesis-world`, `gs-madrona`, `quadrants`, and their heavy transitive
  packages from the lock. `UV_PROJECT_ENVIRONMENT=.venv uv lock --check` ->
  passed; `rg -n "genesis-world|gs-madrona|quadrants" pyproject.toml uv.lock`
  -> no matches. Runtime verification with `uv run --no-sync` reported
  `torch OK`, `mujoco OK`, `genesis MISSING`, `mjlab MISSING`, `src.tasks
  MISSING`, `torch 2.5.1+cu121`, CUDA available on `NVIDIA GeForce RTX 4090`,
  and MuJoCo tiny MJCF compile/step passed with `mujoco 3.11.0`.
- 2026-08-07 `UV_PROJECT_ENVIRONMENT=.venv uv sync --check --locked --extra
  dev --extra training --inexact` still reports the environment outdated only
  because it would replace the editable project metadata and the local-file
  torch wheel with the registry `torch==2.5.1+cu121`. Full sync would require a
  large torch wheel download, so evidence commands continue to use the
  already-functional `.venv` with `uv run --no-sync`.
- 2026-08-07 Searched for a repo-native MuJoCo robot training entry. The
  existing `h200_locomotion_lab.tools.g1_ppo_smoke` is still explicitly backed
  by `VectorizedGenesisBackend`, while the Task040/Task037/Task044 robot
  training/eval routes import MJLab/`src.tasks`. Under the current MuJoCo-only
  decision, full robot training still needs either the missing MJLab MuJoCo
  runtime/tasks/assets or a new direct MuJoCo backend/training entry; the
  existing Genesis-backed smoke is not a valid substitute.
- 2026-08-10 Continued the user-selected local RTX 4090 + MJLab/MuJoCo Warp
  route without updating the NVIDIA driver, without Docker, and without
  downloading external checkpoints/datasets/assets. The earlier
  one-iteration Task040 smoke failure was reproduced as an invalid gate for
  non-empty rollout-start memory, and a two-iteration MJLab smoke showed the
  real remaining correctness drift:
  `max_logprob_abs_error=0.08075332641601562`,
  `max_ratio_abs_error=0.08410346508026123`, with
  `rollout_start_memory_non_empty=true`.
- 2026-08-10 Fixed the remaining Task040 parity bug by storing per-step actor
  observation-normalizer snapshots at `Task040SequenceAwareTrueTxlPPO.act()`
  time and replaying each rollout step with the sampling-time normalizer state
  in `Task038TrueTxlMemoryModel.task040_forward_sequence()`. This preserves
  the existing rollout-start True-TXL memory snapshot, per-env slicing, and
  live inference-cache isolation, while keeping RSL-RL's normalizer update
  behavior during environment collection.
- 2026-08-10 Added regression coverage proving parity depends on both the
  per-env rollout-start memory and per-step actor normalizer snapshots. Also
  fixed `task038_true_txl_multitrial_eval_smoke.py` to pass through
  `--final-window-s` to the Task037 evaluator after a local eval smoke exposed
  the missing argparse field.
- 2026-08-10 Fresh local verification passed:
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m pytest -q -p
  no:cacheprovider tests/test_task040_sequence_txl_ppo_update_smoke.py
  tests/test_task038_true_txl_multitrial_eval_smoke.py
  tests/test_task037_multitrial_contract.py tests/test_ppo_loop.py
  tests/test_task044_hidden_fault_target.py
  tests/test_task044_continuous_fault_eval.py` -> 78 passed;
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m compileall -q src
  tests`, `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m
  h200_locomotion_lab.tools.inspect_agent`, `git diff --check`, and task
  shell syntax checks -> passed.
- 2026-08-10 Fresh Task040 MJLab CUDA smoke/train evidence:
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m
  h200_locomotion_lab.tools.task040_sequence_txl_ppo_update_smoke --device
  cuda:0 --num-envs 2 --rollout-steps 2 --iterations 2 --num-mini-batches 1
  --output-json
  .agent/task/task047-training-eval-correctness-repair/task047_local_4090_task040_smoke_after_normalizer_snapshot_fix_v2.json`
  -> passed, wrote `outputs/task040/sequence_txl_ppo_update_smoke/model_1.pt`,
  recorded `normalizer_replay_mode=per_step_snapshot`,
  `normalizer_snapshot_count=2`, `rollout_start_memory_non_empty=true`,
  `max_logprob_abs_error=0.0`, and `max_ratio_abs_error=0.0`.
- 2026-08-10 Fresh local CUDA support evidence:
  `task047_local_4090_history_buffer_smoke_after_normalizer_snapshot_fix.json`
  -> passed on `cuda:0`; `task047_local_4090_ppo_core_smoke_after_normalizer_snapshot_fix.json`
  -> passed on `cuda:0`; MuJoCo tiny MJCF compile/step smoke reported
  `mujoco 3.11.0`, `nq=7`, `nv=6`, `time=0.01`.
- 2026-08-10 Used the newly trained tiny local checkpoint
  `outputs/task040/sequence_txl_ppo_update_smoke/model_1.pt` for a Task038
  true-TXL multi-trial eval pipeline smoke:
  `task047_local_4090_task038_multitrial_eval_after_final_window_fix.json`
  -> `pass=true`, `pipeline_pass=true`, `metric_schema=task037_multitrial_eval_metrics_v2`,
  and no quality/training/eval/reproduction/superiority claim. This is
  pipeline evidence only, not a policy-quality result.
- 2026-08-10 Repaired the Task041 real-training evidence gate so
  `train_pipeline_pass` now requires strict Task047 sequence replay parity:
  `algorithm_debug.last_logprob_parity.pass=true`, logprob/ratio absolute
  errors within the recorded threshold, and non-empty rollout-start memory.
  Added targeted regression coverage in
  `tests/test_task041_sequence_txl_clean_train.py`; changed-test verification
  with `tests/test_task040_sequence_txl_ppo_update_smoke.py` passed:
  `36 passed in 2.91s`.
- 2026-08-10 Diagnosed mini-batch-sensitive parity drift before the real run:
  `task047_local_4090_probe_task041_env512_step24_iter2_mb4.json` failed the
  strict parity gate (`max_logprob_abs_error=0.00083160400390625`,
  `max_ratio_abs_error=0.0008319616317749023`, threshold `1e-5`), while
  `task047_local_4090_probe_task041_env512_step24_iter2_mb1.json` passed
  (`max_logprob_abs_error=7.62939453125e-06`,
  `max_ratio_abs_error=7.62939453125e-06`). Therefore the current
  parity-safe Task047 training route uses `num_mini_batches=1`; the older
  `mb4` 100-iteration JSON is retained only as diagnostic evidence even though
  its stale summary field reported `train_pipeline_pass=true`.
- 2026-08-10 Ran a real from-scratch Task041 MJLab/MuJoCo Warp training job on
  local RTX 4090, preserving existing user GPU processes and without Docker,
  driver update, external checkpoint, dataset, asset, or repo download:
  `num_envs=512`, `rollout_steps=24`, `iterations=100`, `save_interval=25`,
  `num_mini_batches=1`, `num_learning_epochs=2`, `seed=4100204`,
  `device=cuda:0`, run
  `local_4090_env512_step24_iter100_mb1_seed4100204`.
  Evidence JSON:
  `task047_local_4090_task041_real_train_env512_step24_iter100_mb1.json`.
  Output directory:
  `outputs/task047/real_train/local_4090_env512_step24_iter100_mb1_seed4100204/`.
  Final checkpoint exists at
  `outputs/task047/real_train/local_4090_env512_step24_iter100_mb1_seed4100204/model_99.pt`;
  intermediate checkpoints `model_0.pt`, `model_25.pt`, `model_50.pt`, and
  `model_75.pt` also exist. Training returned normally in `573.7001986503601s`
  with `checkpoint_exists=true`, `learn_returned=true`, `failure_reasons=[]`,
  `train_pipeline_pass=true`, `sequence_update_batches=200`,
  `normalizer_replay_mode=per_step_snapshot`, `normalizer_snapshot_count=24`,
  `rollout_start_memory_non_empty=true`, `stateless_fallback_forward_batches=0`,
  `max_logprob_abs_error=7.62939453125e-06`, and
  `max_ratio_abs_error=7.62939453125e-06`.
- 2026-08-10 Task041 real training changed the True-TXL trainable parameters,
  proving this was not just a no-op checkpoint write:
  `attention_layers_delta_norm=7.35865592956543`,
  `token_projection_delta_norm=2.7628676891326904`, and
  `memory_output_projection_delta_norm=0.7706440687179565`.
  Top-level `pass=false` in the training JSON remains intentionally
  conservative because the script does not promote a policy-quality,
  eval, reproduction, or superiority claim from training completion alone.
- 2026-08-10 A direct Task038 eval attempt against the Task041 checkpoint was
  rejected as the wrong eval route: Task038 defaults constructed
  `memory_latent_dim=128` without the Task041 passthrough/warmstart actor
  shape, while the checkpoint uses `memory_latent_dim=32`,
  `base_obs_passthrough=true`, and `adaptation_warmstart=true`. The mismatch
  is recorded in
  `task047_local_4090_task041_real_train_env512_step24_iter100_mb1_eval.json`
  and is not treated as checkpoint failure or policy-quality evidence.
- 2026-08-10 Ran the matching Task041 eval route for the real checkpoint:
  `task047_local_4090_task041_real_train_env512_step24_iter100_mb1_task041_eval.json`.
  Eval used checkpoint
  `outputs/task047/real_train/local_4090_env512_step24_iter100_mb1_seed4100204/model_99.pt`,
  `num_envs=64`, `steps=120`, `trial_length_s=0.5`, `seed=4101205`,
  `device=cuda:0`, and runner `Task038TrueTxlMemoryK160Runner`.
  Pipeline checks passed:
  `pipeline_pass=true`, `task041_pipeline_pass=true`,
  `sequence_aware_update_train_pipeline_pass=true`,
  `sequence_aware_checkpoint_match=true`, and
  `memory_ablation_mode_match=true`.
  Quality gate failed, so `pass=false` and all promotion claims remain false.
  Failure reasons:
  `final_yaw_vel_error_too_high`,
  `gravity_xy_max_regressed_from_trial0`,
  `root_z_min_regressed_from_trial0`,
  `lin_vel_error_mean_regressed_from_trial0`, and
  `yaw_vel_error_mean_regressed_from_trial0`.
  Aggregate eval metrics: `sample_count=7680`, `completion_count=256`,
  `fall_count=0`, `fall_ratio=0.0`, `zero_fall_ratio=1.0`,
  `lin_vel_error_mean=0.3388870519896348`,
  `yaw_vel_error_mean=0.46774421880642575`,
  `gravity_xy_max=0.22095124423503876`, and
  `root_z_min=0.7704654335975647`.
- 2026-08-10 Final targeted verification after the Task041 gate and log
  updates:
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m pytest -q -p
  no:cacheprovider tests/test_task041_sequence_txl_clean_train.py
  tests/test_task040_sequence_txl_ppo_update_smoke.py
  tests/test_task038_true_txl_multitrial_eval_smoke.py
  tests/test_task037_multitrial_contract.py tests/test_ppo_loop.py
  tests/test_task044_hidden_fault_target.py
  tests/test_task044_continuous_fault_eval.py` -> 96 passed;
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m compileall -q src
  tests` -> passed;
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m
  h200_locomotion_lab.tools.inspect_agent` -> passed;
  `git diff --check` -> passed;
  `find .agent/task -type f -name '*.sh' -exec bash -n {} +` -> passed.
- 2026-08-11 Added
  `h200_locomotion_lab.tools.task047_task041_eval_video` as a Task047 visual
  diagnostic helper. The script reuses the Task041/Task037 MJLab runner and
  actor construction path, records checkpoint/seed/camera provenance, and
  explicitly keeps `quality_claim=false`, `eval_claim=false`,
  `reproduction_claim=false`, and `visual_diagnostic_only=true`.
  `py_compile`, `ruff check`, and `git diff --check` passed for the new helper.
- 2026-08-11 Rendered the Task041 `model_99.pt` checkpoint under the matching
  eval setting (`num_envs=64`, render `env_idx=0`, `steps=120`,
  `trial_length_s=0.5`, `seed=4101205`, fixed command `lin_vel_x=0.4`) to
  `outputs/task047/videos/task041_model99_eval_seed4101205_env64_env0.mp4`
  with provenance JSON
  `outputs/task047/videos/task041_model99_eval_seed4101205_env64_env0.json`.
  The video artifact check passed: `frame_count=61`, `fps=25.0`,
  `video_duration_s=2.44`, `video_bytes=114964`, H.264 640x480, and
  `train_summary_checkpoint_match=true`. The recorder metrics align with the
  prior numeric eval aggregate (`sample_count=7680`, timeout resets only,
  `lin_vel_error_mean=0.33888939023017883`,
  `yaw_vel_error_mean=0.46787989139556885`,
  `gravity_xy_max=0.22093573212623596`,
  `root_z_min=0.7704667448997498`). A mid-frame PNG extracted from the MP4
  showed the robot in frame, so this is not a black-screen render.
- 2026-08-11 Also rendered a longer single-env visual diagnostic at
  `outputs/task047/videos/task041_model99_eval_short.mp4` with JSON
  `outputs/task047/videos/task041_model99_eval_short.json`. This run used
  `num_envs=1`, `steps=150`, `trial_length_s=5.0`, and `seed=4102205`; it
  recorded one fall reset and is retained only as additional behavior
  inspection, not as the matching eval-setting video.
- 2026-08-11 Manual video inspection correction: the matching eval-setting MP4
  is not a useful visual gait check because the numeric eval setting uses
  `trial_length_s=0.5` and timeout-resets every 0.5s. It only verifies the
  short-window eval/render path. A new longer fixed-world-camera diagnostic was
  rendered to
  `outputs/task047/videos/task041_model99_visual_long_world_seed4102205_env1.mp4`
  with JSON
  `outputs/task047/videos/task041_model99_visual_long_world_seed4102205_env1.json`
  (`num_envs=1`, `steps=250`, `trial_length_s=10.0`, `seed=4102205`,
  fixed command `lin_vel_x=0.4`). It recorded `reset_reason_counts={"1": 2}`,
  `root_z_min=0.31180253624916077`, `gravity_xy_max=0.9383290410041809`,
  and visible falls in the extracted contact sheet. This reinforces the
  existing `quality_gate_pass=false` conclusion: the checkpoint trains and
  runs, but it is not a successful locomotion policy.
- 2026-08-11 Route B scratch-lineage long-training chunk completed on the local
  RTX 4090 without Docker, driver update, or external checkpoint/dataset/asset/
  repo download. The run resumed only from the local scratch-lineage checkpoint
  `outputs/task047/real_train/local_4090_env512_step24_iter100_mb1_seed4100204/model_99.pt`
  and ran `num_envs=512`, `rollout_steps=24`, `iterations=500`,
  `save_interval=100`, `num_mini_batches=1`, `num_learning_epochs=2`,
  `seed=4100304`, `device=cuda:0`. Evidence JSON:
  `.agent/task/task047-training-eval-correctness-repair/task047_local_4090_task041_scratchlineage_from_model99_env512_step24_iter500_mb1.json`.
  Checkpoints exist at `model_300.pt`, `model_400.pt`, and final
  `outputs/task047/long_train/local_4090_scratchlineage_from_model99_env512_step24_iter500_mb1_seed4100304/model_499.pt`.
  The command returned normally in `2613.5759496688843s` with
  `checkpoint_exists=true` and `learn_returned=true`, but strict Task047
  training pipeline promotion failed:
  `failure_reasons=["algorithm_debug_logprob_parity_failed",
  "algorithm_debug_logprob_error_too_high",
  "algorithm_debug_ratio_error_too_high"]`,
  `max_logprob_abs_error=0.00003814697265625`, and
  `max_ratio_abs_error=0.00003814697265625` against threshold `1e-5`.
  Stdout training metrics also showed policy collapse rather than gait
  learning: by the final window, mean episode length had fallen to about
  `13` steps and `Episode_Termination/fell_over` was about `120`.
- 2026-08-11 Matching Task041 post-chunk eval for Route B wrote
  `.agent/task/task047-training-eval-correctness-repair/task047_local_4090_task041_scratchlineage_from_model99_env512_step24_iter500_mb1_eval2s.json`
  using final checkpoint `model_499.pt`, `num_envs=64`, `steps=300`,
  `trial_length_s=2.0`, `seed=4101305`, and fixed command `lin_vel_x=0.4`.
  Eval execution pipeline passed, but the quality gate failed:
  `pass=false`, `pipeline_pass=true`, `quality_gate_pass=false`,
  `sequence_aware_update_train_pipeline_pass=false`, aggregate
  `fall_ratio=1.0`, `zero_fall_ratio=0.0`,
  `lin_vel_error_mean=0.7373346236844858`,
  `yaw_vel_error_mean=5.194398358662923`,
  `gravity_xy_max=0.9575986266136169`, and
  `root_z_min=0.7207261919975281`. The final trial recorded
  `fall_ratio=1.0`, `reset_reason_counts={"1": 1564}`,
  `lin_vel_error_mean=0.7807158827781677`,
  `yaw_vel_error_mean=5.391958713531494`, and action mean L2
  `19.13002586364746`.
- 2026-08-11 Rendered the Route B final checkpoint with the long fixed
  world-camera diagnostic:
  `outputs/task047/videos/task041_scratchlineage_iter500_visual_long_world_seed4102305_env1.mp4`
  and provenance JSON
  `outputs/task047/videos/task041_scratchlineage_iter500_visual_long_world_seed4102305_env1.json`.
  The recorder used `num_envs=1`, `steps=250`, `trial_length_s=10.0`,
  `seed=4102305`, fixed command `lin_vel_x=0.4`, and produced 251 frames
  (`5.02s`, `315196` bytes) with
  `train_summary_checkpoint_match=true` and `visual_diagnostic_only=true`.
  It recorded `reset_reason_counts={"1": 62}`,
  `lin_vel_error_mean=0.7624918818473816`,
  `yaw_vel_error_mean=5.257551670074463`,
  `gravity_xy_max=0.9272055625915527`, and
  `root_z_min=0.7435759902000427`. A contact sheet at
  `outputs/task047/videos/task041_scratchlineage_iter500_visual_long_world_seed4102305_env1_contact.jpg`
  was manually inspected and shows repeated stand-then-twist/side-fall resets,
  not locomotion.
- 2026-08-11 Opened subtask 007 after tracing the local runs to a shared eval
  helper that always called `load_env_cfg(task, play=True)`. The Task041 train
  entry point now has a dedicated train-config loader and a play-config gate.
  Historical gait reconstruction is split into Task048 rather than continuing
  the collapsed `model_499.pt` lineage.

## Review

Status: the first Route B scratch-lineage long-training chunk completed and
produced a final checkpoint, eval JSON, MP4, and contact-sheet evidence, but it
does not reproduce a successful Task041 gait. The run also revealed that even
the current `num_mini_batches=1` route can exceed the strict `1e-5`
sequence-replay parity threshold over a longer 500-iteration continuation
(`3.814697265625e-05`), so this chunk is not a Task047 train-pipeline pass.

Task047 is still not a full project-level pass: corrected smoke routes remain
useful, and the earlier 100-iteration scratch-lineage run passed strict
training parity, but the 500-iteration Route B chunk collapses into rapid fall
resets and fails quality eval. Continuing another blind scratch chunk from
`model_499.pt` is not recommended without first diagnosing the reward/
termination/reset or policy-initialization failure mode. Task044 continuous and
Task046 corrected quality/eval claims still require matching local checkpoints
and fresh JSON evidence. No standing, memory-causality, reproduction,
deployment, or superiority claim is promoted.
