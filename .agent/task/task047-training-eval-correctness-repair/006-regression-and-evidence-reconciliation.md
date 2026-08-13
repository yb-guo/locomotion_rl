# 006: Regression and Evidence Reconciliation

## Route

Close Task047 only with combined regression evidence and corrected experiment
provenance. This unit starts after subtasks 001-005 have implementation and
targeted tests.

Run two layers:

1. local/static verification for all affected modules and task inventory;
2. local RTX 4090 smoke/eval reruns where simulator/runtime evidence is
   required.

Retain old JSONs. Corrected outputs use new paths or explicit schema/version
fields and link back to their historical comparison inputs.

## Acceptance

- Targeted local pytest covers all five regressions.
- `inspect_agent`, Python compilation, and task shell syntax checks pass.
- Corrected Task040 smoke records non-empty rollout-start memory and old/new
  log-probability parity before update.
- Corrected Task037/Task046 eval records pre-reset terminal metric provenance.
- Corrected Task046 reward smoke records fall-conditioned eligible samples; a
  new quality claim requires new training plus matrix eval.
- Corrected Task044 continuous eval records actual runner identity and retains
  the unchanged no-reset gate.
- Task040-Task046 task reviews are amended only where conclusions materially
  change; no old negative or diagnostic result is erased.
- Project/task inventory inconsistencies encountered during closure are listed
  separately from code defects and fixed without renumbering historical tasks.
- Portable compact summaries are stored in the repo when licensing/size allow;
  otherwise missing-local-artifact limits are explicit.

## Evidence Matrix

| Area | Local/static evidence | Local RTX 4090 runtime evidence | Claim boundary |
|---|---|---|---|
| TXL PPO | parity and cache lifecycle tests | corrected sequence-update smoke, blocked until MJLab/tasks/assets exist locally | PPO correctness only |
| Multi-trial metrics | reset-pose fake env | known falling checkpoint rerun, blocked until runtime/assets/checkpoint exist locally | metric provenance only |
| Legacy GAE | analytic mixed-mask tests | no Genesis runtime gate under the current MuJoCo-only route | legacy return correctness only |
| Task046 reward | reset-reason state-machine tests | fall-conditioned consumer smoke, blocked until runtime/assets/checkpoint/free memory exist locally | no quality claim without retrain |
| Task044 runner | wrong-runner false-pass test | corrected continuous eval, blocked until runtime/assets/checkpoint exist locally | pipeline identity only |

## Log

- 2026-08-07 Opened as the final Task047 closure unit.
- 2026-08-07 Opening note: the initial local shell lacked pytest/torch/numpy and
  simulator runtimes. No missing-dependency result may be recorded as a
  functional pass or fail.
- 2026-08-07 Created repo-local uv environment with
  `UV_PROJECT_ENVIRONMENT=.venv uv sync --extra dev --extra training`.
  `.venv/` is ignored by git and `uv.lock` was generated for reproducibility.
  Genesis/MuJoCo/MJLab simulator extras were not installed and no assets,
  checkpoints, datasets, or upstream repos were downloaded.
- 2026-08-07 Local targeted regression:
  `UV_PROJECT_ENVIRONMENT=.venv uv run pytest -q -p no:cacheprovider
  tests/test_task040_sequence_txl_ppo_update_smoke.py
  tests/test_task037_multitrial_contract.py tests/test_ppo_loop.py
  tests/test_task044_hidden_fault_target.py
  tests/test_task044_continuous_fault_eval.py` -> 57 passed, with one local
  CUDA driver warning from torch during a CPU-path optimizer test.
- 2026-08-07 Full local pytest initially exposed one Task040 debug-snapshot
  compatibility issue in a lightweight `object.__new__` fake model and one
  pre-existing cross-platform MJCF `meshdir` absolute-path bug. Both were fixed
  without simulator assets or downloads.
- 2026-08-07 Full local regression:
  `UV_PROJECT_ENVIRONMENT=.venv uv run pytest -q -p no:cacheprovider` -> 714
  passed, with one local CUDA driver warning from torch CUDA initialization.
- 2026-08-07 Local inventory/static checks:
  `UV_PROJECT_ENVIRONMENT=.venv uv run python -m
  h200_locomotion_lab.tools.inspect_agent` -> passed;
  `UV_PROJECT_ENVIRONMENT=.venv uv run python -m compileall -q src tests` ->
  passed; `find .agent/task -type f -name '*.sh' -exec bash -n {} +` ->
  passed; `git diff --check` -> passed.
- 2026-08-07 User replaced the H200 gate with local RTX 4090 training. Local
  `nvidia-smi` evidence: `NVIDIA GeForce RTX 4090`, driver `535.309.01`, CUDA
  runtime `12.2`, `45838MiB / 49140MiB` already used. Existing processes were
  left untouched.
- 2026-08-07 Local uv/CUDA evidence after manual torch wheel repair:
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python` reported
  `torch 2.5.1+cu121`, CUDA build `12.1`, CUDA available, device
  `NVIDIA GeForce RTX 4090`; `uv pip check` reported all installed packages
  compatible.
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
- 2026-08-07 Local runtime blockers:
  `genesis`, `mujoco`, `mjlab`, and `src.tasks` are missing in `.venv`; the
  configured G1 asset
  `/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_27dof_nohand.xml`
  is absent. These block full local robot training/eval for Task040,
  Task037/Task046, and Task044 without explicit simulator/asset/checkpoint
  setup.
- 2026-08-07 Local RTX 4090 CUDA smoke:
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m
  h200_locomotion_lab.tools.task033_history_buffer_smoke --num-envs 64
  --history-len 4 --steps 8 --benchmark-steps 16 --device cuda
  --output-json
  .agent/task/task047-training-eval-correctness-repair/task047_local_4090_history_buffer_smoke.json`
  -> passed.
- 2026-08-07 Added `task047_local_cuda_ppo_core_smoke.py` and ran:
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m
  h200_locomotion_lab.tools.task047_local_cuda_ppo_core_smoke --device cuda:0
  --output-json
  .agent/task/task047-training-eval-correctness-repair/task047_local_4090_ppo_core_smoke.json`
  -> passed. Scope is CUDA PPO-core only, not robot locomotion quality:
  2 PPO updates, timeout terminal values on `cuda:0`, actor/value parameters
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
- 2026-08-07 Reconciled dependency/project metadata with the MuJoCo-only route:
  `training` now includes `mujoco>=3.2`, the advertised `genesis` extra was
  removed from `pyproject.toml`, and README/`.agent/doc` no longer present
  Genesis as a current local training path. Historical Genesis code is retained
  as legacy/optional adapter code only.
- 2026-08-07 Lock/runtime verification after dependency reconciliation:
  `UV_PROJECT_ENVIRONMENT=.venv uv lock` -> resolved 61 packages and removed
  Genesis packages/transitives from the lock; `UV_PROJECT_ENVIRONMENT=.venv uv
  lock --check` -> passed; `rg -n "genesis-world|gs-madrona|quadrants"
  pyproject.toml uv.lock` -> no matches; `UV_PROJECT_ENVIRONMENT=.venv uv run
  --no-sync python -m pip check` -> passed; module check reported `torch OK`,
  `mujoco OK`, `genesis MISSING`, `mjlab MISSING`, `src.tasks MISSING`;
  MuJoCo tiny MJCF compile/step smoke passed. `uv sync --check --locked
  --extra dev --extra training --inexact` remains outdated only on editable
  project metadata and local-file torch-wheel source, not on missing MuJoCo or
  Genesis dependency requirements.
- 2026-08-07 MuJoCo-only training-entry inventory: `g1_ppo_smoke.py` remains a
  Genesis-backed legacy smoke, not a MuJoCo trainer; Task040/Task037/Task044
  robot runtime paths require MJLab/`src.tasks` plus G1 assets/checkpoints.
  Therefore no full robot training/eval claim can be promoted from the current
  installed dependency state.
- 2026-08-10 Local RTX 4090 + MJLab/MuJoCo Warp route is now runnable for the
  Task040/Task038 smoke paths without updating the driver or using Docker.
  Fresh verification: targeted Task047 pytest matrix including Task038/Task040
  wrappers -> 78 passed; compileall, inspect_agent, git diff check, and task
  shell syntax checks -> passed; history-buffer CUDA smoke, PPO-core CUDA
  smoke, and MuJoCo tiny MJCF smoke -> passed.
- 2026-08-10 Fresh robot-runtime JSONs added without overwriting old evidence:
  `task047_local_4090_task040_smoke_after_normalizer_snapshot_fix_v2.json`
  passed the corrected non-empty rollout-start memory and per-step normalizer
  parity gate; `task047_local_4090_task038_multitrial_eval_after_final_window_fix.json`
  passed the Task038/Task037 multi-trial eval pipeline using the tiny local
  Task040 smoke checkpoint. These are smoke/pipeline results only.
- 2026-08-10 Tightened Task041 real-training evidence reconciliation:
  `evaluate_train_pipeline_pass()` now rejects summaries missing strict
  Task047 logprob/ratio parity or non-empty rollout-start memory. Added tests
  in `tests/test_task041_sequence_txl_clean_train.py`; changed-test regression
  with `tests/test_task040_sequence_txl_ppo_update_smoke.py` passed:
  `36 passed in 2.91s`.
- 2026-08-10 Mini-batch probe result: `num_mini_batches=4` is not currently a
  valid Task047 parity-safe route for the Task041 real-training gate. The
  `mb4` probe recorded `max_logprob_abs_error=0.00083160400390625` and
  `max_ratio_abs_error=0.0008319616317749023` against threshold `1e-5`; the
  `mb1` probe recorded `7.62939453125e-06` for both and passed. The 100-iter
  `mb4` JSON is retained as diagnostic evidence but is not promoted.
- 2026-08-10 Real local RTX 4090 Task041 MJLab/MuJoCo Warp training completed
  from scratch without Docker, driver update, or external checkpoint/dataset/
  asset/repo download:
  `num_envs=512`, `rollout_steps=24`, `iterations=100`,
  `num_mini_batches=1`, `num_learning_epochs=2`, `seed=4100204`,
  `device=cuda:0`.
  Evidence JSON:
  `task047_local_4090_task041_real_train_env512_step24_iter100_mb1.json`.
  Log/checkpoint directory:
  `outputs/task047/real_train/local_4090_env512_step24_iter100_mb1_seed4100204/`.
  Final checkpoint:
  `outputs/task047/real_train/local_4090_env512_step24_iter100_mb1_seed4100204/model_99.pt`.
  Evidence fields: `train_pipeline_pass=true`, `failure_reasons=[]`,
  `checkpoint_exists=true`, `learn_returned=true`,
  `wall_time_s=573.7001986503601`, `sequence_update_batches=200`,
  `normalizer_replay_mode=per_step_snapshot`,
  `normalizer_snapshot_count=24`,
  `rollout_start_memory_non_empty=true`,
  `stateless_fallback_forward_batches=0`,
  `max_logprob_abs_error=7.62939453125e-06`, and
  `max_ratio_abs_error=7.62939453125e-06`.
- 2026-08-10 The real training updated trainable True-TXL parameters:
  `attention_layers_delta_norm=7.35865592956543`,
  `token_projection_delta_norm=2.7628676891326904`,
  `memory_output_projection_delta_norm=0.7706440687179565`.
  The training JSON still has top-level `pass=false` by design because no
  quality/eval/reproduction/superiority claim is promoted from training alone.
- 2026-08-10 Eval route reconciliation: a direct Task038 eval command against
  the Task041 checkpoint failed due to actor shape/config mismatch
  (`memory_latent_dim=128` default eval actor versus the checkpoint's
  `memory_latent_dim=32`, `base_obs_passthrough=true`,
  `adaptation_warmstart=true`). This is invalid-route evidence only. The
  matching Task041 eval route then completed and wrote
  `task047_local_4090_task041_real_train_env512_step24_iter100_mb1_task041_eval.json`.
  It passed pipeline/provenance gates (`pipeline_pass=true`,
  `task041_pipeline_pass=true`,
  `sequence_aware_update_train_pipeline_pass=true`,
  `sequence_aware_checkpoint_match=true`,
  `memory_ablation_mode_match=true`) but failed the quality gate:
  `final_yaw_vel_error_too_high`,
  `gravity_xy_max_regressed_from_trial0`,
  `root_z_min_regressed_from_trial0`,
  `lin_vel_error_mean_regressed_from_trial0`, and
  `yaw_vel_error_mean_regressed_from_trial0`.
  Aggregate metrics: `sample_count=7680`, `completion_count=256`,
  `fall_count=0`, `fall_ratio=0.0`, `zero_fall_ratio=1.0`,
  `lin_vel_error_mean=0.3388870519896348`,
  `yaw_vel_error_mean=0.46774421880642575`,
  `gravity_xy_max=0.22095124423503876`, and
  `root_z_min=0.7704654335975647`.
- 2026-08-10 Final targeted verification after Task041 gate and evidence-log
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
- 2026-08-11 Added and ran a Task047 Task041 video recorder for visual
  inspection only:
  `h200_locomotion_lab.tools.task047_task041_eval_video`. The matching eval
  setting render used `model_99.pt`, `num_envs=64`, render `env_idx=0`,
  `steps=120`, `trial_length_s=0.5`, `seed=4101205`, and fixed command
  `lin_vel_x=0.4`. Output MP4:
  `outputs/task047/videos/task041_model99_eval_seed4101205_env64_env0.mp4`;
  provenance JSON:
  `outputs/task047/videos/task041_model99_eval_seed4101205_env64_env0.json`.
  Artifact verification read 61 H.264 frames at 640x480 and extracted a
  non-black mid-frame. JSON recorded `pass=true`,
  `visual_diagnostic_only=true`, `quality_claim=false`, `eval_claim=false`,
  `reproduction_claim=false`, `video_bytes=114964`,
  `train_summary_checkpoint_match=true`, and timeout resets only
  (`reset_reason_counts={"2": 256}`).
- 2026-08-11 Manual video inspection found that the matching eval-setting MP4
  is not a useful gait-quality video because `trial_length_s=0.5` causes
  repeated timeout resets. A longer fixed-world-camera diagnostic was rendered:
  `outputs/task047/videos/task041_model99_visual_long_world_seed4102205_env1.mp4`
  with JSON
  `outputs/task047/videos/task041_model99_visual_long_world_seed4102205_env1.json`.
  It used `num_envs=1`, `steps=250`, `trial_length_s=10.0`, `seed=4102205`,
  and fixed command `lin_vel_x=0.4`; it recorded two fall resets
  (`reset_reason_counts={"1": 2}`), `root_z_min=0.31180253624916077`, and
  `gravity_xy_max=0.9383290410041809`. The visual evidence must therefore be
  read as a policy-quality failure, not a successful locomotion result.
- 2026-08-11 Route B scratch-lineage long-training chunk completed on the local
  RTX 4090, continuing only from the local scratch-lineage
  `model_99.pt` checkpoint and not from an external checkpoint. The command ran
  `num_envs=512`, `rollout_steps=24`, `iterations=500`,
  `save_interval=100`, `num_mini_batches=1`, `num_learning_epochs=2`,
  `seed=4100304`, and `device=cuda:0`. Evidence JSON:
  `task047_local_4090_task041_scratchlineage_from_model99_env512_step24_iter500_mb1.json`.
  Output directory:
  `outputs/task047/long_train/local_4090_scratchlineage_from_model99_env512_step24_iter500_mb1_seed4100304/`.
  Final checkpoint:
  `outputs/task047/long_train/local_4090_scratchlineage_from_model99_env512_step24_iter500_mb1_seed4100304/model_499.pt`.
  The run returned normally in `2613.5759496688843s`, but
  `train_pipeline_pass=false` because strict PPO replay parity drifted above
  the Task047 gate:
  `max_logprob_abs_error=0.00003814697265625`,
  `max_ratio_abs_error=0.00003814697265625`, threshold `1e-5`.
  By stdout observation near the end of training, mean episode length had
  collapsed to about `13` steps and `Episode_Termination/fell_over` was about
  `120`, so the apparent reward improvement was not locomotion learning.
- 2026-08-11 Route B post-chunk eval wrote
  `task047_local_4090_task041_scratchlineage_from_model99_env512_step24_iter500_mb1_eval2s.json`
  for `model_499.pt` with `num_envs=64`, `steps=300`,
  `trial_length_s=2.0`, `seed=4101305`, and command `lin_vel_x=0.4`.
  The eval execution pipeline itself passed, but final promotion failed:
  `pass=false`, `pipeline_pass=true`, `quality_gate_pass=false`,
  `sequence_aware_update_train_pipeline_pass=false`,
  `failure_reasons=["final_fall_ratio_too_high",
  "final_gravity_xy_too_high", "final_lin_vel_error_too_high",
  "final_yaw_vel_error_too_high", "lin_vel_error_mean_regressed_from_trial0",
  "yaw_vel_error_mean_regressed_from_trial0",
  "train_summary_pipeline_not_passed"]`. Aggregate metrics:
  `fall_ratio=1.0`, `zero_fall_ratio=0.0`,
  `lin_vel_error_mean=0.7373346236844858`,
  `yaw_vel_error_mean=5.194398358662923`,
  `gravity_xy_max=0.9575986266136169`, and
  `root_z_min=0.7207261919975281`. The final trial recorded
  `reset_reason_counts={"1": 1564}` and action mean L2
  `19.13002586364746`.
- 2026-08-11 Route B video diagnostic rendered
  `outputs/task047/videos/task041_scratchlineage_iter500_visual_long_world_seed4102305_env1.mp4`
  with JSON
  `outputs/task047/videos/task041_scratchlineage_iter500_visual_long_world_seed4102305_env1.json`
  and contact sheet
  `outputs/task047/videos/task041_scratchlineage_iter500_visual_long_world_seed4102305_env1_contact.jpg`.
  It used `num_envs=1`, `steps=250`, `trial_length_s=10.0`,
  `seed=4102305`, fixed command `lin_vel_x=0.4`, and produced 251 frames
  over `5.02s`. Video JSON recorded `reset_reason_counts={"1": 62}`,
  `lin_vel_error_mean=0.7624918818473816`,
  `yaw_vel_error_mean=5.257551670074463`,
  `gravity_xy_max=0.9272055625915527`, and
  `root_z_min=0.7435759902000427`. Manual contact-sheet inspection shows
  repeated stand-then-twist/side-fall resets, not a gait.

## Review

Status: local regression plus real local RTX 4090 Task041 training/eval/video
evidence has been reconciled. The earlier 100-iteration scratch-lineage
training run remains the last strict train-pipeline pass, while the 500-iter
Route B continuation produced a checkpoint but failed strict parity and
collapsed into rapid fall resets. `num_mini_batches=4` remains a diagnosed
strict-parity drift route; the longer `num_mini_batches=1` continuation now
also needs investigation before being promoted as a parity-safe long-training
route.

Task047 still cannot be marked as a full quality/reproduction pass. The real
Task041 checkpoints run under MJLab/MuJoCo Warp on the local RTX 4090, but the
latest Route B long chunk fails quality eval with `fall_ratio=1.0` and the
manual video shows no usable gait. Continuing another blind scratch chunk from
`model_499.pt` is not recommended until the reward/termination/reset or
initialization failure mode is diagnosed. Task044 continuous and Task046
corrected quality/eval claims still require matching checkpoint evidence. No
standing, memory-causality, reproduction, deployment, or superiority claim is
promoted.
