# 002 Overhead Harness

## Route

Measure whether the shared history path is cheap enough before relying on it.

Benchmarks:

- baseline current MLP without history;
- buffer-only with no policy input change;
- StackMLP `K=4`;
- StackMLP `K=8`;
- GRU smoke;
- LocoFormer-style token smoke.

Metrics:

- env count;
- horizon/steps;
- policy steps per second;
- env steps per second;
- policy forward time if measurable;
- GPU memory before/after;
- actor input dim or token count;
- history length;
- JSON path for each run.

Summary JSON schema:

- `summary_json_path`
- `repo_commit`
- `h200_checkout`
- `host`
- `gpu_name`
- `env_id`
- `checkpoint`
- `seed`
- `benchmark_modes`
- per mode:
  - `mode`
  - `env_count`
  - `horizon_steps`
  - `warmup_steps`
  - `repeat_count`
  - `history_len`
  - `actor_frame_dim`
  - `actor_input_dim` or `token_count`
  - `action_dim`
  - `env_steps_per_s`
  - `policy_steps_per_s`
  - `policy_forward_ms_mean`
  - `policy_forward_ms_p95`
  - `gpu_mem_before_mb`
  - `gpu_mem_peak_mb`
  - `gpu_mem_after_mb`
  - `overhead_pct_vs_baseline`
  - `same_env_scheduler_eval_semantics`
  - `no_fault_label_leak`
  - `json_path`
  - `pass`
  - `failed_gates`

Initial overhead gates:

- buffer-only overhead target: `<= 5%`;
- StackMLP `K=4` target: `<= 15%`;
- StackMLP `K=8` target: `<= 25%`;
- GRU/LocoFormer-style are exploratory and must report cost, not pass/fail
  purely on overhead.

Measurement order:

1. Run buffer-only micro-benchmark first with the Task033 CLI to prove the
   shared ring buffer is cheap and GPU-resident.
2. Run baseline MLP denominator in the MJLab adapter with the same env count,
   horizon, GPU, and checkpoint intended for StackMLP.
3. Run StackMLP `K=4`, then `K=8` only if `K=4` is within gate.
4. Run GRU and LocoFormer-style as smoke/cost reports after their consumers
   exist.

## Log

- 2026-05-28 Planned. This benchmark must run before full training.
- 2026-05-28 Added the buffer-only micro-benchmark path to
  `task033_history_buffer_smoke.py` via `--benchmark-steps`.
- 2026-05-28 H200 buffer-only micro-benchmark passed on `NVIDIA H20D`:
  8192 envs, `history_len=4`, `actor_frame_dim=135`, stack input dim `540`,
  `benchmark_steps=200`, `elapsed_s=0.020585347898304462`,
  `history_frames_per_sec=79590590.74901274`, and
  `cuda_memory_allocated_bytes=79691776`.
  Evidence JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task033/history_buffer_smoke/task033_history_buffer_cuda_smoke.json`.
- 2026-05-28 H200 MJLab env8192 one-iteration overhead smokes:
  - baseline MLP:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_history_overhead/2026-05-28_11-50-01_033_baseline_mlp_env8192_iter1_gpu0_seed3303310`,
    `11908` steps/s.
  - buffer-only K4:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_history_overhead/2026-05-28_11-50-59_033_bufferonly_k4_env8192_iter1_gpu0_seed3303311`,
    `11680` steps/s, `1.9%` overhead vs baseline, within the `5%` gate.
  - StackMLP K4:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_history_overhead/2026-05-28_11-51-58_033_stackmlp_k4_env8192_iter1_gpu0_seed3303312`,
    actor input dim `540`, `10805` steps/s, `9.3%` overhead vs baseline,
    within the `15%` gate.
  - StackMLP K8:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_history_overhead/2026-05-28_11-53-15_033_stackmlp_k8_env8192_iter1_gpu0_seed3303313`,
    actor input dim `1080`, `11596` steps/s, `2.6%` overhead vs baseline,
    within the `25%` gate.
  - GRU K4:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_history_overhead/2026-05-28_12-47-40_033_gru_k4_env8192_iter1_gpu1_seed3303314`,
    actor history input dim `540`, recurrent hidden dim `256`, `52312`
    steps/s. This is a construction/cost smoke, not a quality claim.
  - token K4:
    `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_history_overhead/2026-05-28_12-48-44_033_token_k4_env8192_iter1_gpu1_seed3303315`,
    actor history input dim `540`, token projection `Linear(135, 128)`,
    `56745` steps/s. This is a construction/cost smoke, not a quality claim.
  These are initial one-iteration overhead smokes, not statistically robust
  throughput benchmarks.

## Review

Status: initial overhead gate passed for buffer-only K4 and StackMLP K4/K8.
GRU K4 and token K4 also have env8192 construction/cost smokes. This subtask
does not prove trained GRU/token policy quality.
