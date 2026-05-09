# Route

Task: task011-genesis-official-batched-api-decision

Goal: validate the H200 + Genesis official MJCF batched/tensor API baseline
using the Franka asset before testing locomotion assets.

Scope:

- Use the prepared official Franka asset.
- Build with `n_envs=1,16,256,1024` subject to stop rules.
- Use default asset/morph parameters first.
- Verify tensor action write, tensor state read, and selected joint reset.
- Record build and steady-state metrics separately.
- Record GPU/memory snapshots before and after benchmark sections.

Environment:

- interactive remote sessions must enter `/root/agent_workspace/safe_agent/agent_shell.sh`
- non-interactive remote commands must run through `/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/<project> && <command>'`
- benchmark commands must prefer physical GPU 1 with `CUDA_VISIBLE_DEVICES=1`
- record that physical GPU 1 maps to logical `cuda:0` inside the process
- record `nvidia-smi` GPU/memory snapshot before and after the benchmark section
- code, generated files, and intermediate outputs must stay under `/root/agent_workspace/project`
- datasets must be accessed only through `/root/agent_workspace/datasets`
- dataset links must be created with `/root/agent_workspace/safe_agent/link_dataset.sh`
- do not write to or delete from `/mnt/workspace` or `/mnt/workspace1`
- do not manually clear `LD_PRELOAD`

Verify:

- `status=ok|blocked|failed` for each attempted `n_envs`.
- `cuda_visible_devices=1`, `physical_gpu=1`, and `logical_cuda_device=cuda:0`
  are recorded or a GPU isolation blocker is recorded.
- `action_device`, `qpos_device`, and DOF state devices are recorded.
- `tensor_device_ok=true` is required for pass.
- Selected joint reset changes only target envs.
- Throughput metrics use the task-level definitions.

No Hack:

- Do not use render/GIF/SONIC/ONNX/planner.
- Do not enable `convexify`, `decimate`, or `performance_mode` before the
  default baseline.
- Do not continue to Go2 if Franka fails by stop rule.

Hardware: H200/Linux target via `myserver`.

# Log

- 2026-05-09 local: Added independent probe tool
  `h200_locomotion_lab.tools.genesis_official_batched_api_probe`. It does not
  import or use `GenesisG1SceneBackend`; local focused verification:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_genesis_official_batched_api_probe.py -q -p no:cacheprovider`
  -> 16 passed after reviewer fixes.
- 2026-05-09 H200 guarded focused verification in remote project
  `/root/agent_workspace/project/h200-locomotion-lab-task011-genesis-official-batched-api-decision`:
  `PYTHONPATH=src python -m pytest tests/test_genesis_official_batched_api_probe.py -q -p no:cacheprovider`
  -> 16 passed after reviewer fixes.
- Environment evidence saved under `outputs/task011/`:
  `env_versions.txt` reports `genesis-world==0.4.6` and `torch==2.5.1`;
  `gpu_inventory.txt` reports two NVIDIA H200 GPUs with 143771 MiB each.
  Benchmark runs used `CUDA_VISIBLE_DEVICES=1`, so physical GPU 1 was exposed
  as logical `cuda:0`.
- H200 guarded benchmark command shape:
  `/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/h200-locomotion-lab-task011-genesis-official-batched-api-decision && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src timeout ... python -m h200_locomotion_lab.tools.genesis_official_batched_api_probe --asset-kind franka --asset /root/agent_workspace/project/genesis_assets/genesis-world-0.4.6/genesis/assets/xml/franka_emika_panda/panda.xml --n-envs ... --warmup-policy-steps 20 --measure-policy-steps 100 --decimation 4'`
- Default Franka output files:
  `outputs/task011/franka/default_n1.txt`,
  `outputs/task011/franka/default_n16.txt`,
  `outputs/task011/franka/default_n256.txt`,
  `outputs/task011/franka/default_n1024.txt`.

| n_envs | status | build_time_s | measure_time_s | policy_steps_per_sec | env_policy_steps_per_sec | env_sim_steps_per_sec | tensor_device_ok | selected_reset_changes_only_target_envs |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | ok | 10.328132 | 0.432988 | 230.953171 | 230.953171 | 923.812686 | true | false (not applicable for one env) |
| 16 | ok | 52.939369 | 0.452470 | 221.009161 | 3536.146575 | 14144.586300 | true | true |
| 256 | ok | 20.996516 | 0.418496 | 238.951166 | 61171.498607 | 244685.994430 | true | true |
| 1024 | ok | 21.236085 | 0.453136 | 220.684411 | 225980.836743 | 903923.346973 | true | true |

- Every default Franka run emitted:
  `cuda_visible_devices=1`, `physical_gpu=1`,
  `logical_cuda_device=cuda:0`, `backend=cuda`,
  `includes_build_time=false`, `includes_state_read=true`,
  `includes_action_write=true`, `includes_reward=false`,
  `includes_render=false`, and all recorded tensor devices as `cuda:0`.
- Selected reset was actually exercised for `n_envs>=16` and changed only the
  target env. The `n_envs=1` run records
  `selected_reset_time_s=not_applicable_n_envs_lt_2` and does not claim
  target-only verification.
- Non-blocking warnings: Genesis emitted `torch<2.8.0` warning under
  `torch==2.5.1`, MJCF tendon approximations for Franka fingers, qpos0 joint
  limit warning, solver time-constant adjustment warning, and neutral
  self-collision filtering warning.
- Result: Franka official MJCF batched/tensor API baseline passed through
  `n_envs=1024`; stop rule permits Go2.

# Review

Status: passed.

Read-only review completed after the probe fixes. No blocking findings remain.
Franka evidence satisfies the subtask requirements: independent tool, guarded
H200 runs, GPU 1 isolation, build/steady-state timing separation, CUDA tensor
device checks, selected joint reset verified for batched runs, and no
render/GIF/SONIC/ONNX/planner/PPO work in the benchmark loop.
