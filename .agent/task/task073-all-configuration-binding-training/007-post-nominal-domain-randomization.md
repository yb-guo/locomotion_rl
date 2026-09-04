# 007 — Post-nominal domain randomization

## Route

1. 每个 case 只有在自己的 numerical/video/verifier nominal gate 通过后才可进入 randomization；
   未通过的型号不得用随机化或 curriculum “救活”。G1/Go2 可在 Task072 pass 后进入本 subtask，
   但其 Task072 nominal binding 是只读 base，randomization 必须是 versioned overlay。
2. 以 source-bound nominal center 为中心做增量 ablation，固定顺序为：
   - correlated mass + local COM（并保持 inertia/geometry coherence）；
   - contact friction；
   - coherent motor tuple，包括 transmission-side effort/speed/gain/armature/friction/delay。
3. 每次只新增一个声明的 randomization group，记录范围、分布、相关性、seed 与 provenance；不得把
   torque、speed、gear、gain 和 armature 独立乱采成不可能组合。
4. 每阶段重训并同时评估 nominal center 与 randomized holdout；要求 nominal walking 不发生未声明
   regression，并保存 checkpoint、20 s metrics、视频与 verifier。对 G1/Go2，nominal 对照的
   descriptor/physics/motor/action/reward identity 必须与 Task072 freeze 一致；否则先重跑 Task072。
5. runtime weak/dead/latency/lock fault process 不属于本 subtask；它仍是 nominal/randomized motor
   stack 之上的后续任务，不得混入 domain-randomized pass claim。

## Log

- 2026-08-27：所有 Task073 case 均未达到本 subtask 入口；没有 randomization training 已启动。

## Code implementation

在 task-local pipeline 新增 frozen `RandomizationOverlay(version, base_binding_sha256, groups, seed)` 和
`sample_binding()`。一个 `MjModel` 不能给多个 `MjData` 不同物理参数，因此不要在 shared model 上
假装 per-env randomization：每个 sampled binding 编译自己的 `WholeBodyMuJoCoShard`，再用已有
`WholeBodyRolloutMux` 合并 shard batch。

随机化是累计三阶段，每阶段只**新增**一组，v1 范围是本仓库 conservative engineering prior，
不是 named-robot system identification：

1. `mass_com`：mirrored link group 共用 `s_mass ~ U[0.9,1.1]`；mass 和 diagonal inertia 同乘
   `s_mass`，COM 每轴偏移不超过 link length 的 2%，左右 y offset 镜像；
2. `friction`：在 stage-1 上增加每个 contact role 共用 `s_mu ~ U[0.8,1.2]`，MuJoCo friction tuple
   同比例缩放并保持正；
3. `motor`：在 stage-2 上按 coherent motor family 采样 `s_strength ~ U[0.9,1.1]`，effort 与 kp 同乘
   `s_strength`、kd 乘 `sqrt(s_strength)`；另采样 bandwidth `s_bw ~ U[0.95,1.05]`，velocity 乘
   `s_bw`、delay 除以 `s_bw` 并 clamp 到 `[0, 1/control_hz]`；armature 没有 source range 时不随机。

每阶段固定训练 8 个 sampled train shards、每 shard 4 env，由 mux 组成 32 env。`mass_com` 从该 case
nominal selected checkpoint 初始化，`friction` 从通过的 mass_com checkpoint 初始化，`motor` 从通过的
friction checkpoint 初始化；每次都保存 parent checkpoint SHA，不允许跳阶段。每组先跑 2-update smoke，
再跑 1000-update pilot；通过 finite/progression gate 后最多跑 31,200 updates，rollout steps 固定 64、
每 200 updates checkpoint、每 1000 updates evaluation，首次满足 gate 即停止。也就是每阶段上限仍为
63,897,600 transitions，而不是让执行者临时选择预算。

holdout 固定 20 个未见 binding seeds，seed 公式为
`373000 + case_index*100 + sample_index`，每个 seed 跑单环境 20 s deterministic rollout。aggregate 必须
满足 zero-fall ratio `>=0.95`、mean planar error `<=0.35 m/s`、mean yaw error `<=0.35 rad/s`、mean
projected-gravity XY norm `<=0.35`，且至少 19/20 forward displacement 为正。同时重放 exact nominal
center；它必须继续通过原 nominal gate，error metric 允许的恶化为
`max(0.10*abs(nominal_value), 0.02)`，mean forward displacement 下降不得超过 10%，否则 regression
不能晋级。选择 holdout planar-error 中位 seed 与最差 seed，各渲染 8 s 视频并做 agent visual check。
命令：

```bash
TASK_PY=/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
TASK073=.agent/task/task073-all-configuration-binding-training
env PYTHONPATH="$PWD/src" "$TASK_PY" "$TASK073/task073_pipeline.py" randomize \
  --case unitree_g1 --group mass_com
env PYTHONPATH="$PWD/src" "$TASK_PY" -m pytest -q tests/test_task073_randomization.py
```

随后依次运行 `--group friction`、`--group motor`；CLI 自动加载并校验前一 overlay SHA。输出
`randomization/<group>/{overlay.json,train_manifest.json,nominal_eval.json,holdout_eval.json,
video_median.mp4,video_worst.mp4,agent_visual_observation.json,verifier.json}`。前两阶段只记录 evidence，
不改变最终 state；只有 cumulative `motor` 阶段同时通过
nominal non-regression、20-seed holdout、视频与 replay verifier，才以该 verifier SHA 执行
`nominal_passed -> randomized_passed`。无 nominal pass、base SHA 漂移、顺序错误或独立乱采 tuple
时立即停止。

Paired evaluation 的 `sample_index` 固定为 train `0..7`、holdout `8..27`，两集合互斥且边界稳定；
每个累计 randomization group 必须复用同一组 train/holdout index，保证跨 group paired comparison。
seed 公式保持既有约定不变。新增测试要求验证两集合互斥、完整覆盖、顺序稳定，以及各 group index
复用稳定；集合不满足即停止，不得生成 randomization evidence。

## Review

通过条件：18/18 均先有 nominal pass，再有逐阶段、可复现、物理相关且不越过 provenance 的
randomization evidence。只在训练时打开一个全局噪声开关、缺 nominal 对照或把 runtime fault 混入，
均不通过。
