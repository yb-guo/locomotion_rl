# Task061 — RTX 5060 Ti Specialist 正常行走

## Route

在 Task060 的可用本体上，先分别训练一个程序化二足和四足 masked MLP/PPO
specialist。第一步只用固定 topology、窄物理范围和无在线 motor fault；随后
再用同一 specialist 评估 100 条 10 秒 trial。

## Log

- 2026-08-19：Task061 启动；先执行 biped/quadruped 一更新 CUDA smoke，确认
  RTX 5060 Ti 上 PPO、active mask 和 checkpoint 写入闭环。
- 2026-08-19：首轮 CUDA smoke 暴露 MLP actor/value 未随 device 迁移的问题；
  修复 `WholeBodyMLPActorCritic` 构造时的 module `.to(device)`，CPU/GPU 混用
  已消失。
- 2026-08-19：biped 与 quadruped 各完成 1 update × 8 env × 32 rollout steps，
  PPO metrics finite、fall_count=0，并写出 schema-hashed checkpoints：
  `artifacts/biped_smoke.json`、`artifacts/quadruped_smoke.json`。
- 2026-08-19：RTX pilot 各运行 8 updates × 16 env × 64 rollout steps。biped
  pilot 累计 `fall_count=123`，quadruped 累计 `fall_count=15`；两者 metrics
  finite 但尚未达到行走 gate，因此 Task061 继续保持 in-progress。
- 2026-08-19：增加 centered `physics_range_fraction=0.5` 入口，初始阶段关闭
  action delay；增加 grounded nominal reset，按实际 collidable geom 将最低点
  放到 floor margin 上。窄范围 biped pilot 仍有早期跌倒，说明需要继续诊断
  reset pose/reward/control scale，不能提前进入 Task062。
- 2026-08-19：低噪声/低 action-scale ablation 也已执行，metrics finite 但未
  达到 quality gate；对应 JSON 保存在 `artifacts/`。
- 2026-08-19：诊断定位为二足 reset/stance/control contract，而非 2000 个本体
  的编译或数值可用性问题。固定窄范围下，32 个 train-like 二足 seed 中
  `31/32` 在 zero-action、2 秒内跌倒，首次跌倒 control step 为 `23–76`
  （中位数 `38`，约 `0.76s`）；对应四足为 `0/32`。二足 reset 时初始
  `ncon=0`，接近地面的脚通常只有 `0–1` 个，且随机链长造成左右腿不对称。
  grounding margin ablation 没有改变早期跌倒时刻，说明单纯平移最低 geom 不能
  形成静态支撑。当前 nominal position target 在目标姿态附近没有重力补偿，默认
  PD 也偏弱；同时 reward/termination 缺少明确的 base-height、foot-support 和
  terminal-fall shaping。该证据足以阻止进入 Task062，但不应把 pilot 的
  `fall_count` 直接当作最终 100×10s zero-fall 指标。
- 2026-08-19：诊断收敛到 **generator stance contract**。新增两个只读诊断工具
  `tools/whole_body_stance_diagnosis.py` 和 `tools/whole_body_stance_isolation.py`
  （不改 45D action / 193D observation / mask / env-task 接口）。复现命令与证据：

  ```bash
  PYTHONPATH=src python -m h200_locomotion_lab.tools.whole_body_stance_diagnosis \
      --seeds 32 --range-fraction 0.5 --horizon-steps 100 \
      --output-json .agent/task/task061-rtx-specialist-normal-walk/artifacts/stance_diagnosis_32x2_rf05.json
  PYTHONPATH=src python -m h200_locomotion_lab.tools.whole_body_stance_diagnosis \
      --seeds 32 --range-fraction 0.0 --horizon-steps 100 \
      --output-json .agent/task/task061-rtx-specialist-normal-walk/artifacts/stance_diagnosis_32x2_rf00.json
  PYTHONPATH=src python -m h200_locomotion_lab.tools.whole_body_stance_isolation \
      --family biped --seed 557 --range-fraction 0.0 \
      --output-json .agent/task/task061-rtx-specialist-normal-walk/artifacts/stance_isolation_biped_seed557.json
  ```

  - Task060 artifact 自身已经记录了跌倒：`usability_gate_2000x2s.json` 中
    1000 个二足有 `968/1000 (96.8%)` 的 `min_height` 低于 env 自己的跌倒阈值
    `0.35 * nominal_height`，四足为 `36/1000 (3.6%)`；两者仍全部 `passed=true`。
    该 gate 只检查 NaN/爆炸/穿模，因此“可用”与“会走”确实是两回事。
  - `rf=0.5`：二足 `30/32` 零动作跌倒（首次跌倒 step 24–55，中位数 40.5，
    约 0.81s），四足 `0/32`。`rf=0.0`（完全无随机化）二足 `32/32` 跌倒，
    说明不是物理随机化造成的。
  - 支撑多边形：二足 `32/32` 退化（`hull_area=0`，顶点 ≤2），COM 投影
    `0/32` 落在支撑内，中位 margin `-0.111 m`；四足 `0/32` 退化，
    `hull_area` 中位 `0.203 m²`，COM `32/32` 落在支撑内，margin `+0.158 m`。
    二足的“脚”是末端 capsule 的圆端（`_foot` 是 site，不参与碰撞），
    支撑面积恒为 0，静态站立在数学上不可能。
  - reset stance：`ground_nominal_pose` 只对齐单个最低 collidable geom 并留
    `margin=0.015`，因此 reset 时 `ncon` 中位数 0，二足贴地脚数中位数 `1`，
    左右脚底高度差中位 `0.062 m`、最大 `0.158 m`。
  - controller：nominal target 处 `max_abs_ctrl_error=0.0`、
    `max_abs_actuator_force=0.0`，而静态保持所需 `qfrc_bias` 中位 `5.33 N·m`
    （最大 `11.4`）。kp=30 ⇒ 需要中位 `0.188 rad` 下沉才能产生保持力矩。
    力矩能力不是瓶颈：`actuator_saturation_events=0`，
    `seeds_with_actuator_over_force_limit=0`。
  - 隔离阶梯（seed 557，左右镜像、双腿均有 ankle_pitch+ankle_roll，`rf=0`）：
    nominal 0.46s / double-support 0.40s / 接触式 reset 0.38s /
    10×kp+5×kv 0.48s / 精确关节重力补偿 0.30s / 两者同时 0.46s /
    box 脚 0.48s / 展平且脚在髋下 0.52s。全部 2 秒内跌倒，全部无饱和。
    只有对全部腿关节做静态平衡优化 + 有面积的脚才第一次把 COM 送进支撑内
    （margin `+0.011 m`，`hull_area 0.063 m²`），存活时间升到 `1.02s`，仍跌倒。
  - generator 能力分布（1000 个二足）：`42.2%` 双腿都有 ankle；
    `52.6%` 双腿都有 roll 权限（hip_roll 或 ankle_roll）；`22.6%` 至少一条腿
    只有 2 个自由度（hip_pitch+knee_pitch，纯平面腿）；`77.3%` 左右腿关节数
    不相等。四足：`0%` 出现 ≤2 自由度腿，最小 3 自由度。
  - reward masking：零动作二足在跌倒前每步仍拿 `0.733`（上限 1.5 的 49%），
    存活的四足是 `0.800`，差距只有 `9%`，且没有 terminal fall penalty、没有
    base-height / foot-support 项。加上诊断 shaping 后差距扩大到 `33%`
    （0.979 vs 1.303），但两者都不能让二足的任务变得可行。
  - reset 一致性：`reset_max_qpos_delta=0.0`、`reset_max_qvel_delta=0.0`，
    `nan_seeds=0`，`min_joint_limit_margin` 最小 `0.68 rad`。因此
    reset 决定性、数值、关节限位、mask/PPO/device 均不是本次根因。
  - 次要缺陷（已量化，不作为第一优先级）：capsule link 的 `<inertial pos>`
    落在近端关节而不是 capsule 中心，COM 因此系统性偏高；修正后
    COM-到-脚的水平偏移只从 `0.178 m` 变到 `0.161 m`，不是主因。
    `_is_fallen` 用未乘 `global_scale` 的 `nominal_height`，阈值与实际体型
    偏差可达 ±30%。`_foot_geoms` 是整根末端 capsule，小腿擦地不会被记成
    non-foot contact。GAE 把 timeout 也当 terminal。
  - 结论：主要根因是 generator 没有为每个随机本体求解“可静态支撑的
    nominal stance”，也没有保证二足具备站立/行走所需的最小机械结构
    （有面积的脚、ankle、roll 权限、左右对称）。控制器与 reward 都是次要项。
    据此阻止进入 Task062，并新建 `task067-biped-stance-contract`。


## Review

目标 gate：zero-fall ≥0.95、normalized velocity error ≤0.25、non-foot contact
≤0.05、roll/pitch p95 ≤0.45 rad。失败时只诊断 generator、reset、action scale
和 reward，不进入 Task062。

当前状态：CUDA smoke 通过；100×10s specialist quality gate 尚未开始，不能据此
宣称正常行走质量已通过。诊断已完成并归档，主要根因为 generator stance
contract；Task061 阻塞在 `task067-biped-stance-contract` 之后再重跑 pilot。
