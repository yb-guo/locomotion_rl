# 006 — Tier C nominal locomotion training

## Route

对通过 005 的八个人形 candidate 逐 case 使用 family-appropriate biped pose/contact reward、同 tuple
推导的 per-slot action scale、flat terrain 与固定 `vx=0.5 m/s, vy=0, yaw=0`，关闭全部 domain
randomization 与 curriculum，并从随机初始化 PPO 训练。

每个 case 必须独立保存训练 progression、paired zero/untrained baselines、20 s evaluation、8 s 视频、
agent visual check 与 verifier；腰链、不同 root placement 和 23/29/更高 DoF 必须通过 canonical-root
runtime regressions。未通过 005 的 case 不允许用 proxy motor config 开训。

## Log

- 2026-08-27：八个 candidate 的 nominal locomotion training 均未执行。

## Code implementation

`task073_pipeline.py nominal` 读取 005 的 local schema、002 binding verifier 与 per-slot tuple；只有
registry state=`runtime_ready` 才构建 `TopologyLocalEmbodiment`、dynamic-dim policy/trainer 和 biped
reward。仍固定 `(0.5,0,0)`、Task072 stage budgets/seeds、无 randomization/curriculum/fault，并按
train -> render -> eval -> verify 顺序执行。

新增 frozen `Task073CandidateBipedRewardConfig`，version 为 `task073_candidate_biped_v1`。它逐项复用
Task072 biped v1 的公式和权重，并增加 `pose_head_hand = -mean((q-q_nominal)^2)`、weight `0.05`；
fall 仍只 termination。semantic resolver 规则固定为：leg module 再按 slot 的唯一
`hip|knee|ankle` token 分组，waist module -> waist，left/right arm -> arm_wrist，
`head|left_hand|right_hand` -> head_hand。每个 active position slot 必须恰好进入一组，未分类、重复或
空的 load-bearing leg group 都拒绝训练。foot/contact 仍只由 blueprint `foot=true` 与 binding
`contact_role=foot` 交集解析。reward config、group membership 和 SHA 写入每个 case
`nominal/reward_config.json`，测试对 23/29/31/43/55 DoF 全覆盖并复算 component sum。

允许的 `--case` 精确为 `agibot_x1_serial, agibot_x2_ultra, engineai_t800,
engineai_t800pro, limx_hu_d04, booster_t1_23, booster_t1_29, robotera_star1`。示例命令：

```bash
TASK_PY=/home/admin1/workspace/proj/locomotion_rl/.venv/bin/python
TASK073=.agent/task/task073-all-configuration-binding-training
env PYTHONPATH="$PWD/src" "$TASK_PY" "$TASK073/task073_pipeline.py" nominal \
  --case agibot_x1_serial
env PYTHONPATH="$PWD/src" "$TASK_PY" -m pytest -q \
  tests/test_task073_pipeline.py -k 'candidate and nominal'
```

每个 case 输出自己的 dynamic schema hash、checkpoint progression、paired baselines、eval、video、agent
visual observation 和 replay verifier；不能共享 checkpoint。source/config unknown、slot count/hash
漂移、root regression、数值/视频 gate 失败都保持非 `nominal_passed`，但 denominator 不删除。

## Review

通过条件：8/8 分别完成 exact-bound nominal proof；任一 candidate 缺 source/config、使用 guessed
motor、发生 slot truncation 或未通过视频/数值 gate，`nominal_18_complete` 均保持 false。
