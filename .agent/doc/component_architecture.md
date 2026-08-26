# Component Architecture

本文件记录 task / policy / algorithm 的长期边界。它是新实验的规范；旧的
`envs/g1_velocity_tracking_env.py` 已成为 task compatibility shim，
`training/ppo_loop.py` 暂时保留 rollout/diagnostic orchestration compatibility，
不代表新代码可以继续沿用旧耦合。

## Decision

从一个有限时域 MDP 出发：

```text
task = (observation, action, transition, reward, reset, termination, metrics)
policy = action generator pi_theta(action | observation, state)
algorithm = rule that changes trainable state from data
experiment = one explicit composition of task + policy + algorithm + runtime
```

物理 backend 只推进动力学并执行 reset/actuation；reset 的时机和分布属于
task。训练设备、并行环境数、随机种子和训练预算属于 experiment。这样每一条
边界都来自一个独立变化轴，而不是来自文件大小或框架命名。

## Minimal Dependency Rule

允许的依赖方向是：

```text
                        +--> tasks ------+
core contracts --------+--> policies ---+--> experiments
                        +--> algorithms -+

robots --> envs/backends --> tasks
```

- `core` 不依赖 Torch、simulator、robot、PPO 或任何项目层模块。
- Shared numerical kernels such as active-mask distribution math live in the
  neutral top-level module `masked_distribution`, not in `core`.
- `tasks` 可以依赖 `core`、robot 和 backend，不能依赖 policy 或 algorithm。
- `policies` 只实现动作生成，不 import concrete task/backend/algorithm。
- `algorithms` 只实现数据收集/更新规则，不 import concrete task/policy。
- `experiments` 是唯一允许知道所有组件的 composition root。

依赖限制由 `tests/test_component_architecture.py` 的 AST guard 执行，而不是只靠
约定。

## Ownership

| Component | Owns | Must not own |
| --- | --- | --- |
| Task | observation roles, action semantics, reward, reset, termination, task metrics | optimizer, PPO/flow loss, neural-network shape, CUDA budget |
| Policy | network/generative process, recurrent state, action sampling, declared capabilities | reward terms, simulator reset, rollout/update schedule |
| Algorithm | replay/rollout rules, targets/advantages, losses, optimizer update, algorithm metrics | robot constants, observation construction, reward decomposition |
| Experiment | component references, backend/device, `num_envs`, seed, run budget, logging | new task or algorithm semantics |

Observation and action dimensions have one authority: the task. A policy builder receives
those spaces when the experiment is composed; policy YAML must not copy `obs_dim` or
`action_dim`. Compatibility is checked through capabilities, not names. For example, PPO
requires `sample + log_prob + value`; a policy called `flow` is neither accepted nor rejected
because of its name.

## Runtime Data Boundary

`TaskStep` is the output of a task step. `TransitionBatch` is the only generic interaction
record passed to a learning algorithm:

```text
observation, action, reward, next_observation,
terminated, truncated, policy_info, task_metrics, final_observation
```

The generic interaction loop does not know reward component names, fall thresholds, gait
phases, PPO losses, or diffusion losses. Task metrics remain namespaced data; algorithm
metrics are returned separately in `UpdateReport`.

Auto-reset tasks must preserve `final_observation` when truncation bootstrapping needs it.
Policy-specific training tensors such as old log probability or flow time live in
`policy_info`; the task never produces them.

## MIP / Flow / Diffusion / JiT Placement

MIP、flow matching、diffusion，以及“`x0` 相比噪声位于更低维流形”这一 JiT
假设，描述的是动作分布如何参数化和采样，因此放在 `policies/`，不是 task。
它是否能由某个 RL algorithm 更新取决于能力接口：

- 有可用的 `log_prob` 与 value estimate 时，可以与 PPO 类 on-policy 更新组合；
- 只有 `sample` 或 flow/denoise training objective 时，应配 advantage/Q-weighted
  regression 或相应的 flow-policy update，不应为了复用 PPO 假造 likelihood；
- imitation/flow-matching 预训练和 RL fine-tuning 是两个 algorithm 配置，可以复用
  同一个 policy family 与同一个 task。

这使“换 MLP 为 MIP/JiT policy”和“换 PPO 为 flow-aware RL update”成为两个可独立
验证的实验变量。

## Configuration Boundary

Canonical config tree:

```text
configs/
  tasks/          MDP contract and task parameters
  policies/       model/generative-policy parameters
  algorithms/     learning-rule parameters
  experiments/    references plus runtime and run budget
```

The loader rejects missing/unknown structural keys and component paths escaping the config
root. It has no global registry and no implicit name lookup: an experiment names three YAML
paths explicitly.

## Occam Constraints

当前不引入以下机制：

- global registry、plugin manager、DI container 或新的 Hydra layer；
- task-specific trainer class，例如 `G1PPOTrainer`；
- algorithm-specific task class，例如 `PPOG1Env`；
- 为每种 policy 重复 observation/action dimension；
- 为目录整洁而一次性重写全部历史实验。

Python `Protocol`、frozen specs、一个 strict loader 和一个 composition loop 足以表达
当前边界。只有出现第二个真实用例且现有契约无法表达时，才增加新抽象。

## Legacy Migration

迁移采用逐实验替换，不做 big-bang rewrite：

| Legacy path | Destination/role |
| --- | --- |
| `envs/g1_velocity_tracking_env.py` | compatibility shim；实现已移到 `tasks/g1_velocity_tracking.py` |
| `training/ppo_loop.py` | GAE/PPO update 已移到 `algorithms/ppo.py`；legacy collector 尚待按实验迁移 |
| legacy tanh-Gaussian builder | 实现已移到 `policies/tanh_gaussian_actor_critic.py` |
| `agents/` | 其余 policy implementations/adapters 按实验移到 `policies/` |
| task-specific CLI orchestration | `experiments/` composition entrypoints |

迁移一个实验的完成标准是：组件可独立加载、composition validation 通过、最小
interaction/update smoke 通过、旧回归不退化。仅新增目录或 re-export 不算完成。
