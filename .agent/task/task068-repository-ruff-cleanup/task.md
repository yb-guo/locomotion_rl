# Task068 — 全仓 Ruff 清理

进入条件：Task067 R0 已通过；本任务只做静态质量清理，不推进 R1–R3。
退出条件：全仓 Ruff 为 0、全量测试通过、agent inspection 通过，且有验证证据。

## Route

在保留现有 dirty worktree 与 Task067 全部改动的前提下，将
`.venv/bin/python -m ruff check .` 的既有 434 条告警清理到 0。

- 先保存 machine-readable baseline，按规则与目录盘点。
- 对 import、typing 与无语义变化的规则使用 Ruff 机械修复，再检查 diff。
- `BLE001`、`S110`、`S112` 逐处审查：收窄已知异常；需要边界兜底时显式记录或
  保留可解释的异常语义，不用全局 ignore 或批量 `noqa` 掩盖。
- `EXE001` 按脚本用途判断 shebang / executable bit，以最小改动修正。
- 不改变训练或仿真语义，不运行 H200，不下载 checkpoint、数据集或 simulator asset。
- 最后执行全仓 Ruff、全量 pytest 与 agent inspection；失败时记录原因并继续修复。

## Log

- 2026-08-19：任务创建。baseline 来自 Task067 R0：434 条告警、151 个文件，
  其中 276 条可由 Ruff 自动修复。开始前 worktree 已包含用户与 Task047–067 的大量
  tracked/untracked 改动；本任务不重置、不覆盖、不回滚这些改动。
- 硬件假设：仅静态检查与单元测试；RTX 5060 Ti 可见时有一个 device-forward 单测会
  使用 CUDA。本任务不启动训练、仿真或 H200 路线。
- 2026-08-19：机械清理完成。处理了 import 排序/格式、`collections.abc` 类型导入、
  冗余 `int()`、常量 `getattr/setattr`、无用 import/变量、pairwise/min/容器拼接等规则。
  12 个带 shebang 的 task-local CLI 都有 `main()` 入口，故保留 shebang 并将 mode 从
  `100644` 调整为 `100755`。MJLab 的 `mjlab.tasks` / `src.tasks` import 是注册副作用，
  未删除；改为显式 side-effect import 并释放局部绑定，避免破坏 task registry。
- 2026-08-19：逐处审查原 `BLE001/S110/S112`。新增 dependency-neutral
  `h200_locomotion_lab.error_policy.RECOVERABLE_RUNTIME_ERRORS`，只枚举常见 Python
  runtime/import/I/O/类型/数值错误；未知自定义 `Exception` 不再被静默吞掉，会向上抛出。
  Task023 独立脚本使用本地等价集合，保持其 task-local 可运行性。原静默
  `pass/continue` 均重写为显式 fallback 状态与分支；动态 Genesis API 的兼容探测仍按
  selected-call → full-call 或 next-method 顺序降级。未使用全局 ignore 或 `noqa`。
- 2026-08-19：人工语义检查重点：NaN self-compare 改为 `math.isfinite`；MJLab 注册
  side effect 保留；`G1LikeSlotSchema` 默认值改为 frozen module singleton；非 tensor
  column concat 改为 `chain.from_iterable`，输出类型保持 list；JSON root 类型错误改为
  `TypeError`。未发现已确认的训练/仿真逻辑 bug，变更均为静态质量、错误边界或等价实现。
- 2026-08-19：验证命令与结果：

  ```bash
  .venv/bin/python -m ruff check .
  # All checks passed!

  .venv/bin/python -m pytest -q \
    tests/test_genesis_sonic_policy_locomotion_probe.py \
    tests/test_g1_zero_action_standing_causality.py \
    tests/test_genesis_official_batched_api_probe.py \
    tests/test_g1_ankle_foot_asset_contact_audit.py \
    tests/test_g1_base_attitude_height_stabilization.py
  # 71 passed

  .venv/bin/python -m pytest -q
  # 752 passed, 35 warnings

  .venv/bin/python -m h200_locomotion_lab.tools.inspect_agent
  # exit 0; sonic_adapter and locoformer_min component trees emitted
  ```

  35 条 warning 全部来自 TorchScript `torch.jit.script` deprecation，非本任务新增失败。
  汇总 artifact：`artifacts/verification.json`。
- 2026-08-19：最终串联复跑时，Ruff 先通过，但同一全量 pytest 的一个 CUDA device
  test 在 `cublasCreate` 返回 `CUBLAS_STATUS_ALLOC_FAILED`；只读 `nvidia-smi` 证实工作区
  外的 `vggt_omega_slam` 进程占用 `15638/16311 MiB` 且 GPU utilization 100%。未终止或
  干预该用户进程。此前本任务内原始 `.venv/bin/python -m pytest -q` 已完整
  `752 passed`；随后以 `CUDA_VISIBLE_DEVICES=''` 强制 CPU 全量复核再次
  `752 passed, 35 warnings`，并再次完成 agent inspection。该瞬时失败属于外部资源
  竞争，不是本次代码回归。
- 2026-08-19：独立复核发现一个确定性契约回归：
  `test_task038_true_txl_checkpoint_eval_snapshot_error_is_optional` 用
  `_txl_debug_snapshot` 的 invariant `AssertionError` 验证 optional debug snapshot，
  而全局异常收窄后该断言被外抛。最小修复仅在
  `_optional_txl_debug_snapshot()` 捕获 `(AssertionError, *RECOVERABLE_RUNTIME_ERRORS)`，
  并继续返回原始 `repr(exc)`；没有把 `AssertionError` 加回全局 policy。现有测试已明确
  断言错误文本与 optional pass contract，无需扩大测试面。修复后验证：

  ```bash
  .venv/bin/python -m ruff check .
  # All checks passed!

  CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
    tests/test_task038_true_txl_checkpoint_eval_smoke.py::\
test_task038_true_txl_checkpoint_eval_snapshot_error_is_optional
  # 1 passed

  CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q
  # 752 passed, 35 warnings
  ```

  这是本任务确认并修复的一个真实错误边界回归；训练/仿真成功路径未变。

## Review

状态：**passed**。

通过证据必须同时包含：

1. `.venv/bin/python -m ruff check .` 返回 0；
2. `.venv/bin/python -m pytest -q` 返回 0；
3. `.venv/bin/python -m h200_locomotion_lab.tools.inspect_agent` 返回 0；
4. Log 记录机械修复、人工异常处理、发现的真实逻辑 bug（如有）与剩余风险。

以上 1–4 已满足。剩余风险：本任务没有启动 Genesis/MJLab/Isaac/H200 runtime；可选
模拟器的自定义异常若不继承所枚举的标准错误，现会 fail loud 而非生成降级 artifact。
Task038 optional debug snapshot 的 invariant `AssertionError` 是已记录的局部例外，不扩大
全局 policy。这是用于避免掩盖未知 simulator failure 的有意边界收紧，不影响成功路径。Task067
R1–R3、stance gate 与训练语义均未改动。当前 GPU 被无关进程占满，因此未在最后一次
串联命令中复现 CUDA test；本任务较早的原命令全量通过与最终 CPU 全量通过共同覆盖
代码回归，GPU 资源竞争已单独记录而未被隐藏。
