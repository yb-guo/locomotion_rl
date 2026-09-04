# Task065 — Heldout/OOD 最终 Gate

## Route

冻结同一个 checkpoint，评估 heldout topology、doubled dynamics、dynamic fault、
locked/stuck OOD、Berkeley Humanoid、ANYmal C、G1 29DoF 和 Go2。Berkeley/ANYmal
只允许建立 mapping、joint limits 和 nominal pose，不参与训练或 checkpoint 选择。

## Log

- 2026-08-19：排队，依赖 Task064。

## Review

记录 zero-shot/few-shot 指标、median return 和 bootstrap CI；Task048 回归不得
退化。未通过不得进入 Task066 扩规模。
