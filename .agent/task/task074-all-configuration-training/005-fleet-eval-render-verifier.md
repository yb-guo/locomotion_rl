# 005 — Fleet evaluation, render, and verifier

状态：**blocked / waiting_for_18_training_runs**。

## Route

1. 对每个 selected checkpoint 做固定命令、多 episode deterministic evaluation，不读取训练汇总代替
   rollout。
2. 每个 case 生成规定时长视频、contact/motion sheet 和人工视觉结论；站立、拖行、倾倒位移、空转或
   persistent double support 不得冒充 locomotion。
3. 独立 verifier 重新加载 checkpoint，并校验 Task073 asset/contact/stance、Task074 config/source、
   transition count 和 eval/video SHA。
4. 生成 18-case denominator matrix；failed/missing case 保持在分母中，不允许只报告成功子集。

## Log

- 2026-08-30：合同已列出；当前没有训练 checkpoint 可评估。

## Review

通过条件：18/18 都有闭合 lineage、实际 rollout 数值、匹配运动模式的视频和独立 verifier pass；否则
Task074 保持 not_passed。
