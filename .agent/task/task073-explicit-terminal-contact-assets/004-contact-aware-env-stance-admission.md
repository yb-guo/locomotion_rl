# 004 — Contact-aware environment and stance admission

状态：**blocked / waiting_for_003**。

## Route

1. environment 按 logical terminal 聚合 contact、normal force、airtime、touchdown、height、velocity
   与 slip；输出 shape 由 terminal count 决定。
2. non-terminal contact 识别 profile 内全部 geoms；同脚多点接地不能变成多只脚或多次 touchdown。
3. stance/support code 对 sphere、capsule、box、wheel 计算 support points/sole height，再按 terminal
   聚合 support polygon、load 与 center of pressure；每个新 asset 重新求 stance。
4. 对 18 case 逐一执行 compile、reset、finite dynamics、contact grouping、load、stance、no-update
   parameter delta 与最短 one-update smoke；one-update 只验证接线，不声称 locomotion。
5. case 失败时保留证据，不得缩减 denominator 或用旧 stance/checkpoint 绕过。

## Log

- 2026-08-30：合同已列出；尚未运行 admission。

## Review

通过条件：18/18 compiled/reset/stance/no-update/one-update gates 通过，telemetry shape 与
biped/quadruped/wheel contract 一致，无旧 stance/contact identity 混用。
