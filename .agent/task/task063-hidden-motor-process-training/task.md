# Task063 — Hidden Online Motor Process

## Route

在 shared MLP 上加入隐藏 weak/dead/delay/recovery motor process。actor 不接收
motor state 或 failure label；critic/logging 可以接收。先验证逐步 trace，再做
clean-gait degradation 和 dynamic fault recovery。

## Log

- 2026-08-19：排队，依赖 Task062。

## Review

事件分布、onset/duration/recovery 与配置一致；clean gait 下降 ≤10%。不加入
locked/stuck joint 训练，只在 Task065 做 OOD。
