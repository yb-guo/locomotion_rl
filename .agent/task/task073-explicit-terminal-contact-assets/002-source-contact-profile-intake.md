# 002 — Source contact-profile intake

状态：**blocked / waiting_for_001**。

## Route

1. 对 18-case denominator 建立 contact-source registry，记录本地路径、source commit/file SHA、许可
   边界、logical terminal mapping 和可用字段。
2. source-backed profile 读取编译后有效 geom type/pose/size、condim、priority、friction、contact bits
   与 reference site；匿名化时记录完整 local-frame transform 与 scale。
3. 不复制 mesh、texture、logo 或 vendor body identity；contact primitive 使用匿名名称。
4. source evidence 不完整时不得套用 G1 7-capsule 并称为 aligned，只能保持 blocked，或使用预先声明
   的 normalized generated prior 并标记 `source_contact_aligned=false`。
5. 不下载 repo、asset、checkpoint 或 dataset；现有本地 source 不足时记录 blocker。

## Log

- 2026-08-30：合同已列出；尚未执行 18-case intake。

## Review

通过条件：18/18 均有唯一且互斥的 `exact source / generated prior / blocked` decision，未知来源不得
静默进入生成器。
