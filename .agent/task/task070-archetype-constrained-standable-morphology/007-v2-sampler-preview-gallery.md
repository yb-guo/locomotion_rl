# 007 — V2 Sampler Preview Gallery

## Route

1. Build a descriptor-driven sampler that first selects a discrete structural
   center, then samples normalized primitive geometry/physical parameters.
2. Keep topology identity separate from continuous geometry identity.
3. Produce a small preview gallery by structural center and sampling region:
   prior neighborhood, interpolation band where graph-compatible, and bounded
   outward band.
4. Include negative gates proving lower-body-only 12DoF biped cannot count as a
   mature G1/PM01 center.

## Log

- 2026-08-25：完成 seed 0 structural-center inspection gallery 的第一部分：Spot/Go2/Lite3
  quadruped，G1/PM01 wheeled-biped，以及三种 wheeled-quadruped，共八张四视图 sheet。每个
  witness 绑定 source descriptor hash 和 exact motor accounting；execution agent 已逐图本地检查，
  aggregate observation 为
  `artifacts/preview_task070_v2_descriptor_driven_attempt003/quadruped_wheel_leg_agent_visual_observation.json`。
- 本轮只修复用户要求的四足/轮腿 visual witness。尚未实现本 microtask Route 1–3 的完整
  prior/interpolation/outward sampler，也尚未补 12DoF biped negative gate；不能把八张 seed-0
  preview 冒充 R4 population gallery。

## Review

- Pass only if all selected centers appear in the gallery with exact motor
  accounting and descriptor hash binding.
- Fail if different DoF graphs are interpolated as one vector, if continuous
  sampling changes motor count, or if success-only samples hide rejected seeds.
- 当前 scoped review：**IN PROGRESS / NOT PASSED**。preview 可交用户视觉验收，但 sampler、
  region gallery 和 negative gate 仍待后续执行。
