# 006 — Wheel Module Composition

## Route

1. Add terminal wheel modules on top of complete non-wheel descriptors without
   deleting source motors.
2. Verify actuator counts: G1 wheeled biped `29 + 2 = 31`, PM01 wheeled biped
   `23 + 2 = 25`, each quadruped center `12 + 4 = 16`.
3. Derive wheel axis from terminal limb local frame and preserve wheel contact
   cylinder, continuous hinge, and actuator slot.

## Log

- 2026-08-25：实现 v2-only terminal wheel composition。先完整生成 source descriptor 的全部
  non-wheel links/joints/actuators，再按 limb 末端追加 wheel child；source joint/actuator order 保持
  为 blueprint 前缀。
- Exact accounting 已编译验证：G1 `29 + 2 = 31`、PM01 `23 + 2 = 25`、Spot/Go2/Lite3
  各 `12 + 4 = 16`。Manifest 分开记录 source/non-wheel/wheel/total count 和 bijection gate。
- 每个 wheel hinge 使用 terminal child frame 的 local lateral basis `(0, 1, 0)`；该轴继承
  descriptor-preserved source frame，并在 `terminal_wheel_composition.axis_derivation` 留证，
  不是 world-frame attachment。MJCF wheel hinge 为 unlimited continuous，wheel cylinder 为唯一
  contact terminal；视觉复核中移除了会造成“轮+脚球”假双终端的 end-site sphere。
- Attempt003 输出 G1/PM01 wheeled-biped 和三种 wheeled-quadruped 的 XML、descriptor、
  manifest、front/side/oblique/contact PNG 与 sheet。Focused pytest `18 passed`；Ruff passed；
  frozen legacy/Task069 compatibility `256/256` passed。

## Review

- Pass only if wheel augmentation is manifest-visible and source/non-wheel/wheel
  motor accounting is exact.
- Fail if a wheel replaces ankle/knee/foot source motors or if wheel axes are
  hard-coded to a world axis without descriptor-frame evidence.
- 2026-08-25 scoped review：**PASS for wheel-composition/compile/visual-witness microtask**。
  `user_visual_acceptance=false`、`counts_toward_task070_v2_pass=false`，不构成 Task070 pass。
