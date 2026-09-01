# 003 — Fleet asset regeneration

状态：**blocked / waiting_for_002**。

## Route

1. 从冻结 structure、inertial、actuation、visual inputs 与 contact registry 生成新的 versioned
   RobotAsset；不覆盖 Task070/071/072 XML、manifest、stance 或 SHA。
2. 每个 collision primitive 只属于一个 logical terminal；旧大 box 只有 profile 显式选择 `box` 时
   才允许存在。
3. collision geoms 不参与惯性推导；逐 case 验证 body mass、COM、inertia、joint、actuator 与 parent
   contract 相等，除非另有预注册非接触物理 delta。
4. biped、quadruped 与 wheel composition 使用各自 profile；wheel 保留 rolling axis、geometry 和
   velocity/torque semantics。
5. 每个 case 输出 XML、RobotAsset manifest、contact profile、source mapping、compiled audit、contact
   sheet 和互绑 SHA。

## Log

- 2026-08-30：合同已列出；尚未生成资产。

## Review

通过条件：18/18 生成唯一 versioned asset，terminal/primitive mapping 闭合，旧 artifacts 不变，所有
未声明 non-contact physical drift 为零。
