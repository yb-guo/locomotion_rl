# 004: Online Planner Encoder Rollout

## Route

Run:

```text
mjlab state -> SONIC planner -> encoder -> decoder -> ScalarActionBridge
  -> MjlabG1RobotBackend -> mjlab step
```

Reuse task006 planner runner and ONNX wrapper behavior. Render only after
numeric finite/height checks pass.

## Log

- 2026-05-15 Not run. Current H200 workspace does not contain the SONIC ONNX
  artifacts or the C++ planner runner used by task006. The adapter CLI accepts
  these paths through:
  - `--planner`
  - `--planner-runner`
  - `--encoder`
  - `--decoder`
  - `--planner-work-dir`

## Review

Blocked until artifacts are restored or explicitly downloaded.
