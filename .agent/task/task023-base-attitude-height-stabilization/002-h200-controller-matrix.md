# Subtask 002: H200 Controller Matrix

## Route

- Run only after subtask001 passes local tests and read-only review.
- Use guarded H200 commands only.
- Use physical GPU 1 with logical `cuda:0`.
- No PPO.

## H200 Command Shape

```bash
/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/h200-locomotion-lab-task023-base-attitude-height-stabilization && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src <command>'
```

## Required Matrix

- source asset, no stabilizer;
- source asset, attitude-only;
- source asset, height-only;
- source asset, attitude+height;
- regenerated `ankle_roll_larger_spheres`, no stabilizer;
- regenerated `ankle_roll_larger_spheres`, best bounded stabilizer candidate.

## Evidence Required

- first tilt/reset step;
- root height and upright timeline;
- ankle-roll and ankle-pitch contact force summary;
- top joint error summary;
- clipping/saturation summary;
- output paths under `/root/agent_workspace/project`.

## Stop Rules

- If baseline reproduction differs materially from task022, stop and diagnose.
- If a candidate improves by at least 20 policy steps, rerun once.
- If all candidates fail near task022 horizons, stop and record controller
  insufficient.
- If contact force exceeds task022 box-support levels without stability gain,
  stop that candidate.

## Log

- 2026-05-13 Created with task023.

## Review

Status: pending.
