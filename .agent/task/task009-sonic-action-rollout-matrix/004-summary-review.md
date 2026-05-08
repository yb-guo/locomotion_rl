# Route

Task: task009-sonic-action-rollout-matrix

Goal: summarize multi-action rollout evidence and remaining limitations.

Scope:

- Compare walking and dance-like/non-walking rollout outcomes.
- Record local copied artifact paths if any.
- State clearly what is still simulator-only versus real robot execution.

Environment:

- Local docs and H200 artifact paths only.

Verify:

- Every earlier subtask has concrete pass/fail evidence.
- Summary distinguishes official-context walking from command-stress tests.

No Hack:

- Do not mark task passed if any required rollout lacks evidence.

Hardware: local + H200 evidence review.

# Log

- 2026-05-08: Summary of completed task009 rollouts:

  Walking official-context route:

```text
GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_OK
FRAMES 40
PLANNER_CALLS 4
REPLAY_OBS_USED False
REPLAY_TOKEN_USED False
ROOT_Z_MIN 0.7183110117912292
HORIZONTAL_DISPLACEMENT 0.4167171290308901
TOTAL_CONTACT_SWITCHES 6
SINGLE_SUPPORT_FRAMES 26
LOCOMOTION_OBSERVED True
```

  Dance-like sidestep/facing-back stress route:

```text
GENESIS_SONIC_PLANNER_ENCODER_ROLLOUT_PROBE_OK
FRAMES 40
PLANNER_CALLS 4
REPLAY_OBS_USED False
REPLAY_TOKEN_USED False
ROOT_Z_MIN 0.7366165518760681
HORIZONTAL_DISPLACEMENT 0.26222350185643367
YAW_DELTA -2.7321533960550517
TOTAL_CONTACT_SWITCHES 9
SINGLE_SUPPORT_FRAMES 31
LOCOMOTION_OBSERVED True
```

  Local visual artifacts:

```text
.agent/task/task009-sonic-action-rollout-matrix/artifacts/walking_officialctx_40f.gif
.agent/task/task009-sonic-action-rollout-matrix/artifacts/walking_officialctx_40f.mp4
.agent/task/task009-sonic-action-rollout-matrix/artifacts/dance_like_sidestep_facing_back_40f.gif
.agent/task/task009-sonic-action-rollout-matrix/artifacts/dance_like_sidestep_facing_back_40f.mp4
```

  Interpretation:

- Both rollouts use online SONIC planner/encoder/decoder, not replayed decoder
  observations or tokens.
- Both rollouts execute through the task008 profile-backed bridge and Genesis
  G1 backend.
- The dance-like case is a command-stress sidestep/facing route. It is not
  evidence of an official SONIC dance skill.
- This is still simulator execution on Genesis, not real Unitree G1 hardware
  motor publishing.

# Review

Status: passed.

- 2026-05-08: Router summary only; earlier subtask reviews passed and no new
  code was added.
