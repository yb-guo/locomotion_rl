# 010 MJLab Runner Smoke

## Route

Close only the minimum runner wiring loop for the Task038 external MJLab XML
variant tasks. This slice extends the `009` train and held-out XML env cfg
helpers into separate runner-smoke task ids:

- `Unitree-G1-Gripper-Flat-Task038-TrainRunnerSmoke`;
- `Unitree-G1-Gripper-Flat-Task038-HeldoutRunnerSmoke`.

The runner-smoke task ids use `Task037TxlMemoryK160DeterministicRunner` only to
prove runner construction, wrapper insertion, one inference-policy forward pass,
and short zero-action stepping over the Task038 train/held-out XML variants.

This slice does not train, evaluate quality, compare baselines, or make a
reproduction claim. Probe JSON must keep `quality_claim:false`,
`training_claim:false`, and `eval_claim:false`.

## Minimal Closed Loop

Local closed loop:

- patcher MemoryPath tests prove the two runner-smoke task ids and runner import
  are inserted exactly once;
- CLI unit tests prove defaults, pass-gate requirements, fake finite obs/action
  summaries, optional inner/outer reset gating, and claim-boundary fields without
  importing MJLab;
- docs tests prove this slice does not claim LocoFormer reproduction, policy
  quality, training, eval, or true TXL reproduction success.

H200 closed loop, still pending:

- run the train runner-smoke probe against external MJLab and write JSON
  evidence;
- run the held-out runner-smoke probe against external MJLab and write JSON
  evidence;
- each JSON must show expected runner class, `action_dim=31`,
  `total_action_dim=31`, finite policy action with shape
  `[actual_num_envs, 31]`, finite non-empty obs summary, no missing step extras,
  and successful zero-action steps.

## Evidence Gate

Positive pass requires:

- registered runner class equals `Task037TxlMemoryK160DeterministicRunner`;
- `action_dim=31` and `total_action_dim=31`;
- `runner.get_inference_policy(...)` is available and one policy forward returns
  finite actions with recorded shape `[actual_num_envs, expected_action_dim]`;
- `runner.env.reset()` and at least one zero-action step through `runner.env`
  complete, with `steps > 0` and `step_count > 0`;
- required Task037 multi-trial extras are seen on at least one step, not only in
  reset extras;
- done/extras consistency is checked on a step where `episode_done` is present;
- obs summary is non-empty and finite, including actor/critic/actor_history
  leaves when MJLab exposes them;
- no quality, training, or eval claim flags are set.

Inner/outer reset evidence is recorded when seen, but is not required unless the
probe is run with `--require-inner-outer-reset`. The default runner smoke may be
too short to force both reset modes.

## Subagent Ownership

Worker `Task038/010` owns only:

- this `010` doc;
- the narrow `task.md` status append;
- the Task038 external MJLab patcher extension for runner-smoke task ids;
- `src/h200_locomotion_lab/tools/task038_mjlab_runner_smoke_probe.py`;
- `tests/test_task038_mjlab_runner_smoke.py`.

Do not touch `.test_tmp_task021/`. Do not change unrelated Task037/Task038
training, eval, policy, or asset-generation code.

## Failure Exit

If runner construction fails on H200, record the failure JSON and stop this
route. Do not continue into training, checkpoint loading, quality eval, or
reproduction claims. If Isaac/RTX/Vulkan/Kit becomes involved, stop and keep this
MJLab route scoped to MuJoCo/MJLab runner smoke.

## Log

- 2026-05-29 Added local Task038/010 runner-smoke wiring: two external MJLab
  task ids, a no-training runner smoke probe, local MemoryPath/pass-gate tests,
  and this evidence contract. Status remains pending H200 runner smoke evidence.
- 2026-05-29 Reviewer found two false-pass risks before H200: finite but wrong
  policy action shapes could pass, and reset-only/no-step runs could satisfy the
  extras gate. The probe now requires `policy_action_shape` to equal
  `[actual_num_envs, expected_action_dim]`, `steps > 0`, `step_count > 0`,
  required extras observed on step extras, and a step-level
  `done`/`episode_done` consistency check.
- 2026-05-29 Verification after reviewer fixes:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider --basetemp .test_tmp_task038_010_verify2 tests\test_task038_mjlab_runner_smoke.py tests\test_task038_mjlab_variant_env_load.py`
  -> `23 passed in 0.06s`.
- 2026-05-29 H200 external MJLab patch application completed against
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab`
  after adding the runner-smoke task ids.
- 2026-05-29 H200 train runner-smoke probe passed on `cuda:0`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/runner_smoke/train_runner_smoke_env8.json`.
  Evidence: `pass=true`, `runner_cls=Task037TxlMemoryK160DeterministicRunner`,
  `action_dim=31`, `total_action_dim=31`, `policy_action_shape=[8,31]`,
  `policy_action_finite=true`, `step_count=8`,
  `step_required_extras_missing=[]`, `done_episode_consistency_checked=true`,
  `obs.actor.shape=[8,104]`, `obs.actor_history.shape=[8,21600]`,
  `obs.critic.shape=[8,119]`, and all obs leaves finite.
- 2026-05-29 H200 held-out runner-smoke probe passed on `cuda:0`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/runner_smoke/heldout_runner_smoke_env8.json`.
  Evidence: `pass=true`, `runner_cls=Task037TxlMemoryK160DeterministicRunner`,
  `action_dim=31`, `total_action_dim=31`, `policy_action_shape=[8,31]`,
  `policy_action_finite=true`, `step_count=8`,
  `step_required_extras_missing=[]`, `done_episode_consistency_checked=true`,
  `obs.actor.shape=[8,104]`, `obs.actor_history.shape=[8,21600]`,
  `obs.critic.shape=[8,119]`, and all obs leaves finite.

## Review

Status: closed for the `010` runner-smoke-only slice.

Final reviewer verdict:

- H200 train and held-out JSON evidence satisfies the `010` gate.
- No more `010` H200 evidence is needed.
- The evidence closes runner construction, one policy forward, and short
  zero-action stepping only.

This is not true TXL reproduction, LocoFormer reproduction, policy-quality
validation, training, or evaluation.
