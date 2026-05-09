# 002: PPO Core And Math

## Goal

Implement reusable PPO math with a small, testable seam.

## Route

1. Replace placeholder `h200_locomotion_lab.training.ppo_loop` with reusable
   core.
2. Keep torch import lazy:
   - no hard torch import at package/module import time on local Windows;
   - import torch inside torch-required functions or guarded helpers.
3. Add config and result types:
   - PPO config;
   - rollout batch;
   - update diagnostics.
4. Implement:
   - discounted returns;
   - GAE;
   - advantage normalization;
   - clipped policy objective;
   - clipped ratio diagnostics;
   - value loss;
   - entropy;
   - approx KL;
   - gradient clipping.
5. Add tests:
   - GAE known-value case;
   - shape validation;
   - finite loss on tiny fake batch when torch exists;
   - params change after one update when torch exists.

## Stop Rules

- If loss/logprob math requires CPU sync per minibatch, stop and redesign.
- If tanh Gaussian log-prob correction is not finite for near-boundary actions,
  add clamp/epsilon test before H200.
- If local no-torch import fails, do not proceed to H200.

## Verification

- Focused local tests for PPO math.
- Full local pytest.
- H200 focused tests for torch path.

## Log

Pending implementation.

## Review

Status: pending.
