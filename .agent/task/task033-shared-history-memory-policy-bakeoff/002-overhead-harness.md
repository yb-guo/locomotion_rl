# 002 Overhead Harness

## Route

Measure whether the shared history path is cheap enough before relying on it.

Benchmarks:

- baseline current MLP without history;
- buffer-only with no policy input change;
- StackMLP `K=4`;
- StackMLP `K=8`;
- GRU smoke;
- LocoFormer-style token smoke.

Metrics:

- env count;
- horizon/steps;
- policy steps per second;
- env steps per second;
- policy forward time if measurable;
- GPU memory before/after;
- actor input dim or token count;
- history length;
- JSON path for each run.

Initial overhead gates:

- buffer-only overhead target: `<= 5%`;
- StackMLP `K=4` target: `<= 15%`;
- StackMLP `K=8` target: `<= 25%`;
- GRU/LocoFormer-style are exploratory and must report cost, not pass/fail
  purely on overhead.

## Log

- 2026-05-28 Planned. This benchmark must run before full training.

## Review

Status: planned. Do not train all consumers until buffer-only overhead is known.
