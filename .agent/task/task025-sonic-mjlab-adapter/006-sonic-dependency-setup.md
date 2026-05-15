# 006: SONIC Dependency Setup

## Route

Declare the minimal Python dependency set needed by this repo's SONIC adapter
code without pulling in official checkpoints, robot assets, upstream repos, or
large simulator stacks.

Keep C++ deployment dependencies separate from Python extras:

- Python encoder/decoder helpers: repo `sonic` extra.
- Planner runner: external C++ binary with its own ONNX Runtime/TensorRT setup.

## Log

- 2026-05-15 Local dependency check:
  - `onnx`: missing
  - `numpy`: installed
  - `torch`: missing
  - `mujoco`: missing
- 2026-05-15 H200 `unitree-rl-mjlab` conda dependency check:
  - `onnx`: installed
  - `numpy`: installed
  - `torch`: installed
  - `mujoco`: installed
  - `onnxruntime`: missing
- 2026-05-15 H200 environment tool check:
  - `uv`: missing
  - `pip`: `/usr/local/bin/pip`
  - `proxychains4`: `/usr/bin/proxychains4`
- 2026-05-15 Added `pyproject.toml` optional extra:

  ```toml
  sonic = [
    "numpy>=1.26",
    "onnx>=1.16",
  ]
  ```

- 2026-05-15 Documented install command:

  ```bash
  python -m pip install -e ".[sonic]"
  ```
- 2026-05-15 Local install attempt with the new command timed out after 304s.
  Follow-up checks showed:
  - local `onnx`: still missing
  - local `numpy`: installed
  - `python -m pip config list`: `no-index=1`
  - no local `onnx` wheel found under `.external_downloads`, `.agent`, or
    `outputs`

## Review

Dependency boundary is now explicit. The current Python adapter path does not
require Python `onnxruntime` because encoder/decoder execution uses
`onnx.reference.ReferenceEvaluator`. The online planner path still requires a
restored planner ONNX artifact and C++ planner runner.

H200 already satisfies the Python-side `sonic` extra, so no remote Python
install was needed. Local Windows remains missing `onnx` until a wheelhouse or
network-enabled pip index is provided.
