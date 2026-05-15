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
- 2026-05-15 Built a local Linux CPython 3.11 wheelhouse for H200 upload:

  ```bash
  python -m pip download \
    --dest .external_downloads/sonic_wheelhouse_linux_cp311 \
    --platform manylinux2014_x86_64 \
    --python-version 311 \
    --implementation cp \
    --abi cp311 \
    --only-binary=:all: \
    --index-url https://pypi.org/simple \
    "numpy>=1.26" \
    "onnx>=1.16"
  ```

  Downloaded wheels:

  - `numpy-2.2.6-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
  - `onnx-1.19.1-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`
  - `ml_dtypes-0.5.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
  - `protobuf-7.34.1-cp310-abi3-manylinux2014_x86_64.whl`
  - `typing_extensions-4.15.0-py3-none-any.whl`

- 2026-05-15 Uploaded the wheelhouse to H200:

  ```text
  /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/sonic_wheelhouse_linux_cp311
  ```

  Remote SHA256:

  ```text
  c09526488c3a9e8b7a23a388d4974b670a9a3dd40c5c8a61db5593ce9b725bab  ml_dtypes-0.5.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  ba10f8411898fc418a521833e014a77d3ca01c15b0c6cdcce6a0d2897e6dbbdf  numpy-2.2.6-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  1839af08ab4a909e4af936b8149c27f8c64b96138981024e251906e0539d8bf9  onnx-1.19.1-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
  8ff40ce8cd688f7265326b38d5a1bed9bfdf5e6723d49961432f83e21d5713e4  protobuf-7.34.1-cp310-abi3-manylinux2014_x86_64.whl
  f0fa19c6845758ab08074a0cfa8b7aecb71c999ca73d62883bc25cc018c4e548  typing_extensions-4.15.0-py3-none-any.whl
  ```

- 2026-05-15 Ran offline install/confirmation in the H200
  `unitree-rl-mjlab` environment:

  ```bash
  PIP_CONFIG_FILE=/dev/null \
  /mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python \
    -m pip install \
    --no-index \
    --find-links /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/sonic_wheelhouse_linux_cp311 \
    "numpy>=1.26" \
    "onnx>=1.16"
  ```

  Result: requirements were already satisfied:

  - `numpy 2.4.4` from the conda env
  - `onnx 1.19.1` from `/usr/local/lib/python3.11/site-packages`
  - `protobuf 6.33.6`
  - `typing_extensions 4.15.0`
  - `ml_dtypes 0.5.3`

- 2026-05-15 Verified H200 imports:

  ```text
  onnx 1.19.1 /usr/local/lib/python3.11/site-packages/onnx/__init__.py
  numpy 2.4.4 /mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/lib/python3.11/site-packages/numpy/__init__.py
  ReferenceEvaluator ReferenceEvaluator
  ```

## Review

Dependency boundary is now explicit. The current Python adapter path does not
require Python `onnxruntime` because encoder/decoder execution uses
`onnx.reference.ReferenceEvaluator`. The online planner path still requires a
restored planner ONNX artifact and C++ planner runner.

H200 already satisfies the Python-side `sonic` extra. The uploaded wheelhouse is
now available as a repeatable offline fallback for this exact Python 3.11 Linux
target. Local Windows remains missing `onnx` because the uploaded wheelhouse is
Linux-only.

The C++ planner runner dependency was later prepared under task025 using the
official ONNX Runtime Linux x64 1.19.2 tarball. That makes the planner binary
buildable on H200, but it does not provide SONIC model artifacts.
