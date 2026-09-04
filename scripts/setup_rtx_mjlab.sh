#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNITREE_MJLAB_ROOT="${UNITREE_MJLAB_ROOT:-${REPO_ROOT}/.external/unitree_rl_mjlab}"
VENV_ROOT="${RTX_MJLAB_VENV_ROOT:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${VENV_ROOT}/bin/python"
CONSTRAINTS="${REPO_ROOT}/configs/requirements/rtx5060ti-mjlab-constraints.txt"
EXPECTED_UNITREE_MJLAB_REV="${EXPECTED_UNITREE_MJLAB_REV:-1425b15f73bd4095f0df53709d7c389c3eb9e790}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

if [[ -z "${UV_BIN}" ]]; then
  echo "error: uv is required but was not found in PATH" >&2
  exit 2
fi

if [[ ! -d "${UNITREE_MJLAB_ROOT}/.git" || ! -f "${UNITREE_MJLAB_ROOT}/setup.py" ]]; then
  cat >&2 <<EOF
error: official Unitree MJLab checkout not found at:
  ${UNITREE_MJLAB_ROOT}

Create the checkout explicitly, then rerun this script:
  git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git "${UNITREE_MJLAB_ROOT}"
  git -C "${UNITREE_MJLAB_ROOT}" checkout "${EXPECTED_UNITREE_MJLAB_REV}"

This setup script intentionally does not fetch upstream repositories or checkpoints.
EOF
  exit 2
fi

ACTUAL_UNITREE_MJLAB_REV="$(git -C "${UNITREE_MJLAB_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_UNITREE_MJLAB_REV}" != "${EXPECTED_UNITREE_MJLAB_REV}" ]]; then
  cat >&2 <<EOF
error: Unitree MJLab revision mismatch
  expected: ${EXPECTED_UNITREE_MJLAB_REV}
  actual:   ${ACTUAL_UNITREE_MJLAB_REV}

Set EXPECTED_UNITREE_MJLAB_REV only after validating a different upstream revision.
EOF
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  "${UV_BIN}" venv --python 3.11 "${VENV_ROOT}"
fi

PYTHON_SERIES="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_SERIES}" != "3.11" ]]; then
  echo "error: ${PYTHON_BIN} is Python ${PYTHON_SERIES}; Python 3.11 is required" >&2
  exit 2
fi

"${UV_BIN}" pip install \
  --python "${PYTHON_BIN}" \
  --torch-backend cu130 \
  --constraints "${CONSTRAINTS}" \
  --editable "${REPO_ROOT}[dev,training,mujoco]" \
  --editable "${UNITREE_MJLAB_ROOT}" \
  scipy

"${UV_BIN}" pip check --python "${PYTHON_BIN}"

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata

packages = (
    "torch",
    "torchvision",
    "mjlab",
    "mujoco",
    "mujoco-warp",
    "warp-lang",
    "rsl-rl-lib",
    "scipy",
)
for package in packages:
    print(f"{package}=={metadata.version(package)}")
PY

echo
echo "Environment ready: ${PYTHON_BIN}"
echo "Official GPU smoke: ${REPO_ROOT}/scripts/run_rtx_mjlab_smoke.sh"
echo "Task044 migration audit: ${REPO_ROOT}/scripts/check_task044_migration.sh"
