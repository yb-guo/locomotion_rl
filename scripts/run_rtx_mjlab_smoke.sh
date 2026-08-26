#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNITREE_MJLAB_ROOT="${UNITREE_MJLAB_ROOT:-${REPO_ROOT}/.external/unitree_rl_mjlab}"
VENV_ROOT="${RTX_MJLAB_VENV_ROOT:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${VENV_ROOT}/bin/python"
TASK_ID="${RTX_MJLAB_SMOKE_TASK:-Unitree-G1-Flat}"
NUM_ENVS="${RTX_MJLAB_SMOKE_NUM_ENVS:-32}"
STEPS_PER_ENV="${RTX_MJLAB_SMOKE_STEPS_PER_ENV:-8}"
EXPERIMENT_NAME="${RTX_MJLAB_SMOKE_EXPERIMENT:-task047_rtx5060ti_smoke}"
RUN_NAME="${RTX_MJLAB_SMOKE_RUN_NAME:-official_g1_env${NUM_ENVS}_iter1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: ${PYTHON_BIN} is missing; run scripts/setup_rtx_mjlab.sh first" >&2
  exit 2
fi
if [[ ! -f "${UNITREE_MJLAB_ROOT}/scripts/train.py" ]]; then
  echo "error: Unitree MJLab train entry point is missing at ${UNITREE_MJLAB_ROOT}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_DISABLED="true"

"${PYTHON_BIN}" - <<'PY'
import torch

assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
assert "sm_120" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
x = torch.randn((1024, 1024), device="cuda")
y = x @ x.T
assert torch.isfinite(y).all()
print(
    {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "finite_matmul": True,
    }
)
PY

cd "${UNITREE_MJLAB_ROOT}"
"${PYTHON_BIN}" scripts/train.py \
  "${TASK_ID}" \
  --env.scene.num-envs "${NUM_ENVS}" \
  --agent.num-steps-per-env "${STEPS_PER_ENV}" \
  --agent.max-iterations 1 \
  --agent.save-interval 1 \
  --agent.experiment-name "${EXPERIMENT_NAME}" \
  --agent.run-name "${RUN_NAME}" \
  --agent.logger tensorboard \
  --agent.upload-model False
