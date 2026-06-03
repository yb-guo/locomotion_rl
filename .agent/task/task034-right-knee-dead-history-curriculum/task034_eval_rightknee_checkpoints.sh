#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task034_rightknee_frozenbase_train/2026-05-28_14-12-50_034_mixed_rightknee_from_task033_model5378_env8192_iter30_gpu1_seed3403401_lr5e6}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task034/rightknee_eval}"
PY="${PY:-/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python}"
CHECKPOINTS="${CHECKPOINTS:-5380 5385 5390 5395 5400 5405 5407}"
SEED="${SEED:-3403500}"
NUM_ENVS="${NUM_ENVS:-128}"
STEPS="${STEPS:-500}"

cd /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter
for model in ${CHECKPOINTS}; do
  out="${OUTPUT_ROOT}/model${model}_vx2p0_seed${SEED}"
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH=src:/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab:/tmp/task029_ipython_stub \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  "${PY}" -m h200_locomotion_lab.tools.task033_failure_grid_eval_checkpoint \
    --task Unitree-G1-Gripper-Flat-Task033-StackMlpK4-FrozenBase-FocusedDeadGrid-Fast2p0 \
    --checkpoint "${CHECKPOINT_DIR}/model_${model}.pt" \
    --output-dir "${out}" \
    --joints right_knee_joint \
    --lin-vel-x 2.0 \
    --num-envs "${NUM_ENVS}" \
    --steps "${STEPS}" \
    --seed "${SEED}" \
    --device cuda:0 >/tmp/task034_eval_model${model}.log 2>&1
  MODEL="${model}" OUT="${out}" python - <<'PY'
import json
import os

model = os.environ["MODEL"]
path = os.path.join(os.environ["OUT"], "task033_failure_grid_eval_aggregate.json")
data = json.load(open(path, encoding="utf-8"))
failed = data.get("failed") or [{}]
row = failed[0]
print(
    model,
    data.get("pass"),
    data.get("pass_count"),
    row.get("zero_fall_ratio"),
    row.get("lin_vel_error_mean"),
    row.get("yaw_vel_error_mean"),
    row.get("gravity_xy_mean"),
)
PY
done
