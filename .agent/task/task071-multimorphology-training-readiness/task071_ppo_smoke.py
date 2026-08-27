"""Run the bounded Task071 R3 no-update and one-update CUDA smoke."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = ROOT / ".agent/task/task071-multimorphology-training-readiness"
OUT = TASK_DIR / "artifacts"
ARTIFACT_PATH = OUT / "r3_ppo_update_smoke.json"
ARTIFACT_VERSION = "task071_r3_g1_go2_bounded_cuda_ppo_v1"
COMMAND = (
    "UV_CACHE_DIR=/home/admin1/workspace/store/cache/uv "
    "uv run --isolated --locked --offline --python 3.11 "
    "--extra training --extra mujoco python "
    ".agent/task/task071-multimorphology-training-readiness/task071_ppo_smoke.py"
)


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _gpu_identity(torch: Any, device: Any) -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    name, driver, total_mib = (value.strip() for value in query.split(","))
    properties = torch.cuda.get_device_properties(device)
    return {
        "torch_device": str(device),
        "torch_device_name": torch.cuda.get_device_name(device),
        "nvidia_smi_name": name,
        "driver_version": driver,
        "total_memory_mib": int(total_mib),
        "torch_total_memory_bytes": int(properties.total_memory),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
    }


class _TracedEnvironment:
    def __init__(self, shard: Any) -> None:
        self.shard = shard
        self.trial_done: list[list[bool]] = []
        self.context_done: list[list[bool]] = []
        self.falls: list[list[bool]] = []

    @property
    def active_action_mask(self) -> Any:
        return self.shard.active_action_mask

    def reset(self) -> Any:
        return self.shard.reset()

    def step(self, action: Any) -> Any:
        result = self.shard.step(action)
        self.trial_done.append([bool(value) for value in result.trial_done])
        self.context_done.append([bool(value) for value in result.context_done])
        self.falls.append([bool(value) for value in result.metrics["fall"]])
        return result


def _snapshot(policy: Any) -> tuple[Any, ...]:
    return tuple(parameter.detach().clone() for parameter in policy.parameters())


def _parameter_l1_delta(before: tuple[Any, ...], policy: Any) -> float:
    return sum(
        float((old - current.detach()).abs().sum().item())
        for old, current in zip(before, policy.parameters())
    )


def _all_parameters_finite(torch: Any, policy: Any) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters())


def _run_case(
    torch: Any,
    physics: Any,
    env_contract: Any,
    overlay: dict[str, Any],
    r1: dict[str, Any],
    reference_id: str,
    device: Any,
) -> dict[str, Any]:
    import mujoco

    from h200_locomotion_lab.algorithms.ppo import compute_gae, ppo_update
    from h200_locomotion_lab.envs.whole_body_mujoco import (
        WholeBodyMuJoCoShard,
        WholeBodyMuJoCoShardConfig,
    )
    from h200_locomotion_lab.robots.motor_process import MotorProcessConfig
    from h200_locomotion_lab.training.whole_body_ppo import (
        WholeBodyPPOConfig,
        WholeBodyPPOTrainer,
    )

    case = physics.load_frozen_case(reference_id)
    overlay_record = env_contract._record(overlay["records"], reference_id)
    r1_record = env_contract._record(r1["records"], reference_id)
    stance = env_contract._stance_solution(physics, case, overlay, r1)
    xml = env_contract._bound_xml(physics, mujoco, case, overlay_record)
    env_config = WholeBodyMuJoCoShardConfig(
        trial_seconds=0.08,
        context_trials=2,
        action_scale=0.10,
        command_vx_range=(0.0, 0.0),
        command_vy_range=(0.0, 0.0),
        command_yaw_range=(0.0, 0.0),
        seed=7100,
    )
    shard = WholeBodyMuJoCoShard(
        case.blueprint,
        physical=case.physical,
        num_envs=4,
        config=env_config,
        motor_config=MotorProcessConfig(
            control_hz=env_config.control_hz,
            no_event_probability=1.0,
        ),
        model_xml=xml,
        model_xml_sha256=overlay_record["output_xml_sha256"],
        stance_solution=stance,
    )
    traced = _TracedEnvironment(shard)
    ppo_config = WholeBodyPPOConfig(
        rollout_steps=8,
        updates=1,
        hidden_dim=64,
        hidden_layers=2,
        epochs=1,
        minibatch_size=32,
        log_std_init=-2.0,
        device=str(device),
    )
    seed = 71_000 + tuple(physics.CASE_SPECS).index(reference_id)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    trainer = WholeBodyPPOTrainer(
        traced,
        action_mask=shard.active_action_mask,
        config=ppo_config,
    )
    before_rollout = _snapshot(trainer.policy)
    torch.cuda.synchronize(device)
    collect_started = time.perf_counter()
    batch, reward_mean, fall_count = trainer._collect_rollout()
    torch.cuda.synchronize(device)
    collect_seconds = time.perf_counter() - collect_started
    no_update_delta = _parameter_l1_delta(before_rollout, trainer.policy)

    finite_tensors = {
        "observations": bool(torch.isfinite(batch.observations).all()),
        "actions": bool(torch.isfinite(batch.actions).all()),
        "rewards": bool(torch.isfinite(batch.rewards).all()),
        "values": bool(torch.isfinite(batch.values).all()),
        "log_probs": bool(torch.isfinite(batch.log_probs).all()),
        "next_observation": bool(torch.isfinite(trainer.observation).all()),
        "next_value": bool(torch.isfinite(batch.next_value).all()),
    }
    inactive_values = batch.actions.masked_select(~batch.active_action_mask)
    inactive_actions_zero = bool(torch.count_nonzero(inactive_values).item() == 0)
    active_count = len(case.blueprint.actuators)
    active_mask_exact = bool(
        batch.active_action_mask.shape == (8, 4, 45)
        and torch.all(batch.active_action_mask.sum(dim=-1) == active_count)
    )
    action_bounds = bool((batch.actions.abs() <= 1.0).all())
    trial_done_counts = [sum(row) for row in traced.trial_done]
    context_done_counts = [sum(row) for row in traced.context_done]
    fall_counts = [sum(row) for row in traced.falls]
    reset_semantics = bool(
        trial_done_counts == [0, 0, 0, 4, 0, 0, 0, 4]
        and context_done_counts == [0, 0, 0, 0, 0, 0, 0, 4]
        and fall_counts == [0] * 8
        and all(int(value) == 0 for value in shard._trial_step)
        and all(int(value) == 0 for value in shard._trial_index)
        and all(int(value) == 1 for value in shard._context_index)
    )
    no_update_passed = bool(
        no_update_delta == 0.0
        and all(finite_tensors.values())
        and inactive_actions_zero
        and active_mask_exact
        and action_bounds
        and reset_semantics
        and fall_count == 0.0
        and math.isfinite(reward_mean)
    )

    advantages, returns = compute_gae(batch, ppo_config)
    advantages_finite = bool(torch.isfinite(advantages).all())
    returns_finite = bool(torch.isfinite(returns).all())
    before_update = _snapshot(trainer.policy)
    optimizer_step_count = 0
    original_optimizer_step = trainer.optimizer.step

    def counted_optimizer_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal optimizer_step_count
        optimizer_step_count += 1
        return original_optimizer_step(*args, **kwargs)

    trainer.optimizer.step = counted_optimizer_step
    try:
        diagnostics = ppo_update(
            trainer.policy,
            trainer.optimizer,
            batch,
            advantages,
            returns,
            ppo_config,
        )
    finally:
        trainer.optimizer.step = original_optimizer_step
    torch.cuda.synchronize(device)
    parameter_delta = _parameter_l1_delta(before_update, trainer.policy)
    diagnostic_values = asdict(diagnostics)
    diagnostics_finite = all(math.isfinite(float(value)) for value in diagnostic_values.values())
    parameters_finite = _all_parameters_finite(torch, trainer.policy)
    update_passed = bool(
        advantages_finite
        and returns_finite
        and diagnostics_finite
        and diagnostics.grad_norm > 0.0
        and parameter_delta > 0.0
        and parameters_finite
        and optimizer_step_count == 1
    )
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    env_steps = ppo_config.rollout_steps * shard.num_envs
    checks = {
        "exact_r1_xml_and_stance_consumed": bool(
            shard.xml == xml
            and shard.model_xml_sha256 == overlay_record["output_xml_sha256"]
            and stance.solution_hash == shard.stance_solution.solution_hash
            and r1_record["bound_xml_sha256"] == overlay_record["output_xml_sha256"]
        ),
        "no_update_rollout_parameter_immutable": no_update_delta == 0.0,
        "rollout_tensors_finite": all(finite_tensors.values()),
        "inactive_actions_exactly_zero": inactive_actions_zero,
        "active_mask_exact": active_mask_exact,
        "actions_bounded": action_bounds,
        "trial_context_reset_semantics": reset_semantics,
        "rollout_zero_fall": fall_count == 0.0,
        "advantages_returns_finite": advantages_finite and returns_finite,
        "ppo_diagnostics_and_gradient_finite": diagnostics_finite and diagnostics.grad_norm > 0.0,
        "ppo_parameter_delta_positive": parameter_delta > 0.0 and parameters_finite,
        "exactly_one_optimizer_step_observed": optimizer_step_count == 1,
    }
    passed = no_update_passed and update_passed and all(checks.values())
    result = {
        "reference_id": reference_id,
        "family": case.spec["family"],
        "bound_xml_sha256": overlay_record["output_xml_sha256"],
        "stance_profile_sha256": r1_record["stance_profile"]["sha256"],
        "stance_solution_sha256": stance.solution_hash,
        "torch_seed": seed,
        "budget": {
            "num_envs": shard.num_envs,
            "rollout_steps": ppo_config.rollout_steps,
            "env_steps": env_steps,
            "ppo_updates": 1,
            "epochs": ppo_config.epochs,
            "minibatch_size": ppo_config.minibatch_size,
            "hidden_dim": ppo_config.hidden_dim,
            "action_scale": env_config.action_scale,
            "runtime_fault_process": "disabled_for_nominal_smoke",
        },
        "no_update_rollout": {
            "parameter_l1_delta": no_update_delta,
            "finite_tensors": finite_tensors,
            "inactive_actions_exactly_zero": inactive_actions_zero,
            "active_mask_count": active_count,
            "reward_mean": reward_mean,
            "fall_count": fall_count,
            "trial_done_count_by_step": trial_done_counts,
            "context_done_count_by_step": context_done_counts,
            "collect_seconds": collect_seconds,
            "env_steps_per_second": env_steps / collect_seconds,
            "passed": no_update_passed,
        },
        "ppo_update": {
            "advantages_finite": advantages_finite,
            "returns_finite": returns_finite,
            "diagnostics": diagnostic_values,
            "parameters_finite_after_update": parameters_finite,
            "parameter_l1_delta": parameter_delta,
            "completed_update_count": optimizer_step_count,
            "passed": update_passed,
        },
        "gpu_memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_allocated_mib": peak_allocated / 2**20,
            "peak_reserved_mib": peak_reserved / 2**20,
        },
        "checks": checks,
        "passed": passed,
        "failure_reasons": [] if passed else [name for name, value in checks.items() if not value],
    }
    del batch, advantages, returns, trainer, traced, shard
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_ppo_smoke(*, write_artifact: bool = True) -> dict[str, Any]:
    sys.path[:0] = [str(TASK_DIR), str(ROOT / "src")]
    import task071_env_contract as env_contract
    import task071_physics_overlay as physics
    import torch

    from h200_locomotion_lab.envs import whole_body_mujoco
    from h200_locomotion_lab.policies import whole_body_mlp
    from h200_locomotion_lab.training import whole_body_ppo

    _require(torch.cuda.is_available(), "R3 requires CUDA")
    device = torch.device("cuda:0")
    overlay = physics.generate_overlay(write_artifact=write_artifact)
    r1 = physics.run_bound_r1(overlay, write_artifact=write_artifact)
    r2 = env_contract.run_env_contract(
        write_artifact=write_artifact,
        overlay=overlay,
        r1=r1,
    )
    _require(r2["task071_r2_admission_passed"] is True, "R3 requires R2 admission")
    _require(
        r2["overlay"]["sha256"] == physics.json_artifact_sha256(overlay),
        "R2 overlay drift",
    )
    _require(
        r2["r1"]["sha256"] == physics.json_artifact_sha256(r1),
        "R2 R1 drift",
    )
    gpu = _gpu_identity(torch, device)
    records = [
        _run_case(torch, physics, env_contract, overlay, r1, reference_id, device)
        for reference_id in physics.CASE_SPECS
    ]
    admission = len(records) == 2 and all(record["passed"] for record in records)
    payload = {
        "artifact": ARTIFACT_VERSION,
        "task": "task071-multimorphology-training-readiness",
        "runtime": {
            "command": COMMAND,
            "git_head": _git_head(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gpu": gpu,
            "robot_asset_dataset_or_checkpoint_downloads_performed": False,
        },
        "source": {
            "probe_path": str(Path(__file__).relative_to(ROOT)),
            "probe_sha256": _sha256_path(Path(__file__)),
            "environment_path": str(Path(whole_body_mujoco.__file__).relative_to(ROOT)),
            "environment_sha256": _sha256_path(Path(whole_body_mujoco.__file__)),
            "trainer_path": str(Path(whole_body_ppo.__file__).relative_to(ROOT)),
            "trainer_sha256": _sha256_path(Path(whole_body_ppo.__file__)),
            "policy_path": str(Path(whole_body_mlp.__file__).relative_to(ROOT)),
            "policy_sha256": _sha256_path(Path(whole_body_mlp.__file__)),
        },
        "r2": {
            "payload_sha256": _payload_sha256(r2),
            "artifact_path": str(env_contract.ARTIFACT_PATH.relative_to(ROOT)),
            "artifact_sha256": (
                _sha256_path(env_contract.ARTIFACT_PATH) if write_artifact else None
            ),
            "admission": True,
            "r1": r2["r1"],
            "overlay": r2["overlay"],
        },
        "denominator": 2,
        "records": records,
        "summary": {
            "no_update_rollout_passed": sum(
                record["no_update_rollout"]["passed"] for record in records
            ),
            "one_ppo_update_passed": sum(record["ppo_update"]["passed"] for record in records),
            "all_contract_checks_passed": sum(record["passed"] for record in records),
            "total_env_steps": sum(record["budget"]["env_steps"] for record in records),
            "total_completed_ppo_updates": sum(
                record["ppo_update"]["completed_update_count"] for record in records
            ),
            "max_peak_allocated_mib": max(
                record["gpu_memory"]["peak_allocated_mib"] for record in records
            ),
        },
        "task071_r3_representative_admission_passed": admission,
        "failure_reasons": [] if admission else [
            f"{record['reference_id']}:{reason}"
            for record in records
            for reason in record["failure_reasons"]
        ],
        "claim_boundary": {
            "bounded_one_update_smoke_only": True,
            "training_performed": True,
            "ppo_updates_completed_per_case": 1,
            "checkpoint_written": False,
            "long_training_started": False,
            "walking_or_quality_claimed": False,
            "task071_passed": False,
        },
    }
    if write_artifact:
        physics.write_json_artifact(ARTIFACT_PATH, payload)
    return payload


def main() -> int:
    payload = run_ppo_smoke()
    print(
        "Task071 R3: "
        f"admission={payload['task071_r3_representative_admission_passed']}, "
        f"rollout={payload['summary']['no_update_rollout_passed']}/2, "
        f"ppo_update={payload['summary']['one_ppo_update_passed']}/2"
    )
    return 0 if payload["task071_r3_representative_admission_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
