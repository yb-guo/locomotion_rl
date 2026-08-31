"""Fail-closed verifier for Task072 repaired-E3a smoke and optimizer gates."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = TASK_DIR / "artifacts/nominal_v4/unitree_g1"
E2_DIR = ARTIFACT_ROOT / "E2_reward_dt"
REJECTED_DIR = ARTIFACT_ROOT / "E3a_adaptive_kl"
REPAIR_DIR = ARTIFACT_ROOT / "E3a_mjlab_kl_repair"
SMOKE_DIR = REPAIR_DIR / "smoke"

ARTIFACT = "task072_bound_locomotion_proof_v1"
TASK = "task072-bound-g1-go2-locomotion-proof"
CASE = "unitree_g1"
REPAIR = "E3a_mjlab_kl_repair"

NO_UPDATE_PATH = REPAIR_DIR / "no_update_correctness_gate.json"
R4_GATE_PATH = REPAIR_DIR / "r4_repaired_smoke_gate.json"
R4_MANIFEST_PATH = SMOKE_DIR / "run_manifest.json"
R5_MANIFEST_PATH = REPAIR_DIR / "run_manifest.json"

NO_UPDATE_SHA256 = "d2e3650be8e3b112688898bdb8441fb761c14171455a574c1d186adcfbbb80c3"
FIXED_ARTIFACTS = {
    "e2_gate": (
        E2_DIR / "e2_gate.json",
        "fe41b57a91b715254f09625b73339e6403d02c1caa696ce6b73c0a8571b2910e",
    ),
    "e2_run_manifest": (
        E2_DIR / "run_manifest.json",
        "25ef89af5c7460701a8b8f5f2b48de876dbcf34b7405e54dac7471e261b35597",
    ),
    "rejected_e3a_gate": (
        REJECTED_DIR / "e3a_gate.json",
        "3cb5594110432cf6bc4ab10bd4d74acec470468e8fe2cb16d05177ceb1706d1e",
    ),
    "rejected_e3a_progression": (
        REJECTED_DIR / "progression.json",
        "45e9a52ee07a6166fe1ffe2cd1af43a44a4f9bda7fdb3e4f07261782db080d27",
    ),
    "rejected_e3a_run_manifest": (
        REJECTED_DIR / "run_manifest.json",
        "3974dbadfe7d96b234878cc14fe6edfb58def64ea2a154747688bda5f118e33a",
    ),
}
R4_ARTIFACT_SHA256 = {
    "run_manifest": "04d1aba1a459832f704bb3724ec04cc70cf026ed6fef93543cc8ab57a628db40",
    "progression": "28bfe92d62275985623945f56a13e9d91009d91fbfff95fd0d11635a5e500651",
    "initial": "c657a6a7a8bf31c7b798390ee811a953c31f4b78b7adc1b2bcd6c480a9a0da39",
    "update000001": "b1f4922e17298e5f1a22429d255c29fa151e29fcdf4f2ea49a941277db93e83d",
    "update000002": "707f842687aa6a5b9cbdc7b09d8311c91afa5d7a18c10b25d9ed351a990aeeb3",
    "final": "0c851a233f99c930aa78ec290c9a786336502f6f160c7707efd939a0f2273688",
}

SOURCE_PATHS = {
    "environment": "src/h200_locomotion_lab/envs/whole_body_mujoco.py",
    "masked_distribution": "src/h200_locomotion_lab/masked_distribution.py",
    "policy": "src/h200_locomotion_lab/policies/whole_body_mlp.py",
    "ppo_kernel": "src/h200_locomotion_lab/algorithms/ppo.py",
    "pyproject": "pyproject.toml",
    "task072_cli": ".agent/task/task072-bound-g1-go2-locomotion-proof/task072_locomotion_proof.py",
    "trainer": "src/h200_locomotion_lab/training/whole_body_ppo.py",
    "uv_lock": "uv.lock",
}
NO_UPDATE_SOURCE_LABELS = {
    "masked_distribution": "masked_distribution",
    "policy": "policy",
    "ppo_kernel": "ppo_kernel",
    "task072_cli_and_verifier": "task072_cli",
    "trainer": "trainer",
}

CONFIG_DIFF_PATHS = frozenset(
    {
        "configuration.adaptive_kl",
        "configuration.desired_kl",
        "configuration.hard_kl_stop",
        "configuration.ppo.adaptive_kl",
        "configuration.ppo.desired_kl",
        "configuration.ppo.hard_kl_stop",
        "configuration.ppo.target_kl",
        "configuration.target_kl",
        "configuration.variant_id",
    }
)
MANIFEST_DELTA_ALLOWLIST = sorted(CONFIG_DIFF_PATHS - {"configuration.variant_id"})
THRESHOLDS = {
    "approx_kl_mean": 0.015,
    "approx_kl_p95": 0.03,
    "approx_kl_max": 0.05,
    "clip_fraction_mean": 0.20,
    "clip_fraction_p95": 0.35,
}
EXPECTED_POLICY_PARAMETER_COUNT = 243803


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def payload_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> Any:
    raise ValueError(f"invalid JSON constant: {token}")


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
    )
    require(isinstance(value, dict), f"JSON object required: {path}")
    finite_tree(value)
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def finite_tree(value: Any, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"nonfinite JSON number: {path}")


_MISSING = object()


def config_diff(left: Any, right: Any, prefix: str = "configuration") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            paths.extend(
                config_diff(
                    left.get(key, _MISSING),
                    right.get(key, _MISSING),
                    f"{prefix}.{key}",
                )
            )
        return paths
    return [] if left is not _MISSING and right is not _MISSING and left == right else [prefix]


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    require(resolved.is_relative_to(ROOT), f"path escapes repository: {path}")
    return str(resolved.relative_to(ROOT))


def exact_path(value: Any, expected: Path, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} path missing")
    observed = Path(value)
    if not observed.is_absolute():
        observed = ROOT / observed
    require(observed.resolve() == expected.resolve(), f"{label} path mismatch")
    require(expected.resolve().is_relative_to(ROOT), f"{label} path escapes repository")
    return expected.resolve()


def require_sha(path: Path, expected: Any, label: str) -> str:
    require(isinstance(expected, str) and len(expected) == 64, f"{label} SHA missing")
    observed = sha256(path)
    require(observed == expected, f"{label} SHA drift")
    return observed


def guard_output(output: Path, inputs: list[Path]) -> Path:
    resolved = output.resolve()
    require(all(resolved != item.resolve() for item in inputs), "output would overwrite evidence")
    return resolved


def repaired_config(e2_config: dict[str, Any], *, stage: str) -> dict[str, Any]:
    config = copy.deepcopy(e2_config)
    config.update(
        {
            "variant_id": REPAIR,
            "target_kl": None,
            "hard_kl_stop": False,
            "adaptive_kl": True,
            "desired_kl": 0.01,
        }
    )
    config["ppo"].update(
        {
            "target_kl": None,
            "hard_kl_stop": False,
            "adaptive_kl": True,
            "desired_kl": 0.01,
        }
    )
    if stage == "smoke":
        config.update(
            {
                "stage": "smoke",
                "num_envs": 4,
                "rollout_steps": 32,
                "updates": 2,
                "env_steps": 256,
                "checkpoint_every": 1,
            }
        )
        config["ppo"]["rollout_steps"] = 32
    else:
        require(stage == "pilot", f"unsupported repaired stage: {stage}")
    return config


def load_fixed_evidence(no_update_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(no_update_path.resolve() == NO_UPDATE_PATH.resolve(), "noncanonical no-update gate path")
    require_sha(no_update_path, NO_UPDATE_SHA256, "no-update gate")
    no_update = load_json(no_update_path)
    require(no_update.get("schema_version") == 2, "no-update schema mismatch")
    require(no_update.get("variant_id") == REPAIR, "no-update variant mismatch")
    require(no_update.get("r0_r3_passed") is True, "no-update R0-R3 gate did not pass")
    require(no_update.get("training_started") is False, "no-update gate claims training")
    require(no_update.get("smoke_started") is False, "no-update gate claims smoke")

    records = no_update.get("r0", {}).get("fixed_artifacts")
    require(isinstance(records, dict), "no-update fixed_artifacts missing")
    require(set(records) == set(FIXED_ARTIFACTS), "fixed artifact set mismatch")
    observed: dict[str, Any] = {}
    for label, (expected_path, expected_sha) in FIXED_ARTIFACTS.items():
        record = records[label]
        require(isinstance(record, dict), f"invalid fixed artifact record: {label}")
        exact_path(record.get("path"), expected_path, label)
        require(record.get("sha256") == expected_sha, f"{label} anchored SHA mismatch")
        observed[label] = {
            "path": repo_relative(expected_path),
            "sha256": require_sha(expected_path, expected_sha, label),
        }

    rejected_gate = load_json(FIXED_ARTIFACTS["rejected_e3a_gate"][0])
    require(
        rejected_gate.get("r2_e3a_optimizer_gate_passed") is False,
        "rejected E3a no longer rejected",
    )
    e2_manifest = load_json(FIXED_ARTIFACTS["e2_run_manifest"][0])
    return no_update, e2_manifest, observed


def validate_sources(
    manifest: dict[str, Any],
    no_update: dict[str, Any],
) -> dict[str, dict[str, str]]:
    sources = manifest.get("static_lineage", {}).get("sources")
    require(isinstance(sources, dict), "manifest sources missing")
    require(set(sources) == set(SOURCE_PATHS), "manifest source set mismatch")
    output: dict[str, dict[str, str]] = {}
    for label, relative in SOURCE_PATHS.items():
        record = sources[label]
        require(isinstance(record, dict), f"source record invalid: {label}")
        path = ROOT / relative
        exact_path(record.get("path"), path, f"source {label}")
        observed_sha = require_sha(path, record.get("sha256"), f"source {label}")
        output[label] = {"path": relative, "sha256": observed_sha}

    controlled = no_update.get("r0", {}).get("controlled_sources")
    require(isinstance(controlled, dict), "no-update controlled sources missing")
    require(set(controlled) == set(NO_UPDATE_SOURCE_LABELS), "controlled source set mismatch")
    for no_update_label, manifest_label in NO_UPDATE_SOURCE_LABELS.items():
        record = controlled[no_update_label]
        require(isinstance(record, dict), f"controlled source invalid: {no_update_label}")
        expected = output[manifest_label]
        require(record.get("path") == expected["path"], f"controlled source path drift: {no_update_label}")
        require(record.get("sha256") == expected["sha256"], f"controlled source SHA drift: {no_update_label}")
    return output


def validate_manifest(
    manifest: dict[str, Any],
    *,
    expected_config: dict[str, Any],
    e2_config: dict[str, Any],
    no_update: dict[str, Any],
    expected_static_lineage: dict[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    finite_tree(manifest)
    require(manifest.get("artifact") == ARTIFACT, "manifest artifact mismatch")
    require(manifest.get("task") == TASK, "manifest task mismatch")
    require(manifest.get("case") == CASE, "manifest case mismatch")
    require(manifest.get("variant_id") == REPAIR, "manifest variant mismatch")
    require(manifest.get("parent_variant") == "E2_reward_dt", "manifest parent mismatch")
    require(manifest.get("configuration") == expected_config, "manifest configuration drift")
    require(
        manifest.get("parent_config_sha256") == payload_sha(e2_config),
        "parent configuration SHA mismatch",
    )
    require(
        manifest.get("delta_allowlist") == MANIFEST_DELTA_ALLOWLIST,
        "manifest delta allowlist drift",
    )

    static_lineage = manifest.get("static_lineage")
    require(isinstance(static_lineage, dict), "static lineage missing")
    if expected_static_lineage is not None:
        require(static_lineage == expected_static_lineage, "static lineage differs from repaired smoke")
    require(static_lineage.get("artifact") == ARTIFACT, "static lineage artifact mismatch")
    require(static_lineage.get("case") == CASE, "static lineage case mismatch")
    require(
        manifest.get("static_lineage_sha256") == payload_sha(static_lineage),
        "static lineage SHA mismatch",
    )
    sources = validate_sources(manifest, no_update)
    require(
        manifest.get("source_sha256") == payload_sha(static_lineage["sources"]),
        "source-set SHA mismatch",
    )

    expected_identity = {
        "artifact": ARTIFACT,
        "case": CASE,
        "static_lineage_sha256": manifest["static_lineage_sha256"],
        "configuration": expected_config,
    }
    require(manifest.get("run_identity") == expected_identity, "run identity content mismatch")
    require(
        manifest.get("run_identity_sha256") == payload_sha(expected_identity),
        "run identity SHA mismatch",
    )
    claim = manifest.get("claim_boundary", {})
    require(claim.get("training_completed") is True, "training-completed claim missing")
    require(claim.get("evaluation_completed") is False, "unexpected evaluation claim")
    require(claim.get("walking_claimed") is False, "unexpected walking claim")
    require(claim.get("task072_passed") is False, "unexpected Task072 pass claim")
    runtime = manifest.get("runtime", {})
    require("RTX 5060 Ti" in str(runtime.get("gpu")), "non-RTX-5060-Ti runtime")
    require(
        runtime.get("robot_asset_dataset_or_checkpoint_downloads_performed") is False,
        "runtime reports external downloads",
    )
    return sources


def scheduler_decision(kl: float) -> str:
    if kl > 0.02:
        return "decrease"
    if 0.0 < kl < 0.005:
        return "increase"
    return "hold"


def scheduler_lr(before: float, decision: str) -> float:
    factor = 1.5 if decision == "increase" else 1.0 / 1.5 if decision == "decrease" else 1.0
    return min(1e-2, max(1e-5, before * factor))


def close(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    return finite_number(left) and finite_number(right) and math.isclose(
        float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
    )


def validate_reports(
    rows: Any,
    *,
    updates: int,
    minibatches_per_epoch: int,
    env_steps_per_update: int,
    initial_learning_rate: float,
) -> list[dict[str, Any]]:
    require(isinstance(rows, list) and len(rows) == updates, "progression report count mismatch")
    flat: list[dict[str, Any]] = []
    previous_after = initial_learning_rate
    expected_per_update = 4 * minibatches_per_epoch
    expected_layout = [
        (epoch, index)
        for epoch in range(4)
        for index in range(minibatches_per_epoch)
    ]
    for update, row in enumerate(rows, start=1):
        require(isinstance(row, dict), f"report {update} is not an object")
        require(row.get("global_update") == update, f"report {update} global_update mismatch")
        require(
            row.get("env_steps") == update * env_steps_per_update,
            f"report {update} env_steps mismatch",
        )
        require(row.get("epochs_completed") == 4, f"report {update} epoch count mismatch")
        require(
            row.get("minibatches_attempted") == expected_per_update,
            f"report {update} attempted minibatches mismatch",
        )
        require(
            row.get("minibatches_completed") == expected_per_update,
            f"report {update} completed minibatches mismatch",
        )
        require(row.get("early_stopped") is False, f"report {update} early-stopped")
        require(close(row.get("desired_kl"), 0.01), f"report {update} desired KL mismatch")

        minibatches = row.get("minibatches")
        require(
            isinstance(minibatches, list) and len(minibatches) == expected_per_update,
            f"report {update} minibatch record count mismatch",
        )
        layout = [(record.get("epoch"), record.get("index")) for record in minibatches]
        require(layout == expected_layout, f"report {update} minibatch layout mismatch")
        for position, record in enumerate(minibatches):
            require(isinstance(record, dict), f"report {update} minibatch {position} invalid")
            required = {
                "applied",
                "epoch",
                "index",
                "scheduler_kl",
                "scheduler_decision",
                "learning_rate_before",
                "learning_rate_after",
                "approx_kl",
                "clip_fraction",
                "same_policy_identity_error",
            }
            require(required <= set(record), f"report {update} minibatch telemetry missing")
            require(record["applied"] is True, f"report {update} unapplied minibatch")
            numeric = (
                "scheduler_kl",
                "learning_rate_before",
                "learning_rate_after",
                "approx_kl",
                "clip_fraction",
                "same_policy_identity_error",
            )
            require(
                all(finite_number(record[key]) for key in numeric),
                f"report {update} nonfinite/bool minibatch telemetry",
            )
            kl = float(record["scheduler_kl"])
            before = float(record["learning_rate_before"])
            after = float(record["learning_rate_after"])
            approx_kl = float(record["approx_kl"])
            clip_fraction = float(record["clip_fraction"])
            identity_error = float(record["same_policy_identity_error"])
            require(kl >= 0.0, f"report {update} negative scheduler KL")
            # The sampled estimator is theoretically non-negative, but float32
            # cancellation produces tiny negative zero-point values in valid R5
            # telemetry.  Keep aggregate metrics strict while accepting only
            # this evidence-backed per-minibatch noise floor.
            require(approx_kl >= -1e-8, f"report {update} negative approximate KL")
            require(0.0 <= clip_fraction <= 1.0, f"report {update} invalid clip fraction")
            require(abs(identity_error) <= 1e-5, f"report {update} likelihood identity drift")
            require(1e-5 <= before <= 1e-2 and 1e-5 <= after <= 1e-2, "LR escaped bounds")
            require(close(before, previous_after), f"report {update} LR continuity mismatch")
            expected_decision = scheduler_decision(kl)
            require(
                record["scheduler_decision"] == expected_decision,
                f"report {update} scheduler decision mismatch",
            )
            require(
                close(after, scheduler_lr(before, expected_decision)),
                f"report {update} scheduler LR transition mismatch",
            )
            previous_after = after
            flat.append(record)

        first = minibatches[0]
        require(abs(float(first["approx_kl"])) <= 1e-6, f"report {update} first approximate KL")
        require(float(first["clip_fraction"]) == 0.0, f"report {update} first clip fraction")
        require(abs(float(first["scheduler_kl"])) <= 1e-6, f"report {update} first scheduler KL")
        require(
            close(
                row.get("approx_kl"),
                sum(float(item["approx_kl"]) for item in minibatches) / len(minibatches),
                tolerance=1e-8,
            ),
            f"report {update} approximate KL aggregate mismatch",
        )
        require(
            close(
                row.get("clip_fraction"),
                sum(float(item["clip_fraction"]) for item in minibatches) / len(minibatches),
            ),
            f"report {update} clip aggregate mismatch",
        )
        last = minibatches[-1]
        require(close(row.get("scheduler_kl"), last["scheduler_kl"]), f"report {update} scheduler KL aggregate mismatch")
        require(row.get("scheduler_decision") == last["scheduler_decision"], f"report {update} scheduler decision aggregate mismatch")
        require(close(row.get("learning_rate_before"), minibatches[0]["learning_rate_before"]), f"report {update} initial LR aggregate mismatch")
        require(close(row.get("learning_rate_after"), last["learning_rate_after"]), f"report {update} final LR aggregate mismatch")
        require(close(row.get("learning_rate"), last["learning_rate_after"]), f"report {update} learning-rate telemetry mismatch")
    return flat


def checkpoint_equal(left: Any, right: Any) -> bool:
    import torch

    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(checkpoint_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(checkpoint_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def finite_checkpoint_tree(value: Any, path: str = "checkpoint") -> None:
    import torch

    if torch.is_tensor(value):
        require(bool(torch.isfinite(value).all()), f"nonfinite tensor: {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_checkpoint_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_checkpoint_tree(item, f"{path}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"nonfinite checkpoint scalar: {path}")


def validate_checkpoint_reference(
    reference: Any,
    *,
    expected_path: Path,
    update: int,
    env_steps: int,
    expected_sha: str | None = None,
) -> tuple[dict[str, Any], str]:
    import torch

    require(isinstance(reference, dict), f"checkpoint reference missing: update {update}")
    path = exact_path(reference.get("path"), expected_path, f"checkpoint update {update}")
    require(reference.get("global_update") == update, f"checkpoint update {update} metadata mismatch")
    require(reference.get("env_steps") == env_steps, f"checkpoint update {update} env_steps mismatch")
    if expected_sha is not None:
        require(reference.get("sha256") == expected_sha, f"checkpoint update {update} anchored SHA mismatch")
    observed_sha = require_sha(path, reference.get("sha256"), f"checkpoint update {update}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(state, dict), f"checkpoint update {update} payload invalid")
    finite_checkpoint_tree(state)
    require(state.get("artifact") == ARTIFACT, f"checkpoint update {update} artifact mismatch")
    require(isinstance(state.get("policy"), dict), f"checkpoint update {update} policy missing")
    require(isinstance(state.get("optimizer"), dict), f"checkpoint update {update} optimizer missing")
    return state, observed_sha


def validate_checkpoint_lineage(
    state: dict[str, Any],
    *,
    manifest: dict[str, Any],
    update: int,
    env_steps: int,
) -> None:
    expected = {
        "artifact": ARTIFACT,
        "case": CASE,
        "static_lineage_sha256": manifest["static_lineage_sha256"],
        "run_identity_sha256": manifest["run_identity_sha256"],
        "global_update": update,
        "env_steps": env_steps,
    }
    require(state.get("lineage") == expected, f"checkpoint update {update} lineage mismatch")


def policy_delta(initial: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    import torch

    require(initial and set(initial) == set(final), "policy state keys mismatch")
    parameter_count = 0
    squared_l2 = 0.0
    max_abs = 0.0
    for key in initial:
        require(torch.is_tensor(initial[key]) and torch.is_tensor(final[key]), f"policy tensor missing: {key}")
        require(initial[key].shape == final[key].shape, f"policy tensor shape mismatch: {key}")
        delta = final[key] - initial[key]
        require(bool(torch.isfinite(delta).all()), f"nonfinite policy delta: {key}")
        parameter_count += delta.numel()
        squared_l2 += float((delta * delta).sum())
        max_abs = max(max_abs, float(delta.abs().max()))
    return {
        "finite": True,
        "nonzero": max_abs > 0.0,
        "delta_max_abs": max_abs,
        "delta_l2": math.sqrt(squared_l2),
        "parameter_count_compared": parameter_count,
    }


def validate_policy_update(delta: dict[str, Any], *, stage: str) -> dict[str, Any]:
    require(stage in {"R4", "R5"}, "unknown policy update stage")
    require(delta.get("nonzero") is True, f"{stage} policy did not update")
    require(
        delta.get("parameter_count_compared") == EXPECTED_POLICY_PARAMETER_COUNT,
        f"{stage} policy parameter count mismatch",
    )
    return delta


def validate_progression_and_checkpoints(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    stage: str,
    updates: int,
    env_steps_per_update: int,
    minibatches_per_epoch: int,
    checkpoint_updates: list[int],
    expected_progression_sha: str | None = None,
    expected_checkpoint_shas: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, str]]:
    run_dir = manifest_path.parent.resolve()
    progression_ref = manifest.get("progression")
    require(isinstance(progression_ref, dict), "progression reference missing")
    progression_path = exact_path(
        progression_ref.get("path"), run_dir / "progression.json", "progression"
    )
    if expected_progression_sha is not None:
        require(
            progression_ref.get("sha256") == expected_progression_sha,
            "progression anchored SHA mismatch",
        )
    progression_sha = require_sha(progression_path, progression_ref.get("sha256"), "progression")
    progression = load_json(progression_path)
    require(progression.get("artifact") == ARTIFACT, "progression artifact mismatch")
    require(progression.get("case") == CASE, "progression case mismatch")
    require(progression.get("stage") == stage, "progression stage mismatch")
    rows = progression.get("reports")
    flat = validate_reports(
        rows,
        updates=updates,
        minibatches_per_epoch=minibatches_per_epoch,
        env_steps_per_update=env_steps_per_update,
        initial_learning_rate=0.0001,
    )

    references = progression.get("checkpoints")
    require(isinstance(references, list), "checkpoint references missing")
    require(
        [item.get("global_update") for item in references] == checkpoint_updates,
        "checkpoint cadence mismatch",
    )
    states: dict[int, dict[str, Any]] = {}
    checkpoint_shas: dict[str, str] = {}
    for reference, update in zip(references, checkpoint_updates):
        filename = "initial.pt" if update == 0 else f"update{update:06d}.pt"
        env_steps = update * env_steps_per_update
        expected_sha = None if expected_checkpoint_shas is None else expected_checkpoint_shas[
            "initial" if update == 0 else f"update{update:06d}"
        ]
        state, observed_sha = validate_checkpoint_reference(
            reference,
            expected_path=run_dir / filename,
            update=update,
            env_steps=env_steps,
            expected_sha=expected_sha,
        )
        validate_checkpoint_lineage(
            state,
            manifest=manifest,
            update=update,
            env_steps=env_steps,
        )
        states[update] = state
        checkpoint_shas[filename] = observed_sha

    require(manifest.get("update0_checkpoint") == references[0], "manifest update0 cross-link mismatch")
    final_reference = progression.get("final_checkpoint")
    require(manifest.get("final_checkpoint") == final_reference, "manifest final cross-link mismatch")
    final_sha = None if expected_checkpoint_shas is None else expected_checkpoint_shas["final"]
    final_state, observed_final_sha = validate_checkpoint_reference(
        final_reference,
        expected_path=run_dir / "final.pt",
        update=updates,
        env_steps=updates * env_steps_per_update,
        expected_sha=final_sha,
    )
    validate_checkpoint_lineage(
        final_state,
        manifest=manifest,
        update=updates,
        env_steps=updates * env_steps_per_update,
    )
    require(
        checkpoint_equal(states[updates], final_state),
        "final checkpoint differs from final periodic checkpoint",
    )
    checkpoint_shas["final.pt"] = observed_final_sha
    return progression, flat, {**states, -1: final_state}, {
        "progression": progression_sha,
        **checkpoint_shas,
    }


def compute_metrics(minibatches: list[dict[str, Any]]) -> dict[str, float]:
    require(minibatches, "optimizer metrics require minibatches")
    approx = [float(item["approx_kl"]) for item in minibatches]
    clips = [float(item["clip_fraction"]) for item in minibatches]

    def nearest_rank(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        return ordered[math.ceil(probability * len(ordered)) - 1]

    return {
        "approx_kl_mean": sum(approx) / len(approx),
        "approx_kl_p95": nearest_rank(approx, 0.95),
        "approx_kl_max": max(approx),
        "clip_fraction_mean": sum(clips) / len(clips),
        "clip_fraction_p95": nearest_rank(clips, 0.95),
    }


def metrics_pass(metrics: dict[str, float]) -> bool:
    return all(metrics[key] <= threshold for key, threshold in THRESHOLDS.items())


def check_smoke(run_path: Path, no_update_path: Path) -> dict[str, Any]:
    require(run_path.resolve() == R4_MANIFEST_PATH.resolve(), "noncanonical R4 manifest path")
    require_sha(run_path, R4_ARTIFACT_SHA256["run_manifest"], "R4 run manifest")
    no_update, e2_manifest, fixed = load_fixed_evidence(no_update_path)
    e2_config = e2_manifest["configuration"]
    manifest = load_json(run_path)
    sources = validate_manifest(
        manifest,
        expected_config=repaired_config(e2_config, stage="smoke"),
        e2_config=e2_config,
        no_update=no_update,
    )
    _, minibatches, states, observed = validate_progression_and_checkpoints(
        manifest=manifest,
        manifest_path=run_path,
        stage="smoke",
        updates=2,
        env_steps_per_update=128,
        minibatches_per_epoch=1,
        checkpoint_updates=[0, 1, 2],
        expected_progression_sha=R4_ARTIFACT_SHA256["progression"],
        expected_checkpoint_shas=R4_ARTIFACT_SHA256,
    )
    require(len(minibatches) == 8, "R4 minibatch total mismatch")
    delta = validate_policy_update(
        policy_delta(states[0]["policy"], states[-1]["policy"]),
        stage="R4",
    )

    e2_initial_ref = e2_manifest.get("update0_checkpoint")
    e2_initial_state, _ = validate_checkpoint_reference(
        e2_initial_ref,
        expected_path=E2_DIR / "initial.pt",
        update=0,
        env_steps=0,
    )
    require(
        checkpoint_equal(e2_initial_state["policy"], states[0]["policy"]),
        "R4 random initialization differs from E2",
    )
    require(states[0]["optimizer"].get("state") == {}, "R4 update0 optimizer is not empty")

    checkpoint_sha = {
        "initial": observed["initial.pt"],
        "update000001": observed["update000001.pt"],
        "update000002": observed["update000002.pt"],
        "final": observed["final.pt"],
    }
    require(checkpoint_sha == {key: R4_ARTIFACT_SHA256[key] for key in checkpoint_sha}, "R4 checkpoint anchor drift")
    return {
        "schema_version": 3,
        "task": TASK,
        "case": CASE,
        "variant": REPAIR,
        "stage": "smoke",
        "r4_repaired_smoke_passed": True,
        "artifact_paths": {
            "run_manifest": repo_relative(run_path),
            "progression": repo_relative(SMOKE_DIR / "progression.json"),
            "no_update_correctness_gate": repo_relative(no_update_path),
        },
        "artifact_sha256": {
            "run_manifest": R4_ARTIFACT_SHA256["run_manifest"],
            "progression": observed["progression"],
            "no_update_correctness_gate": NO_UPDATE_SHA256,
            "fixed_artifacts": fixed,
            "checkpoints": checkpoint_sha,
        },
        "controlled_sources": sources,
        "verifier": {
            "path": repo_relative(Path(__file__)),
            "sha256": sha256(Path(__file__)),
        },
        "reports": 2,
        "minibatches": 8,
        "observed_transitions": 256,
        "first_minibatch_identity_verified_per_update": True,
        "parameter_update": delta,
        "random_initialization_matches_e2": True,
        "old_rejected_E3a_preserved": True,
        "no_update_gate_preserved": True,
        "claim_boundary": {
            "r5_training_started": False,
            "task072_passed": False,
            "task048_checkpoint_used": False,
            "external_downloads_used": False,
            "h200_used": False,
        },
    }


def validate_r4_gate_payload(gate: dict[str, Any]) -> dict[str, Any]:
    recomputed = check_smoke(R4_MANIFEST_PATH, NO_UPDATE_PATH)
    require(gate == recomputed, "R4 gate differs from deterministic recomputation")
    return recomputed


def require_r4_gate(gate_path: Path) -> dict[str, Any]:
    require(gate_path.resolve() == R4_GATE_PATH.resolve(), "noncanonical R4 gate path")
    return validate_r4_gate_payload(load_json(gate_path))


def check_optimizer(run_path: Path, r4_gate_path: Path) -> dict[str, Any]:
    gate = require_r4_gate(r4_gate_path)
    require(run_path.resolve() == R5_MANIFEST_PATH.resolve(), "noncanonical R5 manifest path")
    no_update, e2_manifest, fixed = load_fixed_evidence(NO_UPDATE_PATH)
    e2_config = e2_manifest["configuration"]
    manifest = load_json(run_path)
    expected_config = repaired_config(e2_config, stage="pilot")
    require(
        set(config_diff(e2_config, expected_config)) == CONFIG_DIFF_PATHS,
        "repaired optimizer config diff contract drift",
    )
    r4_manifest = load_json(R4_MANIFEST_PATH)
    sources = validate_manifest(
        manifest,
        expected_config=expected_config,
        e2_config=e2_config,
        no_update=no_update,
        expected_static_lineage=r4_manifest["static_lineage"],
    )
    _, minibatches, states, observed = validate_progression_and_checkpoints(
        manifest=manifest,
        manifest_path=run_path,
        stage="pilot",
        updates=1000,
        env_steps_per_update=2048,
        minibatches_per_epoch=8,
        checkpoint_updates=[0, 200, 400, 600, 800, 1000],
    )
    require(len(minibatches) == 32000, "R5 minibatch total mismatch")

    r4_initial_reference = r4_manifest["update0_checkpoint"]
    r4_initial_state, _ = validate_checkpoint_reference(
        r4_initial_reference,
        expected_path=SMOKE_DIR / "initial.pt",
        update=0,
        env_steps=0,
        expected_sha=R4_ARTIFACT_SHA256["initial"],
    )
    require(
        checkpoint_equal(states[0]["policy"], r4_initial_state["policy"]),
        "R5 update0 is not the fixed random initialization",
    )
    require(states[0]["optimizer"].get("state") == {}, "R5 update0 optimizer is not empty")
    delta = validate_policy_update(
        policy_delta(states[0]["policy"], states[-1]["policy"]),
        stage="R5",
    )

    metrics = compute_metrics(minibatches)
    passed = metrics_pass(metrics)
    checkpoint_sha = {key: value for key, value in observed.items() if key != "progression"}
    return {
        "schema_version": 3,
        "task": TASK,
        "case": CASE,
        "variant": REPAIR,
        "parent_variant": "E2_reward_dt",
        "r2_e3a_optimizer_gate_passed": passed,
        "r2_next_variant_allowed": (
            "E4a_roll_authority/E4b_contact_geometry" if passed else None
        ),
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "config_diff_paths": sorted(CONFIG_DIFF_PATHS),
        "progression_verified": True,
        "early_stop_fraction": 0.0,
        "artifact_sha256": {
            "run_manifest": sha256(run_path),
            "progression": observed["progression"],
            "r4_gate": sha256(r4_gate_path),
            "fixed_artifacts": fixed,
            "checkpoints": checkpoint_sha,
        },
        "controlled_sources": sources,
        "verifier": {
            "path": repo_relative(Path(__file__)),
            "sha256": sha256(Path(__file__)),
        },
        "parameter_update": delta,
        "claim_boundary": {
            "mechanistic_gate_only": True,
            "r5_training_started": True,
            "e3b_started": False,
            "e4_started": False,
            "task072_passed": False,
            "task048_checkpoint_used": False,
            "external_downloads_used": False,
            "h200_used": False,
        },
        "r4_gate": gate,
    }


def failure_payload(*, gate: str, error: Exception, r5_training_started: bool = False) -> dict[str, Any]:
    payload = {
        "schema_version": 3,
        "task": TASK,
        "case": CASE,
        "variant": REPAIR,
        "failure_reasons": [str(error)],
        "verifier": {
            "path": repo_relative(Path(__file__)),
            "sha256": sha256(Path(__file__)),
        },
        "claim_boundary": {
            "task072_passed": False,
            "e3b_started": False,
            "e4_started": False,
        },
    }
    if gate == "smoke":
        payload["r4_repaired_smoke_passed"] = False
        payload["claim_boundary"]["r5_training_started"] = False
    else:
        payload["r2_e3a_optimizer_gate_passed"] = False
        payload["r2_next_variant_allowed"] = None
        payload["claim_boundary"]["r5_training_started"] = r5_training_started
    return payload


def verify_smoke(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    try:
        guard_output(output, [args.run_manifest, args.no_update_gate])
    except Exception:
        return 1
    try:
        result = check_smoke(args.run_manifest.resolve(), args.no_update_gate.resolve())
    except Exception as error:
        result = failure_payload(gate="smoke", error=error)
    write_json(output, result)
    return 0 if result.get("r4_repaired_smoke_passed") is True else 1


def verify_optimizer(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    try:
        guard_output(output, [args.run_manifest, args.r4_gate])
    except Exception:
        return 1
    try:
        result = check_optimizer(args.run_manifest.resolve(), args.r4_gate.resolve())
    except Exception as error:
        result = failure_payload(
            gate="optimizer",
            error=error,
            r5_training_started=args.run_manifest.is_file(),
        )
    write_json(output, result)
    return 0 if result.get("r2_e3a_optimizer_gate_passed") is True else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("repaired-smoke")
    smoke.add_argument("--run-manifest", type=Path, required=True)
    smoke.add_argument("--no-update-gate", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.set_defaults(function=verify_smoke)
    optimizer = commands.add_parser("repaired-optimizer")
    optimizer.add_argument("--run-manifest", type=Path, required=True)
    optimizer.add_argument("--r4-gate", type=Path, required=True)
    optimizer.add_argument("--output", type=Path, required=True)
    optimizer.set_defaults(function=verify_optimizer)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
