"""Task038 G1-like heldout multi-trial eval JSON contract.

This module is simulator-free. It validates a reviewable JSON summary shape
before any H200 training/eval runner exists, then delegates the full superiority
claim gate to ``task038_claim_contract.evaluate_claim``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from h200_locomotion_lab.robots.g1like_morphology import (
    generate_g1like_morphology_manifest,
    validate_g1like_morphology_manifest,
)
from h200_locomotion_lab.training.task038_claim_contract import (
    ClaimGate,
    ClaimResult,
    evaluate_claim,
)


ATTEMPTED = "attempted"
NOT_ATTEMPTED = "not_attempted"
CONTROL_REFERENCE = "control_reference"
PRIMARY_EVAL = "primary_eval"


@dataclass(frozen=True, slots=True)
class TrialMetrics:
    trial_index: int
    passed: bool
    pass_metric: float
    fall_ratio: float = 0.0
    velocity_tracking: float = 0.0
    root_z: float = 0.0
    action_smoothness: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrialMetrics":
        return cls(
            trial_index=int(data["trial_index"]),
            passed=bool(data["passed"]),
            pass_metric=float(data.get("pass_metric", 1.0 if data["passed"] else 0.0)),
            fall_ratio=float(data.get("fall_ratio", 0.0)),
            velocity_tracking=float(data.get("velocity_tracking", 0.0)),
            root_z=float(data.get("root_z", 0.0)),
            action_smoothness=float(data.get("action_smoothness", 0.0)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "passed": self.passed,
            "pass_metric": self.pass_metric,
            "fall_ratio": self.fall_ratio,
            "velocity_tracking": self.velocity_tracking,
            "root_z": self.root_z,
            "action_smoothness": self.action_smoothness,
        }


@dataclass(frozen=True, slots=True)
class EvalPolicyRow:
    policy: str
    status: str
    variant_id: str
    split: str
    speed_mps: float
    seed: int
    heldout_condition: str | None = None
    trials: tuple[TrialMetrics, ...] = ()
    skipped_reason: str | None = None
    evidence_role: str = PRIMARY_EVAL
    construction_only: bool = False
    txl_debug: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EvalPolicyRow":
        status = str(data.get("status", ATTEMPTED))
        return cls(
            policy=str(data["policy"]),
            status=status,
            variant_id=str(data["variant_id"]),
            split=_normalize_split(str(data["split"])),
            speed_mps=float(data["speed_mps"]),
            seed=int(data["seed"]),
            heldout_condition=_optional_string(data.get("heldout_condition", data.get("condition"))),
            trials=tuple(TrialMetrics.from_mapping(trial) for trial in data.get("trials", ())),
            skipped_reason=_optional_string(data.get("skipped_reason")),
            evidence_role=str(data.get("evidence_role", PRIMARY_EVAL)),
            construction_only=bool(data.get("construction_only", False)),
            txl_debug=dict(data.get("txl_debug", {})),
        )

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "policy": self.policy,
            "status": self.status,
            "variant_id": self.variant_id,
            "split": self.split,
            "speed_mps": self.speed_mps,
            "seed": self.seed,
            "trials": [trial.to_json() for trial in self.trials],
            "evidence_role": self.evidence_role,
            "construction_only": self.construction_only,
        }
        if self.heldout_condition is not None:
            data["heldout_condition"] = self.heldout_condition
        if self.skipped_reason is not None:
            data["skipped_reason"] = self.skipped_reason
        if self.txl_debug:
            data["txl_debug"] = dict(self.txl_debug)
        return data

    @property
    def is_attempted_primary(self) -> bool:
        return self.status == ATTEMPTED and self.evidence_role != CONTROL_REFERENCE

    @property
    def matrix_key(self) -> tuple[str, float, int]:
        return (self.variant_id, self.speed_mps, self.seed)

    @property
    def trial_indices(self) -> tuple[int, ...]:
        return tuple(sorted(trial.trial_index for trial in self.trials))

    @property
    def trial0(self) -> TrialMetrics:
        return _trial_by_index(self.trials, 0)

    @property
    def final_trial(self) -> TrialMetrics:
        if not self.trials:
            raise ValueError(f"attempted row for policy={self.policy} has no trials")
        return max(self.trials, key=lambda trial: trial.trial_index)

    @property
    def trial0_to_final_improvement(self) -> float:
        return self.final_trial.pass_metric - self.trial0.pass_metric


@dataclass(frozen=True, slots=True)
class EvalSummary:
    summary_id: str
    morphology_manifest: Mapping[str, Any]
    rows: tuple[EvalPolicyRow, ...]
    command: str
    hardware_note: str
    json_summary_path: str | None = None
    git_ref: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EvalSummary":
        return cls(
            summary_id=str(data["summary_id"]),
            morphology_manifest=dict(data["morphology_manifest"]),
            rows=tuple(EvalPolicyRow.from_mapping(row) for row in data["rows"]),
            command=str(data.get("command", "")),
            hardware_note=str(data.get("hardware_note", "")),
            json_summary_path=_optional_string(data.get("json_summary_path")),
            git_ref=_optional_string(data.get("git_ref")),
        )

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema": "task038.g1like_heldout_multitrial_eval.v1",
            "summary_id": self.summary_id,
            "morphology_manifest": dict(self.morphology_manifest),
            "rows": [row.to_json() for row in self.rows],
            "command": self.command,
            "hardware_note": self.hardware_note,
        }
        if self.json_summary_path is not None:
            data["json_summary_path"] = self.json_summary_path
        if self.git_ref is not None:
            data["git_ref"] = self.git_ref
        return data


@dataclass(frozen=True, slots=True)
class EvalContractResult:
    schema_passed: bool
    comparable: bool
    local_passed: bool
    can_attempt_superiority_claim: bool
    missing: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    attempted_policies: tuple[str, ...] = ()
    not_attempted_policies: Mapping[str, str] = field(default_factory=dict)
    trial_summaries: tuple[Mapping[str, Any], ...] = ()
    failure_cases: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    claim_result: ClaimResult | None = None

    @property
    def reasons(self) -> tuple[str, ...]:
        return (*self.missing, *self.failures)


def validate_eval_summary(
    summary: Mapping[str, Any] | EvalSummary,
    *,
    required_baselines: tuple[str, ...] = ("MLP", "GRU", "AdaptK160"),
    txl_policy: str = "TXL",
) -> EvalContractResult:
    """Validate the tiny local JSON summary and comparability contract."""

    summary_obj = summary if isinstance(summary, EvalSummary) else EvalSummary.from_mapping(summary)
    missing: list[str] = []
    failures: list[str] = []

    try:
        validate_g1like_morphology_manifest(summary_obj.morphology_manifest)
    except ValueError as exc:
        failures.append(f"invalid morphology_manifest: {exc}")

    if not summary_obj.command:
        missing.append("missing command")
    if not summary_obj.hardware_note:
        missing.append("missing hardware_note")

    variant_lookup = _variant_lookup(summary_obj.morphology_manifest)
    for row in summary_obj.rows:
        _validate_row(row, variant_lookup, missing, failures)

    primary_rows = tuple(row for row in summary_obj.rows if row.evidence_role != CONTROL_REFERENCE)
    attempted = tuple(row for row in primary_rows if row.status == ATTEMPTED)
    skipped = tuple(row for row in primary_rows if row.status == NOT_ATTEMPTED)
    attempted_policies = tuple(sorted({row.policy for row in attempted}))
    skipped_by_policy = {
        row.policy: row.skipped_reason or "not_attempted"
        for row in skipped
    }

    if txl_policy not in attempted_policies:
        missing.append(f"missing attempted TXL policy={txl_policy}")
    attempted_baselines = [policy for policy in required_baselines if policy in attempted_policies]
    if not attempted_baselines:
        missing.append("missing attempted non-TXL baseline")
    for policy in required_baselines:
        if policy not in attempted_policies and policy not in skipped_by_policy:
            missing.append(f"missing baseline policy={policy}; add explicit not_attempted row")

    comparable_failures = _comparability_failures(attempted)
    failures.extend(comparable_failures)

    trial_summaries = tuple(_trial_summary(row) for row in attempted)
    failure_cases = _failure_cases(attempted)
    construction_only_txl = any(
        row.policy == txl_policy and row.construction_only for row in attempted
    )
    if construction_only_txl:
        failures.append("construction-only TXL row cannot produce superiority claim")

    schema_passed = not missing and not failures
    comparable = not comparable_failures and bool(attempted)
    local_passed = schema_passed and comparable

    return EvalContractResult(
        schema_passed=schema_passed,
        comparable=comparable,
        local_passed=local_passed,
        can_attempt_superiority_claim=local_passed and not construction_only_txl,
        missing=tuple(missing),
        failures=tuple(failures),
        attempted_policies=attempted_policies,
        not_attempted_policies=skipped_by_policy,
        trial_summaries=trial_summaries,
        failure_cases=failure_cases,
    )


def evaluate_full_claim_from_summary(
    summary: Mapping[str, Any] | EvalSummary,
    gate: ClaimGate | None = None,
) -> EvalContractResult:
    """Run local contract checks, then delegate full matrix claim to evaluate_claim."""

    gate = gate or ClaimGate()
    local = validate_eval_summary(
        summary,
        required_baselines=gate.baseline_policies,
        txl_policy=gate.txl_policy,
    )
    if not local.can_attempt_superiority_claim:
        return local

    summary_obj = summary if isinstance(summary, EvalSummary) else EvalSummary.from_mapping(summary)
    claim_rows = tuple(
        claim_row
        for row in summary_obj.rows
        if row.status == ATTEMPTED and row.evidence_role != CONTROL_REFERENCE
        for claim_row in _claim_rows(row)
    )
    claim_result = evaluate_claim(claim_rows, gate)
    return EvalContractResult(
        schema_passed=local.schema_passed,
        comparable=local.comparable,
        local_passed=local.local_passed,
        can_attempt_superiority_claim=local.can_attempt_superiority_claim,
        missing=(*local.missing, *claim_result.missing, *claim_result.insufficient),
        failures=(*local.failures, *claim_result.failures),
        attempted_policies=local.attempted_policies,
        not_attempted_policies=local.not_attempted_policies,
        trial_summaries=local.trial_summaries,
        failure_cases=local.failure_cases,
        claim_result=claim_result,
    )


def build_tiny_comparable_fixture() -> EvalSummary:
    """Return a tiny local fixture that passes comparability but not the full gate."""

    manifest = generate_g1like_morphology_manifest(
        seed=38,
        train_count=1,
        heldout_conditions=("link_length",),
    )
    seen = next(variant for variant in manifest["variants"] if variant["split"] == "train")
    heldout = next(variant for variant in manifest["variants"] if variant["split"] == "heldout")
    rows: list[EvalPolicyRow] = []
    for variant in (seen, heldout):
        split = _normalize_split(str(variant["split"]))
        condition = None if split == "seen" else str(variant["heldout_condition"])
        for policy in ("MLP", "TXL"):
            rows.append(
                EvalPolicyRow(
                    policy=policy,
                    status=ATTEMPTED,
                    variant_id=str(variant["variant_id"]),
                    split=split,
                    speed_mps=0.4,
                    seed=0,
                    heldout_condition=condition,
                    trials=(
                        TrialMetrics(
                            trial_index=0,
                            passed=policy == "MLP",
                            pass_metric=0.6 if policy == "MLP" else 0.2,
                        ),
                        TrialMetrics(
                            trial_index=1,
                            passed=policy == "TXL",
                            pass_metric=0.4 if policy == "MLP" else 0.9,
                        ),
                    ),
                    txl_debug=_tiny_txl_debug() if policy == "TXL" else {},
                )
            )
    for policy in ("GRU", "AdaptK160"):
        rows.append(
            EvalPolicyRow(
                policy=policy,
                status=NOT_ATTEMPTED,
                variant_id=str(seen["variant_id"]),
                split="seen",
                speed_mps=0.4,
                seed=0,
                skipped_reason="local fixture records explicit skipped baseline",
            )
        )
    return EvalSummary(
        summary_id="task038-005-tiny-local-fixture",
        morphology_manifest=manifest,
        rows=tuple(rows),
        command=(
            "$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider "
            "tests\\test_task038_eval_contract.py"
        ),
        hardware_note="local contract fixture only; no H200 training, video, or simulator launched",
        json_summary_path=None,
        git_ref=None,
    )


def _validate_row(
    row: EvalPolicyRow,
    variant_lookup: Mapping[str, Mapping[str, Any]],
    missing: list[str],
    failures: list[str],
) -> None:
    if row.status not in (ATTEMPTED, NOT_ATTEMPTED):
        failures.append(f"invalid row status={row.status} policy={row.policy}")
    if row.variant_id not in variant_lookup:
        failures.append(f"unknown variant_id={row.variant_id} policy={row.policy}")
        return
    variant = variant_lookup[row.variant_id]
    expected_split = _normalize_split(str(variant["split"]))
    if row.split != expected_split:
        failures.append(
            f"row split mismatch policy={row.policy} variant_id={row.variant_id} "
            f"row={row.split} manifest={expected_split}"
        )
    if row.split == "heldout":
        expected_condition = str(variant["heldout_condition"])
        if row.heldout_condition != expected_condition:
            failures.append(
                f"heldout_condition mismatch policy={row.policy} variant_id={row.variant_id} "
                f"row={row.heldout_condition} manifest={expected_condition}"
            )
    if row.status == ATTEMPTED:
        if not row.trials:
            failures.append(f"attempted row for policy={row.policy} has no trials")
        elif 0 not in row.trial_indices:
            missing.append(f"attempted row for policy={row.policy} missing trial0")
        elif row.final_trial.trial_index <= 0:
            failures.append(
                f"attempted row for policy={row.policy} needs distinct final trial index > 0"
            )
        if row.policy == "TXL" and row.txl_debug:
            _validate_txl_debug(row, failures)
    elif row.status == NOT_ATTEMPTED and not row.skipped_reason:
        missing.append(f"not_attempted policy={row.policy} missing skipped_reason")


def _validate_txl_debug(row: EvalPolicyRow, failures: list[str]) -> None:
    envs = row.txl_debug.get("envs")
    if not isinstance(envs, list):
        failures.append(f"TXL debug missing envs list policy={row.policy}")
        return
    required_env_keys = {
        "env_id",
        "memory_lengths",
        "inner_reset_events",
        "outer_reset_events",
        "incremental_steps",
    }
    for env in envs:
        if not isinstance(env, Mapping) or not required_env_keys.issubset(env):
            failures.append(f"TXL debug env entry missing cache metadata keys policy={row.policy}")
            return


def _comparability_failures(rows: Iterable[EvalPolicyRow]) -> list[str]:
    rows = tuple(rows)
    failures: list[str] = []
    by_policy: dict[str, set[tuple[str, float, int, tuple[int, ...]]]] = {}
    for row in rows:
        by_policy.setdefault(row.policy, set()).add((*row.matrix_key, row.trial_indices))
    if len(by_policy) <= 1:
        return failures
    expected_policy, expected = next(iter(sorted(by_policy.items())))
    for policy, matrix in sorted(by_policy.items()):
        if matrix != expected:
            missing = sorted(expected - matrix)
            extra = sorted(matrix - expected)
            failures.append(
                "attempted policies use different eval matrix "
                f"policy={policy} expected_from={expected_policy} missing={missing} extra={extra}"
            )
    return failures


def _trial_summary(row: EvalPolicyRow) -> dict[str, Any]:
    final = row.final_trial
    return {
        "policy": row.policy,
        "variant_id": row.variant_id,
        "split": row.split,
        "heldout_condition": row.heldout_condition,
        "speed_mps": row.speed_mps,
        "seed": row.seed,
        "trial0": row.trial0.to_json(),
        "final_trial": final.to_json(),
        "final_pass": final.passed,
        "trial0_to_final_improvement": row.trial0_to_final_improvement,
    }


def _failure_cases(rows: Iterable[EvalPolicyRow]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.final_trial.passed:
            continue
        condition = row.heldout_condition or ("seen" if row.split == "seen" else "unknown")
        key = f"{row.variant_id}|{condition}"
        entry = grouped.setdefault(
            key,
            {
                "variant_id": row.variant_id,
                "heldout_condition": condition,
                "final_failures": 0,
            },
        )
        entry["final_failures"] += 1
    return grouped


def _claim_rows(row: EvalPolicyRow) -> Iterable[dict[str, Any]]:
    for trial in row.trials:
        claim_row = {
            "policy": row.policy,
            "variant_id": row.variant_id,
            "split": row.split,
            "speed_mps": row.speed_mps,
            "seed": row.seed,
            "trial_index": trial.trial_index,
            "passed": trial.passed,
            "pass_metric": trial.pass_metric,
        }
        if row.split == "heldout":
            claim_row["heldout_condition"] = row.heldout_condition
        yield claim_row


def _variant_lookup(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(variant["variant_id"]): variant for variant in manifest.get("variants", [])}


def _trial_by_index(trials: tuple[TrialMetrics, ...], trial_index: int) -> TrialMetrics:
    for trial in trials:
        if trial.trial_index == trial_index:
            return trial
    raise ValueError(f"missing trial_index={trial_index}")


def _normalize_split(split: str) -> str:
    if split == "train":
        return "seen"
    return split


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _tiny_txl_debug() -> dict[str, Any]:
    return {
        "num_envs": 1,
        "num_layers": 1,
        "memory_len": 2,
        "envs": [
            {
                "env_id": 0,
                "episode_index": 0,
                "memory_lengths": (2,),
                "inner_reset_events": 1,
                "outer_reset_events": 0,
                "incremental_steps": 2,
                "segments_appended": 2,
                "tokens_appended": 2,
            }
        ],
    }


__all__ = [
    "ATTEMPTED",
    "CONTROL_REFERENCE",
    "EvalContractResult",
    "EvalPolicyRow",
    "EvalSummary",
    "NOT_ATTEMPTED",
    "PRIMARY_EVAL",
    "TrialMetrics",
    "build_tiny_comparable_fixture",
    "evaluate_full_claim_from_summary",
    "validate_eval_summary",
]
