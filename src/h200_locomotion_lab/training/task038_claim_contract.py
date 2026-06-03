"""Task038 fake-evidence claim gate.

This module intentionally stays independent of simulators and training stacks.
It evaluates small local evidence rows with the same shape later H200 JSON
summaries must provide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


DEFAULT_SPEEDS_MPS = (0.4, 1.2, 2.0)
DEFAULT_BASELINE_POLICIES = ("MLP", "GRU", "AdaptK160")
DEFAULT_HELDOUT_CONDITIONS = (
    "link_length",
    "mass_com_inertia",
    "motor_dynamics",
    "combined",
)


@dataclass(frozen=True, slots=True)
class ClaimGate:
    required_seen_variants: int = 2
    required_heldout_variants: int = 4
    required_speeds_mps: tuple[float, ...] = DEFAULT_SPEEDS_MPS
    min_seeds: int = 3
    min_inner_trials: int = 4
    txl_policy: str = "TXL"
    baseline_policies: tuple[str, ...] = DEFAULT_BASELINE_POLICIES
    required_heldout_conditions: tuple[str, ...] = DEFAULT_HELDOUT_CONDITIONS
    heldout_split: str = "heldout"
    seen_split: str = "seen"
    min_heldout_final_margin_pp: float = 10.0
    max_seen_final_regression_pp: float = 5.0
    metric_pass_threshold: float = 1.0
    pass_keys: tuple[str, ...] = ("passed", "pass", "success", "is_success")
    metric_keys: tuple[str, ...] = ("pass_metric", "metric", "score")

    @property
    def required_policies(self) -> tuple[str, ...]:
        return (*self.baseline_policies, self.txl_policy)

    @property
    def final_trial_index(self) -> int:
        return self.min_inner_trials - 1


@dataclass(frozen=True, slots=True)
class EvidenceRow:
    policy: str
    variant_id: str
    split: str
    speed_mps: float
    seed: int
    trial_index: int
    passed: bool
    heldout_condition: str | None = None


@dataclass(frozen=True, slots=True)
class ClaimResult:
    """Local fake-evidence shape gate result, not an H200 reproduction verdict."""

    passed: bool
    missing: tuple[str, ...] = ()
    insufficient: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    heldout_final_pass_rates: Mapping[str, float] = field(default_factory=dict)
    seen_final_pass_rates: Mapping[str, float] = field(default_factory=dict)
    aggregate_pass_rates: Mapping[str, float] = field(default_factory=dict)

    @property
    def reasons(self) -> tuple[str, ...]:
        return (*self.missing, *self.insufficient, *self.failures)


def evaluate_claim(
    rows: Iterable[Mapping[str, Any] | EvidenceRow],
    gate: ClaimGate | None = None,
) -> ClaimResult:
    """Evaluate fake Task038 claim evidence.

    The claim uses final-trial pass rates. Aggregate pass rates are returned for
    diagnostics only and cannot make a failed final-trial claim pass. The
    returned ``ClaimResult.passed`` means the local evidence shape and numeric
    claim gate passed; it is not a full H200 reproduction pass.
    """

    gate = gate or ClaimGate()
    evidence = tuple(_normalize_row(row, gate) for row in rows)
    missing, insufficient = _coverage_issues(evidence, gate)

    heldout_final_rates = _pass_rates(
        evidence,
        split=gate.heldout_split,
        trial_index=gate.final_trial_index,
    )
    seen_final_rates = _pass_rates(
        evidence,
        split=gate.seen_split,
        trial_index=gate.final_trial_index,
    )
    aggregate_rates = _pass_rates(evidence)

    failures = []
    txl_heldout = heldout_final_rates.get(gate.txl_policy)
    baseline_heldout = {
        policy: heldout_final_rates[policy]
        for policy in gate.baseline_policies
        if policy in heldout_final_rates
    }
    if txl_heldout is None:
        missing.append(f"missing heldout final-trial rows for policy={gate.txl_policy}")
    elif not baseline_heldout:
        missing.append("missing heldout final-trial rows for non-TXL baselines")
    else:
        best_policy, best_rate = max(baseline_heldout.items(), key=lambda item: item[1])
        margin_pp = (txl_heldout - best_rate) * 100.0
        if margin_pp < gate.min_heldout_final_margin_pp:
            failures.append(
                "heldout final-trial TXL margin "
                f"{margin_pp:.1f}pp < {gate.min_heldout_final_margin_pp:.1f}pp "
                f"over best baseline policy={best_policy}"
            )

    txl_seen = seen_final_rates.get(gate.txl_policy)
    baseline_seen = {
        policy: seen_final_rates[policy]
        for policy in gate.baseline_policies
        if policy in seen_final_rates
    }
    if txl_seen is None:
        missing.append(f"missing seen final-trial rows for policy={gate.txl_policy}")
    elif not baseline_seen:
        missing.append("missing seen final-trial rows for non-TXL baselines")
    else:
        best_policy, best_rate = max(baseline_seen.items(), key=lambda item: item[1])
        regression_pp = (best_rate - txl_seen) * 100.0
        if regression_pp > gate.max_seen_final_regression_pp:
            failures.append(
                "seen final-trial TXL regression "
                f"{regression_pp:.1f}pp > {gate.max_seen_final_regression_pp:.1f}pp "
                f"behind best baseline policy={best_policy}"
            )

    return ClaimResult(
        passed=not missing and not insufficient and not failures,
        missing=tuple(missing),
        insufficient=tuple(insufficient),
        failures=tuple(failures),
        heldout_final_pass_rates=heldout_final_rates,
        seen_final_pass_rates=seen_final_rates,
        aggregate_pass_rates=aggregate_rates,
    )


def _normalize_row(row: Mapping[str, Any] | EvidenceRow, gate: ClaimGate) -> EvidenceRow:
    if isinstance(row, EvidenceRow):
        return row

    try:
        passed = _row_passed(row, gate)
        return EvidenceRow(
            policy=str(row["policy"]),
            variant_id=str(row["variant_id"]),
            split=str(row["split"]),
            speed_mps=float(row["speed_mps"]),
            seed=int(row["seed"]),
            trial_index=int(row["trial_index"]),
            passed=passed,
            heldout_condition=_optional_condition(row),
        )
    except KeyError as exc:
        raise ValueError(f"evidence row missing required key: {exc.args[0]}") from exc


def _row_passed(row: Mapping[str, Any], gate: ClaimGate) -> bool:
    for key in gate.pass_keys:
        if key in row:
            return bool(row[key])
    for key in gate.metric_keys:
        if key in row:
            return float(row[key]) >= gate.metric_pass_threshold
    raise ValueError(
        "evidence row needs a boolean pass key or metric key "
        f"(pass_keys={gate.pass_keys}, metric_keys={gate.metric_keys})"
    )


def _coverage_issues(
    rows: tuple[EvidenceRow, ...],
    gate: ClaimGate,
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    insufficient: list[str] = []

    by_policy = {row.policy for row in rows}
    for policy in gate.required_policies:
        if policy not in by_policy:
            missing.append(f"missing required policy={policy}")

    split_variants = _split_variants(rows, gate)
    policy_variants = _policy_variants(rows, gate)
    _check_same_variant_matrix(split_variants, policy_variants, gate, missing)
    _check_variant_count(
        split_variants,
        gate.seen_split,
        gate.required_seen_variants,
        insufficient,
    )
    _check_variant_count(
        split_variants,
        gate.heldout_split,
        gate.required_heldout_variants,
        insufficient,
    )
    _check_heldout_conditions(rows, split_variants, gate, missing)

    for policy in gate.required_policies:
        policy_rows = [row for row in rows if row.policy == policy]
        if not policy_rows:
            continue
        for split in (gate.seen_split, gate.heldout_split):
            for variant_id in sorted(split_variants[split]):
                variant_rows = [
                    row
                    for row in policy_rows
                    if row.split == split and row.variant_id == variant_id
                ]
                if not variant_rows:
                    missing.append(
                        f"missing variant_id={variant_id} for policy={policy} split={split}"
                    )
                    continue
                for speed in gate.required_speeds_mps:
                    speed_rows = [
                        row for row in variant_rows if _same_speed(row.speed_mps, speed)
                    ]
                    if not speed_rows:
                        missing.append(
                            "missing speed_mps="
                            f"{speed:g} for policy={policy} split={split} "
                            f"variant_id={variant_id}"
                        )
                        continue
                    seeds = {row.seed for row in speed_rows}
                    if len(seeds) < gate.min_seeds:
                        insufficient.append(
                            "insufficient seeds "
                            f"policy={policy} split={split} variant_id={variant_id} "
                            f"speed_mps={speed:g} found={len(seeds)} "
                            f"required={gate.min_seeds}"
                        )
                    for seed in sorted(seeds):
                        seed_rows = [row for row in speed_rows if row.seed == seed]
                        trials = {row.trial_index for row in seed_rows}
                        required_trials = set(range(gate.min_inner_trials))
                        missing_trials = sorted(required_trials - trials)
                        if missing_trials:
                            missing.append(
                                "missing trials "
                                f"policy={policy} split={split} variant_id={variant_id} "
                                f"speed_mps={speed:g} seed={seed} "
                                f"trial_index={missing_trials}"
                            )

    return missing, insufficient


def _optional_condition(row: Mapping[str, Any]) -> str | None:
    condition = row.get("heldout_condition", row.get("condition"))
    if condition is None:
        return None
    return str(condition)


def _split_variants(
    rows: tuple[EvidenceRow, ...],
    gate: ClaimGate,
) -> dict[str, set[str]]:
    variants: dict[str, set[str]] = {
        gate.seen_split: set(),
        gate.heldout_split: set(),
    }
    for row in rows:
        if row.split in variants:
            variants[row.split].add(row.variant_id)
    return variants


def _check_same_variant_matrix(
    split_variants: Mapping[str, set[str]],
    policy_variants_by_split: Mapping[tuple[str, str], set[str]],
    gate: ClaimGate,
    missing: list[str],
) -> None:
    for split, expected_variants in split_variants.items():
        for policy in gate.required_policies:
            policy_variants = policy_variants_by_split.get((policy, split), set())
            missing_variants = sorted(expected_variants - policy_variants)
            extra_variants = sorted(policy_variants - expected_variants)
            if missing_variants:
                missing.append(
                    f"policy={policy} split={split} missing matrix variants "
                    f"variant_id={missing_variants}"
                )
            if extra_variants:
                missing.append(
                    f"policy={policy} split={split} has out-of-matrix variants "
                    f"variant_id={extra_variants}"
                )


def _policy_variants(
    rows: tuple[EvidenceRow, ...],
    gate: ClaimGate,
) -> dict[tuple[str, str], set[str]]:
    variants: dict[tuple[str, str], set[str]] = {
        (policy, split): set()
        for policy in gate.required_policies
        for split in (gate.seen_split, gate.heldout_split)
    }
    for row in rows:
        key = (row.policy, row.split)
        if key in variants:
            variants[key].add(row.variant_id)
    return variants


def _check_heldout_conditions(
    rows: tuple[EvidenceRow, ...],
    split_variants: Mapping[str, set[str]],
    gate: ClaimGate,
    missing: list[str],
) -> None:
    if not gate.required_heldout_conditions:
        return

    conditions_by_variant: dict[str, set[str]] = {}
    for row in rows:
        if row.split != gate.heldout_split:
            continue
        if row.heldout_condition is None:
            missing.append(
                f"missing heldout_condition for heldout variant_id={row.variant_id}"
            )
            continue
        conditions_by_variant.setdefault(row.variant_id, set()).add(row.heldout_condition)

    for variant_id in sorted(split_variants[gate.heldout_split]):
        conditions = conditions_by_variant.get(variant_id, set())
        if not conditions:
            missing.append(f"missing heldout_condition for heldout variant_id={variant_id}")
        elif len(conditions) > 1:
            missing.append(
                f"conflicting heldout_condition for heldout variant_id={variant_id}: "
                f"{sorted(conditions)}"
            )

    present_conditions = {
        next(iter(conditions))
        for conditions in conditions_by_variant.values()
        if len(conditions) == 1
    }
    for condition in gate.required_heldout_conditions:
        if condition not in present_conditions:
            missing.append(f"missing heldout condition={condition}")


def _check_variant_count(
    split_variants: Mapping[str, set[str]],
    split: str,
    required: int,
    insufficient: list[str],
) -> None:
    variants = split_variants[split]
    if len(variants) < required:
        insufficient.append(
            f"insufficient {split} variants found={len(variants)} required={required}"
        )


def _pass_rates(
    rows: tuple[EvidenceRow, ...],
    *,
    split: str | None = None,
    trial_index: int | None = None,
) -> dict[str, float]:
    grouped: dict[str, list[bool]] = {}
    for row in rows:
        if split is not None and row.split != split:
            continue
        if trial_index is not None and row.trial_index != trial_index:
            continue
        grouped.setdefault(row.policy, []).append(row.passed)
    return {
        policy: sum(1 for passed in values if passed) / len(values)
        for policy, values in sorted(grouped.items())
        if values
    }


def _same_speed(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-9
