import dataclasses

from h200_locomotion_lab.training import task038_eval_contract as eval_contract
from h200_locomotion_lab.training.task038_claim_contract import ClaimGate
from h200_locomotion_lab.training.task038_eval_contract import (
    ATTEMPTED,
    CONTROL_REFERENCE,
    NOT_ATTEMPTED,
    EvalPolicyRow,
    TrialMetrics,
    build_tiny_comparable_fixture,
    evaluate_full_claim_from_summary,
    validate_eval_summary,
)


def _attempted_rows(summary):
    return [row for row in summary.rows if row.status == ATTEMPTED]


def _replace_row(summary, old_row, new_row):
    return dataclasses.replace(
        summary,
        rows=tuple(new_row if row is old_row else row for row in summary.rows),
    )


def test_tiny_comparable_fixture_passes_local_schema_and_comparability_aggregator():
    summary = build_tiny_comparable_fixture()

    result = validate_eval_summary(summary)

    assert result.local_passed
    assert result.schema_passed
    assert result.comparable
    assert result.attempted_policies == ("MLP", "TXL")
    assert result.not_attempted_policies["GRU"]
    assert result.not_attempted_policies["AdaptK160"]
    assert result.trial_summaries
    txl_summary = next(row for row in result.trial_summaries if row["policy"] == "TXL")
    assert txl_summary["trial0"]["pass_metric"] == 0.2
    assert txl_summary["final_trial"]["pass_metric"] == 0.9
    assert txl_summary["final_pass"] is True
    assert txl_summary["trial0_to_final_improvement"] == 0.7


def test_missing_baseline_without_not_attempted_fails():
    summary = build_tiny_comparable_fixture()
    rows = tuple(row for row in summary.rows if row.policy != "GRU")
    summary = dataclasses.replace(summary, rows=rows)

    result = validate_eval_summary(summary)

    assert not result.local_passed
    assert any("missing baseline policy=GRU" in reason for reason in result.missing)


def test_attempted_policies_on_different_variant_ids_fail():
    summary = build_tiny_comparable_fixture()
    txl_seen = next(row for row in _attempted_rows(summary) if row.policy == "TXL" and row.split == "seen")
    heldout_variant = next(
        variant for variant in summary.morphology_manifest["variants"] if variant["split"] == "heldout"
    )
    broken = dataclasses.replace(
        txl_seen,
        variant_id=heldout_variant["variant_id"],
        split="heldout",
        heldout_condition=heldout_variant["heldout_condition"],
    )
    summary = _replace_row(summary, txl_seen, broken)

    result = validate_eval_summary(summary)

    assert not result.comparable
    assert any("different eval matrix" in reason for reason in result.failures)


def test_construction_only_txl_row_cannot_produce_superiority_claim(monkeypatch):
    summary = build_tiny_comparable_fixture()
    txl_rows = [row for row in _attempted_rows(summary) if row.policy == "TXL"]
    rows = tuple(
        dataclasses.replace(row, construction_only=True) if row in txl_rows else row
        for row in summary.rows
    )
    summary = dataclasses.replace(summary, rows=rows)
    called = False

    def fake_evaluate_claim(rows, gate):
        nonlocal called
        called = True
        raise AssertionError("construction-only TXL must block delegation")

    monkeypatch.setattr(eval_contract, "evaluate_claim", fake_evaluate_claim)

    result = evaluate_full_claim_from_summary(summary)

    assert not called
    assert not result.can_attempt_superiority_claim
    assert any("construction-only TXL" in reason for reason in result.failures)


def test_failure_cases_grouped_by_morphology_condition():
    summary = build_tiny_comparable_fixture()

    result = validate_eval_summary(summary)

    assert result.failure_cases
    heldout_failure = [
        case for case in result.failure_cases.values() if case["heldout_condition"] == "link_length"
    ]
    assert heldout_failure
    assert heldout_failure[0]["final_failures"] == 1


def test_full_claim_matrix_gate_rejects_tiny_fixture_and_delegates(monkeypatch):
    summary = build_tiny_comparable_fixture()
    called = False
    original = eval_contract.evaluate_claim

    def spy_evaluate_claim(rows, gate):
        nonlocal called
        called = True
        return original(rows, gate)

    monkeypatch.setattr(eval_contract, "evaluate_claim", spy_evaluate_claim)

    result = evaluate_full_claim_from_summary(summary)

    assert called
    assert result.claim_result is not None
    assert not result.claim_result.passed
    assert any("insufficient seen variants" in reason for reason in result.missing)
    assert any("insufficient heldout variants" in reason for reason in result.missing)


def test_skipped_not_attempted_entries_are_explicit():
    summary = build_tiny_comparable_fixture()
    skipped = next(row for row in summary.rows if row.status == NOT_ATTEMPTED and row.policy == "GRU")
    summary = _replace_row(summary, skipped, dataclasses.replace(skipped, skipped_reason=None))

    result = validate_eval_summary(summary)

    assert not result.local_passed
    assert any("not_attempted policy=GRU missing skipped_reason" in reason for reason in result.missing)


def test_task037_control_reference_cannot_count_as_heldout_morphology_pass():
    summary = build_tiny_comparable_fixture()
    rows = tuple(row for row in summary.rows if row.policy != "MLP")
    heldout = next(variant for variant in summary.morphology_manifest["variants"] if variant["split"] == "heldout")
    rows = rows + (
        EvalPolicyRow(
            policy="MLP",
            status=ATTEMPTED,
            variant_id=heldout["variant_id"],
            split="heldout",
            speed_mps=0.4,
            seed=0,
            heldout_condition=heldout["heldout_condition"],
            trials=(
                TrialMetrics(trial_index=0, passed=True, pass_metric=1.0),
                TrialMetrics(trial_index=1, passed=True, pass_metric=1.0),
            ),
            evidence_role=CONTROL_REFERENCE,
        ),
    )
    summary = dataclasses.replace(summary, rows=rows)

    local = validate_eval_summary(summary)
    full = evaluate_full_claim_from_summary(summary, ClaimGate(required_speeds_mps=(0.4,)))

    assert not local.local_passed
    assert any("missing attempted non-TXL baseline" in reason for reason in local.missing)
    assert full.claim_result is None


def test_row_split_mismatch_hard_fails_schema_local_and_claim():
    summary = build_tiny_comparable_fixture()
    row = next(row for row in _attempted_rows(summary) if row.policy == "MLP" and row.split == "seen")
    broken = dataclasses.replace(row, split="heldout", heldout_condition="link_length")
    summary = _replace_row(summary, row, broken)

    result = validate_eval_summary(summary)

    assert not result.schema_passed
    assert not result.local_passed
    assert not result.can_attempt_superiority_claim
    assert any("row split mismatch" in reason for reason in result.failures)


def test_heldout_condition_mismatch_hard_fails_schema_local_and_claim():
    summary = build_tiny_comparable_fixture()
    row = next(row for row in _attempted_rows(summary) if row.policy == "MLP" and row.split == "heldout")
    broken = dataclasses.replace(row, heldout_condition="motor_dynamics")
    summary = _replace_row(summary, row, broken)

    result = validate_eval_summary(summary)

    assert not result.schema_passed
    assert not result.local_passed
    assert not result.can_attempt_superiority_claim
    assert any("heldout_condition mismatch" in reason for reason in result.failures)


def test_invalid_txl_debug_metadata_hard_fails_schema_local_and_claim():
    summary = build_tiny_comparable_fixture()
    row = next(row for row in _attempted_rows(summary) if row.policy == "TXL" and row.split == "seen")
    broken = dataclasses.replace(row, txl_debug={"envs": [{"env_id": 0}]})
    summary = _replace_row(summary, row, broken)

    result = validate_eval_summary(summary)

    assert not result.schema_passed
    assert not result.local_passed
    assert not result.can_attempt_superiority_claim
    assert any("TXL debug env entry missing" in reason for reason in result.failures)


def test_attempted_rows_with_only_trial0_fail_distinct_final_trial_requirement():
    summary = build_tiny_comparable_fixture()
    rows = tuple(
        dataclasses.replace(row, trials=(row.trial0,)) if row.status == ATTEMPTED else row
        for row in summary.rows
    )
    summary = dataclasses.replace(summary, rows=rows)

    result = validate_eval_summary(summary)

    assert result.comparable
    assert not result.schema_passed
    assert not result.local_passed
    assert not result.can_attempt_superiority_claim
    assert any("distinct final trial index > 0" in reason for reason in result.failures)
