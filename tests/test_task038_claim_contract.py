from h200_locomotion_lab.training.task038_claim_contract import ClaimGate, evaluate_claim

TINY_GATE = ClaimGate(
    required_seen_variants=1,
    required_heldout_variants=1,
    required_heldout_conditions=("link_length",),
    required_speeds_mps=(0.4,),
    min_seeds=1,
    min_inner_trials=2,
    baseline_policies=("MLP", "GRU"),
)


def _rows(
    *,
    txl_final_passed: bool = True,
    mlp_final_passed: bool = False,
    gru_final_passed: bool = False,
    heldout_condition: str | None = "link_length",
):
    rows = []
    final_passed = {
        "MLP": mlp_final_passed,
        "GRU": gru_final_passed,
        "TXL": txl_final_passed,
    }
    for split, variant_id in (("seen", "seen-a"), ("heldout", "heldout-a")):
        for policy in ("MLP", "GRU", "TXL"):
            for trial_index in (0, 1):
                row = {
                    "policy": policy,
                    "variant_id": variant_id,
                    "split": split,
                    "speed_mps": 0.4,
                    "seed": 0,
                    "trial_index": trial_index,
                    "passed": True if trial_index == 0 else final_passed[policy],
                }
                if split == "heldout" and heldout_condition is not None:
                    row["heldout_condition"] = heldout_condition
                rows.append(row)
    return rows


def test_aggregate_mean_cannot_hide_failed_final_trial_adaptation():
    rows = _rows(txl_final_passed=False, mlp_final_passed=False, gru_final_passed=False)

    result = evaluate_claim(rows, TINY_GATE)

    assert result.aggregate_pass_rates["TXL"] == 0.5
    assert result.heldout_final_pass_rates["TXL"] == 0.0
    assert not result.passed
    assert any("heldout final-trial TXL margin" in reason for reason in result.failures)


def test_txl_heldout_final_rate_without_10pp_margin_fails():
    result = evaluate_claim(
        _rows(txl_final_passed=True, mlp_final_passed=False, gru_final_passed=True),
        TINY_GATE,
    )

    assert result.heldout_final_pass_rates["TXL"] == 1.0
    assert result.heldout_final_pass_rates["GRU"] == 1.0
    assert not result.passed
    assert any("< 10.0pp" in reason for reason in result.failures)


def test_missing_required_coverage_is_reported_instead_of_pass():
    rows = [
        {
            "policy": "TXL",
            "variant_id": "seen-a",
            "split": "seen",
            "speed_mps": 0.4,
            "seed": 0,
            "trial_index": 0,
            "passed": True,
        }
    ]

    result = evaluate_claim(rows, TINY_GATE)

    assert not result.passed
    assert "missing required policy=MLP" in result.missing
    assert "missing required policy=GRU" in result.missing
    assert any("insufficient heldout variants" in reason for reason in result.insufficient)
    assert any("missing heldout condition=link_length" in reason for reason in result.missing)
    assert any("missing trials" in reason for reason in result.missing)


def test_policies_evaluated_on_different_variant_ids_fail():
    rows = _rows()
    for row in rows:
        if row["policy"] == "GRU" and row["split"] == "heldout":
            row["variant_id"] = "heldout-gru-only"

    result = evaluate_claim(rows, TINY_GATE)

    assert not result.passed
    assert any("missing matrix variants" in reason for reason in result.missing)


def test_missing_heldout_condition_fails():
    result = evaluate_claim(_rows(heldout_condition=None), TINY_GATE)

    assert not result.passed
    assert any("missing heldout_condition" in reason for reason in result.missing)
    assert any("missing heldout condition=link_length" in reason for reason in result.missing)


def test_seeds_and_trials_spread_across_variants_fail():
    gate = ClaimGate(
        required_seen_variants=1,
        required_heldout_variants=2,
        required_heldout_conditions=("link_length",),
        required_speeds_mps=(0.4,),
        min_seeds=2,
        min_inner_trials=2,
        baseline_policies=("MLP", "GRU"),
    )
    rows = []
    for policy in ("MLP", "GRU", "TXL"):
        for split, variant_ids in (
            ("seen", ("seen-a",)),
            ("heldout", ("heldout-a", "heldout-b")),
        ):
            for variant_index, variant_id in enumerate(variant_ids):
                for seed in (variant_index,):
                    for trial_index in (0,):
                        row = {
                            "policy": policy,
                            "variant_id": variant_id,
                            "split": split,
                            "speed_mps": 0.4,
                            "seed": seed,
                            "trial_index": trial_index,
                            "passed": policy == "TXL",
                        }
                        if split == "heldout":
                            row["heldout_condition"] = "link_length"
                        rows.append(row)

    result = evaluate_claim(rows, gate)

    assert not result.passed
    assert any("insufficient seeds" in reason for reason in result.insufficient)
    assert any("missing trials" in reason for reason in result.missing)


def test_tiny_two_baseline_relaxed_fixture_can_pass():
    result = evaluate_claim(
        _rows(txl_final_passed=True, mlp_final_passed=False, gru_final_passed=False),
        TINY_GATE,
    )

    assert result.passed
    assert result.missing == ()
    assert result.insufficient == ()
    assert result.failures == ()


def test_default_full_gate_rejects_tiny_fixture():
    result = evaluate_claim(
        _rows(txl_final_passed=True, mlp_final_passed=False, gru_final_passed=False)
    )

    assert not result.passed
    assert any("insufficient seen variants" in reason for reason in result.insufficient)
    assert any("insufficient heldout variants" in reason for reason in result.insufficient)
    assert any("missing required policy=AdaptK160" in reason for reason in result.missing)
    assert any("missing heldout condition=mass_com_inertia" in reason for reason in result.missing)
