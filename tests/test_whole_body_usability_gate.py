from __future__ import annotations

import pytest

from h200_locomotion_lab.tools.whole_body_usability_gate import run_usability_gate


def test_small_procedural_usability_gate_covers_reset_mapping_and_step() -> None:
    pytest.importorskip("mujoco")
    summary = run_usability_gate(seeds=1, duration_seconds=0.04)

    assert summary["record_count"] == 2
    assert summary["passed"] is True
    assert summary["passed_records"] == 2
    assert all(record["mapping_pass"] for record in summary["records"])
    assert all(record["actuator_target_pass"] for record in summary["records"])
    assert all(record["controlled_step_pass"] for record in summary["records"])
