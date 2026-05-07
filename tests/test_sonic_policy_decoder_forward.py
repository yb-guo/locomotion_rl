from pathlib import Path

import pytest

from h200_locomotion_lab.tools.sonic_policy_decoder_forward import (
    is_finite,
    read_obs_csv,
    vector_range,
    zero_obs,
)


def test_zero_obs_has_requested_dimension() -> None:
    assert zero_obs(4) == (0.0, 0.0, 0.0, 0.0)


def test_zero_obs_rejects_non_positive_dimension() -> None:
    with pytest.raises(ValueError, match="obs_dim must be positive"):
        zero_obs(0)


def test_read_obs_csv_with_header() -> None:
    row = read_obs_csv(_fixture_path("obs_994_with_header.csv"), 994)

    assert len(row) == 994
    assert row[0] == 0.0
    assert row[-1] == 0.993


def test_read_obs_csv_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match="expected 994 obs values"):
        read_obs_csv(_fixture_path("obs_wrong_width.csv"), 994)


def test_vector_range_and_finite_check() -> None:
    assert vector_range((-2.0, 0.0, 3.0)) == (-2.0, 3.0, 3.0)
    assert is_finite((0.0, 1.0))


def _fixture_path(name: str) -> Path:
    return Path(__file__).with_name("fixtures") / name
