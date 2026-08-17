"""Tests for season assignment and the season-aware sampling weights.

This is the paper's central mechanism, so the properties that must hold are
tested directly: every season ends up equally likely, whatever the input
imbalance; a missing season does not silently vanish; and the day-of-year
encoding wraps at the year boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from treecover.data.seasons import (
    LEAF_OFF,
    LEAF_ON,
    TRANSITION,
    date_encoding,
    day_of_year,
    month_from_date,
    season_from_month,
    season_weights,
)


@pytest.mark.parametrize(
    ("month", "expected"),
    [
        (1, LEAF_OFF), (2, LEAF_OFF), (3, LEAF_OFF), (11, LEAF_OFF), (12, LEAF_OFF),
        (4, TRANSITION), (5, TRANSITION), (9, TRANSITION), (10, TRANSITION),
        (6, LEAF_ON), (7, LEAF_ON), (8, LEAF_ON),
    ],
)
def test_every_month_maps_to_its_phenological_stage(month, expected):
    assert season_from_month(month) == expected


def test_unknown_month_falls_back_to_the_minority_class():
    """leaf_off is rarest, so a mislabelled patch is up-weighted rather than
    diluting the majority class."""
    assert season_from_month(None) == LEAF_OFF


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("20240730", 7), ("2024-07-30", 7), ("2024/07/30", 7),
        ("2024-12-01 00:00:00", 12), (None, None), ("", None), ("nonsense", None),
    ],
)
def test_month_parsing_accepts_every_format_the_project_produced(value, expected):
    assert month_from_date(value) == expected


def test_weights_equalise_a_heavily_imbalanced_set():
    """The real case: summer dominates. After weighting, the expected draws
    per season must be equal."""
    dates = ["20240715"] * 900 + ["20240420"] * 80 + ["20240115"] * 20
    weights, counts, per_season = season_weights(dates)

    assert counts == {LEAF_ON: 900, TRANSITION: 80, LEAF_OFF: 20}
    for season in (LEAF_ON, TRANSITION, LEAF_OFF):
        drawn = weights[[season_from_month(month_from_date(d)) == season for d in dates]].sum()
        assert drawn == pytest.approx(len(dates) / 3)


def test_rare_season_gets_the_largest_per_patch_weight():
    dates = ["20240715"] * 900 + ["20240420"] * 80 + ["20240115"] * 20
    _, _, per_season = season_weights(dates)
    assert per_season[LEAF_OFF] > per_season[TRANSITION] > per_season[LEAF_ON]


def test_absent_season_gets_zero_and_does_not_divide_by_zero():
    """A two-season dataset must still balance those two, not be scaled by 1/3."""
    dates = ["20240715"] * 50 + ["20240115"] * 50
    weights, counts, per_season = season_weights(dates)
    assert per_season[TRANSITION] == 0.0
    assert TRANSITION not in counts
    assert weights.sum() == pytest.approx(len(dates))


def test_balanced_input_yields_uniform_weights():
    dates = ["20240715"] * 30 + ["20240420"] * 30 + ["20240115"] * 30
    weights, _, _ = season_weights(dates)
    assert np.allclose(weights, weights[0])


def test_empty_input_is_rejected_with_a_clear_message():
    with pytest.raises(ValueError, match="No patches to weight"):
        season_weights([])


def test_unparseable_dates_are_counted_not_dropped():
    weights, counts, _ = season_weights(["nonsense"] * 5 + ["20240715"] * 5)
    assert len(weights) == 10
    assert counts[LEAF_OFF] == 5


# ── day-of-year encoding ─────────────────────────────────────────────────────


def test_day_of_year_handles_leap_years():
    assert day_of_year("2024-12-31") == (366, 366)
    assert day_of_year("2023-12-31") == (365, 365)


def test_encoding_is_cyclical_across_the_year_boundary():
    """31 Dec and 1 Jan must be neighbours, not opposites — the whole reason
    for a sin/cos encoding rather than a linear day number."""
    dec = date_encoding("2023-12-31", 1, 1)[:, 0, 0]
    jan = date_encoding("2023-01-01", 1, 1)[:, 0, 0]
    jul = date_encoding("2023-07-01", 1, 1)[:, 0, 0]
    assert np.linalg.norm(dec - jan) < np.linalg.norm(dec - jul)


def test_encoding_stays_in_the_unit_interval():
    """Values must sit in [0, 1] so Normalize(0.5, 0.5) maps them to [-1, 1]
    like the imagery bands."""
    for date in ("2024-01-01", "2024-03-21", "2024-06-21", "2024-09-23", "2024-12-31"):
        values = date_encoding(date, 2, 2)
        assert values.min() >= 0.0 and values.max() <= 1.0


def test_missing_date_encodes_to_the_normalisation_midpoint():
    """0.5 normalises to exactly 0 — a missing date contributes nothing
    rather than pointing at an arbitrary time of year."""
    values = date_encoding(None, 3, 3)
    assert np.allclose(values, 0.5)
    assert values.shape == (2, 3, 3)
