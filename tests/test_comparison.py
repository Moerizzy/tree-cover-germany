"""Tests for product comparison and aggregation.

The invariant throughout is area weighting. An unweighted mean of tile
percentages shifts the national figure by more than the differences between
products, so every aggregation is checked against a hand-computed weighted
value rather than against a plain mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from treecover.comparison import (
    GRID_DLAT,
    GRID_DLON,
    OUR_COLUMN,
    aggregate_to_grid,
    difference_table,
    rasterise,
)

OTHER = "clms_tcd2023_treecover_pct"


def tiles(**overrides) -> pd.DataFrame:
    """Four tiles in one small area, with deliberately unequal areas."""
    frame = pd.DataFrame(
        {
            "lon_c": [10.001, 10.002, 10.003, 10.004],
            "lat_c": [52.001, 52.002, 52.003, 52.004],
            "tile_area_km2": [1.0, 1.0, 1.0, 1.0],
            OUR_COLUMN: [10.0, 20.0, 30.0, 40.0],
            OTHER: [12.0, 18.0, 33.0, 37.0],
        }
    )
    for key, value in overrides.items():
        frame[key] = value
    return frame


# ── grid aggregation ─────────────────────────────────────────────────────────


def test_equal_area_tiles_average_plainly():
    grid = aggregate_to_grid(tiles(), [OUR_COLUMN], dlon=1.0, dlat=1.0)
    assert len(grid.frame) == 1
    assert grid.frame[OUR_COLUMN].iloc[0] == pytest.approx(25.0)


def test_aggregation_is_area_weighted_not_a_plain_mean():
    """A 9 km² tile at 10 % and a 1 km² tile at 90 % is 18 %, not 50 %."""
    frame = pd.DataFrame(
        {
            "lon_c": [10.001, 10.002],
            "lat_c": [52.001, 52.002],
            "tile_area_km2": [9.0, 1.0],
            OUR_COLUMN: [10.0, 90.0],
        }
    )
    grid = aggregate_to_grid(frame, [OUR_COLUMN], dlon=1.0, dlat=1.0)
    assert grid.frame[OUR_COLUMN].iloc[0] == pytest.approx(18.0)


def test_missing_values_do_not_count_as_zero():
    """A product with gaps is averaged over what it covers; treating a gap
    as 0 % would understate it."""
    frame = tiles()
    frame.loc[0, OTHER] = np.nan
    grid = aggregate_to_grid(frame, [OTHER], dlon=1.0, dlat=1.0)
    assert grid.frame[OTHER].iloc[0] == pytest.approx((18 + 33 + 37) / 3)


def test_finer_grid_splits_cells():
    """Tiles spread over ~0.4°: one cell at 1°, several at the 1 km scale."""
    frame = pd.DataFrame(
        {
            "lon_c": [10.05, 10.20, 10.35, 10.45],
            "lat_c": [52.05, 52.15, 52.25, 52.35],
            "tile_area_km2": [1.0] * 4,
            OUR_COLUMN: [10.0, 20.0, 30.0, 40.0],
        }
    )
    coarse = aggregate_to_grid(frame, [OUR_COLUMN], dlon=1.0, dlat=1.0)
    fine = aggregate_to_grid(frame, [OUR_COLUMN], GRID_DLON, GRID_DLAT)
    assert len(coarse.frame) == 1
    assert len(fine.frame) == 4


def test_totals_are_restricted_to_tiles_where_ours_is_valid():
    """Otherwise a product with wider coverage reports more area purely for
    covering more ground, and that reads as disagreement about trees."""
    frame = tiles()
    frame.loc[3, OUR_COLUMN] = np.nan          # our map has a gap here
    grid = aggregate_to_grid(frame, [OUR_COLUMN, OTHER], dlon=1.0, dlat=1.0)
    # The other product must not count tile 3 either: (12+18+33)/100 km².
    assert grid.totals_km2[OTHER] == pytest.approx(0.63)


def test_totals_convert_percent_and_area_correctly():
    """50 % of 2 km² is 1 km² of tree."""
    frame = pd.DataFrame(
        {"lon_c": [10.0], "lat_c": [52.0], "tile_area_km2": [2.0], OUR_COLUMN: [50.0]}
    )
    grid = aggregate_to_grid(frame, [OUR_COLUMN], dlon=1.0, dlat=1.0)
    assert grid.totals_km2[OUR_COLUMN] == pytest.approx(1.0)


def test_missing_coordinate_column_is_reported():
    with pytest.raises(KeyError, match="lon_c"):
        aggregate_to_grid(pd.DataFrame({"lat_c": [1.0]}), [OUR_COLUMN], 1.0, 1.0)


def test_totals_require_our_column():
    frame = tiles().drop(columns=[OUR_COLUMN])
    with pytest.raises(KeyError, match="comparison baseline"):
        aggregate_to_grid(frame, [OTHER], dlon=1.0, dlat=1.0)


# ── rasterising ──────────────────────────────────────────────────────────────


def test_empty_cells_stay_nan_not_zero():
    """A gap in a product is not the same as no trees there."""
    frame = pd.DataFrame(
        {
            "lon_c": [10.05, 12.05],
            "lat_c": [52.05, 52.05],
            "tile_area_km2": [1.0, 1.0],
            OUR_COLUMN: [10.0, 20.0],
        }
    )
    grid = aggregate_to_grid(frame, [OUR_COLUMN], dlon=1.0, dlat=1.0)
    image = rasterise(grid, OUR_COLUMN)
    assert np.isnan(image).any(), "the cell between the two tiles must be NaN"
    assert np.nanmin(image) == pytest.approx(10.0)


# ── Table 1 ──────────────────────────────────────────────────────────────────


def per_state() -> pd.DataFrame:
    """Two states of very unequal size, to expose unweighted averaging."""
    return pd.DataFrame(
        {
            "admin_name": ["Bayern", "Bremen"],
            "unit_area_km2": [70000.0, 400.0],
            OUR_COLUMN: [35.0, 10.0],
            f"{OUR_COLUMN}_km2": [24500.0, 40.0],
            OTHER: [30.0, 12.0],
            f"{OTHER}_km2": [21000.0, 48.0],
        }
    )


def test_national_mean_is_a_ratio_of_sums_not_a_mean_of_ratios():
    """70,000 km² at 35 % and 400 km² at 10 % is 34.86 %, not 22.5 %.

    Averaging the two state percentages would let Bremen weigh as much as
    Bavaria; weighting by administrative area still assumes each state was
    mapped in full. Only the ratio of the summed areas is exact.
    """
    frame = pd.DataFrame(
        {
            "admin_name": ["big", "small"],
            "unit_area_km2": [70000.0, 400.0],
            OUR_COLUMN: [35.0, 10.0],
            f"{OUR_COLUMN}_km2": [24500.0, 40.0],
            f"{OUR_COLUMN}_ref_km2": [70000.0, 400.0],
        }
    )
    table = difference_table(frame, [OUR_COLUMN])
    assert table["tree_cover_pct"].iloc[0] == pytest.approx(100 * 24540 / 70400)


def test_national_mean_uses_each_products_own_reference_area():
    """A product with gaps is judged on the ground it actually covers."""
    frame = pd.DataFrame(
        {
            "admin_name": ["a"],
            "unit_area_km2": [1000.0],
            OUR_COLUMN: [30.0],
            f"{OUR_COLUMN}_km2": [300.0],
            f"{OUR_COLUMN}_ref_km2": [1000.0],
            OTHER: [50.0],
            f"{OTHER}_km2": [100.0],
            f"{OTHER}_ref_km2": [200.0],  # only covers a fifth of the unit
        }
    )
    table = difference_table(frame, [OUR_COLUMN, OTHER]).set_index("column")
    assert table.loc[OUR_COLUMN, "tree_cover_pct"] == pytest.approx(30.0)
    assert table.loc[OTHER, "tree_cover_pct"] == pytest.approx(50.0)


def test_national_mean_falls_back_to_state_area():
    """Bavaria is not one half of Germany. An unweighted mean would give
    22.5 %; the weighted figure is close to Bavaria's own."""
    table = difference_table(per_state(), [OUR_COLUMN], {OUR_COLUMN: "Ours"})
    expected = (35.0 * 70000 + 10.0 * 400) / 70400
    assert table["tree_cover_pct"].iloc[0] == pytest.approx(expected)
    assert table["tree_cover_pct"].iloc[0] > 34.0


def test_difference_is_reported_both_absolutely_and_relatively():
    table = difference_table(
        per_state(), [OUR_COLUMN, OTHER], {OUR_COLUMN: "Ours", OTHER: "Copernicus"}
    )
    ours = table[table["column"] == OUR_COLUMN].iloc[0]
    other = table[table["column"] == OTHER].iloc[0]
    assert ours["diff_pp"] == pytest.approx(0.0)
    assert other["diff_pp"] == pytest.approx(other["tree_cover_pct"] - ours["tree_cover_pct"])
    assert other["diff_pct"] < 0, "Copernicus reports less cover here"


def test_areas_sum_across_states():
    table = difference_table(per_state(), [OUR_COLUMN], {OUR_COLUMN: "Ours"})
    assert table["tree_area_km2"].iloc[0] == pytest.approx(24540.0)


def test_unweighted_fallback_warns(caplog):
    """Producing an unweighted national figure silently would be worse than
    saying so."""
    frame = per_state().drop(columns="unit_area_km2")
    with caplog.at_level("WARNING"):
        difference_table(frame, [OUR_COLUMN], {OUR_COLUMN: "Ours"})
    assert "unweighted" in caplog.text


def test_table_requires_our_column():
    with pytest.raises(KeyError):
        difference_table(pd.DataFrame({"admin_name": ["X"]}), [OTHER])
