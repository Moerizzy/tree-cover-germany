"""Stage 1: the stratified draw of training tiles.

The invariants worth pinning are the ones a plausible refactor would break
silently: that a tile without a phenological pair can never be drawn, that
the separation constraint is enforced rather than approximated, that the
per-flight cap holds, and that a stratum is filled to its target when the
pool allows it. All of it runs on a synthetic grid — no rasters, no
downloads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from treecover.data.tile_sampling import (
    PUBLISHED_BIN_TARGETS,
    TCD_LABELS,
    assign_splits,
    assign_tcd_bins,
    border_tile_ids,
    has_seasonal_pair,
    months_of,
    seasonal_pattern,
    stratified_sample,
    tiles_from_index,
)

gpd = pytest.importorskip("geopandas")


def make_index(n_east=6, n_north=6, spacing=1000):
    """A synthetic acquisition index: every tile flown in summer and winter."""
    rows = []
    for i in range(n_east):
        for j in range(n_north):
            easting_km = 300 + i
            northing_km = 5800 + j
            tile_id = f"32{easting_km}{northing_km}"
            geometry = box(
                easting_km * 1000, northing_km * 1000,
                easting_km * 1000 + spacing, northing_km * 1000 + spacing,
            )
            for date in ("2023-07-15", "2024-03-10"):
                rows.append({"tile_id": tile_id, "Aktualitaet": date,
                             "geometry": geometry})
    return gpd.GeoDataFrame(rows, crs="EPSG:25832")


def make_candidates(n=40, seed=0):
    """A candidate table ready for the sampler, spread 5 km apart."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        cover = float(rng.uniform(2, 95))
        rows.append(
            {
                "tile_id": f"32{300 + i}5800",
                "mean_tcd": cover,
                "is_urban": i % 4 == 0,
                "has_both": True,
                "flight_dates": [pd.Timestamp("2023-07-15"),
                                 pd.Timestamp("2024-03-10")],
                "x": float(i * 5000),
                "y": 0.0,
                "is_border": False,
                "excluded": False,
            }
        )
    frame = pd.DataFrame(rows)
    frame["tcd_bin"] = assign_tcd_bins(frame["mean_tcd"])
    return frame


# ── Per-tile attributes ──────────────────────────────────────────────────────


def test_index_collapses_to_one_row_per_tile():
    tiles = tiles_from_index(make_index(3, 3))

    assert len(tiles) == 9
    assert tiles["image_count"].eq(2).all()
    assert tiles["n_flight_dates"].eq(2).all()
    assert tiles["has_both"].all()


def test_summer_only_tile_is_not_both_seasons():
    index = make_index(2, 2)
    index = index[index["Aktualitaet"] == "2023-07-15"]

    tiles = tiles_from_index(index)

    assert tiles["has_summer"].all()
    assert not tiles["has_non_summer"].any()
    assert not tiles["has_both"].any()


def test_missing_date_column_names_the_columns_it_found():
    index = make_index(2, 2).rename(columns={"Aktualitaet": "flown"})

    with pytest.raises(KeyError, match="flown"):
        tiles_from_index(index, date_column="Aktualitaet")


def test_interior_tile_needs_all_eight_neighbours():
    ids = [f"32{300 + i}{5800 + j}" for i in range(3) for j in range(3)]

    border = border_tile_ids(ids)

    # Only the centre of a 3 x 3 block has all eight neighbours.
    assert border == set(ids) - {"323015801"}


def test_tcd_bins_cover_the_full_range():
    values = pd.Series([0.0, 5.0, 10.0, 25.0, 50.0, 100.0])

    bins = assign_tcd_bins(values)

    assert list(bins) == [TCD_LABELS[0], TCD_LABELS[0], TCD_LABELS[0],
                          TCD_LABELS[1], TCD_LABELS[2], TCD_LABELS[3]]


# ── Seasonal structure ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "dates, expected",
    [
        (["2023-07-15", "2024-03-10"], "summer_then_non-summer"),
        (["2023-03-10", "2023-07-15"], "non-summer_then_summer"),
        (["2023-03-10", "2023-07-15", "2024-02-01"], "both"),
        (["2023-07-15", "2023-08-01"], "unknown"),
        ([], "unknown"),
    ],
)
def test_seasonal_pattern(dates, expected):
    assert seasonal_pattern([pd.Timestamp(d) for d in dates]) == expected


def test_two_summer_flights_are_not_a_seasonal_pair():
    # The point of the training set is one label under two canopy states.
    # Two July flights carry the same state and teach nothing new.
    assert not has_seasonal_pair([pd.Timestamp("2023-07-15"),
                                  pd.Timestamp("2023-08-01")])


def test_months_are_unique_and_sorted():
    dates = [pd.Timestamp("2024-03-10"), pd.Timestamp("2023-07-15"),
             pd.Timestamp("2022-07-30")]

    assert months_of(dates) == [3, 7]


# ── The draw ─────────────────────────────────────────────────────────────────


def test_draw_respects_the_minimum_distance():
    frame = make_candidates(40)

    run = stratified_sample(frame, n_total=10, min_distance_km=12.0,
                            max_per_flight=None)

    coordinates = frame.set_index("tile_id").loc[run.tile_ids, ["x", "y"]].to_numpy()
    for i, first in enumerate(coordinates):
        for second in coordinates[i + 1:]:
            assert np.hypot(*(first - second)) / 1000 >= 12.0


def test_draw_stops_short_rather_than_violating_the_distance():
    # 40 tiles at 5 km spacing span 195 km; asking for 20 at 50 km apart
    # cannot be satisfied, and that must show up as a short draw.
    frame = make_candidates(40)

    run = stratified_sample(frame, n_total=20, min_distance_km=50.0,
                            max_per_flight=None)

    assert 0 < len(run.tile_ids) < 20
    assert run.stopped_early
    assert run.exhausted


def test_tiles_without_a_seasonal_pair_are_never_drawn():
    frame = make_candidates(20)
    frame.loc[frame.index[:10], "flight_dates"] = pd.Series(
        [[pd.Timestamp("2023-07-15"), pd.Timestamp("2023-08-20")]] * 10,
        index=frame.index[:10],
    )
    unusable = set(frame.loc[frame.index[:10], "tile_id"])

    run = stratified_sample(frame, n_total=10, min_distance_km=0.0,
                            max_per_flight=None)

    assert not unusable & set(run.tile_ids)


def test_border_and_excluded_tiles_are_never_drawn():
    frame = make_candidates(20)
    frame.loc[frame.index[:5], "is_border"] = True
    frame.loc[frame.index[5:10], "excluded"] = True
    blocked = set(frame.loc[frame.index[:10], "tile_id"])

    run = stratified_sample(frame, n_total=10, min_distance_km=0.0,
                            max_per_flight=None)

    assert not blocked & set(run.tile_ids)


def test_min_tcd_drops_treeless_tiles():
    frame = make_candidates(20)
    frame.loc[frame.index[:5], "mean_tcd"] = 0.5
    frame["tcd_bin"] = assign_tcd_bins(frame["mean_tcd"])
    treeless = set(frame.loc[frame.index[:5], "tile_id"])

    run = stratified_sample(frame, n_total=10, min_distance_km=0.0,
                            min_tcd=1.0, max_per_flight=None)

    assert not treeless & set(run.tile_ids)


def test_flight_cap_limits_tiles_per_stratum():
    # Every tile shares one flight pair, so with a cap of two no stratum can
    # contribute more than two tiles — eight in total across the 4 x 2 grid.
    frame = make_candidates(60)

    run = stratified_sample(frame, n_total=40, min_distance_km=0.0,
                            max_per_flight=2)

    assert all(count <= 2 for count in run.bin_counts.values())


def test_targets_are_filled_when_the_pool_allows_it():
    frame = make_candidates(200, seed=3)
    targets = {label: {"urban": 1, "nonurban": 2} for label in TCD_LABELS}

    run = stratified_sample(frame, n_total=12, min_distance_km=0.0,
                            bin_targets=targets, max_per_flight=None)

    for (bin_name, status), count in run.bin_counts.items():
        available = frame[
            (frame["tcd_bin"] == bin_name)
            & (frame["is_urban"] == (status == "urban"))
        ]
        assert count == min(targets[bin_name][status], len(available))


def test_the_same_seed_draws_the_same_tiles():
    frame = make_candidates(60)

    first = stratified_sample(frame, n_total=15, min_distance_km=1.0, seed=100)
    second = stratified_sample(frame, n_total=15, min_distance_km=1.0, seed=100)

    assert first.tile_ids == second.tile_ids


def test_a_different_seed_draws_differently():
    frame = make_candidates(60)

    first = stratified_sample(frame, n_total=15, min_distance_km=1.0, seed=100)
    second = stratified_sample(frame, n_total=15, min_distance_km=1.0, seed=7)

    assert first.tile_ids != second.tile_ids


def test_no_tile_is_drawn_twice():
    frame = make_candidates(60)

    run = stratified_sample(frame, n_total=30, min_distance_km=0.0,
                            max_per_flight=None)

    assert len(set(run.tile_ids)) == len(run.tile_ids)


def test_missing_column_is_named():
    frame = make_candidates(10).drop(columns="mean_tcd")

    with pytest.raises(KeyError, match="mean_tcd"):
        stratified_sample(frame, n_total=5)


def test_empty_pool_raises_rather_than_returning_nothing():
    frame = make_candidates(10)
    frame["mean_tcd"] = 0.0
    frame["tcd_bin"] = assign_tcd_bins(frame["mean_tcd"])

    with pytest.raises(ValueError, match="No tile survives"):
        stratified_sample(frame, n_total=5, min_tcd=1.0)


def test_published_targets_sum_to_the_published_request():
    total = sum(sum(t.values()) for t in PUBLISHED_BIN_TARGETS.values())

    assert total == 200


# ── Split ────────────────────────────────────────────────────────────────────


def test_split_is_disjoint_and_complete():
    pytest.importorskip("sklearn")
    frame = make_candidates(60)

    splits = assign_splits(frame)

    assert set(splits.unique()) <= {"train", "val", "test"}
    assert len(splits) == len(frame)
    assert splits.index.equals(frame.index)


def test_split_keeps_the_density_bins_represented():
    pytest.importorskip("sklearn")
    frame = make_candidates(120, seed=5)

    splits = assign_splits(frame)

    train_bins = set(frame.loc[splits == "train", "tcd_bin"].astype(str))
    assert train_bins == set(frame["tcd_bin"].astype(str))


def test_tiny_strata_go_to_train_whole():
    pytest.importorskip("sklearn")
    frame = make_candidates(40)
    # One tile alone in its bin cannot be split three ways.
    frame.loc[frame.index[0], "mean_tcd"] = 99.9
    frame["tcd_bin"] = assign_tcd_bins(frame["mean_tcd"])
    lonely = frame["tcd_bin"].astype(str).value_counts()
    solitary = lonely[lonely == 1]

    splits = assign_splits(frame)

    for label in solitary.index:
        rows = frame["tcd_bin"].astype(str) == label
        assert (splits[rows] == "train").all()
