"""Which training tiles figure 3 draws.

The figure makes a claim the data has to back: that a training tile is a
*pair* of acquisitions under different canopy conditions sharing one label.
These tests pin the three ways that claim can quietly break — the partner
picked from the wrong end of the phenological axis, a same-season pair
slipping through the filter, and the row selection collapsing onto the
extremes of the cover gradient where the panels are blank.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from treecover.data.observations import Observation
from treecover.figures.training_examples import (
    LABEL_NODATA,
    TilePair,
    pair_observations,
    season_contrast,
    select_examples,
    tile_tree_cover_pct,
)


def observation(tile_id: str, date: str, ndsm: bool = True) -> Observation:
    return Observation(
        obs_id=f"{tile_id}_{date}",
        tile_id=tile_id,
        date=date,
        split="train",
        image_path=Path(f"/images/{tile_id}_{date}.tif"),
        mask_path=Path(f"/masks/{tile_id}.tif"),
        ndsm_path=Path(f"/ndsm/{tile_id}_{date}.tif") if ndsm else None,
    )


def pair(tile_id: str, cover: float, contrast_dates=("2023-07-01", "2023-01-15"),
         is_urban=None):
    summer, other = contrast_dates
    return TilePair(
        tile_id=tile_id,
        label_source=observation(tile_id, summer),
        partner=observation(tile_id, other),
        tree_cover_pct=cover,
        is_urban=is_urban,
    )


# ── the label source ────────────────────────────────────────────────────────

def test_label_source_is_the_summer_acquisition():
    """The label is drawn on summer imagery, so summer must be column one."""
    observations = [
        observation("t1", "2023-03-04"),
        observation("t1", "2023-07-12"),
    ]
    (result,) = pair_observations(observations)
    assert result.label_source.date == "2023-07-12"
    assert result.partner.date == "2023-03-04"
    assert result.label_season == "leaf_on"
    assert result.partner_season == "leaf_off"


def test_tile_without_summer_falls_back_to_the_latest():
    """A tile with no summer image is still drawn, not silently dropped."""
    observations = [
        observation("t1", "2023-02-01"),
        observation("t1", "2023-04-20"),
    ]
    (result,) = pair_observations(observations)
    assert result.label_source.date == "2023-04-20"
    assert result.partner is not None


# ── choosing the partner ────────────────────────────────────────────────────

def test_partner_is_the_most_phenologically_distant_not_the_nearest_in_time():
    """Leaf-off beats transition even though transition is closer in time.

    This is the whole point of the fourth column: the widest available
    contrast is what shows the reader that one label serves both.
    """
    observations = [
        observation("t1", "2023-07-01"),   # label source
        observation("t1", "2023-05-20"),   # transition, 6 weeks away
        observation("t1", "2023-01-10"),   # leaf-off, 6 months away
    ]
    (result,) = pair_observations(observations)
    assert result.partner.date == "2023-01-10"
    assert result.contrast == 2


def test_partner_choice_is_deterministic_across_input_order():
    """Directory iteration order must not change which pair is published."""
    dates = ["2023-07-01", "2023-04-10", "2023-10-05"]
    forward = pair_observations([observation("t1", d) for d in dates])
    reverse = pair_observations([observation("t1", d) for d in reversed(dates)])
    assert forward[0].partner.date == reverse[0].partner.date


def test_single_acquisition_yields_no_partner():
    (result,) = pair_observations([observation("t1", "2023-07-01")])
    assert result.partner is None
    assert result.contrast == 0


def test_season_contrast_is_symmetric_and_bounded():
    assert season_contrast("leaf_on", "leaf_off") == 2
    assert season_contrast("leaf_off", "leaf_on") == 2
    assert season_contrast("leaf_on", "transition") == 1
    assert season_contrast("leaf_on", "leaf_on") == 0


# ── choosing the rows ───────────────────────────────────────────────────────

def test_same_season_pairs_are_dropped():
    """A summer/summer pair cannot illustrate the pairing, so it is skipped."""
    flat = TilePair(
        tile_id="flat",
        label_source=observation("flat", "2023-07-01"),
        partner=observation("flat", "2023-08-02"),
        tree_cover_pct=30.0,
    )
    assert select_examples([flat], count=1) == []
    assert select_examples([flat], count=1, require_contrast=False) == [flat]


def test_tiles_without_ndsm_are_dropped_only_when_required():
    no_height = TilePair(
        tile_id="t1",
        label_source=observation("t1", "2023-07-01", ndsm=False),
        partner=observation("t1", "2023-01-01", ndsm=False),
        tree_cover_pct=30.0,
    )
    assert select_examples([no_height], count=1) == []
    assert select_examples([no_height], count=1, require_ndsm=False) == [no_height]


def test_rows_span_the_gradient_without_landing_on_the_extremes():
    """Bin centres, not endpoints.

    The sparsest tile in the training set has a blank nDSM and an empty
    label; the densest has a solid green label. Neither shows a label being
    made, so the selection must avoid both while still spreading out.
    """
    pairs = [pair(f"t{i:03d}", float(i)) for i in range(101)]  # 0 … 100 % cover
    chosen = select_examples(pairs, count=3)

    covers = [p.tree_cover_pct for p in chosen]
    assert covers == sorted(covers), "rows must read as an ascending gradient"
    assert min(covers) > 0.0 and max(covers) < 100.0, "extremes must be avoided"
    assert covers[0] < covers[1] < covers[2], "rows must actually differ"
    assert covers[2] - covers[0] > 50, "the spread must still cover the range"


def test_higher_contrast_wins_among_neighbours_of_equal_cover():
    """Between two near-identical tiles, publish the one with a real contrast."""
    weak = pair("weak", 50.0, contrast_dates=("2023-07-01", "2023-09-10"))  # 1
    strong = pair("strong", 50.1, contrast_dates=("2023-07-01", "2023-01-10"))  # 2
    chosen = select_examples([weak, strong], count=1)
    assert [p.tile_id for p in chosen] == ["strong"]


def test_fewer_candidates_than_rows_returns_what_exists():
    pairs = [pair("t1", 10.0), pair("t2", 40.0)]
    assert len(select_examples(pairs, count=5)) == 2


def test_selection_never_repeats_a_tile():
    pairs = [pair(f"t{i}", float(i)) for i in range(6)]
    chosen = select_examples(pairs, count=3)
    assert len({p.tile_id for p in chosen}) == len(chosen)


# ── the settlement stratum ──────────────────────────────────────────────────

def urban_and_rural_pool():
    """A sample shaped like the real one: urban tiles are the low-cover minority."""
    rural = [pair(f"r{i:03d}", float(i), is_urban=False) for i in range(1, 100)]
    urban = [pair(f"u{i:03d}", float(i), is_urban=True) for i in range(1, 12)]
    return rural + urban


def test_an_urban_row_is_reserved():
    """Ranking on cover alone returns only rural tiles — urban tiles are a
    low-cover minority, so they never reach a bin centre of the whole set."""
    pairs = urban_and_rural_pool()

    without = select_examples(pairs, count=3, urban_rows=0)
    assert not any(p.is_urban for p in without)

    with_urban = select_examples(pairs, count=3, urban_rows=1)
    assert sum(1 for p in with_urban if p.is_urban) == 1
    assert len(with_urban) == 3, "reserving a row must not cost a row"


def test_urban_shortfall_goes_to_the_rural_rows():
    """Two urban rows requested, one urban tile available: still three rows."""
    pairs = [pair(f"r{i}", float(i), is_urban=False) for i in range(1, 30)]
    pairs.append(pair("u1", 5.0, is_urban=True))

    chosen = select_examples(pairs, count=3, urban_rows=2)
    assert len(chosen) == 3
    assert sum(1 for p in chosen if p.is_urban) == 1


def test_tiles_with_no_recorded_stratum_count_as_non_urban():
    """``None`` is not evidence of an urban tile, so it must not fill an urban row."""
    pairs = [pair(f"t{i}", float(i)) for i in range(1, 30)]  # is_urban=None
    chosen = select_examples(pairs, count=3, urban_rows=1)
    assert len(chosen) == 3
    assert not any(p.is_urban for p in chosen)


def test_rows_still_span_the_gradient_with_a_reserved_urban_row():
    pairs = urban_and_rural_pool()
    covers = [p.tree_cover_pct for p in select_examples(pairs, count=3, urban_rows=1)]
    assert covers == sorted(covers)
    assert max(covers) - min(covers) > 30


# ── flight dates across rows ────────────────────────────────────────────────

def test_rows_prefer_distinct_flight_dates():
    """Neighbouring tiles are often sampled from the same two acquisitions.

    Publishing one flight pair twice makes the training set look like it held
    a handful of dates, so an equally well-placed tile from another flight
    wins.
    """
    shared = ("2023-07-01", "2023-01-15")
    other = ("2023-08-14", "2023-02-20")
    pairs = [
        pair("u1", 10.0, contrast_dates=shared, is_urban=True),
        pair("r1", 40.0, contrast_dates=shared, is_urban=False),
        pair("r2", 40.1, contrast_dates=other, is_urban=False),
        pair("r3", 80.0, contrast_dates=other, is_urban=False),
    ]
    chosen = select_examples(pairs, count=2, urban_rows=1)
    dates = [p.label_source.date for p in chosen]
    assert len(set(dates)) == len(dates), f"repeated flight date in {dates}"


def test_distinct_dates_never_override_the_cover_spread():
    """The date tiebreak works inside the neighbourhood, not across the range."""
    pairs = [pair(f"t{i:03d}", float(i), contrast_dates=("2023-07-01", "2023-01-15"))
             for i in range(1, 100)]
    covers = [p.tree_cover_pct for p in select_examples(pairs, count=3, urban_rows=0)]
    assert covers[2] - covers[0] > 50


# ── the cover statistic ─────────────────────────────────────────────────────

def write_mask(path: Path, data: np.ndarray) -> Path:
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype="uint8", crs="EPSG:25832",
        transform=from_origin(500000, 5800000, 0.2, 0.2),
    ) as dst:
        dst.write(data, 1)
    return path


def test_tree_cover_ignores_nodata_rather_than_counting_it_as_background(tmp_path):
    """Half tree, half nodata is 100 % cover — nodata is not a denominator."""
    data = np.full((100, 100), LABEL_NODATA, dtype=np.uint8)
    data[:50] = 1
    path = write_mask(tmp_path / "mask.tif", data)
    assert tile_tree_cover_pct(path) == pytest.approx(100.0)


def test_tree_cover_of_a_half_tree_tile(tmp_path):
    data = np.zeros((100, 100), dtype=np.uint8)
    data[:, :50] = 1
    path = write_mask(tmp_path / "mask.tif", data)
    assert tile_tree_cover_pct(path) == pytest.approx(50.0, abs=1.0)


def test_wholly_nodata_mask_is_nan_not_zero(tmp_path):
    """NaN sorts out of the way; 0.0 would masquerade as a treeless tile."""
    data = np.full((50, 50), LABEL_NODATA, dtype=np.uint8)
    path = write_mask(tmp_path / "mask.tif", data)
    assert np.isnan(tile_tree_cover_pct(path))


def test_unreadable_mask_is_nan_not_a_crash(tmp_path):
    missing = tmp_path / "does_not_exist.tif"
    assert np.isnan(tile_tree_cover_pct(missing))


def test_decimated_read_matches_the_full_resolution_share(tmp_path):
    """The statistic only ranks tiles, but it must not be wrong doing it."""
    rng = np.random.default_rng(0)
    data = (rng.random((1000, 1000)) < 0.37).astype(np.uint8)
    path = write_mask(tmp_path / "mask.tif", data)
    assert tile_tree_cover_pct(path, max_pixels=256) == pytest.approx(37.0, abs=2.0)
