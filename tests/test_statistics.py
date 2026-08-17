"""Tests for per-tile area statistics and the class-code guard.

The invariant under test is the one the paper's headline number depends on:
percentages are relative to land inside the tile's own state, so sea and
neighbouring states never enter the denominator.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from treecover.constants import (
    NODATA,
    PRED_BACKGROUND,
    PRED_TREE,
    to_binary_tree,
    validate_prediction_codes,
)
from treecover.io.tiles import TileRef
from treecover.statistics import StatsJob, compute_tile_stats, summarise_by_state


@pytest.fixture
def tile(tmp_path):
    """A 100 × 100 tile at 20 cm — 20 m square, 0.04 ha — half of it tree."""
    path = tmp_path / "dop20rgb_32_660_5261_by_file_20240730_pred.tif"
    data = np.zeros((100, 100), dtype=np.uint8)
    data[:50] = PRED_TREE
    with rasterio.open(
        path, "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8",
        crs="EPSG:25832", transform=from_origin(660000, 5261000, 0.2, 0.2),
    ) as dst:
        dst.write(data, 1)
    return TileRef(path, "BY", "2024")


def test_unmasked_tile_reports_half_cover(tile):
    stats = compute_tile_stats(tile, StatsJob(), land_mask=None)
    assert stats.ok
    assert stats.tree_cover_pct == pytest.approx(50.0)
    assert stats.land_status == "no_mask"


def test_areas_use_the_configured_pixel_size(tile):
    """100 × 100 px at 20 cm = 400 m² = 0.04 ha."""
    stats = compute_tile_stats(tile, StatsJob(pixel_size_m=0.20), land_mask=None)
    assert stats.tile_area_ha == pytest.approx(0.04)
    assert stats.tree_area_ha == pytest.approx(0.02)


def test_date_is_recovered_from_the_filename(tile):
    stats = compute_tile_stats(tile, StatsJob(), land_mask=None)
    assert stats.date == "20240730"


class _HalfMask:
    """Stand-in mask marking only the left half of a window as valid."""

    def mask_for_bounds(self, epsg, bounds, height, width, state):
        valid = np.zeros((height, width), dtype=bool)
        valid[:, : width // 2] = True
        return valid, "partial"


class _OutsideMask:
    def mask_for_bounds(self, epsg, bounds, height, width, state):
        return None, "outside"


def test_percentage_is_relative_to_masked_land_only(tile):
    """The tile is 50 % tree overall and the mask keeps the left half, which
    is also 50 % tree — so the percentage must stay 50, not halve."""
    stats = compute_tile_stats(tile, StatsJob(), land_mask=_HalfMask())
    assert stats.land_status == "partial"
    assert stats.land_area_ha == pytest.approx(0.02)
    assert stats.tree_cover_pct == pytest.approx(50.0)


def test_tile_outside_the_state_contributes_nothing(tile):
    """A tile beyond the border must not add land area to this state, or
    border tiles get counted twice across the national total."""
    stats = compute_tile_stats(tile, StatsJob(), land_mask=_OutsideMask())
    assert stats.ok
    assert stats.land_area_ha == 0.0
    assert stats.tree_area_ha == 0.0
    assert stats.tree_cover_pct == 0.0


def test_foreign_class_codes_are_rejected(tmp_path):
    """A raster from the trees-outside-forests classification must fail
    loudly rather than have its codes counted as background."""
    path = tmp_path / "x_pred.tif"
    data = np.full((10, 10), 4, dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=10, width=10, count=1, dtype="uint8",
        crs="EPSG:25832", transform=from_origin(0, 0, 0.2, 0.2),
    ) as dst:
        dst.write(data, 1)

    stats = compute_tile_stats(TileRef(path, "BY", "2024"), StatsJob(), land_mask=None)
    assert not stats.ok
    assert "not part of this repository" in stats.error


def test_foreign_codes_can_be_tolerated_explicitly(tmp_path):
    path = tmp_path / "y_pred.tif"
    data = np.full((10, 10), 4, dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=10, width=10, count=1, dtype="uint8",
        crs="EPSG:25832", transform=from_origin(0, 0, 0.2, 0.2),
    ) as dst:
        dst.write(data, 1)

    stats = compute_tile_stats(
        TileRef(path, "BY", "2024"), StatsJob(strict_codes=False), land_mask=None
    )
    assert stats.ok
    assert stats.tree_cover_pct == 0.0


def test_state_summary_recomputes_the_percentage_from_summed_areas():
    """Averaging per-tile percentages would weight a mostly-water coastal
    tile like a full inland one."""
    rows = [
        {"state": "SH", "land_area_ha": 100.0, "tree_area_ha": 10.0},
        {"state": "SH", "land_area_ha": 1.0, "tree_area_ha": 1.0},
    ]
    summary = summarise_by_state(rows)["SH"]
    assert summary["tiles"] == 2
    # 11 / 101, not the mean of 10 % and 100 %.
    assert summary["tree_cover_pct"] == pytest.approx(100 * 11 / 101)
    assert summary["land_area_km2"] == pytest.approx(1.01)


# ── class codes ──────────────────────────────────────────────────────────────


def test_binary_normalisation_preserves_nodata():
    raster = np.array([[PRED_BACKGROUND, PRED_TREE, NODATA]], dtype=np.uint8)
    assert to_binary_tree(raster).tolist() == [[0, 1, NODATA]]


def test_valid_codes_pass_validation():
    validate_prediction_codes(np.array([[0, 1, NODATA]], dtype=np.uint8))


def test_error_names_the_offending_codes():
    with pytest.raises(ValueError, match=r"\[3, 6\]"):
        validate_prediction_codes(np.array([[0, 1, 3, 6]], dtype=np.uint8))
