"""Tests for observation building and patch extraction.

The properties that matter here are about data leakage and label validity,
not about pixel counts: a tile must never straddle two splits, a label must
never be applied to imagery it does not describe, and a patch must never be
built from an all-nodata window.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from treecover.data.observations import (
    build_observations,
    is_summer,
    parse_acquisition_date,
)
from treecover.data.patches import (
    MASK_NODATA,
    extract_patches,
    region_vrt_map,
    split_map,
)


# ── date parsing ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2023-05-13", datetime(2023, 5, 13)),
        ("2023-05-13 00:00:00", datetime(2023, 5, 13)),
        ("20230513", datetime(2023, 5, 13)),
        ("2018summer", datetime(2018, 7, 15)),
        ("2018_winter", datetime(2018, 1, 15)),
    ],
)
def test_every_date_spelling_in_the_project_parses(value, expected):
    assert parse_acquisition_date(value) == expected


def test_unparseable_date_returns_none_rather_than_raising():
    assert parse_acquisition_date("not-a-date") is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("20230715", True), ("20230601", True), ("20230831", True),
     ("20230501", False), ("20230915", False), ("2018summer", True)],
)
def test_summer_detection_matches_the_label_source_months(value, expected):
    assert is_summer(value) is expected


# ── fixtures ─────────────────────────────────────────────────────────────────


def write_raster(path, array, count=1):
    array = np.atleast_3d(array.T).T if array.ndim == 2 else array
    with rasterio.open(
        path, "w", driver="GTiff",
        height=array.shape[-2], width=array.shape[-1], count=count,
        dtype=array.dtype, crs="EPSG:25832",
        transform=from_origin(600000, 5300000, 0.2, 0.2),
    ) as dst:
        dst.write(array.reshape(count, array.shape[-2], array.shape[-1]))


@pytest.fixture
def dataset(tmp_path):
    """Two tiles: one with four acquisitions, one with a single summer one."""
    images = tmp_path / "img"
    masks = tmp_path / "mask"
    images.mkdir()
    masks.mkdir()

    layout = {
        "t1": ["20210715", "20220110", "20220715", "20230715"],
        "t2": ["20220715"],
    }
    for tile, dates in layout.items():
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[:128] = 1                       # top half is tree
        write_raster(masks / f"{tile}.tif", mask)
        for date in dates:
            write_raster(images / f"{tile}_{date}.tif",
                         np.full((4, 256, 256), 100, dtype=np.uint8), count=4)

    tiles = pd.DataFrame(
        {"tile_id": ["t1", "t2"], "split": ["train", "val"], "Change": ["0", "0"]}
    )
    return images, masks, tiles


# ── observations ─────────────────────────────────────────────────────────────


def test_temporal_filter_keeps_only_dates_near_the_label_source(dataset):
    """The label comes from the latest summer image (2023-07). With a distance
    of 1 the 2022 summer image survives and the 2021 one does not — a 2023
    label does not describe a tile as it was two years earlier."""
    images, masks, tiles = dataset
    observations = build_observations(images, masks, tiles, max_temporal_distance=1)
    kept = {o.date for o in observations if o.tile_id == "t1"}
    assert kept == {"20220715", "20230715"}


def test_zero_distance_keeps_only_the_label_source(dataset):
    images, masks, tiles = dataset
    observations = build_observations(images, masks, tiles, max_temporal_distance=0)
    assert {o.date for o in observations if o.tile_id == "t1"} == {"20230715"}


def test_wide_distance_keeps_every_acquisition(dataset):
    images, masks, tiles = dataset
    observations = build_observations(images, masks, tiles, max_temporal_distance=10)
    assert len([o for o in observations if o.tile_id == "t1"]) == 4


def test_all_observations_of_a_tile_share_one_label(dataset):
    """The whole basis of multi-season training."""
    images, masks, tiles = dataset
    observations = build_observations(images, masks, tiles, max_temporal_distance=10)
    for tile in ("t1", "t2"):
        paths = {o.mask_path for o in observations if o.tile_id == tile}
        assert len(paths) == 1


def test_changed_tiles_are_dropped(dataset):
    images, masks, tiles = dataset
    tiles.loc[tiles["tile_id"] == "t1", "Change"] = "1"
    observations = build_observations(images, masks, tiles)
    assert {o.tile_id for o in observations} == {"t2"}


def test_tiles_outside_the_sample_are_ignored(dataset):
    images, masks, tiles = dataset
    write_raster(images / "t99_20220715.tif",
                 np.zeros((4, 256, 256), dtype=np.uint8), count=4)
    observations = build_observations(images, masks, tiles)
    assert "t99" not in {o.tile_id for o in observations}


def test_a_tile_never_appears_in_two_splits(dataset):
    """A tile in both train and val leaks overlapping patches between them."""
    images, masks, tiles = dataset
    observations = build_observations(images, masks, tiles, max_temporal_distance=10)
    per_tile = {}
    for obs in observations:
        per_tile.setdefault(obs.tile_id, set()).add(obs.split)
    assert all(len(splits) == 1 for splits in per_tile.values())


def test_missing_mask_skips_the_tile_without_crashing(dataset, tmp_path):
    images, masks, tiles = dataset
    (masks / "t2.tif").unlink()
    observations = build_observations(images, masks, tiles)
    assert "t2" not in {o.tile_id for o in observations}
    assert "t1" in {o.tile_id for o in observations}


# ── patches ──────────────────────────────────────────────────────────────────


def test_patch_grid_and_overlap(dataset):
    images, masks, tiles = dataset
    obs = build_observations(images, masks, tiles, max_temporal_distance=0)[0]
    patches = extract_patches(obs, patch_size=128, stride=64)
    # (256 - 128) // 64 + 1 = 3 positions per axis
    assert len(patches) == 9
    assert {p.width for p in patches} == {128}


def test_stride_equal_to_patch_size_gives_no_overlap(dataset):
    images, masks, tiles = dataset
    obs = build_observations(images, masks, tiles, max_temporal_distance=0)[0]
    patches = extract_patches(obs, patch_size=128, stride=128)
    assert len(patches) == 4
    assert {(p.col_off, p.row_off) for p in patches} == {(0, 0), (0, 128), (128, 0), (128, 128)}


def test_tree_cover_is_measured_per_patch(dataset):
    """The fixture's mask is tree in its top half only."""
    images, masks, tiles = dataset
    obs = build_observations(images, masks, tiles, max_temporal_distance=0)[0]
    patches = extract_patches(obs, patch_size=128, stride=128)
    top = [p for p in patches if p.row_off == 0]
    bottom = [p for p in patches if p.row_off == 128]
    assert all(p.tree_cover_pct == 100.0 for p in top)
    assert all(p.tree_cover_pct == 0.0 for p in bottom)


def test_treeless_patches_are_kept_by_default():
    """They are the negative examples that stop the model painting canopy
    onto bare fields — filtering them raises training IoU and hurts the map."""
    import inspect

    assert inspect.signature(extract_patches).parameters["min_tree_cover_pct"].default == 0.0


def test_all_nodata_patches_are_skipped(tmp_path):
    images, masks = tmp_path / "i", tmp_path / "m"
    images.mkdir()
    masks.mkdir()
    mask = np.full((256, 256), MASK_NODATA, dtype=np.uint8)
    mask[:128, :128] = 1                     # only the top-left quadrant is valid
    write_raster(masks / "t1.tif", mask)
    write_raster(images / "t1_20220715.tif", np.zeros((4, 256, 256), np.uint8), count=4)

    tiles = pd.DataFrame({"tile_id": ["t1"], "split": ["train"]})
    obs = build_observations(images, masks, tiles, change_column=None)[0]
    patches = extract_patches(obs, patch_size=128, stride=128)
    assert len(patches) == 1
    assert (patches[0].col_off, patches[0].row_off) == (0, 0)


def test_raster_smaller_than_a_patch_yields_nothing(dataset):
    """Partial windows are not padded — a padded label is not ground truth."""
    images, masks, tiles = dataset
    obs = build_observations(images, masks, tiles, max_temporal_distance=0)[0]
    assert extract_patches(obs, patch_size=512, stride=256) == []


def test_invalid_stride_is_rejected(dataset):
    images, masks, tiles = dataset
    obs = build_observations(images, masks, tiles, max_temporal_distance=0)[0]
    with pytest.raises(ValueError, match="stride must be positive"):
        extract_patches(obs, patch_size=128, stride=0)


# ── stage-4 handover ─────────────────────────────────────────────────────────


def test_artefacts_match_what_stage_four_reads(dataset):
    """region_vrts keys must be exactly the patch region_ids, or training
    fails with a KeyError deep inside the DataLoader."""
    images, masks, tiles = dataset
    observations = build_observations(images, masks, tiles, max_temporal_distance=10)
    vrts = region_vrt_map(observations)
    splits = split_map(observations)

    patches = [p for o in observations for p in extract_patches(o, 128, 128)]
    assert {p.region_id for p in patches} <= set(vrts)
    assert all({"rgbi_vrt", "mask_vrt"} <= set(v) for v in vrts.values())

    assert set(splits) == {"train_regions", "val_regions", "test_regions"}
    assert set(splits["train_regions"]) & set(splits["val_regions"]) == set()
