"""Tests for the neighbourhood context read around each tile.

From the manuscript: *"To avoid boundary artifacts at tile edges, each tile
was read with 256 pixels of spatial context of neighboring tiles."* The
model sees the halo; the written prediction must not include it, and the
georeferencing must shift accordingly — an off-by-one here would displace
every tile in the map by 256 pixels.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from treecover.inference.sources import CONTEXT_PX, TileData, TileTask


def tile_data(context_px: int, size: int = 1000) -> TileData:
    """A square array whose transform starts at a known origin."""
    task = TileTask(tile_id="t", state="BY")
    image = np.zeros((3, size, size), dtype=np.uint8)
    return TileData(
        task=task,
        image=image,
        transform=from_origin(600000.0, 5300000.0, 0.2, 0.2),
        crs=rasterio.crs.CRS.from_epsg(25832),
        context_px=context_px,
    )


def test_context_matches_the_manuscript():
    assert CONTEXT_PX == 256


def test_inner_shape_excludes_the_halo_on_both_sides():
    data = tile_data(context_px=256, size=5512)
    assert data.shape == (5512, 5512)
    assert data.inner_shape == (5000, 5000)


def test_crop_returns_the_tile_extent():
    data = tile_data(context_px=10, size=120)
    prediction = np.zeros((120, 120), dtype=np.uint8)
    prediction[10:110, 10:110] = 1          # the tile itself
    cropped = data.crop(prediction)
    assert cropped.shape == (100, 100)
    assert cropped.all(), "the crop must return exactly the tile region"


def test_crop_works_on_stacked_arrays():
    data = tile_data(context_px=5, size=50)
    stacked = np.zeros((2, 50, 50), dtype=np.float32)
    assert data.crop(stacked).shape == (2, 40, 40)


def test_transform_shifts_by_the_halo():
    """The halo starts 256 px up and left of the tile, so the tile's own
    origin is that far in. Getting this wrong displaces the whole map."""
    data = tile_data(context_px=256)
    outer = data.transform
    inner = data.inner_transform
    assert inner.c == pytest.approx(outer.c + 256 * 0.2)
    assert inner.f == pytest.approx(outer.f - 256 * 0.2)
    assert inner.a == outer.a and inner.e == outer.e


def test_no_context_is_a_no_op():
    data = tile_data(context_px=0, size=100)
    array = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)
    assert data.crop(array).shape == (100, 100)
    assert data.inner_transform == data.transform
    assert data.inner_shape == (100, 100)


def test_cropping_is_the_inverse_of_the_halo():
    """Whatever halo width is used, cropping recovers the original size."""
    for context in (0, 1, 76, 256):
        size = 512 + 2 * context
        data = tile_data(context_px=context, size=size)
        assert data.crop(np.zeros((size, size))).shape == (512, 512)
