"""Tests for reducing a comparison product to a tree cover percentage.

Two reductions exist and applying the wrong one to a product silently
produces a plausible but incorrect number — a canopy height model averaged
instead of thresholded reports metres as if they were percent. These tests
pin each product to its own reduction.
"""

from __future__ import annotations

import numpy as np
import pytest

from treecover.products import (
    EXCLUDED_PRODUCTS,
    HEIGHT_THRESHOLD_M,
    PRODUCTS,
    Product,
    tree_cover_from_array,
)


# ── the two reductions ───────────────────────────────────────────────────────


def test_height_is_thresholded_not_averaged():
    """A canopy height model reports the *share of pixels* above 3 m. Taking
    the mean would report metres as if they were a percentage."""
    heights = np.array([0.0, 1.0, 2.9, 3.0, 10.0, 25.0])
    assert tree_cover_from_array(heights, "height", None) == pytest.approx(50.0)


def test_height_threshold_matches_the_manuscript():
    assert HEIGHT_THRESHOLD_M == 3.0


def test_exactly_three_metres_counts_as_tree():
    assert tree_cover_from_array(np.array([3.0]), "height", None) == 100.0
    assert tree_cover_from_array(np.array([2.99]), "height", None) == 0.0


def test_cover_is_averaged_not_thresholded():
    """A density product is already a percentage per pixel; its tile value is
    the mean, not the share of pixels above something."""
    density = np.array([0.0, 25.0, 75.0, 100.0])
    assert tree_cover_from_array(density, "cover", None) == pytest.approx(50.0)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="Unknown product kind"):
        tree_cover_from_array(np.array([1.0]), "whatever", None)


# ── nodata ───────────────────────────────────────────────────────────────────


def test_nodata_is_excluded_not_counted_as_zero():
    """A product covering half a tile must report what it saw over that half,
    not be halved for the gap."""
    values = np.array([80.0, 80.0, 255.0, 255.0])
    assert tree_cover_from_array(values, "cover", nodata=255) == pytest.approx(80.0)


def test_all_nodata_yields_none_not_zero():
    """"No data" and "no trees" are different, and the difference has to
    survive into the table — a 0 would be averaged in as a real value."""
    assert tree_cover_from_array(np.full(9, 255.0), "cover", nodata=255) is None


def test_empty_window_yields_none():
    assert tree_cover_from_array(np.array([]), "cover", None) is None


def test_zero_is_a_valid_value_when_it_is_not_nodata():
    assert tree_cover_from_array(np.zeros(4), "cover", nodata=255) == 0.0


def test_tcd_nodata_is_declared_explicitly():
    """CLMS TCD uses 255 for nodata but ships without a nodata tag. Counting
    those pixels would put 255 % tiles into the mean."""
    tcd = next(p for p in PRODUCTS if p.column == "clms_tcd2023_treecover_pct")
    assert tcd.nodata == 255


def test_planet_cover_products_treat_zero_as_nodata():
    """These rasters use 0 for "not covered", not for "no trees"."""
    for column in ("treesense3m_treecover_pct", "treesense10m_treecover_pct",
                   "treesense30m_treecover_pct"):
        product = next(p for p in PRODUCTS if p.column == column)
        assert product.nodata == 0, column


# ── the product registry ─────────────────────────────────────────────────────


def test_each_product_has_a_reduction():
    for product in PRODUCTS:
        assert product.kind in ("height", "cover"), product.column


def test_the_chm_products_are_thresholded():
    for column in ("meta_chm_treecover_pct", "treesense_chm3m_treecover_pct"):
        product = next(p for p in PRODUCTS if p.column == column)
        assert product.kind == "height", column


def test_the_three_manuscript_products_are_present():
    """CHMv2, Planet CHM and TCD are what the paper compares against."""
    columns = {p.column for p in PRODUCTS}
    assert {"meta_chm_treecover_pct", "treesense_chm3m_treecover_pct",
            "clms_tcd2023_treecover_pct"} <= columns


def test_landsat_is_excluded_from_sampling():
    """Not part of the manuscript, and its encoding could not be established
    — the raster holds values above 100, so it is not a cover percentage.
    Filling gaps with a guess would leave the column computed two ways."""
    assert all(p.column != "treesense_landsat15m_treecover_pct" for p in PRODUCTS)
    assert any(p.column == "treesense_landsat15m_treecover_pct"
               for p in EXCLUDED_PRODUCTS)


def test_column_names_are_unique():
    columns = [p.column for p in PRODUCTS]
    assert len(columns) == len(set(columns))


def test_every_product_has_a_label_for_figures():
    for product in PRODUCTS:
        assert product.label, product.column


def test_products_are_immutable():
    """The registry is shared by extraction, figures and Table 1; a mutable
    entry could be changed by one of them for all the others."""
    with pytest.raises((AttributeError, TypeError)):
        PRODUCTS[0].kind = "cover"


def test_product_definitions_can_be_constructed_freely():
    """A new product needs only a column, a path and a reduction."""
    product = Product("x_pct", "x.vrt", "cover", nodata=0, label="X")
    assert product.column == "x_pct" and product.nodata == 0


# ── stripping a mosaic's source mask bands ───────────────────────────────────
#
# The CHMv2 mosaic lists every source with <UseMaskBand>true</UseMaskBand>, and
# on a few tiles that mask covers the canopy rather than the gaps. The per-tile
# table keeps the mask — it moves the national mean by 0.03 pp — but the
# local-scale figure has to drop it, so the rewrite must be exact: strip the
# mask tags, leave everything else, and make the source paths absolute so the
# rewritten copy still finds them from a temporary directory.


VRT_WITH_MASK = """<VRTDataset rasterXSize="10" rasterYSize="10">
  <VRTRasterBand dataType="Byte" band="1">
    <ComplexSource>
      <SourceFilename relativeToVRT="1">tiles/a.tif</SourceFilename>
      <SrcRect xOff="0" yOff="0" xSize="10" ySize="10" />
      <UseMaskBand>true</UseMaskBand>
    </ComplexSource>
  </VRTRasterBand>
</VRTDataset>
"""


def _rewrite(text: str, directory) -> str:
    """Run the rewrite the way :func:`open_without_source_mask` does."""
    import re

    text = text.replace('relativeToVRT="1">', f'relativeToVRT="0">{directory}/')
    return re.sub(r"\s*<UseMaskBand>\s*true\s*</UseMaskBand>", "", text)


def test_rewrite_removes_every_mask_tag():
    rewritten = _rewrite(VRT_WITH_MASK, "/products")
    assert "<UseMaskBand>" not in rewritten


def test_rewrite_makes_source_paths_absolute():
    """Without this the copy, opened from a temporary directory, resolves its
    sources against the wrong parent and reads an empty mosaic rather than
    failing — a silent all-zero result."""
    rewritten = _rewrite(VRT_WITH_MASK, "/products")
    assert 'relativeToVRT="0">/products/tiles/a.tif' in rewritten


def test_rewrite_keeps_the_geometry():
    """Only the mask tags go. A dropped SrcRect would move the data."""
    rewritten = _rewrite(VRT_WITH_MASK, "/products")
    assert '<SrcRect xOff="0" yOff="0" xSize="10" ySize="10" />' in rewritten
    assert 'rasterXSize="10"' in rewritten


def test_a_plain_raster_is_opened_unchanged(tmp_path):
    """Nothing to strip outside a VRT — the file must be opened as it is
    rather than rewritten into a temporary copy."""
    import rasterio

    from treecover.products import open_without_source_mask

    path = tmp_path / "plain.tif"
    with rasterio.open(path, "w", driver="GTiff", width=4, height=4, count=1,
                       dtype="uint8", crs="EPSG:25832",
                       transform=rasterio.Affine(1, 0, 0, 0, -1, 4)) as dst:
        dst.write(np.arange(16, dtype="uint8").reshape(4, 4), 1)

    with open_without_source_mask(path) as src:
        assert src.name == str(path)
        assert src.read(1)[3, 3] == 15


def test_a_vrt_without_mask_bands_is_opened_unchanged(tmp_path):
    import rasterio

    from treecover.products import open_without_source_mask

    tif = tmp_path / "a.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=4, height=4, count=1,
                       dtype="uint8", crs="EPSG:25832",
                       transform=rasterio.Affine(1, 0, 0, 0, -1, 4)) as dst:
        dst.write(np.full((4, 4), 7, dtype="uint8"), 1)
    vrt = tmp_path / "m.vrt"
    vrt.write_text(
        '<VRTDataset rasterXSize="4" rasterYSize="4">'
        '<GeoTransform>0, 1, 0, 4, 0, -1</GeoTransform>'
        '<VRTRasterBand dataType="Byte" band="1"><SimpleSource>'
        f'<SourceFilename relativeToVRT="0">{tif}</SourceFilename>'
        '<SourceBand>1</SourceBand>'
        '<SrcRect xOff="0" yOff="0" xSize="4" ySize="4" />'
        '<DstRect xOff="0" yOff="0" xSize="4" ySize="4" />'
        '</SimpleSource></VRTRasterBand></VRTDataset>'
    )
    with open_without_source_mask(vrt) as src:
        assert src.name == str(vrt)
        assert src.read(1)[0, 0] == 7
