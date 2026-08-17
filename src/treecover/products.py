"""Sampling the comparison products over a 1 km tile.

This is the step that builds the per-tile comparison table. Each product is
read over the tile's footprint at its **own** resolution and reduced to a
single tree cover percentage, so products of very different pixel sizes
become comparable — the manuscript's common 1 km grid.

Two reductions, both taken from the manuscript:

* **Canopy height models** — the share of pixels at or above 3 m.
  *"For the two CHM-based products, tree cover was derived by counting all
  pixels with a canopy height above 3 m."*
* **Cover and density products** — the mean of the per-pixel percentage.
  *"For the TCD product, the per-pixel density percentage was multiplied by
  the pixel area to obtain the tree-covered area."* Over a tile of equal
  pixels that is the mean.

Nodata is excluded before reducing, never counted as zero: a product with a
gap over part of a tile must report what it saw, not be penalised for the
gap. A tile where a product has no valid pixel at all yields ``None``, so
the difference between "no trees" and "no data" survives into the table.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds

logger = logging.getLogger(__name__)

__all__ = ["Product", "PRODUCTS", "EXCLUDED_PRODUCTS", "sample_product",
           "tree_cover_from_array", "open_without_source_mask"]

#: Minimum canopy height counted as tree, metres.
HEIGHT_THRESHOLD_M = 3.0


@dataclass(frozen=True)
class Product:
    """One comparison product.

    Attributes:
        column: Output column name, matching the published table.
        relative_path: Location under the products root.
        kind: ``"height"`` (metres, thresholded) or ``"cover"`` (percent,
            averaged).
        nodata: Value marking no data where the raster does not declare one.
            CLMS TCD uses 255 but ships without a nodata tag, so counting
            those pixels would put 255 % tiles into the mean.
        label: Name used in figures and tables.
    """

    column: str
    relative_path: str
    kind: str
    nodata: float | None = None
    label: str = ""


#: The products this pipeline can reproduce. Column names match the
#: published table, so a completed run drops straight back into it.
#:
#: Each was checked against the published values on four Schleswig-Holstein
#: tiles and agrees to within 0.7 percentage points — the residual is window
#: alignment at the product's own resolution.
PRODUCTS = (
    Product("meta_chm_treecover_pct", "meta CHM V2.vrt", "height",
            label="CHMv2"),
    Product("treesense_chm3m_treecover_pct",
            "TreeSense_CanopyHeight_Planet_3M/_vrt_2023.vrt", "height",
            nodata=255, label="Planet CHM"),
    Product("clms_tcd2023_treecover_pct", "CLMS_TCD_2023.vrt", "cover",
            nodata=255, label="TCD"),
    Product("treesense3m_treecover_pct", "TreeSense_TreeCover_Planet_3M.vrt",
            "cover", nodata=0, label="Planet 3 m"),
    Product("treesense10m_treecover_pct", "TreeSense_TreeCover_Planet_10M.vrt",
            "cover", nodata=0, label="Planet 10 m"),
    Product("treesense30m_treecover_pct", "TreeSense_TreeCover_Planet_30M.vrt",
            "cover", nodata=0, label="Planet 30 m"),
)

#: Deliberately **not** sampled.
#:
#: The Landsat product is not part of the manuscript, which compares against
#: CHMv2, Planet CHM and TCD only. It is also not reproducible from the data
#: on hand: the raster holds values up to 145, with 81 % of pixels above 100
#: on a test tile, so it is not a cover percentage — and neither a 3 m
#: threshold, a clip to 0–100, nor any constant scaling reproduces the
#: published column across several tiles.
#:
#: The definition is kept so the column is documented rather than merely
#: absent. Sampling it would fill rows with a guessed value and leave the
#: column computed two different ways, which is worse than a visible gap.
EXCLUDED_PRODUCTS = (
    Product("treesense_landsat15m_treecover_pct",
            "TreeSense_TreeCover_Landsat_15M/_vrt_2023.vrt", "cover",
            nodata=255, label="Landsat 15 m"),
)


@contextmanager
def open_without_source_mask(path: Path):
    """Open a mosaic VRT with its sources' mask bands ignored.

    The CHMv2 mosaic lists every source with ``<UseMaskBand>true</UseMaskBand>``,
    so GDAL replaces masked pixels with zero. On a handful of tiles that mask
    is defective: at full resolution it covers precisely the canopy — over one
    Baden-Württemberg tile it removes 44 % of the pixels, of which 77 % hold
    heights above 3 m, while the same mask read from the overviews excludes
    nothing but empty ground. Honouring it there makes CHMv2 appear to miss a
    forest it does in fact map.

    The effect is local. Across 500 random 1 km tiles the mask changes the
    mean by **0.03 pp**, and fewer than 1 % of tiles move by more than a
    percentage point — which is why the published per-tile table, computed
    with the mask honoured, is unaffected and stays the reference for
    :func:`sample_product`. Only the local-scale figure, where one tile is
    the whole panel, needs the mask out of the way.

    Yields:
        An open dataset. For anything that is not a VRT with mask bands, the
        file itself — there is nothing to strip.
    """
    path = Path(path)
    text = path.read_text(errors="ignore") if path.suffix.lower() == ".vrt" else ""
    if "<UseMaskBand>" not in text:
        with rasterio.open(path) as source:
            yield source
        return

    # Sources are named relative to the original VRT, so they have to be
    # made absolute before the rewritten copy is opened from elsewhere.
    text = text.replace('relativeToVRT="1">', f'relativeToVRT="0">{path.parent}/')
    text = re.sub(r"\s*<UseMaskBand>\s*true\s*</UseMaskBand>", "", text)

    handle, temporary = tempfile.mkstemp(suffix=".vrt")
    os.close(handle)
    try:
        Path(temporary).write_text(text)
        with rasterio.open(temporary) as source:
            yield source
    finally:
        Path(temporary).unlink(missing_ok=True)


def tree_cover_from_array(
    values: np.ndarray, kind: str, nodata: float | None
) -> float | None:
    """Reduce a window of product values to a tree cover percentage.

    Args:
        values: Raw pixel values over the tile.
        kind: ``"height"`` or ``"cover"``.
        nodata: Value to exclude, if any.

    Returns:
        Tree cover in percent, or ``None`` when no valid pixel remains —
        which is not the same as 0 % and must not be recorded as such.
    """
    valid = values
    if nodata is not None:
        valid = values[values != nodata]
    if valid.size == 0:
        return None

    if kind == "height":
        return float(100.0 * (valid >= HEIGHT_THRESHOLD_M).mean())
    if kind == "cover":
        return float(valid.mean())
    raise ValueError(f"Unknown product kind {kind!r}; expected 'height' or 'cover'.")


def sample_product(
    product: Product,
    root: Path,
    bounds: tuple[float, float, float, float],
    bounds_crs: str = "EPSG:4326",
    handles: dict | None = None,
) -> float | None:
    """Read one product over a tile and reduce it to a tree cover percentage.

    Args:
        product: The product to sample.
        root: Directory holding the product rasters.
        bounds: Tile footprint ``(minx, miny, maxx, maxy)``.
        bounds_crs: CRS of ``bounds``.
        handles: Optional cache of open datasets. Reopening a VRT per tile
            dominates runtime over hundreds of thousands of tiles; pass a
            dict and it is reused.

    Returns:
        Tree cover in percent, or ``None`` if the product does not cover the
        tile or cannot be read.
    """
    path = root / product.relative_path
    try:
        source = _open(path, handles)
        if source is None:
            return None

        product_bounds = transform_bounds(bounds_crs, source.crs, *bounds)
        if not _intersects(source.bounds, product_bounds):
            return None

        window = window_from_bounds(*product_bounds, transform=source.transform)
        # boundless with the nodata fill, so a tile partly outside the
        # product is reduced over the part it does cover.
        fill = product.nodata if product.nodata is not None else (source.nodata or 0)
        values = source.read(1, window=window, boundless=True, fill_value=fill)
    except rasterio.RasterioIOError as exc:
        logger.warning("Cannot read %s: %s", path.name, exc)
        return None

    nodata = product.nodata if product.nodata is not None else source.nodata
    return tree_cover_from_array(values, product.kind, nodata)


def _open(path: Path, handles: dict | None):
    """Open a raster, reusing a cached handle when one is offered."""
    key = str(path)
    if handles is not None and key in handles:
        return handles[key]
    if not path.exists():
        logger.warning("Product not found: %s", path)
        if handles is not None:
            handles[key] = None
        return None
    source = rasterio.open(path)
    if handles is not None:
        handles[key] = source
    return source


def _intersects(a, b) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])
