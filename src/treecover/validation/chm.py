"""LiDAR canopy height models and the reference tree masks derived from them.

The reference data for the accuracy assessment is not hand-labelled; it is
derived from the states' airborne laser scanning. For each validation box:

1. Rasterise the point cloud to a **DSM** — the maximum return height per
   cell, i.e. the top of whatever is there.
2. Rasterise the ground-classified returns to a **DTM** and interpolate it
   across the gaps under canopy.
3. **CHM = DSM − DTM**, the height above ground.
4. Threshold at 3 m, close small holes, and optionally subtract buildings.

Both raster steps run over a buffer-expanded extent and are cropped back
afterwards. Without the buffer the DTM interpolation has nothing to work
with at the edges and the CHM shows a frame of spurious height.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rasterio.transform import from_bounds as transform_from_bounds
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, label, minimum_filter

logger = logging.getLogger(__name__)

__all__ = [
    "PointCloud",
    "create_chm",
    "chm_to_tree_mask",
    "fill_small_holes",
    "fill_nodata_by_majority",
    "GROUND_CLASS",
    "TREE_HEIGHT_THRESHOLD_M",
]

#: ASPRS classification code for ground returns.
GROUND_CLASS = 2

#: Minimum height counted as tree, metres. Matches the paper.
TREE_HEIGHT_THRESHOLD_M = 3.0

#: Heights above this are sensor artefacts (birds, noise), not canopy.
MAX_PLAUSIBLE_HEIGHT_M = 100.0


@dataclass
class PointCloud:
    """A minimal point cloud: coordinates, heights, ASPRS classification.

    Deliberately not a laspy object — points from several LAZ tiles get
    merged before rasterising, and this keeps that independent of the
    reader.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    classification: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.x)

    def ground(self) -> PointCloud:
        """The ground-classified subset."""
        if self.classification is None:
            return PointCloud(
                np.array([]), np.array([]), np.array([]), np.array([])
            )
        mask = self.classification == GROUND_CLASS
        return PointCloud(self.x[mask], self.y[mask], self.z[mask], self.classification[mask])

    def clip(self, bounds: tuple[float, float, float, float], buffer: float = 0.0) -> PointCloud:
        """Points within ``bounds`` expanded by ``buffer``."""
        minx, miny, maxx, maxy = bounds
        mask = (
            (self.x >= minx - buffer) & (self.x <= maxx + buffer)
            & (self.y >= miny - buffer) & (self.y <= maxy + buffer)
        )
        return PointCloud(
            self.x[mask], self.y[mask], self.z[mask],
            None if self.classification is None else self.classification[mask],
        )

    @classmethod
    def concatenate(cls, clouds: list[PointCloud]) -> PointCloud:
        """Merge several clouds — a box often straddles two LAZ tiles."""
        clouds = [c for c in clouds if len(c)]
        if not clouds:
            return cls(np.array([]), np.array([]), np.array([]), np.array([]))
        has_class = all(c.classification is not None for c in clouds)
        return cls(
            np.concatenate([c.x for c in clouds]),
            np.concatenate([c.y for c in clouds]),
            np.concatenate([c.z for c in clouds]),
            np.concatenate([c.classification for c in clouds]) if has_class else None,
        )


def _rasterise(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
    minx: float, maxy: float, rows: int, cols: int, resolution: float,
    reducer: str,
) -> np.ndarray:
    """Bin points to a grid, keeping the max or min z per cell.

    ``np.maximum.at`` / ``np.minimum.at`` do this in one vectorised pass;
    the original notebook looped in Python, which dominated runtime on
    dense clouds.
    """
    col = np.clip(((xs - minx) / resolution).astype(int), 0, cols - 1)
    row = np.clip(((maxy - ys) / resolution).astype(int), 0, rows - 1)

    fill = -np.inf if reducer == "max" else np.inf
    grid = np.full((rows, cols), fill, dtype=np.float64)
    op = np.maximum if reducer == "max" else np.minimum
    op.at(grid, (row, col), zs)
    grid[np.isinf(grid)] = np.nan
    return grid


def create_chm(
    points: PointCloud,
    bounds: tuple[float, float, float, float],
    resolution: float = 1.0,
    buffer: float = 0.0,
) -> tuple[np.ndarray, object] | tuple[None, None]:
    """Build a canopy height model from a point cloud.

    Args:
        points: All returns. Ground returns are taken from
            :meth:`PointCloud.ground`.
        bounds: Output extent ``(minx, miny, maxx, maxy)`` in the cloud's CRS.
        resolution: Cell size in CRS units.
        buffer: Extra extent rasterised and then cropped away, in CRS units.
            50 m gives the DTM interpolation enough ground returns across
            tile seams.

    Returns:
        ``(chm, transform)``, or ``(None, None)`` if the cloud is empty or
        the extent degenerate. Heights are clipped to
        ``[0, MAX_PLAUSIBLE_HEIGHT_M]``.
    """
    minx, miny, maxx, maxy = bounds
    if not len(points):
        return None, None

    proc_minx, proc_miny = minx - buffer, miny - buffer
    proc_maxx, proc_maxy = maxx + buffer, maxy + buffer
    proc_cols = int(np.ceil((proc_maxx - proc_minx) / resolution))
    proc_rows = int(np.ceil((proc_maxy - proc_miny) / resolution))
    out_cols = int(np.ceil((maxx - minx) / resolution))
    out_rows = int(np.ceil((maxy - miny) / resolution))
    if min(proc_cols, proc_rows, out_cols, out_rows) <= 0:
        return None, None

    dsm = _rasterise(points.x, points.y, points.z,
                     proc_minx, proc_maxy, proc_rows, proc_cols, resolution, "max")

    ground = points.ground()
    if len(ground):
        dtm = _rasterise(ground.x, ground.y, ground.z,
                         proc_minx, proc_maxy, proc_rows, proc_cols, resolution, "min")
        chm = dsm - _interpolate_ground(dtm, proc_rows, proc_cols)
    else:
        # No ground classification: approximate the terrain with a minimum
        # filter over a ~10 m kernel. Coarser, but better than nothing, and
        # it is what the earlier NRW processing did.
        logger.debug("No ground returns — falling back to a minimum filter")
        kernel = max(3, int(10 / resolution)) | 1  # force odd
        base = np.nan_to_num(dsm, nan=np.nanmin(dsm) if not np.all(np.isnan(dsm)) else 0.0)
        chm = dsm - minimum_filter(base, size=kernel)

    chm = np.clip(chm, 0, MAX_PLAUSIBLE_HEIGHT_M)

    buffer_px = int(buffer / resolution)
    if buffer_px > 0:
        chm = chm[buffer_px : proc_rows - buffer_px, buffer_px : proc_cols - buffer_px]
    chm = chm[:out_rows, :out_cols]

    transform = transform_from_bounds(minx, miny, maxx, maxy, chm.shape[1], chm.shape[0])
    return chm.astype(np.float32), transform


def _interpolate_ground(dtm: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Fill the DTM under canopy by interpolating between ground returns."""
    valid = ~np.isnan(dtm)
    if valid.sum() <= 3:
        return np.nanmedian(dtm[valid]) if valid.any() else 0.0
    yy, xx = np.mgrid[0:rows, 0:cols]
    return griddata(
        (yy[valid], xx[valid]), dtm[valid], (yy, xx),
        method="linear", fill_value=float(np.nanmedian(dtm[valid])),
    )


def fill_small_holes(mask: np.ndarray, max_hole_px: int) -> np.ndarray:
    """Close small holes in both foreground and background.

    Applied in both directions: gaps inside canopy smaller than
    ``max_hole_px`` become canopy, and isolated canopy specks smaller than
    the same threshold become background. The LiDAR CHM is noisy at 1 m and
    without this the reference mask is peppered with single-pixel holes the
    model has no chance of reproducing, which depresses IoU for reasons
    unrelated to model quality.
    """
    filled = mask.astype(bool).copy()
    for foreground in (False, True):
        target = filled if foreground else ~filled
        labels, n = label(target)
        if n == 0:
            continue
        sizes = np.bincount(labels.ravel())
        small = np.isin(labels, np.flatnonzero((sizes <= max_hole_px) & (sizes > 0)))
        small &= target
        filled[small] = not foreground
    return filled


def fill_nodata_by_majority(
    mask: np.ndarray, nodata: np.ndarray, max_hole_px: int
) -> tuple[np.ndarray, np.ndarray]:
    """Assign small nodata patches the majority class of their border.

    Cells with no LiDAR return at all (occlusion, water) leave holes. Small
    ones are filled from their surroundings; large ones stay nodata and are
    excluded from the metrics.

    Returns:
        ``(mask, nodata)`` with the filled patches resolved.
    """
    out_mask = mask.astype(bool).copy()
    out_nodata = nodata.astype(bool).copy()
    labels, n = label(out_nodata)
    if n == 0:
        return out_mask, out_nodata

    for region_id in range(1, n + 1):
        region = labels == region_id
        if region.sum() > max_hole_px:
            continue
        ring = binary_dilation(region) & ~region
        neighbours = out_mask[ring & ~out_nodata]
        if neighbours.size == 0:
            continue
        out_mask[region] = bool(neighbours.mean() >= 0.5)
        out_nodata[region] = False
    return out_mask, out_nodata


def chm_to_tree_mask(
    chm: np.ndarray,
    height_threshold: float = TREE_HEIGHT_THRESHOLD_M,
    max_hole_m2: float = 10.0,
    resolution: float = 1.0,
    buildings: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a CHM into the binary reference tree mask.

    Args:
        chm: Canopy height model, metres. NaN marks no data.
        height_threshold: Minimum height counted as tree.
        max_hole_m2: Holes up to this area are closed.
        resolution: CHM cell size, for converting the area to pixels.
        buildings: Optional boolean building footprint mask. LiDAR cannot
            tell a tree from a roof by height alone, so buildings are
            subtracted where footprints are available.

    Returns:
        ``(tree_mask, nodata_mask)``, both boolean.
    """
    nodata = np.isnan(chm)
    tree = np.nan_to_num(chm, nan=0.0) >= height_threshold

    # No floor here: max_hole_m2=0 must mean "fill nothing", so a caller can
    # get the raw thresholded mask for a sensitivity analysis. Both helpers
    # are natural no-ops at 0. The default (10 m² at 1 m) is unaffected.
    max_hole_px = int(max_hole_m2 / (resolution * resolution))
    if max_hole_px > 0:
        tree = fill_small_holes(tree, max_hole_px)
        tree, nodata = fill_nodata_by_majority(tree, nodata, max_hole_px)

    if buildings is not None:
        tree &= ~buildings.astype(bool)

    tree[nodata] = False
    return tree, nodata
