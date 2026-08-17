"""Assembling the 1 km prediction tiles into a seamless 10 km mosaic.

This is the *lossless* counterpart to :mod:`treecover.merge`'s reprojection
path. Every prediction tile is already on an exact 1 km UTM grid with 20 cm
pixels, and all tiles of one zone share that grid — so a 10 km cell is
built by copying whole tiles into their windows. No resampling happens
anywhere, and a pixel of the output is the same pixel of the input.

That matters for the area figures. Reprojecting a 20 cm binary mask means
nearest-neighbour resampling, which perturbs pixel counts for no gain: in
the native UTM grid a pixel is exactly 0.04 m² of *projected* area, and the
only correction left is the map scale factor, which
:func:`geodesic_area_factor` supplies per cell. Reprojecting to Web
Mercator, as the display product does, would inflate areas by ~2.5× at
Germany's latitude.

The mosaic also settles a question the per-tile tables can only answer by
convention: overlapping coverage. About 2.5 % of 1 km cells are flown by
two states, so a per-tile sum has to mask one of them away. In the mosaic
each ground pixel exists exactly once — :func:`treecover.merge.select_one_per_cell`
picks the winner before anything is written — so a count over the mosaic
cannot double-count by construction.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "CELL_SIZE_M",
    "PIXEL_M",
    "CELL_PX",
    "STRIPE_PX",
    "Placement",
    "cell_id_for",
    "cell_id_from_tile_key",
    "parse_cell_id",
    "cell_bounds",
    "cell_transform",
    "placement_for",
    "plan_cell",
    "geodesic_area_factor",
    "grid_offset",
    "is_copyable",
    "sanity_check_grid",
]

#: Output cell size in metres. The prediction archive is already foldered
#: this way (``UTM33_E4400_N58600``), so the grid is inherited, not invented.
CELL_SIZE_M = 10_000

#: Ground sampling distance of the published map.
PIXEL_M = 0.2

#: Cell width and height in pixels.
CELL_PX = int(CELL_SIZE_M / PIXEL_M)

#: Rows written at a time. One stripe is one row of 1 km tiles, so a stripe
#: is always filled completely before it is handed to the compressor — see
#: the note on strip alignment in ``scripts/11_export_mosaic.py``.
STRIPE_PX = int(1_000 / PIXEL_M)

#: ``UTM32_E4100_N52900`` — zone, then easting and northing in units of
#: 100 m, which is how the archive names its directories.
_CELL_ID_RE = re.compile(r"^UTM(3[23])_E(\d+)_N(\d+)$")


@dataclass(frozen=True)
class Placement:
    """Where one source tile lands inside its 10 km cell.

    ``read_window`` is in *source* pixel coordinates and is only ever a
    proper subset of the tile when the tile overhangs the cell — one tile in
    the archive is 5001 px instead of 5000 and would otherwise write a row
    past the edge.
    """

    path: Path
    row_off: int
    col_off: int
    height: int
    width: int
    read_row_off: int = 0
    read_col_off: int = 0
    trimmed_rows: int = 0

    @property
    def stripe(self) -> int:
        """Index of the stripe this placement belongs to."""
        return self.row_off // STRIPE_PX


def cell_id_for(zone: str | int, easting_km: int, northing_km: int) -> str:
    """Name the 10 km cell containing a 1 km tile at ``easting_km``/``northing_km``.

    Args:
        zone: UTM zone, 32 or 33.
        easting_km: Tile easting in whole kilometres.
        northing_km: Tile northing in whole kilometres.

    Returns:
        A cell id such as ``UTM33_E4400_N58600``.
    """
    east = (int(easting_km) // 10) * 100
    north = (int(northing_km) // 10) * 100
    return f"UTM{zone}_E{east}_N{north}"


def cell_id_from_tile_key(key: tuple[str, str, str]) -> str:
    """Cell id from a :func:`treecover.merge.cell_key` triple."""
    zone, easting, northing = key
    return cell_id_for(zone, int(easting), int(northing))


def parse_cell_id(cell: str) -> tuple[int, float, float]:
    """Split a cell id into its EPSG code and lower-left corner.

    Returns:
        ``(epsg, left_m, bottom_m)``. EPSG is 25832 or 25833 — ETRS89 / UTM,
        the CRS the predictions were written in.

    Raises:
        ValueError: If ``cell`` is not a cell id.
    """
    match = _CELL_ID_RE.match(cell)
    if not match:
        raise ValueError(f"not a 10 km cell id: {cell!r}")
    zone, east, north = match.groups()
    return 25800 + int(zone), int(east) * 100.0, int(north) * 100.0


def cell_bounds(cell: str) -> tuple[float, float, float, float]:
    """``(left, bottom, right, top)`` of a cell, in its own CRS."""
    _, left, bottom = parse_cell_id(cell)
    return left, bottom, left + CELL_SIZE_M, bottom + CELL_SIZE_M


def cell_transform(cell: str):
    """Affine transform of the cell raster, north-up, 20 cm pixels."""
    from affine import Affine

    _, left, bottom = parse_cell_id(cell)
    top = bottom + CELL_SIZE_M
    return Affine(PIXEL_M, 0.0, left, 0.0, -PIXEL_M, top)


def placement_for(
    cell: str, path: Path, left: float, top: float, width: int, height: int
) -> Placement | None:
    """Locate one source tile inside its cell, clipped to the cell.

    Args:
        cell: Target cell id.
        path: Source tile.
        left: Source's western edge, in the cell's CRS.
        top: Source's northern edge.
        width: Source width in pixels.
        height: Source height in pixels.

    Returns:
        A :class:`Placement`, or ``None`` if the tile lies outside the cell
        entirely — which means the filename and the georeferencing disagree
        and the tile must not be written anywhere.
    """
    cell_left, cell_bottom, cell_right, cell_top = cell_bounds(cell)

    col_off = int(round((left - cell_left) / PIXEL_M))
    row_off = int(round((cell_top - top) / PIXEL_M))

    read_col = max(0, -col_off)
    read_row = max(0, -row_off)
    col_off = max(0, col_off)
    row_off = max(0, row_off)

    usable_w = min(width - read_col, CELL_PX - col_off)
    usable_h = min(height - read_row, CELL_PX - row_off)

    # A placement may not cross a stripe boundary: the writer materialises
    # one stripe at a time, so a taller placement has nowhere to go. In this
    # archive the only tiles affected are the 5001 px ones, which cover
    # 1000.2 m and so overlap their southern neighbour by a single row —
    # trimming that row loses nothing, the neighbour supplies it.
    rows_left = STRIPE_PX - (row_off % STRIPE_PX)
    trimmed = max(0, usable_h - rows_left)
    usable_h -= trimmed

    if usable_w <= 0 or usable_h <= 0:
        logger.warning(
            "%s does not overlap %s (bounds %.1f/%.1f); skipped",
            path.name, cell, left, top,
        )
        return None

    if usable_w != width or usable_h != height:
        logger.debug(
            "%s clipped to %dx%d inside %s (source %dx%d)",
            path.name, usable_w, usable_h, cell, width, height,
        )

    return Placement(
        path=path,
        row_off=row_off,
        col_off=col_off,
        height=usable_h,
        width=usable_w,
        read_row_off=read_row,
        read_col_off=read_col,
        trimmed_rows=trimmed,
    )


def plan_cell(placements: list[Placement]) -> dict[int, list[Placement]]:
    """Group placements by output stripe, so each stripe is written once.

    Every placement lies inside exactly one stripe —
    :func:`placement_for` guarantees it by trimming, and reports how much
    it had to trim so that a genuinely oversized tile can be refused rather
    than silently truncated.
    """
    by_stripe: dict[int, list[Placement]] = {}
    for placement in placements:
        by_stripe.setdefault(placement.stripe, []).append(placement)
    return {k: by_stripe[k] for k in sorted(by_stripe)}


def geodesic_area_factor(cell: str) -> float:
    """True ground area of a cell divided by its 100 km² of projected area.

    UTM is conformal, not equal-area: a projected square kilometre is not a
    square kilometre on the ellipsoid. The factor is 1/k², within roughly
    ±0.15 % across Germany, and multiplying a pixel-counted area by it turns
    a projected area into a ground area.

    Returns:
        The correction factor, or ``1.0`` if pyproj is unavailable — the
        caller then reports projected areas, which is a 0.1 % matter and
        must not stop an export.
    """
    try:
        from pyproj import Geod, Transformer
    except ImportError:  # pragma: no cover - pyproj ships with geopandas
        logger.warning("pyproj not available; areas stay projected")
        return 1.0

    epsg, left, bottom = parse_cell_id(cell)
    right, top = left + CELL_SIZE_M, bottom + CELL_SIZE_M
    transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    corners = [(left, bottom), (right, bottom), (right, top), (left, top)]
    lons, lats = zip(*(transformer.transform(x, y) for x, y in corners))

    geod = Geod(ellps="GRS80")
    true_area, _ = geod.polygon_area_perimeter(lons, lats)
    return abs(true_area) / (CELL_SIZE_M * CELL_SIZE_M)


def grid_offset(cell: str, left: float, top: float) -> tuple[float, float]:
    """How far a source tile sits from the nearest pixel of its cell.

    Returned in pixels, each within ±0.5 by construction. Saarland's
    WMS-derived tiles carry ``top = …000.1``, exactly half a pixel out;
    everything else in the archive is exact.
    """
    cell_left, _, _, cell_top = cell_bounds(cell)
    col = (left - cell_left) / PIXEL_M
    row = (cell_top - top) / PIXEL_M
    return col - round(col), row - round(row)


def sanity_check_grid(cell: str, left: float, top: float) -> bool:
    """Whether a source tile sits exactly on the 20 cm grid of its cell.

    Advisory only — :func:`placement_for` snaps to the nearest pixel, and a
    tile that is a fraction of a pixel out is still worth writing. Dropping
    it would punch a hole in the mosaic that the paper's own statistics do
    not have, which is a far larger error than 10 cm of position.
    """
    off_x, off_y = grid_offset(cell, left, top)
    tolerance = 1e-6
    return abs(off_x) < tolerance and abs(off_y) < tolerance


def is_copyable(res: tuple[float, float], skew: tuple[float, float]) -> bool:
    """Whether a source can be copied at all, rather than resampled.

    A different ground sample distance or any rotation means the source
    does not share the output grid, and copying it would misplace every
    pixel. Sub-pixel translation is fine — that only shifts a tile by less
    than its own resolution — but these two are not.
    """
    tolerance = 1e-9
    return (
        abs(res[0] - PIXEL_M) < tolerance
        and abs(res[1] - PIXEL_M) < tolerance
        and abs(skew[0]) < tolerance
        and abs(skew[1]) < tolerance
    )


def format_area(square_metres: float) -> str:
    """Render an area in km² with a thousands separator."""
    return f"{square_metres / 1e6:,.2f} km²"


def stripe_rows(stripe: int) -> tuple[int, int]:
    """``(first_row, row_count)`` of a stripe, clipped to the cell."""
    start = stripe * STRIPE_PX
    return start, min(STRIPE_PX, CELL_PX - start)


def total_stripes() -> int:
    """Number of stripes in a full cell."""
    return math.ceil(CELL_PX / STRIPE_PX)
