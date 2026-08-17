"""Comparing the map against other tree cover products.

Every aggregation here is **area-weighted**, never a plain mean of tile
percentages. Tiles differ in the land area they contribute — a coastal tile
may be a tenth the land of an inland one — so an unweighted mean would let
a sliver of a tile count as much as a whole one and shift the national
figure by more than the differences the comparison is trying to measure.

The comparison baseline is deliberately narrow: a product only enters a
total over tiles where **our** map is also valid. Otherwise a product with
wider coverage would report a larger absolute area purely because it covers
more ground, and the difference would be read as disagreement about trees.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "OUR_COLUMN",
    "GRID_DLON",
    "GRID_DLAT",
    "WEIGHT_COLUMNS",
    "weight_column",
    "aggregate_to_grid",
    "aggregate_by_unit",
    "rasterise",
    "difference_table",
    "GridResult",
]

#: The column holding our own tree cover percentage.
OUR_COLUMN = "our_treecover_pct"

#: Candidate weight columns, best first. ``land_area_km2`` is the land
#: measured inside the tile by :mod:`treecover.statistics` — the area the
#: percentage is actually a percentage *of*. ``tile_area_km2`` is the
#: fallback, and if it was derived from a lon/lat bounding box it is
#: inflated a mean 4.5 % by meridian convergence: harmless for percentages,
#: which are ratios of the same weights, but it lands undiluted in any area
#: computed as ``percentage × weight``.
WEIGHT_COLUMNS = ("land_area_km2", "tile_area_km2")

#: The common grid the manuscript aggregates every product onto: *"we
#: aggregated all products to a common 1 km tile grid, which formed the
#: basis for the scatterplots, maps, and per-state statistics."* Cell width
#: exceeds height because a degree of longitude is shorter than one of
#: latitude at Germany's latitude.
GRID_DLON = 0.020
GRID_DLAT = 0.012


class GridResult:
    """Aggregated grid plus the edges needed to place it on a map."""

    __slots__ = ("frame", "lon_edges", "lat_edges", "totals_km2")

    def __init__(self, frame, lon_edges, lat_edges, totals_km2):
        self.frame = frame
        self.lon_edges = lon_edges
        self.lat_edges = lat_edges
        self.totals_km2 = totals_km2


def weight_column(frame: pd.DataFrame) -> str:
    """Pick the area weight to aggregate by.

    Prefers the measured land area over the tile footprint, so a tile that
    is half sea weighs half as much rather than as much as an inland one.

    Raises:
        KeyError: If the frame carries neither, since an unweighted
            aggregation would silently answer a different question.
    """
    for column in WEIGHT_COLUMNS:
        if column in frame.columns:
            if column != WEIGHT_COLUMNS[0]:
                logger.info(
                    "Weighting by %r; %r is absent. Percentages are unaffected, "
                    "but areas derived from this weight inherit whatever it "
                    "over- or understates.", column, WEIGHT_COLUMNS[0],
                )
            return column
    raise KeyError(
        f"tiles has none of {WEIGHT_COLUMNS}. Every aggregation here is "
        "area-weighted; there is no sensible unweighted default."
    )


def _weighted_mean(
    frame: pd.DataFrame, value_column: str, weight_column: str, by: list[str]
) -> pd.Series:
    """Area-weighted mean of ``value_column``, grouped by ``by``.

    Rows where the value is missing are dropped *before* weighting, so a
    product with gaps is averaged over the tiles it actually covers rather
    than being penalised with implicit zeros.
    """
    subset = frame[by + [weight_column, value_column]].dropna(subset=[value_column]).copy()
    if subset.empty:
        return pd.Series(dtype=float, name=value_column)
    subset["_weighted"] = subset[value_column] * subset[weight_column]
    grouped = subset.groupby(by, observed=True)
    return (grouped["_weighted"].sum() / grouped[weight_column].sum()).rename(value_column)


def aggregate_to_grid(
    tiles: pd.DataFrame,
    product_columns: list[str],
    dlon: float = GRID_DLON,
    dlat: float = GRID_DLAT,
    area_column: str | None = None,
) -> GridResult:
    """Aggregate per-tile percentages onto the common 1 km grid.

    The grid size is a parameter only so it can be varied in tests; the
    manuscript reports one scale, and everything downstream assumes it.

    Args:
        tiles: One row per tile, with ``lon_c``, ``lat_c``, ``area_column``
            and one column per product.
        product_columns: Product columns to aggregate.
        dlon: Cell width in degrees of longitude.
        dlat: Cell height in degrees of latitude.
        area_column: Weight column, in km². Defaults to the best available,
            see :func:`weight_column`.

    Returns:
        A :class:`GridResult`. ``totals_km2`` holds each product's total
        tree area over the tiles where our map is also valid.
    """
    # Coordinates first: a frame without them is malformed in a more basic
    # way than one whose weight column is merely not the preferred one.
    for column in ("lon_c", "lat_c"):
        if column not in tiles.columns:
            raise KeyError(f"tiles has no {column!r} column")
    area_column = area_column or weight_column(tiles)
    if area_column not in tiles.columns:
        raise KeyError(f"tiles has no {area_column!r} column")

    lon_edges = np.arange(
        np.floor(tiles["lon_c"].min() / dlon) * dlon,
        np.ceil(tiles["lon_c"].max() / dlon) * dlon + dlon * 0.5,
        dlon,
    )
    lat_edges = np.arange(
        np.floor(tiles["lat_c"].min() / dlat) * dlat,
        np.ceil(tiles["lat_c"].max() / dlat) * dlat + dlat * 0.5,
        dlat,
    )

    binned = tiles.copy()
    binned["lon_bin"] = pd.cut(binned["lon_c"], bins=lon_edges, labels=False)
    binned["lat_bin"] = pd.cut(binned["lat_c"], bins=lat_edges, labels=False)

    columns = [
        _weighted_mean(binned, column, area_column, ["lat_bin", "lon_bin"])
        for column in product_columns
        if column in binned.columns
    ]
    frame = pd.concat(columns, axis=1).reset_index()
    frame["lon_c"] = lon_edges[frame["lon_bin"].astype(int)] + dlon / 2
    frame["lat_c"] = lat_edges[frame["lat_bin"].astype(int)] + dlat / 2

    return GridResult(frame, lon_edges, lat_edges,
                      _totals_km2(binned, product_columns, area_column))


def _totals_km2(
    tiles: pd.DataFrame, product_columns: list[str], area_column: str
) -> dict[str, float]:
    """Total tree area per product, over tiles where our map is valid too."""
    if OUR_COLUMN not in tiles.columns:
        raise KeyError(
            f"tiles has no {OUR_COLUMN!r} column — the comparison baseline is "
            "defined by where our own map is valid."
        )
    ours_valid = tiles[OUR_COLUMN].notna()
    totals = {}
    for column in product_columns:
        if column not in tiles.columns:
            continue
        mask = ours_valid & tiles[column].notna()
        totals[column] = float(
            (tiles.loc[mask, area_column] * tiles.loc[mask, column] / 100.0).sum()
        )
        dropped = int(ours_valid.sum() - mask.sum())
        if dropped:
            logger.info("%s: %d tile(s) excluded from the total (product has no value)",
                        column, dropped)
    return totals


def rasterise(
    grid: GridResult, column: str
) -> np.ndarray:
    """Place an aggregated column onto a 2-D array for plotting.

    Cells with no data stay NaN rather than becoming zero — a gap in a
    product is not the same as no trees there.
    """
    n_lon = len(grid.lon_edges) - 1
    n_lat = len(grid.lat_edges) - 1
    image = np.full((n_lat, n_lon), np.nan)

    subset = grid.frame[["lon_bin", "lat_bin", column]].dropna()
    rows = subset["lat_bin"].astype(int).to_numpy()
    cols = subset["lon_bin"].astype(int).to_numpy()
    inside = (rows >= 0) & (rows < n_lat) & (cols >= 0) & (cols < n_lon)
    image[rows[inside], cols[inside]] = subset[column].to_numpy()[inside]
    return image


def aggregate_by_unit(
    tiles,
    units,
    product_columns: list[str],
    unit_column: str = "admin_name",
    area_column: str | None = None,
):
    """Aggregate tiles to administrative units by area-weighted mean.

    Args:
        tiles: GeoDataFrame of tile centroids or polygons.
        units: GeoDataFrame of administrative boundaries with ``unit_column``.
        product_columns: Product columns to aggregate.
        unit_column: Name column in ``units``.
        area_column: Weight column, in km².

    Returns:
        ``units`` with one percentage column and one ``_km2`` column per
        product. Units with no overlapping tiles keep NaN rather than 0,
        so an unmapped district is visibly unmapped.
    """
    import geopandas as gpd

    area_column = area_column or weight_column(tiles)
    present = [c for c in product_columns if c in tiles.columns]
    joined = gpd.sjoin(
        tiles[["geometry", area_column] + present],
        units[[unit_column, "geometry"]],
        how="inner",
        predicate="within",
    )
    if joined.empty:
        logger.warning("No tiles fell inside any unit — check the CRS of both inputs.")
        return units.copy()

    ours_valid = joined[OUR_COLUMN].notna() if OUR_COLUMN in joined else True
    comparable = joined[ours_valid]

    means = [
        _weighted_mean(comparable, column, area_column, [unit_column])
        for column in present
    ]
    result = pd.concat(means, axis=1).reset_index()

    for column in present:
        subset = comparable.dropna(subset=[column])
        grouped = subset.assign(
            _km2=subset[column] * subset[area_column] / 100.0
        ).groupby(unit_column)
        result[f"{column}_km2"] = result[unit_column].map(grouped["_km2"].sum())
        # The weight actually behind that area. Products drop out on
        # different tiles, so each carries its own reference area — and a
        # national figure that divides one product's area by another's
        # reference is the error this whole module exists to avoid.
        result[f"{column}_ref_km2"] = result[unit_column].map(
            grouped[area_column].sum()
        )

    return units.merge(result, on=unit_column, how="left")


def difference_table(
    per_unit: pd.DataFrame,
    product_columns: list[str],
    labels: dict[str, str] | None = None,
    unit_column: str = "admin_name",
) -> pd.DataFrame:
    """National summary: each product's cover, area and gap against ours.

    The national mean is itself area-weighted across units, for the same
    reason the per-unit means are: Bavaria is not one sixteenth of Germany.

    Returns:
        One row per product with ``tree_cover_pct``, ``tree_area_km2``,
        ``diff_pp`` (percentage points against ours) and ``diff_pct``
        (relative). This is the paper's Table 1.
    """
    labels = labels or {}
    if OUR_COLUMN not in per_unit.columns:
        raise KeyError(f"per_unit has no {OUR_COLUMN!r} column")

    rows = []
    ours_pct = _national_mean(per_unit, OUR_COLUMN)
    ours_km2 = float(per_unit.get(f"{OUR_COLUMN}_km2", pd.Series(dtype=float)).sum())

    for column in product_columns:
        if column not in per_unit.columns:
            continue
        pct = _national_mean(per_unit, column)
        km2 = float(per_unit.get(f"{column}_km2", pd.Series(dtype=float)).sum())
        rows.append(
            {
                "product": labels.get(column, column),
                "column": column,
                "tree_cover_pct": pct,
                "tree_area_km2": km2,
                "diff_pp": pct - ours_pct,
                "diff_pct": 100.0 * (pct - ours_pct) / ours_pct if ours_pct else np.nan,
                "diff_km2": km2 - ours_km2,
                "units_with_data": int(per_unit[column].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def _national_mean(per_unit: pd.DataFrame, column: str) -> float:
    """National cover fraction: total tree area over total reference area.

    A ratio of sums, not a mean of ratios. The two differ whenever a
    state's administrative area is not the area actually mapped in it — by
    about 0.02 pp nationally here, which is small but is exactly the kind
    of slippage between a percentage and its reference area that this
    module is trying to eliminate.

    Falls back to weighting by administrative area, then to an unweighted
    mean, each with a warning: an unweighted national figure is wrong, and
    producing one silently would be worse than saying so.
    """
    reference = f"{column}_ref_km2"
    area = f"{column}_km2"
    if reference in per_unit.columns and area in per_unit.columns:
        subset = per_unit[[area, reference]].dropna()
        total = subset[reference].sum()
        if total > 0:
            return float(100.0 * subset[area].sum() / total)

    if "unit_area_km2" in per_unit.columns:
        logger.warning(
            "No %r column — weighting by administrative area instead, which "
            "assumes every state was mapped in full.", reference,
        )
        subset = per_unit[[column, "unit_area_km2"]].dropna()
        if subset.empty:
            return float("nan")
        return float(
            (subset[column] * subset["unit_area_km2"]).sum() / subset["unit_area_km2"].sum()
        )

    logger.warning(
        "No 'unit_area_km2' column — falling back to an unweighted mean over units, "
        "which over-weights small states."
    )
    return float(per_unit[column].mean())
