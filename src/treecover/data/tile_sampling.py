"""Stratified selection of the tiles the model was trained on.

The training set is 1 km tiles of one federal state, and which tiles they
are decides what the model ever sees. Drawing them uniformly would return
farmland: Lower Saxony's tile index is dominated by low tree cover, and a
uniform draw carries almost no closed canopy and almost no settlement.

So tiles are drawn to fill a 4 x 2 grid of strata — four Copernicus tree
cover density bins crossed with settlement type — under four constraints
that the notebook this module replaces introduced one at a time:

* **both seasons.** A tile is only useful if the same ground was flown in
  summer *and* outside it. One label mask is drawn on the summer image and
  inherited by the other acquisition, which is what makes season-aware
  training possible without labelling every flight — see
  :mod:`treecover.data.seasons`.
* **a flight before or after summer**, not merely a second summer flight.
* **2 km minimum separation**, so no two training tiles share a flight
  line's illumination and phenology.
* **at most five tiles per flight date per stratum**, so one large flight
  cannot supply a whole stratum.

Two soft preferences run alongside: the acquisition order (summer first vs.
non-summer first) is kept balanced, and tiles carrying under-represented
months are preferred. Both only reorder candidates; neither can override
the constraints above.

The published run asked for 200 tiles and the separation constraint stopped
it at 152 — that is the number in the training-data package, and it is
expected, not a failure.

.. note::
   The draw depends on the row order of the candidate table and on the
   global numpy seed, so an exactly identical tile list requires an
   identical input index. Stage 1 reports the overlap with a reference
   table rather than claiming reproduction it cannot guarantee.
"""

from __future__ import annotations

import glob
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

from .observations import SUMMER_MONTHS

logger = logging.getLogger(__name__)

__all__ = [
    "TCD_BINS",
    "TCD_LABELS",
    "URBAN_CLC_CLASSES",
    "CLC_CLASS_NAMES",
    "PUBLISHED_BIN_TARGETS",
    "PUBLISHED_SETTINGS",
    "PUBLISHED_TILE_COUNT",
    "SamplingRun",
    "tiles_from_index",
    "mean_tcd",
    "dominant_land_cover",
    "assign_tcd_bins",
    "border_tile_ids",
    "months_of",
    "seasonal_pattern",
    "has_seasonal_pair",
    "stratified_sample",
    "assign_splits",
]

#: Copernicus HRL tree cover density bins. The same edges as the validation
#: sampling (:mod:`treecover.validation.sampling`), but labelled ``B1..B4``
#: because the sampler reports and targets them by name.
TCD_BINS = [0, 10, 25, 50, 100]
TCD_LABELS = ["B1: 0-10%", "B2: 10-25%", "B3: 25-50%", "B4: 50-100%"]

#: CORINE classes counted as settlement when classifying a training tile.
#:
#: .. warning::
#:    This is **not** ``range(1, 12)``, the full artificial-surfaces group
#:    that :data:`treecover.validation.sampling.URBAN_CLC_CLASSES` uses.
#:    Dump sites (8) and sport and leisure facilities (11) are missing. The
#:    published training tiles were selected with this list, so it is kept
#:    as it was: widening it here would move tiles between the urban and
#:    rural strata and silently redraw the training set.
URBAN_CLC_CLASSES = (1, 2, 3, 4, 5, 6, 7, 9, 10)

#: CORINE Land Cover level-3 codes, for the readable ``dominant_lc_name``.
CLC_CLASS_NAMES = {
    1: "Continuous urban fabric",
    2: "Discontinuous urban fabric",
    3: "Industrial or commercial units",
    4: "Road and rail networks",
    5: "Port areas",
    6: "Airports",
    7: "Mineral extraction sites",
    8: "Dump sites",
    9: "Construction sites",
    10: "Green urban areas",
    11: "Sport and leisure facilities",
    12: "Non-irrigated arable land",
    13: "Permanently irrigated land",
    14: "Rice fields",
    15: "Vineyards",
    16: "Fruit trees and berry plantations",
    17: "Olive groves",
    18: "Pastures",
    19: "Annual crops with permanent crops",
    20: "Complex cultivation patterns",
    21: "Agriculture with natural vegetation",
    22: "Agro-forestry areas",
    23: "Broad-leaved forest",
    24: "Coniferous forest",
    25: "Mixed forest",
    26: "Natural grasslands",
    27: "Moors and heathland",
    28: "Sclerophyllous vegetation",
    29: "Transitional woodland-shrub",
    30: "Beaches, dunes, sands",
    31: "Bare rocks",
    32: "Sparsely vegetated areas",
    33: "Burnt areas",
    34: "Glaciers and perpetual snow",
    35: "Inland marshes",
    36: "Peat bogs",
    37: "Salt marshes",
    38: "Salines",
    39: "Intertidal flats",
    40: "Water courses",
    41: "Water bodies",
    42: "Coastal lagoons",
    43: "Estuaries",
    44: "Sea and ocean",
}

#: Per-stratum targets of the published run. They sum to 200; the run
#: returned 152 because the separation constraint exhausted several strata.
#: The highest density bin has no urban target — a tile cannot be both
#: closed forest and dominantly built-up.
PUBLISHED_BIN_TARGETS = {
    "B1: 0-10%": {"urban": 12, "nonurban": 38},
    "B2: 10-25%": {"urban": 12, "nonurban": 38},
    "B3: 25-50%": {"urban": 12, "nonurban": 38},
    "B4: 50-100%": {"urban": 0, "nonurban": 50},
}

#: The rest of the published call, kept next to the targets so the whole
#: draw is described in one place.
PUBLISHED_SETTINGS = {
    "n_total": 200,
    "min_distance_km": 2.0,
    "min_tcd": 1.0,
    "max_per_flight": 5,
    "balance_months": True,
    "seed": 100,
}

#: What that call produced, for the self-check at the end of stage 1.
PUBLISHED_TILE_COUNT = 152

#: The eight neighbours a tile needs to count as interior.
_NEIGHBOUR_OFFSETS = (
    (-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (1, 1), (-1, -1), (1, -1),
)


@dataclass
class SamplingRun:
    """The drawn tiles plus what the draw did to get there."""

    tile_ids: list
    eligible: pd.DataFrame
    bin_counts: dict
    temporal_counts: dict
    monthly_counts: dict
    bin_targets: dict
    exhausted: list = field(default_factory=list)

    @property
    def stopped_early(self) -> bool:
        """Whether the separation constraint cut the draw short."""
        return len(self.tile_ids) < sum(
            sum(t.values()) for t in self.bin_targets.values()
        )

    def summary(self) -> str:
        lines = [f"Selected {len(self.tile_ids)} tiles from "
                 f"{len(self.eligible)} eligible", "", "By stratum:"]
        for bin_name, targets in self.bin_targets.items():
            lines.append(f"  {bin_name}")
            for status in ("urban", "nonurban"):
                got = self.bin_counts.get((bin_name, status), 0)
                target = targets[status]
                mark = "" if got >= target else "  (pool or distance exhausted)"
                lines.append(f"    {status:<9} {got:3d} / {target:<3d}{mark}")

        lines += ["", "Acquisition order (kept balanced during the draw):"]
        for pattern, count in self.temporal_counts.items():
            lines.append(f"  {pattern:<24} {count:3d}")

        month_names = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
        total = sum(self.monthly_counts.values())
        lines += ["", "Flight months of the selected tiles:"]
        for month in range(1, 13):
            count = self.monthly_counts.get(month, 0)
            share = 100 * count / total if total else 0.0
            lines.append(f"  {month_names[month - 1]}  {count:3d}  ({share:4.1f} %)")
        return "\n".join(lines)


# ── Per-tile attributes ──────────────────────────────────────────────────────


def tiles_from_index(
    index: gpd.GeoDataFrame,
    tile_id_column: str = "tile_id",
    date_column: str = "Aktualitaet",
) -> gpd.GeoDataFrame:
    """Collapse a per-acquisition tile index into one row per tile.

    The state's index lists one feature per acquisition, so a tile flown
    three times appears three times with the same geometry. Sampling works
    on tiles, and what it needs to know about a tile is which seasons its
    acquisitions cover.

    Summer is :data:`treecover.data.observations.SUMMER_MONTHS` — June to
    August, the months a label mask is drawn from.

    Args:
        index: The acquisition index, with a tile id, an acquisition date
            and a geometry per row.
        tile_id_column: Tile id column.
        date_column: Acquisition date column, parsed with
            :func:`pandas.to_datetime`.

    Returns:
        One row per tile, ordered by tile id, carrying ``has_summer``,
        ``has_non_summer``, ``has_both``, ``image_count``, ``flight_dates``
        (sorted unique), ``n_flight_dates`` and the centroid as ``x``/``y``.

    Raises:
        KeyError: If a required column is missing.
    """
    for column in (tile_id_column, date_column):
        if column not in index.columns:
            raise KeyError(
                f"Column {column!r} not in the tile index. Available: "
                f"{list(index.columns)}"
            )

    frame = index.copy()
    frame[tile_id_column] = frame[tile_id_column].astype(str)
    dates = pd.to_datetime(frame[date_column])
    frame["_is_summer"] = dates.dt.month.isin(SUMMER_MONTHS)
    frame["_date"] = dates

    # Ordered by tile id, which is what groupby gives and what the draw's
    # random sample then walks — see the module note on reproducibility.
    grouped = frame.groupby(tile_id_column, sort=True)
    attributes = pd.DataFrame(
        {
            tile_id_column: [key for key, _ in grouped],
            "has_summer": grouped["_is_summer"].any().to_numpy(),
            "has_non_summer": (~grouped["_is_summer"].all()).to_numpy(),
            "image_count": grouped.size().to_numpy(),
            "flight_dates": [sorted(set(group["_date"])) for _, group in grouped],
        }
    )
    geometries = gpd.GeoSeries(
        [group["geometry"].iloc[0] for _, group in grouped], crs=index.crs
    )
    tiles = gpd.GeoDataFrame(attributes, geometry=geometries, crs=index.crs)
    tiles["has_both"] = tiles["has_summer"] & tiles["has_non_summer"]
    tiles["n_flight_dates"] = tiles["flight_dates"].apply(len)

    centroids = tiles.geometry.centroid
    tiles["x"] = centroids.x
    tiles["y"] = centroids.y

    logger.info(
        "%d acquisitions over %d tiles: %d with both seasons, %d summer only",
        len(index), len(tiles), int(tiles["has_both"].sum()),
        int((tiles["has_summer"] & ~tiles["has_non_summer"]).sum()),
    )
    return tiles


def _zonal_values(dataset, geometry, max_valid: float | None = None) -> np.ndarray:
    """Raster values inside one polygon, as a flat array.

    ``all_touched`` includes every pixel the polygon touches. At 10 m
    (density) and 20 m (land cover) against a 1 km tile that is a boundary
    effect of well under a percent, and it guarantees a tile is never empty.
    """
    from rasterio.mask import mask as rio_mask

    try:
        window, _ = rio_mask(dataset, [geometry], crop=True, all_touched=True)
    except ValueError:
        # Disjoint from the raster — rasterio raises rather than returning
        # an empty window.
        return np.empty(0, dtype=dataset.dtypes[0])

    values = window[0].ravel()
    if max_valid is not None:
        return values[values <= max_valid]
    return values


def _intersects(dataset, bounds) -> bool:
    """Whether a tile's bounds overlap a raster's, without reading pixels."""
    return not (
        bounds[0] > dataset.bounds.right
        or bounds[2] < dataset.bounds.left
        or bounds[1] > dataset.bounds.top
        or bounds[3] < dataset.bounds.bottom
    )


def mean_tcd(
    tiles: gpd.GeoDataFrame,
    tcd_paths: Sequence[Path] | str,
    tile_id_column: str = "tile_id",
    progress_every: int = 1000,
) -> pd.Series:
    """Mean Copernicus tree cover density per tile.

    The product ships as one raster per UTM cell, so a tile near a cell
    border draws pixels from two files; values from all overlapping files
    are pooled before averaging, which is why this is not a plain zonal
    statistic. Values above 100 are the product's nodata and are dropped.

    A tile with no valid pixel gets 0 rather than NaN — that is what the
    original run did, and the ``min_tcd`` filter removes those tiles from
    the draw anyway.

    Args:
        tiles: Tiles to measure, any CRS.
        tcd_paths: The density rasters, or a glob pattern for them.
        tile_id_column: Tile id column.
        progress_every: Log every N tiles. 0 disables.

    Returns:
        Mean density per tile, indexed by tile id.

    Raises:
        FileNotFoundError: If the pattern matches nothing.
    """
    import rasterio

    paths = _resolve_rasters(tcd_paths, "tree cover density")
    with rasterio.open(paths[0]) as first:
        target_crs = first.crs
    projected = tiles.to_crs(target_crs) if tiles.crs != target_crs else tiles

    datasets = [rasterio.open(p) for p in paths]
    try:
        values = []
        for position, (_, row) in enumerate(projected.iterrows(), start=1):
            bounds = row.geometry.bounds
            pooled = [
                _zonal_values(dataset, row.geometry, max_valid=100)
                for dataset in datasets
                if _intersects(dataset, bounds)
            ]
            pooled = [chunk for chunk in pooled if len(chunk)]
            values.append(float(np.concatenate(pooled).mean()) if pooled else 0.0)

            if progress_every and position % progress_every == 0:
                logger.info("Tree cover density: %d/%d tiles", position, len(projected))
    finally:
        for dataset in datasets:
            dataset.close()

    return pd.Series(values, index=tiles[tile_id_column].to_numpy(), name="mean_tcd")


def dominant_land_cover(
    tiles: gpd.GeoDataFrame,
    clc_path: Path,
    tile_id_column: str = "tile_id",
    urban_classes: Iterable[int] = URBAN_CLC_CLASSES,
    progress_every: int = 1000,
) -> pd.DataFrame:
    """The CORINE class covering most of each tile, and its built-up share.

    Two numbers come out of the same read. ``dominant_lc`` is the modal
    class and decides the settlement stratum — a tile counts as urban when
    its *dominant* class is built-up, not when it merely contains houses,
    which is why the urban stratum is small and has to be reserved for.
    ``urban_pct`` is the share of built-up pixels and is carried through to
    the training-data package for reference only.

    Args:
        tiles: Tiles to classify, any CRS.
        clc_path: The CORINE raster, class-coded 1–44.
        tile_id_column: Tile id column.
        urban_classes: Classes counted as settlement. The default is the
            list the published selection used — see
            :data:`URBAN_CLC_CLASSES`.
        progress_every: Log every N tiles. 0 disables.

    Returns:
        ``dominant_lc``, ``dominant_lc_name``, ``urban_pct`` and
        ``is_urban``, indexed by tile id. A tile with no valid pixel gets
        class 0 and ``is_urban`` false.
    """
    import rasterio

    urban = tuple(urban_classes)
    with rasterio.open(clc_path) as dataset:
        projected = (
            tiles.to_crs(dataset.crs) if tiles.crs != dataset.crs else tiles
        )
        rows = []
        for position, (_, row) in enumerate(projected.iterrows(), start=1):
            values = _zonal_values(dataset, row.geometry)
            values = values[values > 0]
            if len(values):
                classes, counts = np.unique(values, return_counts=True)
                dominant = int(classes[int(np.argmax(counts))])
                urban_pct = 100.0 * float(np.isin(values, urban).sum()) / len(values)
            else:
                dominant, urban_pct = 0, 0.0

            rows.append(
                {
                    "dominant_lc": dominant,
                    "dominant_lc_name": CLC_CLASS_NAMES.get(dominant, "No data"),
                    "urban_pct": urban_pct,
                    "is_urban": dominant in urban,
                }
            )

            if progress_every and position % progress_every == 0:
                logger.info("Land cover: %d/%d tiles", position, len(projected))

    return pd.DataFrame(rows, index=tiles[tile_id_column].to_numpy())


def _resolve_rasters(paths: Sequence[Path] | str, what: str) -> list[Path]:
    """Accept either a glob pattern or an explicit list, and check it."""
    if isinstance(paths, (str, Path)) and any(c in str(paths) for c in "*?["):
        resolved = [Path(p) for p in sorted(glob.glob(str(paths)))]
    elif isinstance(paths, (str, Path)):
        resolved = [Path(paths)]
    else:
        resolved = [Path(p) for p in paths]

    if not resolved:
        raise FileNotFoundError(f"No {what} raster matched {paths!r}")
    logger.info("%s: %d raster(s)", what.capitalize(), len(resolved))
    return resolved


def assign_tcd_bins(values) -> pd.Categorical:
    """Bin tree cover density into the four strata, edges inclusive at 0."""
    return pd.cut(values, bins=TCD_BINS, labels=TCD_LABELS, include_lowest=True)


def border_tile_ids(tile_ids: Iterable[str]) -> set:
    """Tiles missing at least one of their eight neighbours.

    A border tile has no imagery on one side, so a patch cut there has
    context on one side only — the same reason inference reads a halo from
    the neighbouring tiles. They are dropped from the draw.

    Ids encode the tile's south-west corner in kilometres: ``323405940`` is
    zone 32, easting 340 km, northing 5940 km. Neighbours are found by
    stepping that pair and re-assembling the id as text, which assumes a
    three-digit easting — true throughout Lower Saxony, and an id whose
    easting reaches four digits simply reads as a border tile rather than
    matching the wrong neighbour.

    Args:
        tile_ids: Every tile the state's index knows, not only the
            candidates. Restricting this set would report interior tiles as
            border ones.

    Returns:
        The subset of ``tile_ids`` that lack a neighbour.
    """
    known = {str(t) for t in tile_ids}
    border = set()
    for tile_id in known:
        if len(tile_id) < 6:
            border.add(tile_id)
            continue
        zone, easting, northing = tile_id[:2], tile_id[2:-4], tile_id[-4:]
        try:
            easting_km, northing_km = int(easting), int(northing)
        except ValueError:
            border.add(tile_id)
            continue

        for dx, dy in _NEIGHBOUR_OFFSETS:
            if f"{zone}{easting_km + dx}{northing_km + dy}" not in known:
                border.add(tile_id)
                break
    return border


# ── Seasonal structure of a tile's acquisitions ──────────────────────────────


def months_of(dates) -> list:
    """Sorted unique months of a tile's acquisitions."""
    if not isinstance(dates, (list, tuple)):
        return []
    return sorted({pd.to_datetime(d).month for d in dates})


def _split_by_season(dates) -> tuple[list, list]:
    """A tile's acquisitions as ``(summer, non_summer)``."""
    summer, non_summer = [], []
    for date in dates:
        parsed = pd.to_datetime(date)
        (summer if parsed.month in SUMMER_MONTHS else non_summer).append(parsed)
    return summer, non_summer


def seasonal_pattern(dates) -> str:
    """How a tile's non-summer flights sit relative to its first summer one.

    ``non-summer_then_summer``, ``summer_then_non-summer``, ``both``, or
    ``unknown`` when the tile does not have both seasons at all. The
    sampler keeps the first two balanced so the training set does not
    consist only of, say, winter flights that predate their label image —
    where anything built or felled in between contradicts the label in one
    direction only.
    """
    if not isinstance(dates, (list, tuple)) or not dates:
        return "unknown"

    summer, non_summer = _split_by_season(dates)
    if not summer or not non_summer:
        return "unknown"

    first_summer = min(summer)
    before = [d for d in non_summer if d < first_summer]
    after = [d for d in non_summer if d > first_summer]
    if before and after:
        return "both"
    if before:
        return "non-summer_then_summer"
    if after:
        return "summer_then_non-summer"
    return "unknown"


def has_seasonal_pair(dates) -> bool:
    """Whether a tile has a non-summer flight before or after its summer one.

    Stricter than ``has_both``: it rules out a non-summer acquisition on
    the same day as the summer one, which would carry the same canopy state
    and teach the model nothing new.
    """
    return seasonal_pattern(dates) != "unknown"


# ── The draw ─────────────────────────────────────────────────────────────────


def stratified_sample(
    frame: pd.DataFrame,
    n_total: int = 200,
    min_distance_km: float = 2.0,
    bin_targets: dict | None = None,
    min_tcd: float = 1.0,
    max_per_flight: int | None = 5,
    balance_months: bool = True,
    seed: int = 100,
    tile_id_column: str = "tile_id",
) -> SamplingRun:
    """Draw training tiles, filling the neediest stratum at every step.

    One tile is taken per iteration, always from the stratum furthest below
    its target, so the strata fill together rather than in sequence and an
    early exhaustion is visible in the report instead of silently shifting
    the composition.

    Within the chosen stratum three filters narrow the pool before the
    draw, in this order: the under-represented acquisition order, the
    per-flight cap, and — if ``balance_months`` — the tiles whose months
    are currently under-represented, kept within 30 % of the best score so
    the pool does not collapse to a single candidate. What survives is
    filtered to tiles at least ``min_distance_km`` from every tile already
    taken, and one is drawn at random.

    A stratum that cannot supply such a tile is closed for good rather than
    retried: its pool cannot grow, and retrying it would spin. When every
    stratum is closed the draw stops short of ``n_total``, which is what
    happened to the published run.

    Args:
        frame: Candidate tiles with ``mean_tcd``, ``tcd_bin``, ``is_urban``,
            ``has_both``, ``flight_dates``, ``x``/``y`` in metres, and
            optionally ``is_border`` and ``excluded``.
        n_total: How many tiles to draw. The targets should sum to it.
        min_distance_km: Minimum centroid separation, strictly enforced.
        bin_targets: ``{tcd_bin: {"urban": n, "nonurban": n}}``. Defaults to
            :data:`PUBLISHED_BIN_TARGETS`.
        min_tcd: Drop tiles below this mean density. The default of 1 %
            removes tiles with no canopy at all, which cannot teach a tree
            class.
        max_per_flight: Cap on tiles sharing one flight date within one
            stratum. ``None`` disables it.
        balance_months: Prefer under-represented flight months.
        seed: Global numpy seed for the draws.
        tile_id_column: Tile id column.

    Returns:
        A :class:`SamplingRun`.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If no tile survives the filters.
    """
    targets = dict(bin_targets or PUBLISHED_BIN_TARGETS)
    required = {tile_id_column, "mean_tcd", "tcd_bin", "is_urban", "flight_dates",
                "x", "y"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Candidate table is missing {sorted(missing)}")

    eligible = _eligible(frame, min_tcd)
    if eligible.empty:
        raise ValueError(
            "No tile survives the filters. Check mean_tcd, the season flags "
            "and the border/exclusion masks."
        )
    eligible = eligible.reset_index(drop=True)
    eligible["tile_months"] = eligible["flight_dates"].apply(months_of)
    eligible["temporal_pattern"] = eligible["flight_dates"].apply(seasonal_pattern)

    logger.info(
        "Eligible: %d tiles; acquisition order %s",
        len(eligible), eligible["temporal_pattern"].value_counts().to_dict(),
    )

    np.random.seed(seed)
    strata = {
        (bin_name, status): eligible[
            (eligible["tcd_bin"] == bin_name)
            & (eligible["is_urban"] == (status == "urban"))
        ]
        for bin_name in targets
        for status in ("urban", "nonurban")
    }
    flight_use = {key: {} for key in strata}
    counts = {key: 0 for key in strata}
    temporal_counts = {"summer_then_non-summer": 0, "non-summer_then_summer": 0,
                       "both": 0}
    monthly_counts = {month: 0 for month in range(1, 13)}

    selected: list = []
    coordinates = np.empty((0, 2))
    exhausted: list = []

    while len(selected) < n_total and strata:
        # Neediest stratum. Ties go to the first in insertion order, i.e. by
        # density bin and urban before rural.
        key = max(strata, key=lambda k: targets[k[0]][k[1]] - counts[k])
        pool = strata[key]
        pool = pool[~pool[tile_id_column].isin(selected)]

        candidate = _pick(
            pool, coordinates, min_distance_km, temporal_counts, monthly_counts,
            flight_use[key], max_per_flight, balance_months,
        )
        if candidate is None:
            # The pool is empty, every remaining tile is too close, or the
            # flight cap blocks them all. None of that can change later.
            del strata[key], flight_use[key]
            exhausted.append(key)
            continue

        selected.append(candidate[tile_id_column])
        counts[key] += 1
        coordinates = np.vstack([coordinates, [[candidate["x"], candidate["y"]]]])

        if candidate["temporal_pattern"] in temporal_counts:
            temporal_counts[candidate["temporal_pattern"]] += 1
        for month in candidate["tile_months"]:
            monthly_counts[month] += 1
        for date in candidate["flight_dates"]:
            flight_use[key][str(date)] = flight_use[key].get(str(date), 0) + 1

    if len(selected) < n_total:
        logger.warning(
            "Stopped at %d of %d tiles: no remaining candidate is %.1f km from "
            "every tile already selected. Strata closed: %s",
            len(selected), n_total, min_distance_km,
            ", ".join(f"{b} {s}" for b, s in exhausted),
        )

    return SamplingRun(
        tile_ids=selected,
        eligible=eligible,
        bin_counts=counts,
        temporal_counts=temporal_counts,
        monthly_counts=monthly_counts,
        bin_targets=targets,
        exhausted=exhausted,
    )


def _eligible(frame: pd.DataFrame, min_tcd: float) -> pd.DataFrame:
    """Apply the hard filters, reporting what each one costs."""
    conditions = [("mean tcd", frame["mean_tcd"] >= min_tcd)]
    if "has_both" in frame.columns:
        conditions.append(("both seasons", frame["has_both"].astype(bool)))
    conditions.append(
        ("flight before/after summer", frame["flight_dates"].apply(has_seasonal_pair))
    )
    if "is_border" in frame.columns:
        conditions.append(("interior", ~frame["is_border"].astype(bool)))
    if "excluded" in frame.columns:
        conditions.append(("not excluded", ~frame["excluded"].astype(bool)))

    mask = pd.Series(True, index=frame.index)
    for label, condition in conditions:
        before = int(mask.sum())
        mask &= condition
        logger.debug("filter %-28s %6d -> %6d", label, before, int(mask.sum()))
    return frame[mask].copy()


def _pick(
    pool: pd.DataFrame,
    coordinates: np.ndarray,
    min_distance_km: float,
    temporal_counts: dict,
    monthly_counts: dict,
    flight_use: dict,
    max_per_flight: int | None,
    balance_months: bool,
):
    """One tile from ``pool``, or None if the stratum cannot serve one."""
    if pool.empty:
        return None

    preferred = _underrepresented_pattern(temporal_counts)
    narrowed = pool
    if preferred is not None:
        by_pattern = pool[pool["temporal_pattern"] == preferred]
        if not by_pattern.empty:
            narrowed = by_pattern

    if max_per_flight is not None:
        narrowed = narrowed[
            narrowed["flight_dates"].apply(
                lambda dates: all(
                    flight_use.get(str(d), 0) < max_per_flight for d in dates
                )
            )
        ]
        if narrowed.empty and preferred is not None:
            # Dropping the soft preference can reopen the stratum; dropping
            # the flight cap cannot, it is a constraint.
            narrowed = pool[
                pool["flight_dates"].apply(
                    lambda dates: all(
                        flight_use.get(str(d), 0) < max_per_flight for d in dates
                    )
                )
            ]
    if narrowed.empty:
        return None

    if balance_months and coordinates.size:
        narrowed = _prefer_underrepresented_months(narrowed, monthly_counts)

    if not coordinates.size:
        return narrowed.sample(1).iloc[0]

    from scipy.spatial.distance import cdist

    distances = cdist(narrowed[["x", "y"]].to_numpy(), coordinates) / 1000.0
    far_enough = distances.min(axis=1) >= min_distance_km
    if not far_enough.any():
        return None
    return narrowed[far_enough].sample(1).iloc[0]


def _underrepresented_pattern(temporal_counts: dict) -> str | None:
    """Which acquisition order is behind, or None when they are level."""
    forward = temporal_counts["summer_then_non-summer"]
    backward = temporal_counts["non-summer_then_summer"]
    if forward < backward:
        return "summer_then_non-summer"
    if backward < forward:
        return "non-summer_then_summer"
    return None


def _prefer_underrepresented_months(
    pool: pd.DataFrame, monthly_counts: dict
) -> pd.DataFrame:
    """Keep the tiles carrying the most under-represented months.

    Scored as the number of a tile's months that are below the current mean
    across all twelve, then thinned to everything within 30 % of the best
    score. Keeping a band rather than the maximum leaves the random draw
    something to choose from — a hard maximum makes the month balance
    decide the tile, and with it the spatial pattern.
    """
    mean_count = float(np.mean(list(monthly_counts.values())))
    scores = pool["tile_months"].apply(
        lambda months: sum(1 for m in months if monthly_counts[m] < mean_count)
    )
    best = scores.max()
    if best > 0:
        return pool[scores >= best * 0.7]
    return pool


# ── Split ────────────────────────────────────────────────────────────────────


def assign_splits(
    tiles: pd.DataFrame,
    stratify_column: str = "tcd_bin",
    test_size: float = 0.4,
    random_state: int = 42,
    min_per_stratum: int = 3,
) -> pd.Series:
    """A 60/20/20 train/val/test assignment, stratified by density bin.

    Assigned per tile, never per patch: every acquisition of a tile shares
    one label mask and patches from one tile overlap, so a finer split puts
    near-copies of the validation data into training.

    .. warning::
       This is the split that ships in ``sampled_tiles_100.gpkg``, and it
       is **not** the split the published model was trained on. That one is
       an 80/20 two-way draw recorded in the training-data package's
       ``patches/observations.csv``, which :mod:`scripts.03_prepare_patches`
       reads by default. Using this column instead moves published training
       tiles into validation and inflates every metric measured afterwards.
       It is reproduced here because it is part of the published file.

    Args:
        tiles: The drawn tiles.
        stratify_column: Column to stratify on.
        test_size: Share held out, split evenly into val and test.
        random_state: Seed.
        min_per_stratum: Strata smaller than this cannot be split three
            ways and go to train whole.

    Returns:
        ``train`` / ``val`` / ``test`` per row, aligned with ``tiles``.

    Raises:
        SystemExit: If scikit-learn is not installed.
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise SystemExit(
            "error: the train/val/test split needs scikit-learn "
            "(pip install scikit-learn). Everything before it has already "
            "been written."
        ) from None

    keys = tiles[stratify_column].astype(str)
    sizes = keys.value_counts()
    small = keys.isin(sizes[sizes < min_per_stratum].index)
    if small.any():
        logger.warning(
            "%d tile(s) sit in strata smaller than %d and go to train whole",
            int(small.sum()), min_per_stratum,
        )

    splittable = tiles[~small]
    splits = pd.Series("train", index=tiles.index, dtype=object)
    if len(splittable) < 3:
        logger.warning("Too few tiles to split — all assigned to train")
        return splits

    _, held_out = train_test_split(
        splittable, test_size=test_size, stratify=keys[~small],
        random_state=random_state,
    )

    # A stratum of three splits 2/1, leaving that class alone in the
    # held-out half — which the second stratified split rejects outright.
    # Halving it without stratification keeps the sizes and the tile-level
    # separation, which is what the split is for; only the bin balance
    # between val and test is given up, and it is reported.
    held_out_keys = keys.loc[held_out.index]
    counts = held_out_keys.value_counts()
    stratify = held_out_keys if counts.min() >= 2 else None
    if stratify is None:
        logger.warning(
            "Splitting val/test without stratification: %s has a single tile in "
            "the held-out half",
            ", ".join(counts[counts < 2].index),
        )

    validation, test = train_test_split(
        held_out, test_size=0.5, stratify=stratify, random_state=random_state,
    )
    splits.loc[validation.index] = "val"
    splits.loc[test.index] = "test"

    logger.info(
        "Split: %d train / %d val / %d test",
        int((splits == "train").sum()), len(validation), len(test),
    )
    return splits
