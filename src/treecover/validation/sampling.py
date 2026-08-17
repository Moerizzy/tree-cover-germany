"""Stratified sampling of validation locations.

Accuracy on a national map is not one number — it varies with tree cover
density, season, terrain, settlement and landscape. A simple random sample
would be dominated by whatever is most common (low-TCD farmland in summer)
and would say nothing about the cases that actually stress the model.

So candidates are drawn to fill every level of every stratification
variable independently, subject to a minimum separation so that two samples
never share the same orthophoto neighbourhood.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "StratumReport",
    "SamplingResult",
    "stratified_sample",
    "make_boxes",
    "resolve_sample_ids",
    "TCD_BINS",
    "TCD_LABELS",
    "ELEV_BINS",
    "ELEV_LABELS",
    "SEASON_OF_MONTH",
]

#: Copernicus HRL tree cover density bins, as used throughout the paper.
TCD_BINS = [0, 10, 25, 50, 100]
TCD_LABELS = ["0-10%", "10-25%", "25-50%", "50-100%"]

ELEV_BINS = [0, 200, 500, 900]
ELEV_LABELS = ["Low (0-200m)", "Mid (200-500m)", "High (500m+)"]

#: Meteorological seasons, used to group acquisition months.
SEASON_OF_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

#: CORINE Land Cover classes 1–11 are artificial surfaces.
URBAN_CLC_CLASSES = tuple(range(1, 12))


@dataclass
class StratumReport:
    """How well one stratification variable was filled."""

    column: str
    counts: dict[object, int]
    target: int

    @property
    def underfilled(self) -> dict[object, int]:
        """Levels that did not reach ``target``."""
        return {k: v for k, v in self.counts.items() if v < self.target}

    def summary(self) -> str:
        lines = [f"{self.column}:"]
        for value, count in sorted(self.counts.items(), key=lambda kv: str(kv[0])):
            status = "ok" if count >= self.target else f"under ({count})"
            lines.append(f"  {str(value):<25} {count:4d} / {self.target}  [{status}]")
        return "\n".join(lines)


@dataclass
class SamplingResult:
    """Selected points plus a per-stratum fulfilment report."""

    samples: gpd.GeoDataFrame
    reports: list[StratumReport] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"Selected {len(self.samples)} samples"]
        parts += [r.summary() for r in self.reports]
        under = {r.column: r.underfilled for r in self.reports if r.underfilled}
        if under:
            parts.append(
                "\nUnderfilled strata (the candidate pool ran out, or the minimum "
                "distance could not be met):"
            )
            for column, levels in under.items():
                parts.append(f"  {column}: {levels}")
        return "\n".join(parts)


def stratified_sample(
    candidates: gpd.GeoDataFrame,
    strata_columns: list[str],
    min_per_stratum: int = 50,
    min_distance_m: float = 500.0,
    random_seed: int = 42,
) -> SamplingResult:
    """Draw samples filling every level of every stratification variable.

    The variables are treated as independent one-dimensional strata rather
    than as a full cross-product. A 4 × 2 × 5 × 12 cross-product would have
    480 cells, most of them empty in reality (there is no high-altitude
    urban winter coastal marsh), and insisting on filling them would stall.
    Filling each dimension separately gives balanced marginals, which is
    what the stratified accuracy tables in the paper report.

    At each step the most underfilled level across all variables is picked
    and one candidate is taken from it. Because a single sample increments
    *every* variable it belongs to, the dimensions fill together rather than
    one at a time.

    Args:
        candidates: Candidate points with the stratification columns and a
            projected CRS (distances are computed in CRS units).
        strata_columns: Columns to balance. Rows with a null in any of them
            are dropped.
        min_per_stratum: Target count per level.
        min_distance_m: Minimum separation between selected points. Guards
            against two samples landing in the same orthophoto and thus
            sharing its illumination and season.
        random_seed: Seed for the candidate shuffle.

    Returns:
        A :class:`SamplingResult`. Levels that could not be filled are
        reported rather than raised — running out of candidates in the
        highest TCD bin is expected, not an error.
    """
    if candidates.crs is None or candidates.crs.is_geographic:
        raise ValueError(
            "candidates must be in a projected CRS; min_distance_m is in CRS units."
        )

    missing = [c for c in strata_columns if c not in candidates.columns]
    if missing:
        raise KeyError(f"Stratification columns not found: {missing}")

    pool = candidates.dropna(subset=strata_columns).copy()
    logger.info("Candidate pool: %d of %d rows have all strata", len(pool), len(candidates))
    if pool.empty:
        return SamplingResult(candidates.iloc[0:0].copy())

    counts: dict[str, dict[object, int]] = {
        col: dict.fromkeys(pool[col].dropna().unique(), 0) for col in strata_columns
    }
    #: Levels we have given up on — pool exhausted or all remaining too close.
    exhausted: set[tuple[str, object]] = set()

    selected_idx: list = []
    coords = np.empty((0, 2), dtype=float)
    rng = np.random.default_rng(random_seed)

    max_iterations = len(pool) * 2
    for iteration in range(max_iterations):
        target = _neediest(counts, exhausted, min_per_stratum)
        if target is None:
            logger.info("All strata reached %d samples", min_per_stratum)
            break
        column, value = target

        available = pool[(pool[column] == value) & (~pool.index.isin(selected_idx))]
        if available.empty:
            exhausted.add(target)
            continue

        order = rng.permutation(len(available))
        picked = None
        for position in order:
            row = available.iloc[position]
            xy = np.array([row.geometry.x, row.geometry.y])
            if len(coords) and np.min(np.hypot(*(coords - xy).T)) < min_distance_m:
                continue
            picked = (available.index[position], row, xy)
            break

        if picked is None:
            # Every remaining candidate in this level is too close to an
            # already-selected point. Nothing more to do here.
            exhausted.add(target)
            continue

        index, row, xy = picked
        selected_idx.append(index)
        coords = np.vstack([coords, xy])
        for col in strata_columns:
            if row[col] in counts[col]:
                counts[col][row[col]] += 1

        if (iteration + 1) % 100 == 0:
            filled = sum(
                1 for col in counts for c in counts[col].values() if c >= min_per_stratum
            )
            total = sum(len(v) for v in counts.values())
            logger.info(
                "iter %d: %d samples, %d/%d strata filled",
                iteration + 1, len(selected_idx), filled, total,
            )
    else:
        logger.warning("Hit the iteration cap (%d) before filling all strata", max_iterations)

    return SamplingResult(
        samples=candidates.loc[selected_idx].copy(),
        reports=[StratumReport(col, counts[col], min_per_stratum) for col in strata_columns],
    )


def _neediest(
    counts: dict[str, dict[object, int]],
    exhausted: set[tuple[str, object]],
    target: int,
) -> tuple[str, object] | None:
    """The least-filled level still worth trying, or None if all are done."""
    best: tuple[str, object] | None = None
    best_count = target
    for column, levels in counts.items():
        for value, count in levels.items():
            if (column, value) in exhausted or count >= target:
                continue
            if count < best_count:
                best_count = count
                best = (column, value)
    return best


def make_boxes(points: gpd.GeoDataFrame, size_m: float = 25.0) -> gpd.GeoDataFrame:
    """Expand sample points into square validation footprints.

    25 m squares are large enough to contain a meaningful mix of canopy and
    background at 20 cm, and small enough that a LiDAR CHM can be built for
    each without downloading the whole tile.

    Args:
        points: Sample points in a projected CRS.
        size_m: Box side length in CRS units.

    Returns:
        The same table with box geometries, plus a ``sample_id`` column if
        one is not already present.
    """
    half = size_m / 2.0
    boxes = points.copy()
    boxes["geometry"] = points.geometry.buffer(half, cap_style=3)
    if "sample_id" not in boxes.columns:
        boxes = boxes.reset_index(drop=True)
        boxes["sample_id"] = boxes.index.astype(int)
    return boxes


def resolve_sample_ids(footprints: gpd.GeoDataFrame) -> pd.Series:
    """The sample id per footprint row, however the state recorded it.

    The three validation sets were produced at different times and do not
    share a schema: Brandenburg and Bavaria carry an explicit ``sample_id``
    column, North Rhine-Westphalia does not and identifies a box purely by
    its row position. Both must map onto the same
    ``tree_mask_sample_XXXX.tif`` filenames.

    Positional order, not the DataFrame index, is what the filenames follow
    — a filtered frame keeps its original index but the masks were numbered
    when the file was written.

    Args:
        footprints: Footprints as read from ``tree_mask_footprints.geojson``.

    Returns:
        Integer ids aligned with ``footprints``.
    """
    if "sample_id" in footprints.columns:
        return footprints["sample_id"].astype(int)
    logger.debug(
        "No 'sample_id' column — falling back to row position, which is how "
        "the NRW masks were numbered."
    )
    return pd.Series(range(len(footprints)), index=footprints.index, dtype=int)


def assign_bins(
    df: pd.DataFrame,
    tcd_column: str = "tcd",
    elevation_column: str = "elevation",
    month_column: str = "month",
    clc_column: str = "clc",
) -> pd.DataFrame:
    """Derive the stratification columns from raw extracted attributes.

    Adds ``tcd_bin``, ``elev_bin``, ``month_bin`` and ``is_urban``. Columns
    whose source is absent are simply not created, so a state without a DEM
    can still be sampled on the remaining variables.
    """
    out = df.copy()
    if tcd_column in out:
        out["tcd_bin"] = pd.cut(
            out[tcd_column], bins=TCD_BINS, labels=TCD_LABELS, include_lowest=True
        )
    if elevation_column in out and out[elevation_column].notna().any():
        out["elev_bin"] = pd.cut(
            out[elevation_column], bins=ELEV_BINS, labels=ELEV_LABELS, include_lowest=True
        )
    if month_column in out:
        out["month_bin"] = out[month_column].map(SEASON_OF_MONTH)
    if clc_column in out:
        out["is_urban"] = out[clc_column].isin(URBAN_CLC_CLASSES)
    return out
