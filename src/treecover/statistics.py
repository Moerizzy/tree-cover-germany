"""Per-tile tree cover area statistics.

Turns the tiled prediction archive into the table every downstream figure
and table reads: one row per tile with its land area, its tree-covered area
and the resulting percentage.

Percentages are relative to **land inside the tile's own state**, not to the
full tile. A tile half of which is the North Sea, or half of which lies in
Lower Saxony, would otherwise report a tree cover diluted by area that
should never have been in the denominator.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from treecover.constants import NODATA, PRED_TREE, validate_prediction_codes
from treecover.io.tiles import TileRef, acquisition_date, find_prediction_tiles
from treecover.masking import StateLandMask

logger = logging.getLogger(__name__)

__all__ = ["StatsJob", "TileStats", "compute_tile_stats", "run"]

M2_PER_HA = 10_000.0


@dataclass
class StatsJob:
    """Settings shared by every worker."""

    land_mask_path: Path | None = None
    gadm_path: Path | None = None
    #: Pixel size in metres. 0.20 for the published map.
    pixel_size_m: float = 0.20
    #: Fail on rasters holding codes this pipeline never writes.
    strict_codes: bool = True


@dataclass
class TileStats:
    """Area statistics for one tile."""

    tile_name: str
    state: str
    year: str
    date: str
    land_status: str
    tile_area_ha: float
    land_area_ha: float
    tree_area_ha: float
    tree_cover_pct: float
    error: str | None = None
    traceback: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_row(self) -> dict[str, Any]:
        row = {
            "tile_name": self.tile_name,
            "state": self.state,
            "year": self.year,
            "date": self.date,
            "land_status": self.land_status,
            "tile_area_ha": self.tile_area_ha,
            "land_area_ha": self.land_area_ha,
            "tree_area_ha": self.tree_area_ha,
            "tree_cover_pct": self.tree_cover_pct,
        }
        if self.error:
            row["error"] = self.error
        return row


# Per-worker globals: building StateLandMask costs seconds and hundreds of MB,
# so it is created once per process rather than once per tile.
_job: StatsJob | None = None
_land_mask: StateLandMask | None = None


def _worker_init(job: StatsJob) -> None:
    global _job, _land_mask
    _job = job
    _land_mask = (
        StateLandMask(job.land_mask_path, job.gadm_path)
        if job.land_mask_path and job.gadm_path
        else None
    )


def compute_tile_stats(
    tile: TileRef, job: StatsJob, land_mask: StateLandMask | None
) -> TileStats:
    """Measure tree cover on one tile.

    Args:
        tile: The prediction tile.
        job: Run settings.
        land_mask: Shared mask, or ``None`` to count the whole tile as land.

    Returns:
        A :class:`TileStats`. Failures are captured, not raised — one
        unreadable tile must not abort a run over hundreds of thousands.
    """
    pred_path, state, year = tile
    stats = TileStats(
        tile_name=pred_path.name,
        state=state,
        year=year,
        date=acquisition_date(pred_path, year),
        land_status="unknown",
        tile_area_ha=0.0,
        land_area_ha=0.0,
        tree_area_ha=0.0,
        tree_cover_pct=0.0,
    )

    try:
        with rasterio.open(pred_path) as src:
            data = src.read(1)
            bounds = tuple(src.bounds)
            height, width = src.height, src.width
            epsg = src.crs.to_epsg() if src.crs else None

        if job.strict_codes:
            validate_prediction_codes(data)

        if land_mask is not None:
            valid, status = land_mask.mask_for_bounds(epsg, bounds, height, width, state)
        else:
            valid, status = True, "no_mask"
        stats.land_status = status

        px_ha = job.pixel_size_m * job.pixel_size_m / M2_PER_HA
        total_px = height * width
        stats.tile_area_ha = total_px * px_ha

        if valid is None:
            # Entirely outside this state's land — contributes nothing.
            return stats

        if valid is True:
            land_px = total_px
            tree_px = int(np.sum(data == PRED_TREE))
        else:
            land_px = int(valid.sum())
            tree_px = int(np.sum((data == PRED_TREE) & valid))

        # Pixels the model never predicted are not land for our purposes.
        land_px -= int(np.sum((data == NODATA) & (valid if valid is not True else True)))
        land_px = max(land_px, 0)

        stats.land_area_ha = land_px * px_ha
        stats.tree_area_ha = tree_px * px_ha
        stats.tree_cover_pct = 100.0 * tree_px / land_px if land_px > 0 else 0.0
    except Exception as exc:  # noqa: BLE001 - one tile must not kill the run
        stats.error = str(exc)
        stats.traceback = traceback.format_exc()
    return stats


def _in_worker(tile: TileRef) -> TileStats:
    assert _job is not None, "worker not initialised"
    return compute_tile_stats(tile, _job, _land_mask)


def run(
    predictions_root: Path,
    job: StatsJob,
    states: list[str] | None = None,
    years: list[str] | None = None,
    workers: int = 1,
    limit: int | None = None,
    progress: bool = True,
) -> list[TileStats]:
    """Measure every matching tile under ``predictions_root``.

    Args:
        predictions_root: Root of the prediction archive.
        job: Settings shared by all workers.
        states: Restrict to these states.
        years: Restrict to these years.
        workers: Process count. 1 runs in-process, keeping tracebacks intact.
        limit: Stop after this many tiles (debugging).
        progress: Show a tqdm bar.

    Returns:
        One :class:`TileStats` per tile, in input order.
    """
    tiles = list(find_prediction_tiles(predictions_root, states, years))
    if limit is not None:
        tiles = tiles[:limit]
    if not tiles:
        return []

    if workers <= 1:
        _worker_init(job)
        return list(_maybe_progress((_in_worker(t) for t in tiles), len(tiles), progress))

    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_worker_init, initargs=(job,)) as pool:
        return list(
            _maybe_progress(pool.imap(_in_worker, tiles, chunksize=8), len(tiles), progress)
        )


def _maybe_progress(iterator, total: int, enabled: bool):
    if not enabled:
        return iterator
    from tqdm import tqdm

    return tqdm(iterator, total=total, unit="tile")


def summarise_by_state(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate per-tile rows into per-state totals.

    Areas sum; the percentage is recomputed from the summed areas rather
    than averaged over tiles, because tiles differ in land area and a plain
    mean would weight a mostly-water coastal tile like an inland one.
    """
    per_state: dict[str, dict[str, float]] = {}
    for row in rows:
        entry = per_state.setdefault(
            row["state"], {"land_area_ha": 0.0, "tree_area_ha": 0.0, "tiles": 0}
        )
        entry["land_area_ha"] += row["land_area_ha"]
        entry["tree_area_ha"] += row["tree_area_ha"]
        entry["tiles"] += 1

    for entry in per_state.values():
        land = entry["land_area_ha"]
        entry["tree_cover_pct"] = 100.0 * entry["tree_area_ha"] / land if land else 0.0
        entry["land_area_km2"] = entry["land_area_ha"] / 100.0
        entry["tree_area_km2"] = entry["tree_area_ha"] / 100.0
    return per_state
