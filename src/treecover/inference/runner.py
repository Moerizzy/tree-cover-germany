"""Driving inference across a state and writing the outputs.

Writes, per tile:

* ``<out_root>/<STATE>/<YEAR>/predictions/<UTM_TILE>/<id>_pred.tif`` — the
  binary mask, matching the layout :mod:`treecover.io.tiles` expects.
* one line in ``tile_results.jsonl`` — tree cover, uncertainty, coverage.

The JSONL report is appended after every tile rather than collected in
memory and written at the end, so a run killed after two days still leaves
a usable record and can be resumed by skipping tiles already listed.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio

from .predictor import InferenceConfig, Predictor
from .sources import TileSource, TileTask

logger = logging.getLogger(__name__)

__all__ = ["InferenceRun", "TileReport", "run_inference", "prediction_path"]

REPORT_NAME = "tile_results.jsonl"


@dataclass
class TileReport:
    """One line of the run report."""

    tile_id: str
    state: str
    year: str | None
    date: str | None
    pred_path: str | None
    tree_cover_pct: float | None = None
    mean_uncertainty: float | None = None
    coverage_pct: float | None = None
    n_patches: int | None = None
    status: str = "ok"
    error: str | None = None


@dataclass
class InferenceRun:
    """Settings for one inference run."""

    out_root: Path
    config: InferenceConfig
    #: Directory level between year and tile; kept configurable so a test
    #: run can write to `predictions_test` without touching real output.
    subdir: str = "predictions"
    #: Fallback year when a tile carries no date.
    default_year: str = "unknown"
    overwrite: bool = False
    #: Also write a float32 uncertainty raster beside each prediction.
    write_uncertainty: bool = False


def prediction_path(run: InferenceRun, task: TileTask, year: str) -> Path:
    """Where a tile's prediction goes.

    Mirrors the archive layout so that
    :func:`treecover.io.tiles.find_prediction_tiles` finds the result
    without any further configuration.
    """
    # Group tiles into 10 km cells the way the rest of the project does.
    return (
        run.out_root / task.state / year / run.subdir / _grid_cell(task.tile_id)
        / f"{task.tile_id}_pred.tif"
    )


# Two tile-id spellings occur in the archive:
#   dop20rgb_32_660_5261_by_file_20240730   zone, easting km, northing km
#   32660_5261                              zone-prefixed easting, northing km
_TILE_ID_SPACED = re.compile(r"(?<!\d)(3[23])_(\d{3})_(\d{4})(?!\d)")
_TILE_ID_JOINED = re.compile(r"(?<!\d)(3[23])(\d{3})_(\d{4})(?!\d)")


def _grid_cell(tile_id: str) -> str:
    """Derive the 10 km grid-cell directory name from a tile id.

    German tile ids carry the UTM zone plus easting and northing in
    kilometres. The archive groups tiles into 10 km cells whose name states
    the coordinates in units of 100 m, so ``32_660_5261`` belongs to
    ``UTM32_E6600_N52600``.

    Ids that match neither spelling fall back to a single ``misc`` bucket
    rather than raising — the grouping is for filesystem hygiene, not
    correctness, and an unrecognised name must not abort a nationwide run.
    """
    for pattern in (_TILE_ID_SPACED, _TILE_ID_JOINED):
        match = pattern.search(tile_id)
        if match:
            zone, east_km, north_km = match.groups()
            return (
                f"UTM{zone}"
                f"_E{int(east_km) // 10 * 100}"
                f"_N{int(north_km) // 10 * 100}"
            )
    return "misc"


def _already_done(path: Path, overwrite: bool) -> bool:
    return path.exists() and not overwrite


def run_inference(
    source: TileSource,
    run: InferenceRun,
    limit: int | None = None,
    progress: bool = True,
) -> list[TileReport]:
    """Predict every tile ``source`` yields.

    Args:
        source: Where imagery comes from.
        run: Output settings.
        limit: Stop after this many tiles (debugging).
        progress: Show a tqdm bar.

    Returns:
        One :class:`TileReport` per tile attempted.
    """
    predictor = Predictor(run.config)
    run.out_root.mkdir(parents=True, exist_ok=True)
    report_path = run.out_root / REPORT_NAME

    tasks = list(source.tasks())
    if limit is not None:
        tasks = tasks[:limit]
    logger.info("%d tile(s) to process", len(tasks))

    iterator = tasks
    if progress:
        from tqdm import tqdm

        iterator = tqdm(tasks, unit="tile")

    reports: list[TileReport] = []
    for task in iterator:
        report = _process(task, source, predictor, run)
        reports.append(report)
        with report_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(report)) + "\n")

    ok = sum(r.status == "ok" for r in reports)
    logger.info("Done: %d/%d succeeded, report at %s", ok, len(reports), report_path)
    return reports


def _process(
    task: TileTask, source: TileSource, predictor: Predictor, run: InferenceRun
) -> TileReport:
    """Predict one tile, converting any failure into a report row."""
    year = task.year or (task.date[:4] if task.date else run.default_year)
    out_path = prediction_path(run, task, year)

    if _already_done(out_path, run.overwrite):
        return TileReport(task.tile_id, task.state, year, task.date,
                          str(out_path), status="skipped")

    try:
        data = source.load(task)
        if data is None:
            return TileReport(task.tile_id, task.state, year, task.date,
                              None, status="unavailable")

        result = predictor.predict(data.image, return_maps=run.write_uncertainty)

        # The model saw a halo of neighbouring imagery; only the tile's own
        # extent is written out.
        prediction = data.crop(result.prediction)
        transform = data.inner_transform

        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_raster(out_path, prediction, transform, data.crs,
                      dtype="uint8", nodata=None)

        if run.write_uncertainty and result.uncertainty_map is not None:
            _write_raster(
                out_path.with_name(out_path.name.replace("_pred.tif", "_unc.tif")),
                data.crop(result.uncertainty_map), transform, data.crs,
                dtype="float32", nodata=None,
            )

        return TileReport(
            tile_id=task.tile_id,
            state=task.state,
            year=year,
            date=task.date,
            pred_path=str(out_path),
            # Reported over the tile itself, not the halo — otherwise a
            # tile's tree cover would include a strip of its neighbours.
            tree_cover_pct=float(100.0 * (prediction == 1).sum() / prediction.size)
            if prediction.size else 0.0,
            mean_uncertainty=result.mean_uncertainty,
            coverage_pct=result.coverage_pct,
            n_patches=result.n_patches,
        )
    except Exception as exc:  # noqa: BLE001 - one tile must not kill the run
        logger.exception("Tile %s failed", task.tile_id)
        return TileReport(task.tile_id, task.state, year, task.date, None,
                          status="error", error=str(exc))


def _write_raster(path: Path, data: np.ndarray, transform, crs, dtype: str, nodata) -> None:
    """Write a single-band raster with LZW compression and internal tiling."""
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    ) as dst:
        dst.write(data.astype(dtype), 1)
