"""Where and when the mapped imagery was acquired.

The acquisition-date figure and the merged raster must agree: if the figure
states that an area came from July 2023, the raster there has to be the
July 2023 tile. They are therefore built from the same selection —
:func:`treecover.merge.select_one_per_cell` — rather than from two
independent passes over the archive, which is how the original notebooks
could disagree.

Tile bounds come from the **filename**, not from opening the raster.
``dop20rgb_32_682_5470_by_file_20230910`` is UTM zone 32, easting 682 km,
northing 5470 km, and tiles are 1 km. Reading 380,000 GeoTIFF headers to
learn something already written in the name costs about an hour.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from treecover.io.tiles import find_prediction_tiles
from treecover.merge import TileCandidate, cell_key, select_one_per_cell, tile_date

logger = logging.getLogger(__name__)

__all__ = ["build_coverage", "TILE_SIZE_M", "SEASON_OF_MONTH"]

#: Side length of one prediction tile, metres.
TILE_SIZE_M = 1000.0

#: Month -> phenological season, for colouring the coverage map.
SEASON_OF_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

_TRANSFORMERS: dict[int, Transformer] = {}


def _to_wgs84(zone: str) -> Transformer:
    """Cached transformer from a UTM zone to WGS 84."""
    epsg = 25832 if zone == "32" else 25833
    if epsg not in _TRANSFORMERS:
        _TRANSFORMERS[epsg] = Transformer.from_crs(epsg, 4326, always_xy=True)
    return _TRANSFORMERS[epsg]


def build_coverage(
    predictions_root: Path,
    states: list[str] | None = None,
    resolve_overlaps: bool = True,
) -> pd.DataFrame:
    """Tabulate every mapped tile with its position and acquisition date.

    Args:
        predictions_root: Root of the prediction archive.
        states: Restrict to these states.
        resolve_overlaps: Keep one tile per 1 km cell, newest acquisition
            date first — the same rule the merge uses. Set ``False`` to see
            the raw archive including the duplicates at state borders.

    Returns:
        One row per cell with ``lon_c``, ``lat_c``, ``date``, ``year``,
        ``month``, ``season`` and ``state``. Tiles whose filename encodes
        no cell, and those with no usable date, are dropped and counted in
        the log — a tile that cannot be placed or dated cannot appear on a
        date map.
    """
    candidates = [
        TileCandidate(path=t.path, state=t.state, year=t.year,
                      date=tile_date(t.path, t.year))
        for t in find_prediction_tiles(predictions_root, states)
    ]
    if not candidates:
        return pd.DataFrame()

    if resolve_overlaps:
        chosen = [choice.winner for choice in select_one_per_cell(candidates)]
    else:
        chosen = candidates

    rows = []
    unplaceable = undated = 0
    for candidate in chosen:
        cell = cell_key(candidate.path)
        if cell is None:
            unplaceable += 1
            continue
        if not candidate.is_dated:
            undated += 1
            continue

        zone, east_km, north_km = cell
        # Tile centre, in the zone's projected CRS.
        easting = int(east_km) * 1000.0 + TILE_SIZE_M / 2
        northing = int(north_km) * 1000.0 + TILE_SIZE_M / 2
        lon, lat = _to_wgs84(zone).transform(easting, northing)

        rows.append(
            {
                "state": candidate.state,
                "date": candidate.date,
                "year": int(candidate.date[:4]),
                "month": int(candidate.date[4:6]),
                "lon_c": lon,
                "lat_c": lat,
            }
        )

    if unplaceable:
        logger.warning("%d tile(s) could not be placed from their filename", unplaceable)
    if undated:
        logger.warning(
            "%d tile(s) carry no acquisition date and are absent from the coverage "
            "map. They are still in the raster — the map understates coverage by "
            "that much.", undated,
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["season"] = frame["month"].map(SEASON_OF_MONTH)
    logger.info("Coverage: %d tile(s), %d state(s), %s–%s",
                len(frame), frame["state"].nunique() if not frame.empty else 0,
                frame["date"].min() if not frame.empty else "-",
                frame["date"].max() if not frame.empty else "-")
    return frame


def rasterise_mode(
    frame: pd.DataFrame, column: str, dlon: float, dlat: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Most frequent value of ``column`` per grid cell.

    The mode, not the mean: a month or a year is categorical, and averaging
    August and February into May would be meaningless.

    Returns:
        ``(image, lon_edges, lat_edges)``. Empty cells are NaN.
    """
    lon_edges = np.arange(
        np.floor(frame["lon_c"].min() / dlon) * dlon,
        np.ceil(frame["lon_c"].max() / dlon) * dlon + dlon, dlon,
    )
    lat_edges = np.arange(
        np.floor(frame["lat_c"].min() / dlat) * dlat,
        np.ceil(frame["lat_c"].max() / dlat) * dlat + dlat, dlat,
    )

    binned = frame.assign(
        lon_bin=pd.cut(frame["lon_c"], bins=lon_edges, labels=False),
        lat_bin=pd.cut(frame["lat_c"], bins=lat_edges, labels=False),
    ).dropna(subset=["lon_bin", "lat_bin", column])

    modes = (
        binned.groupby(["lat_bin", "lon_bin"], observed=True)[column]
        .agg(lambda values: values.mode().iloc[0])
    )

    image = np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan)
    for (lat_bin, lon_bin), value in modes.items():
        image[int(lat_bin), int(lon_bin)] = value
    return image, lon_edges, lat_edges
