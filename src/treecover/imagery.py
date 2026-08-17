"""Locating the orthophoto behind a prediction tile.

The archive stores imagery beside the predictions it produced::

    <STATE>/<YEAR>/RGB/<CELL>/dop20rgb_32_682_5470_by_file_20230910.jp2
    <STATE>/<YEAR>/predictions/<CELL>/dop20rgb_32_682_5470_by_file_20230910_pred.tif

so an ortho is found by name, not by a spatial index — the two differ only
in the folder and the ``_pred`` suffix.

Two wrinkles the layout hides:

* The folder is ``RGB`` in the states that publish three bands and
  ``RGBI`` in the ten that publish four. Both are searched.
* Imagery is JPEG 2000 in most states and GeoTIFF in some.

Reading a window rather than the whole file matters here: a 1 km tile at
20 cm is 5000 × 5000 px, and the figures need a few hundred pixels of it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds

logger = logging.getLogger(__name__)

__all__ = [
    "IMAGE_DIRS",
    "IMAGE_SUFFIXES",
    "ortho_for_prediction",
    "find_prediction_for_point",
    "find_prediction_by_stem",
    "state_from_stem",
    "date_from_stem",
    "read_rgb_window",
    "read_band_window",
]

#: Folder names imagery lives under. ``RGB`` for three-band states,
#: ``RGBI`` for the four-band ones.
IMAGE_DIRS = ("RGB", "RGBI")

#: Formats the states publish.
IMAGE_SUFFIXES = (".jp2", ".tif", ".tiff")

PRED_SUFFIX = "_pred.tif"


def ortho_for_prediction(pred_path: Path) -> Path | None:
    """Find the orthophoto a prediction tile was made from.

    Args:
        pred_path: ``.../<YEAR>/predictions/<CELL>/<stem>_pred.tif``.

    Returns:
        The matching image, or ``None`` if it is not on disk — most states'
        imagery was deleted after inference, so a missing ortho is normal
        rather than an error.
    """
    year_dir = pred_path.parent.parent.parent
    cell = pred_path.parent.name
    stem = pred_path.name
    stem = stem[: -len(PRED_SUFFIX)] if stem.endswith(PRED_SUFFIX) else pred_path.stem

    for folder in IMAGE_DIRS:
        for suffix in IMAGE_SUFFIXES:
            candidate = year_dir / folder / cell / f"{stem}{suffix}"
            if candidate.exists():
                return candidate

    # Some states renamed the imagery slightly; fall back to a glob on the
    # cell directory before giving up.
    for folder in IMAGE_DIRS:
        directory = year_dir / folder / cell
        if not directory.is_dir():
            continue
        matches = sorted(directory.glob(f"{stem}.*"))
        if matches:
            return matches[0]
    return None


def find_prediction_for_point(
    predictions_root: Path, state: str, easting: float, northing: float
) -> Path | None:
    """Find the prediction tile covering a projected coordinate.

    Tile names encode the kilometre grid, so the cell follows from the
    coordinate without opening anything: ``easting 682_450`` is in the
    ``682`` column.

    Args:
        predictions_root: Root of the archive.
        state: State directory name.
        easting: X in the state's projected CRS, metres.
        northing: Y in the same CRS.

    Returns:
        The prediction tile, or ``None``. Where several years cover the
        point, the newest is returned — the same preference the merge uses.
    """
    east_km = int(easting // 1000)
    north_km = int(northing // 1000)
    state_dir = Path(predictions_root) / state
    if not state_dir.is_dir():
        logger.debug("No such state directory: %s", state_dir)
        return None

    matches: list[Path] = []
    for year_dir in sorted(state_dir.iterdir(), reverse=True):
        pred_dir = year_dir / "predictions"
        if not pred_dir.is_dir():
            continue
        # `*_<east>_<north>_*` is specific enough to hit one tile per year.
        matches.extend(pred_dir.glob(f"*/*_{east_km}_{north_km}_*{PRED_SUFFIX}"))
        if matches:
            break
    return sorted(matches)[-1] if matches else None


def state_from_stem(stem: str) -> str | None:
    """Read the state code out of a tile name.

    ``dop20rgbi_33_411_5654_sn_file_20240319`` -> ``'SN'``. The code sits
    between the grid cell and ``file``, which is the one part of the name
    every state writes the same way.
    """
    match = re.search(r"_([a-z]{2})_file", stem, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def date_from_stem(stem: str) -> str | None:
    """Read the acquisition date out of a tile name, as ``YYYY-MM-DD``.

    The date is the only eight-digit run in the name, so it is taken by
    shape rather than by position — the states put it after ``file`` but
    not always with the same separators.
    """
    match = re.search(r"\d{8}", stem)
    if not match:
        return None
    digits = match.group()
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"


def find_prediction_by_stem(
    predictions_root: Path, stem: str, state: str | None = None
) -> Path | None:
    """Find a prediction tile by its file name.

    Args:
        predictions_root: Root of the archive.
        stem: Tile name, with or without the ``_pred`` suffix.
        state: State directory. Derived from the name when omitted —
            searching all sixteen states costs a directory walk each.

    Returns:
        The prediction tile, or ``None``. Where several years hold the same
        name, the newest is returned, as :func:`find_prediction_for_point`
        does.
    """
    stem = stem[: -len("_pred")] if stem.endswith("_pred") else stem
    state = state or state_from_stem(stem)
    root = Path(predictions_root)
    bases = [root / state] if state and (root / state).is_dir() else [root]

    matches: list[Path] = []
    for base in bases:
        matches.extend(base.glob(f"*/predictions/*/{stem}{PRED_SUFFIX}"))
        if not matches:
            matches.extend(base.glob(f"*/*/predictions/*/{stem}{PRED_SUFFIX}"))
    return sorted(matches)[-1] if matches else None


def read_rgb_window(
    path: Path,
    bounds: tuple[float, float, float, float],
    bounds_crs=None,
    max_pixels: int = 2000,
) -> np.ndarray | None:
    """Read an RGB window from an image, as ``(height, width, 3)`` uint8.

    Args:
        path: Image file.
        bounds: ``(minx, miny, maxx, maxy)``.
        bounds_crs: CRS of ``bounds``. ``None`` means they already match
            the image.
        max_pixels: Cap on the longer side. A whole 1 km tile is 5000 px;
            a figure panel is a few hundred, and decoding the full window
            of a JPEG 2000 to then shrink it wastes most of the time.

    Returns:
        The window, or ``None`` if it does not intersect the image. The
        first three bands are taken, so an RGBI image yields its RGB.
    """
    try:
        with rasterio.open(path) as src:
            if bounds_crs is not None and src.crs != bounds_crs:
                from rasterio.warp import transform_bounds

                bounds = transform_bounds(bounds_crs, src.crs, *bounds)

            if not _intersects(src.bounds, bounds):
                return None

            window = window_from_bounds(*bounds, transform=src.transform)
            height = max(1, int(window.height))
            width = max(1, int(window.width))
            scale = min(1.0, max_pixels / max(height, width))
            out_shape = (min(3, src.count), max(1, int(height * scale)),
                         max(1, int(width * scale)))

            data = src.read(
                indexes=list(range(1, out_shape[0] + 1)),
                window=window, out_shape=out_shape,
                boundless=True, fill_value=0,
            )
    except rasterio.RasterioIOError as exc:
        logger.warning("Cannot read %s: %s", Path(path).name, exc)
        return None

    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    return np.transpose(data[:3], (1, 2, 0)).astype(np.uint8)


def read_band_window(
    path: Path,
    bounds: tuple[float, float, float, float],
    bounds_crs=None,
    band: int = 1,
    max_pixels: int = 2000,
    resampling: str = "nearest",
) -> np.ndarray | None:
    """Read one band of a window, as float32 with nodata as ``NaN``.

    The single-band counterpart to :func:`read_rgb_window`, for the layers
    that are a measurement rather than a picture: height models and label
    masks. Same bounds-first contract, because those layers rarely share a
    grid with the orthophoto — an nDSM is 1 m where the DOP is 20 cm, so a
    pixel window read from one is meaningless in the other.

    Args:
        path: Raster file.
        bounds: ``(minx, miny, maxx, maxy)``.
        bounds_crs: CRS of ``bounds``. ``None`` means they already match.
        band: 1-based band index.
        max_pixels: Cap on the longer side, as in :func:`read_rgb_window`.
        resampling: ``"nearest"`` or ``"bilinear"``. Keep nearest for a
            class mask — averaging class codes invents classes that are not
            in the legend. Height may use bilinear.

    Returns:
        The window as float32 with the file's nodata value replaced by
        ``NaN``, or ``None`` if it does not intersect the raster.
    """
    from rasterio.enums import Resampling

    try:
        with rasterio.open(path) as src:
            if bounds_crs is not None and src.crs != bounds_crs:
                from rasterio.warp import transform_bounds

                bounds = transform_bounds(bounds_crs, src.crs, *bounds)

            if not _intersects(src.bounds, bounds):
                return None

            window = window_from_bounds(*bounds, transform=src.transform)
            height = max(1, int(window.height))
            width = max(1, int(window.width))
            scale = min(1.0, max_pixels / max(height, width))
            out_shape = (max(1, int(height * scale)), max(1, int(width * scale)))

            nodata = src.nodatavals[band - 1]
            data = src.read(
                band, window=window, out_shape=out_shape,
                boundless=True, fill_value=0,
                resampling=getattr(Resampling, resampling),
            ).astype(np.float32)
    except rasterio.RasterioIOError as exc:
        logger.warning("Cannot read %s: %s", Path(path).name, exc)
        return None

    if nodata is not None and not np.isnan(nodata):
        data[data == nodata] = np.nan
    return data


def _intersects(a, b) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])
