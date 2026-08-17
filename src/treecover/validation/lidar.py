"""Fetching LiDAR point clouds from the state surveying authorities.

Each federal state serves airborne laser scanning differently — Bavaria as
bare ``.laz``, Brandenburg zipped, NRW as ``.laz`` under a different path
scheme. The URL templates live in ``configs/states.yaml``; what varies here
is only whether the download needs unpacking.

Downloads are cached on disk and written atomically via a ``.tmp`` file, so
an interrupted run leaves no half-file that a later run would happily read
as a valid tile. A cached ZIP is integrity-checked before use, because a
truncated archive is the most common failure mode when a state's server
drops the connection mid-transfer.
"""

from __future__ import annotations

import logging
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .chm import PointCloud

logger = logging.getLogger(__name__)

__all__ = ["download_tile", "read_point_cloud", "load_points_for_bounds"]

_POINT_SUFFIXES = (".laz", ".las")


def download_tile(
    url: str,
    cache_dir: Path,
    max_retries: int = 5,
    retry_delay: float = 10.0,
) -> Path | None:
    """Fetch one LiDAR tile, unpacking it if the state serves a ZIP.

    Args:
        url: Tile URL, from :meth:`treecover.config.StateConfig.lidar_url`.
        cache_dir: Where tiles are kept. Reused across runs.
        max_retries: Attempts before giving up on this tile.
        retry_delay: Seconds between attempts. State servers rate-limit, so
            retrying immediately mostly wastes attempts.

    Returns:
        Path to the ``.laz``/``.las`` file, or ``None`` if it could not be
        obtained. A failed tile is not fatal: the sample it belongs to is
        skipped and reported.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = url.rsplit("/", 1)[-1]
    local = cache_dir / filename

    if filename.lower().endswith(".zip"):
        return _download_and_extract(url, local, cache_dir, max_retries, retry_delay)

    if local.exists() and local.stat().st_size > 0:
        return local
    return local if _fetch(url, local, max_retries, retry_delay) else None


def _fetch(url: str, dest: Path, max_retries: int, retry_delay: float) -> bool:
    """Download to ``dest`` atomically. True on success."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    for attempt in range(1, max_retries + 1):
        try:
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size == 0:
                raise OSError("downloaded file is empty")
            tmp.replace(dest)
            return True
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            tmp.unlink(missing_ok=True)
            if attempt == max_retries:
                logger.warning("Gave up on %s after %d attempts: %s",
                               url, max_retries, exc)
                return False
            logger.debug("Attempt %d/%d failed for %s (%s); retrying in %.0fs",
                         attempt, max_retries, url, exc, retry_delay)
            time.sleep(retry_delay)
    return False


def _download_and_extract(
    url: str, zip_path: Path, cache_dir: Path, max_retries: int, retry_delay: float
) -> Path | None:
    """Fetch a zipped tile and return the point file inside it."""
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path) as archive:
                archive.testzip()
        except (zipfile.BadZipFile, OSError):
            logger.info("Cached archive %s is corrupt — refetching", zip_path.name)
            zip_path.unlink(missing_ok=True)

    if not zip_path.exists() and not _fetch(url, zip_path, max_retries, retry_delay):
        return None

    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = [m for m in archive.namelist()
                       if m.lower().endswith(_POINT_SUFFIXES)]
            if not members:
                logger.warning("No .laz/.las inside %s", zip_path.name)
                return None

            member = members[0]
            extracted = cache_dir / Path(member).name
            if extracted.exists() and extracted.stat().st_size > 0:
                return extracted

            archive.extract(member, cache_dir)
            nested = cache_dir / member
            if nested != extracted and nested.exists():
                shutil.move(str(nested), extracted)
            return extracted
    except (zipfile.BadZipFile, OSError) as exc:
        logger.warning("Cannot extract %s: %s", zip_path.name, exc)
        zip_path.unlink(missing_ok=True)
        return None


def read_point_cloud(path: Path) -> PointCloud | None:
    """Read a ``.laz``/``.las`` file into a :class:`PointCloud`.

    Coordinates are returned already scaled to metres — laspy exposes both
    raw integers and scaled floats, and using the raw ones silently produces
    a CHM in the wrong units.
    """
    try:
        import laspy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading LiDAR needs laspy. Install it with: pip install 'laspy[laszip]'"
        ) from exc

    try:
        with laspy.open(str(path)) as reader:
            las = reader.read()
    except Exception as exc:  # noqa: BLE001 - laspy raises many types
        logger.warning("Cannot read %s: %s", Path(path).name, exc)
        return None

    classification = (
        np.asarray(las.classification) if hasattr(las, "classification") else None
    )
    return PointCloud(
        x=np.asarray(las.x, dtype=np.float64),
        y=np.asarray(las.y, dtype=np.float64),
        z=np.asarray(las.z, dtype=np.float64),
        classification=classification,
    )


def load_points_for_bounds(
    urls: list[str],
    bounds: tuple[float, float, float, float],
    cache_dir: Path,
    buffer_m: float = 50.0,
    cache: dict | None = None,
) -> PointCloud:
    """Collect the points covering ``bounds`` from every overlapping tile.

    A 25 m validation box routinely straddles two 1 km LiDAR tiles, so the
    points are gathered from all of them and merged before rasterising.

    Args:
        urls: Tile URLs that intersect the box.
        bounds: ``(minx, miny, maxx, maxy)`` in the tiles' CRS.
        cache_dir: Download cache.
        buffer_m: Extra margin kept around ``bounds``. The DTM
            interpolation needs ground returns from beyond the box edge,
            otherwise the CHM shows a frame of spurious height.
        cache: Optional in-memory cache keyed by path. Neighbouring samples
            usually share a tile, and re-reading a 100 MB LAZ per sample
            dominates runtime otherwise.

    Returns:
        The merged, clipped cloud. Empty if no tile could be obtained.
    """
    clouds = []
    for url in urls:
        path = download_tile(url, cache_dir)
        if path is None:
            continue

        if cache is not None and str(path) in cache:
            cloud = cache[str(path)]
        else:
            cloud = read_point_cloud(path)
            if cache is not None and cloud is not None:
                cache[str(path)] = cloud
        if cloud is None:
            continue

        clipped = cloud.clip(bounds, buffer=buffer_m)
        if len(clipped):
            clouds.append(clipped)

    return PointCloud.concatenate(clouds)
