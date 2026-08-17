"""Where inference imagery comes from.

The four original inference scripts existed because Germany's states publish
orthophotos three different ways, and each script hardcoded one of them:

* **Local rasters** — the imagery was bulk-downloaded to disk first. Formats
  and resolutions vary per state.
* **A VRT mosaic** — one virtual raster over a whole state, cut into tiles
  on the fly.
* **HTTP per tile** — a GeoJSON tile index carries a download URL column;
  tiles are fetched as needed, optionally with a matching nDOM.

All three yield the same thing: an ``(channels, height, width)`` array at
the model's target resolution, plus the georeferencing needed to write the
prediction back out. That contract is :class:`TileSource`.
"""

from __future__ import annotations

import io
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window

logger = logging.getLogger(__name__)

__all__ = [
    "TileTask",
    "TileData",
    "TileSource",
    "LocalRasterSource",
    "VrtSource",
    "HttpTileIndexSource",
    "resample_to_resolution",
]

_DATE_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")

#: Halo of neighbouring imagery read around each tile, in pixels.
#:
#: From the manuscript: *"To avoid boundary artifacts at tile edges, each
#: tile was read with 256 pixels of spatial context of neighboring tiles."*
#: The model sees the halo; the prediction is cropped back to the tile
#: before writing. Without it, patches at a tile border have no context on
#: one side and their predictions degrade — visible as seams along the
#: tile grid in the merged map.
CONTEXT_PX = 256


@dataclass(frozen=True)
class TileTask:
    """One unit of work: a tile to predict."""

    tile_id: str
    state: str
    #: Source raster, or the VRT the window is read from.
    source: Path | None = None
    #: Window into ``source`` for VRT-backed tiles.
    window: Window | None = None
    #: Download URL for HTTP-backed tiles.
    url: str | None = None
    #: Auxiliary nDOM/nDSM URL, where the model uses a height channel.
    ndsm_url: str | None = None
    #: Acquisition date, ``YYYYMMDD`` where known.
    date: str | None = None
    year: str | None = None


@dataclass
class TileData:
    """Imagery for one tile, at the model's target resolution.

    ``image`` may include a halo of neighbouring imagery; ``context_px``
    says how wide it is on each side. The model sees the halo, but the
    prediction is cropped back to the tile's own extent before it is
    written — see :attr:`inner_transform`.
    """

    task: TileTask
    image: np.ndarray
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    #: Halo width in pixels on each side. 0 means the array is the tile.
    context_px: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[2]

    @property
    def inner_shape(self) -> tuple[int, int]:
        """Shape of the tile itself, halo excluded."""
        height, width = self.shape
        return height - 2 * self.context_px, width - 2 * self.context_px

    @property
    def inner_transform(self) -> rasterio.Affine:
        """Georeferencing of the tile itself, halo excluded."""
        if not self.context_px:
            return self.transform
        return self.transform * rasterio.Affine.translation(
            self.context_px, self.context_px
        )

    def crop(self, array: np.ndarray) -> np.ndarray:
        """Cut the tile's own extent out of a halo-sized result."""
        if not self.context_px:
            return array
        c = self.context_px
        return array[..., c:-c, c:-c]


class TileSource(ABC):
    """Supplies tiles to the inference runner."""

    @abstractmethod
    def tasks(self) -> Iterator[TileTask]:
        """Enumerate the tiles to process, in a deterministic order."""

    @abstractmethod
    def load(self, task: TileTask) -> TileData | None:
        """Fetch and prepare one tile. ``None`` if it is unavailable."""


def resample_to_resolution(
    data: np.ndarray,
    transform: rasterio.Affine,
    crs,
    target_res_m: float,
    resampling: Resampling = Resampling.average,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Resample a raster to ``target_res_m``, returning data and transform.

    Downsampling (e.g. NRW's 10 cm DOP to 20 cm) uses ``average`` rather
    than nearest: averaging 2×2 blocks matches what the training patches saw
    and preserves sub-pixel detail that nearest-neighbour throws away.

    Returns the input untouched when it is already at the target resolution,
    so this is safe to call unconditionally.
    """
    current_res = max(abs(transform.a), abs(transform.e))
    if np.isclose(current_res, target_res_m, atol=1e-6):
        return data, transform

    scale = current_res / target_res_m
    n_bands, height, width = data.shape
    dst_height = max(1, int(round(height * scale)))
    dst_width = max(1, int(round(width * scale)))

    dst = np.zeros((n_bands, dst_height, dst_width), dtype=data.dtype)
    dst_transform = transform * rasterio.Affine.scale(width / dst_width, height / dst_height)

    for band in range(n_bands):
        reproject(
            source=data[band],
            destination=dst[band],
            src_transform=transform,
            src_crs=crs,
            dst_transform=dst_transform,
            dst_crs=crs,
            resampling=resampling,
        )
    return dst, dst_transform


def _date_from_name(path: Path | str) -> str | None:
    """Pull a plausible ``YYYYMMDD`` acquisition date out of a filename."""
    for match in _DATE_RE.finditer(Path(str(path)).stem):
        token = match.group(1)
        year, month, day = int(token[:4]), int(token[4:6]), int(token[6:8])
        if 1990 <= year <= 2099 and 1 <= month <= 12 and 1 <= day <= 31:
            return token
    return None


class LocalRasterSource(TileSource):
    """Orthophotos already on disk, one raster per tile.

    Args:
        root: Directory searched recursively.
        state: State code recorded on each task.
        n_channels: Bands to read. Extra bands in the file are ignored;
            too few is an error at load time.
        target_res_m: Resolution to resample to.
        patterns: Filename globs to accept.
    """

    def __init__(
        self,
        root: Path,
        state: str,
        n_channels: int = 3,
        target_res_m: float = 0.20,
        patterns: tuple[str, ...] = ("*.tif", "*.tiff", "*.jp2"),
    ):
        self.root = Path(root)
        self.state = state
        self.n_channels = n_channels
        self.target_res_m = target_res_m
        self.patterns = patterns

    def tasks(self) -> Iterator[TileTask]:
        seen: set[Path] = set()
        for pattern in self.patterns:
            for path in sorted(self.root.rglob(pattern)):
                if path in seen:
                    continue
                seen.add(path)
                yield TileTask(
                    tile_id=path.stem,
                    state=self.state,
                    source=path,
                    date=_date_from_name(path),
                )

    def load(self, task: TileTask) -> TileData | None:
        try:
            with rasterio.open(task.source) as src:
                if src.count < self.n_channels:
                    logger.warning(
                        "%s has %d band(s), model needs %d — skipping",
                        task.source.name, src.count, self.n_channels,
                    )
                    return None
                data = src.read(list(range(1, self.n_channels + 1)))
                transform, crs = src.transform, src.crs
        except rasterio.RasterioIOError as exc:
            logger.warning("Cannot read %s: %s", task.source, exc)
            return None

        data, transform = resample_to_resolution(data, transform, crs, self.target_res_m)
        return TileData(task, data, transform, crs)


class VrtSource(TileSource):
    """A single VRT mosaic, cut into fixed-size tiles on the fly.

    Used where a state's imagery is one seamless mosaic rather than
    per-tile files.

    Args:
        vrt_path: The mosaic.
        state: State code recorded on each task.
        tile_px: Tile side in pixels at the VRT's own resolution.
        n_channels: Bands to read.
        target_res_m: Resolution to resample to.
        skip_empty: Skip windows that read as entirely zero — a mosaic is
            mostly nodata outside the state and predicting there wastes GPU
            time.
    """

    def __init__(
        self,
        vrt_path: Path,
        state: str,
        tile_px: int = 5000,
        n_channels: int = 3,
        target_res_m: float = 0.20,
        skip_empty: bool = True,
        context_px: int = CONTEXT_PX,
    ):
        self.vrt_path = Path(vrt_path)
        self.state = state
        self.tile_px = tile_px
        self.n_channels = n_channels
        self.target_res_m = target_res_m
        self.skip_empty = skip_empty
        self.context_px = context_px

    def tasks(self) -> Iterator[TileTask]:
        with rasterio.open(self.vrt_path) as src:
            height, width = src.height, src.width
        for row in range(0, height, self.tile_px):
            for col in range(0, width, self.tile_px):
                window = Window(
                    col, row,
                    min(self.tile_px, width - col),
                    min(self.tile_px, height - row),
                )
                yield TileTask(
                    tile_id=f"r{row:07d}_c{col:07d}",
                    state=self.state,
                    source=self.vrt_path,
                    window=window,
                )

    def load(self, task: TileTask) -> TileData | None:
        window = task.window
        context = self.context_px
        # Read the tile plus a halo of neighbouring imagery. boundless=True
        # pads with nodata at the mosaic edge, where there is no neighbour.
        haloed = Window(
            window.col_off - context,
            window.row_off - context,
            window.width + 2 * context,
            window.height + 2 * context,
        )

        with rasterio.open(self.vrt_path) as src:
            data = src.read(
                list(range(1, self.n_channels + 1)),
                window=haloed, boundless=True, fill_value=0,
            )
            transform = src.window_transform(haloed)
            crs = src.crs

        if self.skip_empty and not data[:, context:-context or None,
                                        context:-context or None].any():
            return None

        data, transform = resample_to_resolution(data, transform, crs, self.target_res_m)
        # Resampling changes the pixel count, so scale the halo with it.
        scaled_context = int(round(context * (data.shape[2] / haloed.width)))
        return TileData(task, data, transform, crs, context_px=scaled_context)


class HttpTileIndexSource(TileSource):
    """Tiles downloaded on demand from URLs in a GeoJSON tile index.

    This is how NRW and Niedersachsen were processed: the index carries one
    row per tile with a download URL, and optionally a second URL for the
    nDOM height model used by the 5-channel ablations.

    Args:
        index: GeoJSON/GeoPackage tile index.
        state: State code recorded on each task.
        url_column: Column holding the orthophoto URL.
        id_column: Column holding the tile id.
        date_column: Column holding the acquisition date, if present.
        ndsm_url_column: Column holding the nDOM URL, if the model uses one.
        n_channels: Bands the model expects.
        target_res_m: Resolution to resample to.
        timeout: Per-request timeout, seconds.
        max_retries: Attempts per URL before giving up on a tile.
    """

    def __init__(
        self,
        index: Path,
        state: str,
        url_column: str,
        id_column: str = "id",
        date_column: str | None = "datetime",
        ndsm_url_column: str | None = None,
        n_channels: int = 3,
        target_res_m: float = 0.20,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.index = Path(index)
        self.state = state
        self.url_column = url_column
        self.id_column = id_column
        self.date_column = date_column
        self.ndsm_url_column = ndsm_url_column
        self.n_channels = n_channels
        self.target_res_m = target_res_m
        self.timeout = timeout
        self.max_retries = max_retries

    def tasks(self) -> Iterator[TileTask]:
        from treecover.io.vector import read_vector

        gdf = read_vector(self.index)
        for column in (self.url_column, self.id_column):
            if column not in gdf.columns:
                raise KeyError(
                    f"Column {column!r} not in {self.index.name}. "
                    f"Available: {list(gdf.columns)}"
                )

        for row in gdf.itertuples(index=False):
            date = None
            if self.date_column and hasattr(row, self.date_column):
                raw = getattr(row, self.date_column)
                date = str(raw)[:10].replace("-", "") if raw is not None else None
            yield TileTask(
                tile_id=str(getattr(row, self.id_column)),
                state=self.state,
                url=getattr(row, self.url_column),
                ndsm_url=(
                    getattr(row, self.ndsm_url_column, None) if self.ndsm_url_column else None
                ),
                date=date,
            )

    def _download(self, url: str) -> bytes | None:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response.content
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    logger.warning("Giving up on %s after %d attempts: %s",
                                   url, self.max_retries, exc)
                    return None
                logger.debug("Retry %d/%d for %s", attempt, self.max_retries, url)
        return None

    def load(self, task: TileTask) -> TileData | None:
        content = self._download(task.url)
        if content is None:
            return None

        with rasterio.open(io.BytesIO(content)) as src:
            available = min(src.count, self.n_channels)
            data = src.read(list(range(1, available + 1)))
            transform, crs = src.transform, src.crs

        if data.shape[0] < self.n_channels:
            logger.warning(
                "%s supplied %d band(s), model needs %d — skipping",
                task.tile_id, data.shape[0], self.n_channels,
            )
            return None

        data, transform = resample_to_resolution(data, transform, crs, self.target_res_m)

        if task.ndsm_url:
            ndsm = self._load_ndsm(task, transform, crs, data.shape[1:])
            if ndsm is None:
                return None
            data = np.concatenate([data, ndsm[np.newaxis]], axis=0)

        return TileData(task, data, transform, crs)

    def _load_ndsm(self, task, transform, crs, shape) -> np.ndarray | None:
        """Fetch the nDOM and warp it onto the orthophoto's grid."""
        content = self._download(task.ndsm_url)
        if content is None:
            return None
        with rasterio.open(io.BytesIO(content)) as src:
            dst = np.zeros(shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.average,
            )
        # Training clipped nDSM at zero — negative heights are artefacts.
        return np.clip(np.nan_to_num(dst, nan=0.0), 0, None)
