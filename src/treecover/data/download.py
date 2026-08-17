"""Fetching the training rasters: orthophotos and the two height models.

Stage 1 returns tile ids; stage 3 needs the actual rasters on disk, named
``<tile_id>_<date>.tif``. In between sits this module, which turns a tile
geometry into download URLs and the downloads into the three products the
training set is built from:

* **DOP** — the orthophoto. In Lower Saxony it is published on a **2 km**
  grid while the sample is 1 km, so a DOP is cropped to its tile after
  download. Elsewhere the grids may agree and the crop is a no-op.
* **bDOM** — the image-based surface model, 20 cm, from the same flight as
  the orthophoto.
* **DGM1** — the LiDAR ground model, 1 m, flown once and reused.

``bDOM − DGM1`` is the nDSM the labels were digitised with. It is
image-based apart from the ground component, which is why it is as high
over a roof as over a crown — see :mod:`treecover.figures.training_examples`
for what that means for figure 3.

Two sources resolve the URLs, and the difference between them is the whole
reason both exist:

``easygeodata``
    The `easygeodata.de <https://easygeodata.de>`_ Extract API, which
    indexes the open geodata of all sixteen states behind one bbox query.
    Nothing local is needed and it covers states this project never built
    an index for. **It serves the current acquisition of a tile only.**

``index``
    The state's own acquisition index — the same GeoJSON stage 1 samples
    from — which lists *every* flight over a tile with its URL. This is
    the only route to the historical acquisitions, and the published
    training set is built from pairs of them.

So: the API is the convenient route and the right one for a fresh area;
reproducing the published training set needs the index. A run that asks
for dates the API cannot serve says so rather than quietly fetching one
image per tile.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlencode

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "EASYGEODATA_URL",
    "PRODUCTS",
    "Link",
    "EasyGeoData",
    "tile_key_to_id",
    "tile_id_to_key",
    "links_from_index",
    "match_acquisitions",
    "pair_ground_models",
    "download",
    "crop_to_geometry",
    "normalised_surface",
]

#: Base URL of the Extract API. Overridable so a mirror or a local
#: deployment can be used without touching the code.
EASYGEODATA_URL = "https://easygeodata.de/api"

#: The three products, and where each sits on disk. The directory names
#: are the ones ``paths.yaml`` and stage 3 expect.
PRODUCTS = {
    "dop": "DOP",
    "bdom": "bDOM",
    "dgm": "DGM",
}

#: Written into the nDSM where either input has none.
NDSM_NODATA = -9999.0


@dataclass
class Link:
    """One downloadable raster."""

    tile_id: str
    product: str
    url: str
    date: str | None = None
    state: str | None = None
    gsd: float | None = None
    bands: int | None = None

    @property
    def filename(self) -> str:
        """``<tile_id>_<date>.tif`` — the name stage 3 parses.

        A product without a date (the ground model is flown once and
        reused) still gets one, because the training pipeline pairs files
        by name and an undated DGM would match no orthophoto.
        """
        return f"{self.tile_id}_{self.date}.tif" if self.date else f"{self.tile_id}.tif"

    def as_row(self) -> dict:
        return {
            "tile_id": self.tile_id, "date": self.date, "product": self.product,
            "state": self.state, "gsd": self.gsd, "bands": self.bands, "url": self.url,
        }


def tile_key_to_id(key: str) -> str:
    """``"32_416_5830"`` → ``"324165830"``, the id used everywhere else."""
    return str(key).replace("_", "")


def tile_id_to_key(tile_id: str) -> str:
    """``"324165830"`` → ``"32_416_5830"``, the API's spelling.

    The split is by position, not by separator: zone, then the easting in
    whole kilometres, then the northing. Same assumption as
    :func:`treecover.data.tile_sampling.border_tile_ids` — three-digit
    eastings, which holds across Germany's UTM 32/33 zones.
    """
    text = str(tile_id)
    return f"{text[:2]}_{text[2:-4]}_{text[-4:]}"


class EasyGeoData:
    """Client for the easygeodata Extract API.

    One HTTP request per tile per product. The API answers a bounding box
    with every tile it intersects, so a request for a 1 km tile returns
    its neighbours too — they are filtered out by tile id rather than by
    shrinking the box, because a tile whose bounds are rounded differently
    upstream would otherwise vanish.

    Args:
        base_url: API root.
        timeout: Per-request timeout, seconds.
        retries: Attempts per request. The API is a free service; the
            default backs off rather than hammering it.
        pause: Seconds between requests.
    """

    def __init__(self, base_url: str = EASYGEODATA_URL, timeout: float = 60.0,
                 retries: int = 3, pause: float = 0.2) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.pause = pause
        self._last_request = 0.0

    def _get(self, path: str, **params) -> dict:
        """One GET, retried with linear backoff, returning parsed JSON."""
        import json
        import urllib.error
        import urllib.request

        query = {k: v for k, v in params.items() if v is not None}
        url = f"{self.base_url}{path}?{urlencode(query)}"

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.pause:
                time.sleep(self.pause - elapsed)
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as response:
                    self._last_request = time.monotonic()
                    return json.load(response)
            except urllib.error.HTTPError as error:
                self._last_request = time.monotonic()
                if error.code == 400:
                    # A rejected parameter will be rejected again.
                    raise ValueError(
                        f"easygeodata rejected {url}: {error.reason}. Check the "
                        "product name — the API accepts 'dop' and 'bdom', not the "
                        "state's own spellings."
                    ) from error
                last_error = error
            except Exception as error:  # noqa: BLE001 — network, any failure retries
                self._last_request = time.monotonic()
                last_error = error

            if attempt < self.retries:
                logger.debug("Retry %d/%d for %s: %s", attempt, self.retries, url,
                             last_error)
                time.sleep(self.pause * attempt)

        raise ConnectionError(f"easygeodata unreachable after {self.retries} "
                              f"attempts: {url} ({last_error})")

    def orthophotos(self, bounds, state: str | None = None) -> list:
        """DOP links intersecting ``bounds`` (minx, miny, maxx, maxy, WGS84)."""
        payload = self._get("/dop/links", bbox=_bbox(bounds), state=state)
        return [
            Link(
                tile_id=tile_key_to_id(item["tile_key"]), product="dop",
                url=item["url"], date=item.get("aktualitaet"),
                state=item.get("state"), gsd=item.get("gsd"),
                bands=item.get("bands"),
            )
            for item in payload.get("links") or []
        ]

    def surface_models(self, bounds, state: str | None = None) -> list:
        """bDOM links — the image-based surface model, same flight as the DOP."""
        payload = self._get("/dgm/links", bbox=_bbox(bounds), state=state,
                            produkt="bdom")
        return [
            Link(
                tile_id=tile_key_to_id(item["tile_key"]), product="bdom",
                url=item["url"], date=item.get("aktualitaet"),
                state=item.get("state"), gsd=item.get("gsd"),
            )
            for item in payload.get("links") or []
        ]

    def ground_models(self, bounds, state: str | None = None) -> list:
        """DGM1 links — the LiDAR ground model.

        A different endpoint from the other two: the ground model is not a
        flight product, so the API lists it under ``/dgm/tiles`` with its
        own acquisition date, which is usually years older than the
        imagery. That date is deliberately *not* carried into the filename
        — see :func:`pair_ground_models`.
        """
        payload = self._get("/dgm/tiles", bbox=_bbox(bounds), state=state)
        return [
            Link(
                tile_id=tile_key_to_id(item["tile_key"]), product="dgm",
                url=item["url"], date=item.get("aktualitaet"),
                state=item.get("state"),
            )
            for item in payload.get("tiles") or []
        ]

    def links_for_tile(self, tile_id: str, bounds, products: Iterable[str],
                       state: str | None = None) -> list:
        """Every requested product for one tile, filtered to that tile."""
        wanted = set(products)
        found: list = []
        if "dop" in wanted:
            found += self.orthophotos(bounds, state)
        if "bdom" in wanted:
            found += self.surface_models(bounds, state)
        if "dgm" in wanted:
            found += self.ground_models(bounds, state)

        # The DOP grid is 2 km in some states, so its key names a tile the
        # sample does not contain. Keep those: the crop cuts them to size.
        exact = [link for link in found if link.tile_id == str(tile_id)]
        overlapping = [
            link for link in found
            if link.product == "dop" and link.tile_id != str(tile_id)
        ]
        for link in exact:
            link.tile_id = str(tile_id)
        if not any(link.product == "dop" for link in exact) and overlapping:
            chosen = overlapping[0]
            chosen.tile_id = str(tile_id)
            exact.append(chosen)
        return exact


def _bbox(bounds) -> str:
    """``(minx, miny, maxx, maxy)`` as the API wants it."""
    return "%.6f,%.6f,%.6f,%.6f" % tuple(bounds)


def links_from_index(
    indices: dict,
    tile_ids: Sequence[str],
    tiles=None,
    tile_id_column: str = "tile_id",
    date_column: str = "Aktualitaet",
) -> pd.DataFrame:
    """Resolve URLs from the state's own acquisition indices.

    The route to the *historical* acquisitions the published training set
    is built from. Each index is one feature per flight, with the download
    URL in a product-specific column — ``bdom``, ``rgbi``, ``dgm1`` for
    Lower Saxony's three files.

    The orthophoto index needs a spatial join rather than a tile-id match:
    its tiles are 2 km and carry their own ids, which name a different
    grid from the 1 km sample. The join shrinks each sample tile by 10 m
    first, so a DOP that merely shares a border does not match.

    Args:
        indices: ``{product: (frame, url_column)}`` for any of ``dop``,
            ``bdom``, ``dgm``.
        tile_ids: Tiles to resolve.
        tiles: The sampled tiles with geometry. Required for ``dop``.
        tile_id_column: Tile id column of the indices.
        date_column: Acquisition date column of the indices.

    Returns:
        One row per tile, date and product, with a ``url`` column.
    """
    wanted = {str(t) for t in tile_ids}
    rows = []

    for product, (frame, url_column) in indices.items():
        if frame is None or url_column not in frame.columns:
            logger.warning("No %s index, or no %r column in it", product, url_column)
            continue

        if product == "dop":
            if tiles is None:
                raise ValueError(
                    "The orthophoto index is on a different grid from the sample, "
                    "so resolving it needs the tile geometries."
                )
            matched = _spatial_join(frame, tiles, tile_id_column)
        else:
            matched = frame[frame[tile_id_column].astype(str).isin(wanted)].copy()
            matched["_tile_id"] = matched[tile_id_column].astype(str)

        for _, row in matched.iterrows():
            if row["_tile_id"] not in wanted:
                continue
            date = row.get(date_column)
            rows.append(
                Link(
                    tile_id=row["_tile_id"], product=product, url=row[url_column],
                    date=None if pd.isna(date) else str(pd.to_datetime(date).date()),
                ).as_row()
            )

    frame = pd.DataFrame(rows, columns=["tile_id", "date", "product", "state",
                                        "gsd", "bands", "url"])
    return frame.drop_duplicates(subset=["tile_id", "date", "product"])


def _spatial_join(index, tiles, tile_id_column: str):
    """Match a coarser product grid onto the sampled tiles."""
    import geopandas as gpd

    if index.crs != tiles.crs:
        index = index.to_crs(tiles.crs)

    shrunk = tiles[[tile_id_column, "geometry"]].copy()
    # A DOP sharing only a border covers none of the tile.
    shrunk["geometry"] = shrunk.geometry.buffer(-10.0)

    joined = gpd.sjoin(shrunk, index, how="inner", predicate="intersects")
    left = f"{tile_id_column}_left"
    joined["_tile_id"] = (
        joined[left] if left in joined.columns else joined[tile_id_column]
    ).astype(str)
    return joined


def match_acquisitions(links: pd.DataFrame) -> tuple:
    """Keep only acquisitions where the orthophoto and the surface model agree.

    The nDSM belongs to *its* flight: a March surface model under an August
    orthophoto describes different trees, and a label drawn from the two
    would be wrong in both. The original download step merged the two
    indices on tile **and date** for that reason, and dropped whatever did
    not pair.

    It matters more than it sounds, because the two products are not
    published in step. Asking the API for the newest of each can return a
    2026 orthophoto and a 2025 surface model for the same tile — a pair
    that never existed.

    Args:
        links: Resolved links for all products.

    Returns:
        ``(kept, dropped)``. Ground models are carried through untouched;
        they are matched to imagery dates by :func:`pair_ground_models`.
    """
    products = set(links["product"])
    if not {"dop", "bdom"} <= products:
        return links, links.iloc[0:0]

    dates = {
        product: group.groupby("tile_id")["date"].apply(set).to_dict()
        for product, group in links.groupby("product")
    }
    paired = {
        tile_id: dates["dop"].get(tile_id, set()) & dates["bdom"].get(tile_id, set())
        for tile_id in set(links["tile_id"])
    }

    imagery = links["product"].isin(("dop", "bdom"))
    matches = links.apply(
        lambda row: row["date"] in paired.get(row["tile_id"], set()), axis=1
    )
    keep = ~imagery | matches

    dropped = links[~keep]
    if not dropped.empty:
        logger.warning(
            "%d file(s) dropped: the orthophoto and the surface model of that tile "
            "carry different dates, so they are not one acquisition",
            len(dropped),
        )
    return links[keep].copy(), dropped.copy()


def pair_ground_models(links: pd.DataFrame) -> pd.DataFrame:
    """Give each ground model the dates of the imagery it will be used with.

    The ground model is flown once and reused by every acquisition of that
    tile. The training pipeline pairs files by name, so one DGM has to
    land on disk once per imagery date — under the imagery's date, not its
    own. Its true acquisition date stays in the log.

    Args:
        links: Resolved links for all products.

    Returns:
        The same table with the ground-model rows expanded, and their
        original date preserved as ``source_date``.
    """
    if links.empty or "dgm" not in set(links["product"]):
        return links

    imagery = links[links["product"] != "dgm"]
    dates = (
        imagery.dropna(subset=["date"])
        .groupby("tile_id")["date"].apply(lambda values: sorted(set(values)))
        .to_dict()
    )

    expanded = []
    for _, row in links[links["product"] == "dgm"].iterrows():
        for date in dates.get(row["tile_id"], [row["date"]]):
            copy = row.copy()
            copy["source_date"] = row["date"]
            copy["date"] = date
            expanded.append(copy)

    others = links[links["product"] != "dgm"].copy()
    others["source_date"] = others["date"]
    if not expanded:
        return others
    return pd.concat([others, pd.DataFrame(expanded)], ignore_index=True)


def download(url: str, destination: Path, retries: int = 3, timeout: float = 300.0,
             chunk: int = 1 << 16) -> tuple:
    """Fetch one file, streaming it to disk.

    Writes to a ``.part`` file and renames on success, so an interrupted
    run leaves no truncated raster that a later run would skip as done.

    Returns:
        ``(status, error)`` where status is ``skipped``, ``success`` or
        ``failed``.
    """
    import urllib.request

    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0:
        return "skipped", None

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, \
                    open(partial, "wb") as handle:
                while True:
                    block = response.read(chunk)
                    if not block:
                        break
                    handle.write(block)
            if partial.stat().st_size == 0:
                raise OSError("empty response")
            partial.replace(destination)
            return "success", None
        except Exception as error:  # noqa: BLE001 — any failure is worth a retry
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(attempt)

    return "failed", str(last_error)


def crop_to_geometry(source: Path, geometry, destination: Path) -> bool:
    """Cut a raster to one tile's footprint.

    Lower Saxony publishes orthophotos on a 2 km grid, so three quarters
    of a downloaded DOP belongs to other tiles. Cropping keeps the patch
    extractor from reading imagery its label mask does not cover.

    Returns:
        Whether the crop was written.
    """
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import mapping

    try:
        with rasterio.open(source) as dataset:
            window, transform = rio_mask(dataset, [mapping(geometry)], crop=True,
                                         all_touched=False)
            profile = dataset.profile.copy()

        profile.update(height=window.shape[1], width=window.shape[2],
                       transform=transform)
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(destination, "w", **profile) as out:
            out.write(window)
        return True
    except Exception as error:  # noqa: BLE001 — a bad tile must not stop the run
        logger.warning("Could not crop %s: %s", Path(source).name, error)
        return False


def normalised_surface(surface: Path, ground: Path, destination: Path,
                       nodata: float = NDSM_NODATA) -> tuple:
    """``nDSM = surface − ground``, on the surface model's grid.

    The ground model is 1 m and the surface model 20 cm, so the ground is
    resampled bilinearly onto the finer grid — nearest neighbour would
    print the 1 m grid onto every slope as 5 x 5 pixel steps, and the
    labels were drawn while looking at this raster.

    Returns:
        ``(status, error)``.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    destination = Path(destination)
    if destination.exists():
        return "skipped", None

    try:
        with rasterio.open(surface) as surface_src:
            surface_data = surface_src.read(1)
            profile = surface_src.profile.copy()
            surface_nodata = surface_src.nodata
            transform, crs, shape = (surface_src.transform, surface_src.crs,
                                     surface_data.shape)

        with rasterio.open(ground) as ground_src:
            resampled = np.empty(shape, dtype=np.float32)
            reproject(
                source=rasterio.band(ground_src, 1),
                src_transform=ground_src.transform, src_crs=ground_src.crs,
                destination=resampled, dst_transform=transform, dst_crs=crs,
                resampling=Resampling.bilinear,
            )
            ground_nodata = ground_src.nodata

        heights = surface_data.astype(np.float32) - resampled
        invalid = np.zeros(shape, dtype=bool)
        for values, value in ((surface_data, surface_nodata), (resampled, ground_nodata)):
            if value is not None:
                invalid |= values == value
        heights[invalid] = nodata

        profile.update(dtype="float32", nodata=nodata, count=1)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(destination, "w", **profile) as out:
            out.write(heights, 1)
        return "success", None
    except Exception as error:  # noqa: BLE001 — report and carry on
        return "failed", str(error)
