#!/usr/bin/env python3
"""Fetch the orthophotos and height models for the sampled tiles.

Takes the tile table stage 1 draws and puts three rasters per acquisition
on disk, named ``<tile_id>_<date>.tif`` — which is what stage 3 parses:

```
<out>/DOP/    orthophoto, cropped to the 1 km tile
<out>/bDOM/   image-based surface model, 20 cm
<out>/DGM/    LiDAR ground model, 1 m
<out>/nDSM/   bDOM − DGM, the raster the labels were digitised with
<out>/logs/   download_log.csv and ndsm_log.csv — what came from where
```

URLs come from one of two sources, and the choice matters:

``--source easygeodata`` (default)
    The easygeodata.de Extract API, one bbox query per tile. Covers all
    sixteen states with no local index, and is the right choice for a
    fresh area. It serves the **current** acquisition of a tile only.

``--source index``
    The state's own acquisition index — the GeoJSON stage 1 samples from —
    which lists every flight over a tile. The published training set is
    built from *pairs* of acquisitions, one summer and one not, so
    reproducing it needs this route. Lower Saxony publishes three such
    files, one per product.

A run over the published tiles via the API therefore fetches roughly half
the acquisitions the training set uses, and says so at the end rather than
leaving it to be discovered in stage 3.

What this stage cannot fetch is the label masks: they were drawn by hand
and are published with the paper's training-data package. Point stage 3 at
that package's ``labels/``.

Examples::

    # The published tiles, current imagery, via the API
    python scripts/02_download_dataset.py --out data/Sampling

    # The published training set: every acquisition, from the state index
    python scripts/02_download_dataset.py --source index --out data/Sampling \\
        --index-dop  .../lgln-opengeodata-dop20.geojson \\
        --index-bdom .../lgln-opengeodata-bdom20.geojson \\
        --index-dgm  .../lgln-opengeodata-dgm1.geojson

    # What would be fetched, and how much of it
    python scripts/02_download_dataset.py --dry-run --limit 5 -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from treecover.config import load_paths
from treecover.data.download import (
    EASYGEODATA_URL,
    PRODUCTS,
    EasyGeoData,
    crop_to_geometry,
    download,
    links_from_index,
    match_acquisitions,
    normalised_surface,
    pair_ground_models,
)
from treecover.io.vector import read_vector

logger = logging.getLogger(__name__)

#: URL column per product in Lower Saxony's indices. Other states name
#: theirs differently; ``--url-column`` overrides.
INDEX_URL_COLUMNS = {"dop": "rgbi", "bdom": "bdom", "dgm": "dgm1"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    src = p.add_argument_group("input")
    src.add_argument("--tiles", type=Path, default=None,
                     help="Sampled tiles from stage 1. "
                          "Default: training_data.tiles from paths.yaml.")
    src.add_argument("--tile-id-column", default="tile_id")
    src.add_argument("--tile-ids", nargs="*", default=None, metavar="ID",
                     help="Restrict to these tiles.")
    src.add_argument("--source", choices=("easygeodata", "index"),
                     default="easygeodata",
                     help="Where download URLs come from. 'index' is the only "
                          "route to historical acquisitions — see above.")
    src.add_argument("--products", nargs="+", default=["dop", "bdom", "dgm"],
                     choices=sorted(PRODUCTS),
                     help="Which products to fetch.")
    src.add_argument("--state", default=None, metavar="CODE",
                     help="Restrict the API to one state, e.g. NI. Only needed "
                          "where two states publish over the same ground.")
    src.add_argument("--limit", type=int, default=None, metavar="N",
                     help="Stop after N tiles.")

    idx = p.add_argument_group("index source")
    idx.add_argument("--index-dop", type=Path, default=None)
    idx.add_argument("--index-bdom", type=Path, default=None)
    idx.add_argument("--index-dgm", type=Path, default=None)
    idx.add_argument("--date-column", default="Aktualitaet",
                     help="Acquisition date column of the indices.")
    idx.add_argument("--url-column", nargs="*", default=None, metavar="PRODUCT=COLUMN",
                     help="Override the URL column of an index, e.g. dop=rgbi.")

    api = p.add_argument_group("API")
    api.add_argument("--api-url", default=EASYGEODATA_URL)
    api.add_argument("--pause", type=float, default=0.2,
                     help="Seconds between API requests. It is a free service.")
    api.add_argument("--retries", type=int, default=3)

    out = p.add_argument_group("output")
    out.add_argument("--out", type=Path, default=None,
                     help="Output directory. Default: training_root/Sampling.")
    out.add_argument("--no-ndsm", action="store_true",
                     help="Skip the bDOM − DGM step.")
    out.add_argument("--allow-unpaired", action="store_true",
                     help="Keep an orthophoto whose surface model carries another "
                          "date. They are two flights, and the nDSM would describe "
                          "different trees from the image it is stacked with.")
    out.add_argument("--no-crop", action="store_true",
                     help="Keep orthophotos at their published extent. In states "
                          "publishing on a 2 km grid that is four times the tile.")
    out.add_argument("--dry-run", action="store_true",
                     help="Resolve the URLs, write the log, download nothing.")
    out.add_argument("--no-progress", action="store_true")
    out.add_argument("-v", "--verbose", action="store_true")
    return p


def resolve_links(tiles, args) -> pd.DataFrame:
    """The download table: one row per tile, date and product."""
    if args.source == "index":
        return links_from_index(
            _load_indices(args), tiles[args.tile_id_column].astype(str),
            tiles=tiles, tile_id_column=args.tile_id_column,
            date_column=args.date_column,
        )

    client = EasyGeoData(args.api_url, retries=args.retries, pause=args.pause)
    wgs84 = tiles.to_crs("EPSG:4326")
    rows = []
    for position, ((_, tile), (_, bounds_row)) in enumerate(
            zip(tiles.iterrows(), wgs84.iterrows()), start=1):
        tile_id = str(tile[args.tile_id_column])
        try:
            links = client.links_for_tile(
                tile_id, bounds_row.geometry.bounds, args.products, args.state
            )
        except (ConnectionError, ValueError) as error:
            logger.error("Tile %s: %s", tile_id, error)
            continue

        rows += [link.as_row() for link in links]
        if not args.no_progress and position % 10 == 0:
            logger.info("Resolved %d/%d tiles", position, len(tiles))

    return pd.DataFrame(rows, columns=["tile_id", "date", "product", "state",
                                       "gsd", "bands", "url"])


def _load_indices(args) -> dict:
    """Read the per-product index files named on the command line."""
    overrides = dict(
        pair.split("=", 1) for pair in (args.url_column or []) if "=" in pair
    )
    columns = {**INDEX_URL_COLUMNS, **overrides}

    indices = {}
    for product, path in (("dop", args.index_dop), ("bdom", args.index_bdom),
                          ("dgm", args.index_dgm)):
        if product not in args.products:
            continue
        if path is None:
            logger.warning("--source index but no --index-%s; skipping %s",
                           product, product)
            continue
        indices[product] = (read_vector(path), columns[product])
        logger.info("%s index: %d features from %s", product,
                    len(indices[product][0]), path.name)
    return indices


def fetch(links: pd.DataFrame, tiles, args) -> pd.DataFrame:
    """Download everything in ``links``, cropping orthophotos as they land."""
    geometries = dict(zip(tiles[args.tile_id_column].astype(str), tiles.geometry))
    log = []

    for position, row in enumerate(links.itertuples(index=False), start=1):
        directory = args.out / PRODUCTS[row.product]
        target = directory / f"{row.tile_id}_{row.date}.tif"

        if args.dry_run:
            log.append({**row._asdict(), "status": "planned", "error": None,
                        "path": str(target)})
            continue

        needs_crop = row.product == "dop" and not args.no_crop
        if needs_crop and target.exists():
            status, error = "skipped", None
        elif needs_crop:
            staging = directory / f".{row.tile_id}_{row.date}.download.tif"
            status, error = download(row.url, staging, retries=args.retries)
            if status == "success":
                if crop_to_geometry(staging, geometries[row.tile_id], target):
                    staging.unlink(missing_ok=True)
                else:
                    # Better a whole 2 km tile than none: stage 3 reads the
                    # window it needs anyway, it just reads more of it.
                    staging.replace(target)
                    error = "crop failed, kept the full extent"
            else:
                staging.unlink(missing_ok=True)
        else:
            status, error = download(row.url, target, retries=args.retries)

        log.append({**row._asdict(), "status": status, "error": error,
                    "path": str(target)})

        if not args.no_progress and position % 25 == 0:
            done = sum(1 for entry in log if entry["status"] in ("success", "skipped"))
            logger.info("%d/%d files, %d in place", position, len(links), done)

    return pd.DataFrame(log)


def build_ndsm(args) -> pd.DataFrame:
    """bDOM − DGM for every surface model on disk that has a ground model."""
    surfaces = sorted((args.out / PRODUCTS["bdom"]).glob("*.tif"))
    log = []
    for path in surfaces:
        ground = args.out / PRODUCTS["dgm"] / path.name
        destination = args.out / "nDSM" / path.name
        if not ground.exists():
            log.append({"file": path.name, "status": "failed",
                        "error": "no ground model for this tile and date"})
            continue
        status, error = normalised_surface(path, ground, destination)
        log.append({"file": path.name, "status": status, "error": error})
    return pd.DataFrame(log)


def report(links: pd.DataFrame, download_log: pd.DataFrame,
           ndsm_log: pd.DataFrame, tiles, args) -> None:
    """Say what landed, and — the useful part — what did not."""
    print(f"\nResolved {len(links)} file(s) for {links['tile_id'].nunique()} "
          f"of {len(tiles)} tile(s)")
    for product, group in links.groupby("product"):
        dates = group["date"].nunique()
        print(f"  {PRODUCTS[product]:<6} {len(group):5d} file(s), "
              f"{group['tile_id'].nunique():4d} tile(s), {dates:3d} date(s)")

    if not download_log.empty and not args.dry_run:
        print("\nDownloads:")
        for status, count in download_log["status"].value_counts().items():
            print(f"  {status:<9} {count}")
        failures = download_log[download_log["status"] == "failed"]
        for row in failures.head(5).itertuples(index=False):
            print(f"    {row.tile_id} {row.date} {row.product}: {row.error}")
        if len(failures) > 5:
            print(f"    … and {len(failures) - 5} more, see the log")

    if not ndsm_log.empty:
        print("\nnDSM:")
        for status, count in ndsm_log["status"].value_counts().items():
            print(f"  {status:<9} {count}")

    _report_missing_acquisitions(links, tiles, args)


def _report_missing_acquisitions(links: pd.DataFrame, tiles, args) -> None:
    """Compare what was fetched against the dates the tile table records.

    The API serves one acquisition per tile, and a training set built on
    one acquisition per tile is exactly the thing this paper argues
    against. Nothing downstream would flag it, so this does.
    """
    if "flight_dates_str" not in tiles.columns or links.empty:
        return

    expected = {}
    for _, tile in tiles.iterrows():
        dates = [d.strip()[:10] for d in str(tile["flight_dates_str"]).split(",")
                 if d.strip()]
        if dates:
            expected[str(tile[args.tile_id_column])] = set(dates)
    if not expected:
        return

    imagery = links[links["product"] != "dgm"]
    fetched = imagery.groupby("tile_id")["date"].apply(set).to_dict()
    missing = {
        tile_id: dates - fetched.get(tile_id, set())
        for tile_id, dates in expected.items()
    }
    total_missing = sum(len(v) for v in missing.values())
    if not total_missing:
        print("\nEvery acquisition the tile table records was resolved.")
        return

    total_expected = sum(len(v) for v in expected.values())
    print(f"\n{total_missing} of {total_expected} acquisitions in the tile table "
          f"were not resolved.")
    if args.source == "easygeodata":
        print("  Expected: the API serves the current acquisition of a tile, and\n"
              "  the training set pairs a summer image with a non-summer one. Use\n"
              "  --source index with the state's acquisition index for the rest.")
    else:
        print("  Check that the index files cover the same period as the tile table.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    paths = load_paths()
    args.tiles = args.tiles or paths.get_path("training_data.tiles")
    if not Path(args.tiles).exists():
        print(f"error: --tiles not found: {args.tiles}", file=sys.stderr)
        return 2
    if args.out is None:
        root = paths.get_value("training_root", "./data")
        args.out = Path(root) / "Sampling"
    args.out = Path(args.out)

    tiles = read_vector(args.tiles)
    if args.tile_id_column not in tiles.columns:
        print(f"error: column {args.tile_id_column!r} not in {Path(args.tiles).name}. "
              f"Available: {list(tiles.columns)}", file=sys.stderr)
        return 2
    if args.tile_ids:
        wanted = {str(t) for t in args.tile_ids}
        tiles = tiles[tiles[args.tile_id_column].astype(str).isin(wanted)]
        if tiles.empty:
            print("error: none of --tile-ids is in the tile table", file=sys.stderr)
            return 2
    if args.limit:
        tiles = tiles.head(args.limit)
    logger.info("Tiles: %d from %s", len(tiles), Path(args.tiles).name)

    links = resolve_links(tiles, args)
    if links.empty:
        print("error: no download URL could be resolved. With --source index, check "
              "--index-* and --url-column; with the API, check connectivity.",
              file=sys.stderr)
        return 1

    dropped = links.iloc[0:0]
    if not args.allow_unpaired:
        links, dropped = match_acquisitions(links)
        if links.empty:
            print("error: no orthophoto shares a date with its surface model. The "
                  "two products are published separately and the API serves the "
                  "newest of each; --source index pairs them properly, and "
                  "--allow-unpaired keeps them anyway.", file=sys.stderr)
            return 1
    links = pair_ground_models(links)

    args.out.mkdir(parents=True, exist_ok=True)
    log_dir = args.out / "logs"
    log_dir.mkdir(exist_ok=True)

    download_log = fetch(links, tiles, args)
    download_log.to_csv(log_dir / "download_log.csv", index=False)
    if not dropped.empty:
        dropped.to_csv(log_dir / "unpaired_acquisitions.csv", index=False)

    ndsm_log = pd.DataFrame()
    if not args.no_ndsm and not args.dry_run and {"bdom", "dgm"} <= set(args.products):
        ndsm_log = build_ndsm(args)
        ndsm_log.to_csv(log_dir / "ndsm_log.csv", index=False)

    report(links, download_log, ndsm_log, tiles, args)
    if not dropped.empty:
        print(f"\n{len(dropped)} file(s) skipped as unpaired — the orthophoto and "
              "the surface\nmodel of that tile carry different dates. See "
              "logs/unpaired_acquisitions.csv.")

    print(f"\nWritten to {args.out}:")
    for product in args.products:
        directory = args.out / PRODUCTS[product]
        if directory.exists():
            print(f"  {PRODUCTS[product]}/{'':<6} {len(list(directory.glob('*.tif')))} file(s)")
    if not ndsm_log.empty:
        print(f"  nDSM/{'':<7} {len(list((args.out / 'nDSM').glob('*.tif')))} file(s)")
    print("  logs/download_log.csv")

    if args.dry_run:
        print("\nDry run — nothing downloaded.")
        return 0

    print(f"\nNext: label the tiles, then python scripts/03_prepare_patches.py "
          f"--images {args.out / 'DOP'} --ndsm {args.out / 'nDSM'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
