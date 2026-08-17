#!/usr/bin/env python3
"""Build or complete the per-tile comparison table.

For each 1 km tile: read our prediction, then sample every comparison
product over the same footprint and reduce each to a tree cover
percentage. This is what makes products of 20 cm to 30 m pixel size
comparable at all.

``--complete`` is the usual mode. The published table was produced while
several states were still being processed, so 6,953 of its 370,533 rows
carry no value — Schleswig-Holstein alone is missing 1,308 tiles, all of
them 2024 and 2025 acquisitions. Completing fills exactly those rows and
leaves every existing value untouched, so the numbers already published do
not move.

``our_pred_path`` in the published table still carries the absolute root of
the external HDD the table was computed on. ``--path-prefix`` rewrites that
root to wherever the archive sits now — read the prefix off the first row
of the table and pass it as ``OLD=NEW``.

Examples::

    # Fill the gaps, keeping every existing value
    python scripts/10_extract_tile_products.py --complete \\
        --in  publication/tiles/tile_treecover_products.csv \\
        --out publication/tiles/tile_treecover_products_complete.csv \\
        --products-root /tf/Other_Tree_Products \\
        --path-prefix /mnt/products/Germany=/tf/Germany

    # Recompute one state from scratch
    python scripts/10_extract_tile_products.py --states SH \\
        --predictions-root /tf/Germany --products-root /tf/Other_Tree_Products \\
        --out sh_tiles.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds

from treecover.comparison import OUR_COLUMN
from treecover.config import load_paths
from treecover.constants import PRED_TREE
from treecover.io.tiles import acquisition_date, find_prediction_tiles
from treecover.products import PRODUCTS, sample_product

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--complete", action="store_true",
                      help="Fill only the rows of --in that have no value.")
    mode.add_argument("--states", nargs="*", metavar="CODE",
                      help="Recompute these states from the archive.")

    p.add_argument("--in", dest="source", type=Path, default=None,
                   help="Existing table, required with --complete.")
    p.add_argument("--out", dest="destination", type=Path, required=True)
    p.add_argument("--predictions-root", type=Path, default=None)
    p.add_argument("--products-root", type=Path, required=True)
    p.add_argument("--path-prefix", default=None, metavar="OLD=NEW",
                   help="Rewrite the root of our_pred_path, e.g. "
                        "/mnt/products/Germany=/tf/Germany")
    p.add_argument("--limit", type=int, default=None, help="Stop after N tiles.")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def our_tree_cover(path: Path) -> tuple[float | None, tuple | None, str | None]:
    """Tree cover of one prediction tile, plus its bounds in WGS 84.

    Returns:
        ``(percent, lonlat_bounds, error)``. Nodata is excluded, so a tile
        the model never covered reports over what it did cover.
    """
    try:
        with rasterio.open(path) as src:
            data = src.read(1)
            bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    except rasterio.RasterioIOError as exc:
        return None, None, str(exc)

    valid = data != 255
    if not valid.any():
        return None, bounds, "all nodata"
    return float(100.0 * (data[valid] == PRED_TREE).mean()), bounds, None


def rewrite(path_value, prefix: tuple[str, str] | None) -> Path | None:
    """Apply ``--path-prefix`` to a recorded prediction path."""
    if not isinstance(path_value, str) or not path_value:
        return None
    if prefix:
        old, new = prefix
        if path_value.startswith(old):
            path_value = new + path_value[len(old):]
    return Path(path_value)


def locate_prediction(filename, state, predictions_root: Path) -> Path | None:
    """Find a tile's prediction from its imagery filename.

    Rows written before their state finished processing carry no
    ``our_pred_path`` — every one of Schleswig-Holstein's 363 remaining gaps
    is of this kind. The prediction is nonetheless on disk under the same
    stem with ``_pred.tif`` appended, so it can be located by name.

    Args:
        filename: Imagery filename, e.g.
            ``dop20rgbi_32_565_5948_sh_file_20250317.jp2``.
        state: State directory name.
        predictions_root: Root of the archive.

    Returns:
        The prediction, or ``None``. Where several years hold the same
        stem, the newest is taken — the same preference the merge uses.
    """
    if not isinstance(filename, str) or not filename or not isinstance(state, str):
        return None
    stem = filename.rsplit(".", 1)[0]
    state_dir = Path(predictions_root) / state
    if not state_dir.is_dir():
        return None
    matches = sorted(state_dir.glob(f"*/predictions/*/{stem}_pred.tif"))
    return matches[-1] if matches else None


def extract_row(pred_path: Path, products_root: Path, handles: dict) -> dict:
    """Everything measured for one tile."""
    cover, bounds, error = our_tree_cover(pred_path)
    row: dict = {OUR_COLUMN: cover}
    if bounds is None:
        row["extract_error"] = error
        return row

    row.update(
        lon_min=bounds[0], lat_min=bounds[1], lon_max=bounds[2], lat_max=bounds[3],
        lon_c=(bounds[0] + bounds[2]) / 2, lat_c=(bounds[1] + bounds[3]) / 2,
    )
    for product in PRODUCTS:
        row[product.column] = sample_product(
            product, products_root, bounds, "EPSG:4326", handles
        )
    if error:
        row["extract_error"] = error
    return row


def complete(args, prefix) -> int:
    """Fill the rows that have no value, leaving the rest untouched."""
    frame = pd.read_csv(args.source)
    if OUR_COLUMN not in frame.columns:
        print(f"error: {args.source.name} has no {OUR_COLUMN!r} column", file=sys.stderr)
        return 2
    if "our_pred_path" not in frame.columns:
        print(f"error: {args.source.name} has no 'our_pred_path' column, so the "
              "prediction for a gap row cannot be located.", file=sys.stderr)
        return 2

    gaps = frame.index[frame[OUR_COLUMN].isna()]
    if args.limit:
        gaps = gaps[: args.limit]
    print(f"{len(frame):,} rows, {len(gaps):,} without a value")
    if not len(gaps):
        frame.to_csv(args.destination, index=False)
        print("Nothing to complete.")
        return 0

    by_state = frame.loc[gaps, "state"].value_counts() if "state" in frame else {}
    for state, count in dict(by_state).items():
        print(f"  {state}: {count:,}")

    handles: dict = {}
    filled = skipped = failed = 0
    iterator = gaps
    if not args.no_progress:
        from tqdm import tqdm

        iterator = tqdm(gaps, unit="tile")

    predictions_root = args.predictions_root or load_paths().get_path("predictions_root")
    for index in iterator:
        pred_path = rewrite(frame.at[index, "our_pred_path"], prefix)
        if pred_path is None or not pred_path.exists():
            # No recorded path, or it points somewhere this machine cannot
            # reach: fall back to locating the prediction by filename.
            pred_path = locate_prediction(
                frame.at[index, "filename"] if "filename" in frame.columns else None,
                frame.at[index, "state"] if "state" in frame.columns else None,
                predictions_root,
            )
        if pred_path is None or not pred_path.exists():
            skipped += 1
            continue
        values = extract_row(pred_path, args.products_root, handles)
        if values.get(OUR_COLUMN) is None:
            failed += 1
            continue
        for column, value in values.items():
            if value is not None and column in frame.columns:
                frame.at[index, column] = value
            elif value is not None:
                frame.loc[index, column] = value
        filled += 1

    for source in handles.values():
        if source is not None:
            source.close()

    frame.to_csv(args.destination, index=False)
    remaining = int(frame[OUR_COLUMN].isna().sum())
    print(f"\nFilled {filled:,}   prediction missing {skipped:,}   unreadable {failed:,}")
    print(f"Rows still without a value: {remaining:,}")
    print(f"-> {args.destination}")
    return 0


def rebuild(args, _prefix) -> int:
    """Recompute whole states from the archive."""
    predictions_root = args.predictions_root or load_paths().get_path("predictions_root")
    tiles = list(find_prediction_tiles(predictions_root, args.states or None))
    if args.limit:
        tiles = tiles[: args.limit]
    if not tiles:
        print("error: no prediction tiles matched", file=sys.stderr)
        return 1
    print(f"{len(tiles):,} tile(s)")

    handles: dict = {}
    rows = []
    iterator = tiles
    if not args.no_progress:
        from tqdm import tqdm

        iterator = tqdm(tiles, unit="tile")

    for tile in iterator:
        values = extract_row(tile.path, args.products_root, handles)
        values.update(
            filename=tile.path.name, state=tile.state, year=tile.year,
            date=acquisition_date(tile.path, tile.year),
            our_pred_path=str(tile.path), tile_area_km2=1.0,
        )
        rows.append(values)

    for source in handles.values():
        if source is not None:
            source.close()

    frame = pd.DataFrame(rows)
    frame.to_csv(args.destination, index=False)
    print(f"\n{len(frame):,} rows -> {args.destination}")
    valid = frame[OUR_COLUMN].notna()
    print(f"With a value: {int(valid.sum()):,}   "
          f"mean {frame.loc[valid, OUR_COLUMN].mean():.2f} %")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    if args.complete and args.source is None:
        print("error: --complete needs --in", file=sys.stderr)
        return 2
    if not args.products_root.is_dir():
        print(f"error: products root not found: {args.products_root}", file=sys.stderr)
        return 2

    prefix = None
    if args.path_prefix:
        if "=" not in args.path_prefix:
            print("error: --path-prefix takes OLD=NEW", file=sys.stderr)
            return 2
        old, _, new = args.path_prefix.partition("=")
        prefix = (old, new)

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    return complete(args, prefix) if args.complete else rebuild(args, prefix)


if __name__ == "__main__":
    raise SystemExit(main())
