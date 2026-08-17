#!/usr/bin/env python3
"""Convert the collected tile table into the columns this pipeline expects.

``tile_treecover_products.csv`` records tile geometry as a bounding box
(``lon_min``/``lat_min``/``lon_max``/``lat_max``). Stage 9 and figures 6–8
work from tile centroids and an area weight instead. This script derives
them, writing a new file rather than editing the original — the original is
what the published numbers were computed from and must stay comparable.

The area weight is 1 km² per tile: the tiles are 1 km squares on the UTM
grid, and that is what they measure. ``--bbox-area`` reproduces what the
original notebook did instead — derive the weight from the tile's lon/lat
bounding box — and is kept only for reproducing published intermediates.

Do not use it for areas. A UTM square is rotated against the meridian by
the meridian convergence, so its lon/lat envelope is larger than the square
itself: a mean 4.5 % across Germany, up to 13 % at the zone edges, summing
to 376,205 km² of "tile area" for a country of 357,596 km². Percentages
survive it — they are ratios of the same weights, so the inflation cancels
— but every area computed from it is a product of the weight and carries
the full error. That is how a draft came to report 32.2 % alongside
120,943 km², two numbers whose reference areas differ by 5 %.

``--land-mask`` adds an ``on_land`` column. Roughly 6,500 tiles of the
archive sit over the North Sea, the Baltic or across a national border —
the flight blocks extend past the coastline. In the published figures they
are absent, but not because they were masked: they were part of the
incomplete aggregation and dropped out of every ``dropna()``. Once the gaps
are filled they reappear, and the coastline turns ragged. Flagging them
explicitly makes the exclusion a decision rather than a side effect of how
far a processing run happened to get.

Usage::

    python scripts/prepare_tile_table.py \\
        --in  publication/tiles/tile_treecover_products.csv \\
        --out publication/tiles/tile_treecover_products_prepared.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from treecover.comparison import OUR_COLUMN

#: Kilometres per degree of latitude, as used by the original notebook.
DEG_TO_KM = 111.32

#: Bounding-box columns the collected table uses.
BBOX_COLUMNS = ("lon_min", "lat_min", "lon_max", "lat_max")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in", dest="source", type=Path, required=True)
    p.add_argument("--out", dest="destination", type=Path, required=True)
    p.add_argument("--constant-area", type=float, default=1.0, metavar="KM2",
                   help="Area weight per tile, in km². Default 1.0 — the tiles "
                        "are 1 km squares on the UTM grid.")
    p.add_argument("--bbox-area", action="store_true",
                   help="Derive the weight from the lon/lat bounding box, as the "
                        "original notebook did. Inflated a mean 4.5 %% by meridian "
                        "convergence; for reproducing old intermediates only.")
    p.add_argument("--drop-incomplete", action="store_true",
                   help="Drop rows where our own tree cover is missing. They "
                        "contribute nothing to any comparison, since every "
                        "total is taken over tiles where our map is valid.")
    p.add_argument("--land-areas", type=Path, default=None, metavar="CSV",
                   help="tile_statistics_all.csv. Adds 'land_area_km2', the land "
                        "measured inside each tile, which stage 9 then prefers as "
                        "the aggregation weight over the tile footprint.")
    p.add_argument("--land-mask", type=Path, default=None, metavar="GPKG",
                   help="Land polygons (or GADM). Adds an 'on_land' column "
                        "marking tiles whose centroid falls on German land.")
    p.add_argument("--land-layer", default=None,
                   help="Layer name for a multi-layer land mask, e.g. ADM_ADM_1.")
    p.add_argument("--drop-offshore", action="store_true",
                   help="Also remove the tiles marked off land. Needs --land-mask.")
    return p


def flag_on_land(frame, mask_path: Path, layer: str | None):
    """Mark tiles whose centroid falls on land.

    The centroid, not the footprint: a 1 km tile straddling the coast is
    counted by where its middle lies, which is the same rule the per-state
    aggregation uses. Testing the footprint would keep every tile that
    touches land at all, and the coastline would stay ragged.

    Uses a spatial join rather than ``points.within(union)``. The OSM coast
    polygons carry hundreds of thousands of vertices, and testing 370,000
    points against their union one at a time takes the better part of an
    hour; the join's R-tree turns that into a minute.
    """
    import geopandas as gpd

    from treecover.io.vector import read_vector

    polygons = read_vector(mask_path, layer=layer)
    if polygons.crs is None:
        polygons = polygons.set_crs("EPSG:4326")

    points = gpd.GeoDataFrame(
        {"_row": np.arange(len(frame))},
        geometry=gpd.points_from_xy(frame["lon_c"], frame["lat_c"]),
        crs="EPSG:4326",
    ).to_crs(polygons.crs)

    hits = gpd.sjoin(points, polygons[["geometry"]], how="inner", predicate="within")
    on_land = np.zeros(len(frame), dtype=bool)
    on_land[hits["_row"].to_numpy()] = True
    return on_land


def _tile_stem(name) -> str:
    """Tile identity shared by the product table and the statistics table.

    One records ``…_20220721.jp2``, the other ``…_20220721_pred.tif``; the
    stem is what they agree on.
    """
    text = str(name)
    for suffix in ("_pred.tif", ".tif", ".jp2"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def attach_land_areas(frame, statistics_csv: Path):
    """Add ``land_area_km2`` from the per-tile statistics.

    This is the area the tile's percentage is a percentage of: land inside
    the tile *and* inside its own state, nodata removed. Weighting by it
    makes a half-drowned coastal tile count for half a tile, which the tile
    footprint cannot express.
    """
    stats = pd.read_csv(statistics_csv, usecols=["state", "tile_name", "land_area_ha"])
    stats["_key"] = stats["state"] + "/" + stats["tile_name"].map(_tile_stem)
    lookup = stats.drop_duplicates("_key").set_index("_key")["land_area_ha"] / 100.0

    key = frame["state"] + "/" + frame["filename"].map(_tile_stem)
    frame["land_area_km2"] = key.map(lookup)
    matched = int(frame["land_area_km2"].notna().sum())
    print(f"Land areas: matched {matched:,} of {len(frame):,} tiles "
          f"({frame['land_area_km2'].sum():,.0f} km² total).")
    if matched < len(frame):
        print(f"warning: {len(frame) - matched:,} tile(s) have no measured land "
              f"area and will drop out of any aggregation weighted by it.",
              file=sys.stderr)
    return frame


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.source.exists():
        print(f"error: {args.source} not found", file=sys.stderr)
        return 2

    frame = pd.read_csv(args.source)
    print(f"Read {len(frame):,} rows, {len(frame.columns)} columns")

    # Fill per row, not per column. A table completed by
    # 10_extract_tile_products.py has centroids on the rows it filled and
    # nothing on the rest, so testing only whether the column exists leaves
    # most rows empty — and they then drop out of every spatial join
    # silently, which looks like a working run over almost no data.
    for column in ("lon_c", "lat_c"):
        if column not in frame.columns:
            frame[column] = pd.NA

    needs_centroid = frame["lon_c"].isna() | frame["lat_c"].isna()
    if needs_centroid.any():
        absent = [c for c in BBOX_COLUMNS if c not in frame.columns]
        if absent:
            print(f"error: {int(needs_centroid.sum()):,} row(s) have no centroid and "
                  f"the bounding box is absent ({absent}).\n"
                  f"       Available: {list(frame.columns)}", file=sys.stderr)
            return 2
        frame.loc[needs_centroid, "lon_c"] = (
            frame.loc[needs_centroid, "lon_min"] + frame.loc[needs_centroid, "lon_max"]
        ) / 2
        frame.loc[needs_centroid, "lat_c"] = (
            frame.loc[needs_centroid, "lat_min"] + frame.loc[needs_centroid, "lat_max"]
        ) / 2
        print(f"Derived lon_c/lat_c for {int(needs_centroid.sum()):,} row(s) "
              "from the bounding box.")

    still_missing = int((frame["lon_c"].isna() | frame["lat_c"].isna()).sum())
    if still_missing:
        print(f"warning: {still_missing:,} row(s) still have no centroid and will "
              "drop out of any spatial aggregation.", file=sys.stderr)

    if "tile_area_km2" not in frame.columns:
        if args.bbox_area:
            # The lon/lat envelope of a UTM square, not the square. Kept so
            # published intermediates can be reproduced exactly; it is not a
            # ground area and must not be used as one.
            lat_km = (frame["lat_max"] - frame["lat_min"]) * DEG_TO_KM
            lon_km = (
                (frame["lon_max"] - frame["lon_min"]) * DEG_TO_KM
                * np.cos(np.deg2rad(frame["lat_c"].astype(float)))
            )
            frame["tile_area_km2"] = lat_km * lon_km
            median = frame["tile_area_km2"].median()
            print(f"Derived tile_area_km2 from the bounding box "
                  f"(median {median:.4f} km²).")
            print(f"warning: these envelopes are inflated by meridian convergence "
                  f"({100 * (median - 1):.1f} %% at the median) and sum to "
                  f"{frame['tile_area_km2'].sum():,.0f} km². Percentages are "
                  f"unaffected; areas derived from them are not.", file=sys.stderr)
        else:
            frame["tile_area_km2"] = args.constant_area
            print(f"Set tile_area_km2 = {args.constant_area} km² per tile.")

    if args.land_areas:
        if not args.land_areas.exists():
            print(f"error: land areas not found: {args.land_areas}", file=sys.stderr)
            return 2
        frame = attach_land_areas(frame, args.land_areas)

    if args.land_mask:
        if not args.land_mask.exists():
            print(f"error: land mask not found: {args.land_mask}", file=sys.stderr)
            return 2
        frame["on_land"] = flag_on_land(frame, args.land_mask, args.land_layer)
        offshore = int((~frame["on_land"]).sum())
        print(f"Land mask: {int(frame['on_land'].sum()):,} tile(s) on land, "
              f"{offshore:,} off it ({100 * offshore / len(frame):.1f} %).")
        if args.drop_offshore:
            frame = frame[frame["on_land"]]
            print(f"Dropped {offshore:,} offshore tile(s).")
    elif args.drop_offshore:
        print("error: --drop-offshore needs --land-mask", file=sys.stderr)
        return 2

    if args.drop_incomplete and OUR_COLUMN in frame.columns:
        before = len(frame)
        frame = frame[frame[OUR_COLUMN].notna()]
        print(f"Dropped {before - len(frame):,} row(s) with no value in "
              f"{OUR_COLUMN!r}.")

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.destination, index=False)
    print(f"\nWrote {len(frame):,} rows -> {args.destination}")

    if OUR_COLUMN in frame.columns:
        valid = frame[[OUR_COLUMN, "tile_area_km2"]].dropna()
        weighted = (valid[OUR_COLUMN] * valid["tile_area_km2"]).sum() / valid[
            "tile_area_km2"
        ].sum()
        print(f"Area-weighted mean {OUR_COLUMN}: {weighted:.2f} % "
              f"over {len(valid):,} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
