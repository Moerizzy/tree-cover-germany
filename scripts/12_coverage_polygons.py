#!/usr/bin/env python3
"""Dissolve the per-tile table into one polygon per acquisition date.

The map is 4,043 rasters and says nothing about *when* each place was
flown. This writes that: one polygon per date, per UTM zone, in the zone's
native CRS, so a reader can see the acquisition mosaic behind the map and
a GIS can join on it. The two files ship beside the mosaic.

The overlap rule is the merge's rule — **newest acquisition wins** (see
:mod:`treecover.merge`). Each date claims only the ground no later date
already claimed, so the polygons tile the mapped area without overlaps and
their areas sum to it. Tiles whose filename carries no date rank last and
form one undated polygon rather than being dropped.

Ported from ``mk_coverage.py``, which produced the published files.

Examples::

    python scripts/12_coverage_polygons.py \\
        --tiles publication/tiles/tile_treecover_products.csv \\
        --out publication/mosaic_utm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from treecover.config import load_paths

logger = logging.getLogger(__name__)

#: Tile edge, metres. The whole archive is on a 1 km grid.
TILE_SIZE_M = 1000.0

#: UTM zone → EPSG code. Germany spans exactly these two.
ZONE_EPSG = {32: 25832, 33: 25833}

#: Dates outside this range are typos or placeholders, not acquisitions.
DATE_RANGE = (20000000, 20301231)

#: Coordinates are rounded to a decimetre. At 20 cm that is below one
#: pixel and it halves the file.
COORDINATE_DECIMALS = 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tiles", type=Path, required=True,
                   help="Per-tile table with date, zone, tile_x_m and tile_y_m.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: results_root from paths.yaml).")
    p.add_argument("--zones", nargs="*", type=int, default=sorted(ZONE_EPSG),
                   help="UTM zones to write.")
    p.add_argument("--name", default="coverage_utm{zone}",
                   help="Output basename; {zone} is substituted.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _round(value, decimals: int = COORDINATE_DECIMALS):
    """Round nested coordinate lists in place of a full geometry rewrite."""
    if isinstance(value, (list, tuple)):
        return [_round(item, decimals) for item in value]
    return round(value, decimals)


def coverage_features(frame: pd.DataFrame, zone: int) -> list:
    """One geometry per date for a zone, newest date claiming its ground first.

    Args:
        frame: Tiles with ``date`` as ``YYYYMMDD`` integers, ``zone`` and
            the tile's south-west corner in metres.
        zone: UTM zone to build.

    Returns:
        ``[(date, geometry), …]`` sorted by date, disjoint by construction.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union

    subset = frame[frame["zone"] == zone]
    if subset.empty:
        return []

    per_date = {
        int(date): unary_union([
            box(x, y, x + TILE_SIZE_M, y + TILE_SIZE_M)
            for x, y in zip(group["tile_x_m"], group["tile_y_m"])
        ])
        for date, group in subset.groupby("date")
    }

    # Newest first, undated last — the same precedence the merge applies.
    order = sorted((d for d in per_date if d > 0), reverse=True)
    order += [d for d in per_date if d == 0]

    features, claimed = [], None
    for date in order:
        geometry = per_date[date]
        remaining = geometry if claimed is None else geometry.difference(claimed)
        claimed = geometry if claimed is None else claimed.union(geometry)
        if not remaining.is_empty:
            features.append((date, remaining))

    features.sort()
    return features


def to_geojson(features: list, zone: int) -> dict:
    """A FeatureCollection with the zone's CRS named the way GDAL expects."""
    from shapely.geometry import mapping

    collection = {
        "type": "FeatureCollection",
        "name": f"tree_cover_germany_coverage_utm{zone}",
        "crs": {"type": "name",
                "properties": {"name": f"urn:ogc:def:crs:EPSG::{ZONE_EPSG[zone]}"}},
        "features": [],
    }
    for date, geometry in features:
        text = str(date)
        properties = (
            {"month": int(text[4:6]),
             "datetime": f"{text[:4]}-{text[4:6]}-{text[6:8]}T00:00:00"}
            if date > 0 else {"month": None, "datetime": None}
        )
        shape = mapping(geometry)
        shape["coordinates"] = _round(shape["coordinates"])
        collection["features"].append(
            {"type": "Feature", "properties": properties, "geometry": shape}
        )
    return collection


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    if not args.tiles.exists():
        print(f"error: --tiles not found: {args.tiles}", file=sys.stderr)
        return 2

    columns = ["date", "zone", "tile_x_m", "tile_y_m"]
    frame = pd.read_csv(args.tiles, usecols=columns)
    frame["date"] = pd.to_numeric(frame["date"], errors="coerce").fillna(0).astype(int)
    unusable = (frame["date"] < DATE_RANGE[0]) | (frame["date"] > DATE_RANGE[1])
    frame.loc[unusable, "date"] = 0
    logger.info("%d tiles, %d without a usable date", len(frame), int(unusable.sum()))

    out_dir = Path(args.out or load_paths().get_path("results_root", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    for zone in args.zones:
        if zone not in ZONE_EPSG:
            print(f"error: no EPSG code for UTM zone {zone}", file=sys.stderr)
            return 2

        features = coverage_features(frame, zone)
        if not features:
            logger.warning("UTM%d: no tiles", zone)
            continue

        area = sum(geometry.area for _, geometry in features) / 1e6
        undated = sum(1 for date, _ in features if date == 0)
        path = out_dir / f"{args.name.format(zone=zone)}.geojson"
        path.write_text(json.dumps(to_geojson(features, zone)), encoding="utf-8")

        print(f"UTM{zone}: {len(features)} polygon(s) "
              f"({undated} undated), {area:,.1f} km² → {path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
