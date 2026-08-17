#!/usr/bin/env python3
"""Compare the map against other tree cover products.

Produces the paper's Results tables:

* ``grid_1km.csv``          area-weighted tree cover per 1 km cell, per product
* ``per_state.csv``         per Bundesland, with tree area in km²
* ``per_district.csv``      per Landkreis (with ``--districts``)
* ``table1.csv``            national summary and difference against ours

All aggregation is area-weighted, and a product only enters a total over
tiles where our own map is valid — otherwise a product with wider coverage
reports a larger area purely for covering more ground.

Examples::

    python scripts/09_compare_products.py --tiles tiles_with_treecover.csv
    python scripts/09_compare_products.py --tiles tiles.csv --states gadm41_DEU.gpkg
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from treecover.comparison import (
    OUR_COLUMN,
    WEIGHT_COLUMNS,
    aggregate_by_unit,
    aggregate_to_grid,
    difference_table,
    weight_column,
)
from treecover.config import load_paths
from treecover.figures.style import PRODUCT_LABELS
from treecover.io.vector import read_vector

logger = logging.getLogger(__name__)

#: Ours first, then the three comparison products of the manuscript.
#: Names come from treecover.figures.style, the single source for them.
PRODUCTS = dict(PRODUCT_LABELS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tiles", type=Path, required=True,
                   help="Per-tile CSV with lon_c, lat_c, tile_area_km2 and one "
                        "tree-cover column per product.")
    p.add_argument("--states", type=Path, default=None,
                   help="Bundesland boundaries (GADM level 1).")
    p.add_argument("--states-layer", default="ADM_ADM_1",
                   help="Layer name for multi-layer formats such as GeoPackage.")
    p.add_argument("--districts", type=Path, default=None,
                   help="Landkreis boundaries (GADM level 2).")
    p.add_argument("--districts-layer", default="ADM_ADM_2")
    p.add_argument("--name-column", default="NAME_1",
                   help="Name column in the boundary file.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: <results_root>/comparison).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def load_units(path: Path, name_column: str, layer: str | None = None,
               equal_area_epsg: int = 3035):
    """Read boundaries and attach each unit's land area in km².

    The area is computed in an equal-area projection (ETRS89-LAEA by
    default); measuring it in degrees would make northern states look
    smaller than they are and skew every area-weighted national mean.
    """
    units = read_vector(path, layer=layer)
    if name_column not in units.columns:
        raise KeyError(
            f"{path.name} has no {name_column!r} column. Available: {list(units.columns)}"
        )
    units = units.rename(columns={name_column: "admin_name"})
    units["unit_area_km2"] = units.to_crs(equal_area_epsg).geometry.area / 1e6
    return units[["admin_name", "unit_area_km2", "geometry"]]


def tiles_to_geoframe(tiles: pd.DataFrame):
    """Tile centroids as a GeoDataFrame in WGS 84."""
    import geopandas as gpd

    return gpd.GeoDataFrame(
        tiles,
        geometry=gpd.points_from_xy(tiles["lon_c"], tiles["lat_c"]),
        crs="EPSG:4326",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    if not args.tiles.exists():
        print(f"error: {args.tiles} not found", file=sys.stderr)
        return 2

    tiles = pd.read_csv(args.tiles)
    if OUR_COLUMN not in tiles.columns:
        print(f"error: {args.tiles.name} has no {OUR_COLUMN!r} column.\n"
              f"       Available: {list(tiles.columns)}", file=sys.stderr)
        return 2
    if not any(c in tiles.columns for c in WEIGHT_COLUMNS):
        # 1 km tiles unless told otherwise; stated rather than assumed silently.
        logger.warning("No %s column — assuming 1 km² per tile.",
                       " or ".join(repr(c) for c in WEIGHT_COLUMNS))
        tiles["tile_area_km2"] = 1.0
    weights = weight_column(tiles)

    present = [c for c in PRODUCTS if c in tiles.columns]
    missing = [c for c in PRODUCTS if c not in tiles.columns]
    print(f"Tiles      : {len(tiles):,}")
    print(f"Weight     : {weights} ({tiles[weights].sum():,.0f} km² total)")
    print(f"Products   : {', '.join(PRODUCTS[c] for c in present)}")
    if missing:
        print(f"Not present: {', '.join(PRODUCTS[c] for c in missing)}")

    out_dir = args.out or (load_paths().get_path("results_root", "./results") / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = aggregate_to_grid(tiles, present)
    grid.frame.to_csv(out_dir / "grid_1km.csv", index=False)
    print(f"\n1 km grid: {len(grid.frame):,} cells")
    for column in present:
        if column in grid.totals_km2:
            print(f"  {PRODUCTS[column]:<22} {grid.totals_km2[column]:>10,.0f} km²")

    if not args.states:
        print("\nNo --states given; skipping the administrative summary and Table 1.")
        return 0

    try:
        states = load_units(args.states, args.name_column, args.states_layer)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    tiles_gdf = tiles_to_geoframe(tiles).to_crs(states.crs)
    per_state = aggregate_by_unit(tiles_gdf, states, present)
    per_state.drop(columns="geometry").to_csv(out_dir / "per_state.csv", index=False)
    print(f"\nPer state: {int(per_state[OUR_COLUMN].notna().sum())} of {len(per_state)} "
          "units have data")

    if args.districts:
        districts = load_units(args.districts, "NAME_2", args.districts_layer)
        per_district = aggregate_by_unit(tiles_gdf.to_crs(districts.crs), districts, present)
        per_district.drop(columns="geometry").to_csv(
            out_dir / "per_district.csv", index=False
        )
        print(f"Per district: {int(per_district[OUR_COLUMN].notna().sum())} of "
              f"{len(per_district)} units have data")

    table1 = difference_table(per_state.drop(columns="geometry"), present, PRODUCTS)
    table1.to_csv(out_dir / "table1.csv", index=False)

    print("\nTable 1 — national summary")
    print(f"{'product':<22} {'cover %':>9} {'area km²':>12} {'Δ pp':>8} {'Δ %':>8}")
    for _, row in table1.iterrows():
        print(f"{row['product']:<22} {row['tree_cover_pct']:>9.2f} "
              f"{row['tree_area_km2']:>12,.0f} {row['diff_pp']:>+8.2f} "
              f"{row['diff_pct']:>+8.1f}")
    print(f"\nWritten to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
