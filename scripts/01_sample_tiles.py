#!/usr/bin/env python3
"""Draw the training tiles: stratified by tree cover density and settlement.

Reads a state's acquisition index — one feature per flight over a 1 km tile
— and returns the tiles to label, filling a 4 x 2 grid of Copernicus
density bins crossed with settlement type. Only tiles flown both in summer
and outside it are eligible: one label mask is drawn on the summer image
and inherited by the other acquisition, which is what makes season-aware
training possible without labelling every flight.

The published run drew from Lower Saxony's bDOM index, asked for 200 tiles
and got 152 — the 2 km separation constraint exhausted the strata first.
Those 152 are ``sampled_tiles_100.gpkg`` of the training-data package, and
stage 3 turns them into patches.

The output carries every column of that file but one: ``Change``, which
flags the tiles whose ground changed between the two acquisitions, was set
by eye in QGIS after the draw and before labelling. Stage 3 drops the rows
it marks, so a fresh draw needs the same inspection pass.

Two rasters are needed besides the index: Copernicus HRL tree cover density
for the density stratum, and CORINE Land Cover 2018 for the settlement one.
Reading them is the slow part — around an hour for a state — so
``--save-attributes`` caches the per-tile table and ``--attributes`` reuses
it, which makes re-drawing with different targets a matter of seconds.

.. note::
   The draw is not bit-reproducible across inputs. It walks the candidate
   table in order and draws with the global numpy seed, so a tile index
   that has gained flights since 2024 gives a different, equally valid set.
   ``--compare`` reports the overlap with a reference table instead of
   pretending otherwise.

Examples::

    # Paths from the sampling block of paths.yaml
    python scripts/01_sample_tiles.py --out results/sampling

    # Cache the expensive raster reads, then re-draw cheaply
    python scripts/01_sample_tiles.py --save-attributes tile_attributes.csv \\
        --out results/sampling
    python scripts/01_sample_tiles.py --attributes tile_attributes.csv \\
        --n-total 50 --out results/sampling_small

    # Reproduce the published draw and check it against the package
    python scripts/01_sample_tiles.py --out results/sampling \\
        --exclude .../gadm41_DEU.gpkg --exclude-where "NAME_1 == 'Bremen'" \\
        --compare /mnt/publication/training/sampled_tiles_100.gpkg
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from treecover.config import load_paths
from treecover.data.tile_sampling import (
    PUBLISHED_BIN_TARGETS,
    PUBLISHED_SETTINGS,
    PUBLISHED_TILE_COUNT,
    assign_splits,
    assign_tcd_bins,
    border_tile_ids,
    dominant_land_cover,
    mean_tcd,
    stratified_sample,
    tiles_from_index,
)
from treecover.io.vector import read_vector

logger = logging.getLogger(__name__)

#: Columns of the published GeoPackage, in its order.
EXPORT_COLUMNS = [
    "tile_id", "split", "mean_tcd", "tcd_bin", "is_urban", "urban_pct",
    "dominant_lc", "dominant_lc_name", "image_count", "n_flight_dates",
    "flight_dates_str",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    src = p.add_argument_group("input")
    src.add_argument("--index", type=Path, default=None,
                     help="Acquisition index: one feature per flight over a tile, "
                          "with a tile id, a date and a geometry. "
                          "Default: sampling.tile_index from paths.yaml.")
    src.add_argument("--tile-id-column", default="tile_id")
    src.add_argument("--date-column", default="Aktualitaet",
                     help="Acquisition date column of the index.")
    src.add_argument("--tcd", default=None, metavar="GLOB",
                     help="Copernicus HRL tree cover density raster(s). "
                          "Default: sampling.tcd_pattern.")
    src.add_argument("--clc", type=Path, default=None,
                     help="CORINE Land Cover 2018, class-coded. Default: sampling.clc.")
    src.add_argument("--attributes", type=Path, default=None, metavar="FILE",
                     help="Reuse a per-tile attribute table written by "
                          "--save-attributes, instead of reading the rasters again.")
    src.add_argument("--exclude", type=Path, default=None, metavar="VECTOR",
                     help="Drop tiles intersecting this geometry. The published run "
                          "excluded Bremen, which is enclosed by Lower Saxony and "
                          "publishes its own orthophotos.")
    src.add_argument("--exclude-layer", default=None,
                     help="Layer of --exclude, for multi-layer files. GADM 4.1 "
                          "carries the states in ADM_ADM_1.")
    src.add_argument("--exclude-where", default=None, metavar="QUERY",
                     help="Pandas query selecting the features of --exclude to use, "
                          "e.g. \"NAME_1 == 'Bremen'\" for a GADM level-1 file.")
    src.add_argument("--limit", type=int, default=None, metavar="N",
                     help="Read the rasters for the first N candidate tiles only. "
                          "For smoke-testing the stage; the draw it produces is not "
                          "a training set.")

    sel = p.add_argument_group("selection")
    sel.add_argument("--n-total", type=int, default=PUBLISHED_SETTINGS["n_total"],
                     help="Tiles to draw. The published run asked for %(default)s "
                          "and the separation constraint stopped it at 152.")
    sel.add_argument("--min-distance-km", type=float,
                     default=PUBLISHED_SETTINGS["min_distance_km"],
                     help="Minimum separation between drawn tiles. Strictly enforced: "
                          "the draw stops rather than violating it.")
    sel.add_argument("--min-tcd", type=float, default=PUBLISHED_SETTINGS["min_tcd"],
                     help="Drop tiles below this mean tree cover density (%%).")
    sel.add_argument("--max-per-flight", type=int,
                     default=PUBLISHED_SETTINGS["max_per_flight"],
                     help="Cap on tiles sharing one flight date within a stratum. "
                          "0 disables the cap.")
    sel.add_argument("--no-month-balance", action="store_true",
                     help="Stop preferring tiles with under-represented flight months.")
    sel.add_argument("--bin-targets", default=None, metavar="JSON",
                     help="Per-stratum targets as JSON, e.g. "
                          "'{\"B1: 0-10%%\": {\"urban\": 6, \"nonurban\": 19}}'. "
                          "Default: the published targets.")
    sel.add_argument("--seed", type=int, default=PUBLISHED_SETTINGS["seed"],
                     help="Global numpy seed for the draws.")
    sel.add_argument("--no-split", action="store_true",
                     help="Skip the train/val/test column. It is not the split the "
                          "published model used — see the module docs of "
                          "treecover.data.tile_sampling.assign_splits.")

    out = p.add_argument_group("output")
    out.add_argument("--out", type=Path, default=None,
                     help="Output directory (default: results_root from paths.yaml).")
    out.add_argument("--name", default="sampled_tiles",
                     help="Basename of the GeoPackage and CSV.")
    out.add_argument("--save-attributes", type=Path, default=None, metavar="FILE",
                     help="Write the per-tile attribute table, so a re-draw with "
                          "other targets needs no raster reads. Give it a vector "
                          "suffix (.gpkg, .geojson) to keep the tile geometries — a "
                          ".csv can be inspected but not drawn from on its own.")
    out.add_argument("--compare", type=Path, default=None, metavar="TILES",
                     help="Reference tile table to report the overlap against, "
                          "e.g. the published sampled_tiles_100.gpkg.")
    out.add_argument("--force", action="store_true",
                     help="Overwrite an existing output. Without it the run refuses: "
                          "re-drawing changes which tiles were labelled.")
    out.add_argument("-v", "--verbose", action="store_true")
    return p


def build_attributes(args) -> pd.DataFrame:
    """Per-tile table: seasons, density, land cover, border and exclusion.

    This is the expensive half of the stage — every tile is read out of two
    raster products — and it depends only on the state, not on the sampling
    targets. Hence ``--save-attributes``.
    """
    index = read_vector(args.index)
    logger.info("Index: %d acquisitions from %s", len(index), args.index.name)

    tiles = tiles_from_index(index, args.tile_id_column, args.date_column)

    # Border detection needs every tile the state knows, not just the
    # candidates — otherwise the edge of the candidate set reads as the
    # edge of the state.
    borders = border_tile_ids(tiles[args.tile_id_column])

    # The rasters are only read for tiles that can be used at all. This is
    # what makes the stage take an hour instead of most of a day.
    candidates = tiles[tiles["has_both"]].copy()
    if args.limit:
        candidates = candidates.head(args.limit).copy()
        logger.warning("--limit %d: a smoke test, not a training set", args.limit)
    logger.info("Reading rasters for %d tiles with both seasons", len(candidates))

    density = mean_tcd(candidates, args.tcd, args.tile_id_column)
    land_cover = dominant_land_cover(candidates, args.clc, args.tile_id_column)

    candidates["mean_tcd"] = density.to_numpy()
    for column in land_cover.columns:
        candidates[column] = land_cover[column].to_numpy()
    candidates["tcd_bin"] = assign_tcd_bins(candidates["mean_tcd"])
    candidates["is_border"] = candidates[args.tile_id_column].isin(borders)
    candidates["excluded"] = _exclusion_mask(candidates, args)

    return candidates


def _exclusion_mask(candidates, args) -> pd.Series:
    """Which candidates intersect the exclusion geometry."""
    if args.exclude is None:
        return pd.Series(False, index=candidates.index)

    excluded = read_vector(args.exclude, layer=args.exclude_layer)
    if args.exclude_where:
        excluded = excluded.query(args.exclude_where)
        if excluded.empty:
            raise SystemExit(
                f"error: --exclude-where {args.exclude_where!r} matched no feature "
                f"in {args.exclude.name}"
            )
    if excluded.crs != candidates.crs:
        excluded = excluded.to_crs(candidates.crs)

    # union_all() from GeoPandas 1.0, unary_union before it.
    geometry = excluded.geometry
    dissolved = (geometry.union_all() if hasattr(geometry, "union_all")
                 else geometry.unary_union)

    mask = candidates.geometry.intersects(dissolved)
    logger.info("Excluded by %s: %d of %d tiles", args.exclude.name, int(mask.sum()),
                len(candidates))
    if mask.mean() > 0.25:
        # The usual cause is a multi-layer file read at the wrong level:
        # GADM's ADM_ADM_0 is the whole country, not one state.
        logger.warning(
            "%.0f %% of the candidates are excluded. Check --exclude-layer and "
            "--exclude-where — a country-level layer excludes everything.",
            100 * mask.mean(),
        )
    return mask


#: Attribute-cache suffixes that keep the tile geometries.
VECTOR_SUFFIXES = (".gpkg", ".geojson", ".json", ".fgb", ".shp")


def save_attributes(attributes, path: Path) -> None:
    """Cache the per-tile table so a re-draw needs no raster reads.

    Written as a vector file by default, because the drawn tiles have to be
    exported with their geometry and re-reading the index to recover it
    would defeat the point of the cache.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = attributes.copy()
    # A list column has no representation in either format.
    frame["flight_dates"] = frame["flight_dates"].apply(
        lambda dates: ", ".join(str(d) for d in dates)
    )
    if path.suffix.lower() in VECTOR_SUFFIXES:
        frame["tcd_bin"] = frame["tcd_bin"].astype(str)
        frame.to_file(path)
    else:
        frame.drop(columns="geometry").to_csv(path, index=False)
        logger.warning(
            "%s holds no geometry, so a run using --attributes %s cannot write the "
            "drawn tiles. Use a .gpkg for that.", path.name, path.name,
        )
    logger.info("Attributes written to %s", path)


def load_attributes(path: Path, tile_id_column: str) -> pd.DataFrame:
    """Read back a cached attribute table, restoring the list column.

    ``flight_dates`` is a list per row and survives the round trip only as
    text, so it is parsed back — the sampler needs the individual dates for
    the per-flight cap and the month balance.
    """
    if path.suffix.lower() in VECTOR_SUFFIXES:
        frame = read_vector(path)
    else:
        frame = pd.read_csv(path)
    frame[tile_id_column] = frame[tile_id_column].astype(str)
    if "flight_dates" in frame.columns:
        frame["flight_dates"] = frame["flight_dates"].apply(_parse_dates)
    for column in ("has_both", "has_summer", "has_non_summer", "is_urban",
                   "is_border", "excluded"):
        if column in frame.columns:
            frame[column] = frame[column].astype(bool)
    logger.info("Attributes: %d tiles from %s", len(frame), path.name)
    return frame


def _parse_dates(value) -> list:
    """A CSV cell back into a list of timestamps."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        items = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        items = [part.strip() for part in value.split(",")]
    return [pd.to_datetime(item) for item in items if str(item).strip()]


def to_export_frame(run, attributes, args):
    """The drawn tiles in the column layout of the published GeoPackage."""
    drawn = attributes[attributes[args.tile_id_column].isin(run.tile_ids)].copy()
    drawn = drawn.sort_values(args.tile_id_column).reset_index(drop=True)

    drawn["tcd_bin"] = drawn["tcd_bin"].astype(str)
    drawn["flight_dates_str"] = drawn["flight_dates"].apply(
        lambda dates: ", ".join(str(d) for d in dates)
    )
    drawn["split"] = (
        "train" if args.no_split else assign_splits(drawn).to_numpy()
    )

    columns = [c for c in EXPORT_COLUMNS if c in drawn.columns]
    return drawn[columns + ["geometry"]]


def report_overlap(drawn, run, reference_path: Path, tile_id_column: str) -> None:
    """Say plainly how much of a reference draw this run reproduced.

    Silence would be the failure mode. A draw from a tile index that has
    grown since 2024 is a perfectly good training set and nothing
    downstream would notice it is a different one.

    The number that matters is not the overlap but how many reference tiles
    were *eligible*: a reference tile the filters reject is a defect in
    this stage, whereas one that was merely not picked is the random draw
    working on a pool that has changed.
    """
    reference = read_vector(reference_path)
    if tile_id_column not in reference.columns:
        print(f"\n--compare: {reference_path.name} has no {tile_id_column!r} column",
              file=sys.stderr)
        return

    reference_ids = set(reference[tile_id_column].astype(str))
    drawn_ids = set(drawn[tile_id_column].astype(str))
    eligible_ids = set(run.eligible[tile_id_column].astype(str))
    shared = reference_ids & drawn_ids
    rejected = reference_ids - eligible_ids

    print(f"\nAgainst {reference_path.name}: {len(shared)} of {len(reference_ids)} "
          f"tiles in common ({100 * len(shared) / max(len(reference_ids), 1):.0f} %), "
          f"{len(reference_ids) - len(rejected)} of them eligible here")

    if rejected:
        print(f"  {len(rejected)} reference tile(s) the filters reject: "
              f"{', '.join(sorted(rejected)[:8])}"
              f"{' …' if len(rejected) > 8 else ''}")
        print("  Those were labelled and trained on, so a filter here is stricter\n"
              "  than the one that drew them. Worth looking at.")
    elif len(shared) < len(reference_ids):
        print(f"  Every reference tile passes the filters; {len(reference_ids - drawn_ids)} "
              f"were simply not drawn,\n  and {len(drawn_ids - reference_ids)} others "
              "were. Expected unless the tile index is the\n  2024 one byte for byte — "
              "the draw walks the candidate table in order, so\n  one added flight "
              "shifts everything after it.")


def _configured(paths, key: str) -> Path | None:
    """A path from paths.yaml, or None when the key is not set."""
    value = paths.get_value(key)
    return Path(str(value)) if value else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    paths = load_paths()
    out_dir = Path(args.out or paths.get_path("results_root", "./results"))
    gpkg = out_dir / f"{args.name}.gpkg"
    if gpkg.exists() and not args.force:
        print(f"error: {gpkg} exists. Re-drawing gives a different training set, "
              "so this needs --force.", file=sys.stderr)
        return 2

    if args.attributes:
        if not args.attributes.exists():
            print(f"error: --attributes not found: {args.attributes}", file=sys.stderr)
            return 2
        attributes = load_attributes(args.attributes, args.tile_id_column)
    else:
        args.index = args.index or _configured(paths, "sampling.tile_index")
        args.tcd = args.tcd or paths.get_value("sampling.tcd_pattern")
        args.clc = args.clc or _configured(paths, "sampling.clc")
        args.exclude = args.exclude or _configured(paths, "sampling.exclude")

        for label, path in (("--index", args.index), ("--clc", args.clc)):
            if path is None or not Path(path).exists():
                print(f"error: {label} not found: {path}. Set it on the command line "
                      "or in the sampling block of paths.yaml.", file=sys.stderr)
                return 2
        if not args.tcd:
            print("error: --tcd not set and sampling.tcd_pattern is missing from "
                  "paths.yaml.", file=sys.stderr)
            return 2
        attributes = build_attributes(args)

        if args.save_attributes:
            save_attributes(attributes, args.save_attributes)

    if "geometry" not in attributes.columns:
        print("error: the attribute table has no geometry, so the drawn tiles cannot "
              "be written. Re-run without --attributes, or cache to a .gpkg.",
              file=sys.stderr)
        return 2

    targets = PUBLISHED_BIN_TARGETS
    if args.bin_targets:
        targets = json.loads(args.bin_targets)

    run = stratified_sample(
        attributes,
        n_total=args.n_total,
        min_distance_km=args.min_distance_km,
        bin_targets=targets,
        min_tcd=args.min_tcd,
        max_per_flight=args.max_per_flight or None,
        balance_months=not args.no_month_balance,
        seed=args.seed,
        tile_id_column=args.tile_id_column,
    )
    if not run.tile_ids:
        print("error: no tile could be drawn. Lower --min-tcd or --min-distance-km, "
              "or check that the index has tiles flown in two seasons.",
              file=sys.stderr)
        return 1

    drawn = to_export_frame(run, attributes, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    drawn.to_file(gpkg, driver="GPKG")
    csv_path = out_dir / f"{args.name}.csv"
    drawn.drop(columns="geometry").to_csv(csv_path, index=False)

    print(run.summary())
    print(f"\nWritten to {out_dir}:\n  {gpkg.name}\n  {csv_path.name}")

    if len(run.tile_ids) == PUBLISHED_TILE_COUNT:
        print(f"\n{PUBLISHED_TILE_COUNT} tiles — the size of the published draw.")
    elif run.stopped_early:
        print(f"\n{len(run.tile_ids)} tiles rather than the {args.n_total} asked for: "
              f"the {args.min_distance_km} km separation exhausted the strata. The "
              f"published run ended the same way, at {PUBLISHED_TILE_COUNT}.")

    if args.compare:
        report_overlap(drawn, run, args.compare, args.tile_id_column)

    print(f"\nNext: label the tiles, then "
          f"python scripts/03_prepare_patches.py --tiles {gpkg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
