#!/usr/bin/env python3
"""Build the training patch table from downloaded imagery and labels.

Takes the sampled tiles plus the orthophotos, label masks and optional
height models fetched by stage 2, and writes the three artefacts stage 4
consumes:

* ``patches.csv``       one row per patch: region, window, date, split, cover
* ``region_vrts.json``  region_id -> imagery / mask / nDSM paths
* ``splits.json``       train / val / test region lists

Filenames must be ``<tile_id>_<date>.tif``. Every acquisition of a tile
shares that tile's label mask, which is what makes multi-season training
possible — see :mod:`treecover.data.observations`.

.. warning::
   The ``split`` column in ``sampled_tiles_100.gpkg`` is **not** the split
   the published model was trained on. It is an earlier three-way draw
   (train 73 / val 23 / test 23) whose classes cut across both published
   ones; using it puts published training tiles into validation and inflates
   every metric measured afterwards. The real assignment lives in the
   training-data package's ``patches/observations.csv``, which this script
   reads by default — see :func:`apply_published_splits`.

Run with no arguments to reproduce the published training set exactly:
117 tiles, 245 observations, 19,845 patches (15,957 train / 3,888 val).
The run reports whether it did.

Examples::

    # Paths come from the training_data block of paths.yaml
    python scripts/03_prepare_patches.py

    python scripts/03_prepare_patches.py \\
        --tiles publication/training/sampled_tiles_100.gpkg \\
        --images .../DOP --masks publication/training/labels

    # Denser sampling, drop patches that are mostly nodata
    python scripts/03_prepare_patches.py \\
        --patch-size 512 --stride 128 --max-nodata-frac 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from treecover.config import load_paths
from treecover.data.observations import build_observations
from treecover.data.patches import extract_all, region_vrt_map, split_map
from treecover.data.seasons import month_from_date, season_from_month
from treecover.io.vector import read_vector

logger = logging.getLogger(__name__)

#: Extraction settings of the published training set, from the original run's
#: ``experiment_config.json``. Non-overlapping: stride equals patch size.
#: A stride of 256 gives four times the patches and a different training set.
PUBLISHED_PATCH_SIZE = 512
PUBLISHED_STRIDE = 512

#: What the published run produced, for the self-check at the end.
PUBLISHED_TOTALS = {"tiles": 117, "observations": 245, "patches": 19845,
                    "train_patches": 15957, "val_patches": 3888}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Defaults come from the training_data block of paths.yaml — the layout
    # of the published training-data package.
    src = p.add_argument_group("input")
    src.add_argument("--tiles", type=Path, default=None,
                     help="Sampled tiles (GeoPackage/GeoJSON) with tile_id and "
                          "split. Default: training_data.tiles from paths.yaml.")
    src.add_argument("--images", type=Path, default=None,
                     help="Orthophotos named <tile_id>_<date>.tif. "
                          "Default: training_data.images.")
    src.add_argument("--masks", type=Path, default=None,
                     help="Label masks, one per tile. Default: training_data.labels.")
    src.add_argument("--ndsm", type=Path, default=None,
                     help="Height models, for the RGB_nDSM / RGBI_nDSM "
                          "configurations. Default: training_data.ndsm.")
    src.add_argument("--tile-id-column", default="tile_id")
    src.add_argument("--split-column", default="split")
    src.add_argument("--splits-from", type=Path, default=None,
                     help="CSV with tile_id and split columns that overrides the "
                          "tile table's own split, and restricts the run to the "
                          "tiles it lists. Default: the published "
                          "observations.csv from training_data.patches. Pass '' "
                          "to use the tile table's split column instead — see the "
                          "warning in the module docstring before you do.")
    src.add_argument("--change-column", default="Change",
                     help="Tiles flagged '1' here changed between label and imagery "
                          "and are dropped. Pass '' to disable.")

    ext = p.add_argument_group("extraction")
    ext.add_argument("--patch-size", type=int, default=PUBLISHED_PATCH_SIZE)
    ext.add_argument("--stride", type=int, default=PUBLISHED_STRIDE,
                     help="Step between patches. The default equals the patch "
                          "size, i.e. non-overlapping, which is what the "
                          "manuscript describes and what the published model was "
                          "trained on. Half the patch size gives 50%% overlap and "
                          "roughly four times the patches.")
    ext.add_argument("--max-temporal-distance", type=int, default=1,
                     help="How many acquisitions from the label source an observation "
                          "may be. 0 keeps only the label source itself.")
    ext.add_argument("--min-tree-cover", type=float, default=0.0,
                     help="Drop patches below this canopy %%. Leave at 0 — treeless "
                          "patches are what stop the model hallucinating canopy.")
    ext.add_argument("--max-nodata-frac", type=float, default=1.0,
                     help="Drop patches whose label exceeds this nodata fraction.")

    out = p.add_argument_group("output")
    out.add_argument("--out", type=Path, default=None,
                     help="Output directory (default: patches_dir from paths.yaml).")
    out.add_argument("--no-progress", action="store_true")
    out.add_argument("-v", "--verbose", action="store_true")
    return p


def apply_published_splits(tiles, path: Path, tile_id_column: str,
                           split_column: str):
    """Replace the tile table's split with the one an observation table records.

    The ``split`` column shipped in ``sampled_tiles_100.gpkg`` is an earlier
    three-way draw — train 73 / val 23 / test 23 — and it does **not** match
    the split the published model was trained on. Cross-tabulating the two
    shows the published train and val sets cutting across all three of its
    classes, so the published run drew a fresh 80/20 at tile level and never
    used that column.

    Taking the split from the published ``observations.csv`` instead does two
    things at once: it restores the real assignment, and it restricts the run
    to the 117 tiles that run actually used. The two extra tiles the column
    would add (``323995948`` and ``324215950``) keep only summer acquisitions
    after the temporal-distance filter, so they carry no phenological pair —
    which is presumably why the published run dropped them.

    Args:
        tiles: The sampled-tile table.
        path: CSV holding ``tile_id`` and ``split``.
        tile_id_column: Tile id column in ``tiles``.
        split_column: Split column to overwrite in ``tiles``.

    Returns:
        ``tiles`` restricted to the listed tiles, with their split.
    """
    reference = pd.read_csv(path)
    missing = {"tile_id", "split"} - set(reference.columns)
    if missing:
        raise SystemExit(
            f"error: {path} has no {', '.join(sorted(missing))} column; it "
            "cannot supply the splits."
        )

    assignment = dict(
        zip(reference["tile_id"].astype(str), reference["split"].astype(str))
    )
    ids = tiles[tile_id_column].astype(str)
    before, before_counts = len(tiles), tiles[split_column].value_counts().to_dict()
    tiles = tiles[ids.isin(assignment)].copy()
    tiles[split_column] = tiles[tile_id_column].astype(str).map(assignment)

    logger.info(
        "Splits taken from %s: %d of %d tiles kept, %s (was %s)",
        path.name, len(tiles), before,
        tiles[split_column].value_counts().to_dict(), before_counts,
    )
    if len(tiles) == 0:
        raise SystemExit(
            f"error: no tile id in {path} matches the tile table. Check that "
            "both use the same id format."
        )
    return tiles


def summarise(patches_df: pd.DataFrame) -> None:
    """Report what the split and season balance actually came out as."""
    print(f"\nPatches: {len(patches_df)} from "
          f"{patches_df['region_id'].nunique()} observation(s) / "
          f"{patches_df['tile_id'].nunique()} tile(s)")

    print("\nBy split:")
    for split, group in patches_df.groupby("split"):
        print(f"  {split:<6} {len(group):6d} patches, "
              f"{group['tile_id'].nunique():3d} tiles, "
              f"mean tree cover {group['tree_cover_pct'].mean():5.1f} %")

    seasons = patches_df["date"].map(lambda d: season_from_month(month_from_date(d)))
    print("\nBy season (training patches — the weighted sampler equalises these):")
    train = seasons[patches_df["split"] == "train"]
    if train.empty:
        print("  none")
        return
    for season, count in train.value_counts().items():
        print(f"  {season:<12} {count:6d}  ({100 * count / len(train):4.1f} %)")

    empty = (patches_df["tree_cover_pct"] == 0).sum()
    print(f"\nTreeless patches: {empty} ({100 * empty / len(patches_df):.1f} %) — "
          "kept on purpose, they teach the negative case")

    _compare_to_published(patches_df)


def _compare_to_published(patches_df: pd.DataFrame) -> None:
    """Say plainly whether this run reproduced the published training set.

    Silence here would be the failure mode: a different stride or a different
    split still produces a perfectly valid patch table, just not the one the
    published model was trained on, and nothing downstream would notice.
    """
    per_split = patches_df["split"].value_counts().to_dict()
    actual = {
        "tiles": patches_df["tile_id"].nunique(),
        "observations": patches_df["region_id"].nunique(),
        "patches": len(patches_df),
        "train_patches": per_split.get("train", 0),
        "val_patches": per_split.get("val", 0),
    }
    differences = {k: (v, actual[k]) for k, v in PUBLISHED_TOTALS.items()
                   if actual[k] != v}

    if not differences:
        print("\nMatches the published training set exactly "
              f"({PUBLISHED_TOTALS['patches']:,} patches, "
              f"{PUBLISHED_TOTALS['observations']} observations, "
              f"{PUBLISHED_TOTALS['tiles']} tiles).")
        return

    print("\nDiffers from the published training set:")
    for key, (expected, got) in differences.items():
        print(f"  {key:<14} published {expected:>7,}   this run {got:>7,}")
    print("  Expected if you changed --stride, --patch-size, --splits-from or\n"
          "  --max-temporal-distance. Otherwise the inputs differ from the\n"
          "  published training-data package.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    paths = load_paths()
    args.tiles = args.tiles or paths.get_path("training_data.tiles")
    args.images = args.images or paths.get_path("training_data.images")
    args.masks = args.masks or paths.get_path("training_data.labels")
    if args.ndsm is None:
        configured = paths.get_value("training_data.ndsm")
        args.ndsm = Path(configured) if configured else None

    for label, path in (("--tiles", args.tiles), ("--images", args.images),
                        ("--masks", args.masks)):
        if not path.exists():
            print(f"error: {label} not found: {path}", file=sys.stderr)
            return 2

    tiles = read_vector(args.tiles)
    for column in (args.tile_id_column, args.split_column):
        if column not in tiles.columns:
            print(f"error: column {column!r} not in {args.tiles.name}. "
                  f"Available: {list(tiles.columns)}", file=sys.stderr)
            return 2

    splits_from = args.splits_from
    if splits_from is None:
        candidate = paths.get_value("training_data.patches")
        if candidate:
            candidate = Path(candidate) / "observations.csv"
            splits_from = candidate if candidate.exists() else None
    if splits_from:
        tiles = apply_published_splits(tiles, splits_from,
                                       args.tile_id_column, args.split_column)
    else:
        logger.warning(
            "Using the %r column of %s. That column is an earlier three-way "
            "draw and is NOT the split the published model was trained on — it "
            "puts published training tiles into validation. Point --splits-from "
            "at the published observations.csv to reproduce the paper.",
            args.split_column, args.tiles.name,
        )

    observations = build_observations(
        image_dir=args.images,
        mask_dir=args.masks,
        tiles=tiles,
        ndsm_dir=args.ndsm,
        max_temporal_distance=args.max_temporal_distance,
        tile_id_column=args.tile_id_column,
        split_column=args.split_column,
        change_column=args.change_column or None,
    )
    if not observations:
        print("error: no observations matched. Check that image filenames are "
              "<tile_id>_<date>.tif and that the tile ids line up.", file=sys.stderr)
        return 1

    patches = extract_all(
        observations,
        patch_size=args.patch_size,
        stride=args.stride,
        min_tree_cover_pct=args.min_tree_cover,
        max_nodata_frac=args.max_nodata_frac,
        progress=not args.no_progress,
    )
    if not patches:
        print("error: no patches extracted — is --patch-size larger than the rasters?",
              file=sys.stderr)
        return 1

    out_dir = args.out or load_paths().get_path("patches_dir", "./patches")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patches_df = pd.DataFrame([p.as_row() for p in patches])
    patches_df.to_csv(out_dir / "patches.csv", index=False)
    (out_dir / "region_vrts.json").write_text(
        json.dumps(region_vrt_map(observations), indent=2), encoding="utf-8"
    )
    (out_dir / "splits.json").write_text(
        json.dumps(split_map(observations), indent=2), encoding="utf-8"
    )

    summarise(patches_df)
    print(f"\nWritten to {out_dir}:")
    for name in ("patches.csv", "region_vrts.json", "splits.json"):
        print(f"  {name}")
    print(f"\nNext: python scripts/04_train.py --patches-dir {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
