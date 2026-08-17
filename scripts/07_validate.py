#!/usr/bin/env python3
"""Validate the map against LiDAR reference data.

Three subcommands, run in order. They are separate commands rather than one
because two hard boundaries sit between them:

``sample``
    Draw stratified validation locations and expand them to 25 m boxes.
    Runs in minutes and must run **once** — re-drawing changes the sample
    set and with it every number in the results.

``reference``
    Download LiDAR, build a CHM per box, threshold it into a binary tree
    mask. Hours to days, and state servers drop connections, so this is
    resumable and expected to be re-run until it completes.

    *Between* ``reference`` and ``score`` sits a manual step: open
    ``tree_mask_footprints.geojson`` in QGIS and set ``exclude = 1`` on
    boxes where the LiDAR and the orthophoto disagree because something
    changed between the two acquisitions. ``score`` honours that flag.

``score``
    Model vs. reference: IoU, F1, precision, recall, broken down by stratum.
    Minutes.

``summarise``
    The same tables from per-sample metrics already on disk, without the
    rasters. For rebuilding the paper's accuracy table from the published
    per-sample CSVs, where the predictions are terabytes away.

Examples::

    python scripts/07_validate.py sample --state BB --n-per-stratum 50
    python scripts/07_validate.py reference --state BB --workers 4
    python scripts/07_validate.py score --all-states
    python scripts/07_validate.py summarise --metrics-dir publication/validation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds as window_from_bounds

from treecover.config import load_paths, load_states, validation_states
from treecover.constants import NODATA
from treecover.io.tiles import find_prediction_tiles
from treecover.io.vector import read_vector
from treecover.validation import (
    aggregate,
    binary_metrics,
    by_stratum,
    chm_to_tree_mask,
    create_chm,
    make_boxes,
    resolve_sample_ids,
    score_zero_reference,
    stratified_sample,
)
from treecover.validation.lidar import load_points_for_bounds
from treecover.validation.metrics import reduce_prediction_to_reference

logger = logging.getLogger(__name__)

#: Stratification columns broken out in the per-stratum table.
STRATA = ("tcd_bin", "month_bin", "is_urban", "elev_bin", "landscape")

#: A box that was actually scored. ``score`` writes ``ok``; the published
#: per-sample files, written by the original notebook, write ``success``.
SCORED_STATUSES = ("ok", "success")

FOOTPRINTS_NAME = "tree_mask_footprints.geojson"


# ══════════════════════════════════════════════════════════════════════════════
# sample
# ══════════════════════════════════════════════════════════════════════════════


def cmd_sample(args, paths) -> int:
    """Draw stratified validation locations."""
    state = load_states().resolve(args.state)
    candidates = read_vector(args.candidates)

    strata = [c for c in args.strata if c in candidates.columns]
    missing = [c for c in args.strata if c not in candidates.columns]
    if missing:
        logger.warning("Not stratifying on absent column(s): %s", ", ".join(missing))
    if not strata:
        print(f"error: none of {args.strata} are in {args.candidates.name}. "
              f"Available: {list(candidates.columns)}", file=sys.stderr)
        return 2

    result = stratified_sample(
        candidates,
        strata_columns=strata,
        min_per_stratum=args.n_per_stratum,
        min_distance_m=args.min_distance,
        random_seed=args.seed,
    )
    if result.samples.empty:
        print("error: no samples selected — check the candidate pool.", file=sys.stderr)
        return 1

    boxes = make_boxes(result.samples, size_m=args.box_size)
    boxes["exclude"] = 0          # set to 1 in QGIS after visual inspection
    boxes["state"] = state

    out_dir = _state_dir(args, paths, state)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validation_samples.geojson"
    if out_path.exists() and not args.force:
        print(f"error: {out_path} already exists. Re-drawing would change every "
              "number that follows.\n       Pass --force if that is really intended.",
              file=sys.stderr)
        return 2

    _write_geojson(boxes, out_path)
    print(result.summary())
    print(f"\n{len(boxes)} boxes of {args.box_size:.0f} m -> {out_path}")
    print(f"Next: python scripts/07_validate.py reference --state {state}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# reference
# ══════════════════════════════════════════════════════════════════════════════


def cmd_reference(args, paths) -> int:
    """Download LiDAR and build a reference tree mask per box."""
    state_cfg = load_states()[args.state]
    state = state_cfg.code

    state_dir = _state_dir(args, paths, state)
    samples_path = args.samples or (state_dir / "validation_samples.geojson")
    if not samples_path.exists():
        print(f"error: no sample set at {samples_path}. Run the 'sample' step first.",
              file=sys.stderr)
        return 2

    samples = read_vector(samples_path)
    tile_index = read_vector(args.lidar_index) if args.lidar_index else None
    if tile_index is None and not state_cfg.has_lidar:
        print(f"error: {state} has no LiDAR URL template in states.yaml and no "
              "--lidar-index was given.", file=sys.stderr)
        return 2

    masks_dir = state_dir / paths.get_value("tree_mask_subdir", "tree_masks")
    cache_dir = args.cache or (state_dir / paths.get_value("lidar_cache_subdir",
                                                           "lidar_cache"))
    masks_dir.mkdir(parents=True, exist_ok=True)

    buildings = _load_buildings(args.buildings) if args.buildings else None
    point_cache: dict = {}
    records = []
    built = skipped = failed = 0

    iterator = samples.iterrows()
    if not args.no_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, total=len(samples), unit="box")

    for _, row in iterator:
        sample_id = int(row.get("sample_id", _))
        mask_path = masks_dir / f"tree_mask_sample_{sample_id:04d}.tif"
        if mask_path.exists() and not args.force:
            skipped += 1
            records.append(_footprint_record(row, sample_id, mask_path, "cached"))
            continue

        urls = _lidar_urls(row, state_cfg, tile_index)
        if not urls:
            failed += 1
            records.append(_footprint_record(row, sample_id, None, "no_lidar_tile"))
            continue

        bounds = row.geometry.bounds
        points = load_points_for_bounds(
            urls, bounds, cache_dir, buffer_m=args.buffer, cache=point_cache
        )
        if not len(points):
            failed += 1
            records.append(_footprint_record(row, sample_id, None, "no_points"))
            continue

        chm, transform = create_chm(
            points, bounds, resolution=args.resolution, buffer=args.buffer
        )
        if chm is None:
            failed += 1
            records.append(_footprint_record(row, sample_id, None, "chm_failed"))
            continue

        tree, nodata = chm_to_tree_mask(
            chm,
            height_threshold=args.height_threshold,
            max_hole_m2=args.max_hole,
            resolution=args.resolution,
            buildings=_rasterise_buildings(buildings, transform, chm.shape)
            if buildings is not None else None,
        )

        out = np.where(nodata, NODATA, tree.astype(np.uint8))
        _write_mask(mask_path, out, transform, samples.crs)
        built += 1
        records.append(_footprint_record(row, sample_id, mask_path, "ok"))

    footprints = gpd.GeoDataFrame(records, geometry="geometry", crs=samples.crs)
    footprints_path = masks_dir / FOOTPRINTS_NAME
    _write_geojson(footprints, footprints_path)

    print(f"\n{built} mask(s) built, {skipped} already present, {failed} failed")
    if failed:
        print(footprints[footprints["status"] != "ok"]["status"]
              .value_counts().to_string())
    print(f"Footprints: {footprints_path}")
    print("\nBefore scoring: open the footprints in QGIS and set exclude = 1 on "
          "boxes where\nthe LiDAR and the orthophoto disagree because the ground "
          "changed between them.")
    return 0 if built or skipped else 1


def _lidar_urls(row, state_cfg, tile_index) -> list[str]:
    """URLs of the LiDAR tiles a box intersects."""
    if tile_index is not None:
        hits = tile_index[tile_index.intersects(row.geometry)]
        if "url" in hits.columns:
            return [u for u in hits["url"].tolist() if u]
        return [
            state_cfg.lidar_url(str(t))
            for t in hits.get("tile_name", hits.get("id", []))
            if state_cfg.lidar_url(str(t))
        ]

    # No index: derive the 1 km tile name from the box centroid.
    centre = row.geometry.centroid
    east_km = int(centre.x // 1000)
    north_km = int(centre.y // 1000)
    tile_name = f"{state_cfg.utm_zone}{east_km:03d}_{north_km:04d}"
    url = state_cfg.lidar_url(tile_name)
    return [url] if url else []


def _footprint_record(row, sample_id: int, mask_path: Path | None, status: str) -> dict:
    """One row of the footprint table, carrying the strata through."""
    record = {
        "sample_id": sample_id,
        "geometry": row.geometry,
        "status": status,
        "mask_path": str(mask_path) if mask_path else None,
        "exclude": int(row.get("exclude", 0) or 0),
    }
    for column in STRATA + ("state", "month", "tcd", "elevation"):
        if column in row.index:
            record[column] = row[column]
    return record


def _load_buildings(path: Path):
    """Building footprints used to subtract roofs from the reference."""
    gdf = read_vector(path)
    logger.info("Loaded %d building footprint(s)", len(gdf))
    return gdf


def _rasterise_buildings(buildings, transform, shape):
    """Burn building polygons onto the CHM grid."""
    from rasterio.features import geometry_mask

    if buildings is None or buildings.empty:
        return None
    return geometry_mask(
        buildings.geometry, out_shape=shape, transform=transform, invert=True
    )


def _write_mask(path: Path, data: np.ndarray, transform, crs) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype="uint8", crs=crs, transform=transform, nodata=NODATA,
        compress="lzw",
    ) as dst:
        dst.write(data.astype(np.uint8), 1)


# ══════════════════════════════════════════════════════════════════════════════
# score
# ══════════════════════════════════════════════════════════════════════════════


def cmd_score(args, paths) -> int:
    """Compare predictions against the reference masks."""
    registry = load_states()
    states = list(validation_states()) if args.all_states else [registry.resolve(args.state)]

    frames = []
    for state in states:
        try:
            frames.append(_score_state(state, args, paths))
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    per_sample = pd.concat(frames, ignore_index=True)
    scored = per_sample[per_sample["status"] == "ok"].copy()
    if scored.empty:
        print("No boxes could be scored.", file=sys.stderr)
        print(per_sample["status"].value_counts().to_string(), file=sys.stderr)
        return 1

    if not args.keep_empty:
        empty = scored["reference_cover_pct"] == 0
        if empty.any():
            logger.info("Excluding %d box(es) with 0 %% reference cover "
                        "(--keep-empty to keep them)", int(empty.sum()))
            scored = scored[~empty]

    out_dir = args.out or (
        paths.get_path("results_root", "./results") / "validation"
        / ("all" if args.all_states else states[0])
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample.to_csv(out_dir / "validation_metrics_per_sample.csv", index=False)
    summary = aggregate(scored)
    summary.to_csv(out_dir / "validation_metrics_summary.csv", index=False)

    strata_frames = []
    for column in STRATA:
        if column in scored.columns and scored[column].notna().any():
            frame = by_stratum(scored, column)
            frame.insert(0, "stratum", column)
            strata_frames.append(frame.rename(columns={column: "level"}))
    if strata_frames:
        pd.concat(strata_frames, ignore_index=True).to_csv(
            out_dir / "validation_metrics_by_strata.csv", index=False
        )

    print(f"\nScored {len(scored)} of {len(per_sample)} boxes across {len(states)} state(s)")
    print(per_sample["status"].value_counts().to_string())
    print(f"\n{summary.to_string(index=False, float_format=lambda v: f'{v:.4f}')}")
    print(f"\nWritten to {out_dir}")
    return 0


def cmd_summarise(args, paths) -> int:
    """Rebuild the accuracy tables from per-sample metrics already on disk.

    ``score`` needs the predictions and the reference masks; this needs
    neither. The published release ships the per-sample metrics precisely
    so the paper's table can be checked without the terabytes behind it,
    and the aggregation is the same function either way.
    """
    directory = Path(args.metrics_dir)
    frames = []
    for path in sorted(directory.glob(args.pattern)):
        frame = pd.read_csv(path)
        # The published files prefix the per-box metrics with `overall_`;
        # aggregate() names them plainly.
        frame = frame.rename(columns={
            column: column[len("overall_"):] for column in frame.columns
            if column.startswith("overall_")
        })
        if "state" not in frame.columns:
            frame["state"] = path.stem.split("_")[-1]
        frames.append(frame)

    if not frames:
        print(f"error: no file matches {args.pattern} in {directory}",
              file=sys.stderr)
        return 2

    per_sample = pd.concat(frames, ignore_index=True)
    if "status" in per_sample.columns:
        # The published files say "success" where this pipeline says "ok".
        scored = per_sample["status"].isin(SCORED_STATUSES)
        logger.info("Scored boxes: %d of %d (%s)", int(scored.sum()),
                    len(per_sample),
                    per_sample.loc[~scored, "status"].value_counts().to_dict())
        per_sample = per_sample[scored]

    # A third of the boxes hold no reference tree at all, and what is done
    # with them decides the headline number — see score_zero_reference.
    if args.empty == "score":
        per_sample = score_zero_reference(per_sample, prefix="")
    elif args.empty == "drop" and "lidar_tree_cover_pct" in per_sample.columns:
        empty = per_sample["lidar_tree_cover_pct"] == 0
        logger.info("Dropping %d box(es) with 0 %% reference cover",
                    int(empty.sum()))
        per_sample = per_sample[~empty]

    if per_sample.empty:
        print("error: nothing left to summarise", file=sys.stderr)
        return 1

    summary = aggregate(per_sample)
    summary.insert(0, "state", "all")
    per_state = []
    for state, group in per_sample.groupby("state"):
        frame = aggregate(group)
        frame.insert(0, "state", state)
        per_state.append(frame)
    summary = pd.concat([summary] + per_state, ignore_index=True)

    out = Path(args.out) if args.out else directory / "metrics_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    strata_frames = []
    for column in STRATA:
        if column in per_sample.columns and per_sample[column].notna().any():
            frame = by_stratum(per_sample, column)
            frame.insert(0, "stratum", column)
            strata_frames.append(frame.rename(columns={column: "level"}))
    if strata_frames:
        strata_path = out.with_name(out.stem + "_by_strata.csv")
        pd.concat(strata_frames, ignore_index=True).to_csv(strata_path, index=False)

    print(f"\n{len(per_sample)} boxes across "
          f"{per_sample['state'].nunique()} state(s)\n")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWritten to {out}")
    return 0


def _score_state(state: str, args, paths) -> pd.DataFrame:
    masks_dir = args.masks_dir or (
        _state_dir(args, paths, state) / paths.get_value("tree_mask_subdir", "tree_masks")
    )
    footprints_path = args.footprints or (masks_dir / FOOTPRINTS_NAME)
    if not footprints_path.exists():
        raise FileNotFoundError(
            f"No footprints for {state} at {footprints_path}. "
            "Run the 'reference' step first."
        )

    footprints = read_vector(footprints_path)
    # NRW records no sample_id; there the row position is the id.
    footprints = footprints.assign(sample_id=resolve_sample_ids(footprints))
    if "exclude" in footprints.columns:
        keep = ~footprints["exclude"].fillna(0).astype(bool)
        if (~keep).any():
            logger.info("Dropping %d box(es) flagged exclude=1", int((~keep).sum()))
        footprints = footprints[keep]

    predictions_root = args.predictions_root or paths.get_path("predictions_root")
    tiles = list(find_prediction_tiles(predictions_root,
                                       states=[load_states()[state].pred_dir]))
    logger.info("%s: %d box(es), %d prediction tile(s)", state, len(footprints), len(tiles))

    factor = int(round(args.reference_res / args.model_res))
    rows = []

    for _, row in footprints.iterrows():
        sample_id = int(row["sample_id"])
        mask_path = masks_dir / f"tree_mask_sample_{sample_id:04d}.tif"
        if not mask_path.exists():
            rows.append({"sample_id": sample_id, "state": state, "status": "no_reference"})
            continue

        with rasterio.open(mask_path) as src:
            reference_raw = src.read(1)
            ref_crs = src.crs

        valid = reference_raw != NODATA
        reference = (reference_raw == 1) & valid

        prediction = _extract_prediction(row.geometry, tiles, ref_crs)
        if prediction is None:
            rows.append({"sample_id": sample_id, "state": state, "status": "no_prediction"})
            continue

        predicted = reduce_prediction_to_reference(
            prediction, reference.shape, factor
        )
        metrics = binary_metrics(reference, predicted, valid)

        record = {"sample_id": sample_id, "state": state, "status": "ok",
                  **metrics.as_dict()}
        for column in STRATA:
            if column in footprints.columns:
                record[column] = row[column]
        rows.append(record)

    return pd.DataFrame(rows)


def _extract_prediction(geometry, tiles, target_crs) -> np.ndarray | None:
    """Read the prediction covering a box, merging across tile edges."""
    bounds = geometry.bounds
    pieces = []
    for tile in tiles:
        with rasterio.open(tile.path) as src:
            if src.crs != target_crs or not _overlaps(src.bounds, bounds):
                continue
            window = window_from_bounds(*bounds, transform=src.transform)
            pieces.append(src.read(1, window=window, boundless=True, fill_value=NODATA))

    if not pieces:
        return None
    merged = pieces[0].copy()
    for data in pieces[1:]:
        if data.shape == merged.shape:
            gap = merged == NODATA
            merged[gap] = data[gap]
    return merged


def _overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


# ══════════════════════════════════════════════════════════════════════════════
# plumbing
# ══════════════════════════════════════════════════════════════════════════════


def _state_dir(args, paths, state: str) -> Path:
    return Path(getattr(args, "state_dir", None) or paths.get_path("validation_root")) / state


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Write GeoJSON, coercing types the driver cannot represent."""
    out = gdf.copy()
    for column in out.columns:
        if column == "geometry":
            continue
        if str(out[column].dtype) in ("category", "object"):
            out[column] = out[column].astype(str).replace({"nan": None, "None": None})
    out.to_file(path, driver="GeoJSON")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--state-dir", type=Path, default=None,
                   help="Override validation_root from paths.yaml.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="Draw stratified validation locations.")
    s.add_argument("--state", required=True)
    s.add_argument("--candidates", type=Path, required=True,
                   help="Candidate points with the stratification columns.")
    s.add_argument("--strata", nargs="*", default=list(STRATA))
    s.add_argument("--n-per-stratum", type=int, default=50)
    s.add_argument("--min-distance", type=float, default=500.0,
                   help="Minimum separation in CRS units, so two boxes never share "
                        "an orthophoto.")
    s.add_argument("--box-size", type=float, default=25.0)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--force", action="store_true",
                   help="Overwrite an existing sample set. This changes every "
                        "number that follows — be sure.")
    s.set_defaults(func=cmd_sample)

    r = sub.add_parser("reference", help="LiDAR -> CHM -> reference tree mask.")
    r.add_argument("--state", required=True)
    r.add_argument("--samples", type=Path, default=None)
    r.add_argument("--lidar-index", type=Path, default=None,
                   help="Tile index with LiDAR URLs. Without it, tile names are "
                        "derived from the box centroid.")
    r.add_argument("--buildings", type=Path, default=None,
                   help="Building footprints. LiDAR cannot tell a roof from a "
                        "canopy by height alone.")
    r.add_argument("--cache", type=Path, default=None, help="LiDAR download cache.")
    r.add_argument("--resolution", type=float, default=1.0, help="CHM resolution, m.")
    r.add_argument("--buffer", type=float, default=50.0,
                   help="Margin in metres for the DTM interpolation.")
    r.add_argument("--height-threshold", type=float, default=3.0)
    r.add_argument("--max-hole", type=float, default=10.0,
                   help="Close holes up to this area, m². 0 disables.")
    r.add_argument("--force", action="store_true", help="Rebuild existing masks.")
    r.add_argument("--no-progress", action="store_true")
    r.set_defaults(func=cmd_reference)

    c = sub.add_parser("score", help="Model vs. reference metrics.")
    sel = c.add_mutually_exclusive_group(required=True)
    sel.add_argument("--state")
    sel.add_argument("--all-states", action="store_true")
    c.add_argument("--masks-dir", type=Path, default=None)
    c.add_argument("--footprints", type=Path, default=None)
    c.add_argument("--predictions-root", type=Path, default=None)
    c.add_argument("--out", type=Path, default=None)
    c.add_argument("--reference-res", type=float, default=1.0)
    c.add_argument("--model-res", type=float, default=0.20)
    c.add_argument("--keep-empty", action="store_true",
                   help="Keep boxes with 0 %% reference cover. They inflate the mean "
                        "IoU, since a correct empty box scores 1.0.")
    c.set_defaults(func=cmd_score)

    s = sub.add_parser("summarise", help="Aggregate per-sample metrics on disk.")
    s.add_argument("--metrics-dir", type=Path, required=True,
                   help="Directory of per-sample CSVs, one per state.")
    s.add_argument("--pattern", default="metrics_per_sample_*.csv",
                   help="Which files to read.")
    s.add_argument("--out", type=Path, default=None,
                   help="Output CSV (default: <metrics-dir>/metrics_summary.csv).")
    s.add_argument("--empty", choices=("score", "drop", "keep"), default="score",
                   help="What to do with boxes holding no reference tree. 'score' "
                        "resolves them as correct negatives or false positives and "
                        "is what the published numbers do; 'drop' measures the model "
                        "only where trees are; 'keep' takes the file's own values, "
                        "where an undefined IoU reads as 0. Default: %(default)s.")
    s.set_defaults(func=cmd_summarise)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )
    return args.func(args, load_paths())


if __name__ == "__main__":
    raise SystemExit(main())
