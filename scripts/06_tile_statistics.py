#!/usr/bin/env python3
"""Measure tree cover per tile, masked to land and state borders.

Produces the table every downstream figure and table reads: one row per
tile with its land area, tree-covered area and percentage, plus a per-state
summary.

Masking matters for the totals. Without it, tiles straddling a state border
are counted twice and coastal tiles count open water as treeless land —
both inflate the denominator. The paper's 115,202 km² of tree cover on
356,381 km² of mapped land — 32.33 % — are computed with masking on.

**Tree cover only — this reads the prediction rasters and nothing else.**
The released ``tile_statistics_all.csv`` did not come from here: its
``forest_ha``/``tof_ha`` columns give it away as output of the
trees-outside-forests run, which needs a ``_tof.tif`` beside every
prediction. 362 sound Schleswig-Holstein tiles have no such companion and
so never reached the table, silently. Requiring only the prediction is the
point of this script.

Tiles that cannot be read are counted and written to
``tile_statistics_failed.csv`` rather than dropped, so the archive count
and the table always reconcile.

Examples::

    python scripts/06_tile_statistics.py --all
    python scripts/06_tile_statistics.py --states BY SN --workers 8
    python scripts/06_tile_statistics.py --limit 20 --no-land-mask   # quick look
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd

from treecover.config import load_paths
from treecover.statistics import StatsJob, run, summarise_by_state


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    src = p.add_argument_group("input selection")
    src.add_argument("--predictions-root", type=Path, default=None,
                     help="Override predictions_root from paths.yaml.")
    src.add_argument("--states", nargs="*", default=None, metavar="CODE")
    src.add_argument("--years", nargs="*", default=None, metavar="YYYY")
    src.add_argument("--all", action="store_true",
                     help="Process everything matched (the default with no filter).")
    src.add_argument("--limit", type=int, default=None, help="Stop after N tiles.")

    mask = p.add_argument_group("masking")
    mask.add_argument("--no-land-mask", action="store_true",
                      help="Skip land/border masking. Faster, but the totals then "
                           "double-count border tiles and count sea as land.")
    mask.add_argument("--land-mask", type=Path, default=None, help="Override masks.land.")
    mask.add_argument("--gadm", type=Path, default=None, help="Override masks.gadm.")

    run_g = p.add_argument_group("execution")
    run_g.add_argument("--pixel-size", type=float, default=0.20,
                       help="Ground resolution in metres.")
    run_g.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    run_g.add_argument("--allow-unknown-codes", action="store_true",
                       help="Do not fail on rasters holding codes above 1. Those come "
                            "from the trees-outside-forests classification, which is "
                            "not part of this pipeline.")
    run_g.add_argument("--out", type=Path, default=None,
                       help="Output directory (default: <results_root>/statistics).")
    run_g.add_argument("--no-progress", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_paths()

    predictions_root = args.predictions_root or paths.get_path("predictions_root")

    if args.no_land_mask:
        land_mask_path = gadm_path = None
    else:
        land_mask_path = args.land_mask or paths.get_path("masks.land")
        gadm_path = args.gadm or paths.get_path("masks.gadm")
        for label, path in (("masks.land", land_mask_path), ("masks.gadm", gadm_path)):
            if not path.exists():
                print(f"error: {label} not found: {path}", file=sys.stderr)
                print("       Fix paths.yaml, or pass --no-land-mask and treat the "
                      "totals as indicative only.", file=sys.stderr)
                return 2

    job = StatsJob(
        land_mask_path=land_mask_path,
        gadm_path=gadm_path,
        pixel_size_m=args.pixel_size,
        strict_codes=not args.allow_unknown_codes,
    )

    print(f"Predictions : {predictions_root}")
    print(f"Masking     : {'off' if args.no_land_mask else 'land + state borders'}")
    print(f"Workers     : {args.workers}")

    results = run(
        predictions_root, job,
        states=args.states, years=args.years,
        workers=args.workers, limit=args.limit, progress=not args.no_progress,
    )
    if not results:
        print("No prediction tiles matched.", file=sys.stderr)
        return 1

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    print(f"\n{len(ok)}/{len(results)} tiles measured.")

    out_dir = args.out or (paths.get_path("results_root", "./results") / "statistics")
    out_dir.mkdir(parents=True, exist_ok=True)

    # An unreadable tile is not a tile without trees. Dropping it silently
    # shrinks the reference area and nothing in the output says so — that is
    # how 92 truncated rasters left the published table without a trace. Every
    # failure is written out, so archive minus table is always nameable.
    failed_csv = out_dir / "tile_statistics_failed.csv"
    if failed:
        pd.DataFrame(
            [{"state": r.state, "year": r.year, "tile_name": r.tile_name,
              "error": r.error} for r in failed]
        ).to_csv(failed_csv, index=False)
        print(f"{len(failed)} unreadable, listed in {failed_csv}", file=sys.stderr)
        for r in failed[:5]:
            print(f"  {r.state}/{r.year}/{r.tile_name}: {r.error}", file=sys.stderr)
    elif failed_csv.exists():
        failed_csv.unlink()  # a clean run must not leave a stale list behind

    if not ok:
        return 1

    rows = [r.as_row() for r in ok]
    tiles_csv = out_dir / "tile_statistics.csv"
    pd.DataFrame(rows).to_csv(tiles_csv, index=False)

    per_state = summarise_by_state(rows)
    state_df = (
        pd.DataFrame(per_state).T.reset_index().rename(columns={"index": "state"})
        .sort_values("state")
    )
    state_df.to_csv(out_dir / "state_statistics.csv", index=False)

    print(f"\n{'state':<6} {'tiles':>8} {'land km²':>12} {'tree km²':>12} {'cover %':>9}")
    for _, row in state_df.iterrows():
        print(f"{row['state']:<6} {int(row['tiles']):>8} {row['land_area_km2']:>12,.0f} "
              f"{row['tree_area_km2']:>12,.0f} {row['tree_cover_pct']:>9.2f}")

    land = state_df["land_area_km2"].sum()
    tree = state_df["tree_area_km2"].sum()
    print(f"{'TOTAL':<6} {int(state_df['tiles'].sum()):>8} {land:>12,.0f} "
          f"{tree:>12,.0f} {100 * tree / land if land else 0:>9.2f}")
    print(f"\nWritten to {out_dir}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
