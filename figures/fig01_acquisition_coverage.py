#!/usr/bin/env python3
"""Figure 1 — spatial distribution of acquisition dates across Germany.

(a) Coverage by acquisition month, coloured by phenological season.
(b) Coverage by acquisition year.

Both are categorical maps, so each cell takes the **mode** of the tiles
inside it, never a mean — averaging August and February into May would say
nothing. Month colours are grouped by season with a distinct hue per
season and a lightness step within it, so the seasonal pattern reads at a
glance while individual months stay separable.

The map is built from the same tile selection as the merged raster
(newest acquisition date wins per 1 km cell), so the two cannot disagree
about which acquisition covers a given place.

Usage::

    python figures/fig01_acquisition_coverage.py --predictions-root /tf/Germany
    python figures/fig01_acquisition_coverage.py --coverage coverage.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from treecover.comparison import GRID_DLAT, GRID_DLON
from treecover.config import load_paths
from treecover.coverage import build_coverage, rasterise_mode
from treecover.figures.style import INK, INK_SECONDARY, SIZES, apply_style, save

logger = logging.getLogger(__name__)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

#: One hue per season, stepped in lightness within it. Winter and summer
#: sit at opposite ends of the palette so the dominant summer bias in the
#: survey is immediately visible.
MONTH_COLORS = {
    12: "#08306b", 1: "#2171b5", 2: "#6baed6",     # winter — blues
    3: "#00441b", 4: "#238b45", 5: "#74c476",      # spring — greens
    6: "#8c2d04", 7: "#d94801", 8: "#fd8d3c",      # summer — oranges
    9: "#4a1486", 10: "#6a51a3", 11: "#9e9ac8",    # autumn — purples
}

#: Sequential ramp for years — a year is ordinal, unlike a month.
YEAR_CMAP = "viridis"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--predictions-root", type=Path, default=None,
                        help="Scan the archive (slow: ~380,000 filenames).")
    source.add_argument("--coverage", type=Path, default=None,
                        help="A coverage CSV written by an earlier run.")
    p.add_argument("--states", nargs="*", default=None)
    p.add_argument("--save-coverage", type=Path, default=None,
                   help="Write the coverage table so later runs can skip the scan.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--name", default="fig01_acquisition_coverage")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def draw_month_panel(ax, frame: pd.DataFrame, extent_kwargs) -> list[Patch]:
    """Coverage by acquisition month, with a season-grouped legend."""
    image, lon_edges, lat_edges = rasterise_mode(frame, "month", GRID_DLON, GRID_DLAT)

    months = sorted(frame["month"].unique())
    colours = ListedColormap([MONTH_COLORS[m] for m in months])
    # Boundaries at half-integers so each month gets exactly one colour.
    norm = BoundaryNorm([m - 0.5 for m in months] + [months[-1] + 0.5], colours.N)

    ax.imshow(image, origin="lower",
              extent=(lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]),
              cmap=colours, norm=norm, interpolation="nearest", **extent_kwargs)

    counts = frame["month"].value_counts()
    return [
        Patch(facecolor=MONTH_COLORS[m], edgecolor="none",
              label=f"{MONTH_NAMES[m - 1]} ({_share(counts[m], len(frame))})")
        for m in months
    ]


def _share(count: int, total: int) -> str:
    """Format a share honestly.

    Rounding to whole percent prints "0 %" for the leaf-off months, which
    have thousands of tiles each — and those months are precisely the ones
    the paper is about. Small shares therefore get a decimal, and anything
    below 0.05 % is shown as "< 0.1 %" rather than as zero.
    """
    share = 100.0 * count / total
    if share >= 10:
        return f"{share:.0f} %"
    if share >= 0.1:
        return f"{share:.1f} %"
    return "< 0.1 %"


def draw_year_panel(ax, frame: pd.DataFrame, extent_kwargs):
    """Coverage by acquisition year."""
    image, lon_edges, lat_edges = rasterise_mode(frame, "year", GRID_DLON, GRID_DLAT)
    years = sorted(frame["year"].unique())
    norm = BoundaryNorm([y - 0.5 for y in years] + [years[-1] + 0.5],
                        len(years), extend="neither")
    cmap = plt.get_cmap(YEAR_CMAP, len(years))

    handle = ax.imshow(
        image, origin="lower",
        extent=(lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]),
        cmap=cmap, norm=norm, interpolation="nearest", **extent_kwargs,
    )
    return handle, years


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    if args.coverage:
        if not args.coverage.exists():
            print(f"error: {args.coverage} not found", file=sys.stderr)
            return 2
        frame = pd.read_csv(args.coverage)
    else:
        root = args.predictions_root or load_paths().get_path("predictions_root")
        print(f"Scanning {root} — this reads ~380,000 filenames, not the rasters.")
        frame = build_coverage(root, states=args.states)

    if frame.empty:
        print("error: no dated tiles found", file=sys.stderr)
        return 1

    if args.save_coverage:
        args.save_coverage.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.save_coverage, index=False)
        print(f"Coverage table -> {args.save_coverage}")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(SIZES.double, 4.4))
    # 1° lon is shorter than 1° lat at ~51 °N; without this Germany is squashed.
    extent_kwargs = {"aspect": 1.0 / np.cos(np.deg2rad(float(frame["lat_c"].mean())))}

    month_handles = draw_month_panel(axes[0], frame, extent_kwargs)
    axes[0].set_title("(a) Acquisition month", loc="left", color=INK)
    _strip(axes[0])
    axes[0].legend(handles=month_handles, loc="upper left", bbox_to_anchor=(1.0, 1.0),
                   fontsize=5.5, labelcolor=INK_SECONDARY, handlelength=1.0,
                   handleheight=1.0, borderpad=0.2, labelspacing=0.25)

    year_handle, years = draw_year_panel(axes[1], frame, extent_kwargs)
    axes[1].set_title("(b) Acquisition year", loc="left", color=INK)
    _strip(axes[1])
    bar = fig.colorbar(year_handle, ax=axes[1], orientation="vertical",
                       fraction=0.04, pad=0.02, ticks=years)
    bar.ax.set_yticklabels([str(y) for y in years], fontsize=6)

    paths = save(fig, args.name, args.out_dir)

    print(f"\n{len(frame):,} tiles, {frame['state'].nunique()} states, "
          f"{frame['date'].min()}–{frame['date'].max()}")
    print("\nBy season:")
    for season, count in frame["season"].value_counts().items():
        print(f"  {season:<8} {count:>8,}  ({100 * count / len(frame):4.1f} %)")
    print("\nBy year:")
    for year, count in frame["year"].value_counts().sort_index().items():
        print(f"  {year}     {count:>8,}  ({100 * count / len(frame):4.1f} %)")
    print(f"\n-> {', '.join(str(p) for p in paths)}")
    return 0


def _strip(ax) -> None:
    """Maps carry no axes furniture — the legend is the only scale needed."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


if __name__ == "__main__":
    raise SystemExit(main())
