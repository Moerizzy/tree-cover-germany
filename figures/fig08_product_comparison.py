#!/usr/bin/env python3
"""Figure 8 — our map and its difference against each reference product.

Replicates ``plot_treecover_maps_multiscale.ipynb``, cell 18
(``all_products_1km_raster.png``).

A 2 × 2 grid: (a) our tree cover in absolute terms, (b)–(d) the difference
of each reference product against it.

.. note::
   The figure currently in the manuscript is a 1 × 4 row and labels its
   panels "Liu et al. 2023" and "Tree Cover Density 2023". That is an older
   rendering; the notebook is the current state and is what this script
   follows. Colour ramps and the ±20 pp difference range agree between the
   two — only the layout and the panel names differ.

Details that change what the reader sees, all taken from the notebook:

* **The difference scale is fixed at ±20 pp**, not derived from the data.
  A percentile-based range would rescale whenever the input changes and two
  versions of the figure would no longer be comparable.
* **Display cells are 0.02° square**, and the aspect is corrected with
  ``1/cos(latitude)``. This is a display binning, not the 1 km analysis
  grid — 0.02° is about 1.4 km east–west and 2.2 km north–south at 51 °N.
  The 1 km grid of the manuscript is the tile grid itself, which the per-
  tile table already is.
* **Empty cells are white**, via ``set_bad`` on a masked array. A gap must
  not read as "no trees" on a green ramp or as "no difference" on a
  diverging one.

Usage::

    python figures/fig08_product_comparison.py --tiles tiles_with_treecover.csv
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from treecover.comparison import OUR_COLUMN, aggregate_to_grid, rasterise

#: Panel order: ours first, then the three comparison products.
PANEL_PRODUCTS = [
    (OUR_COLUMN, "Ours"),
    ("meta_chm_treecover_pct", "CHMv2"),
    ("treesense_chm3m_treecover_pct", "Planet CHM"),
    ("clms_tcd2023_treecover_pct", "TCD"),
]

#: Display cell size in degrees, square as in the notebook.
DISPLAY_DEG = 0.02

#: Fixed difference range, percentage points.
DIFF_LIMIT_PP = 20.0

PANEL_LETTERS = "abcdefgh"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", type=Path, required=True,
                   help="Per-tile CSV with lon_c, lat_c and one column per product.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--name", default="fig08_product_comparison")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.tiles.exists():
        print(f"error: {args.tiles} not found", file=sys.stderr)
        return 2

    frame = pd.read_csv(args.tiles)
    for column in ("lon_c", "lat_c"):
        if column not in frame.columns:
            print(f"error: {args.tiles.name} has no {column!r} column", file=sys.stderr)
            return 2
    if "tile_area_km2" not in frame.columns:
        frame["tile_area_km2"] = 1.0

    panels = [(c, label) for c, label in PANEL_PRODUCTS if c in frame.columns]
    if not panels or panels[0][0] != OUR_COLUMN:
        print(f"error: {OUR_COLUMN!r} must be present", file=sys.stderr)
        return 2

    grid = aggregate_to_grid(frame, [c for c, _ in panels],
                             dlon=DISPLAY_DEG, dlat=DISPLAY_DEG)
    rasters = {c: np.ma.masked_invalid(rasterise(grid, c)) for c, _ in panels}

    greens = copy.copy(plt.cm.Greens)
    greens.set_bad("white")
    diverging = copy.copy(plt.cm.RdBu_r)
    diverging.set_bad("white")
    diff_norm = TwoSlopeNorm(vmin=-DIFF_LIMIT_PP, vcenter=0, vmax=DIFF_LIMIT_PP)

    # 1° of longitude is shorter than 1° of latitude; without this Germany
    # comes out squashed east to west.
    aspect = 1.0 / np.cos(np.deg2rad((grid.lat_edges[0] + grid.lat_edges[-1]) / 2))

    fig, axes = plt.subplots(2, 2, figsize=(9, 11),
                             gridspec_kw={"wspace": 0.01, "hspace": 0.4})
    flat = axes.flatten()

    print(f"Figure 8 — {len(frame):,} tiles -> "
          f"{len(grid.lon_edges) - 1}x{len(grid.lat_edges) - 1} display cells")

    for index, (ax, (column, label)) in enumerate(zip(flat, panels)):
        if column == OUR_COLUMN:
            ax.pcolormesh(grid.lon_edges, grid.lat_edges, rasters[column],
                          cmap=greens, vmin=0, vmax=100, shading="flat")
            ax.set_title(label, fontsize=11)
        else:
            difference = np.ma.masked_invalid(rasters[column] - rasters[OUR_COLUMN])
            ax.pcolormesh(grid.lon_edges, grid.lat_edges, difference,
                          cmap=diverging, norm=diff_norm, shading="flat")
            ax.set_title(label, fontsize=12)
            print(f"  {label:<12} median {np.ma.median(difference):+6.2f} pp "
                  f"over {int(difference.count()):,} cells")
        ax.set_facecolor("white")
        ax.set_aspect(aspect)
        ax.set_axis_off()
        ax.text(0.02, 0.98, f"({PANEL_LETTERS[index]})", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")

    for ax in flat[len(panels):]:
        ax.set_visible(False)

    fig.subplots_adjust(bottom=0.12)
    absolute = plt.cm.ScalarMappable(cmap=greens, norm=plt.Normalize(0, 100))
    absolute.set_array([])
    relative = plt.cm.ScalarMappable(cmap=diverging, norm=diff_norm)
    relative.set_array([])

    bar_absolute = fig.colorbar(absolute, ax=flat[0], orientation="horizontal",
                                fraction=0.05, pad=0.04, shrink=0.9)
    bar_relative = fig.colorbar(relative, ax=flat[1:len(panels)].tolist(),
                                orientation="horizontal", fraction=0.05,
                                pad=0.04, shrink=0.6)
    bar_absolute.set_label("Tree cover (%)", fontsize=9)
    bar_relative.set_label("Difference vs. Ours (pp)", fontsize=9)

    out_dir = args.out_dir or Path("figures/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{args.name}.png"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / f"{args.name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
