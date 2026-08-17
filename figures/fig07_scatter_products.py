#!/usr/bin/env python3
"""Figure 7 — tree cover per grid cell, ours against each reference product.

Replicates ``plot_treecover_maps_multiscale.ipynb``, cell 6, at the
``~1 km (tiles)`` scale (0.02°).

Two details are easy to get wrong and both change the number in the title:

* Points are **aggregated grid cells**, not raw tiles. The notebook runs
  ``aggregate_to_grid`` first, so roughly 370,000 tiles become about 120,000
  cells and the cloud is correspondingly tighter.
* **R² is the squared Pearson correlation**, not ``1 − SS_res/SS_tot``
  against the 1:1 line. The first asks how well the product tracks ours in
  shape, the second how well it matches in value. A product with a constant
  offset scores well on the first and badly on the second; the published
  figure reports the first.

Plain markers rather than a density plot, as in the notebook: ``s=4`` at
40 % opacity, which lets the overplotted core darken on its own.

Usage::

    python figures/fig07_scatter_products.py --tiles tiles_with_treecover.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from treecover.comparison import OUR_COLUMN, aggregate_to_grid

#: Comparison products, in the manuscript order.
COMPARISON_PRODUCTS = [
    ("meta_chm_treecover_pct", "CHMv2"),
    ("treesense_chm3m_treecover_pct", "Planet CHM"),
    ("clms_tcd2023_treecover_pct", "TCD"),
]

#: Grid cell size in degrees — the notebook's "~1 km (tiles)" scale. Square,
#: not 1 km exactly; 0.02° avoids the empty-bin stripes a finer grid produces.
SCALE_DEG = 0.02

POINT_COLOR = "steelblue"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", type=Path, required=True)
    p.add_argument("--scale-deg", type=float, default=SCALE_DEG)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--name", default="fig07_scatter_products")
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
    if OUR_COLUMN not in frame.columns:
        print(f"error: {OUR_COLUMN!r} missing", file=sys.stderr)
        return 2

    present = [(c, label) for c, label in COMPARISON_PRODUCTS if c in frame.columns]
    if not present:
        print("error: no comparison product present", file=sys.stderr)
        return 2

    grid = aggregate_to_grid(frame, [OUR_COLUMN] + [c for c, _ in present],
                             dlon=args.scale_deg, dlat=args.scale_deg)
    cells = grid.frame
    print(f"Figure 7 — {len(frame):,} tiles -> {len(cells):,} grid cells "
          f"at {args.scale_deg}°")

    fig, axes = plt.subplots(1, len(present), figsize=(4.5 * len(present), 4.5),
                             sharey=False)
    axes = np.atleast_1d(axes)

    for ax, (column, label) in zip(axes, present):
        subset = cells[[OUR_COLUMN, column]].dropna()
        x = subset[OUR_COLUMN].to_numpy()
        y = subset[column].to_numpy()

        # Squared Pearson correlation, as in the notebook.
        r2 = np.corrcoef(x, y)[0, 1] ** 2 if len(x) > 1 else float("nan")

        ax.scatter(x, y, s=4, alpha=0.4, linewidths=0, color=POINT_COLOR)
        ax.plot([0, 100], [0, 100], "k--", linewidth=1, label="1:1")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Ours: Tree Cover (%)", fontsize=10)
        ax.set_ylabel(f"{label} Tree Cover (%)", fontsize=10)
        ax.set_title(f"{label}\nR²={r2:.3f}", fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        print(f"  {label:<12} n={len(x):>8,}  R²={r2:.3f}  "
              f"bias={np.mean(y - x):+6.2f} pp")

    plt.tight_layout()
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
