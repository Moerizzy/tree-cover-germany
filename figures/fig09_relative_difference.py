#!/usr/bin/env python3
"""Figure 9 — relative difference per product, split by our tree cover density.

Replicates ``plot_treecover_maps_multiscale.ipynb``, cell 21
(``rel_diff_distribution_per_tcd_bin_1km_tiles.png``).

Four panels, one per tree cover bin. Within each, the distribution of
``(product − ours) / ours × 100 %`` across 1 km tiles, drawn as a **curve
over 1 % bins** rather than as a box or violin — the shape of the tail is
the argument, and a summary statistic would hide it.

Three conventions, all from the notebook, each of which changes the figure:

* Tiles where our cover is **0 % are excluded**: the relative difference is
  undefined there.
* All three products are drawn over the **same tiles**. The mask requires
  every product to be valid, so the curves are directly comparable rather
  than each being over its own subset.
* The y-axis is the **share of tiles per 1 % bin**, normalised within each
  panel. Panels hold very different tile counts, and raw counts would make
  the largest bin look like the widest distribution.

Usage::

    python figures/fig09_relative_difference.py --tiles tiles_with_treecover.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from treecover.comparison import OUR_COLUMN

#: Comparison products with their published labels and colours.
PRODUCTS = [
    ("meta_chm_treecover_pct", "CHMv2", "#3fa96b"),
    ("treesense_chm3m_treecover_pct", "Planet CHM", "#e07b4a"),
    ("clms_tcd2023_treecover_pct", "TCD", "#7a8ec6"),
]

TCD_EDGES = [0, 10, 25, 50, 100]
CLIP_PCT = 100.0
PANEL_LETTERS = "abcd"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--name", default="fig09_relative_difference")
    return p


def bin_labels() -> list[str]:
    """``[0, 10)%`` … ``[50, 100]%`` — the last bin closed, as in the notebook."""
    return [
        f"[{TCD_EDGES[i]}, {TCD_EDGES[i + 1]})%"
        if i < len(TCD_EDGES) - 2
        else f"[{TCD_EDGES[i]}, {TCD_EDGES[i + 1]}]%"
        for i in range(len(TCD_EDGES) - 1)
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.tiles.exists():
        print(f"error: {args.tiles} not found", file=sys.stderr)
        return 2

    frame = pd.read_csv(args.tiles)
    if OUR_COLUMN not in frame.columns:
        print(f"error: {OUR_COLUMN!r} missing", file=sys.stderr)
        return 2
    present = [(c, label, colour) for c, label, colour in PRODUCTS
               if c in frame.columns]
    if not present:
        print("error: no comparison product present", file=sys.stderr)
        return 2

    # One joint mask, so every curve is over the same tiles.
    mask = frame[OUR_COLUMN].notna() & (frame[OUR_COLUMN] > 0)
    for column, _, _ in present:
        mask &= frame[column].notna()
    subset = frame.loc[mask, [OUR_COLUMN] + [c for c, _, _ in present]].copy()
    print(f"Figure 9 — {len(subset):,} of {len(frame):,} tiles "
          "(ours > 0 and every product valid)")

    for column, _, _ in present:
        subset[f"_rd_{column}"] = (
            (subset[column] - subset[OUR_COLUMN]) / subset[OUR_COLUMN] * 100.0
        ).clip(-CLIP_PCT, CLIP_PCT)

    # right=False gives half-open bins; 100 % is put back into the last one.
    subset["_bin"] = pd.cut(subset[OUR_COLUMN], bins=TCD_EDGES, right=False,
                            include_lowest=True)
    subset.loc[subset[OUR_COLUMN] == 100, "_bin"] = subset["_bin"].cat.categories[-1]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    edges = np.arange(-100, 101, 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    for index, (label, ax) in enumerate(zip(bin_labels(), axes.flat)):
        interval = subset["_bin"].cat.categories[index]
        rows = subset[subset["_bin"] == interval]
        count = len(rows)

        for column, product_label, colour in present:
            values = rows[f"_rd_{column}"].to_numpy()
            histogram, _ = np.histogram(values, bins=edges)
            ax.plot(centres, histogram / max(count, 1) * 100.0,
                    color=colour, lw=1.4, label=product_label)

        ax.axvline(0, ls="--", color="black", lw=0.8)
        ax.set_title(f"Ours ∈ {label}  (n={count:,})", fontsize=11)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_xlim(-100, 100)
        ax.text(0.02, 0.96, f"({PANEL_LETTERS[index]})", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")

        medians = "  ".join(
            f"{lbl}: {np.median(rows[f'_rd_{c}']):+6.1f} %"
            for c, lbl, _ in present
        )
        print(f"  {label:<13} n={count:>8,}  {medians}")

    for ax in axes[:, 0]:
        ax.set_ylabel("Share per 1 % bin (%)")
    for ax in axes[-1, :]:
        ax.set_xlabel("rel. diff. vs. Ours (%)")

    handles = [plt.Line2D([0], [0], color=colour, lw=1.6, label=label)
               for _, label, colour in present]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=len(present), frameon=False, fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

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
