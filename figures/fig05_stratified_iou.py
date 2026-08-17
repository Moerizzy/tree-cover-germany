#!/usr/bin/env python3
"""Figure 5 — IoU distribution by month, reference tree cover and land cover.

Replicates ``09_validation_results_summary_new.ipynb``, cell 11.

Three details carry meaning and are reproduced rather than reinterpreted:

* **Twelve months, not four seasons.** The panel shows every acquisition
  month separately; grouping them into seasons would hide that the survey
  is dominated by a handful of months.
* **``inner="box"``** draws a miniature box plot inside each violin — the
  median, the interquartile range and 1.5× IQR whiskers. That is what the
  published caption describes, and it is why the violins are not bare.
* **The mean is a white line**, drawn over the violin. Mean and median
  differ where the distribution is skewed, which is the interesting case;
  showing only one would hide it.

Panel widths are proportional to their category count, so a violin is the
same width in all three panels and the eye is not misled into reading the
twelve-category panel as more tightly distributed.

Usage::

    python figures/fig05_stratified_iou.py --metrics-dir publication/validation
"""

from __future__ import annotations

import argparse
import calendar
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from treecover.figures.validation_data import load_validation_metrics

VIOLIN_WIDTH = 0.7
#: Panel width per category, inches — equal violin width across panels.
PER_CAT_INCHES = 0.7
MEAN_COLOR = "white"
MEAN_LW = 2.4

METRIC = "overall_iou"

#: Colourblind-safe month colours, shared with the acquisition-date figure.
MONTH_COLORS = {
    12: "#4C72B0", 1: "#8AB4DC", 2: "#C6D9EE",
    3: "#B8E0C2", 4: "#5FBF7D", 5: "#117733",
    6: "#F0E442", 7: "#E69F00", 8: "#B4560A",
    9: "#D55E00", 10: "#AA3377", 11: "#661100",
}
MONTH_NUMBER = {name: i for i, name in enumerate(calendar.month_name) if name}

LAND_PALETTE = {"Urban": "#BDBDBD", "Non-urban": "#5D9B59"}

#: TCD bins arrive as "B1: 0-10%"; the prefix is an internal ordering key.
_TCD_PREFIX = re.compile(r"^\s*B\d+\s*:\s*")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-dir", type=Path, required=True)
    p.add_argument("--exclude-zero-lidar", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--name", default="fig05_stratified_iou")
    return p


def clean_tcd_label(label) -> str:
    return _TCD_PREFIX.sub("", str(label))


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="left")


def violin(ax, data, column: str, order: list, palette) -> None:
    """One panel: violins with an inner box, plus an explicit mean line."""
    sns.violinplot(
        data=data, x=column, y=METRIC, order=order, ax=ax,
        width=VIOLIN_WIDTH, palette=palette, hue=column, hue_order=order,
        dodge=False, legend=False, inner="box", linewidth=1,
        cut=0, density_norm="width",
    )
    means = data.groupby(column, observed=True)[METRIC].mean()
    half = VIOLIN_WIDTH / 2
    for position, category in enumerate(order):
        if category in means.index:
            ax.hlines(means.loc[category], position - half, position + half,
                      color=MEAN_COLOR, linewidth=MEAN_LW, zorder=5)
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("IoU")


def collect_panels(frame):
    """Panels present in the data, each with its category order and palette."""
    panels = []
    if "month_bin" in frame.columns:
        present = set(frame["month_bin"].dropna().unique())
        order = [m for m in list(calendar.month_name)[1:] if m in present]
        if order:
            panels.append(("month", order,
                           {m: MONTH_COLORS[MONTH_NUMBER[m]] for m in order}))
    if "tcd_bin" in frame.columns:
        order = sorted(frame["tcd_bin"].dropna().unique())
        if order:
            cmap = plt.cm.YlGn
            panels.append(("tcd", order,
                           {label: cmap(0.25 + 0.75 * i / max(len(order) - 1, 1))
                            for i, label in enumerate(order)}))
    if "is_urban" in frame.columns:
        panels.append(("urban", ["Urban", "Non-urban"], LAND_PALETTE))
    return panels


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frame = load_validation_metrics(
            args.metrics_dir, exclude_zero_lidar=args.exclude_zero_lidar
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if METRIC not in frame.columns:
        print(f"error: no {METRIC!r} column. Available: {list(frame.columns)}",
              file=sys.stderr)
        return 2

    panels = collect_panels(frame)
    if not panels:
        print("error: no stratification columns available", file=sys.stderr)
        return 2

    counts = [len(order) for _, order, _ in panels]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(sum(counts) * PER_CAT_INCHES + 2.0, 5),
        gridspec_kw={"width_ratios": counts},
    )
    axes = np.atleast_1d(axes)

    for ax, (key, order, palette), label in zip(axes, panels, ["(a)", "(b)", "(c)"]):
        if key == "month":
            violin(ax, frame[frame["month_bin"].isin(order)], "month_bin", order, palette)
            ax.set_title("Month")
            ax.set_xlabel("Month")
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([m[:3] for m in order], rotation=0)
        elif key == "tcd":
            violin(ax, frame[frame["tcd_bin"].isin(order)], "tcd_bin", order, palette)
            ax.set_title("Tree Cover")
            ax.set_xlabel("Reference tree cover [%]")
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([clean_tcd_label(v) for v in order],
                               rotation=45, ha="right")
        else:
            land = frame.assign(
                land_type=frame["is_urban"].map({True: "Urban", False: "Non-urban"})
            )
            violin(ax, land, "land_type", order, palette)
            ax.set_title("Landcover")
            ax.set_xlabel("")
        add_panel_label(ax, label)

    # No legend, matching the published figure. The notebook builds a
    # ``Line2D`` handle for the mean and then never passes it to a legend
    # call, so the white mean line is unlabelled in the manuscript. Adding
    # one here would not help either: a white line on a white legend patch
    # is invisible. If the mean should be labelled, the caption is the place.
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    out_dir = args.out_dir or Path("figures/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{args.name}.png"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / f"{args.name}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Figure 5 — {len(frame):,} patches, mean IoU {frame[METRIC].mean():.4f}")
    for key, order, _ in panels:
        column = {"month": "month_bin", "tcd": "tcd_bin", "urban": "is_urban"}[key]
        if column == "is_urban":
            continue
        stats = frame.groupby(column, observed=True)[METRIC].agg(["mean", "size"])
        print(f"\n  {key}:")
        for category in order:
            if category in stats.index:
                row = stats.loc[category]
                print(f"    {clean_tcd_label(category):<14} "
                      f"mean {row['mean']:.3f}  n={int(row['size']):,}")
    print(f"\n-> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
