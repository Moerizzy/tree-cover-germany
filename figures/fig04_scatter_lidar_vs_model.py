#!/usr/bin/env python3
"""Figure 4 — predicted vs. LiDAR tree cover per test patch, by state.

Replicates ``09_validation_results_summary_new.ipynb``, cell 7. Layout,
colours, statistics and their formatting follow that code rather than being
redesigned, so a regenerated figure drops into the manuscript unchanged.

Notes on details that are easy to get wrong:

* All three panels use the **same** blue. The notebook has per-state colours
  commented out in favour of one; keeping three hues would imply the panels
  encode something they do not.
* **RMSE is computed against the OLS fit**, not against the 1:1 line. Those
  are different quantities — the first measures scatter about the trend, the
  second includes the bias. The published figure reports the first.
* Zero-LiDAR patches are **kept** (``EXCLUDE_ZERO_LIDAR = False`` in the
  notebook). A patch where the LiDAR finds no trees and the model agrees is
  a correct prediction and belongs in the scatter.

Usage::

    python figures/fig04_scatter_lidar_vs_model.py \\
        --metrics-dir publication/validation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from treecover.figures.validation_data import STATE_NAMES, STATES, load_validation_metrics

#: One colour for every panel, as in the notebook.
POINT_COLOR = "#2196F3"

PANEL_LABELS = ["(a)", "(b)", "(c)"]

X_COLUMN = "lidar_tree_cover_pct"
Y_COLUMN = "model_tree_cover_pct"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-dir", type=Path, required=True,
                   help="Directory with metrics_per_sample_<STATE>.csv "
                        "(or validation_metrics_<STATE>.csv).")
    p.add_argument("--exclude-zero-lidar", action="store_true",
                   help="Drop patches where the LiDAR reports no trees. The "
                        "published figure keeps them.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--name", default="fig04_scatter_lidar_vs_model")
    return p


def fit_statistics(x: np.ndarray, y: np.ndarray) -> dict:
    """Slope, intercept, R², RMSE about the fit, and mean bias."""
    if len(x) < 2:
        return dict(slope=np.nan, intercept=np.nan, r2=np.nan,
                    rmse=np.nan, mean_bias=np.nan, n=len(x))
    coefficients = np.polyfit(x, y, 1)
    predicted = np.polyval(coefficients, x)
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return dict(
        slope=float(coefficients[0]),
        intercept=float(coefficients[1]),
        r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        # About the fit, as in the notebook — not about the 1:1 line.
        rmse=float(np.sqrt(np.mean((y - predicted) ** 2))),
        mean_bias=float(np.mean(y - x)),
        n=len(x),
        coefficients=coefficients,
    )


def draw_panel(ax, data: pd.DataFrame, state: str, panel_label: str) -> dict:
    x = data[X_COLUMN].to_numpy(dtype=float)
    y = data[Y_COLUMN].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    ax.scatter(x, y, s=22, color=POINT_COLOR, alpha=1, edgecolors="none", zorder=3)
    ax.plot([0, 100], [0, 100], "k--", lw=2.2, label="1:1", zorder=5)

    stats = fit_statistics(x, y)
    if np.isfinite(stats["slope"]):
        line_x = np.linspace(-5, 105, 200)
        ax.plot(line_x, np.polyval(stats["coefficients"], line_x),
                color="red", lw=1.5, label="OLS fit", zorder=6, clip_on=True)

    sign = "+" if stats["intercept"] >= 0 else "−"
    ax.text(
        0.03, 0.97,
        f"y = {stats['slope']:.2f}x {sign} {abs(stats['intercept']):.1f}\n"
        f"R² = {stats['r2']:.2f}\n"
        f"RMSE = {stats['rmse']:.1f} pp\n"
        f"bias = {stats['mean_bias']:+.1f} pp\n",
        transform=ax.transAxes, fontsize=14, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="0.7", alpha=0.9),
        zorder=10,
    )

    ax.set_title(STATE_NAMES[state], fontsize=18)
    ax.text(-0.08, 1.04, panel_label, transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom", ha="left")
    ax.set_xlabel("LiDAR tree cover [%]", fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="lower right", frameon=True, framealpha=1.0,
              facecolor="white", fontsize=12)
    return stats


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frame = load_validation_metrics(
            args.metrics_dir, exclude_zero_lidar=args.exclude_zero_lidar
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    present = [s for s in STATES if s in set(frame["state"])]
    if not present:
        print(f"error: no rows for {STATES}; found "
              f"{sorted(set(frame['state']))}", file=sys.stderr)
        return 2

    fig, axes = plt.subplots(1, len(present), figsize=(18, 6),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    results = {}
    for ax, state, label in zip(axes, present, PANEL_LABELS):
        results[state] = draw_panel(ax, frame[frame["state"] == state], state, label)
    axes[0].set_ylabel("Model tree cover [%]", fontsize=14)
    plt.tight_layout()

    out_dir = args.out_dir or Path("figures/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{args.name}.png"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / f"{args.name}.pdf", bbox_inches="tight")
    plt.close(fig)

    for state, stats in results.items():
        print(f"\n--- {STATE_NAMES[state]} ---")
        print(f"slope     = {stats['slope']:.2f}")
        print(f"intercept = {stats['intercept']:+.1f} pp")
        print(f"R²        = {stats['r2']:.2f}")
        print(f"RMSE      = {stats['rmse']:.1f} pp")
        print(f"mean bias = {stats['mean_bias']:+.1f} pp  (model − LiDAR)")
        print(f"n         = {stats['n']:,}")
    print(f"\n-> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
