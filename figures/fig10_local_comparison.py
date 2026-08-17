#!/usr/bin/env python3
"""Figure 10 — local-scale comparison against the reference products.

Replicates ``plot_ortho_prediction_examples.ipynb``, the final cell
(``combined_products_comparison.png``).

**One row per layer, one column per scene** — orthophoto, our prediction,
and each reference product beneath one another, so a reader compares
downwards within a scene rather than across the page.

Each scene is the **centre 50 % of one 1 km tile**: 2500 x 2500 px at 20 cm,
500 m on a side. Not a lon/lat box — the tile is the unit the whole pipeline
works in, and cropping its middle keeps the panel clear of the seams where
neighbouring flight strips meet.

Native resolution is the point. Aggregating the products to a common grid,
as the quantitative comparison does, is what makes them comparable — and
exactly what hides why they differ. Shown as they are, a 10 m Sentinel-2
pixel next to a 20 cm prediction explains the systematic gap in sparsely
wooded areas better than any statistic.

Two deliberate departures from the notebook, both recorded here so the
difference against the published figure is a decision and not a drift:

* **Every layer is binary.** The notebook drew the two canopy height models
  on a 0–30 m ramp and TCD on a 0–100 % one, which asks the reader to
  compare a height against a density against a mask. Thresholding them —
  3 m for the height products, :data:`DENSITY_THRESHOLD_PCT` for TCD —
  puts every panel on the same question: tree or not.
* **Nearest-neighbour resampling**, where the notebook used bilinear.
  Interpolating a 10 m product up to a 20 cm grid and *then* thresholding
  draws a smooth boundary through pixels that do not exist, which flatters
  the coarse products at exactly the scale this figure exists to show.
  ``--resampling bilinear`` restores the notebook's behaviour.

One thing this figure does **not** share with the per-tile table: it ignores
the mosaics' per-source mask bands. The CHMv2 mask is defective on a small
number of tiles — over scene (b) it hides 44 % of the pixels, three quarters
of which are canopy above 3 m — and honouring it there would show CHMv2
missing a forest it maps perfectly well. Nationally the mask is worth
0.03 pp, so the table keeps it and this figure drops it; ``--source-mask
honour`` makes the two agree. See
:func:`treecover.products.open_without_source_mask`.

Usage::

    python figures/fig10_local_comparison.py \\
        --products-root /tf/Other_Tree_Products --predictions-root /tf/Germany

    # redraw the scenes as the notebook did, instead of using the published ones
    python figures/fig10_local_comparison.py --draw-seed 42 ...
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.windows
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds

from treecover.config import load_paths
from treecover.figures.style import INK, INK_SECONDARY
from treecover.imagery import (
    date_from_stem,
    find_prediction_by_stem,
    ortho_for_prediction,
    state_from_stem,
)
from treecover.products import HEIGHT_THRESHOLD_M, PRODUCTS, open_without_source_mask

logger = logging.getLogger(__name__)

#: The scenes of the published figure.
#:
#: The notebook chose them at random: ``random.sample(matched, 3)`` under
#: ``seed = 42`` over the sorted list of prediction/orthophoto pairs, with
#: the third replaced by ``random.choice(remaining)`` under ``seed = 142``.
#: That draw is reproducible only as long as the archive holds exactly the
#: same 380,213 pairs — one deleted tile shifts every index after it and
#: the figure silently becomes a different figure. Naming the three tiles
#: pins them; ``--draw-seed`` re-runs the draw for anyone who wants to see
#: it reproduce.
PUBLISHED_TILES = (
    "dop20rgbi_33_411_5654_sn_file_20240319",  # (a) SN — urban, leaf-off
    "dop20rgbi_32_573_5458_bw_file_20240730",  # (b) BW — forest edge, leaf-on
    "dop20rgbi_32_518_6027_sh_file_20240514",  # (c) SH — agricultural, hedgerows
)

#: The three comparison products, taken from :mod:`treecover.products` so
#: their paths, value types and nodata cannot drift from the extraction
#: that builds the quantitative comparison.
PANEL_COLUMNS = (
    "meta_chm_treecover_pct",         # CHMv2
    "treesense_chm3m_treecover_pct",  # Planet CHM
    "clms_tcd2023_treecover_pct",     # TCD
)

#: Density above which a pixel counts as tree, percent.
#:
#: Every panel is a binary mask, so the density product needs a threshold
#: that the height products get for free from the 3 m rule. This is a
#: display choice and not the reduction the quantitative comparison uses —
#: there TCD enters as a density and is averaged, with no threshold at all.
#: 50 % means "more tree than not" within a pixel.
DENSITY_THRESHOLD_PCT = 50.0

#: Green of the notebook's tree overlay, on white.
TREE_COLOR = (0.2, 0.85, 0.3)

#: Fraction of the tile kept, centred. 0.5 of a 1 km tile is 500 m.
CROP_FRACTION = 0.5

RESAMPLING = {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", nargs="+", default=list(PUBLISHED_TILES), metavar="STEM",
                   help="Prediction tile names, one per column. Defaults to the "
                        "three scenes of the published figure.")
    p.add_argument("--draw-seed", type=int, default=None, metavar="SEED",
                   help="Redraw the scenes as the notebook did instead of using "
                        "--tiles. Walks the whole archive; takes minutes.")
    p.add_argument("--draw-count", type=int, default=3,
                   help="Number of scenes to draw with --draw-seed.")
    p.add_argument("--products-root", type=Path, required=True)
    p.add_argument("--predictions-root", type=Path, default=None)
    p.add_argument("--density-threshold", type=float, default=DENSITY_THRESHOLD_PCT,
                   help="Density above which a pixel counts as tree, percent. "
                        "Applies to TCD; the height products use the 3 m rule.")
    p.add_argument("--resampling", choices=sorted(RESAMPLING), default="nearest",
                   help="How products are put on the display grid. The notebook "
                        "used bilinear; nearest is honest about pixel size.")
    p.add_argument("--source-mask", choices=("ignore", "honour"), default="ignore",
                   help="Whether the mosaics' per-source mask bands are applied. "
                        "The CHMv2 mask is defective on a few tiles, one of them "
                        "under scene (b); 'honour' matches the per-tile table, "
                        "'ignore' matches the published figure and the raster.")
    p.add_argument("--dpi", type=int, default=600,
                   help="600 as in the notebook, which yields a ~47 MB PNG. "
                        "Pass 150 for a version to look at.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--name", default="fig10_local_comparison")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


# --------------------------------------------------------------------------
# Scene selection
# --------------------------------------------------------------------------

def draw_scenes(predictions_root: Path, seed: int, count: int) -> list[str]:
    """Reproduce the notebook's random draw over the archive.

    Kept because it is the provenance of :data:`PUBLISHED_TILES` and the
    only way to check that those three are still what the seed yields.
    Walks every prediction and every JPEG 2000 under the root, so it costs
    minutes; the tile names it prints are what belongs in ``--tiles``.
    """
    predictions = sorted(p for p in predictions_root.rglob("*.tif")
                         if "predictions" in p.parts)
    orthos = {p.stem: p for p in predictions_root.rglob("*.jp2")}
    matched = [(p, orthos[p.stem.replace("_pred", "")]) for p in predictions
               if p.stem.replace("_pred", "") in orthos]
    logger.info("%d prediction/orthophoto pairs", len(matched))

    random.seed(seed)
    sample = random.sample(matched, min(count, len(matched)))
    if len(sample) >= 3:
        # The notebook replaced the third pick under a second seed.
        remaining = [m for m in matched if m not in sample]
        random.seed(seed + 100)
        sample[2] = random.choice(remaining)
    return [pred.stem for pred, _ in sample]


def crop_window(source, fraction: float):
    """The centred window covering ``fraction`` of a raster, and its bounds."""
    margin = (1.0 - fraction) / 2
    col_off = int(source.width * margin)
    row_off = int(source.height * margin)
    width = int(source.width * fraction)
    height = int(source.height * fraction)
    window = Window(col_off, row_off, width, height)
    return window, rasterio.windows.bounds(window, source.transform)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def read_scene(pred_path: Path, fraction: float):
    """Read the orthophoto crop and the matching prediction mask.

    The extent comes from the orthophoto, and the prediction is read over
    those same bounds, so the two panels cannot end up a pixel apart if the
    prediction was written on a slightly different grid.

    Returns:
        ``(rgb, mask, bounds, crs, pixel_size_m)``. ``rgb`` is ``None``
        where the imagery was deleted after inference, which is the normal
        state for most states.
    """
    with rasterio.open(pred_path) as src:
        pred_window, bounds = crop_window(src, fraction)
        crs = src.crs
        pixel_size = abs(src.transform.a)
        mask = src.read(1, window=pred_window) == 1

    height, width = mask.shape
    rgb = None
    ortho_path = ortho_for_prediction(pred_path)
    if ortho_path is None:
        logger.warning("No orthophoto for %s", pred_path.name)
    else:
        with rasterio.open(ortho_path) as src:
            window = window_from_bounds(*bounds, transform=src.transform)
            bands = min(3, src.count)
            data = src.read(
                indexes=list(range(1, bands + 1)), window=window,
                out_shape=(bands, height, width),
                resampling=Resampling.bilinear,
                boundless=True, fill_value=0,
            )
            pixel_size = abs(src.transform.a)
        if data.shape[0] < 3:
            data = np.repeat(data, 3, axis=0)
        rgb = np.transpose(data[:3], (1, 2, 0)).astype(np.uint8)

    return rgb, mask, bounds, crs, pixel_size


def read_product_mask(product, root: Path, bounds, dst_crs, shape,
                      density_threshold: float, resampling,
                      source_mask: str = "ignore"):
    """Read one product over a scene and reduce it to a binary tree mask.

    Nodata is excluded before thresholding, so a pixel the product never
    saw is not counted as treeless.

    Args:
        source_mask: ``"ignore"`` strips the mosaic's per-source mask bands,
            ``"honour"`` keeps them. See
            :func:`treecover.products.open_without_source_mask` — the CHMv2
            mask is defective on a few tiles, one of which the published
            figure happens to sit on.

    Returns:
        A boolean mask, or ``None`` if the product does not cover the scene
        — drawn as a gap rather than as absence of trees.
    """
    path = root / product.relative_path
    if not path.exists():
        logger.warning("Product not found: %s", path)
        return None

    height, width = shape
    opener = (open_without_source_mask(path) if source_mask == "ignore"
              else rasterio.open(path))
    try:
        with opener as src:
            source_bounds = transform_bounds(dst_crs, src.crs, *bounds)
            if not _intersects(src.bounds, source_bounds):
                logger.info("%s does not cover this scene", product.label)
                return None

            nodata = product.nodata if product.nodata is not None else src.nodata
            fill = nodata if nodata is not None else 0
            window = window_from_bounds(*source_bounds, transform=src.transform)
            data = src.read(1, window=window, boundless=True,
                            fill_value=fill).astype(np.float32)
            source_transform = src.window_transform(window)
            source_crs = src.crs
    except rasterio.RasterioIOError as exc:
        logger.warning("Cannot read %s: %s", path.name, exc)
        return None

    if nodata is not None:
        data[data == nodata] = np.nan

    destination = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=data, destination=destination,
        src_transform=source_transform, src_crs=source_crs,
        dst_transform=transform_from_bounds(*bounds, width, height),
        dst_crs=dst_crs, resampling=resampling,
        src_nodata=np.nan, dst_nodata=np.nan,
    )

    threshold = HEIGHT_THRESHOLD_M if product.kind == "height" else density_threshold
    return np.where(np.isnan(destination), False, destination >= threshold)


def _intersects(a, b) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def draw_mask(ax, mask) -> None:
    """Draw a binary mask as green on white, or say the data is absent."""
    if mask is None:
        ax.text(0.5, 0.5, "not available", ha="center", va="center",
                transform=ax.transAxes, color=INK_SECONDARY, fontsize=14)
        ax.set_facecolor("#f5f5f5")
        return
    image = np.ones(mask.shape + (3,), dtype=np.float32)
    image[mask] = TREE_COLOR
    ax.imshow(image, interpolation="nearest")


def scale_bar(ax, ground_width_m: float) -> None:
    """A white scale bar in the corner of a panel.

    The length is the largest round number below a third of the panel, as
    the notebook chose it — 100 m over a 500 m scene.
    """
    length_m = 50
    for candidate in (50, 100, 200, 500, 1000):
        if candidate / ground_width_m < 0.35:
            length_m = candidate
    fraction = length_m / ground_width_m

    x0, y0 = 0.05, 0.07
    ax.plot([x0, x0 + fraction], [y0, y0], color="white", lw=3.5,
            transform=ax.transAxes, solid_capstyle="butt")
    for x in (x0, x0 + fraction):
        ax.plot([x, x], [y0 - 0.02, y0 + 0.02], color="white", lw=2,
                transform=ax.transAxes)
    label = f"{length_m} m" if length_m < 1000 else f"{length_m // 1000} km"
    ax.text(x0 + fraction / 2, y0 + 0.03, label, color="white", ha="center",
            va="bottom", fontsize=14, fontweight="bold", transform=ax.transAxes)


def tree_legend(ax) -> None:
    """The one legend the figure needs, now that every mask row is binary."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    swatch_w, swatch_h = 0.30, 0.05
    for y, (colour, label) in zip((0.54, 0.42),
                                  ((TREE_COLOR, "Tree"), ("white", "No tree"))):
        ax.add_patch(mpatches.Rectangle((0.08, y), swatch_w, swatch_h,
                                        facecolor=colour, edgecolor="grey",
                                        lw=0.8, transform=ax.transAxes))
        ax.text(0.62, y + swatch_h / 2, label, ha="center", va="center",
                fontsize=13, rotation=90, color=INK, transform=ax.transAxes)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    predictions_root = args.predictions_root or load_paths().get_path("predictions_root")
    predictions_root = Path(predictions_root)

    stems = (draw_scenes(predictions_root, args.draw_seed, args.draw_count)
             if args.draw_seed is not None else list(args.tiles))

    by_column = {p.column: p for p in PRODUCTS}
    products = [by_column[c] for c in PANEL_COLUMNS if c in by_column]
    absent = [p.label for p in products
              if not (args.products_root / p.relative_path).exists()]
    if absent:
        print(f"Not found under {args.products_root}: {', '.join(absent)}",
              file=sys.stderr)

    scenes = []
    for stem in stems:
        pred_path = find_prediction_by_stem(predictions_root, stem)
        if pred_path is None:
            print(f"error: no prediction named {stem!r} under {predictions_root}",
                  file=sys.stderr)
            return 2
        scenes.append((stem, pred_path))

    rows = ["Orthophoto", "Prediction (ours)"] + [
        f"{p.label} (≥ {HEIGHT_THRESHOLD_M:.0f} m)" if p.kind == "height"
        else f"{p.label} (≥ {args.density_threshold:.0f} %)"
        for p in products
    ]
    n_rows, n_cols = len(rows), len(scenes)

    fig = plt.figure(figsize=(6 * n_cols + 0.8, 6 * n_rows))
    grid = fig.add_gridspec(n_rows, n_cols + 1,
                            width_ratios=[1] * n_cols + [0.12],
                            hspace=0.05, wspace=0.02)
    axes = [[fig.add_subplot(grid[r, c]) for c in range(n_cols)]
            for r in range(n_rows)]
    # One legend for the whole binary block, spanning every mask row.
    ax_blank = fig.add_subplot(grid[0, n_cols])
    ax_blank.axis("off")
    tree_legend(fig.add_subplot(grid[1:, n_cols]))

    print(f"Figure 10 — {n_cols} scene(s), {len(products)} product(s); all binarised "
          f"(height >= {HEIGHT_THRESHOLD_M:.0f} m, density >= "
          f"{args.density_threshold:.0f} %), {args.resampling} resampling, "
          f"source masks "
          f"{'ignored' if args.source_mask == 'ignore' else 'honoured'}")

    ground_width_m = None
    for column, (stem, pred_path) in enumerate(scenes):
        rgb, prediction, bounds, crs, pixel_size = read_scene(pred_path, CROP_FRACTION)
        ground_width_m = prediction.shape[1] * pixel_size

        if rgb is None:
            axes[0][column].text(0.5, 0.5, "orthophoto deleted", ha="center",
                                 va="center", transform=axes[0][column].transAxes,
                                 color=INK_SECONDARY, fontsize=14)
            axes[0][column].set_facecolor("#f5f5f5")
        else:
            axes[0][column].imshow(rgb, interpolation="nearest")
        draw_mask(axes[1][column], prediction)

        shares = [("ours", 100.0 * prediction.mean())]
        for row, product in enumerate(products, start=2):
            mask = read_product_mask(product, args.products_root, bounds, crs,
                                     prediction.shape, args.density_threshold,
                                     RESAMPLING[args.resampling], args.source_mask)
            draw_mask(axes[row][column], mask)
            shares.append((product.label,
                           None if mask is None else 100.0 * mask.mean()))

        state = state_from_stem(stem) or "?"
        date = date_from_stem(stem) or "?"
        axes[0][column].set_title(f"({'abcdefghij'[column]})  {state}  —  {date}",
                                  fontsize=18, pad=8, color=INK)
        for row in range(n_rows):
            axes[row][column].set_axis_off()

        print(f"  ({'abcdefghij'[column]}) {state} {date}  {stem}")
        print("      " + "   ".join(
            f"{name} {value:.1f} %" if value is not None else f"{name} n/a"
            for name, value in shares))

    for row, label in enumerate(rows):
        axes[row][0].text(-0.04, 0.5, label, transform=axes[row][0].transAxes,
                          fontsize=18, va="center", ha="right", rotation=90,
                          color=INK)

    if ground_width_m:
        scale_bar(axes[0][-1], ground_width_m)

    out_dir = Path(args.out_dir) if args.out_dir else Path("figures/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.name}.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {path} ({path.stat().st_size / 1e6:.1f} MB at {args.dpi} dpi)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
