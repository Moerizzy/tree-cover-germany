#!/usr/bin/env python3
"""Figure 6 — example validation tiles.

Replicates ``09_validation_results_summary_new.ipynb``, the
"Figure 5 — Example Validation Tiles" cell (``figure5_example_tiles.png``;
the manuscript still numbers it 5).

Four rows, one per validation box, four columns: the orthophoto, the
LiDAR-derived reference mask, the model prediction and a pixel-wise
difference. Rows are labelled by **acquisition date** only — the point of
the row order is the seasons, from full leaf-on in August down to leaf-off
in early April, which is the heterogeneity the paper is about.

The four boxes are named in :data:`PUBLISHED_SAMPLES` rather than chosen by
a rule. The notebook selected them by hand, and no ranking reproduces them:
(a) and (c) are near-perfect, (b) is a hard forest edge and (d) is the
failure case — a leaf-off April tile where the model finds a quarter of the
canopy the LiDAR sees. ``--samples`` names others; ``--pick-by-iou`` falls
back to the automatic best/median/worst draw the rebuild used before the
published selection was recovered.

Four details of the drawing are load-bearing and are the notebook's:

* **The prediction panel is drawn at its native 20 cm**, not on the
  reference grid. Reference and difference are 1 m, so the three mask
  panels deliberately differ in blockiness — that *is* the resolution
  argument.
* **The orthophoto is cut with the footprint polygon**, not with its
  bounding box. The polygon is stored in WGS 84 and is a hair off
  axis-aligned once projected, which is why the published panels carry a
  thin dark wedge along one edge. Reading the bounding box instead removes
  the wedge and resamples the photo by up to half a pixel, which at 20 cm
  changes a sixth of the pixels in a canopy.
* **The prediction is put on the reference grid by nearest neighbour**
  (``--reduce nearest``, the default here). The validation pipeline uses a
  majority vote over each 5 x 5 block, which is the better reduction and
  the one behind every number in the paper; on these four boxes the two
  disagree on 3 to 11 of 625 cells. ``--reduce majority`` draws the panel
  the way the metrics are computed.
* **Nodata in the LiDAR mask counts as background**, where the metrics
  exclude it. The published figure has a four-entry legend and none of these
  four boxes contains a single nodata pixel, so the two treatments cannot
  disagree on this figure.

Usage::

    python figures/fig06_example_tiles.py --masks-root .../lidar_masks \\
        --predictions-root /tf/Germany
    python figures/fig06_example_tiles.py --masks-root ... --samples BB:62 BY:104
    python figures/fig06_example_tiles.py --masks-root ... --pick-by-iou 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.mask
from matplotlib.colors import ListedColormap
from pyproj import Transformer
from scipy.ndimage import zoom
from shapely.geometry import box as shapely_box
from shapely.ops import transform as shapely_transform

from treecover.config import load_paths, load_states
from treecover.constants import NODATA, PRED_TREE, validate_prediction_codes
from treecover.figures.style import save
from treecover.imagery import (
    date_from_stem,
    find_prediction_for_point,
    ortho_for_prediction,
)
from treecover.io.vector import read_vector
from treecover.validation import resolve_sample_ids
from treecover.validation.metrics import binary_metrics, majority_vote_downsample

logger = logging.getLogger(__name__)

#: The four boxes of the published figure, as panels (a) to (d).
#:
#: The notebook lists them as ``[('BY', 82), ('BY', 50), ('NRW', 166),
#: ('NRW', 84)]``; the mask directories spell North Rhine-Westphalia ``NW``,
#: which is the spelling used here. They resolve to the acquisition dates
#: printed in the published figure — 2025-08-13, 2025-05-11, 2025-04-06 and
#: 2025-04-07 — which is what makes the identification checkable, since the
#: figure itself carries no sample ids.
PUBLISHED_SAMPLES = (("BY", 82), ("BY", 50), ("NW", 166), ("NW", 84))

#: Difference categories, in the notebook's order and colours: agreement on
#: background is near-white so the eye goes to the two error types.
DIFF_TN, DIFF_TP, DIFF_FP, DIFF_FN = 0, 1, 2, 3
DIFF_COLORS = ["#F5F5F5", "#4CAF50", "#F44336", "#FF9800"]
DIFF_LABELS = ["True negative (TN)", "True positive (TP)",
               "False positive (FP)", "False negative (FN)"]
#: Legend order, which is not the raster order — the errors read first.
DIFF_LEGEND_ORDER = (DIFF_TP, DIFF_FP, DIFF_FN, DIFF_TN)
#: Only the near-white true-negative swatch needs an outline to be visible.
DIFF_LEGEND_EDGE = {DIFF_TN: "#CCCCCC"}

#: The style the notebook set once at the top and every figure in it
#: inherited. It is why the titles and row labels are dark grey rather than
#: black; matplotlib's own default would change the published figure.
NOTEBOOK_STYLE = "seaborn-v0_8-whitegrid"

#: Binary masks use matplotlib's ``Greens`` — the notebook's choice, and not
#: :data:`treecover.figures.style.MASK_CMAP`, so this figure keeps the exact
#: greens the reviewers saw.
MASK_CMAP = "Greens"

#: Percentiles the orthophoto is stretched between, per band. A 25 m box of
#: closed canopy has almost no dynamic range at 8 bit; without the stretch
#: three of the four panels are near-black.
STRETCH_PERCENTILES = (2, 98)

COL_TITLE_FONTSIZE = 13
ROW_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 11

PANEL_LABELS = "abcdefgh"

#: Canonical state codes, matching configs/states.yaml and the directory
#: names under validation/lidar_masks/. "NRW" is an accepted alias but the
#: directories use "NW".
STATE_ORDER = ("NW", "BB", "BY")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--masks-root", type=Path, required=True,
                   help="Directory holding one subdirectory of LiDAR masks per state.")
    p.add_argument("--predictions-root", type=Path, default=None)
    p.add_argument("--samples", nargs="*", default=None, metavar="STATE:ID",
                   help="Specific boxes, e.g. BB:62 BY:104. Defaults to the four "
                        "of the published figure.")
    p.add_argument("--pick-by-iou", type=int, default=0, metavar="N",
                   help="Ignore the published selection and take N boxes per "
                        "state by IoU instead (1 = median, 3 = best, median, "
                        "worst). Scores every box in the state; takes minutes.")
    p.add_argument("--reduce", choices=("nearest", "majority"), default="nearest",
                   help="How the 20 cm prediction is put on the 1 m reference "
                        "grid for the difference panel. 'nearest' is the "
                        "published figure, 'majority' is what the metrics use.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--name", default="fig06_example_tiles")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def load_boxes(masks_root: Path, state: str):
    """Footprints for one state, excluding the boxes flagged in QGIS."""
    directory = masks_root / state
    footprints = directory / "tree_mask_footprints.geojson"
    if not footprints.exists():
        logger.warning("No footprints for %s at %s", state, footprints)
        return None, directory
    boxes = read_vector(footprints)
    # Assign ids before filtering: the mask filenames were numbered when the
    # file was written, so a filtered frame must keep the original numbering.
    boxes = boxes.assign(sample_id=resolve_sample_ids(boxes))
    if "exclude" in boxes.columns:
        boxes = boxes[~boxes["exclude"].fillna(0).astype(bool)]
    return boxes, directory


def score_box(mask_path: Path, predictions_root: Path, pred_dir: str,
              geometry=None, geometry_crs=None, reduce: str = "nearest"):
    """Load reference and prediction for one box. None if either is missing.

    ``prediction`` is the raster as predicted, at 20 cm; ``predicted`` is the
    same put on the 1 m reference grid by ``reduce``, and is what both the
    difference panel and the printed metrics are computed on.

    The prediction is cut with the footprint polygon, like the orthophoto —
    see :func:`read_ortho`. Cutting the two differently would put the photo
    and the mask over it a pixel apart.
    """
    with rasterio.open(mask_path) as src:
        reference_raw = src.read(1)
        bounds = tuple(src.bounds)
        crs = src.crs

    valid = reference_raw != NODATA
    reference = (reference_raw == PRED_TREE) & valid

    centre_x = (bounds[0] + bounds[2]) / 2
    centre_y = (bounds[1] + bounds[3]) / 2
    pred_path = find_prediction_for_point(predictions_root, pred_dir, centre_x, centre_y)
    if pred_path is None:
        return None

    with rasterio.open(pred_path) as src:
        if geometry is not None:
            shape = reproject_geometry(geometry, geometry_crs, src.crs)
            prediction_raw = rasterio.mask.mask(src, [shape], crop=True,
                                                nodata=NODATA)[0][0]
        else:
            from rasterio.windows import from_bounds as window_from_bounds

            window = window_from_bounds(*bounds, transform=src.transform)
            prediction_raw = src.read(1, window=window, boundless=True,
                                      fill_value=NODATA)

    validate_prediction_codes(prediction_raw)
    predicted = to_reference_grid(prediction_raw == PRED_TREE, reference.shape, reduce)
    # The polygon cut leaves a wedge of nodata along one edge. It belongs in
    # the difference panel — the published figure shows it as a strip of
    # false negatives — but not in a metric, so the printed IoU excludes it.
    covered = to_reference_grid(prediction_raw != NODATA, reference.shape, reduce)

    return {
        "reference": reference,
        "valid": valid,
        "prediction": prediction_raw == PRED_TREE,
        "predicted": predicted,
        "bounds": bounds,
        "crs": crs,
        "pred_path": pred_path,
        "metrics": binary_metrics(reference, predicted, valid & covered),
    }


def to_reference_grid(mask: np.ndarray, reference_shape, reduce: str) -> np.ndarray:
    """Put a 20 cm boolean mask on the 1 m reference grid.

    ``"nearest"`` samples one prediction pixel per reference cell, as the
    notebook did. ``"majority"`` takes the mode of each block, as
    :mod:`treecover.validation.metrics` does for every number in the paper —
    a single 20 cm pixel cannot say whether a square metre is canopy. The
    two differ only on cells the model splits.
    """
    if reduce == "majority":
        factor = max(1, round(mask.shape[0] / max(reference_shape[0], 1)))
        resized = majority_vote_downsample(mask.astype(np.uint8), factor).astype(bool)
    else:
        scale = (reference_shape[0] / mask.shape[0], reference_shape[1] / mask.shape[1])
        resized = zoom(mask.astype(np.float32), scale, order=0).astype(bool)
    if resized.shape == tuple(reference_shape):
        return resized
    # zoom rounds each axis independently and can land a pixel out.
    out = np.zeros(reference_shape, dtype=bool)
    rows = min(resized.shape[0], reference_shape[0])
    cols = min(resized.shape[1], reference_shape[1])
    out[:rows, :cols] = resized[:rows, :cols]
    return out


def difference_image(result) -> np.ndarray:
    """Categorical agreement map on the reference grid: TN / TP / FP / FN.

    Nodata is left in whichever agreement class it falls into, as the
    notebook had it — see the module docstring.
    """
    reference, predicted = result["reference"], result["predicted"]
    image = np.full(reference.shape, DIFF_TN, dtype=np.uint8)
    image[reference & predicted] = DIFF_TP
    image[~reference & predicted] = DIFF_FP
    image[reference & ~predicted] = DIFF_FN
    return image


def read_ortho(path: Path, geometry, geometry_crs) -> np.ndarray | None:
    """Cut the orthophoto to a footprint polygon, as ``(h, w, 3)``.

    ``rasterio.mask`` with ``crop=True`` rather than a windowed read: it
    keeps the photo on its own pixel grid — no half-pixel resample — and
    zeroes the corners the projected polygon does not cover, which is the
    dark wedge along the edge of the published panels.
    """
    try:
        with rasterio.open(path) as src:
            shape = reproject_geometry(geometry, geometry_crs, src.crs)
            data, _ = rasterio.mask.mask(src, [shape], crop=True, nodata=0)
    except (rasterio.RasterioIOError, ValueError) as exc:
        logger.warning("Cannot read %s: %s", Path(path).name, exc)
        return None

    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    return np.moveaxis(data[:3], 0, -1)


def reproject_geometry(geometry, src_crs, dst_crs):
    """Move a shapely geometry between two CRSs."""
    if src_crs is None or dst_crs is None:
        return geometry
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely_transform(lambda x, y: transformer.transform(x, y), geometry)


def stretch_rgb(image: np.ndarray) -> np.ndarray | None:
    """Percentile-stretch an RGB image to 0–1, per band.

    Zero is the fill outside the footprint and past the edge of a tile, and
    is excluded from the percentiles, so a partly covered box is stretched
    on the pixels it actually has.
    """
    if image is None:
        return None
    out = image.astype(np.float32)
    for band in range(out.shape[2]):
        positive = out[..., band][out[..., band] > 0]
        if positive.size == 0:
            continue
        low, high = np.percentile(positive, STRETCH_PERCENTILES)
        out[..., band] = np.clip((out[..., band] - low) / max(high - low, 1e-6), 0, 1)
    return out


def collect_samples(masks_root: Path, predictions_root: Path, wanted,
                    reduce: str = "nearest"):
    """Load the named boxes, in the order given."""
    registry = load_states()
    cache: dict[str, tuple] = {}
    rows = []

    for state, sample_id in wanted:
        if state not in cache:
            cache[state] = load_boxes(masks_root, state)
        boxes, directory = cache[state]
        if boxes is None:
            continue
        match = boxes[boxes["sample_id"].astype(int) == sample_id]
        if match.empty:
            logger.warning("%s: box %d is not in the footprints (excluded?)",
                           state, sample_id)
        mask_path = directory / f"tree_mask_sample_{sample_id:04d}.tif"
        if not mask_path.exists():
            logger.warning("%s: no mask at %s", state, mask_path)
            continue
        geometry = None if match.empty else match.geometry.iloc[0]
        result = score_box(mask_path, predictions_root, registry[state].pred_dir,
                           geometry, boxes.crs, reduce)
        if result is None:
            logger.warning("%s: no prediction covers box %d", state, sample_id)
            continue
        result.update(state=state, sample_id=sample_id,
                      geometry=geometry, geometry_crs=boxes.crs)
        rows.append(result)
    return rows


def pick_by_iou(masks_root: Path, predictions_root: Path, states, per_state: int,
                reduce: str = "nearest"):
    """The automatic draw: best, median and worst box per state by IoU.

    Kept because it is how a *new* validation set would be inspected, and
    because it is the honest way to check that the published four are not
    flattering. It does not reproduce the published figure.
    """
    registry = load_states()
    rows = []

    for state in states:
        boxes, directory = load_boxes(masks_root, state)
        if boxes is None:
            continue
        pred_dir = registry[state].pred_dir

        scored = []
        for _, row in boxes.iterrows():
            sample_id = int(row["sample_id"])
            mask_path = directory / f"tree_mask_sample_{sample_id:04d}.tif"
            if not mask_path.exists():
                continue
            result = score_box(mask_path, predictions_root, pred_dir,
                               row.geometry, boxes.crs, reduce)
            if result is None:
                continue
            result.update(state=state, sample_id=sample_id,
                          geometry=row.geometry, geometry_crs=boxes.crs)
            scored.append(result)

        if not scored:
            logger.warning("%s: no box could be scored", state)
            continue

        scored.sort(key=lambda r: r["metrics"].iou)
        candidates = [scored[-1], scored[len(scored) // 2], scored[0]]
        seen, picks = set(), []
        for candidate in candidates[:per_state]:
            if candidate["sample_id"] not in seen:
                seen.add(candidate["sample_id"])
                picks.append(candidate)
        rows.extend(picks)
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    if not args.masks_root.is_dir():
        print(f"error: {args.masks_root} is not a directory", file=sys.stderr)
        return 2
    predictions_root = args.predictions_root or load_paths().get_path("predictions_root")

    if args.pick_by_iou:
        states = [s for s in STATE_ORDER if (args.masks_root / s).is_dir()]
        rows = pick_by_iou(args.masks_root, predictions_root, states,
                           args.pick_by_iou, args.reduce)
    else:
        wanted = list(PUBLISHED_SAMPLES)
        if args.samples:
            try:
                wanted = [(s.split(":")[0].upper(), int(s.split(":")[1]))
                          for s in args.samples]
            except (IndexError, ValueError):
                print("error: --samples takes STATE:ID pairs, e.g. BB:62",
                      file=sys.stderr)
                return 2
        # "NRW" is the project's name for the state, "NW" the directory's.
        registry = load_states()
        wanted = [(registry.resolve(s), i) for s, i in wanted]
        rows = collect_samples(args.masks_root, predictions_root, wanted,
                               args.reduce)

    if not rows:
        print("error: no box could be drawn", file=sys.stderr)
        return 1

    # Not treecover.figures.style.apply_style(): this figure is four rows of
    # imagery at the notebook's own size and type scale, and the shared style
    # turns on constrained layout, which tight_layout(rect=...) then fights.
    try:
        plt.style.use(NOTEBOOK_STYLE)
    except OSError:
        logger.warning("Style %s is not available in matplotlib %s; titles and "
                       "labels will come out black rather than dark grey.",
                       NOTEBOOK_STYLE, plt.matplotlib.__version__)
    plt.rcParams.update({
        "figure.constrained_layout.use": False,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(13, n_rows * 3.2))
    axes = np.atleast_2d(axes)

    titles = ["Ortho (RGB)", "LiDAR mask", "Model prediction", "Difference"]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=COL_TITLE_FONTSIZE, pad=6)

    print(f"Figure 6 — {n_rows} box(es)")

    for index, result in enumerate(rows):
        image_path = ortho_for_prediction(result["pred_path"])
        ortho = None
        if image_path is not None:
            footprint = result.get("geometry")
            if footprint is None:
                # No footprint row for this box — fall back to the mask's own
                # extent, which is the same square minus the projection wedge.
                footprint = shapely_box(*result["bounds"])
                crs = result["crs"]
            else:
                crs = result["geometry_crs"]
            ortho = stretch_rgb(read_ortho(image_path, footprint, crs))

        _panel_image(axes[index, 0], ortho)
        # Nodata is drawn as background, matching the difference panel.
        axes[index, 1].imshow(result["reference"].astype(float), cmap=MASK_CMAP,
                              vmin=0, vmax=1, interpolation="nearest")
        # At 20 cm, where it was predicted — the reference and difference
        # panels are 1 m, and the figure says so by looking different.
        axes[index, 2].imshow(result["prediction"].astype(float), cmap=MASK_CMAP,
                              vmin=0, vmax=1, interpolation="nearest")
        axes[index, 3].imshow(difference_image(result),
                              cmap=ListedColormap(DIFF_COLORS), vmin=0, vmax=3,
                              interpolation="nearest")

        stem = (image_path or result["pred_path"]).stem
        date = date_from_stem(stem) or "n/a"
        axes[index, 0].set_ylabel(f"({PANEL_LABELS[index]})  {date}",
                                  fontsize=ROW_LABEL_FONTSIZE)
        for column in range(4):
            _strip(axes[index, column])

        metrics = result["metrics"]
        print(f"  ({PANEL_LABELS[index]}) {result['state']:<4} #{result['sample_id']:<4} "
              f"{date}  IoU {metrics.iou:.3f}  ref {metrics.reference_cover_pct:5.1f} % "
              f" pred {metrics.predicted_cover_pct:5.1f} %"
              f"{'' if image_path else '   (no orthophoto on disk)'}")

    handles = [mpatches.Patch(color=DIFF_COLORS[c], label=DIFF_LABELS[c],
                              ec=DIFF_LEGEND_EDGE.get(c, DIFF_COLORS[c]))
               for c in DIFF_LEGEND_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=LEGEND_FONTSIZE, frameon=True,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    paths = save(fig, args.name, args.out_dir)
    print(f"-> {', '.join(str(p) for p in paths)}")
    return 0


def _panel_image(ax, image) -> None:
    """Draw the orthophoto, or say plainly that it is not on disk."""
    if image is None:
        ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                transform=ax.transAxes, color="grey", fontsize=10)
        ax.set_facecolor("#EEEEEE")
    else:
        ax.imshow(image, interpolation="nearest")


def _strip(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


if __name__ == "__main__":
    raise SystemExit(main())
