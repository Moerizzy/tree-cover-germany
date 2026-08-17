#!/usr/bin/env python3
"""Figure 3 — training areas in Lower Saxony, and how their labels were made.

One row per training tile, four columns: the summer orthophoto the label
was drawn on, the nDSM that supported the drawing, the resulting label
mask, and the paired non-summer acquisition that inherits the same label.

The fourth column is the point of the figure. A training tile in this study
is not one image but a *pair* of the same ground under different canopy
conditions, sharing one ground truth — that sharing is what lets the model
see winter canopy without anyone labelling winter imagery. Putting the two
acquisitions at opposite ends of the row makes the seasonal difference and
the identical label visible in the same glance.

Two things this figure deliberately does not do:

**No threshold contour on the nDSM.** The training labels were digitised by
hand with the nDSM and the near-infrared band as aids, not thresholded out
of the height model. Drawing a 3 m contour here would claim a mechanism the
manuscript does not — that is the *validation* reference (figure 6), which
really is a thresholded LiDAR CHM.

**The height ramp is not green.** An nDSM is just as high over a roof as
over a crown; a green ramp would read as canopy and hide the ambiguity the
near-infrared band exists to resolve.

The nDSM in Lower Saxony is the image-based surface model (``dom1``) minus
the LiDAR ground model (``dgm1``). The column is labelled just ``nDSM`` --
the provenance belongs in the caption, not on the panel. Pass
``--height-label`` if a state's height model needs a different name.

Usage::

    # Paths come from the training_data block of paths.yaml
    python figures/fig03_training_examples.py

    python figures/fig03_training_examples.py --list-candidates
    python figures/fig03_training_examples.py --tile-ids 324935816 324165830
    python figures/fig03_training_examples.py \\
        --tiles publication/training/sampled_tiles_100.gpkg \\
        --masks publication/training/labels \\
        --images .../DOP --ndsm .../nDSM
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from treecover.config import load_paths
from treecover.data.observations import build_observations
from treecover.figures.style import (
    HEIGHT,
    INK,
    INK_SECONDARY,
    MASK_CMAP,
    NODATA_GREY,
    SIZES,
    TREE_GREEN,
    apply_style,
    save,
)
from treecover.figures.training_examples import (
    LABEL_NODATA,
    SEASON_LABELS,
    pair_observations,
    select_examples,
    tile_tree_cover_pct,
)
from treecover.imagery import read_band_window, read_rgb_window
from treecover.io.vector import read_vector

logger = logging.getLogger(__name__)

#: Column order. The label sits between the two acquisitions on purpose:
#: it is what they share, and the eye reads it as belonging to both.
COLUMN_TITLES = (
    "Orthophoto (label source)",
    "nDSM",
    "Training label",
    "Paired acquisition",
)

#: Height range of the nDSM ramp, metres. Fixed rather than per-tile, so a
#: colour means the same height in every row — a per-row stretch would make
#: a hedge in a flat tile look like a mature stand.
HEIGHT_VMIN, HEIGHT_VMAX = 0.0, 30.0

#: Side of the square window drawn from each tile, metres. A full 1 km tile
#: reproduced at column width is 190 mm / 5000 px — individual crowns are
#: gone, and the label column becomes a green smear. 500 m is the same
#: choice figure 10 makes, for the same reason.
DEFAULT_EXTENT_M = 500.0

#: The tiles in the published figure, in row order.
#:
#: Pinned rather than left to :func:`select_examples`, for the reason figure
#: 10 pins its scenes: the selection depends on the whole candidate set, so
#: one tile added to or dropped from the training table would silently
#: republish a different figure. The algorithm proposed these three and is
#: still what ``--auto-select`` runs; a human then chose (b) from
#: ``--list-candidates`` for its hedgerow structure.
#:
#: (a) urban village, (b) hedgerows and woodland edge, (c) closed forest —
#: 18 %, 31 % and 63 % canopy, one acquisition pair each.
PUBLISHED_TILES = ("324935816", "324165830", "324415876")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # All four default to the training-data package described in paths.yaml,
    # so the published figure is one command. Imagery and height models are
    # not in that package — those keys point at wherever the state data was
    # downloaded to.
    src = p.add_argument_group("input")
    src.add_argument("--tiles", type=Path, default=None,
                     help="Sampled tiles (GeoPackage/GeoJSON) with tile_id and "
                          "split. Default: training_data.tiles from paths.yaml.")
    src.add_argument("--images", type=Path, default=None,
                     help="Orthophotos named <tile_id>_<date>.tif. "
                          "Default: training_data.images.")
    src.add_argument("--masks", type=Path, default=None,
                     help="Training label masks, one per tile. "
                          "Default: training_data.labels.")
    src.add_argument("--ndsm", type=Path, default=None,
                     help="Height models named <tile_id>_<date>.tif. "
                          "Default: training_data.ndsm. Without them the nDSM "
                          "column cannot be drawn.")
    src.add_argument("--tile-id-column", default="tile_id")
    src.add_argument("--split-column", default="split")
    src.add_argument("--change-column", default="Change",
                     help="Tiles flagged '1' here changed between label and "
                          "imagery and are dropped. Pass '' to disable.")

    sel = p.add_argument_group("selection")
    sel.add_argument("--tile-ids", nargs="*", default=None, metavar="TILE_ID",
                     help="Draw these tiles, in this order. Default: the "
                          "published three (see PUBLISHED_TILES).")
    sel.add_argument("--auto-select", action="store_true",
                     help="Re-run the selection instead of drawing the published "
                          "tiles: rows spread across the tree cover gradient, "
                          "stratified by settlement type.")
    sel.add_argument("--rows", type=int, default=3,
                     help="Number of tiles to draw when picking automatically.")
    sel.add_argument("--urban-rows", type=int, default=1,
                     help="How many rows to reserve for the urban stratum — the "
                          "second axis the training set was sampled on. Ranking "
                          "on tree cover alone returns only rural tiles. 0 turns "
                          "the split off.")
    sel.add_argument("--split", default="train",
                     help="Restrict to one split. '' for any.")
    sel.add_argument("--extent-m", type=float, default=DEFAULT_EXTENT_M,
                     help="Side of the square window drawn from each tile, metres. "
                          "0 draws the whole tile.")
    sel.add_argument("--allow-same-season", action="store_true",
                     help="Keep tiles whose two acquisitions fall in the same "
                          "phenological stage. They cannot show the pairing.")
    sel.add_argument("--list-candidates", action="store_true",
                     help="Print every eligible tile with its cover, stratum and "
                          "acquisition pair, then exit. Use it to choose tiles "
                          "for --tile-ids instead of taking the automatic pick.")

    out = p.add_argument_group("output")
    out.add_argument("--height-label", default="nDSM",
                     help="What the nDSM column is called in the figure.")
    out.add_argument("--out-dir", type=Path, default=None)
    out.add_argument("--name", default="fig03_training_examples")
    out.add_argument("-v", "--verbose", action="store_true")
    return p


def short_date(value) -> str:
    """``2024-06-26 00:00:00`` -> ``2024-06-26``.

    The download stage wrote midnight timestamps into the filenames, and the
    observation table carries the filename through verbatim. A time of day
    on an aerial acquisition is noise in a caption, and 00:00:00 is not even
    the real one.
    """
    text = str(value).strip()
    return text.split(" ")[0] if " " in text else text


def _is_urban(attributes: dict | None) -> bool | None:
    """The tile's settlement stratum, or ``None`` if the table does not say.

    ``None`` rather than ``False``: a tile with no stratum recorded is not
    evidence of a rural tile, and the selection treats the two differently.
    """
    if not attributes or attributes.get("is_urban") is None:
        return None
    return str(attributes["is_urban"]).strip().lower() in ("true", "1", "1.0", "yes")


def stratum_label(attributes: dict | None) -> str:
    """The tile's sampling stratum, as the manuscript names it.

    The training set was drawn along tree cover density × settlement type,
    so saying which cell of that design a row came from is what stops the
    figure looking like three tiles someone liked.
    """
    if not attributes:
        return ""
    parts = []
    tcd_bin = attributes.get("tcd_bin")
    if tcd_bin:
        # "B3: 25-50%" -> "TCD 25-50 %", with a proper en dash.
        text = str(tcd_bin).split(":")[-1].strip().replace("-", "–")
        parts.append(f"TCD {text.replace('%', ' %')}")
    urban = attributes.get("is_urban")
    if urban is not None:
        parts.append("urban" if str(urban).lower() in ("true", "1") else "non-urban")
    return " · ".join(parts)


def window_bounds(path: Path, extent_m: float):
    """The square window drawn from a tile, and the CRS it is in.

    Centred on the tile. ``extent_m <= 0``, or a tile smaller than the
    requested window, gives the tile's full extent rather than a window
    reaching outside it.
    """
    import rasterio

    with rasterio.open(path) as src:
        bounds = tuple(src.bounds)
        crs = src.crs

    if extent_m <= 0:
        return bounds, crs
    minx, miny, maxx, maxy = bounds
    half = min(extent_m, maxx - minx, maxy - miny) / 2.0
    centre_x, centre_y = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    return (centre_x - half, centre_y - half, centre_x + half, centre_y + half), crs


def label_panel(mask: np.ndarray | None):
    """Label mask as an RGB image: white background, green canopy, grey nodata.

    Built explicitly rather than by giving the colormap a third entry,
    because nodata is not a class between background and tree — a colormap
    with three ordered colours would place it on the same scale as the two
    that mean something.
    """
    from matplotlib.colors import to_rgb

    if mask is None:
        return None
    nodata = mask == LABEL_NODATA
    tree = (~nodata) & (mask != 0)

    image = np.ones(mask.shape + (3,), dtype=np.float32)
    image[tree] = to_rgb(TREE_GREEN)
    image[nodata] = to_rgb(NODATA_GREY)
    return image


def load_row(pair, extent_m: float) -> dict:
    """Read the four panels for one tile."""
    bounds, crs = window_bounds(pair.label_source.mask_path, extent_m)

    ndsm = None
    if pair.label_source.ndsm_path is not None:
        ndsm = read_band_window(pair.label_source.ndsm_path, bounds, crs,
                                resampling="bilinear")

    mask = read_band_window(pair.label_source.mask_path, bounds, crs,
                            resampling="nearest")

    return {
        "pair": pair,
        "source_rgb": read_rgb_window(pair.label_source.image_path, bounds, crs),
        "ndsm": ndsm,
        "label": None if mask is None else mask.astype(np.int16),
        "partner_rgb": (None if pair.partner is None
                        else read_rgb_window(pair.partner.image_path, bounds, crs)),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    paths = load_paths()
    args.tiles = args.tiles or paths.get_path("training_data.tiles")
    args.images = args.images or paths.get_path("training_data.images")
    args.masks = args.masks or paths.get_path("training_data.labels")
    if args.ndsm is None:
        # Optional: the figure still draws without it, minus one column.
        configured = paths.get_value("training_data.ndsm")
        args.ndsm = Path(configured) if configured else None

    for label, path in (("--tiles", args.tiles), ("--images", args.images),
                        ("--masks", args.masks)):
        if not path.exists():
            print(f"error: {label} not found: {path}", file=sys.stderr)
            return 2
    if args.ndsm is not None and not args.ndsm.exists():
        logger.warning("No height models at %s — the nDSM column will be empty. "
                       "Set training_data.ndsm in paths.yaml or pass --ndsm.",
                       args.ndsm)
        args.ndsm = None

    tiles = read_vector(args.tiles)
    for column in (args.tile_id_column, args.split_column):
        if column not in tiles.columns:
            print(f"error: column {column!r} not in {args.tiles.name}. "
                  f"Available: {list(tiles.columns)}", file=sys.stderr)
            return 2

    # max_temporal_distance=1 is stage 3's default and the pairing the
    # training set was built under; widening it here would draw a pair the
    # model never saw.
    observations = build_observations(
        image_dir=args.images,
        mask_dir=args.masks,
        tiles=tiles,
        ndsm_dir=args.ndsm,
        max_temporal_distance=1,
        tile_id_column=args.tile_id_column,
        split_column=args.split_column,
        change_column=args.change_column or None,
    )
    if args.split:
        observations = [o for o in observations if o.split == args.split]
    if not observations:
        print("error: no observations matched. Check that image filenames are "
              "<tile_id>_<date>.tif and that --split is right.", file=sys.stderr)
        return 1

    # Stratum attributes, keyed exactly as the tile ids come out of the
    # filenames. Needed before the selection, which stratifies on them.
    attributes = {
        str(row[args.tile_id_column]): row.to_dict()
        for _, row in tiles.drop(columns="geometry", errors="ignore").iterrows()
    }

    pairs = pair_observations(observations)
    for pair in pairs:
        pair.is_urban = _is_urban(attributes.get(pair.tile_id))

    if args.list_candidates:
        for pair in pairs:
            pair.tree_cover_pct = tile_tree_cover_pct(pair.label_source.mask_path)
        return _list_candidates(pairs, attributes, args)

    wanted = args.tile_ids
    if wanted is None and not args.auto_select:
        by_id = {p.tile_id: p for p in pairs}
        absent = [t for t in PUBLISHED_TILES if t not in by_id]
        if absent:
            # The published tiles are not in this training table. Drawing a
            # different set under the same figure name would be worse than
            # saying so and falling back to the algorithm.
            logger.warning(
                "Published tile(s) %s are not in this training set — falling "
                "back to automatic selection. The figure will not match the "
                "manuscript.", ", ".join(absent),
            )
        else:
            wanted = list(PUBLISHED_TILES)

    if wanted:
        by_id = {p.tile_id: p for p in pairs}
        missing = [t for t in wanted if t not in by_id]
        if missing:
            print(f"error: no observations for tile(s) {', '.join(missing)}",
                  file=sys.stderr)
            return 1
        chosen = [by_id[t] for t in wanted]
        for pair in chosen:
            pair.tree_cover_pct = tile_tree_cover_pct(pair.label_source.mask_path)
    else:
        for pair in pairs:
            pair.tree_cover_pct = tile_tree_cover_pct(pair.label_source.mask_path)
        chosen = select_examples(
            pairs,
            count=args.rows,
            require_contrast=not args.allow_same_season,
            require_ndsm=args.ndsm is not None,
            urban_rows=args.urban_rows,
        )

    if not chosen:
        print("error: no training tile could be drawn. Every candidate lacked a "
              "paired acquisition, a seasonal contrast or an nDSM — see the log, "
              "and try --allow-same-season.", file=sys.stderr)
        return 1

    rows = [load_row(pair, args.extent_m) for pair in chosen]

    apply_style()
    fig, axes = plt.subplots(len(rows), 4, figsize=(SIZES.double, 2.05 * len(rows)))
    axes = np.atleast_2d(axes)

    titles = list(COLUMN_TITLES)
    titles[1] = args.height_label
    window = f"{args.extent_m:.0f} m window" if args.extent_m > 0 else "full tile"
    print(f"Figure 3 — {len(rows)} training tile(s), {window}")

    height_cmap = HEIGHT.copy()
    # NaN (no return, or outside the height model) reads as the nodata grey
    # rather than as ground level, which would invent a flat surface.
    height_cmap.set_bad(NODATA_GREY)

    height_image = None
    for index, row in enumerate(rows):
        pair = row["pair"]

        _panel_rgb(axes[index, 0], row["source_rgb"], "orthophoto")
        drawn = _panel_height(axes[index, 1], row["ndsm"], height_cmap)
        height_image = drawn if drawn is not None else height_image
        _panel_rgb(axes[index, 2], label_panel(row["label"]), "label mask")
        _panel_rgb(axes[index, 3], row["partner_rgb"], "paired acquisition")

        cover = pair.tree_cover_pct
        cover_text = f"{cover:.0f} % cover" if not np.isnan(cover) else "cover unknown"
        stratum = stratum_label(attributes.get(pair.tile_id))

        # Row letter inside the panel rather than a label beside it: the
        # caption still needs to name a row, but a margin column of tile ids
        # and strata is metadata, not figure content. The full detail goes to
        # stdout, where it can be pasted into the caption.
        axes[index, 0].text(
            0.03, 0.955, f"({'abcdefghij'[index]})", transform=axes[index, 0].transAxes,
            ha="left", va="top", fontsize=7.5, color=INK,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                      boxstyle="square,pad=0.18"),
        )

        # Per-panel dates, because the two acquisitions are the comparison
        # and a single row label could not carry both.
        if row["source_rgb"] is not None:
            _stamp(axes[index, 0], f"{short_date(pair.label_source.date)}   "
                                   f"{SEASON_LABELS[pair.label_season]}")
        if pair.partner is not None and row["partner_rgb"] is not None:
            _stamp(axes[index, 3], f"{short_date(pair.partner.date)}   "
                                   f"{SEASON_LABELS[pair.partner_season]}")

        for column in range(4):
            # The nDSM and label panels are mostly white, so without a hair
            # rule their edges dissolve into the page and the reader cannot
            # tell an empty panel from a missing one.
            _strip(axes[index, column], frame=column in (1, 2))
            if index == 0:
                axes[index, column].set_title(titles[column], loc="left", color=INK)

        print(f"  ({'abcdefghij'[index]}) {pair.tile_id:<12} {cover:5.1f} % cover  "
              f"{stratum or 'stratum unknown':<28} "
              f"{short_date(pair.label_source.date)} ({SEASON_LABELS[pair.label_season]}) -> "
              f"{short_date(pair.partner.date) if pair.partner else '—'} "
              f"({SEASON_LABELS[pair.partner_season] if pair.partner else 'none'})"
              f"{'' if row['ndsm'] is not None else '   (no nDSM on disk)'}")

    # Colorbar under its own column, legend under the label column. Side by
    # side rather than stacked: stacking put the colorbar's label and the
    # legend's entries on the same line and they overlapped.
    if height_image is not None:
        bar = fig.colorbar(height_image, ax=axes[:, 1].tolist(), location="bottom",
                           fraction=0.035, pad=0.015, aspect=28)
        bar.set_label("Height above ground (m)", color=INK_SECONDARY, fontsize=6.5)
        bar.ax.tick_params(labelsize=6, colors=INK_SECONDARY)
        bar.outline.set_visible(False)

    handles = [
        Patch(facecolor=TREE_GREEN, edgecolor="#cccccc", linewidth=0.3, label="Tree"),
        Patch(facecolor="#ffffff", edgecolor="#cccccc", linewidth=0.3, label="Background"),
        Patch(facecolor=NODATA_GREY, edgecolor="#cccccc", linewidth=0.3, label="No label"),
    ]
    axes[-1, 2].legend(handles=handles, loc="upper center", ncol=3,
                       bbox_to_anchor=(0.5, -0.03), labelcolor=INK_SECONDARY,
                       fontsize=6.5, handlelength=1.1, handleheight=1.1,
                       columnspacing=1.2, borderpad=0.2)

    paths = save(fig, args.name, args.out_dir)
    print(f"-> {', '.join(str(p) for p in paths)}")
    return 0


def _list_candidates(pairs, attributes, args) -> int:
    """Print every tile the automatic pick was choosing between.

    The automatic selection is defensible but it cannot know that a tile is
    under snow or that two rows look alike. This is how a human overrides it
    without reading the code: eyeball the list, then pin the choice with
    ``--tile-ids``, which is what the published figure should use anyway.
    """
    eligible = [p for p in pairs if p.partner is not None]
    if not args.allow_same_season:
        eligible = [p for p in eligible if p.contrast > 0]
    if args.ndsm is not None:
        eligible = [p for p in eligible if p.has_ndsm]
    eligible.sort(key=lambda p: (float("inf") if np.isnan(p.tree_cover_pct)
                                 else p.tree_cover_pct))

    print(f"{len(eligible)} eligible tile(s) of {len(pairs)}\n")
    print(f"{'tile_id':<12} {'cover':>7}  {'stratum':<28} "
          f"{'label source':<12} {'partner':<12} contrast")
    for pair in eligible:
        cover = pair.tree_cover_pct
        print(f"{pair.tile_id:<12} "
              f"{'   n/a' if np.isnan(cover) else f'{cover:6.1f}'} %  "
              f"{stratum_label(attributes.get(pair.tile_id)) or '—':<28} "
              f"{short_date(pair.label_source.date):<12} "
              f"{short_date(pair.partner.date):<12} {pair.contrast}")
    print("\nPin a choice with:  --tile-ids " + " ".join(
        p.tile_id for p in eligible[:3]))
    return 0


def _panel_rgb(ax, image, what: str) -> None:
    """Draw an image panel, or say plainly that it is not on disk."""
    if image is None:
        ax.text(0.5, 0.5, f"{what}\nnot available", ha="center", va="center",
                transform=ax.transAxes, color=INK_SECONDARY, fontsize=6.5)
        ax.set_facecolor("#f5f5f5")
    else:
        ax.imshow(image, interpolation="nearest")


def _panel_height(ax, ndsm, cmap):
    """Draw the nDSM panel and return the image, for the shared colorbar."""
    if ndsm is None:
        ax.text(0.5, 0.5, "nDSM\nnot available", ha="center", va="center",
                transform=ax.transAxes, color=INK_SECONDARY, fontsize=6.5)
        ax.set_facecolor("#f5f5f5")
        return None
    return ax.imshow(np.ma.masked_invalid(ndsm), cmap=cmap,
                     vmin=HEIGHT_VMIN, vmax=HEIGHT_VMAX, interpolation="nearest")


def _stamp(ax, text: str) -> None:
    """Acquisition date and season, bottom-left inside the panel."""
    ax.text(0.03, 0.03, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=5.5, color=INK,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                      boxstyle="square,pad=0.2"))


def _strip(ax, frame: bool = False) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(frame)
        if frame:
            spine.set_color("#cccccc")
            spine.set_linewidth(0.4)


if __name__ == "__main__":
    raise SystemExit(main())
