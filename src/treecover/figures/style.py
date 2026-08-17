"""Shared figure styling.

Every paper figure imports from here, so a change to a colour or a font
size propagates rather than being made in six places — which is how the
original notebooks ended up with three different greens for "our product".

Colour choices follow four rules:

* **Categorical** hues are assigned in a fixed order and never cycled. The
  order is Okabe–Ito, designed to stay distinguishable under protanopia,
  deuteranopia and tritanopia. A ninth series would need a different
  encoding, not a ninth hue.
* **Sequential** (tree cover %) is a single hue, light to dark.
* **Diverging** (difference against our product) is two hues with a neutral
  grey midpoint — never a hue at the middle, never a rainbow.
* **Status/identity never rides on colour alone** — series are also direct-
  labelled or given distinct markers.

.. note::
   ``DIVERGING`` defaults to red–blue because that is what the submitted
   manuscript used, and the figures here must reproduce it. Red–blue is the
   weaker choice for protanopia, where red darkens towards the midpoint.
   :data:`DIVERGING_CVD` (purple–orange) is the better option for any new
   figure; pass ``diverging=DIVERGING_CVD`` to switch.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm

__all__ = [
    "OKABE_ITO",
    "PRODUCT_COLORS",
    "PRODUCT_LABELS",
    "SEQUENTIAL",
    "HEIGHT",
    "DIVERGING",
    "DIVERGING_CVD",
    "SEASON_COLORS",
    "TREE_GREEN",
    "MASK_CMAP",
    "NODATA_GREY",
    "apply_style",
    "diverging_norm",
    "figure_path",
    "save",
]

#: Okabe & Ito (2008) qualitative palette, in its canonical order.
#: Assign by index; never cycle past the end.
OKABE_ITO = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

#: Product column -> fixed hue, so a figure that drops a product does not
#: repaint the survivors. Keyed by *column*, not by label, so a renamed
#: caption cannot silently change a colour.
#:
#: The three comparison products use orange / bluish-green / sky-blue —
#: the widest-separated triple in Okabe–Ito, and the safest under all three
#: CVD types. Vermillion and orange are the one weak pair in the palette and
#: are deliberately not used together.
PRODUCT_COLORS = {
    "our_treecover_pct": "#009E73",                # bluish green
    "meta_chm_treecover_pct": "#3fa96b",           # CHMv2
    "treesense_chm3m_treecover_pct": "#e07b4a",    # Planet CHM
    "clms_tcd2023_treecover_pct": "#7a8ec6",       # TCD
}

#: Product column -> the manuscript's name for it. Captions, legends and
#: Table 1 read from here, so the figures and the text cannot drift apart.
#:
#: .. warning::
#:    "Planet CHM" is ``treesense_chm3m_treecover_pct`` — the *canopy
#:    height* product of Liu et al. (2025), thresholded at 3 m. It is not
#:    ``treesense3m_treecover_pct``, which is the separate TreeSense tree
#:    *cover* raster at 3 m. Nationally the two differ by nine percentage
#:    points (36.1 % against 27.4 %) and by the sign of their difference
#:    against our map, so confusing them turns an overestimate into an
#:    underestimate without anything looking wrong.
PRODUCT_LABELS = {
    "our_treecover_pct": "Ours (20 cm)",
    "meta_chm_treecover_pct": "CHMv2",
    "treesense_chm3m_treecover_pct": "Planet CHM",
    "clms_tcd2023_treecover_pct": "TCD",
}

#: Single-hue ramp for tree cover percentage.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "treecover_greens", ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"]
)

#: Single-hue ramp for height above ground (nDSM / CHM), light to dark.
#:
#: Purple rather than green on purpose. Height and tree cover appear side by
#: side in the training-data figure, and a green height ramp would read as
#: "this panel already shows canopy" — but an nDSM is high over roofs too,
#: which is exactly the ambiguity the near-infrared band resolves. A hue the
#: eye does not associate with vegetation keeps the two panels doing
#: different jobs. Single hue, so it survives all three CVD types.
HEIGHT = LinearSegmentedColormap.from_list(
    "ndsm_height", ["#fcfbfd", "#dadaeb", "#9e9ac8", "#6a51a3", "#3f007d"]
)

#: The one green that means "tree" throughout the paper, and the two-value
#: colormap binary masks are drawn with. Background is white rather than a
#: second hue so the eye goes to the canopy; both the LiDAR reference and the
#: training labels use it, which is what makes the two figures comparable.
TREE_GREEN = "#1b7837"
MASK_CMAP = ListedColormap(["#ffffff", TREE_GREEN])

#: Nodata in a mask panel. Grey, never white — a nodata pixel is excluded
#: from every metric and must not read as "agreed background".
NODATA_GREY = "#bdbdbd"

#: Red–blue, as used in the manuscript. Neutral grey at the midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "treecover_diff", ["#b2182b", "#ef8a62", "#f0f0f0", "#67a9cf", "#2166ac"]
)

#: Purple–orange. Preferred for new figures; safe under all three CVD types.
DIVERGING_CVD = LinearSegmentedColormap.from_list(
    "treecover_diff_cvd", ["#b35806", "#f1a340", "#f0f0f0", "#998ec3", "#542788"]
)

#: One hue per meteorological season, for the acquisition-date figures.
SEASON_COLORS = {
    "Winter": "#0072B2",
    "Spring": "#009E73",
    "Summer": "#E69F00",
    "Autumn": "#D55E00",
}

#: Ink colours. Text never wears a series colour — the mark beside it carries
#: identity.
INK = "#1a1a1a"
INK_SECONDARY = "#4d4d4d"
INK_MUTED = "#808080"
GRID = "#d9d9d9"


@dataclass(frozen=True)
class FigureSize:
    """Widths matching the Remote Sensing of Environment column grid."""

    single: float = 3.54   # 90 mm
    double: float = 7.48   # 190 mm


SIZES = FigureSize()


def apply_style(base_font_size: float = 8.0) -> None:
    """Set matplotlib defaults for print figures.

    Small type and thin marks: an RSE figure is reproduced at column width,
    so anything sized for a screen comes out heavy. Grid and spines are
    recessive so the data carries the contrast.

    Args:
        base_font_size: Body text size in points. 8 pt is about the smallest
            that stays legible in print at 90 mm.
    """
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": base_font_size,
            "axes.titlesize": base_font_size + 1,
            "axes.labelsize": base_font_size,
            "xtick.labelsize": base_font_size - 1,
            "ytick.labelsize": base_font_size - 1,
            "legend.fontsize": base_font_size - 1,
            "axes.edgecolor": INK_SECONDARY,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.7,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )


def diverging_norm(vmin: float, vmax: float, center: float = 0.0) -> TwoSlopeNorm:
    """Diverging norm pinned so the neutral colour sits exactly at ``center``.

    Without pinning, an asymmetric data range puts zero somewhere off-centre
    in the ramp and a small positive difference reads as a large one.
    """
    # TwoSlopeNorm requires vmin < center < vmax strictly.
    span = max(abs(vmin - center), abs(vmax - center), 1e-6)
    return TwoSlopeNorm(vmin=center - span, vcenter=center, vmax=center + span)


def figure_path(name: str, out_dir=None, ext: str = "png"):
    """Resolve an output path, creating the directory."""
    from pathlib import Path

    from treecover.config import load_paths

    directory = Path(out_dir) if out_dir else load_paths().get_path(
        "figures_root", "./figures/output"
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}.{ext}"


def save(fig, name: str, out_dir=None, formats: tuple[str, ...] = ("png", "pdf")) -> list:
    """Write a figure in each requested format and return the paths.

    PDF alongside PNG because journals want vector where possible, and a
    600 dpi raster where the figure contains a map.
    """
    written = []
    for ext in formats:
        path = figure_path(name, out_dir, ext)
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return written
