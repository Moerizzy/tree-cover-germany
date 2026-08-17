"""Shared styling for the paper figures.

The figure scripts themselves live in ``figures/`` at the repository root —
one script per figure, each runnable on its own. This package holds only
what they share, so a colour or a font size is defined once.
"""

from .style import (
    DIVERGING,
    DIVERGING_CVD,
    HEIGHT,
    MASK_CMAP,
    NODATA_GREY,
    OKABE_ITO,
    PRODUCT_COLORS,
    PRODUCT_LABELS,
    SEASON_COLORS,
    SEQUENTIAL,
    SIZES,
    TREE_GREEN,
    apply_style,
    diverging_norm,
    figure_path,
    save,
)

__all__ = [
    "apply_style",
    "save",
    "figure_path",
    "diverging_norm",
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
    "load_validation_metrics",
    "STATES",
    "STATE_NAMES",
    "SIZES",
    "TilePair",
    "pair_observations",
    "select_examples",
    "tile_tree_cover_pct",
]

from .training_examples import (  # noqa: E402
    TilePair,
    pair_observations,
    select_examples,
    tile_tree_cover_pct,
)
from .validation_data import STATE_NAMES, STATES, load_validation_metrics  # noqa: E402
