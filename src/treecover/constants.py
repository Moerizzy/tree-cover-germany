"""Raster class codes — the single source of truth for the whole pipeline.

The map is binary: a pixel is tree or it is not. Import these names rather
than writing the numbers locally, so a change here reaches every stage.

.. note::
   Earlier versions of this project also carried a six-code scheme that
   split trees outside forests into tree / patch / linear. That
   classification belongs to a different paper (Lucas et al. 2025) and has
   been removed from this repository. A raster containing codes above 1 is
   from that work, not from this pipeline —
   :func:`validate_prediction_codes` says so rather than silently reading
   the extra codes as background.
"""

from __future__ import annotations

import numpy as np

#: Not tree.
PRED_BACKGROUND = 0
#: Tree.
PRED_TREE = 1

#: Every valid value in a prediction raster, nodata aside.
VALID_PREDICTION_CODES = (PRED_BACKGROUND, PRED_TREE)

#: Value marking pixels outside the land/state mask, or without data.
NODATA = 255

CLASS_NAMES = {
    PRED_BACKGROUND: "background",
    PRED_TREE: "tree",
}

#: Used consistently across QGIS styles and figures.
CLASS_COLORS = {
    PRED_BACKGROUND: "#ffffff",
    PRED_TREE: "#1b7837",
}


def to_binary_tree(prediction: np.ndarray, nodata: int = NODATA) -> np.ndarray:
    """Normalise a prediction raster to a strict 0/1 mask.

    Nodata pixels stay nodata so callers can mask them out before computing
    metrics or areas.

    Args:
        prediction: Raster of prediction codes.
        nodata: Value marking invalid pixels.

    Returns:
        ``uint8`` array containing only :data:`PRED_BACKGROUND`,
        :data:`PRED_TREE` and ``nodata``.
    """
    out = np.full(prediction.shape, PRED_BACKGROUND, dtype=np.uint8)
    out[prediction == PRED_TREE] = PRED_TREE
    out[prediction == nodata] = nodata
    return out


def validate_prediction_codes(prediction: np.ndarray, nodata: int = NODATA) -> None:
    """Raise if ``prediction`` holds values this pipeline never writes.

    Guards against feeding in a raster from the trees-outside-forests
    classification, whose codes 3–6 would otherwise be counted as
    background and quietly deflate every tree-cover figure.

    Raises:
        ValueError: If unexpected class codes are present.
    """
    present = set(np.unique(prediction).tolist()) - {nodata}
    unexpected = present - set(VALID_PREDICTION_CODES)
    if not unexpected:
        return
    raise ValueError(
        f"Prediction raster contains class code(s) {sorted(unexpected)}; this "
        f"pipeline only writes {list(VALID_PREDICTION_CODES)}. Codes above 1 come "
        "from the trees-outside-forests classification of Lucas et al. (2025), "
        "which is not part of this repository."
    )
