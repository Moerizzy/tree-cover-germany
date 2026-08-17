"""Wall-to-wall tree cover mapping of Germany at 20 cm resolution.

Reference implementation for Lucas, Brandt & Waske, *Overcoming seasonal
heterogeneity in national aerial surveys: 20 cm resolution tree cover
mapping of Germany*.

The pipeline runs in six stages, each with a CLI under ``scripts/``:

===  ==========================  ==========================================
1    ``01_sample_tiles``         Stratified selection of training tiles
2    ``02_download_dataset``     Fetch DOP / nDSM / labels for those tiles
3    ``03_prepare_patches``      Sliding-window patch extraction
4    ``04_train``                SegFormer training, season-aware sampling
5    ``05_inference``            Nationwide moving-window inference
6    ``06_tile_statistics``      Tree cover area per tile, land/border masked
===  ==========================  ==========================================

Validation against LiDAR (``07_validate``), merging to publication tiles
(``08_merge_reproject``) and the paper figures (``figures/``) sit alongside.
"""

__version__ = "1.0.0"

from treecover.constants import (
    CLASS_COLORS,
    CLASS_NAMES,
    NODATA,
    PRED_BACKGROUND,
    PRED_TREE,
    to_binary_tree,
    validate_prediction_codes,
)

__all__ = [
    "__version__",
    "PRED_BACKGROUND",
    "PRED_TREE",
    "NODATA",
    "CLASS_NAMES",
    "CLASS_COLORS",
    "to_binary_tree",
    "validate_prediction_codes",
]
