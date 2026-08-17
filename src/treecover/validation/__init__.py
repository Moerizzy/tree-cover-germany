"""Validation against LiDAR-derived reference data.

Driven by ``scripts/07_validate.py``, which has three subcommands:

``sample``
    Stratified selection of validation locations, expanded to 25 m boxes.
    Run once — re-drawing changes every number that follows.
``reference``
    LiDAR download → CHM → binary reference tree mask per box. Long-running
    and resumable. A manual QGIS pass follows it, setting ``exclude = 1``
    where the ground changed between the LiDAR and the orthophoto.
``score``
    Model vs. reference: IoU, F1, precision, recall, per stratum.

Ported from ``08_full_validation_pipeline_state_{BB,BY}.ipynb`` and
``08b_validation_TOF_vs_LiDAR.ipynb``, which carried all three as parts 1–3
of one notebook.
"""

from .chm import (
    GROUND_CLASS,
    TREE_HEIGHT_THRESHOLD_M,
    PointCloud,
    chm_to_tree_mask,
    create_chm,
    fill_nodata_by_majority,
    fill_small_holes,
)
from .metrics import (
    BinaryMetrics,
    aggregate,
    binary_metrics,
    by_stratum,
    score_zero_reference,
    majority_vote_downsample,
    reduce_prediction_to_reference,
)
from .lidar import download_tile, load_points_for_bounds, read_point_cloud
from .sampling import (
    ELEV_BINS,
    ELEV_LABELS,
    SEASON_OF_MONTH,
    TCD_BINS,
    TCD_LABELS,
    SamplingResult,
    StratumReport,
    assign_bins,
    make_boxes,
    resolve_sample_ids,
    stratified_sample,
)

__all__ = [
    "PointCloud",
    "download_tile",
    "read_point_cloud",
    "load_points_for_bounds",
    "create_chm",
    "chm_to_tree_mask",
    "fill_small_holes",
    "fill_nodata_by_majority",
    "GROUND_CLASS",
    "TREE_HEIGHT_THRESHOLD_M",
    "BinaryMetrics",
    "binary_metrics",
    "majority_vote_downsample",
    "reduce_prediction_to_reference",
    "aggregate",
    "by_stratum",
    "score_zero_reference",
    "stratified_sample",
    "make_boxes",
    "resolve_sample_ids",
    "assign_bins",
    "SamplingResult",
    "StratumReport",
    "TCD_BINS",
    "TCD_LABELS",
    "ELEV_BINS",
    "ELEV_LABELS",
    "SEASON_OF_MONTH",
]
