"""Comparing model predictions against the LiDAR reference.

Two resolutions meet here: the model predicts at 20 cm, the LiDAR CHM is
built at 1 m. The model is reduced to the reference grid by **majority
vote** over each 5 × 5 block, not by nearest-neighbour sampling — a single
pixel cannot represent whether a square metre is canopy.

Predictions are checked against the expected class codes before scoring.
Feeding in a raster from a different classification would otherwise have
its extra codes counted as background, quietly deflating recall.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

from treecover.constants import NODATA, PRED_TREE, validate_prediction_codes

__all__ = [
    "BinaryMetrics",
    "majority_vote_downsample",
    "binary_metrics",
    "reduce_prediction_to_reference",
    "aggregate",
    "by_stratum",
    "score_zero_reference",
]


@dataclass
class BinaryMetrics:
    """Confusion-matrix-derived metrics for the tree class."""

    iou: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    tp: int
    fp: int
    fn: int
    tn: int
    n_valid: int
    reference_cover_pct: float
    predicted_cover_pct: float

    def as_dict(self) -> dict:
        return asdict(self)


def majority_vote_downsample(array: np.ndarray, factor: int) -> np.ndarray:
    """Reduce a label raster by ``factor`` using the per-block mode.

    Args:
        array: 2-D label array.
        factor: Block size. 5 takes 20 cm predictions to a 1 m grid.

    Returns:
        The reduced array. Trailing rows/columns that do not fill a whole
        block are dropped rather than partially averaged.
    """
    if factor <= 1:
        return array
    height = array.shape[0] // factor * factor
    width = array.shape[1] // factor * factor
    trimmed = array[:height, :width]
    blocks = trimmed.reshape(height // factor, factor, width // factor, factor)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(height // factor, width // factor, -1)
    return stats.mode(blocks, axis=2, keepdims=False).mode.astype(array.dtype)


def reduce_prediction_to_reference(
    prediction: np.ndarray,
    reference_shape: tuple[int, int],
    factor: int,
    strict_codes: bool = True,
) -> np.ndarray:
    """Bring a model prediction onto the reference grid as a binary mask.

    Args:
        prediction: Model output, ``*_pred.tif``.
        reference_shape: Shape of the LiDAR mask to match.
        factor: Resolution ratio, e.g. 5 for 20 cm vs 1 m.
        strict_codes: Reject rasters holding codes this pipeline never
            writes, instead of counting them as background.

    Returns:
        A boolean array of ``reference_shape``. Missing rows/columns after
        downsampling are padded with False rather than raising, because a
        one-pixel shape mismatch between a CHM and a prediction window is
        routine.
    """
    if strict_codes:
        validate_prediction_codes(prediction)

    reduced = majority_vote_downsample(prediction, factor)
    out = np.zeros(reference_shape, dtype=bool)
    rows = min(reduced.shape[0], reference_shape[0])
    cols = min(reduced.shape[1], reference_shape[1])
    out[:rows, :cols] = reduced[:rows, :cols] == PRED_TREE
    return out


def binary_metrics(
    reference: np.ndarray, predicted: np.ndarray, valid: np.ndarray | None = None
) -> BinaryMetrics:
    """Confusion-matrix metrics for the tree class over the valid pixels.

    Args:
        reference: Boolean reference tree mask.
        predicted: Boolean predicted tree mask, same shape.
        valid: Pixels to score. Defaults to all. Pass the inverse of the
            LiDAR nodata mask so occluded cells do not count as background.

    Returns:
        The metrics. Where a metric is undefined — no positives in either
        mask, i.e. both agree there are no trees — it is reported as 1.0
        (perfect agreement) rather than 0.0 or NaN. This matters: treeless
        validation boxes are common, and scoring them 0 would drag the
        mean down for tiles the model got exactly right.
    """
    if reference.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {predicted.shape}")

    if valid is None:
        valid = np.ones(reference.shape, dtype=bool)
    ref = reference[valid].astype(bool)
    pred = predicted[valid].astype(bool)
    n = ref.size

    tp = int(np.sum(ref & pred))
    fp = int(np.sum(~ref & pred))
    fn = int(np.sum(ref & ~pred))
    tn = int(np.sum(~ref & ~pred))

    union = tp + fp + fn
    iou = 1.0 if union == 0 else tp / union
    denom_f1 = 2 * tp + fp + fn
    f1 = 1.0 if denom_f1 == 0 else 2 * tp / denom_f1
    precision = 1.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 1.0 if tp + fn == 0 else tp / (tp + fn)
    accuracy = 1.0 if n == 0 else (tp + tn) / n

    return BinaryMetrics(
        iou=iou, f1=f1, precision=precision, recall=recall, accuracy=accuracy,
        tp=tp, fp=fp, fn=fn, tn=tn, n_valid=n,
        reference_cover_pct=100.0 * ref.sum() / n if n else 0.0,
        predicted_cover_pct=100.0 * pred.sum() / n if n else 0.0,
    )


def score_zero_reference(
    frame: pd.DataFrame,
    reference_column: str = "lidar_tree_cover_pct",
    model_column: str = "model_tree_cover_pct",
    prefix: str = "overall_",
) -> pd.DataFrame:
    """Score the boxes where the reference reports no trees at all.

    IoU is undefined there — the union is empty — and a third of the
    validation boxes are in that state. Dropping them measures the model
    only where trees are, which is the easier half of the problem; the
    published numbers keep them and resolve the undefined case explicitly:

    * the model also finds nothing → a correct negative, every metric 1.0
    * the model finds trees → a false positive, IoU/F1/precision/recall
      0.0 and accuracy the share of background it did leave alone

    This is what makes the paper's mean IoU 0.844 rather than 0.771, so it
    is defined once here and used by both the figures and the summary
    table rather than being reimplemented in either.

    Args:
        frame: Per-sample metrics.
        reference_column: Reference cover in percent.
        model_column: Predicted cover in percent.
        prefix: Prefix of the metric columns, ``""`` when they are plain.

    Returns:
        The frame with those rows rescored. Rows with a non-empty
        reference are untouched.
    """
    if reference_column not in frame or model_column not in frame:
        return frame

    frame = frame.copy()
    empty = frame[reference_column] == 0
    correct_negative = empty & (frame[model_column] == 0)
    false_positive = empty & (frame[model_column] > 0)

    names = {name: f"{prefix}{name}" for name in
             ("iou", "f1", "precision", "recall", "accuracy")}
    for column in names.values():
        if column in frame.columns:
            frame.loc[correct_negative, column] = 1.0
    for name in ("iou", "f1", "precision", "recall"):
        column = names[name]
        if column in frame.columns:
            frame.loc[false_positive, column] = 0.0
    if names["accuracy"] in frame.columns:
        frame.loc[false_positive, names["accuracy"]] = (
            1.0 - frame.loc[false_positive, model_column] / 100.0
        )

    logger.info("Empty-reference boxes: %d correct negatives, %d false positives",
                int(correct_negative.sum()), int(false_positive.sum()))
    return frame


def aggregate(per_sample: pd.DataFrame) -> pd.DataFrame:
    """Summarise per-sample metrics two ways.

    **Per-sample mean ± std** weights every validation box equally, which is
    what the stratified tables report.

    **Pooled** sums the confusion matrices first and derives the metrics
    once, which weights every *pixel* equally. The headline IoU and F1 in
    the paper are the pooled figures; they are lower than the per-sample
    mean because large-canopy boxes dominate the pixel count.

    Args:
        per_sample: One row per validation sample, with at least the
            ``tp``/``fp``/``fn``/``tn`` columns.

    Returns:
        A tidy frame with a ``metric``, ``mean``, ``std`` and ``pooled``
        column.
    """
    metric_cols = ["iou", "f1", "precision", "recall", "accuracy"]
    present = [c for c in metric_cols if c in per_sample.columns]

    pooled_counts = {k: int(per_sample[k].sum()) for k in ("tp", "fp", "fn", "tn")}
    pooled = _from_counts(**pooled_counts)

    return pd.DataFrame(
        {
            "metric": present,
            "mean": [per_sample[c].mean() for c in present],
            "std": [per_sample[c].std() for c in present],
            "pooled": [pooled[c] for c in present],
            "n_samples": len(per_sample),
        }
    )


def _from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """Derive metrics from summed confusion-matrix counts."""
    union = tp + fp + fn
    total = tp + fp + fn + tn
    return {
        "iou": 1.0 if union == 0 else tp / union,
        "f1": 1.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn),
        "precision": 1.0 if tp + fp == 0 else tp / (tp + fp),
        "recall": 1.0 if tp + fn == 0 else tp / (tp + fn),
        "accuracy": 1.0 if total == 0 else (tp + tn) / total,
    }


def by_stratum(
    per_sample: pd.DataFrame, column: str, metrics: tuple[str, ...] = ("iou", "f1")
) -> pd.DataFrame:
    """Break metrics down by one stratification variable.

    Reports both the per-sample mean and the pooled value per level, plus
    the sample count — a level with four samples deserves less weight in
    interpretation than one with two hundred, and the count makes that
    visible.
    """
    if column not in per_sample.columns:
        raise KeyError(f"{column!r} not in the per-sample table")

    rows = []
    for level, group in per_sample.groupby(column, observed=True, dropna=True):
        pooled = _from_counts(*(int(group[k].sum()) for k in ("tp", "fp", "fn", "tn")))
        row = {column: level, "n_samples": len(group)}
        for metric in metrics:
            if metric in group.columns:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std()
            row[f"{metric}_pooled"] = pooled[metric]
        row["reference_cover_pct"] = group.get(
            "reference_cover_pct", pd.Series(dtype=float)
        ).mean()
        row["predicted_cover_pct"] = group.get(
            "predicted_cover_pct", pd.Series(dtype=float)
        ).mean()
        rows.append(row)

    return pd.DataFrame(rows).sort_values(column).reset_index(drop=True)


def mask_nodata(reference: np.ndarray, nodata_value: int = NODATA) -> np.ndarray:
    """Validity mask for a reference raster read from disk."""
    return reference != nodata_value
