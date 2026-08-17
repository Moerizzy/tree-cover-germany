"""Tests for the validation metrics and the CHM → tree mask step."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from treecover.validation import (
    PointCloud,
    aggregate,
    binary_metrics,
    by_stratum,
    chm_to_tree_mask,
    create_chm,
    fill_small_holes,
    majority_vote_downsample,
)
from treecover.validation.metrics import reduce_prediction_to_reference


# ── metrics ──────────────────────────────────────────────────────────────────


def test_perfect_agreement_scores_one():
    mask = np.array([[True, False], [True, True]])
    m = binary_metrics(mask, mask)
    assert m.iou == 1.0 and m.f1 == 1.0 and m.accuracy == 1.0


def test_complete_disagreement_scores_zero_iou():
    ref = np.array([[True, True], [True, True]])
    m = binary_metrics(ref, ~ref)
    assert m.iou == 0.0 and m.tp == 0 and m.fn == 4


def test_both_empty_counts_as_perfect_not_zero():
    """A treeless box the model also calls treeless is a correct prediction.
    Scoring it 0 (or NaN) would drag down the mean for boxes we got right."""
    empty = np.zeros((4, 4), dtype=bool)
    m = binary_metrics(empty, empty)
    assert m.iou == 1.0 and m.f1 == 1.0 and m.precision == 1.0 and m.recall == 1.0


def test_false_positives_on_an_empty_reference_score_zero():
    """But predicting trees where there are none must not score 1.0."""
    ref = np.zeros((4, 4), dtype=bool)
    pred = np.ones((4, 4), dtype=bool)
    m = binary_metrics(ref, pred)
    assert m.iou == 0.0 and m.precision == 0.0
    assert m.recall == 1.0, "no positives to miss"


def test_invalid_pixels_are_excluded_entirely():
    """LiDAR occlusion must not count as background agreement."""
    ref = np.array([[True, True], [False, False]])
    pred = np.array([[True, False], [False, False]])
    valid = np.array([[True, False], [True, True]])
    m = binary_metrics(ref, pred, valid)
    assert m.n_valid == 3
    assert m.iou == 1.0, "the only disagreement sits on an invalid pixel"


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        binary_metrics(np.zeros((2, 2), bool), np.zeros((3, 3), bool))


# ── resolution reduction ─────────────────────────────────────────────────────


def test_majority_vote_takes_the_block_mode():
    arr = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.uint8)
    assert majority_vote_downsample(arr, 2).tolist() == [[1, 0], [1, 0]]


def test_majority_vote_drops_partial_trailing_blocks():
    arr = np.ones((5, 5), dtype=np.uint8)
    assert majority_vote_downsample(arr, 2).shape == (2, 2)


def test_reduction_from_binary_prediction():
    """20 cm prediction to a 1 m reference grid is a factor-5 majority vote."""
    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[:5, :5] = 1                      # one whole reference cell is tree
    out = reduce_prediction_to_reference(pred, (2, 2), factor=5)
    assert out.tolist() == [[True, False], [False, False]]


def test_reduction_rejects_a_raster_from_another_classification():
    """Codes above 1 come from the trees-outside-forests work, which is not
    part of this pipeline. Counting them as background would deflate recall
    without any error being raised."""
    foreign = np.full((10, 10), 4, dtype=np.uint8)
    with pytest.raises(ValueError, match="not part of this repository"):
        reduce_prediction_to_reference(foreign, (2, 2), factor=5)


def test_unknown_codes_can_be_tolerated_explicitly():
    foreign = np.full((10, 10), 4, dtype=np.uint8)
    out = reduce_prediction_to_reference(foreign, (2, 2), factor=5, strict_codes=False)
    assert not out.any(), "codes other than 1 are not tree"


def test_reduction_pads_rather_than_failing_on_a_one_pixel_mismatch():
    """A CHM and a prediction window differing by a pixel is routine."""
    pred = np.ones((10, 10), dtype=np.uint8)
    out = reduce_prediction_to_reference(pred, (3, 3), factor=5)
    assert out.shape == (3, 3)
    assert out[2, 2] == False  # noqa: E712 - padded, not predicted


# ── aggregation ──────────────────────────────────────────────────────────────


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "iou": [1.0, 0.5, 0.0],
            "f1": [1.0, 0.667, 0.0],
            "precision": [1.0, 0.5, 0.0],
            "recall": [1.0, 1.0, 0.0],
            "accuracy": [1.0, 0.75, 0.5],
            "tp": [10, 5, 0], "fp": [0, 5, 0], "fn": [0, 0, 10], "tn": [90, 90, 90],
            "tcd_bin": ["0-10%", "0-10%", "50-100%"],
            "reference_cover_pct": [10.0, 5.0, 10.0],
            "predicted_cover_pct": [10.0, 10.0, 0.0],
        }
    )


def test_pooled_differs_from_the_per_sample_mean():
    """Pooled weights pixels, the mean weights boxes. The paper reports both
    and they are not interchangeable."""
    summary = aggregate(_frame())
    iou = summary[summary["metric"] == "iou"].iloc[0]
    assert iou["mean"] == pytest.approx(0.5)
    # pooled: tp=15, fp=5, fn=10 -> 15/30
    assert iou["pooled"] == pytest.approx(0.5)
    assert summary["n_samples"].iloc[0] == 3


def test_by_stratum_reports_counts_alongside_metrics():
    out = by_stratum(_frame(), "tcd_bin")
    assert set(out["tcd_bin"]) == {"0-10%", "50-100%"}
    assert out.loc[out["tcd_bin"] == "0-10%", "n_samples"].iloc[0] == 2
    assert "iou_mean" in out.columns and "iou_pooled" in out.columns


def test_by_stratum_rejects_an_unknown_column():
    with pytest.raises(KeyError):
        by_stratum(_frame(), "not_a_column")


# ── CHM ──────────────────────────────────────────────────────────────────────


def test_chm_is_height_above_ground_not_above_sea_level():
    """A 10 m tree on a 100 m hill must come out as 10 m, not 110."""
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0, 20, n)
    y = rng.uniform(0, 20, n)
    terrain = 100.0

    ground = PointCloud(x, y, np.full(n, terrain), np.full(n, 2))
    canopy_mask = (x > 8) & (x < 12) & (y > 8) & (y < 12)
    canopy = PointCloud(
        x[canopy_mask], y[canopy_mask], np.full(canopy_mask.sum(), terrain + 10.0),
        np.full(canopy_mask.sum(), 5),
    )
    cloud = PointCloud.concatenate([ground, canopy])

    chm, transform = create_chm(cloud, (0, 0, 20, 20), resolution=1.0)
    assert chm is not None
    assert chm.max() == pytest.approx(10.0, abs=0.5)
    assert chm.min() == pytest.approx(0.0, abs=0.5)


def test_create_chm_returns_none_for_an_empty_cloud():
    empty = PointCloud(np.array([]), np.array([]), np.array([]), np.array([]))
    assert create_chm(empty, (0, 0, 10, 10)) == (None, None)


def test_tree_mask_thresholds_at_three_metres():
    chm = np.array([[0.0, 2.9], [3.1, 12.0]], dtype=np.float32)
    tree, nodata = chm_to_tree_mask(chm, max_hole_m2=0.0)
    assert tree.tolist() == [[False, False], [True, True]]
    assert not nodata.any()


def test_buildings_are_subtracted_from_the_reference():
    """LiDAR cannot tell a roof from a canopy by height alone."""
    chm = np.full((4, 4), 10.0, dtype=np.float32)
    buildings = np.zeros((4, 4), dtype=bool)
    buildings[0, 0] = True
    tree, _ = chm_to_tree_mask(chm, max_hole_m2=0.0, buildings=buildings)
    assert not tree[0, 0]
    assert tree[3, 3]


def test_small_holes_are_closed_in_both_directions():
    mask = np.ones((9, 9), dtype=bool)
    mask[4, 4] = False                    # 1 px hole in canopy -> filled
    speck = np.zeros((9, 9), dtype=bool)
    speck[4, 4] = True                    # 1 px canopy speck -> removed

    assert fill_small_holes(mask, max_hole_px=2).all()
    assert not fill_small_holes(speck, max_hole_px=2).any()


def test_large_holes_survive():
    mask = np.ones((12, 12), dtype=bool)
    mask[3:9, 3:9] = False                # 36 px, above the threshold
    assert not fill_small_holes(mask, max_hole_px=4)[5, 5]


def test_nodata_stays_out_of_the_tree_mask():
    chm = np.array([[np.nan, 10.0], [10.0, 10.0]], dtype=np.float32)
    tree, nodata = chm_to_tree_mask(chm, max_hole_m2=0.0)
    assert nodata[0, 0]
    assert not tree[0, 0], "nodata must not be scored as tree"
