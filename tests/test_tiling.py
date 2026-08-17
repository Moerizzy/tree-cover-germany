"""Tests for patch planning and logit stitching.

No model or GPU involved — this is pure geometry and array bookkeeping,
which is exactly where the four original inference scripts could have
drifted apart without anyone noticing.
"""

from __future__ import annotations

import numpy as np
import pytest

from treecover.inference.tiling import LogitAccumulator, normalise, plan_patches


def covered_mask(patches, height, width) -> np.ndarray:
    """Which pixels at least one patch writes back to."""
    mask = np.zeros((height, width), dtype=int)
    for p in patches:
        r0 = p.row_start + p.ir_start
        r1 = min(p.row_start + p.ir_end, height)
        c0 = p.col_start + p.ic_start
        c1 = min(p.col_start + p.ic_end, width)
        mask[r0:r1, c0:c1] += 1
    return mask


@pytest.mark.parametrize(
    ("height", "width"),
    [(1024, 1024), (1500, 1100), (5000, 5000), (700, 512), (920, 920), (513, 4999)],
)
def test_every_pixel_is_covered_exactly_once(height, width):
    """Full coverage with no gaps — a gap would leave background stripes."""
    patches = plan_patches(height, width, patch_size=512, inner_fraction=0.7)
    counts = covered_mask(patches, height, width)
    assert counts.min() >= 1, "uncovered pixels would predict as background"
    assert counts.max() == 1, "double-written pixels would be counted twice"


@pytest.mark.parametrize("size", range(256, 1400, 7))
def test_no_size_leaves_a_gap(size):
    """Regression: striding naively to the end leaves a hole for sizes where
    ``size % stride`` lands between the stride and half a patch — e.g. 920,
    where pixels 796..920 were never predicted and came out as background."""
    patches = plan_patches(size, size, patch_size=512, inner_fraction=0.7)
    counts = covered_mask(patches, size, size)
    assert counts.min() >= 1, f"gap at size {size}"
    assert counts.max() == 1, f"overlap at size {size}"


def test_patches_never_read_outside_their_own_extent():
    """inner_end must stay within the patch, or the accumulator slices past it."""
    for size in (600, 920, 1024, 1500, 5000):
        for p in plan_patches(size, size, patch_size=512, inner_fraction=0.7):
            assert 0 <= p.ir_start < p.ir_end <= 512
            assert 0 <= p.ic_start < p.ic_end <= 512


def test_edge_patches_keep_their_margin():
    """At the tile border there is no neighbour to supply a better centre."""
    patches = plan_patches(1024, 1024, patch_size=512, inner_fraction=0.7)
    top_left = next(p for p in patches if p.row_start == 0 and p.col_start == 0)
    assert top_left.ir_start == 0
    assert top_left.ic_start == 0

    interior = [p for p in patches if p.row_start > 0 and p.col_start > 0]
    assert interior, "expected at least one interior patch"
    assert all(p.ir_start > 0 and p.ic_start > 0 for p in interior)


def test_tiny_tile_yields_no_patches():
    """Below half a patch in both dimensions there is nothing worth predicting."""
    assert plan_patches(100, 100, patch_size=512) == []


def test_one_pixel_overhang_does_not_create_a_sliver_patch():
    """A tile one pixel wider than a patch gets a second full-size patch
    anchored at the edge, not a 1-pixel column of mostly padding."""
    patches = plan_patches(512, 513, patch_size=512, inner_fraction=0.7)
    assert all(p.width >= 512 * 0.5 for p in patches)
    assert covered_mask(patches, 512, 513).min() >= 1


def test_invalid_inner_fraction_is_rejected():
    with pytest.raises(ValueError, match="no stride"):
        plan_patches(1024, 1024, patch_size=512, inner_fraction=0.0)


def test_accumulator_averages_overlapping_logits():
    """Two patches disagreeing on a pixel must average, not last-write-wins."""
    acc = LogitAccumulator(4, 4, n_classes=2)
    patch_a = plan_patches(4, 4, patch_size=4, inner_fraction=1.0)[0]

    acc.add(patch_a, np.stack([np.full((4, 4), 1.0), np.full((4, 4), 3.0)]))
    acc.add(patch_a, np.stack([np.full((4, 4), 3.0), np.full((4, 4), 1.0)]))

    mean = acc.mean_logits()
    assert np.allclose(mean[0], 2.0)
    assert np.allclose(mean[1], 2.0)


def test_prediction_is_argmax_of_averaged_logits():
    acc = LogitAccumulator(2, 2, n_classes=2)
    patch = plan_patches(2, 2, patch_size=2, inner_fraction=1.0)[0]
    logits = np.zeros((2, 2, 2), dtype=np.float32)
    logits[1, 0, 0] = 5.0          # class 1 wins at (0, 0)
    logits[0, 1, 1] = 5.0          # class 0 wins at (1, 1)
    acc.add(patch, logits)
    assert acc.prediction().tolist() == [[1, 0], [0, 0]]


def test_uncovered_pixels_are_marked_invalid():
    acc = LogitAccumulator(10, 10)
    assert not acc.valid.any()
    assert np.all(acc.mean_logits() == 0), "uncovered pixels must be 0, not NaN"


def test_uncertainty_is_bounded_and_peaks_on_ties():
    """A tie between two classes is maximal uncertainty: 1 - 0.5 = 0.5."""
    acc = LogitAccumulator(2, 2, n_classes=2)
    patch = plan_patches(2, 2, patch_size=2, inner_fraction=1.0)[0]
    logits = np.zeros((2, 2, 2), dtype=np.float32)
    logits[1, 0, 0] = 50.0         # confident
    acc.add(patch, logits)

    unc = acc.uncertainty()
    assert np.all((unc >= 0) & (unc <= 1))
    assert unc[0, 0] < 1e-6, "a 50-logit margin should be near-certain"
    assert unc[1, 1] == pytest.approx(0.5), "an exact tie should be maximally uncertain"


def test_uncertainty_survives_large_logits():
    """The max-subtraction trick must keep exp() from overflowing."""
    acc = LogitAccumulator(1, 1, n_classes=2)
    patch = plan_patches(1, 1, patch_size=1, inner_fraction=1.0)[0]
    acc.add(patch, np.array([[[1000.0]], [[999.0]]], dtype=np.float32))
    assert np.isfinite(acc.uncertainty()).all()


def test_normalise_maps_byte_range_to_minus_one_to_one():
    """Must match albumentations.Normalize(mean=0.5, std=0.5) from training."""
    data = np.array([[[0, 127.5, 255]]], dtype=np.float32)
    out = normalise(data)
    assert out[0, 0, 0] == pytest.approx(-1.0)
    assert out[0, 0, 1] == pytest.approx(0.0)
    assert out[0, 0, 2] == pytest.approx(1.0)


def test_normalise_treats_nodata_as_mid_grey():
    """NaN must not poison the patch it sits in."""
    out = normalise(np.array([[[np.nan]]], dtype=np.float32))
    assert np.isfinite(out).all()
    assert out[0, 0, 0] == pytest.approx(-1.0)
