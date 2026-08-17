"""Moving-window patch generation and logit stitching.

A tile is larger than the network's input, so it is cut into overlapping
patches whose predictions are then merged. Two things make the seams
disappear:

**Inner-fraction blending.** Only the central ``inner_fraction`` of each
patch is written back. SegFormer predictions degrade near a patch border
because the receptive field is truncated there, so the margin is discarded
in favour of a neighbouring patch's centre — except at the tile edge, where
there is no neighbour and the margin is kept.

**Logit averaging, not vote averaging.** Overlapping regions accumulate raw
logits and are argmaxed once at the end. Averaging hard labels instead
would throw away confidence and produce blocky artefacts along the overlap.

Both the nationwide inference and the tile-level evaluation in the paper use
this module, so their numbers are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Patch", "plan_patches", "LogitAccumulator", "normalise"]


@dataclass(frozen=True)
class Patch:
    """One patch, with the sub-window of it that gets written back.

    Attributes:
        row_start, row_end, col_start, col_end: Patch extent in tile pixels.
        ir_start, ir_end, ic_start, ic_end: The kept (inner) region, in
            patch-local pixels.
    """

    row_start: int
    row_end: int
    col_start: int
    col_end: int
    ir_start: int
    ir_end: int
    ic_start: int
    ic_end: int

    @property
    def height(self) -> int:
        return self.row_end - self.row_start

    @property
    def width(self) -> int:
        return self.col_end - self.col_start


def _plan_axis(
    size: int, patch_size: int, stride: int, margin: int, min_coverage: float
) -> list[tuple[int, int, int, int]]:
    """Lay out one axis as ``(start, end, inner_start, inner_end)`` spans.

    The last patch is anchored against the far edge rather than left
    wherever the stride happens to land, and each patch's inner region
    begins exactly where its predecessor's ended. Together these guarantee
    the inner regions tile the axis exactly once for *any* size.

    The naive version — striding to the end and dropping any final patch
    shorter than ``min_coverage`` — leaves a gap whenever
    ``size % stride`` falls between ``patch_size - 2 * margin`` and
    ``patch_size * min_coverage``. Those pixels get no prediction at all
    and silently come out as background. It does not bite at the standard
    5000 px tile, but it does for VRT windows at a mosaic edge and for
    states with irregular tile sizes.

    Returns:
        Spans in ascending order, or empty if the axis is too short to
        predict at all.
    """
    if size < patch_size * min_coverage:
        return []
    if size <= patch_size:
        return [(0, size, 0, size)]

    starts = list(range(0, size - patch_size + 1, stride))
    if starts[-1] + patch_size < size:
        starts.append(size - patch_size)

    spans: list[tuple[int, int, int, int]] = []
    prev_inner_end_abs = 0
    for i, start in enumerate(starts):
        end = start + patch_size
        inner_start = 0 if i == 0 else prev_inner_end_abs - start
        inner_end = (size - start) if i == len(starts) - 1 else (patch_size - margin)
        spans.append((start, end, inner_start, inner_end))
        prev_inner_end_abs = start + inner_end
    return spans


def plan_patches(
    tile_height: int,
    tile_width: int,
    patch_size: int = 512,
    inner_fraction: float = 0.7,
    min_coverage: float = 0.5,
) -> list[Patch]:
    """Lay out overlapping patches covering a tile exactly once.

    Args:
        tile_height: Tile height in pixels.
        tile_width: Tile width in pixels.
        patch_size: Square patch side, pixels.
        inner_fraction: Fraction of each patch kept. 0.7 discards a 15 %
            margin on every side, where the truncated receptive field makes
            predictions less reliable.
        min_coverage: A tile shorter than this fraction of ``patch_size``
            in either dimension is skipped entirely — a 20-pixel sliver is
            mostly padding and its prediction is not trustworthy.

    Returns:
        Patches in row-major order, whose inner regions partition the tile.
        Empty if the tile is too small to predict.
    """
    margin = int(patch_size * (1 - inner_fraction) / 2)
    stride = patch_size - 2 * margin
    if stride <= 0:
        raise ValueError(
            f"inner_fraction={inner_fraction} leaves no stride at patch_size={patch_size}"
        )

    rows = _plan_axis(tile_height, patch_size, stride, margin, min_coverage)
    cols = _plan_axis(tile_width, patch_size, stride, margin, min_coverage)

    return [
        Patch(
            row_start=r_start,
            row_end=min(r_end, tile_height),
            col_start=c_start,
            col_end=min(c_end, tile_width),
            ir_start=ir_start,
            ir_end=ir_end,
            ic_start=ic_start,
            ic_end=ic_end,
        )
        for r_start, r_end, ir_start, ir_end in rows
        for c_start, c_end, ic_start, ic_end in cols
    ]


class LogitAccumulator:
    """Sums patch logits into a full-tile array and reduces them once.

    Kept separate from the inference loop so it can be unit-tested without
    a GPU or a model.
    """

    def __init__(self, tile_height: int, tile_width: int, n_classes: int = 2):
        self.shape = (tile_height, tile_width)
        self._sum = np.zeros((n_classes, tile_height, tile_width), dtype=np.float32)
        self._count = np.zeros((tile_height, tile_width), dtype=np.uint16)

    def add(self, patch: Patch, logits: np.ndarray) -> None:
        """Accumulate one patch's logits over its inner region.

        Args:
            patch: The patch these logits belong to.
            logits: ``(n_classes, patch_size, patch_size)``.
        """
        h, w = self.shape
        ar_start = patch.row_start + patch.ir_start
        ar_end = min(patch.row_start + patch.ir_end, h)
        ac_start = patch.col_start + patch.ic_start
        ac_end = min(patch.col_start + patch.ic_end, w)

        # The patch may have been padded; clip the source slice to match.
        pr_end = patch.ir_start + (ar_end - ar_start)
        pc_end = patch.ic_start + (ac_end - ac_start)

        self._sum[:, ar_start:ar_end, ac_start:ac_end] += logits[
            :, patch.ir_start : pr_end, patch.ic_start : pc_end
        ]
        self._count[ar_start:ar_end, ac_start:ac_end] += 1

    @property
    def valid(self) -> np.ndarray:
        """Pixels covered by at least one patch."""
        return self._count > 0

    def mean_logits(self) -> np.ndarray:
        """Per-pixel mean logits. Uncovered pixels are zero, not NaN."""
        safe = np.maximum(self._count.astype(np.float32), 1.0)
        return self._sum / safe[np.newaxis, :, :]

    def prediction(self) -> np.ndarray:
        """Hard class labels from the averaged logits."""
        return np.argmax(self.mean_logits(), axis=0).astype(np.uint8)

    def uncertainty(self) -> np.ndarray:
        """``1 - max softmax probability`` per pixel.

        Computed with the max-subtraction trick so that large logits cannot
        overflow ``exp``.
        """
        logits = self.mean_logits()
        shifted = logits - logits.max(axis=0, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.clip(exp.sum(axis=0, keepdims=True), 1e-12, None)
        return (1.0 - probs.max(axis=0)).astype(np.float32)


def normalise(image: np.ndarray) -> np.ndarray:
    """Scale 8-bit imagery to [-1, 1], matching training.

    Training used ``albumentations.Normalize(mean=0.5, std=0.5)`` with the
    default ``max_pixel_value=255``, i.e. ``(x / 255 - 0.5) / 0.5``. Applying
    anything else at inference time silently shifts the input distribution,
    which is one of the easier ways to lose several points of IoU without
    any error being raised.

    NaNs become 0 before scaling, so nodata reads as mid-grey rather than
    poisoning the patch.
    """
    return (np.nan_to_num(image, nan=0.0).astype(np.float32) / 255.0 - 0.5) / 0.5
