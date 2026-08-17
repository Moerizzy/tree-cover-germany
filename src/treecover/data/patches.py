"""Sliding-window patch extraction.

Patches are *metadata*, not files: each is a window into an observation's
raster. A 1 km tile at 20 cm is 5000 × 5000 pixels, so materialising every
512 × 512 patch with 50 % overlap would multiply the training set on disk
by four for no benefit. Reading the window at load time costs a seek.

Two patches are dropped at extraction time rather than at training time:
those that are entirely nodata, and — optionally — those with no canopy at
all. The latter is a real trade-off, documented on
:func:`extract_patches`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import rasterio
from rasterio.windows import Window

from .observations import Observation

logger = logging.getLogger(__name__)

__all__ = ["Patch", "extract_patches", "extract_all", "MASK_NODATA"]

#: Nodata in the label masks produced by the earlier inference stage.
MASK_NODATA = 255


@dataclass
class Patch:
    """One training patch: a window into an observation."""

    region_id: str
    patch_id: str
    tile_id: str
    date: str
    split: str
    col_off: int
    row_off: int
    width: int
    height: int
    tree_cover_pct: float

    @property
    def window(self) -> Window:
        return Window(self.col_off, self.row_off, self.width, self.height)

    def as_row(self) -> dict:
        return asdict(self)


def extract_patches(
    observation: Observation,
    patch_size: int = 512,
    stride: int = 256,
    min_tree_cover_pct: float = 0.0,
    max_nodata_frac: float = 1.0,
) -> list[Patch]:
    """Lay a sliding window over one observation.

    Args:
        observation: The tile-date to cut up.
        patch_size: Square patch side, pixels.
        stride: Step between patches. ``patch_size // 2`` gives 50 % overlap,
            which augments the training set and means no ground feature only
            ever appears at a patch border.
        min_tree_cover_pct: Drop patches below this canopy fraction.
            **Leave at 0.** Treeless patches are what teach the model not to
            hallucinate canopy on bare fields and roofs; filtering them out
            raises training IoU and lowers real-world precision.
        max_nodata_frac: Drop patches whose label is more than this fraction
            nodata. 1.0 keeps everything except all-nodata patches.

    Returns:
        Patches in row-major order. Empty if the raster is smaller than one
        patch — partial windows are not padded, because a padded label is
        not ground truth.
    """
    if stride <= 0:
        raise ValueError("stride must be positive")

    patches: list[Patch] = []
    with rasterio.open(observation.mask_path) as mask_src:
        height, width = mask_src.height, mask_src.width

        rows = max(0, (height - patch_size) // stride + 1)
        cols = max(0, (width - patch_size) // stride + 1)
        if rows == 0 or cols == 0:
            logger.debug("%s is %dx%d, smaller than one %d patch — skipped",
                         observation.obs_id, width, height, patch_size)
            return []

        for i in range(rows):
            for j in range(cols):
                row_off = i * stride
                col_off = j * stride
                window = Window(col_off, row_off, patch_size, patch_size)
                mask = mask_src.read(1, window=window)

                valid = mask != MASK_NODATA
                n_valid = int(valid.sum())
                if n_valid == 0:
                    continue
                if 1.0 - n_valid / mask.size > max_nodata_frac:
                    continue

                tree_cover = 100.0 * int((mask[valid] != 0).sum()) / n_valid
                if tree_cover < min_tree_cover_pct:
                    continue

                patches.append(
                    Patch(
                        region_id=observation.obs_id,
                        patch_id=f"{observation.obs_id}_p{len(patches):04d}",
                        tile_id=observation.tile_id,
                        date=observation.date,
                        split=observation.split,
                        col_off=col_off,
                        row_off=row_off,
                        width=patch_size,
                        height=patch_size,
                        tree_cover_pct=round(tree_cover, 2),
                    )
                )
    return patches


def extract_all(
    observations: list[Observation],
    patch_size: int = 512,
    stride: int = 256,
    min_tree_cover_pct: float = 0.0,
    max_nodata_frac: float = 1.0,
    progress: bool = True,
) -> list[Patch]:
    """Extract patches from every observation.

    A raster that cannot be opened is logged and skipped — one corrupt tile
    out of hundreds must not abort an hour of extraction.
    """
    iterator = observations
    if progress:
        from tqdm import tqdm

        iterator = tqdm(observations, unit="obs", desc="Extracting patches")

    patches: list[Patch] = []
    failed = 0
    for observation in iterator:
        try:
            patches.extend(
                extract_patches(
                    observation, patch_size, stride, min_tree_cover_pct, max_nodata_frac
                )
            )
        except rasterio.RasterioIOError as exc:
            failed += 1
            logger.warning("Cannot read %s: %s", observation.mask_path, exc)

    if failed:
        logger.warning("%d observation(s) failed to read", failed)
    return patches


def region_vrt_map(observations: list[Observation]) -> dict[str, dict[str, str]]:
    """Map each observation to the rasters its patches read from.

    The keys match ``Patch.region_id`` and the structure is what
    :class:`treecover.data.dataset.TreeCoverDataset` expects. Despite the
    name these are plain rasters here; the field names are kept because the
    dataset accepts a VRT just as well, and mosaicked states need one.
    """
    mapping: dict[str, dict[str, str]] = {}
    for observation in observations:
        entry = {
            "rgbi_vrt": str(observation.image_path),
            "mask_vrt": str(observation.mask_path),
        }
        if observation.ndsm_path is not None:
            entry["ndsm_vrt"] = str(observation.ndsm_path)
        mapping[observation.obs_id] = entry
    return mapping


def split_map(observations: list[Observation]) -> dict[str, list[str]]:
    """Group observation ids by split, in the form stage 4 reads."""
    groups: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for observation in observations:
        groups.setdefault(observation.split, []).append(observation.obs_id)
    return {
        "train_regions": sorted(groups.get("train", [])),
        "val_regions": sorted(groups.get("val", [])),
        "test_regions": sorted(groups.get("test", [])),
    }
