"""The training dataset and its augmentation pipeline.

Patches are not stored as files. Each is a window into a per-observation
VRT, so the 20 cm imagery stays on disk in its original tiles and only the
requested window is read. A tile-observation is one acquisition of one tile;
the same label mask is shared by every date of a tile, which is what makes
multi-season training possible at all.

Four input configurations exist, from the paper's ablation. The published
model is ``RGB``; the others need bands or a height model that most states
do not publish, which is why the transferable model is the RGB one even
though RGBI+nDSM scored higher where it was available.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

from .seasons import date_encoding

logger = logging.getLogger(__name__)

__all__ = [
    "DATA_CONFIGS",
    "n_input_channels",
    "TreeCoverDataset",
    "training_augmentation",
    "validation_augmentation",
]

#: Input configuration -> base band count (before any date-encoding channels).
DATA_CONFIGS = {
    "RGB": 3,
    "RGBI": 4,
    "RGB_nDSM": 4,
    "RGBI_nDSM": 5,
}


def n_input_channels(data_config: str, include_date_encoding: bool = False) -> int:
    """Channel count the model must accept for a configuration."""
    if data_config not in DATA_CONFIGS:
        raise ValueError(
            f"Unknown data_config {data_config!r}. Choose from {sorted(DATA_CONFIGS)}."
        )
    return DATA_CONFIGS[data_config] + (2 if include_date_encoding else 0)


def _as_window(patch: dict) -> Window:
    """Build a rasterio Window from whichever form the patch table stores.

    Patch tables round-tripped through CSV lose tuples, so a window may
    arrive as a string, as four columns, or as a real sequence.
    """
    raw = patch.get("window")
    if isinstance(raw, str):
        # A CSV round-trip turns (col, row, w, h) into its repr.
        numbers = [int(float(n)) for n in raw.strip("()[] ").split(",")]
        return Window(*numbers)
    if raw is not None and not isinstance(raw, str):
        return Window(*raw)
    return Window(
        patch["col_off"], patch["row_off"], patch["width"], patch["height"]
    )


class TreeCoverDataset:
    """Patches read on demand from per-observation VRTs.

    Args:
        patches: Patch records. Each needs ``region_id`` and a window; a
            ``date`` is required only when ``include_date_encoding`` is set.
        region_vrts: ``region_id`` (as ``str``) -> ``{"rgbi_vrt", "mask_vrt",
            "ndsm_vrt"}`` paths.
        data_config: One of :data:`DATA_CONFIGS`.
        transform: Albumentations pipeline. When ``None`` the raw arrays are
            returned as tensors without normalisation — useful for
            inspecting inputs, not for training.
        include_date_encoding: Append the two cyclical day-of-year channels.

    Note:
        Not a subclass of :class:`torch.utils.data.Dataset` — ``DataLoader``
        duck-types on ``__len__``/``__getitem__``, and staying independent
        means the patch table can be inspected without importing torch.
    """

    def __init__(
        self,
        patches: list[dict],
        region_vrts: dict[str, dict],
        data_config: str = "RGB",
        transform: Any = None,
        include_date_encoding: bool = False,
    ):
        self.patches = patches
        self.region_vrts = region_vrts
        self.data_config = data_config
        self.transform = transform
        self.include_date_encoding = include_date_encoding
        self.n_channels = n_input_channels(data_config, include_date_encoding)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        import torch

        patch = self.patches[idx]
        vrts = self.region_vrts[str(patch["region_id"])]
        window = _as_window(patch)

        with rasterio.open(vrts["rgbi_vrt"]) as src:
            rgbi = src.read(window=window).astype(np.float32)
        with rasterio.open(vrts["mask_vrt"]) as src:
            mask = src.read(1, window=window)

        # Labels come from an earlier model's output, which may carry class
        # codes rather than 0/1. Anything non-zero is tree.
        mask = (mask != 0).astype(np.uint8)

        image = self._assemble_channels(rgbi, vrts, window)
        image = np.nan_to_num(image, nan=0.0)

        if self.include_date_encoding:
            image = np.vstack(
                [image, date_encoding(patch.get("date"), image.shape[1], image.shape[2])]
            )

        image = np.transpose(image, (1, 2, 0))  # albumentations wants HWC

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            return augmented["image"], augmented["mask"].long()

        return (
            torch.from_numpy(image).permute(2, 0, 1),
            torch.from_numpy(mask).long(),
        )

    def _assemble_channels(self, rgbi: np.ndarray, vrts: dict, window: Window) -> np.ndarray:
        """Select and stack the bands for this configuration."""
        if self.data_config == "RGB":
            return rgbi[:3]
        if self.data_config == "RGBI":
            return rgbi

        with rasterio.open(vrts["ndsm_vrt"]) as src:
            ndsm = src.read(1, window=window).astype(np.float32)
        # Negative heights are interpolation artefacts, not depressions.
        # They also broke RandomGamma augmentation with NaNs, which is why
        # that transform is absent from the training pipeline.
        ndsm = np.clip(np.nan_to_num(ndsm, nan=0.0), 0, None)

        base = rgbi[:3] if self.data_config == "RGB_nDSM" else rgbi
        return np.vstack([base, ndsm[np.newaxis, :, :]])


def training_augmentation(n_channels: int):
    """Augmentation for training.

    Geometric transforms only, plus mild brightness/contrast jitter and
    coarse dropout. Deliberately no ``RandomGamma``: it produces NaNs on the
    nDSM channel, which carries physical heights rather than 0–255 values.

    ``Normalize(mean=0.5, std=0.5)`` maps to [-1, 1]; inference must use the
    identical scaling — see :func:`treecover.inference.tiling.normalise`.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.0625, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
            A.Normalize(mean=[0.5] * n_channels, std=[0.5] * n_channels),
            ToTensorV2(),
        ]
    )


def validation_augmentation(n_channels: int):
    """Normalisation only — validation must measure the model, not the jitter."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.Normalize(mean=[0.5] * n_channels, std=[0.5] * n_channels),
            ToTensorV2(),
        ]
    )
