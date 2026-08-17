"""Running the model over a tile.

This is the piece the four original ``04_run_inference_*.py`` scripts each
carried their own copy of. They differed only in where the imagery came
from — a local folder, a VRT mosaic, or an HTTP download — never in how the
network was applied. That difference now lives in
:mod:`treecover.inference.sources`; the inference itself happens once, here.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from treecover.models.segformer import (
    create_segformer,
    extract_logits,
    infer_backbone_from_path,
    load_checkpoint,
)

from .tiling import LogitAccumulator, Patch, normalise, plan_patches

logger = logging.getLogger(__name__)

__all__ = ["InferenceConfig", "TilePrediction", "Predictor"]


@dataclass
class InferenceConfig:
    """Settings for a nationwide inference run.

    Attributes:
        checkpoint: ``.pth`` weights file.
        backbone: Hugging Face backbone id. ``None`` infers it from the
            checkpoint filename.
        n_channels: Input bands the checkpoint expects. 3 = RGB (published
            model), 4 = RGBI, 5 = RGBI + nDSM.
        patch_size: Network input size, pixels.
        inner_fraction: Fraction of each patch kept when stitching.
        batch_size: Patches per forward pass.
        dataloader_workers: Worker processes prefetching patches. 0 keeps
            everything in the main process, which is easier to debug.
        device: ``"cuda"``, ``"cuda:1"``, ``"cpu"`` …
        amp: Use float16 autocast on CUDA. Roughly halves runtime with no
            measurable accuracy cost for this model.
        compute_uncertainty: Also return mean softmax uncertainty. Costs an
            extra pass over the tile.
        target_resolution_m: Resolution the model expects. Imagery at a
            different resolution is resampled by the source.
    """

    checkpoint: Path
    backbone: str | None = None
    n_channels: int = 3
    patch_size: int = 512
    inner_fraction: float = 0.7
    batch_size: int = 8
    dataloader_workers: int = 4
    device: str = "cuda"
    amp: bool = True
    compute_uncertainty: bool = True
    target_resolution_m: float = 0.20


@dataclass
class TilePrediction:
    """Result of running the model over one tile."""

    prediction: np.ndarray
    tree_cover_pct: float
    coverage_pct: float
    n_patches: int
    mean_uncertainty: float
    uncertainty_map: np.ndarray | None = None
    valid_mask: np.ndarray | None = None


class _PatchDataset(torch.utils.data.Dataset):
    """Serves padded patches to a DataLoader so reads overlap with compute."""

    def __init__(self, image: np.ndarray, patches: list[Patch], patch_size: int):
        self.image = image
        self.patches = patches
        self.patch_size = patch_size
        self.n_channels = image.shape[0]

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        p = self.patches[idx]
        out = np.zeros((self.n_channels, self.patch_size, self.patch_size), dtype=np.float32)
        src = self.image[:, p.row_start : p.row_end, p.col_start : p.col_end]
        out[:, : src.shape[1], : src.shape[2]] = src
        return torch.from_numpy(out), idx


class Predictor:
    """A loaded model plus the logic to apply it to a tile.

    Build once per process and reuse — constructing the model and moving it
    to the GPU costs seconds.

    Example::

        predictor = Predictor(InferenceConfig(checkpoint=Path("model.pth")))
        result = predictor.predict(rgb)          # (3, H, W) uint8
        print(result.tree_cover_pct)
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() or "cpu" in config.device else "cpu"
        )
        if str(self.device) != config.device:
            logger.warning("CUDA unavailable — falling back to %s", self.device)

        backbone = config.backbone or infer_backbone_from_path(config.checkpoint)
        model = create_segformer(n_channels=config.n_channels, backbone=backbone)
        load_checkpoint(model, config.checkpoint, self.device)
        model.eval().to(self.device)
        if self.device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        self.model = model
        logger.info("Loaded %s (%d channels) on %s", backbone, config.n_channels, self.device)

    @property
    def _use_amp(self) -> bool:
        return self.config.amp and self.device.type == "cuda"

    def predict(self, image: np.ndarray, return_maps: bool = False) -> TilePrediction:
        """Run moving-window inference over one tile.

        Args:
            image: ``(channels, height, width)``, raw 8-bit values. It is
                normalised here — do not pre-scale it.
            return_maps: Also return the per-pixel uncertainty and validity
                masks.

        Returns:
            A :class:`TilePrediction`. A tile too small for any patch yields
            an all-background prediction with ``n_patches == 0`` rather than
            raising, so one undersized tile cannot stop a nationwide run.
        """
        if image.ndim != 3:
            raise ValueError(f"expected (channels, height, width), got shape {image.shape}")
        if image.shape[0] != self.config.n_channels:
            raise ValueError(
                f"model expects {self.config.n_channels} channel(s), got {image.shape[0]}"
            )

        _, height, width = image.shape
        patches = plan_patches(
            height, width, self.config.patch_size, self.config.inner_fraction
        )
        if not patches:
            return TilePrediction(
                prediction=np.zeros((height, width), dtype=np.uint8),
                tree_cover_pct=0.0,
                coverage_pct=0.0,
                n_patches=0,
                mean_uncertainty=1.0,
                uncertainty_map=np.ones((height, width), np.float32) if return_maps else None,
                valid_mask=np.zeros((height, width), bool) if return_maps else None,
            )

        accumulator = LogitAccumulator(height, width)
        loader = torch.utils.data.DataLoader(
            _PatchDataset(normalise(image), patches, self.config.patch_size),
            batch_size=self.config.batch_size,
            num_workers=self.config.dataloader_workers,
            pin_memory=self.device.type == "cuda",
            prefetch_factor=2 if self.config.dataloader_workers > 0 else None,
            shuffle=False,
        )

        for batch, indices in loader:
            batch = batch.to(self.device, non_blocking=self.device.type == "cuda")
            if self.device.type == "cuda":
                batch = batch.contiguous(memory_format=torch.channels_last)

            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self._use_amp
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                logits = extract_logits(self.model(batch))
                # SegFormer emits logits at 1/4 resolution.
                if logits.shape[-2:] != (self.config.patch_size, self.config.patch_size):
                    logits = torch.nn.functional.interpolate(
                        logits,
                        size=(self.config.patch_size, self.config.patch_size),
                        mode="bilinear",
                        align_corners=False,
                    )
            logits_np = logits.float().cpu().numpy()

            for i, idx in enumerate(indices.tolist()):
                accumulator.add(patches[idx], logits_np[i])

            del batch, logits, logits_np

        prediction = accumulator.prediction()
        valid = accumulator.valid
        n_px = prediction.size

        want_uncertainty = self.config.compute_uncertainty or return_maps
        if want_uncertainty:
            uncertainty = accumulator.uncertainty()
            mean_uncertainty = (
                float(uncertainty[valid].mean()) if valid.any() else float(uncertainty.mean())
            )
        else:
            uncertainty, mean_uncertainty = None, 1.0

        return TilePrediction(
            prediction=prediction,
            tree_cover_pct=float((prediction == 1).sum() / n_px * 100.0) if n_px else 0.0,
            coverage_pct=float(valid.sum() / n_px * 100.0) if n_px else 0.0,
            n_patches=len(patches),
            mean_uncertainty=mean_uncertainty,
            uncertainty_map=uncertainty if return_maps else None,
            valid_mask=valid if return_maps else None,
        )
