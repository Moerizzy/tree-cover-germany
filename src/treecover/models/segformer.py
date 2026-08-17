"""SegFormer construction and checkpoint loading.

Shared by training and inference so that a model is built the same way in
both — a mismatch here is silent and shows up only as degraded accuracy.

The published model is ``mit-b5`` initialised from the `restor/tcd-segformer`
tree-crown-delineation weights and fine-tuned on RGB. Channel counts other
than 3 are supported for the RGBI and RGBI+nDSM ablations reported in the
paper.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation

logger = logging.getLogger(__name__)

__all__ = [
    "create_segformer",
    "load_checkpoint",
    "extract_logits",
    "infer_backbone_from_path",
    "N_CLASSES",
    "TCD_PRETRAINED",
]

#: Binary segmentation: background and tree.
N_CLASSES = 2

#: Tree-crown-delineation weights used to initialise the published model.
TCD_PRETRAINED = "restor/tcd-segformer-mit-b5"

_BACKBONE_RE = re.compile(r"segformer[_-]?b([0-5])")


def create_segformer(
    n_channels: int = 3,
    backbone: str = "nvidia/mit-b5",
    use_tcd_pretrained: bool = False,
) -> SegformerForSemanticSegmentation:
    """Build a SegFormer adapted to ``n_channels`` input bands.

    For more than three channels the first patch-embedding convolution is
    replaced. RGB weights are copied across, and each extra channel is
    seeded with the *mean* of the RGB weights rather than random values —
    an unseeded channel starts as noise and measurably slows convergence.

    Args:
        n_channels: Input bands. 3 = RGB, 4 = RGBI, 5 = RGBI + nDSM.
        backbone: Hugging Face model id, ``nvidia/mit-b0`` … ``mit-b5``.
        use_tcd_pretrained: Start from :data:`TCD_PRETRAINED` instead of the
            plain ImageNet backbone. This is what the published model used.

    Returns:
        The model, on CPU and in training mode.
    """
    source = TCD_PRETRAINED if use_tcd_pretrained else backbone
    logger.info("Building SegFormer from %s with %d input channel(s)", source, n_channels)
    model = SegformerForSemanticSegmentation.from_pretrained(
        source, num_labels=N_CLASSES, ignore_mismatched_sizes=True
    )

    if n_channels == 3:
        return model

    old_conv = model.segformer.encoder.patch_embeddings[0].proj
    new_conv = nn.Conv2d(
        in_channels=n_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
    )
    with torch.no_grad():
        copy_n = min(3, n_channels)
        new_conv.weight[:, :copy_n] = old_conv.weight[:, :copy_n].clone()
        if n_channels > 3:
            rgb_mean = old_conv.weight.mean(dim=1, keepdim=True)
            for i in range(3, n_channels):
                new_conv.weight[:, i : i + 1] = rgb_mean.clone()
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    model.segformer.encoder.patch_embeddings[0].proj = new_conv
    return model


def extract_logits(outputs) -> torch.Tensor:
    """Unwrap logits from whatever the model returned.

    Transformers returns a ``SemanticSegmenterOutput``; a scripted or
    unwrapped model returns a bare tensor.
    """
    if isinstance(outputs, torch.Tensor):
        return outputs
    return getattr(outputs, "logits", outputs)


def load_checkpoint(
    model: nn.Module, checkpoint_path: str | Path, device: str | torch.device = "cpu"
) -> nn.Module:
    """Load weights, tolerating the checkpoint layouts this project produced.

    Accepts a bare ``state_dict``, or a dict wrapping one under
    ``state_dict`` / ``model_state_dict``, and strips a ``module.`` prefix
    left behind by ``DataParallel``.

    Args:
        model: Model whose architecture matches the checkpoint.
        checkpoint_path: ``.pth`` file.
        device: Map location for the load.

    Returns:
        ``model``, with weights loaded in place.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break

    if any(k.startswith("module.") for k in checkpoint):
        checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint)
    return model


def infer_backbone_from_path(checkpoint_path: str | Path, default: str = "nvidia/mit-b5") -> str:
    """Guess the backbone from a checkpoint filename.

    The training runs encode it in the name, e.g.
    ``segformer_b5_tcd_..._RGB_3ch_nodateenc_bs16_lr1e-04_adamw.pth`` →
    ``nvidia/mit-b5``. Falls back to ``default`` when the name says nothing.
    """
    match = _BACKBONE_RE.search(Path(checkpoint_path).name.lower())
    return f"nvidia/mit-b{match.group(1)}" if match else default
