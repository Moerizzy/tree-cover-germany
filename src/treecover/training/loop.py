"""The training loop.

Model selection is on **validation IoU of the tree class**, not on loss.
Loss is dominated by the background class — most pixels in most patches are
not tree — so a checkpoint chosen by loss can be worse at the thing the map
is for. Early stopping watches the same metric.

Validation IoU is computed by accumulating intersection and union counts
across the whole set and dividing once, rather than averaging per-batch
IoU. Per-batch averaging over-weights batches that happen to contain little
canopy, where a handful of pixels swing the ratio.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["TrainConfig", "EpochRecord", "TrainResult", "train"]


@dataclass
class TrainConfig:
    """Hyperparameters. Defaults reproduce the published model."""

    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    patience: int = 5
    device: str = "cuda"
    amp: bool = True
    #: Gradient clipping norm. 0 disables.
    clip_grad_norm: float = 1.0
    seed: int = 42


@dataclass
class EpochRecord:
    """One row of the training history."""

    epoch: int
    train_loss: float
    val_loss: float
    val_iou: float
    seconds: float
    improved: bool


@dataclass
class TrainResult:
    """Outcome of a training run."""

    best_val_iou: float
    best_epoch: int
    checkpoint: Path
    history: list[EpochRecord] = field(default_factory=list)
    stopped_early: bool = False

    def to_json(self, path: Path) -> None:
        """Write the history beside the checkpoint, for the record."""
        path.write_text(
            json.dumps(
                {
                    "best_val_iou": self.best_val_iou,
                    "best_epoch": self.best_epoch,
                    "checkpoint": str(self.checkpoint),
                    "stopped_early": self.stopped_early,
                    "history": [asdict(r) for r in self.history],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def train(
    model,
    train_loader,
    val_loader,
    checkpoint_path: Path,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Fit ``model``, keeping the checkpoint with the best validation IoU.

    Args:
        model: SegFormer from :func:`treecover.models.create_segformer`.
        train_loader: Training batches. Pass the season-weighted sampler here.
        val_loader: Validation batches, unweighted and unaugmented.
        checkpoint_path: Where the best weights are written. Overwritten
            each time validation IoU improves.
        config: Hyperparameters.

    Returns:
        A :class:`TrainResult` with the full per-epoch history.
    """
    import torch
    from torch import nn

    config = config or TrainConfig()
    torch.manual_seed(config.seed)

    device = torch.device(
        config.device if (torch.cuda.is_available() or "cpu" in config.device) else "cpu"
    )
    if str(device) != config.device:
        logger.warning("CUDA unavailable — training on %s", device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_iou = -1.0
    best_epoch = -1
    stale = 0
    history: list[EpochRecord] = []

    try:
        for epoch in range(1, config.epochs + 1):
            started = time.monotonic()
            train_loss = _train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler, use_amp, config
            )
            val_loss, val_iou = _validate(model, val_loader, criterion, device, use_amp)

            improved = val_iou > best_iou
            if improved:
                best_iou, best_epoch, stale = val_iou, epoch, 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                stale += 1

            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_iou=val_iou,
                seconds=time.monotonic() - started,
                improved=improved,
            )
            history.append(record)
            logger.info(
                "epoch %3d/%d  train %.4f  val %.4f  IoU %.4f  %s (%.0fs)",
                epoch, config.epochs, train_loss, val_loss, val_iou,
                "saved" if improved else f"no improvement ({stale}/{config.patience})",
                record.seconds,
            )

            if stale >= config.patience:
                logger.info("Early stopping at epoch %d; best IoU %.4f (epoch %d)",
                            epoch, best_iou, best_epoch)
                return TrainResult(best_iou, best_epoch, checkpoint_path, history, True)
    finally:
        # Free the GPU even on an exception — an OOM here otherwise strands
        # several GB and the next run in the same session fails too.
        if device.type == "cuda":
            import gc

            model.to("cpu")
            del optimizer
            torch.cuda.empty_cache()
            gc.collect()

    return TrainResult(best_iou, best_epoch, checkpoint_path, history, False)


def _train_one_epoch(
    model, loader, optimizer, criterion, device, scaler, use_amp, config
) -> float:
    """One pass over the training set. Returns the mean batch loss."""
    import torch

    from treecover.models.segformer import extract_logits

    model.train()
    total = 0.0
    batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            logits = _upsample_to(extract_logits(model(images)), masks.shape[-2:])
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        if config.clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        batches += 1

    return total / max(batches, 1)


def _validate(model, loader, criterion, device, use_amp) -> tuple[float, float]:
    """Mean loss and pooled tree-class IoU over the validation set."""
    import torch

    from treecover.models.segformer import extract_logits

    model.eval()
    total_loss = 0.0
    batches = 0
    intersection = 0
    union = 0

    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with _autocast(device, use_amp):
                logits = _upsample_to(extract_logits(model(images)), masks.shape[-2:])
                loss = criterion(logits, masks)

            total_loss += loss.item()
            batches += 1

            predicted_tree = logits.argmax(dim=1) == 1
            actual_tree = masks == 1
            intersection += int((predicted_tree & actual_tree).sum())
            union += int((predicted_tree | actual_tree).sum())

    # union == 0 means neither the model nor the labels found any tree in the
    # entire validation set: perfect agreement, not an undefined ratio.
    iou = 1.0 if union == 0 else intersection / union
    return total_loss / max(batches, 1), iou


def _autocast(device, enabled: bool):
    """Half-precision context, or a no-op.

    ``torch.autocast`` validates its dtype on ``__enter__`` even when
    ``enabled=False``, and CPU autocast rejects float16 — so passing the
    flag through would make ``--device cpu`` crash regardless of ``--no-amp``.
    Returning a real no-op instead keeps CPU runs working, which matters
    because that is how the pipeline gets smoke-tested.
    """
    import torch

    if not enabled:
        return nullcontext()
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _upsample_to(logits, size):
    """Bring SegFormer's quarter-resolution logits back to the label size."""
    import torch

    if logits.shape[-2:] == tuple(size):
        return logits
    return torch.nn.functional.interpolate(
        logits, size=tuple(size), mode="bilinear", align_corners=False
    )
