#!/usr/bin/env python3
"""Train the SegFormer tree cover model.

Reads the outputs of ``03_prepare_patches.py``:

* ``patches.csv``        one row per training patch (region, window, date)
* ``region_vrts.json``   region_id -> imagery / mask / nDSM VRT paths
* ``splits.json``        train / val / test region lists

Splits are at *tile* level, never patch level: patches from one tile overlap,
so a patch-level split leaks the validation set into training and inflates
every metric.

Training patches are drawn with a season-aware weighted sampler so leaf-off,
transition and leaf-on imagery appear about equally often per epoch. That is
the paper's central mechanism — see :mod:`treecover.data.seasons`.

The published model::

    python scripts/04_train.py --data-config RGB --backbone nvidia/mit-b5 \\
        --tcd-pretrained --batch-size 16 --lr 1e-4 --name segformer_b5_tcd_RGB

Quick check that the wiring works, on CPU::

    python scripts/04_train.py --epochs 1 --limit 32 --batch-size 2 --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from treecover.config import load_paths
from treecover.data import season_weights
from treecover.data.dataset import (
    DATA_CONFIGS,
    TreeCoverDataset,
    n_input_channels,
    training_augmentation,
    validation_augmentation,
)
from treecover.models import create_segformer
from treecover.training import TrainConfig, train

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    data = p.add_argument_group("data")
    data.add_argument("--patches-dir", type=Path, default=None,
                      help="Directory holding patches.csv, region_vrts.json, splits.json "
                           "(default: patches_dir from paths.yaml).")
    data.add_argument("--data-config", default="RGB", choices=sorted(DATA_CONFIGS),
                      help="Input bands. RGB is the published, transferable model.")
    data.add_argument("--date-encoding", action="store_true",
                      help="Append two cyclical day-of-year channels. The published "
                           "model does NOT use this — it did not improve accuracy.")
    data.add_argument("--limit", type=int, default=None,
                      help="Use only N training patches — for smoke tests.")

    model = p.add_argument_group("model")
    model.add_argument("--backbone", default="nvidia/mit-b5",
                       help="SegFormer backbone, mit-b0 … mit-b5.")
    model.add_argument("--tcd-pretrained", action="store_true",
                       help="Initialise from restor/tcd-segformer instead of ImageNet. "
                            "The published model used this.")

    opt = p.add_argument_group("optimisation")
    opt.add_argument("--epochs", type=int, default=50)
    opt.add_argument("--batch-size", type=int, default=16)
    opt.add_argument("--lr", type=float, default=1e-4)
    opt.add_argument("--weight-decay", type=float, default=0.01)
    opt.add_argument("--patience", type=int, default=5,
                     help="Epochs without validation-IoU improvement before stopping.")
    opt.add_argument("--no-season-weighting", action="store_true",
                     help="Sample uniformly instead. Reproduces the ablation showing "
                          "why the weighting matters; not for a production model.")

    run = p.add_argument_group("execution")
    run.add_argument("--out", type=Path, default=None,
                     help="Where checkpoints go (default: weights_dir from paths.yaml).")
    run.add_argument("--name", default=None,
                     help="Checkpoint basename (default: derived from the settings).")
    run.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    run.add_argument("--device", default="cuda")
    run.add_argument("--no-amp", dest="amp", action="store_false")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("-v", "--verbose", action="store_true")
    return p


def load_inputs(patches_dir: Path):
    """Read the three artefacts stage 3 produces."""
    required = {
        "patches": patches_dir / "patches.csv",
        "region_vrts": patches_dir / "region_vrts.json",
        "splits": patches_dir / "splits.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing patch artefacts:\n  " + "\n  ".join(missing)
            + "\n\nRun scripts/03_prepare_patches.py first."
        )

    patches = pd.read_csv(required["patches"]).to_dict("records")
    region_vrts = json.loads(required["region_vrts"].read_text())
    splits = json.loads(required["splits"].read_text())
    return patches, region_vrts, splits


def split_patches(patches: list[dict], splits: dict) -> tuple[list, list]:
    """Partition patches by the tile-level split."""
    train_regions = set(map(str, splits["train_regions"]))
    val_regions = set(map(str, splits["val_regions"]))

    train = [p for p in patches if str(p["region_id"]) in train_regions]
    val = [p for p in patches if str(p["region_id"]) in val_regions]

    overlap = train_regions & val_regions
    if overlap:
        raise ValueError(
            f"{len(overlap)} region(s) appear in both the train and val split "
            f"(e.g. {sorted(overlap)[:3]}). Patches from one tile overlap, so this "
            "leaks validation data into training."
        )
    return train, val


def default_name(args) -> str:
    """Checkpoint name encoding the settings, as the original runs did."""
    backbone = args.backbone.split("/")[-1].replace("mit-", "")
    channels = n_input_channels(args.data_config, args.date_encoding)
    encoding = "dateenc" if args.date_encoding else "nodateenc"
    tcd = "_tcd" if args.tcd_pretrained else ""
    return (
        f"segformer_{backbone}{tcd}_{args.data_config}_{channels}ch_{encoding}"
        f"_bs{args.batch_size}_lr{args.lr:.0e}_adamw"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    paths = load_paths()
    patches_dir = args.patches_dir or paths.get_path("patches_dir")
    try:
        patches, region_vrts, splits = load_inputs(patches_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    train_patches, val_patches = split_patches(patches, splits)
    if args.limit:
        train_patches = train_patches[: args.limit]
        val_patches = val_patches[: max(1, args.limit // 4)]

    if not train_patches or not val_patches:
        print(f"error: empty split (train={len(train_patches)}, val={len(val_patches)})",
              file=sys.stderr)
        return 2

    n_channels = n_input_channels(args.data_config, args.date_encoding)
    print(f"Patches    : {len(train_patches)} train / {len(val_patches)} val")
    print(f"Input      : {args.data_config} -> {n_channels} channel(s)")
    print(f"Backbone   : {args.backbone}"
          f"{'  (TCD-pretrained)' if args.tcd_pretrained else ''}")

    train_dataset = TreeCoverDataset(
        train_patches, region_vrts, args.data_config,
        transform=training_augmentation(n_channels),
        include_date_encoding=args.date_encoding,
    )
    val_dataset = TreeCoverDataset(
        val_patches, region_vrts, args.data_config,
        transform=validation_augmentation(n_channels),
        include_date_encoding=args.date_encoding,
    )

    if args.no_season_weighting:
        sampler, shuffle = None, True
        print("Sampling   : uniform (season weighting disabled)")
    else:
        weights, counts, per_season = season_weights([p.get("date") for p in train_patches])
        sampler, shuffle = (
            WeightedRandomSampler(
                weights=torch.DoubleTensor(weights),
                num_samples=len(weights),
                replacement=True,
            ),
            False,
        )
        print(f"Sampling   : season-aware  counts={counts}")
        print(f"             weights={ {k: round(v, 3) for k, v in per_season.items()} }")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=shuffle,
        num_workers=args.workers, pin_memory=args.device.startswith("cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
    )

    model = create_segformer(
        n_channels=n_channels,
        backbone=args.backbone,
        use_tcd_pretrained=args.tcd_pretrained,
    )

    out_dir = args.out or paths.get_path("weights_dir", "./weights")
    name = args.name or default_name(args)
    checkpoint = Path(out_dir) / f"{name}.pth"
    print(f"Checkpoint : {checkpoint}\n")

    result = train(
        model, train_loader, val_loader, checkpoint,
        TrainConfig(
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=args.device,
            amp=args.amp,
            seed=args.seed,
        ),
    )
    result.to_json(checkpoint.with_suffix(".history.json"))

    print(f"\nBest validation IoU {result.best_val_iou:.4f} at epoch {result.best_epoch}"
          f"{' (early stop)' if result.stopped_early else ''}")
    print(f"Weights  : {result.checkpoint}")
    print(f"History  : {checkpoint.with_suffix('.history.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
