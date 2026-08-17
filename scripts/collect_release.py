#!/usr/bin/env python3
"""Gather the release into one directory and say what is still missing.

Everything the paper publishes — the map, the model, the training data,
the validation reference, the per-tile tables and the figures — belongs in
one directory that can be uploaded as it stands. Most of it is written
there by the pipeline stages; the rest arrives from elsewhere, and this
step is what turns a directory that happens to hold the right files into
one that is checked against an inventory.

The inventory lives in :mod:`treecover.release`, not here, so the README
and the manifest cannot disagree with each other.

What it does:

* copies anything named with ``--add SRC=DEST`` into the release,
* checks every inventory entry, sizing directories and hashing files,
* writes ``MANIFEST.csv``,
* reports what is missing, what is empty and what is present but unlisted.

Missing entries are reported, not fixed: a figure that was never rendered
has to be rendered by its own script, and inventing a placeholder here
would hide that.

Examples::

    python scripts/collect_release.py --root /tf/moritz_lucas/publication

    # Bring the figures in from wherever they were rendered
    python scripts/collect_release.py --root publication \\
        --add figures/output/fig01_acquisition_coverage.png=figures/

    # Fast pass over a multi-gigabyte release
    python scripts/collect_release.py --root publication --no-checksums
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

from treecover.config import load_paths
from treecover.release import (
    FIGURE_STEMS,
    RELEASE,
    human_size,
    inspect_release,
    manifest,
    unlisted,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", type=Path, default=None,
                   help="The release directory. Default: publication_root from "
                        "paths.yaml, else ./publication.")
    p.add_argument("--add", nargs="*", default=None, metavar="SRC=DEST",
                   help="Copy SRC into the release at DEST. A DEST ending in / is "
                        "a directory and keeps the source filename.")
    p.add_argument("--checksums", dest="checksums", action="store_true", default=True,
                   help="SHA-256 for every file within --checksum-limit (default).")
    p.add_argument("--no-checksums", dest="checksums", action="store_false")
    p.add_argument("--checksum-limit", type=float, default=512.0, metavar="MB",
                   help="Files above this size are sized but not hashed.")
    p.add_argument("--prune", action="store_true",
                   help="Delete .ipynb_checkpoints and leftover .part files. "
                        "Nothing else is ever removed.")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Where to write the manifest (default: <root>/MANIFEST.csv).")
    p.add_argument("--dry-run", action="store_true",
                   help="Report, but copy nothing and write no manifest.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def copy_in(root: Path, specs, dry_run: bool = False) -> int:
    """Copy ``SRC=DEST`` pairs into the release. Returns how many landed."""
    copied = 0
    for spec in specs:
        if "=" not in spec:
            print(f"error: --add expects SRC=DEST, got {spec!r}", file=sys.stderr)
            continue
        source_text, destination_text = spec.split("=", 1)
        source = Path(source_text)
        if not source.exists():
            print(f"error: --add source not found: {source}", file=sys.stderr)
            continue

        destination = root / destination_text
        if destination_text.endswith("/") or destination.is_dir():
            destination = destination / source.name

        if dry_run:
            print(f"  would copy {source} -> {destination.relative_to(root)}")
            copied += 1
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
        logger.info("copied %s -> %s", source, destination.relative_to(root))
        copied += 1
    return copied


def prune(root: Path, dry_run: bool = False) -> list:
    """Remove working clutter — checkpoint directories and partial downloads."""
    removed = []
    for path in list(root.rglob(".ipynb_checkpoints")) + list(root.rglob("*.part")):
        removed.append(str(path.relative_to(root)))
        if not dry_run:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    return removed


def report_figures(root: Path) -> None:
    """Name the figures that are absent, since the inventory only counts them."""
    directory = root / "figures"
    if not directory.exists():
        return
    stems = {path.stem for path in directory.glob("*.png")}
    missing = [stem for stem in FIGURE_STEMS if stem not in stems]
    if not missing:
        print("\nFigures: all ten present.")
        return

    print(f"\nFigures: {len(FIGURE_STEMS) - len(missing)} of {len(FIGURE_STEMS)} "
          "present. Missing:")
    for stem in missing:
        number = stem[3:5]
        hint = ("a QGIS composition — export it from "
                "Paper/Map_Validation_Training/map_training_validation_tree_Cover.qgz"
                if number == "02" else f"render it with figures/{stem}.py")
        print(f"  {stem}.png — {hint}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    paths = load_paths()
    root = Path(args.root or paths.get_value("publication_root", "./publication"))
    if not root.exists():
        print(f"error: release directory not found: {root}", file=sys.stderr)
        return 2
    print(f"Release: {root}")

    if args.add:
        print("\nCopying in:")
        copy_in(root, args.add, args.dry_run)

    if args.prune:
        removed = prune(root, args.dry_run)
        print(f"\nPruned {len(removed)} clutter entr{'y' if len(removed) == 1 else 'ies'}"
              + (":" if removed else ""))
        for path in removed[:10]:
            print(f"  {path}")

    statuses = inspect_release(root, checksums=args.checksums,
                               checksum_limit=int(args.checksum_limit * 1024 * 1024))

    present = [s for s in statuses if s.present]
    missing = [s for s in statuses if not s.present]
    total = sum(s.size_bytes for s in present)

    print(f"\n{len(present)} of {len(statuses)} inventory entries present, "
          f"{human_size(total)} in total\n")
    for status in statuses:
        mark = "ok     " if status.present else (
            "MISSING" if status.item.required else "absent ")
        size = human_size(status.size_bytes) if status.present else ""
        files = f"{status.files:>6} file(s)" if status.item.is_directory else ""
        print(f"  {mark} {status.item.path:<42} {size:>10} {files}"
              + (f"  {status.note}" if status.note else ""))

    report_figures(root)

    extra = unlisted(root)
    if extra:
        print("\nPresent but not in the inventory:")
        for name, why in extra:
            print(f"  {name:<42} {why}")

    written = None
    if not args.dry_run:
        destination = args.manifest or root / "MANIFEST.csv"
        pd.DataFrame(manifest(statuses)).to_csv(destination, index=False)
        written = destination
        print(f"\nManifest written to {destination}")

    # The manifest cannot list itself as present — it is written after the
    # walk. Reporting it as missing in the same run that wrote it would be
    # the one failure that never goes away.
    required_missing = [
        s for s in missing
        if s.item.required and not (written and s.item.path == written.name)
    ]
    if required_missing:
        print(f"\n{len(required_missing)} required entr"
              f"{'y is' if len(required_missing) == 1 else 'ies are'} missing — the "
              "release is not complete:")
        for status in required_missing:
            print(f"  {status.item.path:<42} {status.item.what}")
            if status.item.stage:
                print(f"    produced by {status.item.stage}")
        return 1

    print("\nEvery required entry is present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
