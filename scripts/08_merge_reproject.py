#!/usr/bin/env python3
"""Merge 1 km prediction tiles into 10 km GeoTIFFs and reproject to EPSG:3857.

Tiles are grouped by their 10 km UTM grid cell across state boundaries.
Tiles cut at a state border carry 0 where data is missing, so
``gdalbuildvrt -srcnodata 0`` fills the gap from the neighbouring state's
tile.

**Overlap resolution.** About 2.5 % of 1 km cells are covered twice, almost
always because two states flew the same ground. One tile per cell is chosen
*before* GDAL is invoked, by newest acquisition date — see
:mod:`treecover.merge`. Earlier revisions of this script left the choice to
``gdalbuildvrt`` source ordering and sorted the file list three different
ways, which could put a tile in the map that the published
acquisition-date figure says is not there.

Output uses CCITTFAX4, which is the right compression for 1-bit rasters —
COG/DEFLATE inflates them by an order of magnitude.

The run is resumable: ``_tile_counts.json`` in the output directory records
how many source tiles went into each cell, so a cell is rebuilt only when
new tiles have appeared.

Examples::

    python scripts/08_merge_reproject.py
    python scripts/08_merge_reproject.py --workers 8 --input-dir /data/predictions
    python scripts/08_merge_reproject.py --dry-run --report overlaps.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from subprocess import CalledProcessError, run as run_cmd

from treecover.config import load_paths
from treecover.merge import TileCandidate, select_one_per_cell, tile_date

logger = logging.getLogger(__name__)

#: Above this, an output is assumed to come from truncated inputs. A correct
#: 10 km 1-bit CCITTFAX4 tile is a few MB; tens of MB means the compressor
#: was fed noise.
BLOAT_LIMIT_MB = 50.0
MAX_ATTEMPTS = 5
GRID_PREFIX = "UTM"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-dir", type=Path, default=None, help="Override merge.input_dir.")
    p.add_argument("--output-dir", type=Path, default=None, help="Override merge.output_dir.")
    p.add_argument("--temp-dir", type=Path, default=None, help="Override merge.temp_dir.")
    p.add_argument("--workers", type=int, default=4, help="Grid cells processed concurrently.")
    p.add_argument("--force", action="store_true", help="Rebuild cells even if up to date.")
    p.add_argument("--dry-run", action="store_true", help="List what would be built, then stop.")
    p.add_argument("--report", type=Path, default=None,
                   help="Write a CSV of every contested cell and what was dropped.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def collect_candidates(input_dir: Path) -> list[TileCandidate]:
    """Find every prediction tile under ``input_dir``.

    State and year are read from the path where the archive layout provides
    them (``<STATE>/<YEAR>/...``); tiles outside that layout still take part,
    they just carry empty metadata.
    """
    candidates: list[TileCandidate] = []
    for tif in sorted(input_dir.rglob("*.tif")):
        parts = tif.relative_to(input_dir).parts
        state = parts[0] if len(parts) > 1 else ""
        year = parts[1] if len(parts) > 2 else ""
        candidates.append(
            TileCandidate(path=tif, state=state, year=year, date=tile_date(tif, year))
        )
    return candidates


def group_by_output_cell(paths: list[Path]) -> dict[str, list[Path]]:
    """Group the selected tiles by their 10 km output cell.

    The cell name is the parent directory (e.g. ``UTM32_E4100_N52900``);
    tiles outside such a directory are ignored, since there is no output
    tile for them to belong to.
    """
    cells: dict[str, list[Path]] = {}
    for path in paths:
        cell = path.parent.name
        if cell.startswith(GRID_PREFIX):
            cells.setdefault(cell, []).append(path)
    return cells


def write_overlap_report(choices, path: Path) -> int:
    """Record every contested cell. Returns the number of contests."""
    contested = [c for c in choices if c.contested]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell", "winner", "winner_state", "winner_date",
                         "dropped", "dropped_state", "dropped_date"])
        for choice in contested:
            for rejected in choice.rejected:
                writer.writerow([
                    "_".join(choice.cell),
                    choice.winner.path.name, choice.winner.state, choice.winner.date,
                    rejected.path.name, rejected.state, rejected.date,
                ])
    return len(contested)


def load_state(state_file: Path) -> dict[str, int]:
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {}


def process_cell(
    cell: str, tifs: list[Path], out_dir: Path, temp_dir: Path
) -> tuple[str, bool, str]:
    """Build one 10 km output tile, dropping corrupt inputs and retrying.

    GDAL reports a corrupt source by path in stderr; that file is renamed to
    ``.tif.corrupted`` and the merge retried without it, up to
    :data:`MAX_ATTEMPTS` times.

    Returns:
        ``(cell, success, message)``.
    """
    out_path = out_dir / f"{cell}.tif"
    # Overlaps were already resolved in select_one_per_cell, so at most one
    # tile per 1 km cell reaches here and the VRT source order no longer
    # decides anything. Sorted only for reproducible file lists.
    good = sorted(tifs)
    vrt = temp_dir / f"{cell}.vrt"
    flist = temp_dir / f"{cell}.txt"

    for _ in range(MAX_ATTEMPTS):
        if not good:
            return cell, False, "no usable source tiles"
        try:
            flist.write_text("\n".join(str(t) for t in good))
            run_cmd(
                ["gdalbuildvrt", "-srcnodata", "0", "-vrtnodata", "0",
                 "-input_file_list", str(flist), str(vrt)],
                check=True, capture_output=True, text=True,
            )
            run_cmd(
                ["gdalwarp", "-t_srs", "EPSG:3857", "-r", "near", "-of", "GTiff",
                 "-co", "COMPRESS=CCITTFAX4", "-co", "NBITS=1", "-co", "TILED=YES",
                 "-co", "BLOCKXSIZE=512", "-co", "BLOCKYSIZE=512",
                 "-srcnodata", "0", "-dstnodata", "0", str(vrt), str(out_path)],
                check=True, capture_output=True, text=True,
            )
            run_cmd(
                ["gdaladdo", "--config", "COMPRESS_OVERVIEW", "CCITTFAX4",
                 str(out_path), "2", "4", "8", "16", "32", "64"],
                check=True, capture_output=True, text=True,
            )

            mb = out_path.stat().st_size / (1024 * 1024)
            if mb > BLOAT_LIMIT_MB:
                out_path.unlink()
                return cell, False, f"bloated output ({mb:.0f} MB) — inputs likely truncated"

            dropped = len(tifs) - len(good)
            note = f" ({dropped} corrupt dropped)" if dropped else ""
            return cell, True, f"{len(good)} tiles -> {mb:.1f} MB{note}"

        except CalledProcessError as exc:
            out_path.unlink(missing_ok=True)
            bad = _find_bad_tile(exc.stderr, good)
            if bad is None:
                return cell, False, (exc.stderr or "")[:200]
            good = [t for t in good if t != bad]
            if bad.exists():
                bad.rename(bad.with_suffix(".tif.corrupted"))
        finally:
            vrt.unlink(missing_ok=True)
            flist.unlink(missing_ok=True)

    return cell, False, "too many corrupt tiles"


def _find_bad_tile(stderr: str, candidates: list[Path]) -> Path | None:
    """Identify which source tile GDAL choked on, from its stderr."""
    if not stderr:
        return None
    for line in stderr.splitlines():
        match = re.search(r"(\S+\.tif)\b", line)
        if not match:
            continue
        reported = Path(match.group(1))
        for tile in candidates:
            # stderr may truncate long paths, so fall back to matching the stem.
            if tile == reported or tile.name == reported.name or reported.stem in tile.name:
                return tile
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )
    paths = load_paths()

    input_dir = args.input_dir or paths.get_path("merge.input_dir")
    output_dir = args.output_dir or paths.get_path("merge.output_dir")
    temp_dir = args.temp_dir or paths.get_path("merge.temp_dir", "/tmp/treecover_merge")

    if not input_dir.is_dir():
        print(f"error: input directory not found: {input_dir}", file=sys.stderr)
        return 2

    candidates = collect_candidates(input_dir)
    if not candidates:
        print(f"No .tif files found under {input_dir}.", file=sys.stderr)
        return 1

    # Resolve overlaps explicitly, so the outcome does not depend on how
    # gdalbuildvrt happens to order its sources.
    choices = select_one_per_cell(candidates)
    selected = [c.winner.path for c in choices]
    dropped = sum(len(c.rejected) for c in choices)

    if args.report:
        contested = write_overlap_report(choices, args.report)
        print(f"Overlap report: {contested} contested cell(s) -> {args.report}")

    cells = group_by_output_cell(selected)
    if not cells:
        print(f"No tiles under {GRID_PREFIX}* subdirectories in {input_dir}.",
              file=sys.stderr)
        return 1

    total = sum(len(v) for v in cells.values())
    print(f"{len(candidates)} source tiles -> {total} selected "
          f"({dropped} dropped as overlaps, newest acquisition date wins)")
    print(f"{len(cells)} output grid cells")

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / "_tile_counts.json"
    state = load_state(state_file)

    todo = []
    for cell, tifs in sorted(cells.items()):
        out_path = output_dir / f"{cell}.tif"
        if out_path.exists() and not args.force and state.get(cell) == len(tifs):
            continue
        if out_path.exists():
            out_path.unlink()
        todo.append((cell, tifs))

    print(f"{len(todo)} to build, {len(cells) - len(todo)} up to date")
    if args.dry_run:
        for cell, tifs in todo:
            print(f"  {cell}: {len(tifs)} tiles")
        return 0
    if not todo:
        return 0

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_cell, cell, tifs, output_dir, temp_dir): (cell, tifs)
            for cell, tifs in todo
        }
        for n, future in enumerate(as_completed(futures), 1):
            cell, tifs = futures[future]
            _, ok, message = future.result()
            if ok:
                state[cell] = len(tifs)
                state_file.write_text(json.dumps(state))
                print(f"[{n}/{len(todo)}] {cell}: {message}")
            else:
                failures += 1
                print(f"[{n}/{len(todo)}] FAIL {cell}: {message}", file=sys.stderr)

    outputs = list(output_dir.glob("*.tif"))
    total_mb = sum(f.stat().st_size for f in outputs) / (1024 * 1024)
    print(f"\nDone: {len(outputs)} tiles, {total_mb:.0f} MB in {output_dir}")
    if failures:
        print(f"{failures} cells failed.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
