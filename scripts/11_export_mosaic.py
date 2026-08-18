#!/usr/bin/env python3
"""Export the predictions as a seamless 10 km mosaic in native UTM.

This is the archival product: the 20 cm binary tree mask, merged across
state boundaries, one GeoTIFF per 10 km grid cell, in the CRS the model
wrote (EPSG:25832 west, 25833 east). **Nothing is resampled.** All tiles of
a zone share one 20 cm grid, so merging is a copy, and a pixel of the
export is a pixel of the prediction.

Contrast with ``08_merge_reproject.py``, which warps to EPSG:3857 for web
display. That product must not be used for areas — Web Mercator inflates
them by about 2.5× at Germany's latitude, latitude-dependently. This one
can: a pixel is exactly 0.04 m² of projected area, and the per-cell
geodesic factor in the output table converts that to ground area.

**Overlaps.** About 2.5 % of 1 km cells are flown by two states. One tile
per cell is chosen before anything is written, by newest acquisition date
(:mod:`treecover.merge`), so every ground pixel exists exactly once and a
count over the mosaic cannot double-count.

Alongside the rasters it writes ``cell_statistics.csv``, ``tile_index.geojson``
and a ``README.md`` describing both.

The run is resumable: ``_export_state.json`` records how many source tiles
went into each finished cell, and a cell is rebuilt only if that count
changed or ``--force`` is given.

Examples::

    python scripts/11_export_mosaic.py --predictions-root /tf/Germany \\
        --out /mnt/publication/mosaic_utm --workers 8
    python scripts/11_export_mosaic.py --out /tmp/m --limit-cells 3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from treecover.constants import PRED_TREE
from treecover.merge import TileCandidate, cell_key, select_one_per_cell, tile_date
from treecover.mosaic import (
    CELL_PX,
    CELL_SIZE_M,
    PIXEL_M,
    Placement,
    cell_id_from_tile_key,
    cell_transform,
    geodesic_area_factor,
    grid_offset,
    is_copyable,
    parse_cell_id,
    placement_for,
    plan_cell,
    sanity_check_grid,
    stripe_rows,
    total_stripes,
)

logger = logging.getLogger(__name__)

#: Rows per TIFF strip. Must divide the stripe height, so a strip is never
#: half-written when the next stripe arrives — CCITTFAX4 cannot revisit an
#: emitted strip. Tiled TIFF is not an option here: TIFF requires tile
#: dimensions to be multiples of 16, and no multiple of 16 divides 5000.
ROWS_PER_STRIP = 250

#: How far a tile may overhang its stripe before it is refused instead of
#: trimmed. The archive's odd tiles are 5001 px — one row of redundant
#: overlap with the neighbour; anything beyond a handful of pixels is not a
#: 1 km tile and must not be silently truncated.
MAX_TRIM_PX = 8

#: Creation options for the 1-bit output. CCITTFAX4 is the right codec for
#: bilevel rasters; DEFLATE inflates these by an order of magnitude.
CREATION = {
    "compress": "CCITTFAX4",
    "nbits": 1,
    "tiled": False,
    "blockysize": ROWS_PER_STRIP,
    "bigtiff": "IF_SAFER",
}


@dataclass
class CellResult:
    """What one output cell ended up containing."""

    cell: str
    epsg: int = 0
    n_source_tiles: int = 0
    covered_px: int = 0
    tree_px: int = 0
    states: set = field(default_factory=set)
    dates: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    snapped: int = 0
    max_snap_px: float = 0.0
    trimmed: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_row(self) -> dict:
        factor = geodesic_area_factor(self.cell) if self.ok else 1.0
        px_m2 = PIXEL_M * PIXEL_M
        covered = self.covered_px * px_m2
        tree = self.tree_px * px_m2
        return {
            "cell": self.cell,
            "epsg": self.epsg,
            "n_source_tiles": self.n_source_tiles,
            "covered_km2_projected": covered / 1e6,
            "tree_km2_projected": tree / 1e6,
            "geodesic_factor": factor,
            "covered_km2": covered * factor / 1e6,
            "tree_km2": tree * factor / 1e6,
            "tree_cover_pct": 100.0 * self.tree_px / self.covered_px
            if self.covered_px
            else 0.0,
            "states": ",".join(sorted(self.states)),
            "date_min": min(self.dates) if self.dates else "",
            "date_max": max(self.dates) if self.dates else "",
            "n_skipped": len(self.skipped),
            "n_snapped": self.snapped,
            "max_snap_m": round(self.max_snap_px * PIXEL_M, 3),
            "n_trimmed": self.trimmed,
        }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_argument_group("input")
    src.add_argument("--predictions-root", type=Path, default=None,
                     help="Root of the prediction archive (default: paths.yaml).")
    src.add_argument("--states", nargs="*", default=None, metavar="CODE")
    src.add_argument("--limit-cells", type=int, default=None,
                     help="Build only the first N cells (debugging).")

    out = p.add_argument_group("output")
    out.add_argument("--out", type=Path, required=True, help="Output directory.")
    out.add_argument("--force", action="store_true", help="Rebuild finished cells.")
    out.add_argument("--dry-run", action="store_true",
                     help="Report the plan and stop before writing rasters.")
    out.add_argument("--tables-only", action="store_true",
                     help="Skip raster writing; rebuild the CSV, GeoJSON and README "
                          "from the cells already on disk.")

    run = p.add_argument_group("execution")
    run.add_argument("--workers", type=int, default=6,
                     help="Cells built concurrently. Each holds a 250 MB stripe.")
    run.add_argument("-v", "--verbose", action="store_true")
    return p


def collect_candidates(root: Path, states: list[str] | None) -> list[TileCandidate]:
    """Find every prediction tile under ``<root>/<STATE>/<YEAR>/predictions/``."""
    candidates: list[TileCandidate] = []
    for state_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if states and state_dir.name not in states:
            continue
        for year_dir in sorted(p for p in state_dir.iterdir() if p.is_dir()):
            predictions = year_dir / "predictions"
            if not predictions.is_dir():
                continue
            for tif in sorted(predictions.rglob("*.tif")):
                candidates.append(
                    TileCandidate(
                        path=tif,
                        state=state_dir.name,
                        year=year_dir.name,
                        date=tile_date(tif, year_dir.name),
                    )
                )
    return candidates


def group_by_cell(winners: list[TileCandidate]) -> dict[str, list[TileCandidate]]:
    """Group the selected tiles by the 10 km cell they belong to."""
    cells: dict[str, list[TileCandidate]] = {}
    for candidate in winners:
        key = cell_key(candidate.path)
        if key is None:
            logger.warning("%s carries no cell id; not exported", candidate.path.name)
            continue
        cells.setdefault(cell_id_from_tile_key(key), []).append(candidate)
    return cells


def _open_placements(
    cell: str, candidates: list[TileCandidate], result: CellResult
) -> list[Placement]:
    """Read each source tile's georeferencing and locate it in the cell.

    Tiles that cannot be opened, sit off the 20 cm grid or fall outside the
    cell are recorded in ``result.skipped`` rather than raising: one bad
    tile out of 380,000 must not cost a whole cell.
    """
    import rasterio

    epsg, _, _ = parse_cell_id(cell)
    placements: list[Placement] = []

    for candidate in candidates:
        try:
            with rasterio.open(candidate.path) as src:
                src_epsg = src.crs.to_epsg() if src.crs else None
                left, top = src.transform.c, src.transform.f
                width, height = src.width, src.height
                res = src.res
                skew = (src.transform.b, src.transform.d)
        except Exception as exc:  # noqa: BLE001 - corrupt tiles are known
            result.skipped.append(f"{candidate.path.name}: unreadable ({exc})")
            continue

        if src_epsg != epsg:
            result.skipped.append(
                f"{candidate.path.name}: EPSG {src_epsg}, cell is {epsg}"
            )
            continue
        if not is_copyable(res, skew):
            result.skipped.append(
                f"{candidate.path.name}: res {res}, skew {skew} — would need resampling"
            )
            continue

        # Saarland's WMS tiles sit half a pixel out in y. Snapping shifts
        # them by 10 cm; refusing them would leave a hole in the mosaic
        # that the paper's per-tile statistics do not have.
        if not sanity_check_grid(cell, left, top):
            offset = grid_offset(cell, left, top)
            result.snapped += 1
            result.max_snap_px = max(
                result.max_snap_px, max(abs(offset[0]), abs(offset[1]))
            )

        placement = placement_for(cell, candidate.path, left, top, width, height)
        if placement is None:
            result.skipped.append(f"{candidate.path.name}: outside {cell}")
            continue

        # A 1 km tile stored as 5001 px overhangs its neighbour by one row,
        # which is trimmed. Anything larger is not a 1 km tile at all, and
        # truncating it would drop real ground without saying so.
        if placement.trimmed_rows > MAX_TRIM_PX:
            result.skipped.append(
                f"{candidate.path.name}: spans {placement.trimmed_rows} rows past "
                "its stripe — not a 1 km tile"
            )
            continue
        if placement.trimmed_rows:
            result.trimmed += 1

        placements.append(placement)
        result.states.add(candidate.state)
        if candidate.is_dated:
            result.dates.append(candidate.date)

    return placements


def build_cell(cell: str, candidates: list[TileCandidate], out_dir: Path) -> CellResult:
    """Write one 10 km mosaic tile and measure what went into it.

    Stripes are materialised one at a time and written top to bottom, so
    memory stays at one stripe (250 MB) and every TIFF strip is complete
    before the compressor sees it.
    """
    import rasterio

    result = CellResult(cell=cell)
    try:
        epsg, _, _ = parse_cell_id(cell)
        result.epsg = epsg

        placements = _open_placements(cell, candidates, result)
        if not placements:
            result.error = "no usable source tiles"
            return result
        result.n_source_tiles = len(placements)

        by_stripe = plan_cell(placements)
        out_path = out_dir / f"UTM{epsg - 25800}" / f"{cell}.tif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        partial = out_path.with_suffix(".tif.partial")

        profile = {
            "driver": "GTiff",
            "width": CELL_PX,
            "height": CELL_PX,
            "count": 1,
            "dtype": "uint8",
            "crs": f"EPSG:{epsg}",
            "transform": cell_transform(cell),
            **CREATION,
        }

        with rasterio.open(partial, "w", **profile) as dst:
            for stripe in range(total_stripes()):
                first_row, n_rows = stripe_rows(stripe)
                band = np.zeros((n_rows, CELL_PX), dtype=np.uint8)

                for placement in by_stripe.get(stripe, []):
                    _place(placement, band, first_row, result)

                dst.write(
                    band,
                    1,
                    window=rasterio.windows.Window(0, first_row, CELL_PX, n_rows),
                )
                result.tree_px += int(np.count_nonzero(band))
                del band

        partial.replace(out_path)
    except Exception as exc:  # noqa: BLE001 - one cell must not kill the run
        result.error = f"{type(exc).__name__}: {exc}"
        logger.debug("cell %s failed", cell, exc_info=True)
    return result


def _place(placement: Placement, band: np.ndarray, first_row: int, result: CellResult):
    """Copy one source tile into the stripe buffer.

    Values other than 0/1 abort the tile rather than being counted as
    background — a raster holding higher codes comes from the
    trees-outside-forests classification and is not this map.
    """
    import rasterio
    from rasterio.windows import Window

    try:
        with rasterio.open(placement.path) as src:
            data = src.read(
                1,
                window=Window(
                    placement.read_col_off,
                    placement.read_row_off,
                    placement.width,
                    placement.height,
                ),
            )
    except Exception as exc:  # noqa: BLE001
        result.skipped.append(f"{placement.path.name}: read failed ({exc})")
        return

    unexpected = set(np.unique(data).tolist()) - {0, PRED_TREE}
    if unexpected:
        result.skipped.append(
            f"{placement.path.name}: unexpected codes {sorted(unexpected)}"
        )
        return

    row = placement.row_off - first_row
    band[row : row + placement.height,
         placement.col_off : placement.col_off + placement.width] = data
    result.covered_px += placement.height * placement.width


_OUT_DIR: Path | None = None


def _worker_init(out_dir: Path) -> None:
    global _OUT_DIR
    _OUT_DIR = out_dir


def _in_worker(task: tuple[str, list[TileCandidate]]) -> CellResult:
    """Build one cell. The task carries its own tiles and nothing else.

    Handing the whole cell index to every worker instead costs a gigabyte
    per process — 370,000 candidates pickled sixteen times over — and puts
    the run within reach of the OOM killer on a shared container.
    """
    assert _OUT_DIR is not None, "worker not initialised"
    cell, candidates = task
    return build_cell(cell, candidates, _OUT_DIR)


def finalise(out_dir: Path, results: list[CellResult], cells: dict) -> dict:
    """Merge this run into the cell table, then rewrite every document.

    Cells built in an earlier run keep the row they were measured with when
    they were written; only the cells rebuilt now are replaced. Re-reading
    eleven gigabytes of raster to reproduce rows that are already correct
    would be the alternative, and it would measure nothing new.
    """
    import pandas as pd

    table = out_dir / "cell_statistics.csv"
    fresh = pd.DataFrame([r.as_row() for r in results if r.ok])

    if table.exists():
        kept = pd.read_csv(table)
        if not fresh.empty:
            kept = kept[~kept["cell"].isin(fresh["cell"])]
        frame = pd.concat([kept, fresh], ignore_index=True) if not fresh.empty else kept
    elif not fresh.empty:
        frame = fresh
    else:
        # Nothing measured and nothing on record: fall back to counting the
        # rasters themselves rather than writing an empty table.
        frame = pd.DataFrame([r.as_row() for r in _results_from_disk(cells, out_dir, [])])

    if frame.empty:
        logger.warning("no cells to report; documents not written")
        return {}

    frame = frame.sort_values("cell").reset_index(drop=True)
    for column in ("n_snapped", "max_snap_m", "n_trimmed", "n_skipped"):
        if column not in frame.columns:
            frame[column] = 0
        frame[column] = frame[column].fillna(0)

    frame.to_csv(table, index=False)
    totals = _totals_from_frame(frame)
    (out_dir / "export_summary.json").write_text(json.dumps(totals, indent=2))
    _write_geojson(frame, out_dir)
    write_readme(out_dir, frame)
    return totals


def _write_geojson(frame, out_dir: Path) -> None:
    """One polygon per 10 km cell, carrying that cell's statistics."""
    features = []
    for row in frame.to_dict("records"):
        features.append(
            {
                "type": "Feature",
                "geometry": _cell_polygon(row["cell"]),
                "properties": {k: _plain(v) for k, v in row.items()},
            }
        )
    geojson = {
        "type": "FeatureCollection",
        "name": "tree_cover_germany_mosaic_10km",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }
    (out_dir / "tile_index.geojson").write_text(
        json.dumps(geojson), encoding="utf-8"
    )


#: Written verbatim into the export directory. The numbers are filled from
#: the measured table, so the README cannot drift from the data it
#: describes — that drift is exactly how the paper ended up quoting a
#: percentage and an area that implied two different reference areas.
README_TEMPLATE = """\
# Tree cover of Germany — 20 cm mosaic

The binary tree-cover mask of the accompanying manuscript, merged from the
per-tile predictions into a seamless 10 km grid.

Generated by `scripts/11_export_mosaic.py`.

## What is here

```
UTM32/  {n32:>5} GeoTIFFs   EPSG:25832, western Germany
UTM33/  {n33:>5} GeoTIFFs   EPSG:25833, eastern Germany
cell_statistics.csv        one row per cell, the table below
tile_index.geojson         one polygon per cell, same columns, WGS 84
export_summary.json        national totals
```

| | |
|---|---|
| cells | {cells:,} |
| source tiles merged | {source_tiles:,} |
| mapped area | {covered_km2:,.0f} km² |
| tree cover | {tree_km2:,.0f} km² |
| tree cover of mapped area | {tree_cover_pct:.2f} % |

## Raster specification

| | |
|---|---|
| CRS | ETRS89 / UTM 32N (EPSG:25832) and 33N (EPSG:25833) |
| pixel | 0.2 m × 0.2 m, north-up, aligned to the UTM grid |
| size | 50,000 × 50,000 px = 10 km × 10 km |
| values | `1` tree, `0` not tree |
| type | 1-bit unsigned, CCITTFAX4, {rows_per_strip} rows per strip |
| nodata | none — see the caveat below |

Cell names give the south-west corner in units of 100 m:
`UTM33_E4400_N58600` is 440,000 / 5,860,000 in EPSG:25833.

**Nothing was resampled.** Every prediction tile already sits on this exact
20 cm grid, so merging copied whole tiles into their windows. A pixel of
this mosaic is a pixel of the model output, bit for bit.

## Computing areas

A pixel is exactly 0.04 m² of *projected* area, so

```
tree area (projected) = count_of_ones * 0.04 m²
```

UTM is conformal, not equal-area, so projected area is not ground area.
Multiply by the cell's `geodesic_factor` from `cell_statistics.csv` — the
true area of the cell on the GRS80 ellipsoid divided by its 100 km² of
projected area. Across Germany the correction stays within ±0.2 %:

```
tree area (ground) = count_of_ones * 0.04 m² * geodesic_factor
```

Do **not** compute areas from a Web Mercator (EPSG:3857) version of this
product. That projection inflates area by roughly 2.5× at Germany's
latitude, and latitude-dependently.

**Caveat — the denominator.** A 1-bit raster cannot distinguish "not tree"
from "not mapped": both are `0`. Cells on the coast and the national border
are only partly covered. Take the reference area from `covered_km2` in the
table (derived from the source tiles that actually went in), not from the
{cell_km2:,.0f} km² footprint of the cell. The columns are:

| column | meaning |
|---|---|
| `cell` | cell id, also the file name |
| `epsg` | 25832 or 25833 |
| `n_source_tiles` | 1 km prediction tiles merged into this cell |
| `covered_km2` | ground area actually mapped |
| `tree_km2` | ground area classified as tree |
| `tree_cover_pct` | `tree_km2 / covered_km2` |
| `geodesic_factor` | projected → ground area correction |
| `covered_km2_projected`, `tree_km2_projected` | the same before correction |
| `states` | federal states contributing tiles |
| `date_min`, `date_max` | acquisition dates of those tiles |
| `n_skipped` | source tiles refused (unreadable, wrong CRS or resolution) |
| `n_snapped`, `max_snap_m` | tiles nudged onto the grid, and by how far |
| `n_trimmed` | tiles whose redundant overhang row was dropped |

**Sub-pixel snapping.** {snapped:,} of {source_tiles:,} source tiles are
georeferenced a fraction of a pixel off this grid, by at most
{max_snap_m:.2f} m — half a pixel. They are snapped to the nearest pixel
rather than refused: a 10 cm shift is well inside the resolution, whereas
dropping them would leave holes that the paper's own per-tile statistics do
not have.

**Overhanging tiles.** A few tiles are stored as 5001 px, covering 1000.2 m
and so overlapping their southern neighbour by one row. That row is
trimmed; the neighbour supplies it. {trimmed:,} tiles were trimmed this
way. A tile overhanging by more than {max_trim} px is refused instead —
it is not a 1 km tile, and truncating it would drop real ground silently.

## Overlapping coverage — read this before summing

About 2.5 % of 1 km cells were flown by two states. **Within a zone** the
winner is the **newest acquisition date**, decided before anything is
written, with a deterministic tiebreak. Of {selected_from:,} prediction
tiles, {dropped:,} were dropped as the older side of an overlap and
{source_tiles:,} were merged.

**Across the zone seam this does not apply.** The zone is a property of the
delivering state, not of geography: Brandenburg, Berlin, Mecklenburg-
Vorpommern, Saxony and Thuringia deliver in EPSG:25833, every other state
in 25832 — including Saxony-Anhalt, which lies between them. Along those
state borders both neighbours flew the same ground, and the two tiles carry
different zone identifiers, so the overlap rule above never sees them.

Measured on the tile geometries, **3,175 km² of ground is present in both
`UTM32/` and `UTM33/`**, carrying **785 km² of tree cover, 0.68 % of the
national total**. The largest seams are Saxony-Anhalt–Brandenburg
(829 km²), Thuringia–Saxony-Anhalt (438), Saxony–Saxony-Anhalt (434) and
Thuringia–Bavaria (388). The two sides were flown on different dates in
98.5 % of cases and disagree accordingly: 30.0 % against 24.7 % tree cover
over the same ground.

So: summing `tree_km2` over every cell in both directories double-counts by
0.68 %. For a national figure use the state-masked per-tile statistics
(`statistics/tile_statistics_all.csv`), where each tile is clipped to its
own state and the double coverage cannot arise; that route gives
115,202 km² of tree on 356,381 km² of mapped land. Per cell and per zone
the numbers in `cell_statistics.csv` are exact.
"""


def write_readme(out_dir: Path, frame) -> None:
    """Describe the export, with its own measurements filled in."""
    summary_path = out_dir / "export_summary.json"
    totals = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    provenance_path = out_dir / "_provenance.json"
    provenance = (
        json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    )

    covered = float(frame["covered_km2"].sum())
    tree = float(frame["tree_km2"].sum())
    text = README_TEMPLATE.format(
        n32=int((frame["epsg"] == 25832).sum()),
        n33=int((frame["epsg"] == 25833).sum()),
        cells=len(frame),
        source_tiles=int(frame["n_source_tiles"].sum()),
        covered_km2=covered,
        tree_km2=tree,
        tree_cover_pct=100.0 * tree / covered if covered else 0.0,
        rows_per_strip=ROWS_PER_STRIP,
        cell_km2=(CELL_SIZE_M / 1000) ** 2,
        selected_from=provenance.get("candidates", totals.get("source_tiles", 0)),
        dropped=provenance.get("dropped", 0),
        snapped=int(frame["n_snapped"].sum()),
        max_snap_m=float(frame["max_snap_m"].max()),
        trimmed=int(frame["n_trimmed"].sum()),
        max_trim=MAX_TRIM_PX,
    )
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _plain(value):
    """NumPy scalars are not JSON-serialisable; plain Python is."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 6)
    return value


def _cell_polygon(cell: str) -> dict:
    """Cell envelope as a WGS 84 ring, densified so the edges stay true.

    A 10 km UTM square is not a quadrilateral in geographic coordinates;
    four corners would cut the corner by a few metres. Ten points per edge
    keeps the error below the pixel size.
    """
    from pyproj import Transformer

    from treecover.mosaic import CELL_SIZE_M

    epsg, left, bottom = parse_cell_id(cell)
    right, top = left + CELL_SIZE_M, bottom + CELL_SIZE_M
    steps = 10
    ring_xy = []
    for i in range(steps):
        ring_xy.append((left + (right - left) * i / steps, bottom))
    for i in range(steps):
        ring_xy.append((right, bottom + (top - bottom) * i / steps))
    for i in range(steps):
        ring_xy.append((right - (right - left) * i / steps, top))
    for i in range(steps):
        ring_xy.append((left, top - (top - bottom) * i / steps))
    ring_xy.append(ring_xy[0])

    transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    ring = [list(transformer.transform(x, y)) for x, y in ring_xy]
    return {"type": "Polygon", "coordinates": [ring]}


def _totals_from_frame(frame) -> dict:
    """National totals from an already-measured cell table."""
    covered = float(frame["covered_km2"].sum())
    tree = float(frame["tree_km2"].sum())
    return {
        "cells": int(len(frame)),
        "source_tiles": int(frame["n_source_tiles"].sum()),
        "covered_km2": covered,
        "tree_km2": tree,
        "tree_cover_pct": 100.0 * tree / covered if covered else 0.0,
        "covered_km2_projected": float(frame["covered_km2_projected"].sum()),
        "tree_km2_projected": float(frame["tree_km2_projected"].sum()),
        "skipped_tiles": int(frame["n_skipped"].sum()),
        "snapped_tiles": int(frame["n_snapped"].sum()),
        "max_snap_m": float(frame["max_snap_m"].max()),
        "trimmed_tiles": int(frame["n_trimmed"].sum()),
    }


def summarise(results: list[CellResult]) -> dict:
    """National totals over the finished cells."""
    rows = [r.as_row() for r in results if r.ok]
    covered = sum(r["covered_km2"] for r in rows)
    tree = sum(r["tree_km2"] for r in rows)
    return {
        "cells": len(rows),
        "source_tiles": sum(r["n_source_tiles"] for r in rows),
        "covered_km2": covered,
        "tree_km2": tree,
        "tree_cover_pct": 100.0 * tree / covered if covered else 0.0,
        "covered_km2_projected": sum(r["covered_km2_projected"] for r in rows),
        "tree_km2_projected": sum(r["tree_km2_projected"] for r in rows),
        "skipped_tiles": sum(r["n_skipped"] for r in rows),
        "snapped_tiles": sum(r["n_snapped"] for r in rows),
        "max_snap_m": max((r["max_snap_m"] for r in rows), default=0.0),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    root = args.predictions_root
    if root is None:
        from treecover.config import load_paths

        root = load_paths().get_path("predictions_root")
    if not root.is_dir():
        print(f"error: predictions root not found: {root}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    state_file = args.out / "_export_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}

    print(f"Predictions : {root}")
    print(f"Output      : {args.out}")

    candidates = collect_candidates(root, args.states)
    if not candidates:
        print(f"No prediction tiles under {root}.", file=sys.stderr)
        return 1

    choices = select_one_per_cell(candidates)
    winners = [c.winner for c in choices]
    dropped = sum(len(c.rejected) for c in choices)
    cells = group_by_cell(winners)

    print(f"{len(candidates):,} tiles -> {len(winners):,} selected "
          f"({dropped:,} dropped as overlaps)")
    print(f"{len(cells):,} output cells")

    # Recorded so a later --tables-only run can still describe where the
    # rasters came from without rescanning 380,000 files.
    (args.out / "_provenance.json").write_text(
        json.dumps({"candidates": len(candidates), "dropped": dropped,
                    "selected": len(winners), "cells": len(cells)}, indent=2)
    )

    todo = sorted(cells)
    if not args.force:
        todo = [c for c in todo
                if state.get(c) != len(cells[c])
                or not (args.out / f"UTM{parse_cell_id(c)[0] - 25800}" / f"{c}.tif").exists()]

    # Held back by --limit-cells is not the same as finished, and reporting
    # the two together would make a truncated debug run look complete.
    outstanding = len(todo)
    if args.limit_cells:
        todo = todo[: args.limit_cells]
    if args.tables_only:
        todo = []

    held_back = outstanding - len(todo)
    print(f"{len(todo):,} to build, {len(cells) - outstanding:,} up to date"
          + (f", {held_back:,} not attempted this run" if held_back else ""))

    if args.dry_run:
        for cell in todo[:20]:
            print(f"  {cell}: {len(cells[cell])} tiles")
        return 0

    results: list[CellResult] = []
    failures = 0
    if todo:
        ctx = mp.get_context("spawn")
        tasks = ((cell, cells[cell]) for cell in todo)
        with ctx.Pool(
            max(1, args.workers), initializer=_worker_init, initargs=(args.out,)
        ) as pool:
            for n, result in enumerate(pool.imap_unordered(_in_worker, tasks), 1):
                results.append(result)
                if result.ok:
                    state[result.cell] = len(cells[result.cell])
                    if n % 25 == 0 or n == len(todo):
                        state_file.write_text(json.dumps(state))
                    print(f"[{n}/{len(todo)}] {result.cell}: "
                          f"{result.n_source_tiles} tiles, "
                          f"{result.tree_px * 0.04 / 1e6:,.2f} km² tree"
                          + (f", {len(result.skipped)} skipped" if result.skipped else ""))
                else:
                    failures += 1
                    # Without the reasons a failed cell is undiagnosable:
                    # "no usable source tiles" says nothing about whether
                    # the tiles were corrupt, off-grid or in another CRS.
                    reasons = "; ".join(result.skipped[:3])
                    print(f"[{n}/{len(todo)}] FAIL {result.cell}: {result.error}"
                          + (f" | {reasons}" if reasons else ""),
                          file=sys.stderr)
        state_file.write_text(json.dumps(state))

    totals = finalise(args.out, results, cells)

    print("\nMosaic totals")
    print(f"  cells            {totals['cells']:,}")
    print(f"  source tiles     {totals['source_tiles']:,}")
    print(f"  covered          {totals['covered_km2']:,.0f} km²")
    print(f"  tree cover       {totals['tree_km2']:,.0f} km²  "
          f"({totals['tree_cover_pct']:.2f} % of covered)")
    print(f"  projected areas  {totals['tree_km2_projected']:,.0f} km² tree, "
          f"{totals['covered_km2_projected']:,.0f} km² covered")
    if totals["skipped_tiles"]:
        print(f"  skipped tiles    {totals['skipped_tiles']:,}")
    if failures:
        print(f"\n{failures} cell(s) failed.", file=sys.stderr)
    return 1 if failures else 0


def _results_from_disk(
    cells: dict, out_dir: Path, fresh: list[CellResult]
) -> list[CellResult]:
    """Re-measure cells finished in an earlier run, so the tables are complete.

    Counting the ones of a 1-bit raster is cheap next to building it, and
    it means the published table is measured from the published raster
    rather than from a log of what was intended.
    """
    import rasterio

    have = {r.cell: r for r in fresh if r.ok}
    results = list(fresh)
    for cell in sorted(cells):
        if cell in have:
            continue
        epsg, _, _ = parse_cell_id(cell)
        path = out_dir / f"UTM{epsg - 25800}" / f"{cell}.tif"
        if not path.exists():
            continue
        result = CellResult(cell=cell, epsg=epsg, n_source_tiles=len(cells[cell]))
        result.states = {c.state for c in cells[cell]}
        result.dates = [c.date for c in cells[cell] if c.is_dated]
        try:
            with rasterio.open(path) as src:
                tree = 0
                for _, window in src.block_windows(1):
                    tree += int(np.count_nonzero(src.read(1, window=window)))
            result.tree_px = tree
            # Coverage is a property of the sources, not of a 1-bit raster:
            # an uncovered pixel and a treeless one are both 0. Each source
            # tile contributes its own 1 km².
            result.covered_px = len(cells[cell]) * int((1_000 / PIXEL_M) ** 2)
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
        results.append(result)
    return results


if __name__ == "__main__":
    raise SystemExit(main())
