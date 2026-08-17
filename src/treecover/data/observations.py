"""Building the tile-observation table.

A **tile** is a 1 km cell that was sampled for training. An **observation**
is one acquisition of that tile — the same tile flown in May 2022 and in
February 2023 gives two observations. Every observation of a tile shares
*one* label mask, produced from the tile's latest summer image where the
canopy is most legible.

That sharing is what makes multi-season training possible: the model sees
the same ground truth under leaf-on, transition and leaf-off imagery and
must learn to find the same trees in all of them. It also creates the two
constraints this module enforces.

**Temporal distance.** A label from summer 2023 does not describe a tile
flown in 2019 — trees were felled, hedges grew. Observations more than
``max_temporal_distance`` acquisitions away from the label source are
dropped rather than trained on as if the label still held.

**Tile-level splits.** Patches from one tile overlap, and observations of
one tile share a label. Splitting anywhere below tile level puts near-copies
of validation data into training and inflates every metric that follows.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "Observation",
    "parse_acquisition_date",
    "is_summer",
    "label_source_index",
    "build_observations",
    "SUMMER_MONTHS",
]

#: Months whose imagery is used as the label source. Full canopy makes the
#: reference mask most reliable.
SUMMER_MONTHS = (6, 7, 8)

_SEASON_RE = re.compile(r"(\d{4})[_-]?(summer|winter|spring|autumn)", re.IGNORECASE)
_SEASON_MONTH = {"winter": 1, "spring": 4, "summer": 7, "autumn": 10}


@dataclass
class Observation:
    """One acquisition of one tile."""

    obs_id: str
    tile_id: str
    date: str
    split: str
    image_path: Path
    mask_path: Path
    ndsm_path: Path | None = None


def parse_acquisition_date(value) -> datetime | None:
    """Parse the several date spellings the download stages produced.

    Handles ISO dates, ``YYYYMMDD``, and the season labels (``2018summer``)
    used before per-flight dates were available. Season labels map to a
    representative month so they still sort chronologically.

    Returns ``None`` when nothing parses, so a stray file is skipped rather
    than crashing the run.
    """
    text = str(value).strip()

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    match = _SEASON_RE.search(text)
    if match:
        return datetime(int(match.group(1)), _SEASON_MONTH[match.group(2).lower()], 15)

    parsed = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(parsed) else parsed.to_pydatetime()


def is_summer(value) -> bool:
    """Whether an acquisition falls in the label-source months."""
    parsed = parse_acquisition_date(value)
    return parsed is not None and parsed.month in SUMMER_MONTHS


def label_source_index(dates: list[str]) -> int | None:
    """Index of the tile's label source: its latest summer acquisition.

    Falls back to the latest acquisition of any season when a tile has no
    summer image at all, rather than dropping the tile.

    Public because the training-data figure has to mark the same
    acquisition as the label source that the pipeline labelled from. Two
    copies of this rule would let the figure caption and the training set
    disagree about which image the ground truth was drawn on.
    """
    if not dates:
        return None
    summer = [i for i, d in enumerate(dates) if is_summer(d)]
    return summer[-1] if summer else len(dates) - 1


def build_observations(
    image_dir: Path,
    mask_dir: Path,
    tiles: pd.DataFrame,
    ndsm_dir: Path | None = None,
    max_temporal_distance: int = 1,
    image_pattern: str = "*.tif",
    tile_id_column: str = "tile_id",
    split_column: str = "split",
    change_column: str | None = "Change",
) -> list[Observation]:
    """Assemble the observation table from files on disk.

    Image filenames must be ``<tile_id>_<date>.tif``; the label mask is
    looked up per tile, not per date.

    Args:
        image_dir: Orthophoto tiles.
        mask_dir: Label masks, one per tile.
        tiles: Sampled tiles carrying the tile id and the split assignment.
        ndsm_dir: Optional height models, matched per tile-date.
        max_temporal_distance: How many acquisitions away from the label
            source an observation may be. 1 keeps the neighbouring dates
            either side. 0 keeps only the label source itself.
        image_pattern: Glob for image files.
        tile_id_column: Column in ``tiles`` holding the tile id.
        split_column: Column holding ``train``/``val``/``test``.
        change_column: Tiles flagged ``'1'`` here changed between the label
            date and the imagery, and are dropped entirely. Pass ``None``
            to skip the check.

    Returns:
        Observations in deterministic order.
    """
    tiles = tiles.copy()
    if change_column and change_column in tiles.columns:
        before = len(tiles)
        tiles = tiles[tiles[change_column].astype(str) != "1"]
        if before != len(tiles):
            logger.info("Dropped %d tile(s) flagged %s='1'", before - len(tiles), change_column)

    split_of = dict(
        zip(tiles[tile_id_column].astype(str), tiles[split_column].astype(str))
    )
    logger.info("%d sampled tiles; splits %s", len(split_of),
                tiles[split_column].value_counts().to_dict())

    # Group image files by tile, chronologically.
    per_tile: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    skipped_unnamed = skipped_unsampled = 0
    for path in sorted(Path(image_dir).glob(image_pattern)):
        parts = path.stem.split("_")
        if len(parts) != 2:
            skipped_unnamed += 1
            continue
        tile_id, date = parts
        if tile_id not in split_of:
            skipped_unsampled += 1
            continue
        per_tile[tile_id].append((date, path))

    if skipped_unnamed:
        logger.info("Skipped %d file(s) not named <tile_id>_<date>", skipped_unnamed)
    if skipped_unsampled:
        logger.debug("Skipped %d file(s) for tiles outside the sample", skipped_unsampled)

    observations: list[Observation] = []
    dropped_far = missing_mask = 0

    for tile_id in sorted(per_tile):
        entries = sorted(per_tile[tile_id], key=lambda e: parse_acquisition_date(e[0])
                         or datetime.min)
        dates = [d for d, _ in entries]

        mask_path = _find_mask(mask_dir, tile_id)
        if mask_path is None:
            missing_mask += 1
            continue

        source = label_source_index(dates)
        if source is None:
            continue

        for index, (date, image_path) in enumerate(entries):
            if abs(index - source) > max_temporal_distance:
                dropped_far += 1
                continue
            observations.append(
                Observation(
                    obs_id=f"{tile_id}_{date}",
                    tile_id=tile_id,
                    date=date,
                    split=split_of[tile_id],
                    image_path=image_path,
                    mask_path=mask_path,
                    ndsm_path=_find_ndsm(ndsm_dir, tile_id, date),
                )
            )

    if missing_mask:
        logger.warning("%d tile(s) had no label mask and were skipped", missing_mask)
    if dropped_far:
        logger.info(
            "Dropped %d observation(s) more than %d acquisition(s) from their label "
            "source — the mask no longer describes that imagery",
            dropped_far, max_temporal_distance,
        )
    logger.info("Built %d observation(s) across %d tile(s)",
                len(observations), len({o.tile_id for o in observations}))
    return observations


def _find_mask(mask_dir: Path, tile_id: str) -> Path | None:
    """Locate a tile's label mask, whatever suffix the export used."""
    candidates = sorted(Path(mask_dir).glob(f"{tile_id}*.tif"))
    return candidates[0] if candidates else None


def _find_ndsm(ndsm_dir: Path | None, tile_id: str, date: str) -> Path | None:
    """Locate the height model for a tile-date, if height is being used."""
    if ndsm_dir is None:
        return None
    exact = Path(ndsm_dir) / f"{tile_id}_{date}.tif"
    if exact.exists():
        return exact
    candidates = sorted(Path(ndsm_dir).glob(f"{tile_id}_{date}*.tif"))
    return candidates[0] if candidates else None
