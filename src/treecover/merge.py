"""Resolving which tile wins where coverage overlaps.

About 2.5 % of the 1 km cells in the archive are covered by more than one
prediction tile. Almost all of these sit on state borders, where two states
both fly the same ground: a cell can hold a Brandenburg tile from
2023-05-08 *and* a Mecklenburg-Vorpommern tile of the same area.

The original code left this to GDAL. Three revisions existed, each feeding
``gdalbuildvrt`` a differently sorted file list and relying on its source
ordering to break the tie: alphabetically by path, by file modification
time, and by acquisition date. Applied to the archive the three disagree
about every one of the 9,679 contested cells.

The manuscript does not prescribe a rule — its *Nationwide mapping* section
describes tile-level inference and says nothing about merging. Only one of
the three is a property of the imagery, though: modification time depends
on when inference was last re-run, and path order on the state code.
Neither would survive re-running the pipeline. Worse, they could contradict
the published acquisition-date figure, which states that an area came from
July 2023 while the raster actually showed a 2021 tile.

So there is one rule: **newest acquisition date wins**, with a
deterministic tiebreak so two runs over an unchanged archive produce the
same map. The choice is made here, before GDAL is involved, so VRT
source-ordering semantics decide nothing.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "TileCandidate",
    "CellChoice",
    "cell_key",
    "select_one_per_cell",
    "tile_date",
]

#: 1 km cell identity inside a filename: zone, easting km, northing km.
_CELL_RE = re.compile(r"(?<!\d)(3[23])_(\d{3})_(\d{4})(?!\d)")

#: Acquisition date, 8 digits not adjacent to further digits.
_DATE_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")

#: A tile whose filename carries no usable date.
UNDATED = "00000000"


@dataclass(frozen=True)
class TileCandidate:
    """One prediction tile competing for a cell."""

    path: Path
    state: str
    year: str
    date: str

    @property
    def is_dated(self) -> bool:
        """Whether a real acquisition date could be read from the filename."""
        return self.date != UNDATED and not self.date.endswith("0000")


@dataclass
class CellChoice:
    """The winning tile for a cell, and what it beat."""

    cell: tuple[str, str, str]
    winner: TileCandidate
    rejected: list[TileCandidate]

    @property
    def contested(self) -> bool:
        return bool(self.rejected)


def cell_key(path: Path) -> tuple[str, str, str] | None:
    """Identify the 1 km cell a tile covers, ignoring state and date.

    Returns ``None`` for filenames that do not encode a cell; those are
    passed through unmerged rather than silently grouped together.
    """
    match = _CELL_RE.search(path.name)
    return match.groups() if match else None


def tile_date(path: Path, year_fallback: str = "") -> str:
    """Acquisition date as sortable ``YYYYMMDD``.

    Falls back to ``<year>0000`` from the directory, then to
    :data:`UNDATED`. Both fallbacks sort before any real date, so a tile
    with a known date always beats one without.
    """
    for match in _DATE_RE.finditer(path.stem):
        token = match.group(1)
        year, month, day = int(token[:4]), int(token[4:6]), int(token[6:8])
        if 1990 <= year <= 2099 and 1 <= month <= 12 and 1 <= day <= 31:
            return token
    if year_fallback.isdigit() and len(year_fallback) == 4 and year_fallback != "0000":
        return f"{year_fallback}0000"
    return UNDATED


def _rank(candidate: TileCandidate):
    """Ranking key — higher sorts later, and the last one wins.

    Date first; state and filename only break ties, so the outcome is
    identical on every run and every machine.
    """
    return (candidate.date, candidate.state, candidate.path.name)


def select_one_per_cell(candidates: list[TileCandidate]) -> list[CellChoice]:
    """Choose exactly one tile per 1 km cell, newest acquisition date first.

    Args:
        candidates: Every prediction tile in scope.

    Returns:
        One :class:`CellChoice` per cell, in cell order.
    """
    by_cell: dict[tuple[str, str, str], list[TileCandidate]] = defaultdict(list)
    unkeyed: list[TileCandidate] = []
    for candidate in candidates:
        key = cell_key(candidate.path)
        if key is None:
            unkeyed.append(candidate)
        else:
            by_cell[key].append(candidate)

    if unkeyed:
        logger.warning(
            "%d tile(s) carry no recognisable cell id and are kept as-is; "
            "overlaps among them are not resolved (e.g. %s)",
            len(unkeyed), unkeyed[0].path.name,
        )

    choices: list[CellChoice] = []
    for cell in sorted(by_cell):
        ranked = sorted(by_cell[cell], key=_rank)
        choices.append(CellChoice(cell, ranked[-1], ranked[:-1]))

    for candidate in unkeyed:
        choices.append(CellChoice(("", "", ""), candidate, []))

    _report(choices)
    return choices


def _report(choices: list[CellChoice]) -> None:
    """Log how many cells were contested and how they resolved."""
    contested = [c for c in choices if c.contested]
    if not contested:
        logger.info("%d cell(s), none contested", len(choices))
        return

    logger.info(
        "%d cell(s); %d contested (%.1f %%), resolved by newest acquisition date",
        len(choices), len(contested), 100 * len(contested) / len(choices),
    )

    # A contest decided in favour of an undated tile is worth flagging: the
    # date rule cannot actually rank it, so the tiebreak did the work.
    undated_winners = [c for c in contested if not c.winner.is_dated]
    if undated_winners:
        logger.warning(
            "%d contested cell(s) were won by a tile with no acquisition date "
            "(e.g. %s). The date rule could not rank these — the tiebreak "
            "decided. Check whether those tiles should carry a date.",
            len(undated_winners), undated_winners[0].winner.path.name,
        )

    losers_by_state: dict[str, int] = defaultdict(int)
    for choice in contested:
        for rejected in choice.rejected:
            losers_by_state[rejected.state] += 1
    logger.info("Tiles dropped per state: %s", dict(sorted(losers_by_state.items())))
