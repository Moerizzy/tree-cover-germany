"""Choosing which training tiles the training-data figure draws.

The figure has one job: show that a training tile is not a single image but
a *pair* — one summer acquisition carrying the label, and one non-summer
acquisition of the same ground that inherits it. Everything here exists to
pick pairs that actually demonstrate that, and to fail loudly when they do
not exist rather than drawing a summer image twice.

Two rules decide the selection.

**The label source is not re-derived.** It comes from
:func:`~treecover.data.observations.label_source_index`, the same function
:func:`~treecover.data.observations.build_observations` used when the
training set was assembled. A figure that picked its own "summer image"
could caption an acquisition as the label source that the pipeline never
labelled from.

**The partner is the most phenologically distant acquisition, not the
nearest in time.** The three stages sit on an axis — leaf-off, transition,
leaf-on — and the contrast worth showing is the widest one available for
that tile. A tile whose only partner is another leaf-on image has no
contrast to show and is skipped by default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.observations import Observation, label_source_index
from ..data.seasons import LEAF_OFF, LEAF_ON, TRANSITION, month_from_date, season_from_month

logger = logging.getLogger(__name__)

__all__ = [
    "TilePair",
    "SEASON_LABELS",
    "PHENOLOGY_RANK",
    "season_contrast",
    "pair_observations",
    "tile_tree_cover_pct",
    "select_examples",
]

#: Display spelling of the three stages. The code uses underscores; a figure
#: caption uses hyphens, and the manuscript's own wording is "leaf-on".
SEASON_LABELS = {
    LEAF_OFF: "leaf-off",
    TRANSITION: "transition",
    LEAF_ON: "leaf-on",
}

#: The stages ordered along the phenological axis, so the distance between
#: two of them is a subtraction. Leaf-off to leaf-on is the full span (2);
#: either to transition is half of it (1).
PHENOLOGY_RANK = {LEAF_OFF: 0, TRANSITION: 1, LEAF_ON: 2}

#: Nodata in the training label masks, as written by the labelling stage.
#: Matches :data:`treecover.data.patches.MASK_NODATA`.
LABEL_NODATA = 255

#: How far either side of a target rank :func:`select_examples` may look for
#: a better seasonal contrast. Two ranks out of a hundred-odd tiles moves the
#: canopy share by a fraction of a point, so the spread survives.
NEIGHBOURHOOD = 2


@dataclass
class TilePair:
    """One training tile as the figure draws it: label source plus partner.

    Attributes:
        tile_id: The 1 km tile.
        label_source: The acquisition the label was drawn on — summer,
            except for tiles that have no summer image at all.
        partner: The acquisition that inherits the label, or ``None`` when
            the tile has only one acquisition left after the temporal
            distance filter.
        tree_cover_pct: Canopy share of the label mask, valid pixels only.
            Drives the spread in :func:`select_examples`.
        is_urban: The tile's settlement stratum, from the sampled-tile
            table. ``None`` when the table does not say, which makes the
            tile eligible for the non-urban rows only.
    """

    tile_id: str
    label_source: Observation
    partner: Observation | None = None
    tree_cover_pct: float = float("nan")
    is_urban: bool | None = None

    @property
    def label_season(self) -> str:
        return season_from_month(month_from_date(self.label_source.date))

    @property
    def partner_season(self) -> str | None:
        if self.partner is None:
            return None
        return season_from_month(month_from_date(self.partner.date))

    @property
    def contrast(self) -> int:
        """Phenological distance between the two acquisitions, 0–2."""
        if self.partner is None:
            return 0
        return season_contrast(self.label_season, self.partner_season)

    @property
    def has_ndsm(self) -> bool:
        return self.label_source.ndsm_path is not None


def season_contrast(first: str, second: str) -> int:
    """Distance between two phenological stages along the leaf-off↔leaf-on axis.

    ``0`` means the same stage — nothing for the figure to show.
    """
    return abs(PHENOLOGY_RANK.get(first, 0) - PHENOLOGY_RANK.get(second, 0))


def pair_observations(observations: list[Observation]) -> list[TilePair]:
    """Group observations into one pair per tile.

    Args:
        observations: The table stage 3 built, for any mix of tiles.

    Returns:
        One :class:`TilePair` per tile, tile id order. Tiles whose partner
        is missing are returned with ``partner=None`` rather than dropped,
        so the caller can report how many were skipped and why.
    """
    by_tile: dict[str, list[Observation]] = {}
    for observation in observations:
        by_tile.setdefault(observation.tile_id, []).append(observation)

    pairs: list[TilePair] = []
    for tile_id in sorted(by_tile):
        entries = sorted(by_tile[tile_id], key=lambda o: str(o.date))
        index = label_source_index([o.date for o in entries])
        if index is None:
            continue
        source = entries[index]
        source_season = season_from_month(month_from_date(source.date))

        # Widest phenological contrast wins; the later acquisition breaks a
        # tie, so the choice does not depend on directory iteration order.
        candidates = [o for i, o in enumerate(entries) if i != index]
        partner = max(
            candidates,
            key=lambda o: (
                season_contrast(source_season,
                                season_from_month(month_from_date(o.date))),
                str(o.date),
            ),
            default=None,
        )
        pairs.append(TilePair(tile_id=tile_id, label_source=source, partner=partner))
    return pairs


def tile_tree_cover_pct(mask_path: Path, max_pixels: int = 512) -> float:
    """Canopy share of a label mask, in percent of valid pixels.

    Read decimated: the statistic only has to rank tiles against each other,
    and a training tile is 5000 × 5000 px at 20 cm. Reading all 117 at full
    resolution to sort three of them to the top costs minutes for a number
    that is stable to a tenth of a percent at 512 px.

    Returns:
        Percent canopy, or ``NaN`` if the mask is unreadable or wholly
        nodata — ``NaN`` sorts out of the way in :func:`select_examples`
        instead of masquerading as a treeless tile.
    """
    import rasterio

    try:
        with rasterio.open(mask_path) as src:
            scale = min(1.0, max_pixels / max(src.height, src.width, 1))
            shape = (max(1, int(src.height * scale)), max(1, int(src.width * scale)))
            mask = src.read(1, out_shape=shape)
    except (rasterio.RasterioIOError, ValueError) as exc:
        logger.warning("Cannot read label mask %s: %s", Path(mask_path).name, exc)
        return float("nan")

    valid = mask != LABEL_NODATA
    if not valid.any():
        return float("nan")
    return float(100.0 * np.count_nonzero(mask[valid]) / valid.sum())


def select_examples(
    pairs: list[TilePair],
    count: int = 3,
    require_contrast: bool = True,
    require_ndsm: bool = True,
    urban_rows: int = 1,
) -> list[TilePair]:
    """Pick the tiles to draw, across both axes the training set was drawn on.

    The sample was stratified along tree cover density **and** settlement
    type, so the figure is too. ``urban_rows`` of the rows are drawn from
    the urban stratum and the rest from the non-urban one; within each
    stratum the tiles are sorted by canopy share and sampled at evenly
    spaced quantiles. Rows chosen on cover alone come out entirely rural —
    urban tiles are the minority of the sample — and would show a sampling
    design the paper does not claim.

    The quantiles are the *bin centres* ``(i + 0.5) / count``, not the
    endpoints. Endpoints put the extremes in the figure, and at this sample
    size the extremes are a heath tile with 2 % canopy — whose nDSM and
    label panels are both blank — and a closed-canopy tile whose label is a
    solid green square. Neither shows a label being made. The centres keep
    the same spread while landing on tiles that have something in them.

    Args:
        pairs: Candidates, with ``tree_cover_pct`` and ``is_urban`` filled in.
        count: How many rows the figure has.
        require_contrast: Drop tiles whose two acquisitions fall in the same
            phenological stage. They cannot illustrate the pairing, which is
            the point of the figure.
        require_ndsm: Drop tiles with no height model on disk, which would
            leave the nDSM column empty.
        urban_rows: How many rows to reserve for the urban stratum. Clamped
            to what the candidates actually offer, and to ``count``; 0 turns
            the split off and ranks every tile together.

    Returns:
        Up to ``count`` pairs, ordered by ascending tree cover so the rows
        read as a gradient. Fewer than ``count`` if the candidates run out.
    """
    candidates = [p for p in pairs if p.partner is not None]
    dropped_single = len(pairs) - len(candidates)

    if require_contrast:
        before = len(candidates)
        candidates = [p for p in candidates if p.contrast > 0]
        dropped_flat = before - len(candidates)
    else:
        dropped_flat = 0

    if require_ndsm:
        before = len(candidates)
        candidates = [p for p in candidates if p.has_ndsm]
        dropped_ndsm = before - len(candidates)
    else:
        dropped_ndsm = 0

    for label, n in (("no partner acquisition", dropped_single),
                     ("no seasonal contrast", dropped_flat),
                     ("no nDSM on disk", dropped_ndsm)):
        if n:
            logger.info("Skipped %d tile(s): %s", n, label)

    scored = [p for p in candidates if not np.isnan(p.tree_cover_pct)]
    if not scored:
        # Better a figure of unranked tiles than no figure: without cover
        # statistics the spread is meaningless, but the pairs are still valid.
        logger.warning("No tile has a readable label mask — falling back to "
                       "tile id order, rows will not span the cover gradient")
        return candidates[:count]

    scored.sort(key=lambda p: p.tree_cover_pct)
    if count >= len(scored):
        return scored

    urban = [p for p in scored if p.is_urban]
    rural = [p for p in scored if not p.is_urban]

    n_urban = max(0, min(urban_rows, count, len(urban)))
    if urban_rows and n_urban < urban_rows:
        logger.warning(
            "Only %d urban tile(s) qualify, %d row(s) requested — the "
            "shortfall goes to non-urban rows", len(urban), urban_rows,
        )
    # Whatever the urban stratum cannot fill stays with the non-urban one,
    # so the figure keeps its row count rather than losing a row. The urban
    # rows go first and the rural ones then avoid their flight dates.
    chosen = _spread(urban, n_urban)
    chosen += _spread(rural, count - n_urban,
                      avoid_dates={p.label_source.date for p in chosen})
    return sorted(chosen, key=lambda p: p.tree_cover_pct)


def _spread(
    pool: list[TilePair], count: int, avoid_dates: set[str] | None = None
) -> list[TilePair]:
    """``count`` tiles from ``pool``, evenly spread across its cover range.

    ``pool`` must already be sorted by canopy share. Targets are the bin
    centres; around each, the best tile within :data:`NEIGHBOURHOOD` ranks
    wins on three keys in order.

    First a flight date no other row already uses. Neighbouring tiles were
    often sampled from the same two acquisitions, so without this a figure
    can show one flight pair twice and read as though the training set held
    a handful of dates. Then the widest seasonal contrast — a summer/winter
    pair says more than one two weeks apart in late August. Then closeness
    to the target rank, which keeps the spread.

    All three tiebreaks move canopy share by a fraction of a percentage
    point, so the gradient the rows are chosen for survives them.
    """
    if count <= 0 or not pool:
        return []
    if count >= len(pool):
        return list(pool)

    used_dates = set(avoid_dates or ())
    last = len(pool) - 1
    targets = [min(last, int(round(((i + 0.5) / count) * last))) for i in range(count)]

    chosen: list[TilePair] = []
    taken: set[int] = set()
    for target in targets:
        window = [i for i in range(max(0, target - NEIGHBOURHOOD),
                                   min(last, target + NEIGHBOURHOOD) + 1)
                  if i not in taken]
        if not window:
            continue
        best = max(window, key=lambda i: (
            pool[i].label_source.date not in used_dates,
            pool[i].contrast,
            -abs(i - target),
        ))
        taken.add(best)
        used_dates.add(pool[best].label_source.date)
        chosen.append(pool[best])
    return chosen
