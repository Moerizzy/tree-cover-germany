"""Phenological seasons and season-aware sampling weights.

This is the paper's central mechanism. National aerial surveys are flown
over years, and the resulting imagery is dominated by summer acquisitions:
leaf-off and transition imagery is a small minority of any training set
drawn uniformly. A model trained on that distribution learns "tree = green
blob" and degrades badly on bare winter canopy.

Weighting each patch by the inverse frequency of its season makes every
phenological stage equally likely to be drawn per epoch, without discarding
data. That, and not the architecture, is what lets a model trained in one
federal state transfer to all sixteen.

The three-way split is coarser than meteorological seasons on purpose. What
matters to the model is canopy state, and April and October look alike to it
even though one is spring and the other autumn.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "Season",
    "SEASON_OF_MONTH",
    "season_from_month",
    "month_from_date",
    "day_of_year",
    "season_weights",
    "date_encoding",
]

#: The three phenological stages.
LEAF_OFF = "leaf_off"
TRANSITION = "transition"
LEAF_ON = "leaf_on"
Season = str

#: Month → phenological stage. Nov–Mar is bare canopy; Jun–Aug is full
#: canopy; the months between are partial and are where the model is most
#: likely to fail, which is why they get their own class.
SEASON_OF_MONTH = {
    1: LEAF_OFF, 2: LEAF_OFF, 3: LEAF_OFF,
    4: TRANSITION, 5: TRANSITION,
    6: LEAF_ON, 7: LEAF_ON, 8: LEAF_ON,
    9: TRANSITION, 10: TRANSITION,
    11: LEAF_OFF, 12: LEAF_OFF,
}

SEASONS = (LEAF_OFF, TRANSITION, LEAF_ON)


def month_from_date(value) -> int | None:
    """Month from a date in any of the formats the project produced.

    Accepts ``YYYYMMDD``, ``YYYY-MM-DD``, a datetime, or a pandas
    Timestamp. Returns ``None`` for anything unparseable, so a patch with a
    missing date degrades to the default season rather than crashing a
    training run at epoch 12.
    """
    if value is None:
        return None
    text = str(value).replace("-", "").replace("/", "").strip()
    if len(text) >= 6 and text[:6].isdigit():
        month = int(text[4:6])
        if 1 <= month <= 12:
            return month
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else int(parsed.month)


def season_from_month(month: int | None) -> Season:
    """Phenological stage for a month.

    An unknown month falls back to ``leaf_off``: that is the minority class,
    so a mislabelled patch is up-weighted rather than diluting the majority.
    """
    if month is None:
        return LEAF_OFF
    return SEASON_OF_MONTH.get(month, LEAF_OFF)


def day_of_year(value) -> tuple[int | None, int | None]:
    """``(day_of_year, days_in_year)`` for a date, or ``(None, None)``."""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None, None
    days = 366 if bool(getattr(parsed, "is_leap_year", False)) else 365
    return int(parsed.dayofyear), days


def season_weights(dates) -> tuple[np.ndarray, dict[str, int], dict[str, float]]:
    """Per-patch sampling weights that equalise the three seasons.

    Each patch is weighted by ``N / (K * n_season)`` where ``N`` is the
    patch count, ``K`` the number of seasons actually present and
    ``n_season`` the count in that patch's season. Feeding these to a
    :class:`~torch.utils.data.WeightedRandomSampler` with replacement draws
    each season roughly equally often per epoch.

    Seasons absent from the data get weight 0 rather than dividing by zero,
    and ``K`` counts only the present ones — otherwise a two-season dataset
    would be under-sampled by a third.

    Args:
        dates: One acquisition date per training patch, in patch order.

    Returns:
        ``(weights, counts, per_season_weight)``. ``weights`` aligns with
        ``dates``; the other two are for logging and for the reproducibility
        record.
    """
    seasons = [season_from_month(month_from_date(d)) for d in dates]
    counts = Counter(seasons)
    total = len(seasons)
    present = sum(1 for s in SEASONS if counts.get(s, 0) > 0)

    if total == 0 or present == 0:
        raise ValueError("No patches to weight — check the split and the date column.")

    per_season = {
        s: (total / (present * counts[s])) if counts.get(s, 0) > 0 else 0.0 for s in SEASONS
    }
    weights = np.array([per_season[s] for s in seasons], dtype=np.float64)

    logger.info(
        "Season-aware sampling over %d patches: leaf-off %d, transition %d, leaf-on %d",
        total, counts.get(LEAF_OFF, 0), counts.get(TRANSITION, 0), counts.get(LEAF_ON, 0),
    )
    if present < len(SEASONS):
        missing = [s for s in SEASONS if counts.get(s, 0) == 0]
        logger.warning(
            "No training patches for season(s) %s. The model will not see that "
            "canopy state — expect degraded accuracy on imagery acquired then.",
            ", ".join(missing),
        )

    return weights, dict(counts), per_season


def date_encoding(value, height: int, width: int) -> np.ndarray:
    """Two constant channels encoding day-of-year cyclically.

    ``sin`` and ``cos`` of the day-of-year angle, each rescaled to [0, 1] so
    the same ``Normalize(mean=0.5, std=0.5)`` applies as to the imagery
    bands. Cyclical rather than linear so that 31 December and 1 January are
    adjacent instead of maximally far apart.

    An unparseable date encodes as ``(0.5, 0.5)``, the value both channels
    normalise to zero at — a missing date contributes nothing rather than
    pointing at an arbitrary time of year.

    .. note::
       The published model does **not** use this (its checkpoint is named
       ``nodateenc``). Adding the channels did not improve accuracy — see
       the ablation in the paper. It is kept because the ablation is part of
       the record.
    """
    doy, days = day_of_year(value)
    if doy is None or not days:
        sin_val = cos_val = 0.5
    else:
        angle = 2 * np.pi * max(1, min(doy, days)) / days
        sin_val = (np.sin(angle) + 1.0) / 2.0
        cos_val = (np.cos(angle) + 1.0) / 2.0

    return np.stack(
        [
            np.full((height, width), sin_val, dtype=np.float32),
            np.full((height, width), cos_val, dtype=np.float32),
        ],
        axis=0,
    )
