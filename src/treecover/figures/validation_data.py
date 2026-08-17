"""Loading the per-state validation metrics for figures 4 and 5.

Replicates the preparation in ``09_validation_results_summary_new.ipynb``
cells 3 and 5, which both figures depend on. Two parts of it change the
numbers and are therefore not incidental:

**Zero-LiDAR patches are kept and scored explicitly.** Where the LiDAR finds
no trees, IoU is undefined by the usual formula. The notebook assigns them
by hand: if the model also finds none it is a correct negative and scores
1.0; if the model finds trees it is a false positive and scores 0.0, with
accuracy set to the share of background it did get right. Dropping these
patches instead would remove exactly the cases that test whether the model
hallucinates canopy.

**Month labels are normalised.** The three states record the acquisition
month as ``5``, ``5.0``, ``Month_05`` or ``May`` depending on when their
validation ran. All become the full month name, or the figures show four
different spellings of the same month as four categories.
"""

from __future__ import annotations

import calendar
import logging
import re
from pathlib import Path

import pandas as pd

from treecover.validation.metrics import score_zero_reference

logger = logging.getLogger(__name__)

__all__ = [
    "STATES",
    "STATE_NAMES",
    "METRIC_COLUMNS",
    "month_to_name",
    "load_validation_metrics",
]

#: Panel order in the manuscript.
STATES = ("NRW", "BB", "BY")

STATE_NAMES = {
    "NRW": "North Rhine-Westphalia",
    "BB": "Brandenburg",
    "BY": "Bavaria",
}

METRIC_COLUMNS = (
    "overall_iou",
    "overall_f1",
    "overall_precision",
    "overall_recall",
    "overall_accuracy",
)

LIDAR_COLUMN = "lidar_tree_cover_pct"
MODEL_COLUMN = "model_tree_cover_pct"

_MONTH_RE = re.compile(r"[Mm]onth_(\d+)")


def month_to_name(value) -> str:
    """Normalise any month spelling to a full name.

    ``5``, ``5.0``, ``'Month_05'`` and ``'May'`` all become ``'May'``.
    Anything unrecognisable becomes ``'Unknown'`` rather than being dropped,
    so a bad label is visible in the figure instead of silently shrinking
    the sample.
    """
    if value is None:
        return "Unknown"
    text = str(value).strip()
    if text in calendar.month_name:
        return text

    match = _MONTH_RE.match(text)
    if match and 1 <= int(match.group(1)) <= 12:
        return calendar.month_name[int(match.group(1))]

    try:
        number = int(float(text))
    except (TypeError, ValueError):
        return "Unknown"
    return calendar.month_name[number] if 1 <= number <= 12 else "Unknown"


def _find_csv(directory: Path, state: str) -> Path | None:
    """Locate a state's metrics file under either naming convention."""
    for name in (f"metrics_per_sample_{state}.csv",
                 f"validation_metrics_{state}.csv"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    # NRW is stored as NW in the collected data.
    if state == "NRW":
        return _find_csv(directory, "NW")
    return None


def _score_zero_lidar(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign metrics to patches where the LiDAR reports no trees.

    The rule itself lives in
    :func:`treecover.validation.metrics.score_zero_reference`, because the
    accuracy table of the paper and these figures have to agree about it —
    it is the difference between a mean IoU of 0.844 and one of 0.771.
    """
    return score_zero_reference(frame, reference_column=LIDAR_COLUMN,
                                model_column=MODEL_COLUMN, prefix="overall_")


def load_validation_metrics(
    directory: Path, exclude_zero_lidar: bool = False
) -> pd.DataFrame:
    """Load and prepare the per-state validation metrics.

    Args:
        directory: Holds one CSV per state.
        exclude_zero_lidar: Drop patches where the LiDAR reports no trees.
            The published figures keep them.

    Returns:
        One frame for all states, with a ``state`` column, normalised month
        labels and zero-LiDAR patches scored.

    Raises:
        FileNotFoundError: If no state file is found at all.
    """
    directory = Path(directory)
    frames = []
    for state in STATES:
        path = _find_csv(directory, state)
        if path is None:
            logger.warning("No metrics file for %s in %s", state, directory)
            continue
        frame = pd.read_csv(path)
        frame["state"] = state
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No validation metrics found in {directory}. Expected "
            "metrics_per_sample_<STATE>.csv or validation_metrics_<STATE>.csv."
        )

    combined = pd.concat(frames, ignore_index=True)

    if "month_bin" in combined.columns:
        combined["month_bin"] = combined["month_bin"].apply(month_to_name)
    elif "month" in combined.columns:
        combined["month_bin"] = combined["month"].apply(month_to_name)

    if "success" in combined.columns:
        combined = combined[combined["success"].fillna(True).astype(bool)]

    combined = _score_zero_lidar(combined)

    if exclude_zero_lidar and LIDAR_COLUMN in combined.columns:
        before = len(combined)
        combined = combined[combined[LIDAR_COLUMN] > 0]
        logger.info("Excluded %d zero-LiDAR patch(es)", before - len(combined))

    logger.info("Loaded %d patch(es) across %d state(s)",
                len(combined), combined["state"].nunique())
    return combined
