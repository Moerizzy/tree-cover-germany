"""Training data: patches, seasons, and the dataset that serves them.

Season handling (:mod:`~treecover.data.seasons`) is pure numpy/pandas and
imports without torch. The dataset itself defers torch and albumentations
to first use.
"""

from __future__ import annotations

from .observations import (
    SUMMER_MONTHS,
    Observation,
    build_observations,
    is_summer,
    label_source_index,
    parse_acquisition_date,
)
from .patches import Patch, extract_all, extract_patches, region_vrt_map, split_map
from .tile_sampling import (
    PUBLISHED_BIN_TARGETS,
    SamplingRun,
    assign_splits,
    assign_tcd_bins,
    border_tile_ids,
    seasonal_pattern,
    stratified_sample,
    tiles_from_index,
)
from .seasons import (
    LEAF_OFF,
    LEAF_ON,
    SEASON_OF_MONTH,
    SEASONS,
    TRANSITION,
    date_encoding,
    day_of_year,
    month_from_date,
    season_from_month,
    season_weights,
)

__all__ = [
    "SEASONS",
    "SEASON_OF_MONTH",
    "LEAF_OFF",
    "TRANSITION",
    "LEAF_ON",
    "season_from_month",
    "month_from_date",
    "day_of_year",
    "season_weights",
    "date_encoding",
    "Observation",
    "build_observations",
    "parse_acquisition_date",
    "is_summer",
    "label_source_index",
    "SUMMER_MONTHS",
    "Patch",
    "extract_patches",
    "extract_all",
    "region_vrt_map",
    "split_map",
    "SamplingRun",
    "tiles_from_index",
    "stratified_sample",
    "assign_splits",
    "assign_tcd_bins",
    "border_tile_ids",
    "seasonal_pattern",
    "PUBLISHED_BIN_TARGETS",
    "TreeCoverDataset",
    "training_augmentation",
    "validation_augmentation",
    "DATA_CONFIGS",
    "n_input_channels",
]

_LAZY = {
    "TreeCoverDataset": ".dataset",
    "training_augmentation": ".dataset",
    "validation_augmentation": ".dataset",
    "DATA_CONFIGS": ".dataset",
    "n_input_channels": ".dataset",
}


def __getattr__(name: str):
    """Defer the dataset module so seasons stay importable without torch."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
