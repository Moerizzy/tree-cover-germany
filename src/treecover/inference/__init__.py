"""Nationwide moving-window inference.

Replaces the four ``04_run_inference*.py`` variants. They differed only in
where imagery came from, which is now :mod:`~treecover.inference.sources`;
the model application itself lives once in
:mod:`~treecover.inference.predictor`.

Torch and transformers are imported lazily. The patch geometry in
:mod:`~treecover.inference.tiling` is pure numpy, and the validation and
figure code depends on it — requiring the deep-learning stack just to lay
out patches would mean nobody can reproduce a figure without a GPU install.
Touching :class:`Predictor` or :func:`run_inference` triggers the import;
everything else does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .sources import (
    HttpTileIndexSource,
    LocalRasterSource,
    TileData,
    TileSource,
    TileTask,
    VrtSource,
    resample_to_resolution,
)
from .tiling import LogitAccumulator, Patch, normalise, plan_patches

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .predictor import InferenceConfig, Predictor, TilePrediction
    from .runner import InferenceRun, TileReport, run_inference

#: Names resolved on first access, mapped to the module providing them.
_LAZY = {
    "InferenceConfig": ".predictor",
    "Predictor": ".predictor",
    "TilePrediction": ".predictor",
    "InferenceRun": ".runner",
    "TileReport": ".runner",
    "run_inference": ".runner",
}

__all__ = [
    "TileSource",
    "TileTask",
    "TileData",
    "LocalRasterSource",
    "VrtSource",
    "HttpTileIndexSource",
    "resample_to_resolution",
    "Patch",
    "plan_patches",
    "LogitAccumulator",
    "normalise",
    *_LAZY,
]


def __getattr__(name: str):
    """Import torch-dependent names on first use (PEP 562)."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
