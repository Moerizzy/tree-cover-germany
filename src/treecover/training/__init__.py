"""Model training. Needs the ``dl`` extra."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .loop import EpochRecord, TrainConfig, TrainResult, train

__all__ = ["train", "TrainConfig", "TrainResult", "EpochRecord"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(".loop", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
