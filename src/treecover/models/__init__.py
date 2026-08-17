"""Model construction, shared by training and inference.

Everything here needs torch and transformers, so the imports are deferred:
merely importing :mod:`treecover` must not require the deep-learning stack.
Install it with ``pip install -e ".[dl]"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .segformer import (
        N_CLASSES,
        TCD_PRETRAINED,
        create_segformer,
        extract_logits,
        infer_backbone_from_path,
        load_checkpoint,
    )

__all__ = [
    "create_segformer",
    "load_checkpoint",
    "extract_logits",
    "infer_backbone_from_path",
    "N_CLASSES",
    "TCD_PRETRAINED",
]


def __getattr__(name: str):
    """Import from :mod:`.segformer` on first use (PEP 562)."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(".segformer", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
