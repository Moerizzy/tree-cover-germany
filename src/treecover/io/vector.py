"""Reading vector data across incompatible GeoPandas/Fiona combinations.

GeoPandas 0.13 calls ``fiona.path``, which Fiona 1.10 removed. That pairing
is exactly what the project's inference container ships, and it makes plain
``geopandas.read_file`` raise ``AttributeError: module 'fiona' has no
attribute 'path'`` on every call.

Rather than pin versions the container cannot change, everything in this
package reads vectors through :func:`read_vector`, which prefers the
``pyogrio`` engine (unaffected, and the default from GeoPandas 0.14) and
falls back to Fiona. If both fail it raises with the actual fix rather than
the confusing ``fiona.path`` traceback.

The original notebooks worked around this by hand-parsing GeoJSON with the
``json`` module, which only worked for GeoJSON and lost the CRS.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd

logger = logging.getLogger(__name__)

__all__ = ["read_vector", "available_engine"]


@lru_cache(maxsize=1)
def available_engine() -> str | None:
    """The vector engine to use, or ``None`` if GeoPandas must choose.

    Resolved once per process: probing costs an import and the answer
    cannot change mid-run.
    """
    try:
        import pyogrio  # noqa: F401

        return "pyogrio"
    except ImportError:
        pass

    try:
        import fiona

        # GeoPandas < 0.14 reaches into fiona.path, dropped in Fiona 1.10.
        if not hasattr(fiona, "path") and _geopandas_version() < (0, 14):
            logger.warning(
                "GeoPandas %s with Fiona %s is a broken combination "
                "(GeoPandas calls the removed fiona.path). Install pyogrio "
                "(`pip install pyogrio`) or upgrade GeoPandas to >= 0.14.",
                gpd.__version__, fiona.__version__,
            )
        return "fiona"
    except ImportError:
        return None


def _geopandas_version() -> tuple[int, ...]:
    parts = []
    for chunk in gpd.__version__.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def read_vector(path: str | Path, layer: str | None = None, **kwargs: Any) -> gpd.GeoDataFrame:
    """Read a vector file with whichever engine works in this environment.

    Args:
        path: GeoPackage, GeoJSON, Shapefile — anything GDAL reads.
        layer: Layer name, for multi-layer formats such as GeoPackage.
        **kwargs: Passed through to :func:`geopandas.read_file`.

    Returns:
        The layer as a GeoDataFrame.

    Raises:
        RuntimeError: If no engine can read the file, with the concrete fix
            rather than an ``AttributeError`` from deep inside GeoPandas.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")

    if layer is not None:
        kwargs["layer"] = layer

    engine = available_engine()
    attempts = [engine] if engine else []
    # Try the other engine too — a file may be readable by one and not the other.
    attempts += [e for e in ("pyogrio", "fiona", None) if e not in attempts]

    errors: list[str] = []
    for candidate in attempts:
        try:
            if candidate is None:
                return gpd.read_file(path, **kwargs)
            return gpd.read_file(path, engine=candidate, **kwargs)
        except ImportError as exc:
            errors.append(f"{candidate}: not installed ({exc})")
        except AttributeError as exc:
            # The signature of the GeoPandas/Fiona mismatch.
            errors.append(f"{candidate}: {exc}")
        except (TypeError, ValueError) as exc:
            # Older GeoPandas has no `engine` parameter at all.
            if "engine" in str(exc):
                errors.append(f"{candidate}: engine parameter unsupported")
                continue
            raise

    raise RuntimeError(
        f"Could not read {path.name} with any vector engine.\n"
        + "\n".join(f"  - {e}" for e in errors)
        + "\n\nMost likely cause: GeoPandas < 0.14 together with Fiona >= 1.10.\n"
        "Fix with either:  pip install pyogrio   (recommended)\n"
        "             or:  pip install 'geopandas>=0.14'"
    )
