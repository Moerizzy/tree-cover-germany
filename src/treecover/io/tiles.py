"""Discovery and naming for the tiled prediction archive.

The archive follows one layout throughout::

    <predictions_root>/<STATE>/<YEAR>/predictions/<UTM_TILE>/<name>_pred.tif

and derived products mirror it, swapping the ``predictions`` level for the
product name::

    <predictions_root>/<STATE>/<YEAR>/<product>/<UTM_TILE>/<name>_<product>.tif

Everything that needs to walk, name or date a tile does it through this
module, so the layout is described once.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile

__all__ = [
    "TileRef",
    "find_prediction_tiles",
    "derived_path",
    "acquisition_date",
    "build_predictions_vrt",
    "predictions_vrt_path",
]

# A plausible acquisition date: 8 digits not adjacent to further digits.
_DATE_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")

PRED_SUFFIX = "_pred.tif"


class TileRef:
    """A prediction tile together with the state and year it belongs to."""

    __slots__ = ("path", "state", "year")

    def __init__(self, path: Path, state: str, year: str):
        self.path = path
        self.state = state
        self.year = year

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TileRef({self.path.name!r}, {self.state!r}, {self.year!r})"

    def __iter__(self):
        """Unpack as ``path, state, year``."""
        return iter((self.path, self.state, self.year))


def find_prediction_tiles(
    predictions_root: Path,
    states: list[str] | None = None,
    years: list[str] | None = None,
) -> Iterator[TileRef]:
    """Yield every prediction tile under ``predictions_root``.

    Args:
        predictions_root: Directory holding one subdirectory per state.
        states: Restrict to these state directory names (case-insensitive).
        years: Restrict to these year directory names.

    Yields:
        :class:`TileRef` in a deterministic order, so a run can be resumed
        or sharded reproducibly.
    """
    if not predictions_root.is_dir():
        raise FileNotFoundError(f"Predictions root does not exist: {predictions_root}")

    state_dirs = sorted(d for d in predictions_root.iterdir() if d.is_dir())
    if states:
        wanted = {s.upper() for s in states}
        state_dirs = [d for d in state_dirs if d.name.upper() in wanted]

    for state_dir in state_dirs:
        for year_dir in sorted(d for d in state_dir.iterdir() if d.is_dir()):
            if years and year_dir.name not in years:
                continue
            pred_dir = year_dir / "predictions"
            if not pred_dir.is_dir():
                continue
            for tif in sorted(pred_dir.rglob(f"*{PRED_SUFFIX}")):
                yield TileRef(tif, state_dir.name, year_dir.name)


def derived_path(pred_path: Path, subdir: str, suffix: str) -> Path:
    """Mirror a prediction tile's path into a derived-product directory.

    ``.../2024/predictions/UTM32_E4100_N52900/x_pred.tif``
    becomes ``.../2024/<subdir>/UTM32_E4100_N52900/x<suffix>``.
    """
    year_dir = pred_path.parent.parent.parent
    utm_tile = pred_path.parent.name
    name = pred_path.name
    if name.endswith(PRED_SUFFIX):
        out_name = name[: -len(PRED_SUFFIX)] + suffix
    else:
        out_name = pred_path.stem + suffix
    return year_dir / subdir / utm_tile / out_name


def acquisition_date(pred_path: Path, year_fallback: str) -> str:
    """Extract ``YYYYMMDD`` from a tile filename.

    Falls back to ``<year>0000`` when the name carries no date token, and to
    ``00000000`` when the year directory is not numeric either. The result
    is always 8 digits so that lexical order equals chronological order.
    """
    for match in _DATE_RE.finditer(pred_path.stem):
        token = match.group(1)
        year, month, day = int(token[:4]), int(token[4:6]), int(token[6:8])
        if 1990 <= year <= 2099 and 1 <= month <= 12 and 1 <= day <= 31:
            return token
    if year_fallback.isdigit() and len(year_fallback) == 4:
        return f"{year_fallback}0000"
    return "00000000"


def predictions_vrt_path(pred_path: Path) -> Path:
    """Path of the per-(state, year) predictions VRT for this tile."""
    return pred_path.parent.parent.parent / "vrt_predictions.vrt"


def build_predictions_vrt(
    pred_dir: Path, vrt_path: Path, overwrite: bool = False
) -> Path | None:
    """Build a VRT covering every prediction tile under ``pred_dir``.

    A derived product may need a buffered halo around each tile so that
    features crossing a tile edge are classified from their full extent
    rather than being cut in two. The halo comes from this VRT.

    Args:
        pred_dir: Directory to scan recursively for ``*_pred.tif``.
        vrt_path: Where to write the VRT.
        overwrite: Rebuild even if the VRT already exists.

    Returns:
        The VRT path, or ``None`` if no source tiles were found.

    Raises:
        RuntimeError: If ``gdalbuildvrt`` is unavailable or fails.
    """
    if vrt_path.exists() and not overwrite:
        return vrt_path

    sources = sorted(pred_dir.rglob(f"*{PRED_SUFFIX}"))
    if not sources:
        return None
    if shutil.which("gdalbuildvrt") is None:
        raise RuntimeError(
            "gdalbuildvrt not found on PATH. Install GDAL, or run with --buffer 0 "
            "to classify each tile in isolation."
        )

    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write("\n".join(str(p) for p in sources))
        list_file = tmp.name
    try:
        proc = subprocess.run(
            ["gdalbuildvrt", "-overwrite", str(vrt_path), "-input_file_list", list_file],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout).strip())
    finally:
        Path(list_file).unlink(missing_ok=True)
    return vrt_path
