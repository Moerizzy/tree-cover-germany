"""Land and state-border masking.

Before a prediction tile contributes to any area figure it is clipped to
the intersection of

1. the OSM land polygons for Germany, and
2. the GADM state border for the tile's state.

Without this the national and per-state totals are wrong in two ways that
both inflate them: tiles straddling a state border are counted twice, once
for each state, and coastal tiles count open water as land that happens to
have no trees on it. The paper's headline figures — 32.33 % of the mapped
land, 115,202 km² of tree cover — depend on this masking being applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds as transform_from_bounds
from shapely.geometry import box as shapely_box
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union
from shapely.strtree import STRtree

from treecover.config import load_states
from treecover.io.vector import read_vector

Coverage = Literal["outside", "fully_inside", "partial", "no_crs"]

__all__ = ["StateLandMask", "Coverage"]


class StateLandMask:
    """Combined land + state-border mask, queried per raster window.

    Loading the source layers is expensive, so one instance is built per
    worker process and reused for every tile that worker handles. Per-state
    intersections are cached lazily on first use.

    Args:
        land_gpkg: OSM land polygons clipped to the German border.
        gadm_gpkg: GADM 4.1 Germany. Level-1 layer ``ADM_ADM_1`` is read.
    """

    def __init__(self, land_gpkg: Path, gadm_gpkg: Path):
        land = read_vector(land_gpkg)
        if land.crs is None or land.crs.to_epsg() != 4326:
            land = land.to_crs(4326)
        self._land_union = unary_union(list(land.geometry.values))

        gadm = read_vector(gadm_gpkg, layer="ADM_ADM_1")
        if gadm.crs is None or gadm.crs.to_epsg() != 4326:
            gadm = gadm.to_crs(4326)
        # HASC_1 looks like 'DE.BY'; we key on the suffix.
        self._geoms_by_hasc = {
            row.HASC_1.split(".")[-1]: row.geometry for row in gadm.itertuples(index=False)
        }

        self._states = load_states()
        self._state_cache: dict[str, tuple[list, STRtree | None]] = {}
        self._transformers: dict[int, Transformer] = {}

    def _transformer_from_4326(self, dst_epsg: int) -> Transformer:
        if dst_epsg not in self._transformers:
            self._transformers[dst_epsg] = Transformer.from_crs(
                4326, dst_epsg, always_xy=True
            )
        return self._transformers[dst_epsg]

    def _state_index(self, state_code: str) -> tuple[list, STRtree | None]:
        """Return (polygons, STRtree) for the land part of ``state_code``."""
        code = state_code.upper()
        if code in self._state_cache:
            return self._state_cache[code]

        hasc = self._states[code].gadm_hasc
        if hasc not in self._geoms_by_hasc:
            raise KeyError(
                f"State {code!r} maps to HASC {hasc!r}, which is not present in the "
                "GADM layer. Check configs/states.yaml."
            )

        masked = self._geoms_by_hasc[hasc].intersection(self._land_union)
        if masked.is_empty:
            geoms: list = []
        elif masked.geom_type == "Polygon":
            geoms = [masked]
        else:
            geoms = [g for g in getattr(masked, "geoms", [masked]) if not g.is_empty]

        self._state_cache[code] = (geoms, STRtree(geoms) if geoms else None)
        return self._state_cache[code]

    def _coverage(self, state_code: str, lonlat_bbox: tuple[float, float, float, float]):
        """Classify how a lon/lat bbox sits relative to the state's land area."""
        geoms, tree = self._state_index(state_code)
        if not geoms or tree is None:
            return "outside", None

        bbox_geom = shapely_box(*lonlat_bbox)
        hits = tree.query(bbox_geom)
        if len(hits) == 0:
            return "outside", None

        local = [g for g in (geoms[i].intersection(bbox_geom) for i in hits) if not g.is_empty]
        if not local:
            return "outside", None

        union = unary_union(local)
        if union.contains(bbox_geom):
            return "fully_inside", union
        return "partial", union

    def mask_for_bounds(
        self,
        src_epsg: int | None,
        bounds: tuple[float, float, float, float],
        height: int,
        width: int,
        state_code: str,
    ) -> tuple[np.ndarray | bool | None, Coverage]:
        """Build a validity mask for an arbitrary raster window.

        Returns:
            ``(mask, status)`` where the mask is ``None`` when the window
            lies entirely outside (skip the tile), ``True`` when it lies
            entirely inside (no masking needed — the common case, and the
            reason this is not always an array), or a boolean array with
            ``True`` marking valid pixels.
        """
        if src_epsg is None:
            return True, "no_crs"

        to_wgs84 = Transformer.from_crs(src_epsg, 4326, always_xy=True)
        left, bottom, right, top = bounds
        lons, lats = to_wgs84.transform([left, right, left, right], [bottom, bottom, top, top])
        lonlat_bbox = (min(lons), min(lats), max(lons), max(lats))

        status, union = self._coverage(state_code, lonlat_bbox)
        if status == "outside":
            return None, status
        if status == "fully_inside":
            return True, status

        to_src = self._transformer_from_4326(src_epsg)
        union_local = shp_transform(lambda x, y, z=None: to_src.transform(x, y), union)
        transform = transform_from_bounds(left, bottom, right, top, width, height)
        mask = geometry_mask(
            [union_local],
            out_shape=(height, width),
            transform=transform,
            invert=True,
            all_touched=False,
        )
        return mask, status

    def mask_for_dataset(self, src, state_code: str):
        """Convenience wrapper around :meth:`mask_for_bounds` for a rasterio dataset."""
        src_epsg = src.crs.to_epsg() if src.crs else None
        return self.mask_for_bounds(
            src_epsg, tuple(src.bounds), src.height, src.width, state_code
        )
