"""What the published release consists of, and whether it is complete.

The paper's data sits in one directory, and the reason to describe that
directory in code rather than in prose is that prose drifts: the release
README claimed the model weights were elsewhere for two days after they
were copied in, and nothing noticed.

:data:`RELEASE` is the inventory — one entry per file or directory, each
saying what it is, whether the release is incomplete without it, and which
stage of the pipeline produces it. :func:`inspect_release` walks it and
reports what is present, what is missing and what is there but unlisted,
which is how a stray checkpoint or a leftover ``.ipynb_checkpoints``
surfaces.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "Item",
    "RELEASE",
    "Status",
    "inspect_release",
    "manifest",
    "checksum",
    "directory_size",
]


@dataclass(frozen=True)
class Item:
    """One entry of the release.

    Args:
        path: Location relative to the release root. A trailing ``/``
            marks a directory.
        what: One line for the README and the manifest.
        stage: The pipeline step that writes it, for the provenance
            column. ``None`` for hand-made or third-party inputs.
        required: Whether the release is incomplete without it.
        pattern: For directories, the glob its contents should match, so
            an empty or wrongly filled directory is reported rather than
            counted as present.
    """

    path: str
    what: str
    stage: str | None = None
    required: bool = True
    pattern: str = "*"

    @property
    def is_directory(self) -> bool:
        return self.path.endswith("/")


#: The release, in the order the README lists it.
RELEASE = (
    Item("README.md", "What is here, where each file came from, how to cite it",
         stage=None),
    Item("MANIFEST.csv", "This inventory, with sizes and checksums",
         stage="collect_release"),

    # ── The map ──
    Item("mosaic_utm/", "The 20 cm tree cover mask, 10 km GeoTIFFs in native UTM",
         stage="11_export_mosaic", pattern="UTM3*/*.tif"),
    Item("mosaic_utm/coverage_utm32.geojson",
         "Acquisition date per area, EPSG:25832", stage="12_coverage_polygons"),
    Item("mosaic_utm/coverage_utm33.geojson",
         "Acquisition date per area, EPSG:25833", stage="12_coverage_polygons"),
    Item("mosaic_utm/README.md", "Raster specification and caveats", stage=None),

    # ── The model ──
    Item("weights/", "The published SegFormer b5 checkpoint", stage="04_train",
         pattern="*.pth"),
    Item("training_runs_ablations.zip",
         "Metrics, configs and histories of the ablation runs", stage="04_train"),

    # ── Training data ──
    Item("training/sampled_tiles_100.gpkg",
         "The 152 sampled training tiles with their strata", stage="01_sample_tiles"),
    Item("training/sampled_tiles_100.csv", "The same table without geometry",
         stage="01_sample_tiles"),
    Item("training/labels/", "152 hand-drawn label masks, 5000 x 5000 px at 20 cm",
         stage=None, pattern="*.tif"),
    Item("training/patches/", "observations.csv, patches_metadata.csv, split_info.json",
         stage="03_prepare_patches", pattern="*"),
    Item("training/logs/", "Which tile-date came from which URL, and when",
         stage="02_download_dataset", pattern="*.csv"),
    Item("training/imagery/", "DOP and nDSM of the three tiles figure 3 draws",
         stage="02_download_dataset", required=False, pattern="**/*.tif"),
    Item("training/README.md", "Column meanings and the split warning", stage=None),
    Item("fig03_data.zip",
         "training/imagery packaged for download — the same six rasters",
         stage="02_download_dataset", required=False),

    # ── Validation ──
    Item("validation/lidar_masks/", "460 LiDAR reference masks and their footprints",
         stage="07_validate", pattern="*/*.tif"),
    Item("validation/metrics_per_sample_NW.csv", "Per-box metrics, North Rhine-Westphalia",
         stage="07_validate"),
    Item("validation/metrics_per_sample_BB.csv", "Per-box metrics, Brandenburg",
         stage="07_validate"),
    Item("validation/metrics_per_sample_BY.csv", "Per-box metrics, Bavaria",
         stage="07_validate"),
    Item("validation/metrics_summary.csv", "The accuracy table of the paper",
         stage="07_validate"),

    # ── Per-tile tables and results ──
    Item("tiles/tile_treecover_products.csv",
         "One row per 1 km tile: our cover and every comparison product",
         stage="10_extract_tile_products"),
    Item("tiles/tile_treecover_products_v2.csv",
         "The same, completed and land-masked (centroids, bbox areas)",
         stage="prepare_tile_table", required=False),
    Item("tiles/tile_treecover_products_v3.csv",
         "v2 with measured land areas — the weights the aggregation uses",
         stage="prepare_tile_table"),
    Item("tiles/tile_acquisition_coverage.csv",
         "Acquisition date, month and season per 1 km cell", stage="fig01"),
    Item("statistics/tile_statistics_all.csv", "Tree cover area per prediction tile",
         stage="06_tile_statistics"),
    Item("statistics/state_summary.csv", "The same, aggregated per state",
         stage="06_tile_statistics"),
    Item("results/table1.csv", "Table 1: our cover against the three products",
         stage="09_compare_products"),
    Item("results/per_state.csv", "Cover per federal state and product",
         stage="09_compare_products"),
    Item("results/grid_1km.csv", "The common 1 km comparison grid",
         stage="09_compare_products"),

    # ── Masks and figures ──
    Item("masks/germany_land_in_border.gpkg",
         "OSM land polygons clipped to the border — keeps sea out of the areas",
         stage=None),
    Item("masks/gadm41_DEU.gpkg", "GADM 4.1 administrative boundaries", stage=None),
    Item("masks/germany_border.geojson", "The national border as a single polygon",
         stage=None),
    Item("masks/germany_land_osm.gpkg",
         "OSM land polygons before clipping — the source of the mask above",
         stage=None, required=False),

    # ── Third-party inputs the pipeline reads ──
    # Copied in so the release stands on its own; see the licence note in the
    # README before redistributing any of them.
    Item("auxiliary/corine/U2018_CLC2018_V2020_20u1.tiff",
         "CORINE Land Cover 2018, 20 m — the urban/non-urban split of the strata",
         stage=None),
    Item("auxiliary/tcd/",
         "Copernicus HRL Tree Cover Density 2023, 10 m — the density strata of the "
         "sample and the TCD comparison product",
         stage=None, pattern="*.tiff"),
    Item("auxiliary/tile_index/lgln-opengeodata-bdom20.geojson",
         "Lower Saxony's bDOM acquisition index — the population stage 1 draws from",
         stage=None),

    Item("figures/", "The paper's figures as rendered", stage="figures/", pattern="*.png"),
)

#: Figures the release should carry, by their file stem.
FIGURE_STEMS = (
    "fig01_acquisition_coverage",
    "fig02_training_validation_areas",
    "fig03_training_examples",
    "fig04_scatter_lidar_vs_model",
    "fig05_stratified_iou",
    "fig06_example_tiles",
    "fig07_scatter_products",
    "fig08_product_comparison",
    "fig09_relative_difference",
    "fig10_local_comparison",
)

#: Directories that are working clutter, never release content.
CLUTTER = (".ipynb_checkpoints", "__pycache__")


@dataclass
class Status:
    """What one inventory entry looks like on disk."""

    item: Item
    present: bool
    size_bytes: int = 0
    files: int = 0
    digest: str | None = None
    note: str = ""

    @property
    def missing_and_required(self) -> bool:
        return self.item.required and not self.present


def directory_size(path: Path, pattern: str = "*") -> tuple:
    """``(bytes, file count)`` over a glob, ignoring clutter."""
    total = count = 0
    for entry in path.glob(pattern):
        if entry.is_file() and not any(part in CLUTTER for part in entry.parts):
            total += entry.stat().st_size
            count += 1
    return total, count


def checksum(path: Path, limit_bytes: int = 512 * 1024 * 1024) -> str | None:
    """SHA-256 of a file, or ``None`` when it is larger than ``limit_bytes``.

    Hashing 11 GB of rasters to publish a manifest costs more than it is
    worth; the size and the file count already catch a truncated copy.
    """
    if path.stat().st_size > limit_bytes:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_release(root: Path, items=RELEASE, checksums: bool = True,
                    checksum_limit: int = 512 * 1024 * 1024) -> list:
    """Walk the inventory and describe what is actually there.

    Args:
        root: The release directory.
        items: Inventory to check.
        checksums: Whether to hash files within the limit.
        checksum_limit: Files above this size are sized but not hashed.

    Returns:
        One :class:`Status` per item, in inventory order.
    """
    root = Path(root)
    statuses = []
    for item in items:
        path = root / item.path.rstrip("/")
        if not path.exists():
            statuses.append(Status(item, present=False))
            continue

        if item.is_directory:
            size, count = directory_size(path, item.pattern)
            statuses.append(Status(
                item, present=count > 0, size_bytes=size, files=count,
                note="" if count else f"no file matches {item.pattern}",
            ))
        else:
            size = path.stat().st_size
            statuses.append(Status(
                item, present=size > 0, size_bytes=size, files=1,
                digest=checksum(path, checksum_limit) if checksums else None,
                note="" if size else "empty file",
            ))
    return statuses


def unlisted(root: Path, items=RELEASE) -> list:
    """Top-level entries of the release that the inventory does not mention.

    Not an error — it is how a new artefact announces itself — but it has
    to be either added to :data:`RELEASE` or removed before publishing,
    because the README is generated from the inventory and would not
    mention it.
    """
    listed = {Path(item.path.rstrip("/")).parts[0] for item in items}
    found = []
    for entry in sorted(Path(root).iterdir()):
        if entry.name in CLUTTER:
            found.append((entry.name, "working clutter, delete before publishing"))
        elif entry.name not in listed:
            found.append((entry.name, "not in the inventory"))
    return found


def manifest(statuses: list) -> list:
    """The inventory as rows for ``MANIFEST.csv``."""
    return [
        {
            "path": status.item.path,
            "what": status.item.what,
            "produced_by": status.item.stage or "",
            "required": status.item.required,
            "present": status.present,
            "files": status.files,
            "size_bytes": status.size_bytes,
            "sha256": status.digest or "",
            "note": status.note,
        }
        for status in statuses
    ]


def human_size(size_bytes: float) -> str:
    """``1.2 GB`` — for the report and the README table."""
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if size_bytes < 1024 or unit == "TB":
            return f"{size_bytes:.0f} {unit}" if unit == "B" else f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
