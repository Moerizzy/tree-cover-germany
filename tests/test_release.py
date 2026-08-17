"""The release inventory, and the coverage polygons that ship with the map."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from treecover.release import (
    CLUTTER,
    FIGURE_STEMS,
    RELEASE,
    Item,
    checksum,
    directory_size,
    human_size,
    inspect_release,
    manifest,
    unlisted,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    """Import a numbered CLI, whose name is not a valid module name."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def release(tmp_path):
    """A release holding one file and one directory of the inventory."""
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    labels = tmp_path / "training" / "labels"
    labels.mkdir(parents=True)
    (labels / "324165830_SegFormerb5_RGBI_nDSM.tif").write_bytes(b"x" * 100)
    return tmp_path


ITEMS = (
    Item("README.md", "the guide"),
    Item("training/labels/", "label masks", pattern="*.tif"),
    Item("results/table1.csv", "Table 1"),
    Item("figures/", "figures", required=False, pattern="*.png"),
)


# ── The inventory ────────────────────────────────────────────────────────────


def test_present_and_missing_are_both_reported(release):
    statuses = inspect_release(release, ITEMS)

    by_path = {s.item.path: s for s in statuses}
    assert by_path["README.md"].present
    assert by_path["training/labels/"].present
    assert not by_path["results/table1.csv"].present
    assert by_path["results/table1.csv"].missing_and_required


def test_an_optional_entry_is_not_a_failure(release):
    statuses = inspect_release(release, ITEMS)

    figures = next(s for s in statuses if s.item.path == "figures/")
    assert not figures.present
    assert not figures.missing_and_required


def test_a_directory_with_no_matching_file_counts_as_missing(release):
    # An empty labels/ would otherwise pass as present and the release
    # would ship without the one thing that cannot be re-derived.
    for path in (release / "training" / "labels").glob("*"):
        path.unlink()

    status = next(s for s in inspect_release(release, ITEMS)
                  if s.item.path == "training/labels/")

    assert not status.present
    assert "*.tif" in status.note


def test_an_empty_file_counts_as_missing(release):
    (release / "README.md").write_text("", encoding="utf-8")

    status = next(s for s in inspect_release(release, ITEMS)
                  if s.item.path == "README.md")

    assert not status.present
    assert status.note == "empty file"


def test_clutter_does_not_count_towards_a_directory(release):
    checkpoints = release / "training" / "labels" / ".ipynb_checkpoints"
    checkpoints.mkdir()
    (checkpoints / "stale.tif").write_bytes(b"x" * 5000)

    size, count = directory_size(release / "training" / "labels", "**/*.tif")

    assert count == 1
    assert size == 100


def test_unlisted_entries_are_surfaced(release):
    (release / "leftover.zip").write_bytes(b"x")
    (release / ".ipynb_checkpoints").mkdir()

    found = dict(unlisted(release, ITEMS))

    assert "leftover.zip" in found
    assert "clutter" in found[".ipynb_checkpoints"]


def test_manifest_has_a_row_per_entry(release):
    rows = manifest(inspect_release(release, ITEMS))

    frame = pd.DataFrame(rows)
    assert len(frame) == len(ITEMS)
    assert set(frame.columns) >= {"path", "what", "produced_by", "present", "sha256"}


def test_large_files_are_sized_but_not_hashed(release):
    digest = checksum(release / "README.md", limit_bytes=1)

    assert digest is None
    assert checksum(release / "README.md", limit_bytes=1024).startswith("2cf24d")


def test_every_inventory_entry_names_what_it_is():
    for item in RELEASE:
        assert item.what, f"{item.path} has no description"
        assert not item.path.startswith("/"), f"{item.path} is not relative"


def test_the_figure_list_matches_the_figure_scripts():
    scripts = {path.stem for path in (SCRIPTS.parent / "figures").glob("fig*.py")}

    # Figure 2 is a QGIS composition and has no script; every other stem
    # in the release must be something the repo can re-render.
    assert set(FIGURE_STEMS) - scripts == {"fig02_training_validation_areas"}


@pytest.mark.parametrize("size, expected", [(512, "512 B"), (2048, "2.0 kB"),
                                            (5 * 1024 ** 3, "5.0 GB")])
def test_human_size(size, expected):
    assert human_size(size) == expected


def test_clutter_names_are_the_ones_git_ignores():
    assert ".ipynb_checkpoints" in CLUTTER and "__pycache__" in CLUTTER


# ── Coverage polygons ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def coverage():
    return load_script("12_coverage_polygons")


def tile_frame(rows):
    return pd.DataFrame(rows, columns=["date", "zone", "tile_x_m", "tile_y_m"])


def test_two_dates_give_two_disjoint_polygons(coverage):
    frame = tile_frame([
        (20230625, 32, 400000, 5830000),
        (20250420, 32, 401000, 5830000),
    ])

    features = coverage.coverage_features(frame, 32)

    assert [date for date, _ in features] == [20230625, 20250420]
    assert sum(geometry.area for _, geometry in features) == 2e6


def test_the_newer_acquisition_claims_the_overlap(coverage):
    # Same ground flown twice. The merge keeps the newer tile, so the
    # coverage map must say the newer date and count the area once.
    frame = tile_frame([
        (20230625, 32, 400000, 5830000),
        (20250420, 32, 400000, 5830000),
    ])

    features = coverage.coverage_features(frame, 32)

    assert [date for date, _ in features] == [20250420]
    assert features[0][1].area == 1e6


def test_undated_tiles_rank_last_but_are_kept(coverage):
    frame = tile_frame([
        (0, 32, 400000, 5830000),
        (20250420, 32, 400000, 5830000),
        (0, 32, 402000, 5830000),
    ])

    features = coverage.coverage_features(frame, 32)

    dates = [date for date, _ in features]
    assert dates == [0, 20250420]
    # The undated tile under the dated one is claimed by the date.
    assert dict(features)[0].area == 1e6


def test_the_other_zone_is_not_mixed_in(coverage):
    frame = tile_frame([
        (20250420, 32, 400000, 5830000),
        (20250420, 33, 400000, 5830000),
    ])

    assert len(coverage.coverage_features(frame, 33)) == 1
    assert coverage.coverage_features(frame, 33)[0][1].area == 1e6


def test_geojson_names_the_zone_crs(coverage):
    frame = tile_frame([(20250420, 33, 400000, 5830000)])

    collection = coverage.to_geojson(coverage.coverage_features(frame, 33), 33)

    assert collection["crs"]["properties"]["name"].endswith("25833")
    assert collection["features"][0]["properties"]["datetime"] == "2025-04-20T00:00:00"
    assert collection["features"][0]["properties"]["month"] == 4


def test_an_undated_polygon_carries_no_date(coverage):
    frame = tile_frame([(0, 32, 400000, 5830000)])

    collection = coverage.to_geojson(coverage.coverage_features(frame, 32), 32)

    assert collection["features"][0]["properties"] == {"month": None, "datetime": None}
