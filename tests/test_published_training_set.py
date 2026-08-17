"""The settings that reproduce the published training set.

Stage 3 with no arguments must rebuild exactly what the published model was
trained on: 117 tiles, 245 observations, 19,845 patches. Two settings decide
that, and both have a plausible-looking wrong value that produces a valid
patch table nobody would question.

**Stride.** 512, i.e. non-overlapping, as the manuscript says. A stride of
256 is the natural default for a segmentation dataset and gives roughly four
times the patches — a bigger, differently distributed training set.

**Split.** From the published ``observations.csv``, not from the ``split``
column of ``sampled_tiles_100.gpkg``. That column is an earlier three-way
draw whose classes cut across both published ones, so using it moves
published *training* tiles into validation. Nothing downstream can detect
that; it just quietly inflates the validation metrics.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from treecover.data.patches import Patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_stage3():
    """Import ``03_prepare_patches.py``, whose name is not a valid module."""
    spec = importlib.util.spec_from_file_location(
        "stage3", SCRIPTS / "03_prepare_patches.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage3"] = module
    spec.loader.exec_module(module)
    return module


stage3 = load_stage3()


# ── the pinned settings ─────────────────────────────────────────────────────

def test_patches_are_non_overlapping_by_default():
    """Stride equals patch size — the manuscript's 'non-overlapping'."""
    assert stage3.PUBLISHED_STRIDE == stage3.PUBLISHED_PATCH_SIZE == 512

    parser = stage3.build_parser()
    defaults = parser.parse_args(["--tiles", "t", "--images", "i", "--masks", "m"])
    assert defaults.stride == 512, "a 256 stride quadruples the training set"
    assert defaults.patch_size == 512


def test_published_totals_match_the_manuscript():
    """The numbers the run checks itself against are the ones in the paper."""
    assert stage3.PUBLISHED_TOTALS == {
        "tiles": 117,
        "observations": 245,
        "patches": 19845,
        "train_patches": 15957,
        "val_patches": 3888,
    }
    totals = stage3.PUBLISHED_TOTALS
    assert totals["train_patches"] + totals["val_patches"] == totals["patches"]


# ── taking the split from the published observation table ───────────────────

@pytest.fixture
def tile_table():
    """A tile table carrying the gpkg's stale three-way split."""
    return pd.DataFrame({
        "tile_id": ["100", "200", "300", "400"],
        "split": ["train", "val", "test", "train"],
        "mean_tcd": [10.0, 20.0, 30.0, 40.0],
    })


@pytest.fixture
def observations_csv(tmp_path):
    """The published assignment: cuts across the table's own classes."""
    path = tmp_path / "observations.csv"
    pd.DataFrame({
        "tile_id": ["100", "100", "200", "300"],
        "split": ["val", "val", "train", "train"],
        "date": ["2023-07-01", "2023-01-10", "2023-07-02", "2023-07-03"],
    }).to_csv(path, index=False)
    return path


def test_published_split_overrides_the_tile_table(tile_table, observations_csv):
    result = stage3.apply_published_splits(tile_table, observations_csv,
                                           "tile_id", "split")
    assignment = dict(zip(result["tile_id"].astype(str), result["split"]))
    assert assignment == {"100": "val", "200": "train", "300": "train"}, (
        "tile 100 is 'train' in the table and 'val' in the published run; "
        "the published run wins"
    )


def test_tiles_absent_from_the_published_run_are_dropped(tile_table, observations_csv):
    """Tile 400 is in the table but was not used, so it must not reappear."""
    result = stage3.apply_published_splits(tile_table, observations_csv,
                                           "tile_id", "split")
    assert "400" not in set(result["tile_id"].astype(str))
    assert len(result) == 3


def test_one_row_per_tile_even_though_a_tile_has_several_observations(
    tile_table, observations_csv
):
    """observations.csv lists tile 100 twice; the tile table stays one row."""
    result = stage3.apply_published_splits(tile_table, observations_csv,
                                           "tile_id", "split")
    assert result["tile_id"].is_unique


def test_integer_tile_ids_still_match(tmp_path):
    """CSV readers turn '100' into 100; the join must survive that."""
    tiles = pd.DataFrame({"tile_id": ["100", "200"], "split": ["train", "train"]})
    path = tmp_path / "obs.csv"
    pd.DataFrame({"tile_id": [100, 200], "split": ["val", "train"]}).to_csv(
        path, index=False
    )
    result = stage3.apply_published_splits(tiles, path, "tile_id", "split")
    assert len(result) == 2
    assert dict(zip(result["tile_id"], result["split"])) == {"100": "val",
                                                             "200": "train"}


def test_a_reference_without_a_split_column_fails_loudly(tile_table, tmp_path):
    path = tmp_path / "no_split.csv"
    pd.DataFrame({"tile_id": ["100"]}).to_csv(path, index=False)
    with pytest.raises(SystemExit, match="split"):
        stage3.apply_published_splits(tile_table, path, "tile_id", "split")


def test_no_overlapping_tile_ids_fails_loudly(tile_table, tmp_path):
    """Silently producing an empty training set would be the worst outcome."""
    path = tmp_path / "other.csv"
    pd.DataFrame({"tile_id": ["999"], "split": ["train"]}).to_csv(path, index=False)
    with pytest.raises(SystemExit, match="matches the tile table"):
        stage3.apply_published_splits(tile_table, path, "tile_id", "split")


# ── the self-check ──────────────────────────────────────────────────────────

def patch_frame(n_train: int, n_val: int, tiles: int, observations: int):
    rows = []
    for i in range(n_train + n_val):
        rows.append(Patch(
            region_id=f"r{i % observations}", patch_id=f"p{i}",
            tile_id=f"t{i % tiles}", date="2023-07-01",
            split="train" if i < n_train else "val",
            col_off=0, row_off=0, width=512, height=512, tree_cover_pct=10.0,
        ).as_row())
    return pd.DataFrame(rows)


def test_self_check_confirms_an_exact_reproduction(capsys):
    totals = stage3.PUBLISHED_TOTALS
    frame = patch_frame(totals["train_patches"], totals["val_patches"],
                        totals["tiles"], totals["observations"])
    stage3._compare_to_published(frame)
    assert "Matches the published training set exactly" in capsys.readouterr().out


def test_self_check_reports_a_wrong_stride_rather_than_staying_silent(capsys):
    """Four times the patches is what a 256 stride looks like."""
    totals = stage3.PUBLISHED_TOTALS
    frame = patch_frame(totals["train_patches"] * 4, totals["val_patches"] * 4,
                        totals["tiles"], totals["observations"])
    stage3._compare_to_published(frame)
    out = capsys.readouterr().out
    assert "Differs from the published training set" in out
    assert "patches" in out and "19,845" in out
