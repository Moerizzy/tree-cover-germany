"""Tests for locating imagery and reading tile names.

Figure 10 pins its scenes by tile name rather than by coordinate, so the
name parsers are load-bearing: a state read wrong picks a prediction from
the wrong archive branch, and a date read wrong mislabels a panel with a
season that contradicts what the orthophoto shows.
"""

from __future__ import annotations

import pytest

from treecover.imagery import (
    IMAGE_DIRS,
    date_from_stem,
    find_prediction_by_stem,
    ortho_for_prediction,
    state_from_stem,
)


# ── reading a tile name ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "stem, state",
    [
        ("dop20rgbi_33_411_5654_sn_file_20240319", "SN"),
        ("dop20rgbi_32_573_5458_bw_file_20240730", "BW"),
        ("dop20rgb_32_682_5470_by_file_20230910", "BY"),
    ],
)
def test_state_is_read_from_the_name(stem, state):
    assert state_from_stem(stem) == state


def test_state_survives_the_pred_suffix():
    assert state_from_stem("dop20rgbi_32_518_6027_sh_file_20240514_pred") == "SH"


def test_state_is_none_when_the_name_does_not_carry_one():
    """Better than a wrong guess: the caller then has to say which state."""
    assert state_from_stem("some_other_raster_20240101") is None


@pytest.mark.parametrize(
    "stem, date",
    [
        ("dop20rgbi_33_411_5654_sn_file_20240319", "2024-03-19"),
        ("dop20rgbi_32_573_5458_bw_file_20240730", "2024-07-30"),
    ],
)
def test_date_is_read_from_the_name(stem, date):
    assert date_from_stem(stem) == date


def test_date_is_none_when_the_name_has_no_eight_digit_run():
    assert date_from_stem("dop20rgbi_32_573_5458_bw") is None


# ── locating a prediction by name ────────────────────────────────────────────


def _make_tile(root, state, year, cell, stem):
    directory = root / state / year / "predictions" / cell
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}_pred.tif"
    path.write_bytes(b"")
    return path


def test_prediction_is_found_by_name(tmp_path):
    stem = "dop20rgbi_32_573_5458_bw_file_20240730"
    expected = _make_tile(tmp_path, "BW", "2024", "UTM32_E5700_N54500", stem)
    assert find_prediction_by_stem(tmp_path, stem) == expected


def test_the_pred_suffix_is_accepted(tmp_path):
    """The name printed by a run already carries it; requiring it stripped
    would make the run's own output unusable as input."""
    stem = "dop20rgbi_32_573_5458_bw_file_20240730"
    expected = _make_tile(tmp_path, "BW", "2024", "UTM32_E5700_N54500", stem)
    assert find_prediction_by_stem(tmp_path, f"{stem}_pred") == expected


def test_a_missing_tile_yields_none_rather_than_raising(tmp_path):
    assert find_prediction_by_stem(tmp_path, "dop20rgbi_32_1_1_bw_file_20240101") is None


def test_the_newest_year_wins(tmp_path):
    """Same rule as the merge: where two acquisitions carry a name, the
    later one is the one the published map shows."""
    stem = "dop20rgbi_32_573_5458_bw_file_20240730"
    _make_tile(tmp_path, "BW", "2023", "UTM32_E5700_N54500", stem)
    newer = _make_tile(tmp_path, "BW", "2024", "UTM32_E5700_N54500", stem)
    assert find_prediction_by_stem(tmp_path, stem) == newer


def test_an_explicit_state_overrides_the_name(tmp_path):
    stem = "dop20rgbi_32_573_5458_bw_file_20240730"
    expected = _make_tile(tmp_path, "XX", "2024", "UTM32_E5700_N54500", stem)
    assert find_prediction_by_stem(tmp_path, stem, state="XX") == expected


# ── the orthophoto beside it ─────────────────────────────────────────────────


@pytest.mark.parametrize("folder", IMAGE_DIRS)
def test_ortho_is_found_in_either_band_folder(tmp_path, folder):
    """Five states publish three bands under RGB and eleven publish four
    under RGBI; a figure that searched only one would lose two thirds of
    the country."""
    stem = "dop20rgbi_32_573_5458_bw_file_20240730"
    pred = _make_tile(tmp_path, "BW", "2024", "UTM32_E5700_N54500", stem)
    image_dir = tmp_path / "BW" / "2024" / folder / "UTM32_E5700_N54500"
    image_dir.mkdir(parents=True)
    image = image_dir / f"{stem}.jp2"
    image.write_bytes(b"")
    assert ortho_for_prediction(pred) == image


def test_a_deleted_ortho_yields_none(tmp_path):
    """Most states' imagery was removed after inference. That is the normal
    state of the archive, not an error."""
    stem = "dop20rgbi_32_573_5458_bw_file_20240730"
    pred = _make_tile(tmp_path, "BW", "2024", "UTM32_E5700_N54500", stem)
    assert ortho_for_prediction(pred) is None
