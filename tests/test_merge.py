"""Tests for overlap resolution when merging tiles.

Roughly 2.5 % of 1 km cells are covered twice, almost always at a state
border. Which tile wins used to depend on how ``gdalbuildvrt`` ordered its
sources, and three revisions of the merge script sorted the file list three
different ways. These tests pin the rule down.
"""

from __future__ import annotations

from pathlib import Path

from treecover.merge import (
    TileCandidate,
    cell_key,
    select_one_per_cell,
    tile_date,
)


def candidate(name: str, state: str = "BB", year: str = "2023") -> TileCandidate:
    path = Path(f"/data/{state}/{year}/predictions/UTM33_E4100_N59300/{name}")
    return TileCandidate(path=path, state=state, year=year, date=tile_date(path, year))


# ── cell identity ────────────────────────────────────────────────────────────


def test_cell_key_ignores_state_and_date():
    """The same ground flown by two states must map to one cell."""
    bb = Path("dop20rgbi_33_418_5931_bb_file_20230508_pred.tif")
    mv = Path("dop20rgbi_33_418_5931_mv_file_00000000_pred.tif")
    assert cell_key(bb) == cell_key(mv) == ("33", "418", "5931")


def test_neighbouring_cells_differ():
    a = Path("dop20rgbi_33_418_5931_bb_file_20230508_pred.tif")
    b = Path("dop20rgbi_33_418_5932_bb_file_20230508_pred.tif")
    assert cell_key(a) != cell_key(b)


def test_unrecognisable_name_has_no_cell():
    assert cell_key(Path("something_else.tif")) is None


# ── date extraction ──────────────────────────────────────────────────────────


def test_date_comes_from_the_filename():
    assert tile_date(Path("dop20rgbi_33_418_5931_bb_file_20230508_pred.tif")) == "20230508"


def test_date_falls_back_to_the_year_directory():
    assert tile_date(Path("x_pred.tif"), year_fallback="2024") == "20240000"


def test_undated_tile_sorts_before_every_real_date():
    """MV tiles in the archive carry neither a date nor a usable year."""
    undated = tile_date(Path("dop20rgbi_33_418_5931_mv_file_00000000_pred.tif"), "0000")
    assert undated == "00000000"
    assert undated < "19900101"


# ── selection ────────────────────────────────────────────────────────────────


def test_uncontested_cells_pass_through():
    choices = select_one_per_cell([candidate("dop20rgbi_33_418_5931_bb_file_20230508_pred.tif")])
    assert len(choices) == 1
    assert not choices[0].contested


def test_newest_acquisition_date_wins():
    old = candidate("dop20rgbi_33_418_5931_bb_file_20210508_pred.tif", "BB", "2021")
    new = candidate("dop20rgbi_33_418_5931_mv_file_20230612_pred.tif", "MV", "2023")
    choice = select_one_per_cell([old, new])[0]
    assert choice.winner is new
    assert choice.rejected == [old]


def test_order_of_input_does_not_matter():
    """The rule must be a property of the tiles, not of the scan order."""
    old = candidate("dop20rgbi_33_418_5931_bb_file_20210508_pred.tif", "BB", "2021")
    new = candidate("dop20rgbi_33_418_5931_mv_file_20230612_pred.tif", "MV", "2023")
    assert select_one_per_cell([old, new])[0].winner is new
    assert select_one_per_cell([new, old])[0].winner is new


def test_dated_tile_beats_undated_one():
    """The real archive case: BB/2023 against MV with no date at all."""
    dated = candidate("dop20rgbi_33_418_5931_bb_file_20230508_pred.tif", "BB", "2023")
    undated = candidate("dop20rgbi_33_418_5931_mv_file_00000000_pred.tif", "MV", "0000")
    choice = select_one_per_cell([dated, undated])[0]
    assert choice.winner is dated


def test_ties_resolve_deterministically():
    """Two tiles with the same date must always give the same winner, or the
    published map changes between runs over an unchanged archive."""
    a = candidate("dop20rgbi_33_418_5931_bb_file_20230508_pred.tif", "BB", "2023")
    b = candidate("dop20rgbi_33_418_5931_mv_file_20230508_pred.tif", "MV", "2023")
    first = select_one_per_cell([a, b])[0].winner
    second = select_one_per_cell([b, a])[0].winner
    assert first.path == second.path


def test_exactly_one_tile_survives_per_cell():
    tiles = [
        candidate("dop20rgbi_33_418_5931_bb_file_20230508_pred.tif", "BB", "2023"),
        candidate("dop20rgbi_33_418_5931_mv_file_00000000_pred.tif", "MV", "0000"),
        candidate("dop20rgbi_33_418_5932_bb_file_20230508_pred.tif", "BB", "2023"),
    ]
    choices = select_one_per_cell(tiles)
    assert len(choices) == 2
    assert sum(len(c.rejected) for c in choices) == 1


def test_date_beats_alphabetical_order():
    """A regression guard on the rule itself. Alphabetically /data/MV/…
    sorts after /data/BB/…, so path order would hand this cell to MV; the
    date rule gives it to the dated BB tile."""
    bb = candidate("dop20rgbi_33_418_5931_bb_file_20230508_pred.tif", "BB", "2023")
    mv = candidate("dop20rgbi_33_418_5931_mv_file_00000000_pred.tif", "MV", "0000")
    assert str(mv.path) > str(bb.path)
    assert select_one_per_cell([bb, mv])[0].winner is bb


def test_tiles_without_a_cell_id_are_kept():
    """Passing them through unmerged is safer than grouping them together."""
    odd = TileCandidate(Path("/data/x/odd_name.tif"), "x", "", "00000000")
    choices = select_one_per_cell([odd])
    assert len(choices) == 1
    assert choices[0].winner is odd
