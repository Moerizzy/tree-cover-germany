"""The mosaic must be a copy, not a reprojection.

Every test here defends one of two invariants: a source pixel lands on the
output pixel it belongs to, and an area derived from the result is a real
area. Both are what makes the mosaic usable for the paper's km² figures.
"""

from pathlib import Path

import pytest

from treecover.mosaic import (
    CELL_PX,
    CELL_SIZE_M,
    PIXEL_M,
    Placement,
    cell_bounds,
    cell_id_for,
    cell_id_from_tile_key,
    cell_transform,
    geodesic_area_factor,
    grid_offset,
    is_copyable,
    parse_cell_id,
    placement_for,
    plan_cell,
    sanity_check_grid,
    stripe_rows,
    total_stripes,
)

CELL = "UTM33_E4400_N58600"


def test_cell_id_matches_the_archive_directory_naming():
    """The 1 km tile 33_445_5861 lives in UTM33_E4400_N58600 on disk."""
    assert cell_id_for(33, 445, 5861) == CELL
    assert cell_id_from_tile_key(("33", "445", "5861")) == CELL


def test_cell_id_floors_rather_than_rounds():
    """449 and 440 share a cell; 450 starts the next one."""
    assert cell_id_for(32, 440, 5300) == "UTM32_E4400_N53000"
    assert cell_id_for(32, 449, 5309) == "UTM32_E4400_N53000"
    assert cell_id_for(32, 450, 5310) == "UTM32_E4500_N53100"


def test_parse_cell_id_gives_the_utm_crs_and_corner():
    epsg, left, bottom = parse_cell_id(CELL)
    assert epsg == 25833
    assert (left, bottom) == (440_000.0, 5_860_000.0)
    assert cell_bounds(CELL) == (440_000.0, 5_860_000.0, 450_000.0, 5_870_000.0)


def test_parse_cell_id_rejects_anything_else():
    with pytest.raises(ValueError):
        parse_cell_id("UTM31_E4400_N58600")
    with pytest.raises(ValueError):
        parse_cell_id("tile_1.tif")


def test_transform_is_north_up_at_20_cm():
    transform = cell_transform(CELL)
    assert transform.a == pytest.approx(PIXEL_M)
    assert transform.e == pytest.approx(-PIXEL_M)
    # Origin is the north-west corner, not the south-west one.
    assert transform.c == pytest.approx(440_000.0)
    assert transform.f == pytest.approx(5_870_000.0)


def test_a_tile_lands_on_the_pixel_its_coordinates_name():
    """445000/5862000 is 5 km east and 8 km south of the cell's NW corner."""
    placement = placement_for(CELL, Path("t.tif"), 445_000.0, 5_862_000.0, 5000, 5000)
    assert placement is not None
    assert placement.col_off == 25_000
    assert placement.row_off == 40_000
    assert (placement.height, placement.width) == (5000, 5000)
    assert (placement.read_row_off, placement.read_col_off) == (0, 0)


def test_the_north_west_tile_starts_at_the_origin():
    placement = placement_for(CELL, Path("t.tif"), 440_000.0, 5_870_000.0, 5000, 5000)
    assert (placement.row_off, placement.col_off) == (0, 0)


def test_an_oversized_tile_is_clipped_not_rejected():
    """One tile in the archive is 5001 px and would write past the edge."""
    placement = placement_for(CELL, Path("t.tif"), 449_000.0, 5_861_000.0, 5001, 5001)
    assert placement is not None
    assert placement.col_off == 45_000
    assert placement.col_off + placement.width == CELL_PX
    assert placement.row_off + placement.height == CELL_PX


def test_a_placement_never_crosses_a_stripe_boundary():
    """The writer materialises one stripe at a time; a taller placement
    has nowhere to go, and broadcasting it aborted 24 cells before this
    was enforced here."""
    placement = placement_for(CELL, Path("t.tif"), 445_000.0, 5_866_000.0, 5001, 5001)
    assert placement is not None
    start, count = stripe_rows(placement.stripe)
    assert placement.row_off >= start
    assert placement.row_off + placement.height <= start + count
    assert placement.trimmed_rows == 1


def test_a_normal_tile_is_not_trimmed():
    placement = placement_for(CELL, Path("t.tif"), 445_000.0, 5_866_000.0, 5000, 5000)
    assert placement.trimmed_rows == 0
    assert placement.height == 5000


def test_every_placement_of_a_full_cell_stays_inside_its_stripe():
    """The invariant, checked over a whole cell rather than one tile."""
    for e in range(440, 450):
        for n in range(5860, 5870):
            p = placement_for(CELL, Path("t.tif"), e * 1000.0, (n + 1) * 1000.0, 5000, 5000)
            assert p is not None
            start, count = stripe_rows(p.stripe)
            assert start <= p.row_off
            assert p.row_off + p.height <= start + count


def test_a_tile_overhanging_the_north_west_is_read_from_an_offset():
    """Clipping happens on the read side, so the write stays inside the cell."""
    placement = placement_for(CELL, Path("t.tif"), 439_800.0, 5_870_200.0, 5000, 5000)
    assert (placement.row_off, placement.col_off) == (0, 0)
    assert (placement.read_row_off, placement.read_col_off) == (1000, 1000)
    assert (placement.height, placement.width) == (4000, 4000)


def test_a_tile_outside_the_cell_is_refused():
    """Silently writing it somewhere would corrupt a neighbouring cell."""
    assert placement_for(CELL, Path("t.tif"), 470_000.0, 5_862_000.0, 5000, 5000) is None


def test_off_grid_tiles_are_detected():
    """Saarland's WMS tiles sit half a pixel out; that must be visible."""
    assert sanity_check_grid(CELL, 445_000.0, 5_862_000.0)
    assert not sanity_check_grid(CELL, 445_000.1, 5_862_000.0)
    assert not sanity_check_grid(CELL, 445_000.0, 5_862_000.1)


def test_grid_offset_reports_the_sub_pixel_shift():
    """0.1 m at a 0.2 m pixel is half a pixel, in y as in x."""
    assert grid_offset(CELL, 445_000.0, 5_862_000.0) == pytest.approx((0.0, 0.0))
    dx, dy = grid_offset(CELL, 445_000.1, 5_862_000.1)
    assert abs(dx) == pytest.approx(0.5)
    assert abs(dy) == pytest.approx(0.5)


def test_an_off_grid_tile_is_still_placed():
    """It is snapped, not dropped — a hole would be the larger error.

    The paper's per-tile statistics count these tiles, so a mosaic that
    omitted them could not reproduce its own area figures.
    """
    placement = placement_for(CELL, Path("sl.tif"), 445_000.1, 5_862_000.1, 5000, 5000)
    assert placement is not None
    assert placement.col_off == 25_000  # snapped to the nearest pixel
    assert placement.row_off == 40_000


def test_a_different_resolution_is_not_copyable():
    """10 cm pixels would need resampling, which this path refuses to do."""
    assert is_copyable((0.2, 0.2), (0.0, 0.0))
    assert not is_copyable((0.1, 0.1), (0.0, 0.0))
    assert not is_copyable((0.2, 0.4), (0.0, 0.0))


def test_a_rotated_source_is_not_copyable():
    """Any skew means the source does not share the output grid."""
    assert not is_copyable((0.2, 0.2), (0.001, 0.0))
    assert not is_copyable((0.2, 0.2), (0.0, 0.001))


def test_stripes_tile_the_cell_exactly():
    """No gap and no overlap, or the mosaic would drop or double rows."""
    covered = 0
    for stripe in range(total_stripes()):
        first, count = stripe_rows(stripe)
        assert first == covered
        covered += count
    assert covered == CELL_PX


def test_placements_are_grouped_by_the_stripe_they_start_in():
    tiles = [
        Placement(Path("a.tif"), row_off=0, col_off=0, height=5000, width=5000),
        Placement(Path("b.tif"), row_off=0, col_off=5000, height=5000, width=5000),
        Placement(Path("c.tif"), row_off=5000, col_off=0, height=5000, width=5000),
    ]
    planned = plan_cell(tiles)
    assert list(planned) == [0, 1]
    assert len(planned[0]) == 2
    assert len(planned[1]) == 1


def test_geodesic_factor_is_close_to_one_but_not_one():
    """UTM is conformal: a projected km² is not a km² on the ground.

    Across Germany the correction stays inside ±0.2 %, which is why the
    5 % discrepancy in the paper's Table 1 could never have come from here.
    """
    factor = geodesic_area_factor(CELL)
    assert 0.998 < factor < 1.002
    assert factor != 1.0


def test_the_factor_grows_away_from_the_central_meridian():
    """k is smallest on the central meridian, so 1/k² is largest there."""
    central = geodesic_area_factor("UTM33_E5000_N58600")
    edge = geodesic_area_factor("UTM33_E3000_N58600")
    assert central > edge


def test_a_cell_is_a_hundred_square_kilometres_of_projected_area():
    assert (CELL_PX * PIXEL_M) ** 2 == pytest.approx(CELL_SIZE_M**2)
    assert CELL_PX * PIXEL_M == 10_000
