"""Tests for configuration loading and the state registry."""

from __future__ import annotations

import pytest

from treecover.config import (
    CONFIG_DIR,
    load_paths,
    load_states,
    strip_zone_prefix,
    validation_states,
    zone_dash,
)


def test_all_sixteen_states_are_defined():
    states = load_states()
    assert len(states) == 16


def test_brandenburg_hasc_differs_from_its_code():
    """The one state where the GADM key is not the folder code — BB vs BR."""
    assert load_states()["BB"].gadm_hasc == "BR"


def test_nrw_alias_resolves_to_the_official_key():
    """The project used NRW and NW interchangeably; both must work."""
    states = load_states()
    assert states.resolve("NRW") == "NW"
    assert states["NRW"] is states["NW"]
    assert states["NW"].pred_dir == "NW"


def test_unknown_state_lists_the_valid_codes():
    with pytest.raises(KeyError, match="Known:"):
        load_states()["XX"]


def test_eastern_states_use_utm_33():
    states = load_states()
    for code in ("BB", "MV", "SN", "ST"):
        assert states[code].epsg == 25833, code
        assert states[code].utm_zone == 33, code
    assert states["BY"].epsg == 25832


def test_bavaria_lidar_url_strips_the_zone_prefix():
    url = load_states()["BY"].lidar_url("32706_5585")
    assert url == "https://geodaten.bayern.de/odd_data/laser/706_5585.laz"


def test_brandenburg_lidar_url_keeps_the_zone_and_swaps_the_separator():
    url = load_states()["BB"].lidar_url("33304_5862")
    assert url.endswith("als_33304-5862.zip")


def test_states_without_a_lidar_feed_report_it():
    states = load_states()
    assert states["BY"].has_lidar
    assert not states["SH"].has_lidar
    assert states["SH"].lidar_url("32500_5900") is None


def test_validation_states_match_the_paper():
    assert set(validation_states()) == {"NW", "BB", "BY"}


@pytest.mark.parametrize(
    ("tile", "expected"),
    [("32706_5585", "706_5585"), ("706_5585", "706_5585")],
)
def test_strip_zone_prefix_is_idempotent(tile, expected):
    assert strip_zone_prefix(tile) == expected


def test_zone_dash():
    assert zone_dash("33304_5862") == "33304-5862"


def test_paths_interpolation_expands_references():
    # The example file, not whatever paths.yaml this machine happens to
    # have: a host that only fills in the keys it needs would fail a test
    # about interpolation for no reason.
    paths = load_paths(CONFIG_DIR / "paths.example.yaml")
    clc = str(paths.get_path("sampling.clc"))
    assert "${" not in clc, "unexpanded ${...} left in the resolved path"


def test_env_override_wins_over_the_file(monkeypatch):
    monkeypatch.setenv("TREECOVER_PREDICTIONS_ROOT", "/tmp/somewhere-else")
    assert str(load_paths().get_path("predictions_root")) == "/tmp/somewhere-else"


def test_missing_key_names_the_fix():
    with pytest.raises(KeyError, match="paths.yaml"):
        load_paths().get_path("definitely.not.a.key")
