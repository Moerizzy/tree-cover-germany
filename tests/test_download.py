"""Stage 2: resolving download URLs and turning the rasters into an nDSM.

Nothing here touches the network. The API client is exercised against
recorded payloads, and the raster maths against synthetic GeoTIFFs — the
invariants that matter are that a 2 km orthophoto still finds its 1 km
tile, that one ground model lands once per acquisition, and that a canopy
10 m above a 100 m hill measures 10 m and not 110.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from treecover.data.download import (
    EasyGeoData,
    Link,
    crop_to_geometry,
    download,
    links_from_index,
    match_acquisitions,
    normalised_surface,
    pair_ground_models,
    tile_id_to_key,
    tile_key_to_id,
)

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")

from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import box  # noqa: E402

DOP_PAYLOAD = {
    "count": 1, "mit_datum": 1,
    "links": [{
        "state": "NI", "tile_key": "32_416_5830", "gsd": 0.2, "bands": 4,
        "aktualitaet": "2025-04-20", "datum_typ": "befliegung",
        "url": "https://example.invalid/dop20rgb_32_416_5830_2_ni_2025-04-20.tif",
    }],
}
BDOM_PAYLOAD = {
    "count": 1,
    "links": [{
        "state": "NI", "produkt": "bdom", "tile_key": "32_416_5830", "gsd": 0.2,
        "aktualitaet": "2025-04-20",
        "url": "https://example.invalid/bdom20_32_416_5830_1_ni_2025-04-20.tif",
    }],
}
DGM_PAYLOAD = {
    "count": 1,
    "tiles": [{
        "state": "NI", "tile_key": "32_416_5830", "aktualitaet": "2017-03-14",
        "url": "https://example.invalid/dgm1_32_416_5830.tif",
    }],
}


class FakeResponse(io.BytesIO):
    """Enough of an HTTP response for urlopen's context-manager use."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture
def fake_api(monkeypatch):
    """Answer any API request from the recorded payloads, recording the URLs."""
    calls = []

    def urlopen(url, timeout=None):
        calls.append(url)
        if "/dop/links" in url:
            payload = DOP_PAYLOAD
        elif "/dgm/links" in url:
            payload = BDOM_PAYLOAD
        elif "/dgm/tiles" in url:
            payload = DGM_PAYLOAD
        else:
            raise AssertionError(f"unexpected request: {url}")
        return FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return calls


def write_raster(path, values, pixel_size, origin=(400000.0, 5830000.0), nodata=None):
    """A single-band GeoTIFF in EPSG:25832."""
    values = np.asarray(values, dtype="float32")
    profile = {
        "driver": "GTiff", "height": values.shape[0], "width": values.shape[1],
        "count": 1, "dtype": "float32", "crs": "EPSG:25832",
        "transform": from_origin(origin[0], origin[1], pixel_size, pixel_size),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)
    return path


# ── Tile ids ─────────────────────────────────────────────────────────────────


def test_tile_key_round_trip():
    assert tile_key_to_id("32_416_5830") == "324165830"
    assert tile_id_to_key("324165830") == "32_416_5830"


def test_link_filename_is_what_stage_three_parses():
    link = Link(tile_id="324165830", product="dop", url="…", date="2025-04-20")

    assert link.filename == "324165830_2025-04-20.tif"


# ── API client ───────────────────────────────────────────────────────────────


def test_client_parses_each_endpoint(fake_api):
    client = EasyGeoData(pause=0.0)
    bounds = (7.75, 52.61, 7.77, 52.62)

    dop = client.orthophotos(bounds)
    bdom = client.surface_models(bounds)
    dgm = client.ground_models(bounds)

    assert [l.product for l in dop + bdom + dgm] == ["dop", "bdom", "dgm"]
    assert dop[0].tile_id == "324165830"
    assert dop[0].date == "2025-04-20"
    assert dop[0].bands == 4
    # The ground model is older than the imagery — that is normal, it is
    # flown once and reused.
    assert dgm[0].date == "2017-03-14"


def test_client_sends_the_bbox_and_the_product(fake_api):
    client = EasyGeoData(pause=0.0)

    client.surface_models((7.75, 52.61, 7.77, 52.62))

    assert "produkt=bdom" in fake_api[0]
    assert "bbox=7.750000%2C52.610000%2C7.770000%2C52.620000" in fake_api[0]


def test_links_for_tile_collects_every_product(fake_api):
    client = EasyGeoData(pause=0.0)

    links = client.links_for_tile("324165830", (7.75, 52.61, 7.77, 52.62),
                                  ["dop", "bdom", "dgm"])

    assert sorted(l.product for l in links) == ["bdom", "dgm", "dop"]
    assert {l.tile_id for l in links} == {"324165830"}


def test_a_coarser_orthophoto_tile_is_still_kept(monkeypatch):
    # Lower Saxony publishes DOP on a 2 km grid, so the key names a
    # different tile. Dropping it would leave every tile without imagery.
    payload = {"links": [dict(DOP_PAYLOAD["links"][0], tile_key="32_416_5828")]}

    def urlopen(url, timeout=None):
        return FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    links = EasyGeoData(pause=0.0).links_for_tile(
        "324165830", (7.75, 52.61, 7.77, 52.62), ["dop"]
    )

    assert [l.tile_id for l in links] == ["324165830"]


def test_a_rejected_parameter_is_not_retried(monkeypatch):
    import urllib.error

    calls = []

    def urlopen(url, timeout=None):
        calls.append(url)
        raise urllib.error.HTTPError(url, 400, "Bad Request", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    with pytest.raises(ValueError, match="rejected"):
        EasyGeoData(pause=0.0, retries=3).orthophotos((7.75, 52.61, 7.77, 52.62))
    assert len(calls) == 1


def test_an_unreachable_api_raises_after_its_retries(monkeypatch):
    calls = []

    def urlopen(url, timeout=None):
        calls.append(url)
        raise OSError("network is down")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    with pytest.raises(ConnectionError, match="unreachable"):
        EasyGeoData(pause=0.0, retries=2).orthophotos((7.75, 52.61, 7.77, 52.62))
    assert len(calls) == 2


# ── Ground models ────────────────────────────────────────────────────────────


def test_an_orthophoto_without_a_matching_surface_model_is_dropped():
    # The API serves the newest of each product independently, so this
    # pair is two different flights and the nDSM would describe other trees.
    links = pd.DataFrame([
        {"tile_id": "1", "date": "2026-03-06", "product": "dop", "url": "a"},
        {"tile_id": "1", "date": "2025-03-06", "product": "bdom", "url": "b"},
        {"tile_id": "2", "date": "2025-04-20", "product": "dop", "url": "c"},
        {"tile_id": "2", "date": "2025-04-20", "product": "bdom", "url": "d"},
    ])

    kept, dropped = match_acquisitions(links)

    assert set(kept["tile_id"]) == {"2"}
    assert len(dropped) == 2


def test_ground_models_survive_the_pairing_check():
    links = pd.DataFrame([
        {"tile_id": "1", "date": "2025-04-20", "product": "dop", "url": "a"},
        {"tile_id": "1", "date": "2025-04-20", "product": "bdom", "url": "b"},
        {"tile_id": "1", "date": "2017-03-14", "product": "dgm", "url": "c"},
    ])

    kept, dropped = match_acquisitions(links)

    assert dropped.empty
    assert set(kept["product"]) == {"dop", "bdom", "dgm"}


def test_pairing_is_skipped_when_only_one_product_was_asked_for():
    links = pd.DataFrame([
        {"tile_id": "1", "date": "2026-03-06", "product": "dop", "url": "a"},
    ])

    kept, dropped = match_acquisitions(links)

    assert len(kept) == 1
    assert dropped.empty


def test_one_ground_model_lands_once_per_acquisition():
    links = pd.DataFrame([
        {"tile_id": "1", "date": "2023-06-25", "product": "dop", "url": "a"},
        {"tile_id": "1", "date": "2025-04-20", "product": "dop", "url": "b"},
        {"tile_id": "1", "date": "2017-03-14", "product": "dgm", "url": "c"},
    ])

    paired = pair_ground_models(links)

    ground = paired[paired["product"] == "dgm"]
    assert sorted(ground["date"]) == ["2023-06-25", "2025-04-20"]
    # Its real acquisition date survives for the log.
    assert set(ground["source_date"]) == {"2017-03-14"}


def test_pairing_without_ground_models_changes_nothing():
    links = pd.DataFrame([
        {"tile_id": "1", "date": "2023-06-25", "product": "dop", "url": "a"},
    ])

    assert len(pair_ground_models(links)) == 1


# ── The index route ──────────────────────────────────────────────────────────


def make_tiles():
    return gpd.GeoDataFrame(
        {"tile_id": ["324165830"]},
        geometry=[box(416000, 5830000, 417000, 5831000)],
        crs="EPSG:25832",
    )


def test_index_matches_the_surface_model_by_tile_id():
    index = gpd.GeoDataFrame(
        {"tile_id": ["324165830", "324165831"],
         "Aktualitaet": ["2023-06-25", "2023-06-25"],
         "bdom": ["url-a", "url-b"]},
        geometry=[box(416000, 5830000, 417000, 5831000),
                  box(416000, 5831000, 417000, 5832000)],
        crs="EPSG:25832",
    )

    links = links_from_index({"bdom": (index, "bdom")}, ["324165830"])

    assert list(links["url"]) == ["url-a"]
    assert list(links["date"]) == ["2023-06-25"]


def test_a_two_kilometre_orthophoto_is_matched_spatially():
    # Its own id names a different grid, so only the geometry can match it.
    index = gpd.GeoDataFrame(
        {"tile_id": ["32_416_5830_2km"], "Aktualitaet": ["2023-06-25"],
         "rgbi": ["url-dop"]},
        geometry=[box(416000, 5830000, 418000, 5832000)],
        crs="EPSG:25832",
    )

    links = links_from_index({"dop": (index, "rgbi")}, ["324165830"],
                             tiles=make_tiles())

    assert list(links["url"]) == ["url-dop"]
    assert list(links["tile_id"]) == ["324165830"]


def test_a_neighbour_that_only_shares_a_border_does_not_match():
    index = gpd.GeoDataFrame(
        {"tile_id": ["x"], "Aktualitaet": ["2023-06-25"], "rgbi": ["url-dop"]},
        geometry=[box(417000, 5830000, 419000, 5832000)],
        crs="EPSG:25832",
    )

    links = links_from_index({"dop": (index, "rgbi")}, ["324165830"],
                             tiles=make_tiles())

    assert links.empty


def test_resolving_orthophotos_without_geometry_says_why():
    index = gpd.GeoDataFrame(
        {"tile_id": ["a"], "Aktualitaet": ["2023-06-25"], "rgbi": ["u"]},
        geometry=[box(0, 0, 1, 1)], crs="EPSG:25832",
    )

    with pytest.raises(ValueError, match="different grid"):
        links_from_index({"dop": (index, "rgbi")}, ["324165830"])


# ── Downloading ──────────────────────────────────────────────────────────────


def test_download_writes_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda url, timeout=None: FakeResponse(b"raster bytes"))
    target = tmp_path / "tile.tif"

    status, error = download("https://example.invalid/x.tif", target)

    assert (status, error) == ("success", None)
    assert target.read_bytes() == b"raster bytes"


def test_an_existing_file_is_not_fetched_again(tmp_path, monkeypatch):
    def urlopen(url, timeout=None):
        raise AssertionError("should not download")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    target = tmp_path / "tile.tif"
    target.write_bytes(b"already here")

    assert download("https://example.invalid/x.tif", target)[0] == "skipped"


def test_a_failed_download_leaves_no_partial_file(tmp_path, monkeypatch):
    # A truncated raster is worse than none: the next run would skip it.
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda url, timeout=None: (_ for _ in ()).throw(OSError("boom")))
    target = tmp_path / "tile.tif"

    status, error = download("https://example.invalid/x.tif", target, retries=1)

    assert status == "failed"
    assert "boom" in error
    assert not target.exists()
    assert list(tmp_path.glob("*.part")) == []


# ── Rasters ──────────────────────────────────────────────────────────────────


def test_crop_cuts_a_coarse_tile_down_to_its_tile(tmp_path):
    source = write_raster(tmp_path / "dop.tif", np.ones((100, 100)), pixel_size=20.0)
    tile = box(400000, 5828000, 401000, 5829000)

    assert crop_to_geometry(source, tile, tmp_path / "cropped.tif")

    with rasterio.open(tmp_path / "cropped.tif") as dataset:
        assert dataset.shape == (50, 50)


def test_canopy_over_a_hill_measures_its_own_height(tmp_path):
    # 100 m hill, 10 m canopy. The nDSM must say 10.
    surface = write_raster(tmp_path / "bdom.tif", np.full((25, 25), 110.0),
                           pixel_size=0.2)
    ground = write_raster(tmp_path / "dgm.tif", np.full((5, 5), 100.0),
                          pixel_size=1.0)

    status, error = normalised_surface(surface, ground, tmp_path / "ndsm.tif")

    assert (status, error) == ("success", None)
    with rasterio.open(tmp_path / "ndsm.tif") as dataset:
        heights = dataset.read(1)
        assert dataset.res == (0.2, 0.2)
        assert np.allclose(heights, 10.0)


def test_nodata_survives_the_subtraction(tmp_path):
    values = np.full((25, 25), 110.0)
    values[0, 0] = -9999.0
    surface = write_raster(tmp_path / "bdom.tif", values, pixel_size=0.2,
                           nodata=-9999.0)
    ground = write_raster(tmp_path / "dgm.tif", np.full((5, 5), 100.0),
                          pixel_size=1.0)

    normalised_surface(surface, ground, tmp_path / "ndsm.tif")

    with rasterio.open(tmp_path / "ndsm.tif") as dataset:
        heights = dataset.read(1)
    assert heights[0, 0] == -9999.0
    assert np.isclose(heights[10, 10], 10.0)


def test_a_missing_ground_model_is_reported_not_raised(tmp_path):
    surface = write_raster(tmp_path / "bdom.tif", np.ones((5, 5)), pixel_size=0.2)

    status, error = normalised_surface(surface, tmp_path / "absent.tif",
                                       tmp_path / "ndsm.tif")

    assert status == "failed"
    assert error
