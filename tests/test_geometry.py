"""Projection round-trip tests.

The prior project trained for months on imagery centred 1,100 km from Singapore
because a hard-coded tile index was never checked against a map. These tests
assert the transform against real, independently-known coordinates.
"""
import json

import numpy as np
import pytest

from data.geometry import GridSpec, haversine_km

# Independently known Singapore landmarks (not derived from the grid config).
LANDMARKS = {
    "Changi Airport": (103.9915, 1.3644),
    "Jurong West":    (103.7050, 1.3404),
    "Marina Bay":     (103.8600, 1.2830),
    "Woodlands":      (103.7860, 1.4360),
}


def test_self_test_passes(cfg):
    GridSpec.from_config(cfg).self_test()


def test_roundtrip_random_points(cfg):
    g = GridSpec.from_config(cfg)
    rng = np.random.default_rng(0)
    lon = rng.uniform(g.lon_min, g.lon_max, 500)
    lat = rng.uniform(g.lat_min, g.lat_max, 500)
    i, j = g.lonlat_to_ij(lon, lat)
    lon2, lat2 = g.ij_to_lonlat(i, j)
    assert np.allclose(lon, lon2, atol=1e-9)
    assert np.allclose(lat, lat2, atol=1e-9)


def test_landmarks_inside_grid_and_ordered(cfg):
    """Every landmark must fall inside the grid, and relative geography must hold."""
    g = GridSpec.from_config(cfg)
    idx = {}
    for name, (lon, lat) in LANDMARKS.items():
        assert g.contains(lon, lat), f"{name} falls outside the configured grid"
        i, j = g.nearest_ij(lon, lat)
        idx[name] = (int(i), int(j))

    # Changi is EAST of Jurong -> larger column index
    assert idx["Changi Airport"][0] > idx["Jurong West"][0]
    # Woodlands is NORTH of Marina Bay -> larger row index (lat increases with j)
    assert idx["Woodlands"][1] > idx["Marina Bay"][1]


def test_cell_centre_is_within_half_a_cell(cfg):
    g = GridSpec.from_config(cfg)
    for lon, lat in LANDMARKS.values():
        i, j = g.nearest_ij(lon, lat)
        clon, clat = g.ij_to_lonlat(i, j)
        assert abs(clon - lon) <= g.dlon / 2 + 1e-9
        assert abs(clat - lat) <= g.dlat / 2 + 1e-9


def test_grid_actually_covers_singapore(cfg):
    """Guards the 1,100-km failure: the grid centre must be ON Singapore."""
    g = GridSpec.from_config(cfg)
    site = (cfg["site"]["longitude"], cfg["site"]["latitude"])
    clon = 0.5 * (g.lon_min + g.lon_max)
    clat = 0.5 * (g.lat_min + g.lat_max)
    assert haversine_km(clon, clat, *site) < 25.0, "grid centre is not near the site"


def test_real_gauges_land_in_grid(cfg, root):
    """Fetched gauge coordinates must rasterise inside the grid."""
    p = root / "artifacts/raw/nea/stations_rainfall.json"
    if not p.exists():
        pytest.skip("no fetched station metadata yet")
    meta = json.loads(p.read_text())
    g = GridSpec.from_config(cfg)
    lons = [m["longitude"] for m in meta.values() if m["longitude"] is not None]
    lats = [m["latitude"] for m in meta.values() if m["latitude"] is not None]
    assert len(lons) >= 30, f"only {len(lons)} stations with coordinates"
    inside = g.contains(np.array(lons), np.array(lats))
    assert inside.all(), f"{(~inside).sum()} gauges fall outside the grid extent"
    # and the network should span tens of km, not hundreds (sanity on units)
    span = haversine_km(min(lons), np.mean(lats), max(lons), np.mean(lats))
    assert 20 < span < 120, f"implausible network span {span:.1f} km"


def test_display_flip_is_north_up(cfg):
    g = GridSpec.from_config(cfg)
    a = np.zeros((g.ny, g.nx), dtype=np.float32)
    a[-1, 0] = 1.0                       # northernmost row in storage order
    d = g.to_display(a)
    assert d[0, 0] == 1.0, "to_display() did not put north at row 0"
