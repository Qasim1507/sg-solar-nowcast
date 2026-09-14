"""API contract tests, including DEGRADED states.

Degraded states are the ones that mislead - a dashboard that silently renders a
stale or empty forecast as if it were current is the failure this suite exists to
prevent. So every endpoint is checked twice: once normally, once with its data
source removed.
"""
import json
import os

import pytest

# Tests must be deterministic and offline-capable. Replay mode drives the
# forecast through the stored dataset instead of the live fetchers, which is both
# fast and exactly the path a defence demo uses.
os.environ["NO_SCHEDULER"] = "1"
os.environ["REPLAY_MODE"] = "1"

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

import api.main as apimain  # noqa: E402

apimain.REPLAY_MODE = True          # in case the module was imported earlier
client = TestClient(apimain.app)


@pytest.fixture(autouse=True)
def _force_replay(monkeypatch):
    monkeypatch.setattr(apimain, "REPLAY_MODE", True, raising=False)

ENDPOINTS = ["/api/health", "/api/model/info", "/api/verification",
             "/api/stations", "/api/forecast/latest", "/api/raingrid/latest"]


# ------------------------------------------------------------------ contract
@pytest.mark.parametrize("ep", ["/api/health", "/api/model/info", "/api/verification"])
def test_endpoint_returns_json_with_as_of(ep):
    r = client.get(ep)
    assert r.status_code in (200, 503)
    body = r.json()
    assert "as_of" in body, f"{ep} has no as_of timestamp"


def test_every_endpoint_reports_data_age():
    """A response must never leave the caller guessing how old its inputs are."""
    for ep in ["/api/health", "/api/model/info", "/api/verification"]:
        body = client.get(ep).json()
        assert "data_age_minutes" in body, f"{ep} omits data_age_minutes"


def test_health_schema():
    b = client.get("/api/health").json()
    for k in ("sources", "warnings", "healthy", "replay_mode", "data_age_minutes"):
        assert k in b
    assert isinstance(b["warnings"], list)
    assert isinstance(b["healthy"], bool)
    for name in ("era5", "nwp", "nea"):
        assert name in b["sources"]


def test_model_info_schema():
    b = client.get("/api/model/info").json()
    assert "degraded" in b
    if not b["degraded"]:
        for k in ("checkpoint", "variant", "track", "quantiles", "horizons",
                  "target_product", "nominal_interval", "known_limitations"):
            assert k in b, f"model/info missing {k}"
        assert b["target_product"].startswith("ERA5"), "target must be labelled ERA5"


def test_verification_declares_it_is_not_independent_truth():
    b = client.get("/api/verification").json()
    assert "note" in b
    assert "not independent ground truth" in b["note"]


def test_verification_nominal_level_is_derived_not_assumed(cfg):
    b = client.get("/api/verification").json()
    expected = round(max(cfg["quantiles"]) - min(cfg["quantiles"]), 4)
    assert b["nominal_interval"] == expected
    assert b["nominal_interval"] != 0.9 or expected == 0.9


def test_stations_schema():
    """Uses the stored station roster; the live overlay is allowed to fail."""
    b = client.get("/api/stations").json()
    assert "stations" in b and isinstance(b["stations"], list)
    assert "n_known" in b and "n_online" in b
    for s in b["stations"][:5]:
        for k in ("id", "name", "lat", "lon", "value", "online"):
            assert k in s


# ---------------------------------------------------------------- forecast
def test_forecast_latest_schema_or_degraded():
    r = client.get("/api/forecast/latest")
    b = r.json()
    if r.status_code == 503 or b.get("degraded"):
        assert "error" in b and "horizons" in b
        return
    for k in ("as_of", "issue_time", "horizons", "quantiles", "nominal_interval",
              "diagnostics", "model", "target_product", "data_age_minutes"):
        assert k in b, f"forecast missing {k}"
    assert b["timezone"].startswith("SGT")
    for h in b["horizons"]:
        for k in ("horizon_h", "valid_time", "quantiles", "median", "clearsky_ghi"):
            assert k in h


def test_fan_chart_quantiles_are_monotonic():
    """Snapshot test: the quantiles as RENDERED must never cross."""
    b = client.get("/api/forecast/latest").json()
    if b.get("degraded"):
        pytest.skip("no forecast available")
    qs = [str(q) for q in b["quantiles"]]
    for h in b["horizons"]:
        vals = [h["quantiles"][q] for q in qs]
        assert vals == sorted(vals), (
            f"quantiles cross at t+{h['horizon_h']}h: {vals}")
        assert h["lower"] <= h["median"] <= h["upper"]


def test_forecast_never_hides_its_age():
    b = client.get("/api/forecast/latest").json()
    if b.get("degraded"):
        pytest.skip("no forecast available")
    assert "forecast" in b["data_age_minutes"]
    assert "stale" in b


# ------------------------------------------------------------ degraded modes
def test_absent_model_degrades_gracefully(monkeypatch):
    """No checkpoint: endpoints must say so, not 500."""
    monkeypatch.setattr(apimain, "latest_checkpoint", lambda: None)
    monkeypatch.setitem(apimain._state, "forecaster", None)
    monkeypatch.setitem(apimain._state, "error", None)

    r = client.get("/api/model/info")
    assert r.status_code == 200
    assert r.json()["degraded"] is True

    r = client.get("/api/forecast/latest")
    b = r.json()
    # a cached forecast may still be served, but it must be marked
    assert ("degraded" in b) or ("error" in b)

    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["checkpoint"] is None

    apimain._state["forecaster"] = None


def test_absent_data_sources_are_reported_as_warnings(monkeypatch, tmp_path):
    """Missing raw files must surface as explicit warnings, not silence."""
    monkeypatch.setitem(apimain.CFG["paths"], "raw", str(tmp_path / "nowhere"))
    b = client.get("/api/health").json()
    assert b["healthy"] is False
    assert len(b["warnings"]) >= 1
    assert any("not fetched" in w for w in b["warnings"])


def test_absent_evaluation_reports_degraded(monkeypatch, tmp_path):
    monkeypatch.setitem(apimain.CFG["paths"], "reports", str(tmp_path / "none"))
    b = client.get("/api/verification").json()
    assert b["degraded"] is True
    assert b["tracks"] == {}


def test_stale_forecast_is_flagged(monkeypatch):
    """An old cached forecast must be marked stale, never presented as current."""
    stale = {
        "as_of": "2020-01-01 09:00:00", "issue_time": "2020-01-01 09:00:00",
        "timezone": "SGT (UTC+8)", "mode": "replay", "horizons": [],
        "quantiles": [0.1, 0.5, 0.9], "nominal_interval": 0.8,
        "diagnostics": {}, "model": {}, "data_age_minutes": {},
        "target_product": "ERA5 reanalysis",
    }
    monkeypatch.setattr(apimain, "cached_forecast", lambda: stale)
    b = client.get("/api/forecast/latest").json()
    assert b["stale"] is True, "a years-old forecast was not flagged stale"
    assert b["data_age_minutes"]["forecast"] > 1000


def test_partial_raingrid_degrades(monkeypatch):
    """If the gauge fetch fails, the map endpoint must degrade, not crash."""
    def boom(*a, **k):
        raise RuntimeError("gauge API unavailable")
    f = apimain.get_forecaster()
    if f is None:
        pytest.skip("no model")
    monkeypatch.setattr(type(f), "_live_grid", boom)
    b = client.get("/api/raingrid/latest").json()
    assert b["degraded"] is True
    assert "error" in b
    assert b["grid"] == [] and b["stations"] == []


def test_history_handles_missing_dataset(monkeypatch, tmp_path):
    monkeypatch.setitem(apimain.CFG["paths"], "processed", str(tmp_path / "none"))
    b = client.get("/api/forecast/history?days=2").json()
    assert b.get("degraded") is True or b.get("items") == []


def test_history_rejects_absurd_ranges():
    assert client.get("/api/forecast/history?days=0").status_code == 422
    assert client.get("/api/forecast/history?days=999").status_code == 422


# ------------------------------------------------------------------ frontend
def test_frontend_is_served():
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    assert "Singapore GHI Nowcast" in html
    for panel in ("fan", "map", "verification", "calibration", "model-info",
                  "diagnostics", "health", "limitations"):
        assert f'id="{panel}"' in html, f"panel #{panel} missing from the page"


def test_frontend_labels_timezone_and_product():
    html = client.get("/").text
    assert "SGT" in html
    assert "ERA5 reanalysis" in html, "observed GHI is not labelled as reanalysis"
    assert "consistency check" in html


def test_cache_from_a_different_checkpoint_is_discarded(monkeypatch, tmp_path):
    """A cached forecast must not outlive the model that produced it.

    After retraining, a stale cache would still be served - correctly flagged
    stale, but attributed to a checkpoint that may no longer exist.
    """
    stale = {
        "as_of": "2020-01-01 09:00:00", "issue_time": "2020-01-01 09:00:00",
        "timezone": "SGT (UTC+8)", "mode": "replay", "horizons": [],
        "quantiles": [0.1, 0.5, 0.9], "nominal_interval": 0.8,
        "diagnostics": {}, "data_age_minutes": {},
        "model": {"checkpoint": "SOME_OLD_MODEL.pt", "track": "B"},
        "target_product": "ERA5 reanalysis",
    }
    cache = tmp_path / "forecast_latest.json"
    cache.write_text(json.dumps(stale))
    monkeypatch.setattr(apimain, "CACHE", cache)
    assert apimain.cached_forecast() is None, "cache from another checkpoint was reused"

    # and a cache matching the current checkpoint IS reused
    ck = apimain.latest_checkpoint()
    if ck is not None:
        stale["model"]["checkpoint"] = ck.name
        cache.write_text(json.dumps(stale))
        assert apimain.cached_forecast() is not None
