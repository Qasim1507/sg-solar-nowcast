"""Rolling verification: the daylight gate, lane separation, and scorer agreement.

The panel this feeds is the thing a reader will trust most, so the failure modes
that matter are the quiet ones: scoring night rows the model was never fitted for,
letting the provisional lane pass as ERA5, or drifting away from the metrics
evaluate.py reports.
"""
import numpy as np
import pandas as pd
import pytest

import metrics
import storage
import verification


@pytest.fixture
def conn(tmp_path):
    c = storage.connect(tmp_path / "v.db")
    yield c
    c.close()


def _seed(conn, n=40, source="era5", mode="replay", model_id="catboost_A_kt",
          start="2026-09-01T09:00:00"):
    rng = np.random.default_rng(0)
    t0 = pd.Timestamp(start)
    actuals = []
    for i in range(n):
        issue = t0 + pd.Timedelta(hours=i)
        valid = issue + pd.Timedelta(hours=1)
        med = 500.0 + rng.normal(0, 60)
        storage.record_forecast(
            conn, issue_time=str(issue), valid_time=str(valid), horizon_h=1,
            model_id=model_id,
            values=[med - 90, med - 45, med, med + 45, med + 90],
            median=med, clearsky=900.0, kt_now=0.8, mode=mode)
        a = med + rng.normal(0, 50)
        storage.record_outcome(conn, str(valid), source, a)
        actuals.append((med, a))
    return actuals


# ---------------------------------------------------------------- daylight gate
def test_daylight_gate_rejects_night(cfg):
    import api.main as apimain
    assert apimain.in_daylight(pd.Timestamp("2026-09-14 19:00:00")) is False
    assert apimain.in_daylight(pd.Timestamp("2026-09-14 03:00:00")) is False


def test_daylight_gate_accepts_the_training_window(cfg):
    import api.main as apimain
    lo = cfg["daylight"]["start_hour"]
    hi = cfg["daylight"]["end_hour"]
    for hour in range(lo, hi + 1):
        assert apimain.in_daylight(pd.Timestamp(f"2026-09-14 {hour:02d}:00:00")) is True
    assert apimain.in_daylight(pd.Timestamp(f"2026-09-14 {lo - 1:02d}:00:00")) is False
    assert apimain.in_daylight(pd.Timestamp(f"2026-09-14 {hi + 1:02d}:00:00")) is False


# ------------------------------------------------------------------- scoring
def test_scorer_agrees_with_metrics_full_report(conn, cfg):
    """The rolling scorer must not drift from what evaluate.py reports."""
    _seed(conn, n=30)
    now = pd.Timestamp("2026-09-03T12:00:00")
    rep = verification.score_window(conn, cfg, window="7d", source="era5", now=now)

    rows = storage.scored_rows(conn, "era5", verification.window_start("7d", now))
    pred = np.array([r["quantiles"] for r in rows], dtype=float)
    y = np.array([r["actual"] for r in rows], dtype=float)
    cs = np.array([r["clearsky"] for r in rows], dtype=float)
    kt = np.array([r["kt_now"] for r in rows], dtype=float)
    expect = metrics.full_report(pred, y, cs, kt, cfg["quantiles"],
                                 [tuple(b) for b in cfg["eval"]["cloud_bins"]])

    got = rep["horizons"]["t+1h"]
    assert got["n"] == expect["n"]
    assert got["mae"] == pytest.approx(expect["mae"])
    assert got["pinball"] == pytest.approx(expect["pinball"])
    assert got["coverage"] == pytest.approx(expect["coverage"])


def test_empty_store_degrades_rather_than_raising(conn, cfg):
    rep = verification.score_window(conn, cfg, window="7d", source="era5")
    assert rep["horizons"] == {} and rep["n_total"] == 0


def test_unknown_window_is_rejected(conn, cfg):
    with pytest.raises(ValueError):
        verification.score_window(conn, cfg, window="all-time", source="era5")


# -------------------------------------------------------------- lane labelling
def test_analysis_lane_is_flagged_provisional(conn, cfg):
    _seed(conn, n=10, source="analysis")
    now = pd.Timestamp("2026-09-03T12:00:00")
    prov = verification.score_window(conn, cfg, window="7d", source="analysis", now=now)
    assert prov["provisional"] is True
    assert "NOT ERA5" in prov["source_label"] or "PROVISIONAL" in prov["source_label"]

    final = verification.score_window(conn, cfg, window="7d", source="era5", now=now)
    assert final["provisional"] is False
    assert final["n_total"] == 0, "analysis outcomes leaked into the era5 lane"


def test_replay_and_live_counts_are_reported_separately(conn, cfg):
    """A populated panel must never present backfilled rows as a live record."""
    _seed(conn, n=10, mode="replay", start="2026-09-01T09:00:00")
    _seed(conn, n=5, mode="live", start="2026-09-02T09:00:00")
    now = pd.Timestamp("2026-09-03T12:00:00")
    rep = verification.score_window(conn, cfg, window="7d", source="era5", now=now)
    h = rep["horizons"]["t+1h"]
    assert h["n_live"] == 5 and h["n_replay"] == 10
    assert h["n"] == h["n_live"] + h["n_replay"]


def test_window_narrows_the_sample(conn, cfg):
    _seed(conn, n=60, start="2026-08-20T09:00:00")
    now = pd.Timestamp("2026-08-23T12:00:00")
    wide = verification.score_window(conn, cfg, window="30d", source="era5", now=now)
    narrow = verification.score_window(conn, cfg, window="24h", source="era5", now=now)
    assert narrow["n_total"] < wide["n_total"]


# ---------------------------------------------------------------- timeline
def test_timeline_is_chronological_and_horizon_filtered(conn, cfg):
    _seed(conn, n=12)
    now = pd.Timestamp("2026-09-03T12:00:00")
    pts = verification.timeline(conn, window="7d", source="era5", horizon=1, now=now)
    assert len(pts) == 12
    assert [p["valid_time"] for p in pts] == sorted(p["valid_time"] for p in pts)
    assert all(p["lower"] <= p["median"] <= p["upper"] for p in pts)
    assert verification.timeline(conn, window="7d", source="era5", horizon=3, now=now) == []


def test_outcome_timestamps_are_normalised_before_matching(conn, monkeypatch, cfg):
    """Open-Meteo returns '...T15:00'; the store keys on '... 15:00:00'.

    Regression guard: an un-normalised comparison matches nothing, and the lane
    stays permanently empty while the job logs a cheerful '0 of N filled'.
    """
    import api.main as apimain

    storage.record_forecast(
        conn, issue_time="2026-09-14T14:00:00", valid_time="2026-09-14 15:00:00",
        horizon_h=1, model_id="catboost_A_kt", values=[1, 2, 3, 4, 5],
        median=3.0, clearsky=900.0, kt_now=0.8, mode="live")

    monkeypatch.setattr(apimain, "get_store", lambda: conn)
    monkeypatch.setattr(apimain, "_fetch_ghi",
                        lambda url, start, end, model=None: [("2026-09-14T15:00", 612.0)])
    monkeypatch.setattr(apimain, "sgt_now",
                        lambda: pd.Timestamp("2026-09-14 17:00:00"))

    assert apimain.collect_outcomes("analysis") == 1
    rows = storage.scored_rows(conn, "analysis", "2026-09-14")
    assert len(rows) == 1 and rows[0]["actual"] == 612.0


def test_provisional_lane_pins_an_explicit_nwp_model(conn, monkeypatch, cfg):
    """The provisional lane must never fetch Open-Meteo's best_match default.

    best_match returns values bit-identical to the ERA5 archive (verified in
    data/fetch_nwp.py, guarded by tests/test_nwp_leakage.py). An unpinned request
    makes the 'provisional' lane a relabelled copy of the final one - it agreed
    with ERA5 on 271/271 rows to 0.000000 before this was pinned.
    """
    import api.main as apimain

    storage.record_forecast(
        conn, issue_time="2026-09-14T14:00:00", valid_time="2026-09-14 15:00:00",
        horizon_h=1, model_id="catboost_A_kt", values=[1, 2, 3, 4, 5],
        median=3.0, clearsky=900.0, kt_now=0.8, mode="live")

    seen = {}

    def fake_fetch(url, start, end, model=None):
        seen["url"], seen["model"] = url, model
        return [("2026-09-14T15:00", 600.0)]

    monkeypatch.setattr(apimain, "_fetch_ghi", fake_fetch)
    monkeypatch.setattr(apimain, "get_store", lambda: conn)
    monkeypatch.setattr(apimain, "sgt_now", lambda: pd.Timestamp("2026-09-14 17:00:00"))

    apimain.collect_outcomes("analysis")
    assert seen["model"], "provisional lane fetched with no models= (best_match)"
    assert seen["model"] != "best_match"
    assert seen["model"] == cfg["nwp"]["models"][0]

    # The ERA5 lane hits the archive endpoint, where no model pin applies. It needs
    # a row older than archive_lag_days - anything newer is correctly not yet due.
    storage.record_forecast(
        conn, issue_time="2026-09-01T14:00:00", valid_time="2026-09-01 15:00:00",
        horizon_h=1, model_id="catboost_A_kt", values=[1, 2, 3, 4, 5],
        median=3.0, clearsky=900.0, kt_now=0.8, mode="live")
    apimain.collect_outcomes("era5")
    assert seen["url"] == cfg["weather"]["archive_url"] and seen["model"] is None
