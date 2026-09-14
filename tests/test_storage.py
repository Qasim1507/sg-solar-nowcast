"""Forecast store: idempotency, lane separation, and the mode distinction.

The store exists so live forecasts can be scored as outcomes arrive. Two things
must hold or the verification panel lies: re-issuing a forecast must not inflate
the sample count, and the provisional ('analysis') lane must never be silently
mixed with the real ('era5') one.
"""
import json

import pytest

import storage


@pytest.fixture
def conn(tmp_path):
    c = storage.connect(tmp_path / "store.db")
    yield c
    c.close()


def _pkg(issue="2024-06-15T11:00:00", base=500.0):
    """A Forecaster._package()-shaped result. Valid times derive from the issue
    time, as they do in the real thing."""
    day, hhmmss = issue.split("T")
    hour = int(hhmmss[:2])
    return {
        "issue_time": issue,
        "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
        "kt_now": 0.82,
        "horizons": [
            {"horizon_h": h, "valid_time": f"{day}T{hour + h:02d}:00:00",
             "quantiles": {"0.1": base - 80, "0.25": base - 40, "0.5": base,
                           "0.75": base + 40, "0.9": base + 80},
             "median": base, "clearsky_ghi": 900.0}
            for h in (1, 2, 3)
        ],
    }


def test_reissuing_the_same_forecast_does_not_duplicate(conn):
    """A scheduler that fires twice must not double the sample count."""
    storage.record_package(conn, _pkg(), "catboost_A_kt")
    storage.record_package(conn, _pkg(), "catboost_A_kt")
    assert storage.summary(conn)["forecasts"] == 3


def test_reissue_overwrites_with_the_newer_values(conn):
    storage.record_package(conn, _pkg(base=500.0), "catboost_A_kt")
    storage.record_package(conn, _pkg(base=610.0), "catboost_A_kt")
    storage.record_outcome(conn, "2024-06-15T12:00:00", "era5", 555.0)
    rows = storage.scored_rows(conn, "era5", "2024-06-15")
    assert len(rows) == 1
    assert rows[0]["median"] == 610.0


def test_two_models_coexist_at_the_same_issue_time(conn):
    storage.record_package(conn, _pkg(), "catboost_A_kt")
    storage.record_package(conn, _pkg(), "A_gated_seed0")
    s = storage.summary(conn)
    assert s["forecasts"] == 6
    assert s["models"] == ["A_gated_seed0", "catboost_A_kt"]


def test_lanes_are_scored_separately(conn):
    """The provisional lane must never leak into the era5 lane."""
    storage.record_package(conn, _pkg(), "catboost_A_kt")
    storage.record_outcome(conn, "2024-06-15T12:00:00", "analysis", 480.0)
    storage.record_outcome(conn, "2024-06-15T13:00:00", "era5", 512.0)

    prov = storage.scored_rows(conn, "analysis", "2024-06-15")
    final = storage.scored_rows(conn, "era5", "2024-06-15")
    assert [r["actual"] for r in prov] == [480.0]
    assert [r["actual"] for r in final] == [512.0]


def test_unscored_forecasts_are_excluded(conn):
    """A forecast with no outcome yet must not appear as if it were scored."""
    storage.record_package(conn, _pkg(), "catboost_A_kt")
    assert storage.scored_rows(conn, "era5", "2024-06-15") == []


def test_missing_outcomes_lists_only_what_is_due(conn):
    storage.record_package(conn, _pkg(), "catboost_A_kt")
    storage.record_outcome(conn, "2024-06-15T12:00:00", "era5", 512.0)
    due = storage.missing_outcomes(conn, "era5", before="2024-06-15T13:30:00")
    assert due == ["2024-06-15T13:00:00"]          # 14:00 is not due yet, 12:00 is done


def test_replay_rows_stay_distinguishable_from_live(conn):
    """Backfilled numbers must never be passed off as a live track record."""
    storage.record_package(conn, _pkg("2024-06-15T11:00:00"), "catboost_A_kt", mode="replay")
    storage.record_package(conn, _pkg("2026-09-14T11:00:00"), "catboost_A_kt", mode="live")
    s = storage.summary(conn)
    assert s["replay"] == 3 and s["live"] == 3

    storage.record_outcome(conn, "2024-06-15T12:00:00", "era5", 500.0)
    assert storage.scored_rows(conn, "era5", "2024-06-15", mode="live") == []
    assert len(storage.scored_rows(conn, "era5", "2024-06-15", mode="replay")) == 1


def test_quantiles_round_trip_in_configured_order(conn):
    storage.record_package(conn, _pkg(base=500.0), "catboost_A_kt")
    storage.record_outcome(conn, "2024-06-15T12:00:00", "era5", 500.0)
    q = storage.scored_rows(conn, "era5", "2024-06-15")[0]["quantiles"]
    assert q == [420.0, 460.0, 500.0, 540.0, 580.0]
    assert q == sorted(q), "quantiles must survive storage in ascending order"


def test_bad_enum_values_are_rejected(conn):
    with pytest.raises(ValueError):
        storage.record_outcome(conn, "2024-06-15T12:00:00", "guess", 1.0)
    with pytest.raises(ValueError):
        storage.record_forecast(conn, issue_time="t", valid_time="t", horizon_h=1,
                                model_id="m", values=[1, 2, 3, 4, 5], mode="pretend")


def test_bulk_outcomes_skip_nulls_and_nans(conn):
    n = storage.record_outcomes(conn, [
        ("2024-06-15T12:00:00", 500.0),
        ("2024-06-15T13:00:00", None),
        ("2024-06-15T14:00:00", float("nan")),
    ], "analysis")
    assert n == 1


def test_restore_preserves_timestamps(conn, tmp_path):
    """A no-op CSV round-trip must produce a zero-line diff.

    record_forecast stamps created_at=now by default. If a restore from CSV lets it
    do that, every row's timestamp is rewritten and the exported CSV diffs in full
    on every scheduled run - 630 changed lines for 3 new forecasts, which makes the
    git-backed store unusable.
    """
    import sys
    sys.path.insert(0, str(tmp_path.parent))
    from scripts.store_sync import export, import_

    storage.record_package(conn, _pkg(), "catboost_A_kt")
    storage.record_outcome(conn, "2024-06-15T12:00:00", "era5", 500.0)
    export(conn, tmp_path)
    first = (tmp_path / "forecasts.csv").read_text()
    first_o = (tmp_path / "outcomes.csv").read_text()

    rebuilt = storage.connect(tmp_path / "rebuilt.db")
    import_(rebuilt, tmp_path)
    out = tmp_path / "again"
    export(rebuilt, out)

    assert (out / "forecasts.csv").read_text() == first
    assert (out / "outcomes.csv").read_text() == first_o
    rebuilt.close()
