"""Append-only store of issued forecasts and the outcomes they are scored against.

Until now nothing the model forecast was ever kept: the scheduler overwrote a
single `forecast_latest.json`, `/api/forecast/history` replayed the stored dataset
(a backtest, not a log), and `/api/verification` served whatever `evaluate.py` last
wrote offline. None of that can answer "how has this model actually done this week".

TWO OUTCOME SOURCES, NEVER MIXED
--------------------------------
The training target is ERA5, which lags ~2 days, so live forecasts cannot be
scored against truth as they land. Outcomes therefore arrive in two lanes:

  'analysis'  Open-Meteo forecast-API analysis, available ~1h after valid time.
              PROVISIONAL: it is a model product that the forecaster itself
              consumes, so scoring against it partly measures agreement with the
              same source. Always labelled.
  'era5'      The real target, ~2-3 days later. This is what REPORT.md is scored
              against.

Callers pick a lane explicitly; nothing here ever averages the two.

`mode` on a forecast is 'live' (genuinely issued at that moment) or 'replay'
(backfilled by re-running the model over historical rows), so a populated panel
can never pass backfilled numbers off as a live track record.

sqlite3 is used because it ships with Python - the store needs windowed joins,
which parquet append cannot do, and no new dependency is worth a table this small.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    issue_time     TEXT    NOT NULL,
    valid_time     TEXT    NOT NULL,
    horizon_h      INTEGER NOT NULL,
    model_id       TEXT    NOT NULL,
    quantiles_json TEXT    NOT NULL,   -- predicted values, in configured quantile order
    median         REAL,
    clearsky       REAL,
    kt_now         REAL,
    mode           TEXT    NOT NULL,   -- 'live' | 'replay'
    created_at     TEXT    NOT NULL,
    PRIMARY KEY (issue_time, horizon_h, model_id)
);
CREATE INDEX IF NOT EXISTS idx_forecasts_valid ON forecasts(valid_time);
CREATE INDEX IF NOT EXISTS idx_forecasts_model ON forecasts(model_id, valid_time);

CREATE TABLE IF NOT EXISTS outcomes (
    valid_time TEXT NOT NULL,
    source     TEXT NOT NULL,          -- 'analysis' | 'era5'
    ghi        REAL NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (valid_time, source)
);
"""

SOURCES = ("analysis", "era5")
MODES = ("live", "replay")


def default_path(cfg: dict, root: Path | None = None) -> Path:
    root = Path(root or Path(__file__).resolve().parent)
    return root / "artifacts" / "forecast_store.db"


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # the scheduler writes while requests read; WAL lets both proceed
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def record_forecast(conn: sqlite3.Connection, *, issue_time: str, valid_time: str,
                    horizon_h: int, model_id: str, values: list[float],
                    median: float | None = None, clearsky: float | None = None,
                    kt_now: float | None = None, mode: str = "live",
                    created_at: str | None = None) -> None:
    """Upsert one (issue_time, horizon, model) forecast.

    Re-issuing the same issue time replaces rather than duplicates, so a scheduler
    that fires twice in a minute cannot inflate the sample count.

    `created_at` defaults to now, but a restore from CSV MUST pass the original:
    otherwise rebuilding the store rewrites every row's timestamp and the exported
    CSV diffs in full on every run, which is unusable as a git-backed store.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    conn.execute(
        """INSERT INTO forecasts (issue_time, valid_time, horizon_h, model_id,
                                  quantiles_json, median, clearsky, kt_now, mode, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(issue_time, horizon_h, model_id) DO UPDATE SET
               valid_time=excluded.valid_time, quantiles_json=excluded.quantiles_json,
               median=excluded.median, clearsky=excluded.clearsky,
               kt_now=excluded.kt_now, mode=excluded.mode, created_at=excluded.created_at""",
        (issue_time, valid_time, int(horizon_h), model_id,
         json.dumps([float(v) for v in values]),
         None if median is None else float(median),
         None if clearsky is None else float(clearsky),
         None if kt_now is None else float(kt_now),
         mode, created_at or _now()),
    )
    conn.commit()


def record_package(conn: sqlite3.Connection, pkg: dict, model_id: str,
                   mode: str = "live") -> int:
    """Store every horizon of a Forecaster._package() result. Returns rows written."""
    qs = [str(q) for q in pkg["quantiles"]]
    n = 0
    for hz in pkg["horizons"]:
        record_forecast(
            conn,
            issue_time=str(pkg["issue_time"]),
            valid_time=str(hz["valid_time"]),
            horizon_h=int(hz["horizon_h"]),
            model_id=model_id,
            values=[hz["quantiles"][q] for q in qs],
            median=hz.get("median"),
            clearsky=hz.get("clearsky_ghi"),
            kt_now=pkg.get("kt_now"),
            mode=mode,
        )
        n += 1
    return n


def record_outcome(conn: sqlite3.Connection, valid_time: str, source: str,
                   ghi: float, fetched_at: str | None = None) -> None:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    conn.execute(
        """INSERT INTO outcomes (valid_time, source, ghi, fetched_at) VALUES (?,?,?,?)
           ON CONFLICT(valid_time, source) DO UPDATE SET
               ghi=excluded.ghi, fetched_at=excluded.fetched_at""",
        (valid_time, source, float(ghi), fetched_at or _now()),
    )
    conn.commit()


def record_outcomes(conn: sqlite3.Connection, rows, source: str) -> int:
    """Bulk upsert. Rows are (valid_time, ghi) or (valid_time, ghi, fetched_at).

    Non-finite values are skipped. Pass fetched_at when restoring from CSV - see
    record_forecast on why rewritten timestamps make the git-backed store churn.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    payload = []
    for r in rows:
        t, g = r[0], r[1]
        fetched = r[2] if len(r) > 2 else None
        if g is None or float(g) != float(g):
            continue
        payload.append((str(t), source, float(g), fetched or _now()))
    if not payload:
        return 0
    conn.executemany(
        """INSERT INTO outcomes (valid_time, source, ghi, fetched_at) VALUES (?,?,?,?)
           ON CONFLICT(valid_time, source) DO UPDATE SET
               ghi=excluded.ghi, fetched_at=excluded.fetched_at""",
        payload,
    )
    conn.commit()
    return len(payload)


def missing_outcomes(conn: sqlite3.Connection, source: str, before: str) -> list[str]:
    """Valid times that have a forecast but no outcome yet from `source`."""
    cur = conn.execute(
        """SELECT DISTINCT f.valid_time FROM forecasts f
           LEFT JOIN outcomes o ON o.valid_time = f.valid_time AND o.source = ?
           WHERE o.valid_time IS NULL AND f.valid_time <= ?
           ORDER BY f.valid_time""",
        (source, before),
    )
    return [r["valid_time"] for r in cur.fetchall()]


def scored_rows(conn: sqlite3.Connection, source: str, since: str,
                model_id: str | None = None, mode: str | None = None) -> list[dict]:
    """Forecasts joined to their outcome from one lane, newest window first."""
    sql = """SELECT f.issue_time, f.valid_time, f.horizon_h, f.model_id, f.mode,
                    f.quantiles_json, f.median, f.clearsky, f.kt_now, o.ghi AS actual
             FROM forecasts f
             JOIN outcomes o ON o.valid_time = f.valid_time AND o.source = ?
             WHERE f.valid_time >= ?"""
    args: list = [source, since]
    if model_id:
        sql += " AND f.model_id = ?"
        args.append(model_id)
    if mode:
        sql += " AND f.mode = ?"
        args.append(mode)
    sql += " ORDER BY f.valid_time"
    out = []
    for r in conn.execute(sql, args).fetchall():
        d = dict(r)
        d["quantiles"] = json.loads(d.pop("quantiles_json"))
        out.append(d)
    return out


def summary(conn: sqlite3.Connection) -> dict:
    f = conn.execute(
        """SELECT COUNT(*) n, MIN(issue_time) first, MAX(issue_time) last,
                  SUM(mode='live') n_live, SUM(mode='replay') n_replay
           FROM forecasts"""
    ).fetchone()
    o = {r["source"]: r["n"] for r in conn.execute(
        "SELECT source, COUNT(*) n FROM outcomes GROUP BY source").fetchall()}
    models = [r["model_id"] for r in conn.execute(
        "SELECT DISTINCT model_id FROM forecasts ORDER BY model_id").fetchall()]
    return {
        "forecasts": f["n"] or 0,
        "live": f["n_live"] or 0,
        "replay": f["n_replay"] or 0,
        "first_issue": f["first"], "last_issue": f["last"],
        "outcomes": o, "models": models,
    }
