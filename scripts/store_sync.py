"""Move the forecast store between sqlite (runtime) and CSV (durable, in git).

GitHub Actions runners are ephemeral: the sqlite file the scheduler writes is gone
the moment the job ends. Git is the only free durable store available, and a binary
.db committed every 15 minutes would bloat the repo and diff as noise. So the CSVs
are the source of truth in the repo and sqlite is rebuilt from them for each run.

    import : CSV -> sqlite   (before a cycle)
    export : sqlite -> CSV   (after, then commit)

Both are idempotent. `storage.record_forecast` / `record_outcome` upsert on their
primary keys, so importing the same CSV twice changes nothing, and rows are written
in a stable sort order so a run that adds three forecasts produces a three-line diff
rather than a reshuffled file.

Usage:
    python scripts/store_sync.py import
    python scripts/store_sync.py export
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import storage  # noqa: E402
from data._client import load_config, log, setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_store"

FORECAST_COLS = ["issue_time", "valid_time", "horizon_h", "model_id",
                 "quantiles_json", "median", "clearsky", "kt_now", "mode", "created_at"]
OUTCOME_COLS = ["valid_time", "source", "ghi", "fetched_at"]


def export(conn, data_dir: Path) -> tuple[int, int]:
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = conn.execute(
        f"SELECT {', '.join(FORECAST_COLS)} FROM forecasts "
        "ORDER BY issue_time, horizon_h, model_id").fetchall()
    with open(data_dir / "forecasts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FORECAST_COLS)
        w.writerows([[r[c] for c in FORECAST_COLS] for r in rows])

    orows = conn.execute(
        f"SELECT {', '.join(OUTCOME_COLS)} FROM outcomes "
        "ORDER BY valid_time, source").fetchall()
    with open(data_dir / "outcomes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(OUTCOME_COLS)
        w.writerows([[r[c] for c in OUTCOME_COLS] for r in orows])

    return len(rows), len(orows)


def import_(conn, data_dir: Path) -> tuple[int, int]:
    nf = no = 0
    fp = data_dir / "forecasts.csv"
    if fp.exists():
        with open(fp, newline="") as f:
            for r in csv.DictReader(f):
                storage.record_forecast(
                    conn,
                    issue_time=r["issue_time"], valid_time=r["valid_time"],
                    horizon_h=int(r["horizon_h"]), model_id=r["model_id"],
                    values=json.loads(r["quantiles_json"]),
                    median=_f(r["median"]), clearsky=_f(r["clearsky"]),
                    kt_now=_f(r["kt_now"]), mode=r["mode"],
                    created_at=r.get("created_at") or None)
                nf += 1
    op = data_dir / "outcomes.csv"
    if op.exists():
        with open(op, newline="") as f:
            rows = [(r["valid_time"], float(r["ghi"]), r["source"],
                     r.get("fetched_at") or None) for r in csv.DictReader(f)]
        for source in storage.SOURCES:
            batch = [(t, g, fa) for t, g, s, fa in rows if s == source]
            if batch:
                no += storage.record_outcomes(conn, batch, source)
    return nf, no


def _f(v):
    return None if v in ("", None) else float(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["import", "export"])
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    setup_logging()
    cfg = load_config()
    data_dir = Path(args.data_dir) if args.data_dir else DATA
    conn = storage.connect(storage.default_path(cfg))

    if args.action == "export":
        nf, no = export(conn, data_dir)
        log.info("exported %d forecasts, %d outcomes -> %s", nf, no, data_dir)
    else:
        nf, no = import_(conn, data_dir)
        log.info("imported %d forecast rows, %d outcomes from %s", nf, no, data_dir)
    log.info("store: %s", storage.summary(conn))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
