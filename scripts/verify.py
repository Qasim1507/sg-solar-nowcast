"""SCORE PAST FORECASTS. Run this a few hours after scripts/forecast.py.

    python scripts/verify.py

One command. It fetches whatever truth has become available, joins it to the
forecasts already stored, and writes data_store/verification.csv - one row per
(issue_time, horizon, truth source) with the prediction, what actually happened,
and the error.

TWO TRUTH SOURCES, AND THEY ARRIVE AT DIFFERENT TIMES
-----------------------------------------------------
  analysis  ~1 hour behind. Available when you run this the same afternoon, but
            PROVISIONAL: it is a model product the forecaster itself consumes, and
            it disagrees with ERA5 by ~97 W/m2 on average. Treat it as a liveness
            check, not a quality measure.
  era5      ~3 days behind. The real target, and what REPORT.md is scored against.

So running this daily scores today's forecasts provisionally AND fills in the
final ERA5 numbers for forecasts made about three days ago. One habit covers both;
nothing extra to remember.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("NO_SCHEDULER", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import storage  # noqa: E402
from data._client import log, setup_logging  # noqa: E402
from scripts.store_sync import DATA, export, import_  # noqa: E402

COLS = ["issue_time", "valid_time", "horizon_h", "source", "mode",
        "median", "lower", "upper", "actual", "error", "abs_error", "in_interval"]


def write_verification(conn, cfg, data_dir: Path) -> int:
    """Rebuild verification.csv from the store. Derived, so a full rewrite is safe."""
    rows = []
    for source in storage.SOURCES:
        for r in storage.scored_rows(conn, source, "0000"):
            q = r["quantiles"]
            med = r["median"] if r["median"] is not None else q[len(q) // 2]
            lo, hi = q[0], q[-1]
            actual = r["actual"]
            rows.append({
                "issue_time": r["issue_time"], "valid_time": r["valid_time"],
                "horizon_h": r["horizon_h"], "source": source, "mode": r["mode"],
                "median": round(med, 2), "lower": round(lo, 2), "upper": round(hi, 2),
                "actual": round(actual, 2),
                "error": round(med - actual, 2),
                "abs_error": round(abs(med - actual), 2),
                "in_interval": int(lo <= actual <= hi),
            })
    rows.sort(key=lambda r: (r["issue_time"], r["horizon_h"], r["source"]))
    path = data_dir / "verification.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    setup_logging()
    import api.main as api
    import verification

    data_dir = Path(args.data_dir) if args.data_dir else DATA
    store = api.get_store()
    if store is None:
        log.error("could not open the forecast store")
        return 1
    import_(store, data_dir)

    got = {}
    for source in ("analysis", "era5"):
        got[source] = api.collect_outcomes(source)
    print()
    print(f"  New outcomes: {got['analysis']} provisional, {got['era5']} final (ERA5)")

    n = write_verification(store, api.CFG, data_dir)
    export(store, data_dir)

    for source, label in (("era5", "ERA5 (final)"), ("analysis", "analysis (PROVISIONAL)")):
        rep = verification.score_window(store, api.CFG, window="30d", source=source,
                                        model_id=api.model_id(), mode="live",
                                        now=api.sgt_now())
        print()
        print(f"  {label} - live forecasts only, last 30 days")
        if not rep["horizons"]:
            lag = "arrives ~3 days after the forecast hour" if source == "era5" \
                  else "arrives ~1 hour after the forecast hour"
            print(f"    nothing scored yet ({lag})")
            continue
        print(f"    {'horizon':<10}{'n':>4}{'MAE':>9}{'pinball':>9}{'coverage':>10}")
        for hz, r in rep["horizons"].items():
            print(f"    {hz:<10}{r['n']:>4}{r['mae']:>9.1f}{r['pinball']:>9.2f}"
                  f"{r['coverage']:>10.3f}")

    print(f"\n  Wrote {n} scored rows -> {data_dir / 'verification.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
