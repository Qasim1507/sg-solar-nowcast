"""MAKE A FORECAST. Run this once a day, any time between 08:00 and 17:00 SGT.

    python scripts/forecast.py

One command, no setup. It reads data_store/*.csv, fetches live weather and gauge
data, predicts t+1/2/3h with the served CatBoost model, and writes the CSVs back.

Nothing is scored here - that is scripts/verify.py, run later once the hours you
forecast have actually happened.

Outside 08:00-17:00 SGT it refuses: the model was trained only on daylight issue
times, so a night forecast would be extrapolation dressed up as a prediction.
Pass --force to override for testing.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("NO_SCHEDULER", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import storage  # noqa: E402
from data._client import log, setup_logging  # noqa: E402
from scripts.store_sync import DATA, export, import_  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="issue even outside 08:00-17:00 SGT (testing only)")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    setup_logging()
    import api.main as api

    data_dir = Path(args.data_dir) if args.data_dir else DATA
    store = api.get_store()
    if store is None:
        log.error("could not open the forecast store")
        return 1
    import_(store, data_dir)

    if args.force:
        api.in_daylight = lambda t=None: True

    out = api.issue_forecast()
    if out is None:
        now = api.sgt_now()
        log.error("no forecast issued - it is %s SGT, outside the 08:00-17:00 "
                  "window the model was trained on (use --force to override)",
                  now.strftime("%H:%M"))
        return 1

    print()
    print(f"  Forecast issued {out['issue_time']} SGT")
    print(f"  {'horizon':<10}{'valid':<8}{'median':>10}{'80% interval':>22}")
    print("  " + "-" * 50)
    for h in out["horizons"]:
        lo, hi = h["lower"], h["upper"]
        print(f"  t+{h['horizon_h']}h{'':<6}{str(h['valid_time'])[11:16]:<8}"
              f"{h['median']:>9.1f}{'':>3}[{lo:>7.1f}, {hi:>7.1f}]")
    print(f"\n  W/m2. kt_now={out['kt_now']:.2f}"
          f"  ·  gauges reporting: {out['diagnostics']['n_stations_reporting']}")
    if out.get("bimodal_warning"):
        print("  NOTE: currently cloudy (k_t < 0.4). Outcomes are close to bimodal "
              "here -\n        it either stays overcast or clears. Expect a wide interval.")

    nf, no = export(store, data_dir)
    log.info("wrote %d forecast rows, %d outcomes -> %s", nf, no, data_dir)
    print(f"\n  Verify after the forecast hours have passed:  python scripts/verify.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
