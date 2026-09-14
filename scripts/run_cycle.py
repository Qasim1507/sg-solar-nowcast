"""One scheduled cycle: issue a forecast, then collect whatever truth has landed.

This is what GitHub Actions runs every 15 minutes instead of keeping uvicorn alive.
It calls the SAME functions the in-process scheduler calls - api.main.issue_forecast
and api.main.collect_outcomes - so the daylight gate, the store write, the lane
separation and the best_match guard all behave identically. Nothing is reimplemented
here; this only sequences them and reports.

NO_SCHEDULER=1 is set before importing api.main so the APScheduler startup hook does
not fire inside a batch job.

Usage:
    python scripts/run_cycle.py
    python scripts/run_cycle.py --skip-outcomes
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-outcomes", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="issue even outside the daylight window (testing only)")
    args = ap.parse_args()

    setup_logging()
    import api.main as api

    if args.force:
        api.in_daylight = lambda t=None: True

    out = api.issue_forecast()
    if out is None:
        log.info("no forecast issued (outside 08:00-17:00 SGT, or the fetch failed)")
    else:
        med = [round(h["median"], 1) for h in out["horizons"]]
        log.info("issued %s  medians=%s", out["issue_time"], med)

    if not args.skip_outcomes:
        for source in ("analysis", "era5"):
            n = api.collect_outcomes(source)
            log.info("outcomes[%s]: %d new", source, n)

    store = api.get_store()
    if store is not None:
        log.info("store: %s", storage.summary(store))
    # A failed cycle must not fail the workflow: a transient API outage should skip
    # one issuance, not stop the schedule or leave the committed CSVs half-written.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
