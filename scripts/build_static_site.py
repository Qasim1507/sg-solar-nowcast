"""Render the dashboard as a static site: no server, nothing to keep alive.

The FastAPI app is fine locally but needs a process running to answer requests, and
free hosts either sleep or charge for that. Since the scheduled job already runs the
model every cycle, it can just as well write the JSON each panel reads, and the page
becomes plain files on GitHub Pages - permanently up, no runtime.

Every endpoint the frontend calls gets one file under site/data/. The names must
match the mapping in api/static/app.js::getJSON when window.STATIC_BUILD is set.

ONE PANEL IS DROPPED: /api/raingrid/series, the 3-hour scrubber. It is 36 live
5-minute frames per request; precomputing it would add ~40 KB x 40 runs/day to the
deploy. The frontend hides the slider when the file is absent rather than erroring.

Usage:
    python scripts/build_static_site.py [--out site]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("NO_SCHEDULER", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data._client import log, setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WINDOWS = ("24h", "7d", "30d")
SOURCES = ("era5", "analysis")


def _write(path: Path, payload) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, default=float)
    path.write_text(text)
    return len(text)


def build(out_dir: Path) -> None:
    import api.main as api

    data = out_dir / "data"
    total = 0

    # Each of these mirrors one endpoint. They are called directly rather than
    # through HTTP so the build needs no server; the functions already return
    # plain dicts and already degrade instead of raising.
    panels = {
        "forecast_latest": lambda: api.forecast_latest(refresh=False),
        "health": api.health,
        "model_info": api.model_info,
        "verification": api.verification,
        "raingrid_latest": api.raingrid_latest,
    }
    for name, fn in panels.items():
        try:
            payload = fn()
        except Exception as e:                       # a dead panel must not kill the build
            log.warning("panel %s failed: %s", name, str(e)[:160])
            payload = {"degraded": True, "error": str(e)[:200]}
        total += _write(data / f"{name}.json", payload)
        log.info("  %-22s %s", name + ".json",
                 "degraded" if payload.get("degraded") else "ok")

    # rolling verification: one file per (window, source), each carrying all three
    # horizons' timelines so the horizon selector needs no extra request
    for window in WINDOWS:
        for source in SOURCES:
            try:
                payload = api.verification_rolling(window=window, source=source, horizon=1)
                payload["timelines"] = {}
                store = api.get_store()
                if store is not None:
                    import verification
                    for h in (1, 2, 3):
                        payload["timelines"][str(h)] = verification.timeline(
                            store, window=window, source=source,
                            model_id=api.model_id(), horizon=h, now=api.sgt_now())
                payload.pop("timeline", None)
            except Exception as e:
                log.warning("rolling %s/%s failed: %s", window, source, str(e)[:160])
                payload = {"degraded": True, "error": str(e)[:200],
                           "horizons": {}, "timelines": {}}
            total += _write(data / f"rolling_{window}_{source}.json", payload)

    # static assets, with the build flag injected so getJSON reads files not /api
    static = ROOT / "api" / "static"
    for item in static.iterdir():
        if item.is_file():
            shutil.copy2(item, out_dir / item.name)
        else:
            shutil.copytree(item, out_dir / item.name, dirs_exist_ok=True)

    index = out_dir / "index.html"
    html = index.read_text()
    flag = '<script>window.STATIC_BUILD = true;</script>'
    if flag not in html:
        html = html.replace("<script src=\"/static/app.js\"></script>",
                            f"{flag}\n<script src=\"app.js\"></script>")
        html = html.replace('href="/static/style.css"', 'href="style.css"')
        index.write_text(html)

    # Pages serves _-prefixed paths oddly under Jekyll; disable it
    (out_dir / ".nojekyll").write_text("")
    log.info("wrote %s  (%.1f KB of JSON)", out_dir, total / 1024)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site")
    args = ap.parse_args()
    setup_logging()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    build(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
