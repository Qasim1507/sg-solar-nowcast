# Multimodal Solar Irradiance Nowcasting — Singapore

Probabilistic Global Horizontal Irradiance (GHI) forecasts for Singapore
(**1.3521 N, 103.8198 E**) at **t+1h, t+2h and t+3h**, fusing three modalities:

| Branch | Source | Model |
|---|---|---|
| Temporal | Open-Meteo ERA5 archive (point weather) | BiLSTM |
| Spatial | NEA rain-gauge network via data.gov.sg v2 | small CNN + cross-attention |
| Head inputs | Open-Meteo historical-forecast NWP + clear-sky | concatenated at fusion |

Output is a **quantile fan** (`0.10, 0.25, 0.50, 0.75, 0.90`), not a point estimate,
trained with pinball loss. Daylight only (08:00–17:00 SGT). Everything is
config-driven via `config.yaml`.

---

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e .

python data/fetch_weather.py                 # ERA5 archive + pvlib clear-sky
python data/fetch_nwp.py                     # NWP (explicit models only)
python data/fetch_nea.py --start 2020-01-01  # gauge backfill (long; resumable)
python data/build_grids.py                   # gauge points -> rasters + mask
python data/build_dataset.py                 # aligned splits + train_stats.json

pytest                                       # correctness gates
python baselines.py --track A                # baselines BEFORE any deep model
python train.py --track A --ablation         # 4 variants x 3 seeds
python evaluate.py --track A

uvicorn api.main:app                         # dashboard at http://127.0.0.1:8000
```

Replay a stored day (no network, works at night — use this for a defence):

```bash
REPLAY_DATE=2024-06-15 uvicorn api.main:app
```

---

## What was verified about the data sources

All three APIs were probed directly before any code was written. Several findings
changed the design:

**The default NWP path leaks the target.** The Historical Forecast API's default
(`best_match`) returns values *bit-identical* to the ERA5 archive target, confirmed
at the same grid cell (1.3708 N, 103.802 E) on 2024-01-15, 2024-06-15, 2025-03-10
and 2026-08-20. Feeding it as "the forecast of the target hour" would hand the model
its own answer. Every NWP request therefore pins an explicit `models=`;
`best_match` is refused in code, and `tests/test_nwp_leakage.py` is the regression
guard.

**Archive depths differ per source**, which is why the project runs two tracks:

| Source | Earliest usable | Note |
|---|---|---|
| ERA5 archive | 2020 and earlier | ~2-day lag |
| NEA rainfall | 2020-01-01 | station count varies 51 → 88 |
| NWP `gfs_seamless` | ~2021-04-01 | full Track A coverage, live API parity |
| NWP `icon_seamless` | ~2023-01-01 | optional second model |
| NWP `ecmwf_ifs025` | ~mid-2024 | most accurate, shortest archive |

**Operational quirks handled in `data/_client.py` and `data/fetch_nea.py`:**

- The rate limiter returns **HTTP 200 with `code: 24` in the body**.
  `raise_for_status()` sails straight past it, so the client inspects the JSON code.
- The pagination token is `base64("offset=N")` and **crafted offsets work**, so the
  fetcher seeks straight to the daylight window: 5 requests/day instead of ~12.
- **Partial days exist** (2021-01-01 and 2023-01-01 both end at 07:55). The manifest
  records actual vs expected frame counts so these surface loudly.
- Station membership changes over time and even between adjacent 5-minute frames, so
  readings are joined to stations by `stationId` **per response**.
- The newest 5-minute frame is usually **partially reported** (3 of 88 gauges).
  Live inference takes the most recent frame with adequate coverage and reports its
  true age.

---

## Two evaluation tracks

No NWP archive exists before ~2021-04, but ERA5 and the gauge network both reach
2020. Rather than truncate the record, the dataset spans 2020-01-01 → present with a
per-row `nwp_ok` flag:

- **Track A (headline)** — 2021-04 onward, all three modalities present. Every
  model, baseline and deep, is scored on **identical rows**.
- **Track B (long record)** — 2020-01-01 onward, no NWP branch. Quantifies what the
  extra ~15 months buys the temporal and spatial branches.

---

## Correctness constraints, and why each exists

Each of these is a bug that was **measured** in a prior satellite-based attempt.
They are enforced in code and pinned by tests.

| Constraint | Prior failure | Enforced by |
|---|---|---|
| Time-indexed targets/lags, never positional `shift()` | 10/20/30% of t+1/2/3h targets contaminated; `ghi_lag1` stale by 15h on 10% of rows | `data/build_dataset.py::assert_alignment`, `tests/test_alignment.py` |
| Quantile head + pinball, not Gaussian NLL | cloudy-row spread was 52–63% of reality; 90% intervals covered 81.4% on cloudy days | `models/head.py`, `tests/test_quantiles.py` |
| Pad lookbacks with the training mean | zero-padding sent inputs in at −12.6σ | `data/dataset.py`, `tests/test_padding.py` |
| Filter before splitting | unusable tail landed entirely in test, collapsing it to 18 samples | `data/build_dataset.py`, `min_test_samples` assertion |
| Geometry self-tests | prior project trained on imagery centred **1,100 km** from Singapore | `data/geometry.py::self_test`, `tests/test_geometry.py` |
| Stratified calibration as a first-class metric | aggregate coverage hid a cloudy-only failure | `metrics.py::stratified_report` |
| CNN parameter budget | 14.4M image-branch params against ~5,300 samples | `models/spatial.py` (~81k params) |

Two further constraints were added after they showed up here:

- **Night targets are excluded explicitly.** Targets are built on the full hourly
  frame so the time-reindex is correct, which means a 17:00 issue time gets a real —
  but nocturnal — t+3h value of ~0 W/m², not NaN. Near-zero targets are trivially
  easy and inflate every metric, so the target *hour* is filtered directly rather
  than left to whatever the grid join happens to drop.
- **The alignment verifier refuses to pass vacuously.** Its first version compared
  only where both sides were non-NaN, which silently skipped exactly the
  day-boundary rows where a positional shift goes wrong — it passed a frame built
  with `shift()`. It now requires a full hourly reference, treats a non-NaN value
  whose hour is absent from that reference as an error, and raises if it cannot
  verify a minimum fraction of rows.

### A note on the nominal interval

With `QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]`, the central band `[q0.10, q0.90]` is
an **80%** prediction interval, not 90%. Scoring it against a 90% target would make
a perfectly calibrated model look ~10 points broken. The nominal level is therefore
**derived from the quantile levels** everywhere and reported under that name. To get
a true 90% interval, add `0.05` and `0.95` to `quantiles` in `config.yaml`.

---

## Dashboard

`uvicorn api.main:app` serves both the API and the single-page frontend
(Plotly.js + Leaflet, no build step).

### Served model

The dashboard serves **CatBoost + k_t**, which beats every deep variant at every
horizon on Track A (t+1h 18.26, t+2h 25.22, t+3h 27.28 pinball, 3-seed means, seed
sd <= 0.08). Save it before first run; the torch checkpoint is a fallback:

```bash
python baselines.py --track A --save        # -> artifacts/checkpoints/catboost_A_kt.*
```

### Continuous issuance and rolling verification

The scheduler issues a forecast every `api.refresh_minutes`, **only between 08:00
and 17:00 SGT** - night targets were excluded from training and are ~0 W/m2, so
issuing after dark would pad the metrics with rows the model was never fitted for.
Every forecast is appended to `artifacts/forecast_store.db` (sqlite), and outcomes
are joined in as they arrive.

**Truth arrives in two lanes, and they are never mixed:**

| Lane | Source | Latency | What it means |
|---|---|---|---|
| `analysis` | Open-Meteo forecast-API analysis | ~1 h | **Provisional.** A model product this forecaster also consumes, so it flatters. |
| `era5` | ERA5 archive | ~2-3 days | The real target, scored the way `REPORT.md` is. |

`GET /api/verification/rolling?window=24h|7d|30d&source=era5|analysis` reports
per-horizon metrics over a moving window, via the same `metrics.full_report` that
`evaluate.py` uses. It is distinct from `/api/verification`, which serves the static
offline evaluation.

Rows carry `mode`: `live` (genuinely issued then) or `replay` (backfilled). The API
reports the two counts separately so a backfilled panel can never read as a live
track record. To populate it on day one:

```bash
python scripts/backfill_forecast_store.py --track A --days 30
```

Note a short window is dominated by recent weather rather than model quality - the
last 30 days of the record have mean k_t 0.93 against 0.89 across the full test
period, which alone moves MAE substantially.

---

## Daily operation

Two commands. Run the first in the morning, the second a few hours later.

```bash
python scripts/forecast.py      # 08:00-17:00 SGT: predict t+1/2/3h
python scripts/verify.py        # a few hours later: score what has happened
```

Neither needs a server, a database or any setup step - both read and write
`data_store/*.csv` directly.

### What each does

`forecast.py` fetches live Open-Meteo analysis, NEA gauge readings and GFS NWP,
runs the served CatBoost + k_t model, prints the three horizons, and appends to
`data_store/forecasts.csv`. It refuses outside 08:00-17:00 SGT, because the model
was trained only on daylight issue times and a night forecast is extrapolation.

`verify.py` fetches whatever truth has arrived, scores every stored forecast
against it, and writes `data_store/verification.csv` - one row per
(issue_time, horizon, source) with the prediction, the actual, the error, and
whether the outcome fell inside the 80% interval.

### The two truth sources arrive at different times

| Source | Lag | What it is |
|---|---|---|
| `analysis` | ~1 hour | **Provisional.** A model product the forecaster also consumes; it disagrees with ERA5 by ~97 W/m2 on average. A liveness check, not a quality measure. |
| `era5` | ~3 days | The real target, scored the way `REPORT.md` is. |

Running `verify.py` three hours after forecasting only fills the provisional lane -
ERA5 does not exist yet. That resolves itself: because you run it daily, each run
also backfills the final ERA5 numbers for forecasts made about three days earlier.
One habit covers both lanes.

### Why this is manual

It was on a GitHub Actions cron. GitHub delivered roughly **one of ten** requested
slots per day, hours late, and `schedule` carries no SLA on free runners. The
workflow is still in the repo for `workflow_dispatch`, so the published dashboard
can be rebuilt from the Actions tab when you want it refreshed.

Note also that `predict.gather_live_inputs` floors the issue time to the hour and
the store is keyed on `(issue_time, horizon, model)`, so running `forecast.py`
twice within the same hour updates one row rather than adding a second - there is
no value in running it more than hourly.

### What the static build gives up

The 3-hour rain scrubber (`/api/raingrid/series`) is live-only: it is 36 fresh
5-minute frames per request and cannot be published every cycle without bloating the
deploy. The static page hides the slider and says so. Run `uvicorn api.main:app`
locally for the full interactive version.

| Endpoint | Returns |
|---|---|
| `GET /api/forecast/latest` | current forecast, all quantiles, clear-sky, diagnostics |
| `GET /api/forecast/history?days=N` | past forecasts joined to observed outcomes |
| `GET /api/raingrid/latest` | interpolated grid + raw station values + mask |
| `GET /api/raingrid/series?minutes=N` | 5-minute frames for the map time slider |
| `GET /api/stations` | station metadata, last reading, online flag |
| `GET /api/verification` | rolling skill, coverage, stratified calibration |
| `GET /api/model/info` | checkpoint identity, split sizes, features, limitations |
| `GET /api/health` | per-source freshness and staleness warnings |

Every response carries `as_of` and `data_age_minutes`. A stale forecast presented as
current is the failure mode the API is designed against, so degraded states are
explicit fields rather than omissions, and `tests/test_api.py` exercises stale,
partial and absent data.

Panels: quantile fan chart, rain-field map (masked cells hatched, motion arrow,
3-hour scrubber), verification, stratified calibration, model/data status, and
collapsible diagnostics. Dark and light themes, SGT labelled everywhere.

---

## Known limitations

- **The gauge network spans only ~50 km** (measured: 28 km N–S × 47 km E–W). It
  describes current conditions *over* the island, not what is arriving from
  upwind, so its value decays with horizon. The NWP head inputs are what cover
  that gap, and that is the main reason they are included.
- **The rain field is extremely sparse.** A typical midday frame has 0–1 wet gauges
  out of 60–88, and the cross-correlation motion vector is undefined for roughly
  95% of frames — so the physics gate leans on clear-sky index and cloud cover far
  more than on motion.
- **The target is ERA5 reanalysis, not a pyranometer.** It is a modelled product.
  Verification uses the same product the model was trained on, which makes it a
  consistency check, not independent ground truth. The dashboard states this.
- **Live inference has train/serve skew.** ERA5 lags ~2 days, so live mode takes
  recent observations from the Open-Meteo forecast API's analysis (`past_days`)
  instead. That is a different product from the ERA5 the model trained on. It is
  disclosed in the API response and on the dashboard rather than hidden.
- **NWP lead time is approximate.** The historical-forecast endpoint serves the
  archived series for a day, not specifically the run issued at time *t* targeting
  *t+1/2/3h*. A true issue-time archive was not available for this period; this is
  documented rather than silently replaced with reanalysis.
- **Single-site target.** One grid cell of ERA5, not a spatial irradiance field, so
  the model says nothing about spatial variability of irradiance across the island.
- **Station metadata is accumulated across years** (106 known IDs, 88 currently
  online), so historical grids are built from a sparser network than today's.
- **Outside 08:00–17:00 SGT the model is out of its training window** and outputs are
  dominated by the clear-sky clamp. The dashboard shows a prominent warning.

---

## Repository layout

```
config.yaml            site, grid extent/resolution, horizons, quantiles, splits
data/
  _client.py           retrying/rate-limited HTTP, in-body error codes, resume state
  geometry.py          lat/lon <-> grid transforms + self-test
  fetch_weather.py     ERA5 archive -> hourly CSV + pvlib clear-sky
  fetch_nea.py         data.gov.sg v2, pagination, resumable, manifest
  fetch_nwp.py         historical-forecast, explicit models, leakage check
  build_grids.py       gauge points -> (4,64,64) .npy + mask + motion + manifest
  build_dataset.py     time-indexed join, daylight filter, splits, train_stats.json
  dataset.py           torch Dataset: lookback windows, mean padding
models/                temporal.py spatial.py gate.py head.py fusion.py
metrics.py             point, probabilistic, stratified calibration
baselines.py  train.py  evaluate.py  predict.py
api/                   main.py + static/ (dashboard)
tests/                 geometry, alignment, padding, quantiles, leakage, dataset, API
```
