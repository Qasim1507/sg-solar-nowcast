# forecast-data

Durable state for the nowcaster, kept off `main` so its history stays readable.

- `data_store/forecasts.csv` — every forecast issued, one row per (issue_time, horizon, model)
- `data_store/outcomes.csv` — observed GHI, one row per (valid_time, source)

Written by `.github/workflows/nowcast.yml` every 15 minutes during SGT daylight.
`mode` distinguishes `live` (genuinely issued then) from `replay` (backfilled), and
`source` separates `era5` (the real target, ~2-3 day lag) from `analysis`
(provisional, ~1 h). The two are never mixed.

Do not edit by hand — `scripts/store_sync.py` rebuilds sqlite from these.
