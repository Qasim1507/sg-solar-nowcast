# Evaluation report

Singapore GHI nowcasting, t+1/2/3h, probabilistic (quantile) forecasts.
Generated 2026-09-09 (final run, post both correctness fixes). Within a track, every model is scored on **identical rows**,
and every metric carries its sample size.

Quantiles are `[0.10, 0.25, 0.50, 0.75, 0.90]`, so the central band is an
**80% interval**, not 90% (see README). "coverage" is against 0.80.
"spread/reality" is in **k_t space**; 1.00 is the goal, below ~0.70 means the model
is hedging toward the mean.

---

## 1. Data

Backfill recovered **16,817 / 17,080 issue times (98.5%)** over 2020-01-01 →
2026-09-05. 21 days are absent from the NEA archive entirely (two clusters in Jun
and Jul–Aug 2020), 39 are partial.

| track | modalities | rows | train | val | test | test period |
|---|---|---|---|---|---|---|
| A (headline) | temporal + spatial + NWP | 13,667 | 9,566 | 2,050 | 2,051 | 2025-11-15 → 2026-09-05 |
| B (long record) | temporal + spatial | 16,817 | 11,771 | 2,523 | 2,523 | 2025-09-08 → 2026-09-05 |

Track B's test split spans a full year and is seasonally complete. Both are ~4×
the `min_test_samples` floor of 500; training data is roughly double the prior
project's entire dataset (~5,300 rows).

---

## 2. Two correctness fixes that invalidated earlier results

Both were found after a full set of results had already been produced, and both
changed numbers rather than merely tidying code. Everything in this report is
post-fix.

### 2a. Clear-sky was misaligned by a full hour

pvlib clear-sky was evaluated at `timestamp + 30 min`. Open-Meteo's ERA5
`shortwave_radiation` is the **mean of the preceding hour** — the row labelled
17:00 covers 16:00–17:00.

Detected by a physical invariant: median k_t ran **0.19 at 08:00 to 1.46 at 17:00**,
and a clear-sky index must not trend with time of day. Cross-correlating ERA5 GHI
against clear-sky confirmed the true offset (r = 0.877 at −30 min vs 0.801 at +30).

While live, the 1.25× clear-sky clamp bound on **68.7% of t+3h targets issued at
14:00**, collapsing the predictive interval to a single point, and every k_t-space
calibration figure was distorted.

Fixed by **integrating clear-sky over the preceding hour** (midpoint rule, 5-min
steps) rather than sampling a point — necessary because irradiance is strongly
convex near sunrise. Median k_t is now flat at 0.78–0.87 from 09:00–17:00 and the
clamp binds on 0.0% of rows. Pinned by `tests/test_clearsky.py`.

### 2b. The lookback window was 83% synthetic

The sample frame holds only **issue times 08:00–14:00** (7 rows/day), because a
sample needs all three horizons inside daylight. `build_window` walked back
hour-by-hour *through that frame*, so it hit a gap after at most 7 steps:

```
real steps per window: min=1  median=4  max=7   (of 24)
mean real fraction: 16.5%
windows below the 50% floor: 100.0%
```

The BiLSTM was training on windows that were **83.5% mean-padding**, and every
window was below the floor meant to reject them — `strict_padding=False` had
disabled the guard. The existing padding tests passed throughout because they
exercised `build_window` against a synthetic frame and never asserted anything
about the real dataset.

Fixed by building the lookback from the full **daylight observation series**
(08:00–17:00, every day), stepping back one daylight hour at a time across the
overnight gap — which is what "24 daylight steps" meant. Windows are now 100% real
and span ~3 days. Reaching back past a split boundary is correct rather than
leakage: at inference time those observations exist, and stepping backwards cannot
admit anything after *t*. Pinned by three new tests that assert on the **real**
dataset.

This bug was surfaced by the dashboard, which displayed a
"lookback window is only 29% real data" banner that no test had ever checked.

## 3. Headline results

### Track A, ranked by pinball (n = 2,051 identical rows)

| horizon | winner | pinball | runner-up | pinball |
|---|---|---|---|---|
| t+1h | lightgbm tabular | **19.06** | deep temporal | 19.51 |
| t+2h | **deep concat** | **25.97** | deep gated | 26.04 |
| t+3h | lightgbm tab+grid | **27.99** | lightgbm tabular | 28.51 |

**t+3h, full table**

| model | MAE | pinball | coverage | spread/reality |
|---|---|---|---|---|
| **lightgbm tab+grid** | **79.0** | **27.99** | 0.774 | 0.74 |
| lightgbm tabular | 81.2 | 28.51 | 0.769 | 0.75 |
| deep: gated | 81.8 | 29.07 | 0.795 | 0.69 |
| deep: spatial | 96.0 | 34.00 | 0.844 | 0.99 |
| deep: temporal | 82.2 | 34.23 | 0.895 | 1.60 |
| deep: concat | 81.0 | 36.57 | 0.908 | 2.00 |
| smart persistence | 123.3 | 46.56 | 0.830 | 1.28 |
| persistence | 262.3 | 79.46 | 0.766 | 1.69 |

**t+2h — the deep models win here**

| model | MAE | pinball | coverage | spread/reality |
|---|---|---|---|---|
| **deep: concat** | 74.0 | **25.97** | 0.827 | 0.66 |
| deep: gated | 73.1 | 26.04 | 0.816 | 0.64 |
| lightgbm tab+grid | 73.5 | 26.11 | 0.774 | 0.68 |
| deep: temporal | 74.8 | 26.22 | 0.812 | 0.65 |
| lightgbm tabular | 75.0 | 26.43 | 0.784 | 0.69 |

### Track B (no NWP, n = 2,523), t+3h

| model | MAE | pinball | coverage | spread/reality |
|---|---|---|---|---|
| **lightgbm tab+grid** | **83.1** | **28.90** | 0.771 | 0.72 |
| lightgbm tabular | 84.5 | 29.45 | 0.774 | 0.73 |
| deep: gated | 85.6 | 30.30 | 0.800 | 0.73 |
| deep: concat | 85.9 | 32.74 | 0.861 | 1.17 |
| deep: temporal | 86.2 | 33.13 | 0.873 | 1.20 |
| deep: spatial | 150.2 | 51.54 | 0.838 | 1.58 |

### Verdict

**The result is horizon-dependent and is not a clean win for either side.**

- At **t+1h** LightGBM wins, but only just (19.06 vs 19.51). Before the lookback
  fix the gap was 19.06 vs 21.78, so repairing the temporal branch closed most of
  it — the one place where that fix visibly moved the ranking.
- At **t+2h** the multimodal deep models win on Track A (25.97 vs 26.11). Gated and
  concat both beat every baseline.
- At **t+3h** LightGBM wins by ~4% (27.99 vs 29.07), a wider margin than the ~1%
  seen before the fix.
- On **Track B** (no NWP) LightGBM wins at every horizon.

So the multimodal model earns its place **only at t+2h, and only when NWP is
available**. Everywhere else a gradient-boosted tree on the same features is as
good or better. That is close to the prior project's finding rather than a
reversal of it.

## 4. What each modality is worth

LightGBM on **identical Track A test rows (n = 2,051)**, adding one modality at a
time. Nothing else changes, so this is the cleanest measurement in the study.

| feature set | #feat | t+1h MAE | t+2h MAE | t+3h MAE |
|---|---|---|---|---|
| tabular only | 15 | 53.2 | 76.8 | 84.6 |
| + rain-grid scalars | 23 | 52.8 | 75.8 | 84.1 |
| + NWP at target hour | 33 | 53.3 | **73.5** | **79.0** |

- **NWP is by far the largest single gain**, and its value grows with horizon
  exactly as the design predicted: −0.0 / −2.3 / **−5.1 W/m²** at t+1/2/3h.
- **The rain-gauge field is marginal**: −0.4 / −1.0 / −0.5 W/m². It is not nothing,
  but it is an order of magnitude smaller than NWP.
- **At t+1h neither modality helps.** Recent irradiance persistence dominates.

The Track A vs Track B gap (t+3h MAE 79.0 vs 83.1) points the same way, but those
are different test sets; the table above is the defensible number.

---

## 5. Calibration

Post-fix, calibration is broadly good and no longer shows the prior project's
cloudy-day failure. Stratified by cloud regime, Track A t+3h:

| model | regime | n | coverage | spread/reality | bias |
|---|---|---|---|---|---|
| lightgbm tab+grid | cloudy | 100 | 0.780 | 0.86 | −9.4 |
| lightgbm tab+grid | mixed | 279 | 0.767 | 0.86 | +26.5 |
| lightgbm tab+grid | clear | 1,672 | 0.775 | 0.82 | +17.6 |
| deep: gated | cloudy | 100 | 0.837 | 0.80 | — |
| deep: gated | mixed | 279 | 0.809 | 0.89 | — |
| deep: gated | clear | 1,672 | 0.840 | 0.85 | — |

Coverage is **within ~4 points of nominal in every regime for both models**, and
spread/reality sits at 0.80–0.89 — mild under-dispersion, nothing like the 0.52–0.63
collapse the prior project measured on cloudy rows. **No regime triggers the
hedging flag (<0.70).**

Note the cloudy stratum is small (n = 100 at t+3h) because the corrected k_t puts
only 8.5% of rows below 0.4; those numbers carry real sampling uncertainty.

**Two deep variants are now badly over-dispersed at t+3h**: `concat` reaches
spread/reality 2.00 (coverage 0.908) and `temporal` 1.60 (0.895) — intervals twice
as wide as reality warrants, which is why their pinball scores are poor despite
competitive MAE. `gated` (0.69) and the LightGBM models (0.74–0.75) stay close to
nominal. Combined with the seed spread in §6, the deep variants' t+3h behaviour
should be read as unstable rather than as a clean ranking.

There remains a systematic **positive bias in mixed and clear conditions**
(+26.5 and +17.6 W/m² for LightGBM) — both models are slightly optimistic when the
sky is not already overcast.

---

## 6. Seed variance — why 3 seeds is not optional

Track A pinball across seeds:

| variant | horizon | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| temporal | t+1h | 19.57 | 19.37 | 19.58 |
| temporal | t+3h | 36.60 | **28.82** | 37.26 |
| gated | t+1h | **19.95** | 27.95 | 27.73 |
| gated | t+3h | 29.45 | 28.90 | 28.87 |
| concat | t+1h | 19.41 | 19.50 | **27.81** |
| concat | t+3h | 37.40 | 36.01 | 36.29 |

Several variants show a single seed landing ~40% away from its siblings, and it is
a *different* seed each time. `temporal` at t+3h ranges 28.82–37.26; a single-seed
run could have placed it first or last. Any variant ordering from fewer than three
seeds should be treated as unfounded — exactly the failure the prior project hit.

Note `gated` is the most stable variant at t+3h (28.87–29.45), which is part of why
it is the best-performing deep variant there.

## 7. Physics gate

The gate did **not** collapse. α range across seeds:

| track | seed 0 | seed 1 | seed 2 | threshold | status |
|---|---|---|---|---|---|
| A | 0.221 | 0.198 | 0.243 | 0.10 | ok |
| B | 0.141 | 0.265 | 0.184 | 0.10 | ok |

Median α ≈ 0.48–0.53, so the model genuinely mixes both branches. But Track B seed 0
has a range of only 0.141 — not collapsed, yet close enough to the 0.10 threshold
that the gate is doing little work there. Worth watching rather than declaring
healthy.

Its motion inputs are largely inert regardless: the cross-correlation displacement
is undefined on ~88% of frames because the rain field is so sparse, so α is driven
almost entirely by k_t and cloud cover.

---

## 8. The spatial branch is weak, and the reason is in the data

Spatial-only is the worst deep variant everywhere (Track A t+3h MAE 96.0; Track B
150.2). The cause is measurable and not architectural:

- a typical midday frame has **0–1 wet gauges out of 60–88**
- **~88% of frames** yield no usable motion vector
- the network spans only **28 km N–S × 47 km E–W**

The grid *scalars* (mean, max, wet fraction, spatial std) do help LightGBM slightly,
so a little signal exists — it is just low-dimensional enough that an 81k-parameter
CNN over a 64×64 raster is the wrong instrument. A handful of summary statistics
captures what is there.

---

## 9. Honest caveats

- The target is **ERA5 reanalysis**, not pyranometer measurement, and verification
  uses the same product the model trained on. It is a consistency check, not
  independent ground truth.
- **NWP lead time is approximate**: the historical-forecast endpoint serves the
  archived series for a day, not the run issued exactly at *t*.
- Track A and Track B test sets differ, so cross-track numbers are indicative; §4
  is the defensible NWP measurement.
- The **cloudy stratum is small** (n = 100 at t+3h on Track A).
- **k_t at 08:00 remains unreliable** — the Ineichen model underestimates clear-sky
  at very low sun even after hour-integration, so 08:00 rows sit high in the clear
  bin. Hours 09:00–17:00 are well behaved.
- Live inference has train/serve skew (ERA5 lags ~2 days; live mode substitutes the
  forecast API's analysis). Disclosed in the API response and dashboard.
- The dashboard **has** now been verified in a browser: fan chart, rain map with
  mask hatching and gauge markers, time slider, and the staleness/degradation
  banners all render. Its basemap was switched from CARTO (which began requiring an
  API key and returns a watermarked tile on HTTP 200) to keyless Esri Canvas tiles
  with an OpenStreetMap fallback.
- See README → Known limitations for the full list.

---

## 10. Reproducing

```bash
pytest                                   # 85 correctness tests
python baselines.py --track A
python train.py --track A --ablation     # 4 variants x 3 seeds
python evaluate.py --track A
```

Raw outputs: `artifacts/reports/eval_A.txt`, `eval_B.txt`, `evaluation_A.json`,
`evaluation_B.json`, `baselines_A.txt`, `baselines_B.txt`.
