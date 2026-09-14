/* Dashboard logic. Vanilla JS + Plotly + Leaflet, no build step - this must still
   run in two years when someone reopens the thesis repo. */
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);
const SGT = " SGT";

/* Times from the API are naive SGT wall-clock strings. Never render a naive
   timestamp without its zone label. */
function hhmm(ts) {
  if (!ts) return "—";
  const m = String(ts).match(/(\d{2}):(\d{2})/);
  return m ? `${m[1]}:${m[2]}` : String(ts);
}
function dayTime(ts) {
  if (!ts) return "—";
  return String(ts).replace("T", " ").slice(0, 16) + SGT;
}
function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* ------------------------------------------------------------------ theme */
function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  localStorage.setItem("theme", t);
  $("theme").innerHTML = t === "dark" ? "&#9788; Light" : "&#9789; Dark";
  if (state.forecast) drawFan(state.forecast);
  if (state.map) setBasemap();
}
$("theme").onclick = () => {
  const cur = document.documentElement.getAttribute("data-theme");
  applyTheme(cur === "dark" ? "light" : "dark");
};

const state = { forecast: null, map: null, gridLayer: null, maskLayer: null,
                stationLayer: null, motionLayer: null, series: null, playing: null,
                baseLayer: null, usedFallback: false };

/* ---------------------------------------------------------------- banners */
function banner(msg, kind = "") {
  const d = document.createElement("div");
  d.className = "banner " + kind;
  d.innerHTML = `<span>${kind === "err" ? "&#9888;" : kind === "ok" ? "&#10003;" : "&#9888;"}</span><span>${msg}</span>`;
  $("banners").appendChild(d);
}

/* ----------------------------------------------------------- 1. FAN CHART */
function drawFan(f) {
  const acc = css("--accent"), acc2 = css("--accent-2");
  const b80 = css("--band-80"), b50 = css("--band-50");
  const fg = css("--fg"), grid = css("--border");

  const hs = f.horizons || [];
  const x = hs.map((h) => hhmm(h.valid_time));
  const qs = (f.quantiles || []).map(String);
  const at = (h, q) => (h.quantiles && h.quantiles[q] !== undefined) ? h.quantiles[q] : null;

  // anchor the fan at the last observation so the bands start from "now"
  const obs = (f.observed_history || []).filter((o) => o.ghi !== null);
  const obsX = obs.map((o) => hhmm(o.timestamp));
  const obsY = obs.map((o) => o.ghi);
  const anchorX = obsX.length ? obsX[obsX.length - 1] : hhmm(f.issue_time);
  const anchorY = obsY.length ? obsY[obsY.length - 1] : null;

  const pre = (arr) => (anchorY === null ? arr : [anchorY].concat(arr));
  const preX = (arr) => (anchorY === null ? arr : [anchorX].concat(arr));

  const traces = [];
  const lo80 = hs.map((h) => at(h, qs[0])), hi80 = hs.map((h) => at(h, qs[qs.length - 1]));
  const lo50 = hs.map((h) => at(h, qs[1])), hi50 = hs.map((h) => at(h, qs[qs.length - 2]));

  traces.push({ x: preX(x), y: pre(hi80), name: "90th pct", mode: "lines",
    line: { width: 0, color: acc }, hoverinfo: "skip", showlegend: false });
  traces.push({ x: preX(x), y: pre(lo80), name: "10–90% band", mode: "lines",
    line: { width: 0, color: acc }, fill: "tonexty", fillcolor: b80,
    hovertemplate: "10th pct %{y:.0f} W/m²<extra></extra>" });
  traces.push({ x: preX(x), y: pre(hi50), name: "75th pct", mode: "lines",
    line: { width: 0, color: acc }, hoverinfo: "skip", showlegend: false });
  traces.push({ x: preX(x), y: pre(lo50), name: "25–75% band", mode: "lines",
    line: { width: 0, color: acc }, fill: "tonexty", fillcolor: b50,
    fillpattern: { shape: "/", size: 5, solidity: 0.25 },
    hovertemplate: "25th pct %{y:.0f} W/m²<extra></extra>" });
  traces.push({ x: preX(x), y: pre(hs.map((h) => h.median)), name: "median forecast",
    mode: "lines+markers", line: { color: acc, width: 3 }, marker: { size: 7 },
    hovertemplate: "median %{y:.0f} W/m²<extra></extra>" });
  traces.push({ x: preX(x), y: pre(hs.map((h) => h.clearsky_ghi)), name: "clear-sky GHI",
    mode: "lines", line: { color: acc2, width: 2, dash: "dash" },
    hovertemplate: "clear-sky %{y:.0f} W/m²<extra></extra>" });
  if (obs.length) {
    traces.unshift({ x: obsX, y: obsY, name: "observed (ERA5 reanalysis)",
      mode: "lines", line: { color: fg, width: 2.5 },
      hovertemplate: "observed %{y:.0f} W/m²<extra></extra>" });
  }
  const actuals = hs.filter((h) => h.actual_ghi !== null && h.actual_ghi !== undefined);
  if (actuals.length) {
    traces.push({ x: actuals.map((h) => hhmm(h.valid_time)),
      y: actuals.map((h) => h.actual_ghi), name: "outcome (ERA5)", mode: "markers",
      marker: { color: fg, size: 10, symbol: "x", line: { width: 2 } },
      hovertemplate: "outcome %{y:.0f} W/m²<extra></extra>" });
  }

  Plotly.react("fan", traces, {
    margin: { l: 62, r: 14, t: 10, b: 46 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: fg, size: 13 },
    xaxis: { title: { text: "Valid time (SGT)" }, gridcolor: grid, type: "category" },
    yaxis: { title: { text: "GHI (W/m²)" }, gridcolor: grid, rangemode: "tozero" },
    legend: { orientation: "h", y: -0.22, font: { size: 12 } },
    hovermode: "x unified",
  }, { displayModeBar: false, responsive: true });

  const note = $("kt-note");
  if (f.kt_now !== null && f.kt_now < 0.4) {
    note.classList.remove("hidden");
    note.innerHTML = `<strong>Currently overcast (k<sub>t</sub> = ${fmt(f.kt_now, 2)}).</strong>
      In these conditions the 2-hour outcome is close to <strong>bimodal</strong> &mdash; it
      either stays overcast (k<sub>t</sub> &asymp; 0.3) or clears (k<sub>t</sub> &asymp; 0.75).
      The wide band reflects real uncertainty, not model weakness.`;
  } else { note.classList.add("hidden"); }
}

/* ------------------------------------------------------------- 2. RAIN MAP */
/* Basemap tiles.
   CARTO's basemaps.cartocdn.com now requires an API key - it still answers HTTP 200
   but stamps "API key required" across the tile, so the failure looks like a data
   problem rather than an auth one. Esri's Canvas Gray needs no key, is muted enough
   that the rain field reads clearly on top of it, and ships light AND dark variants
   that track the dashboard theme. OpenStreetMap is the fallback.
   Note Esri orders its path {z}/{y}/{x}, not the usual {z}/{x}/{y}. */
const BASEMAPS = {
  light: {
    url: "https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri",
  },
  dark: {
    url: "https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri",
  },
  fallback: {
    url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: "&copy; OpenStreetMap contributors",
  },
};

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
}

function setBasemap() {
  if (!state.map) return;
  if (state.baseLayer) state.map.removeLayer(state.baseLayer);
  const spec = BASEMAPS[currentTheme()];
  let failed = 0;
  state.baseLayer = L.tileLayer(spec.url, {
    attribution: spec.attribution, maxZoom: 18, crossOrigin: true,
  });
  // If the provider starts refusing tiles, fall back rather than showing a
  // half-blank map. The data overlays are vectors and render regardless.
  state.baseLayer.on("tileerror", () => {
    if (++failed === 4 && !state.usedFallback) {
      state.usedFallback = true;
      state.map.removeLayer(state.baseLayer);
      state.baseLayer = L.tileLayer(BASEMAPS.fallback.url, {
        attribution: BASEMAPS.fallback.attribution, maxZoom: 18,
      }).addTo(state.map);
    }
  });
  state.baseLayer.addTo(state.map);
  state.baseLayer.bringToBack();
}

function initMap() {
  if (state.map) return;
  state.map = L.map("map", { zoomControl: true, attributionControl: true })
               .setView([1.3521, 103.8198], 11);
  setBasemap();
}

function rainColor(v) {
  if (v <= 0.001) return null;
  if (v < 0.05) return "#c7e9ff";
  if (v < 0.15) return "#7fc4f5";
  if (v < 0.4) return "#3f8fd8";
  if (v < 1.0) return "#2b60b5";
  return "#7b2bb5";
}

function drawMap(rg) {
  initMap();
  ["gridLayer", "maskLayer", "stationLayer", "motionLayer"].forEach((k) => {
    if (state[k]) { state.map.removeLayer(state[k]); state[k] = null; }
  });
  if (rg.degraded || !rg.grid || !rg.grid.length) {
    $("map-meta").textContent = "Rain field unavailable: " + (rg.error || "no data");
    return;
  }
  const b = rg.extent, ny = rg.ny, nx = rg.nx;
  const dLat = (b.lat_max - b.lat_min) / ny, dLon = (b.lon_max - b.lon_min) / nx;

  const gridG = L.layerGroup(), maskG = L.layerGroup();
  for (let r = 0; r < ny; r++) {
    // row 0 of the returned array is NORTH (server already flipped it)
    const latHi = b.lat_max - r * dLat, latLo = latHi - dLat;
    for (let c = 0; c < nx; c++) {
      const lonLo = b.lon_min + c * dLon, lonHi = lonLo + dLon;
      const bounds = [[latLo, lonLo], [latHi, lonHi]];
      if (rg.mask[r] && rg.mask[r][c] === 0) {
        maskG.addLayer(L.rectangle(bounds, { stroke: false, fillColor: "#888",
          fillOpacity: 0.35, className: "mask-cell", interactive: false }));
        continue;
      }
      const col = rainColor(rg.grid[r][c]);
      if (col) {
        gridG.addLayer(L.rectangle(bounds, { stroke: false, fillColor: col,
          fillOpacity: 0.6, interactive: false }));
      }
    }
  }
  state.gridLayer = gridG; state.maskLayer = maskG;
  if ($("layer-grid").checked) gridG.addTo(state.map);
  if ($("layer-mask").checked) maskG.addTo(state.map);

  const stG = L.layerGroup();
  (rg.stations || []).forEach((s) => {
    if (s.lat === null || s.lon === null) return;
    const v = s.value || 0;
    const col = rainColor(v) || "#9aa7b5";
    stG.addLayer(L.circleMarker([s.lat, s.lon], {
      radius: 4 + Math.min(8, v * 6), color: "#fff", weight: 1.2,
      fillColor: col, fillOpacity: 0.9,
    }).bindTooltip(`<strong>${s.name || s.id}</strong><br>${fmt(v, 2)} mm / 5 min`));
  });
  state.stationLayer = stG;
  if ($("layer-stations").checked) stG.addTo(state.map);

  // motion arrow feeding the physics gate
  const mv = rg.motion || {};
  if (mv.valid) {
    const cLat = (b.lat_min + b.lat_max) / 2, cLon = (b.lon_min + b.lon_max) / 2;
    const arrow = L.polyline([[cLat, cLon],
      [cLat + mv.vy * dLat * 3, cLon + mv.vx * dLon * 3]],
      { color: css("--accent-2"), weight: 4, opacity: 0.95 });
    state.motionLayer = L.layerGroup([arrow]).addTo(state.map);
    $("motion-readout").textContent = `motion vector: (${fmt(mv.vx, 1)}, ${fmt(mv.vy, 1)}) cells`;
  } else {
    $("motion-readout").innerHTML =
      `motion vector: <span class="flag-warn">undefined</span> (field too dry to cross-correlate)`;
  }

  const skipped = rg.newest_frame_skipped
    ? ` · newest frame ${hhmm(rg.newest_frame_skipped)} skipped (partially reported)` : "";
  $("map-meta").innerHTML =
    `${rg.n_stations_reporting} gauges reporting · reading ${hhmm(rg.latest_reading)}${SGT}`
    + ` · age ${fmt((rg.data_age_minutes || {}).nea, 0)} min · wet cells `
    + `${fmt(100 * (rg.wet_fraction || 0), 1)}% · grid ${nx}×${ny} `
    + `(${fmt(rg.cell_km[0], 2)}×${fmt(rg.cell_km[1], 2)} km)${skipped}`;
}
["layer-grid", "layer-stations", "layer-mask"].forEach((id) => {
  document.addEventListener("change", (e) => {
    if (e.target.id !== id) return;
    const key = { "layer-grid": "gridLayer", "layer-stations": "stationLayer",
                  "layer-mask": "maskLayer" }[id];
    if (!state[key]) return;
    if (e.target.checked) state[key].addTo(state.map); else state.map.removeLayer(state[key]);
  });
});

/* ------------------------------------------ 2b. TIME SLIDER OVER RAIN FIELDS */
function paintFrame(frame, mask, meta) {
  ["gridLayer"].forEach((k) => {
    if (state[k]) { state.map.removeLayer(state[k]); state[k] = null; }
  });
  const b = meta.extent, ny = meta.ny, nx = meta.nx;
  const dLat = (b.lat_max - b.lat_min) / ny, dLon = (b.lon_max - b.lon_min) / nx;
  const g = L.layerGroup();
  for (let r = 0; r < ny; r++) {
    const latHi = b.lat_max - r * dLat, latLo = latHi - dLat;
    for (let c = 0; c < nx; c++) {
      if (mask && mask[r] && mask[r][c] === 0) continue;
      const col = rainColor(frame.grid[r][c]);
      if (!col) continue;
      const lonLo = b.lon_min + c * dLon;
      g.addLayer(L.rectangle([[latLo, lonLo], [latHi, lonLo + dLon]],
        { stroke: false, fillColor: col, fillOpacity: 0.6, interactive: false }));
    }
  }
  state.gridLayer = g;
  if ($("layer-grid").checked) g.addTo(state.map);
  $("slider-label").textContent =
    `${hhmm(frame.timestamp)}${SGT} · ${fmt(frame.age_minutes, 0)} min ago`;
}

async function loadSeries() {
  const sl = $("time-slider");
  try {
    const s = await getJSON("/api/raingrid/series?minutes=180");
    state.series = s;
    if (s.degraded || !s.n) {
      sl.disabled = true; $("play").disabled = true;
      if (STATIC) {
        const row = document.querySelector(".slider-row");
        if (row) row.hidden = true;
        $("map-meta").textContent +=
          " · time scrubber needs the live API (run uvicorn locally)";
      } else {
        $("slider-label").textContent = "series unavailable";
      }
      return;
    }
    sl.disabled = false; $("play").disabled = false;
    sl.min = 0; sl.max = s.n - 1; sl.value = s.n - 1;
    const wet = s.frames.filter((f) => f.wet_fraction > 0.001).length;
    $("slider-label").textContent =
      `${hhmm(s.frames[s.n - 1].timestamp)}${SGT} · latest`;
    sl.oninput = () => paintFrame(s.frames[+sl.value], s.mask, s);
    $("play").onclick = () => {
      if (state.playing) { clearInterval(state.playing); state.playing = null;
        $("play").innerHTML = "&#9654; Play"; return; }
      $("play").innerHTML = "&#10074;&#10074; Pause";
      let i = 0;
      state.playing = setInterval(() => {
        sl.value = i; paintFrame(s.frames[i], s.mask, s);
        i = (i + 1) % s.n;
      }, 220);
    };
    const note = ` · ${wet}/${s.n} of the last 3h frames have any rain at all`;
    $("map-meta").innerHTML += note;
  } catch (e) {
    sl.disabled = true; $("play").disabled = true;
    $("slider-label").textContent = "series failed";
  }
}

/* ------------------------------------------------- 3+4. VERIFICATION TABLES */
function table(headers, rows) {
  let h = "<table><thead><tr>" + headers.map((x) => `<th>${x}</th>`).join("") +
          "</tr></thead><tbody>";
  h += rows.map((r) => "<tr>" + r.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("");
  return h + "</tbody></table>";
}

function skillClass(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  return v > 0.02 ? "flag-good" : (v < -0.02 ? "flag-bad" : "flag-warn");
}
function spreadCell(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const pct = (100 * v).toFixed(0) + "%";
  if (v < 0.70) return `<span class="flag-bad">&#9660; ${pct}</span>`;
  if (v < 0.85 || v > 1.35) return `<span class="flag-warn">&#9679; ${pct}</span>`;
  return `<span class="flag-good">&#9679; ${pct}</span>`;
}

function renderVerification(v) {
  const box = $("verification"), cal = $("calibration");
  if (v.degraded || !Object.keys(v.tracks || {}).length) {
    box.innerHTML = `<p class="meta">No evaluation report yet. Run
      <code>python evaluate.py --track A</code>.</p>`;
    cal.innerHTML = box.innerHTML;
    return;
  }
  const track = v.tracks.A ? "A" : Object.keys(v.tracks)[0];
  const t = v.tracks[track];
  const rows = [];
  const entries = [];
  Object.entries(t.baselines || {}).forEach(([name, perH]) => entries.push([name, perH, "baseline"]));
  (t.models || []).forEach((m) => entries.push([`${m.variant} (seed ${m.seed})`, m.horizons, "model"]));

  entries.forEach(([name, perH]) => {
    Object.entries(perH || {}).forEach(([hz, m]) => {
      rows.push([name, hz, `${m.n} <span class="n-badge">rows</span>`,
        fmt(m.mae, 1), fmt(m.rmse, 1), fmt(m.pinball, 2), fmt(m.crps, 2),
        fmt(m.coverage, 3),
        `<span class="${skillClass(m.skill_vs_ref)}">${fmt(m.skill_vs_ref, 3)}</span>`]);
    });
  });
  box.innerHTML = table(
    ["model", "horizon", "n", "MAE (W/m²)", "RMSE", "pinball", "CRPS",
     `coverage (nominal ${fmt(v.nominal_interval, 2)})`, "skill vs smart persistence"],
    rows) + `<p class="meta">${v.note}</p>`;

  const crows = [];
  entries.forEach(([name, perH]) => {
    Object.entries(perH || {}).forEach(([hz, m]) => {
      (m.stratified || []).forEach((s) => {
        if (!s.n) return;
        crows.push([name, hz, s.regime, `${s.n} <span class="n-badge">rows</span>`,
          fmt(s.coverage, 3), spreadCell(s.spread_over_reality),
          fmt(s.mae, 1), fmt(s.bias, 1)]);
      });
    });
  });
  cal.innerHTML = table(
    ["model", "horizon", "regime", "n", "coverage", "spread / reality (k_t)",
     "MAE (W/m²)", "bias"], crows) +
    `<p class="meta">&#9660; below 70% = hedging toward the mean. &#9679; near 100% = well
     dispersed. Coverage is compared against the nominal
     ${fmt(v.nominal_interval, 2)} band implied by the quantiles
     (0.10–0.90 is an 80% interval, not 90%).</p>`;
}

/* --------------------------------------------------- 5. MODEL / DATA STATUS */
function renderModel(mi) {
  const box = $("model-info");
  if (mi.degraded) {
    box.innerHTML = `<p class="meta">Model unavailable: ${mi.error || "unknown"}</p>`;
    renderLimitations(mi.known_limitations || []);
    return;
  }
  const rows = [
    ["Checkpoint", mi.checkpoint], ["Variant", mi.variant],
    ["Track", mi.track], ["Seed", mi.seed],
    ["Trained at", (mi.trained_at || "").replace("T", " ") + SGT],
    ["Train / val / test", `${mi.n_train} / ${mi.n_val} / ${mi.n_test}`],
    ["Parameters", (mi.params && mi.params.total) ? mi.params.total.toLocaleString() : "—"],
    ["Quantiles", (mi.quantiles || []).join(", ")],
    ["Central band", `${fmt(mi.nominal_interval, 2)} nominal`],
    ["Horizons", (mi.horizons || []).map((h) => "t+" + h + "h").join(", ")],
    ["NWP models", (mi.nwp_models || []).join(", ")],
    ["Target product", mi.target_product],
    ["Daylight window", `${mi.daylight_window[0]}:00–${mi.daylight_window[1]}:00 SGT`],
    ["Features", (mi.features || []).join(", ")],
  ];
  box.innerHTML = table(["field", "value"], rows.map(([k, v]) => [k, v === null || v === undefined ? "—" : v]));
  renderLimitations(mi.known_limitations || []);
}

function renderLimitations(list) {
  $("limitations").innerHTML = list.length
    ? "<ul>" + list.map((l) => `<li>${l}</li>`).join("") + "</ul>"
    : `<p class="meta">See README.md &rarr; Known limitations.</p>`;
}

function renderHealth(h) {
  const rows = Object.entries(h.sources || {}).map(([k, s]) => [
    k.toUpperCase(),
    s.error ? `<span class="flag-bad">&#9888; ${s.error}</span>`
            : (s.last ? dayTime(s.last) : (s.last_day || "—")),
    s.age_minutes !== undefined && s.age_minutes !== null ? fmt(s.age_minutes, 0) + " min" :
      (s.days !== undefined ? `${s.complete_days}/${s.days} complete days` : "—"),
    s.threshold ? fmt(s.threshold, 0) + " min" : "—",
    s.error ? "<span class='flag-bad'>&#9888; missing</span>"
            : (s.age_minutes !== null && s.age_minutes !== undefined && s.threshold &&
               s.age_minutes > s.threshold
                ? "<span class='flag-warn'>&#9679; stale</span>"
                : "<span class='flag-good'>&#9679; ok</span>"),
  ]);
  $("health").innerHTML = table(["source", "latest", "age", "threshold", "status"], rows);
  h.warnings.forEach((w) => banner(w, "err"));
  if (h.replay_mode) {
    banner(`<strong>Replay mode.</strong> Serving a stored day` +
      (h.replay_date ? ` (${h.replay_date})` : "") +
      ` through the exact training pipeline &mdash; no live network needed.`, "ok");
  }
}

/* --------------------------------------------------------- 6. DIAGNOSTICS */
function renderDiagnostics(f) {
  const d = f.diagnostics || {};
  const alpha = d.gate_alpha;
  const rows = [
    ["Gate α (this forecast)", alpha === null || alpha === undefined ? "— (variant has no gate)" : fmt(alpha, 3)],
    ["Lookback real steps", `${d.lookback_real_steps} / ${d.lookback_required} (${fmt(100 * d.lookback_real_fraction, 0)}% real)`],
    ["Gauges reporting", d.n_stations_reporting],
    ["Motion vector", d.motion_valid ? `(${fmt(d.motion_vx, 1)}, ${fmt(d.motion_vy, 1)}) cells` :
      `<span class="flag-warn">undefined &mdash; field too dry</span>`],
    ["Grid wet fraction", fmt(100 * d.grid_wet_fraction, 2) + "%"],
    ["Grid mask coverage", fmt(100 * d.grid_mask_fraction, 1) + "%"],
    ["Observation product", f.observation_product],
    ["Target product", f.target_product],
    ["Mode", f.mode],
  ];
  $("diagnostics").innerHTML = table(["diagnostic", "value"], rows);
  if (d.lookback_real_fraction !== undefined && d.lookback_real_fraction < 0.5) {
    banner(`Lookback window is only ${fmt(100 * d.lookback_real_fraction, 0)}% real data —
            the rest is mean-padded. Treat this forecast with caution.`, "err");
  }
}

/* --------------------------------------------------------------- summary */
function renderSummary(f) {
  const hs = f.horizons || [];
  if (!hs.length) { $("summary").textContent = "No forecast available."; return; }
  const qs = (f.quantiles || []).map(String);
  const mid = hs[Math.min(1, hs.length - 1)];
  const lo = mid.quantiles[qs[0]], hi = mid.quantiles[qs[qs.length - 1]];
  const kt = f.kt_now;
  const sky = kt === null ? "unknown sky" :
    (kt < 0.4 ? "overcast" : kt < 0.7 ? "partly cloudy" : "mostly clear");
  $("summary").innerHTML =
    `Next ${hs.length} hours: <strong>${sky}</strong>, irradiance likely
     <strong>${fmt(lo, 0)}–${fmt(hi, 0)} W/m²</strong> at
     <strong>${hhmm(mid.valid_time)}${SGT}</strong>
     (${fmt(100 * f.nominal_interval, 0)}% interval).`;
  $("issue-line").innerHTML =
    `Issued ${dayTime(f.issue_time)} · mode <strong>${f.mode}</strong>
     · model ${f.model.variant} / track ${f.model.track}
     · k<sub>t</sub> now ${fmt(kt, 2)}
     · forecast age ${fmt((f.data_age_minutes || {}).forecast, 0)} min`;

  if (f.outside_training_window) {
    banner(`<strong>Outside the training window
      (${document.title ? "08:00–17:00" : ""} SGT).</strong> Outputs here are dominated
      by the clear-sky clamp rather than by the model.`, "err");
  }
  if (f.stale) banner("This forecast is stale relative to its refresh cadence.", "err");
}

/* -------------------------------------------------------------- bootstrap */
// In the static build (GitHub Pages) there is no server: the scheduled job wrote
// one JSON file per endpoint. Map /api/... onto those files. The FastAPI path is
// untouched, so local dev keeps live refresh and the 3-hour scrubber.
const STATIC = (typeof window !== "undefined" && window.STATIC_BUILD === true);

function staticPath(url) {
  const [path, qs] = url.split("?");
  const q = new URLSearchParams(qs || "");
  if (path === "/api/verification/rolling") {
    return `data/rolling_${q.get("window") || "7d"}_${q.get("source") || "era5"}.json`;
  }
  const map = {
    "/api/forecast/latest": "forecast_latest",
    "/api/health": "health",
    "/api/model/info": "model_info",
    "/api/verification": "verification",
    "/api/raingrid/latest": "raingrid_latest",
  };
  // /api/raingrid/series is deliberately NOT built - 36 live frames per request is
  // too much to publish every cycle. Returning null makes getJSON degrade, and
  // loadSeries already disables the scrubber on a degraded payload.
  return map[path] ? `data/${map[path]}.json` : null;
}

async function getJSON(url) {
  if (STATIC) {
    const p = staticPath(url);
    if (!p) return { degraded: true, n: 0, error: "not available in the static build" };
    const r = await fetch(p);
    if (!r.ok) return { degraded: true, n: 0, error: `${p}: HTTP ${r.status}` };
    return r.json();
  }
  const r = await fetch(url);
  return r.json();
}

// ------------------------------------------------- live rolling verification
function renderRolling(v) {
  const badge = $("rv-badge");
  if (v.provisional) {
    badge.className = "pill pill-warn";
    badge.textContent = "PROVISIONAL — not ERA5";
  } else {
    badge.className = "pill pill-good";
    badge.textContent = "ERA5 — final";
  }

  if (v.degraded || !v.horizons || Object.keys(v.horizons).length === 0) {
    $("rolling").innerHTML = `<p class="hint">${v.error || "No scored forecasts yet."}</p>`;
    Plotly.purge("rv-plot");
    $("rv-store").textContent = "";
    return;
  }

  const rows = Object.entries(v.horizons).map(([hz, r]) => [
    hz, r.n,
    `${r.n_live} live / ${r.n_replay} replay`,
    fmt(r.mae, 1), fmt(r.pinball, 2), fmt(r.crps, 2),
    fmt(r.coverage, 3), spreadCell(r.dispersion_spread_over_reality),
  ]);
  $("rolling").innerHTML = table(
    ["Horizon", "n", "Source of rows", "MAE", "Pinball", "CRPS",
     `Coverage (nom ${fmt(v.horizons[Object.keys(v.horizons)[0]].nominal_level, 2)})`,
     "Spread/reality"],
    rows);

  const st = v.store || {};
  $("rv-store").innerHTML =
    `Store holds <strong>${st.forecasts || 0}</strong> forecasts ` +
    `(${st.live || 0} issued live, ${st.replay || 0} backfilled replay), ` +
    `last issued ${st.last_issue || "—"}${SGT}. ` +
    `Backfilled rows are a backtest, not a live track record. ` +
    `A short window is dominated by recent weather, not model quality.`;

  drawRollingPlot(v);
}

function drawRollingPlot(v) {
  const pts = v.timeline || [];
  if (!pts.length) { Plotly.purge("rv-plot"); return; }
  const fg = css("--fg"), grid = css("--border");
  const acc = css("--accent"), b80 = css("--band-80");
  const x = pts.map((p) => dayTime(p.valid_time));

  const traces = [
    { x, y: pts.map((p) => p.lower), type: "scatter", mode: "lines",
      line: { width: 0, color: acc }, hoverinfo: "skip", showlegend: false },
    { x, y: pts.map((p) => p.upper), type: "scatter", mode: "lines",
      line: { width: 0, color: acc }, fill: "tonexty", fillcolor: b80,
      name: `${fmt(100 * (v.horizons[Object.keys(v.horizons)[0]] || {}).nominal_level, 0)}% interval`,
      hoverinfo: "skip" },
    { x, y: pts.map((p) => p.median), type: "scatter", mode: "lines",
      line: { color: acc, width: 2 }, name: "Forecast median" },
    { x, y: pts.map((p) => p.actual), type: "scatter", mode: "lines",
      line: { color: fg, width: 2, dash: "dot" },
      name: v.provisional ? "Analysis (provisional)" : "ERA5 actual" },
  ];

  Plotly.react("rv-plot", traces, {
    margin: { l: 55, r: 15, t: 10, b: 70 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: fg, size: 13 },
    xaxis: { title: { text: "Valid time (SGT)" }, gridcolor: grid, type: "category",
             nticks: 10 },
    yaxis: { title: { text: "GHI (W/m²)" }, gridcolor: grid, rangemode: "tozero" },
    legend: { orientation: "h", y: -0.3 },
    height: 320,
  }, { displayModeBar: false, responsive: true });
}

function loadRolling() {
  const w = $("rv-window").value, src = $("rv-source").value, h = $("rv-horizon").value;
  return getJSON(`/api/verification/rolling?window=${w}&source=${src}&horizon=${h}`)
    .then((v) => {
      if (v.timelines) v.timeline = v.timelines[String(h)] || [];
      renderRolling(v);
    })
    .catch((e) => { $("rolling").innerHTML = `<p class="hint">Rolling verification failed: ${e}</p>`; });
}

async function loadAll(refresh = false) {
  $("banners").innerHTML = "";
  try {
    const f = await getJSON("/api/forecast/latest" + (refresh ? "?refresh=true" : ""));
    state.forecast = f;
    if (f.degraded) {
      banner(`No forecast: ${f.error}`, "err");
      $("summary").textContent = "No forecast available — train a model first.";
    } else {
      renderSummary(f); drawFan(f); renderDiagnostics(f);
    }
  } catch (e) { banner("Forecast request failed: " + e, "err"); }

  getJSON("/api/health").then(renderHealth).catch(() => {});
  getJSON("/api/model/info").then(renderModel).catch(() => {});
  getJSON("/api/verification").then(renderVerification).catch(() => {});
  loadRolling();
  getJSON("/api/raingrid/latest").then((rg) => { drawMap(rg); loadSeries(); })
    .catch((e) => { $("map-meta").textContent = "Rain field failed: " + e; });
}

["rv-window", "rv-source", "rv-horizon"].forEach((id) => { $(id).onchange = loadRolling; });
if (STATIC) {
  const b = $("refresh");
  b.disabled = true;
  b.title = "Static build - the page is rebuilt by the scheduled job";
} else {
  $("refresh").onclick = () => loadAll(true);
}
applyTheme(localStorage.getItem("theme") || "dark");
loadAll();
// The static build changes only when the scheduled job republishes it, so a reload
// every 15 min is enough; the live API is worth polling more often.
setInterval(() => loadAll(false), (STATIC ? 15 : 5) * 60 * 1000);
