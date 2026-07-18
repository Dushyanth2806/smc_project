const $ = (id) => document.getElementById(id);

function setStatus(el, msg, cls) {
  el.textContent = msg;
  el.className = "status" + (cls ? " " + cls : "");
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

async function postForm(url, formData) {
  const res = await fetch(url, { method: "POST", body: formData });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function renderMetrics(metrics) {
  const el = $("train-metrics");
  const rows = [];
  rows.push(`<div class="forecast-grid">
    <div class="cell"><div class="label">Rows (train/test)</div><div class="value">${metrics.rows_train} / ${metrics.rows_test}</div></div>
    <div class="cell"><div class="label">Accuracy</div><div class="value">${metrics.accuracy !== undefined ? (metrics.accuracy * 100).toFixed(1) + "%" : "n/a"}</div></div>
    <div class="cell"><div class="label">Event F1</div><div class="value">${metrics.event_detection ? (metrics.event_detection.f1 * 100).toFixed(1) + "%" : "n/a"}</div></div>
    <div class="cell"><div class="label">Horizon</div><div class="value">${metrics.horizon_bars} bar(s)</div></div>
  </div>`);

  if (metrics.classification_report) {
    let table = `<table class="metrics-table"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>`;
    for (const [name, vals] of Object.entries(metrics.classification_report)) {
      table += `<tr><td>${name}</td><td>${(vals.precision * 100).toFixed(1)}%</td><td>${(vals.recall * 100).toFixed(1)}%</td><td>${(vals["f1-score"] * 100).toFixed(1)}%</td><td>${vals.support}</td></tr>`;
    }
    table += `</tbody></table>`;
    rows.push(table);
  }

  if (metrics.top_features) {
    let table = `<h3 style="font-size:0.9rem;margin-bottom:0.3rem;">Top features</h3><table class="metrics-table"><thead><tr><th>Feature</th><th>Importance</th></tr></thead><tbody>`;
    for (const f of metrics.top_features) {
      table += `<tr><td>${f.feature}</td><td>${f.importance}</td></tr>`;
    }
    table += `</tbody></table>`;
    rows.push(table);
  }

  el.innerHTML = rows.join("");
  $("train-results").classList.remove("hidden");
}

function renderForecast(forecast) {
  if (!forecast) {
    $("forecast-card").classList.add("hidden");
    return;
  }
  const probs = Object.entries(forecast.probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([name, p]) => `<div class="cell"><div class="label">${name}</div><div class="value">${(p * 100).toFixed(1)}%</div></div>`)
    .join("");

  $("forecast-content").innerHTML = `
    <p>Based on bar closing at <strong>${new Date(forecast.based_on_bar_time).toLocaleString()}</strong>,
    forecasting <strong>${forecast.horizon_bars}</strong> bar(s) ahead:</p>
    <p><span class="badge ${forecast.predicted_label}">${forecast.predicted_label.replace("_", " ")}</span>
    &nbsp; confidence ${(forecast.confidence * 100).toFixed(1)}%</p>
    <div class="forecast-grid">${probs}</div>
  `;
  $("forecast-card").classList.remove("hidden");
}

let lastPredictionData = null;
const chartToggles = { bos: true, choch: true, swing: true, internal: true, pred: true, ob: true, fvg: true };

function eventArrowColor(direction) {
  return direction === "bullish" ? "#22c55e" : "#ef4444";
}

// Draws EVERY BOS/CHoCH as an arrow FROM the broken pivot point TO the bar
// that broke it (the standard SMC visualization), rather than a bare
// marker at a single point.
//
// Color always encodes DIRECTION (green = bullish break, red = bearish
// break) - it never encodes BOS vs CHoCH. A green CHoCH is normal: it
// means "price had been trending down, then broke UP through a pivot",
// i.e. a bullish reversal signal. To tell BOS and CHoCH apart at a
// glance without relying on tiny text, BOS uses a solid filled arrowhead
// and CHoCH uses a hollow/open arrowhead; the text label ("BOS"/"CHoCH")
// is always drawn too, just smaller for the more frequent internal
// (5-bar lookback) breaks than for swing (50-bar lookback) ones.
function buildBreakAnnotations(events) {
  return events
    .filter((e) => chartToggles[e.type === "BOS" ? "bos" : "choch"])
    .filter((e) => chartToggles[e.scope])
    .map((e) => {
      const color = eventArrowColor(e.direction);
      const isSwing = e.scope === "swing";
      const isBOS = e.type === "BOS";
      return {
        x: e.breakout_time, y: e.pivot_level,
        ax: e.pivot_time, ay: e.pivot_level,
        xref: "x", yref: "y", axref: "x", ayref: "y",
        showarrow: true,
        arrowhead: isBOS ? 3 : 6, // 3 = solid filled triangle (BOS), 6 = open/hollow (CHoCH)
        arrowsize: isSwing ? 1.1 : 0.85,
        arrowwidth: isSwing ? 1.8 : 1.1,
        opacity: isSwing ? 1 : 0.85,
        arrowcolor: color,
        text: e.type,
        font: { color, size: isSwing ? 11 : 9 },
        bgcolor: isSwing ? "rgba(15,17,21,0.8)" : "rgba(15,17,21,0.55)",
        bordercolor: isSwing ? color : "rgba(0,0,0,0)",
        borderwidth: isSwing ? 1 : 0,
        borderpad: 2,
        yshift: e.direction === "bullish" ? (isSwing ? 12 : 6) : (isSwing ? -12 : -6),
        standoff: 2,
      };
    });
}

// Order blocks and FVGs are drawn as shaded rectangles (the standard SMC
// visual) rather than arrows, since they're zones/ranges, not single
// points. Both are ground truth from the rule engine (order_blocks.py /
// fvg.py) - not the ML model - same as the BOS/CHoCH arrows.
function buildOrderBlockShapes(orderBlocks) {
  const shapes = [];
  const annotations = [];
  orderBlocks
    .filter((ob) => chartToggles.ob && chartToggles[ob.scope])
    .forEach((ob) => {
      const color = ob.bias === "bullish" ? "34,197,94" : "239,68,68";
      shapes.push({
        type: "rect", xref: "x", yref: "y",
        x0: ob.time, x1: ob.end_time, y0: ob.bottom, y1: ob.top,
        fillcolor: `rgba(${color},${ob.mitigated ? 0.06 : 0.16})`,
        line: { color: `rgba(${color},${ob.mitigated ? 0.3 : 0.7})`, width: 1 },
      });
      annotations.push({
        x: ob.time, y: ob.top, xref: "x", yref: "y",
        text: `OB${ob.mitigated ? " (mitigated)" : ""}`,
        showarrow: false, xanchor: "left", yanchor: "bottom",
        font: { color: `rgba(${color},0.9)`, size: 9 },
        bgcolor: "rgba(15,17,21,0.6)",
      });
    });
  return { shapes, annotations };
}

function buildFVGShapes(fvgs) {
  const shapes = [];
  const annotations = [];
  if (!chartToggles.fvg) return { shapes, annotations };
  fvgs.forEach((g) => {
    const color = g.bias === "bullish" ? "34,197,94" : "239,68,68";
    const top = Math.max(g.top, g.bottom);
    const bottom = Math.min(g.top, g.bottom);
    shapes.push({
      type: "rect", xref: "x", yref: "y",
      x0: g.left_time, x1: g.right_time, y0: bottom, y1: top,
      fillcolor: `rgba(${color},${g.mitigated ? 0.05 : 0.12})`,
      line: { color: `rgba(${color},${g.mitigated ? 0.25 : 0.55})`, width: 1, dash: "dot" },
    });
    annotations.push({
      x: g.left_time, y: top, xref: "x", yref: "y",
      text: `FVG${g.mitigated ? " (mitigated)" : ""}`,
      showarrow: false, xanchor: "left", yanchor: "top",
      font: { color: `rgba(${color},0.85)`, size: 9 },
      bgcolor: "rgba(15,17,21,0.6)",
    });
  });
  return { shapes, annotations };
}

function buildChart() {
  const data = lastPredictionData;
  if (!data) return;
  const c = data.candles;

  const candleTrace = {
    type: "candlestick",
    x: c.time, open: c.open, high: c.high, low: c.low, close: c.close,
    name: "Price",
    increasing: { line: { color: "#22c55e" } },
    decreasing: { line: { color: "#ef4444" } },
  };

  const traces = [candleTrace];

  if (chartToggles.pred) {
    const predEvents = (data.predictions || []).filter((p) => p.predicted_label !== "NONE");
    const priceByTime = {};
    c.time.forEach((t, i) => { priceByTime[t] = c.close[i]; });
    traces.push({
      type: "scatter", mode: "markers", name: "Predicted next-bar event",
      x: predEvents.map((p) => p.time),
      y: predEvents.map((p) => priceByTime[p.time]),
      text: predEvents.map((p) => `Predicted: ${p.predicted_label} (${(p.confidence * 100).toFixed(0)}%)`),
      marker: { color: "#4f8cff", size: 7, symbol: "circle-open", line: { width: 2 } },
    });
  }

  const breakAnnotations = buildBreakAnnotations(data.actual_events || []);
  const obResult = buildOrderBlockShapes(data.order_blocks || []);
  const fvgResult = buildFVGShapes(data.fair_value_gaps || []);

  const layout = {
    paper_bgcolor: "#171a21", plot_bgcolor: "#171a21",
    font: { color: "#e6e9ef" },
    xaxis: { rangeslider: { visible: false }, gridcolor: "#2a2f3a" },
    yaxis: { gridcolor: "#2a2f3a" },
    margin: { t: 30, l: 50, r: 20, b: 40 },
    legend: { orientation: "h" },
    shapes: [...obResult.shapes, ...fvgResult.shapes],
    annotations: [...breakAnnotations, ...obResult.annotations, ...fvgResult.annotations],
  };

  Plotly.react("chart", traces, layout, { responsive: true });
}

function renderChart(data) {
  lastPredictionData = data;
  buildChart();
}

function wireToggle(btnId, key) {
  $(btnId).addEventListener("click", () => {
    chartToggles[key] = !chartToggles[key];
    $(btnId).classList.toggle("active", chartToggles[key]);
    buildChart();
  });
}

wireToggle("toggle-bos", "bos");
wireToggle("toggle-choch", "choch");
wireToggle("toggle-swing", "swing");
wireToggle("toggle-internal", "internal");
wireToggle("toggle-pred", "pred");
wireToggle("toggle-ob", "ob");
wireToggle("toggle-fvg", "fvg");

async function refreshModelStatus() {
  try {
    const res = await fetch("/api/model/status");
    const data = await res.json();
    if (data.trained) renderMetrics(data.metrics);
  } catch (e) { /* ignore on first load */ }
}

$("train-tv-btn").addEventListener("click", async () => {
  const btn = $("train-tv-btn");
  btn.disabled = true;
  setStatus($("train-status"), "Fetching data from TradingView and training...");
  try {
    const data = await postJSON("/api/train/tradingview", {
      symbol: $("train-tv-symbol").value,
      exchange: $("train-tv-exchange").value,
      interval: $("train-tv-interval").value,
      n_bars: Number($("train-tv-bars").value),
      horizon: Number($("train-tv-horizon").value),
    });
    setStatus($("train-status"), "Model trained successfully.", "ok");
    renderMetrics(data.metrics);
  } catch (e) {
    setStatus($("train-status"), e.message, "error");
  } finally {
    btn.disabled = false;
  }
});

$("train-csv-btn").addEventListener("click", async () => {
  const btn = $("train-csv-btn");
  const file = $("train-csv-file").files[0];
  if (!file) { setStatus($("train-status"), "Choose a CSV file first.", "error"); return; }
  btn.disabled = true;
  setStatus($("train-status"), "Uploading CSV and training...");
  try {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("horizon", $("train-csv-horizon").value);
    const data = await postForm("/api/train/csv", fd);
    setStatus($("train-status"), "Model trained successfully.", "ok");
    renderMetrics(data.metrics);
  } catch (e) {
    setStatus($("train-status"), e.message, "error");
  } finally {
    btn.disabled = false;
  }
});

function renderDataTable(data) {
  const c = data.candles;
  const order = $("data-table-order").value;
  const predByTime = {};
  (data.predictions || []).forEach((p) => { predByTime[p.time] = p; });

  const n = c.time.length;
  const indices = [...Array(n).keys()];
  if (order === "desc") indices.reverse();

  let rows = "";
  indices.forEach((i, pos) => {
    const t = c.time[i];
    const pred = predByTime[t];
    const isLatest = order === "desc" ? pos === 0 : i === n - 1;
    const predCell = pred
      ? `<span class="badge ${pred.predicted_label}">${pred.predicted_label.replace("_", " ")}</span> ${(pred.confidence * 100).toFixed(0)}%`
      : "";
    rows += `<tr class="${isLatest ? "latest-row" : ""}">
      <td>${new Date(t).toLocaleString()}</td>
      <td>${c.open[i].toFixed(4)}</td>
      <td>${c.high[i].toFixed(4)}</td>
      <td>${c.low[i].toFixed(4)}</td>
      <td>${c.close[i].toFixed(4)}</td>
      <td>${Math.round(c.volume[i]).toLocaleString()}</td>
      <td>${predCell}</td>
    </tr>`;
  });

  $("data-table-wrap").innerHTML = `<table>
    <thead><tr><th>Time</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th><th>Model prediction (next bar)</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
  $("data-row-count").textContent = n;
}
$("data-table-order").addEventListener("change", () => { if (lastPredictionData) renderDataTable(lastPredictionData); });

function renderAll(data) {
  renderForecast(data.next_bar_forecast);
  renderChart(data);
  renderDataTable(data);
}

async function runTVPredict({ silent } = {}) {
  const btn = $("pred-tv-btn");
  if (!silent) btn.disabled = true;
  if (!silent) setStatus($("pred-status"), "Fetching data from TradingView and predicting...");
  try {
    const data = await postJSON("/api/predict/tradingview", {
      symbol: $("pred-tv-symbol").value,
      exchange: $("pred-tv-exchange").value,
      interval: $("pred-tv-interval").value,
      n_bars: Number($("pred-tv-bars").value),
    });
    setStatus(
      $("pred-status"),
      `Loaded ${data.candles.time.length} bars.` + (silent ? ` (auto-refreshed ${new Date().toLocaleTimeString()})` : ""),
      "ok",
    );
    renderAll(data);
    return true;
  } catch (e) {
    setStatus($("pred-status"), e.message, "error");
    return false;
  } finally {
    if (!silent) btn.disabled = false;
  }
}

$("pred-tv-btn").addEventListener("click", () => runTVPredict());

$("pred-csv-btn").addEventListener("click", async () => {
  const btn = $("pred-csv-btn");
  const file = $("pred-csv-file").files[0];
  if (!file) { setStatus($("pred-status"), "Choose a CSV file first.", "error"); return; }
  btn.disabled = true;
  setStatus($("pred-status"), "Uploading CSV and predicting...");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const data = await postForm("/api/predict/csv", fd);
    setStatus($("pred-status"), `Loaded ${data.candles.time.length} bars.`, "ok");
    renderAll(data);
  } catch (e) {
    setStatus($("pred-status"), e.message, "error");
  } finally {
    btn.disabled = false;
  }
});

// ---- Live auto-refresh: repeatedly re-fetches the TradingView symbol on a
// timer and re-renders the chart/table/forecast in place. tvDatafeed is a
// polling client, not a websocket push feed into this app, so "live" here
// means "always showing the latest bar TradingView currently has" rather
// than tick-by-tick streaming.
let liveTimer = null;

function stopLive() {
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = null;
  $("live-toggle-btn").textContent = "Start live updates";
  $("live-indicator").textContent = "";
  $("live-indicator").classList.remove("live-on");
}

function startLive() {
  const seconds = Math.max(10, Number($("live-interval").value) || 30);
  stopLive();
  liveTimer = setInterval(() => runTVPredict({ silent: true }), seconds * 1000);
  $("live-toggle-btn").textContent = "Stop live updates";
  $("live-indicator").textContent = `Live: refreshing every ${seconds}s`;
  $("live-indicator").classList.add("live-on");
  runTVPredict({ silent: true });
}

$("live-toggle-btn").addEventListener("click", () => {
  if (liveTimer) stopLive();
  else startLive();
});

refreshModelStatus();
