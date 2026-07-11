# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Two layered pieces living under `smc/`:

1. **`smc/*.py`** — a faithful Python port of the LuxAlgo _Smart Money Concepts_ Pine Script v5 indicator. Deterministic, rule-based: given OHLCV data it detects swing/internal market structure (BOS/CHoCH), order blocks, fair value gaps, equal highs/lows, and premium/discount zones — exactly reproducing what the original indicator draws on a TradingView chart.
2. **`smc/webapp/`** — a Flask + scikit-learn app built on top of (1). It uses the rule-based engine as a _ground-truth label generator_, trains a `RandomForestClassifier` to forecast the _next_ bar's BOS/CHoCH (rather than just detecting it after the fact), and serves a single-page UI (Plotly charts) for training/predicting from TradingView data or an uploaded CSV.

Original Pine Script is © LuxAlgo, CC BY-NC-SA 4.0 (non-commercial) — this port reproduces signal logic for personal/educational/research use.

## Commands

All commands assume the venv at `smc/venv` (Windows paths; adjust activation for other shells).

```powershell
cd smc
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt        # includes tvdatafeed installed from GitHub (not on PyPI)
```

Run the rule-based engine standalone (also serves as the closest thing to a smoke test — no formal test suite/pytest exists in this repo):

```powershell
python main.py                         # synthetic demo data, writes smc_chart.html
python main.py path/to/ohlcv.csv       # your own data (columns: time,open,high,low,close,volume)
```

Run the ML web app:

```powershell
cd webapp
python app.py                          # serves http://127.0.0.1:5000, debug=True (auto-reloads on file change)
```

There is no lint/build/CI config in this repo — no `pytest.ini`, `pyproject.toml`, or linter config exists.

## TradingView credentials (webapp only)

`smc/webapp/ml_engine.py` pulls live data via the **unofficial** `tvDatafeed` package (installed via `pip install git+https://github.com/rongardF/tvdatafeed.git` — not on PyPI). It works anonymously with reduced reliability/limits, or authenticates using `TV_USERNAME`/`TV_PASSWORD` env vars loaded from `smc/webapp/.env` (via `python-dotenv`, wired in `app.py`).

- `smc/webapp/.env` holds real credentials and is git-ignored (`*.env` in `C:\Users\ADMIN\.gitignore`, which is the actual root of this git repo — see "Repo root" note below). `smc/webapp/.env.example` is the safe, empty template that _is_ tracked.
- Anonymous access is known to be flaky for larger intraday pulls; `fetch_tradingview()` retries up to 3 times, forcing a client reconnect between attempts, before failing.
- tvDatafeed's login itself is currently unreliable for many accounts (`error while signin`, a widely-reported upstream issue unrelated to credential correctness) and silently falls back to anonymous mode on failure — don't assume `.env` credentials are actually being used without checking server logs.

## Repo root note

The git repository root is `C:\Users\ADMIN` (the whole Windows user profile), **not** this project directory — `smc_project` is a subfolder within it. Be deliberate about `git add` scope; never use blanket `git add -A`/`git add .` from outside `smc_project` without checking `git status` first, since the tree includes unrelated personal directories.

## Architecture: the rule-based engine (`smc/`)

The core design tension the codebase resolves: Pine Script's BOS/CHoCH, order-block, and trend-bias logic is inherently _stateful and bar-sequential_ (each bar's classification depends on the running trend bias, which only changes when a break occurs), while most of the supporting math (ATR, rolling highest/lowest, leg/pivot determination, FVG thresholds) is pure and vectorizable. The codebase keeps these separate:

- **Vectorized math** lives in `utils.py` (ATR, true range, crossover/crossunder) and `pivots.py`'s `compute_leg_series`/`leg_transitions` (pandas rolling ops + `ffill`, no loops).
- **Stateful bar-by-bar walks** live in `market_structure.py`'s `MarketStructureEngine.run()` and `order_blocks.py`'s `OrderBlockManager` — these iterate `for i in range(n)` exactly mirroring Pine's own execution model, because bias/order-block state can't be vectorized away.

`indicators.py`'s `SmartMoneyConcepts.run()` is the orchestrator. **Its execution order matters and deliberately mirrors the bottom "MUTABLE VARIABLES & EXECUTION" section of the original Pine script**: per bar, swing structure → internal structure → equal-high/low → `displayStructure(internal=True)` (may emit an order-block creation request) → `displayStructure(internal=False)` → delete-mitigated-order-blocks(internal) → delete-mitigated-order-blocks(swing). Reordering these breaks fidelity to the original indicator, so don't casually reshuffle calls in `run()`.

A subtlety worth knowing before touching `pivots.py`: Pine's `leg(size)` is called from three independent call-sites (swing structure, internal structure, equal-high/low), and each maintains **its own independent state** (`var leg = 0` is call-site-local, not global) — reproduced as three separate calls to `compute_leg_series()`, not one shared computation.

Key modules and what each owns (see `smc/README.md` for the full Pine-feature-to-module table and per-feature fidelity notes — e.g. how order block anchor candles, `ta.atr(200)` seeding, and MTF Daily/Weekly/Monthly levels are reproduced without lookahead bias):

- `config.py` — dataclass config mirroring every Pine `input.*()`, grouped to match Pine's `group=` sections (`SwingStructureConfig`, `InternalStructureConfig`, `OrderBlockConfig`, etc.)
- `market_structure.py` — swing/internal BOS/CHoCH detection, trend bias, trailing extremes. Also emits `StructureBreakEvent` records carrying the full pivot→breakout geometry (pivot time/price, breakout time) for each BOS/CHoCH — this is what lets the webapp draw a break as an arrow rather than a bare marker.
- `order_blocks.py`, `fvg.py`, `equal_highs_lows.py`, `zones.py` — order blocks (creation/mitigation), fair value gaps, equal highs/lows, premium/discount/equilibrium zones + MTF levels.
- `indicators.py` — `SmartMoneyConcepts.run(df) -> SMCResult`, the single entry point tying everything together and producing the enriched output DataFrame + order block/FVG lists + alert booleans.
- `plotting.py` — Plotly chart builder for the standalone `main.py` demo.

## Architecture: the ML layer (`smc/webapp/`)

`ml_engine.py` is the whole ML pipeline; `app.py` is a thin Flask wrapper exposing it as JSON endpoints. Everything routes through the rule-based engine above via `run_smc_engine()`, which returns the full `SMCResult` (not just the DataFrame) so both the enriched OHLCV frame _and_ the pivot→breakout event geometry are available.

Design points that aren't obvious from a single file:

- **Labels are horizon-shifted to avoid trivial/leaky prediction.** `build_labels(out, horizon)` labels bar `i` with whichever BOS/CHoCH event fires at bar `i + horizon` (default horizon=1: predict the _next_ bar). Predicting bar `i`'s own event from bar `i`'s own close-vs-pivot crossover would need no ML at all — it's directly computable by the rule engine.
- **Only the swing-level (`internal=False`) structure series feeds the classifier label**, since it's less noisy than internal (5-bar lookback) structure — but `predict()` returns _both_ swing and internal `structure_break_events` (tagged with `scope: "swing"|"internal"`) for the chart, since the frontend renders both densities with different visual weight.
- **`normalize_ohlcv_dataframe()` is the single ingestion chokepoint** for both TradingView fetches and CSV uploads — it tolerates missing `volume` (synthesized as a flat placeholder, since volume isn't used by the rule engine at all, only by two optional ML features), numeric Unix-epoch-seconds timestamps vs. ISO strings (auto-detected — don't assume `pd.to_datetime` defaults are safe on raw integers, they'll misparse as nanoseconds), and various column-name aliases.
- **`FEATURE_COLUMNS` are all backward-looking only** (rolling windows, `.shift()`, no centering) — this is required for the horizon-shift label scheme to be valid; adding a new feature that peeks forward silently invalidates the whole train/predict pipeline.
- The trained model bundle (`RandomForestClassifier` + feature columns + label map + metrics) is pickled via `joblib` to `smc/webapp/models/bos_choch_model.joblib` — a single global model file, overwritten on each retrain (no versioning/multi-model support).

Frontend (`templates/index.html`, `static/app.js`, `static/style.css`) is a single vanilla-JS page, no build step. Chart rendering conventions worth preserving if extending it: annotation **color always encodes direction** (bullish/bearish), never event type; **arrowhead shape encodes BOS (solid) vs CHoCH (hollow)**; swing vs internal structure are independently toggleable and rendered at different visual weight (internal = thinner/lighter, since it's far denser).
