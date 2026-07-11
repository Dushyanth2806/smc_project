# Smart Money Concepts — Python Port

A faithful Python re-implementation of the LuxAlgo **Smart Money Concepts**
Pine Script v5 indicator (`SMC_Indicator.pine`), rebuilt as an idiomatic,
type-hinted, modular Python package instead of a line-by-line transliteration.

> The original Pine Script is © LuxAlgo, licensed CC BY-NC-SA 4.0
> (non-commercial). This port reproduces its *signal logic* for personal /
> educational / research use; keep the same license terms in mind if you
> redistribute this code.

## What it reproduces

| Pine feature | Python module |
|---|---|
| Swing & Internal market structure (`leg()`, `getCurrentStructure`, `displayStructure`) | `market_structure.py` |
| BOS / CHoCH detection & trend bias | `market_structure.py` |
| Order Blocks (swing + internal), storage, mitigation | `order_blocks.py` |
| Fair Value Gaps (3-candle gap, auto threshold, mitigation) | `fvg.py` |
| Equal Highs / Equal Lows | `equal_highs_lows.py` (built on pivots from `market_structure.py`) |
| Premium / Discount / Equilibrium zones | `zones.py` |
| Daily / Weekly / Monthly MTF levels | `zones.py` |
| ATR / Cumulative Mean Range volatility filters | `utils.py` |
| Pivot / trend / trailing-extremes data types | `pivots.py` |
| Alerts (`alertcondition(...)`) | `MarketStructureEngine.alerts` dict |
| Visualization | `plotting.py` (Plotly) |

## Project layout

```
smc/
├── main.py               # demo / validation script (entry point)
├── indicators.py         # SmartMoneyConcepts orchestrator + SMCResult
├── market_structure.py   # swing/internal BOS/CHoCH state machine
├── order_blocks.py        # order block creation & mitigation
├── fvg.py                # fair value gap detection & mitigation
├── equal_highs_lows.py   # equal high/low post-processing
├── zones.py              # premium/discount zones + MTF levels
├── pivots.py             # Pivot/Trend/TrailingExtremes + leg() detection
├── utils.py              # ATR, true range, crossover helpers
├── plotting.py           # Plotly chart builder
├── config.py             # dataclass-based configuration (mirrors Pine inputs)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.12+.

## Quick start

```python
import pandas as pd
from config import SMCConfig
from indicators import SmartMoneyConcepts
from plotting import plot_smc

# DataFrame with columns: time, open, high, low, close, volume
df = pd.read_csv("your_ohlcv.csv")

engine = SmartMoneyConcepts(SMCConfig())
result = engine.run(df)

print(result.latest_signals())

fig = plot_smc(result)
fig.write_html("smc_chart.html")
```

Or simply:

```bash
python main.py                     # runs on bundled synthetic demo data
python main.py path/to/ohlcv.csv   # runs on your own data
```

`main.py` will print the latest signal snapshot, summary counts (BOS/CHoCH,
order blocks, FVGs), and write an interactive `smc_chart.html`.

## Output columns

`SMCResult.df` (a copy of your input DataFrame, indexed by `time`) gains:

| Column | Meaning |
|---|---|
| `atr`, `volatility_measure`, `high_volatility_bar` | Volatility diagnostics |
| `trend`, `internal_trend` | Current swing / internal bias: `+1` bullish, `-1` bearish, `0` undetermined |
| `bos`, `choch` | Non-zero (`+1`/`-1`) on the bar a swing BOS/CHoCH fired |
| `internal_bos`, `internal_choch` | Same, for internal structure |
| `swing_high`, `swing_low` | Price level marked on the (lagged) pivot bar, else `NaN` |
| `internal_high`, `internal_low` | Same, for internal pivots |
| `equal_high`, `equal_low` | Boolean flag: an EQH/EQL pair was confirmed on this bar |
| `equal_high_level`, `equal_low_level` | Associated price level |
| `bullish_ob`, `bearish_ob` | Boolean flag: a new order block of that bias was created this bar |
| `bullish_fvg`, `bearish_fvg` | Boolean flag: a new Fair Value Gap was confirmed this bar |
| `premium_top/bottom`, `discount_top/bottom`, `equilibrium_top/bottom/level` | Zone boundaries |
| `premium_zone`, `discount_zone`, `equilibrium` | Whether the bar's close falls in each zone |

Full order block / FVG objects (including mitigation status) are available on
`result.swing_order_blocks`, `result.internal_order_blocks`, and
`result.fair_value_gaps` — these carry more detail (exact anchor candle,
mitigation bar) than can be flattened into single boolean columns.

Alert booleans (one array per bar, matching every Pine `alertcondition`) are
available at `result.alerts`, e.g. `result.alerts["swing_bullish_bos"]`.

## Design notes / fidelity

- **Execution model**: BOS/CHoCH, order-block storage and mitigation are
  inherently *stateful, bar-sequential* computations in Pine (they depend on
  the current trend bias and on a live watch-list of unmitigated order
  blocks). `market_structure.py` and `indicators.py` therefore walk the
  DataFrame bar-by-bar exactly as Pine does, while every purely
  mathematical sub-computation (ATR, rolling highest/lowest, leg
  determination, FVG thresholding) is fully vectorized with pandas/numpy.
- **`leg(size)`**: Pine's `var leg = 0` state is call-site-local; because
  `leg()` is invoked from three separate places (swing, internal, and equal
  high/low structure), each maintains its own independent leg state —
  reproduced with three independent calls to `compute_leg_series`.
- **`ta.atr(200)`**: reproduced as Wilder's RMA of true range, seeded with a
  simple mean over the first 200 bars (matching TradingView's `ta.rma`
  seeding behavior).
- **Order block anchor candle**: exactly reproduces `storeOrdeBlock` — the
  candle with the most extreme *volatility-filtered* ("parsed") high
  (bearish OB) or low (bullish OB) between the pivot bar and the breakout
  bar.
- **Fair Value Gaps**: implemented for the case `fairValueGapsTimeframeInput
  == ''` (FVG timeframe = chart timeframe), which is unambiguous to
  reproduce without a live multi-timeframe data feed. For genuine
  higher-timeframe FVGs, resample your OHLCV to the target timeframe first
  and run `FairValueGapDetector` on that frame (the detection math is
  identical either way) — see the docstring in `fvg.py`.
- **MTF Daily/Weekly/Monthly levels**: Pine's `request.security(...,
  lookahead_on)` pulls the *previous completed* higher-timeframe bar's
  high/low. Reproduced via `groupby(index.to_period(...))` + `shift(1)`,
  broadcast back across the base-timeframe bars of the current period —
  mathematically equivalent, no lookahead bias.
- Things intentionally **not** reproduced (display-only in Pine, no signal
  value): label/line/box drawing minutiae, exact colors per historical vs.
  present mode, and the `varip` real-time-only bar-index bookkeeping used
  purely to detect "new bar" events for redrawing MTF levels efficiently in
  a live chart.

## Testing

`main.py` doubles as a smoke test: it generates synthetic OHLCV data (or
loads your CSV), runs the full pipeline, and asserts (implicitly, by not
crashing and by printing sane counts) that structure detection, order
blocks, FVGs, and zones all compute end-to-end.
