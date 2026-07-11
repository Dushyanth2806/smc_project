"""
utils.py
--------
Vectorized numerical helpers that reproduce Pine Script's built-in `ta.*`
functions used throughout `SMC_Indicator.pine`.

Pine's `ta.*` functions operate on a per-bar "streaming" basis but are pure
functions of the bar's history, so they can be reproduced with pandas/numpy
vectorized operations without any loss of accuracy. The genuinely stateful
parts of the indicator (pivots, trend, order blocks, FVGs) are NOT vectorized
here — they live in the engine classes that walk the DataFrame bar-by-bar,
exactly mirroring Pine's execution model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """Reproduce Pine's `ta.tr` (true range).

    ``tr = max(high - low, abs(high - close[1]), abs(low - close[1]))``

    The first bar has no previous close, so Pine treats `close[1]` as `na`
    and simply falls back to `high - low` for that single bar.
    """
    prev_close = df["close"].shift(1)
    range1 = df["high"] - df["low"]
    range2 = (df["high"] - prev_close).abs()
    range3 = (df["low"] - prev_close).abs()
    tr = pd.concat([range1, range2, range3], axis=1).max(axis=1)
    tr.iloc[0] = range1.iloc[0]
    return tr


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    """Reproduce Pine's `ta.atr(length)`.

    Pine's ATR is an RMA (Wilder's moving average, alpha = 1/length) of the
    true range, seeded by a simple mean of the first `length` true-range
    values (this is how TradingView's RMA seeds its recursive average).
    """
    tr = true_range(df)
    alpha = 1.0 / length
    # RMA seed: simple mean of the first `length` values, then recursive
    # exponential smoothing afterwards - this matches ta.rma() semantics.
    rma = tr.copy()
    rma.iloc[:] = np.nan
    if len(tr) == 0:
        return rma
    seed_n = min(length, len(tr))
    rma.iloc[seed_n - 1] = tr.iloc[:seed_n].mean()
    for i in range(seed_n, len(tr)):
        rma.iloc[i] = alpha * tr.iloc[i] + (1 - alpha) * rma.iloc[i - 1]
    # Backfill the warm-up period with the seed value so downstream code
    # (which needs an ATR value on every bar) never sees NaN, matching the
    # practical effect of Pine plotting "no value" only very briefly.
    rma = rma.bfill()
    return rma


def cumulative_mean_range(df: pd.DataFrame) -> pd.Series:
    """Reproduce Pine's `ta.cum(ta.tr) / bar_index` (the RANGE volatility
    filter option). `bar_index` is 0-based, so this divides by
    (position + 1) using a 1-based running count to avoid division by zero
    on the very first bar, matching the effective magnitude Pine produces
    once `bar_index > 0`.
    """
    tr = true_range(df)
    cum_tr = tr.cumsum()
    bar_count = pd.Series(np.arange(1, len(df) + 1), index=df.index)
    return cum_tr / bar_count


def rolling_highest(series: pd.Series, length: int) -> pd.Series:
    """Reproduce Pine's single-argument `ta.highest(length)`, which uses
    `high` as its implicit source and looks back over `length` bars
    INCLUDING the current bar.
    """
    return series.rolling(window=length, min_periods=1).max()


def rolling_lowest(series: pd.Series, length: int) -> pd.Series:
    """Reproduce Pine's single-argument `ta.lowest(length)` (implicit
    source `low`)."""
    return series.rolling(window=length, min_periods=1).min()


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """Reproduce Pine's `ta.crossover(a, b)`: true on the bar where `a`
    was <= `b` on the previous bar and is > `b` on the current bar."""
    prev_a, prev_b = a.shift(1), b.shift(1)
    return (prev_a <= prev_b) & (a > b)


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """Reproduce Pine's `ta.crossunder(a, b)`."""
    prev_a, prev_b = a.shift(1), b.shift(1)
    return (prev_a >= prev_b) & (a < b)


def avg(a: float, b: float) -> float:
    """Reproduce Pine's `math.avg(a, b)`."""
    return (a + b) / 2.0


def safe_get(series: pd.Series, idx: int, default=np.nan):
    """Bounds-checked positional access, useful when walking arrays with
    an offset that could go negative (mirrors Pine's `na` on out-of-range
    history references)."""
    if idx < 0 or idx >= len(series):
        return default
    return series.iloc[idx]
