"""
zones.py
--------
Reproduces:

    * `drawPremiumDiscountZones()` - Premium / Equilibrium / Discount
      bands computed from the trailing swing extremes.
    * `drawLevels(timeframe, ...)` - previous Daily/Weekly/Monthly
      high & low levels (the "Highs & Lows MTF" group).

Premium/Discount zone math (from Pine, `trailing.top` / `trailing.bottom`
being the running swing extremes maintained by
`market_structure.MarketStructureEngine`)::

    premium_bottom     = 0.95 * top + 0.05 * bottom
    premium_top        = top
    equilibrium_top    = 0.525 * top + 0.475 * bottom
    equilibrium_bottom = 0.525 * bottom + 0.475 * top
    equilibrium_level  = avg(top, bottom)
    discount_top       = 0.95 * bottom + 0.05 * top
    discount_bottom    = bottom
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_premium_discount_zones(
    trailing_top: np.ndarray, trailing_bottom: np.ndarray
) -> dict[str, np.ndarray]:
    """Vectorized reproduction of `drawPremiumDiscountZones`. Returns the
    band boundaries for every bar (NaN until the first trailing extremes
    are established)."""
    top = trailing_top
    bottom = trailing_bottom

    premium_top = top
    premium_bottom = 0.95 * top + 0.05 * bottom

    equilibrium_top = 0.525 * top + 0.475 * bottom
    equilibrium_bottom = 0.525 * bottom + 0.475 * top
    equilibrium_level = (top + bottom) / 2.0

    discount_top = 0.95 * bottom + 0.05 * top
    discount_bottom = bottom

    return {
        "premium_top": premium_top,
        "premium_bottom": premium_bottom,
        "equilibrium_top": equilibrium_top,
        "equilibrium_bottom": equilibrium_bottom,
        "equilibrium_level": equilibrium_level,
        "discount_top": discount_top,
        "discount_bottom": discount_bottom,
    }


def classify_zone(close: pd.Series, zones: dict[str, np.ndarray]) -> pd.DataFrame:
    """Convenience helper (not present verbatim in Pine, which only
    draws static bands): classifies each bar's close as being in the
    premium, discount, or equilibrium band. Useful for building simple
    "buy the discount / sell the premium" filters on top of the raw
    zone boundaries."""
    premium = close.to_numpy() >= zones["premium_bottom"]
    discount = close.to_numpy() <= zones["discount_top"]
    equilibrium = (~premium) & (~discount)
    return pd.DataFrame(
        {"premium_zone": premium, "discount_zone": discount, "equilibrium": equilibrium},
        index=close.index,
    )


def _previous_period_high_low(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Group rows by calendar period (`rule`: 'D', 'W', or 'M'), take the
    PRIOR period's high/low, and broadcast it across every bar of the
    current period - reproducing Pine's "previous Daily/Weekly/Monthly
    high/low" MTF levels without needing a live `request.security` feed."""
    period_key = df.index.to_period(rule)

    period_high = df["high"].groupby(period_key).max()
    period_low = df["low"].groupby(period_key).min()

    prev_high_by_period = period_high.shift(1)
    prev_low_by_period = period_low.shift(1)

    prev_high_aligned = period_key.map(prev_high_by_period).to_numpy()
    prev_low_aligned = period_key.map(prev_low_by_period).to_numpy()

    return pd.DataFrame(
        {"prev_high": prev_high_aligned, "prev_low": prev_low_aligned}, index=df.index
    )


def compute_mtf_levels(
    df: pd.DataFrame, daily: bool = False, weekly: bool = False, monthly: bool = False
) -> dict[str, pd.DataFrame]:
    """Reproduce the Daily/Weekly/Monthly "Highs & Lows MTF" group.
    Requires a DatetimeIndex. Returns a dict keyed by 'D' / 'W' / 'M',
    each value a DataFrame with `prev_high` / `prev_low` columns aligned
    to the input index.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("compute_mtf_levels requires a DatetimeIndex (set `time` as the index).")

    results: dict[str, pd.DataFrame] = {}
    if daily:
        results["D"] = _previous_period_high_low(df, "D")
    if weekly:
        results["W"] = _previous_period_high_low(df, "W")
    if monthly:
        results["M"] = _previous_period_high_low(df, "M")
    return results
