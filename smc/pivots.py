"""
pivots.py
---------
Reproduces Pine's `leg()`, `startOfNewLeg()`, `startOfBullishLeg()`,
`startOfBearishLeg()` functions, plus the `pivot`, `trend`, and
`trailingExtremes` user-defined types (UDTs) from `SMC_Indicator.pine`.

Pine's `leg(size)`:

    leg(int size) =>
        var leg = 0
        newLegHigh = high[size] > ta.highest(size)
        newLegLow  = low[size]  < ta.lowest(size)
        if newLegHigh
            leg := BEARISH_LEG   // 0
        else if newLegLow
            leg := BULLISH_LEG   // 1
        leg

`ta.highest(size)` / `ta.lowest(size)` (single-arg form) use `high` / `low`
as their implicit source and look back `size` bars INCLUDING the current
bar. `high[size]` reads the high `size` bars in the past. So a new
"bearish leg" begins when the high from `size` bars ago is still the
highest high seen in the trailing `size`-bar window (i.e. no bar since has
closed above it) - Pine's classic lagging pivot/fractal confirmation.

Because `leg` only changes when one of the two boolean conditions fires
(otherwise it holds its previous value via `var`), the whole state machine
reduces to a "carry-forward" vectorized operation, which we implement with
`ffill`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils import rolling_highest, rolling_lowest

BULLISH_LEG = 1
BEARISH_LEG = 0

BULLISH = 1
BEARISH = -1


@dataclass
class Pivot:
    """Mirrors Pine's `pivot` UDT.

    Attributes
    ----------
    current_level : float
        Price level of the most recently confirmed pivot.
    last_level : float
        Price level of the pivot immediately before `current_level`
        (used to classify HH/HL/LH/LL).
    crossed : bool
        Whether price has already broken this pivot (prevents firing the
        same BOS/CHoCH twice off one pivot).
    bar_time : optional
        Timestamp (DataFrame index value) of the pivot bar.
    bar_index : int
        Integer bar position of the pivot bar.
    """

    current_level: float = np.nan
    last_level: float = np.nan
    crossed: bool = False
    bar_time: Optional[object] = None
    bar_index: int = -1


@dataclass
class TrendState:
    """Mirrors Pine's `trend` UDT: a single directional bias, +1 bullish,
    -1 bearish, 0 = undetermined (no BOS/CHoCH observed yet)."""

    bias: int = 0


@dataclass
class TrailingExtremes:
    """Mirrors Pine's `trailingExtremes` UDT used for the "Strong/Weak
    High/Low" markers and the Premium/Discount zone boundaries."""

    top: float = np.nan
    bottom: float = np.nan
    bar_time: Optional[object] = None
    bar_index: int = -1
    last_top_time: Optional[object] = None
    last_bottom_time: Optional[object] = None


def compute_leg_series(high: pd.Series, low: pd.Series, size: int) -> pd.Series:
    """Vectorized reproduction of Pine's `leg(size)` across an entire
    series.

    Returns
    -------
    pd.Series[int]
        0 (BEARISH_LEG) / 1 (BULLISH_LEG) leg value for every bar.
    """
    highest = rolling_highest(high, size)
    lowest = rolling_lowest(low, size)

    high_shifted = high.shift(size)
    low_shifted = low.shift(size)

    new_leg_high = high_shifted > highest
    new_leg_low = low_shifted < lowest

    # Raw signal: 0 where a new bearish leg starts, 1 where a new bullish
    # leg starts, NaN everywhere else (meaning "carry previous value").
    raw = pd.Series(np.nan, index=high.index)
    raw[new_leg_high] = BEARISH_LEG
    # `new_leg_high` is checked first in Pine's if/else-if, so it takes
    # priority when both conditions are (impossibly) true on the same bar.
    raw[new_leg_low & ~new_leg_high] = BULLISH_LEG

    leg = raw.ffill().fillna(BEARISH_LEG).astype(int)
    return leg


def leg_transitions(leg: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Reproduce `startOfNewLeg`, `startOfBullishLeg`, `startOfBearishLeg`.

    Returns
    -------
    (new_pivot, pivot_low, pivot_high) : tuple of boolean Series
        `pivot_low`  -> leg changed 0->1 (a swing LOW was just confirmed)
        `pivot_high` -> leg changed 1->0 (a swing HIGH was just confirmed)
    """
    change = leg.diff()
    new_pivot = change.fillna(0) != 0
    pivot_low = change == 1  # startOfBullishLeg: ta.change(leg) == +1
    pivot_high = change == -1  # startOfBearishLeg: ta.change(leg) == -1
    return new_pivot.fillna(False), pivot_low.fillna(False), pivot_high.fillna(False)
