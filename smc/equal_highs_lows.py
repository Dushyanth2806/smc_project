"""
equal_highs_lows.py
--------------------
Post-processes the `EqualLevelEvent`s produced by
`market_structure.MarketStructureEngine` (the equal-high/low branch of
Pine's `getCurrentStructure(size, equalHighLow=true)`) into flat
DataFrame columns.

Pine's equal-high/low detection reuses the exact same pivot-confirmation
machinery as swing/internal structure (`leg()` / `startOfNewLeg()`), just
with its own dedicated `equalHigh` / `equalLow` pivot objects and a
short default lookback (`equalHighsLowsLengthInput`, default 3). Two
consecutive pivots of the same type are flagged "equal" when they lie
within `equalHighsLowsThresholdInput * ATR` of one another — see
`drawEqualHighLow` in the Pine source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_structure import EqualLevelEvent


def build_equal_hl_columns(
    index: pd.Index, events: list[EqualLevelEvent]
) -> dict[str, np.ndarray]:
    """Builds `equal_high`, `equal_high_level`, `equal_low`,
    `equal_low_level` arrays flagging the bar on which each equal
    high/low pair was confirmed (mirrors where Pine draws the EQH/EQL
    label and connecting dotted line)."""
    n = len(index)
    equal_high = np.zeros(n, dtype=bool)
    equal_high_level = np.full(n, np.nan)
    equal_low = np.zeros(n, dtype=bool)
    equal_low_level = np.full(n, np.nan)

    for ev in events:
        if ev.is_high:
            equal_high[ev.bar_index] = True
            equal_high_level[ev.bar_index] = ev.level
        else:
            equal_low[ev.bar_index] = True
            equal_low_level[ev.bar_index] = ev.level

    return {
        "equal_high": equal_high,
        "equal_high_level": equal_high_level,
        "equal_low": equal_low,
        "equal_low_level": equal_low_level,
    }
