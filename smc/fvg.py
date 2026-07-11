"""
fvg.py
------
Reproduces Pine's Fair Value Gap (FVG) detection from
`drawFairValueGaps()` / `deleteFairValueGaps()`.

Pine's original implementation pulls three bars via `request.security`
on a (possibly higher) timeframe. This module implements the
mathematically equivalent, fully vectorized computation for the case
where the FVG timeframe equals the chart's own timeframe (the Pine
default, `fairValueGapsTimeframeInput = ''`), which is also the only
case that is unambiguous to reproduce without a live TradingView feed.

For a genuine multi-timeframe FVG (e.g. plotting 1H FVGs on a 5m chart),
resample your OHLCV DataFrame to the target timeframe first, run
`FairValueGapDetector` on the resampled frame, then map gaps back onto
your base-timeframe index (each higher-timeframe bar corresponds to a
contiguous slice of base-timeframe bars) — the detection math itself is
identical either way.

Formulae (bar `i` = current, `i-1` = previous, `i-2` = two bars back)::

    bar_delta_percent = (close[i-1] - open[i-1]) / (open[i-1] * 100)
    threshold = auto ? (2 * cumsum(|bar_delta_percent|) / (bar_index)) : 0

    bullish_fvg = low[i] > high[i-2] and close[i-1] > high[i-2]
                  and bar_delta_percent > threshold
    bearish_fvg = high[i] < low[i-2] and close[i-1] < low[i-2]
                  and -bar_delta_percent > threshold
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from pivots import BEARISH, BULLISH


@dataclass
class FairValueGap:
    """Mirrors Pine's `fairValueGap` UDT (field names kept 1:1, including
    the fact that `top`/`bottom` are simply the two box coordinates as
    Pine assigns them — for a bearish gap `top` numerically ends up below
    `bottom`; this matches the original script's box-drawing math)."""

    top: float
    bottom: float
    bias: int  # BULLISH (+1) / BEARISH (-1)
    created_at_index: int
    left_time: object
    right_time: object
    mitigated: bool = False
    mitigated_at_index: Optional[int] = None

    @property
    def gap_high(self) -> float:
        return max(self.top, self.bottom)

    @property
    def gap_low(self) -> float:
        return min(self.top, self.bottom)

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0


class FairValueGapDetector:
    """Detects and mitigates Fair Value Gaps across an OHLC DataFrame."""

    def __init__(self, df: pd.DataFrame, auto_threshold: bool = True):
        self.df = df
        self.auto_threshold = auto_threshold
        self.high = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self.close = df["close"].to_numpy(dtype=float)
        self.open = df["open"].to_numpy(dtype=float)
        self.index = df.index
        self.n = len(df)

    def detect(self) -> tuple[np.ndarray, np.ndarray, list[FairValueGap]]:
        """Returns
        -------
        (bullish_flags, bearish_flags, gaps)
            Boolean arrays flagging the bar a gap was confirmed on, plus
            the full list of `FairValueGap` objects (unmitigated status
            populated by a subsequent call to `apply_mitigation`).
        """
        n = self.n
        bullish_flags = np.zeros(n, dtype=bool)
        bearish_flags = np.zeros(n, dtype=bool)
        gaps: list[FairValueGap] = []
        if n < 3:
            return bullish_flags, bearish_flags, gaps

        prev_close = self.close[:-1]  # close[i-1] aligned to i via slicing below
        prev_open = self.open[:-1]

        bar_delta_percent = np.full(n, np.nan)
        bar_delta_percent[1:] = (self.close[:-1] - self.open[:-1]) / (self.open[:-1] * 100.0)

        if self.auto_threshold:
            abs_delta = np.nan_to_num(np.abs(bar_delta_percent), nan=0.0)
            cum = np.cumsum(abs_delta)
            bar_count = np.arange(1, n + 1)  # bar_index is 0-based in Pine; guard div-by-zero
            threshold = 2.0 * cum / np.maximum(bar_count, 1)
        else:
            threshold = np.zeros(n)

        for i in range(2, n):
            last_close, last_open = self.close[i - 1], self.open[i - 1]
            current_low, current_high = self.low[i], self.high[i]
            last2_high, last2_low = self.high[i - 2], self.low[i - 2]
            delta_pct = bar_delta_percent[i - 1] if not np.isnan(bar_delta_percent[i - 1]) else 0.0
            # Note: Pine's threshold is evaluated with the *current* bar's
            # cumulative threshold value (index i), matching `ta.cum(...)`
            # being sampled on the current bar.
            thresh = threshold[i]

            is_bullish = (
                current_low > last2_high
                and last_close > last2_high
                and delta_pct > thresh
            )
            is_bearish = (
                current_high < last2_low
                and last_close < last2_low
                and -delta_pct > thresh
            )

            if is_bullish:
                bullish_flags[i] = True
                gaps.append(
                    FairValueGap(
                        top=current_low,
                        bottom=last2_high,
                        bias=BULLISH,
                        created_at_index=i,
                        left_time=self.index[i - 1],
                        right_time=self.index[i],
                    )
                )
            if is_bearish:
                bearish_flags[i] = True
                gaps.append(
                    FairValueGap(
                        top=current_high,
                        bottom=last2_low,
                        bias=BEARISH,
                        created_at_index=i,
                        left_time=self.index[i - 1],
                        right_time=self.index[i],
                    )
                )

        return bullish_flags, bearish_flags, gaps

    def apply_mitigation(self, gaps: list[FairValueGap]) -> list[FairValueGap]:
        """Reproduce `deleteFairValueGaps`: a gap is mitigated the moment
        price trades back through its unfilled boundary. Marks
        `mitigated` / `mitigated_at_index` in place and also returns the
        list for convenience."""
        for gap in gaps:
            start = gap.created_at_index + 1
            if start >= self.n:
                continue
            if gap.bias == BULLISH:
                # invalidated once low trades back below the gap's bottom
                breach = self.low[start:] < gap.bottom
            else:
                breach = self.high[start:] > gap.top
            hit = np.argmax(breach) if breach.any() else -1
            if hit >= 0:
                gap.mitigated = True
                gap.mitigated_at_index = start + hit
        return gaps
