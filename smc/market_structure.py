"""
market_structure.py
--------------------
Reproduces the swing/internal market-structure logic of
`SMC_Indicator.pine`:

    * `getCurrentStructure(size, equalHighLow, internal)`
    * `displayStructure(internal)`  (the BOS / CHoCH crossover detector)
    * trailing extremes (`updateTrailingExtremes`) used for Strong/Weak
      High/Low markers and Premium/Discount zones.

This module walks the OHLC series bar-by-bar (exactly like Pine's own
execution model) because BOS/CHoCH detection is fundamentally stateful:
whether a breakout is labelled BOS or CHoCH depends on the *current*
trend bias, which itself only changes when a breakout occurs.

Order-block *creation requests* are emitted here (as `OrderBlockEvent`)
but the actual order-block bookkeeping (storage, FIFO limits, mitigation)
lives in `order_blocks.py` — `indicators.py` wires the two together bar
by bar, mirroring the interleaved execution order of the original script:

    1. getCurrentStructure(swing)
    2. getCurrentStructure(internal)
    3. getCurrentStructure(equal)         [optional]
    4. displayStructure(internal=True)    -> may emit internal OB request
    5. displayStructure(internal=False)   -> may emit swing OB request
    6. deleteOrderBlocks(internal=True)
    7. deleteOrderBlocks(internal=False)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import SMCConfig
from pivots import (
    BEARISH,
    BULLISH,
    Pivot,
    TrailingExtremes,
    TrendState,
    compute_leg_series,
    leg_transitions,
)

BOS = "BOS"
CHOCH = "CHoCH"


@dataclass
class OrderBlockEvent:
    """Emitted whenever a BOS/CHoCH breakout fires and order-block
    tracking is applicable, mirroring Pine's `storeOrdeBlock(...)` call
    site inside `displayStructure`."""

    bar_index: int  # bar at which the breakout was confirmed
    pivot_bar_index: int  # bar_index of the pivot that was broken
    bias: int  # BULLISH (+1) or BEARISH (-1)
    internal: bool


@dataclass
class StructureBreakEvent:
    """Emitted whenever a BOS/CHoCH breakout fires. Unlike `OrderBlockEvent`
    (which only exists to drive order-block bookkeeping), this carries the
    full "from pivot -> to breakout" geometry needed to draw the break as
    an arrow/line on a chart, mirroring how SMC indicators visually
    connect the broken structure point to the bar that broke it."""

    pivot_bar_index: int
    pivot_time: object
    pivot_level: float  # price level of the pivot that was broken

    bar_index: int  # bar at which the breakout was confirmed
    bar_time: object
    breakout_close: float  # close price on the breakout bar

    tag: str  # BOS | CHoCH
    bias: int  # BULLISH (+1) or BEARISH (-1)
    internal: bool


@dataclass
class EqualLevelEvent:
    """Emitted when an Equal-High or Equal-Low pair is confirmed."""

    bar_index: int
    pivot_bar_index: int
    level: float
    is_high: bool


@dataclass
class StructureResult:
    """All per-bar output series produced by `MarketStructureEngine.run()`."""

    swing_trend: np.ndarray
    internal_trend: np.ndarray

    swing_bos: np.ndarray  # +1/-1 direction on bars where a swing BOS fired, else 0
    swing_choch: np.ndarray
    internal_bos: np.ndarray
    internal_choch: np.ndarray

    swing_high_level: np.ndarray  # price marked at the pivot bar, NaN elsewhere
    swing_low_level: np.ndarray
    internal_high_level: np.ndarray
    internal_low_level: np.ndarray

    trailing_top: np.ndarray
    trailing_bottom: np.ndarray

    order_block_events: list[OrderBlockEvent]
    equal_events: list[EqualLevelEvent]
    structure_break_events: list[StructureBreakEvent]

    alerts: dict[str, np.ndarray]


class MarketStructureEngine:
    """Bar-by-bar reproduction of Pine's swing/internal structure state
    machine."""

    def __init__(self, df: pd.DataFrame, config: SMCConfig, atr: pd.Series):
        self.df = df
        self.config = config
        self.atr = atr.to_numpy()
        self.n = len(df)

        self.high = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self.close = df["close"].to_numpy(dtype=float)
        self.open = df["open"].to_numpy(dtype=float)
        self.index = df.index

        # Pre-compute the vectorized leg (pivot-confirmation) series for
        # each of the three independent state machines Pine instantiates
        # via separate call-sites of `leg(size)`.
        self._swing_leg = compute_leg_series(df["high"], df["low"], config.swing.length)
        self._internal_leg = compute_leg_series(df["high"], df["low"], config.internal.length)
        self._equal_leg = compute_leg_series(df["high"], df["low"], config.equal_hl.length)

        _, self._swing_pl, self._swing_ph = leg_transitions(self._swing_leg)
        _, self._internal_pl, self._internal_ph = leg_transitions(self._internal_leg)
        _, self._equal_pl, self._equal_ph = leg_transitions(self._equal_leg)

    def run(self) -> StructureResult:
        n = self.n
        swing_trend = np.zeros(n, dtype=int)
        internal_trend = np.zeros(n, dtype=int)
        swing_bos = np.zeros(n, dtype=int)
        swing_choch = np.zeros(n, dtype=int)
        internal_bos = np.zeros(n, dtype=int)
        internal_choch = np.zeros(n, dtype=int)
        swing_high_level = np.full(n, np.nan)
        swing_low_level = np.full(n, np.nan)
        internal_high_level = np.full(n, np.nan)
        internal_low_level = np.full(n, np.nan)
        trailing_top = np.full(n, np.nan)
        trailing_bottom = np.full(n, np.nan)

        order_block_events: list[OrderBlockEvent] = []
        equal_events: list[EqualLevelEvent] = []
        structure_break_events: list[StructureBreakEvent] = []

        alert_fields = [
            "internal_bullish_bos", "internal_bearish_bos",
            "internal_bullish_choch", "internal_bearish_choch",
            "swing_bullish_bos", "swing_bearish_bos",
            "swing_bullish_choch", "swing_bearish_choch",
            "equal_highs", "equal_lows",
        ]
        alerts = {f: np.zeros(n, dtype=bool) for f in alert_fields}

        swing_high, swing_low = Pivot(), Pivot()
        internal_high, internal_low = Pivot(), Pivot()
        equal_high, equal_low = Pivot(), Pivot()
        swing_trend_state, internal_trend_state = TrendState(), TrendState()
        trailing = TrailingExtremes()

        want_trailing = self.config.swing.show_high_low_swings or self.config.zones.show
        want_equal = self.config.equal_hl.show
        threshold = self.config.equal_hl.threshold

        for i in range(n):
            t = self.index[i]
            h, l, c = self.high[i], self.low[i], self.close[i]
            a = self.atr[i]

            # ---- 1. update trailing extremes (running max/min) --------
            if want_trailing:
                new_top = h if np.isnan(trailing.top) else max(h, trailing.top)
                if new_top == h:
                    trailing.last_top_time = t
                trailing.top = new_top
                new_bottom = l if np.isnan(trailing.bottom) else min(l, trailing.bottom)
                if new_bottom == l:
                    trailing.last_bottom_time = t
                trailing.bottom = new_bottom

            # ---- 2. getCurrentStructure(swing) -------------------------
            self._update_structure(
                i, t,
                pivot_low=swing_low, pivot_high=swing_high,
                is_pivot_low=self._swing_pl.iloc[i], is_pivot_high=self._swing_ph.iloc[i],
                new_pivot=(self._swing_pl.iloc[i] or self._swing_ph.iloc[i]),
                size=self.config.swing.length,
                equal_high_low=False,
                trailing=trailing,
                update_trailing=True,
            )

            # ---- 3. getCurrentStructure(internal) ----------------------
            self._update_structure(
                i, t,
                pivot_low=internal_low, pivot_high=internal_high,
                is_pivot_low=self._internal_pl.iloc[i], is_pivot_high=self._internal_ph.iloc[i],
                new_pivot=(self._internal_pl.iloc[i] or self._internal_ph.iloc[i]),
                size=self.config.internal.length,
                equal_high_low=False,
                trailing=None,
                update_trailing=False,
            )

            # ---- 4. getCurrentStructure(equal) --------------------------
            if want_equal:
                size = self.config.equal_hl.length
                if self._equal_pl.iloc[i] and size <= i:
                    low_size = self.low[i - size]
                    if not np.isnan(equal_low.current_level) and abs(equal_low.current_level - low_size) < threshold * a:
                        equal_events.append(EqualLevelEvent(i, equal_low.bar_index, low_size, is_high=False))
                        alerts["equal_lows"][i] = True
                    equal_low.last_level = equal_low.current_level
                    equal_low.current_level = low_size
                    equal_low.crossed = False
                    equal_low.bar_time = self.index[i - size]
                    equal_low.bar_index = i - size
                elif self._equal_ph.iloc[i] and size <= i:
                    high_size = self.high[i - size]
                    if not np.isnan(equal_high.current_level) and abs(equal_high.current_level - high_size) < threshold * a:
                        equal_events.append(EqualLevelEvent(i, equal_high.bar_index, high_size, is_high=True))
                        alerts["equal_highs"][i] = True
                    equal_high.last_level = equal_high.current_level
                    equal_high.current_level = high_size
                    equal_high.crossed = False
                    equal_high.bar_time = self.index[i - size]
                    equal_high.bar_index = i - size

            # ---- 5. displayStructure(internal=True) --------------------
            self._display_structure(
                i, c,
                pivot_high=internal_high, pivot_low=internal_low,
                trend_state=internal_trend_state,
                internal=True,
                other_high=swing_high, other_low=swing_low,
                bos_arr=internal_bos, choch_arr=internal_choch,
                alerts=alerts, order_block_events=order_block_events,
                structure_break_events=structure_break_events,
            )

            # ---- 6. displayStructure(internal=False) --------------------
            self._display_structure(
                i, c,
                pivot_high=swing_high, pivot_low=swing_low,
                trend_state=swing_trend_state,
                internal=False,
                other_high=None, other_low=None,
                bos_arr=swing_bos, choch_arr=swing_choch,
                alerts=alerts, order_block_events=order_block_events,
                structure_break_events=structure_break_events,
            )

            # ---- record per-bar snapshots --------------------------------
            swing_trend[i] = swing_trend_state.bias
            internal_trend[i] = internal_trend_state.bias
            if want_trailing:
                trailing_top[i] = trailing.top
                trailing_bottom[i] = trailing.bottom

            # Mark pivot levels at the bar they occurred on (lagged by
            # `size`), useful for plotting swing/internal structure points.
            if self._swing_pl.iloc[i] and self.config.swing.length <= i:
                swing_low_level[i - self.config.swing.length] = self.low[i - self.config.swing.length]
            if self._swing_ph.iloc[i] and self.config.swing.length <= i:
                swing_high_level[i - self.config.swing.length] = self.high[i - self.config.swing.length]
            if self._internal_pl.iloc[i] and self.config.internal.length <= i:
                internal_low_level[i - self.config.internal.length] = self.low[i - self.config.internal.length]
            if self._internal_ph.iloc[i] and self.config.internal.length <= i:
                internal_high_level[i - self.config.internal.length] = self.high[i - self.config.internal.length]

        return StructureResult(
            swing_trend=swing_trend,
            internal_trend=internal_trend,
            swing_bos=swing_bos,
            swing_choch=swing_choch,
            internal_bos=internal_bos,
            internal_choch=internal_choch,
            swing_high_level=swing_high_level,
            swing_low_level=swing_low_level,
            internal_high_level=internal_high_level,
            internal_low_level=internal_low_level,
            trailing_top=trailing_top,
            trailing_bottom=trailing_bottom,
            order_block_events=order_block_events,
            equal_events=equal_events,
            structure_break_events=structure_break_events,
            alerts=alerts,
        )

    # -----------------------------------------------------------------
    def _update_structure(
        self, i: int, t,
        pivot_low: Pivot, pivot_high: Pivot,
        is_pivot_low: bool, is_pivot_high: bool, new_pivot: bool,
        size: int, equal_high_low: bool,
        trailing: Optional[TrailingExtremes], update_trailing: bool,
    ) -> None:
        """Reproduce the pivot-updating half of `getCurrentStructure`
        (excludes the equal-high/low branch, handled separately since it
        needs its own dedicated Pivot objects)."""
        if not new_pivot or size > i:
            return

        if is_pivot_low:
            low_size = self.low[i - size]
            pivot_low.last_level = pivot_low.current_level
            pivot_low.current_level = low_size
            pivot_low.crossed = False
            pivot_low.bar_time = self.index[i - size]
            pivot_low.bar_index = i - size
            if update_trailing and trailing is not None:
                trailing.bottom = pivot_low.current_level
                trailing.bar_time = pivot_low.bar_time
                trailing.bar_index = pivot_low.bar_index
                trailing.last_bottom_time = pivot_low.bar_time
        else:
            high_size = self.high[i - size]
            pivot_high.last_level = pivot_high.current_level
            pivot_high.current_level = high_size
            pivot_high.crossed = False
            pivot_high.bar_time = self.index[i - size]
            pivot_high.bar_index = i - size
            if update_trailing and trailing is not None:
                trailing.top = pivot_high.current_level
                trailing.bar_time = pivot_high.bar_time
                trailing.bar_index = pivot_high.bar_index
                trailing.last_top_time = pivot_high.bar_time

    # -----------------------------------------------------------------
    def _display_structure(
        self, i: int, close: float,
        pivot_high: Pivot, pivot_low: Pivot,
        trend_state: TrendState, internal: bool,
        other_high: Optional[Pivot], other_low: Optional[Pivot],
        bos_arr: np.ndarray, choch_arr: np.ndarray,
        alerts: dict[str, np.ndarray],
        order_block_events: list[OrderBlockEvent],
        structure_break_events: list[StructureBreakEvent],
    ) -> None:
        """Reproduce `displayStructure(internal)`: detects BOS/CHoCH via
        a crossover/crossunder of `close` against the stored pivot level,
        classifies BOS vs CHoCH from the current trend bias, and (for
        internal structure) applies the same-level "extraCondition"
        confluence filter against the swing structure."""
        prev_close = self.close[i - 1] if i > 0 else np.nan

        # ---- bullish side: crossover(close, pivot_high.current_level) --
        level = pivot_high.current_level
        if not np.isnan(level) and not pivot_high.crossed:
            crossover = (not np.isnan(prev_close)) and prev_close <= level and close > level
            extra_condition = True
            if internal and other_high is not None:
                extra_condition = pivot_high.current_level != other_high.current_level
                if self.config.internal.filter_confluence:
                    high, low, open_ = self.high[i], self.low[i], self.open[i]
                    bullish_bar = (high - max(close, open_)) > min(close, open_ - low)
                    extra_condition = extra_condition and bullish_bar
            if crossover and extra_condition:
                tag = CHOCH if trend_state.bias == BEARISH else BOS
                pivot_high.crossed = True
                trend_state.bias = BULLISH
                direction = 1
                if tag == BOS:
                    bos_arr[i] = direction
                else:
                    choch_arr[i] = direction
                prefix = "internal" if internal else "swing"
                alerts[f"{prefix}_bullish_{'choch' if tag == CHOCH else 'bos'}"][i] = True
                order_block_events.append(
                    OrderBlockEvent(bar_index=i, pivot_bar_index=pivot_high.bar_index, bias=BULLISH, internal=internal)
                )
                structure_break_events.append(
                    StructureBreakEvent(
                        pivot_bar_index=pivot_high.bar_index,
                        pivot_time=self.index[pivot_high.bar_index],
                        pivot_level=level,
                        bar_index=i,
                        bar_time=self.index[i],
                        breakout_close=close,
                        tag=tag,
                        bias=BULLISH,
                        internal=internal,
                    )
                )

        # ---- bearish side: crossunder(close, pivot_low.current_level) --
        level = pivot_low.current_level
        if not np.isnan(level) and not pivot_low.crossed:
            crossunder = (not np.isnan(prev_close)) and prev_close >= level and close < level
            extra_condition = True
            if internal and other_low is not None:
                extra_condition = (pivot_low.current_level != other_low.current_level)
                if self.config.internal.filter_confluence:
                    high, low, open_ = self.high[i], self.low[i], self.open[i]
                    bearish_bar = (high - max(close, open_)) < min(close, open_ - low)
                    extra_condition = extra_condition and bearish_bar
            if crossunder and extra_condition:
                tag = CHOCH if trend_state.bias == BULLISH else BOS
                pivot_low.crossed = True
                trend_state.bias = BEARISH
                direction = -1
                if tag == BOS:
                    bos_arr[i] = direction
                else:
                    choch_arr[i] = direction
                prefix = "internal" if internal else "swing"
                alerts[f"{prefix}_bearish_{'choch' if tag == CHOCH else 'bos'}"][i] = True
                order_block_events.append(
                    OrderBlockEvent(bar_index=i, pivot_bar_index=pivot_low.bar_index, bias=BEARISH, internal=internal)
                )
                structure_break_events.append(
                    StructureBreakEvent(
                        pivot_bar_index=pivot_low.bar_index,
                        pivot_time=self.index[pivot_low.bar_index],
                        pivot_level=level,
                        bar_index=i,
                        bar_time=self.index[i],
                        breakout_close=close,
                        tag=tag,
                        bias=BEARISH,
                        internal=internal,
                    )
                )
