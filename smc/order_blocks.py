"""
order_blocks.py
----------------
Reproduces Pine's order-block lifecycle:

    * `storeOrdeBlock(pivot, internal, bias)`  - locate the candle with
      the most extreme parsed high/low between the pivot bar and the
      breakout bar, and register it as a new order block.
    * `deleteOrderBlocks(internal)`            - remove (mitigate) order
      blocks once price trades through them.
    * `drawOrderBlocks(internal)`              - (display only) keep the
      most recent N order blocks visible.

Volatility filtering ("parsed" highs/lows) is reproduced in
`indicators.py` (it needs the ATR / cumulative-mean-range series that is
also shared by other modules) and passed in here as plain arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import OrderBlockMitigation
from pivots import BEARISH, BULLISH


@dataclass
class OrderBlock:
    """Mirrors Pine's `orderBlock` UDT."""

    bar_high: float
    bar_low: float
    bar_time: object
    bar_index: int
    bias: int  # BULLISH (+1) / BEARISH (-1)
    internal: bool
    created_at_index: int  # bar index of the breakout that created this OB
    mitigated: bool = False
    mitigated_at_index: Optional[int] = None


class OrderBlockManager:
    """Stateful FIFO-ish container reproducing Pine's
    `swingOrderBlocks` / `internalOrderBlocks` arrays, including
    mitigation removal.

    Order blocks are prepended (`array.unshift`) when created, and Pine
    caps the backing array at 100 entries (oldest dropped). We keep the
    same cap for parity, and additionally expose `max_display` (mirrors
    `internalOrderBlocksSizeInput` / `swingOrderBlocksSizeInput`) so
    callers can slice down to the "currently visible" set exactly like
    `drawOrderBlocks` does.
    """

    MAX_STORED = 100

    def __init__(
        self,
        parsed_highs: np.ndarray,
        parsed_lows: np.ndarray,
        times: pd.Index,
        mitigation: OrderBlockMitigation,
        max_display: int,
    ):
        self.parsed_highs = parsed_highs
        self.parsed_lows = parsed_lows
        self.times = times
        self.mitigation = mitigation
        self.max_display = max_display

        self.active: list[OrderBlock] = []
        self.all_created: list[OrderBlock] = []  # full history, incl. mitigated

    # -----------------------------------------------------------------
    def create(self, pivot_bar_index: int, breakout_bar_index: int, bias: int, internal: bool) -> None:
        """Reproduce `storeOrdeBlock`: within [pivot_bar_index,
        breakout_bar_index), find the candle with the most extreme
        parsed high (bearish OB) or parsed low (bullish OB)."""
        if pivot_bar_index < 0 or breakout_bar_index <= pivot_bar_index:
            return

        window_high = self.parsed_highs[pivot_bar_index:breakout_bar_index]
        window_low = self.parsed_lows[pivot_bar_index:breakout_bar_index]
        if len(window_high) == 0:
            return

        if bias == BEARISH:
            offset = int(np.argmax(window_high))
        else:
            offset = int(np.argmin(window_low))
        parsed_index = pivot_bar_index + offset

        ob = OrderBlock(
            bar_high=float(self.parsed_highs[parsed_index]),
            bar_low=float(self.parsed_lows[parsed_index]),
            bar_time=self.times[parsed_index],
            bar_index=parsed_index,
            bias=bias,
            internal=internal,
            created_at_index=breakout_bar_index,
        )
        self.active.insert(0, ob)  # array.unshift
        if len(self.active) > self.MAX_STORED:
            self.active.pop()  # array.pop (drop oldest, i.e. last element)
        self.all_created.append(ob)

    # -----------------------------------------------------------------
    def delete_mitigated(self, bar_index: int, close: float, high: float, low: float) -> list[OrderBlock]:
        """Reproduce `deleteOrderBlocks`: remove any order block that has
        been traded through, based on the configured mitigation source.

        Returns the list of order blocks mitigated on this bar (useful
        for alert reproduction).
        """
        bearish_source = close if self.mitigation == OrderBlockMitigation.CLOSE else high
        bullish_source = close if self.mitigation == OrderBlockMitigation.CLOSE else low

        newly_mitigated = []
        survivors = []
        for ob in self.active:
            crossed = False
            if ob.bias == BEARISH and bearish_source > ob.bar_high:
                crossed = True
            elif ob.bias == BULLISH and bullish_source < ob.bar_low:
                crossed = True

            if crossed:
                ob.mitigated = True
                ob.mitigated_at_index = bar_index
                newly_mitigated.append(ob)
            else:
                survivors.append(ob)

        self.active = survivors
        return newly_mitigated

    # -----------------------------------------------------------------
    def visible(self) -> list[OrderBlock]:
        """Reproduce `drawOrderBlocks`: the most recent `max_display`
        active order blocks (already most-recent-first since we
        `insert(0, ...)` on creation)."""
        return self.active[: self.max_display]
