"""
indicators.py
-------------
Top-level orchestrator reproducing the full `SMC_Indicator.pine` script.

`SmartMoneyConcepts.run(df)` takes an OHLCV DataFrame and returns an
`SMCResult` containing:

    * an enriched copy of the input DataFrame with one column per
      indicator output (bos, choch, trend, internal_trend, swing_high,
      swing_low, equal_high, equal_low, bullish_ob, bearish_ob,
      bullish_fvg, bearish_fvg, premium_zone, discount_zone,
      equilibrium, ...)
    * the full lists of order blocks / fair value gaps (for plotting or
      further analysis)
    * a dict of boolean alert series, one per Pine `alertcondition(...)`

Execution order intentionally mirrors the bottom of `SMC_Indicator.pine`
(the "MUTABLE VARIABLES & EXECUTION" section) bar by bar, since Pine's
order-block store/mitigate calls are interleaved with structure
detection within a single per-bar pass.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import OrderBlockFilter, SMCConfig
from equal_highs_lows import build_equal_hl_columns
from fvg import FairValueGap, FairValueGapDetector
from market_structure import MarketStructureEngine, StructureResult
from order_blocks import OrderBlock, OrderBlockManager
from pivots import BULLISH
from utils import atr as compute_atr
from utils import cumulative_mean_range
from zones import classify_zone, compute_mtf_levels, compute_premium_discount_zones

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass
class SMCResult:
    """Container for everything `SmartMoneyConcepts.run()` produces."""

    df: pd.DataFrame
    swing_order_blocks: list[OrderBlock]
    internal_order_blocks: list[OrderBlock]
    fair_value_gaps: list[FairValueGap]
    alerts: dict[str, np.ndarray]
    structure_break_events: list = field(default_factory=list)
    mtf_levels: dict[str, pd.DataFrame] = field(default_factory=dict)
    config: SMCConfig = None

    def latest_signals(self) -> dict:
        """Convenience accessor mirroring what a trader would check on
        the most recently closed bar."""
        last = self.df.iloc[-1]
        return {
            "time": self.df.index[-1],
            "close": last["close"],
            "trend": int(last["trend"]),
            "internal_trend": int(last["internal_trend"]),
            "bos": bool(last["bos"] != 0),
            "choch": bool(last["choch"] != 0),
            "equal_high": bool(last["equal_high"]),
            "equal_low": bool(last["equal_low"]),
            "premium_zone": bool(last.get("premium_zone", False)),
            "discount_zone": bool(last.get("discount_zone", False)),
            "equilibrium": bool(last.get("equilibrium", False)),
            "active_bullish_order_blocks": sum(
                1 for ob in self.swing_order_blocks + self.internal_order_blocks
                if ob.bias == BULLISH and not ob.mitigated
            ),
            "active_bearish_order_blocks": sum(
                1 for ob in self.swing_order_blocks + self.internal_order_blocks
                if ob.bias != BULLISH and not ob.mitigated
            ),
        }


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns and ensure `time` is the DataFrame index (as
    required by the task spec), sorted ascending."""
    df = df.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "DataFrame must either have a 'time' column or already be indexed by time."
        )

    df = df.sort_index()
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype(float)
    return df


class SmartMoneyConcepts:
    """Python reproduction of the LuxAlgo `Smart Money Concepts` Pine
    Script indicator."""

    def __init__(self, config: SMCConfig | None = None):
        self.config = config or SMCConfig()

    def run(self, ohlcv: pd.DataFrame) -> SMCResult:
        df = _prepare_dataframe(ohlcv)
        n = len(df)
        cfg = self.config

        # ---- volatility measure & "parsed" (volatility-filtered) highs/lows
        atr_series = compute_atr(df, cfg.atr_length)
        if cfg.order_blocks.filter_method == OrderBlockFilter.ATR:
            volatility_measure = atr_series
        else:
            volatility_measure = cumulative_mean_range(df)

        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        vol = volatility_measure.to_numpy(dtype=float)

        high_volatility_bar = (high - low) >= (2.0 * vol)
        parsed_high = np.where(high_volatility_bar, low, high)
        parsed_low = np.where(high_volatility_bar, high, low)

        # ---- market structure (swing / internal BOS-CHoCH, trailing extremes)
        structure_engine = MarketStructureEngine(df, cfg, atr_series)
        structure: StructureResult = structure_engine.run()

        # ---- order blocks: replay events bar-by-bar, interleaving
        #      creation and mitigation exactly like the Pine execution order
        internal_ob_mgr = OrderBlockManager(
            parsed_high, parsed_low, df.index,
            cfg.order_blocks.mitigation_method, cfg.order_blocks.internal_count,
        )
        swing_ob_mgr = OrderBlockManager(
            parsed_high, parsed_low, df.index,
            cfg.order_blocks.mitigation_method, cfg.order_blocks.swing_count,
        )

        events_by_bar: dict[int, list] = defaultdict(list)
        for ev in structure.order_block_events:
            events_by_bar[ev.bar_index].append(ev)

        bullish_ob_created = np.zeros(n, dtype=bool)
        bearish_ob_created = np.zeros(n, dtype=bool)

        for i in range(n):
            for ev in events_by_bar.get(i, []):
                mgr = internal_ob_mgr if ev.internal else swing_ob_mgr
                enabled = (
                    cfg.order_blocks.show_internal if ev.internal else cfg.order_blocks.show_swing
                )
                if not enabled:
                    continue
                mgr.create(ev.pivot_bar_index, ev.bar_index, ev.bias, ev.internal)
                if ev.bias == BULLISH:
                    bullish_ob_created[i] = True
                else:
                    bearish_ob_created[i] = True

            if cfg.order_blocks.show_internal:
                internal_ob_mgr.delete_mitigated(i, close[i], high[i], low[i])
            if cfg.order_blocks.show_swing:
                swing_ob_mgr.delete_mitigated(i, close[i], high[i], low[i])

        # ---- fair value gaps ------------------------------------------------
        bullish_fvg = np.zeros(n, dtype=bool)
        bearish_fvg = np.zeros(n, dtype=bool)
        fvgs: list[FairValueGap] = []
        if cfg.fvg.show:
            detector = FairValueGapDetector(df, auto_threshold=cfg.fvg.auto_threshold)
            bullish_fvg, bearish_fvg, fvgs = detector.detect()
            detector.apply_mitigation(fvgs)

        # ---- equal highs / lows ---------------------------------------------
        equal_cols = build_equal_hl_columns(df.index, structure.equal_events)

        # ---- premium / discount / equilibrium zones --------------------------
        zone_cols = compute_premium_discount_zones(structure.trailing_top, structure.trailing_bottom)
        zone_flags = classify_zone(df["close"], zone_cols) if cfg.zones.show else pd.DataFrame(
            {"premium_zone": False, "discount_zone": False, "equilibrium": False}, index=df.index
        )

        # ---- MTF daily/weekly/monthly levels ----------------------------------
        mtf_levels = {}
        if cfg.mtf_levels.show_daily or cfg.mtf_levels.show_weekly or cfg.mtf_levels.show_monthly:
            mtf_levels = compute_mtf_levels(
                df,
                daily=cfg.mtf_levels.show_daily,
                weekly=cfg.mtf_levels.show_weekly,
                monthly=cfg.mtf_levels.show_monthly,
            )

        # ---- assemble output DataFrame -----------------------------------------
        out = df.copy()
        out["atr"] = atr_series.to_numpy()
        out["volatility_measure"] = vol
        out["high_volatility_bar"] = high_volatility_bar

        out["trend"] = structure.swing_trend
        out["internal_trend"] = structure.internal_trend
        out["bos"] = structure.swing_bos
        out["choch"] = structure.swing_choch
        out["internal_bos"] = structure.internal_bos
        out["internal_choch"] = structure.internal_choch

        out["swing_high"] = structure.swing_high_level
        out["swing_low"] = structure.swing_low_level
        out["internal_high"] = structure.internal_high_level
        out["internal_low"] = structure.internal_low_level

        out["equal_high"] = equal_cols["equal_high"]
        out["equal_high_level"] = equal_cols["equal_high_level"]
        out["equal_low"] = equal_cols["equal_low"]
        out["equal_low_level"] = equal_cols["equal_low_level"]

        out["bullish_ob"] = bullish_ob_created
        out["bearish_ob"] = bearish_ob_created

        out["bullish_fvg"] = bullish_fvg
        out["bearish_fvg"] = bearish_fvg

        for col, values in zone_cols.items():
            out[col] = values
        out["premium_zone"] = zone_flags["premium_zone"].to_numpy()
        out["discount_zone"] = zone_flags["discount_zone"].to_numpy()
        out["equilibrium"] = zone_flags["equilibrium"].to_numpy()

        if cfg.show_trend_candles:
            out["candle_color"] = np.where(out["internal_trend"] >= 0, "bullish", "bearish")

        return SMCResult(
            df=out,
            swing_order_blocks=swing_ob_mgr.all_created,
            internal_order_blocks=internal_ob_mgr.all_created,
            fair_value_gaps=fvgs,
            alerts=structure.alerts,
            structure_break_events=structure.structure_break_events,
            mtf_levels=mtf_levels,
            config=cfg,
        )
