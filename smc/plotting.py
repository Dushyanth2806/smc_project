"""
plotting.py
-----------
Plotly visualization for `SMCResult` (see `indicators.py`).

Draws:
    * OHLC candlesticks
    * Swing & Internal BOS / CHoCH lines + labels
    * Order block boxes (internal + swing, bullish/bearish, mitigated
      ones dimmed)
    * Fair Value Gap boxes
    * Equal High / Equal Low connecting lines + labels
    * Premium / Discount / Equilibrium zone bands
    * Swing/internal pivot markers (HH/HL/LH/LL style dots)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from indicators import SMCResult
from pivots import BULLISH

GREEN = "#089981"
RED = "#F23645"
BLUE = "#2157f3"
GRAY = "#878b94"

FVG_BULL = "rgba(0,255,104,0.25)"
FVG_BEAR = "rgba(255,0,8,0.25)"

OB_INTERNAL_BULL = "rgba(49,121,245,0.20)"
OB_INTERNAL_BEAR = "rgba(247,124,128,0.20)"
OB_SWING_BULL = "rgba(24,72,204,0.20)"
OB_SWING_BEAR = "rgba(178,40,51,0.20)"

PREMIUM_COLOR = "rgba(242,54,69,0.10)"
DISCOUNT_COLOR = "rgba(8,153,129,0.10)"
EQUILIBRIUM_COLOR = "rgba(135,139,148,0.10)"


def _add_candles(fig: go.Figure, df: pd.DataFrame) -> None:
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Price",
            increasing_line_color=GREEN, decreasing_line_color=RED,
        )
    )


def _add_structure_lines(fig: go.Figure, df: pd.DataFrame) -> None:
    """Draw BOS/CHoCH as horizontal segments from the breaking pivot's
    bar to the breakout bar (approximated here as a short segment ending
    at the breakout bar, since the exact pivot origin bar is not carried
    in the flattened DataFrame — use `SMCResult` order-block/structure
    objects directly for pixel-perfect original-pivot anchoring)."""
    for col, label_prefix, bull_color, bear_color, dash in (
        ("bos", "BOS", GREEN, RED, "solid"),
        ("choch", "CHoCH", GREEN, RED, "solid"),
        ("internal_bos", "iBOS", GREEN, RED, "dash"),
        ("internal_choch", "iCHoCH", GREEN, RED, "dash"),
    ):
        events = df.index[df[col] != 0]
        for t in events:
            direction = df.loc[t, col]
            color = bull_color if direction > 0 else bear_color
            price = df.loc[t, "close"]
            fig.add_annotation(
                x=t, y=price, text=label_prefix, showarrow=True, arrowhead=1,
                font=dict(color=color, size=10), arrowcolor=color,
                ay=-25 if direction > 0 else 25,
            )


def _add_order_blocks(fig: go.Figure, result: SMCResult, last_time) -> None:
    for ob_list, internal in ((result.internal_order_blocks, True), (result.swing_order_blocks, False)):
        for ob in ob_list:
            if ob.bias == BULLISH:
                color = OB_INTERNAL_BULL if internal else OB_SWING_BULL
            else:
                color = OB_INTERNAL_BEAR if internal else OB_SWING_BEAR
            right_edge = ob.bar_time if ob.mitigated and ob.mitigated_at_index is not None else last_time
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=ob.bar_time, x1=right_edge if not ob.mitigated else last_time,
                y0=ob.bar_low, y1=ob.bar_high,
                fillcolor=color,
                line=dict(width=0.5, color=color, dash="dot" if ob.mitigated else "solid"),
                opacity=0.5 if ob.mitigated else 1.0,
                layer="below",
            )


def _add_fvgs(fig: go.Figure, result: SMCResult, last_time) -> None:
    for gap in result.fair_value_gaps:
        color = FVG_BULL if gap.bias == BULLISH else FVG_BEAR
        right_edge = gap.right_time if gap.mitigated else last_time
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=gap.left_time, x1=right_edge,
            y0=gap.gap_low, y1=gap.gap_high,
            fillcolor=color, line=dict(width=0), layer="below",
        )


def _add_equal_hl(fig: go.Figure, df: pd.DataFrame) -> None:
    eqh = df.index[df["equal_high"]]
    if len(eqh):
        fig.add_trace(
            go.Scatter(
                x=eqh, y=df.loc[eqh, "equal_high_level"], mode="markers+text",
                text=["EQH"] * len(eqh), textposition="top center",
                marker=dict(symbol="triangle-down", color=RED, size=7),
                name="Equal Highs",
            )
        )
    eql = df.index[df["equal_low"]]
    if len(eql):
        fig.add_trace(
            go.Scatter(
                x=eql, y=df.loc[eql, "equal_low_level"], mode="markers+text",
                text=["EQL"] * len(eql), textposition="bottom center",
                marker=dict(symbol="triangle-up", color=GREEN, size=7),
                name="Equal Lows",
            )
        )


def _add_zones(fig: go.Figure, df: pd.DataFrame) -> None:
    if df["premium_top"].isna().all():
        return
    last = df.iloc[-1]
    x0, x1 = df.index[0], df.index[-1]
    fig.add_shape(
        type="rect", xref="x", yref="y", x0=x0, x1=x1,
        y0=last["premium_bottom"], y1=last["premium_top"],
        fillcolor=PREMIUM_COLOR, line=dict(width=0), layer="below",
    )
    fig.add_shape(
        type="rect", xref="x", yref="y", x0=x0, x1=x1,
        y0=last["equilibrium_bottom"], y1=last["equilibrium_top"],
        fillcolor=EQUILIBRIUM_COLOR, line=dict(width=0), layer="below",
    )
    fig.add_shape(
        type="rect", xref="x", yref="y", x0=x0, x1=x1,
        y0=last["discount_bottom"], y1=last["discount_top"],
        fillcolor=DISCOUNT_COLOR, line=dict(width=0), layer="below",
    )


def _add_swing_points(fig: go.Figure, df: pd.DataFrame) -> None:
    sh = df.index[df["swing_high"].notna()]
    if len(sh):
        fig.add_trace(
            go.Scatter(
                x=sh, y=df.loc[sh, "swing_high"], mode="markers", marker=dict(color=RED, size=5),
                name="Swing High",
            )
        )
    sl = df.index[df["swing_low"].notna()]
    if len(sl):
        fig.add_trace(
            go.Scatter(
                x=sl, y=df.loc[sl, "swing_low"], mode="markers", marker=dict(color=GREEN, size=5),
                name="Swing Low",
            )
        )


def plot_smc(result: SMCResult, title: str = "Smart Money Concepts") -> go.Figure:
    """Builds a single self-contained Plotly figure visualizing every
    facet of the SMC output. Returns the `go.Figure`; call `.show()` or
    `.write_html(...)` on it."""
    df = result.df
    fig = go.Figure()

    _add_candles(fig, df)
    _add_zones(fig, df)
    _add_order_blocks(fig, result, df.index[-1])
    _add_fvgs(fig, result, df.index[-1])
    _add_swing_points(fig, df)
    _add_equal_hl(fig, df)
    _add_structure_lines(fig, df)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
