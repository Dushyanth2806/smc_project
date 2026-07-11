"""
config.py
---------
Configuration objects for the Smart Money Concepts (SMC) engine.

Every tunable parameter that existed as an `input.*()` call in the original
LuxAlgo Pine Script (`SMC_Indicator.pine`) is represented here as a typed,
documented dataclass field. Grouping mirrors the Pine `group = ...` sections:

    SMART_GROUP    -> SMCConfig (top-level mode/style)
    INTERNAL_GROUP -> InternalStructureConfig
    SWING_GROUP    -> SwingStructureConfig
    BLOCKS_GROUP   -> OrderBlockConfig
    EQUAL_GROUP    -> EqualHighLowConfig
    GAPS_GROUP     -> FairValueGapConfig
    LEVELS_GROUP   -> MTFLevelsConfig
    ZONES_GROUP    -> PremiumDiscountConfig

Only visual-only inputs (colors, label sizes, line styles used purely for
TradingView drawing objects) are kept as plain fields so that `plotting.py`
can still respect user preferences, but they never influence signal logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Mode(str, Enum):
    """Pine `modeInput`: HISTORICAL keeps all past structure drawings,
    PRESENT only keeps the most recent one. This only affects plotting,
    never the underlying signal series."""

    HISTORICAL = "Historical"
    PRESENT = "Present"


class Style(str, Enum):
    """Pine `styleInput`: COLORED uses the configured bullish/bearish
    colors, MONOCHROME forces a fixed grayscale palette."""

    COLORED = "Colored"
    MONOCHROME = "Monochrome"


class OrderBlockFilter(str, Enum):
    """Pine `orderBlockFilterInput`: which volatility measure is used to
    discard "high volatility" bars when locating order block candles."""

    ATR = "Atr"
    RANGE = "Cumulative Mean Range"


class OrderBlockMitigation(str, Enum):
    """Pine `orderBlockMitigationInput`: which price series is compared
    against an order block's high/low to decide whether it is mitigated
    (invalidated)."""

    CLOSE = "Close"
    HIGHLOW = "High/Low"


class LineStyle(str, Enum):
    SOLID = "solid"
    DASHED = "dash"
    DOTTED = "dot"


@dataclass
class InternalStructureConfig:
    """Mirrors Pine `INTERNAL_GROUP` inputs. Internal structure uses a
    fixed, short lookback (5 bars in the original script) to react quickly
    to short-term swings."""

    show: bool = True
    length: int = 5  # Pine hardcodes `getCurrentStructure(5, false, true)`
    filter_confluence: bool = False  # internalFilterConfluenceInput
    bullish_display: str = "All"  # ALL | BOS | CHOCH
    bearish_display: str = "All"


@dataclass
class SwingStructureConfig:
    """Mirrors Pine `SWING_GROUP` inputs."""

    show: bool = True
    length: int = 50  # swingsLengthInput (minval 10 in Pine)
    bullish_display: str = "All"
    bearish_display: str = "All"
    show_swing_points: bool = False
    show_high_low_swings: bool = True


@dataclass
class OrderBlockConfig:
    """Mirrors Pine `BLOCKS_GROUP` inputs."""

    show_internal: bool = True
    internal_count: int = 5  # max internal order blocks retained for display
    show_swing: bool = False
    swing_count: int = 5
    filter_method: OrderBlockFilter = OrderBlockFilter.ATR
    mitigation_method: OrderBlockMitigation = OrderBlockMitigation.HIGHLOW


@dataclass
class EqualHighLowConfig:
    """Mirrors Pine `EQUAL_GROUP` inputs."""

    show: bool = True
    length: int = 3  # bars used to confirm equal highs/lows (pivot lag)
    threshold: float = 0.1  # sensitivity as a fraction of ATR, range (0, 0.5)


@dataclass
class FairValueGapConfig:
    """Mirrors Pine `GAPS_GROUP` inputs."""

    show: bool = False
    auto_threshold: bool = True
    timeframe: str = ""  # '' = same timeframe as input data (no resampling)
    extend_bars: int = 1  # how many bars to extend FVG boxes to the right


@dataclass
class MTFLevelsConfig:
    """Mirrors Pine `LEVELS_GROUP` inputs. Daily/Weekly/Monthly high-low
    levels computed via higher-timeframe resampling."""

    show_daily: bool = False
    daily_style: LineStyle = LineStyle.SOLID
    show_weekly: bool = False
    weekly_style: LineStyle = LineStyle.SOLID
    show_monthly: bool = False
    monthly_style: LineStyle = LineStyle.SOLID


@dataclass
class PremiumDiscountConfig:
    """Mirrors Pine `ZONES_GROUP` inputs."""

    show: bool = False


@dataclass
class SMCConfig:
    """Top level configuration object aggregating every sub-config.

    Parameters
    ----------
    mode:
        HISTORICAL or PRESENT (display-only, kept for API completeness).
    style:
        COLORED or MONOCHROME (display-only palette selector).
    show_trend_candles:
        Pine `showTrendInput` - whether to compute a trend-colored candle
        series (`internal_trend` column drives this in our output).
    atr_length:
        Pine `ta.atr(200)` - fixed window used for the ATR volatility
        measure that gates high-volatility-bar filtering and equal
        high/low threshold sizing.
    """

    mode: Mode = Mode.HISTORICAL
    style: Style = Style.COLORED
    show_trend_candles: bool = False
    atr_length: int = 200

    internal: InternalStructureConfig = field(default_factory=InternalStructureConfig)
    swing: SwingStructureConfig = field(default_factory=SwingStructureConfig)
    order_blocks: OrderBlockConfig = field(default_factory=OrderBlockConfig)
    equal_hl: EqualHighLowConfig = field(default_factory=EqualHighLowConfig)
    fvg: FairValueGapConfig = field(default_factory=FairValueGapConfig)
    mtf_levels: MTFLevelsConfig = field(default_factory=MTFLevelsConfig)
    zones: PremiumDiscountConfig = field(default_factory=PremiumDiscountConfig)
