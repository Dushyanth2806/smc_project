"""
main.py
-------
Validation / demo script:

    1. Loads OHLCV data (a bundled synthetic sample if no CSV is given)
    2. Runs the SmartMoneyConcepts engine
    3. Prints the latest signals
    4. Produces an interactive Plotly chart (`smc_chart.html`)

Usage
-----
    python main.py                     # synthetic demo data
    python main.py path/to/ohlcv.csv   # your own data (columns:
                                        # time,open,high,low,close,volume)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from config import (
    EqualHighLowConfig,
    FairValueGapConfig,
    OrderBlockConfig,
    PremiumDiscountConfig,
    SMCConfig,
)
from indicators import SmartMoneyConcepts
from plotting import plot_smc


def generate_synthetic_ohlcv(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Generates a plausible-looking OHLCV series (geometric random walk
    with occasional trend/volatility regimes) purely for demonstration
    and smoke-testing purposes."""
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2023-01-01", periods=n, freq="4h")

    returns = rng.normal(loc=0.0002, scale=0.006, size=n)
    # inject a few short trending / choppy regimes so structure has
    # something interesting to detect
    for start in range(0, n, 80):
        regime = rng.choice(["up", "down", "chop"])
        length = min(40, n - start)
        if regime == "up":
            returns[start:start + length] += 0.0035
        elif regime == "down":
            returns[start:start + length] -= 0.0035

    close = 100 * np.exp(np.cumsum(returns))
    open_ = np.empty(n)
    open_[0] = close[0] * (1 - returns[0])
    open_[1:] = close[:-1]

    high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.004, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.004, size=n))
    volume = rng.uniform(1_000, 10_000, size=n)

    return pd.DataFrame(
        {"time": dt, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def load_ohlcv(path: str | None) -> pd.DataFrame:
    if path is None:
        print("No CSV path given - using synthetic demo data.")
        return generate_synthetic_ohlcv()
    df = pd.read_csv(path)
    expected = {"time", "open", "high", "low", "close", "volume"}
    if not expected.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(expected)}, got {list(df.columns)}")
    return df


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    ohlcv = load_ohlcv(csv_path)

    config = SMCConfig(
        equal_hl=EqualHighLowConfig(show=True, length=3, threshold=0.1),
        order_blocks=OrderBlockConfig(show_internal=True, show_swing=True),
        fvg=FairValueGapConfig(show=True, auto_threshold=True),
        zones=PremiumDiscountConfig(show=True),
    )

    engine = SmartMoneyConcepts(config)
    result = engine.run(ohlcv)

    print("=" * 60)
    print("Latest signals")
    print("=" * 60)
    for k, v in result.latest_signals().items():
        print(f"{k:>28}: {v}")

    print()
    print(f"Total swing order blocks created:    {len(result.swing_order_blocks)}")
    print(f"Total internal order blocks created: {len(result.internal_order_blocks)}")
    print(f"Total fair value gaps created:       {len(result.fair_value_gaps)}")
    print(f"Bullish BOS count:                   {int((result.df['bos'] > 0).sum())}")
    print(f"Bearish BOS count:                   {int((result.df['bos'] < 0).sum())}")
    print(f"Bullish CHoCH count:                 {int((result.df['choch'] > 0).sum())}")
    print(f"Bearish CHoCH count:                 {int((result.df['choch'] < 0).sum())}")

    fig = plot_smc(result, title="Smart Money Concepts (Python port)")
    out_path = "smc_chart.html"
    fig.write_html(out_path)
    print(f"\nChart written to {out_path}")


if __name__ == "__main__":
    main()
