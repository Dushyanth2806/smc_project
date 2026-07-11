"""
ml_engine.py
------------
Machine-learning layer built on top of the rule-based Smart Money Concepts
(SMC) engine in the parent `smc/` package.

The rule-based engine (`indicators.SmartMoneyConcepts`) computes BOS/CHoCH
*after the fact*, purely from crossovers of `close` against confirmed pivot
levels. That's used here as a ground-truth label generator: we walk history,
compute BOS/CHoCH with the deterministic engine, then train a classifier to
predict whether a BOS/CHoCH will occur on the *next* bar using only
information available up to the current bar (no look-ahead).

Two entry points feed the classifier:
    * `fetch_tradingview(...)`   -> live/historical OHLCV pulled from TradingView (via tvDatafeed)
    * a user-uploaded CSV        -> parsed by `normalize_ohlcv_dataframe(...)`

`fetch_yfinance(...)` is kept around unused as a fallback/reference but is
no longer wired into the web app's routes.

Both funnel into the same `train_model(...)` / `predict(...)` pipeline.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ml_engine")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# The rule-based engine lives one directory up (smc/).
PARENT_DIR = Path(__file__).resolve().parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from config import (  # noqa: E402
    EqualHighLowConfig,
    FairValueGapConfig,
    OrderBlockConfig,
    PremiumDiscountConfig,
    SMCConfig,
)
from indicators import SmartMoneyConcepts  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "bos_choch_model.joblib"

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

# 5-class target: no event, or one of the four directional BOS/CHoCH events.
LABEL_NAMES = {
    0: "NONE",
    1: "BULLISH_BOS",
    2: "BEARISH_BOS",
    3: "BULLISH_CHOCH",
    4: "BEARISH_CHOCH",
}

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_5", "ret_10",
    "hl_range", "oc_range", "gap",
    "body_ratio", "upper_wick", "lower_wick",
    "vol_chg", "vol_ma_ratio",
    "close_over_sma10", "close_over_sma20", "close_over_sma50",
    "atr_pct", "rsi_14",
    "dist_to_high20", "dist_to_low20",
    "momentum_10", "volatility_10",
    "trend", "internal_trend",
    "dist_to_premium_top", "dist_to_discount_bottom", "dist_to_equilibrium",
]


def _smc_config() -> SMCConfig:
    """A lean but complete configuration: structure + equal HL + order
    blocks + zones all enabled, since several of them feed derived
    features (premium/discount/equilibrium distances)."""
    return SMCConfig(
        equal_hl=EqualHighLowConfig(show=True, length=3, threshold=0.1),
        order_blocks=OrderBlockConfig(show_internal=True, show_swing=True),
        fvg=FairValueGapConfig(show=False),
        zones=PremiumDiscountConfig(show=True),
    )


def normalize_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Accepts a loosely-formatted OHLCV DataFrame (from an uploaded CSV or
    a yfinance download) and returns one with exactly the columns the SMC
    engine expects: a `time` column plus lower-cased open/high/low/close/
    volume. Extra columns (prev_close, adj close, symbol, ...) are dropped
    since prev_close etc. are re-derived internally to guarantee
    consistency with the rest of the feature pipeline.
    """
    df = df.copy()

    # yfinance sometimes returns a MultiIndex column frame for single tickers.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Find a time-like column, or fall back to the index.
    time_col = None
    for candidate in ("time", "date", "datetime", "timestamp"):
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col is not None:
        raw_time = df[time_col]
        if pd.api.types.is_numeric_dtype(raw_time):
            # Exports like TradingView's "time" column are Unix epoch
            # seconds (e.g. 1674186300), not the nanosecond epoch
            # pd.to_datetime assumes by default - misparsing this as ns
            # silently lands every timestamp near 1970-01-01.
            df["time"] = pd.to_datetime(raw_time.astype("int64"), unit="s", utc=True)
        else:
            df["time"] = pd.to_datetime(raw_time, utc=True, errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        df["time"] = df.index
    else:
        df = df.reset_index(drop=True)
        df["time"] = pd.date_range("2000-01-01", periods=len(df), freq="D")

    rename_map = {}
    if "close" not in df.columns and "adj_close" in df.columns:
        rename_map["adj_close"] = "close"
    for alt, canonical in {
        "vol": "volume", "vol.": "volume", "shares_traded": "volume", "quantity": "volume",
    }.items():
        if canonical not in df.columns and alt in df.columns:
            rename_map[alt] = canonical
    df = df.rename(columns=rename_map)

    # Volume isn't used anywhere in the rule-based BOS/CHoCH structure
    # logic (only OHLC drives pivots/crossovers) - it only feeds a couple
    # of optional ML features (vol_chg, vol_ma_ratio). So a dataset
    # without it (e.g. many charting-tool exports) can still be used:
    # synthesize a flat placeholder column instead of hard-failing.
    if "volume" not in df.columns:
        df["volume"] = 1.0

    required_price_cols = ("open", "high", "low", "close")
    missing = [c for c in required_price_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Data is missing required columns {missing}. "
            f"Found columns: {sorted(df.columns)}"
        )

    out = df[["time", *REQUIRED_COLUMNS]].copy()
    for col in REQUIRED_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["time", *required_price_cols]).sort_values("time").reset_index(drop=True)
    out["volume"] = out["volume"].fillna(1.0)
    if out.empty:
        raise ValueError(
            "No usable rows remained after parsing this file - check that 'time' "
            "and the open/high/low/close columns contain valid values."
        )
    return out


def fetch_yfinance(ticker: str, period: str = "60d", interval: str = "15m") -> pd.DataFrame:
    """Pulls OHLCV history directly from Yahoo Finance via `yfinance`."""
    import yfinance as yf

    raw = yf.download(
        tickers=ticker, period=period, interval=interval,
        auto_adjust=False, progress=False, multi_level_index=False,
    )
    if raw is None or raw.empty:
        raise ValueError(
            f"yfinance returned no data for ticker={ticker!r}, period={period!r}, "
            f"interval={interval!r}. Check the symbol and that the period/interval "
            f"combination is valid (Yahoo limits intraday history)."
        )
    raw = raw.reset_index()
    raw = raw.rename(columns={raw.columns[0]: "time"})
    return normalize_ohlcv_dataframe(raw)


_TV_CLIENT = None  # lazy singleton - the websocket handshake is slow, reuse it across requests

_TV_INTERVAL_MAP = {
    "1m": "in_1_minute",
    "5m": "in_5_minute",
    "15m": "in_15_minute",
    "30m": "in_30_minute",
    "1h": "in_1_hour",
    "1d": "in_daily",
}


def _get_tv_client():
    global _TV_CLIENT
    if _TV_CLIENT is None:
        from tvDatafeed import TvDatafeed

        import os
        username = os.environ.get("TV_USERNAME")
        password = os.environ.get("TV_PASSWORD")
        _TV_CLIENT = TvDatafeed(username, password) if username and password else TvDatafeed()
    return _TV_CLIENT


def fetch_tradingview(symbol: str, exchange: str = "NSE", interval: str = "15m", n_bars: int = 1000) -> pd.DataFrame:
    """Pulls OHLCV history directly from TradingView via the (unofficial)
    `tvDatafeed` client - the same bar data TradingView's own charts use.
    Works anonymously with reduced limits, or set TV_USERNAME/TV_PASSWORD
    environment variables to use a logged-in session."""
    from tvDatafeed import Interval

    if interval not in _TV_INTERVAL_MAP:
        raise ValueError(f"Unsupported interval {interval!r}. Use one of {sorted(_TV_INTERVAL_MAP)}.")
    tv_interval = getattr(Interval, _TV_INTERVAL_MAP[interval])

    # The anonymous (no-login) tvDatafeed websocket session is flaky - it
    # occasionally returns an empty response for a request it would happily
    # serve on retry, especially for larger intraday pulls. Retry a couple
    # of times, forcing a fresh client/reconnect after the first failure,
    # before surfacing an error to the user.
    global _TV_CLIENT
    last_error: Optional[Exception] = None
    for attempt in range(3):
        logger.info("fetch_tradingview: %s:%s interval=%s n_bars=%s attempt=%d", exchange, symbol, interval, n_bars, attempt + 1)
        try:
            tv = _get_tv_client()
            raw = tv.get_hist(symbol=symbol, exchange=exchange, interval=tv_interval, n_bars=n_bars)
        except Exception as exc:  # noqa: BLE001
            raw = None
            last_error = exc
            logger.warning("fetch_tradingview: attempt %d raised %r", attempt + 1, exc)
        if raw is not None and not raw.empty:
            logger.info(
                "fetch_tradingview: got %d raw rows, range %s -> %s",
                len(raw), raw.index.min(), raw.index.max(),
            )
            raw = raw.reset_index().rename(columns={"datetime": "time"})
            out = normalize_ohlcv_dataframe(raw)
            logger.info("fetch_tradingview: normalized to %d rows\n%s", len(out), out.to_string(max_rows=10))
            return out
        logger.warning("fetch_tradingview: attempt %d returned empty/None, reconnecting", attempt + 1)
        _TV_CLIENT = None  # drop the stale session so the next attempt reconnects

    detail = f" (last error: {last_error})" if last_error else ""
    raise ValueError(
        f"TradingView returned no data for symbol={symbol!r}, exchange={exchange!r}, "
        f"interval={interval!r}, n_bars={n_bars} after 3 attempts{detail}. This is usually "
        f"transient flakiness in the anonymous tvDatafeed session - try again, request fewer "
        f"bars, or set TV_USERNAME/TV_PASSWORD for a more reliable logged-in session. Also "
        f"double check the symbol/exchange combination (e.g. symbol='INFY', exchange='NSE')."
    )


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def run_smc_engine(df: pd.DataFrame):
    """Runs the deterministic SMC engine and returns its full result
    object: `.df` is the enriched OHLCV frame (indexed by time), and
    `.structure_break_events` carries each BOS/CHoCH's pivot->breakout
    geometry (used to draw the break as an arrow rather than a bare
    marker)."""
    engine = SmartMoneyConcepts(_smc_config())
    return engine.run(df)


def build_features(out: pd.DataFrame) -> pd.DataFrame:
    """Builds the ML feature matrix from the SMC engine's output
    DataFrame. Every feature at row `i` only depends on bars `<= i`, so it
    is safe to pair with a label that looks at bar `i + horizon`."""
    close, open_, high, low, volume = (
        out["close"], out["open"], out["high"], out["low"], out["volume"],
    )

    feat = pd.DataFrame(index=out.index)
    feat["ret_1"] = close.pct_change(1)
    feat["ret_3"] = close.pct_change(3)
    feat["ret_5"] = close.pct_change(5)
    feat["ret_10"] = close.pct_change(10)

    rng = (high - low).replace(0, np.nan)
    feat["hl_range"] = rng / close
    feat["oc_range"] = (close - open_) / open_
    feat["gap"] = (open_ - close.shift(1)) / close.shift(1)
    feat["body_ratio"] = (close - open_).abs() / rng
    feat["upper_wick"] = (high - np.maximum(close, open_)) / rng
    feat["lower_wick"] = (np.minimum(close, open_) - low) / rng

    feat["vol_chg"] = volume.pct_change(1)
    feat["vol_ma_ratio"] = volume / volume.rolling(20, min_periods=1).mean()

    sma10 = close.rolling(10, min_periods=1).mean()
    sma20 = close.rolling(20, min_periods=1).mean()
    sma50 = close.rolling(50, min_periods=1).mean()
    feat["close_over_sma10"] = close / sma10 - 1
    feat["close_over_sma20"] = close / sma20 - 1
    feat["close_over_sma50"] = close / sma50 - 1

    feat["atr_pct"] = out["atr"] / close
    feat["rsi_14"] = _rsi(close, 14)

    high20 = high.rolling(20, min_periods=1).max()
    low20 = low.rolling(20, min_periods=1).min()
    feat["dist_to_high20"] = (high20 - close) / close
    feat["dist_to_low20"] = (close - low20) / close

    feat["momentum_10"] = close - close.shift(10)
    feat["volatility_10"] = close.pct_change().rolling(10, min_periods=1).std()

    feat["trend"] = out["trend"]
    feat["internal_trend"] = out["internal_trend"]

    feat["dist_to_premium_top"] = (out["premium_top"] - close) / close
    feat["dist_to_discount_bottom"] = (close - out["discount_bottom"]) / close
    feat["dist_to_equilibrium"] = (close - out["equilibrium_level"]) / close

    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat[FEATURE_COLUMNS]


def build_labels(out: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Class label for bar `i`: which BOS/CHoCH event (if any) fires on
    bar `i + horizon` of the *swing* structure series. Predicting the
    next bar's event (rather than the current bar's, which is knowable
    directly from `close` vs the pivot level with no ML needed) is what
    makes this a genuine forecasting task."""
    bos = out["bos"].shift(-horizon)
    choch = out["choch"].shift(-horizon)

    label = pd.Series(0, index=out.index, dtype=int)
    label[bos > 0] = 1
    label[bos < 0] = 2
    label[choch > 0] = 3
    label[choch < 0] = 4
    # Rows where the shifted target fell off the end of the series are
    # undefined, not "no event" - mark them so callers can drop them.
    label.iloc[len(label) - horizon:] = -1
    return label


def _prepare_training_frame(raw_df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    df = normalize_ohlcv_dataframe(raw_df)
    if len(df) < 100:
        raise ValueError(
            f"Need at least 100 bars to train a useful model, got {len(df)}. "
            "Pick a longer period / smaller interval, or upload more history."
        )
    out = run_smc_engine(df).df
    X = build_features(out)
    y = build_labels(out, horizon=horizon)

    mask = (y != -1) & X.notna().all(axis=1)
    return X.loc[mask], y.loc[mask]


def train_model(raw_df: pd.DataFrame, horizon: int = 1, source: str = "unknown") -> dict:
    """Trains a RandomForest classifier on `raw_df`, saves it to disk, and
    returns a JSON-serializable metrics summary."""
    X, y = _prepare_training_frame(raw_df, horizon)

    if y.nunique() < 2:
        raise ValueError(
            "Training data only contains a single class (likely no BOS/CHoCH "
            "events were detected). Try a longer history or different symbol."
        )

    split = int(len(X) * 0.8)
    split = max(split, 1)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    metrics: dict = {
        "rows_total": int(len(X)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "class_counts": {LABEL_NAMES[k]: int(v) for k, v in y.value_counts().sort_index().items()},
        "horizon_bars": horizon,
        "source": source,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    if len(X_test) > 0 and y_test.nunique() >= 1:
        y_pred = model.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            labels=list(LABEL_NAMES.keys()),
            target_names=list(LABEL_NAMES.values()),
            output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred, labels=list(LABEL_NAMES.keys()))
        metrics["accuracy"] = report["accuracy"]
        metrics["classification_report"] = {
            name: {k: v for k, v in vals.items()}
            for name, vals in report.items()
            if name in LABEL_NAMES.values()
        }
        metrics["confusion_matrix"] = cm.tolist()
        metrics["confusion_matrix_labels"] = list(LABEL_NAMES.values())

        non_none_mask = y_test != 0
        event_pred_mask = pd.Series(y_pred, index=y_test.index) != 0
        tp = int((non_none_mask & event_pred_mask).sum())
        fp = int((~non_none_mask & event_pred_mask).sum())
        fn = int((non_none_mask & ~event_pred_mask).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics["event_detection"] = {"precision": precision, "recall": recall, "f1": f1}

    importances = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_.tolist()),
        key=lambda kv: kv[1], reverse=True,
    )
    metrics["top_features"] = [{"feature": f, "importance": round(v, 4)} for f, v in importances[:10]]

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "label_names": LABEL_NAMES,
        "horizon": horizon,
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_PATH)

    return metrics


def load_model_bundle() -> Optional[dict]:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict(raw_df: pd.DataFrame) -> dict:
    """Runs the trained model over `raw_df`, returning candlestick data,
    the rule-based (ground-truth) BOS/CHoCH markers, and the model's
    per-bar predicted class + probabilities, plus a headline "next bar"
    forecast off the most recent row."""
    bundle = load_model_bundle()
    if bundle is None:
        raise RuntimeError("No trained model found. Train a model first.")

    df = normalize_ohlcv_dataframe(raw_df)
    smc_result = run_smc_engine(df)
    out = smc_result.df
    X = build_features(out)

    valid_mask = X.notna().all(axis=1)
    X_valid = X.loc[valid_mask, bundle["feature_columns"]]

    model = bundle["model"]
    label_names = bundle["label_names"]

    proba = model.predict_proba(X_valid)
    classes = model.classes_
    pred_class = classes[np.argmax(proba, axis=1)]

    pred_label = pd.Series("NONE", index=out.index, dtype=object)
    pred_confidence = pd.Series(0.0, index=out.index)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for i, cls in zip(X_valid.index, pred_class):
        pred_label.loc[i] = label_names[int(cls)]
    for i, row_proba in zip(X_valid.index, proba):
        pred_confidence.loc[i] = float(row_proba.max())

    prob_columns = {}
    for cls_id, name in label_names.items():
        if cls_id in class_to_idx:
            col = pd.Series(0.0, index=out.index)
            col.loc[X_valid.index] = proba[:, class_to_idx[cls_id]]
            prob_columns[name] = col
        else:
            prob_columns[name] = pd.Series(0.0, index=out.index)

    candles = {
        "time": [t.isoformat() for t in out.index],
        "open": out["open"].round(6).tolist(),
        "high": out["high"].round(6).tolist(),
        "low": out["low"].round(6).tolist(),
        "close": out["close"].round(6).tolist(),
        "volume": out["volume"].round(2).tolist(),
    }

    # Every structure break (both the long-lookback "swing" series, which
    # matches the `bos`/`choch` columns on `out`, and the short-lookback
    # "internal" series, which fires far more often) carries the full
    # pivot -> breakout geometry, so the UI can draw each one as an arrow
    # from the broken pivot to the bar that broke it instead of a bare
    # marker. `scope` lets the frontend filter/style swing vs internal
    # breaks independently.
    actual_events = [
        {
            "type": ev.tag,  # "BOS" | "CHoCH"
            "scope": "internal" if ev.internal else "swing",
            "direction": "bullish" if ev.bias > 0 else "bearish",
            "pivot_time": pd.Timestamp(ev.pivot_time).isoformat(),
            "pivot_level": float(ev.pivot_level),
            "breakout_time": pd.Timestamp(ev.bar_time).isoformat(),
        }
        for ev in smc_result.structure_break_events
    ]

    predictions = []
    for i, t in enumerate(out.index):
        if not valid_mask.iloc[i]:
            continue
        predictions.append({
            "time": t.isoformat(),
            "predicted_label": pred_label.iloc[i],
            "confidence": round(float(pred_confidence.iloc[i]), 4),
            "probabilities": {name: round(float(col.iloc[i]), 4) for name, col in prob_columns.items()},
        })

    last_valid_idx = X_valid.index[-1] if len(X_valid) else None
    next_bar_forecast = None
    if last_valid_idx is not None:
        i = out.index.get_loc(last_valid_idx)
        next_bar_forecast = {
            "based_on_bar_time": out.index[i].isoformat(),
            "horizon_bars": bundle["horizon"],
            "predicted_label": pred_label.iloc[i],
            "confidence": round(float(pred_confidence.iloc[i]), 4),
            "probabilities": {name: round(float(col.iloc[i]), 4) for name, col in prob_columns.items()},
        }

    return {
        "candles": candles,
        "actual_events": actual_events,
        "predictions": predictions,
        "next_bar_forecast": next_bar_forecast,
        "model_metrics": bundle["metrics"],
    }
