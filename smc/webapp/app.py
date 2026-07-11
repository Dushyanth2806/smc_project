"""
app.py
------
Flask web app exposing the BOS/CHoCH ML pipeline (`ml_engine.py`) through
a small JSON API plus a single-page UI (`templates/index.html`):

    GET  /                          -> UI
    GET  /api/model/status          -> is a model trained? with what metrics?
    POST /api/train/tradingview     -> {symbol, exchange, interval, n_bars, horizon} -> train
    POST /api/train/csv             -> multipart CSV upload -> train
    POST /api/predict/tradingview   -> {symbol, exchange, interval, n_bars} -> predict
    POST /api/predict/csv           -> multipart CSV upload -> predict

Run with:  python app.py   (serves on http://127.0.0.1:5000)
"""

from __future__ import annotations

import io
import logging
import traceback
import warnings
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from dotenv import load_dotenv

# Loads TV_USERNAME / TV_PASSWORD (and anything else) from a local .env
# file next to this script, if one exists. The .env file itself is
# git-ignored - never commit real credentials into a tracked file.
load_dotenv(Path(__file__).resolve().parent / ".env")

import pandas as pd
from flask import Flask, jsonify, render_template, request

import ml_engine

# Harmless sklearn/joblib version-mismatch warning triggered by
# RandomForestClassifier's internal n_jobs=-1 parallelism - doesn't
# affect correctness, just noisy in the server log.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model/status")
def model_status():
    bundle = ml_engine.load_model_bundle()
    if bundle is None:
        return jsonify({"trained": False})
    return jsonify({"trained": True, "metrics": bundle["metrics"]})


def _read_uploaded_csv() -> pd.DataFrame:
    file = request.files.get("file")
    if file is None or file.filename == "":
        raise ValueError("No CSV file was uploaded (form field name must be 'file').")
    return pd.read_csv(io.BytesIO(file.read()))


@app.route("/api/train/tradingview", methods=["POST"])
def train_tradingview():
    body = request.get_json(force=True, silent=True) or {}
    symbol = str(body.get("symbol", "")).strip()
    exchange = str(body.get("exchange", "NSE")).strip()
    interval = str(body.get("interval", "15m")).strip()
    n_bars = int(body.get("n_bars", 2000))
    horizon = int(body.get("horizon", 1))

    if not symbol:
        return jsonify({"error": "symbol is required"}), 400

    try:
        raw = ml_engine.fetch_tradingview(symbol, exchange=exchange, interval=interval, n_bars=n_bars)
        metrics = ml_engine.train_model(
            raw, horizon=horizon, source=f"tradingview:{exchange}:{symbol}:{interval}:{n_bars}bars"
        )
        return jsonify({"ok": True, "metrics": metrics})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 400


@app.route("/api/train/csv", methods=["POST"])
def train_csv():
    horizon = int(request.form.get("horizon", 1))
    try:
        raw = _read_uploaded_csv()
        metrics = ml_engine.train_model(raw, horizon=horizon, source="csv_upload")
        return jsonify({"ok": True, "metrics": metrics})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 400


@app.route("/api/predict/tradingview", methods=["POST"])
def predict_tradingview():
    body = request.get_json(force=True, silent=True) or {}
    symbol = str(body.get("symbol", "")).strip()
    exchange = str(body.get("exchange", "NSE")).strip()
    interval = str(body.get("interval", "15m")).strip()
    n_bars = int(body.get("n_bars", 300))

    if not symbol:
        return jsonify({"error": "symbol is required"}), 400

    try:
        raw = ml_engine.fetch_tradingview(symbol, exchange=exchange, interval=interval, n_bars=n_bars)
        result = ml_engine.predict(raw)
        return jsonify({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 400


@app.route("/api/predict/csv", methods=["POST"])
def predict_csv():
    try:
        raw = _read_uploaded_csv()
        result = ml_engine.predict(raw)
        return jsonify({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
