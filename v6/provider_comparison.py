"""Independent Twelve Data and Yahoo Finance evidence for EUR/USD.

This module compares research-feed observations only. It does not select a
preferred provider, create a trading signal, or average prices into a quote.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

import pandas as pd
import yfinance as yf

from .data_quality import normalise_candles
from .twelvedata import fetch_candles as fetch_twelve_data_candles


YAHOO_SYMBOL = "EURUSD=X"
YAHOO_INTERVALS = {
    "M15": ("15m", "60d"),
    "H1": ("60m", "730d"),
}
TIMEFRAME_PIPS = 0.0001
# Provisional research thresholds, not universal market-quality standards.
# Keep the observed deviation visible even when it exceeds this tolerance.
PROVIDER_DEVIATION_TOLERANCE_PIPS = {"M15": 0.5, "H1": 1.0}


def _deviation_tolerance(timeframe: str) -> float:
    """Read a non-negative per-timeframe override, falling back to safe defaults."""
    default = PROVIDER_DEVIATION_TOLERANCE_PIPS[timeframe]
    raw = os.getenv(f"EURUSD_PROVIDER_TOLERANCE_{timeframe}_PIPS", "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if pd.notna(value) and value >= 0 and value != float("inf") else default


def _observations(frame: pd.DataFrame, timeframe: str, provider: str,
                  now: pd.Timestamp) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean, quality = normalise_candles(
        frame, timeframe, provider, now=now, minimum_candles=1
    )
    delta = pd.Timedelta(minutes=15) if timeframe == "M15" else pd.Timedelta(hours=1)
    rows = []
    retrieved = datetime.now(timezone.utc).isoformat()
    for timestamp, candle in clean.iterrows():
        rows.append({
            "provider": provider,
            "symbol": "EUR/USD",
            "timeframe": timeframe,
            "candle_open_utc": timestamp.isoformat(),
            "candle_close_utc": (timestamp + delta).isoformat(),
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
            "retrieved_at_utc": retrieved,
            "quality": "valid_completed",
            "error": None,
        })
    return rows, quality.as_dict()


def compare_provider_frames(twelve_frame: pd.DataFrame | None,
                            yahoo_frame: pd.DataFrame | None,
                            timeframe: str, *,
                            now: pd.Timestamp | None = None) -> dict[str, Any]:
    """Preserve each feed and compare only exact matching completed candle opens."""
    tf = timeframe.upper()
    if tf not in YAHOO_INTERVALS:
        raise ValueError(f"Unsupported comparison timeframe: {timeframe}")
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")

    providers: dict[str, dict[str, Any]] = {}
    for name, frame in (("Twelve Data", twelve_frame), ("Yahoo Finance", yahoo_frame)):
        if frame is None or frame.empty:
            providers[name] = {
                "status": "unavailable", "observations": [],
                "quality": None, "error": "No data returned",
            }
            continue
        try:
            rows, quality = _observations(frame, tf, name, current)
            providers[name] = {
                "status": "available" if rows else "unavailable",
                "observations": rows,
                "quality": quality,
                "error": None if rows else "No completed valid candles",
            }
        except Exception as exc:
            providers[name] = {
                "status": "unavailable", "observations": [],
                "quality": None, "error": f"Invalid provider data: {type(exc).__name__}",
            }

    left = {row["candle_open_utc"]: row for row in providers["Twelve Data"]["observations"]}
    right = {row["candle_open_utc"]: row for row in providers["Yahoo Finance"]["observations"]}
    common = sorted(set(left).intersection(right))
    latest_diff = None
    if common:
        latest = common[-1]
        latest_diff = abs(left[latest]["close"] - right[latest]["close"]) / TIMEFRAME_PIPS

    tolerance = _deviation_tolerance(tf)
    deviation_status = (
        "not_comparable" if latest_diff is None else
        "within_tolerance" if latest_diff <= tolerance else
        "above_tolerance"
    )

    both_available = all(providers[name]["status"] == "available" for name in providers)
    comparison_status = (
        "compared" if common else
        "no_matching_candles" if both_available else
        "insufficient_provider_data"
    )
    return {
        "symbol": "EUR/USD",
        "timeframe": tf,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "providers": providers,
        "comparison": {
            "status": comparison_status,
            "matched_candles": len(common),
            "latest_matched_candle_open_utc": common[-1] if common else None,
            "latest_close_difference_pips": latest_diff,
            "difference_unit": "pips (0.0001 EUR/USD price units)",
            "deviation_tolerance_pips": tolerance if latest_diff is not None else None,
            "deviation_status": deviation_status,
            "deviation_tolerance_note": (
                "Provisional environment-configurable research threshold; exceeding it flags "
                "provider disagreement but does not reject either feed."
            ),
            "consensus_close": None,
            "consensus_note": "No consensus quote is produced in Phase 1A; provider observations remain separate.",
        },
        "limitations": [
            "Twelve Data and Yahoo Finance are independent research feeds, not broker bid/ask quotes.",
            "Exact timestamp matching does not prove identical venue, liquidity or candle-session conventions.",
            "No provider is assumed to be correct when prices disagree.",
            "Only completed, valid OHLC candles are included.",
        ],
    }


def fetch_provider_evidence(timeframe: str, *,
                            now: pd.Timestamp | None = None) -> dict[str, Any]:
    """Fetch both providers independently; either result survives the other's failure."""
    tf = timeframe.upper()
    if tf not in YAHOO_INTERVALS:
        raise ValueError(f"Unsupported comparison timeframe: {timeframe}")
    interval, period = YAHOO_INTERVALS[tf]
    twelve_frame = None
    yahoo_frame = None
    twelve_error = None
    yahoo_error = None

    try:
        twelve_frame = fetch_twelve_data_candles(tf)
    except Exception as exc:
        twelve_error = f"Twelve Data request failed: {type(exc).__name__}"

    try:
        yahoo_frame = yf.Ticker(YAHOO_SYMBOL).history(
            period=period, interval=interval, auto_adjust=False,
            actions=False, raise_errors=True,
        )
        if yahoo_frame is not None and yahoo_frame.empty:
            yahoo_frame = None
            yahoo_error = "Yahoo Finance returned no candles"
    except Exception as exc:
        yahoo_error = f"Yahoo Finance request failed: {type(exc).__name__}"

    result = compare_provider_frames(twelve_frame, yahoo_frame, tf, now=now)
    if twelve_error:
        result["providers"]["Twelve Data"]["error"] = twelve_error
    if yahoo_error:
        result["providers"]["Yahoo Finance"]["error"] = yahoo_error
    if twelve_frame is None and yahoo_frame is None:
        result["comparison"]["status"] = "unavailable"
    return result
