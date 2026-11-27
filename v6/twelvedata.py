"""Twelve Data EUR/USD candle adapter.

Uses one request per interval, a bounded HTTP timeout, and explicit provider
error handling. Never logs or returns the API key.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://api.twelvedata.com/time_series"
SYMBOL = "EUR/USD"
INTERVALS = {"M15": "15min", "H1": "1h", "D1": "1day"}


class TwelveDataError(RuntimeError):
    """Twelve Data is not configured or did not return usable candles."""


def fetch_candles(timeframe: str, *, outputsize: int = 5000) -> pd.DataFrame:
    """Fetch EUR/USD candles from Twelve Data, normalized to a UTC index."""
    api_key = os.getenv("TWELVE_DATA_KEY", "").strip()
    if not api_key:
        raise TwelveDataError("TWELVE_DATA_KEY is not configured")

    tf = timeframe.upper()
    interval = INTERVALS.get(tf)
    if interval is None:
        raise ValueError(f"Twelve Data does not directly provide {tf}")

    try:
        response = requests.get(
            BASE_URL,
            params={
                "symbol": SYMBOL,
                "interval": interval,
                "outputsize": outputsize,
                "timezone": "UTC",
                "format": "JSON",
                "apikey": api_key,
            },
            timeout=(4, 12),
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except requests.Timeout as exc:
        raise TwelveDataError("Twelve Data request timed out") from exc
    except requests.RequestException as exc:
        raise TwelveDataError(f"Twelve Data transport error: {type(exc).__name__}") from exc
    except ValueError as exc:
        raise TwelveDataError("Twelve Data returned invalid JSON") from exc

    if payload.get("status") == "error" or "code" in payload and "values" not in payload:
        message = str(payload.get("message", "provider rejected the request"))
        code = payload.get("code", "unknown")
        # Avoid echoing request URLs or credentials in error messages.
        raise TwelveDataError(f"Twelve Data error {code}: {message[:180]}")

    values = payload.get("values")
    if not isinstance(values, list) or not values:
        raise TwelveDataError("Twelve Data returned no candle values")

    frame = pd.DataFrame(values)
    required = {"datetime", "open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        raise TwelveDataError("Twelve Data response is missing OHLC fields")

    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["datetime", "open", "high", "low", "close"])
    frame = frame.set_index("datetime").sort_index()
    frame = frame.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close"
    })
    frame.attrs["source"] = "Twelve Data"
    frame.attrs["symbol"] = SYMBOL
    if frame.empty:
        raise TwelveDataError("Twelve Data returned no valid OHLC candles")
    return frame
