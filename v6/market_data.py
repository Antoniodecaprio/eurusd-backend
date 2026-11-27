"""Best-effort Yahoo Finance EUR/USD candle adapter for v6.

Yahoo Finance is a free research feed, not an executable broker quote source.
Do not use it to calculate final MT5 entry prices, spread, pip value or fills.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from .data_quality import normalise_candles, resample_completed


SYMBOL = "EURUSD=X"
SOURCE = "Yahoo Finance (yfinance)"
INTERVALS = {
    "M15": ("15m", "60d"),
    "H1": ("60m", "730d"),
    "D1": ("1d", "max"),
}


class MarketDataError(RuntimeError):
    """Raised when the upstream provider cannot supply usable candle data."""


def _download(interval: str, period: str) -> pd.DataFrame:
    try:
        frame = yf.Ticker(SYMBOL).history(
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=False,
            raise_errors=True,
        )
    except Exception as exc:
        raise MarketDataError(f"{SOURCE} request failed: {type(exc).__name__}") from exc
    if frame is None or frame.empty:
        raise MarketDataError(f"{SOURCE} returned no {interval} candles")
    return frame


def fetch_timeframe(
    timeframe: str,
    *,
    now: pd.Timestamp | None = None,
    minimum_candles: int = 100,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch and validate one timeframe; H4 is aggregated from H1 candles.

    D1 is kept in the provider's daily-session convention. Its session
    boundary is not claimed to match every MT5 broker's daily candle.
    """
    tf = timeframe.upper()
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")

    if tf == "H4":
        base = _download(*INTERVALS["H1"])
        clean_h1, base_quality = normalise_candles(
            base, "H1", SOURCE, now=current, minimum_candles=minimum_candles * 4
        )
        candles = resample_completed(clean_h1, "H1", "H4", now=current)
        last_close = (
            candles.index[-1] + pd.Timedelta(hours=4)
            if not candles.empty else None
        )
        age = (
            max(0.0, (current - last_close).total_seconds())
            if last_close is not None else None
        )
        ready = bool(len(candles) >= minimum_candles and age is not None
                     and age <= pd.Timedelta(hours=10).total_seconds())
        quality = {
            "timeframe": "H4",
            "source": SOURCE,
            "base_timeframe": "H1",
            "candle_count": int(len(candles)),
            "last_candle_close_utc": last_close.isoformat() if last_close is not None else None,
            "age_seconds": age,
            "stale": not ready and last_close is not None,
            "ready": ready,
            "reasons": [] if ready else ["insufficient_or_stale_completed_h4_candles"],
            "base_quality": base_quality.as_dict(),
        }
    else:
        if tf not in INTERVALS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        interval, period = INTERVALS[tf]
        raw = _download(interval, period)
        candles, q = normalise_candles(
            raw, tf, SOURCE, now=current, minimum_candles=minimum_candles
        )
        quality = q.as_dict()
        if tf == "D1":
            quality["session_boundary"] = "Yahoo Finance provider-defined daily session"
            quality["broker_alignment_verified"] = False

    if candles.empty:
        raise MarketDataError(f"No completed {tf} candles available")
    return candles, {
        "symbol": SYMBOL,
        "timeframe": tf,
        "source": SOURCE,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "quality": quality,
        "limitations": [
            "Free third-party research feed; availability and history limits may change.",
            "Not a broker quote; spread, slippage and executable prices are unavailable.",
            "Provider data may be delayed, revised, rate-limited or temporarily unavailable.",
        ],
    }


def fetch_market_snapshot(
    timeframes: tuple[str, ...] = ("M15", "H1", "H4", "D1"),
    *,
    minimum_candles: int = 100,
) -> dict[str, Any]:
    """Fetch requested horizons independently and report per-timeframe failures."""
    result: dict[str, Any] = {
        "symbol": SYMBOL,
        "source": SOURCE,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "timeframes": {},
    }
    for timeframe in timeframes:
        tf = timeframe.upper()
        try:
            candles, metadata = fetch_timeframe(tf, minimum_candles=minimum_candles)
            result["timeframes"][tf] = {
                "status": "ready" if metadata["quality"]["ready"] else "not_ready",
                "metadata": metadata,
                "candles": [
                    {
                        "time": timestamp.isoformat(),
                        "open": float(row.open),
                        "high": float(row.high),
                        "low": float(row.low),
                        "close": float(row.close),
                    }
                    for timestamp, row in candles.iterrows()
                ],
            }
        except (MarketDataError, ValueError) as exc:
            result["timeframes"][tf] = {
                "status": "unavailable",
                "error": str(exc),
                "candles": [],
            }
    return result
