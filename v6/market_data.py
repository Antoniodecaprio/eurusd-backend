"""Best-effort Yahoo Finance EUR/USD candle adapter for v6.

Yahoo Finance is a free research feed, not an executable broker quote source.
Do not use it to calculate final MT5 entry prices, spread, pip value or fills.
"""
from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from time import monotonic
from typing import Any

import pandas as pd
import yfinance as yf

from .data_quality import DataQuality, normalise_candles, resample_completed


SYMBOL = "EURUSD=X"
SOURCE = "Yahoo Finance (yfinance)"
# Short-lived process cache reduces duplicate provider calls and softens rate limits.
_CACHE_TTL_SECONDS = {"15m": 60, "60m": 300, "1d": 3600}
_cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
_rate_limited_until = 0.0
_cache_lock = Lock()
INTERVALS = {
    "M15": ("15m", "60d"),
    "H1": ("60m", "730d"),
    "D1": ("1d", "max"),
}


class MarketDataError(RuntimeError):
    """Raised when the upstream provider cannot supply usable candle data."""


def _download(interval: str, period: str) -> pd.DataFrame:
    """Download with a small in-process cache and rate-limit cooldown."""
    global _rate_limited_until
    key = (interval, period)
    now_mono = monotonic()
    with _cache_lock:
        cached = _cache.get(key)
        if cached and now_mono - cached[0] <= _CACHE_TTL_SECONDS.get(interval, 60):
            return cached[1].copy()
        if now_mono < _rate_limited_until:
            if cached:
                return cached[1].copy()
            raise MarketDataError(f"{SOURCE} rate-limit cooldown active for {interval}")

    try:
        frame = yf.Ticker(SYMBOL).history(
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=False,
            raise_errors=True,
        )
    except Exception as exc:
        name = type(exc).__name__
        if "RateLimit" in name or "TooManyRequests" in name:
            with _cache_lock:
                _rate_limited_until = monotonic() + 120
                cached = _cache.get(key)
            if cached:
                return cached[1].copy()
        raise MarketDataError(f"{SOURCE} request failed: {name}") from exc
    if frame is None or frame.empty:
        raise MarketDataError(f"{SOURCE} returned no {interval} candles")
    with _cache_lock:
        _cache[key] = (monotonic(), frame.copy())
        _rate_limited_until = 0.0
    return frame


def fetch_timeframe(
    timeframe: str,
    *,
    now: pd.Timestamp | None = None,
    minimum_candles: int = 100,
    base_h1: tuple[pd.DataFrame, DataQuality] | None = None,
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
        if base_h1 is None:
            base = _download(*INTERVALS["H1"])
            clean_h1, base_quality = normalise_candles(
                base, "H1", SOURCE, now=current, minimum_candles=minimum_candles * 4
            )
        else:
            clean_h1, base_quality = base_h1
        candles = resample_completed(clean_h1, "H1", "H4", now=current)
        last_close = (
            candles.index[-1] + pd.Timedelta(hours=4)
            if not candles.empty else None
        )
        age = (
            max(0.0, (current - last_close).total_seconds())
            if last_close is not None else None
        )
        h4_stale = age is None or age > pd.Timedelta(hours=6).total_seconds()
        ready = bool(len(candles) >= minimum_candles and not h4_stale and base_quality.ready)
        quality = {
            "timeframe": "H4",
            "source": SOURCE,
            "base_timeframe": "H1",
            "candle_count": int(len(candles)),
            "last_candle_close_utc": last_close.isoformat() if last_close is not None else None,
            "age_seconds": age,
            "stale": h4_stale,
            "ready": ready,
            "reasons": [] if ready else ["h4_or_base_h1_quality_check_failed"],
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
