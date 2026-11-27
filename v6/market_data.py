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
from .twelvedata import TwelveDataError, fetch_candles as fetch_twelve_data_candles


SYMBOL = "EURUSD=X"
SOURCE = "Twelve Data primary; Yahoo Finance fallback"
# Short-lived process cache reduces duplicate provider calls and softens rate limits.
_CACHE_TTL_SECONDS = {"15m": 60, "60m": 300, "1d": 3600}
_cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
_rate_limited_until: dict[tuple[str, str], float] = {}
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
    key = (interval, period)
    now_mono = monotonic()
    with _cache_lock:
        cached = _cache.get(key)
        if cached and now_mono - cached[0] <= _CACHE_TTL_SECONDS.get(interval, 60):
            return cached[1].copy()
        if now_mono < _rate_limited_until.get(key, 0.0):
            if cached:
                return cached[1].copy()
            raise MarketDataError(f"{SOURCE} rate-limit cooldown active for {interval}")

    frame = None
    provider_error = None
    # Prefer Twelve Data when configured. If it is unavailable or rate-limited,
    # use Yahoo as a clearly labelled research fallback.
    if os.getenv("TWELVE_DATA_KEY", "").strip():
        tf = {"15m": "M15", "60m": "H1", "1d": "D1"}.get(interval)
        try:
            frame = fetch_twelve_data_candles(tf)
            frame.attrs["source"] = "Twelve Data"
        except (TwelveDataError, ValueError) as exc:
            provider_error = str(exc)
            if "429" in provider_error or "rate" in provider_error.lower() or "credit" in provider_error.lower():
                with _cache_lock:
                    _rate_limited_until[key] = monotonic() + 120

    if frame is None:
        try:
            frame = yf.Ticker(SYMBOL).history(
                period=period,
                interval=interval,
                auto_adjust=False,
                actions=False,
                raise_errors=True,
            )
            if frame is not None and not frame.empty:
                frame.attrs["source"] = "Yahoo Finance (yfinance)"
        except Exception as exc:
            name = type(exc).__name__
            if "RateLimit" in name or "TooManyRequests" in name:
                with _cache_lock:
                    _rate_limited_until[key] = monotonic() + 120
                provider_error = provider_error or f"Yahoo rate limit: {name}"
            else:
                provider_error = provider_error or f"Yahoo request failed: {name}"

    if frame is None or frame.empty:
        raise MarketDataError(provider_error or f"{SOURCE} returned no {interval} candles")
    with _cache_lock:
        _cache[key] = (monotonic(), frame.copy())
        _rate_limited_until.pop(key, None)
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
    include_candles: bool = True,
) -> dict[str, Any]:
    """Fetch provider intervals concurrently, derive H4 from H1, and report failures."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    result: dict[str, Any] = {
        "symbol": SYMBOL,
        "source": SOURCE,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "timeframes": {},
    }
    requested = tuple(dict.fromkeys(tf.upper() for tf in timeframes))
    provider_jobs = {
        tf: INTERVALS[tf]
        for tf in requested
        if tf in ("M15", "H1", "D1")
    }
    raw_frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    # Parallelize independent provider calls; H4 is derived locally from H1.
    with ThreadPoolExecutor(max_workers=max(1, len(provider_jobs))) as pool:
        futures = {
            pool.submit(_download, *interval): tf
            for tf, interval in provider_jobs.items()
        }
        for future in as_completed(futures):
            tf = futures[future]
            try:
                raw_frames[tf] = future.result()
            except (MarketDataError, ValueError) as exc:
                errors[tf] = str(exc)
            except Exception as exc:
                errors[tf] = f"Provider request failed: {type(exc).__name__}"

    h1_base: tuple[pd.DataFrame, DataQuality] | None = None
    if "H1" in raw_frames:
        h1_clean, h1_quality = normalise_candles(
            raw_frames["H1"], "H1", SOURCE, minimum_candles=minimum_candles
        )
        h1_base = (h1_clean, h1_quality)

    for tf in requested:
        try:
            if tf == "H4":
                if h1_base is None:
                    raise MarketDataError(errors.get("H1", "H1 data unavailable for H4 aggregation"))
                candles, metadata = fetch_timeframe(
                    "H4", minimum_candles=minimum_candles, base_h1=h1_base
                )
            elif tf in errors:
                raise MarketDataError(errors[tf])
            elif tf in raw_frames:
                candles, q = normalise_candles(
                    raw_frames[tf], tf, SOURCE, minimum_candles=minimum_candles
                )
                quality = q.as_dict()
                if tf == "D1":
                    quality["session_boundary"] = "Yahoo Finance provider-defined daily session"
                    quality["broker_alignment_verified"] = False
                metadata = {
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
            else:
                raise ValueError(f"Unsupported timeframe: {tf}")

            item = {
                "status": "ready" if metadata["quality"]["ready"] else "not_ready",
                "metadata": metadata,
            }
            if include_candles:
                item["candles"] = [
                    {
                        "time": timestamp.isoformat(),
                        "open": float(row.open),
                        "high": float(row.high),
                        "low": float(row.low),
                        "close": float(row.close),
                    }
                    for timestamp, row in candles.iterrows()
                ]
            result["timeframes"][tf] = item
        except (MarketDataError, ValueError) as exc:
            result["timeframes"][tf] = {
                "status": "unavailable",
                "error": str(exc),
                **({"candles": []} if include_candles else {}),
            }
    return result
