"""Version 6 market-data validation primitives.

This module deliberately does not fetch prices or emit trading signals. It
normalises provider candles, removes incomplete bars, and reports data quality
so forecast code can fail safely before using market observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


INTERVALS = {
    "M15": pd.Timedelta(minutes=15),
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
}

REQUIRED_OHLC = ("open", "high", "low", "close")
MAX_AGE = {
    "M15": pd.Timedelta(minutes=30),
    "H1": pd.Timedelta(minutes=90),
    "H4": pd.Timedelta(hours=6),
    "D1": pd.Timedelta(hours=36),
}


@dataclass(frozen=True)
class DataQuality:
    timeframe: str
    source: str
    candle_count: int
    last_candle_close_utc: Optional[str]
    age_seconds: Optional[float]
    stale: bool
    duplicate_count: int
    invalid_count: int
    missing_interval_count: int
    ready: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "timeframe": self.timeframe,
            "source": self.source,
            "candle_count": self.candle_count,
            "last_candle_close_utc": self.last_candle_close_utc,
            "age_seconds": self.age_seconds,
            "stale": self.stale,
            "duplicate_count": self.duplicate_count,
            "invalid_count": self.invalid_count,
            "missing_interval_count": self.missing_interval_count,
            "missing_interval_interpretation": (
                "Raw elapsed-time gaps; not session-aware, so weekends and market closures may be counted."
                if self.timeframe == "D1"
                else "Raw elapsed-time gaps; FX market closures and provider/session gaps may be counted."
            ),
            "ready": self.ready,
            "reasons": list(self.reasons),
        }


def interval_delta(timeframe: str) -> pd.Timedelta:
    try:
        return INTERVALS[timeframe.upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe}") from exc


def normalise_candles(
    frame: pd.DataFrame,
    timeframe: str,
    source: str,
    now: Optional[pd.Timestamp] = None,
    stale_after_intervals: float | None = None,
    minimum_candles: int = 100,
) -> tuple[pd.DataFrame, DataQuality]:
    """Return validated, completed OHLC candles indexed by UTC candle-open time.

    Candle timestamps are interpreted as candle-open times. An observation is
    complete only when open_time + interval <= now. Missing intervals are
    counted as diagnostics, not automatically treated as errors, because FX
    markets close on weekends and have provider/session gaps.
    """
    delta = interval_delta(timeframe)
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")

    reasons: list[str] = []
    if frame is None or frame.empty:
        empty = pd.DataFrame(columns=list(REQUIRED_OHLC))
        quality = DataQuality(timeframe.upper(), source, 0, None, None, True,
                              0, 0, 0, False, ("no_candles_received",))
        return empty, quality

    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(col[-1]).strip().lower() for col in data.columns]
    else:
        data.columns = [str(col).strip().lower() for col in data.columns]
    missing = [col for col in REQUIRED_OHLC if col not in data.columns]
    if missing:
        raise ValueError("Missing OHLC columns: " + ", ".join(missing))

    raw_index = pd.to_datetime(data.index, utc=True, errors="coerce")
    data.index = pd.DatetimeIndex(raw_index)
    data = data.loc[~data.index.isna()].copy()
    duplicate_count = int(data.index.duplicated(keep="last").sum())
    data = data.loc[~data.index.duplicated(keep="last")].sort_index()

    for col in REQUIRED_OHLC:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    numeric = data[list(REQUIRED_OHLC)].to_numpy(dtype=float)
    finite = np.isfinite(numeric).all(axis=1)
    positive = (data[list(REQUIRED_OHLC)] > 0).all(axis=1)
    envelope = (
        (data["high"] >= data[["open", "close", "low"]].max(axis=1))
        & (data["low"] <= data[["open", "close", "high"]].min(axis=1))
    )
    valid = finite & positive.to_numpy() & envelope.to_numpy()
    invalid_count = int((~valid).sum())
    data = data.loc[valid].copy()

    # Keep only bars whose full interval has elapsed.
    data = data.loc[(data.index + delta) <= current].copy()
    if data.empty:
        reasons.append("no_completed_candles")
        last_close = None
        age_seconds = None
        stale = True
        missing_count = 0
    else:
        last_open = data.index[-1]
        last_close_ts = last_open + delta
        age_seconds = max(0.0, (current - last_close_ts).total_seconds())
        stale = age_seconds > MAX_AGE[timeframe.upper()].total_seconds()
        last_close = last_close_ts.isoformat()
        if stale:
            reasons.append("latest_completed_candle_is_stale")
        if duplicate_count:
            reasons.append("duplicate_timestamps_removed")
        if invalid_count:
            reasons.append("invalid_ohlc_rows_removed")
        # Count gaps in the observed series; FX weekends naturally create gaps.
        diffs = data.index.to_series().diff().dropna()
        missing_count = int(
            sum(max(0, int(round(diff / delta)) - 1) for diff in diffs if diff > delta)
        )

    if len(data) < minimum_candles:
        reasons.append("insufficient_completed_candles")
    ready = bool(
        len(data) >= minimum_candles
        and not stale
        and duplicate_count == 0
        and invalid_count == 0
    )
    if duplicate_count:
        reasons.append("duplicate_rows_present")
    if invalid_count:
        reasons.append("invalid_rows_present")
    quality = DataQuality(
        timeframe=timeframe.upper(),
        source=source,
        candle_count=int(len(data)),
        last_candle_close_utc=last_close,
        age_seconds=age_seconds,
        stale=stale,
        duplicate_count=duplicate_count,
        invalid_count=invalid_count,
        missing_interval_count=missing_count,
        ready=ready,
        reasons=tuple(dict.fromkeys(reasons)),
    )
    return data, quality


def resample_completed(
    frame: pd.DataFrame,
    source_timeframe: str,
    target_timeframe: str,
    now: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Aggregate smaller candles into UTC-aligned, fully completed larger bars.

    This is appropriate for intraday M15->H1/H4 aggregation. D1 should be
    sourced from a provider's FX daily candles so its broker/session boundary
    is not guessed from UTC midnight.
    """
    source_delta = interval_delta(source_timeframe)
    target_delta = interval_delta(target_timeframe)
    if target_delta <= source_delta or target_timeframe.upper() == "D1":
        raise ValueError("Target must be a larger intraday timeframe (H1 or H4)")
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(REQUIRED_OHLC))

    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(col[-1]).strip().lower() for col in data.columns]
    else:
        data.columns = [str(col).strip().lower() for col in data.columns]
    missing = [col for col in REQUIRED_OHLC if col not in data.columns]
    if missing:
        raise ValueError("Missing OHLC columns: " + ", ".join(missing))
    data = data.sort_index()
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    data = data.loc[(data.index + source_delta) <= current]
    if data.empty:
        return pd.DataFrame(columns=list(REQUIRED_OHLC))

    bars = data.resample(target_delta, origin="epoch", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    counts = data["close"].resample(
        target_delta, origin="epoch", label="left", closed="left"
    ).count()
    expected = int(target_delta / source_delta)
    bars = bars.loc[counts == expected].dropna(subset=list(REQUIRED_OHLC))
    bars = bars.loc[(bars.index + target_delta) <= current]
    return bars
