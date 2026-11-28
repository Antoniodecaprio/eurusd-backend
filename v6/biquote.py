"""biquote public EUR/USD OHLC adapter (read-only)."""
from __future__ import annotations

import pandas as pd
import requests

BASE_URL = "https://biquote.io/api/EURUSD/ohlc"
INTERVALS = {"M15": "15m", "H1": "1h"}


class BiquoteError(RuntimeError):
    """biquote is unavailable or returned an invalid candle payload."""


def fetch_candles(timeframe: str, *, limit: int = 192) -> pd.DataFrame:
    tf = timeframe.upper()
    interval = INTERVALS.get(tf)
    if interval is None:
        raise ValueError(f"biquote does not support {timeframe}")
    try:
        response = requests.get(
            BASE_URL, params={"interval": interval, "limit": min(max(limit, 1), 192 if tf == "M15" else 168)},
            timeout=(4, 12),
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise BiquoteError("biquote request timed out") from exc
    except requests.RequestException as exc:
        raise BiquoteError(f"biquote transport error: {type(exc).__name__}") from exc
    except ValueError as exc:
        raise BiquoteError("biquote returned invalid JSON") from exc

    bars = payload.get("bars") if isinstance(payload, dict) else None
    if not isinstance(bars, list):
        raise BiquoteError("biquote response is missing bars")
    rows = []
    for bar in bars:
        if not isinstance(bar, dict) or bar.get("isOpen") is True:
            continue
        rows.append(bar)
    if not rows:
        raise BiquoteError("biquote returned no completed candles")

    frame = pd.DataFrame(rows)
    required = {"openTime", "open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        raise BiquoteError("biquote candles are missing OHLC fields")
    frame["openTime"] = pd.to_datetime(frame["openTime"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["openTime", "open", "high", "low", "close"])
    frame = frame.set_index("openTime").sort_index()
    frame = frame.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"})
    frame.attrs.update({"source":"biquote", "symbol":"EURUSD", "price_basis":"mid",
                        "upstream_lineage":"Yahoo bootstrap + realtime tick aggregation + MT5 historical feeds"})
    if frame.empty:
        raise BiquoteError("biquote returned no valid completed OHLC candles")
    return frame
