"""biquote live EUR/USD quote adapter and optional SignalR stream.

Live quotes are kept separate from OHLC research candles. Upstream broker identity
is deliberately marked unverified until biquote confirms the exact MT5 source.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import threading
from datetime import datetime, timezone
from typing import Any, Callable

import requests

BASE_URL = "https://biquote.io"
QUOTE_URL = f"{BASE_URL}/api/EURUSD"
HUB_URL = f"{BASE_URL}/hubs/tick"
MAX_QUOTE_AGE_SECONDS = float(os.getenv("BIQUOTE_MAX_QUOTE_AGE_SECONDS", "30"))
MAX_SPREAD_PIPS = float(os.getenv("BIQUOTE_MAX_SPREAD_PIPS", "5.0"))
PIP_SIZE = 0.0001
logger = logging.getLogger("eurusd_v6.biquote_live")


class BiquoteLiveError(RuntimeError):
    """biquote quote is unavailable or invalid."""


def _parse_timestamp(value: Any) -> datetime:
    if value is None:
        raise BiquoteLiveError("quote_timestamp_missing")
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError) as exc:
        raise BiquoteLiveError("quote_timestamp_invalid") from exc


def validate_quote(payload: dict[str, Any], *, received_at: datetime | None = None) -> dict[str, Any]:
    """Validate a raw biquote tick without inventing a broker identity."""
    if not isinstance(payload, dict):
        raise BiquoteLiveError("quote_payload_invalid")
    symbol = str(payload.get("symbol", "EURUSD")).upper().replace("/", "")
    if symbol != "EURUSD":
        raise BiquoteLiveError("unexpected_symbol")
    try:
        bid, ask = float(payload["bid"]), float(payload["ask"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BiquoteLiveError("bid_ask_missing_or_invalid") from exc
    if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask <= 0:
        raise BiquoteLiveError("non_positive_or_non_finite_price")
    if ask < bid:
        raise BiquoteLiveError("ask_below_bid")
    stamp = _parse_timestamp(
        payload.get("timestamp") or payload.get("time") or payload.get("updatedAt")
    )
    received = received_at or datetime.now(timezone.utc)
    if received.tzinfo is None:
        received = received.replace(tzinfo=timezone.utc)
    received = received.astimezone(timezone.utc)
    age = (received - stamp).total_seconds()
    # Small future skew is tolerated; materially future-dated ticks are rejected.
    if age < -5:
        raise BiquoteLiveError("quote_timestamp_in_future")
    spread_pips = (ask - bid) / PIP_SIZE
    if spread_pips > MAX_SPREAD_PIPS:
        raise BiquoteLiveError("spread_above_configured_limit")
    return {
        "provider": "biquote",
        "symbol": "EURUSD",
        "instrument": "EUR/USD",
        "data_type": "live_quote",
        "upstream_source": str(payload.get("source") or "biquote_source_unspecified"),
        "provider_market_state": payload.get("marketState"),
        "provider_stale": bool(payload.get("stale", False)),
        "market_open": str(payload.get("marketState", "unknown")).lower() == "open",
        "provider_quote_age_seconds": payload.get("quoteAgeSeconds"),
        "source_verified": False,
        "price_basis": "bid_ask",
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2.0,
        "spread": ask - bid,
        "spread_pips": spread_pips,
        "timestamp_utc": stamp.isoformat(),
        "received_at_utc": received.isoformat(),
        "age_seconds": max(0.0, age),
        "is_live": age <= MAX_QUOTE_AGE_SECONDS and not bool(payload.get("stale", False)) and str(payload.get("marketState", "unknown")).lower() == "open",
        "is_stale": age > MAX_QUOTE_AGE_SECONDS or bool(payload.get("stale", False)),
        "historical_consensus_eligible": False,
        "trading_enabled": False,
    }


def fetch_live_quote(*, timeout: tuple[float, float] = (4, 10)) -> dict[str, Any]:
    """Fetch one REST quote; intended for initial snapshot and recovery only."""
    try:
        response = requests.get(QUOTE_URL, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise BiquoteLiveError("biquote_quote_timeout") from exc
    except requests.RequestException as exc:
        raise BiquoteLiveError(f"biquote_transport_error:{type(exc).__name__}") from exc
    except ValueError as exc:
        raise BiquoteLiveError("biquote_invalid_json") from exc
    return validate_quote(payload)


class BiquoteSignalRStream:
    """Managed SignalR consumer; reconnects and re-subscribes after reconnects."""

    def __init__(self, on_quote: Callable[[dict[str, Any]], None]):
        self.on_quote = on_quote
        self._client = None
        self.connected = False
        self.last_connected_at: str | None = None
        self.last_message_at: str | None = None
        self.last_error: str | None = None
        self.reconnect_count = 0

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "connected": self.connected,
            "last_connected_at_utc": self.last_connected_at,
            "last_message_at_utc": self.last_message_at,
            "last_error": self.last_error,
            "reconnect_count": self.reconnect_count,
        }

    async def run(self) -> None:
        try:
            from pysignalr.client import SignalRClient
        except ImportError as exc:
            self.last_error = "pysignalr_not_installed"
            raise RuntimeError("Install pysignalr to enable the biquote SignalR stream") from exc

        client = SignalRClient(HUB_URL)
        self._client = client

        async def receive(message: Any) -> None:
            candidate = message[0] if isinstance(message, list) and message else message
            if not isinstance(candidate, dict):
                logger.warning("Ignoring malformed biquote SignalR event")
                return
            try:
                quote = validate_quote(candidate)
            except BiquoteLiveError as exc:
                self.last_error = str(exc)
                logger.warning("Rejected biquote SignalR tick: %s", str(exc))
                return
            self.last_message_at = datetime.now(timezone.utc).isoformat()
            self.last_error = None
            self.on_quote(quote)

        async def on_open() -> None:
            self.connected = True
            self.last_connected_at = datetime.now(timezone.utc).isoformat()
            await client.send("Subscribe", [["EURUSD"]])

        async def on_close() -> None:
            self.connected = False
            self.reconnect_count += 1

        async def on_error(error: Any) -> None:
            self.last_error = str(getattr(error, "error", error))[:300]
            logger.warning("biquote SignalR error: %s", self.last_error)

        client.on("ReceiveTick", receive)
        client.on_open(on_open)
        client.on_close(on_close)
        client.on_error(on_error)
        await client.run()
        self.connected = False


class BiquoteStreamSupervisor:
    """Owns one background stream task and exposes safe operational status."""

    def __init__(self, store: "LiveQuoteStore"):
        self.store = store
        self.stream = BiquoteSignalRStream(store.update)
        self._task: asyncio.Task | None = None
        self.enabled = False

    async def start(self) -> None:
        self.enabled = True
        self._task = asyncio.create_task(self._run_forever(), name="biquote-signalr")

    async def _run_forever(self) -> None:
        delay = 2.0
        while self.enabled:
            try:
                await self.stream.run()
                if self.enabled:
                    self.stream.last_error = "signalr_client_stopped"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.stream.last_error = f"{type(exc).__name__}: {exc}"[:300]
                logger.exception("biquote SignalR worker failed")
            if self.enabled:
                self.stream.connected = False
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)

    async def stop(self) -> None:
        self.enabled = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self.stream.connected = False

    def status(self) -> dict[str, Any]:
        result = self.stream.status()
        result["enabled"] = self.enabled
        result["task_running"] = bool(self._task and not self._task.done())
        return result


class LiveQuoteStore:
    """Thread-safe latest-quote cache used by the API and stream callback."""

    def __init__(self):
        self._lock = threading.Lock()
        self._quote: dict[str, Any] | None = None

    def update(self, quote: dict[str, Any]) -> None:
        with self._lock:
            self._quote = dict(quote)

    def get(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._quote) if self._quote else None


live_quote_store = LiveQuoteStore()


biquote_stream_supervisor = BiquoteStreamSupervisor(live_quote_store)
