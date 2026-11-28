"""Build a safe live-market context for the forecasting application.

This is an observation/feature contract, not a forecast model. Live bid/ask
is kept distinct from completed historical OHLC and is never silently written
into the OU calibration or historical training series.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_forecast_market_context(
    quote: dict[str, Any] | None,
    *,
    now: datetime | None = None,
    max_age_seconds: float = 30.0,
) -> dict[str, Any]:
    """Return an explicit live spot context and fail-closed model-use flags."""
    retrieved = now or datetime.now(timezone.utc)
    if retrieved.tzinfo is None:
        retrieved = retrieved.replace(tzinfo=timezone.utc)
    retrieved = retrieved.astimezone(timezone.utc)

    base = {
        "schema_version": "1.0",
        "symbol": "EUR/USD",
        "retrieved_at_utc": retrieved.isoformat(),
        "context_type": "live_market_observation",
        "historical_data_modified": False,
        "historical_consensus_eligible": False,
        "training_data_eligible": False,
        "ou_parameters_modified": False,
        "trading_enabled": False,
    }
    if not quote:
        return {
            **base,
            "status": "unavailable",
            "reason": "no_live_quote_received",
            "live_quote": None,
            "forecast_handoff": {
                "spot_candidate": None,
                "spot_candidate_status": "unavailable",
                "automatic_model_injection": False,
                "note": "No live quote is available; do not substitute a fabricated or stale price.",
            },
        }

    try:
        bid = float(quote["bid"])
        ask = float(quote["ask"])
        stamp = datetime.fromisoformat(
            str(quote["timestamp_utc"]).replace("Z", "+00:00")
        )
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        stamp = stamp.astimezone(timezone.utc)
    except (KeyError, TypeError, ValueError):
        return {
            **base,
            "status": "rejected",
            "reason": "cached_quote_schema_invalid",
            "live_quote": None,
            "forecast_handoff": {
                "spot_candidate": None,
                "spot_candidate_status": "rejected",
                "automatic_model_injection": False,
            },
        }

    age = (retrieved - stamp).total_seconds()
    valid = (
        bid > 0 and ask >= bid and age >= -5
        and bool(quote.get("market_open"))
        and not bool(quote.get("provider_stale"))
        and age <= max_age_seconds
    )
    quote_view = {
        "provider": quote.get("provider", "unknown"),
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2.0,
        "spread": ask - bid,
        "spread_pips": (ask - bid) / 0.0001,
        "timestamp_utc": stamp.isoformat(),
        "age_seconds": max(0.0, age),
        "is_fresh": valid,
        "market_open": bool(quote.get("market_open")),
        "source_verified": bool(quote.get("source_verified", False)),
    }
    return {
        **base,
        "status": "ready" if valid else "not_ready",
        "reason": None if valid else "quote_stale_closed_or_invalid_for_live_context",
        "live_quote": quote_view,
        "forecast_handoff": {
            # Candidate only: OU model code is not present in this repository branch.
            "spot_candidate": quote_view["mid"] if valid else None,
            "spot_candidate_status": "candidate_only" if valid else "unavailable",
            "automatic_model_injection": False,
            "requires_model_owner_approval": True,
            "note": (
                "This live midpoint may be evaluated as the current spot input for a "
                "separately validated forecast run. It is not the OU equilibrium, "
                "does not recalibrate parameters, and must not be appended to historical candles."
            ),
        },
    }
