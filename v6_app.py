"""EUR/USD v6 staging API.

This first hosted milestone exposes market-data readiness and diagnostics.
It intentionally does not manufacture trading forecasts or execute trades.
"""
from datetime import datetime, timezone
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from v6.market_data import fetch_market_snapshot

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("eurusd_v6")

app = FastAPI(
    title="EUR/USD Distribution Forecaster v6",
    version="6.0.0-dev",
    description=(
        "v6 research/staging API. Market-data diagnostics only; "
        "forecast signals and trade execution are not enabled in this milestone."
    ),
)

origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or [],
    allow_credentials=bool(origins and "*" not in origins),
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "service": "EUR/USD Distribution Forecaster",
        "version": app.version,
        "environment": "staging",
        "mode": "research_only",
        "trading_enabled": False,
        "message": "v6 staging API is running.",
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "eurusd-v6",
        "version": app.version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/v6/status")
def status():
    return {
        "version": app.version,
        "environment": "staging",
        "mode": "research_only",
        "supported_timeframes": ["M15", "H1", "H4", "D1"],
        "market_data_provider": "Yahoo Finance (research feed)",
        "forecast_models_enabled": False,
        "calendar_verified": False,
        "mt5_execution_enabled": False,
        "live_trading_enabled": False,
        "demo_risk_limit_pct": 0.5,
    }

@app.get("/v6/market-snapshot")
def market_snapshot():
    """Fetch independent timeframe snapshots and return quality metadata."""
    try:
        result = fetch_market_snapshot(("M15", "H1", "H4", "D1"))
    except Exception as exc:
        logger.exception("Unexpected market snapshot failure")
        raise HTTPException(
            status_code=502,
            detail=f"Market snapshot failed: {type(exc).__name__}",
        ) from exc

    states = {
        timeframe: item.get("status", "unknown")
        for timeframe, item in result.get("timeframes", {}).items()
    }
    result["overall_status"] = (
        "ready" if states and all(v == "ready" for v in states.values())
        else "partial_or_unavailable"
    )
    result["trading_enabled"] = False
    return result
