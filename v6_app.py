"""EUR/USD v6 staging API.

This first hosted milestone exposes market-data readiness and diagnostics.
It intentionally does not manufacture trading forecasts or execute trades.
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from v6.market_data import fetch_market_snapshot, fetch_timeframe, MarketDataError
from v6.provider_comparison import fetch_provider_evidence
from v6.forecast_context import build_forecast_market_context
from v6.spot_backtest import run_spot_anchor_backtest
from v6.biquote_live import (BiquoteLiveError, fetch_live_quote, live_quote_store, biquote_stream_supervisor)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("eurusd_v6")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    stream_enabled = os.getenv("BIQUOTE_STREAM_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    if stream_enabled:
        await biquote_stream_supervisor.start()
        logger.info("biquote SignalR stream supervisor started")
    try:
        yield
    finally:
        if stream_enabled:
            await biquote_stream_supervisor.stop()
            logger.info("biquote SignalR stream supervisor stopped")


app = FastAPI(
    title="EUR/USD Distribution Forecaster v6",
    lifespan=lifespan,
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
        "market_data_provider": ("Twelve Data primary; Yahoo Finance fallback" if os.getenv("TWELVE_DATA_KEY", "").strip() else "Yahoo Finance (research feed; Twelve Data key not configured)"),
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

@app.get("/v6/market-quality")
def market_quality():
    """Return compact per-timeframe diagnostics without candle arrays.

    This endpoint is intended for quick health review and sharing; it still
    fetches fresh provider data and does not cache or certify trading readiness.
    """
    try:
        result = fetch_market_snapshot(("M15", "H1", "H4", "D1"), include_candles=False)
    except Exception as exc:
        logger.exception("Unexpected market quality failure")
        raise HTTPException(
            status_code=502,
            detail=f"Market quality check failed: {type(exc).__name__}",
        ) from exc

    summary = {}
    for timeframe, item in result.get("timeframes", {}).items():
        metadata = item.get("metadata") or {}
        quality = metadata.get("quality") or {}
        # H4 includes its H1 source-quality report; retain the key diagnostic
        # fields without returning any candle arrays.
        summary[timeframe] = {
            "status": item.get("status", "unknown"),
            "source": quality.get("source", metadata.get("source", "unknown")),
            "candle_count": quality.get("candle_count"),
            "last_candle_close_utc": quality.get("last_candle_close_utc"),
            "age_seconds": quality.get("age_seconds"),
            "stale": quality.get("stale"),
            "missing_interval_count": quality.get("missing_interval_count"),
            "missing_interval_interpretation": quality.get("missing_interval_interpretation"),
            "duplicate_count": quality.get("duplicate_count"),
            "invalid_count": quality.get("invalid_count"),
            "ready": quality.get("ready"),
            "reasons": quality.get("reasons", []),
            "session_boundary": quality.get("session_boundary"),
            "broker_alignment_verified": quality.get("broker_alignment_verified"),
            "base_quality": quality.get("base_quality"),
            "error": item.get("error"),
            "limitations": metadata.get("limitations", []),
        }

    statuses = [v["status"] for v in summary.values()]
    return {
        "service": "eurusd-v6",
        "version": app.version,
        "retrieved_at_utc": result.get("retrieved_at_utc"),
        "symbol": result.get("symbol"),
        "source": result.get("source"),
        "overall_status": (
            "ready" if statuses and all(s == "ready" for s in statuses)
            else "partial_or_unavailable"
        ),
        "trading_enabled": False,
        "timeframes": summary,
    }


@app.get("/v6/market-snapshot-lite")
def market_snapshot_lite(
    timeframes: str = Query("M15,H1,H4,D1", description="Comma-separated: M15,H1,H4,D1"),
    limit: int = Query(100, ge=1, le=500, description="Candles per timeframe, maximum 500"),
):
    """Mobile-friendly snapshot: selected timeframes and bounded candle arrays."""
    requested = tuple(dict.fromkeys(
        part.strip().upper() for part in timeframes.split(",") if part.strip()
    ))
    allowed = {"M15", "H1", "H4", "D1"}
    if not requested or any(tf not in allowed for tf in requested):
        raise HTTPException(status_code=400, detail="Use one or more timeframes from M15,H1,H4,D1")

    try:
        result = fetch_market_snapshot(requested, include_candles=True)
    except Exception as exc:
        logger.exception("Unexpected compact market snapshot failure")
        raise HTTPException(
            status_code=502,
            detail=f"Market snapshot failed: {type(exc).__name__}",
        ) from exc

    for timeframe, item in result.get("timeframes", {}).items():
        candles = item.get("candles") or []
        item["candles_total_returnable"] = len(candles)
        item["candles_limit"] = limit
        item["candles"] = candles[-limit:]

    statuses = [item.get("status", "unknown") for item in result.get("timeframes", {}).values()]
    result["overall_status"] = (
        "ready" if statuses and all(status == "ready" for status in statuses)
        else "partial_or_unavailable"
    )
    result["trading_enabled"] = False
    result["response_mode"] = "compact"
    result["note"] = "Candle arrays are capped per timeframe for mobile viewing; research data only."
    return result


@app.get("/v6/provider-evidence")
def provider_evidence(
    timeframe: str = Query("M15", description="M15 or H1"),
):
    """Compare independent EUR/USD research-feed observations; no trading signal."""
    requested = timeframe.strip().upper()
    if requested not in {"M15", "H1"}:
        raise HTTPException(status_code=400, detail="Use M15 or H1")
    try:
        return fetch_provider_evidence(requested)
    except Exception as exc:
        logger.exception("Unexpected provider evidence failure")
        raise HTTPException(
            status_code=502,
            detail=f"Provider evidence failed: {type(exc).__name__}",
        ) from exc


@app.get("/v6/live-quote")
def biquote_live_quote():
    """Fetch a current biquote bid/ask snapshot; separate from historical evidence."""
    try:
        quote = fetch_live_quote()
        live_quote_store.update(quote)
    except BiquoteLiveError as exc:
        logger.warning("biquote live quote unavailable: %s", str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    quote["note"] = (
        "biquote live quote snapshot. The reported MT5 source is not verified as "
        "the user's Headway execution feed. Not used in historical consensus or trade execution."
    )
    return quote


@app.get("/v6/live-feed-status")
def biquote_live_feed_status():
    """Operational status for the managed biquote stream and latest validated tick."""
    quote = live_quote_store.get()
    return {
        "provider": "biquote",
        "stream": biquote_stream_supervisor.status(),
        "latest_quote": quote,
        "historical_consensus_eligible": False,
        "trading_enabled": False,
    }


@app.get("/v6/forecast-market-context")
def forecast_market_context():
    """Expose a validated live-spot candidate without changing model parameters."""
    quote = live_quote_store.get()
    return build_forecast_market_context(quote)


@app.get("/v6/spot-anchor-backtest")
def spot_anchor_backtest(
    window: int = Query(500, ge=100, le=2000),
    step: int = Query(4, ge=1, le=24),
    max_origins: int = Query(1200, ge=30, le=3000),
):
    """Compare OU spot anchors using walk-forward historical H1 observations."""
    try:
        candles, metadata = fetch_timeframe("H1", minimum_candles=window + 200)
        if not metadata.get("quality", {}).get("ready"):
            raise HTTPException(
                status_code=503,
                detail={"message": "H1 research history is not ready", "quality": metadata.get("quality")},
            )
        result = run_spot_anchor_backtest(
            candles, window=window, step=step, max_origins=max_origins
        )
        result["data_source"] = metadata.get("source")
        result["data_quality"] = metadata.get("quality")
        result["data_retrieved_at_utc"] = metadata.get("retrieved_at_utc")
        return result
    except HTTPException:
        raise
    except (ValueError, MarketDataError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Spot-anchor backtest failed")
        raise HTTPException(status_code=502, detail=f"Backtest failed: {type(exc).__name__}") from exc
