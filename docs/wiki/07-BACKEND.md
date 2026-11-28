# Backend and Deployment

## Recorded v6 stack
FastAPI entry `v6_app.py`; package modules noted include `v6/market_data.py`, `v6/data_quality.py`, `v6/twelvedata.py`; `requirements-v6.txt`; CI workflow `.github/workflows/backend-checks.yml`. Exact code and test coverage must be rechecked before claims.

## Staging environment
Repository `Antoniodecaprio/eurusd-backend`, branch `v6-development`; Render staging `https://eurusd-v6-staging.onrender.com`. Build: `pip install -r requirements-v6.txt`. Start: `uvicorn v6_app:app --host 0.0.0.0 --port $PORT`. Health endpoint `/health`. Legacy v4.4 URL `https://eurusd-api-ok33.onrender.com` must remain untouched unless explicitly authorized.

## Documented routes
`/v6/status` reports configured primary, not necessarily actual source for a request. `/v6/market-quality` reports source/quality per timeframe. `/v6/market-snapshot-lite` compact payload, default limit 100, range 1–500.

## Cache/cooldown noted
M15 TTL 60s; H1 300s; D1 3600s; provider rate-limit cooldown 120s. M15/H1/D1 supported; H4 derived from H1.

## Deployment
Green CI does not prove Render runs that commit. Verify deployment commit and health. Details above are from project context and should be re-audited against source and live service.

## Current v6 release candidate additions
- Live quote: `GET /v6/live-quote`; read-only BiQuote snapshot, separate from historical candles.
- Stream status: `GET /v6/live-feed-status`.
- Forecast context: `GET /v6/forecast-market-context`; candidate spot only, no model injection.
- Research backtest: `GET /v6/spot-anchor-backtest`; historical H1 walk-forward comparison only.
- The BiQuote SignalR worker is opt-in. Set `BIQUOTE_STREAM_ENABLED=true` only after the provider connection is verified in the target deployment. It defaults off so an unavailable optional streaming dependency/provider cannot destabilize API startup.
- Set `CORS_ORIGINS` to the exact FlutterFlow web origins that need API access. Do not use wildcard origins with credentials.
- v6 dependencies: `requirements-v6.txt`; CI installs the full `requirements.txt` so it exercises both legacy imports and the v6 SignalR dependency.
