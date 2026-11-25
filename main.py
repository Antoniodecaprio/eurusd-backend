"""EUR/USD Quant Forecaster API — research prototype v4.0

Important:
- This application is a research/education tool, not investment advice.
- Model probabilities are uncalibrated and do not establish profitability.
- Signals are deliberately gated: no verified calendar, no reliable validation,
  unavailable macro state, or invalid market data => HOLD.
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
import math
import os
import threading
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from scipy.stats import linregress, norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss
from statsmodels.tsa.stattools import adfuller
from fredapi import Fred

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eurusd_quant")

APP_VERSION = "4.0.0"
app = FastAPI(title="EUR/USD Quant Forecaster", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "").strip()
JOURNAL_FILE = Path(os.getenv("JOURNAL_FILE", "forecast_journal.csv"))
JOURNAL_LOCK = threading.Lock()

CALENDAR_URL = "https://showmytrades.com/api/public/calendar/v1"
CALENDAR_SOURCE = "ShowMyTrades public economic calendar"
CALENDAR_ATTRIBUTION_URL = "https://showmytrades.com/tools/economic-calendar"

MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "150"))
MIN_VALIDATION_ROWS = int(os.getenv("MIN_VALIDATION_ROWS", "50"))
MIN_BRIER_SKILL_SCORE = float(os.getenv("MIN_BRIER_SKILL_SCORE", "0.0"))
MACRO_NEUTRAL_BAND = float(os.getenv("MACRO_NEUTRAL_BAND", "0.005"))
CALENDAR_FRESHNESS_MINUTES = int(os.getenv("CALENDAR_FRESHNESS_MINUTES", "20"))


class ShortTermRequest(BaseModel):
    forecast_days: int = Field(default=3, ge=1, le=7)


class ForecastRequest(BaseModel):
    horizon_years: float = Field(default=3.0, gt=0.0, le=10.0)
    pi_us: float = Field(default=0.02, ge=-0.05, le=0.25)
    pi_ez: float = Field(default=0.02, ge=-0.05, le=0.25)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_utc(value: Any) -> Optional[dt.datetime]:
    if value is None or value == "":
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="raise")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def true_range_series(df: pd.DataFrame) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    if len(tr):
        tr.iloc[0] = high_low.iloc[0]
    return tr.astype(float)


# ---------------------------------------------------------------------------
# FRED macro data and macro model
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def fetch_macro_data_cached(cache_key: str) -> pd.DataFrame:
    del cache_key 
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY is not configured")

    fred = Fred(api_key=FRED_API_KEY)

    def month_end(series: pd.Series) -> pd.Series:
        series = pd.Series(series).dropna().copy()
        index = pd.DatetimeIndex(pd.to_datetime(series.index))
        if index.tz is not None:
            index = index.tz_localize(None)
        series.index = index
        return series.resample("ME").last()

    spot = month_end(fred.get_series("DEXUSEU"))
    us_cpi = month_end(fred.get_series("CPIAUCSL")).shift(1)
    ez_cpi = month_end(fred.get_series("CP0000EZ19M086NEST")).shift(1)

    df = pd.DataFrame({"spot": spot, "us_cpi": us_cpi, "ez_cpi": ez_cpi}).dropna()
    df = df.loc[df.index >= "1999-01-01"].copy()
    df = df[(df["spot"] > 0) & (df["us_cpi"] > 0) & (df["ez_cpi"] > 0)].copy()
    if len(df) < 60:
        raise ValueError("Insufficient aligned FRED history for macro model")

    df["s"] = np.log(df["spot"])
    df["p_us"] = np.log(df["us_cpi"])
    df["p_ez"] = np.log(df["ez_cpi"])
    df["q"] = df["s"] + df["p_us"] - df["p_ez"]
    if not np.isfinite(df["q"]).all():
        raise ValueError("Non-finite values in macro real-exchange-rate series")
    return df


def get_macro_state(df_in: pd.DataFrame) -> Tuple[float, int, float, Optional[float], str, float]:
    df = df_in.copy()
    if "q" not in df or len(df) < 60:
        raise ValueError("Insufficient observations for macro-state estimation")

    q = pd.to_numeric(df["q"], errors="coerce").dropna()
    q_lag = q.shift(1)
    pair = pd.DataFrame({"q": q, "q_lag": q_lag}).dropna()
    if len(pair) < 48 or pair["q"].nunique() < 2:
        raise ValueError("Insufficient variation for AR(1) macro estimation")

    slope, intercept, _, _, _ = linregress(pair["q_lag"], pair["q"])
    try:
        adf_p = float(adfuller(q, regression="c", autolag="AIC")[1])
    except Exception:
        adf_p = 1.0

    residuals = pair["q"] - (intercept + slope * pair["q_lag"])
    residual_sigma = finite_float(residuals.std(ddof=1)) or 0.0
    
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError("Macro AR(1) estimates are non-finite")

    if 0.0 < slope < 1.0 and adf_p < 0.05 and residual_sigma > 0:
        phi = float(slope)
        q_bar = float(intercept / (1.0 - phi))
        deviation = float(q.iloc[-1] - q_bar)
        sigma_eps = residual_sigma
        model_type = "AR(1)_mean_reverting"
    else:
        phi = 1.0
        q_bar = None
        deviation = 0.0
        monthly_changes = q.diff().dropna()
        sigma_eps = finite_float(monthly_changes.std(ddof=1)) or 0.0
        model_type = "random_walk_fallback"

    if not math.isfinite(sigma_eps) or sigma_eps < 0:
        raise ValueError("Invalid macro innovation volatility")
    return deviation, -1, phi, q_bar, model_type, float(sigma_eps)


def classify_macro_bias(deviation: float, model_available: bool) -> str:
    if not model_available:
        return "Unavailable"
    if deviation > MACRO_NEUTRAL_BAND:
        return "Bearish"
    if deviation < -MACRO_NEUTRAL_BAND:
        return "Bullish"
    return "Neutral"


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"status": "API is running - Quant v4.0", "version": APP_VERSION}


@app.post("/forecast")
def get_forecast(req: ForecastRequest) -> Dict[str, Any]:
    try:
        cache_key = utc_now().strftime("%Y-%m-%d-%H")
        df = fetch_macro_data_cached(cache_key).copy()
        current_spot = float(df["spot"].iloc[-1])
        deviation, _, phi, q_bar, model_type, sigma_eps = get_macro_state(df)
        months = req.horizon_years * 12.0

        if model_type == "AR(1)_mean_reverting" and q_bar is not None:
            current_q = float(df["q"].iloc[-1])
            half_life = math.log(0.5) / math.log(phi) / 12.0
            phi_n = phi ** months
            drift_q = (phi_n - 1.0) * (current_q - q_bar)
            ppp_anchor = float(np.exp(q_bar - df["p_us"].iloc[-1] + df["p_ez"].iloc[-1]))
            var_q = (sigma_eps ** 2) * (1.0 - phi ** (2.0 * months)) / (1.0 - phi ** 2)
        else:
            half_life = None
            drift_q = float(df["q"].diff().mean()) * months
            ppp_anchor = None
            var_q = (sigma_eps ** 2) * months

        drift_infl = (req.pi_ez - req.pi_us) * req.horizon_years
        mean_delta = float(drift_q + drift_infl)
        var_q = max(float(var_q), 0.0)
        std_delta = math.sqrt(var_q)
        p_up = float(norm.cdf(mean_delta / std_delta)) if std_delta > 0 else 0.5
        log_spot = math.log(current_spot)
        median_rate = float(math.exp(log_spot + mean_delta))
        expected_rate = float(math.exp(log_spot + mean_delta + var_q / 2.0))
        
        historical_data = [
            {"date": index.strftime("%Y-%m-%d"), "price": float(row["spot"])}
            for index, row in df.tail(60).iterrows()
        ]
        
        macro_bias = classify_macro_bias(deviation, model_type == "AR(1)_mean_reverting")

        return {
            "status": "success",
            "data": {
                "current_state": {
                    "spot_rate": current_spot,
                    "ppp_equilibrium_estimate": ppp_anchor,
                    "deviation_q": float(deviation),
                    "half_life_years": float(half_life) if half_life is not None else None,
                    "macro_model": model_type,
                    "macro_bias": macro_bias,
                    "macro_neutral_band": MACRO_NEUTRAL_BAND,
                    "data_source_warning": "FRED monthly series; CPI timing approximated by one-month lag."
                },
                "forecast": {
                    "p_up": p_up,
                    "expected_rate": expected_rate,
                    "median_rate": median_rate,
                    "ci_90_low": float(math.exp(log_spot + mean_delta - 1.645 * std_delta)),
                    "ci_90_high": float(math.exp(log_spot + mean_delta + 1.645 * std_delta))
                },
                "historical_data": historical_data
            }
        }
    except Exception as exc:
        logger.exception("Macro forecast failed")
        raise HTTPException(status_code=503, detail={"status": "error", "message": "Macro forecast unavailable."}) from exc


# ---------------------------------------------------------------------------
# Economic calendar — documented ShowMyTrades API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _get_calendar_payload(cache_bucket: str, days: int) -> Dict[str, Any]:
    del cache_bucket
    response = requests.get(
        CALENDAR_URL,
        params={"currency": "EUR,USD", "impact": "high,medium", "days": int(days), "format": "json"},
        headers={"Accept": "application/json", "User-Agent": f"EURUSD-Quant-Forecaster/{APP_VERSION}"},
        timeout=(3.0, 8.0)
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Calendar response is not a JSON object")
    return payload


def fetch_calendar_risk(forecast_days: int, forecast_candles: int) -> Dict[str, Any]:
    now = utc_now()
    estimated_calendar_horizon = dt.timedelta(days=(forecast_candles / 24.0) * (7.0 / 5.0))
    horizon_end = now + estimated_calendar_horizon
    request_days = min(30, max(1, int(math.ceil(estimated_calendar_horizon.total_seconds() / 86400.0)) + 1))
    bucket = str(int(time.time() // (15 * 60)))

    result: Dict[str, Any] = {
        "calendar_available": False,
        "event_risk": "UNKNOWN",
        "calendar_source": CALENDAR_SOURCE,
        "calendar_attribution_url": CALENDAR_ATTRIBUTION_URL,
        "calendar_checked_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "calendar_window_end": horizon_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "calendar_upcoming_event_count": 0,
        "calendar_event_summary": [],
        "calendar_warning": "UNKNOWN means the filter cannot safely verify coverage."
    }

    try:
        payload = _get_calendar_payload(bucket, request_days)
        required_top = {"server_time", "generated_at", "from", "to", "count", "events"}
        if not required_top.issubset(payload.keys()):
            raise ValueError("Calendar response is missing required metadata")
        if not isinstance(payload["events"], list):
            raise ValueError("Calendar events field is not a list")

        server_time = dt.datetime.fromtimestamp(int(payload["server_time"]), tz=dt.timezone.utc)
        generated_at = parse_utc(payload["generated_at"])
        coverage_start = parse_utc(payload["from"])
        coverage_end = parse_utc(payload["to"])
        
        if not generated_at or not coverage_start or not coverage_end:
            raise ValueError("Calendar coverage timestamps are invalid")
        if abs((now - server_time).total_seconds()) > CALENDAR_FRESHNESS_MINUTES * 60:
            raise ValueError("Calendar server clock is stale or inconsistent")
        if (now - generated_at).total_seconds() > CALENDAR_FRESHNESS_MINUTES * 60:
            raise ValueError("Calendar snapshot is too old")
        if generated_at > now + dt.timedelta(minutes=5):
            raise ValueError("Calendar generation timestamp is in the future")
        if coverage_start > now + dt.timedelta(minutes=5) or coverage_start < now - dt.timedelta(hours=12):
            raise ValueError("Calendar start does not cover the current time")
        if coverage_end < horizon_end - dt.timedelta(minutes=5):
            raise ValueError("Calendar response does not cover the forecast horizon")
        if int(payload["count"]) != len(payload["events"]):
            raise ValueError("Calendar event count does not match event list")

        in_window = []
        for event in payload["events"]:
            if not isinstance(event, dict):
                raise ValueError("Calendar contains a malformed event")
            currency = str(event.get("currency", "")).upper()
            impact = str(event.get("impact", "")).lower()
            title = str(event.get("title", "")).strip()
            status = str(event.get("status", "scheduled")).lower()
            confidence = str(event.get("confidence", "")).lower()
            event_time = parse_utc(event.get("time"))
            
            try:
                timestamp = int(event.get("ts"))
                ts_time = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
            except Exception as exc:
                raise ValueError("Calendar event has an invalid Unix timestamp") from exc

            if currency not in {"EUR", "USD"}: raise ValueError("Unexpected currency")
            if impact not in {"high", "medium"}: raise ValueError("Unexpected impact classification")
            if not title or event_time is None: raise ValueError("Missing title or valid time")
            if status not in {"scheduled", "moved", "tentative"}: raise ValueError("Unknown status")
            if abs((event_time - ts_time).total_seconds()) > 120: raise ValueError("Time mismatch")

            event_item = {
                "time": event_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "currency": currency,
                "impact": impact.upper(),
                "title": title,
                "status": status.upper(),
                "confidence": confidence or None
            }
            if now <= event_time <= horizon_end:
                in_window.append(event_item)

        high_events = [e for e in in_window if e["impact"] == "HIGH"]
        medium_events = [e for e in in_window if e["impact"] == "MEDIUM"]
        uncertain_events = [e for e in in_window if e["status"] == "TENTATIVE" or e["confidence"] == "sources_disagree"]

        if high_events or uncertain_events:
            risk = "HIGH"
        elif medium_events:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        result.update({
            "calendar_available": True,
            "event_risk": risk,
            "calendar_upcoming_event_count": len(in_window),
            "calendar_event_summary": in_window[:12],
            "calendar_coverage_from": coverage_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "calendar_coverage_to": coverage_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "calendar_snapshot_time": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        })
        return result
    except Exception as exc:
        logger.warning("Economic calendar unavailable or invalid: %s", exc)
        result["calendar_error"] = str(exc)[:240]
        return result


# ---------------------------------------------------------------------------
# Technical indicators — Wilder definitions
# ---------------------------------------------------------------------------

def calculate_wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    delta = values.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = np.full(len(values), np.nan, dtype=float)
    avg_loss = np.full(len(values), np.nan, dtype=float)

    if len(values) > period:
        avg_gain[period] = float(gain.iloc[1:period + 1].mean())
        avg_loss[period] = float(loss.iloc[1:period + 1].mean())
        for index in range(period + 1, len(values)):
            avg_gain[index] = ((avg_gain[index - 1] * (period - 1)) + gain.iloc[index]) / period
            avg_loss[index] = ((avg_loss[index - 1] * (period - 1)) + loss.iloc[index]) / period

    rs = avg_gain / np.where(avg_loss == 0.0, np.nan, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.where((avg_loss == 0.0) & (avg_gain > 0.0), 100.0, rsi)
    rsi = np.where((avg_gain == 0.0) & (avg_loss > 0.0), 0.0, rsi)
    rsi = np.where((avg_gain == 0.0) & (avg_loss == 0.0), 50.0, rsi)
    return pd.Series(rsi, index=values.index, name="rsi")


def calculate_true_wilder_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tr = true_range_series(df)
    atr = np.full(len(tr), np.nan, dtype=float)
    if len(tr) > window:
        seed = tr.iloc[1:window + 1]
        if seed.notna().all():
            atr[window] = float(seed.mean())
            for index in range(window + 1, len(tr)):
                atr[index] = ((atr[index - 1] * (window - 1)) + float(tr.iloc[index])) / window
    return pd.Series(atr, index=df.index, name="atr")


def calculate_true_wilder_adx(df: pd.DataFrame, window: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index, dtype=float)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index, dtype=float)
    tr = true_range_series(df)

    tr_s = np.full(len(df), np.nan, dtype=float)
    pdm_s = np.full(len(df), np.nan, dtype=float)
    mdm_s = np.full(len(df), np.nan, dtype=float)
    if len(df) > window:
        tr_seed = tr.iloc[1:window + 1]
        if tr_seed.notna().all():
            tr_s[window] = float(tr_seed.mean())
            pdm_s[window] = float(plus_dm.iloc[1:window + 1].mean())
            mdm_s[window] = float(minus_dm.iloc[1:window + 1].mean())
            for index in range(window + 1, len(df)):
                tr_s[index] = ((tr_s[index - 1] * (window - 1)) + float(tr.iloc[index])) / window
                pdm_s[index] = ((pdm_s[index - 1] * (window - 1)) + float(plus_dm.iloc[index])) / window
                mdm_s[index] = ((mdm_s[index - 1] * (window - 1)) + float(minus_dm.iloc[index])) / window

    tr_smoothed = pd.Series(tr_s, index=df.index)
    plus_smoothed = pd.Series(pdm_s, index=df.index)
    minus_smoothed = pd.Series(mdm_s, index=df.index)
    plus_di = 100.0 * plus_smoothed / (tr_smoothed + 1e-12)
    minus_di = 100.0 * minus_smoothed / (tr_smoothed + 1e-12)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)

    adx = np.full(len(df), np.nan, dtype=float)
    start_idx = 2 * window - 1
    if len(dx) > start_idx:
        seed_dx = dx.iloc[window:start_idx + 1]
        if seed_dx.notna().all():
            adx[start_idx] = float(seed_dx.mean())
            for index in range(start_idx + 1, len(dx)):
                adx[index] = ((adx[index - 1] * (window - 1)) + float(dx.iloc[index])) / window

    adx_series = pd.Series(adx, index=df.index, name="adx")
    rising = adx_series > adx_series.shift(1)
    return adx_series, plus_di.rename("plus_di"), minus_di.rename("minus_di"), rising.rename("adx_rising")


# ---------------------------------------------------------------------------
# Market data ingestion and feature construction
# ---------------------------------------------------------------------------

def normalize_ohlc(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    df = frame.copy()
    rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    df = df.rename(columns=rename_map)
    required = ["Open", "High", "Low", "Close"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"{source} OHLC data is missing required columns")
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[required].dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df["High"] >= df["Low"]) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0)]
    return df.astype(float)


def fetch_eurusd_data() -> Tuple[pd.DataFrame, str]:
    df_eur = pd.DataFrame()
    source = ""

    if TWELVE_DATA_KEY:
        try:
            response = requests.get(
                "https://api.twelvedata.com/time_series",
                params={"symbol": "EUR/USD", "interval": "1h", "outputsize": 1000, "timezone": "UTC", "apikey": TWELVE_DATA_KEY},
                timeout=(3.0, 8.0)
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and isinstance(payload.get("values"), list):
                raw = pd.DataFrame(payload["values"])
                raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True, errors="coerce")
                raw = raw.dropna(subset=["datetime"]).set_index("datetime").sort_index()
                df_eur = normalize_ohlc(raw, "Twelve Data")
                if len(df_eur) >= 200:
                    source = "Twelve Data"
        except Exception as exc:
            logger.warning("Twelve Data EUR/USD fetch failed: %s", exc)

    if df_eur.empty or len(df_eur) < 200:
        try:
            raw_yf = yf.Ticker("EURUSD=X").history(period="90d", interval="1h", auto_adjust=False, prepost=False, timeout=8)
            if not raw_yf.empty:
                if raw_yf.index.tz is None:
                    raw_yf.index = raw_yf.index.tz_localize("UTC")
                else:
                    raw_yf.index = raw_yf.index.tz_convert("UTC")
                df_eur = normalize_ohlc(raw_yf, "Yahoo Finance")
                if len(df_eur) >= 200:
                    source = "Yahoo Finance"
        except Exception as exc:
            logger.warning("Yahoo Finance EUR/USD fallback failed: %s", exc)

    if df_eur.empty or len(df_eur) < 200:
        raise ValueError("Insufficient EUR/USD hourly OHLC data from available providers")

    now = pd.Timestamp.now(tz="UTC")
    latest_bar = df_eur.index[-1]
    current_hour_start = now.floor("h")
    if latest_bar >= current_hour_start:
        df_eur = df_eur.iloc[:-1].copy()

    if len(df_eur) < 200:
        raise ValueError("Insufficient completed EUR/USD candles after removing live candle")
    return df_eur, source


def fetch_dxy_data() -> pd.DataFrame:
    try:
        raw = yf.Ticker("DX-Y.NYB").history(period="90d", interval="1h", auto_adjust=False, prepost=False, timeout=8)
        if raw.empty:
            return pd.DataFrame()
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        else:
            raw.index = raw.index.tz_convert("UTC")
        frame = raw[["Close"]].copy()
        frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
        return frame.dropna().sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]
    except Exception as exc:
        logger.warning("DXY fetch failed; continuing without DXY: %s", exc)
        return pd.DataFrame()


def add_market_features(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    df = df_in.copy()
    df["Clean_Close"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ret_1h"] = df["Clean_Close"].pct_change(1)
    df["ret_24h"] = df["Clean_Close"].pct_change(24)
    df["sma_50"] = df["Clean_Close"].rolling(window=50, min_periods=50).mean()
    df["dist_sma"] = (df["Clean_Close"] - df["sma_50"]) / df["sma_50"]
    df["trend_direction"] = np.where(df["Clean_Close"] > df["sma_50"], 1, -1)
    df["std_50"] = df["Clean_Close"].rolling(window=50, min_periods=50).std()
    df["z_score"] = (df["Clean_Close"] - df["sma_50"]) / (df["std_50"] + 1e-12)
    df["rsi"] = calculate_wilder_rsi(df["Close"], period=14)
    df["rsi_extreme_low"] = (df["rsi"] <= 5).astype(int)
    df["rsi_extreme_high"] = (df["rsi"] >= 95).astype(int)
    df["atr"] = calculate_true_wilder_atr(df, window=14)
    df["adx"], df["plus_di"], df["minus_di"], df["adx_rising"] = calculate_true_wilder_adx(df, window=14)

    df["range_size"] = df["High"] - df["Low"]
    df["range_mean"] = df["range_size"].rolling(window=24, min_periods=24).mean()
    df["range_spike"] = (df["range_size"] > (df["range_mean"] * 1.5)).astype(int)
    df["prev_high"] = df["High"].shift(1)
    df["prev_low"] = df["Low"].shift(1)
    df["bearish_sweep"] = np.where((df["High"] > df["prev_high"]) & (df["Close"] < df["prev_high"]) & (df["range_spike"] == 1), -1, 0)
    df["bullish_sweep"] = np.where((df["Low"] < df["prev_low"]) & (df["Close"] > df["prev_low"]) & (df["range_spike"] == 1), 1, 0)
    df["net_sweep"] = df["bullish_sweep"] + df["bearish_sweep"]

    swing_window = 5
    df["swing_high"] = df["High"].rolling(window=swing_window, min_periods=swing_window).max()
    df["swing_low"] = df["Low"].rolling(window=swing_window, min_periods=swing_window).min()
    df["near_swing_high"] = (df["Close"] > df["swing_high"] * 0.998).astype(int)
    df["near_swing_low"] = (df["Close"] < df["swing_low"] * 1.002).astype(int)

    dxy_frame = fetch_dxy_data()
    dxy_available = False
    if not dxy_frame.empty:
        aligned = df[["Close"]].join(dxy_frame.rename(columns={"Close": "Close_DXY"}), how="left")
        aligned["Close_DXY"] = aligned["Close_DXY"].ffill(limit=4)
        paired_count = int(aligned["Close_DXY"].notna().sum())
        latest_dxy = finite_float(aligned["Close_DXY"].iloc[-1])
        if paired_count >= 100 and latest_dxy is not None and latest_dxy > 0:
            df["Close_DXY"] = aligned["Close_DXY"]
            df["Clean_DXY"] = df["Close_DXY"].ewm(span=10, adjust=False).mean()
            df["dxy_ret_1h"] = df["Clean_DXY"].pct_change(1)
            df["dxy_corr"] = df["ret_1h"].rolling(window=48, min_periods=24).corr(df["dxy_ret_1h"])
            dxy_available = True

    if not dxy_available:
        df["dxy_ret_1h"] = np.nan
        df["dxy_corr"] = np.nan

    return df, dxy_available


def get_mtf_alignment(close: pd.Series) -> Tuple[str, str, str]:
    h4 = close.resample("4h").last().dropna()
    d1 = close.resample("1D").last().dropna()
    if len(h4): h4 = h4.iloc[:-1]
    if len(d1): d1 = d1.iloc[:-1]

    h4_trend = "Unknown"
    d1_trend = "Unknown"
    if len(h4) >= 20:
        avg = h4.rolling(20, min_periods=20).mean().iloc[-1]
        if pd.notna(avg): h4_trend = "Up" if h4.iloc[-1] > avg else "Down"
    if len(d1) >= 10:
        avg = d1.rolling(10, min_periods=10).mean().iloc[-1]
        if pd.notna(avg): d1_trend = "Up" if d1.iloc[-1] > avg else "Down"

    if h4_trend == "Unknown" or d1_trend == "Unknown":
        alignment = "Unknown"
    elif h4_trend == d1_trend == "Up":
        alignment = "Strong Bullish"
    elif h4_trend == d1_trend == "Down":
        alignment = "Strong Bearish"
    else:
        alignment = "Conflict"
    return h4_trend, d1_trend, alignment


def get_reference_price_discrepancy(current_price: float) -> Tuple[bool, Optional[float], Optional[float]]:
    try:
        response = requests.get("https://api.frankfurter.app/latest", params={"from": "EUR", "to": "USD"}, timeout=(2.0, 4.0))
        response.raise_for_status()
        payload = response.json()
        reference_rate = finite_float(payload.get("rates", {}).get("USD"))
        if reference_rate is None or current_price <= 0:
            return False, None, None
        discrepancy_pct = abs(reference_rate - current_price) / current_price * 100.0
        return discrepancy_pct > 1.0, reference_rate, discrepancy_pct
    except Exception:
        return False, None, None


# ---------------------------------------------------------------------------
# Model training, validation and signal logic
# ---------------------------------------------------------------------------

FEATURES_BASE = [
    "ret_1h", "ret_24h", "dist_sma", "rsi", "rsi_extreme_low", "rsi_extreme_high",
    "range_spike", "net_sweep", "near_swing_high", "near_swing_low",
    "trend_direction", "z_score"
]


def make_xgb(seed: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=80, learning_rate=0.04, max_depth=3, min_child_weight=3,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", n_jobs=1, random_state=seed
    )


def make_rf(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=5,
        max_features="sqrt", n_jobs=1, random_state=seed,
        class_weight="balanced_subsample"
    )


def estimate_zscore_half_life(df: pd.DataFrame) -> Optional[float]:
    try:
        recent = df["z_score"].dropna().tail(100)
        if len(recent) < 50: return None
        deltas = recent.index.to_series().diff().dropna()
        if not (deltas == pd.Timedelta(hours=1)).all(): return None
        x = recent.shift(1).dropna()
        y = recent.iloc[1:]
        if len(x) < 40 or x.std() < 1e-10: return None
        fit = linregress(x.to_numpy(), y.to_numpy())
        if 0.0 < fit.slope < 1.0 and fit.pvalue < 0.05 and fit.rvalue ** 2 >= 0.10:
            return float(math.log(0.5) / math.log(fit.slope))
    except Exception:
        return None
    return None


def build_training_frame(df: pd.DataFrame, features: List[str], periods_ahead: int) -> pd.DataFrame:
    frame = df.copy()
    future_close = frame["Close"].shift(-periods_ahead)
    frame["target_end_time"] = pd.Series(frame.index, index=frame.index).shift(-periods_ahead)
    frame["target"] = np.where(future_close.notna(), (future_close > frame["Close"]).astype(float), np.nan)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    train_frame = frame.dropna(subset=features + ["target", "target_end_time"]).copy()
    train_frame["target"] = train_frame["target"].astype(int)
    return train_frame


def evaluate_models(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "validation_brier_score": None, "validation_accuracy": None,
        "baseline_brier_score": None, "baseline_accuracy": None,
        "brier_skill_score": None, "validation_sample_count": int(len(test_df)),
        "validation_status": "INSUFFICIENT_DATA"
    }
    if len(test_df) < MIN_VALIDATION_ROWS: return metrics
    if train_df["target"].nunique() < 2 or test_df["target"].nunique() < 1: return metrics

    eval_xgb = make_xgb()
    eval_rf = make_rf()
    eval_xgb.fit(train_df[features], train_df["target"])
    eval_rf.fit(train_df[features], train_df["target"])
    
    xgb_probs = eval_xgb.predict_proba(test_df[features])[:, 1]
    rf_probs = eval_rf.predict_proba(test_df[features])[:, 1]
    ensemble_probs = (xgb_probs + rf_probs) / 2.0

    brier = float(brier_score_loss(test_df["target"], ensemble_probs))
    accuracy = float(accuracy_score(test_df["target"], (ensemble_probs >= 0.5).astype(int)))
    baseline_prob = float(train_df["target"].mean())
    baseline_probs = np.full(len(test_df), baseline_prob, dtype=float)
    baseline_brier = float(brier_score_loss(test_df["target"], baseline_probs))
    baseline_accuracy = float(accuracy_score(test_df["target"], (baseline_probs >= 0.5).astype(int)))
    bss = float(1.0 - (brier / baseline_brier)) if baseline_brier > 0 else None

    metrics.update({
        "validation_brier_score": brier, "validation_accuracy": accuracy,
        "baseline_brier_score": baseline_brier, "baseline_accuracy": baseline_accuracy,
        "brier_skill_score": bss,
        "validation_status": "BEATS_BASELINE_ON_HOLDOUT" if bss is not None and bss > 0 else "DOES_NOT_BEAT_BASELINE_ON_HOLDOUT"
    })
    return metrics


def fit_final_models(all_labelled_df: pd.DataFrame, features: List[str]) -> Tuple[xgb.XGBClassifier, RandomForestClassifier]:
    if len(all_labelled_df) < MIN_TRAIN_ROWS: raise ValueError(f"Insufficient labelled training data (N={len(all_labelled_df)})")
    if all_labelled_df["target"].nunique() < 2: raise ValueError("Full labelled training data contains only one target class")
    final_xgb = make_xgb(seed=43)
    final_rf = make_rf(seed=43)
    final_xgb.fit(all_labelled_df[features], all_labelled_df["target"])
    final_rf.fit(all_labelled_df[features], all_labelled_df["target"])
    return final_xgb, final_rf


def log_forecast_to_csv(record: Dict[str, Any]) -> None:
    with JOURNAL_LOCK:
        JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_exists = JOURNAL_FILE.is_file()
        header_ok = True
        if file_exists:
            try:
                with JOURNAL_FILE.open("r", newline="", encoding="utf-8") as handle:
                    reader = csv.reader(handle)
                    existing_header = next(reader, [])
                header_ok = existing_header == list(record.keys())
            except Exception:
                header_ok = False

        if file_exists and not header_ok:
            archive = JOURNAL_FILE.with_name(f"{JOURNAL_FILE.name}.{utc_now().strftime('%Y%m%dT%H%M%S')}")
            try:
                JOURNAL_FILE.replace(archive)
                file_exists = False
            except Exception as exc:
                logger.exception("Could not rotate journal schema: %s", exc)
                return

        try:
            with JOURNAL_FILE.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
                if not file_exists: writer.writeheader()
                writer.writerow(record)
        except Exception as exc:
            logger.exception("Could not write forecast journal: %s", exc)


@app.post("/short_term_forecast")
def get_short_term_forecast(req: ShortTermRequest) -> Dict[str, Any]:
    try:
        # 1) Macro state.
        macro_available = False
        macro_bias = "Unavailable"
        macro_dev = 0.0
        macro_model = "unavailable"
        try:
            macro_key = utc_now().strftime("%Y-%m-%d-%H")
            df_macro = fetch_macro_data_cached(macro_key).copy()
            macro_dev, _, _, _, macro_model, _ = get_macro_state(df_macro)
            macro_available = macro_model == "AR(1)_mean_reverting"
            macro_bias = classify_macro_bias(macro_dev, macro_available)
        except Exception as exc:
            logger.warning("Macro context unavailable for micro forecast: %s", exc)

        # 2) Market data and features.
        df_eur, eur_data_source = fetch_eurusd_data()
        df = df_eur.copy()
        df["Close_EUR"] = df["Close"]
        df, dxy_available = add_market_features(df)
        if len(df) < 200: raise ValueError("Insufficient hourly history after feature construction")

        current_price = finite_float(df["Close"].iloc[-1])
        if current_price is None or current_price <= 0: raise ValueError("Latest EUR/USD price is invalid")

        data_discrepancy, reference_rate, reference_discrepancy_pct = get_reference_price_discrepancy(current_price)
        _, _, mtf_alignment = get_mtf_alignment(df["Close"])

        # 3) Targets and leakage-aware chronological holdout.
        periods_ahead = req.forecast_days * 24
        features = FEATURES_BASE.copy()
        if dxy_available: features.extend(["dxy_ret_1h", "dxy_corr"])

        live_row = df[features].iloc[-1:]
        if live_row.isna().any().any() or not np.isfinite(live_row.to_numpy(dtype=float)).all():
            if dxy_available and any(live_row[column].isna().any() for column in ["dxy_ret_1h", "dxy_corr"]):
                dxy_available = False
                features = FEATURES_BASE.copy()
                live_row = df[features].iloc[-1:]
            if live_row.isna().any().any() or not np.isfinite(live_row.to_numpy(dtype=float)).all():
                raise ValueError("Live feature vector contains missing or non-finite values")

        full_labeled_df = build_training_frame(df, features, periods_ahead)
        if len(full_labeled_df) < MIN_TRAIN_ROWS + MIN_VALIDATION_ROWS:
            raise ValueError(f"Insufficient labelled history for train/validation (N={len(full_labeled_df)})")

        split_idx = int(len(full_labeled_df) * 0.75)
        split_idx = min(max(split_idx, 1), len(full_labeled_df) - 1)
        val_start_time = full_labeled_df.index[split_idx]
        
        train_split_df = full_labeled_df.loc[full_labeled_df["target_end_time"] < val_start_time].copy()
        test_split_df = full_labeled_df.loc[full_labeled_df.index >= val_start_time].copy()

        if len(train_split_df) < MIN_TRAIN_ROWS: raise ValueError(f"Insufficient post-purge training rows (N={len(train_split_df)})")
        if train_split_df["target"].nunique() < 2: raise ValueError("Post-purge training data contains only one class")

        validation = evaluate_models(train_split_df, test_split_df, features)
        
        # Deploy models are trained on the ENTIRE labeled history (train + test).
        # Since we only predict live_row (which occurs after all labels in full_labeled_df),
        # this is safe from leakage and maximizes training data.
        final_xgb, final_rf = fit_final_models(full_labeled_df, features)
        
        xgb_probability = float(final_xgb.predict_proba(live_row)[0, 1])
        rf_probability = float(final_rf.predict_proba(live_row)[0, 1])
        probability_up = float((xgb_probability + rf_probability) / 2.0)
        model_disagreement = float(abs(xgb_probability - rf_probability))
        target_up_rate = float(train_split_df["target"].mean())

        # 4) Indicator integrity and market regime.
        current_atr = finite_float(df["atr"].iloc[-1])
        atr_method = "WILDER_ATR"
        if current_atr is None or current_atr <= 0:
            fallback_tr = true_range_series(df).tail(14)
            fallback_atr = finite_float(fallback_tr.mean())
            if fallback_atr is None or fallback_atr <= 0: raise ValueError("Invalid ATR and invalid true-range fallback; forecast aborted")
            current_atr = fallback_atr
            atr_method = "SMA_TRUE_RANGE_FALLBACK"

        current_rsi = finite_float(df["rsi"].iloc[-1])
        current_trend = finite_float(df["trend_direction"].iloc[-1])
        current_z = finite_float(df["z_score"].iloc[-1])
        current_adx_num = finite_float(df["adx"].iloc[-1])
        current_plus_di_num = finite_float(df["plus_di"].iloc[-1])
        current_minus_di_num = finite_float(df["minus_di"].iloc[-1])
        
        if any(value is None for value in [current_rsi, current_trend, current_z, current_adx_num, current_plus_di_num, current_minus_di_num]):
            raise ValueError("One or more latest technical indicators are invalid")

        current_trend_int = int(current_trend)
        current_adx = float(current_adx_num)
        current_plus_di = float(current_plus_di_num)
        current_minus_di = float(current_minus_di_num)
        adx_is_rising = bool(df["adx_rising"].iloc[-1]) if pd.notna(df["adx_rising"].iloc[-1]) else False

        if current_adx > 25:
            market_regime = "Strong Bullish Trend" if current_plus_di > current_minus_di else "Strong Bearish Trend"
        elif current_adx > 20:
            market_regime = "Weak Trend"
        else:
            market_regime = "Ranging"

        # 5) Calendar filter.
        calendar = fetch_calendar_risk(req.forecast_days, periods_ahead)
        event_risk = calendar["event_risk"]
        calendar_available = bool(calendar["calendar_available"])

        # 6) Validate model usefulness.
        bss = validation["brier_skill_score"]
        validation_sample_count = int(validation["validation_sample_count"])
        validation_pass = (
            bss is not None and bss > MIN_BRIER_SKILL_SCORE and
            validation_sample_count >= MIN_VALIDATION_ROWS
        )

        raw_signal = "HOLD"
        expected_move = current_atr * 1.5
        spread_assumption = 0.00015
        
        if (
            validation_pass and macro_available and market_regime != "Unknown" and
            calendar_available and event_risk == "LOW" and expected_move > spread_assumption
        ):
            if probability_up > 0.60: raw_signal = "BUY"
            elif probability_up < 0.40: raw_signal = "SELL"

        signal = raw_signal
        horizon_conflict = "Aligned"
        if macro_available:
            if macro_bias == "Neutral":
                horizon_conflict = "Macro Neutral"
            elif (macro_bias == "Bullish" and raw_signal == "SELL") or (macro_bias == "Bearish" and raw_signal == "BUY"):
                horizon_conflict = "Conflicting Horizons"
                signal = "HOLD"

        if signal == "SELL":
            if current_trend_int == 1 or current_z < -2.5 or (current_rsi < 15 and current_z > -1.5): signal = "HOLD"
            if int(df["net_sweep"].iloc[-1]) == 1: signal = "HOLD"
        elif signal == "BUY":
            if current_trend_int == -1 or current_z > 2.5 or (current_rsi > 85 and current_z < 1.5): signal = "HOLD"
            if int(df["net_sweep"].iloc[-1]) == -1: signal = "HOLD"

        if signal == "BUY":
            target_price = current_price + (3.0 * current_atr)
            stop_loss = current_price - (1.5 * current_atr)
        elif signal == "SELL":
            target_price = current_price - (3.0 * current_atr)
            stop_loss = current_price + (1.5 * current_atr)
        else:
            target_price = None
            stop_loss = None

        dxy_corr_value = finite_float(df["dxy_corr"].iloc[-1]) if dxy_available else None
        reversal_half_life = estimate_zscore_half_life(df)
        
        history = [
            {"date": index.strftime("%Y-%m-%d %H:%M:%S"), "price": float(row["Close"])}
            for index, row in df.tail(48).iterrows()
        ]

        journal_record = {
            "timestamp": utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "spot": current_price,
            "prob_up": round(probability_up, 6),
            "signal": signal,
            "target": round(target_price, 5) if target_price is not None else None,
            "stop": round(stop_loss, 5) if stop_loss is not None else None,
            "mtf_trend": mtf_alignment,
            "regime": market_regime,
            "adx": round(current_adx, 2),
            "brier_skill_score": round(bss, 6) if bss is not None else None,
            "event_risk": event_risk
        }
        log_forecast_to_csv(journal_record)

        data: Dict[str, Any] = {
            "current_price": current_price,
            "forecast_days": req.forecast_days,
            "probability_up": probability_up,
            "xgb_probability": xgb_probability,
            "rf_probability": rf_probability,
            "model_disagreement": model_disagreement,
            "target_up_rate": target_up_rate,
            "validation_brier_score": validation["validation_brier_score"],
            "validation_accuracy": validation["validation_accuracy"],
            "baseline_brier_score": validation["baseline_brier_score"],
            "baseline_accuracy": validation["baseline_accuracy"],
            "brier_skill_score": bss,
            "validation_sample_count": validation_sample_count,
            "validation_status": validation["validation_status"],
            "validation_method": "single_chronological_holdout_with_overlapping_horizons",
            "independent_walk_forward_validated": False,
            "signal": signal,
            "raw_signal": raw_signal,
            "horizon_conflict": horizon_conflict,
            "mtf_trend": mtf_alignment,
            "data_discrepancy": data_discrepancy,
            "reference_price": reference_rate,
            "reference_discrepancy_pct": reference_discrepancy_pct,
            "reference_price_warning": "Frankfurter is a daily reference; timestamp differences can create apparent discrepancies.",
            "eurusd_data_source": eur_data_source,
            "market_regime": market_regime,
            "adx_value": current_adx,
            "plus_di": current_plus_di,
            "minus_di": current_minus_di,
            "adx_rising": adx_is_rising,
            "calendar_available": calendar_available,
            "target_price": float(target_price) if target_price is not None else None,
            "stop_loss": float(stop_loss) if stop_loss is not None else None,
            "last_rsi": float(current_rsi),
            "macro_bias": macro_bias,
            "macro_model": macro_model,
            "macro_deviation_q": float(macro_dev) if macro_available else None,
            "macro_neutral_band": MACRO_NEUTRAL_BAND,
            "whale_trap_detected": bool(int(df["net_sweep"].iloc[-1]) != 0),
            "event_risk": event_risk,
            "dxy_available": dxy_available,
            "dxy_correlation": dxy_corr_value,
            "trend_direction": "Up" if current_trend_int == 1 else "Down",
            "estimated_zscore_half_life_hours": reversal_half_life,
            "recommended_position_size_pct": 0.0,
            "model_weights": {"xgb": 0.5, "rf": 0.5},
            "probabilities_calibrated": False,
            "trading_profitability_validated": False,
            "forecast_horizon_type": "trading_candles",
            "forecast_horizon_candles": periods_ahead,
            "forecast_horizon_requested_days": req.forecast_days,
            "atr_value": current_atr,
            "atr_method": atr_method,
            "spread_assumption": spread_assumption,
            "history": history
        }
        data.update(calendar)
        return {"status": "success", "data": data}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Short-term forecast failed")
        raise HTTPException(status_code=503, detail={"status": "error", "message": str(exc)[:300]}) from exc


# ---------------------------------------------------------------------------
# Journal endpoints
# ---------------------------------------------------------------------------

@app.get("/journal")
def get_journal() -> Dict[str, Any]:
    if not JOURNAL_FILE.is_file():
        return {"status": "success", "journal": []}
    try:
        with JOURNAL_FILE.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        return {"status": "success", "journal": rows[-20:]}
    except Exception as exc:
        logger.exception("Could not read journal")
        raise HTTPException(status_code=500, detail="Could not read forecast journal") from exc


@app.get("/export_journal")
def export_journal() -> FileResponse:
    if not JOURNAL_FILE.is_file():
        raise HTTPException(status_code=404, detail="Journal is empty")
    return FileResponse(path=str(JOURNAL_FILE), media_type="text/csv", filename="forecast_journal_backup.csv")


@app.post("/backtest")
def backtest(req: ShortTermRequest) -> Dict[str, Any]:
    return {
        "status": "not_implemented",
        "message": "Backtest is a placeholder; no historical strategy profitability has been computed.",
        "forecast_days": req.forecast_days
    }