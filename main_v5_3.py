"""EUR/USD Distribution Forecaster - refurbished FastAPI backend.

The original public endpoints and their principal response fields are retained:
    GET  /
    POST /forecast
    POST /backtest
    POST /short_term_forecast

Important modelling notes:
- The macro engine is an AR(1) model for a log real-exchange-rate proxy. It
  requires a stable positive slope interval and an ADF rejection of a unit root;
  otherwise it falls back to a random-walk-with-drift specification rather than
  forcing mean reversion by clipping the coefficient.
- CPI observations are lagged one month as a conservative release-timing proxy.
  FRED series are revised historical series, not ALFRED real-time vintages.
- The short-term model uses purged, expanding-window out-of-fold predictions to
  fit sigmoid probability calibrators. The latest unlabelled row is kept for the
  live forecast; incomplete future labels are never converted into class zero.
- Position sizing is deliberately disabled. A probability that the future close
  is higher is not the same as the probability that a take-profit is reached
  before a stop-loss.
"""

from __future__ import annotations

import csv
import io
import json
import hmac
import logging
import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import yfinance as yf
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fredapi import Fred
from pydantic import BaseModel, Field
from scipy.stats import linregress, norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("eurusd_forecaster")

app = FastAPI(
    title="EUR/USD Distribution Forecaster API",
    version="5.3.0",
    description="Macro distribution forecasts and short-term EUR/USD analysis.",
)

# Configure exact origins in production, e.g.:
# CORS_ORIGINS=https://your-app.flutterflow.app,https://your-custom-domain.com
_cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
_allow_all_origins = "*" in _cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all_origins else _cors_origins,
    allow_credentials=not _allow_all_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

FRED_API_KEY = os.getenv("FRED_API_KEY")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY")
SPREAD_PRICE = float(os.getenv("EURUSD_SPREAD_PRICE", "0.00015"))
MACRO_NEUTRAL_BAND = float(os.getenv("MACRO_NEUTRAL_BAND", "0.03"))
REQUIRE_VERIFIED_EVENT_CALENDAR = os.getenv(
    "REQUIRE_VERIFIED_EVENT_CALENDAR", "true"
).strip().lower() in {"1", "true", "yes", "on"}
KAMPALA_TZ = ZoneInfo(os.getenv("DISPLAY_TIMEZONE", "Africa/Kampala"))
SHOWMYTRADES_CALENDAR_URL = os.getenv(
    "SHOWMYTRADES_CALENDAR_URL",
    "https://showmytrades.com/api/public/calendar/v1",
)
GDELT_DOC_URL = os.getenv(
    "GDELT_DOC_URL", "https://api.gdeltproject.org/api/v2/doc/doc"
)
GDELT_TIMESPAN = os.getenv("GDELT_TIMESPAN", "7d")
GDELT_QUERY = os.getenv(
    "GDELT_QUERY",
    '(ECB OR "European Central Bank" OR euro OR Fed OR FOMC OR "Federal Reserve" OR "US dollar" OR EURUSD) '
    '(inflation OR interest OR rates OR employment OR GDP OR tariffs OR sanctions OR trade OR energy OR geopolitical OR economy)',
)
CALENDAR_CACHE_MINUTES = max(5, int(os.getenv("CALENDAR_CACHE_MINUTES", "15")))
NEWS_CACHE_MINUTES = max(1, int(os.getenv("NEWS_CACHE_MINUTES", "10")))

# Small per-instance caches stop refreshes from repeatedly calling free public APIs.
_calendar_cache: dict[str, Any] = {"expires": None, "result": None, "forecast_days": None}
_news_cache: dict[str, Any] = {"expires": None, "result": None, "max_articles": None}


# ---------------------------------------------------------------------------
# v5.3 reliability, persistence and notification configuration
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN", "").strip()
ENABLE_TELEGRAM_ALERTS = os.getenv("ENABLE_TELEGRAM_ALERTS", "false").lower() in {"1", "true", "yes", "on"}
ALERT_COOLDOWN_SECONDS = max(60, int(os.getenv("ALERT_COOLDOWN_SECONDS", "14400")))
JOURNAL_FILE = Path(os.getenv("JOURNAL_FILE", "forecast_journal.jsonl"))
MAX_JOURNAL_ROWS = max(1, int(os.getenv("MAX_JOURNAL_ROWS", "200")))
SUPABASE_TIMEOUT_SECONDS = max(2, int(os.getenv("SUPABASE_TIMEOUT_SECONDS", "8")))


def _safe_json(value: Any) -> Any:
    """Convert NumPy/Pandas values to JSON-safe Python values without hiding nulls."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.isoformat()
    if isinstance(value, (str, int, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _supabase_headers(prefer: Optional[str] = None) -> dict[str, str]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Supabase is not configured")
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _supabase_rest(path: str, *, method: str = "GET", payload: Any = None,
                   params: Optional[dict[str, Any]] = None, prefer: Optional[str] = None) -> Any:
    url = f"{SUPABASE_URL}/rest/v1/{path.lstrip('/')}"
    response = requests.request(
        method, url, headers=_supabase_headers(prefer), json=payload,
        params=params, timeout=SUPABASE_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Supabase REST {method} {path} failed ({response.status_code}): {response.text[:400]}")
    if not response.text:
        return None
    try:
        return response.json()
    except ValueError:
        return response.text


def _append_local_journal(row: dict[str, Any]) -> None:
    JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_safe_json(row), separators=(",", ":")) + "\n")


def _persist_forecast_journal(data: dict[str, Any]) -> tuple[bool, str]:
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "created_at": now,
        "symbol": "EURUSD",
        "forecast_days": data.get("forecast_days"),
        "signal": data.get("signal", "HOLD"),
        "current_price": data.get("current_price"),
        "probability_up": data.get("probability_up"),
        "payload": _safe_json(data),
    }
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            _supabase_rest("forecast_journal", method="POST", payload=row, prefer="return=minimal")
            return True, "supabase"
        except Exception as exc:
            logger.exception("Supabase journal insert failed")
            # The local copy is a best-effort safety net; Render's local disk may be ephemeral.
            try:
                _append_local_journal(row)
                return False, f"local_fallback_after_supabase_error_{type(exc).__name__}"
            except Exception:
                logger.exception("Local journal fallback also failed")
                return False, "journal_write_failed"
    try:
        _append_local_journal(row)
        return True, "local_jsonl_non_durable_on_ephemeral_hosts"
    except Exception:
        logger.exception("Local journal write failed")
        return False, "journal_write_failed"


def _telegram_send(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Telegram bot token/chat ID not configured")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    response = requests.post(
        url,
        json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram rejected message: {payload}")


def _complete_alert_outbox(outbox_id: int, success: bool, error: Optional[str] = None) -> None:
    _supabase_rest(
        "rpc/complete_forecast_alert",
        method="POST",
        payload={"p_outbox_id": int(outbox_id), "p_success": bool(success), "p_error": (error or "")[:500]},
    )


def _deliver_pending_alerts(limit: int = 3) -> int:
    """Retry pending outbox messages. A DB failure blocks delivery rather than bypassing cooldown."""
    if not (ENABLE_TELEGRAM_ALERTS and SUPABASE_URL and SUPABASE_KEY and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return 0
    rows = _supabase_rest(
        "forecast_alert_outbox",
        params={"select": "id,message,status", "status": "eq.pending", "order": "created_at.asc", "limit": str(limit)},
    ) or []
    delivered = 0
    for row in rows:
        outbox_id = row.get("id")
        if not outbox_id:
            continue
        try:
            _telegram_send(str(row.get("message") or "EUR/USD forecast alert"))
            _complete_alert_outbox(int(outbox_id), True)
            delivered += 1
        except Exception as exc:
            logger.error("Telegram outbox delivery failed for id=%s (%s)", outbox_id, type(exc).__name__)
            try:
                _complete_alert_outbox(int(outbox_id), False, str(exc))
            except Exception:
                logger.exception("Could not update failed Telegram outbox row")
    return delivered


def _queue_and_deliver_alert(data: dict[str, Any]) -> dict[str, Any]:
    """Queue a transition atomically in Postgres, then deliver through Telegram."""
    if not ENABLE_TELEGRAM_ALERTS:
        return {"enabled": False, "status": "disabled_by_configuration"}
    if not all((SUPABASE_URL, SUPABASE_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)):
        return {"enabled": True, "status": "blocked_missing_supabase_or_telegram_configuration"}

    # First retry older pending messages so a temporary Telegram outage does not lose them.
    try:
        _deliver_pending_alerts(limit=2)
    except Exception:
        logger.exception("Pending Telegram alert retry failed")

    signal = str(data.get("signal", "HOLD")).upper()
    if signal not in {"BUY", "SELL", "HOLD"}:
        return {"enabled": True, "status": "blocked_invalid_signal"}

    p_up = data.get("probability_up")
    price = data.get("current_price")
    message = (
        f"EUR/USD forecast state: {signal}\n"
        f"Price: {price if price is not None else 'unavailable'}\n"
        f"Model P(close higher): {p_up if p_up is not None else 'unavailable'}\n"
        f"Event risk: {data.get('event_risk', 'UNKNOWN')}\n"
        f"Data as of: {data.get('data_as_of', 'unknown')}\n"
        "Research estimate only. Not an instruction to trade; verify spreads, news and execution."
    )
    try:
        result = _supabase_rest(
            "rpc/claim_forecast_alert",
            method="POST",
            payload={
                "p_symbol": "EURUSD",
                "p_signal": signal,
                "p_cooldown_seconds": ALERT_COOLDOWN_SECONDS,
                "p_message": message,
            },
        )
        if isinstance(result, list) and result:
            result = result[0]
        if not isinstance(result, dict) or not result.get("queued"):
            return {"enabled": True, "status": "not_queued", "reason": (result or {}).get("reason") if isinstance(result, dict) else "unexpected_rpc_response"}
        _deliver_pending_alerts(limit=3)
        return {"enabled": True, "status": "queued", "outbox_id": result.get("outbox_id")}
    except Exception as exc:
        logger.exception("Atomic alert queue operation failed; notification blocked")
        return {"enabled": True, "status": "blocked_database_or_delivery_error", "error_type": type(exc).__name__}


def _read_local_journal() -> list[dict[str, Any]]:
    if not JOURNAL_FILE.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with JOURNAL_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL journal line")
    return rows[-MAX_JOURNAL_ROWS:]


def _read_journal_rows(limit: int = MAX_JOURNAL_ROWS) -> tuple[list[dict[str, Any]], str]:
    limit = max(1, min(int(limit), 1000))
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            rows = _supabase_rest(
                "forecast_journal",
                params={"select": "id,created_at,symbol,forecast_days,signal,current_price,probability_up,payload", "order": "created_at.desc", "limit": str(limit)},
            ) or []
            return list(rows), "supabase"
        except Exception as exc:
            logger.error("Supabase journal read failed (%s); trying local fallback", type(exc).__name__)
            local_rows = _read_local_journal()
            if local_rows:
                return list(reversed(local_rows[-limit:])), "local_jsonl_fallback_supabase_unavailable"
            raise
    return list(reversed(_read_local_journal()[-limit:])), "local_jsonl_non_durable_on_ephemeral_hosts"



# ============================================================================
# REQUEST MODELS
# ============================================================================

class ForecastRequest(BaseModel):
    horizon_years: float = Field(default=3.0, gt=0.0, le=10.0)
    pi_us: float = Field(default=0.02, ge=-0.05, le=0.25)
    pi_ez: float = Field(default=0.02, ge=-0.05, le=0.25)


class BacktestRequest(BaseModel):
    # Supports both the macro backtest request and the legacy short-term
    # FlutterFlow request without altering the /forecast request schema.
    horizon_years: Optional[float] = Field(default=None, gt=0.0, le=10.0)
    pi_us: float = Field(default=0.02, ge=-0.05, le=0.25)
    pi_ez: float = Field(default=0.02, ge=-0.05, le=0.25)
    forecast_days: Optional[int] = Field(default=None, ge=1, le=14)


class ShortTermRequest(BaseModel):
    forecast_days: int = Field(default=3, ge=1, le=14)


# ============================================================================
# COMMON UTILITIES
# ============================================================================


def _finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert a value to a finite Python float, otherwise return default."""
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _utc_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index, utc=True, errors="coerce")
    return pd.DatetimeIndex(idx)


def _clean_ohlc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardise a Yahoo/Twelve Data OHLC frame and its UTC index."""
    if frame is None or frame.empty:
        return pd.DataFrame()

    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).strip().title() for c in out.columns]

    required = ["Open", "High", "Low", "Close"]
    if not all(col in out.columns for col in required):
        return pd.DataFrame()

    out.index = _utc_index(out.index)
    out = out.loc[~out.index.isna()].copy()
    for col in required + (["Volume"] if "Volume" in out.columns else []):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=required)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[(out["Close"] > 0) & (out["Open"] > 0) & (out["High"] > 0) & (out["Low"] > 0)]
    out = out[(out["High"] >= out[["Open", "Close", "Low"]].max(axis=1)) & (out["Low"] <= out[["Open", "Close", "High"]].min(axis=1))]
    return out


def _drop_current_hour(frame: pd.DataFrame) -> pd.DataFrame:
    """Avoid training/inference on a possibly incomplete current hourly candle."""
    if frame.empty:
        return frame
    current_hour = pd.Timestamp.now(tz="UTC").floor("h")
    return frame.loc[frame.index < current_hour].copy()


def _log_exception(message: str, exc: Exception) -> None:
    # Do not log raw provider exception text: request URLs may contain API keys.
    logger.error("%s (%s)", message, type(exc).__name__)


# ============================================================================
# MACRO DATA AND MODEL
# ============================================================================


def _to_month_end(series: pd.Series, name: str) -> pd.Series:
    series = pd.Series(series).dropna().astype(float)
    series.index = pd.to_datetime(series.index)
    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_localize(None)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.resample("ME").last().rename(name)


def fetch_macro_data() -> pd.DataFrame:
    """Fetch spot and CPI data; CPI is lagged one month for release timing.

    DEXUSEU is USD per EUR. The CPI lag is a practical approximation, not an
    exact release-calendar reconstruction. FRED revised history is also used.
    """
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY environment variable is not set")

    fred = Fred(api_key=FRED_API_KEY)
    spot = _to_month_end(fred.get_series("DEXUSEU"), "spot")
    us_cpi = _to_month_end(fred.get_series("CPIAUCSL"), "us_cpi").shift(1)
    ez_cpi = _to_month_end(
        fred.get_series("CP0000EZ19M086NEST"), "ez_cpi"
    ).shift(1)

    df = pd.concat([spot, us_cpi, ez_cpi], axis=1).dropna()
    df = df.loc[df.index >= pd.Timestamp("1999-01-01")].copy()
    df = df[(df["spot"] > 0) & (df["us_cpi"] > 0) & (df["ez_cpi"] > 0)]

    if len(df) < 60:
        raise RuntimeError(
            f"Insufficient aligned monthly macro data: {len(df)} observations"
        )

    df["s"] = np.log(df["spot"])
    df["p_us"] = np.log(df["us_cpi"])
    df["p_ez"] = np.log(df["ez_cpi"])
    # q = log(USD per EUR) + log(US prices) - log(Euro Area prices)
    df["q"] = df["s"] + df["p_us"] - df["p_ez"]
    return df.replace([np.inf, -np.inf], np.nan).dropna()


@dataclass
class MacroParameters:
    model_type: str
    phi: float
    intercept: float
    q_bar: Optional[float]
    sigma_monthly: float
    drift_monthly: float
    half_life_years: Optional[float]
    mean_reversion_supported: bool
    observations: int
    adf_p_value: Optional[float]


def _fit_macro_parameters(q_values: pd.Series) -> MacroParameters:
    q = pd.Series(q_values).dropna().astype(float)
    q_lag = q.shift(1)
    paired = pd.concat([q.rename("q"), q_lag.rename("q_lag")], axis=1).dropna()
    if len(paired) < 48:
        raise RuntimeError("At least 48 monthly observations are required to fit the macro model")

    regression = linregress(paired["q_lag"].to_numpy(), paired["q"].to_numpy())
    phi_raw = float(regression.slope)
    intercept_raw = float(regression.intercept)
    phi_se = float(regression.stderr) if math.isfinite(float(regression.stderr)) else float("inf")
    phi_lower_95 = phi_raw - 1.96 * phi_se
    phi_upper_95 = phi_raw + 1.96 * phi_se
    try:
        adf_p_value = float(adfuller(q.to_numpy(), regression="c", autolag="AIC")[1])
    except Exception as exc:
        logger.warning("ADF stationarity test could not be completed (%s)", type(exc).__name__)
        adf_p_value = 1.0

    # Do not clip an unstable coefficient into a mean-reverting one. Require
    # both an approximate 95% slope interval inside (0, 1) and an ADF rejection
    # of a unit root. This is still a model diagnostic, not proof of PPP.
    if (
        0.0 < phi_raw < 1.0
        and phi_lower_95 > 0.0
        and phi_upper_95 < 1.0
        and adf_p_value < 0.05
        and math.isfinite(intercept_raw)
    ):
        q_bar = intercept_raw / (1.0 - phi_raw)
        residuals = paired["q"] - (intercept_raw + phi_raw * paired["q_lag"])
        sigma_monthly = float(residuals.std(ddof=1))
        if not math.isfinite(sigma_monthly) or sigma_monthly <= 0:
            raise RuntimeError("Macro AR(1) residual volatility is invalid")
        half_life = math.log(0.5) / math.log(phi_raw) / 12.0
        return MacroParameters(
            model_type="AR(1)_mean_reverting",
            phi=phi_raw,
            intercept=intercept_raw,
            q_bar=float(q_bar),
            sigma_monthly=sigma_monthly,
            drift_monthly=0.0,
            half_life_years=float(half_life),
            mean_reversion_supported=True,
            observations=len(paired),
            adf_p_value=adf_p_value,
        )

    # No statistically admissible positive mean-reversion coefficient: use a
    # random walk with the observed monthly drift and monthly innovation sigma.
    changes = q.diff().dropna()
    drift = float(changes.mean())
    sigma = float(changes.std(ddof=1))
    if not math.isfinite(sigma) or sigma <= 0:
        raise RuntimeError("Macro random-walk volatility is invalid")
    return MacroParameters(
        model_type="random_walk_fallback",
        phi=1.0,
        intercept=drift,
        q_bar=None,
        sigma_monthly=sigma,
        drift_monthly=drift,
        half_life_years=None,
        mean_reversion_supported=False,
        observations=len(changes),
        adf_p_value=adf_p_value,
    )


def _macro_volatility_regime(df: pd.DataFrame) -> tuple[int, Optional[float]]:
    """Return stable labels: 0=lower-volatility, 1=higher-volatility."""
    returns = df["s"].diff()
    rolling_vol = returns.rolling(window=6, min_periods=6).std().dropna()
    if len(rolling_vol) < 30:
        return 0, None

    values = rolling_vol.to_numpy().reshape(-1, 1)
    try:
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            random_state=42,
            n_init=5,
            reg_covar=1e-8,
        )
        gmm.fit(values)
        order = np.argsort(gmm.means_.ravel())
        probabilities = gmm.predict_proba(values[-1:])[0]
        high_component = int(order[-1])
        high_probability = float(probabilities[high_component])
        return (1 if high_probability >= 0.5 else 0), high_probability
    except Exception as exc:  # regime is diagnostic; it must not break the forecast
        logger.warning("Macro volatility-regime estimation failed (%s)", type(exc).__name__)
        return 0, None


def _macro_distribution(
    params: MacroParameters,
    current_q: float,
    horizon_months: float,
    horizon_years: float,
    pi_us: float,
    pi_ez: float,
) -> tuple[float, float]:
    """Return expected log spot change and its variance."""
    if params.mean_reversion_supported and params.q_bar is not None:
        phi_n = params.phi ** horizon_months
        expected_q_change = (phi_n - 1.0) * (current_q - params.q_bar)
        variance_q = (
            params.sigma_monthly**2
            * (1.0 - params.phi ** (2.0 * horizon_months))
            / (1.0 - params.phi**2)
        )
    else:
        expected_q_change = params.drift_monthly * horizon_months
        variance_q = params.sigma_monthly**2 * horizon_months

    # From q = s + p_US - p_EZ, nominal log spot drift includes pi_EZ - pi_US.
    inflation_differential = (pi_ez - pi_us) * horizon_years
    mean_log_change = expected_q_change + inflation_differential
    return float(mean_log_change), float(max(variance_q, 0.0))


def get_macro_state() -> dict[str, Any]:
    df = fetch_macro_data()
    params = _fit_macro_parameters(df["q"])
    regime, high_probability = _macro_volatility_regime(df)
    current_q = float(df["q"].iloc[-1])
    deviation = (
        float(current_q - params.q_bar)
        if params.mean_reversion_supported and params.q_bar is not None
        else None
    )
    return {
        "deviation_q": deviation,
        "current_regime": regime,
        "high_volatility_probability": high_probability,
        "mean_reversion_supported": params.mean_reversion_supported,
        "half_life_years": params.half_life_years,
        "model_type": params.model_type,
        "data_as_of": df.index[-1].strftime("%Y-%m-%d"),
    }


# ============================================================================
# MACRO ENDPOINTS
# ============================================================================

@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "status": "API is running",
        "engine": "EUR/USD Distribution Forecaster v5.3 candidate",
        "version": "5.3.0",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness check. Does not imply external providers are reachable."""
    return {"status": "ok", "version": "5.3.0", "timestamp_utc": datetime.now(timezone.utc).isoformat()}


@app.get("/ready")
def readiness() -> dict[str, Any]:
    """Configuration readiness, deliberately separate from API liveness."""
    telegram_configured = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and SUPABASE_URL and SUPABASE_KEY)
    core_ready = bool(FRED_API_KEY)
    alerts_ready = (not ENABLE_TELEGRAM_ALERTS) or telegram_configured
    return {
        "status": "ready" if core_ready and alerts_ready else "degraded",
        "checks": {
            "fred_configured": core_ready,
            "twelve_data_configured": bool(TWELVE_DATA_KEY),
            "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
            "telegram_enabled": bool(ENABLE_TELEGRAM_ALERTS),
            "telegram_configured": telegram_configured,
        },
        "note": "Configuration-only readiness; provider connectivity is verified at request time.",
    }


@app.post("/forecast")
def get_forecast(req: ForecastRequest) -> dict[str, Any]:
    try:
        df = fetch_macro_data()
        params = _fit_macro_parameters(df["q"])
        current_spot = float(df["spot"].iloc[-1])
        current_q = float(df["q"].iloc[-1])
        regime, high_probability = _macro_volatility_regime(df)

        horizon_months = req.horizon_years * 12.0
        mean_delta, variance = _macro_distribution(
            params=params,
            current_q=current_q,
            horizon_months=horizon_months,
            horizon_years=req.horizon_years,
            pi_us=req.pi_us,
            pi_ez=req.pi_ez,
        )
        std_delta = math.sqrt(max(variance, 0.0))
        log_median = math.log(current_spot) + mean_delta
        median_rate = math.exp(log_median)
        expected_rate = math.exp(log_median + 0.5 * variance)
        p_up = float(norm.cdf(mean_delta / std_delta)) if std_delta > 0 else 0.5
        z90 = 1.6448536269514722

        ppp_anchor = (
            float(np.exp(params.q_bar - df["p_us"].iloc[-1] + df["p_ez"].iloc[-1]))
            if params.mean_reversion_supported and params.q_bar is not None
            else None
        )
        deviation = (
            float(current_q - params.q_bar)
            if params.mean_reversion_supported and params.q_bar is not None
            else None
        )

        historical_data = [
            {"date": d.strftime("%Y-%m-%d"), "price": float(row["spot"])}
            for d, row in df.tail(60).iterrows()
        ]

        fan_chart: list[dict[str, Any]] = []
        steps = 12
        for i in range(1, steps + 1):
            years_ahead = req.horizon_years * i / steps
            months_ahead = years_ahead * 12.0
            m_delta, v_delta = _macro_distribution(
                params=params,
                current_q=current_q,
                horizon_months=months_ahead,
                horizon_years=years_ahead,
                pi_us=macro_req.pi_us,
                pi_ez=macro_req.pi_ez,
            )
            sd = math.sqrt(max(v_delta, 0.0))
            log_mid = math.log(current_spot) + m_delta
            median = math.exp(log_mid)
            arithmetic_mean = math.exp(log_mid + 0.5 * v_delta)
            future_date = (
                datetime.now(timezone.utc)
                + timedelta(days=years_ahead * 365.2425)
            ).date().isoformat()
            fan_chart.append(
                {
                    "months_ahead": int(round(months_ahead)),
                    "date": future_date,
                    "median": float(median),
                    "mean": float(arithmetic_mean),
                    "q05": float(math.exp(log_mid - z90 * sd)),
                    "q25": float(math.exp(log_mid - 0.6744897501960817 * sd)),
                    "q75": float(math.exp(log_mid + 0.6744897501960817 * sd)),
                    "q95": float(math.exp(log_mid + z90 * sd)),
                }
            )

        return {
            "status": "success",
            "data": {
                "inputs": req.model_dump(),
                "current_state": {
                    "spot_rate": current_spot,
                    "ppp_fair_value": ppp_anchor,
                    "deviation_q": deviation,
                    "half_life_years": params.half_life_years,
                    "current_regime": regime,
                    "total_regimes": 2,
                    "regime_label": "High volatility" if regime == 1 else "Low volatility",
                    "high_volatility_probability": high_probability,
                    "macro_model": params.model_type,
                    "phi_monthly": float(params.phi),
                    "sigma_monthly": float(params.sigma_monthly),
                    "mean_reversion_supported": params.mean_reversion_supported,
                    "observations_used": params.observations,
                    "adf_p_value": params.adf_p_value,
                    "data_as_of": df.index[-1].strftime("%Y-%m-%d"),
                    "cpi_release_lag_months": 1,
                    "cpi_vintage_warning": (
                        "FRED revised history is used; this is not an ALFRED real-time-vintage backtest."
                    ),
                },
                "forecast": {
                    "p_up": p_up,
                    "expected_rate": float(expected_rate),
                    "median_rate": float(median_rate),
                    "ci_90_low": float(math.exp(log_median - z90 * std_delta)),
                    "ci_90_high": float(math.exp(log_median + z90 * std_delta)),
                    "log_change_mean": mean_delta,
                    "log_change_variance": variance,
                },
                "historical_data": historical_data,
                "fan_chart": fan_chart,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        _log_exception("/forecast failed", exc)
        raise HTTPException(status_code=503, detail="Forecast calculation failed; check provider configuration and server logs.") from exc


def _require_admin_token(provided_token: Optional[str]) -> None:
    if not ADMIN_API_TOKEN:
        raise HTTPException(status_code=503, detail="Journal access is disabled until ADMIN_API_TOKEN is configured.")
    if not provided_token or not hmac.compare_digest(provided_token, ADMIN_API_TOKEN):
        raise HTTPException(status_code=401, detail="Valid X-API-Token header required.")


@app.get("/journal")
def get_journal(
    limit: int = Query(default=20, ge=1, le=1000),
    x_api_token: Optional[str] = Header(default=None, alias="X-API-Token"),
) -> dict[str, Any]:
    _require_admin_token(x_api_token)
    try:
        rows, source = _read_journal_rows(limit)
        return {"status": "success", "journal": _safe_json(rows), "count": len(rows), "source": source}
    except Exception as exc:
        logger.exception("Journal read failed")
        raise HTTPException(status_code=503, detail="Journal storage is unavailable") from exc


@app.get("/export_journal")
def export_journal(
    limit: int = Query(default=1000, ge=1, le=10000),
    x_api_token: Optional[str] = Header(default=None, alias="X-API-Token"),
) -> Response:
    _require_admin_token(x_api_token)
    try:
        rows, source = _read_journal_rows(limit)
        output = io.StringIO()
        fieldnames = ["id", "created_at", "symbol", "forecast_days", "signal", "current_price", "probability_up", "payload"]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = dict(row)
            if isinstance(clean.get("payload"), (dict, list)):
                clean["payload"] = json.dumps(_safe_json(clean["payload"]), separators=(",", ":"))
            writer.writerow({key: clean.get(key) for key in fieldnames})
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="eurusd_forecast_journal.csv"', "X-Journal-Source": source},
        )
    except Exception as exc:
        logger.exception("Journal export failed")
        raise HTTPException(status_code=503, detail="Journal export is unavailable") from exc


@app.post("/backtest")
def run_backtest(req: BacktestRequest) -> dict[str, Any]:
    if req.forecast_days is not None and req.horizon_years is None:
        return {
            "status": "not_implemented",
            "message": (
                "A short-term trading-strategy backtest is not implemented yet. "
                "The macro distribution backtest is available by sending horizon_years instead."
            ),
            "forecast_days": req.forecast_days,
            "trading_profitability_validated": False,
        }
    macro_req = ForecastRequest(
        horizon_years=req.horizon_years if req.horizon_years is not None else 3.0,
        pi_us=req.pi_us,
        pi_ez=req.pi_ez,
    )
    try:
        df = fetch_macro_data()
        horizon_months = max(1, int(round(macro_req.horizon_years * 12.0)))
        minimum_training_months = 48
        maximum_tests = len(df) - horizon_months - minimum_training_months
        if maximum_tests < 12:
            raise ValueError(
                "Not enough history for this horizon. Reduce horizon_years or use more data."
            )

        test_periods = min(60, maximum_tests)
        test_start = len(df) - horizon_months - test_periods
        probabilities: list[float] = []
        baseline_probabilities: list[float] = []
        actuals: list[int] = []
        hits_50: list[int] = []
        hits_90: list[int] = []
        z90 = 1.6448536269514722
        z50 = 0.6744897501960817

        for i in range(test_start, len(df) - horizon_months):
            # At origin i, data through month i is observable and can be used
            # to estimate parameters; the future outcome remains untouched.
            train_q = df["q"].iloc[: i + 1]
            params = _fit_macro_parameters(train_q)
            current_q = float(df["q"].iloc[i])
            current_spot = float(df["spot"].iloc[i])
            mean_delta, variance = _macro_distribution(
                params=params,
                current_q=current_q,
                horizon_months=float(horizon_months),
                horizon_years=horizon_months / 12.0,
                pi_us=macro_req.pi_us,
                pi_ez=macro_req.pi_ez,
            )
            std_delta = math.sqrt(max(variance, 0.0))
            log_mid = math.log(current_spot) + mean_delta
            p_up = float(norm.cdf(mean_delta / std_delta)) if std_delta > 0 else 0.5
            q05 = math.exp(log_mid - z90 * std_delta)
            q25 = math.exp(log_mid - z50 * std_delta)
            q75 = math.exp(log_mid + z50 * std_delta)
            q95 = math.exp(log_mid + z90 * std_delta)

            actual_spot = float(df["spot"].iloc[i + horizon_months])
            actual_direction = 1 if actual_spot > current_spot else 0
            probabilities.append(p_up)
            actuals.append(actual_direction)

            # Causal expanding prevalence baseline: only include direction labels
            # whose future endpoint was already observable at forecast origin i.
            past_origins = df["spot"].iloc[: i - horizon_months + 1].to_numpy(dtype=float)
            matured_futures = df["spot"].iloc[horizon_months : i + 1].to_numpy(dtype=float)
            matured_labels = (matured_futures > past_origins).astype(int)
            baseline_probabilities.append(float(np.mean(matured_labels)) if len(matured_labels) else 0.5)
            hits_50.append(int(q25 <= actual_spot <= q75))
            hits_90.append(int(q05 <= actual_spot <= q95))

        probs = np.asarray(probabilities, dtype=float)
        actual = np.asarray(actuals, dtype=int)
        brier = float(np.mean((probs - actual) ** 2))
        actual_up_rate = float(np.mean(actual))
        baseline_probs = np.asarray(baseline_probabilities, dtype=float)
        baseline_brier = float(np.mean((baseline_probs - actual) ** 2))
        brier_skill = (
            float(1.0 - brier / baseline_brier)
            if baseline_brier > 0
            else None
        )
        cov_50 = float(np.mean(hits_50))
        cov_90 = float(np.mean(hits_90))

        bins = np.linspace(0.0, 1.0, 6)
        ece = 0.0
        calibration_curve: list[dict[str, float]] = []
        for b in range(len(bins) - 1):
            mask = (probs >= bins[b]) & (
                probs < bins[b + 1] if b < len(bins) - 2 else probs <= bins[b + 1]
            )
            if np.any(mask):
                avg_pred = float(np.mean(probs[mask]))
                avg_actual = float(np.mean(actual[mask]))
                ece += (float(np.sum(mask)) / len(probs)) * abs(avg_pred - avg_actual)
                calibration_curve.append(
                    {
                        "predicted": avg_pred,
                        "actual": avg_actual,
                        "count": int(np.sum(mask)),
                    }
                )
            else:
                midpoint = float((bins[b] + bins[b + 1]) / 2.0)
                calibration_curve.append(
                    {"predicted": midpoint, "actual": midpoint, "count": 0}
                )
        reliability_index = float(1.0 - min(max(ece, 0.0), 1.0))

        return {
            "status": "success",
            "data": {
                "horizon_years": macro_req.horizon_years,
                "n_tests": len(probs),
                "brier_score": brier,
                "coverage_50": cov_50,
                "coverage_90": cov_90,
                "reliability_index": reliability_index,
                "calibration_curve": calibration_curve,
                "baseline_brier_score": baseline_brier,
                "brier_skill_score": brier_skill,
                "actual_up_rate": actual_up_rate,
                "mean_baseline_up_probability": float(np.mean(baseline_probs)),
                "mean_predicted_up_probability": float(np.mean(probs)),
                "test_start": df.index[test_start].strftime("%Y-%m-%d"),
                "test_end": df.index[len(df) - horizon_months - 1].strftime("%Y-%m-%d"),
                "methodology_note": (
                    "Expanding-origin monthly backtest. Brier skill is compared with a causal expanding "
                    "historical directional-prevalence baseline. Forecast windows overlap, so test outcomes "
                    "are not independent. CPI is lagged one month but uses revised FRED history, "
                    "not point-in-time ALFRED vintages."
                ),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        _log_exception("/backtest failed", exc)
        raise HTTPException(status_code=503, detail="Macro backtest failed; check data-provider configuration and server logs.") from exc


# ============================================================================
# SHORT-TERM DATA FETCHING
# ============================================================================


def _fetch_twelve_data_ohlc(symbol: str, outputsize: int = 5000) -> pd.DataFrame:
    if not TWELVE_DATA_KEY:
        return pd.DataFrame()
    url = "https://api.twelvedata.com/time_series"
    response = requests.get(
        url,
        params={
            "symbol": symbol,
            "interval": "1h",
            "outputsize": outputsize,
            "apikey": TWELVE_DATA_KEY,
            "timezone": "UTC",
        },
        timeout=12,
    )
    response.raise_for_status()
    payload = response.json()
    if "values" not in payload:
        raise RuntimeError(
            f"Twelve Data returned no candle values for {symbol}: "
            f"{payload.get('message', payload.get('code', 'unknown response'))}"
        )
    raw = pd.DataFrame(payload["values"])
    if "datetime" not in raw.columns:
        raise RuntimeError(f"Twelve Data response for {symbol} has no datetime field")
    raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True, errors="coerce")
    raw = raw.set_index("datetime")
    return _clean_ohlc_frame(raw)


def _fetch_yahoo_ohlc(symbol: str, period: str = "60d") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    raw = ticker.history(
        period=period,
        interval="1h",
        auto_adjust=False,
        actions=False,
        timeout=12,
    )
    return _clean_ohlc_frame(raw)


def fetch_hourly_market_data() -> tuple[pd.DataFrame, str, bool, str]:
    """Fetch EUR/USD and, separately, real DXY data when available."""
    eur = pd.DataFrame()
    eur_source = ""
    failures: list[str] = []

    if TWELVE_DATA_KEY:
        try:
            eur = _fetch_twelve_data_ohlc("EUR/USD", outputsize=5000)
            if not eur.empty:
                eur_source = "Twelve Data"
        except Exception as exc:
            failures.append(f"Twelve Data EUR/USD ({type(exc).__name__})")
            logger.warning("Twelve Data EUR/USD fetch failed (%s)", type(exc).__name__)

    # Try Yahoo as a second source when the first provider returns a short
    # history. Prefer whichever source supplies more usable candles.
    if eur.empty or len(eur) < 1000:
        try:
            yahoo_eur = _fetch_yahoo_ohlc("EURUSD=X", period="60d")
            if len(yahoo_eur) > len(eur):
                eur = yahoo_eur
                eur_source = "Yahoo Finance fallback"
        except Exception as exc:
            failures.append(f"Yahoo EUR/USD ({type(exc).__name__})")
            logger.warning("Yahoo EUR/USD fetch failed (%s)", type(exc).__name__)

    eur = _drop_current_hour(eur)
    if eur.empty or len(eur) < 100:
        detail = "; ".join(failures[-2:])
        raise RuntimeError(
            "Insufficient completed hourly EUR/USD candles from available providers"
            + (f" ({detail})" if detail else "")
        )

    dxy = pd.DataFrame()
    dxy_source = "unavailable"
    try:
        dxy = _fetch_yahoo_ohlc("DX-Y.NYB", period="60d")
        dxy = _drop_current_hour(dxy)
        if not dxy.empty:
            dxy_source = "Yahoo Finance"
    except Exception as exc:
        logger.warning("DXY fetch failed; DXY features will be marked unavailable (%s)", type(exc).__name__)

    # Do not substitute EUR/USD for DXY. Align only genuine DXY observations,
    # carrying the most recent observation forward for at most two hours.
    out = eur[["Open", "High", "Low", "Close"]].copy()
    out["Volume"] = eur["Volume"] if "Volume" in eur.columns else np.nan
    if not dxy.empty and "Close" in dxy.columns:
        aligned_dxy = dxy["Close"].reindex(
            out.index,
            method="ffill",
            tolerance=pd.Timedelta(hours=2),
        )
        out["Close_DXY"] = aligned_dxy
        dxy_available = bool(aligned_dxy.notna().any())
    else:
        out["Close_DXY"] = np.nan
        dxy_available = False

    return out, eur_source, dxy_available, dxy_source


# ============================================================================
# EVENT CALENDAR + NEWS INTELLIGENCE
# ============================================================================


def _parse_event_time(event: dict[str, Any]) -> Optional[pd.Timestamp]:
    """Parse ShowMyTrades event time, preferring its Unix UTC timestamp."""
    raw_ts = event.get("ts")
    if raw_ts not in (None, ""):
        try:
            return pd.to_datetime(float(raw_ts), unit="s", utc=True, errors="coerce")
        except (TypeError, ValueError, OverflowError):
            pass

    raw = event.get("time") or event.get("datetime") or event.get("date")
    if raw is None:
        return None
    parsed = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _normalise_calendar_event(event: dict[str, Any]) -> dict[str, Any]:
    """Return a FlutterFlow-friendly event with explicit UTC and Kampala times."""
    event_time = _parse_event_time(event)
    if event_time is None or pd.isna(event_time):
        utc_iso = None
        kampala_iso = None
        kampala_display = None
    else:
        event_time = pd.Timestamp(event_time).tz_convert("UTC")
        kampala_time = event_time.tz_convert(KAMPALA_TZ)
        utc_iso = event_time.isoformat()
        kampala_iso = kampala_time.isoformat()
        kampala_display = kampala_time.strftime("%a, %d %b %Y · %H:%M EAT")

    return {
        "id": str(event.get("id") or ""),
        "title": str(event.get("title") or "Economic event"),
        "currency": str(event.get("currency") or "").upper(),
        "impact": str(event.get("impact") or "unknown").lower(),
        "datetime_utc": utc_iso,
        "datetime_kampala": kampala_iso,
        "datetime_kampala_display": kampala_display,
        "forecast": str(event.get("forecast") or ""),
        "previous": str(event.get("previous") or ""),
        "actual": str(event.get("actual") or ""),
        "status": str(event.get("status") or "scheduled"),
        "moved_from_utc": str(event.get("moved_from") or "") or None,
        "period": str(event.get("period") or "") or None,
        "confidence": str(event.get("confidence") or "unknown"),
    }


def get_showmytrades_calendar(days: int) -> dict[str, Any]:
    """Fetch and cache EUR/USD high- and medium-impact events; no API key needed."""
    days = max(1, min(int(days), 30))
    now = datetime.now(timezone.utc)
    cached_result = _calendar_cache.get("result")
    expires = _calendar_cache.get("expires")
    if (
        cached_result is not None
        and expires is not None
        and _calendar_cache.get("forecast_days") == days
        and now < expires
    ):
        return cached_result

    try:
        response = requests.get(
            SHOWMYTRADES_CALENDAR_URL,
            params={
                "currency": "EUR,USD",
                "impact": "high,medium",
                "days": days,
                "format": "json",
            },
            headers={
                "Accept": "application/json",
                "User-Agent": "EURUSD-Distribution-Forecaster/2.1 (personal research)",
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        raw_events = payload.get("events") if isinstance(payload, dict) else None
        if not isinstance(raw_events, list):
            raise RuntimeError("ShowMyTrades response did not contain an events list")

        events = [
            _normalise_calendar_event(event)
            for event in raw_events
            if isinstance(event, dict)
        ]
        events.sort(
            key=lambda item: item.get("datetime_utc") or "9999-12-31T23:59:59+00:00"
        )
        result = {
            "status": "available",
            "provider": "ShowMyTrades",
            "currencies": ["EUR", "USD"],
            "days_ahead": days,
            "retrieved_at_utc": now.isoformat(),
            "server_time_utc": payload.get("server_time_iso"),
            "coverage_start_utc": payload.get("from"),
            "coverage_end_utc": payload.get("to"),
            "provider_event_count": payload.get("count", len(events)),
            "events": events,
            "event_count": len(events),
            "attribution": payload.get("attribution") or {
                "required": False,
                "text": "Economic calendar by ShowMyTrades",
                "url": "https://showmytrades.com/tools/economic-calendar",
            },
            "documentation": "https://showmytrades.com/docs/api/economic-calendar",
        }
    except Exception as exc:
        logger.warning("ShowMyTrades economic-calendar fetch failed (%s)", type(exc).__name__)
        result = {
            "status": "unavailable",
            "provider": "ShowMyTrades",
            "currencies": ["EUR", "USD"],
            "days_ahead": days,
            "retrieved_at_utc": now.isoformat(),
            "events": [],
            "event_count": 0,
            "error": str(exc)[:300],
            "documentation": "https://showmytrades.com/docs/api/economic-calendar",
        }

    # Do not let a transient provider outage poison the normal 15-minute cache.
    cache_minutes = CALENDAR_CACHE_MINUTES if result.get("status") == "available" else 1
    _calendar_cache.update({
        "expires": now + timedelta(minutes=cache_minutes),
        "result": result,
        "forecast_days": days,
    })
    return result


def get_event_risk(forecast_days: int) -> dict[str, Any]:
    """Classify event risk only when provider coverage and event times are verifiable."""
    calendar = get_showmytrades_calendar(forecast_days)
    if calendar.get("status") != "available":
        return {
            "event_risk": "UNKNOWN",
            "calendar_status": "ShowMyTrades calendar unavailable; event risk cannot be verified",
            "calendar_provider": "ShowMyTrades",
            "calendar_currencies": ["EUR", "USD"],
            "high_impact_events": [],
            "medium_impact_events": [],
            "upcoming_events": [],
            "calendar_error": calendar.get("error"),
        }

    # Use the provider's own UTC clock and coverage window, not Render's clock.
    now = pd.to_datetime(calendar.get("server_time_utc"), utc=True, errors="coerce")
    coverage_start = pd.to_datetime(calendar.get("coverage_start_utc"), utc=True, errors="coerce")
    coverage_end = pd.to_datetime(calendar.get("coverage_end_utc"), utc=True, errors="coerce")
    if pd.isna(now) or pd.isna(coverage_start) or pd.isna(coverage_end):
        return {
            "event_risk": "UNKNOWN",
            "calendar_status": "Provider response did not include verifiable server time and coverage bounds",
            "calendar_provider": "ShowMyTrades",
            "calendar_currencies": ["EUR", "USD"],
            "high_impact_events": [],
            "medium_impact_events": [],
            "upcoming_events": [],
        }

    window_end = now + timedelta(days=forecast_days)
    coverage_ok = bool(coverage_start <= now and coverage_end >= window_end)
    if not coverage_ok:
        return {
            "event_risk": "UNKNOWN",
            "calendar_status": "Provider coverage window does not fully cover the requested forecast horizon",
            "calendar_provider": "ShowMyTrades",
            "calendar_currencies": ["EUR", "USD"],
            "high_impact_events": [],
            "medium_impact_events": [],
            "upcoming_events": [],
            "calendar_start_utc": now.isoformat(),
            "calendar_end_utc": window_end.isoformat(),
            "provider_coverage_start_utc": coverage_start.isoformat(),
            "provider_coverage_end_utc": coverage_end.isoformat(),
        }

    high_events: list[dict[str, Any]] = []
    medium_events: list[dict[str, Any]] = []
    upcoming_events: list[dict[str, Any]] = []
    malformed_event_time = False
    malformed_event_metadata = False

    for event in calendar.get("events", []):
        event_time = pd.to_datetime(event.get("datetime_utc"), utc=True, errors="coerce")
        if pd.isna(event_time):
            malformed_event_time = True
            continue
        if now <= event_time <= window_end:
            upcoming_events.append(event)
            currency = str(event.get("currency", "")).upper()
            impact = str(event.get("impact", "")).lower()
            if impact not in {"high", "medium"} or currency not in {"EUR", "USD"}:
                malformed_event_metadata = True
            elif impact == "high":
                high_events.append(event)
            elif impact == "medium":
                medium_events.append(event)

    if malformed_event_time or malformed_event_metadata:
        risk = "UNKNOWN"
        status = "One or more event records have unverified time, currency or impact metadata"
    elif high_events:
        risk = "HIGH"
        status = "One or more high-impact EUR/USD events are scheduled in the forecast window"
    elif medium_events:
        risk = "MEDIUM"
        status = "No high-impact release found, but medium-impact EUR/USD events are scheduled"
    else:
        # Empty event lists are valid only when the full requested window is verified.
        risk = "LOW"
        status = "Verified provider coverage contains no high- or medium-impact EUR/USD event in the forecast window"

    return {
        "event_risk": risk,
        "calendar_status": status,
        "calendar_provider": "ShowMyTrades",
        "calendar_currencies": ["EUR", "USD"],
        "high_impact_events": high_events,
        "medium_impact_events": medium_events,
        "upcoming_events": upcoming_events,
        "calendar_start_utc": now.isoformat(),
        "calendar_end_utc": window_end.isoformat(),
        "provider_coverage_start_utc": coverage_start.isoformat(),
        "provider_coverage_end_utc": coverage_end.isoformat(),
        "calendar_days_ahead": forecast_days,
        "calendar_event_count": len(calendar.get("events", [])),
    }


def _article_currency_relevance(title: str, url: str = "") -> str:
    text = f"{title} {url}".lower()
    eur_terms = ("euro", "eur", "ecb", "european central bank", "eurozone", "euro area")
    usd_terms = ("dollar", "usd", "federal reserve", "fed ", "fomc", "united states", " u.s.")
    eur_match = any(term in text for term in eur_terms)
    usd_match = any(term in text for term in usd_terms)
    if eur_match and usd_match:
        return "EUR_AND_USD"
    if eur_match:
        return "EUR"
    if usd_match:
        return "USD"
    return "MACRO_RELATED"


def get_gdelt_news(max_articles: int = 50) -> dict[str, Any]:
    """Fetch recent EUR/USD macro headlines from GDELT DOC 2.0, without a key."""
    max_articles = max(1, min(int(max_articles), 75))
    now = datetime.now(timezone.utc)
    cached_result = _news_cache.get("result")
    expires = _news_cache.get("expires")
    if (
        cached_result is not None
        and expires is not None
        and _news_cache.get("max_articles") == max_articles
        and now < expires
    ):
        return cached_result

    params = {
        "query": GDELT_QUERY,
        "mode": "ArtList",
        "format": "json",
        "timespan": GDELT_TIMESPAN,
        "maxrecords": max_articles,
        "sort": "DateDesc",
    }
    try:
        response = requests.get(
            GDELT_DOC_URL,
            params=params,
            headers={
                "Accept": "application/json",
                "User-Agent": "EURUSD-Distribution-Forecaster/2.1 (personal research)",
            },
            timeout=20,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "json" not in content_type:
            raise RuntimeError(
                "GDELT returned a non-JSON response; it may have rejected the query or rate-limited this request"
            )
        payload = response.json()
        raw_articles = payload.get("articles") if isinstance(payload, dict) else None
        if not isinstance(raw_articles, list):
            raise RuntimeError("GDELT response did not contain an articles list")

        articles: list[dict[str, Any]] = []
        tones: list[float] = []
        for article in raw_articles:
            if not isinstance(article, dict):
                continue
            title = str(article.get("title") or "Untitled news article")
            url = str(article.get("url") or "")
            raw_date = article.get("seendate") or article.get("date")
            published = pd.to_datetime(raw_date, utc=True, errors="coerce")
            published_utc = None if pd.isna(published) else pd.Timestamp(published).isoformat()
            published_kampala = (
                None
                if pd.isna(published)
                else pd.Timestamp(published).tz_convert(KAMPALA_TZ).isoformat()
            )
            article_tone = _finite_float(article.get("tone"), None)
            if article_tone is not None:
                tones.append(article_tone)
            articles.append({
                "title": title,
                "url": url,
                "source_domain": str(article.get("domain") or ""),
                "published_at_utc": published_utc,
                "published_at_kampala": published_kampala,
                "source_country": str(article.get("sourcecountry") or ""),
                "language": str(article.get("language") or ""),
                "currency_relevance": _article_currency_relevance(title, url),
                "article_tone": article_tone,
            })

        result = {
            "status": "available",
            "provider": "GDELT DOC 2.0",
            "retrieved_at_utc": now.isoformat(),
            "timespan": GDELT_TIMESPAN,
            "query": GDELT_QUERY,
            "article_count": len(articles),
            "articles": articles,
            "mean_article_tone": (
                float(np.mean(tones)) if tones else None
            ),
            "tone_note": (
                "GDELT article tone is a media-language measure, not a EUR/USD bullish or bearish signal."
                if tones
                else "Per-article tone was not supplied in this response."
            ),
            "documentation": "https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/",
        }
    except Exception as exc:
        logger.warning("GDELT news fetch failed (%s)", type(exc).__name__)
        result = {
            "status": "unavailable",
            "provider": "GDELT DOC 2.0",
            "retrieved_at_utc": now.isoformat(),
            "timespan": GDELT_TIMESPAN,
            "query": GDELT_QUERY,
            "article_count": 0,
            "articles": [],
            "error": str(exc)[:300],
            "documentation": "https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/",
        }

    _news_cache.update({
        "expires": now + timedelta(minutes=NEWS_CACHE_MINUTES),
        "result": result,
        "max_articles": max_articles,
    })
    return result


@app.get("/news_intelligence")
def news_intelligence(
    days: int = Query(default=7, ge=1, le=30),
    max_articles: int = Query(default=50, ge=1, le=75),
) -> dict[str, Any]:
    """Combined event calendar + current news endpoint for the FlutterFlow UI."""
    calendar = get_showmytrades_calendar(days)
    event_risk = get_event_risk(days)
    news = get_gdelt_news(max_articles)
    calendar_ok = calendar.get("status") == "available"
    news_ok = news.get("status") == "available"
    overall_status = "success" if calendar_ok and news_ok else "partial" if calendar_ok or news_ok else "unavailable"

    return {
        "status": overall_status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "display_timezone": "Africa/Kampala",
        "calendar": {**calendar, **event_risk},
        "news": news,
        "interpretation_notes": [
            "The calendar identifies scheduled releases; the news feed supplies related headlines.",
            "A headline mentioning a currency is not automatically bullish or bearish for that currency.",
            "Compare released actual data with the pre-release consensus and assess the central-bank implications.",
            "News tone measures media language and must not be treated as a trading probability.",
        ],
    }


# ============================================================================
# SHORT-TERM FEATURE ENGINEERING
# ============================================================================


def _wilder_smooth(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's recursive smoothing, seeded with the first window SMA."""
    values = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < window:
        return pd.Series(output, index=series.index, dtype=float)

    start = int(valid[0])
    seed_end = start + window - 1
    if seed_end >= len(values) or not np.isfinite(values[start:seed_end + 1]).all():
        return pd.Series(output, index=series.index, dtype=float)

    output[seed_end] = float(np.mean(values[start:seed_end + 1]))
    for i in range(seed_end + 1, len(values)):
        if not np.isfinite(values[i]):
            output[i] = output[i - 1]
            continue
        output[i] = ((window - 1) * output[i - 1] + values[i]) / window
    return pd.Series(output, index=series.index, dtype=float)


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return _wilder_smooth(true_range, window=window)


def engineer_short_term_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = raw.copy().sort_index()
    close = df["Close"].astype(float)
    df["Clean_Close"] = close.ewm(span=10, adjust=False).mean()
    df["ret_1h"] = np.log(df["Clean_Close"]).diff()
    df["ret_24h"] = np.log(df["Clean_Close"]).diff(24)

    df["dxy_available"] = df["Close_DXY"].notna().astype(int)
    if df["Close_DXY"].notna().sum() >= 30:
        df["Clean_DXY"] = df["Close_DXY"].ffill(limit=2).ewm(span=10, adjust=False).mean()
        df["dxy_ret_1h"] = np.log(df["Clean_DXY"]).diff()
        df["dxy_corr"] = df["ret_1h"].rolling(24, min_periods=12).corr(df["dxy_ret_1h"])
    else:
        # Missing DXY is explicit: zero-valued return/correlation plus a
        # dxy_available flag. No synthetic DXY price series is manufactured.
        df["dxy_ret_1h"] = 0.0
        df["dxy_corr"] = 0.0
        df["dxy_available"] = 0

    df["sma_50"] = df["Clean_Close"].rolling(50, min_periods=50).mean()
    df["dist_sma"] = (df["Clean_Close"] - df["sma_50"]) / df["sma_50"]
    df["trend_direction"] = np.where(df["Clean_Close"] >= df["sma_50"], 1, -1)
    df["std_50"] = df["Clean_Close"].rolling(50, min_periods=50).std()
    df["z_score"] = (df["Clean_Close"] - df["sma_50"]) / (df["std_50"] + 1e-12)

    delta = df["Clean_Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    both_flat = (avg_gain <= 1e-12) & (avg_loss <= 1e-12)
    df.loc[both_flat, "rsi"] = 50.0
    df.loc[(avg_loss <= 1e-12) & ~both_flat, "rsi"] = 100.0

    df["true_range"] = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = calculate_atr(df, window=14)

    # Wilder directional movement. This is a trend-strength diagnostic, not a signal by itself.
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr_wilder = df["true_range"] if "true_range" in df else pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_wilder = calculate_atr(df, window=14)
    plus_smoothed = _wilder_smooth(plus_dm, window=14)
    minus_smoothed = _wilder_smooth(minus_dm, window=14)
    df["plus_di"] = 100.0 * plus_smoothed / (atr_wilder + 1e-12)
    df["minus_di"] = 100.0 * minus_smoothed / (atr_wilder + 1e-12)
    dx = 100.0 * (df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"] + 1e-12)
    df["adx"] = _wilder_smooth(dx, window=14)

    # H4/D1 features use only completed aggregate candles. They are forward-filled
    # to the hourly frame only after each higher-timeframe candle has closed.
    def completed_mtf_direction(
        rule: str, fast_span: int, slow_span: int, minimum_bars: int
    ) -> pd.Series:
        if not isinstance(df.index, pd.DatetimeIndex):
            return pd.Series(0, index=df.index, dtype=float)
        bars = df["Close"].resample(rule, label="right", closed="left").last().dropna()
        boundary = pd.Timestamp.now(tz="UTC").floor(rule)
        # A bar labelled with a future closing boundary is still incomplete.
        bars = bars.loc[bars.index <= boundary]
        if len(bars) < minimum_bars:
            return pd.Series(0, index=df.index, dtype=float)
        fast = bars.ewm(span=fast_span, adjust=False, min_periods=fast_span).mean()
        slow = bars.ewm(span=slow_span, adjust=False, min_periods=slow_span).mean()
        direction = pd.Series(np.where(fast >= slow, 1.0, -1.0), index=bars.index)
        return direction.reindex(df.index, method="ffill").fillna(0.0)

    df["mtf_h4"] = completed_mtf_direction("4h", 20, 50, 50)
    # Yahoo's hourly endpoint commonly provides only ~60 calendar days; a
    # 10/30 EMA comparison makes the daily filter usable without pretending
    # that 40 observations support a 50-day EMA.
    df["mtf_d1"] = completed_mtf_direction("1D", 10, 30, 30)
    rolling_range_median = df["true_range"].rolling(24, min_periods=12).median()
    df["range_expansion"] = df["true_range"] / (rolling_range_median + 1e-12)

    # Actual volume is used only if the provider supplies non-zero volume.
    volume = (
        pd.to_numeric(df["Volume"], errors="coerce")
        if "Volume" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    genuine_volume = bool(volume.notna().any() and (volume.fillna(0) > 0).any())
    if genuine_volume:
        volume_median = volume.rolling(24, min_periods=12).median()
        df["volume_spike"] = (volume > (volume_median * 1.5)).astype(float)
        df["volume_available"] = 1.0
    else:
        df["volume_spike"] = 0.0
        df["volume_available"] = 0.0

    # A 20-candle prior range is used, excluding the current candle.
    df["prev_high"] = df["High"].shift(1).rolling(20, min_periods=20).max()
    df["prev_low"] = df["Low"].shift(1).rolling(20, min_periods=20).min()
    expansion_confirm = df["range_expansion"] >= 1.5
    df["bearish_sweep"] = np.where(
        (df["High"] > df["prev_high"])
        & (df["Close"] < df["prev_high"])
        & expansion_confirm,
        -1,
        0,
    )
    df["bullish_sweep"] = np.where(
        (df["Low"] < df["prev_low"])
        & (df["Close"] > df["prev_low"])
        & expansion_confirm,
        1,
        0,
    )
    df["net_sweep"] = df["bullish_sweep"] + df["bearish_sweep"]

    df["near_swing_high"] = (
        df["Close"] >= df["prev_high"] * 0.998
    ).astype(float)
    df["near_swing_low"] = (
        df["Close"] <= df["prev_low"] * 1.002
    ).astype(float)
    df["atr_pct"] = df["atr"] / df["Close"]
    df["volatility_24h"] = df["ret_1h"].rolling(24, min_periods=12).std()

    feature_columns = [
        "ret_1h",
        "ret_24h",
        "dxy_ret_1h",
        "dxy_corr",
        "dxy_available",
        "dist_sma",
        "rsi",
        "range_expansion",
        "volume_spike",
        "volume_available",
        "net_sweep",
        "near_swing_high",
        "near_swing_low",
        "trend_direction",
        "z_score",
        "atr_pct",
        "volatility_24h",
        "adx",
        "plus_di",
        "minus_di",
        "mtf_h4",
        "mtf_d1",
    ]

    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    return df, feature_columns


# ============================================================================
# SHORT-TERM MODEL, WALK-FORWARD CALIBRATION AND ENSEMBLE
# ============================================================================


def _new_xgb_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )


def _new_rf_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=5,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def _fit_platt_calibrator(probabilities: np.ndarray, labels: np.ndarray) -> Optional[LogisticRegression]:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-5, 1.0 - 1e-5)
    y = np.asarray(labels, dtype=int)
    if len(p) < 30 or len(np.unique(y)) < 2:
        return None
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    calibrator.fit(logits, y)
    return calibrator


def _apply_platt_calibrator(
    model: Any,
    calibrator: Optional[LogisticRegression],
    x: pd.DataFrame,
) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    raw = np.clip(raw, 1e-5, 1.0 - 1e-5)
    if calibrator is None:
        return raw
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    return np.asarray(calibrator.predict_proba(logits)[:, 1], dtype=float)


def _walk_forward_oof_predictions(
    x: pd.DataFrame,
    y: pd.Series,
    purge_gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create out-of-fold probabilities using expanding, purged time splits."""
    n = len(x)
    minimum_train = max(100, 2 * purge_gap)
    remaining = n - minimum_train - purge_gap
    if remaining < 30:
        raise RuntimeError(
            "Not enough labelled hourly history for purged walk-forward calibration. "
            "Use a shorter forecast horizon or obtain more hourly candles."
        )

    chunk_size = max(30, int(math.ceil(remaining / 3.0)))
    xgb_oof: list[float] = []
    rf_oof: list[float] = []
    y_oof: list[int] = []
    cursor = minimum_train

    while cursor + purge_gap < n:
        validation_start = cursor + purge_gap
        validation_end = min(validation_start + chunk_size, n)
        if validation_end <= validation_start:
            break

        x_fold_train = x.iloc[:cursor]
        y_fold_train = y.iloc[:cursor]
        x_fold_val = x.iloc[validation_start:validation_end]
        y_fold_val = y.iloc[validation_start:validation_end]
        if len(x_fold_val) == 0 or len(np.unique(y_fold_train)) < 2:
            cursor = validation_end
            continue

        fold_xgb = _new_xgb_model()
        fold_rf = _new_rf_model()
        fold_xgb.fit(x_fold_train, y_fold_train)
        fold_rf.fit(x_fold_train, y_fold_train)
        xgb_oof.extend(fold_xgb.predict_proba(x_fold_val)[:, 1].astype(float).tolist())
        rf_oof.extend(fold_rf.predict_proba(x_fold_val)[:, 1].astype(float).tolist())
        y_oof.extend(y_fold_val.astype(int).tolist())
        cursor = validation_end

    if len(y_oof) < 30 or len(np.unique(y_oof)) < 2:
        raise RuntimeError(
            "Walk-forward calibration produced too few usable observations or only one class"
        )

    return (
        np.asarray(xgb_oof, dtype=float),
        np.asarray(rf_oof, dtype=float),
        np.asarray(y_oof, dtype=int),
        np.asarray([len(y_oof)], dtype=int),
    )


def _estimate_reversal_hours(z_values: pd.Series) -> tuple[float, bool]:
    z = pd.Series(z_values).dropna().tail(150)
    paired = pd.concat([z.rename("z"), z.shift(1).rename("lag")], axis=1).dropna()
    if len(paired) < 30:
        return 24.0, False
    slope = float(linregress(paired["lag"], paired["z"]).slope)
    if 0.0 < slope < 1.0:
        half_life = math.log(0.5) / math.log(slope)
        if math.isfinite(half_life) and half_life > 0:
            return float(half_life), True
    return 24.0, False


@app.post("/short_term_forecast")
def get_short_term_forecast(req: ShortTermRequest) -> dict[str, Any]:
    try:
        # 1. Macro state is a decision filter, not copied into every historical
        # ML row. That prevents the latest macro observation leaking backward.
        try:
            macro_state = get_macro_state()
            macro_dev = macro_state["deviation_q"]
            macro_regime = int(macro_state["current_regime"])
            macro_available = bool(macro_state["mean_reversion_supported"] and macro_dev is not None)
        except Exception as exc:
            logger.warning("Macro state unavailable for short-term filter (%s)", type(exc).__name__)
            macro_state = {}
            macro_dev = None
            macro_regime = 0
            macro_available = False

        # 2. Economic calendar. Unknown/partial coverage is not treated as LOW.
        calendar = get_event_risk(req.forecast_days)
        event_risk = str(calendar["event_risk"])

        # 3. Market data.
        raw, eur_source, dxy_available, dxy_source = fetch_hourly_market_data()
        data_as_of = raw.index[-1].isoformat()

        # 4. Causal feature engineering and target creation.
        feature_frame, features = engineer_short_term_features(raw)
        periods_ahead = int(req.forecast_days * 24)
        future_close = feature_frame["Close"].shift(-periods_ahead)
        target = pd.Series(np.nan, index=feature_frame.index, dtype=float)
        known_future = future_close.notna()
        target.loc[known_future] = (
            future_close.loc[known_future] > feature_frame.loc[known_future, "Close"]
        ).astype(int)
        feature_frame["target"] = target

        # Live row is retained independently of target availability.
        x_live_frame = feature_frame[features].replace([np.inf, -np.inf], np.nan).dropna()
        if x_live_frame.empty:
            raise RuntimeError("Latest candle does not have a complete feature vector")
        x_live = x_live_frame.iloc[[-1]]
        live_index = x_live.index[-1]

        labelled = feature_frame[features + ["target"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna(subset=features + ["target"])
        labelled["target"] = labelled["target"].astype(int)
        if len(labelled) < max(250, periods_ahead * 3):
            raise RuntimeError(
                f"Only {len(labelled)} labelled candles are available. More hourly history "
                "is required for the requested forecast horizon."
            )

        x_all = labelled[features].astype(float)
        y_all = labelled["target"].astype(int)
        if len(np.unique(y_all)) < 2:
            raise RuntimeError("Training labels contain only one class; forecast is unavailable")

        # 5. Purged chronological holdout. The training labels must mature before
        # the validation period begins; purge at least the forecast horizon.
        split_idx = int(len(labelled) * 0.80)
        train_end = split_idx - periods_ahead
        if train_end < max(150, periods_ahead * 2):
            raise RuntimeError(
                "Not enough pre-validation training data after purging overlapping "
                "forecast labels. Reduce forecast_days or increase market history."
            )
        x_train_pool = x_all.iloc[:train_end]
        y_train_pool = y_all.iloc[:train_end]
        x_validation = x_all.iloc[split_idx:]
        y_validation = y_all.iloc[split_idx:]
        if len(x_validation) < 25 or len(np.unique(y_train_pool)) < 2:
            raise RuntimeError("Insufficient chronological validation data")

        # 6. Fit sigmoid calibrators from purged expanding-window OOF predictions.
        xgb_oof, rf_oof, y_oof, _ = _walk_forward_oof_predictions(
            x_train_pool, y_train_pool, purge_gap=periods_ahead
        )
        xgb_calibrator = _fit_platt_calibrator(xgb_oof, y_oof)
        rf_calibrator = _fit_platt_calibrator(rf_oof, y_oof)

        # Validate the model on a later untouched time block.
        validation_xgb = _new_xgb_model()
        validation_rf = _new_rf_model()
        validation_xgb.fit(x_train_pool, y_train_pool)
        validation_rf.fit(x_train_pool, y_train_pool)
        val_xgb_prob = _apply_platt_calibrator(validation_xgb, xgb_calibrator, x_validation)
        val_rf_prob = _apply_platt_calibrator(validation_rf, rf_calibrator, x_validation)
        val_ensemble = 0.5 * val_xgb_prob + 0.5 * val_rf_prob
        xgb_brier = float(brier_score_loss(y_validation, val_xgb_prob))
        rf_brier = float(brier_score_loss(y_validation, val_rf_prob))
        validation_brier = float(brier_score_loss(y_validation, val_ensemble))
        baseline_probability = float(np.mean(y_train_pool))
        baseline_predictions = np.full(len(y_validation), baseline_probability, dtype=float)
        baseline_brier = float(brier_score_loss(y_validation, baseline_predictions))
        brier_skill_score = (float(1.0 - validation_brier / baseline_brier) if baseline_brier > 0 else None)
        validation_accuracy = float(np.mean((val_ensemble >= 0.5).astype(int) == y_validation.to_numpy()))
        baseline_accuracy = float(np.mean((baseline_predictions >= 0.5).astype(int) == y_validation.to_numpy()))
        validation_gate_pass = bool(
            len(y_validation) >= 50 and brier_skill_score is not None and brier_skill_score > 0.0
        )

        # 7. Refit base learners on every fully labelled observation. The most
        # recent live row has no target and is never included in this fit.
        xgb_model = _new_xgb_model()
        rf_model = _new_rf_model()
        xgb_model.fit(x_all, y_all)
        rf_model.fit(x_all, y_all)
        live_xgb_prob = float(
            _apply_platt_calibrator(xgb_model, xgb_calibrator, x_live)[0]
        )
        live_rf_prob = float(
            _apply_platt_calibrator(rf_model, rf_calibrator, x_live)[0]
        )
        # Equal weights avoid optimising ensemble weights on a small validation set.
        xgb_weight = 0.5
        rf_weight = 0.5
        p_up = float(np.clip(xgb_weight * live_xgb_prob + rf_weight * live_rf_prob, 0.01, 0.99))

        # 8. Current market state and technical context.
        current_row = feature_frame.loc[live_index]
        current_price = float(current_row["Close"])
        current_atr = _finite_float(current_row["atr"], None)
        current_rsi = _finite_float(current_row["rsi"], 50.0)
        current_z = _finite_float(current_row["z_score"], 0.0)
        current_trend = int(current_row["trend_direction"])
        net_sweep = int(current_row["net_sweep"])
        dxy_corr = _finite_float(current_row["dxy_corr"], 0.0) or 0.0
        range_expansion = _finite_float(current_row["range_expansion"], 0.0) or 0.0
        reversal_hours, reversal_supported = _estimate_reversal_hours(feature_frame["z_score"])

        if current_atr is None or current_atr <= 0:
            raise RuntimeError("ATR is unavailable for the latest candle")

        # The probability is NOT manually shifted by RSI. RSI is already a model
        # feature; adding another hand-set probability boost would double-count it.
        expected_move = current_atr * math.sqrt(float(periods_ahead))
        signal = "HOLD"
        macro_bias = "Unavailable"
        if macro_available and macro_dev is not None:
            if macro_dev > 0.005:
                macro_bias = "Bearish"
            elif macro_dev < -0.005:
                macro_bias = "Bullish"
            else:
                macro_bias = "Neutral"

        # Event calendar must be confirmed LOW. Missing/partial calendar coverage
        # deliberately produces HOLD rather than silently permitting a trade.
        event_filter_allows_signal = (
            event_risk == "LOW"
            or (event_risk == "UNKNOWN" and not REQUIRE_VERIFIED_EVENT_CALENDAR)
        )
        if event_filter_allows_signal and macro_available and macro_dev is not None:
            if p_up > 0.60 and macro_dev <= MACRO_NEUTRAL_BAND and expected_move > SPREAD_PRICE:
                signal = "BUY"
                if net_sweep == -1:
                    signal = "HOLD"
            elif p_up < 0.40 and macro_dev >= -MACRO_NEUTRAL_BAND and expected_move > SPREAD_PRICE:
                signal = "SELL"
                if net_sweep == 1:
                    signal = "HOLD"

        # Trend and extreme-distance filters.
        if signal == "SELL" and current_trend == 1:
            signal = "HOLD"
        if signal == "BUY" and current_trend == -1:
            signal = "HOLD"
        if signal == "SELL" and current_z < -2.0:
            signal = "HOLD"
        if signal == "BUY" and current_z > 2.0:
            signal = "HOLD"

        raw_signal = signal
        if not validation_gate_pass:
            signal = "HOLD"
        # 9. ATR-based reference levels. They are not a guarantee of execution.
        if signal == "BUY":
            target_price = current_price + (3.0 * current_atr)
            stop_loss = current_price - (1.5 * current_atr)
        elif signal == "SELL":
            target_price = current_price - (3.0 * current_atr)
            stop_loss = current_price + (1.5 * current_atr)
        else:
            # Preserve the original numeric response fields for the FlutterFlow UI.
            target_price = current_price + (1.5 * current_atr)
            stop_loss = current_price - (1.5 * current_atr)

        # Position sizing is disabled because close-direction probability does not
        # estimate target-before-stop probability. Keep the legacy field at 0.0.
        kelly_fraction = 0.0
        recommended_position_size_pct = 0.0

        history = [
            {
                "date": d.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "price": float(row["Close"]),
            }
            for d, row in feature_frame.tail(48).iterrows()
        ]

        response_data = {
                "current_price": current_price,
                "forecast_days": req.forecast_days,
                "probability_up": p_up,
                "probability_down": float(1.0 - p_up),
                "signal": signal,
                "target_price": float(target_price),
                "stop_loss": float(stop_loss),
                "last_rsi": float(current_rsi),
                "macro_bias": macro_bias,
                # Legacy field retained for UI compatibility. This is a price/range
                # rejection heuristic, not evidence of institutional/"whale" orders.
                "whale_trap_detected": bool(net_sweep != 0),
                "liquidity_sweep_detected": bool(net_sweep != 0),
                "event_risk": event_risk,
                "dxy_correlation": float(dxy_corr),
                "trend_direction": "Up" if current_trend == 1 else "Down",
                "reversal_expectation_hours": float(reversal_hours),
                "recommended_position_size_pct": recommended_position_size_pct,
                "model_weights": {"xgb": xgb_weight, "rf": rf_weight},
                "history": history,
                # Additional diagnostics; existing top-level fields remain intact.
                "model_validation": {
                    "xgb_brier_score": xgb_brier,
                    "rf_brier_score": rf_brier,
                    "ensemble_brier_score": validation_brier,
                    "validation_observations": int(len(y_validation)),
                    "calibration_method": "purged expanding-window Platt scaling",
                    "calibration_observations": int(len(y_oof)),
                    "calibration_status": (
                        "sigmoid calibrated"
                        if xgb_calibrator is not None and rf_calibrator is not None
                        else "calibration unavailable for one or both models; raw probabilities used there"
                    ),
                },
                "risk_diagnostics": {
                    "position_sizing_status": "DISABLED_PENDING_TARGET_BEFORE_STOP_MODEL",
                    "event_filter_bypassed": bool(
                        event_risk == "UNKNOWN" and not REQUIRE_VERIFIED_EVENT_CALENDAR
                    ),
                    "verified_event_calendar_required": REQUIRE_VERIFIED_EVENT_CALENDAR,
                    "position_sizing_note": (
                        "A forecast that the future close is higher/lower is not the probability "
                        "of reaching take-profit before stop-loss. No account allocation is recommended."
                    ),
                    "spread_price_assumption": SPREAD_PRICE,
                    "spread_source": "EURUSD_SPREAD_PRICE environment variable or default assumption",
                    "atr": float(current_atr),
                    "range_expansion_ratio": float(range_expansion),
                    "volume_data_available": bool(float(current_row["volume_available"]) > 0),
                    "reversal_estimate_supported": reversal_supported,
                    "forecast_target_definition": (
                        f"Close direction after {periods_ahead} hourly candles; weekends and market closures "
                        "mean this is not exactly the same as calendar days."
                    ),
                    "probability_note": (
                        "Probability is model-estimated P(future close > current close), not a guaranteed "
                        "trade win probability or target-before-stop probability."
                    ),
                },
                "data_diagnostics": {
                    "eurusd_source": eur_source,
                    "dxy_source": dxy_source,
                    "dxy_available": dxy_available,
                    "data_as_of": data_as_of,
                    "completed_hourly_candles": int(len(feature_frame)),
                    "labelled_candles": int(len(labelled)),
                    "macro_model_available": macro_available,
                    "macro_regime": macro_regime,
                    "macro_model_type": macro_state.get("model_type"),
                },
                "event_diagnostics": calendar,
        }
        mtf_h4 = int(current_row.get("mtf_h4", 0))
        mtf_d1 = int(current_row.get("mtf_d1", 0))
        adx_now = _finite_float(current_row.get("adx"), None)
        adx_prev = _finite_float(feature_frame["adx"].iloc[-2], None) if len(feature_frame) > 1 else None
        plus_di = _finite_float(current_row.get("plus_di"), None)
        minus_di = _finite_float(current_row.get("minus_di"), None)
        if mtf_h4 == 0 or mtf_d1 == 0:
            mtf_label = "Unknown"
        elif mtf_h4 == mtf_d1 == 1:
            mtf_label = "Bullish"
        elif mtf_h4 == mtf_d1 == -1:
            mtf_label = "Bearish"
        else:
            mtf_label = "Mixed"

        # Keep legacy field names used by earlier FlutterFlow bindings.
        response_data.update({
            "raw_signal": raw_signal,
            "xgb_probability": float(live_xgb_prob),
            "rf_probability": float(live_rf_prob),
            "model_disagreement": float(abs(live_xgb_prob - live_rf_prob)),
            "target_up_rate": baseline_probability,
            "validation_brier_score": validation_brier,
            "validation_accuracy": validation_accuracy,
            "baseline_brier_score": baseline_brier,
            "baseline_accuracy": baseline_accuracy,
            "brier_skill_score": brier_skill_score,
            "validation_sample_count": int(len(y_validation)),
            "validation_status": "PRELIMINARY_POSITIVE_BSS" if validation_gate_pass else "FAILED_OR_INSUFFICIENT_VALIDATION",
            "validation_method": "purged_expanding_window_calibration_plus_chronological_holdout",
            "independent_walk_forward_validated": False,
            "walk_forward_calibration_used": True,
            "horizon_conflict": bool(mtf_h4 != 0 and mtf_d1 != 0 and mtf_h4 != mtf_d1),
            "mtf_trend": mtf_label,
            "data_discrepancy": False,
            "data_discrepancy_check_available": False,
            "reference_price": None,
            "reference_discrepancy_pct": None,
            "reference_price_warning": "No independent live reference-price comparison is enabled in v5.3.",
            "eurusd_data_source": eur_source,
            "market_regime": "High volatility" if int(macro_regime) == 1 else "Low volatility" if macro_available else "Unknown",
            "adx_value": adx_now,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "adx_rising": bool(adx_now is not None and adx_prev is not None and adx_now > adx_prev),
            "calendar_available": event_risk != "UNKNOWN",
            "macro_model": macro_state.get("model_type"),
            "macro_deviation_q": macro_dev,
            "macro_neutral_band": MACRO_NEUTRAL_BAND,
            "dxy_available": bool(dxy_available),
            "estimated_zscore_half_life_hours": float(reversal_hours) if reversal_supported else None,
            "probabilities_calibrated": bool(xgb_calibrator is not None and rf_calibrator is not None),
            "trading_profitability_validated": False,
            "forecast_horizon_type": "hourly_candles",
            "forecast_horizon_candles": int(periods_ahead),
            "forecast_horizon_requested_days": int(req.forecast_days),
            "atr_value": float(current_atr),
            "atr_method": "14-period Wilder smoothing with an initial SMA seed",
            "spread_assumption": SPREAD_PRICE,
            "data_as_of": data_as_of,
        })
        response_data["model_validation"].update({
            "brier_skill_score": brier_skill_score,
            "baseline_probability": baseline_probability,
            "baseline_brier_score": baseline_brier,
            "validation_accuracy": validation_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "validation_gate_pass": validation_gate_pass,
            "interpretation": "Positive holdout Brier skill is a basic gate only; it does not establish profitability or durable predictive edge.",
        })

        journal_ok, journal_source = _persist_forecast_journal(response_data)
        response_data["journal_status"] = {"saved": journal_ok, "source": journal_source}
        alert_status = _queue_and_deliver_alert(response_data)
        response_data["alert_status"] = alert_status

        return {"status": "success", "data": _safe_json(response_data)}

    except HTTPException:
        raise
    except Exception as exc:
        _log_exception("/short_term_forecast failed", exc)
        raise HTTPException(status_code=503, detail="Short-term forecast failed; check market-data configuration and server logs.") from exc
