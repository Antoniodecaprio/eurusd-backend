from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from fredapi import Fred
from scipy.stats import norm, linregress
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, accuracy_score
import yfinance as yf
import xgboost as xgb
import os
import requests
import warnings
import datetime
import logging
import csv
import io
import json
from statsmodels.tsa.stattools import adfuller
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

app = FastAPI(title="EUR/USD Quant Forecaster", version="4.3.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRED_API_KEY = os.getenv("FRED_API_KEY")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ENABLE_TELEGRAM_ALERTS = os.getenv("ENABLE_TELEGRAM_ALERTS", "false").strip().lower() == "true"

JOURNAL_LEGACY_FIELDS = [
    "timestamp", "spot", "prob_up", "signal", "target", "stop",
    "mtf_trend", "regime", "adx"
]

JOURNAL_FILE = "forecast_journal.csv"
STATE_FILE = "last_signal_state.json"

class ShortTermRequest(BaseModel):
    forecast_days: int = Field(default=3, ge=1, le=7)

class ForecastRequest(BaseModel):
    horizon_years: float = Field(default=3.0, gt=0.0, le=10.0)
    pi_us: float = Field(default=0.02, ge=-0.05, le=0.25)
    pi_ez: float = Field(default=0.02, ge=-0.05, le=0.25)

# ==========================================
# HELPER FUNCTIONS & ADAPTERS
# ==========================================

def adapt_for_flutter(data: dict) -> dict:
    """Converts None to 0.0 for FlutterFlow's strict typing."""
    safe_data = data.copy()
    numeric_keys = [
        'current_price', 'target_price', 'stop_loss', 'probability_up', 'xgb_probability',
        'rf_probability', 'model_disagreement', 'target_up_rate', 'validation_brier_score',
        'validation_accuracy', 'baseline_brier_score', 'baseline_accuracy', 'brier_skill_score',
        'adx_value', 'plus_di', 'minus_di', 'last_rsi', 'dxy_correlation',
        'estimated_zscore_half_life_hours', 'atr_value', 'macro_deviation_q',
        'reference_price', 'reference_discrepancy_pct'
    ]

    for k in numeric_keys:
        if k in safe_data:
            val = safe_data[k]
            if val is None or not np.isfinite(val):
                safe_data[k] = 0.0
                safe_data[f"{k}_available"] = False
            else:
                safe_data[f"{k}_available"] = True

    if safe_data.get('trend_direction') not in ['Up', 'Down']:
        safe_data['trend_direction'] = 'Unknown'

    return safe_data

def get_last_signal():
    """Reads the last known signal to prevent notification spam."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f).get("last_signal", "HOLD")
        except Exception:
            return "HOLD"
    return "HOLD"

def update_last_signal(signal):
    """Saves the current signal."""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump({"last_signal": signal}, f)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")

def send_telegram_alert(message: str):
    """Sends a message to Telegram."""
    if not ENABLE_TELEGRAM_ALERTS or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=3)
    except Exception as e:
        logger.error(f"Telegram alert failed: {e}")

@lru_cache(maxsize=1)
def fetch_macro_data_cached(cache_buster: str):
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY not set")
    fred = Fred(api_key=FRED_API_KEY)
    def to_month_end(series):
        series = pd.Series(series).dropna()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        return series.resample('ME').last()

    spot = to_month_end(fred.get_series('DEXUSEU'))
    us_cpi = to_month_end(fred.get_series('CPIAUCSL')).shift(1)
    ez_cpi = to_month_end(fred.get_series('CP0000EZ19M086NEST')).shift(1)

    df = pd.DataFrame({'spot': spot, 'us_cpi': us_cpi, 'ez_cpi': ez_cpi}).dropna()
    df = df[df.index >= '1999-01-01']
    df['s'] = np.log(df['spot'])
    df['p_us'] = np.log(df['us_cpi'])
    df['p_ez'] = np.log(df['ez_cpi'])
    df['q'] = df['s'] + df['p_us'] - df['p_ez']
    return df

def get_macro_state(df):
    df['q_lag'] = df['q'].shift(1)
    df_ols = df.dropna().copy()
    slope, intercept, _, _, _ = linregress(df_ols['q_lag'], df_ols['q'])

    try:
        adf_p = adfuller(df['q'], regression="c", autolag="AIC")[1]
    except:
        adf_p = 1.0

    df_ols['residuals'] = df_ols['q'] - (intercept + slope * df_ols['q_lag'])
    gmm = GaussianMixture(n_components=2, random_state=42)
    valid_resids = df_ols['residuals'].dropna()
    gmm.fit(valid_resids.values.reshape(-1, 1))
    regimes = gmm.predict(valid_resids.values.reshape(-1, 1))
    current_regime = int(regimes[-1])

    if 0.0 < slope < 1.0 and adf_p < 0.05:
        phi = slope
        q_bar = intercept / (1 - phi)
        deviation = df['q'].iloc[-1] - q_bar
        model_type = "AR(1)_mean_reverting"
        df_ols['regime'] = regimes
        current_resids = df_ols[df_ols['regime'] == current_regime]['residuals']
        sigma_eps = current_resids.std() if len(current_resids) > 10 else df_ols['residuals'].std()
    else:
        phi = 1.0
        q_bar = None
        deviation = 0.0
        model_type = "random_walk_fallback"
        sigma_eps = df['q'].diff().std() * np.sqrt(12)
    return deviation, current_regime, phi, q_bar, model_type, sigma_eps

def calculate_wilder_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = np.full(len(series), np.nan)
    avg_loss = np.full(len(series), np.nan)

    if len(series) > period:
        avg_gain[period] = gain.iloc[1:period+1].mean()
        avg_loss[period] = loss.iloc[1:period+1].mean()
        for i in range(period + 1, len(series)):
            avg_gain[i] = ((avg_gain[i-1] * (period - 1)) + gain.iloc[i]) / period
            avg_loss[i] = ((avg_loss[i-1] * (period - 1)) + loss.iloc[i]) / period

    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where((avg_loss == 0) & (avg_gain > 0), 100, rsi)
    rsi = np.where((avg_gain == 0) & (avg_loss > 0), 0, rsi)
    rsi = np.where((avg_gain == 0) & (avg_loss == 0), 50, rsi)
    return pd.Series(rsi, index=series.index)

def calculate_true_wilder_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))

    atr = np.full(len(tr), np.nan)
    if len(tr) > window:
        atr[window] = tr.iloc[1:window+1].mean()
        for i in range(window + 1, len(tr)):
            atr[i] = ((atr[i-1] * (window - 1)) + tr.iloc[i]) / window
    return pd.Series(atr, index=df.index)

def calculate_true_wilder_adx(df, window=14):
    up_move = df['High'] - df['High'].shift(1)
    down_move = df['Low'].shift(1) - df['Low']

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.maximum(df['High'] - df['Low'],
                    np.maximum(np.abs(df['High'] - df['Close'].shift(1)),
                               np.abs(df['Low'] - df['Close'].shift(1))))

    tr_s = np.full(len(tr), np.nan)
    pdm_s = np.full(len(tr), np.nan)
    mdm_s = np.full(len(tr), np.nan)

    if len(tr) > window:
        tr_s[window] = tr.iloc[1:window+1].mean()
        pdm_s[window] = pd.Series(plus_dm).iloc[1:window+1].mean()
        mdm_s[window] = pd.Series(minus_dm).iloc[1:window+1].mean()

        for i in range(window+1, len(tr)):
            tr_s[i] = ((tr_s[i-1] * (window-1)) + tr.iloc[i]) / window
            pdm_s[i] = ((pdm_s[i-1] * (window-1)) + plus_dm[i]) / window
            mdm_s[i] = ((mdm_s[i-1] * (window-1)) + minus_dm[i]) / window

    pdi = 100 * (pd.Series(pdm_s, index=df.index) / (pd.Series(tr_s, index=df.index) + 1e-9))
    mdi = 100 * (pd.Series(mdm_s, index=df.index) / (pd.Series(tr_s, index=df.index) + 1e-9))

    dx = 100 * (np.abs(pdi - mdi) / (pdi + mdi + 1e-9))

    adx = np.full(len(dx), np.nan)
    start_idx = 2 * window - 1
    if len(dx) > start_idx:
        adx[start_idx] = dx.iloc[window:start_idx+1].mean()
        for i in range(start_idx+1, len(dx)):
            adx[i] = ((adx[i-1] * (window-1)) + dx.iloc[i]) / window

    adx_s = pd.Series(adx, index=df.index)
    return adx_s, pdi, mdi, adx_s > adx_s.shift(1)

def log_forecast_to_csv(record):
    file_exists = os.path.isfile(JOURNAL_FILE)
    header_ok = True
    if file_exists:
        try:
            with open(JOURNAL_FILE, 'r') as f:
                first_line = f.readline().strip()
                existing_cols = first_line.split(',')
                if existing_cols != list(record.keys()):
                    header_ok = False
        except Exception:
            header_ok = False

    if not header_ok:
        try:
            os.rename(JOURNAL_FILE, f"{JOURNAL_FILE}.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
            file_exists = False
        except Exception as e:
            logger.error(f"Failed to rotate journal: {e}")

    try:
        with open(JOURNAL_FILE, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
    except Exception as e:
        logger.error(f"Failed to write to journal: {e}")

def _supabase_is_configured():
    return bool(SUPABASE_URL and SUPABASE_KEY)

def _supabase_headers(prefer=None):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers

def _legacy_row_from_supabase(row):
    spot = row.get("spot_price")
    if spot is None:
        spot = row.get("current_price")

    probability = row.get("prob_up")
    if probability is None:
        probability = row.get("probability_up")

    legacy = {
        "timestamp": row.get("timestamp") or row.get("created_at") or "",
        "spot": spot,
        "prob_up": probability,
        "signal": row.get("signal"),
        "target": row.get("target_price"),
        "stop": row.get("stop_loss"),
        "mtf_trend": row.get("mtf_trend"),
        "regime": row.get("regime"),
        "adx": row.get("adx"),
    }
    return {
        key: "" if value is None else str(value)
        for key, value in legacy.items()
    }

def _read_local_journal():
    if not os.path.isfile(JOURNAL_FILE):
        return []
    try:
        with open(JOURNAL_FILE, mode="r", newline="") as file:
            reader = csv.DictReader(file)
            return [
                {key: (row.get(key) or "") for key in JOURNAL_LEGACY_FIELDS}
                for row in reader
            ]
    except Exception:
        logger.exception("Failed to read local CSV journal")
        return []

def _fetch_supabase_journal(limit=20, offset=0, order="timestamp.desc"):
    url = f"{SUPABASE_URL}/rest/v1/forecast_journal"
    params = {
        "select": "timestamp,created_at,spot_price,current_price,prob_up,probability_up,signal,target_price,stop_loss,mtf_trend,regime,adx",
        "order": order,
        "limit": str(limit),
        "offset": str(offset),
    }
    response = requests.get(
        url,
        headers=_supabase_headers(),
        params=params,
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected Supabase journal response")
    return payload

def save_journal_record(record):
    """Write to Supabase; preserve a local CSV fallback if that fails."""
    if _supabase_is_configured():
        try:
            spot = record.get("spot")
            probability = record.get("prob_up")
            database_record = {
                "timestamp": record.get("timestamp"),
                "spot_price": spot,
                "prob_up": probability,
                "signal": record.get("signal"),
                "target_price": record.get("target"),
                "stop_loss": record.get("stop"),
                "mtf_trend": record.get("mtf_trend"),
                "regime": record.get("regime"),
                "adx": record.get("adx"),
                "current_price": spot,
                "forecast_days": record.get("forecast_days"),
                "probability_up": probability,
                "symbol": "EUR/USD",
                "payload": {"legacy_record": record},
            }
            url = f"{SUPABASE_URL}/rest/v1/forecast_journal"
            response = requests.post(
                url,
                headers=_supabase_headers("return=minimal"),
                json=database_record,
                timeout=10,
            )
            response.raise_for_status()
            logger.info("Forecast journal record saved to Supabase")
            return
        except Exception:
            logger.exception("Supabase journal write failed; using local CSV fallback")

    log_forecast_to_csv(record)

def get_journal_rows(limit=None):
    """Return legacy-shaped journal rows, oldest first."""
    if _supabase_is_configured():
        try:
            if limit is not None:
                db_rows = _fetch_supabase_journal(
                    limit=limit, offset=0, order="timestamp.desc"
                )
                return [_legacy_row_from_supabase(row) for row in reversed(db_rows)]

            all_rows = []
            page_size = 1000
            max_rows = 50000
            while len(all_rows) < max_rows:
                db_rows = _fetch_supabase_journal(
                    limit=min(page_size, max_rows - len(all_rows)),
                    offset=len(all_rows),
                    order="timestamp.asc",
                )
                all_rows.extend(_legacy_row_from_supabase(row) for row in db_rows)
                if len(db_rows) < page_size:
                    break
            if len(all_rows) >= max_rows:
                logger.warning("Journal export reached the 50,000-row safety limit")
            return all_rows
        except Exception:
            logger.exception("Supabase journal read failed; trying local CSV fallback")

    rows = _read_local_journal()
    return rows[-limit:] if limit is not None else rows

def build_journal_csv_response(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=JOURNAL_LEGACY_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in JOURNAL_LEGACY_FIELDS})
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="forecast_journal_backup.csv"'},
    )

# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/")
def read_root():
    return {"status": "API is running - Quant v4.3.2"}

@app.post("/forecast")
def get_forecast(req: ForecastRequest):
    try:
        cache_key = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        df = fetch_macro_data_cached(cache_key).copy()
        current_spot = float(df['spot'].iloc[-1])
        deviation, _, phi, q_bar, model_type, sigma_eps = get_macro_state(df)

        if model_type == "AR(1)_mean_reverting":
            current_q = np.log(current_spot) + df['p_us'].iloc[-1] - df['p_ez'].iloc[-1]
            half_life = np.log(0.5) / np.log(phi) / 12
            phi_n = phi ** (req.horizon_years * 12)
            drift_q = (phi_n - 1) * (current_q - q_bar)
            ppp_anchor = float(np.exp(q_bar - df['p_us'].iloc[-1] + df['p_ez'].iloc[-1]))
            var_q = (sigma_eps ** 2) * (1 - phi ** (2 * req.horizon_years * 12)) / (1 - phi ** 2)
        else:
            half_life = None
            drift_q = df['q'].diff().mean() * (req.horizon_years * 12)
            ppp_anchor = None
            var_q = (sigma_eps ** 2) * req.horizon_years

        drift_infl = (req.pi_ez - req.pi_us) * req.horizon_years
        mean_delta = drift_q + drift_infl
        std_delta = np.sqrt(var_q)
        p_up = norm.cdf(mean_delta / std_delta) if std_delta > 0 else 0.5
        median_rate = float(np.exp(np.log(current_spot) + mean_delta))
        expected_rate = float(np.exp(np.log(current_spot) + mean_delta + (var_q / 2)))
        historical_data = [{"date": d.strftime("%Y-%m-%d"), "price": float(r['spot'])} for d, r in df.tail(60).iterrows()]

        macro_bias = "Neutral"
        if model_type == "AR(1)_mean_reverting":
            macro_bias = "Bearish" if deviation > 0 else "Bullish"

        raw_response = {
            "current_state": {
                "spot_rate": current_spot,
                "ppp_equilibrium_estimate": ppp_anchor,
                "deviation_q": float(deviation),
                "half_life_years": half_life,
                "macro_model": model_type,
                "macro_bias": macro_bias,
                "data_source_warning": "FRED API used."
            },
            "forecast": {
                "p_up": float(p_up),
                "expected_rate": expected_rate,
                "median_rate": median_rate,
                "ci_90_low": float(np.exp(np.log(current_spot) + mean_delta - 1.645 * std_delta)),
                "ci_90_high": float(np.exp(np.log(current_spot) + mean_delta + 1.645 * std_delta))
            },
            "historical_data": historical_data
        }

        return {"status": "success", "data": adapt_for_flutter(raw_response)}
    except Exception as e:
        logger.exception("Macro Forecast Error")
        raise HTTPException(status_code=500, detail={"status": "error", "message": "Internal calculation error"})

@app.post("/short_term_forecast")
def get_short_term_forecast(req: ShortTermRequest):
    try:
        # 1. Macro
        macro_available = False
        macro_bias = "Unavailable"
        macro_dev = 0.0
        try:
            cache_key = datetime.datetime.now().strftime("%Y-%m-%d-%H")
            df_macro = fetch_macro_data_cached(cache_key).copy()
            macro_dev_temp, _, _, _, macro_model, _ = get_macro_state(df_macro)
            if macro_model == "AR(1)_mean_reverting":
                macro_available = True
                macro_dev = macro_dev_temp
                macro_bias = "Bearish" if macro_dev > 0 else "Bullish"
        except Exception:
            pass

        # 2. Economic Calendar — fail-closed until coverage is proven
        event_risk = "UNKNOWN"
        calendar_available = False

        calendar_diagnostics = {
            "response_received": False,
            "events_returned": 0,
            "meta_present": False,
            "meta_covers_horizon": False,
            "span_check_passed": False,
            "density_check_passed": False,
            "coverage_evidence": "none",
            "reason": "not_attempted",
        }

        if TWELVE_DATA_KEY:
            try:
                # Use requests params for safe URL encoding
                cal_url = "https://api.twelvedata.com/economic_calendar"
                cal_params = {
                    "country": "United States,Euro Area",
                    "apikey": TWELVE_DATA_KEY
                }
                cal_response = requests.get(cal_url, params=cal_params, timeout=5)
                
                try:
                    cal_res = cal_response.json()
                except Exception:
                    cal_res = {"raw_text": cal_response.text}

                # --- Force diagnostic capture BEFORE raising status ---
                SAMPLE_FILE = "calendar_response_sample.json"
                try:
                    with open(SAMPLE_FILE, "w") as f:
                        json.dump(
                            {
                                "status_code": cal_response.status_code,
                                "request_url_redacted": cal_response.url.replace(TWELVE_DATA_KEY, "***"),
                                "response_body": cal_res,
                            },
                            f,
                            indent=2,
                            default=str,
                        )
                except Exception as e:
                    logger.error(f"Failed to write diagnostic file: {e}")

                cal_response.raise_for_status()
                calendar_diagnostics["response_received"] = True

                now = datetime.datetime.now(datetime.timezone.utc)
                horizon_end = now + datetime.timedelta(days=req.forecast_days)

                events = cal_res.get("data", [])
                meta = (
                    cal_res.get("meta")
                    or cal_res.get("pagination")
                    or cal_res.get("range")
                    or {}
                )
                calendar_diagnostics["meta_present"] = bool(meta)
                calendar_diagnostics["events_returned"] = (
                    len(events) if isinstance(events, list) else 0
                )

                # --- Coverage proof A: explicit meta range ---
                meta_covers = False
                meta_end_raw = meta.get("end_date") or meta.get("end")
                if meta_end_raw:
                    try:
                        meta_end_ts = pd.to_datetime(
                            meta_end_raw, utc=True, errors="coerce"
                        )
                        if not pd.isna(meta_end_ts):
                            meta_covers = (
                                meta_end_ts.to_pydatetime() >= horizon_end
                            )
                    except Exception:
                        pass
                calendar_diagnostics["meta_covers_horizon"] = meta_covers

                # --- Parse events ---
                parsed = []
                if isinstance(events, list) and events:
                    for event in events:
                        raw_time = event.get("date")
                        if not raw_time:
                            continue
                        try:
                            t = pd.to_datetime(
                                raw_time, utc=True, errors="coerce"
                            )
                            if pd.isna(t):
                                continue
                            parsed.append(
                                {
                                    "time": t.to_pydatetime(),
                                    "impact": str(
                                        event.get("impact", "")
                                    ).strip().lower(),
                                }
                            )
                        except Exception:
                            continue

                # --- Coverage proof B: structural span + density ---
                span_covers = False
                density_ok = False
                if parsed:
                    has_past = any(e["time"] <= now for e in parsed)
                    has_future = any(e["time"] >= horizon_end for e in parsed)
                    span_covers = has_past and has_future

                    unique_days = {e["time"].date() for e in parsed}
                    density_ok = len(unique_days) >= 2 or len(parsed) >= 3

                calendar_diagnostics["span_check_passed"] = span_covers
                calendar_diagnostics["density_check_passed"] = density_ok

                # --- Decide ---
                if meta_covers:
                    calendar_diagnostics["coverage_evidence"] = "meta_range"
                elif span_covers and density_ok:
                    calendar_diagnostics["coverage_evidence"] = "structural_span"

                if calendar_diagnostics["coverage_evidence"] != "none":
                    calendar_available = True

                    high_impact_found = any(
                        e["impact"] == "high"
                        and now <= e["time"] <= horizon_end
                        for e in parsed
                    )
                    event_risk = "HIGH" if high_impact_found else "LOW"

                    calendar_diagnostics["reason"] = (
                        f"coverage proven via "
                        f"{calendar_diagnostics['coverage_evidence']}"
                    )
                else:
                    calendar_diagnostics["reason"] = (
                        "no positive coverage evidence — "
                        "defaulting to UNKNOWN (fail-closed)"
                    )

                if not calendar_available:
                    logger.warning(
                        "Calendar coverage unproven: %s",
                        calendar_diagnostics,
                    )

            except Exception:
                calendar_diagnostics["reason"] = "request_or_parse_failure"
                logger.exception(
                    "Economic calendar request failed — event risk UNKNOWN"
                )

        # 3. Data Fetch
        df_eur = pd.DataFrame()
        df_dxy = pd.DataFrame()

        if TWELVE_DATA_KEY:
            try:
                url = f"https://api.twelvedata.com/time_series?symbol=EUR/USD&interval=1h&outputsize=1000&apikey={TWELVE_DATA_KEY}"
                response = requests.get(url, timeout=5).json()
                if 'values' in response:
                    df_eur = pd.DataFrame(response['values'])
                    df_eur['datetime'] = pd.to_datetime(df_eur['datetime'], utc=True)
                    df_eur = df_eur.set_index('datetime').sort_index()
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df_eur.columns:
                            df_eur[col] = pd.to_numeric(df_eur[col])
                    df_eur = df_eur.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
            except Exception:
                pass

        try:
            df_dxy = yf.Ticker("DX-Y.NYB").history(period="90d", interval="1h", timeout=5)
            if not df_dxy.empty:
                df_dxy.index = df_dxy.index.tz_convert('UTC') if df_dxy.index.tz else df_dxy.index.tz_localize('UTC')
        except Exception:
            pass

        if df_eur.empty or len(df_eur) < 200:
            try:
                df_eur = yf.Ticker("EURUSD=X").history(period="90d", interval="1h", timeout=5)
                if not df_eur.empty:
                    df_eur.index = df_eur.index.tz_convert('UTC') if df_eur.index.tz else df_eur.index.tz_localize('UTC')
            except Exception:
                pass

        if df_eur.empty or len(df_eur) < 200:
            raise ValueError("Data unavailable for EUR")

        # --- PATCH 5 APPLIED HERE ---
        now_utc = pd.Timestamp.now(tz="UTC")
        current_hour_start = now_utc.floor("h")

        df_eur = df_eur[
            df_eur.index < current_hour_start
        ].copy()

        if len(df_eur) < 200:
            raise ValueError(
                "Insufficient completed hourly EUR/USD candles."
            )

        current_price = float(df_eur['Close'].iloc[-1])
        # ----------------------------

        # --- CHATGPT'S DXY PATCH 1 APPLIED HERE ---
        df_eur = df_eur[['Open', 'High', 'Low', 'Close']].rename(
            columns={'Close': 'Close_EUR'}
        ).sort_index()

        df_eur = df_eur[~df_eur.index.duplicated(keep='last')]
        df_eur.index.name = 'datetime'

        df = df_eur.copy()
        dxy_available = False

        if not df_dxy.empty:
            try:
                df_dxy = df_dxy[['Close']].rename(
                    columns={'Close': 'Close_DXY'}
                ).sort_index()

                df_dxy = df_dxy[~df_dxy.index.duplicated(keep='last')]
                df_dxy.index.name = 'datetime'

                left = df_eur.reset_index().sort_values('datetime')
                right = df_dxy.reset_index().sort_values('datetime')

                merged = pd.merge_asof(
                    left,
                    right,
                    on='datetime',
                    direction='backward',
                    tolerance=pd.Timedelta('2h')
                )

                df = merged.set_index('datetime')

                matched_count = int(df['Close_DXY'].notna().sum())
                minimum_matches = max(100, int(len(df) * 0.30))

                latest_dxy_time = df['Close_DXY'].last_valid_index()

                latest_dxy_age = (
                    df.index[-1] - latest_dxy_time
                    if latest_dxy_time is not None
                    else pd.Timedelta.max
                )

                dxy_available = (
                    matched_count >= minimum_matches
                    and latest_dxy_age <= pd.Timedelta('2h')
                )

                if not dxy_available:
                    logger.warning(
                        "DXY data rejected: insufficient matched coverage "
                        "or latest observation is stale."
                    )
                    df = df_eur.copy()

            except Exception:
                logger.exception("DXY alignment failed; continuing without DXY.")
                df = df_eur.copy()
                dxy_available = False
        # ---------------------------------------

        df['Close'] = df['Close_EUR']

        # --- PATCH 7 APPLIED HERE ---
        # Build higher-timeframe candles using completed hourly data only.
        current_hour_start = pd.Timestamp.now(tz="UTC").floor("h")

        df_h4 = df['Close'].resample('4h').last().dropna()
        df_h4 = df_h4[
            (df_h4.index + pd.Timedelta(hours=4)) <= current_hour_start
        ]

        df_d1 = df['Close'].resample('1D').last().dropna()
        df_d1 = df_d1[
            (df_d1.index + pd.Timedelta(days=1)) <= current_hour_start
        ]
        # ----------------------------

        h4_trend, d1_trend = "Unknown", "Unknown"
        if len(df_h4) > 20:
            h4_sma = df_h4.rolling(20).mean().iloc[-1]
            if not pd.isna(h4_sma): h4_trend = "Up" if df_h4.iloc[-1] > h4_sma else "Down"
        if len(df_d1) > 10:
            d1_sma = df_d1.rolling(10).mean().iloc[-1]
            if not pd.isna(d1_sma): d1_trend = "Up" if df_d1.iloc[-1] > d1_sma else "Down"
        mtf_alignment = "Unknown"
        if h4_trend != "Unknown" and d1_trend != "Unknown":
            if h4_trend == "Up" and d1_trend == "Up": mtf_alignment = "Strong Bullish"
            elif h4_trend == "Down" and d1_trend == "Down": mtf_alignment = "Strong Bearish"
            else: mtf_alignment = "Conflict"

        # --- CHATGPT'S DXY PATCH 2 APPLIED HERE ---
        df['Clean_Close'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['ret_1h'] = df['Clean_Close'].pct_change(1)
        df['ret_24h'] = df['Clean_Close'].pct_change(24)

        if dxy_available:
            df['Clean_DXY'] = df['Close_DXY'].ewm(
                span=10,
                adjust=False,
                min_periods=10
            ).mean()

            df['dxy_ret_1h'] = df['Clean_DXY'].pct_change(
                periods=1,
                fill_method=None
            )

            df['dxy_corr'] = df['ret_1h'].rolling(
                window=24,
                min_periods=18
            ).corr(df['dxy_ret_1h'])

            latest_dxy_features = df[
                ['dxy_ret_1h', 'dxy_corr']
            ].iloc[-1]

            if not np.isfinite(latest_dxy_features).all():
                logger.warning(
                    "Latest DXY features are incomplete. "
                    "Disabling DXY for this forecast."
                )
                dxy_available = False

        if not dxy_available:
            df['dxy_ret_1h'] = 0.0
            df['dxy_corr'] = 0.0
        # ------------------------------------------

        df['sma_50'] = df['Clean_Close'].rolling(window=50).mean()
        df['dist_sma'] = (df['Clean_Close'] - df['sma_50']) / df['sma_50']
        df['trend_direction'] = np.where(df['Clean_Close'] > df['sma_50'], 1, -1)
        df['std_50'] = df['Clean_Close'].rolling(window=50).std()
        df['z_score'] = (df['Clean_Close'] - df['sma_50']) / (df['std_50'] + 1e-9)

        df['rsi'] = calculate_wilder_rsi(df['Close'], period=14)
        df['rsi_extreme_low'] = (df['rsi'] <= 5).astype(int)
        df['rsi_extreme_high'] = (df['rsi'] >= 95).astype(int)

        df['atr'] = calculate_true_wilder_atr(df, window=14)
        df['adx'], df['plus_di'], df['minus_di'], df['adx_rising'] = calculate_true_wilder_adx(df)

        current_adx = float(df['adx'].iloc[-1]) if not pd.isna(df['adx'].iloc[-1]) else 0.0
        market_regime = "Unknown"
        if current_adx > 25:
            market_regime = "Strong Bullish Trend" if df['plus_di'].iloc[-1] > df['minus_di'].iloc[-1] else "Strong Bearish Trend"
        elif current_adx > 20:
            market_regime = "Weak Trend"
        elif current_adx > 0:
            market_regime = "Ranging"

        df['range_size'] = df['High'] - df['Low']
        df['range_mean'] = df['range_size'].rolling(window=24).mean()
        df['range_spike'] = np.where(df['range_size'] > (df['range_mean'] * 1.5), 1, 0)

        df['prev_high'] = df['High'].shift(1)
        df['prev_low'] = df['Low'].shift(1)
        df['bearish_sweep'] = np.where((df['High'] > df['prev_high']) & (df['Close'] < df['prev_high']) & (df['range_spike'] == 1), -1, 0)
        df['bullish_sweep'] = np.where((df['Low'] < df['prev_low']) & (df['Close'] > df['prev_low']) & (df['range_spike'] == 1), 1, 0)
        df['net_sweep'] = df['bullish_sweep'] + df['bearish_sweep']

        window = 5
        df['swing_high'] = df['High'].rolling(window=window).max()
        df['swing_low'] = df['Low'].rolling(window=window).min()
        df['near_swing_high'] = np.where(df['Close'] > df['swing_high'] * 0.998, 1, 0)
        df['near_swing_low'] = np.where(df['Close'] < df['swing_low'] * 1.002, 1, 0)

        periods_ahead = req.forecast_days * 24

        df['target_end_time'] = df.index.to_series().shift(-periods_ahead)
        df['target'] = np.where(
            df['Close'].shift(-periods_ahead).notna(),
            (df['Close'].shift(-periods_ahead) > df['Close']).astype(float),
            np.nan
        )

        features = ['ret_1h', 'ret_24h', 'dist_sma', 'rsi', 'rsi_extreme_low', 'rsi_extreme_high',
                    'range_spike', 'net_sweep', 'near_swing_high', 'near_swing_low', 'trend_direction', 'z_score']

        if dxy_available:
            features.extend(['dxy_ret_1h', 'dxy_corr'])

        X_live = df[features].iloc[-1:].copy()
        if X_live.isna().any().any() or not np.isfinite(X_live).all().all():
            raise ValueError("Live feature vector contains missing or non-finite values.")

        df_train = df.dropna(subset=features + ['target']).copy()

        # --- PATCH 3 APPLIED HERE ---
        if len(df_train) < 250:
            raise ValueError(
                "Insufficient historical samples for reliable validation."
            )

        split_idx = int(len(df_train) * 0.75)

        if split_idx <= 0 or split_idx >= len(df_train):
            raise ValueError("Invalid chronological validation split.")

        val_start_time = df_train.index[split_idx]

        train_df = df_train[
            df_train['target_end_time'] < val_start_time
        ].copy()

        test_df = df_train[
            df_train.index >= val_start_time
        ].copy()

        if len(train_df) < 150:
            raise ValueError(
                "Insufficient post-purge training data."
            )

        if len(test_df) < 10:
            raise ValueError(
                "Insufficient validation samples."
            )

        if train_df['target'].nunique() < 2:
            raise ValueError(
                "Training data contains only one target class."
            )
        # ----------------------------

        base_xgb = xgb.XGBClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, objective='binary:logistic', eval_metric='logloss')
        base_rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)

        base_xgb.fit(train_df[features], train_df['target'])
        base_rf.fit(train_df[features], train_df['target'])

        brier, acc, baseline_brier, baseline_acc, bss = None, None, None, None, None
        val_sample_count = len(test_df)

        # --- PATCH 4 (FIRST REPLACEMENT) APPLIED HERE ---
        if val_sample_count >= 100:
            test_preds = (base_xgb.predict_proba(test_df[features])[:, 1] * 0.5) + (base_rf.predict_proba(test_df[features])[:, 1] * 0.5)
            brier = float(brier_score_loss(test_df['target'], test_preds))
            acc = float(accuracy_score(test_df['target'], (test_preds > 0.5).astype(int)))
            baseline_prob = float(train_df['target'].mean())
            baseline_preds = np.full(len(test_df), baseline_prob)
            baseline_brier = float(brier_score_loss(test_df['target'], baseline_preds))
            baseline_acc = float(accuracy_score(test_df['target'], (baseline_preds > 0.5).astype(int)))
            if baseline_brier > 0:
                bss = float(1.0 - (brier / baseline_brier))
        # ---------------------------------------------------

        final_xgb = xgb.XGBClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, objective='binary:logistic', eval_metric='logloss')
        final_rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        final_xgb.fit(df_train[features], df_train['target'])
        final_rf.fit(df_train[features], df_train['target'])

        p_up_xgb = float(final_xgb.predict_proba(X_live)[0][1])
        p_up_rf = float(final_rf.predict_proba(X_live)[0][1])
        p_up_pure = (p_up_xgb * 0.5) + (p_up_rf * 0.5)
        model_disagreement = float(abs(p_up_xgb - p_up_rf))

        current_atr = float(df['atr'].iloc[-1]) if not pd.isna(df['atr'].iloc[-1]) else None

        if current_atr is None or not np.isfinite(current_atr) or current_atr <= 0:
            fallback_tr = (df['High'] - df['Low']).tail(14).mean()
            if not np.isfinite(fallback_tr) or fallback_tr <= 0:
                raise ValueError("Invalid ATR and fallback. Forecast aborted.")
            current_atr = fallback_tr

        current_rsi = float(df['rsi'].iloc[-1]) if not pd.isna(df['rsi'].iloc[-1]) else 50.0
        current_trend = int(df['trend_direction'].iloc[-1]) if not pd.isna(df['trend_direction'].iloc[-1]) else 0
        current_z = float(df['z_score'].iloc[-1]) if not pd.isna(df['z_score'].iloc[-1]) else 0.0
        current_plus_di = float(df['plus_di'].iloc[-1]) if not pd.isna(df['plus_di'].iloc[-1]) else 0.0
        current_minus_di = float(df['minus_di'].iloc[-1]) if not pd.isna(df['minus_di'].iloc[-1]) else 0.0
        adx_is_rising = bool(df['adx_rising'].iloc[-1]) if not pd.isna(df['adx_rising'].iloc[-1]) else False

        SPREAD_PIPS = 0.00015
        expected_move = current_atr * 1.5

        # --- PATCH 4 (SECOND REPLACEMENT) APPLIED HERE ---
        validation_pass = (
            bss is not None
            and bss > 0.0
            and val_sample_count >= 100
            and test_df['target'].nunique() >= 2
        )
        # -------------------------------------------------

        raw_signal = "HOLD"
        if validation_pass and macro_available and market_regime != "Unknown" and calendar_available and event_risk == "LOW" and expected_move > SPREAD_PIPS:
            if p_up_pure > 0.60: raw_signal = "BUY"
            elif p_up_pure < 0.40: raw_signal = "SELL"

        signal = raw_signal
        horizon_conflict = "Aligned"
        if macro_available:
            if (macro_bias == "Bullish" and raw_signal == "SELL") or (macro_bias == "Bearish" and raw_signal == "BUY"):
                horizon_conflict = "Conflicting Horizons"
                signal = "HOLD"

        if signal == "SELL":
            if current_trend == 1 or current_z < -2.5: signal = "HOLD"
        elif signal == "BUY":
            if current_trend == -1 or current_z > 2.5: signal = "HOLD"

        if signal == "BUY":
            target_price = current_price + (3.0 * current_atr)
            stop_loss = current_price - (1.5 * current_atr)
        elif signal == "SELL":
            target_price = current_price - (3.0 * current_atr)
            stop_loss = current_price + (1.5 * current_atr)
        else:
            target_price = None
            stop_loss = None

        history = [{"date": d.strftime("%Y-%m-%d %H:%M:%S"), "price": float(r['Close'])} for d, r in df.tail(48).iterrows()]

        journal_record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "spot": current_price, "prob_up": round(p_up_pure, 4), "signal": signal,
            "target": round(target_price, 5) if target_price else None, "stop": round(stop_loss, 5) if stop_loss else None,
            "mtf_trend": mtf_alignment, "regime": market_regime, "adx": round(current_adx, 2), "forecast_days": req.forecast_days
        }
        save_journal_record(journal_record)

        last_signal = get_last_signal()

        if signal != last_signal:
            if signal in ["BUY", "SELL"]:
                tp_str = f"{target_price:.5f}" if target_price is not None else "N/A"
                sl_str = f"{stop_loss:.5f}" if stop_loss is not None else "N/A"
                bss_str = f"{bss*100:.1f}%" if bss is not None else "N/A"

                alert_msg = (
                    f"🚨 *NEW {signal} SIGNAL* 🚨\n\n"
                    f"💵 *Asset:* EUR/USD\n"
                    f"📍 *Entry Price:* {current_price:.5f}\n"
                    f"🎯 *Target:* {tp_str}\n"
                    f"🛑 *Stop Loss:* {sl_str}\n"
                    f"📊 *Prob Up:* {p_up_pure*100:.1f}%\n"
                    f"⚖️ *Brier Edge:* {bss_str}"
                )
                send_telegram_alert(alert_msg)
            elif signal == "HOLD" and last_signal in ["BUY", "SELL"]:
                send_telegram_alert(f"🔔 *Trade Closed / Signal Faded* 🔔\nEUR/USD is now back to HOLD.")

            update_last_signal(signal)

        # --- PATCH 6 APPLIED HERE ---
        raw_response = {
            "current_price": current_price,
            "forecast_days": req.forecast_days,
            "probability_up": float(p_up_pure),
            "xgb_probability": p_up_xgb,
            "rf_probability": p_up_rf,
            "model_disagreement": model_disagreement,
            "target_up_rate": float(train_df['target'].mean()),
            "validation_brier_score": brier,
            "validation_accuracy": acc,
            "baseline_brier_score": baseline_brier,
            "baseline_accuracy": baseline_acc,
            "brier_skill_score": bss,
            "validation_sample_count": val_sample_count,
            "signal": signal,
            "raw_signal": raw_signal,
            "horizon_conflict": horizon_conflict,
            "mtf_trend": mtf_alignment,
            "market_regime": market_regime,
            "adx_value": current_adx,
            "plus_di": current_plus_di,
            "minus_di": current_minus_di,
            "adx_rising": adx_is_rising,
            "calendar_available": calendar_available,
            "calendar_diagnostics": calendar_diagnostics,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "last_rsi": float(current_rsi),
            "macro_bias": macro_bias,
            "whale_trap_detected": bool(df['net_sweep'].iloc[-1] != 0),
            "event_risk": event_risk,
            "dxy_available": dxy_available,
            "dxy_correlation": float(df['dxy_corr'].iloc[-1]) if dxy_available and not pd.isna(df['dxy_corr'].iloc[-1]) else None,
            "trend_direction": "Up" if current_trend == 1 else "Down" if current_trend == -1 else "Unknown",
            "estimated_zscore_half_life_hours": 0.0,
            "recommended_position_size_pct": 0.0,
            "model_weights": {"xgb": 0.5, "rf": 0.5},
            "forecast_horizon_type": "trading_candles",
            "forecast_horizon_candles": periods_ahead,
            "forecast_horizon_requested_days": req.forecast_days,
            "atr_value": current_atr,
            "history": history,
            "validation_pass": validation_pass,
            "forecast_valid": True,
            "response_state": "normal" if validation_pass else "validation_failed"
        }
        # ----------------------------

        return {"status": "success", "data": adapt_for_flutter(raw_response)}

    except Exception as e:
        logger.exception("Short-term forecast failed")
        fallback_response = {
            "current_price": 0.0, "forecast_days": req.forecast_days,
            "probability_up": 0.0, "xgb_probability": 0.0, "rf_probability": 0.0,
            "model_disagreement": 0.0, "target_up_rate": 0.0,
            "validation_brier_score": None, "validation_accuracy": None,
            "baseline_brier_score": None, "baseline_accuracy": None,
            "brier_skill_score": None, "validation_sample_count": 0,
            "signal": "HOLD", "raw_signal": "HOLD",
            "horizon_conflict": "Unknown", "mtf_trend": "Unknown",
            "market_regime": "Unknown", "adx_value": None,
            "plus_di": None, "minus_di": None, "adx_rising": False,
            "calendar_available": False,
            "calendar_diagnostics": {
                "response_received": False, "events_returned": 0,
                "meta_present": False, "meta_covers_horizon": False,
                "span_check_passed": False, "density_check_passed": False,
                "coverage_evidence": "none", "reason": "fallback_path"
            },
            "target_price": None, "stop_loss": None,
            "last_rsi": 0.0, "macro_bias": "Unavailable", "whale_trap_detected": False,
            "event_risk": "UNKNOWN", "dxy_available": False, "dxy_correlation": None,
            "trend_direction": "Unknown", "estimated_zscore_half_life_hours": None,
            "recommended_position_size_pct": 0.0, "model_weights": {"xgb": 0.5, "rf": 0.5},
            "forecast_horizon_type": "trading_candles", "forecast_horizon_candles": req.forecast_days * 24,
            "forecast_horizon_requested_days": req.forecast_days, "atr_value": 0.0,
            "history": [],
            "validation_pass": False,
            "forecast_valid": False,
            "response_state": "degraded",
            "error_code": "CALCULATION_ERROR"
        }
        return {"status": "degraded", "data": adapt_for_flutter(fallback_response)}

@app.get("/journal")
def get_journal():
    return {"status": "success", "journal": get_journal_rows(limit=20)}

@app.get("/export_journal")
def export_journal():
    rows = get_journal_rows(limit=None)
    if not rows:
        raise HTTPException(status_code=404, detail="Journal is empty")
    return build_journal_csv_response(rows)

@app.post("/backtest")
def backtest(req: ShortTermRequest):
    return {"status": "success", "message": "Backtest endpoint placeholder. Awaiting full chronological engine."}

@app.get("/diagnostics/calendar")
def get_calendar_diagnostic():
    """Exposes the captured calendar sample to the browser for mobile inspection."""
    SAMPLE_FILE = "calendar_response_sample.json"
    if os.path.exists(SAMPLE_FILE):
        try:
            with open(SAMPLE_FILE, "r") as f:
                data = json.load(f)
            return {"status": "success", "data": data}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read diagnostic file: {e}")
    else:
        raise HTTPException(
            status_code=404, 
            detail="Diagnostic file not found. Please call /short_term_forecast at least once to trigger a calendar fetch."
        )