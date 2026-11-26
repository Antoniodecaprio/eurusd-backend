from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from fredapi import Fred
from scipy.stats import norm, linregress
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, accuracy_score
import yfinance as yf
import xgboost as xgb
import os
import requests
import warnings
import datetime
import logging
from statsmodels.tsa.stattools import adfuller
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

app = FastAPI(title="EUR/USD Quant Forecaster", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ENV VARIABLES
FRED_API_KEY = os.getenv("FRED_API_KEY")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

class ShortTermRequest(BaseModel):
    forecast_days: int = Field(default=3, ge=1, le=7)

class ForecastRequest(BaseModel):
    horizon_years: float = Field(default=3.0, gt=0.0, le=10.0)
    pi_us: float = Field(default=0.02, ge=-0.05, le=0.25)
    pi_ez: float = Field(default=0.02, ge=-0.05, le=0.25)

# ==========================================
# SUPABASE & TELEGRAM HELPERS
# ==========================================

def get_supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

def get_last_signal_db():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return "HOLD", 0.0
    try:
        url = f"{SUPABASE_URL}/rest/v1/signal_state?instrument=eq.EUR/USD"
        res = requests.get(url, headers=get_supabase_headers(), timeout=5)
        data = res.json()
        if data and len(data) > 0:
            last_alert = data[0].get("last_alert_at")
            if last_alert:
                # Convert ISO string to timestamp
                last_alert_time = datetime.datetime.fromisoformat(last_alert.replace('Z', '+00:00')).timestamp()
            else:
                last_alert_time = 0.0
            return data[0]["last_signal"], last_alert_time
    except Exception as e:
        logger.error(f"Supabase GET error: {e}")
    return "HOLD", 0.0

def update_last_signal_db(signal, timestamp):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        url = f"{SUPABASE_URL}/rest/v1/signal_state?instrument=eq.EUR/USD"
        dt_str = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc).isoformat()
        payload = {"last_signal": signal, "last_alert_at": dt_str, "updated_at": "now()"}
        requests.patch(url, headers=get_supabase_headers(), json=payload, timeout=5)
    except Exception as e:
        logger.error(f"Supabase PATCH error: {e}")

def log_forecast_to_db(record):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        url = f"{SUPABASE_URL}/rest/v1/forecast_journal"
        requests.post(url, headers=get_supabase_headers(), json=record, timeout=5)
    except Exception as e:
        logger.error(f"Supabase POST error: {e}")

def send_telegram_alert(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=3)
    except Exception as e:
        logger.error(f"Telegram alert failed: {e}")

# ==========================================
# ADAPTER & MATH HELPERS
# ==========================================

def adapt_for_flutter(data: dict) -> dict:
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

@lru_cache(maxsize=1)
def fetch_macro_data_cached(cache_buster: str):
    if not FRED_API_KEY: raise ValueError("FRED_API_KEY not set")
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
    try: adf_p = adfuller(df['q'], regression="c", autolag="AIC")[1]
    except: adf_p = 1.0

    df_ols['residuals'] = df_ols['q'] - (intercept + slope * df_ols['q_lag'])
    
    if 0.0 < slope < 1.0 and adf_p < 0.05:
        phi = slope
        q_bar = intercept / (1 - phi)
        deviation = df['q'].iloc[-1] - q_bar
        model_type = "AR(1)_mean_reverting"
        sigma_eps = df_ols['residuals'].std()
    else:
        phi = 1.0
        q_bar = None
        deviation = 0.0
        model_type = "random_walk_fallback"
        sigma_eps = df['q'].diff().std() * np.sqrt(12)
    return deviation, -1, phi, q_bar, model_type, sigma_eps

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
    tr = np.maximum(df['High'] - df['Low'], np.maximum(np.abs(df['High'] - df['Close'].shift(1)), np.abs(df['Low'] - df['Close'].shift(1))))
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

# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/")
def read_root():
    return {"status": "API is running - Quant v5.0 (Supabase)"}

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
        macro_bias = "Bearish" if (deviation > 0 and model_type == "AR(1)_mean_reverting") else "Bullish" if (deviation < 0 and model_type == "AR(1)_mean_reverting") else "Neutral"

        raw_response = {
            "current_state": {
                "spot_rate": current_spot, "ppp_equilibrium_estimate": ppp_anchor,
                "deviation_q": float(deviation), "half_life_years": half_life,
                "macro_model": model_type, "macro_bias": macro_bias
            },
            "forecast": {
                "p_up": float(p_up), "expected_rate": expected_rate, "median_rate": median_rate,
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
        macro_available, macro_bias, macro_dev = False, "Unavailable", 0.0
        try:
            cache_key = datetime.datetime.now().strftime("%Y-%m-%d-%H")
            df_macro = fetch_macro_data_cached(cache_key).copy()
            macro_dev_temp, _, _, _, macro_model, _ = get_macro_state(df_macro)
            if macro_model == "AR(1)_mean_reverting":
                macro_available = True
                macro_dev = macro_dev_temp
                macro_bias = "Bearish" if macro_dev > 0 else "Bullish"
        except Exception: pass

        # 2. Calendar
        event_risk, calendar_available = "UNKNOWN", False
        if TWELVE_DATA_KEY:
            try:
                cal_url = f"https://api.twelvedata.com/economic_calendar?country=US,EZ&apikey={TWELVE_DATA_KEY}"
                cal_res = requests.get(cal_url, timeout=4).json()
                if 'data' in cal_res and len(cal_res['data']) > 0:
                    valid_upcoming_events = 0
                    now = datetime.datetime.now(datetime.timezone.utc)
                    for event in cal_res['data']:
                        raw_time = event.get('date')
                        if raw_time:
                            try:
                                event_time = pd.to_datetime(raw_time)
                                if event_time.tzinfo is None: event_time = event_time.tz_localize('UTC')
                                if event_time >= now: 
                                    valid_upcoming_events += 1
                                    if event.get('impact') == 'High' and event_time <= now + datetime.timedelta(days=req.forecast_days):
                                        event_risk = "HIGH"
                            except Exception: continue
                    if valid_upcoming_events > 0:
                        calendar_available = True
                        if event_risk != "HIGH": event_risk = "LOW"
            except Exception: pass

        # 3. Data Fetch
        df_eur, df_dxy = pd.DataFrame(), pd.DataFrame()
        if TWELVE_DATA_KEY:
            try:
                url = f"https://api.twelvedata.com/time_series?symbol=EUR/USD&interval=1h&outputsize=1000&apikey={TWELVE_DATA_KEY}"
                response = requests.get(url, timeout=5).json()
                if 'values' in response:
                    df_eur = pd.DataFrame(response['values'])
                    df_eur['datetime'] = pd.to_datetime(df_eur['datetime'], utc=True) 
                    df_eur = df_eur.set_index('datetime').sort_index()
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df_eur.columns: df_eur[col] = pd.to_numeric(df_eur[col])
                    df_eur = df_eur.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
            except Exception: pass

        try:
            df_dxy = yf.Ticker("DX-Y.NYB").history(period="90d", interval="1h", timeout=5)
            if not df_dxy.empty:
                df_dxy.index = df_dxy.index.tz_convert('UTC') if df_dxy.index.tz else df_dxy.index.tz_localize('UTC')
        except Exception: pass

        if df_eur.empty or len(df_eur) < 200:
            try:
                df_eur = yf.Ticker("EURUSD=X").history(period="90d", interval="1h", timeout=5)
                if not df_eur.empty:
                    df_eur.index = df_eur.index.tz_convert('UTC') if df_eur.index.tz else df_eur.index.tz_localize('UTC')
            except Exception: pass

        if df_eur.empty or len(df_eur) < 200: raise ValueError("Data unavailable for EUR")

        dxy_available = not df_dxy.empty
        df_eur = df_eur[['Open', 'High', 'Low', 'Close']].rename(columns={'Close': 'Close_EUR'})
        df = df_eur.copy()
        if dxy_available:
            df_dxy = df_dxy[['Close']].rename(columns={'Close': 'Close_DXY'})
            df = df_eur.join(df_dxy, how='left')
            df['Close_DXY'] = df['Close_DXY'].ffill(limit=4)

        df = df.iloc[:-1].copy()
        df['Close'] = df['Close_EUR']
        current_price = float(df['Close'].iloc[-1])

        # MTF
        df_h4 = df['Close'].resample('4h').last().dropna().iloc[:-1]
        df_d1 = df['Close'].resample('1D').last().dropna().iloc[:-1]
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

        # Features
        df['Clean_Close'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['ret_1h'] = df['Clean_Close'].pct_change(1)
        df['ret_24h'] = df['Clean_Close'].pct_change(24)
        if dxy_available:
            df['Clean_DXY'] = df['Close_DXY'].ewm(span=10, adjust=False).mean()
            df['dxy_ret_1h'] = df['Clean_DXY'].pct_change(1)
            df['dxy_corr'] = df['ret_1h'].rolling(24).corr(df['dxy_ret_1h'])
        else:
            df['dxy_ret_1h'], df['dxy_corr'] = 0.0, 0.0

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
        if current_adx > 25: market_regime = "Strong Bullish Trend" if df['plus_di'].iloc[-1] > df['minus_di'].iloc[-1] else "Strong Bearish Trend"
        elif current_adx > 20: market_regime = "Weak Trend"
        elif current_adx > 0: market_regime = "Ranging"

        df['range_size'] = df['High'] - df['Low']
        df['range_mean'] = df['range_size'].rolling(window=24).mean()
        df['range_spike'] = np.where(df['range_size'] > (df['range_mean'] * 1.5), 1, 0)
        df['prev_high'], df['prev_low'] = df['High'].shift(1), df['Low'].shift(1)
        df['bearish_sweep'] = np.where((df['High'] > df['prev_high']) & (df['Close'] < df['prev_high']) & (df['range_spike'] == 1), -1, 0)
        df['bullish_sweep'] = np.where((df['Low'] < df['prev_low']) & (df['Close'] > df['prev_low']) & (df['range_spike'] == 1), 1, 0)
        df['net_sweep'] = df['bullish_sweep'] + df['bearish_sweep']

        window = 5
        df['swing_high'], df['swing_low'] = df['High'].rolling(window=window).max(), df['Low'].rolling(window=window).min()
        df['near_swing_high'] = np.where(df['Close'] > df['swing_high'] * 0.998, 1, 0)
        df['near_swing_low'] = np.where(df['Close'] < df['swing_low'] * 1.002, 1, 0)

        periods_ahead = req.forecast_days * 24 
        df['target_end_time'] = df.index.to_series().shift(-periods_ahead)
        df['target'] = np.where(df['Close'].shift(-periods_ahead).notna(), (df['Close'].shift(-periods_ahead) > df['Close']).astype(float), np.nan)
        
        features = ['ret_1h', 'ret_24h', 'dist_sma', 'rsi', 'rsi_extreme_low', 'rsi_extreme_high', 
                    'range_spike', 'net_sweep', 'near_swing_high', 'near_swing_low', 'trend_direction', 'z_score']
        if dxy_available: features.extend(['dxy_ret_1h', 'dxy_corr'])
        
        X_live = df[features].iloc[-1:].copy()
        if X_live.isna().any().any() or not np.isfinite(X_live).all().all(): raise ValueError("Live feature vector contains missing/non-finite values.")

        df_train = df.dropna(subset=features + ['target']).copy()
        split_idx = int(len(df_train) * 0.75)
        val_start_time = df_train.index[split_idx]
        train_df = df_train[df_train['target_end_time'] < val_start_time]
        test_df = df_train[df_train.index >= val_start_time]
        
        if len(train_df) < 150: raise ValueError("Insufficient post-purge training data.")

        base_xgb = xgb.XGBClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, objective='binary:logistic', eval_metric='logloss')
        base_rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        base_xgb.fit(train_df[features], train_df['target'])
        base_rf.fit(train_df[features], train_df['target'])

        brier, acc, baseline_brier, baseline_acc, bss = None, None, None, None, None
        val_sample_count = len(test_df)
        if val_sample_count > 10:
            test_preds = (base_xgb.predict_proba(test_df[features])[:, 1] * 0.5) + (base_rf.predict_proba(test_df[features])[:, 1] * 0.5)
            brier = float(brier_score_loss(test_df['target'], test_preds))
            acc = float(accuracy_score(test_df['target'], (test_preds > 0.5).astype(int)))
            baseline_prob = float(train_df['target'].mean())
            baseline_preds = np.full(len(test_df), baseline_prob)
            baseline_brier = float(brier_score_loss(test_df['target'], baseline_preds))
            baseline_acc = float(accuracy_score(test_df['target'], (baseline_preds > 0.5).astype(int)))
            if baseline_brier > 0: bss = float(1.0 - (brier / baseline_brier))

        final_xgb, final_rf = xgb.XGBClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, objective='binary:logistic'), RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        final_xgb.fit(df_train[features], df_train['target'])
        final_rf.fit(df_train[features], df_train['target'])

        p_up_xgb = float(final_xgb.predict_proba(X_live)[0][1])
        p_up_rf = float(final_rf.predict_proba(X_live)[0][1])
        p_up_pure = (p_up_xgb * 0.5) + (p_up_rf * 0.5)

        current_atr = float(df['atr'].iloc[-1]) if not pd.isna(df['atr'].iloc[-1]) else None
        if current_atr is None or not np.isfinite(current_atr) or current_atr <= 0:
            fallback_tr = (df['High'] - df['Low']).tail(14).mean() 
            if not np.isfinite(fallback_tr) or fallback_tr <= 0: raise ValueError("Invalid ATR")
            current_atr = fallback_tr

        current_rsi = float(df['rsi'].iloc[-1]) if not pd.isna(df['rsi'].iloc[-1]) else 50.0
        current_trend = int(df['trend_direction'].iloc[-1]) if not pd.isna(df['trend_direction'].iloc[-1]) else 0
        current_z = float(df['z_score'].iloc[-1]) if not pd.isna(df['z_score'].iloc[-1]) else 0.0
        current_plus_di = float(df['plus_di'].iloc[-1]) if not pd.isna(df['plus_di'].iloc[-1]) else 0.0
        current_minus_di = float(df['minus_di'].iloc[-1]) if not pd.isna(df['minus_di'].iloc[-1]) else 0.0
        adx_is_rising = bool(df['adx_rising'].iloc[-1]) if not pd.isna(df['adx_rising'].iloc[-1]) else False

        SPREAD_PIPS = 0.00015 
        expected_move = current_atr * 1.5 
        validation_pass = (bss is not None and bss > 0.0 and val_sample_count >= 10)
        
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

        target_price = None
        stop_loss = None
        if signal == "BUY":
            target_price = current_price + (3.0 * current_atr) 
            stop_loss = current_price - (1.5 * current_atr)
        elif signal == "SELL":
            target_price = current_price - (3.0 * current_atr)
            stop_loss = current_price + (1.5 * current_atr)

        # --- SUPABASE & TELEGRAM LOGIC ---
        last_signal, last_alert_time = get_last_signal_db()
        current_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
        cooldown_period = 14400 
        can_send_alert = (current_time - last_alert_time) > cooldown_period

        if signal != last_signal:
            if signal in ["BUY", "SELL"] and can_send_alert:
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
                update_last_signal_db(signal, current_time)
            elif signal == "HOLD" and last_signal in ["BUY", "SELL"]:
                send_telegram_alert(f"🔔 *Trade Closed / Signal Faded* 🔔\nEUR/USD is now back to HOLD.")
                update_last_signal_db(signal, current_time)
            elif signal != last_signal:
                update_last_signal_db(signal, last_alert_time)

        # Write to Supabase Journal
        journal_record = {
            "spot_price": current_price, "prob_up": round(p_up_pure, 4), "signal": signal,
            "target_price": round(target_price, 5) if target_price else None, "stop_loss": round(stop_loss, 5) if stop_loss else None,
            "mtf_trend": mtf_alignment, "regime": market_regime, "adx": round(current_adx, 2)
        }
        log_forecast_to_db(journal_record)
        # ---------------------------------

        raw_response = {
            "current_price": current_price, "forecast_days": req.forecast_days,
            "probability_up": float(p_up_pure), "xgb_probability": p_up_xgb, "rf_probability": p_up_rf,
            "model_disagreement": float(abs(p_up_xgb - p_up_rf)), "target_up_rate": float(train_df['target'].mean()),
            "validation_brier_score": brier, "validation_accuracy": acc, "baseline_brier_score": baseline_brier,
            "baseline_accuracy": baseline_acc, "brier_skill_score": bss, "validation_sample_count": val_sample_count,
            "signal": signal, "raw_signal": raw_signal, "horizon_conflict": horizon_conflict, "mtf_trend": mtf_alignment,
            "market_regime": market_regime, "adx_value": current_adx, "plus_di": current_plus_di,
            "minus_di": current_minus_di, "adx_rising": adx_is_rising, "calendar_available": calendar_available,
            "target_price": target_price, "stop_loss": stop_loss, "last_rsi": float(current_rsi),
            "macro_bias": macro_bias, "whale_trap_detected": bool(df['net_sweep'].iloc[-1] != 0),
            "event_risk": event_risk, "dxy_available": dxy_available, "dxy_correlation": float(df['dxy_corr'].iloc[-1]) if dxy_available and not pd.isna(df['dxy_corr'].iloc[-1]) else None,
            "trend_direction": "Up" if current_trend == 1 else "Down" if current_trend == -1 else "Unknown",
            "estimated_zscore_half_life_hours": 0.0, "recommended_position_size_pct": 0.0,
            "model_weights": {"xgb": 0.5, "rf": 0.5}, "forecast_horizon_type": "trading_candles",
            "forecast_horizon_candles": periods_ahead, "forecast_horizon_requested_days": req.forecast_days,
            "atr_value": current_atr, "history": [], "forecast_valid": True, "response_state": "normal"
        }
        return {"status": "success", "data": adapt_for_flutter(raw_response)}

    except Exception as e:
        logger.exception("Short-term forecast failed")
        fallback_response = {
            "current_price": 0.0, "forecast_days": req.forecast_days, "probability_up": 0.0, "xgb_probability": 0.0,
            "rf_probability": 0.0, "model_disagreement": 0.0, "target_up_rate": 0.0, "validation_brier_score": None,
            "validation_accuracy": None, "baseline_brier_score": None, "baseline_accuracy": None, "brier_skill_score": None,
            "validation_sample_count": 0, "signal": "HOLD", "raw_signal": "HOLD", "horizon_conflict": "Unknown",
            "mtf_trend": "Unknown", "market_regime": "Unknown", "adx_value": None, "plus_di": None, "minus_di": None,
            "adx_rising": False, "calendar_available": False, "target_price": None, "stop_loss": None, "last_rsi": 0.0,
            "macro_bias": "Unavailable", "whale_trap_detected": False, "event_risk": "UNKNOWN", "dxy_available": False,
            "dxy_correlation": None, "trend_direction": "Unknown", "estimated_zscore_half_life_hours": None,
            "recommended_position_size_pct": 0.0, "model_weights": {"xgb": 0.5, "rf": 0.5},
            "forecast_horizon_type": "trading_candles", "forecast_horizon_candles": req.forecast_days * 24,
            "forecast_horizon_requested_days": req.forecast_days, "atr_value": 0.0, "history": [],
            "forecast_valid": False, "response_state": "degraded", "error_code": "CALCULATION_ERROR"
        }
        return {"status": "degraded", "data": adapt_for_flutter(fallback_response)}

@app.get("/journal")
def get_journal():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"status": "success", "journal": []}
    try:
        url = f"{SUPABASE_URL}/rest/v1/forecast_journal?select=*&order=timestamp.desc&limit=20"
        res = requests.get(url, headers=get_supabase_headers(), timeout=5)
        return {"status": "success", "journal": res.json()}
    except Exception:
        return {"status": "success", "journal": []}

@app.post("/backtest")
def backtest(req: ShortTermRequest):
    return {"status": "success", "message": "Backtest endpoint placeholder."}