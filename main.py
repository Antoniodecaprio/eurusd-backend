from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from fredapi import Fred
from scipy.stats import norm, linregress
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf
import xgboost as xgb
import os
import requests
import warnings
import datetime
import logging
import csv
from statsmodels.tsa.stattools import adfuller
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

app = FastAPI(title="EUR/USD Quant Forecaster", version="3.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRED_API_KEY = os.getenv("FRED_API_KEY")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY")

JOURNAL_FILE = "forecast_journal.csv"

class ForecastRequest(BaseModel):
    horizon_years: float = Field(default=3.0, gt=0.0, le=10.0)
    pi_us: float = Field(default=0.02, ge=-0.05, le=0.25)
    pi_ez: float = Field(default=0.02, ge=-0.05, le=0.25)

class ShortTermRequest(BaseModel):
    forecast_days: int = Field(default=3, ge=1, le=14)

# ==========================================
# HELPER & CACHING FUNCTIONS
# ==========================================

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

def calculate_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=window).mean()

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

def log_forecast_to_csv(record):
    """Lightweight Forecast Journal"""
    file_exists = os.path.isfile(JOURNAL_FILE)
    try:
        with open(JOURNAL_FILE, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
    except Exception as e:
        logger.error(f"Failed to write to journal: {e}")

# ==========================================
# MACRO ENDPOINT
# ==========================================

@app.get("/")
def read_root():
    return {"status": "API is running - Quant v3.3"}

@app.post("/forecast")
def get_forecast(req: ForecastRequest):
    try:
        cache_key = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        df = fetch_macro_data_cached(cache_key).copy()
        current_spot = float(df['spot'].iloc[-1])
        
        deviation, current_regime, phi, q_bar, model_type, sigma_eps = get_macro_state(df)
        
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

        return {
            "status": "success",
            "data": {
                "current_state": {
                    "spot_rate": current_spot,
                    "ppp_equilibrium_estimate": ppp_anchor,
                    "deviation_q": float(deviation),
                    "half_life_years": float(half_life) if half_life else None,
                    "macro_model": model_type,
                    "macro_bias": macro_bias
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
        }
    except Exception as e:
        logger.error(f"Macro Forecast Error: {str(e)}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": "Internal calculation error"})

# ==========================================
# MICRO ENDPOINT (SHORT-TERM ML)
# ==========================================

@app.post("/short_term_forecast")
def get_short_term_forecast(req: ShortTermRequest):
    try:
        # 1. Macro State
        try:
            cache_key = datetime.datetime.now().strftime("%Y-%m-%d-%H")
            df_macro = fetch_macro_data_cached(cache_key).copy()
            macro_dev, _, _, _, macro_model, _ = get_macro_state(df_macro)
            macro_bias = "Bearish" if (macro_dev > 0 and macro_model == "AR(1)_mean_reverting") else "Bullish" if (macro_dev < 0 and macro_model == "AR(1)_mean_reverting") else "Unavailable"
        except Exception:
            macro_dev, macro_bias = 0.0, "Unavailable"

        # 2. Event Risk
        event_risk = "UNKNOWN"
        if TWELVE_DATA_KEY:
            try:
                cal_url = f"https://api.twelvedata.com/economic_calendar?country=US,EZ&apikey={TWELVE_DATA_KEY}"
                cal_res = requests.get(cal_url, timeout=4).json()
                if 'data' in cal_res:
                    event_risk = "LOW"
                    now = datetime.datetime.now(datetime.timezone.utc)
                    for event in cal_res['data']:
                        raw_time = event.get('date')
                        if raw_time:
                            event_time = pd.to_datetime(raw_time)
                            if event_time.tzinfo is None:
                                event_time = event_time.tz_localize('UTC')
                            if event.get('impact') == 'High' and now <= event_time <= now + datetime.timedelta(days=req.forecast_days):
                                event_risk = "HIGH"
                                break
            except Exception:
                pass

        # 3. Data Fetching
        df_eur = pd.DataFrame()
        df_dxy = pd.DataFrame()
        
        if TWELVE_DATA_KEY:
            try:
                url = f"https://api.twelvedata.com/time_series?symbol=EUR/USD&interval=1h&outputsize=500&apikey={TWELVE_DATA_KEY}"
                response = requests.get(url, timeout=5).json()
                if 'values' in response:
                    df_eur = pd.DataFrame(response['values'])
                    df_eur['datetime'] = pd.to_datetime(df_eur['datetime'])
                    df_eur = df_eur.set_index('datetime').sort_index()
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df_eur.columns:
                            df_eur[col] = pd.to_numeric(df_eur[col])
                    df_eur = df_eur.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
            except Exception:
                pass

        try:
            df_dxy = yf.Ticker("DX-Y.NYB").history(period="60d", interval="1h", timeout=5)
        except Exception:
            pass

        if df_eur.empty or len(df_eur) < 100:
            try:
                df_eur = yf.Ticker("EURUSD=X").history(period="60d", interval="1h", timeout=5)
            except Exception:
                pass

        if df_eur.empty or len(df_eur) < 100:
            raise ValueError("Data unavailable for EUR")

        dxy_available = not df_dxy.empty
        
        df_eur = df_eur[['Open', 'High', 'Low', 'Close']].rename(columns={'Close': 'Close_EUR'})
        if df_eur.index.tz is not None: df_eur.index = df_eur.index.tz_localize(None)
        
        if dxy_available:
            df_dxy = df_dxy[['Close']].rename(columns={'Close': 'Close_DXY'})
            if df_dxy.index.tz is not None: df_dxy.index = df_dxy.index.tz_localize(None)
            df = df_eur.join(df_dxy, how='left').ffill().dropna()
        else:
            df = df_eur.copy()
            df['Close_DXY'] = np.nan

        df['Close'] = df['Close_EUR']
        current_price = float(df['Close'].iloc[-1])

        # PHASE 1.5: Frankfurter Data Check (No API key needed)
        data_discrepancy = False
        try:
            frank_res = requests.get("https://api.frankfurter.app/latest?from=EUR&to=USD", timeout=3).json()
            frank_rate = float(frank_res['rates']['USD'])
            if abs(frank_rate - current_price) / current_price > 0.01: # > 1% difference
                data_discrepancy = True
        except Exception:
            frank_rate = None

        # PHASE 1.5: Multi-Timeframe Alignment
        df_h4 = df['Close'].resample('4h').last().dropna()
        df_d1 = df['Close'].resample('1D').last().dropna()
        
        h4_trend = "Up" if df_h4.iloc[-1] > df_h4.rolling(20).mean().iloc[-1] else "Down"
        d1_trend = "Up" if df_d1.iloc[-1] > df_d1.rolling(10).mean().iloc[-1] else "Down"
        
        mtf_alignment = "Conflict"
        if h4_trend == "Up" and d1_trend == "Up": mtf_alignment = "Strong Bullish"
        elif h4_trend == "Down" and d1_trend == "Down": mtf_alignment = "Strong Bearish"

        # Feature Engineering
        df['Clean_Close'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['ret_1h'] = df['Clean_Close'].pct_change(1)
        df['ret_24h'] = df['Clean_Close'].pct_change(24)
        
        if dxy_available:
            df['Clean_DXY'] = df['Close_DXY'].ewm(span=10, adjust=False).mean()
            df['dxy_ret_1h'] = df['Clean_DXY'].pct_change(1)
            df['dxy_corr'] = df['ret_1h'].rolling(24).corr(df['dxy_ret_1h'])
        else:
            df['dxy_ret_1h'] = 0.0
            df['dxy_corr'] = 0.0

        df['sma_50'] = df['Clean_Close'].rolling(window=50).mean()
        df['dist_sma'] = (df['Clean_Close'] - df['sma_50']) / df['sma_50']
        df['trend_direction'] = np.where(df['Clean_Close'] > df['sma_50'], 1, -1)
        df['std_50'] = df['Clean_Close'].rolling(window=50).std()
        df['z_score'] = (df['Clean_Close'] - df['sma_50']) / (df['std_50'] + 1e-9)
        
        df['rsi'] = calculate_wilder_rsi(df['Close'], period=14)
        df['rsi_extreme_low'] = (df['rsi'] <= 5).astype(int)
        df['rsi_extreme_high'] = (df['rsi'] >= 95).astype(int)

        df['atr'] = calculate_atr(df, window=14)
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
        df['target'] = np.where(
            df['Close'].shift(-periods_ahead).notna(),
            (df['Close'].shift(-periods_ahead) > df['Close']).astype(float),
            np.nan
        )
        
        features = ['ret_1h', 'ret_24h', 'dist_sma', 'rsi', 'rsi_extreme_low', 'rsi_extreme_high', 
                    'range_spike', 'net_sweep', 'near_swing_high', 'near_swing_low', 'trend_direction', 'z_score']
        
        if dxy_available:
            features.extend(['dxy_ret_1h', 'dxy_corr'])
        
        df_live = df.iloc[:-1].copy() 
        X_live = df_live[features].iloc[-1:].copy()
        df_train = df.dropna(subset=features + ['target']).copy()
        
        if len(df_train) < 150:
            raise ValueError("Not enough clean data to train model")

        split_idx = int(len(df_train) * 0.75)
        purge_time = df_train.index[split_idx] - pd.Timedelta(hours=periods_ahead)
        train_df = df_train[df_train.index <= purge_time]
        
        target_up_rate = float(train_df['target'].mean())

        base_xgb = xgb.XGBClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, objective='binary:logistic', eval_metric='logloss')
        base_rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)

        base_xgb.fit(train_df[features], train_df['target'])
        base_rf.fit(train_df[features], train_df['target'])

        p_up_xgb = float(base_xgb.predict_proba(X_live)[0][1])
        p_up_rf = float(base_rf.predict_proba(X_live)[0][1])
        p_up_pure = (p_up_xgb * 0.5) + (p_up_rf * 0.5)
        model_disagreement = float(abs(p_up_xgb - p_up_rf))

        current_atr = float(df['atr'].iloc[-1])
        current_rsi = float(df['rsi'].iloc[-1])
        current_trend = int(df['trend_direction'].iloc[-1])
        current_z = float(df['z_score'].iloc[-1])
        
        recent_df = df.tail(100).copy()
        recent_df['x_lag'] = recent_df['z_score'].shift(1)
        recent_df = recent_df.dropna()
        slope_z, _, _, _, _ = linregress(recent_df['x_lag'], recent_df['z_score'])
        phi_z = float(np.clip(slope_z, 0.01, 0.999))
        reversal_hours = float(np.log(0.5) / np.log(phi_z))
        
        SPREAD_PIPS = 0.00015 
        expected_move = current_atr * 1.5 
        
        raw_signal = "HOLD"
        if event_risk in ["HIGH", "UNKNOWN"]:
            raw_signal = "HOLD" 
        elif p_up_pure > 0.60 and macro_dev <= 0.03 and expected_move > SPREAD_PIPS:
            raw_signal = "BUY"
            if df['net_sweep'].iloc[-1] == -1: raw_signal = "HOLD"
        elif p_up_pure < 0.40 and macro_dev >= -0.03 and expected_move > SPREAD_PIPS:
            raw_signal = "SELL"
            if df['net_sweep'].iloc[-1] == 1: raw_signal = "HOLD"

        signal = raw_signal
        
        horizon_conflict = "Aligned"
        if macro_bias != "Unavailable":
            if (macro_bias == "Bullish" and raw_signal == "SELL") or (macro_bias == "Bearish" and raw_signal == "BUY"):
                horizon_conflict = "Conflicting Horizons"
            elif macro_bias == "Neutral":
                horizon_conflict = "Macro Neutral"

        if signal == "SELL":
            if current_trend == 1: signal = "HOLD" 
            if current_z < -2.5: signal = "HOLD" 
            if current_rsi < 15 and current_z > -1.5: signal = "HOLD" 
        elif signal == "BUY":
            if current_trend == -1: signal = "HOLD" 
            if current_z > 2.5: signal = "HOLD" 
            if current_rsi > 85 and current_z < 1.5: signal = "HOLD" 

        if signal == "BUY":
            target_price = current_price + (3.0 * current_atr) 
            stop_loss = current_price - (1.5 * current_atr)
        elif signal == "SELL":
            target_price = current_price - (3.0 * current_atr)
            stop_loss = current_price + (1.5 * current_atr)
        else:
            target_price = current_price + (1.5 * current_atr)
            stop_loss = current_price - (1.5 * current_atr)

        history = [{"date": d.strftime("%Y-%m-%d %H:%M:%S"), "price": float(r['Close'])} for d, r in df.tail(48).iterrows()]

        # PHASE 1.5: Log to Journal
        journal_record = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "spot": current_price,
            "prob_up": round(p_up_pure, 4),
            "signal": signal,
            "target": round(target_price, 5),
            "stop": round(stop_loss, 5),
            "mtf_trend": mtf_alignment
        }
        log_forecast_to_csv(journal_record)

        return {
            "status": "success",
            "data": {
                "current_price": current_price,
                "forecast_days": req.forecast_days,
                "probability_up": float(p_up_pure),
                "xgb_probability": p_up_xgb,
                "rf_probability": p_up_rf,
                "model_disagreement": model_disagreement,
                "target_up_rate": target_up_rate,
                "signal": signal,
                "raw_signal": raw_signal,
                "horizon_conflict": horizon_conflict,
                "mtf_trend": mtf_alignment,
                "data_discrepancy": data_discrepancy,
                "target_price": float(target_price),
                "stop_loss": float(stop_loss),
                "last_rsi": float(current_rsi),
                "macro_bias": macro_bias,
                "whale_trap_detected": bool(df['net_sweep'].iloc[-1] != 0),
                "event_risk": event_risk,
                "dxy_available": dxy_available,
                "dxy_correlation": float(df['dxy_corr'].iloc[-1]) if dxy_available else 0.0,
                "trend_direction": "Up" if current_trend == 1 else "Down",
                "reversal_expectation_hours": reversal_hours,
                "recommended_position_size_pct": 0.0,
                "model_weights": {"xgb": 0.5, "rf": 0.5},
                "history": history
            }
        }
    except Exception as e:
        logger.error(f"Micro Forecast Error: {str(e)}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.get("/journal")
def get_journal():
    """Read the last 20 forecasts"""
    if not os.path.isfile(JOURNAL_FILE):
        return {"status": "success", "journal": []}
    try:
        with open(JOURNAL_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return {"status": "success", "journal": rows[-20:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))