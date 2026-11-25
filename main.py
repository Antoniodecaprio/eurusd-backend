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
from statsmodels.tsa.stattools import adfuller
from functools import lru_cache

# Setup strict logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

app = FastAPI(title="EUR/USD Quant Forecaster", version="3.0.0")

# Fixed CORS: Removed wildcard credentials leak
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Restrict this to your FlutterFlow domain in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRED_API_KEY = os.getenv("FRED_API_KEY")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY")

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
    """Caches FRED data to prevent Render timeouts. Cache updates every 12 hours."""
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
    df_ols = df.dropna()
    
    slope, intercept, _, _, _ = linregress(df_ols['q_lag'], df_ols['q'])
    
    try:
        adf_p = adfuller(df['q'], regression="c", autolag="AIC")[1]
    except:
        adf_p = 1.0

    df['q_ret'] = df['q'].diff()
    gmm = GaussianMixture(n_components=2, random_state=42)
    valid_returns = df['q_ret'].dropna()
    gmm.fit(valid_returns.values.reshape(-1, 1))
    regimes = gmm.predict(valid_returns.values.reshape(-1, 1))
    current_regime = int(regimes[-1])
    
    if 0.0 < slope < 1.0 and adf_p < 0.05:
        phi = slope
        q_bar = intercept / (1 - phi)
        deviation = df['q'].iloc[-1] - q_bar
        model_type = "AR(1)_mean_reverting"
        
        # REVIEWER FIX: AR(1) Innovation Variance
        df_ols['residuals'] = df_ols['q'] - (intercept + phi * df_ols['q_lag'])
        df_ols['regime'] = regimes
        current_resids = df_ols[df_ols['regime'] == current_regime]['residuals']
        # Fallback to global residual variance if regime sample is too small
        sigma_eps = current_resids.std() if len(current_resids) > 10 else df_ols['residuals'].std()
        
    else:
        phi = 1.0
        q_bar = None
        deviation = 0.0
        model_type = "random_walk_fallback"
        drift = df['q'].diff().mean()
        # RW variance (annualized)
        sigma_eps = df['q'].diff().std() * np.sqrt(12)
        
    return deviation, current_regime, phi, q_bar, model_type, sigma_eps

def calculate_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=window).mean()

# ==========================================
# MACRO ENDPOINT
# ==========================================

@app.get("/")
def read_root():
    return {"status": "API is running - Quant v3.0"}

@app.post("/forecast")
def get_forecast(req: ForecastRequest):
    try:
        # 12-hour cache buster
        cache_key = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        df = fetch_macro_data_cached(cache_key)
        current_spot = float(df['spot'].iloc[-1])
        
        deviation, current_regime, phi, q_bar, model_type, sigma_eps = get_macro_state(df)
        
        if model_type == "AR(1)_mean_reverting":
            current_q = np.log(current_spot) + df['p_us'].iloc[-1] - df['p_ez'].iloc[-1]
            half_life = np.log(0.5) / np.log(phi) / 12 # in years
            phi_n = phi ** (req.horizon_years * 12)
            drift_q = (phi_n - 1) * (current_q - q_bar)
            ppp_anchor = float(np.exp(q_bar - df['p_us'].iloc[-1] + df['p_ez'].iloc[-1]))
            
            # REVIEWER FIX: Finite Geometric Sum for AR(1) Variance
            var_q = (sigma_eps ** 2) * (1 - phi ** (2 * req.horizon_years * 12)) / (1 - phi ** 2)
        else:
            half_life = None
            drift_q = df['q'].diff().mean() * (req.horizon_years * 12)
            ppp_anchor = None
            # Random Walk variance
            var_q = (sigma_eps ** 2) * req.horizon_years

        drift_infl = (req.pi_ez - req.pi_us) * req.horizon_years
        mean_delta = drift_q + drift_infl
        std_delta = np.sqrt(var_q)
        
        p_up = norm.cdf(mean_delta / std_delta) if std_delta > 0 else 0.5
        
        # REVIEWER FIX: Median vs Expected Value in Lognormal Distribution
        median_rate = float(np.exp(np.log(current_spot) + mean_delta))
        expected_rate = float(np.exp(np.log(current_spot) + mean_delta + (var_q / 2)))
        
        historical_data = [{"date": d.strftime("%Y-%m-%d"), "price": float(r['spot'])} for d, r in df.tail(60).iterrows()]
        
        # REVIEWER FIX: Corrected Macro Bias Direction (Positive dev = Overvalued EUR = Bearish)
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
            df_macro = fetch_macro_data_cached(cache_key)
            macro_dev, _, _, _, _, _ = get_macro_state(df_macro)
        except Exception:
            macro_dev = 0.0

        # 2. Economic Calendar Check (Fails to UNKNOWN, not LOW)
        event_risk = "UNKNOWN"
        if TWELVE_DATA_KEY:
            try:
                cal_url = f"https://api.twelvedata.com/economic_calendar?country=US,EZ&apikey={TWELVE_DATA_KEY}"
                cal_res = requests.get(cal_url, timeout=4).json()
                if 'data' in cal_res:
                    event_risk = "LOW"
                    now = datetime.datetime.now(datetime.timezone.utc)
                    for event in cal_res['data']:
                        # Only flag events within the forecast window
                        event_time = pd.to_datetime(event.get('date')).tz_localize('UTC')
                        if event.get('impact') == 'High' and now <= event_time <= now + datetime.timedelta(days=req.forecast_days):
                            event_risk = "HIGH"
                            break
            except Exception:
                pass # Remains UNKNOWN

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

        # REVIEWER FIX: Do not fabricate DXY. Exclude if missing.
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
        
        # 4. Feature Engineering (Removed fabricated volume)
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
        
        delta = df['Clean_Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9) 
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df['atr'] = calculate_atr(df, window=14)
        
        # REVIEWER FIX: Rename "Whale" to actual metric (Range Expansion)
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

        df['macro_dev'] = macro_dev

        # REVIEWER FIX: Target Construction (Preserve NaNs)
        periods_ahead = req.forecast_days * 24 
        df['target'] = np.where(
            df['Close'].shift(-periods_ahead).notna(),
            (df['Close'].shift(-periods_ahead) > df['Close']).astype(float),
            np.nan
        )
        
        features = ['ret_1h', 'dxy_ret_1h', 'dxy_corr', 'ret_24h', 'dist_sma', 'rsi', 'range_spike', 'net_sweep', 
                    'macro_dev', 'near_swing_high', 'near_swing_low', 'trend_direction', 'z_score']
        
        # Shift features to predict the NEXT candle
        df[features] = df[features].shift(1) 
        
        # Split LIVE data (unlabelled) from TRAIN data (labelled)
        X_live = df[features].iloc[-1:].copy()
        df_train = df.dropna(subset=features + ['target']).copy()
        
        if len(df_train) < 150:
            raise ValueError("Not enough clean data to train model")

        # REVIEWER FIX: Purged Split
        split_idx = int(len(df_train) * 0.75)
        train_df = df_train.iloc[:split_idx - periods_ahead] 
        
        base_xgb = xgb.XGBClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, objective='binary:logistic', eval_metric='logloss')
        base_rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)

        base_xgb.fit(train_df[features], train_df['target'])
        base_rf.fit(train_df[features], train_df['target'])

        # 6. Prediction
        xgb_weight = 0.5
        rf_weight = 0.5
        p_up_pure = (base_xgb.predict_proba(X_live)[0][1] * xgb_weight) + (base_rf.predict_proba(X_live)[0][1] * rf_weight)

        current_price = float(df['Close'].iloc[-1])
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
        
        # 7. Signal Logic
        SPREAD_PIPS = 0.00015 
        expected_move = current_atr * 1.5 
        
        rsi_low_bound = max(df['rsi'].tail(500).quantile(0.05), 25.0)
        rsi_high_bound = min(df['rsi'].tail(500).quantile(0.95), 75.0)
        
        signal = "HOLD"
        
        # REVIEWER FIX: Macro direction aligned with q formula. High q = Overvalued = SELL bias.
        if event_risk in ["HIGH", "UNKNOWN"]:
            signal = "HOLD" # Block signals if event risk is unsafe or unknown
        elif p_up_pure > 0.60 and macro_dev <= 0.03 and expected_move > SPREAD_PIPS:
            signal = "BUY"
            if df['net_sweep'].iloc[-1] == -1: signal = "HOLD"
        elif p_up_pure < 0.40 and macro_dev >= -0.03 and expected_move > SPREAD_PIPS:
            signal = "SELL"
            if df['net_sweep'].iloc[-1] == 1: signal = "HOLD"

        # Vetoes
        if signal == "SELL" and current_trend == 1: signal = "HOLD" 
        if signal == "BUY" and current_trend == -1: signal = "HOLD" 
        if signal == "SELL" and current_z < -2.0: signal = "HOLD" 
        if signal == "BUY" and current_z > 2.0: signal = "HOLD" 
        if signal == "BUY" and current_rsi > 75: signal = "HOLD"
        if signal == "SELL" and current_rsi < 25: signal = "HOLD"

        # 8. REVIEWER FIX: Kelly Sizing Disabled
        kelly_fraction = 0.0 # Safely disabled per review

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

        return {
            "status": "success",
            "data": {
                "current_price": current_price,
                "probability_up": float(p_up_pure),
                "signal": signal,
                "target_price": float(target_price),
                "stop_loss": float(stop_loss),
                "last_rsi": float(current_rsi),
                "macro_bias": "Bearish" if macro_dev > 0 else "Bullish", # Corrected mapping
                "sweep_detected": bool(df['net_sweep'].iloc[-1] != 0), # Renamed
                "event_risk": event_risk,
                "dxy_available": dxy_available, # Added flag for UI
                "trend_direction": "Up" if current_trend == 1 else "Down",
                "reversal_expectation_hours": reversal_hours,
                "recommended_position_size_pct": 0.0, # Hardcoded to 0
                "history": history
            }
        }
    except Exception as e:
        logger.error(f"Micro Forecast Error: {str(e)}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})