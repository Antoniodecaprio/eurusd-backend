from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from fredapi import Fred
from scipy.stats import norm, linregress
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
import yfinance as yf
import xgboost as xgb
import os
import datetime
import requests
import warnings
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings('ignore')

app = FastAPI(title="EUR/USD Forecaster", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
# HELPER FUNCTIONS
# ==========================================

def fetch_macro_data():
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

def get_macro_state():
    df = fetch_macro_data()
    df['q_lag'] = df['q'].shift(1)
    df_ols = df.dropna()
    
    slope, intercept, _, _, _ = linregress(df_ols['q_lag'], df_ols['q'])
    
    # ADF Stationarity Test
    try:
        adf_p = adfuller(df['q'], regression="c", autolag="AIC")[1]
    except:
        adf_p = 1.0

    # Use AR(1) if stationary, else Random Walk
    if 0.0 < slope < 1.0 and adf_p < 0.05:
        phi = slope
        q_bar = intercept / (1 - phi)
        deviation = df['q'].iloc[-1] - q_bar
        model_type = "AR(1)_mean_reverting"
        drift = 0.0
    else:
        phi = 1.0
        q_bar = None
        deviation = 0.0
        model_type = "random_walk_fallback"
        drift = df['q'].diff().mean()
    
    df['returns'] = df['s'].diff()
    df['vol'] = df['returns'].rolling(window=6).std().bfill()
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(df['vol'].values.reshape(-1, 1))
    regimes = gmm.predict(df['vol'].values.reshape(-1, 1))
    current_regime = int(regimes[-1])
    
    return deviation, current_regime, phi, q_bar, model_type, drift

def calculate_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=window).mean()

# ==========================================
# MACRO ENDPOINTS
# ==========================================

@app.get("/")
def read_root():
    return {"status": "API is running - Optimized v2.2"}

@app.post("/forecast")
def get_forecast(req: ForecastRequest):
    try:
        df = fetch_macro_data()
        current_spot = float(df['spot'].iloc[-1])
        deviation, current_regime, phi, q_bar, model_type, drift = get_macro_state()
        
        dt = 1 / 12
        n_periods = req.horizon_years / dt
        
        if model_type == "AR(1)_mean_reverting":
            current_q = np.log(current_spot) + df['p_us'].iloc[-1] - df['p_ez'].iloc[-1]
            half_life = np.log(0.5) / np.log(phi) * dt
            residuals = df['q'].dropna() - (intercept + phi * df['q'].shift(1).dropna())
            sigma = float(np.std(residuals) * np.sqrt(12))
            phi_n = phi ** n_periods
            drift_q = (phi_n - 1) * (current_q - q_bar)
            ppp_anchor = float(np.exp(q_bar - df['p_us'].iloc[-1] + df['p_ez'].iloc[-1]))
        else:
            half_life = None
            sigma = float(df['q'].diff().std() * np.sqrt(12))
            drift_q = drift * n_periods
            ppp_anchor = None

        drift_infl = (req.pi_ez - req.pi_us) * req.horizon_years
        mean_delta = drift_q + drift_infl
        var_q = (sigma ** 2) * n_periods
        std_delta = np.sqrt(var_q)
        
        p_up = norm.cdf(mean_delta / std_delta) if std_delta > 0 else 0.5
        expected_rate = float(np.exp(np.log(current_spot) + mean_delta))
        
        historical_data = [{"date": d.strftime("%Y-%m-%d"), "price": float(r['spot'])} for d, r in df.tail(60).iterrows()]
        
        return {
            "status": "success",
            "data": {
                "inputs": req.model_dump(),
                "current_state": {
                    "spot_rate": current_spot,
                    "ppp_fair_value": ppp_anchor,
                    "deviation_q": deviation,
                    "half_life_years": half_life,
                    "current_regime": current_regime,
                    "total_regimes": 2,
                    "macro_model": model_type
                },
                "forecast": {
                    "p_up": float(p_up),
                    "expected_rate": expected_rate,
                    "median_rate": expected_rate,
                    "ci_90_low": float(np.exp(np.log(current_spot) + mean_delta - 1.645 * std_delta)),
                    "ci_90_high": float(np.exp(np.log(current_spot) + mean_delta + 1.645 * std_delta))
                },
                "historical_data": historical_data,
                "fan_chart": []
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# MICRO ENDPOINT (SHORT-TERM)
# ==========================================

@app.post("/short_term_forecast")
def get_short_term_forecast(req: ShortTermRequest):
    try:
        # 1. Macro State
        try:
            macro_dev, macro_regime, _, _, _, _ = get_macro_state()
        except Exception:
            macro_dev, macro_regime = 0.0, 0 

        # 2. Economic Calendar (Twelve Data)
        event_risk = "LOW"
        if TWELVE_DATA_KEY:
            try:
                cal_url = f"https://api.twelvedata.com/economic_calendar?country=US&apikey={TWELVE_DATA_KEY}"
                cal_res = requests.get(cal_url, timeout=4).json()
                if 'data' in cal_res:
                    for event in cal_res['data']:
                        if event.get('impact') == 'High':
                            event_risk = "HIGH"
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
            
        if df_dxy.empty:
            df = df_eur.copy()
            df['Close_DXY'] = df['Close'] 
        else:
            df_eur = df_eur[['Open', 'High', 'Low', 'Close']].rename(columns={'Close': 'Close_EUR'})
            df_dxy = df_dxy[['Close']].rename(columns={'Close': 'Close_DXY'})
            
            if df_eur.index.tz is not None: df_eur.index = df_eur.index.tz_localize(None)
            if df_dxy.index.tz is not None: df_dxy.index = df_dxy.index.tz_localize(None)
            df = df_eur.join(df_dxy, how='outer').ffill().dropna()

        if 'Volume' not in df.columns or df['Volume'].sum() == 0:
            df['Volume'] = df['High'] - df['Low'] 
            
        df['Close'] = df['Close_EUR'] if 'Close_EUR' in df.columns else df['Close']
        
        # 4. Feature Engineering
        df['Clean_Close'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['ret_1h'] = df['Clean_Close'].pct_change(1)
        df['ret_24h'] = df['Clean_Close'].pct_change(24)
        
        if 'Close_DXY' in df.columns:
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
        df['vol_mean'] = df['Volume'].rolling(window=24).mean()
        df['whale_volume'] = np.where(df['Volume'] > (df['vol_mean'] * 1.5), 1, 0)
        df['prev_high'] = df['High'].shift(1)
        df['prev_low'] = df['Low'].shift(1)
        df['bearish_sweep'] = np.where((df['High'] > df['prev_high']) & (df['Close'] < df['prev_high']) & (df['whale_volume'] == 1), -1, 0)
        df['bullish_sweep'] = np.where((df['Low'] < df['prev_low']) & (df['Close'] > df['prev_low']) & (df['whale_volume'] == 1), 1, 0)
        df['net_sweep'] = df['bullish_sweep'] + df['bearish_sweep']

        window = 5
        df['swing_high'] = df['High'].rolling(window=window).max()
        df['swing_low'] = df['Low'].rolling(window=window).min()
        df['near_swing_high'] = np.where(df['Close'] > df['swing_high'] * 0.998, 1, 0)
        df['near_swing_low'] = np.where(df['Close'] < df['swing_low'] * 1.002, 1, 0)

        df['macro_dev'] = macro_dev
        df['macro_regime'] = macro_regime

        periods_ahead = req.forecast_days * 24 
        df['target'] = (df['Close'].shift(-periods_ahead) > df['Close']).astype(int)
        
        features = ['ret_1h', 'dxy_ret_1h', 'dxy_corr', 'ret_24h', 'dist_sma', 'rsi', 'whale_volume', 'net_sweep', 
                    'macro_dev', 'macro_regime', 'near_swing_high', 'near_swing_low', 'trend_direction', 'z_score']
        features = [f for f in features if f in df.columns]
        
        df[features] = df[features].shift(1) 
        df = df.dropna()
        
        if len(df) < 100:
            raise ValueError("Not enough data")

        # 5. FAST ML Training (No heavy loops, no timeouts)
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx]
        
        base_xgb = xgb.XGBClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, objective='binary:logistic', eval_metric='logloss')
        base_rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)

        base_xgb.fit(train_df[features], train_df['target'])
        base_rf.fit(train_df[features], train_df['target'])

        xgb_weight = 0.5
        rf_weight = 0.5

        X_live = df[features].iloc[-1:]
        p_up_xgb = base_xgb.predict_proba(X_live)[0][1]
        p_up_rf = base_rf.predict_proba(X_live)[0][1]
        
        p_up = (p_up_xgb * xgb_weight) + (p_up_rf * rf_weight)

        current_price = float(df['Close'].iloc[-1])
        current_atr = float(df['atr'].iloc[-1])
        
        recent_df = df.tail(100).copy()
        recent_df['x_lag'] = recent_df['z_score'].shift(1)
        recent_df = recent_df.dropna()
        slope_z, _, _, _, _ = linregress(recent_df['x_lag'], recent_df['z_score'])
        phi_z = float(np.clip(slope_z, 0.01, 0.999))
        reversal_hours = float(np.log(0.5) / np.log(phi_z))
        
        # 6. Dynamic RSI & Signal Logic
        SPREAD_PIPS = 0.00015 
        expected_move = current_atr * 1.5 
        current_rsi = float(df['rsi'].iloc[-1])
        
        rsi_low_bound = max(df['rsi'].tail(500).quantile(0.05), 25.0)
        rsi_high_bound = min(df['rsi'].tail(500).quantile(0.95), 75.0)
        
        p_up_adjusted = p_up
        rsi_weight = 0.60 
        
        if current_rsi < rsi_low_bound:
            oversold_score = (rsi_low_bound - current_rsi) / rsi_low_bound
            p_up_adjusted += (rsi_weight * oversold_score)
            
        if current_rsi > rsi_high_bound:
            overbought_score = (current_rsi - rsi_high_bound) / (100 - rsi_high_bound)
            p_up_adjusted -= (rsi_weight * overbought_score)
            
        p_up_adjusted = float(np.clip(p_up_adjusted, 0.01, 0.99))
        
        signal = "HOLD"
        if event_risk == "HIGH":
            signal = "HOLD" 
        elif p_up_adjusted > 0.60 and macro_dev >= -0.03 and expected_move > SPREAD_PIPS:
            signal = "BUY"
            if df['net_sweep'].iloc[-1] == -1: signal = "HOLD"
        elif p_up_adjusted < 0.40 and macro_dev <= 0.03 and expected_move > SPREAD_PIPS:
            signal = "SELL"
            if df['net_sweep'].iloc[-1] == 1: signal = "HOLD"

        current_trend = int(df['trend_direction'].iloc[-1])
        current_z = float(df['z_score'].iloc[-1])
        
        if signal == "SELL" and current_trend == 1: signal = "HOLD" 
        if signal == "BUY" and current_trend == -1: signal = "HOLD" 
        if signal == "SELL" and current_z < -2.0: signal = "HOLD" 
        if signal == "BUY" and current_z > 2.0: signal = "HOLD" 

        # 7. Kelly Sizing (Re-enabled)
        risk_reward_ratio = 2.0 
        kelly_fraction = 0.0
        if signal == "BUY":
            kelly_fraction = (p_up_adjusted - ((1 - p_up_adjusted) / risk_reward_ratio))
        elif signal == "SELL":
            kelly_fraction = ((1 - p_up_adjusted) - (p_up_adjusted / risk_reward_ratio))
        kelly_fraction = max(0.0, min(kelly_fraction, 0.05))

        # 8. ATR Risk Management (Dynamic)
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
                "forecast_days": req.forecast_days,
                "probability_up": float(p_up_adjusted),
                "signal": signal,
                "target_price": float(target_price),
                "stop_loss": float(stop_loss),
                "last_rsi": float(current_rsi),
                "macro_bias": "Bullish" if macro_dev > 0 else "Bearish",
                "whale_trap_detected": bool(df['net_sweep'].iloc[-1] != 0),
                "event_risk": event_risk,
                "dxy_correlation": float(df['dxy_corr'].iloc[-1]) if 'dxy_corr' in df.columns else 0.0,
                "trend_direction": "Up" if current_trend == 1 else "Down",
                "reversal_expectation_hours": reversal_hours,
                "recommended_position_size_pct": float(kelly_fraction * 100),
                "model_weights": {"xgb": float(xgb_weight), "rf": float(rf_weight)},
                "history": history
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 