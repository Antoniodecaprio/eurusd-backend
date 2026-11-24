from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from fredapi import Fred
from scipy.stats import norm, linregress
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import yfinance as yf
import xgboost as xgb
import os
import datetime
import requests
import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRED_API_KEY = os.getenv("FRED_API_KEY")

# ==========================================
# DATA MODELS
# ==========================================

class ForecastRequest(BaseModel):
    horizon_years: float = 3.0
    pi_us: float = 0.02
    pi_ez: float = 0.02

class ShortTermRequest(BaseModel):
    forecast_days: int = 3

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def fetch_macro_data():
    """Fetches FRED data for the long-term OU model."""
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY environment variable not set")
    fred = Fred(api_key=FRED_API_KEY)
    spot_series = fred.get_series('DEXUSEU').dropna()
    us_cpi = fred.get_series('CPIAUCSL').dropna()
    ez_cpi = fred.get_series('CP0000EZ19M086NEST').dropna()
    
    df = pd.DataFrame({'spot': spot_series, 'us_cpi': us_cpi, 'ez_cpi': ez_cpi})
    df = df.resample('ME').last().dropna()
    df = df[df.index >= '1999-01-01']
    
    df['s'] = np.log(df['spot'])
    df['p_us'] = np.log(df['us_cpi'])
    df['p_ez'] = np.log(df['ez_cpi'])
    df['q'] = df['s'] + df['p_us'] - df['p_ez']
    return df

def get_macro_state():
    """Calculates current Macro Deviation and Regime for integration into Micro model."""
    df = fetch_macro_data()
    df['q_lag'] = df['q'].shift(1)
    df_ols = df.dropna()
    
    slope, intercept, _, _, _ = linregress(df_ols['q_lag'], df_ols['q'])
    phi = float(np.clip(slope, 0.01, 0.999))
    q_bar = float(intercept / (1 - phi))
    
    current_q = float(df['q'].iloc[-1])
    deviation = current_q - q_bar 
    
    df['returns'] = df['s'].diff()
    df['vol'] = df['returns'].rolling(window=6).std().bfill()
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(df['vol'].values.reshape(-1, 1))
    regimes = gmm.predict(df['vol'].values.reshape(-1, 1))
    current_regime = int(regimes[-1])
    
    return deviation, current_regime

def calculate_atr(df, window=14):
    """Calculates Average True Range for dynamic stop-losses."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=window).mean()

# ==========================================
# MACRO ENDPOINTS (LONG-TERM)
# ==========================================

@app.get("/")
def read_root():
    return {"status": "API is running - Elite Quant Engine Active"}

@app.post("/forecast")
def get_forecast(req: ForecastRequest):
    try:
        df = fetch_macro_data()
        current_spot = float(df['spot'].iloc[-1])
        current_q = float(df['q'].iloc[-1])
        
        df['q_lag'] = df['q'].shift(1)
        df_ols = df.dropna()
        slope, intercept, _, _, _ = linregress(df_ols['q_lag'], df_ols['q'])
        
        phi = float(np.clip(slope, 0.01, 0.999))
        q_bar = float(intercept / (1 - phi))
        dt = 1 / 12
        half_life = np.log(0.5) / np.log(phi) * dt
        residuals = df_ols['q'] - (intercept + phi * df_ols['q_lag'])
        sigma = float(np.std(residuals) * np.sqrt(12))
        
        df['returns'] = df['s'].diff()
        df['vol'] = df['returns'].rolling(window=6).std().bfill()
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(df['vol'].values.reshape(-1, 1))
        regimes = gmm.predict(df['vol'].values.reshape(-1, 1))
        current_regime = int(regimes[-1])
        
        ppp_anchor = float(np.exp(q_bar - df['p_us'].iloc[-1] + df['p_ez'].iloc[-1]))

        n_periods = req.horizon_years / dt
        phi_n = phi ** n_periods
        drift_q = (phi_n - 1) * (current_q - q_bar)
        drift_infl = (req.pi_ez - req.pi_us) * req.horizon_years
        mean_delta = drift_q + drift_infl
        
        var_q = (sigma ** 2) * (1 - phi ** (2 * n_periods)) / (1 - phi ** 2)
        std_delta = np.sqrt(var_q)
        
        p_up = norm.cdf(mean_delta / std_delta) if std_delta > 0 else 0.5
        expected_rate = float(np.exp(np.log(current_spot) + mean_delta))
        
        historical_data = [{"date": d.strftime("%Y-%m-%d"), "price": float(r['spot'])} for d, r in df.tail(60).iterrows()]
        fan_chart = []
        steps = 12 
        for i in range(1, steps + 1):
            h = (req.horizon_years / steps) * i
            n = h / dt
            m_drift = (phi ** n - 1) * (current_q - q_bar) + drift_infl * (h / req.horizon_years)
            v_q = (sigma ** 2) * (1 - phi ** (2 * n)) / (1 - phi ** 2)
            s_delta = np.sqrt(v_q)
            future_date = (datetime.date.today() + datetime.timedelta(days=int(h*365))).strftime("%Y-%m-%d")
            fan_chart.append({
                "months_ahead": int(h * 12),
                "date": future_date,
                "median": float(np.exp(np.log(current_spot) + m_drift)),
                "mean": float(np.exp(np.log(current_spot) + m_drift)),
                "q05": float(np.exp(np.log(current_spot) + m_drift - 1.645 * s_delta)),
                "q25": float(np.exp(np.log(current_spot) + m_drift - 0.674 * s_delta)),
                "q75": float(np.exp(np.log(current_spot) + m_drift + 0.674 * s_delta)),
                "q95": float(np.exp(np.log(current_spot) + m_drift + 1.645 * s_delta)),
            })

        return {
            "status": "success",
            "data": {
                "inputs": req.model_dump(),
                "current_state": {
                    "spot_rate": current_spot,
                    "ppp_fair_value": ppp_anchor,
                    "deviation_q": current_q - q_bar,
                    "half_life_years": float(half_life),
                    "current_regime": current_regime,
                    "total_regimes": 2
                },
                "forecast": {
                    "p_up": float(p_up),
                    "expected_rate": expected_rate,
                    "median_rate": expected_rate,
                    "ci_90_low": float(np.exp(np.log(current_spot) + mean_delta - 1.645 * std_delta)),
                    "ci_90_high": float(np.exp(np.log(current_spot) + mean_delta + 1.645 * std_delta))
                },
                "historical_data": historical_data,
                "fan_chart": fan_chart
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backtest")
def run_backtest(req: ForecastRequest):
    try:
        df = fetch_macro_data()
        horizon_months = int(req.horizon_years * 12)
        test_periods = 60
        
        if len(df) < test_periods + horizon_months:
            raise ValueError("Not enough data for this backtest horizon")
            
        probs_up, actuals, hits_50, hits_90 = [], [], [], []
        
        for i in range(len(df) - test_periods - horizon_months, len(df) - horizon_months):
            train_df = df.iloc[:i].dropna()
            
            slope, intercept, _, _, _ = linregress(train_df['q_lag'], train_df['q'])
            phi = float(np.clip(slope, 0.01, 0.999))
            q_bar = float(intercept / (1 - phi))
            sigma = float(np.std(train_df['q'] - (intercept + phi * train_df['q_lag'])) * np.sqrt(12))
            dt = 1/12
            
            current_q = float(df['q'].iloc[i])
            current_spot = float(df['spot'].iloc[i])
            
            n = horizon_months
            phi_n = phi ** n
            mean_delta = (phi_n - 1) * (current_q - q_bar)
            var_q = (sigma ** 2) * (1 - phi ** (2 * n)) / (1 - phi ** 2)
            std_delta = np.sqrt(var_q)
            
            p_up = norm.cdf(mean_delta / std_delta) if std_delta > 0 else 0.5
            expected_log = np.log(current_spot) + mean_delta
            
            q05 = np.exp(expected_log - 1.645 * std_delta)
            q25 = np.exp(expected_log - 0.674 * std_delta)
            q75 = np.exp(expected_log + 0.674 * std_delta)
            q95 = np.exp(expected_log + 1.645 * std_delta)
            
            actual_spot = float(df['spot'].iloc[i + horizon_months])
            actual_direction = 1 if actual_spot > current_spot else 0
            
            probs_up.append(p_up)
            actuals.append(actual_direction)
            hits_50.append(1 if (actual_spot >= q25 and actual_spot <= q75) else 0)
            hits_90.append(1 if (actual_spot >= q05 and actual_spot <= q95) else 0)

        probs_up = np.array(probs_up)
        actuals = np.array(actuals)
        
        brier_score = float(np.mean((probs_up - actuals) ** 2))
        cov_50 = float(np.mean(hits_50))
        cov_90 = float(np.mean(hits_90))
        
        bins = np.linspace(0, 1, 6)
        ece = 0
        for b in range(len(bins)-1):
            mask = (probs_up >= bins[b]) & (probs_up < bins[b+1])
            if np.sum(mask) > 0:
                avg_pred = np.mean(probs_up[mask])
                avg_actual = np.mean(actuals[mask])
                ece += (np.sum(mask) / len(probs_up)) * abs(avg_pred - avg_actual)
        reliability_index = float(1.0 - ece)

        calibration_curve = []
        for b in range(len(bins)-1):
            mask = (probs_up >= bins[b]) & (probs_up < bins[b+1])
            if np.sum(mask) > 0:
                calibration_curve.append({
                    "predicted": float(np.mean(probs_up[mask])),
                    "actual": float(np.mean(actuals[mask]))
                })
            else:
                calibration_curve.append({"predicted": float((bins[b]+bins[b+1])/2), "actual": float((bins[b]+bins[b+1])/2)})

        return {
            "status": "success",
            "data": {
                "horizon_years": req.horizon_years,
                "n_tests": len(probs_up),
                "brier_score": brier_score,
                "coverage_50": cov_50,
                "coverage_90": cov_90,
                "reliability_index": reliability_index,
                "calibration_curve": calibration_curve
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# MICRO ENDPOINT (SHORT-TERM ML ENGINE)
# ==========================================

@app.post("/short_term_forecast")
def get_short_term_forecast(req: ShortTermRequest):
    try:
        # 1. FUSION: Get Macro Bias First
        try:
            macro_dev, macro_regime = get_macro_state()
        except Exception:
            macro_dev, macro_regime = 0.0, 0 

        # 2. NEWS & EVENT RISK FILTER
        twelvedata_key = os.getenv("TWELVE_DATA_KEY")
        event_risk = "LOW"
        if twelvedata_key:
            try:
                cal_url = f"https://api.twelvedata.com/economic_calendar?country=US&apikey={twelvedata_key}"
                cal_res = requests.get(cal_url, timeout=5).json()
                if 'data' in cal_res:
                    for event in cal_res['data']:
                        if event.get('impact') == 'High':
                            event_risk = "HIGH"
            except Exception:
                pass

        # 3. ROBUST DATA FETCHING (EUR/USD + DXY)
        df_eur = pd.DataFrame()
        df_dxy = pd.DataFrame()
        
        if twelvedata_key:
            try:
                url = f"https://api.twelvedata.com/time_series?symbol=EUR/USD&interval=1h&outputsize=500&apikey={twelvedata_key}"
                response = requests.get(url, timeout=10)
                json_data = response.json()
                
                if 'values' in json_data:
                    df_eur = pd.DataFrame(json_data['values'])
                    df_eur['datetime'] = pd.to_datetime(df_eur['datetime'])
                    df_eur = df_eur.set_index('datetime').sort_index()
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df_eur.columns:
                            df_eur[col] = pd.to_numeric(df_eur[col])
                    df_eur = df_eur.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
            except Exception as e:
                print(f"Twelve Data EUR failed: {e}")

        # Fetch DXY (Dollar Index)
        try:
            ticker_dxy = yf.Ticker("DX-Y.NYB")
            df_dxy = ticker_dxy.history(period="60d", interval="1h")
        except Exception:
            pass

        # Fallbacks if Twelve Data fails
        if df_eur.empty or len(df_eur) < 100:
            ticker_eur = yf.Ticker("EURUSD=X")
            df_eur = ticker_eur.history(period="60d", interval="1h")

        if df_eur.empty or df_dxy.empty or len(df_eur) < 100:
            # Fallback: If DXY fails, just use EUR/USD
            if df_eur.empty or len(df_eur) < 100:
                raise ValueError("Data unavailable from all sources")
            df = df_eur.copy()
            df['Close_DXY'] = df['Close'] # Dummy fallback
        else:
            # Align EUR and DXY data
            df_eur = df_eur[['Open', 'High', 'Low', 'Close']].rename(columns={'Close': 'Close_EUR'})
            df_dxy = df_dxy[['Close']].rename(columns={'Close': 'Close_DXY'})
            df = df_eur.join(df_dxy, how='inner').dropna()

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
        
        # RSI (Fixed Division by Zero)
        delta = df['Clean_Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9) # Prevent division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df['atr'] = calculate_atr(df, window=14)
        df['vol_mean'] = df['Volume'].rolling(window=24).mean()
        df['whale_volume'] = np.where(df['Volume'] > (df['vol_mean'] * 1.5), 1, 0)

        df['prev_high'] = df['High'].shift(1)
        df['prev_low'] = df['Low'].shift(1)
        df['bearish_sweep'] = np.where((df['High'] > df['prev_high']) & (df['Close'] < df['prev_high']) & (df['whale_volume'] == 1), -1, 0)
        df['bullish_sweep'] = np.where((df['Low'] < df['prev_low']) & (df['Close'] > df['prev_low']) & (df['whale_volume'] == 1), 1, 0)
        df['net_sweep'] = df['bullish_sweep'] + df['bearish_sweep']

        # ZigZag Pivot Features (Accuracy Upgrade)
        window = 5
        df['swing_high'] = df['High'].rolling(window=window).max()
        df['swing_low'] = df['Low'].rolling(window=window).min()
        df['near_swing_high'] = np.where(df['Close'] > df['swing_high'] * 0.998, 1, 0)
        df['near_swing_low'] = np.where(df['Close'] < df['swing_low'] * 1.002, 1, 0)

        # Macro Injection
        df['macro_dev'] = macro_dev
        df['macro_regime'] = macro_regime

        # Target (3 days ahead)
        periods_ahead = req.forecast_days * 24 
        df['target'] = (df['Close'].shift(-periods_ahead) > df['Close']).astype(int)
        
        features = ['ret_1h', 'dxy_ret_1h', 'dxy_corr', 'ret_24h', 'dist_sma', 'rsi', 'whale_volume', 'net_sweep', 
                    'macro_dev', 'macro_regime', 'near_swing_high', 'near_swing_low']
        
        # Drop missing features if DXY failed
        features = [f for f in features if f in df.columns]
        
        df[features] = df[features].shift(1) # Prevent leakage
        df = df.dropna()
        
        if len(df) < 100:
            raise ValueError("Not enough valid data after feature engineering")

        # 5. WALK-FORWARD VALIDATION & CALIBRATION
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        X_train, y_train = train_df[features], train_df['target']
        X_val, y_val = val_df[features], val_df['target']

        # Base Models
        base_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, objective='binary:logistic', eval_metric='logloss')
        base_rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)

        # Calibrated Models (Accuracy Upgrade)
        xgb_model = CalibratedClassifierCV(base_xgb, method='sigmoid', cv=3)
        rf_model = CalibratedClassifierCV(base_rf, method='sigmoid', cv=3)

        xgb_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)

        # Calculate Brier Scores
        xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
        rf_preds = rf_model.predict_proba(X_val)[:, 1]
        
        xgb_brier = brier_score_loss(y_val, xgb_preds)
        rf_brier = brier_score_loss(y_val, rf_preds)

        # Dynamic Weights
        xgb_weight = (1 / (xgb_brier + 1e-5))
        rf_weight = (1 / (rf_brier + 1e-5))
        total_weight = xgb_weight + rf_weight
        xgb_weight /= total_weight
        rf_weight /= total_weight

        # Retrain on FULL dataset before live prediction
        xgb_model.fit(df[features], df['target'])
        rf_model.fit(df[features], df['target'])

        X_live = df[features].iloc[-1:]
        p_up_xgb = xgb_model.predict_proba(X_live)[0][1]
        p_up_rf = rf_model.predict_proba(X_live)[0][1]
        
        # Final Weighted Probability
        p_up = (p_up_xgb * xgb_weight) + (p_up_rf * rf_weight)

        current_price = float(df['Close'].iloc[-1])
        current_atr = float(df['atr'].iloc[-1])
        
        # 6. TRANSACTION COSTS, NEWS & SIGNAL LOGIC
        SPREAD_PIPS = 0.00015 
        signal = "HOLD"
        expected_move = current_atr * 1.5 
        
        if event_risk == "HIGH":
            signal = "HOLD" # Override if major news is coming
        elif p_up > 0.60 and macro_dev >= -0.03 and expected_move > SPREAD_PIPS:
            signal = "BUY"
            if df['net_sweep'].iloc[-1] == -1: signal = "HOLD"
        elif p_up < 0.40 and macro_dev <= 0.03 and expected_move > SPREAD_PIPS:
            signal = "SELL"
            if df['net_sweep'].iloc[-1] == 1: signal = "HOLD"

        # 7. DYNAMIC KELLY POSITION SIZING
        risk_reward_ratio = 2.0 
        kelly_fraction = 0.0
        
        if signal == "BUY":
            kelly_fraction = (p_up - ((1 - p_up) / risk_reward_ratio))
        elif signal == "SELL":
            kelly_fraction = ((1 - p_up) - (p_up / risk_reward_ratio))
            
        kelly_fraction = max(0.0, min(kelly_fraction, 0.05))

        # 8. ATR RISK MANAGEMENT (Fixed HOLD targets)
        if signal == "BUY":
            target_price = current_price + (3.0 * current_atr) 
            stop_loss = current_price - (1.5 * current_atr)
        elif signal == "SELL":
            target_price = current_price - (3.0 * current_atr)
            stop_loss = current_price + (1.5 * current_atr)
        else:
            # Show ATR band even on HOLD to prevent UI bug
            target_price = current_price + (1.5 * current_atr)
            stop_loss = current_price - (1.5 * current_atr)

        # Format history for mini-chart
        history = [{"date": d.strftime("%Y-%m-%d %H:%M:%S"), "price": float(r['Close'])} for d, r in df.tail(48).iterrows()]

        return {
            "status": "success",
            "data": {
                "current_price": current_price,
                "forecast_days": req.forecast_days,
                "probability_up": float(p_up),
                "signal": signal,
                "target_price": float(target_price),
                "stop_loss": float(stop_loss),
                "last_rsi": float(df['rsi'].iloc[-1]),
                "macro_bias": "Bullish" if macro_dev > 0 else "Bearish",
                "whale_trap_detected": bool(df['net_sweep'].iloc[-1] != 0),
                "event_risk": event_risk,
                "dxy_correlation": float(df['dxy_corr'].iloc[-1]) if 'dxy_corr' in df.columns else 0.0,
                "recommended_position_size_pct": float(kelly_fraction * 100),
                "model_weights": {"xgb": float(xgb_weight), "rf": float(rf_weight)},
                "history": history
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))