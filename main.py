from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from fredapi import Fred
from scipy.stats import norm, linregress
from sklearn.mixture import GaussianMixture
import yfinance as yf
import xgboost as xgb
import os
import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRED_API_KEY = os.getenv("FRED_API_KEY")

class ForecastRequest(BaseModel):
    horizon_years: float = 3.0
    pi_us: float = 0.02
    pi_ez: float = 0.02

class ShortTermRequest(BaseModel):
    forecast_days: int = 3

def fetch_and_prep_data():
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY environment variable is not set")
    
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

@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.post("/forecast")
def get_forecast(req: ForecastRequest):
    try:
        df = fetch_and_prep_data()
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
        
        historical_data = []
        for date, row in df.tail(60).iterrows():
            historical_data.append({"date": date.strftime("%Y-%m-%d"), "price": float(row['spot'])})

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
        df = fetch_and_prep_data()
        horizon_months = int(req.horizon_years * 12)
        test_periods = 60
        
        if len(df) < test_periods + horizon_months:
            raise ValueError("Not enough data for this backtest horizon")
            
        probs_up = []
        actuals = []
        hits_50 = []
        hits_90 = []
        
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

@app.post("/short_term_forecast")
def get_short_term_forecast(req: ShortTermRequest):
    try:
        ticker = yf.Ticker("EURUSD=X")
        df = ticker.history(period="60d", interval="1h")
        
        if df.empty or len(df) < 100:
            raise ValueError("Not enough intraday data returned from Yahoo Finance")
            
        df = df[['Close']].copy()
        
        df['ret_1h'] = df['Close'].pct_change(1)
        df['ret_4h'] = df['Close'].pct_change(4)
        df['ret_24h'] = df['Close'].pct_change(24)
        
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['dist_sma'] = (df['Close'] - df['sma_10']) / df['sma_10']
        
        df['volatility'] = df['ret_1h'].rolling(window=24).std()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        periods_ahead = req.forecast_days * 24 
        df['target'] = (df['Close'].shift(-periods_ahead) > df['Close']).astype(int)
        
        df = df.dropna()
        
        if len(df) < 50:
            raise ValueError("Not enough valid data after feature engineering")
            
        features = ['ret_1h', 'ret_4h', 'ret_24h', 'dist_sma', 'volatility', 'rsi']
        X = df[features]
        y = df['target']
        
        X_train = X.iloc[:-1]
        y_train = y.iloc[:-1]
        X_live = X.iloc[-1:]
        
        model = xgb.XGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=3,
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        p_up = float(model.predict_proba(X_live)[0][1])
        current_price = float(df['Close'].iloc[-1])
        
        if p_up > 0.6:
            signal = "BUY"
            target_price = current_price * 1.005
            stop_loss = current_price * 0.995
        elif p_up < 0.4:
            signal = "SELL"
            target_price = current_price * 0.995
            stop_loss = current_price * 1.005
        else:
            signal = "HOLD"
            target_price = current_price
            stop_loss = current_price

        return {
            "status": "success",
            "data": {
                "current_price": current_price,
                "forecast_days": req.forecast_days,
                "probability_up": p_up,
                "signal": signal,
                "target_price": target_price,
                "stop_loss": stop_loss,
                "model_features": features,
                "last_rsi": float(df['rsi'].iloc[-1])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))