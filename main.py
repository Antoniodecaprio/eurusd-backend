from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from fredapi import Fred
from scipy.stats import norm
from scipy.stats import linregress
from sklearn.mixture import GaussianMixture
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

@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.post("/forecast")
def get_forecast(req: ForecastRequest):
    try:
        if not FRED_API_KEY:
            raise ValueError("FRED_API_KEY environment variable is not set")
        
        fred = Fred(api_key=FRED_API_KEY)
        
        # 1. Fetch Data (Spot, US CPI, EZ HICP)
        spot_series = fred.get_series('DEXUSEU').dropna()
        us_cpi = fred.get_series('CPIAUCSL').dropna()
        ez_cpi = fred.get_series('CP0000EZ19M086NEST').dropna()
        
        # Align data to monthly frequency
        df = pd.DataFrame({'spot': spot_series, 'us_cpi': us_cpi, 'ez_cpi': ez_cpi})
        df = df.resample('ME').last().dropna() # Month-End frequency
        df = df[df.index >= '1999-01-01'] # Official EUR era
        
        if len(df) < 36:
            raise ValueError("Not enough historical data to estimate parameters")

        # 2. Calculate Real Exchange Rate (q = s + p_us - p_ez)
        df['s'] = np.log(df['spot'])
        df['p_us'] = np.log(df['us_cpi'])
        df['p_ez'] = np.log(df['ez_cpi'])
        df['q'] = df['s'] + df['p_us'] - df['p_ez']
        
        current_spot = float(df['spot'].iloc[-1])
        current_q = float(df['q'].iloc[-1])
        
        # 3. Estimate Dynamic OU Parameters via OLS
        # q_t = c + phi * q_{t-1} + error
        df['q_lag'] = df['q'].shift(1)
        df_ols = df.dropna()
        slope, intercept, r_value, p_value, std_err = linregress(df_ols['q_lag'], df_ols['q'])
        
        # Clip phi to avoid explosive or perfectly stationary processes
        phi = float(np.clip(slope, 0.01, 0.999))
        q_bar = float(intercept / (1 - phi)) # Long-term mean (PPP Anchor)
        
        # Convert to continuous time (Annualized)
        dt = 1 / 12 # monthly steps
        theta = -np.log(phi) / dt
        half_life = np.log(0.5) / np.log(phi) * dt
        
        # Residual volatility
        residuals = df_ols['q'] - (intercept + phi * df_ols['q_lag'])
        sigma = float(np.std(residuals) * np.sqrt(12)) # Annualized
        
        # 4. Regime Detection (GMM on rolling volatility of monthly returns)
        df['returns'] = df['s'].diff()
        df['vol'] = df['returns'].rolling(window=6).std().bfill()
        
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(df['vol'].values.reshape(-1, 1))
        regimes = gmm.predict(df['vol'].values.reshape(-1, 1))
        current_regime = int(regimes[-1])
        
        # Reconstruct PPP Fair Value from q_bar
        ppp_anchor = float(np.exp(q_bar - df['p_us'].iloc[-1] + df['p_ez'].iloc[-1]))

        # 5. Generate Forecast Distribution
        n_periods = req.horizon_years / dt
        phi_n = phi ** n_periods
        
        drift_q = (phi_n - 1) * (current_q - q_bar)
        drift_infl = (req.pi_ez - req.pi_us) * req.horizon_years
        mean_delta = drift_q + drift_infl
        
        var_q = (sigma ** 2) * (1 - phi ** (2 * n_periods)) / (1 - phi ** 2)
        std_delta = np.sqrt(var_q)
        
        p_up = norm.cdf(mean_delta / std_delta) if std_delta > 0 else 0.5
        expected_rate = float(np.exp(np.log(current_spot) + mean_delta))
        # 5.5 Build Historical Data for the Chart (Last 60 Months)
historical_data = []
df_hist = df.tail(60) # Grab last 5 years
for date, row in df_hist.iterrows():
    historical_data.append({
        "date": date.strftime("%Y-%m-%d"),
        "price": float(row['spot'])
    })

        # 6. Build Fan Chart Data
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