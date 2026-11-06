from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

app = FastAPI()

# Allow FlutterFlow to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        # Fetch real data (EUR/USD exchange rate via Yahoo Finance)
        ticker = yf.Ticker("EURUSD=X")
        hist = ticker.history(period="10y", interval="1mo")
        if hist.empty:
            raise ValueError("No data returned from Yahoo Finance")
            
        current_spot = float(hist['Close'].iloc[-1])
        
        # Hardcode PPP anchor and half-life derived from our macro analysis
        ppp_anchor = 1.1420 
        half_life = 2.8
        sigma = 0.035
        dt = 1 / 12
        
        # OU parameters
        phi = 0.5 ** (dt / half_life) # Adjusted phi for monthly steps
        n_periods = req.horizon_years / dt
        q_t = np.log(current_spot) - np.log(ppp_anchor)
        
        # Calculate Drift & Variance
        drift_q = (phi ** n_periods - 1) * (q_t)
        drift_infl = (req.pi_ez - req.pi_us) * req.horizon_years
        mean_delta = drift_q + drift_infl
        
        var_q = (sigma ** 2) * (1 - phi ** (2 * n_periods)) / (1 - phi ** 2)
        std_delta = np.sqrt(var_q)
        
        # Probabilities & Intervals
        p_up = norm.cdf(mean_delta / std_delta) if std_delta > 0 else 0.5
        expected_rate = np.exp(np.log(current_spot) + mean_delta)
        ci_90_low = np.exp(np.log(current_spot) + mean_delta - 1.645 * std_delta)
        ci_90_high = np.exp(np.log(current_spot) + mean_delta + 1.645 * std_delta)
        
        # Build the Fan Chart Data
        fan_chart = []
        steps = 12 
        for i in range(1, steps + 1):
            h = (req.horizon_years / steps) * i
            n = h / dt
            m_drift = (phi ** n - 1) * q_t + drift_infl * (h / req.horizon_years)
            v_q = (sigma ** 2) * (1 - phi ** (2 * n)) / (1 - phi ** 2)
            s_delta = np.sqrt(v_q)
            
            fan_chart.append({
                "months_ahead": int(h * 12),
                "date": str(pd.Timestamp.now() + pd.DateOffset(months=int(h * 12)))[:10],
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
                    "deviation_q": float(q_t),
                    "half_life_years": half_life,
                    "current_regime": 1,
                    "total_regimes": 2
                },
                "forecast": {
                    "p_up": float(p_up),
                    "expected_rate": float(expected_rate),
                    "median_rate": float(expected_rate),
                    "ci_90_low": float(ci_90_low),
                    "ci_90_high": float(ci_90_high)
                },
                "fan_chart": fan_chart
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))