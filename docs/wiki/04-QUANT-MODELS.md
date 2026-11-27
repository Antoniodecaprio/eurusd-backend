# Quant Models

## Original long-horizon concept
Ornstein–Uhlenbeck mean reversion on log EUR/USD, a PPP equilibrium anchor, two-regime Gaussian-mixture volatility and probabilistic fan charts extending up to 10 years.

## Historical illustrative parameters only
Previously discussed: PPP equilibrium ~1.1420, mean-reversion speed ~0.2476/year, half-life ~2.8 years, spot 1.11433, expected 1.1245, and probability EUR/USD above spot 58.5%. These are historical examples—not current market values, validated parameters or forecasts.

## Model meaning and caveats
OU pulls toward an assumed equilibrium; the anchor and mean-reversion assumption can be wrong or regime-sensitive, particularly at shorter horizons. Long-horizon uncertainty should expand. Never present a point target without horizon, distribution/range, probability definition and assumptions.

## Later short-term direction
M15/H1/H4/D1 analysis was introduced. Candidate features discussed: RSI, ATR, trend, z-scores, volatility regimes and macro factors. Random Forest and XGBoost were ideas. Their discussion is not proof they exist or perform well. Regimes and features must avoid look-ahead leakage. Keep simple baselines and prove any complex model adds out-of-sample value.