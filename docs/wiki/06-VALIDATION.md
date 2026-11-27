# Validation Standards

A chart that looks convincing is not proof. Forecast probabilities need out-of-sample evaluation and baselines.

## Historical v4.4 result (preserve)
Brier score 0.3823 vs baseline 0.2126; Brier skill -0.7981; validation failed; forecast validity false; HOLD. Do not erase or reinterpret as success. v6 needs independent testing.

## Design
Chronological splits and walk-forward; no random shuffling. Features must only use information available at forecast time. Address overlapping horizons with purging/embargo where appropriate. Keep an untouched final holdout. Fit scaling/feature selection/calibration only within training data. Predeclare gates.

## Metrics and baselines
Brier, log loss, reliability/calibration, probability buckets, interval coverage/width, MAE/RMSE. Strategy tests (if applicable): drawdown, turnover, costs and risk-adjusted metrics. Compare no-change/random-walk, persistence, unconditional probabilities and basic volatility models.

## Realism
Include bid/ask, spread, commission, slippage, financing/swap and execution delay for strategy evaluation. Promotion criteria (sample size, calibration, baseline improvement, stability, drawdown) must be justified before results. Monitor drift and disable/downgrade degraded models.